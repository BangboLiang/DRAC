from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from llama3_comm.astra_emit import (
    build_dag_reconfiguration_events,
    render_mutable_network_yaml,
)
from llama3_comm.peer_plan import (
    AbstractPeerPlanEvent,
    abstract_peer_edges_for_trace_op,
    build_abstract_peer_plan,
    instantiate_concrete_edge_state,
)
from llama3_comm.trace_ir import CommTriggerRef, TraceBundle, TraceOp, TraceRankGraph
from llama3_comm.rank_lift import (
    collect_domain_edge_templates,
    edge_bandwidth_state_for_share,
)
from llama3_comm.trace_ingest import (
    load_trace_bundle,
    select_representative_rank,
    stable_topological_ops,
)
from llama3_comm.trace_to_comm import build_comm_nodes_from_rank_trace


def _rank_from_coords(
    tp_rank: int, dp_rank: int, pp_rank: int, *, tp: int, dp: int
) -> int:
    return tp_rank + dp_rank * tp + pp_rank * tp * dp


def _tp_group(rank: int, *, tp: int, dp: int, pp: int) -> list[int]:
    pp_rank = rank // (tp * dp)
    rem = rank % (tp * dp)
    dp_rank = rem // tp
    return [
        _rank_from_coords(tp_rank, dp_rank, pp_rank, tp=tp, dp=dp)
        for tp_rank in range(tp)
    ]


def _dp_group(rank: int, *, tp: int, dp: int, pp: int) -> list[int]:
    pp_rank = rank // (tp * dp)
    rem = rank % (tp * dp)
    tp_rank = rem % tp
    return [
        _rank_from_coords(tp_rank, dp_rank, pp_rank, tp=tp, dp=dp)
        for dp_rank in range(dp)
    ]


def _make_native_trace_dir(root: Path) -> Path:
    tp = 2
    dp = 2
    pp = 3
    num_ranks = tp * dp * pp
    trace_dir = root / "trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": "synthetic_training_trace/v1",
        "summary": {"num_ranks": num_ranks},
    }
    (trace_dir / "trace_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    for rank in range(num_ranks):
        pp_rank = rank // (tp * dp)
        rem = rank % (tp * dp)
        dp_rank = rem // tp
        tp_rank = rem % tp
        coords = {
            "global_rank": rank,
            "tp_rank": tp_rank,
            "cp_rank": 0,
            "dp_rank": dp_rank,
            "pp_rank": pp_rank,
        }
        prev_peer = (
            None
            if pp_rank == 0
            else _rank_from_coords(tp_rank, dp_rank, pp_rank - 1, tp=tp, dp=dp)
        )
        next_peer = (
            None
            if pp_rank == pp - 1
            else _rank_from_coords(tp_rank, dp_rank, pp_rank + 1, tp=tp, dp=dp)
        )
        tp_group = _tp_group(rank, tp=tp, dp=dp, pp=pp)
        dp_group = _dp_group(rank, tp=tp, dp=dp, pp=pp)

        events = []

        def add_event(
            event_type: str, op_name: str, duration_us: int | None, **extra
        ) -> None:
            event_id = f"r{rank:03d}_e{len(events):06d}"
            predecessors = [] if not events else [events[-1]["id"]]
            event = {
                "id": event_id,
                "event_type": event_type,
                "phase": extra.pop("phase", "FORWARD"),
                "rank": rank,
                "coordinates": coords,
                "microbatch_id": 0,
                "virtual_pipeline_chunk_id": 0,
                "layer_id": 0,
                "op_name": op_name,
                "predecessors": predecessors,
                "payload_bytes": int(extra.pop("payload_bytes", 0)),
                "flops": float(extra.pop("flops", 0.0)),
                "duration_us": duration_us,
                "communicator": extra.pop("communicator", None),
            }
            if "peer_rank" in extra:
                event["peer_rank"] = extra.pop("peer_rank")
            event.update(extra)
            events.append(event)

        add_event("COMPUTE", "compute_0", 1000, flops=1.0)
        if prev_peer is not None:
            add_event(
                "RECV",
                "pp_recv_forward",
                None,
                payload_bytes=128,
                peer_rank=prev_peer,
                phase="FORWARD",
            )
        add_event("COMPUTE", "compute_1", 2000, flops=2.0)
        add_event(
            "COLLECTIVE",
            "tp_all_gather_attention",
            None,
            payload_bytes=256,
            communicator={"group_type": "TP", "ranks": tp_group, "size": len(tp_group)},
            phase="FORWARD",
        )
        add_event("COMPUTE", "compute_2", 3000, flops=3.0)
        if next_peer is not None:
            add_event(
                "SEND",
                "pp_send_forward",
                None,
                payload_bytes=512,
                peer_rank=next_peer,
                phase="FORWARD",
            )
        add_event("COMPUTE", "compute_3", 4000, flops=4.0)
        if next_peer is not None:
            add_event(
                "RECV",
                "pp_recv_backward",
                None,
                payload_bytes=512,
                peer_rank=next_peer,
                phase="BACKWARD",
            )
        add_event("COMPUTE", "compute_4", 5000, flops=5.0)
        add_event(
            "COLLECTIVE",
            "dp_reduce_scatter_grad",
            None,
            payload_bytes=1024,
            communicator={
                "group_type": "DP_CP",
                "ranks": dp_group,
                "size": len(dp_group),
            },
            phase="BACKWARD",
        )
        add_event("COMPUTE", "compute_5", 6000, flops=6.0)
        if prev_peer is not None:
            add_event(
                "SEND",
                "pp_send_backward",
                None,
                payload_bytes=256,
                peer_rank=prev_peer,
                phase="BACKWARD",
            )

        trace = {
            "schema_version": "synthetic_training_trace/v1",
            "rank": rank,
            "coordinates": coords,
            "events": events,
        }
        (trace_dir / f"trace_rank_{rank:03d}.json").write_text(
            json.dumps(trace, indent=2), encoding="utf-8"
        )
    return trace_dir


class TraceReconfigPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.trace_dir = _make_native_trace_dir(self.root)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_load_and_select_middle_representative(self) -> None:
        bundle = load_trace_bundle(self.trace_dir)
        self.assertEqual(bundle.num_ranks, 12)
        representative = select_representative_rank(bundle, policy="middle")
        self.assertEqual(representative, 4)
        ordered = stable_topological_ops(bundle.ranks[representative])
        self.assertEqual([op.raw_index for op in ordered], list(range(len(ordered))))

    def test_build_comm_nodes_from_trace_rank(self) -> None:
        bundle = load_trace_bundle(self.trace_dir)
        nodes, refs, skipped = build_comm_nodes_from_rank_trace(
            bundle.ranks[4], profile="mixed"
        )
        self.assertEqual(skipped, [])
        self.assertEqual(
            [node.domain for node in nodes], ["pp", "tp", "pp", "pp", "dp", "pp"]
        )
        self.assertEqual(
            [round(node.gap_before_ms, 3) for node in nodes],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        self.assertEqual(refs[0].et_node_id, 2)
        self.assertEqual(refs[-1].et_node_id, 12)

    def test_collect_edge_templates_and_bandwidth_state(self) -> None:
        bundle = load_trace_bundle(self.trace_dir)
        templates = collect_domain_edge_templates(bundle, profile="mixed")
        self.assertIn((4, 5), templates["tp"])
        self.assertIn((5, 4), templates["tp"])
        self.assertIn((0, 4), templates["pp"])
        self.assertIn((4, 8), templates["pp"])
        self.assertIn((4, 0), templates["pp"])
        self.assertIn((4, 6), templates["dp"])

        state = edge_bandwidth_state_for_share(
            templates,
            {"tp": 0.5, "pp": 0.25, "dp": 0.25},
            total_bandwidth_gbps=100.0,
        )
        self.assertAlmostEqual(state[(4, 5)], 50.0)
        self.assertAlmostEqual(state[(4, 8)], 12.5)
        self.assertAlmostEqual(state[(4, 0)], 12.5)
        self.assertAlmostEqual(state[(4, 6)], 25.0)

    def test_abstract_peer_plan_and_instantiation(self) -> None:
        bundle = load_trace_bundle(self.trace_dir)
        nodes, refs, _ = build_comm_nodes_from_rank_trace(
            bundle.ranks[4], profile="mixed"
        )
        from llama3_comm import SystemConfig, solve_best_link_plan_for_bw_segment

        segment = solve_best_link_plan_for_bw_segment(
            nodes,
            0,
            len(nodes) - 1,
            bw_share={"tp": 0.5, "pp": 0.25, "dp": 0.25},
            bw_units=None,
            sys=SystemConfig(
                bandwidth_GBps=100.0,
                latency_us=2.0,
                unit_bw_GBps=0.0,
                asym_min_reverse_units=1,
                reconfig_ms=0.0,
                link_batch_ms=0.0,
                degree_k_total=4,
            ),
        )
        plan = build_abstract_peer_plan(nodes, refs, [segment])
        self.assertEqual(len(plan), 6)
        self.assertEqual(plan[0].active_peers_by_domain["pp"], ("prev:recv",))
        self.assertEqual(plan[1].active_peers_by_domain["tp"], ("next", "prev"))
        self.assertEqual(plan[4].active_peers_by_domain["dp"], ("p1",))

        tp_edges = abstract_peer_edges_for_trace_op(
            bundle.ranks[4].ops[3], profile="mixed"
        )
        self.assertEqual(tp_edges["prev"], {(4, 5)})
        self.assertEqual(tp_edges["next"], {(4, 5)})

        pp_state = instantiate_concrete_edge_state(
            bundle,
            profile="mixed",
            plan_event=plan[0],
            total_bandwidth_gbps=100.0,
        )
        self.assertIn((0, 4), pp_state)
        self.assertNotIn((4, 0), pp_state)

        tp_state = instantiate_concrete_edge_state(
            bundle,
            profile="mixed",
            plan_event=plan[1],
            total_bandwidth_gbps=100.0,
        )
        self.assertAlmostEqual(tp_state[(4, 5)], 50.0)

        dp_state = instantiate_concrete_edge_state(
            bundle,
            profile="mixed",
            plan_event=plan[4],
            total_bandwidth_gbps=100.0,
        )
        self.assertAlmostEqual(dp_state[(4, 6)], 25.0)

    def test_emit_runtime_reconfiguration_events(self) -> None:
        bundle = load_trace_bundle(self.trace_dir)
        _, refs, _ = build_comm_nodes_from_rank_trace(bundle.ranks[4], profile="mixed")
        events = build_dag_reconfiguration_events(
            states=[
                {(4, 5): 50.0, (4, 8): 20.0},
                {(4, 5): 25.0, (4, 0): 10.0},
            ],
            trigger_refs=[refs[0], refs[1]],
            trigger_rank=4,
            trigger_phase="start",
            latency_ns=2000.0,
        )
        self.assertEqual(
            {event["action"]["type"] for event in events},
            {"set-bandwidth", "remove-link", "add-link"},
        )
        self.assertTrue(all(event["node-id"] == refs[1].et_node_id for event in events))
        yaml_text = render_mutable_network_yaml(
            npus_count=12,
            default_bandwidth_gbps=100.0,
            default_latency_ns=2000.0,
            initial_state={(4, 5): 50.0, (4, 8): 20.0},
        )
        self.assertIn("topology: [ Mutable ]", yaml_text)
        self.assertIn("src: 4, dest: 5, bandwidth: 50", yaml_text)

    def test_hypercube_instantiation_preserves_all_stage_edges_for_equivalent_groups(
        self,
    ) -> None:
        ranks = {}
        groups = ([0, 1, 2, 3], [4, 5, 6, 7])
        for group in groups:
            for rank in group:
                op = TraceOp(
                    uid=f"r{rank}_tp0",
                    et_node_id=1,
                    name="tp_all_gather",
                    event_type="COLLECTIVE",
                    phase="FORWARD",
                    rank=rank,
                    coordinates={
                        "tp_rank": group.index(rank),
                        "dp_rank": 0,
                        "pp_rank": 0,
                    },
                    predecessors=(),
                    payload_bytes=256,
                    flops=0.0,
                    duration_us=None,
                    group_type="TP",
                    group_ranks=tuple(group),
                    peer_rank=None,
                    raw_index=0,
                )
                ranks[rank] = TraceRankGraph(
                    rank=rank,
                    coordinates=op.coordinates,
                    ops=(op,),
                    event_id_to_et_node_id={op.uid: op.et_node_id},
                )
        bundle = TraceBundle(metadata={}, ranks=ranks)
        plan_event = AbstractPeerPlanEvent(
            comm_node_idx=0,
            trigger=CommTriggerRef(
                rank=0,
                event_uid="r0_tp0",
                et_node_id=1,
                op_name="tp_all_gather",
                domain="tp",
                raw_index=0,
            ),
            bw_share={"tp": 1.0},
            degree_split={"tp": 1},
            active_peers_by_domain={"tp": ("p1",), "pp": (), "dp": ()},
        )

        state = instantiate_concrete_edge_state(
            bundle,
            profile="hypercube",
            plan_event=plan_event,
            total_bandwidth_gbps=80.0,
        )

        self.assertIn((0, 1), state)
        self.assertIn((0, 2), state)
        self.assertIn((1, 0), state)
        self.assertIn((1, 3), state)
        self.assertIn((4, 5), state)
        self.assertIn((4, 6), state)
        self.assertAlmostEqual(state[(0, 1)], 40.0)
        self.assertAlmostEqual(state[(0, 2)], 40.0)

    def test_cli_end_to_end_outputs_bundle(self) -> None:
        out_dir = self.root / "out"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "trace_reconfig_plan.py"),
            "--trace-dir",
            str(self.trace_dir),
            "--output-dir",
            str(out_dir),
            "--planner",
            "one-shot",
            "--collective-profile",
            "mixed",
            "--bandwidth-gbps",
            "100",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        self.assertIn('"representative_rank": 4', result.stdout)
        self.assertTrue((out_dir / "network_cfg.yml").exists())
        self.assertTrue((out_dir / "system_cfg.json").exists())
        self.assertTrue((out_dir / "plan_summary.json").exists())
        with (out_dir / "plan_summary.json").open("r", encoding="utf-8") as fh:
            summary = json.load(fh)
        self.assertEqual(summary["representative_rank"], 4)
        self.assertEqual(summary["num_comm_nodes"], 6)
        self.assertEqual(summary["num_abstract_peer_events"], 6)
        self.assertIn("abstract_peer_plan", summary)


if __name__ == "__main__":
    unittest.main()
