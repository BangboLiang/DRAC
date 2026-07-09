#!/usr/bin/env python3
"""Trace-driven communication planning and Astra config emission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llama3_comm import (
    SystemConfig,
    build_abstract_peer_plan,
    build_comm_nodes_from_rank_trace,
    build_dynamic_events,
    densify_edge_states,
    instantiate_concrete_edge_state,
    load_trace_bundle,
    plan_segments,
    render_mutable_network_yaml,
    segments_to_summary,
    select_representative_rank,
    write_astra_plan_bundle,
)


def astra_native_collective_impls(profile: str) -> dict[str, list[str]]:
    """Return Astra native collective implementation names for one profile."""
    prof = str(profile).strip().lower()
    if prof in {"ring_asym", "ring_sym"}:
        impl = "ring"
    elif prof == "hypercube":
        impl = "halvingDoubling"
    elif prof == "mixed":
        impl = "ring"
    elif prof == "tree":
        raise ValueError(
            "Astra-sim native collectives do not expose a general 'tree' implementation "
            "for all-gather / reduce-scatter / all-reduce. Supported native names in this "
            "checkout are ring, doubleBinaryTree, halvingDoubling, direct, and variants. "
            "doubleBinaryTree is all-reduce-only, so the repo cannot safely export a native "
            "tree system_cfg.json for profile='tree'."
        )
    else:
        raise ValueError(f"Unknown collective profile: {profile}")

    return {
        "all-reduce-implementation": [impl],
        "all-gather-implementation": [impl],
        "reduce-scatter-implementation": [impl],
        "all-to-all-implementation": ["ring"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan communication reconfiguration from Megatron trace JSON and emit Astra mutable-topology configs"
    )
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--representative-policy",
        default="middle",
        choices=["middle", "first", "last", "only", "rank0"],
    )
    parser.add_argument("--representative-rank", type=int, default=None)
    parser.add_argument(
        "--collective-profile",
        default="mixed",
        choices=["mixed", "ring_asym", "ring_sym", "hypercube", "tree"],
    )
    parser.add_argument(
        "--planner",
        default="preplanned",
        choices=["preplanned", "fast-preplanned", "one-shot", "static"],
    )
    parser.add_argument("--bandwidth-gbps", type=float, default=240.0)
    parser.add_argument("--latency-us", type=float, default=2.0)
    parser.add_argument("--unit-bw-gbps", type=float, default=0.0)
    parser.add_argument("--asym-min-reverse-units", type=int, default=1)
    parser.add_argument("--reconfig-ms", type=float, default=0.0)
    parser.add_argument("--link-batch-ms", type=float, default=0.0)
    parser.add_argument("--degree-k-total", type=int, default=0)
    parser.add_argument("--bw-grid-step", type=float, default=0.01)
    parser.add_argument("--astra-default-latency-ns", type=float, default=None)
    parser.add_argument("--astra-preserve-routes", action="store_true")
    parser.add_argument("--astra-floor-bandwidth-gbps", type=float, default=1e-3)
    parser.add_argument("--rank-scoped-triggers", action="store_true")
    args = parser.parse_args()

    bundle = load_trace_bundle(args.trace_dir)
    representative_rank = (
        int(args.representative_rank)
        if args.representative_rank is not None
        else select_representative_rank(bundle, policy=str(args.representative_policy))
    )
    rank_graph = bundle.ranks[representative_rank]
    nodes, refs, skipped = build_comm_nodes_from_rank_trace(
        rank_graph, profile=str(args.collective_profile)
    )
    if not nodes:
        raise RuntimeError("No TP/PP/DP communication nodes were extracted")

    sys = SystemConfig(
        bandwidth_GBps=float(args.bandwidth_gbps),
        latency_us=float(args.latency_us),
        unit_bw_GBps=float(args.unit_bw_gbps),
        asym_min_reverse_units=int(args.asym_min_reverse_units),
        reconfig_ms=float(args.reconfig_ms),
        link_batch_ms=float(args.link_batch_ms),
        degree_k_total=int(args.degree_k_total),
    )
    segments = plan_segments(
        str(args.planner), nodes, sys, bw_grid_step=float(args.bw_grid_step)
    )
    abstract_plan = build_abstract_peer_plan(nodes, refs, segments)
    states = [
        instantiate_concrete_edge_state(
            bundle,
            profile=str(args.collective_profile),
            plan_event=event,
            total_bandwidth_gbps=float(args.bandwidth_gbps),
        )
        for event in abstract_plan
    ]
    if args.astra_preserve_routes:
        states = densify_edge_states(
            states, floor_bandwidth_gbps=float(args.astra_floor_bandwidth_gbps)
        )

    default_latency_ns = (
        float(args.astra_default_latency_ns)
        if args.astra_default_latency_ns is not None
        else float(args.latency_us) * 1000.0
    )
    initial_state, dag_events, role_templates = build_dynamic_events(
        bundle=bundle,
        profile=str(args.collective_profile),
        representative_rank=representative_rank,
        abstract_plan=abstract_plan,
        states=states,
        default_latency_ns=default_latency_ns,
        rank_scoped_triggers=bool(args.rank_scoped_triggers),
        use_role_templates=(
            bool(args.rank_scoped_triggers) and not bool(args.astra_preserve_routes)
        ),
        planner=str(args.planner),
        sys=sys,
        bw_grid_step=float(args.bw_grid_step),
    )
    network_yaml = render_mutable_network_yaml(
        npus_count=bundle.num_ranks,
        default_bandwidth_gbps=float(args.bandwidth_gbps),
        default_latency_ns=default_latency_ns,
        initial_state=initial_state,
    )
    system_json = {
        "scheduling-policy": "LIFO",
        "endpoint-delay": 10,
        "active-chunks-per-dimension": 2,
        "preferred-dataset-splits": 4,
        **astra_native_collective_impls(str(args.collective_profile)),
        "collective-optimization": "localBWAware",
        "local-mem-bw": 900,
        "boost-mode": 0,
        "dag-reconfiguration-events": dag_events,
    }
    summary_json = {
        "schema_version": "trace_reconfig_plan/v1",
        "trace_dir": str(Path(args.trace_dir).resolve()),
        "representative_rank": representative_rank,
        "representative_coordinates": dict(rank_graph.coordinates),
        "planner": str(args.planner),
        "collective_profile": str(args.collective_profile),
        "network": {
            "bandwidth_gbps": float(args.bandwidth_gbps),
            "latency_us": float(args.latency_us),
            "astra_default_latency_ns": float(default_latency_ns),
            "astra_preserve_routes": bool(args.astra_preserve_routes),
            "astra_floor_bandwidth_gbps": float(args.astra_floor_bandwidth_gbps),
            "rank_scoped_triggers": bool(args.rank_scoped_triggers),
        },
        "num_comm_nodes": len(nodes),
        "num_abstract_peer_events": len(abstract_plan),
        "num_role_templates": len(role_templates),
        "skipped_event_uids": skipped,
        "segments": segments_to_summary(segments, refs),
        "abstract_peer_plan": [
            {
                "comm_node_idx": int(event.comm_node_idx),
                "trigger": {
                    "rank": int(event.trigger.rank),
                    "event_uid": str(event.trigger.event_uid),
                    "et_node_id": int(event.trigger.et_node_id),
                    "op_name": str(event.trigger.op_name),
                },
                "bw_share": dict(event.bw_share),
                "degree_split": dict(event.degree_split),
                "active_peers_by_domain": {
                    domain: list(peers)
                    for domain, peers in sorted(event.active_peers_by_domain.items())
                },
            }
            for event in abstract_plan
        ],
        "role_templates": {
            role: {
                "representative_rank": int(template.representative_rank),
                "num_comm_nodes": len(template.nodes),
                "num_segments": len(template.segments),
            }
            for role, template in sorted(role_templates.items())
        },
        "dag_reconfiguration_event_count": len(dag_events),
    }
    write_astra_plan_bundle(
        output_dir=args.output_dir,
        network_yaml=network_yaml,
        system_json=system_json,
        summary_json=summary_json,
    )
    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir).resolve()),
                "representative_rank": representative_rank,
                "num_comm_nodes": len(nodes),
                "num_abstract_peer_events": len(abstract_plan),
                "num_role_templates": len(role_templates),
                "num_segments": len(segments),
                "dag_reconfiguration_event_count": len(dag_events),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
