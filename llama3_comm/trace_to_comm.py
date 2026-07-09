"""Convert normalized traces into planner comm-node sequences.

Introduced by `trace_reconfig_plan.py`.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .execution import CommNode
from .trace_ingest import stable_topological_ops
from .trace_ir import CommTriggerRef, TraceRankGraph


def collective_profile_choices(profile: str) -> Dict[str, Tuple[str, str]]:
    """Return algo/link-type choices for trace-driven planning.

    Introduced by `trace_reconfig_plan.py`.
    """
    prof = str(profile).strip().lower()
    if prof == "mixed":
        return {
            "tp_allgather": ("ring", "asymmetric"),
            "tp_reducescatter": ("ring", "asymmetric"),
            "tp_allreduce": ("ring", "asymmetric"),
            "pp_p2p": ("p2p", "asymmetric"),
            "dp_reducescatter": ("rh", "symmetric"),
            "dp_allgather": ("rd", "symmetric"),
            "dp_allreduce": ("recursive_doubling", "symmetric"),
        }
    if prof == "ring_asym":
        return {
            "tp_allgather": ("ring", "asymmetric"),
            "tp_reducescatter": ("ring", "asymmetric"),
            "tp_allreduce": ("ring", "asymmetric"),
            "pp_p2p": ("p2p", "asymmetric"),
            "dp_reducescatter": ("ring", "asymmetric"),
            "dp_allgather": ("ring", "asymmetric"),
            "dp_allreduce": ("ring", "asymmetric"),
        }
    if prof == "ring_sym":
        return {
            "tp_allgather": ("ring", "symmetric"),
            "tp_reducescatter": ("ring", "symmetric"),
            "tp_allreduce": ("ring", "symmetric"),
            "pp_p2p": ("p2p", "symmetric"),
            "dp_reducescatter": ("ring", "symmetric"),
            "dp_allgather": ("ring", "symmetric"),
            "dp_allreduce": ("ring", "symmetric"),
        }
    if prof == "hypercube":
        return {
            "tp_allgather": ("rd", "symmetric"),
            "tp_reducescatter": ("rh", "symmetric"),
            "tp_allreduce": ("recursive_doubling", "symmetric"),
            "pp_p2p": ("p2p", "symmetric"),
            "dp_reducescatter": ("rh", "symmetric"),
            "dp_allgather": ("rd", "symmetric"),
            "dp_allreduce": ("recursive_doubling", "symmetric"),
        }
    if prof == "tree":
        return {
            "tp_allgather": ("tree", "symmetric"),
            "tp_reducescatter": ("tree", "symmetric"),
            "tp_allreduce": ("tree", "symmetric"),
            "pp_p2p": ("p2p", "symmetric"),
            "dp_reducescatter": ("tree", "symmetric"),
            "dp_allgather": ("tree", "symmetric"),
            "dp_allreduce": ("tree", "symmetric"),
        }
    raise ValueError(f"Unknown collective profile: {profile}")


def infer_collective_op(op_name: str) -> str | None:
    """Infer the communication operation from an op name.

    Introduced by `trace_reconfig_plan.py`.
    """
    lower = str(op_name).lower()
    if "all_gather" in lower or "allgather" in lower:
        return "allgather"
    if "reduce_scatter" in lower or "reducescatter" in lower:
        return "reducescatter"
    if "all_reduce" in lower or "allreduce" in lower:
        return "allreduce"
    if "broadcast" in lower:
        return "allgather"
    return None


def _event_duration_ms(duration_us: float | None) -> float:
    if duration_us is None:
        return 0.0
    return max(0.0, float(duration_us) / 1000.0)


def _domain_for_event(event_type: str, group_type: str | None) -> str | None:
    et = str(event_type).upper()
    gt = None if group_type is None else str(group_type).upper()
    if et in {"SEND", "RECV"}:
        return "pp"
    if et != "COLLECTIVE":
        return None
    if gt == "TP":
        return "tp"
    if gt in {"DP", "DP_CP"}:
        return "dp"
    if gt == "PP":
        return "pp"
    return None


def build_comm_nodes_from_rank_trace(
    rank_graph: TraceRankGraph,
    profile: str,
    *,
    include_domains: Tuple[str, ...] = ("tp", "pp", "dp"),
) -> Tuple[List[CommNode], List[CommTriggerRef], List[str]]:
    """Convert one rank-local DAG into a serialized comm-node stream.

    Introduced by `trace_reconfig_plan.py`.
    """
    choices = collective_profile_choices(profile)
    ordered = stable_topological_ops(rank_graph)
    nodes: List[CommNode] = []
    refs: List[CommTriggerRef] = []
    skipped: List[str] = []
    pending_gap_ms = 0.0

    for op in ordered:
        domain = _domain_for_event(op.event_type, op.group_type)
        if domain is None:
            pending_gap_ms += _event_duration_ms(op.duration_us)
            continue
        if domain not in include_domains:
            skipped.append(op.uid)
            pending_gap_ms += _event_duration_ms(op.duration_us)
            continue

        if domain == "pp":
            planner_op = "p2p"
            algo, link_type = choices["pp_p2p"]
            nodes_count = 2
        else:
            planner_op = infer_collective_op(op.name)
            if planner_op is None:
                skipped.append(op.uid)
                pending_gap_ms += _event_duration_ms(op.duration_us)
                continue
            key = f"{domain}_{planner_op}"
            algo, link_type = choices[key]
            nodes_count = max(1, len(op.group_ranks))

        payload_bytes = max(1, int(op.payload_bytes or 0))
        label = f"ET{op.et_node_id}:{op.name}"
        nodes.append(
            CommNode(
                name=label,
                domain=domain,
                payload_bytes=float(payload_bytes),
                nodes=int(nodes_count),
                op=planner_op,
                algo=str(algo),
                link_type=str(link_type),
                count=1,
                gap_before_ms=float(pending_gap_ms),
            )
        )
        refs.append(
            CommTriggerRef(
                rank=rank_graph.rank,
                event_uid=op.uid,
                et_node_id=op.et_node_id,
                op_name=op.name,
                domain=domain,
                raw_index=op.raw_index,
            )
        )
        pending_gap_ms = 0.0
    return nodes, refs, skipped
