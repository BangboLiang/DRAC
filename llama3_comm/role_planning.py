"""Role-aware planning helpers for trace-driven Astra emission.

Introduced by `trace_reconfig_plan.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

from .astra_emit import (
    build_dag_reconfiguration_events,
    build_rank_scoped_dag_reconfiguration_events,
)
from .config import SystemConfig
from .execution import BWSegmentPlan, CommNode
from .peer_plan import (
    AbstractPeerPlanEvent,
    abstract_peer_edges_for_trace_op,
    build_abstract_peer_plan,
    _build_comm_node_for_trace_op,
    _find_op_by_trigger,
    _match_equivalent_rank_op,
    _selected_edges_for_op,
)
from .solvers import (
    fast_preplanned_partition,
    preplanned_dp_partition,
    solve_best_link_plan_for_bw_segment,
    solve_min_delay_bw_split,
)
from .trace_ingest import select_representative_rank
from .trace_ir import CommTriggerRef, TraceBundle, TraceRankGraph, role_key_for_rank
from .trace_to_comm import build_comm_nodes_from_rank_trace

Edge = Tuple[int, int]


@dataclass(frozen=True)
class RoleTemplatePlan:
    role: str
    representative_rank: int
    nodes: Tuple[CommNode, ...]
    refs: Tuple[CommTriggerRef, ...]
    skipped: Tuple[str, ...]
    segments: Tuple[BWSegmentPlan, ...]
    abstract_plan: Tuple[AbstractPeerPlanEvent, ...]


def equal_share() -> Dict[str, float]:
    return {"tp": 1.0 / 3.0, "pp": 1.0 / 3.0, "dp": 1.0 / 3.0}


def plan_segments(
    planner: str,
    nodes: Sequence[CommNode],
    sys: SystemConfig,
    bw_grid_step: float,
) -> List[BWSegmentPlan]:
    planner_norm = str(planner).strip().lower()
    if planner_norm == "preplanned":
        return preplanned_dp_partition(nodes, sys, bw_grid_step=bw_grid_step)
    if planner_norm == "fast-preplanned":
        return fast_preplanned_partition(nodes, sys, bw_grid_step=bw_grid_step)
    if planner_norm == "one-shot":
        bw, units, _ = solve_min_delay_bw_split(nodes, sys, bw_grid_step=bw_grid_step)
        return [
            solve_best_link_plan_for_bw_segment(
                nodes, 0, len(nodes) - 1, bw_share=bw, bw_units=units, sys=sys
            )
        ]
    if planner_norm == "static":
        bw = equal_share()
        return [
            solve_best_link_plan_for_bw_segment(
                nodes, 0, len(nodes) - 1, bw_share=bw, bw_units=None, sys=sys
            )
        ]
    raise ValueError(f"Unsupported planner: {planner}")


def segments_to_summary(
    segments: Sequence[BWSegmentPlan], refs: Sequence[CommTriggerRef]
) -> List[Dict[str, object]]:
    summary = []
    for idx, seg in enumerate(segments):
        trigger = refs[seg.start_idx]
        summary.append(
            {
                "segment_index": idx,
                "start_idx": int(seg.start_idx),
                "end_idx": int(seg.end_idx),
                "bw_share": dict(seg.bw_share),
                "trigger": {
                    "rank": int(trigger.rank),
                    "event_uid": str(trigger.event_uid),
                    "et_node_id": int(trigger.et_node_id),
                    "op_name": str(trigger.op_name),
                },
                "total_ms": float(seg.total_ms),
                "comm_time_ms": float(seg.comm_time_ms),
                "exposed_bw_boundary_ms": float(seg.exposed_bw_boundary_ms),
                "exposed_link_boundaries_ms": float(seg.exposed_link_boundaries_ms),
                "internal_retune_ms": float(seg.internal_retune_ms),
            }
        )
    return summary


def plan_template_for_rank(
    rank_graph: TraceRankGraph,
    profile: str,
    planner: str,
    sys: SystemConfig,
    bw_grid_step: float,
) -> RoleTemplatePlan:
    nodes, refs, skipped = build_comm_nodes_from_rank_trace(rank_graph, profile=profile)
    segments = plan_segments(planner, nodes, sys, bw_grid_step=bw_grid_step)
    abstract_plan = build_abstract_peer_plan(nodes, refs, segments)
    return RoleTemplatePlan(
        role=role_key_for_rank(rank_graph, 1 + rank_graph.pp_rank),
        representative_rank=int(rank_graph.rank),
        nodes=tuple(nodes),
        refs=tuple(refs),
        skipped=tuple(skipped),
        segments=tuple(segments),
        abstract_plan=tuple(abstract_plan),
    )


def event_signature(node: CommNode, ref: CommTriggerRef):
    if ref.domain != "pp":
        return (str(ref.domain), str(node.op))
    lower = str(ref.op_name).lower()
    direction = "send" if "send" in lower else "recv"
    phase = "backward" if "backward" in lower else "forward"
    return (str(ref.domain), direction, phase)


def previous_ref(rank_graph: TraceRankGraph, ref: CommTriggerRef) -> CommTriggerRef:
    ordered_ops = sorted(rank_graph.ops, key=lambda op: op.raw_index)
    by_et = {op.et_node_id: op for op in ordered_ops}
    op = by_et.get(ref.et_node_id)
    if op is None or op.raw_index == 0:
        return ref
    prev_op = ordered_ops[op.raw_index - 1]
    return CommTriggerRef(
        rank=ref.rank,
        event_uid=str(prev_op.uid),
        et_node_id=int(prev_op.et_node_id),
        op_name=str(prev_op.name),
        domain=ref.domain,
        raw_index=int(prev_op.raw_index),
    )


def align_rank_refs_to_template(
    rank_graph: TraceRankGraph,
    profile: str,
    template_refs: Sequence[CommTriggerRef],
    template_nodes: Sequence[CommNode],
) -> List[CommTriggerRef | None]:
    nodes, refs, _ = build_comm_nodes_from_rank_trace(rank_graph, profile=profile)
    local_sig = [event_signature(node, ref) for node, ref in zip(nodes, refs)]
    template_sig = [
        event_signature(node, ref) for node, ref in zip(template_nodes, template_refs)
    ]
    aligned: List[CommTriggerRef | None] = []
    cursor = 0
    for target_sig in template_sig:
        found = None
        while cursor < len(local_sig):
            if local_sig[cursor] == target_sig:
                found = previous_ref(rank_graph, refs[cursor])
                cursor += 1
                break
            cursor += 1
        aligned.append(found)
    return aligned


def instantiate_rank_edge_state(
    rank_graph: TraceRankGraph,
    profile: str,
    trigger_ref: CommTriggerRef,
    active_peers_by_domain: Mapping[str, Sequence[str]],
    bw_share: Mapping[str, float],
    total_bandwidth_gbps: float,
) -> Dict[Edge, float]:
    state: Dict[Edge, float] = {}
    trigger_op = _find_op_by_trigger(rank_graph, trigger_ref)
    if trigger_op is None:
        return state
    for domain, labels in active_peers_by_domain.items():
        share = float(bw_share.get(domain, 0.0))
        if share <= 0.0 or not labels:
            continue
        op = _match_equivalent_rank_op(rank_graph, profile, trigger_op)
        if op is None:
            continue
        node = _build_comm_node_for_trace_op(op, profile)
        if node is None or str(node.domain) != str(domain):
            continue
        edges = {
            edge
            for edge in _selected_edges_for_op(op, profile, set(labels))
            if edge[0] == rank_graph.rank
        }
        if not edges:
            continue
        per_edge_bw = float(total_bandwidth_gbps) * share / float(len(edges))
        for edge in sorted(edges):
            state[edge] = float(state.get(edge, 0.0)) + float(per_edge_bw)
    return state


def build_role_templates(
    bundle: TraceBundle,
    profile: str,
    planner: str,
    sys: SystemConfig,
    bw_grid_step: float,
) -> Dict[str, RoleTemplatePlan]:
    templates: Dict[str, RoleTemplatePlan] = {}
    for policy in ["first", "middle", "last", "only"]:
        try:
            rank = select_representative_rank(bundle, policy=policy)
        except Exception:
            continue
        role = role_key_for_rank(bundle.ranks[rank], bundle.pp_size)
        if role in templates:
            continue
        plan = plan_template_for_rank(
            bundle.ranks[rank],
            profile=profile,
            planner=planner,
            sys=sys,
            bw_grid_step=bw_grid_step,
        )
        templates[role] = RoleTemplatePlan(
            role=role,
            representative_rank=plan.representative_rank,
            nodes=plan.nodes,
            refs=plan.refs,
            skipped=plan.skipped,
            segments=plan.segments,
            abstract_plan=plan.abstract_plan,
        )
    return templates


def build_rank_trigger_refs(
    bundle: TraceBundle, profile: str
) -> Dict[int, List[CommTriggerRef | None]]:
    representative_rank = select_representative_rank(bundle, policy="middle")
    rep_nodes, rep_refs, _ = build_comm_nodes_from_rank_trace(
        bundle.ranks[representative_rank], profile=profile
    )
    rank_trigger_refs = {}
    for rank, rank_graph in bundle.ranks.items():
        rank_trigger_refs[int(rank)] = align_rank_refs_to_template(
            rank_graph, profile, rep_refs, rep_nodes
        )
    return rank_trigger_refs


def initial_edge_state_from_first_comm(
    bundle: TraceBundle, profile: str, total_bandwidth_gbps: float
) -> Dict[Edge, float]:
    state: Dict[Edge, float] = {}
    for rank_graph in bundle.ranks.values():
        first_comm = None
        for op in sorted(rank_graph.ops, key=lambda op: op.raw_index):
            edges_by_label = abstract_peer_edges_for_trace_op(op, profile)
            if edges_by_label:
                first_comm = edges_by_label
                break
        if not first_comm:
            continue
        edges = sorted({edge for edge_set in first_comm.values() for edge in edge_set})
        if not edges:
            continue
        per_src: Dict[int, List[Edge]] = {}
        for edge in edges:
            per_src.setdefault(edge[0], []).append(edge)
        for src_edges in per_src.values():
            per_edge_bw = float(total_bandwidth_gbps) / float(len(src_edges))
            for edge in src_edges:
                state[edge] = max(float(state.get(edge, 0.0)), float(per_edge_bw))
    return state


def build_role_scoped_events(
    bundle: TraceBundle,
    templates: Mapping[str, RoleTemplatePlan],
    profile: str,
    total_bandwidth_gbps: float,
    latency_ns: float,
) -> Tuple[Dict[Edge, float], List[Dict[str, object]]]:
    events: List[Dict[str, object]] = []
    initial_state: Dict[Edge, float] = {}
    for rank, rank_graph in bundle.ranks.items():
        role = role_key_for_rank(rank_graph, bundle.pp_size)
        template = templates[role]
        aligned_refs = align_rank_refs_to_template(
            rank_graph, profile, template.refs, template.nodes
        )
        states = [
            instantiate_rank_edge_state(
                rank_graph,
                profile,
                aligned_refs[idx]
                if aligned_refs[idx] is not None
                else plan_event.trigger,
                plan_event.active_peers_by_domain,
                plan_event.bw_share,
                total_bandwidth_gbps,
            )
            for idx, plan_event in enumerate(template.abstract_plan)
        ]
        if states:
            initial_state.update(states[0])
        prev = states[0] if states else {}
        for idx in range(1, len(states)):
            new = states[idx]
            trigger = aligned_refs[idx]
            if trigger is None:
                prev = new
                continue
            for edge in sorted(set(prev) | set(new)):
                old_bw = float(prev.get(edge, 0.0))
                new_bw = float(new.get(edge, 0.0))
                action = None
                if old_bw <= 1e-9 and new_bw > 1e-9:
                    action = {
                        "type": "add-link",
                        "src": int(edge[0]),
                        "dest": int(edge[1]),
                        "bandwidth": float(new_bw),
                        "latency": float(latency_ns),
                    }
                elif old_bw > 1e-9 and new_bw <= 1e-9:
                    action = {
                        "type": "remove-link",
                        "src": int(edge[0]),
                        "dest": int(edge[1]),
                    }
                elif abs(old_bw - new_bw) > 1e-9:
                    action = {
                        "type": "set-bandwidth",
                        "src": int(edge[0]),
                        "dest": int(edge[1]),
                        "bandwidth": float(new_bw),
                    }
                if action is not None:
                    events.append(
                        {
                            "rank": int(rank),
                            "node-id": int(trigger.et_node_id),
                            "phase": "finish",
                            "action": action,
                        }
                    )
            prev = new
    return initial_state, events


def build_dynamic_events(
    *,
    bundle: TraceBundle,
    profile: str,
    representative_rank: int,
    abstract_plan: Sequence[AbstractPeerPlanEvent],
    states: List[Mapping[Edge, float]],
    default_latency_ns: float,
    rank_scoped_triggers: bool,
    use_role_templates: bool,
    planner: str,
    sys: SystemConfig,
    bw_grid_step: float,
) -> Tuple[Dict[Edge, float], List[Dict[str, object]], Dict[str, RoleTemplatePlan]]:
    role_templates: Dict[str, RoleTemplatePlan] = {}
    if rank_scoped_triggers and use_role_templates:
        role_templates = build_role_templates(
            bundle, profile=profile, planner=planner, sys=sys, bw_grid_step=bw_grid_step
        )
        initial_state, dag_events = build_role_scoped_events(
            bundle,
            role_templates,
            profile=profile,
            total_bandwidth_gbps=float(sys.bandwidth_GBps),
            latency_ns=default_latency_ns,
        )
        return initial_state, dag_events, role_templates

    initial_state = {
        **initial_edge_state_from_first_comm(
            bundle, profile, float(sys.bandwidth_GBps)
        ),
        **(states[0] if states else {}),
    }
    event_states = [initial_state] + states[1:] if states else []
    fallback_trigger_refs = [event.trigger for event in abstract_plan]
    if rank_scoped_triggers:
        dag_events = build_rank_scoped_dag_reconfiguration_events(
            states=event_states,
            rank_trigger_refs=build_rank_trigger_refs(bundle, profile),
            fallback_trigger_refs=fallback_trigger_refs,
            trigger_phase="finish",
            latency_ns=default_latency_ns,
        )
    else:
        dag_events = build_dag_reconfiguration_events(
            states=event_states,
            trigger_refs=fallback_trigger_refs,
            trigger_rank=representative_rank,
            trigger_phase="start",
            latency_ns=default_latency_ns,
        )
    return initial_state, dag_events, role_templates
