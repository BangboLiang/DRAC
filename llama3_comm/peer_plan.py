"""Abstract peer-plan construction and cluster-level instantiation.

Introduced by `trace_reconfig_plan.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple

from .degree import op_peer_stream
from .execution import BWSegmentPlan, CommNode
from .trace_ir import CommTriggerRef, TraceBundle, TraceOp, TraceRankGraph
from .trace_to_comm import (
    build_comm_nodes_from_rank_trace,
    collective_profile_choices,
    infer_collective_op,
)

Edge = Tuple[int, int]


@dataclass(frozen=True)
class AbstractPeerPlanEvent:
    """Abstract link state requested at a communication-node boundary.

    Introduced by `trace_reconfig_plan.py`.
    """

    comm_node_idx: int
    trigger: CommTriggerRef
    bw_share: Dict[str, float]
    degree_split: Dict[str, int]
    active_peers_by_domain: Dict[str, Tuple[str, ...]]


def _coalesced_peer_stream(node: CommNode) -> List[frozenset[str]]:
    base = op_peer_stream(node)
    c = int(node.count)
    if c <= 0:
        return []
    out: List[frozenset[str]] = []
    last: frozenset[str] | None = None
    for _ in range(c):
        for ps in base:
            if last is not None and ps == last:
                continue
            out.append(ps)
            last = ps
    return out


def _feed_first_batch(
    working: Set[str],
    stream: Sequence[frozenset[str]],
    k_dom: int,
) -> Tuple[Set[str], Set[str]]:
    """Return (start_batch_state, final_state) under the greedy batching model.

    `start_batch_state` is the concrete abstract-peer working set needed at node start.
    `final_state` is the state left after serving the full node.

    Introduced by `trace_reconfig_plan.py`.
    """
    if not stream:
        return set(working), set(working)
    if int(k_dom) <= 0:
        merged = set(working)
        for ps in stream:
            merged |= set(ps)
        return set(merged), set(merged)

    cur = set(working)
    if not cur:
        cur = set(stream[0])
        first_batch = set(cur)
        rest = list(stream[1:])
    else:
        first_batch = set(cur)
        rest = list(stream)

    for ps in rest:
        if len(cur.union(ps)) <= int(k_dom):
            cur |= set(ps)
            first_batch = set(cur)
        else:
            break

    final_state = set(working)
    started = bool(final_state)
    for ps in stream:
        if not started:
            final_state = set(ps)
            started = True
            continue
        if len(final_state.union(ps)) <= int(k_dom):
            final_state |= set(ps)
        else:
            final_state = set(ps)
    return first_batch, final_state


def build_abstract_peer_plan(
    comm_nodes: Sequence[CommNode],
    trigger_refs: Sequence[CommTriggerRef],
    segments: Sequence[BWSegmentPlan],
) -> List[AbstractPeerPlanEvent]:
    """Build a node-start abstract peer plan from BW+link segment decisions.

    Introduced by `trace_reconfig_plan.py`.
    """
    by_node: Dict[int, Tuple[Dict[str, float], Dict[str, int], bool]] = {}
    for bw_seg in segments:
        for link_seg in bw_seg.link_segments:
            for idx in range(link_seg.start_idx, link_seg.end_idx + 1):
                by_node[idx] = (
                    dict(bw_seg.bw_share),
                    dict(link_seg.degree_split),
                    idx == link_seg.start_idx,
                )

    current: Dict[str, Set[str]] = {"tp": set(), "pp": set(), "dp": set()}
    plan: List[AbstractPeerPlanEvent] = []
    for idx, node in enumerate(comm_nodes):
        if idx not in by_node:
            continue
        bw_share, degree_split, is_link_start = by_node[idx]
        if is_link_start:
            current = {"tp": set(), "pp": set(), "dp": set()}
        active: Dict[str, Tuple[str, ...]] = {
            dom: tuple(sorted(current[dom])) for dom in ["tp", "pp", "dp"]
        }
        stream = _coalesced_peer_stream(node)
        start_batch, final_state = _feed_first_batch(
            current.get(node.domain, set()),
            stream,
            int(degree_split.get(node.domain, 0)),
        )
        active[str(node.domain)] = tuple(sorted(start_batch))
        current[str(node.domain)] = set(final_state)
        plan.append(
            AbstractPeerPlanEvent(
                comm_node_idx=idx,
                trigger=trigger_refs[idx],
                bw_share=dict(bw_share),
                degree_split=dict(degree_split),
                active_peers_by_domain=active,
            )
        )
    return plan


def _build_comm_node_for_trace_op(op: TraceOp, profile: str) -> CommNode | None:
    domain: str | None
    et = str(op.event_type).upper()
    gt = None if op.group_type is None else str(op.group_type).upper()
    if et in {"SEND", "RECV"}:
        domain = "pp"
    elif et == "COLLECTIVE" and gt == "TP":
        domain = "tp"
    elif et == "COLLECTIVE" and gt in {"DP", "DP_CP"}:
        domain = "dp"
    elif et == "COLLECTIVE" and gt == "PP":
        domain = "pp"
    else:
        return None

    choices = collective_profile_choices(profile)
    if domain == "pp":
        planner_op = "p2p"
        algo, link_type = choices["pp_p2p"]
        nodes_count = 2
    else:
        planner_op = infer_collective_op(op.name)
        if planner_op is None:
            return None
        algo, link_type = choices[f"{domain}_{planner_op}"]
        nodes_count = max(1, len(op.group_ranks))
    return CommNode(
        name=str(op.name),
        domain=domain,
        payload_bytes=float(max(1, int(op.payload_bytes or 0))),
        nodes=int(nodes_count),
        op=planner_op,
        algo=str(algo),
        link_type=str(link_type),
        count=1,
        gap_before_ms=0.0,
    )


def _ring_neighbors(group: Tuple[int, ...], rank: int) -> Dict[str, int]:
    idx = group.index(rank)
    return {
        "prev": int(group[(idx - 1) % len(group)]),
        "next": int(group[(idx + 1) % len(group)]),
    }


def _hypercube_partner(group: Tuple[int, ...], rank: int, label: str) -> int | None:
    if not label.startswith("p"):
        return None
    try:
        stage = int(label[1:])
    except ValueError:
        return None
    idx = group.index(rank)
    partner_idx = idx ^ (1 << (stage - 1))
    if partner_idx < 0 or partner_idx >= len(group):
        return None
    return int(group[partner_idx])


def _tree_neighbor(group: Tuple[int, ...], rank: int, label: str) -> int | None:
    idx = group.index(rank)
    if label == "parent":
        if idx == 0:
            return None
        return int(group[(idx - 1) // 2])
    if label == "c1":
        child = 2 * idx + 1
        return None if child >= len(group) else int(group[child])
    if label == "c2":
        child = 2 * idx + 2
        return None if child >= len(group) else int(group[child])
    return None


def _find_op_by_trigger(
    rank_graph: TraceRankGraph, trigger: CommTriggerRef
) -> TraceOp | None:
    for op in rank_graph.ops:
        if str(op.uid) == str(trigger.event_uid):
            return op
    for op in rank_graph.ops:
        if int(op.et_node_id) == int(trigger.et_node_id):
            return op
    return None


def _op_signature(op: TraceOp, profile: str) -> tuple[str, ...] | None:
    node = _build_comm_node_for_trace_op(op, profile)
    if node is None:
        return None
    if node.domain == "pp":
        event_type = str(op.event_type).upper()
        phase = str(op.phase).upper()
        return (str(node.domain), event_type, phase)
    return (str(node.domain), str(node.op), str(node.algo), str(len(op.group_ranks)))


def _match_equivalent_rank_op(
    rank_graph: TraceRankGraph,
    profile: str,
    trigger_op: TraceOp,
) -> TraceOp | None:
    target_sig = _op_signature(trigger_op, profile)
    if target_sig is None:
        return None
    ordered = sorted(rank_graph.ops, key=lambda op: op.raw_index)
    raw_index = int(trigger_op.raw_index)
    if 0 <= raw_index < len(ordered):
        candidate = ordered[raw_index]
        if _op_signature(candidate, profile) == target_sig:
            return candidate
    for op in ordered:
        if _op_signature(op, profile) == target_sig:
            return op
    return None


def _preserve_full_op_edge_set(algo: str) -> bool:
    algo_norm = str(algo).strip().lower()
    return algo_norm in {"rd", "rh", "recursive_doubling", "rabenseifner", "tree"}


def _selected_edges_for_op(
    op: TraceOp,
    profile: str,
    desired_labels: Set[str],
) -> Set[Edge]:
    label_to_edges = abstract_peer_edges_for_trace_op(op, profile)
    if not label_to_edges:
        return set()
    node = _build_comm_node_for_trace_op(op, profile)
    if node is None:
        return set()
    selected_labels = (
        set(label_to_edges.keys())
        if _preserve_full_op_edge_set(node.algo)
        else set(desired_labels)
    )
    selected_edges: Set[Edge] = set()
    for label in selected_labels:
        selected_edges.update(label_to_edges.get(label, set()))
    return selected_edges


def abstract_peer_edges_for_trace_op(op: TraceOp, profile: str) -> Dict[str, Set[Edge]]:
    """Return concrete directed edges for each abstract peer label of one trace op.

    Introduced by `trace_reconfig_plan.py`.
    """
    node = _build_comm_node_for_trace_op(op, profile)
    if node is None:
        return {}
    labels = sorted({peer for ps in _coalesced_peer_stream(node) for peer in ps})
    out: Dict[str, Set[Edge]] = {}
    algo = str(node.algo).strip().lower()
    rank = int(op.rank)
    group = tuple(int(x) for x in op.group_ranks)
    for label in labels:
        edges: Set[Edge] = set()
        if node.domain == "pp":
            peer = None if op.peer_rank is None else int(op.peer_rank)
            if peer is None:
                out[label] = set()
                continue
            if label.endswith(":recv") or str(op.event_type).upper() == "RECV":
                edges.add((peer, rank))
            else:
                edges.add((rank, peer))
        elif algo == "ring":
            nbrs = _ring_neighbors(group, rank)
            base = label.split(":", 1)[0]
            if base in nbrs:
                edges.add((rank, int(nbrs[base])))
        elif algo in {"rd", "rh", "recursive_doubling", "rabenseifner"}:
            peer = _hypercube_partner(group, rank, label)
            if peer is not None:
                edges.add((rank, peer))
        elif algo == "tree":
            peer = _tree_neighbor(group, rank, label)
            if peer is not None:
                edges.add((rank, peer))
        out[label] = edges
    return out


def instantiate_concrete_edge_state(
    bundle: TraceBundle,
    profile: str,
    plan_event: AbstractPeerPlanEvent,
    *,
    include_domains: Tuple[str, ...] = ("tp", "pp", "dp"),
    total_bandwidth_gbps: float,
) -> Dict[Edge, float]:
    """Instantiate one abstract node-start plan event into concrete edge bandwidths.

    Introduced by `trace_reconfig_plan.py`.
    """
    state: Dict[Edge, float] = {}
    representative_graph = bundle.ranks.get(int(plan_event.trigger.rank))
    if representative_graph is None:
        return state
    trigger_op = _find_op_by_trigger(representative_graph, plan_event.trigger)
    if trigger_op is None:
        return state
    for domain in include_domains:
        desired = set(plan_event.active_peers_by_domain.get(domain, ()))
        if not desired:
            continue
        share = float(plan_event.bw_share.get(domain, 0.0))
        if share <= 0.0:
            continue
        edges_by_src: MutableMapping[int, Set[Edge]] = {}
        for rank_graph in bundle.ranks.values():
            op = _match_equivalent_rank_op(rank_graph, profile, trigger_op)
            if op is None:
                continue
            node = _build_comm_node_for_trace_op(op, profile)
            if node is None or str(node.domain) != domain:
                continue
            selected_edges = _selected_edges_for_op(op, profile, desired)
            if not selected_edges:
                continue
            for edge in selected_edges:
                edges_by_src.setdefault(int(edge[0]), set()).add(edge)
        for src, src_edges in edges_by_src.items():
            _ = src
            per_edge_bw = float(total_bandwidth_gbps) * share / float(len(src_edges))
            for edge in sorted(src_edges):
                state[edge] = float(state.get(edge, 0.0)) + float(per_edge_bw)
    return state
