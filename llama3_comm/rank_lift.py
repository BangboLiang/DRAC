"""Lift representative planning decisions into all-rank edge states.

Introduced by `trace_reconfig_plan.py`.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, MutableMapping, Set, Tuple

from .trace_ir import TraceBundle, TraceOp
from .trace_to_comm import collective_profile_choices, infer_collective_op

Edge = Tuple[int, int]


def _cyclic_ring_edges(group: Tuple[int, ...]) -> Set[Edge]:
    if len(group) <= 1:
        return set()
    edges = set()
    for idx, src in enumerate(group):
        dst = group[(idx + 1) % len(group)]
        edges.add((int(src), int(dst)))
    return edges


def _hypercube_edges(group: Tuple[int, ...]) -> Set[Edge]:
    edges = set()
    size = len(group)
    if size <= 1:
        return edges
    local = {rank: idx for idx, rank in enumerate(group)}
    for src in group:
        idx = local[src]
        step = 1
        while step < size:
            partner_idx = idx ^ step
            if partner_idx < size:
                edges.add((int(src), int(group[partner_idx])))
            step <<= 1
    return edges


def _tree_edges(group: Tuple[int, ...]) -> Set[Edge]:
    edges = set()
    for idx, parent in enumerate(group):
        left = 2 * idx + 1
        right = 2 * idx + 2
        if left < len(group):
            child = int(group[left])
            edges.add((int(parent), child))
            edges.add((child, int(parent)))
        if right < len(group):
            child = int(group[right])
            edges.add((int(parent), child))
            edges.add((child, int(parent)))
    return edges


def _edges_for_collective(group: Tuple[int, ...], algo: str) -> Set[Edge]:
    algo_norm = str(algo).strip().lower()
    if algo_norm in {"ring", "p2p"}:
        return _cyclic_ring_edges(group)
    if algo_norm in {"rd", "rh", "recursive_doubling", "rabenseifner"}:
        return _hypercube_edges(group)
    if algo_norm == "tree":
        return _tree_edges(group)
    return _hypercube_edges(group)


def _domain_for_op(op: TraceOp) -> str | None:
    et = str(op.event_type).upper()
    gt = None if op.group_type is None else str(op.group_type).upper()
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


def collect_domain_edge_templates(
    bundle: TraceBundle,
    profile: str,
    *,
    include_domains: Tuple[str, ...] = ("tp", "pp", "dp"),
) -> Dict[str, Set[Edge]]:
    """Collect the union of directed logical links used by each domain.

    Introduced by `trace_reconfig_plan.py`.
    """
    choices = collective_profile_choices(profile)
    out: Dict[str, Set[Edge]] = {d: set() for d in include_domains}
    seen_collectives: Set[Tuple[str, str, Tuple[int, ...]]] = set()
    seen_p2p: Set[Edge] = set()
    for rank_graph in bundle.ranks.values():
        for op in rank_graph.ops:
            domain = _domain_for_op(op)
            if domain is None or domain not in include_domains:
                continue
            if domain == "pp":
                if op.peer_rank is not None:
                    if str(op.event_type).upper() == "SEND":
                        seen_p2p.add((int(op.rank), int(op.peer_rank)))
                    else:
                        seen_p2p.add((int(op.peer_rank), int(op.rank)))
                continue
            planner_op = infer_collective_op(op.name)
            if planner_op is None:
                continue
            choice_key = f"{domain}_{planner_op}"
            algo = choices[choice_key][0]
            key = (domain, str(algo), tuple(op.group_ranks))
            if key in seen_collectives:
                continue
            seen_collectives.add(key)
            out[domain].update(_edges_for_collective(tuple(op.group_ranks), str(algo)))
    out["pp"].update(seen_p2p)
    return out


def edge_bandwidth_state_for_share(
    edge_templates: Mapping[str, Set[Edge]],
    bw_share: Mapping[str, float],
    total_bandwidth_gbps: float,
) -> Dict[Edge, float]:
    """Translate a domain bandwidth split into directed-link bandwidths.

    Introduced by `trace_reconfig_plan.py`.
    """
    state: Dict[Edge, float] = defaultdict(float)
    for domain, edges in edge_templates.items():
        share = float(bw_share.get(domain, 0.0))
        if share <= 0.0 or not edges:
            continue
        per_src: MutableMapping[int, List[Edge]] = defaultdict(list)
        for edge in sorted(edges):
            per_src[int(edge[0])].append(edge)
        for src, src_edges in per_src.items():
            _ = src
            per_edge_bw = float(total_bandwidth_gbps) * share / float(len(src_edges))
            for edge in src_edges:
                state[edge] = float(state.get(edge, 0.0)) + float(per_edge_bw)
    return dict(state)
