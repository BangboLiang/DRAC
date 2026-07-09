"""Trace-driven planning IR helpers.

Introduced by `trace_reconfig_plan.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple


@dataclass(frozen=True)
class TraceOp:
    """Normalized rank-local trace event.

    Introduced by `trace_reconfig_plan.py`.
    """

    uid: str
    et_node_id: int
    name: str
    event_type: str
    phase: str
    rank: int
    coordinates: Mapping[str, int]
    predecessors: Tuple[str, ...]
    payload_bytes: int
    flops: float
    duration_us: float | None
    group_type: str | None
    group_ranks: Tuple[int, ...]
    peer_rank: int | None
    raw_index: int


@dataclass(frozen=True)
class TraceRankGraph:
    """Normalized events for one rank.

    Introduced by `trace_reconfig_plan.py`.
    """

    rank: int
    coordinates: Mapping[str, int]
    ops: Tuple[TraceOp, ...]
    event_id_to_et_node_id: Mapping[str, int]

    @property
    def pp_rank(self) -> int:
        return int(self.coordinates.get("pp_rank", 0))

    @property
    def tp_rank(self) -> int:
        return int(self.coordinates.get("tp_rank", 0))

    @property
    def dp_rank(self) -> int:
        return int(self.coordinates.get("dp_rank", 0))

    @property
    def vp_rank(self) -> int:
        return int(self.coordinates.get("virtual_pipeline_chunk_id", 0))


@dataclass(frozen=True)
class TraceBundle:
    """Whole-trace normalized bundle.

    Introduced by `trace_reconfig_plan.py`.
    """

    metadata: Dict[str, Any]
    ranks: Dict[int, TraceRankGraph]

    @property
    def num_ranks(self) -> int:
        return len(self.ranks)

    @property
    def pp_size(self) -> int:
        pp_ranks = {graph.pp_rank for graph in self.ranks.values()}
        return max(pp_ranks) + 1 if pp_ranks else 1


@dataclass(frozen=True)
class CommTriggerRef:
    """Maps a planning node back to a trace/ET node.

    Introduced by `trace_reconfig_plan.py`.
    """

    rank: int
    event_uid: str
    et_node_id: int
    op_name: str
    domain: str
    raw_index: int


def role_key_for_rank(graph: TraceRankGraph, pp_size: int) -> str:
    """Return a coarse PP-role key for representative-rank selection.

    Introduced by `trace_reconfig_plan.py`.
    """
    pp_rank = graph.pp_rank
    if pp_size <= 1:
        pos = "only"
    elif pp_rank == 0:
        pos = "first"
    elif pp_rank == pp_size - 1:
        pos = "last"
    else:
        pos = "middle"
    return f"pp:{pos}"


def sort_ranks_by_coordinates(bundle: TraceBundle) -> List[int]:
    """Return ranks sorted by PP, DP, TP order.

    Introduced by `trace_reconfig_plan.py`.
    """
    return sorted(
        bundle.ranks,
        key=lambda rank: (
            int(bundle.ranks[rank].coordinates.get("pp_rank", 0)),
            int(bundle.ranks[rank].coordinates.get("dp_rank", 0)),
            int(bundle.ranks[rank].coordinates.get("tp_rank", 0)),
            rank,
        ),
    )
