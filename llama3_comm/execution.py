"""Execution-related data classes and helper functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple

from .degree import exposed_boundary_ms, op_peer_stream, PeerSet
from .traffic import estimate_time_ms

if TYPE_CHECKING:
    from .config import SystemConfig


@dataclass(frozen=True)
class CommNode:
    """A domain-tagged communication node in the known schedule (pre-planned strategy)."""

    name: str
    domain: str  # "tp" | "pp" | "dp"
    payload_bytes: float
    nodes: int
    op: str
    algo: str
    link_type: str
    count: int = 1
    gap_before_ms: float = (
        0.0  # compute gap before this comm node (for hiding reconfig)
    )


@dataclass(frozen=True)
class SegmentPlan:
    start_idx: int  # inclusive, 0-based
    end_idx: int  # inclusive, 0-based
    bw_share: Dict[str, float]
    bw_units: Dict[str, int] | None
    comm_time_ms: float
    exposed_reconfig_ms: float


@dataclass(frozen=True)
class LinkSegmentPlan:
    """A link-only segment inside a BW segment (fixed BW split; variable degree split)."""

    start_idx: int  # inclusive, 0-based
    end_idx: int  # inclusive, 0-based
    degree_split: Dict[str, int]  # {"tp":k_tp,"pp":k_pp,"dp":k_dp}
    comm_time_ms: float
    internal_retune_ms: float
    exposed_link_boundary_ms: float  # cost at start (0 if coincides with BW boundary)

    @property
    def total_ms(self) -> float:
        return (
            float(self.exposed_link_boundary_ms)
            + float(self.internal_retune_ms)
            + float(self.comm_time_ms)
        )


@dataclass(frozen=True)
class BWSegmentPlan:
    """A BW segment with a fixed bandwidth split, plus an inner link-only plan."""

    start_idx: int  # inclusive, 0-based
    end_idx: int  # inclusive, 0-based
    bw_share: Dict[str, float]
    bw_units: Dict[str, int] | None
    exposed_bw_boundary_ms: float
    link_segments: List[LinkSegmentPlan]

    @property
    def comm_time_ms(self) -> float:
        return sum(float(ls.comm_time_ms) for ls in self.link_segments)

    @property
    def internal_retune_ms(self) -> float:
        return sum(float(ls.internal_retune_ms) for ls in self.link_segments)

    @property
    def exposed_link_boundaries_ms(self) -> float:
        return sum(float(ls.exposed_link_boundary_ms) for ls in self.link_segments)

    @property
    def total_ms(self) -> float:
        return (
            float(self.exposed_bw_boundary_ms)
            + self.exposed_link_boundaries_ms
            + self.internal_retune_ms
            + self.comm_time_ms
        )


@dataclass(frozen=True)
class TraceEvent:
    """A serialized timeline event for one rank within one iteration."""

    strategy: str  # "preplanned" | "one-shot" | "static"
    kind: str  # "bw_reconfig" | "link_reconfig" | "link_internal" | "comm"
    label: str
    domain: str  # "tp" | "pp" | "dp" | "reconfig"
    start_ms: float
    duration_ms: float
    bw_share: Dict[str, float] | None = None
    bw_units: Dict[str, int] | None = None
    degree_split: Dict[str, int] | None = None

    @property
    def end_ms(self) -> float:
        return self.start_ms + self.duration_ms


def exposed_reconfig_ms_for_segment_start(
    node: CommNode, sys: "SystemConfig", is_first: bool
) -> float:
    """DEPRECATED: Use exposed_boundary_ms(...) with BW/link costs.

    Kept only for backward compatibility with older call sites; it now matches the
    new semantics: including at t=0, boundaries are gap-hidden the same way.
    """
    if sys.reconfig_sec <= 0:
        return 0.0
    _ = is_first  # no special-casing under the new model
    return exposed_boundary_ms(node.gap_before_ms, sys.reconfig_sec * 1000.0)


def _segment_comm_time_ms(
    nodes: List[CommNode],
    bw_share: Dict[str, float],
    sys: "SystemConfig",
) -> float:
    """Compute total communication time for a segment given a fixed bw split."""
    total = 0.0
    for n in nodes:
        b = float(bw_share.get(n.domain, 0.0))
        t = estimate_time_ms(
            n.payload_bytes, n.nodes, b, n.op, n.algo, n.link_type, sys
        )
        total += float(n.count) * t
    return total


def _node_comm_time_ms(
    n: CommNode, bw_share: Dict[str, float], sys: "SystemConfig"
) -> float:
    """Total time (ms) contributed by a CommNode, including its count multiplier."""
    b = float(bw_share.get(n.domain, 0.0))
    t_one = estimate_time_ms(
        n.payload_bytes, n.nodes, b, n.op, n.algo, n.link_type, sys
    )
    return float(n.count) * float(t_one)


def _trace_from_segments(
    strategy: str,
    nodes: List[CommNode],
    sys: "SystemConfig",
    segments: List[BWSegmentPlan],
) -> List[TraceEvent]:
    """Build a serialized comm trace for the given BW+link segmentation."""
    events: List[TraceEvent] = []
    x = 0.0

    def _iter_peer_sets_for_node(n: CommNode) -> List[PeerSet]:
        """Return the coalesced peer-set stream for a node (respecting count)."""
        base = op_peer_stream(n)
        c = int(n.count)
        if c <= 0:
            return []
        out: List[PeerSet] = []
        last: PeerSet | None = None
        for _ in range(c):
            for ps in base:
                if last is not None and ps == last:
                    continue
                out.append(ps)
                last = ps
        return out

    def _apply_greedy(
        working: set[str],
        k_dom: int,
        stream: List[PeerSet],
    ) -> Tuple[set[str], int]:
        """Apply greedy batching; return (new_working_set, overflows)."""
        if k_dom <= 0:
            # Unlimited/ideal; no batching cost.
            return working, 0
        cur = set(working)
        started = len(cur) > 0
        overflows = 0
        for ps in stream:
            if not started:
                cur = set(ps)
                started = True
                continue
            if len(cur.union(ps)) <= k_dom:
                cur |= set(ps)
            else:
                overflows += 1
                cur = set(ps)
        return cur, overflows

    for bw_seg in segments:
        if bw_seg.exposed_bw_boundary_ms > 0:
            events.append(
                TraceEvent(
                    strategy=strategy,
                    kind="bw_reconfig",
                    label="R",
                    domain="reconfig",
                    start_ms=x,
                    duration_ms=float(bw_seg.exposed_bw_boundary_ms),
                    bw_share=dict(bw_seg.bw_share),
                    bw_units=dict(bw_seg.bw_units)
                    if bw_seg.bw_units is not None
                    else None,
                )
            )
            x += float(bw_seg.exposed_bw_boundary_ms)

        for ls in bw_seg.link_segments:
            if ls.exposed_link_boundary_ms > 0:
                events.append(
                    TraceEvent(
                        strategy=strategy,
                        kind="link_reconfig",
                        label="L",
                        domain="reconfig",
                        start_ms=x,
                        duration_ms=float(ls.exposed_link_boundary_ms),
                        bw_share=dict(bw_seg.bw_share),
                        bw_units=dict(bw_seg.bw_units)
                        if bw_seg.bw_units is not None
                        else None,
                        degree_split=dict(ls.degree_split),
                    )
                )
                x += float(ls.exposed_link_boundary_ms)

            seg_nodes = nodes[ls.start_idx : ls.end_idx + 1]
            # Replay greedy batching to place internal retune (L*) events before overflow nodes.
            working_sets: Dict[str, set[str]] = {
                "tp": set(),
                "pp": set(),
                "dp": set(),
            }
            link_internal_ms = float(sys.link_batch_sec) * 1000.0
            for n in seg_nodes:
                domain = str(n.domain)
                k_dom = int(ls.degree_split.get(domain, 0))
                if link_internal_ms > 0 and k_dom > 0:
                    stream = _iter_peer_sets_for_node(n)
                    new_working, overflows = _apply_greedy(
                        working_sets.get(domain, set()), k_dom, stream
                    )
                    if overflows > 0:
                        for _ in range(overflows):
                            events.append(
                                TraceEvent(
                                    strategy=strategy,
                                    kind="link_internal",
                                    label="L*",
                                    domain="reconfig",
                                    start_ms=x,
                                    duration_ms=link_internal_ms,
                                    bw_share=dict(bw_seg.bw_share),
                                    bw_units=dict(bw_seg.bw_units)
                                    if bw_seg.bw_units is not None
                                    else None,
                                    degree_split=dict(ls.degree_split),
                                )
                            )
                            x += float(link_internal_ms)
                    working_sets[domain] = new_working
                else:
                    # Still advance the working set if k_dom is limited but retune is free.
                    if k_dom > 0:
                        stream = _iter_peer_sets_for_node(n)
                        new_working, _overflows = _apply_greedy(
                            working_sets.get(domain, set()), k_dom, stream
                        )
                        working_sets[domain] = new_working

                t_ms = _node_comm_time_ms(n, bw_seg.bw_share, sys)
                events.append(
                    TraceEvent(
                        strategy=strategy,
                        kind="comm",
                        label=n.name,
                        domain=n.domain,
                        start_ms=x,
                        duration_ms=float(t_ms),
                        bw_share=dict(bw_seg.bw_share),
                        bw_units=dict(bw_seg.bw_units)
                        if bw_seg.bw_units is not None
                        else None,
                        degree_split=dict(ls.degree_split),
                    )
                )
                x += float(t_ms)
    return events
