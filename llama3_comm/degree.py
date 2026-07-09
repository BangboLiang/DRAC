"""Degree/K and link retune modeling functions."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Tuple

from .traffic import _ceil_log2

if TYPE_CHECKING:
    from .execution import CommNode

PeerSet = frozenset[str]


def _pp_peer_from_name(label: str, link_type: str) -> str:
    """Infer PP neighbor direction from the comm-node label.

    For asymmetric links, encode direction to model the need for separate peer
    circuits (recv vs send) when switching between FWD and BWD.
    """
    name = str(label).lower()
    is_fwd = ("fwd" in name) or ("forward" in name)
    is_bwd = ("bwd" in name) or ("backward" in name)
    is_recv = "recv" in name
    is_send = "send" in name
    direction = None
    if is_recv:
        direction = "recv"
    elif is_send:
        direction = "send"

    if is_fwd and is_recv:
        peer = "prev"
    elif is_fwd and is_send:
        peer = "next"
    elif is_bwd and is_recv:
        peer = "next"
    elif is_bwd and is_send:
        peer = "prev"
    else:
        peer = "peer"

    if str(link_type).strip().lower() == "asymmetric" and direction is not None:
        # For asymmetric links, treat send/recv as distinct logical resources.
        return f"{peer}:{direction}"
    return peer


def op_peer_stream(n: "CommNode") -> List[PeerSet]:
    """Return the ordered peer-set request stream for a comm node (one 'call' of that node).

    The stream is domain-agnostic; degree feasibility and batching are handled elsewhere.

    Conventions (synthetic partner IDs):
    - Ring: {prev,next}
    - RD/RH: {p1},{p2},...,{pm} where m=ceil(log2(nodes))
    - Rabenseifner: {p1},...,{pm},{pm},...,{p1}
    - P2P: {peer} (PP uses {prev}/{next} inferred from the label when available)
    - Tree (binary): {parent},{c1},{c2} (bounded partner set; degree model treats as distinct peers)
    """
    algo = str(n.algo).strip().lower()
    op = str(n.op).strip().lower()

    if op == "p2p" or algo == "p2p":
        if n.domain == "pp":
            return [frozenset({_pp_peer_from_name(n.name, n.link_type)})]
        return [frozenset({"peer"})]

    if algo == "ring":
        # Simultaneous neighbors.
        return [frozenset({"prev", "next"})]

    if algo == "tree":
        # Binary tree needs Parent + 2 Children simultaneously for pipelining.
        # Enforce degree constraint K>=3 by grouping them in one set.
        return [frozenset({"parent", "c1", "c2"})]

    if algo in ["rd", "rh", "recursive_doubling", "rd_allreduce"]:
        m = _ceil_log2(int(n.nodes))
        return [frozenset({f"p{s}"}) for s in range(1, m + 1)]

    if algo == "rabenseifner":
        m = _ceil_log2(int(n.nodes))
        fwd = [frozenset({f"p{s}"}) for s in range(1, m + 1)]
        return fwd + list(reversed(fwd))

    raise ValueError(f"Unsupported algo for peer stream: op={op} algo={algo}")


def build_domain_stream(
    comm_nodes: List["CommNode"], i: int, j: int, domain: str
) -> List[PeerSet]:
    """Build the domain-filtered peer-set stream for interval [i..j], with optional coalescing."""
    out: List[PeerSet] = []
    last: PeerSet | None = None
    dom = str(domain)
    for t in range(i, j + 1):
        n = comm_nodes[t]
        if n.domain != dom:
            continue
        base = op_peer_stream(n)
        c = int(n.count)
        if c <= 0:
            continue
        # Repeat node's op-stream count times, collapsing identical adjacent sets.
        for _ in range(c):
            for ps in base:
                if last is not None and ps == last:
                    continue
                out.append(ps)
                last = ps
    return out


def calc_batches(stream: List[PeerSet], k_dom: int) -> int:
    """Greedy batcher (optimal under the batch model).

    Returns number of configured batches needed to serve the stream with degree k_dom.
    If infeasible (any peer_set size > k_dom), returns a large sentinel (inf-like).
    """
    k = int(k_dom)
    if k <= 0:
        return 0 if len(stream) == 0 else 10**18

    if not stream:
        return 0

    batches = 1
    working: set[str] = set()
    for ps in stream:
        if len(ps) > k:
            return 10**18
        if len(working.union(ps)) <= k:
            working |= set(ps)
        else:
            batches += 1
            working = set(ps)
    return batches


def critical_degrees(stream: List[PeerSet], k_total: int) -> List[int]:
    """Return the 'critical' k values worth enumerating: where batches(k) strictly decreases."""
    K = int(k_total)
    if K <= 0:
        return []
    if not stream:
        return [0]

    min_k = max(1, max((len(ps) for ps in stream), default=1))
    if min_k > K:
        return []

    out: List[int] = []
    prev_b: int | None = None
    for k in range(min_k, K + 1):
        b = calc_batches(stream, k)
        if prev_b is None or b < prev_b:
            out.append(k)
            prev_b = b
    return out


def exposed_boundary_ms(gap_before_ms: float, boundary_ms: float) -> float:
    """Gap hiding rule for any boundary (including at t=0)."""
    gap = max(0.0, float(gap_before_ms))
    return max(0.0, float(boundary_ms) - gap)


def _calc_batches_interval_fast(
    comm_nodes: List["CommNode"],
    start_idx: int,
    end_idx: int,
    domain: str,
    k_dom: int,
) -> int:
    """Compute calc_batches(stream_dom(start,end), k_dom) without materializing the stream.

    This is critical for RD/RH, where the stream is a sequence of distinct singletons and
    comm_nodes often have large 'count' multipliers. Materializing repeats can explode.

    Notes:
    - This function preserves the semantics of the greedy batcher from calc_batches(), including
      carry-over of the current working_set across consecutive comm nodes in the interval.
    - It *does* treat identical peer-sets repeated many times (e.g., ring/p2p) efficiently.
    """
    k = int(k_dom)
    if k <= 0:
        # If the domain is active, infeasible; if no requests, return 0.
        active = any(
            (start_idx <= t <= end_idx) and (comm_nodes[t].domain == domain)
            for t in range(start_idx, end_idx + 1)
        )
        return 0 if not active else 10**18

    batches = 0
    working: set[str] = set()
    started = False

    def _feed_peer_set(ps: PeerSet) -> None:
        nonlocal batches, working, started
        if len(ps) > k:
            batches = 10**18
            return
        if not started:
            started = True
            batches = 1
            working = set(ps)
            return
        if len(working.union(ps)) <= k:
            working |= set(ps)
        else:
            batches += 1
            working = set(ps)

    def _feed_singletons_cycle(m: int, reps: int) -> None:
        """Feed the singleton sequence p1..pm repeated reps times, with an O(m) + O(1) fast-path."""
        nonlocal batches, working, started
        if reps <= 0:
            return
        if batches >= 10**18:
            return

        # Simulate ONE cycle to correctly account for current working_set carry-in.
        for s in range(1, m + 1):
            _feed_peer_set(frozenset({f"p{s}"}))
            if batches >= 10**18:
                return

        if reps <= 1:
            return

        if k >= m:
            # After one cycle, working_set is subset of {p1..pm} and fits; repeats add no new batches.
            return

        # For k < m, each additional cycle requires ceil(m/k) new batches.
        add_per_cycle = int(math.ceil(m / k))
        batches += (reps - 1) * add_per_cycle

        # End working_set after any full cycle: last min(k,m) peers in order.
        last = list(range(max(1, m - k + 1), m + 1))
        working = {f"p{s}" for s in last}
        started = True

    def _feed_node_peer_stream(n: "CommNode") -> None:
        """Feed the node's peer-set stream using standard greedy batching."""
        c = int(n.count)
        if c <= 0:
            return
        base = op_peer_stream(n)
        last_local: PeerSet | None = None
        for _ in range(c):
            for ps in base:
                if last_local is not None and ps == last_local:
                    continue
                _feed_peer_set(ps)
                if batches >= 10**18:
                    return
                last_local = ps

    for t in range(start_idx, end_idx + 1):
        n = comm_nodes[t]
        if n.domain != domain:
            continue

        algo = str(n.algo).strip().lower()

        if algo == "tree":
            # Short bounded peer stream; cheap to feed explicitly.
            _feed_node_peer_stream(n)
            continue

        if algo in ["rd", "rh", "recursive_doubling", "rd_allreduce"]:
            m = _ceil_log2(int(n.nodes))
            _feed_singletons_cycle(m, int(n.count))
            continue

        if algo == "rabenseifner":
            # For now, fall back to explicit per-stage feeding (2m is tiny).
            m = _ceil_log2(int(n.nodes))
            seq = list(range(1, m + 1)) + list(range(m, 0, -1))
            for _ in range(int(n.count)):
                for s in seq:
                    _feed_peer_set(frozenset({f"p{s}"}))
                    if batches >= 10**18:
                        break
                if batches >= 10**18:
                    break
            continue

        # Default: feed standard peer stream (ring/p2p/etc).
        _feed_node_peer_stream(n)
        continue

    return 0 if not started else batches
