"""Solver functions for bandwidth splits and DP partition planning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple

from .degree import (
    _calc_batches_interval_fast,
    exposed_boundary_ms,
    op_peer_stream,
    PeerSet,
)
from .execution import (
    BWSegmentPlan,
    CommNode,
    LinkSegmentPlan,
    TraceEvent,
    _node_comm_time_ms,
    _trace_from_segments,
)
from .traffic import _ceil_log2, estimate_time_ms

if TYPE_CHECKING:
    from .config import SystemConfig


@dataclass(frozen=True)
class _FastBlock:
    start_idx: int
    end_idx: int
    key: str


# NOTE: Overlap-aware objective helpers
# Introduced by llama3_modular.py DP bucket/overlap approximation (scheme-2).
# These helpers affect only the solver objective (BW/link partition) and keep
# trace generation serialized for readability.
#
# The behavior is gated by sys.objective_dp_gap_overlap to preserve legacy results.


def _node_objective_time_ms(
    n: CommNode, bw_share: Dict[str, float], sys: "SystemConfig"
) -> float:
    """Per-node time used by the solver objective (ms).

    Convention (approximation):
    - TP/PP comm is blocking.
    - DP comm can be partially hidden by compute modeled as n.gap_before_ms:
        exposed_dp_ms = max(0, dp_comm_ms - gap_before_ms)

    We intentionally do not model a running backlog here; that would require an
    extra DP state dimension. This is the simpler additive approximation.
    """

    b = float(bw_share.get(str(n.domain), 0.0))
    t_one = float(
        estimate_time_ms(
            n.payload_bytes, n.nodes, b, str(n.op), str(n.algo), str(n.link_type), sys
        )
    )

    c = int(n.count)
    if c <= 0:
        return 0.0

    if str(n.domain) == "dp" and bool(getattr(sys, "objective_dp_gap_overlap", False)):
        g = max(0.0, float(n.gap_before_ms))
        return float(c) * max(0.0, float(t_one) - g)

    return float(c) * float(t_one)


def _segment_objective_time_ms(
    nodes: List[CommNode], bw_share: Dict[str, float], sys: "SystemConfig"
) -> float:
    return float(sum(_node_objective_time_ms(n, bw_share, sys) for n in nodes))


def _node_continuous_terms_ms(n: CommNode, sys: "SystemConfig") -> Tuple[float, float]:
    """Return a continuous surrogate decomposition time ~= const + coeff / bw_share.

    This is used only by the fast pre-planned approximation. It deliberately ignores
    bandwidth quantization and uses the continuous fluid model even when the exact
    solver is in lane mode; the exact per-segment solve is run after the approximate
    partition is chosen.
    """
    if int(n.nodes) <= 1:
        return 0.0, 0.0

    setup_sec = 0.0
    pattern = (
        "ring"
        if str(n.algo) == "ring"
        else (
            "tree"
            if str(n.algo) == "tree"
            else ("p2p" if str(n.op) == "p2p" else "1peer")
        )
    )
    eta = 1.0
    if pattern == "ring" or pattern == "p2p" or pattern == "tree":
        eta = 1.0 if str(n.link_type) == "asymmetric" else 0.5

    bw_base = max(1.0, float(sys.bw_bytes_sec))
    op = str(n.op)
    algo = str(n.algo)
    nodes = int(n.nodes)
    M = float(n.payload_bytes)
    lat_ms = 0.0
    coeff_ms = 0.0

    if op == "p2p":
        lat_ms = (1.0 * float(sys.latency_sec) + setup_sec) * 1000.0
        coeff_ms = (M / (eta * bw_base)) * 1000.0
    elif op == "allgather":
        if algo == "ring":
            lat_ms = ((nodes - 1) * float(sys.latency_sec) + setup_sec) * 1000.0
            coeff_ms = (M * (nodes - 1) / nodes / (eta * bw_base)) * 1000.0
        elif algo == "rd":
            lat_ms = (_ceil_log2(nodes) * float(sys.latency_sec) + setup_sec) * 1000.0
            coeff_ms = (M * (nodes - 1) / nodes / (eta * bw_base)) * 1000.0
        elif algo == "tree":
            lat_ms = (
                2 * _ceil_log2(nodes) * float(sys.latency_sec) + setup_sec
            ) * 1000.0
            coeff_ms = (2.0 * M / (eta * bw_base)) * 1000.0
        else:
            t = estimate_time_ms(M, nodes, 1.0, op, algo, str(n.link_type), sys)
            return 0.0, float(t)
    elif op == "reducescatter":
        if algo == "ring":
            lat_ms = ((nodes - 1) * float(sys.latency_sec) + setup_sec) * 1000.0
            coeff_ms = (M * (nodes - 1) / nodes / (eta * bw_base)) * 1000.0
        elif algo == "rh":
            lat_ms = (_ceil_log2(nodes) * float(sys.latency_sec) + setup_sec) * 1000.0
            coeff_ms = (M * (nodes - 1) / nodes / (eta * bw_base)) * 1000.0
        elif algo == "tree":
            lat_ms = (
                2 * _ceil_log2(nodes) * float(sys.latency_sec) + setup_sec
            ) * 1000.0
            coeff_ms = (2.0 * M / (eta * bw_base)) * 1000.0
        else:
            t = estimate_time_ms(M, nodes, 1.0, op, algo, str(n.link_type), sys)
            return 0.0, float(t)
    elif op == "allreduce":
        if algo == "ring":
            lat_ms = (2 * (nodes - 1) * float(sys.latency_sec) + setup_sec) * 1000.0
            coeff_ms = (2 * M * (nodes - 1) / nodes / (eta * bw_base)) * 1000.0
        elif algo == "rabenseifner":
            lat_ms = (
                2 * _ceil_log2(nodes) * float(sys.latency_sec) + setup_sec
            ) * 1000.0
            coeff_ms = (2 * M * (nodes - 1) / nodes / (eta * bw_base)) * 1000.0
        elif algo in ["recursive_doubling", "rd_allreduce"]:
            steps = _ceil_log2(nodes)
            lat_ms = (steps * float(sys.latency_sec) + setup_sec) * 1000.0
            coeff_ms = (steps * M / (eta * bw_base)) * 1000.0
        elif algo == "tree":
            lat_ms = (
                2 * _ceil_log2(nodes) * float(sys.latency_sec) + setup_sec
            ) * 1000.0
            coeff_ms = (2.0 * M / (eta * bw_base)) * 1000.0
        else:
            t = estimate_time_ms(M, nodes, 1.0, op, algo, str(n.link_type), sys)
            return 0.0, float(t)
    else:
        t = estimate_time_ms(M, nodes, 1.0, op, algo, str(n.link_type), sys)
        return 0.0, float(t)

    return float(n.count) * lat_ms, float(n.count) * coeff_ms


def _fast_block_key(n: CommNode) -> str:
    parts = str(n.name).split(":")
    if len(parts) >= 3 and parts[1].startswith("L"):
        return ":".join(parts[:3])
    if len(parts) >= 3 and parts[1] == "PP":
        return ":".join(parts[:3])
    if len(parts) >= 2 and parts[0] == "OPT":
        return ":".join(parts[:2])
    if len(parts) >= 2:
        return ":".join(parts[:2])
    return str(n.name)


def _build_fast_blocks(comm_nodes: List[CommNode]) -> List[_FastBlock]:
    if not comm_nodes:
        return []
    blocks: List[_FastBlock] = []
    cur_key = _fast_block_key(comm_nodes[0])
    start = 0
    for idx in range(1, len(comm_nodes)):
        key = _fast_block_key(comm_nodes[idx])
        if key != cur_key:
            blocks.append(_FastBlock(start_idx=start, end_idx=idx - 1, key=cur_key))
            start = idx
            cur_key = key
    blocks.append(_FastBlock(start_idx=start, end_idx=len(comm_nodes) - 1, key=cur_key))
    return blocks


def _approx_interval_objective_ms(
    consts: Dict[str, float],
    coeffs: Dict[str, float],
    gaps: Dict[str, float],
    bw_share: Dict[str, float],
    sys: "SystemConfig",
) -> float:
    total = 0.0
    for dom in ["tp", "pp", "dp"]:
        coeff = float(coeffs.get(dom, 0.0))
        const = float(consts.get(dom, 0.0))
        if coeff <= 0.0 and const <= 0.0:
            continue
        b = max(1e-9, float(bw_share.get(dom, 0.0)))
        val = const + coeff / b
        if dom == "dp" and bool(getattr(sys, "objective_dp_gap_overlap", False)):
            val = max(0.0, val - float(gaps.get(dom, 0.0)))
        total += val
    return total


def _candidate_bw_shares_from_center(
    center: Dict[str, float],
    active: List[str],
    sys: "SystemConfig",
    local_step: float,
) -> List[Tuple[Dict[str, float], Dict[str, int] | None]]:
    domains = ["tp", "pp", "dp"]
    out: List[Tuple[Dict[str, float], Dict[str, int] | None]] = []
    seen: set[Tuple[int, int, int]] = set()

    def _push_from_bw(bw: Dict[str, float]) -> None:
        key = tuple(int(round(float(bw.get(d, 0.0)) * 1000000.0)) for d in domains)
        if key in seen:
            return
        seen.add(key)
        out.append((dict(bw), None))

    def _push_from_units(units: Dict[str, int], total_units: int) -> None:
        key = tuple(int(units.get(d, 0)) for d in domains)
        if key in seen:
            return
        seen.add(key)
        out.append(
            (
                {d: float(units.get(d, 0)) / float(total_units) for d in domains},
                dict(units),
            )
        )

    if sys.unit_bw_GBps > 0 and sys.total_bw_units is not None:
        total_units = int(sys.total_bw_units)
        active_count = len(active)
        min_u = 1 if total_units >= active_count else 0
        base_units = {
            d: int(round(float(center.get(d, 0.0)) * total_units)) for d in domains
        }
        for d in domains:
            if d not in active:
                base_units[d] = 0
        diff = total_units - sum(base_units.values())
        if active:
            base_units[active[0]] += diff
        radius = 2
        tp0 = base_units.get("tp", 0)
        pp0 = base_units.get("pp", 0)
        for du_tp in range(-radius, radius + 1):
            for du_pp in range(-radius, radius + 1):
                u_tp = tp0 + du_tp if "tp" in active else 0
                u_pp = pp0 + du_pp if "pp" in active else 0
                if u_tp < (min_u if "tp" in active else 0):
                    continue
                if u_pp < (min_u if "pp" in active else 0):
                    continue
                u_dp = total_units - u_tp - u_pp
                if "dp" in active and u_dp < min_u:
                    continue
                if "dp" not in active and u_dp != 0:
                    continue
                units = {"tp": u_tp, "pp": u_pp, "dp": u_dp}
                _push_from_units(units, total_units)
    else:
        if len(active) == 1:
            d0 = active[0]
            _push_from_bw({d: (1.0 if d == d0 else 0.0) for d in domains})
            return out

        step = max(0.01, float(local_step))
        offsets = [-2 * step, -step, 0.0, step, 2 * step]
        c_tp = float(center.get("tp", 0.0))
        c_pp = float(center.get("pp", 0.0))
        for off_tp in offsets:
            for off_pp in offsets:
                if len(active) == 2:
                    d0, d1 = active
                    b0 = float(center.get(d0, 0.5)) + off_tp
                    b1 = 1.0 - b0
                    if b0 <= 0.0 or b1 <= 0.0:
                        continue
                    bw = {d: 0.0 for d in domains}
                    bw[d0] = b0
                    bw[d1] = b1
                    _push_from_bw(bw)
                    continue

                b_tp = c_tp + off_tp
                b_pp = c_pp + off_pp
                b_dp = 1.0 - b_tp - b_pp
                if any(
                    float(x) <= 0.0
                    for d, x in [("tp", b_tp), ("pp", b_pp), ("dp", b_dp)]
                    if d in active
                ):
                    continue
                bw = {"tp": 0.0, "pp": 0.0, "dp": 0.0}
                if "tp" in active:
                    bw["tp"] = b_tp
                if "pp" in active:
                    bw["pp"] = b_pp
                if "dp" in active:
                    bw["dp"] = b_dp
                _push_from_bw(bw)

    return out


def _fast_bw_solve_from_terms(
    consts: Dict[str, float],
    coeffs: Dict[str, float],
    gaps: Dict[str, float],
    sys: "SystemConfig",
    local_step: float,
) -> Tuple[Dict[str, float], Dict[str, int] | None, float]:
    domains = ["tp", "pp", "dp"]
    active = [
        d
        for d in domains
        if float(coeffs.get(d, 0.0)) > 0.0 or float(consts.get(d, 0.0)) > 0.0
    ]
    if not active:
        bw = {d: 0.0 for d in domains}
        return bw, None, 0.0
    if len(active) == 1:
        d0 = active[0]
        bw = {d: (1.0 if d == d0 else 0.0) for d in domains}
        units = None
        if sys.unit_bw_GBps > 0 and sys.total_bw_units is not None:
            units = {d: (sys.total_bw_units if d == d0 else 0) for d in domains}
        return bw, units, _approx_interval_objective_ms(consts, coeffs, gaps, bw, sys)

    weights = {
        d: max(1e-9, math.sqrt(max(0.0, float(coeffs.get(d, 0.0))))) for d in active
    }
    norm = sum(weights.values())
    center = {d: 0.0 for d in domains}
    if norm <= 0.0:
        for d in active:
            center[d] = 1.0 / len(active)
    else:
        for d in active:
            center[d] = float(weights[d]) / norm

    best_bw = {d: 0.0 for d in domains}
    best_units = None
    best_cost = float("inf")
    candidates = _candidate_bw_shares_from_center(
        center, active, sys, local_step=local_step
    )
    if not candidates:
        candidates = [(center, None)]
    for bw, units in candidates:
        cost = _approx_interval_objective_ms(consts, coeffs, gaps, bw, sys)
        if cost < best_cost:
            best_cost = cost
            best_bw = dict(bw)
            best_units = dict(units) if units is not None else None
    return best_bw, best_units, best_cost


def solve_min_delay_bw_split(
    segment_nodes: List[CommNode],
    sys: "SystemConfig",
    bw_grid_step: float = 0.01,
) -> Tuple[Dict[str, float], Dict[str, int] | None, float]:
    """Solve the segment bandwidth split by minimizing *modeled completion time*.

    This intentionally does NOT use sqrt(W_d). Instead it directly minimizes:

        min_{b_d >= 0, sum b_d = 1}  sum_{node in segment} T_node(b_domain(node))

    where T_node is estimate_time_ms() (collective completion time under this model).

    Returns (bw_share, bw_units_or_none, best_comm_time_ms).
    """
    domains = ["tp", "pp", "dp"]
    active = [d for d in domains if any(n.domain == d for n in segment_nodes)]

    if len(active) == 0:
        bw = {d: 0.0 for d in domains}
        return bw, None, 0.0

    if len(active) == 1:
        d0 = active[0]
        bw = {d: (1.0 if d == d0 else 0.0) for d in domains}
        units = None
        if sys.unit_bw_GBps > 0 and sys.total_bw_units is not None:
            units = {d: (sys.total_bw_units if d == d0 else 0) for d in domains}
        return bw, units, _segment_objective_time_ms(segment_nodes, bw, sys)

    best_bw: Dict[str, float] | None = None
    best_units: Dict[str, int] | None = None
    best_t = float("inf")

    if sys.unit_bw_GBps > 0:
        if sys.total_bw_units is None:
            raise ValueError("unit_bw_GBps>0 but total_bw_units is None")
        total_units = sys.total_bw_units

        # Require >=1 unit per active domain if feasible; else allow zeros.
        min_u = 1 if total_units >= len(active) else 0

        for u_tp in range(min_u if "tp" in active else 0, total_units + 1):
            for u_pp in range(min_u if "pp" in active else 0, total_units - u_tp + 1):
                u_dp = total_units - u_tp - u_pp
                if "dp" in active and u_dp < min_u:
                    continue
                if "dp" not in active and u_dp != 0:
                    continue
                units = {"tp": u_tp, "pp": u_pp, "dp": u_dp}
                bw = {d: units[d] / total_units for d in domains}
                t = _segment_objective_time_ms(segment_nodes, bw, sys)
                if t < best_t:
                    best_t = t
                    best_bw = bw
                    best_units = units
    else:
        step = float(bw_grid_step)
        if not (0 < step <= 0.2):
            raise ValueError("bw_grid_step must be in (0, 0.2]")

        # Ensure active domains get >= step share (b=0 yields inf via estimate_time_ms()).
        if len(active) == 2:
            d0, d1 = active
            for k in range(1, int(1.0 / step)):
                b0 = k * step
                b1 = 1.0 - b0
                if b1 < step:
                    continue
                bw = {d: 0.0 for d in domains}
                bw[d0] = b0
                bw[d1] = b1
                t = _segment_objective_time_ms(segment_nodes, bw, sys)
                if t < best_t:
                    best_t = t
                    best_bw = bw
                    best_units = None
        else:
            # 3 active domains
            for k_tp in range(1, int(1.0 / step)):
                b_tp = k_tp * step
                for k_pp in range(1, int((1.0 - b_tp) / step)):
                    b_pp = k_pp * step
                    b_dp = 1.0 - b_tp - b_pp
                    if b_dp < step:
                        continue
                    bw = {"tp": b_tp, "pp": b_pp, "dp": b_dp}
                    t = _segment_objective_time_ms(segment_nodes, bw, sys)
                    if t < best_t:
                        best_t = t
                        best_bw = bw
                        best_units = None

    if best_bw is None:
        # Fallback: equal split across active domains.
        bw = {d: 0.0 for d in domains}
        for d in active:
            bw[d] = 1.0 / len(active)
        return bw, None, _segment_objective_time_ms(segment_nodes, bw, sys)

    return best_bw, best_units, best_t


def _bw_boundary_ms_at(
    comm_nodes: List[CommNode], idx: int, sys: "SystemConfig"
) -> float:
    return exposed_boundary_ms(comm_nodes[idx].gap_before_ms, sys.reconfig_sec * 1000.0)


def _link_boundary_ms_at(
    comm_nodes: List[CommNode], idx: int, sys: "SystemConfig"
) -> float:
    return exposed_boundary_ms(
        comm_nodes[idx].gap_before_ms, sys.link_batch_sec * 1000.0
    )


def _best_degree_split_for_interval(
    comm_nodes: List[CommNode],
    p: int,
    q: int,
    sys: "SystemConfig",
    stream_cache: Dict[Tuple[int, int, str], List[PeerSet]],
    best_cache: Dict[Tuple[int, int], Tuple[float, Dict[str, int]]],
) -> Tuple[float, Dict[str, int]]:
    """Return (best_internal_retune_ms, best_degree_split) for [p..q] under total K."""
    key = (p, q)
    if key in best_cache:
        return best_cache[key]

    K = int(sys.degree_k_total)
    link_ms = float(sys.link_batch_sec) * 1000.0
    domains = ["tp", "pp", "dp"]

    # We keep stream_cache for optional debugging, but use a non-materializing batches calculator
    # to avoid exploding on RD/RH with large 'count' multipliers.
    def is_active(dom: str) -> bool:
        return any(comm_nodes[t].domain == dom for t in range(p, q + 1))

    active = {d: is_active(d) for d in domains}

    # If degree is unlimited/ideal, internal batching overhead can be 0.
    def _min_k_required(dom: str) -> int:
        min_k = 1
        for t in range(p, q + 1):
            n = comm_nodes[t]
            if n.domain != dom:
                continue
            algo = str(n.algo).strip().lower()
            op = str(n.op).strip().lower()
            if algo == "ring":
                min_k = max(min_k, 2)
            elif algo == "tree":
                # Tree uses parent + up to 2 children simultaneously.
                min_k = max(min_k, 3)
            elif op == "p2p" or algo == "p2p":
                min_k = max(min_k, 1)
            else:
                # RD/RH/Rab use singleton peer-sets.
                min_k = max(min_k, 1)
        return min_k

    if K <= 0:
        # Choose minimal feasible split for readability (not used in cost).
        split = {d: 0 for d in domains}
        for d in domains:
            if active[d]:
                split[d] = _min_k_required(d)
        best_cache[key] = (0.0, split)
        return best_cache[key]

    # Enumerate only critical degrees per domain.
    crit: Dict[str, List[int]] = {}
    for d in domains:
        if not active[d]:
            crit[d] = [0]
        else:
            # Find critical k where batches(k) strictly decreases.
            # Also enforce feasibility: k must be >= max peer_set size in the interval.
            min_k = _min_k_required(d)
            if min_k > K:
                ks = []
            else:
                ks = []
                prev_b: int | None = None
                for k_try in range(min_k, K + 1):
                    b = _calc_batches_interval_fast(comm_nodes, p, q, d, k_try)
                    if prev_b is None or b < prev_b:
                        ks.append(k_try)
                        prev_b = b
            if not ks:
                best_cache[key] = (float("inf"), {dd: 0 for dd in domains})
                return best_cache[key]
            crit[d] = ks

    best_internal = float("inf")
    best_split = {d: 0 for d in domains}

    for k_tp in crit["tp"]:
        for k_pp in crit["pp"]:
            for k_dp in crit["dp"]:
                if k_tp + k_pp + k_dp > K:
                    continue
                if active["tp"] and k_tp <= 0:
                    continue
                if active["pp"] and k_pp <= 0:
                    continue
                if active["dp"] and k_dp <= 0:
                    continue

                b_tp = _calc_batches_interval_fast(comm_nodes, p, q, "tp", k_tp)
                b_pp = _calc_batches_interval_fast(comm_nodes, p, q, "pp", k_pp)
                b_dp = _calc_batches_interval_fast(comm_nodes, p, q, "dp", k_dp)
                if b_tp >= 10**18 or b_pp >= 10**18 or b_dp >= 10**18:
                    continue

                internal = 0.0
                internal += max(0, b_tp - 1) * link_ms
                internal += max(0, b_pp - 1) * link_ms
                internal += max(0, b_dp - 1) * link_ms

                if internal < best_internal:
                    best_internal = internal
                    best_split = {"tp": int(k_tp), "pp": int(k_pp), "dp": int(k_dp)}

    best_cache[key] = (best_internal, best_split)
    return best_cache[key]


def solve_best_link_only_plan(
    comm_nodes: List[CommNode],
    start_idx: int,
    end_idx: int,
    bw_share: Dict[str, float],
    sys: "SystemConfig",
) -> Tuple[List[LinkSegmentPlan], float]:
    """Inner DP: choose link-only boundaries + degree splits inside a fixed BW segment [start_idx..end_idx]."""
    if start_idx > end_idx:
        return [], 0.0

    # Precompute objective-time prefix sums under fixed bw_share.
    t_node: List[float] = []
    for t in range(start_idx, end_idx + 1):
        t_node.append(_node_objective_time_ms(comm_nodes[t], bw_share, sys))
    pref = [0.0]
    for v in t_node:
        pref.append(pref[-1] + float(v))

    def comm_ms(p: int, q: int) -> float:
        return float(pref[(q - start_idx) + 1] - pref[p - start_idx])

    # Caches for interval stream construction and best k split.
    stream_cache: Dict[Tuple[int, int, str], List[PeerSet]] = {}
    best_k_cache: Dict[Tuple[int, int], Tuple[float, Dict[str, int]]] = {}

    # DP over positions within [start_idx..end_idx].
    L = end_idx - start_idx + 1
    opt = [float("inf")] * (L + 1)  # opt[x] cost for first x nodes
    prev = [-1] * (L + 1)
    chosen: Dict[int, Tuple[int, int, Dict[str, int], float, float, float]] = {}
    # chosen[end_x] = (start_x, p_abs, deg_split, comm_ms, internal_ms, boundary_ms)
    opt[0] = 0.0

    for end_x in range(1, L + 1):
        q = start_idx + end_x - 1
        for start_x in range(1, end_x + 1):
            p = start_idx + start_x - 1
            boundary = (
                0.0 if p == start_idx else _link_boundary_ms_at(comm_nodes, p, sys)
            )

            internal, deg_split = _best_degree_split_for_interval(
                comm_nodes,
                p,
                q,
                sys,
                stream_cache=stream_cache,
                best_cache=best_k_cache,
            )
            if math.isinf(internal):
                continue
            c_ms = comm_ms(p, q)
            seg_cost = c_ms + float(internal)

            cand = float(opt[start_x - 1]) + float(boundary) + float(seg_cost)
            if cand < opt[end_x]:
                opt[end_x] = cand
                prev[end_x] = start_x - 1
                chosen[end_x] = (
                    start_x - 1,
                    p,
                    deg_split,
                    c_ms,
                    float(internal),
                    boundary,
                )

    if math.isinf(opt[L]):
        return [], float("inf")

    # Reconstruct link segments.
    link_segments: List[LinkSegmentPlan] = []
    cur = L
    while cur > 0:
        if cur not in chosen:
            raise RuntimeError("Inner link DP reconstruction failed")
        start_x0, p, deg_split, c_ms, internal_ms, boundary = chosen[cur]
        q = start_idx + cur - 1
        link_segments.append(
            LinkSegmentPlan(
                start_idx=p,
                end_idx=q,
                degree_split=dict(deg_split),
                comm_time_ms=float(c_ms),
                internal_retune_ms=float(internal_ms),
                exposed_link_boundary_ms=float(boundary),
            )
        )
        cur = start_x0
    link_segments.reverse()
    return link_segments, float(opt[L])


def solve_best_link_plan_for_bw_segment(
    comm_nodes: List[CommNode],
    start_idx: int,
    end_idx: int,
    bw_share: Dict[str, float],
    bw_units: Dict[str, int] | None,
    sys: "SystemConfig",
) -> BWSegmentPlan:
    """Build a BW segment plan [start_idx..end_idx] with inner optimal link-only segmentation."""
    bw_boundary = _bw_boundary_ms_at(comm_nodes, start_idx, sys) if comm_nodes else 0.0
    link_segments, inner_cost = solve_best_link_only_plan(
        comm_nodes, start_idx, end_idx, bw_share=bw_share, sys=sys
    )
    if math.isinf(inner_cost):
        # Infeasible due to degree constraints; keep empty plan.
        link_segments = []
    return BWSegmentPlan(
        start_idx=start_idx,
        end_idx=end_idx,
        bw_share=dict(bw_share),
        bw_units=dict(bw_units) if bw_units is not None else None,
        exposed_bw_boundary_ms=float(bw_boundary),
        link_segments=link_segments,
    )


def preplanned_dp_partition(
    comm_nodes: List[CommNode],
    sys: "SystemConfig",
    bw_grid_step: float = 0.01,
) -> List[BWSegmentPlan]:
    """Outer DP: choose BW segments; each BW segment is evaluated via the inner link-only DP.

        OPT[j] = min_{1<=i<=j} ( OPT[i-1] + t_r'(i) + L[i,j] )

    Here:
      - BW boundary cost at i is gap-hidden (including i=0): max(0, T_segment_reconfig-gap_before[i])
      - L_BW(i,j) is obtained by:
          (1) choosing best bandwidth split b* for nodes[i..j]
          (2) running inner DP to choose link-only cuts + degree splits under b*
    """
    n = len(comm_nodes)
    if n == 0:
        return []

    seg_cache: Dict[
        Tuple[int, int], Tuple[Dict[str, float], Dict[str, int] | None]
    ] = {}
    bw_plan_cache: Dict[Tuple[int, int], Tuple[BWSegmentPlan, float]] = {}

    def seg_bw_solve(i: int, j: int) -> Tuple[Dict[str, float], Dict[str, int] | None]:
        key = (i, j)
        if key in seg_cache:
            return seg_cache[key]
        bw, units, _t = solve_min_delay_bw_split(
            comm_nodes[i : j + 1], sys, bw_grid_step=bw_grid_step
        )
        seg_cache[key] = (bw, units)
        return bw, units

    opt = [float("inf")] * (n + 1)
    prev = [-1] * (n + 1)
    chosen: Dict[int, Tuple[int, BWSegmentPlan, float]] = {}

    opt[0] = 0.0
    for j in range(1, n + 1):
        for i in range(1, j + 1):
            s = i - 1
            e = j - 1
            if (s, e) in bw_plan_cache:
                bw_seg, seg_cost = bw_plan_cache[(s, e)]
            else:
                bw, units = seg_bw_solve(s, e)
                bw_seg = solve_best_link_plan_for_bw_segment(
                    comm_nodes, s, e, bw_share=bw, bw_units=units, sys=sys
                )
                # Inner plan cost excludes BW boundary; outer DP adds BW boundary cost.
                seg_cost = float("inf")
                if bw_seg.link_segments:
                    seg_cost = sum(
                        ls.total_ms for ls in bw_seg.link_segments
                    )  # includes link boundaries + internal + comm
                elif s <= e:
                    # Possible if [s..e] has no nodes (shouldn't happen) or infeasible.
                    seg_cost = float("inf")
                bw_plan_cache[(s, e)] = (bw_seg, seg_cost)

            rc = _bw_boundary_ms_at(comm_nodes, s, sys)
            cand = float(opt[i - 1]) + float(rc) + float(seg_cost)
            if cand < opt[j]:
                opt[j] = cand
                prev[j] = i - 1
                chosen[j] = (i - 1, bw_seg, rc)

    segments: List[BWSegmentPlan] = []
    cur = n
    while cur > 0:
        if cur not in chosen:
            raise RuntimeError("DP reconstruction failed")
        start, bw_seg, rc = chosen[cur]
        # Ensure the stored BW boundary cost matches what the outer DP used.
        bw_seg = BWSegmentPlan(
            start_idx=bw_seg.start_idx,
            end_idx=bw_seg.end_idx,
            bw_share=dict(bw_seg.bw_share),
            bw_units=dict(bw_seg.bw_units) if bw_seg.bw_units is not None else None,
            exposed_bw_boundary_ms=float(rc),
            link_segments=list(bw_seg.link_segments),
        )
        segments.append(bw_seg)
        cur = start
    segments.reverse()
    return segments


def fast_preplanned_partition(
    comm_nodes: List[CommNode],
    sys: "SystemConfig",
    bw_grid_step: float = 0.01,
    fast_local_bw_step: float = 0.05,
    refine_block_radius: int = 2,
) -> List[BWSegmentPlan]:
    """Approximate multi-segment planner for full comm-node schedules.

    Strategy:
    - collapse the known schedule into macro blocks at natural communication boundaries
    - run an outer DP over block intervals using a fast continuous surrogate objective
    - exactify only the selected segments with the existing per-segment bandwidth and link solvers
    - locally refine each internal boundary over a small neighboring block window
    """
    if not comm_nodes:
        return []

    blocks = _build_fast_blocks(comm_nodes)
    m = len(blocks)
    domains = ["tp", "pp", "dp"]

    pref_const = {d: [0.0] * (m + 1) for d in domains}
    pref_coeff = {d: [0.0] * (m + 1) for d in domains}
    pref_gap = {d: [0.0] * (m + 1) for d in domains}

    for bi, blk in enumerate(blocks, start=1):
        block_const = {d: 0.0 for d in domains}
        block_coeff = {d: 0.0 for d in domains}
        block_gap = {d: 0.0 for d in domains}
        for idx in range(blk.start_idx, blk.end_idx + 1):
            n = comm_nodes[idx]
            c_ms, k_ms = _node_continuous_terms_ms(n, sys)
            block_const[str(n.domain)] += float(c_ms)
            block_coeff[str(n.domain)] += float(k_ms)
            if str(n.domain) == "dp" and bool(
                getattr(sys, "objective_dp_gap_overlap", False)
            ):
                block_gap["dp"] += float(n.count) * max(0.0, float(n.gap_before_ms))

        for d in domains:
            pref_const[d][bi] = pref_const[d][bi - 1] + block_const[d]
            pref_coeff[d][bi] = pref_coeff[d][bi - 1] + block_coeff[d]
            pref_gap[d][bi] = pref_gap[d][bi - 1] + block_gap[d]

    def _interval_terms(
        a_blk: int, b_blk: int
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        consts = {d: pref_const[d][b_blk + 1] - pref_const[d][a_blk] for d in domains}
        coeffs = {d: pref_coeff[d][b_blk + 1] - pref_coeff[d][a_blk] for d in domains}
        gaps = {d: pref_gap[d][b_blk + 1] - pref_gap[d][a_blk] for d in domains}
        return consts, coeffs, gaps

    approx_cache: Dict[Tuple[int, int], float] = {}
    opt = [float("inf")] * (m + 1)
    prev = [-1] * (m + 1)
    opt[0] = 0.0

    for j in range(1, m + 1):
        for i in range(1, j + 1):
            a_blk = i - 1
            b_blk = j - 1
            key = (a_blk, b_blk)
            if key not in approx_cache:
                consts, coeffs, gaps = _interval_terms(a_blk, b_blk)
                _bw, _units, approx_comm = _fast_bw_solve_from_terms(
                    consts, coeffs, gaps, sys, local_step=fast_local_bw_step
                )
                rc = _bw_boundary_ms_at(comm_nodes, blocks[a_blk].start_idx, sys)
                approx_cache[key] = float(rc) + float(approx_comm)
            cand = float(opt[i - 1]) + float(approx_cache[key])
            if cand < opt[j]:
                opt[j] = cand
                prev[j] = i - 1

    if math.isinf(opt[m]):
        return preplanned_dp_partition(comm_nodes, sys, bw_grid_step=bw_grid_step)

    block_segments: List[Tuple[int, int]] = []
    cur = m
    while cur > 0:
        start = prev[cur]
        if start < 0:
            raise RuntimeError("Fast preplanned reconstruction failed")
        block_segments.append((start, cur - 1))
        cur = start
    block_segments.reverse()

    exact_cache: Dict[Tuple[int, int], BWSegmentPlan] = {}

    def _exact_seg(a_blk: int, b_blk: int) -> BWSegmentPlan:
        key = (a_blk, b_blk)
        if key in exact_cache:
            return exact_cache[key]
        start_idx = blocks[a_blk].start_idx
        end_idx = blocks[b_blk].end_idx
        bw, units, _comm_ms = solve_min_delay_bw_split(
            comm_nodes[start_idx : end_idx + 1], sys, bw_grid_step=bw_grid_step
        )
        seg = solve_best_link_plan_for_bw_segment(
            comm_nodes,
            start_idx,
            end_idx,
            bw_share=bw,
            bw_units=units,
            sys=sys,
        )
        exact_cache[key] = seg
        return seg

    if int(refine_block_radius) > 0 and len(block_segments) >= 2:
        changed = True
        passes = 0
        while changed and passes < 2:
            changed = False
            passes += 1
            for sidx in range(len(block_segments) - 1):
                left_a, left_b = block_segments[sidx]
                right_a, right_b = block_segments[sidx + 1]
                best_boundary = left_b
                best_total = (
                    _exact_seg(left_a, left_b).total_ms
                    + _exact_seg(right_a, right_b).total_ms
                )
                lo = max(left_a, left_b - int(refine_block_radius))
                hi = min(right_b - 1, left_b + int(refine_block_radius))
                for new_left_b in range(lo, hi + 1):
                    new_right_a = new_left_b + 1
                    cand_total = (
                        _exact_seg(left_a, new_left_b).total_ms
                        + _exact_seg(new_right_a, right_b).total_ms
                    )
                    if cand_total < best_total:
                        best_total = cand_total
                        best_boundary = new_left_b
                if best_boundary != left_b:
                    block_segments[sidx] = (left_a, best_boundary)
                    block_segments[sidx + 1] = (best_boundary + 1, right_b)
                    changed = True

    return [_exact_seg(a_blk, b_blk) for a_blk, b_blk in block_segments]


def _trace_one_shot(
    strategy: str,
    nodes: List[CommNode],
    sys: "SystemConfig",
    bw_grid_step: float,
) -> Tuple[List[TraceEvent], Dict[str, float], Dict[str, int] | None]:
    """One BW segment over the whole schedule; link-only plan chosen by inner DP."""
    bw, units, _comm_ms = solve_min_delay_bw_split(
        nodes, sys, bw_grid_step=bw_grid_step
    )
    bw_seg = solve_best_link_plan_for_bw_segment(
        nodes, 0, len(nodes) - 1, bw, units, sys
    )
    events = _trace_from_segments(
        strategy=strategy, nodes=nodes, sys=sys, segments=[bw_seg]
    )
    return events, bw, units


def _trace_static(
    strategy: str,
    nodes: List[CommNode],
    sys: "SystemConfig",
    bw_share: Dict[str, float],
    bw_units: Dict[str, int] | None,
    include_initial_reconfig: bool,
) -> List[TraceEvent]:
    """Static: fixed bandwidth split across the whole schedule; link-only plan by inner DP."""
    _ = include_initial_reconfig  # deprecated; initial BW boundary handled uniformly via BW plan
    if not nodes:
        return []
    bw_seg = solve_best_link_plan_for_bw_segment(
        nodes, 0, len(nodes) - 1, dict(bw_share), bw_units, sys
    )
    return _trace_from_segments(
        strategy=strategy, nodes=nodes, sys=sys, segments=[bw_seg]
    )
