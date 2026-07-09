from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# ================= 1. System & Model Configuration =================


class SystemConfig:
    def __init__(
        self,
        bandwidth_GBps: float,
        latency_us: float,
        unit_bw_GBps: float = 0.0,
        asym_min_reverse_units: int = 1,
        reconfig_ms: float = 0.0,
        max_peers_per_collective: int = 0,
        peer_switch_us: float = 0.0,
    ) -> None:
        """System/network parameters.

        :param bandwidth_GBps: per-direction injection bandwidth (GB/s).
        :param latency_us: per-step latency (microseconds).
        :param unit_bw_GBps: bandwidth allocation granularity ("lambda") in GB/s.
            - 0.0 => continuous (original fluid model)
            - >0  => quantize shares into integer units of size unit_bw_GBps
        :param asym_min_reverse_units: when unit_bw_GBps>0 and link_type='asymmetric' for
            ring/p2p patterns, reserve this many units for reverse/control, reducing payload BW.
        :param reconfig_ms: end-to-end bandwidth reconfiguration latency (milliseconds) when
            starting a new bandwidth-split segment in the pre-planned strategy.
        :param max_peers_per_collective: how many distinct peer circuits can be kept established
            simultaneously during a single collective call. 0 => unlimited/ideal (default).
        :param peer_switch_us: extra time to reconfigure/retune/rewire to a new peer *inside* a
            collective step sequence (microseconds).
        """
        self.bandwidth_GBps = float(bandwidth_GBps)
        self.latency_us = float(latency_us)

        self.bw_bytes_sec = self.bandwidth_GBps * 1e9
        self.latency_sec = self.latency_us * 1e-6

        self.unit_bw_GBps = float(unit_bw_GBps)
        self.asym_min_reverse_units = int(asym_min_reverse_units)

        self.reconfig_sec = float(reconfig_ms) * 1e-3

        self.max_peers_per_collective = int(max_peers_per_collective)
        self.peer_switch_sec = float(peer_switch_us) * 1e-6

        # Precompute number of discrete units ("lanes") if enabled.
        if self.unit_bw_GBps > 0:
            unit_bytes = self.unit_bw_GBps * 1e9
            self.total_bw_units = int(self.bw_bytes_sec // unit_bytes + 1e-12)
            self.total_bw_units = max(1, self.total_bw_units)
        else:
            self.total_bw_units = None


class ModelConfig:
    def __init__(
        self,
        layers: int = 126,
        hidden: int = 16384,
        seq: int = 8192,
        total_params: float = 405e9,
        bytes_per_act: int = 2,  # BF16
        bytes_per_param: int = 2,  # BF16
    ) -> None:
        """
        Llama 3 405B (8K pre-training, CP=1) modeling inputs.

        Notes:
        - For comm volumes we primarily need boundary activation tensor sizes and parameter shard sizes.
        - total_params is taken as ~405B (paper/model report); we do not re-derive it from architecture.
        """
        self.layers = int(layers)
        self.hidden = int(hidden)
        self.seq = int(seq)
        self.total_params = float(total_params)
        self.bytes_per_act = int(bytes_per_act)
        self.bytes_per_param = int(bytes_per_param)


class ParallelConfig:
    def __init__(
        self,
        tp: int = 8,
        pp: int = 16,
        dp: int = 128,  # ZeRO-2/DP group size
        global_batch_seqs: int = 2048,
        microbatch_seqs: int = 1,
    ) -> None:
        # TP×PP×DP = 8×16×128 = 16k GPUs for the 8K phase.
        self.tp = int(tp)
        self.pp = int(pp)
        self.dp = int(dp)
        self.global_batch_seqs = int(global_batch_seqs)
        self.microbatch_seqs = int(microbatch_seqs)

        # "DP replicas" (a.k.a. number of DP groups) = TP×PP in the 3D layout.
        self.dp_groups = self.tp * self.pp
        self.batch_per_dp_group = self.global_batch_seqs / self.dp_groups
        self.microbatches_per_step = int(self.batch_per_dp_group / self.microbatch_seqs)


# ================= 2. Traffic & Efficiency Logic =================


MiB = 1024**2
GiB = 1024**3


def quantize_share_dict(
    bw_share_req: Dict[str, float],
    total_units: int,
    min_units_per_domain: int = 0,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Quantize bandwidth shares into integer 'units' (lanes).

    Uses a largest-remainder (Hamilton) style rounding so that sum(units)==total_units.

    :param bw_share_req: dict domain->float share (ideally sums to 1.0).
    :param total_units: total number of discrete units available.
    :param min_units_per_domain: minimum units to allocate to any domain with share>0.
    :return: (bw_share_eff, units_alloc) where:
        - bw_share_eff: dict domain->effective share (= units_alloc[d] / total_units)
        - units_alloc: dict domain->int units
    """
    if total_units <= 0:
        raise ValueError("total_units must be positive")

    domains = list(bw_share_req.keys())
    desired = {d: float(bw_share_req[d]) * total_units for d in domains}
    units = {d: int(math.floor(desired[d])) for d in domains}

    active = [d for d in domains if bw_share_req[d] > 0]
    if min_units_per_domain > 0:
        for d in active:
            units[d] = max(units[d], min_units_per_domain)

    # Adjust down if we over-allocated due to min_units_per_domain.
    min_floor = {d: (min_units_per_domain if d in active else 0) for d in domains}
    used = sum(units.values())
    while used > total_units:
        # Remove from the most over-allocated domain (units - desired), but never below its minimum.
        candidates = [d for d in domains if units[d] > min_floor[d]]
        if not candidates:
            break
        d = max(candidates, key=lambda x: (units[x] - desired[x], units[x]))
        units[d] -= 1
        used -= 1

    # Distribute remaining units by largest fractional remainder.
    used = sum(units.values())
    rem = total_units - used
    if rem > 0:
        remainders = sorted(
            domains, key=lambda d: (desired[d] - units[d]), reverse=True
        )
        for i in range(rem):
            units[remainders[i % len(remainders)]] += 1

    # Final fixup for any numerical corner cases.
    used = sum(units.values())
    if used != total_units:
        if used < total_units:
            d = max(domains, key=lambda x: bw_share_req[x])
            units[d] += total_units - used
        else:
            # Reduce from the largest allocation.
            d = max(domains, key=lambda x: units[x])
            units[d] = max(min_floor[d], units[d] - (used - total_units))

    bw_share_eff: Dict[str, float] = {d: units[d] / total_units for d in domains}
    return bw_share_eff, units


def llama3_405b_payloads(
    mod: ModelConfig, par: ParallelConfig
) -> Tuple[int, int, float]:
    """
    Communication payloads (bytes) for Llama 3 405B 8K, CP=1.

    - A_full: boundary activation per microbatch, shape [S, d] in BF16
    - A_shard: sequence-parallel style TP shard, [S/TP, d] in BF16
    - P_local: local parameter/gradient shard bytes (sharded by TP×PP)
    """
    a_full = mod.seq * mod.hidden * mod.bytes_per_act  # bytes
    a_shard = (mod.seq / par.tp) * mod.hidden * mod.bytes_per_act  # bytes
    p_local_params = (
        mod.total_params / (par.tp * par.pp)
    ) * mod.bytes_per_param  # bytes

    return int(a_full), int(a_shard), float(p_local_params)


def get_efficiency(pattern: str, link_type: str) -> float:
    """
    Returns bandwidth efficiency factor (eta).

    This script is modeling *port/bandwidth coupling* for different patterns:

    - 'ring' and 'p2p' (chain-style): need simultaneous send/recv adjacency each step.
      * symmetric (coupled Tx/Rx) effectively burns 2 ports => eta=0.5
      * asymmetric (decoupled Tx vs Rx) can do (Tx->next, Rx<-prev) with 1 port => eta=1.0

    - 1-peer-at-a-time (recursive doubling/halving, Rabenseifner) uses eta=1.0 here.
    """
    if pattern in ["ring", "p2p"]:
        return 1.0 if link_type == "asymmetric" else 0.5

    return 1.0


def _ceil_log2(n: int) -> int:
    return int(math.ceil(math.log2(n)))


def _distinct_partners(nodes: int, op: str, algo: str) -> int:
    """Number of distinct peer partners used across stages of a collective.

    - ring/p2p: fixed neighbors => 0
    - recursive doubling/halving: log2(p) distinct partners
    - Rabenseifner: 2*log2(p) steps but same partner set reused => log2(p) distinct partners
    """
    if nodes <= 1:
        return 0
    lg = _ceil_log2(nodes)
    if algo in ["rd", "rh", "recursive_doubling", "rd_allreduce"]:
        return lg
    if algo == "rabenseifner":
        return lg
    return 0


def intra_collective_reconfig_overhead_sec(
    nodes: int, op: str, algo: str, sys: SystemConfig
) -> float:
    """Extra time to pay for intra-collective peer switching (seconds), once per call."""
    if nodes <= 1:
        return 0.0
    if sys.peer_switch_sec <= 0:
        return 0.0
    if sys.max_peers_per_collective <= 0:  # 0 => ideal/unlimited (backwards compatible)
        return 0.0
    d = _distinct_partners(nodes, op, algo)
    if d <= 0:
        return 0.0
    reconfigs = max(0, d - sys.max_peers_per_collective)
    return reconfigs * sys.peer_switch_sec


def estimate_time_ms(
    payload_bytes: float | int,
    nodes: int,
    b_d: float,
    op: str,
    algo: str,
    link_type: str,
    sys: SystemConfig,
) -> float:
    """
    Communication time model (ms), per-rank.

    payload_bytes:
      - For AllGather / ReduceScatter / AllReduce: payload is the *full tensor size* M (bytes).
        (E.g., activation tensor [S,d] for TP ops; local param shard bytes for DP ops.)
      - For P2P: payload is message size (bytes) for a single transfer.

    algo choices (as requested):
      - op='allgather'     : 'ring' (sym/asym), 'rd' (recursive doubling)
      - op='reducescatter' : 'ring' (sym/asym), 'rh' (recursive halving)
      - op='p2p'           : 'p2p'  (sym/asym)
      - op='allreduce'     : 'ring' (sym/asym), 'rabenseifner', 'recursive_doubling'
    """
    if nodes <= 1:
        return 0.0

    setup_sec = intra_collective_reconfig_overhead_sec(nodes, op, algo, sys)

    # Effective bandwidth.
    pattern = "ring" if algo == "ring" else ("p2p" if op == "p2p" else "1peer")

    if sys.unit_bw_GBps > 0:
        unit_bytes = sys.unit_bw_GBps * 1e9
        if sys.total_bw_units is None:
            raise ValueError("unit_bw_GBps>0 but total_bw_units is None")
        # Convert share to integer units directly (share = units / total_units).
        k_alloc = int(round(float(b_d) * sys.total_bw_units))
        k_alloc = max(0, k_alloc)

        if pattern in ["ring", "p2p"]:
            if link_type == "symmetric":
                # Symmetric coupling: payload can only use pairs of units.
                k_usable = k_alloc // 2
            else:
                # Asymmetric: reserve a minimal reverse/control slice.
                k_usable = max(0, k_alloc - sys.asym_min_reverse_units)
        else:
            # 1-peer-at-a-time algorithms can use all allocated units.
            k_usable = k_alloc

        bw_eff = k_usable * unit_bytes
    else:
        # Continuous (fluid) model.
        bw_budget = b_d * sys.bw_bytes_sec
        eta = get_efficiency(pattern, link_type)
        bw_eff = eta * bw_budget

    if bw_eff <= 0:
        return float("inf")

    M = float(payload_bytes)

    if op == "p2p":
        steps = 1
        total_sent = M
        return ((steps * sys.latency_sec + setup_sec) + total_sent / bw_eff) * 1000

    if op == "allgather":
        if algo == "ring":
            steps = nodes - 1
            total_sent = M * (nodes - 1) / nodes
            return ((steps * sys.latency_sec + setup_sec) + total_sent / bw_eff) * 1000
        if algo == "rd":
            steps = _ceil_log2(nodes)
            # Bandwidth is the same order as ring: M(1-1/p); latency is better.
            total_sent = M * (nodes - 1) / nodes
            return ((steps * sys.latency_sec + setup_sec) + total_sent / bw_eff) * 1000
        raise ValueError(f"Unsupported allgather algo: {algo}")

    if op == "reducescatter":
        if algo == "ring":
            steps = nodes - 1
            total_sent = M * (nodes - 1) / nodes
            return ((steps * sys.latency_sec + setup_sec) + total_sent / bw_eff) * 1000
        if algo == "rh":
            steps = _ceil_log2(nodes)
            total_sent = M * (nodes - 1) / nodes
            return ((steps * sys.latency_sec + setup_sec) + total_sent / bw_eff) * 1000
        raise ValueError(f"Unsupported reducescatter algo: {algo}")

    if op == "allreduce":
        if algo == "ring":
            steps = 2 * (nodes - 1)
            # Ring allreduce ~= reduce-scatter + allgather: 2M(1-1/p).
            total_sent = 2 * M * (nodes - 1) / nodes
            return ((steps * sys.latency_sec + setup_sec) + total_sent / bw_eff) * 1000
        if algo == "rabenseifner":
            steps = 2 * _ceil_log2(nodes)
            total_sent = 2 * M * (nodes - 1) / nodes
            return ((steps * sys.latency_sec + setup_sec) + total_sent / bw_eff) * 1000
        if algo in ["recursive_doubling", "rd_allreduce"]:
            # Classic recursive-doubling allreduce: log2(p) rounds exchanging full M each round.
            steps = _ceil_log2(nodes)
            total_sent = steps * M
            return ((steps * sys.latency_sec + setup_sec) + total_sent / bw_eff) * 1000
        raise ValueError(f"Unsupported allreduce algo: {algo}")

    raise ValueError(f"Unsupported op: {op}")


def effective_bandwidth_GBps(
    nodes: int,
    b_d: float,
    op: str,
    algo: str,
    link_type: str,
    sys: SystemConfig,
) -> float:
    """Return the *payload-effective* bandwidth used by the model (GB/s).

    This mirrors the bandwidth logic inside estimate_time_ms(), but exposes the
    final payload BW so the caller can print/debug it.
    """
    if nodes <= 1:
        return 0.0

    pattern = "ring" if algo == "ring" else ("p2p" if op == "p2p" else "1peer")

    if sys.unit_bw_GBps > 0:
        unit_bytes = sys.unit_bw_GBps * 1e9
        if sys.total_bw_units is None:
            raise ValueError("unit_bw_GBps>0 but total_bw_units is None")
        k_alloc = int(round(float(b_d) * sys.total_bw_units))
        k_alloc = max(0, k_alloc)

        if pattern in ["ring", "p2p"]:
            if link_type == "symmetric":
                k_usable = k_alloc // 2
            else:
                k_usable = max(0, k_alloc - sys.asym_min_reverse_units)
        else:
            k_usable = k_alloc

        return (k_usable * unit_bytes) / 1e9

    bw_budget = b_d * sys.bw_bytes_sec
    eta = get_efficiency(pattern, link_type)
    return (eta * bw_budget) / 1e9


# ================= 3\. Execution & Comparison =================


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
class TraceEvent:
    """A serialized timeline event for one rank within one iteration."""

    strategy: str  # "preplanned" | "one-shot" | "static"
    kind: str  # "reconfig" | "comm"
    label: str
    domain: str  # "tp" | "pp" | "dp" | "reconfig"
    start_ms: float
    duration_ms: float
    bw_share: Dict[str, float] | None = None
    bw_units: Dict[str, int] | None = None

    @property
    def end_ms(self) -> float:
        return self.start_ms + self.duration_ms


def exposed_reconfig_ms_for_segment_start(
    node: CommNode, sys: SystemConfig, is_first: bool
) -> float:
    """Exposed reconfiguration time (ms) at the start of a segment, after hiding by compute gap."""
    if sys.reconfig_sec <= 0:
        return 0.0
    if is_first:
        return sys.reconfig_sec * 1000.0
    gap = max(0.0, float(node.gap_before_ms))
    return max(0.0, sys.reconfig_sec * 1000.0 - gap)


def _segment_comm_time_ms(
    nodes: List[CommNode],
    bw_share: Dict[str, float],
    sys: SystemConfig,
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
    n: CommNode, bw_share: Dict[str, float], sys: SystemConfig
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
    sys: SystemConfig,
    segments: List[SegmentPlan],
) -> List[TraceEvent]:
    """Build a serialized comm trace for the given segmentation."""
    events: List[TraceEvent] = []
    x = 0.0
    for s_i, s in enumerate(segments):
        seg_nodes = nodes[s.start_idx : s.end_idx + 1]
        if not seg_nodes:
            continue

        # Reconfig event at segment start (exposed after any gap hiding).
        if s.exposed_reconfig_ms > 0:
            events.append(
                TraceEvent(
                    strategy=strategy,
                    kind="reconfig",
                    label="R",
                    domain="reconfig",
                    start_ms=x,
                    duration_ms=float(s.exposed_reconfig_ms),
                    bw_share=dict(s.bw_share),
                    bw_units=dict(s.bw_units) if s.bw_units is not None else None,
                )
            )
            x += float(s.exposed_reconfig_ms)

        for n in seg_nodes:
            t_ms = _node_comm_time_ms(n, s.bw_share, sys)
            events.append(
                TraceEvent(
                    strategy=strategy,
                    kind="comm",
                    label=n.name,
                    domain=n.domain,
                    start_ms=x,
                    duration_ms=float(t_ms),
                    bw_share=dict(s.bw_share),
                    bw_units=dict(s.bw_units) if s.bw_units is not None else None,
                )
            )
            x += float(t_ms)
    return events


def _trace_one_shot(
    strategy: str,
    nodes: List[CommNode],
    sys: SystemConfig,
    bw_grid_step: float,
) -> Tuple[List[TraceEvent], Dict[str, float], Dict[str, int] | None]:
    """One segment over the whole schedule (one reconfig at start)."""
    bw, units, _comm_ms = solve_min_delay_bw_split(
        nodes, sys, bw_grid_step=bw_grid_step
    )
    events: List[TraceEvent] = []
    x = 0.0
    rc_ms = (
        exposed_reconfig_ms_for_segment_start(nodes[0], sys, is_first=True)
        if nodes
        else 0.0
    )
    if rc_ms > 0:
        events.append(
            TraceEvent(
                strategy=strategy,
                kind="reconfig",
                label="R",
                domain="reconfig",
                start_ms=x,
                duration_ms=float(rc_ms),
                bw_share=dict(bw),
                bw_units=dict(units) if units is not None else None,
            )
        )
        x += float(rc_ms)

    for n in nodes:
        t_ms = _node_comm_time_ms(n, bw, sys)
        events.append(
            TraceEvent(
                strategy=strategy,
                kind="comm",
                label=n.name,
                domain=n.domain,
                start_ms=x,
                duration_ms=float(t_ms),
                bw_share=dict(bw),
                bw_units=dict(units) if units is not None else None,
            )
        )
        x += float(t_ms)
    return events, bw, units


def _trace_static(
    strategy: str,
    nodes: List[CommNode],
    sys: SystemConfig,
    bw_share: Dict[str, float],
    bw_units: Dict[str, int] | None,
    include_initial_reconfig: bool,
) -> List[TraceEvent]:
    """Static: fixed bandwidth split across the whole schedule (default: no reconfig)."""
    events: List[TraceEvent] = []
    x = 0.0
    if include_initial_reconfig and nodes:
        rc_ms = exposed_reconfig_ms_for_segment_start(nodes[0], sys, is_first=True)
        if rc_ms > 0:
            events.append(
                TraceEvent(
                    strategy=strategy,
                    kind="reconfig",
                    label="R",
                    domain="reconfig",
                    start_ms=x,
                    duration_ms=float(rc_ms),
                    bw_share=dict(bw_share),
                    bw_units=dict(bw_units) if bw_units is not None else None,
                )
            )
            x += float(rc_ms)

    for n in nodes:
        t_ms = _node_comm_time_ms(n, bw_share, sys)
        events.append(
            TraceEvent(
                strategy=strategy,
                kind="comm",
                label=n.name,
                domain=n.domain,
                start_ms=x,
                duration_ms=float(t_ms),
                bw_share=dict(bw_share),
                bw_units=dict(bw_units) if bw_units is not None else None,
            )
        )
        x += float(t_ms)
    return events


def _write_trace_json(path: Path, events: List[TraceEvent]) -> None:
    data = [
        {
            "strategy": e.strategy,
            "kind": e.kind,
            "label": e.label,
            "domain": e.domain,
            "start_ms": e.start_ms,
            "duration_ms": e.duration_ms,
            "end_ms": e.end_ms,
            "bw_share": e.bw_share,
            "bw_units": e.bw_units,
        }
        for e in events
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )


def _write_trace_csv(path: Path, events: List[TraceEvent]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "strategy",
                "kind",
                "domain",
                "label",
                "start_ms",
                "duration_ms",
                "end_ms",
                "bw_tp",
                "bw_pp",
                "bw_dp",
            ]
        )
        for e in events:
            bw = e.bw_share or {}
            w.writerow(
                [
                    e.strategy,
                    e.kind,
                    e.domain,
                    e.label,
                    f"{e.start_ms:.6f}",
                    f"{e.duration_ms:.6f}",
                    f"{e.end_ms:.6f}",
                    f"{float(bw.get('tp', 0.0)):.6f}",
                    f"{float(bw.get('pp', 0.0)):.6f}",
                    f"{float(bw.get('dp', 0.0)):.6f}",
                ]
            )


def _try_plot_rows_png(
    out_path: Path,
    rows: List[Tuple[str, List[TraceEvent]]],
    title: str,
    pp_label_every: int,
    min_marker_ms: float,
    ms_per_inch: float,
    max_width_in: float,
    params_text: str,
) -> bool:
    """Plot an arbitrary list of labeled trace rows with a shared time scale.

    :param rows: list of (row_label, events) pairs. Events should already have absolute start_ms.
    :return: True if plot was written; False if matplotlib unavailable.
    """
    try:
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
    except Exception:
        return False

    COL_TP = "#4DA3D9"
    COL_PP = "#E56B8A"
    COL_DP = "#4FB06E"
    COL_RECFG = "#9AA0A6"

    def _lighten(hex_color: str, t: float = 0.55) -> str:
        """Blend a hex color toward white by factor t in [0,1]."""
        t = max(0.0, min(1.0, float(t)))
        c = hex_color.lstrip("#")
        r = int(c[0:2], 16)
        g = int(c[2:4], 16)
        b = int(c[4:6], 16)
        r2 = int(round(r + (255 - r) * t))
        g2 = int(round(g + (255 - g) * t))
        b2 = int(round(b + (255 - b) * t))
        return f"#{r2:02x}{g2:02x}{b2:02x}"

    COL_TP_L = _lighten(COL_TP, 0.62)
    COL_PP_L = _lighten(COL_PP, 0.62)
    COL_DP_L = _lighten(COL_DP, 0.62)

    total_ms = 0.0
    for _lbl, evs in rows:
        if evs:
            total_ms = max(total_ms, max(e.end_ms for e in evs))
    # Padding space on the right for per-row completion-time annotations.
    pad_ms = max(5.0, 0.02 * total_ms) if total_ms > 0 else 5.0

    def _bw_key(e: TraceEvent) -> Tuple[Tuple[str, float], ...]:
        bw = e.bw_share or {}
        return tuple((k, float(bw.get(k, 0.0))) for k in ["tp", "pp", "dp"])

    def _coalesce_for_plot(evs: List[TraceEvent]) -> List[TraceEvent]:
        """Reduce clutter by coalescing consecutive comm events with same domain and bw split."""
        out: List[TraceEvent] = []
        i = 0
        while i < len(evs):
            e = evs[i]
            if e.kind != "comm":
                out.append(e)
                i += 1
                continue

            # Merge runs of consecutive comm events with same domain + bw split.
            j = i + 1
            dur = float(e.duration_ms)
            while (
                j < len(evs)
                and evs[j].kind == "comm"
                and evs[j].domain == e.domain
                and _bw_key(evs[j]) == _bw_key(e)
            ):
                dur += float(evs[j].duration_ms)
                j += 1

            # Compact label: "MBk\nTP"/"MBk\nPP", DP => "DP\n(RS+AG)" when merging multiple.
            label = e.label
            if e.domain in ["tp", "pp"] and e.label.startswith("MB"):
                # e.label like "MB{n}:TP:AllGather"
                mb = e.label.split(":")[0] if ":" in e.label else e.label.split()[0]
                dom = "TP" if e.domain == "tp" else "PP"
                label = f"{mb}\n{dom}"
            elif e.domain == "dp":
                label = "DP\n(RS+AG)" if (j - i) >= 2 else "DP"

            out.append(
                TraceEvent(
                    strategy=e.strategy,
                    kind=e.kind,
                    label=label,
                    domain=e.domain,
                    start_ms=e.start_ms,
                    duration_ms=dur,
                    bw_share=e.bw_share,
                    bw_units=e.bw_units,
                )
            )
            i = j
        return out

    # Coalesce each row trace for plotting readability.
    plot_rows: List[Tuple[str, List[TraceEvent]]] = [
        (lbl, _coalesce_for_plot(evs)) for lbl, evs in rows
    ]

    # Auto-size width so small PP/reconfig blocks become visible, without wrapping the timeline.
    ms_per_in = max(1.0, float(ms_per_inch))
    w_in = total_ms / ms_per_in
    w_in = max(22.0, float(w_in))
    w_in = min(float(max_width_in), float(w_in))

    # Height scales with number of rows.
    nrows = max(1, len(plot_rows))
    h = 0.75
    row_step = 1.0
    fig_h_in = max(4.2, 1.0 + nrows * 0.85)
    fig, ax = plt.subplots(figsize=(w_in, fig_h_in))

    def _col(domain: str, active: bool) -> str:
        if domain == "tp":
            return COL_TP if active else COL_TP_L
        if domain == "pp":
            return COL_PP if active else COL_PP_L
        if domain == "dp":
            return COL_DP if active else COL_DP_L
        return COL_RECFG

    dom_stack = ["tp", "pp", "dp"]  # bottom -> top

    def _domain_y_span(
        y_base: float, h_total: float, bw: Dict[str, float], domain: str
    ) -> Tuple[float, float]:
        """Return (y0,y1) vertical span for a domain slice within a stacked BW bar."""
        s = (
            float(bw.get("tp", 0.0))
            + float(bw.get("pp", 0.0))
            + float(bw.get("dp", 0.0))
        )
        if s <= 0:
            s = 1.0
        cum = 0.0
        for d in dom_stack:
            frac = float(bw.get(d, 0.0)) / s
            if d == domain:
                y0 = y_base + h_total * cum
                y1 = y_base + h_total * (cum + frac)
                return y0, y1
            cum += frac
        return y_base, y_base + h_total

    for row_i, (row_label, evs) in enumerate(plot_rows):
        # Top-to-bottom order: first row at the top.
        y = float((nrows - 1 - row_i) * row_step)
        # row label
        ax.text(
            -0.01 * max(1.0, total_ms),
            y + h / 2,
            row_label,
            ha="right",
            va="center",
            fontsize=10,
            color="#333333",
        )
        # Completion-time annotation (total = makespan; show comm vs reconfig breakdown).
        if evs:
            makespan = max(float(e.end_ms) for e in evs)
            comm_ms = sum(float(e.duration_ms) for e in evs if e.kind == "comm")
            rc_ms = sum(float(e.duration_ms) for e in evs if e.kind == "reconfig")
            ax.text(
                total_ms + pad_ms,
                y + h / 2,
                f"total={makespan:.1f}ms  (comm {comm_ms:.1f} + R {rc_ms:.1f})",
                ha="left",
                va="center",
                fontsize=9,
                color="#222222",
            )
        for idx, e in enumerate(evs):
            if e.kind == "reconfig":
                # Keep reconfiguration cost as a full-height grey block (separate semantic from BW usage).
                rect = patches.Rectangle(
                    (e.start_ms, y),
                    e.duration_ms,
                    h,
                    linewidth=0.8,
                    edgecolor="white",
                    facecolor=COL_RECFG,
                )
                ax.add_patch(rect)
            else:
                # Bandwidth-sliced view: stack TP/PP/DP vertically by bw share.
                bw = e.bw_share or {}
                # Normalize defensively (should already sum to 1.0).
                s = (
                    float(bw.get("tp", 0.0))
                    + float(bw.get("pp", 0.0))
                    + float(bw.get("dp", 0.0))
                )
                if s <= 0:
                    s = 1.0
                y_off = 0.0
                for d in dom_stack:
                    frac = float(bw.get(d, 0.0)) / s
                    hh = h * max(0.0, frac)
                    if hh <= 0:
                        continue
                    rect = patches.Rectangle(
                        (e.start_ms, y + y_off),
                        e.duration_ms,
                        hh,
                        linewidth=0.8,
                        edgecolor="white",
                        facecolor=_col(d, active=(d == e.domain)),
                    )
                    ax.add_patch(rect)
                    y_off += hh

            # Make very short events visible with a tick (does not change time scaling).
            marker_ms = max(0.0, float(min_marker_ms))
            if marker_ms > 0 and float(e.duration_ms) < marker_ms:
                if e.kind == "comm":
                    bw = e.bw_share or {}
                    y0, y1 = _domain_y_span(y, h, bw, e.domain)
                else:
                    # Reconfig marker spans full height.
                    y0, y1 = y, y + h
                ax.vlines(
                    float(e.start_ms),
                    y0,
                    y1,
                    colors=_col(e.domain, active=True)
                    if e.kind == "comm"
                    else COL_RECFG,
                    linewidth=5.0,
                    zorder=5,
                )

            # Keep labels sparse to avoid clutter.
            if e.kind == "reconfig":
                if e.duration_ms >= 0.6:
                    ax.text(
                        e.start_ms + e.duration_ms / 2,
                        y + h / 2,
                        "R",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white",
                    )
                continue

            lbl = e.label
            if e.domain in ["tp", "dp"]:
                if e.duration_ms >= 5.0:
                    # Place label in the center of the active domain's vertical slice.
                    bw = e.bw_share or {}
                    y0, y1 = _domain_y_span(y, h, bw, e.domain)
                    y_mid = (y0 + y1) / 2.0
                    ax.text(
                        e.start_ms + e.duration_ms / 2,
                        y_mid,
                        lbl,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white",
                    )
            else:
                # PP: label every Nth PP block (after coalescing).
                if (
                    pp_label_every > 0
                    and ((idx + 1) % pp_label_every == 0)
                    and e.duration_ms >= 2.0
                ):
                    bw = e.bw_share or {}
                    y0, y1 = _domain_y_span(y, h, bw, e.domain)
                    y_mid = (y0 + y1) / 2.0
                    ax.text(
                        e.start_ms + e.duration_ms / 2,
                        y_mid,
                        lbl,
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white",
                    )

    ax.set_xlim(0, max(1e-6, total_ms + 3.0 * pad_ms))
    ax.set_ylim(-0.4, float((nrows - 1) * row_step) + h + 0.35)
    ax.set_yticks([])
    ax.set_xlabel("Time (ms)")
    for spine in ax.spines.values():
        spine.set_visible(False)

    note = ""
    if float(min_marker_ms) > 0:
        note = f"; short events (<{float(min_marker_ms):g}ms) marked with thick tick"
    ax.set_title(title + note, fontsize=12)

    if params_text.strip():
        # Parameter block (kept small and unobtrusive).
        ax.text(
            0.005,
            0.99,
            params_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            family="monospace",
            color="#222222",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="#dddddd",
                alpha=0.85,
            ),
        )
    legend = [
        patches.Patch(color=COL_RECFG, label="Reconfig (R)"),
        patches.Patch(color=COL_TP, label="TP BW slice (dark=active, light=idle)"),
        patches.Patch(color=COL_PP, label="PP BW slice (dark=active, light=idle)"),
        patches.Patch(color=COL_DP, label="DP BW slice (dark=active, light=idle)"),
    ]
    ax.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        frameon=False,
        fontsize=9,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def _try_plot_trace_png(
    out_path: Path,
    strategy_to_events: Dict[str, List[TraceEvent]],
    title: str,
    pp_label_every: int,
    min_marker_ms: float,
    ms_per_inch: float,
    max_width_in: float,
    params_text: str,
) -> bool:
    """Return True if plot was written; False if matplotlib unavailable."""
    row_order = ["preplanned", "one-shot", "static"]
    row_names = {"preplanned": "Preplanned", "one-shot": "One-shot", "static": "Static"}
    rows: List[Tuple[str, List[TraceEvent]]] = [
        (row_names[k], strategy_to_events.get(k, [])) for k in row_order
    ]
    return _try_plot_rows_png(
        out_path=out_path,
        rows=rows,
        title=title,
        pp_label_every=pp_label_every,
        min_marker_ms=min_marker_ms,
        ms_per_inch=ms_per_inch,
        max_width_in=max_width_in,
        params_text=params_text,
    )


def solve_min_delay_bw_split(
    segment_nodes: List[CommNode],
    sys: SystemConfig,
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
        return bw, units, _segment_comm_time_ms(segment_nodes, bw, sys)

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
                t = _segment_comm_time_ms(segment_nodes, bw, sys)
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
                t = _segment_comm_time_ms(segment_nodes, bw, sys)
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
                    t = _segment_comm_time_ms(segment_nodes, bw, sys)
                    if t < best_t:
                        best_t = t
                        best_bw = bw
                        best_units = None

    if best_bw is None:
        # Fallback: equal split across active domains.
        bw = {d: 0.0 for d in domains}
        for d in active:
            bw[d] = 1.0 / len(active)
        return bw, None, _segment_comm_time_ms(segment_nodes, bw, sys)

    return best_bw, best_units, best_t


def preplanned_dp_partition(
    comm_nodes: List[CommNode],
    sys: SystemConfig,
    bw_grid_step: float = 0.01,
) -> List[SegmentPlan]:
    """DP partitioning to choose segment starts:

        OPT[j] = min_{1<=i<=j} ( OPT[i-1] + t_r'(i) + L[i,j] )

    where L[i,j] is the segment comm time using the segment's min-delay bw split, and
    t_r'(i) is the exposed reconfig cost at segment start i (after gap hiding).
    """
    n = len(comm_nodes)
    if n == 0:
        return []

    seg_cache: Dict[
        Tuple[int, int], Tuple[Dict[str, float], Dict[str, int] | None, float]
    ] = {}

    def seg_solve(
        i: int, j: int
    ) -> Tuple[Dict[str, float], Dict[str, int] | None, float]:
        key = (i, j)
        if key in seg_cache:
            return seg_cache[key]
        bw, units, t = solve_min_delay_bw_split(
            comm_nodes[i : j + 1], sys, bw_grid_step=bw_grid_step
        )
        seg_cache[key] = (bw, units, t)
        return bw, units, t

    opt = [float("inf")] * (n + 1)
    prev = [-1] * (n + 1)
    chosen: Dict[
        int, Tuple[int, Dict[str, float], Dict[str, int] | None, float, float]
    ] = {}

    opt[0] = 0.0
    for j in range(1, n + 1):
        for i in range(1, j + 1):
            bw, units, seg_t = seg_solve(i - 1, j - 1)
            rc = exposed_reconfig_ms_for_segment_start(
                comm_nodes[i - 1], sys, is_first=(i == 1)
            )
            cand = opt[i - 1] + rc + seg_t
            if cand < opt[j]:
                opt[j] = cand
                prev[j] = i - 1
                chosen[j] = (i - 1, bw, units, seg_t, rc)

    segments: List[SegmentPlan] = []
    cur = n
    while cur > 0:
        if cur not in chosen:
            raise RuntimeError("DP reconstruction failed")
        start, bw, units, seg_t, rc = chosen[cur]
        segments.append(
            SegmentPlan(
                start_idx=start,
                end_idx=cur - 1,
                bw_share=bw,
                bw_units=units,
                comm_time_ms=seg_t,
                exposed_reconfig_ms=rc,
            )
        )
        cur = start
    segments.reverse()
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Llama 3 405B communication model with bandwidth granularity support"
    )
    parser.add_argument(
        "--unit-bw-gbps",
        type=float,
        default=0.0,
        help="Bandwidth allocation granularity (lambda) in GB/s. "
        "0.0 = continuous model (default), >0 = quantize shares into integer units",
    )
    parser.add_argument(
        "--asym-min-reverse-units",
        type=int,
        default=1,
        help="When unit_bw_GBps>0 and link_type='asymmetric' for ring/p2p patterns, "
        "reserve this many units for reverse/control (default: 1)",
    )
    parser.add_argument(
        "--reconfig-ms",
        type=float,
        default=0.0,
        help="End-to-end bandwidth reconfiguration latency (ms) for pre-planned segmentation "
        "(default: 0.0)",
    )
    parser.add_argument(
        "--bw-grid-step",
        type=float,
        default=0.01,
        help="Grid step for solving per-segment min-delay bw splits (continuous model only). "
        "Ignored when --unit-bw-gbps>0. (default: 0.01)",
    )
    parser.add_argument(
        "--preplanned-schedule",
        type=str,
        default="microbatch",
        choices=["microbatch", "compact"],
        help=(
            "Which 'known comm schedule' abstraction to use for the Llama step pre-planned DP. "
            "This controls the *sequence of CommNodes* that the segmentation DP partitions.\n\n"
            "- microbatch: per-microbatch interleaving (TP, TP, PP) repeated for each microbatch, "
            "then DP collectives at end-of-step. More boundary options; DP can decide whether "
            "frequent reconfigs are worth it.\n"
            "- compact: collapse into a few coarse blocks (TP block, PP block, DP block). Fewer "
            "boundary options; faster, but less expressive.\n\n"
            "(default: microbatch)"
        ),
    )
    parser.add_argument(
        "--no-preplanned",
        action="store_true",
        help="Skip computing/printing the pre-planned DP schedule for the Llama step.",
    )
    parser.add_argument(
        "--emit-comm-trace",
        action="store_true",
        help="Emit a one-rank serialized communication timeline for (preplanned, one-shot, static). "
        "Writes JSON/CSV, and optionally a PNG if matplotlib is available.",
    )
    parser.add_argument(
        "--comm-trace-out-dir",
        type=str,
        default="out",
        help="Output directory for comm trace artifacts (default: ./out)",
    )
    parser.add_argument(
        "--comm-trace-prefix",
        type=str,
        default="one_node_comm_sequence_iteration",
        help="File prefix for comm trace outputs (default: one_node_comm_sequence_iteration)",
    )
    parser.add_argument(
        "--comm-trace-no-png",
        action="store_true",
        help="Do not attempt to write the PNG trace plot (JSON/CSV still written).",
    )
    parser.add_argument(
        "--comm-trace-pp-label-every",
        type=int,
        default=4,
        help="Label every Nth PP block in the plot to reduce clutter (default: 4; 0 disables PP labels).",
    )
    parser.add_argument(
        "--comm-trace-plot-min-marker-ms",
        type=float,
        default=1.0,
        help="Draw a thick vertical marker for events shorter than this duration (ms) to make "
        "very small PP/reconfig blocks visible (default: 1.0). Set 0 to disable.",
    )
    parser.add_argument(
        "--comm-trace-plot-ms-per-inch",
        type=float,
        default=800.0,
        help="Plot horizontal scale: milliseconds per inch. Smaller => wider plot and more visible "
        "tiny events (default: 800.0).",
    )
    parser.add_argument(
        "--comm-trace-plot-max-width-inch",
        type=float,
        default=60.0,
        help="Cap the plot width (inches) to avoid absurdly large images (default: 60).",
    )
    parser.add_argument(
        "--comm-trace-plot-no-params",
        action="store_true",
        help="Do not print the run parameters block onto the PNG figure.",
    )
    parser.add_argument(
        "--static-include-initial-reconfig",
        action="store_true",
        help="For the static strategy, include a one-time initial reconfig block at t=0 "
        "(default: off; static is shown as no reconfigs).",
    )
    parser.add_argument(
        "--collective-profiles",
        type=str,
        nargs="+",
        default=["all"],
        choices=["mixed", "ring_asym", "ring_sym", "hypercube", "all"],
        help=(
            "Which collective-algorithm profile(s) to use for the *planning* section "
            "(preplanned/one-shot/static).\n\n"
            "- mixed     : (backwards-compatible default) TP=ring/asym, PP=p2p/asym, DP=hypercube (RH/RD)\n"
            "- ring_asym : ring/asymmetric for TP+DP, p2p/asymmetric for PP\n"
            "- ring_sym  : ring/symmetric for TP+DP, p2p/symmetric for PP\n"
            "- hypercube : TP+DP use RH/RD (1-peer-at-a-time) with symmetric links; PP uses p2p/symmetric\n"
            "- all       : run ring_asym + ring_sym + hypercube (and omit mixed)\n\n"
            "(default: all)"
        ),
    )
    parser.add_argument(
        "--bandwidth-gbps",
        type=float,
        default=240.0,
        help="Per-direction injection bandwidth in GB/s (default: 240.0)",
    )
    parser.add_argument(
        "--latency-us",
        type=float,
        default=2.0,
        help="Per-step latency in microseconds (default: 2.0)",
    )
    parser.add_argument(
        "--max-peers-per-collective",
        type=int,
        default=0,
        help="How many distinct peers can be kept connected simultaneously during a collective "
        "(0=unlimited/ideal, default: 0)",
    )
    parser.add_argument(
        "--peer-switch-us",
        type=float,
        default=0.0,
        help="Extra latency to switch/retune/reconfigure to a new peer inside a collective (us)",
    )
    args = parser.parse_args()

    # Setup: network configuration (defaults preserve previous behavior).
    sys = SystemConfig(
        bandwidth_GBps=args.bandwidth_gbps,
        latency_us=args.latency_us,
        unit_bw_GBps=args.unit_bw_gbps,
        asym_min_reverse_units=args.asym_min_reverse_units,
        reconfig_ms=args.reconfig_ms,
        max_peers_per_collective=args.max_peers_per_collective,
        peer_switch_us=args.peer_switch_us,
    )

    # Llama 3 405B 8K (CP=1) inputs.
    mod = ModelConfig(layers=126, hidden=16384, seq=8192, total_params=405e9)
    par = ParallelConfig(tp=8, pp=16, dp=128, global_batch_seqs=2048, microbatch_seqs=1)

    # Bandwidth shares: portions from the same total BW pool.
    # You can edit these if you want to bias one domain.
    bw_share: Dict[str, float] = {
        "tp": 1.0 / 3.0,
        "pp": 1.0 / 3.0,
        "dp": 1.0 / 3.0,
    }

    # Keep a copy for reporting (requested vs. effective).
    bw_share_requested: Dict[str, float] = dict(bw_share)
    bw_share_units: Dict[str, int] | None = (
        None  # domain->int units when sys.unit_bw_GBps>0
    )

    # Optional: quantize bandwidth shares into integer units (lambda granularity).
    if sys.unit_bw_GBps > 0:
        active_domains = [d for d, v in bw_share_requested.items() if v > 0]
        # Default minimum: try to keep >=1 unit for each active domain (if feasible).
        min_units = 1
        if sys.total_bw_units is not None and sys.total_bw_units < min_units * len(
            active_domains
        ):
            min_units = 0
        bw_share, bw_share_units = quantize_share_dict(
            bw_share_requested, sys.total_bw_units, min_units_per_domain=min_units
        )

    a_full, a_shard, p_local = llama3_405b_payloads(mod, par)

    def _fmt_bytes(x: float) -> str:
        if x >= GiB:
            return f"{x / GiB:.2f} GiB"
        if x >= MiB:
            return f"{x / MiB:.2f} MiB"
        return f"{x:.0f} B"

    def _fmt_mib_compact(x_bytes: float) -> str:
        # For titles/comments: match e.g. "224MiB" / "32MiB" (no space, integer MiB).
        return f"{x_bytes / MiB:.0f}MiB"

    def _fmt_gib(x_bytes: float) -> str:
        # For titles/comments: match e.g. "5.85 GiB".
        return f"{x_bytes / GiB:.2f} GiB"

    tp_peer_shards = par.tp - 1
    tp_recv_bytes = tp_peer_shards * a_shard  # equals M*(tp-1)/tp when M=a_full
    pp_p2p_bytes = a_shard
    dp_sendrecv_bytes = (par.dp - 1) * p_local / par.dp  # M*(dp-1)/dp when M=p_local

    print("=== Llama 3 405B (8K, CP=1) Communication Model ===")
    print(f"Network: {sys.bw_bytes_sec / 1e9:.0f} GB/s, {sys.latency_sec * 1e6:.1f} us")
    if sys.peer_switch_sec > 0:
        max_peers_str = (
            "unlimited/ideal"
            if sys.max_peers_per_collective <= 0
            else str(sys.max_peers_per_collective)
        )
        print(
            f"Intra-collective peer switching: max_peers_per_collective={max_peers_str}, "
            f"peer_switch={sys.peer_switch_sec * 1e6:.2f} us"
        )
    if sys.unit_bw_GBps > 0:
        print(
            f"Bandwidth granularity: unit={sys.unit_bw_GBps:.3f} GB/s, total_units={sys.total_bw_units}, "
            f"asym_min_reverse_units={sys.asym_min_reverse_units}"
        )
        if bw_share_units is not None:
            req = ", ".join(
                [f"{d}={bw_share_requested[d]:.3f}" for d in bw_share_requested]
            )
            eff = ", ".join(
                [
                    f"{d}={bw_share[d]:.3f} ({bw_share_units[d]}/{sys.total_bw_units})"
                    for d in bw_share
                ]
            )
            print(f"bw_share requested: {req}")
            print(f"bw_share effective : {eff}")
    print(f"Parallelism: TP(SP)={par.tp}, PP={par.pp}, DP(ZeRO-2)={par.dp}")
    print(
        f"Batching: global_batch={par.global_batch_seqs}, microbatch={par.microbatch_seqs}, "
        f"microbatches/step={par.microbatches_per_step}"
    )
    print(
        "Payloads at bf16 (per microbatch): "
        "Actually using mixed precision, fp32 for DP RS gradients\n"
        f"A_full={_fmt_bytes(a_full)} | A_shard={_fmt_bytes(a_shard)} | "
        f"P_local_shard={_fmt_bytes(p_local)}"
    )
    print("=" * 78)

    # ---- Per-op comparisons (requested switchability) ----

    comparisons = [
        (
            f"TP AllGather (per call, nodes=TP) each receive {_fmt_mib_compact(tp_recv_bytes)} "
            f"for {tp_peer_shards} activation shards",
            "tp",
            a_full,
            par.tp,
            "allgather",
            [
                ("Ring Asym", "ring", "asymmetric"),
                ("Ring Sym", "ring", "symmetric"),
                ("Recursive Doubling (AG)", "rd", "symmetric"),
            ],
        ),
        (
            f"TP ReduceScatter (per call, nodes=TP) each receive {_fmt_mib_compact(tp_recv_bytes)} "
            f"for {tp_peer_shards} activation shards",
            "tp",
            a_full,
            par.tp,
            "reducescatter",
            [
                ("Ring Asym", "ring", "asymmetric"),
                ("Ring Sym", "ring", "symmetric"),
                ("Recursive Halving (RS)", "rh", "symmetric"),
            ],
        ),
        (
            f"PP P2P boundary (per transfer, 1 hop) TP-sharded, {_fmt_mib_compact(pp_p2p_bytes)} "
            "send/recv per microbatch",
            "pp",
            a_shard,
            2,
            "p2p",
            [
                ("P2P Asym", "p2p", "asymmetric"),
                ("P2P Sym", "p2p", "symmetric"),
            ],
        ),
        (
            f"DP/ZeRO-2 ReduceScatter(dW) (per step call, nodes=DP), send/recv {_fmt_gib(dp_sendrecv_bytes)} parameters",
            "dp",
            p_local,
            par.dp,
            "reducescatter",
            [
                ("Ring Asym", "ring", "asymmetric"),
                ("Ring Sym", "ring", "symmetric"),
                ("Recursive Halving (RS)", "rh", "symmetric"),
            ],
        ),
        (
            f"DP/ZeRO-2 AllGather(W) (per step call, nodes=DP), send/recv {_fmt_gib(dp_sendrecv_bytes)} parameters",
            "dp",
            p_local,
            par.dp,
            "allgather",
            [
                ("Ring Asym", "ring", "asymmetric"),
                ("Ring Sym", "ring", "symmetric"),
                ("Recursive Doubling (AG)", "rd", "symmetric"),
            ],
        ),
        (
            "AllReduce (generic) (per call)",
            "dp",
            p_local,
            par.dp,
            "allreduce",
            [
                ("Ring Asym", "ring", "asymmetric"),
                ("Ring Sym", "ring", "symmetric"),
                ("Rabenseifner", "rabenseifner", "symmetric"),
                ("Recursive Doubling (AR)", "recursive_doubling", "symmetric"),
            ],
        ),
    ]

    for title, domain, payload, nodes, op, variants in comparisons:
        if sys.unit_bw_GBps > 0 and bw_share_units is not None:
            print(
                f"\n[{title}]  bw_share={bw_share_requested[domain]:.2f} -> {bw_share[domain]:.2f}  "
                f"({bw_share_units[domain]}/{sys.total_bw_units} units)"
            )
            print(
                f"{'Configuration':<28} | {'Time (ms)':>10} | {'Eff BW (GB/s)':>13} | {'vs Ring Sym':>10}"
            )
            print("-" * 74)
        else:
            print(f"\n[{title}]  bw_share={bw_share[domain]:.2f}")
            print(f"{'Configuration':<28} | {'Time (ms)':>10} | {'vs Ring Sym':>10}")
            print("-" * 58)

        # Baseline: Ring Sym (when present); else first item.
        base = next((v for v in variants if v[0] == "Ring Sym"), variants[0])
        t_base = estimate_time_ms(
            payload, nodes, bw_share[domain], op, base[1], base[2], sys
        )

        for name, algo, link in variants:
            t = estimate_time_ms(payload, nodes, bw_share[domain], op, algo, link, sys)

            if math.isinf(t_base) or math.isinf(t) or t <= 0:
                speed = float("nan")
            else:
                speed = t_base / t

            if sys.unit_bw_GBps > 0 and bw_share_units is not None:
                eff_bw = effective_bandwidth_GBps(
                    nodes, bw_share[domain], op, algo, link, sys
                )
                t_str = "inf" if math.isinf(t) else f"{t:10.3f}"
                speed_str = "n/a" if math.isnan(speed) else f"{speed:10.2f}x"
                print(f"{name:<28} | {t_str:>10} | {eff_bw:13.1f} | {speed_str:>10}")
            else:
                speed_str = "n/a" if math.isnan(speed) else f"{speed:10.2f}x"
                if math.isinf(t):
                    print(f"{name:<28} | {'inf':>10} | {speed_str:>10}")
                else:
                    print(f"{name:<28} | {t:10.3f} | {speed_str:>10}")

    # ---- Example end-to-end totals (per step, per rank) ----
    #
    # TP comm frequency: paper-level description says ~4 TP collectives per transformer layer.
    # A simple split is 2×AllGather + 2×ReduceScatter per layer (edit if you want a different mix).
    tp_calls_per_layer = {"allgather": 2, "reducescatter": 2}

    def _choices_for_profile(profile: str) -> Dict[str, Tuple[str, str]]:
        """Return algo/link_type choices for the planning schedule.

        Keys:
          - tp_allgather, tp_reducescatter, pp_p2p, dp_reducescatter, dp_allgather
        """
        prof = str(profile).strip().lower()
        if prof == "mixed":
            # Backwards-compatible default: ring/asym for TP+PP, hypercube (RH/RD) for DP.
            return {
                "tp_allgather": ("ring", "asymmetric"),
                "tp_reducescatter": ("ring", "asymmetric"),
                "pp_p2p": ("p2p", "asymmetric"),
                "dp_reducescatter": ("rh", "symmetric"),
                "dp_allgather": ("rd", "symmetric"),
            }
        if prof == "ring_asym":
            return {
                "tp_allgather": ("ring", "asymmetric"),
                "tp_reducescatter": ("ring", "asymmetric"),
                "pp_p2p": ("p2p", "asymmetric"),
                "dp_reducescatter": ("ring", "asymmetric"),
                "dp_allgather": ("ring", "asymmetric"),
            }
        if prof == "ring_sym":
            return {
                "tp_allgather": ("ring", "symmetric"),
                "tp_reducescatter": ("ring", "symmetric"),
                "pp_p2p": ("p2p", "symmetric"),
                "dp_reducescatter": ("ring", "symmetric"),
                "dp_allgather": ("ring", "symmetric"),
            }
        if prof == "hypercube":
            # "1-peer-at-a-time" collectives modeled as eta=1.0 (see get_efficiency()).
            return {
                "tp_allgather": ("rd", "symmetric"),
                "tp_reducescatter": ("rh", "symmetric"),
                "pp_p2p": ("p2p", "symmetric"),
                "dp_reducescatter": ("rh", "symmetric"),
                "dp_allgather": ("rd", "symmetric"),
            }
        raise ValueError(f"Unknown collective profile: {profile}")

    def _profiles_to_run() -> List[str]:
        req = [str(x).strip().lower() for x in (args.collective_profiles or ["mixed"])]
        if "all" in req:
            # Requested sweep over canonical profiles (excluding mixed unless explicitly specified).
            base = ["ring_asym", "ring_sym", "hypercube"]
            extras = [p for p in req if p not in ["all"]]
            out: List[str] = []
            for p in base + extras:
                if p and p not in out:
                    out.append(p)
            return out
        out: List[str] = []
        for p in req:
            if p and p not in out:
                out.append(p)
        return out or ["mixed"]

    def _per_step_totals(
        choices: Dict[str, Tuple[str, str]],
        bw_tp: float,
        bw_pp: float,
        bw_dp: float,
        sys: SystemConfig,
    ) -> Tuple[float, float, float, float, float]:
        """Return (tp_total_ms, pp_total_ms, dp_total_ms, t_tp_ag_ms, t_tp_rs_ms) for one step."""
        t_tp_ag = estimate_time_ms(
            a_full,
            par.tp,
            bw_tp,
            "allgather",
            choices["tp_allgather"][0],
            choices["tp_allgather"][1],
            sys,
        )
        t_tp_rs = estimate_time_ms(
            a_full,
            par.tp,
            bw_tp,
            "reducescatter",
            choices["tp_reducescatter"][0],
            choices["tp_reducescatter"][1],
            sys,
        )
        t_pp_one = estimate_time_ms(
            a_shard,
            2,
            bw_pp,
            "p2p",
            choices["pp_p2p"][0],
            choices["pp_p2p"][1],
            sys,
        )
        t_dp_rs = estimate_time_ms(
            p_local,
            par.dp,
            bw_dp,
            "reducescatter",
            choices["dp_reducescatter"][0],
            choices["dp_reducescatter"][1],
            sys,
        )
        t_dp_ag = estimate_time_ms(
            p_local,
            par.dp,
            bw_dp,
            "allgather",
            choices["dp_allgather"][0],
            choices["dp_allgather"][1],
            sys,
        )

        tp_total = (
            par.microbatches_per_step
            * mod.layers
            * (
                tp_calls_per_layer["allgather"] * t_tp_ag
                + tp_calls_per_layer["reducescatter"] * t_tp_rs
            )
        )
        pp_total = par.microbatches_per_step * (
            2.0 * t_pp_one
        )  # fwd + bwd per microbatch
        dp_total = t_dp_rs + t_dp_ag  # once per step (bucketed in reality; totals add)
        return tp_total, pp_total, dp_total, t_tp_ag, t_tp_rs

    # Per-step totals will be printed per collective profile later (so we can compare profiles).

    # ---- Pre-planned max utilization strategy ----
    #
    # - "When to reconfigure": global DP partitioning over the known comm-node sequence.
    # - "How much bw per domain": within each segment, choose bw shares that minimize the
    #   modeled collective completion time (estimate_time_ms), NOT traffic ratios.

    def _print_segments(
        tag: str,
        nodes: List[CommNode],
        segments: List[SegmentPlan],
        seg_sys: SystemConfig,
    ) -> None:
        print("\n" + "=" * 78)
        print(f"Pre-planned segmentation result ({tag}):")
        total_rc = sum(s.exposed_reconfig_ms for s in segments)
        total_comm = sum(s.comm_time_ms for s in segments)
        print(
            f"Segments={len(segments)} | exposed_reconfig={total_rc:.3f} ms | comm={total_comm:.3f} ms | total={total_rc + total_comm:.3f} ms"
        )
        for k, s in enumerate(segments, start=1):
            span = nodes[s.start_idx : s.end_idx + 1]
            doms = ",".join(sorted({n.domain for n in span}))
            bw_str = ", ".join(
                [f"{d}={s.bw_share.get(d, 0.0):.3f}" for d in ["tp", "pp", "dp"]]
            )
            if s.bw_units is not None and seg_sys.total_bw_units is not None:
                u_str = ", ".join(
                    [
                        f"{d}={s.bw_units.get(d, 0)}/{seg_sys.total_bw_units}"
                        for d in ["tp", "pp", "dp"]
                    ]
                )
                bw_str = f"{bw_str}  | units: {u_str}"
            print(
                f"  seg{k}: nodes[{s.start_idx + 1}..{s.end_idx + 1}] domains={doms}  "
                f"reconfig={s.exposed_reconfig_ms:.3f} ms  comm={s.comm_time_ms:.3f} ms  bw: {bw_str}"
            )

    def _build_llama_nodes(
        schedule: str, choices: Dict[str, Tuple[str, str]]
    ) -> List[CommNode]:
        """Build an abstracted, pre-known comm-node sequence for one training step."""
        if schedule == "compact":
            # One big TP block, one PP block, then DP collectives.
            tp_ag_count = (
                par.microbatches_per_step * mod.layers * tp_calls_per_layer["allgather"]
            )
            tp_rs_count = (
                par.microbatches_per_step
                * mod.layers
                * tp_calls_per_layer["reducescatter"]
            )
            pp_count = par.microbatches_per_step * 2  # fwd + bwd boundary transfers
            return [
                CommNode(
                    name="TP:AllGather",
                    domain="tp",
                    payload_bytes=a_full,
                    nodes=par.tp,
                    op="allgather",
                    algo=str(choices["tp_allgather"][0]),
                    link_type=str(choices["tp_allgather"][1]),
                    count=tp_ag_count,
                    gap_before_ms=0.0,
                ),
                CommNode(
                    name="TP:ReduceScatter",
                    domain="tp",
                    payload_bytes=a_full,
                    nodes=par.tp,
                    op="reducescatter",
                    algo=str(choices["tp_reducescatter"][0]),
                    link_type=str(choices["tp_reducescatter"][1]),
                    count=tp_rs_count,
                    gap_before_ms=0.0,
                ),
                CommNode(
                    name="PP:P2P",
                    domain="pp",
                    payload_bytes=a_shard,
                    nodes=2,
                    op="p2p",
                    algo=str(choices["pp_p2p"][0]),
                    link_type=str(choices["pp_p2p"][1]),
                    count=pp_count,
                    gap_before_ms=0.0,
                ),
                CommNode(
                    name="DP:ReduceScatter(dW)",
                    domain="dp",
                    payload_bytes=p_local,
                    nodes=par.dp,
                    op="reducescatter",
                    algo=str(choices["dp_reducescatter"][0]),
                    link_type=str(choices["dp_reducescatter"][1]),
                    count=1,
                    gap_before_ms=0.0,
                ),
                CommNode(
                    name="DP:AllGather(W)",
                    domain="dp",
                    payload_bytes=p_local,
                    nodes=par.dp,
                    op="allgather",
                    algo=str(choices["dp_allgather"][0]),
                    link_type=str(choices["dp_allgather"][1]),
                    count=1,
                    gap_before_ms=0.0,
                ),
            ]

        # Default: microbatch-level interleaving, which gives DP more realistic boundary choices.
        nodes: List[CommNode] = []
        tp_ag_per_mb = mod.layers * tp_calls_per_layer["allgather"]
        tp_rs_per_mb = mod.layers * tp_calls_per_layer["reducescatter"]
        for mb in range(par.microbatches_per_step):
            nodes.append(
                CommNode(
                    name=f"MB{mb + 1}:TP:AllGather",
                    domain="tp",
                    payload_bytes=a_full,
                    nodes=par.tp,
                    op="allgather",
                    algo=str(choices["tp_allgather"][0]),
                    link_type=str(choices["tp_allgather"][1]),
                    count=tp_ag_per_mb,
                    gap_before_ms=0.0,
                )
            )
            nodes.append(
                CommNode(
                    name=f"MB{mb + 1}:TP:ReduceScatter",
                    domain="tp",
                    payload_bytes=a_full,
                    nodes=par.tp,
                    op="reducescatter",
                    algo=str(choices["tp_reducescatter"][0]),
                    link_type=str(choices["tp_reducescatter"][1]),
                    count=tp_rs_per_mb,
                    gap_before_ms=0.0,
                )
            )
            nodes.append(
                CommNode(
                    name=f"MB{mb + 1}:PP:P2P(fwd+bwd)",
                    domain="pp",
                    payload_bytes=a_shard,
                    nodes=2,
                    op="p2p",
                    algo=str(choices["pp_p2p"][0]),
                    link_type=str(choices["pp_p2p"][1]),
                    count=2,
                    gap_before_ms=0.0,
                )
            )

        # DP collectives at end of step.
        nodes.append(
            CommNode(
                name="DP:ReduceScatter(dW)",
                domain="dp",
                payload_bytes=p_local,
                nodes=par.dp,
                op="reducescatter",
                algo=str(choices["dp_reducescatter"][0]),
                link_type=str(choices["dp_reducescatter"][1]),
                count=1,
                gap_before_ms=0.0,
            )
        )
        nodes.append(
            CommNode(
                name="DP:AllGather(W)",
                domain="dp",
                payload_bytes=p_local,
                nodes=par.dp,
                op="allgather",
                algo=str(choices["dp_allgather"][0]),
                link_type=str(choices["dp_allgather"][1]),
                count=1,
                gap_before_ms=0.0,
            )
        )
        return nodes

    def _one_shot_total(
        nodes: List[CommNode], seg_sys: SystemConfig
    ) -> Tuple[Dict[str, float], float, float]:
        """Single segment over the whole schedule (one reconfig at start)."""
        bw, _units, comm_ms = solve_min_delay_bw_split(
            nodes, seg_sys, bw_grid_step=float(args.bw_grid_step)
        )
        rc_ms = exposed_reconfig_ms_for_segment_start(nodes[0], seg_sys, is_first=True)
        return bw, comm_ms, rc_ms

    def _static_total(
        nodes: List[CommNode], seg_sys: SystemConfig, bw: Dict[str, float]
    ) -> float:
        return _segment_comm_time_ms(nodes, bw, seg_sys)

    if not args.no_preplanned:
        profiles = _profiles_to_run()
        # If we're sweeping multiple profiles, write ONE combined PNG with a shared time scale.
        _combine_png = (
            bool(args.emit_comm_trace)
            and (not args.comm_trace_no_png)
            and (len(profiles) > 1)
        )
        _combined_rows: List[Tuple[str, List[TraceEvent]]] = []
        _combined_params_one_shot_bw: Dict[str, Dict[str, float]] = {}
        for prof in profiles:
            choices = _choices_for_profile(prof)
            tag = f"llama3-step-{args.preplanned_schedule} | profile={prof}"

            # Compute per-step totals for this profile under the current static bw_share.
            tp_total, pp_total, dp_total, t_tp_ag, t_tp_rs = _per_step_totals(
                choices=choices,
                bw_tp=float(bw_share["tp"]),
                bw_pp=float(bw_share["pp"]),
                bw_dp=float(bw_share["dp"]),
                sys=sys,
            )
            print("\n" + "=" * 78)
            print(
                f"Per-step totals (per rank; collective_profile={prof}; using static bw_share):"
            )
            print(
                f"TP total: {tp_total / 1000:.3f} s  "
                f"(per-call AG {t_tp_ag:.3f} ms, RS {t_tp_rs:.3f} ms; {mod.layers} layers, {par.microbatches_per_step} mbs)"
            )
            print(f"PP total: {pp_total / 1000:.3f} s  (fwd+bwd boundary p2p)")
            print(f"DP total: {dp_total / 1000:.3f} s  (RS+AG at step end)")

            llama_nodes = _build_llama_nodes(
                str(args.preplanned_schedule), choices=choices
            )
            segments = preplanned_dp_partition(
                llama_nodes, sys, bw_grid_step=float(args.bw_grid_step)
            )
            _print_segments(tag, llama_nodes, segments, sys)

            bw_all, comm_all, rc_all = _one_shot_total(llama_nodes, sys)
            static_comm = _static_total(llama_nodes, sys, bw_share)

            print("\n" + "-" * 78)
            print(f"Pre-planned vs one-shot vs static(equal-share) [{tag}]:")
            pre_comm = sum(s.comm_time_ms for s in segments)
            pre_rc = sum(s.exposed_reconfig_ms for s in segments)
            print(
                f"  preplanned: comm={pre_comm:.3f} ms  reconfig={pre_rc:.3f} ms  total={(pre_comm + pre_rc):.3f} ms"
            )
            print(
                f"  one-shot  : comm={comm_all:.3f} ms  reconfig={rc_all:.3f} ms  total={(comm_all + rc_all):.3f} ms  "
                f"bw={', '.join([f'{d}={bw_all[d]:.3f}' for d in ['tp', 'pp', 'dp']])}"
            )
            print(
                f"  static    : comm={static_comm:.3f} ms  (bw_share tp/pp/dp all = {bw_share['tp']:.3f})"
            )

            if args.emit_comm_trace:
                # Build per-strategy traces using the same linearized schedule used by the solver.
                preplanned_trace = _trace_from_segments(
                    "preplanned", llama_nodes, sys, segments
                )
                one_shot_trace, one_shot_bw, one_shot_units = _trace_one_shot(
                    "one-shot", llama_nodes, sys, bw_grid_step=float(args.bw_grid_step)
                )
                static_trace = _trace_static(
                    "static",
                    llama_nodes,
                    sys,
                    bw_share=bw_share,
                    bw_units=bw_share_units,
                    include_initial_reconfig=bool(args.static_include_initial_reconfig),
                )

                out_dir = Path(str(args.comm_trace_out_dir))
                prefix = str(args.comm_trace_prefix)
                pfx = f"{prefix}.{prof}"

                # Write machine-readable traces.
                _write_trace_json(out_dir / f"{pfx}.preplanned.json", preplanned_trace)
                _write_trace_json(out_dir / f"{pfx}.one_shot.json", one_shot_trace)
                _write_trace_json(out_dir / f"{pfx}.static.json", static_trace)
                _write_trace_csv(out_dir / f"{pfx}.preplanned.csv", preplanned_trace)
                _write_trace_csv(out_dir / f"{pfx}.one_shot.csv", one_shot_trace)
                _write_trace_csv(out_dir / f"{pfx}.static.csv", static_trace)

                # Plotting:
                # - If multiple profiles are being swept, we create ONE combined PNG later.
                # - Otherwise, we emit a per-profile PNG like before.
                if not args.comm_trace_no_png:
                    if _combine_png:
                        _combined_rows.extend(
                            [
                                (f"{prof} | Preplanned", preplanned_trace),
                                (f"{prof} | One-shot", one_shot_trace),
                                (f"{prof} | Static", static_trace),
                            ]
                        )
                        _combined_params_one_shot_bw[str(prof)] = dict(one_shot_bw)
                    else:
                        # Build a concise parameter block for the figure.
                        params_lines: List[str] = []
                        params_lines.append(
                            f"profile={prof}  schedule={args.preplanned_schedule}  nodes: TP={par.tp}, PP={par.pp}, DP={par.dp}"
                        )
                        params_lines.append(
                            f"mbs/step={par.microbatches_per_step}  layers={mod.layers}  seq={mod.seq}  hidden={mod.hidden}"
                        )
                        params_lines.append(
                            f"net: bw={args.bandwidth_gbps:g}GB/s  lat={args.latency_us:g}us"
                        )
                        if args.reconfig_ms and float(args.reconfig_ms) > 0:
                            params_lines.append(
                                f"reconfig={float(args.reconfig_ms):g}ms"
                            )
                        else:
                            params_lines.append("reconfig=0ms")
                        if args.unit_bw_gbps and float(args.unit_bw_gbps) > 0:
                            params_lines.append(
                                f"unit_bw={float(args.unit_bw_gbps):g}GB/s  total_units={sys.total_bw_units}  "
                                f"asym_min_rev_units={int(args.asym_min_reverse_units)}"
                            )
                        else:
                            params_lines.append(
                                f"unit_bw=0 (continuous)  bw_grid_step={float(args.bw_grid_step):g}"
                            )
                        if args.peer_switch_us and float(args.peer_switch_us) > 0:
                            params_lines.append(
                                f"peer_switch={float(args.peer_switch_us):g}us  max_peers_per_collective={int(args.max_peers_per_collective)}"
                            )
                        # BW splits.
                        params_lines.append(
                            "static_bw: "
                            + ", ".join(
                                [
                                    f"{d}={bw_share.get(d, 0.0):.3f}"
                                    for d in ["tp", "pp", "dp"]
                                ]
                            )
                        )
                        params_lines.append(
                            "one_shot_bw: "
                            + ", ".join(
                                [
                                    f"{d}={one_shot_bw.get(d, 0.0):.3f}"
                                    for d in ["tp", "pp", "dp"]
                                ]
                            )
                        )
                        params_lines.append(
                            f"plot: ms_per_in={float(args.comm_trace_plot_ms_per_inch):g}  "
                            f"min_marker_ms={float(args.comm_trace_plot_min_marker_ms):g}"
                        )
                        params_text = (
                            ""
                            if args.comm_trace_plot_no_params
                            else "\n".join(params_lines)
                        )

                        wrote = _try_plot_trace_png(
                            out_dir / f"{pfx}.png",
                            strategy_to_events={
                                "preplanned": preplanned_trace,
                                "one-shot": one_shot_trace,
                                "static": static_trace,
                            },
                            title=(
                                "One node's serialized communication sequence within one iteration (linearized schedule)\n"
                                f"Profile={prof}; Grey = reconfiguration cost; Blue = TP comm; Pink = PP comm; Green = DP comm"
                            ),
                            pp_label_every=int(args.comm_trace_pp_label_every),
                            min_marker_ms=float(args.comm_trace_plot_min_marker_ms),
                            ms_per_inch=float(args.comm_trace_plot_ms_per_inch),
                            max_width_in=float(args.comm_trace_plot_max_width_inch),
                            params_text=params_text,
                        )
                        if not wrote:
                            print(
                                "\n[comm-trace] matplotlib not available; skipped PNG. "
                                "JSON/CSV traces were still written."
                            )

                # Print the effective one-shot bw split (handy sanity check).
                bw_str = ", ".join(
                    [f"{d}={one_shot_bw.get(d, 0.0):.3f}" for d in ["tp", "pp", "dp"]]
                )
                print("\n" + "-" * 78)
                print(f"[comm-trace] wrote traces to: {out_dir.resolve()}")
                print(f"[comm-trace] profile={prof} one-shot bw: {bw_str}")

        # Combined multi-profile plot under a shared time scale (one figure like the original).
        if (
            _combine_png
            and _combined_rows
            and args.emit_comm_trace
            and (not args.comm_trace_no_png)
        ):
            out_dir = Path(str(args.comm_trace_out_dir))
            prefix = str(args.comm_trace_prefix)
            combined_path = out_dir / f"{prefix}.png"

            # Build a concise parameter block for the combined figure.
            params_lines: List[str] = []
            params_lines.append(
                f"profiles={','.join([str(p) for p in profiles])}  schedule={args.preplanned_schedule}  nodes: TP={par.tp}, PP={par.pp}, DP={par.dp}"
            )
            params_lines.append(
                f"mbs/step={par.microbatches_per_step}  layers={mod.layers}  seq={mod.seq}  hidden={mod.hidden}"
            )
            params_lines.append(
                f"net: bw={args.bandwidth_gbps:g}GB/s  lat={args.latency_us:g}us"
            )
            if args.reconfig_ms and float(args.reconfig_ms) > 0:
                params_lines.append(f"reconfig={float(args.reconfig_ms):g}ms")
            else:
                params_lines.append("reconfig=0ms")
            if args.unit_bw_gbps and float(args.unit_bw_gbps) > 0:
                params_lines.append(
                    f"unit_bw={float(args.unit_bw_gbps):g}GB/s  total_units={sys.total_bw_units}  "
                    f"asym_min_rev_units={int(args.asym_min_reverse_units)}"
                )
            else:
                params_lines.append(
                    f"unit_bw=0 (continuous)  bw_grid_step={float(args.bw_grid_step):g}"
                )
            if args.peer_switch_us and float(args.peer_switch_us) > 0:
                params_lines.append(
                    f"peer_switch={float(args.peer_switch_us):g}us  max_peers_per_collective={int(args.max_peers_per_collective)}"
                )
            params_lines.append(
                "static_bw: "
                + ", ".join(
                    [f"{d}={bw_share.get(d, 0.0):.3f}" for d in ["tp", "pp", "dp"]]
                )
            )
            for p in profiles:
                bw_p = _combined_params_one_shot_bw.get(str(p), {})
                params_lines.append(
                    f"one_shot_bw[{p}]: "
                    + ", ".join(
                        [
                            f"{d}={float(bw_p.get(d, 0.0)):.3f}"
                            for d in ["tp", "pp", "dp"]
                        ]
                    )
                )
            params_lines.append(
                f"plot: ms_per_in={float(args.comm_trace_plot_ms_per_inch):g}  "
                f"min_marker_ms={float(args.comm_trace_plot_min_marker_ms):g}"
            )
            params_text = (
                "" if args.comm_trace_plot_no_params else "\n".join(params_lines)
            )

            wrote = _try_plot_rows_png(
                combined_path,
                rows=_combined_rows,
                title=(
                    "One node's serialized communication sequence within one iteration (linearized schedule)\n"
                    "Grey = reconfiguration cost; Blue = TP comm; Pink = PP comm; Green = DP comm"
                ),
                pp_label_every=int(args.comm_trace_pp_label_every),
                min_marker_ms=float(args.comm_trace_plot_min_marker_ms),
                ms_per_inch=float(args.comm_trace_plot_ms_per_inch),
                max_width_in=float(args.comm_trace_plot_max_width_inch),
                params_text=params_text,
            )
            if wrote:
                print("\n" + "-" * 78)
                print(
                    f"[comm-trace] wrote combined multi-profile plot to: {combined_path.resolve()}"
                )
            else:
                print(
                    "\n[comm-trace] matplotlib not available; skipped combined PNG. "
                    "JSON/CSV traces were still written."
                )


if __name__ == "__main__":
    main()
