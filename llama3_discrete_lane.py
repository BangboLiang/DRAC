from __future__ import annotations

import argparse
import math
from typing import Dict, Tuple

# ================= 1. System & Model Configuration =================


class SystemConfig:
    def __init__(
        self,
        bandwidth_GBps: float,
        latency_us: float,
        unit_bw_GBps: float = 0.0,
        asym_min_reverse_units: int = 1,
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
        remainders = sorted(domains, key=lambda d: (desired[d] - units[d]), reverse=True)
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


def llama3_405b_payloads(mod: ModelConfig, par: ParallelConfig) -> Tuple[int, int, float]:
    """
    Communication payloads (bytes) for Llama 3 405B 8K, CP=1.

    - A_full: boundary activation per microbatch, shape [S, d] in BF16
    - A_shard: sequence-parallel style TP shard, [S/TP, d] in BF16
    - P_local: local parameter/gradient shard bytes (sharded by TP×PP)
    """
    a_full = mod.seq * mod.hidden * mod.bytes_per_act  # bytes
    a_shard = (mod.seq / par.tp) * mod.hidden * mod.bytes_per_act  # bytes
    p_local_params = (mod.total_params / (par.tp * par.pp)) * mod.bytes_per_param  # bytes

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
    bw_budget = b_d * sys.bw_bytes_sec
    pattern = "ring" if algo == "ring" else ("p2p" if op == "p2p" else "1peer")

    if sys.unit_bw_GBps > 0:
        unit_bytes = sys.unit_bw_GBps * 1e9
        # Convert budget to integer units (robust to float noise).
        k_alloc = int(round(bw_budget / unit_bytes))
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

    bw_budget = b_d * sys.bw_bytes_sec
    pattern = "ring" if algo == "ring" else ("p2p" if op == "p2p" else "1peer")

    if sys.unit_bw_GBps > 0:
        unit_bytes = sys.unit_bw_GBps * 1e9
        k_alloc = int(round(bw_budget / unit_bytes))
        k_alloc = max(0, k_alloc)

        if pattern in ["ring", "p2p"]:
            if link_type == "symmetric":
                k_usable = k_alloc // 2
            else:
                k_usable = max(0, k_alloc - sys.asym_min_reverse_units)
        else:
            k_usable = k_alloc

        return (k_usable * unit_bytes) / 1e9

    eta = get_efficiency(pattern, link_type)
    return (eta * bw_budget) / 1e9


# ================= 3\. Execution & Comparison =================


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
    bw_share_units: Dict[str, int] | None = None  # domain->int units when sys.unit_bw_GBps>0

    # Optional: quantize bandwidth shares into integer units (lambda granularity).
    if sys.unit_bw_GBps > 0:
        active_domains = [d for d, v in bw_share_requested.items() if v > 0]
        # Default minimum: try to keep >=1 unit for each active domain (if feasible).
        min_units = 1
        if sys.total_bw_units is not None and sys.total_bw_units < min_units * len(active_domains):
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
    print(f"Network: {sys.bw_bytes_sec/1e9:.0f} GB/s, {sys.latency_sec*1e6:.1f} us")
    if sys.peer_switch_sec > 0:
        max_peers_str = (
            "unlimited/ideal"
            if sys.max_peers_per_collective <= 0
            else str(sys.max_peers_per_collective)
        )
        print(
            f"Intra-collective peer switching: max_peers_per_collective={max_peers_str}, "
            f"peer_switch={sys.peer_switch_sec*1e6:.2f} us"
        )
    if sys.unit_bw_GBps > 0:
        print(
            f"Bandwidth granularity: unit={sys.unit_bw_GBps:.3f} GB/s, total_units={sys.total_bw_units}, "
            f"asym_min_reverse_units={sys.asym_min_reverse_units}"
        )
        if bw_share_units is not None:
            req = ", ".join([f"{d}={bw_share_requested[d]:.3f}" for d in bw_share_requested])
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
        t_base = estimate_time_ms(payload, nodes, bw_share[domain], op, base[1], base[2], sys)

        for name, algo, link in variants:
            t = estimate_time_ms(payload, nodes, bw_share[domain], op, algo, link, sys)

            if math.isinf(t_base) or math.isinf(t) or t <= 0:
                speed = float("nan")
            else:
                speed = t_base / t

            if sys.unit_bw_GBps > 0 and bw_share_units is not None:
                eff_bw = effective_bandwidth_GBps(nodes, bw_share[domain], op, algo, link, sys)
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

    # Default algorithm choices for totals (edit to "switch" quickly).
    choices = {
        "tp_allgather": ("ring", "asymmetric"),
        "tp_reducescatter": ("ring", "asymmetric"),
        "pp_p2p": ("p2p", "asymmetric"),
        "dp_reducescatter": ("rh", "symmetric"),
        "dp_allgather": ("rd", "symmetric"),
    }

    t_tp_ag = estimate_time_ms(
        a_full,
        par.tp,
        bw_share["tp"],
        "allgather",
        choices["tp_allgather"][0],
        choices["tp_allgather"][1],
        sys,
    )
    t_tp_rs = estimate_time_ms(
        a_full,
        par.tp,
        bw_share["tp"],
        "reducescatter",
        choices["tp_reducescatter"][0],
        choices["tp_reducescatter"][1],
        sys,
    )
    t_pp_one = estimate_time_ms(
        a_shard,
        2,
        bw_share["pp"],
        "p2p",
        choices["pp_p2p"][0],
        choices["pp_p2p"][1],
        sys,
    )
    t_dp_rs = estimate_time_ms(
        p_local,
        par.dp,
        bw_share["dp"],
        "reducescatter",
        choices["dp_reducescatter"][0],
        choices["dp_reducescatter"][1],
        sys,
    )
    t_dp_ag = estimate_time_ms(
        p_local,
        par.dp,
        bw_share["dp"],
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
    pp_total = par.microbatches_per_step * (2.0 * t_pp_one)  # fwd + bwd per microbatch
    dp_total = t_dp_rs + t_dp_ag  # once per step (bucketed in reality; totals add)

    print("\n" + "=" * 78)
    print("Per-step totals (per rank; using 'choices' above):")
    print(
        f"TP total: {tp_total/1000:.3f} s  "
        f"(per-call AG {t_tp_ag:.3f} ms, RS {t_tp_rs:.3f} ms; {mod.layers} layers, {par.microbatches_per_step} mbs)"
    )
    print(f"PP total: {pp_total/1000:.3f} s  (per-transfer {t_pp_one:.3f} ms; fwd+bwd)")
    print(f"DP total: {dp_total/1000:.3f} s  (RS {t_dp_rs:.3f} ms + AG {t_dp_ag:.3f} ms)")


if __name__ == "__main__":
    main()