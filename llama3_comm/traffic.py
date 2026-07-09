"""Traffic and efficiency logic for communication modeling."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Tuple

if TYPE_CHECKING:
    from .config import ModelConfig, ParallelConfig, SystemConfig


# ================= Constants =================

MiB = 1024**2
GiB = 1024**3


# ================= Bandwidth Quantization =================


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


# ================= Payload Calculation =================


def llama3_405b_payloads(
    mod: "ModelConfig", par: "ParallelConfig"
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


def llama3_megatron_payloads(
    mod: "ModelConfig", par: "ParallelConfig"
) -> Tuple[int, int, int, int, int]:
    """
    Megatron-style per-layer payloads (bytes) for Llama 3 405B 8K, CP=1.

    Returns:
      - a_full: full-sequence activation tensor [S, d] in BF16
      - a_shard: sequence-parallel TP shard [S/TP, d] in BF16
      - p_layer_tp_bf16: per-layer params per TP rank in BF16 bytes
      - p_layer_tp_fp32: per-layer grads per TP rank in FP32 bytes
      - p_layer_total_params: total per-layer params (elements)
    """
    a_full = mod.seq * mod.hidden * mod.bytes_per_act
    a_shard = (mod.seq / par.tp) * mod.hidden * mod.bytes_per_act

    # Per-layer params (SwiGLU, no biases) following Megatron sizes.
    qkv = mod.hidden * (mod.hidden + 2 * mod.kv_dim)
    attn_out = mod.hidden * mod.hidden
    mlp_fc1 = mod.hidden * (2 * mod.ffn_hidden)
    mlp_fc2 = mod.ffn_hidden * mod.hidden
    lns = 2 * mod.hidden
    per_layer_params = qkv + attn_out + mlp_fc1 + mlp_fc2 + lns

    per_tp_params = per_layer_params / par.tp
    p_layer_tp_bf16 = int(per_tp_params * mod.bytes_per_param)
    p_layer_tp_fp32 = int(per_tp_params * mod.bytes_per_grad)

    return (
        int(a_full),
        int(a_shard),
        p_layer_tp_bf16,
        p_layer_tp_fp32,
        int(per_layer_params),
    )


# ================= Efficiency Logic =================


def get_efficiency(pattern: str, link_type: str) -> float:
    """
    Returns bandwidth efficiency factor (eta).

    This script is modeling *port/bandwidth coupling* for different patterns:

    - 'ring', 'p2p', and 'tree': need simultaneous send/recv within each phase.
      * symmetric (coupled Tx/Rx) effectively burns 2 ports => eta=0.5
      * asymmetric (decoupled Tx vs Rx) can do unidirectional with 1 port => eta=1.0

      Note: tree collectives have opposite-direction phases (e.g., gather up, broadcast down),
      but within each phase nodes do bidirectional communication (e.g., parent receives from
      multiple children). Without inter-phase bandwidth reconfiguration, asymmetric links
      still give eta=1.0 by configuring each phase appropriately.

    - 1-peer-at-a-time (recursive doubling/halving, Rabenseifner) uses eta=1.0 here.
    """
    if pattern in ["ring", "p2p", "tree"]:
        return 1.0 if link_type == "asymmetric" else 0.5

    return 1.0


def _ceil_log2(n: int) -> int:
    return int(math.ceil(math.log2(n)))


def _distinct_partners(nodes: int, op: str, algo: str) -> int:
    """Number of distinct peer partners used across stages of a collective.

    - ring/p2p: fixed neighbors => 0
    - recursive doubling/halving: log2(p) distinct partners
    - Rabenseifner: 2*log2(p) steps but same partner set reused => log2(p) distinct partners
    - tree: bounded constant fan-in/fan-out (parent + up to 2 children) => <=3
    """
    if nodes <= 1:
        return 0
    lg = _ceil_log2(nodes)
    if algo in ["rd", "rh", "recursive_doubling", "rd_allreduce"]:
        return lg
    if algo == "rabenseifner":
        return lg
    if algo == "tree":
        # Binary tree: one parent + up to two children.
        return min(3, max(0, nodes - 1))
    return 0


def intra_collective_reconfig_overhead_sec(
    nodes: int, op: str, algo: str, sys: "SystemConfig"
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
    sys: "SystemConfig",
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
      - tree algorithms:
          - allgather     : 'tree' (gather-to-root, then broadcast)
          - reducescatter : 'tree' (reduce-to-root, then scatter reduced chunks)
          - allreduce     : 'tree' (reduce-to-root, then broadcast full reduced buffer)
    """
    if nodes <= 1:
        return 0.0

    setup_sec = intra_collective_reconfig_overhead_sec(nodes, op, algo, sys)

    # Effective bandwidth.
    pattern = (
        "ring"
        if algo == "ring"
        else ("tree" if algo == "tree" else ("p2p" if op == "p2p" else "1peer"))
    )

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
        if algo == "tree":
            # Two-phase: gather-to-root then broadcast full set down the tree.
            # Steps: 2*ceil(log_2(p)) for two phases (up and down the tree).
            # Bandwidth model: M bytes at effective BW accounting for fanout penalty.
            # The fanout penalty (serializing to 2 children) is in tree_bw, not in data amount.
            steps = 2 * _ceil_log2(nodes)
            total_sent = M
            # Effective BW = bw_eff / 2.0 accounts for 2-child fanout serialization.
            # Combined with get_efficiency() symmetric penalty (0.5), total efficiency is 0.25.
            tree_bw = bw_eff / 2.0
            # Tree requires switching direction (Up -> Down) mid-collective
            reconfig_penalty = sys.reconfig_sec if link_type == "asymmetric" else 0.0
            return (
                (steps * sys.latency_sec + setup_sec + reconfig_penalty)
                + total_sent / tree_bw
            ) * 1000
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
        if algo == "tree":
            # Two-phase: reduce-to-root (full buffer) then scatter reduced chunks.
            # Steps: 2*ceil(log_2(p)) for two phases (up and down the tree).
            # Bandwidth model: M bytes at effective BW accounting for fanout penalty.
            # The fanout penalty (serializing to 2 children) is in tree_bw, not in data amount.
            lg = _ceil_log2(nodes)
            steps = 2 * lg
            total_sent = M
            # Effective BW = bw_eff / 2.0 accounts for 2-child fanout serialization.
            # Combined with get_efficiency() symmetric penalty (0.5), total efficiency is 0.25.
            tree_bw = bw_eff / 2.0
            # Tree requires switching direction (Up -> Down) mid-collective
            reconfig_penalty = sys.reconfig_sec if link_type == "asymmetric" else 0.0
            return (
                (steps * sys.latency_sec + setup_sec + reconfig_penalty)
                + total_sent / tree_bw
            ) * 1000
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
        if algo == "tree":
            # Two-phase: reduce-to-root then broadcast full reduced buffer.
            # Steps: 2*ceil(log_2(p)) for two phases (up and down the tree).
            # Bandwidth model: M bytes at effective BW accounting for fanout penalty.
            # The fanout penalty (serializing to 2 children) is in tree_bw, not in data amount.
            lg = _ceil_log2(nodes)
            steps = 2 * lg
            total_sent = M
            # Effective BW = bw_eff / 2.0 accounts for 2-child fanout serialization.
            # Combined with get_efficiency() symmetric penalty (0.5), total efficiency is 0.25.
            tree_bw = bw_eff / 2.0
            # Tree requires switching direction (Up -> Down) mid-collective
            reconfig_penalty = sys.reconfig_sec if link_type == "asymmetric" else 0.0
            return (
                (steps * sys.latency_sec + setup_sec + reconfig_penalty)
                + total_sent / tree_bw
            ) * 1000
        raise ValueError(f"Unsupported allreduce algo: {algo}")

    raise ValueError(f"Unsupported op: {op}")


def effective_bandwidth_GBps(
    nodes: int,
    b_d: float,
    op: str,
    algo: str,
    link_type: str,
    sys: "SystemConfig",
) -> float:
    """Return the *payload-effective* bandwidth used by the model (GB/s).

    This mirrors the bandwidth logic inside estimate_time_ms(), but exposes the
    final payload BW so the caller can print/debug it.
    """
    if nodes <= 1:
        return 0.0

    pattern = (
        "ring"
        if algo == "ring"
        else ("tree" if algo == "tree" else ("p2p" if op == "p2p" else "1peer"))
    )

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
