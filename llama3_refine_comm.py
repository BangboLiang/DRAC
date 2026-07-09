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
        link_batch_ms: float = 0.0,
        degree_k_total: int = 0,
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
        :param reconfig_ms: BW-boundary ("full reset") latency in milliseconds.
            Updates bandwidth split and initial link topology for the next BW segment.
        :param link_batch_ms: Link-only boundary latency in milliseconds.
            Updates only active peer edges while keeping bandwidth split unchanged.
        :param degree_k_total: Total degree budget K: maximum number of simultaneous bidirectional
            peer edges a node can keep active (sum across TP/PP/DP degree allocations).
            0 => unlimited/ideal (no degree constraint).
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
        self.link_batch_sec = float(link_batch_ms) * 1e-3
        self.degree_k_total = int(degree_k_total)

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
        head_dim: int = 128,
        kv_dim: int = 1024,
        ffn_hidden: int = 53248,
        total_params: float = 405e9,
        bytes_per_act: int = 2,  # BF16
        bytes_per_param: int = 2,  # BF16
        bytes_per_grad: int = 4,  # FP32 for optimizer/grad buckets
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
        self.head_dim = int(head_dim)
        self.kv_dim = int(kv_dim)
        self.ffn_hidden = int(ffn_hidden)
        self.total_params = float(total_params)
        self.bytes_per_act = int(bytes_per_act)
        self.bytes_per_param = int(bytes_per_param)
        self.bytes_per_grad = int(bytes_per_grad)


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


def llama3_megatron_payloads(
    mod: ModelConfig, par: ParallelConfig
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


# ================= 2b. Degree/K + Link Retune Modeling =================

PeerSet = frozenset[str]


def _pp_peer_from_name(label: str, link_type: str) -> str:
    """Infer PP neighbor direction from the comm-node label.

    For asymmetric links, encode direction to model the need for separate peer
    circuits (recv vs send) when switching between FWD and BWD.
    """
    name = str(label).lower()
    is_fwd = "fwd" in name
    is_bwd = "bwd" in name
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
    sys: SystemConfig,
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
    node: CommNode, sys: SystemConfig, is_first: bool
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


def _trace_one_shot(
    strategy: str,
    nodes: List[CommNode],
    sys: SystemConfig,
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
    sys: SystemConfig,
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
            "degree_split": e.degree_split,
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
                "k_tp",
                "k_pp",
                "k_dp",
            ]
        )
        for e in events:
            bw = e.bw_share or {}
            k = e.degree_split or {}
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
                    str(int(k.get("tp", 0))) if "tp" in k else "0",
                    str(int(k.get("pp", 0))) if "pp" in k else "0",
                    str(int(k.get("dp", 0))) if "dp" in k else "0",
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
    COL_BW_RECFG = "#9AA0A6"
    COL_LINK_RECFG = "#B0B5BB"

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
        return COL_BW_RECFG

    dom_stack = ["tp", "pp", "dp"]  # bottom -> top

    def _active_domain_from_bw(bw: Dict[str, float]) -> str:
        # Prefer the highest share; break ties by dom_stack order.
        best = dom_stack[0]
        best_v = float(bw.get(best, 0.0))
        for d in dom_stack[1:]:
            v = float(bw.get(d, 0.0))
            if v > best_v:
                best = d
                best_v = v
        return best

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
            rc_ms = sum(float(e.duration_ms) for e in evs if e.kind != "comm")
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
            if e.kind == "link_internal":
                # Internal retune: full-height hatched block (not domain-colored).
                rect = patches.Rectangle(
                    (e.start_ms, y),
                    e.duration_ms,
                    h,
                    linewidth=0.8,
                    edgecolor="white",
                    facecolor=COL_LINK_RECFG,
                    hatch="///",
                    alpha=0.9,
                )
                ax.add_patch(rect)
            elif e.kind != "comm":
                # Keep reconfiguration cost as a full-height block (separate semantic from BW usage).
                reconfig_color = (
                    COL_BW_RECFG if e.kind == "bw_reconfig" else COL_LINK_RECFG
                )
                rect = patches.Rectangle(
                    (e.start_ms, y),
                    e.duration_ms,
                    h,
                    linewidth=0.8,
                    edgecolor="white",
                    facecolor=reconfig_color,
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

            # Keep labels sparse to avoid clutter.
            if e.kind != "comm":
                if e.duration_ms >= 0.6:
                    if e.kind == "bw_reconfig":
                        label = "R"
                    elif e.kind == "link_reconfig":
                        label = "L"
                    else:
                        label = "L*"
                    ax.text(
                        e.start_ms + e.duration_ms / 2,
                        y + h / 2,
                        label,
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
    fig.suptitle(title + note, fontsize=12, y=0.985)

    if params_text.strip():
        # Parameter block below the title (outside the plot area).
        fig.text(
            0.01,
            0.945,
            params_text,
            ha="left",
            va="top",
            fontsize=8,
            family="monospace",
            color="#222222",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="#dddddd",
                alpha=0.9,
            ),
        )
    legend = [
        patches.Patch(color=COL_BW_RECFG, label="BW reconfig (R)"),
        patches.Patch(color=COL_LINK_RECFG, label="Link reconfig (L)"),
        patches.Patch(
            facecolor="none",
            edgecolor=COL_LINK_RECFG,
            hatch="///",
            label="Internal retune (overlay)",
        ),
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
    if params_text.strip():
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    else:
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
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
    row_names = {
        "preplanned": "Preplanned",
        "one-shot": "One-shot",
        "static": "Even share",
    }
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


def _bw_boundary_ms_at(
    comm_nodes: List[CommNode], idx: int, sys: SystemConfig
) -> float:
    return exposed_boundary_ms(comm_nodes[idx].gap_before_ms, sys.reconfig_sec * 1000.0)


def _link_boundary_ms_at(
    comm_nodes: List[CommNode], idx: int, sys: SystemConfig
) -> float:
    return exposed_boundary_ms(
        comm_nodes[idx].gap_before_ms, sys.link_batch_sec * 1000.0
    )


def _best_degree_split_for_interval(
    comm_nodes: List[CommNode],
    p: int,
    q: int,
    sys: SystemConfig,
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
    sys: SystemConfig,
) -> Tuple[List[LinkSegmentPlan], float]:
    """Inner DP: choose link-only boundaries + degree splits inside a fixed BW segment [start_idx..end_idx]."""
    if start_idx > end_idx:
        return [], 0.0

    # Precompute comm-time prefix sums under fixed bw_share.
    t_node: List[float] = []
    for t in range(start_idx, end_idx + 1):
        t_node.append(_node_comm_time_ms(comm_nodes[t], bw_share, sys))
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
    sys: SystemConfig,
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
    sys: SystemConfig,
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
        "--link-batch-ms",
        type=float,
        default=0.0,
        help="Link-only boundary retune latency (ms). Applies when the planner inserts a link-only "
        "boundary (keeps BW split fixed but resets active peer edges). Also used for internal "
        "batching overhead via calc_batches(). (default: 0.0)",
    )
    parser.add_argument(
        "--degree-k-total",
        type=int,
        default=0,
        help="Total degree budget K: maximum number of simultaneous bidirectional peer edges "
        "a node can keep active across TP/PP/DP (k_tp+k_pp+k_dp<=K). "
        "0 => unlimited/ideal. (default: 0)",
    )
    parser.add_argument(
        "--bw-grid-step",
        type=float,
        default=0.01,
        help="Grid step for solving per-segment min-delay bw splits (continuous model only). "
        "Ignored when --unit-bw-gbps>0. (default: 0.01)",
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
        default=0.0,
        help="Disabled: short-event markers are no longer drawn (kept for compatibility).",
    )
    parser.add_argument(
        "--comm-trace-plot-ms-per-inch",
        type=float,
        default=400.0,
        help="Plot horizontal scale: milliseconds per inch. Smaller => wider plot and more visible "
        "tiny events (default: 400.0).",
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
        help="DEPRECATED/IGNORED: initial BW boundary at t=0 is handled uniformly via gap hiding "
        "for all strategies under the degree+link DP model.",
    )
    parser.add_argument(
        "--collective-profiles",
        type=str,
        nargs="+",
        default=["all"],
        choices=["mixed", "ring_asym", "ring_sym", "hypercube", "tree", "all"],
        help=(
            "Which collective-algorithm profile(s) to use for the *planning* section "
            "(preplanned/one-shot/static).\n\n"
            "- mixed     : (backwards-compatible default) TP=ring/asym, PP=p2p/asym, DP=hypercube (RH/RD)\n"
            "- ring_asym : ring/asymmetric for TP+DP, p2p/asymmetric for PP\n"
            "- ring_sym  : ring/symmetric for TP+DP, p2p/symmetric for PP\n"
            "- hypercube : TP+DP use RH/RD (1-peer-at-a-time) with symmetric links; PP uses p2p/symmetric\n"
            "- tree      : TP+DP use tree collectives (gather/bcast, reduce/scatter) with symmetric links; "
            "PP uses p2p/symmetric\n"
            "- all       : run ring_asym + ring_sym + hypercube + tree (and omit mixed)\n\n"
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
        link_batch_ms=args.link_batch_ms,
        degree_k_total=args.degree_k_total,
        max_peers_per_collective=args.max_peers_per_collective,
        peer_switch_us=args.peer_switch_us,
    )

    # Turn off the legacy intra-collective peer-switching penalty to avoid double counting
    # (link retunes are explicitly modeled via link-only boundaries and calc_batches()).
    if sys.peer_switch_sec > 0 and sys.max_peers_per_collective > 0:
        print(
            "[warn] legacy intra-collective peer switch penalty is enabled, but this script "
            "now models link retunes explicitly; disabling legacy penalty to avoid double-counting."
        )
    sys.max_peers_per_collective = 0
    sys.peer_switch_sec = 0.0

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
        if sys.total_bw_units is None:
            raise ValueError("unit_bw_GBps>0 but total_bw_units is None")
        total_units = sys.total_bw_units
        active_domains = [d for d, v in bw_share_requested.items() if v > 0]
        # Default minimum: try to keep >=1 unit for each active domain (if feasible).
        min_units = 1
        if total_units < min_units * len(active_domains):
            min_units = 0
        bw_share, bw_share_units = quantize_share_dict(
            bw_share_requested, total_units, min_units_per_domain=min_units
        )

    (
        a_full,
        a_shard,
        p_layer_tp_bf16,
        p_layer_tp_fp32,
        p_layer_total_params,
    ) = llama3_megatron_payloads(mod, par)
    ln_grad_bytes = 2 * mod.hidden * mod.bytes_per_grad

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
    dp_sendrecv_bytes_fp32 = (par.dp - 1) * p_layer_tp_fp32 / par.dp
    dp_sendrecv_bytes_bf16 = (par.dp - 1) * p_layer_tp_bf16 / par.dp

    print("=== Llama 3 405B (8K, CP=1) Communication Model ===")
    print(f"Network: {sys.bw_bytes_sec / 1e9:.0f} GB/s, {sys.latency_sec * 1e6:.1f} us")
    print(
        f"Boundaries: T_segment_reconfig={sys.reconfig_sec * 1e3:.3f} ms, "
        f"T_link_batch={sys.link_batch_sec * 1e3:.3f} ms"
    )
    deg_str = (
        "unlimited/ideal" if sys.degree_k_total <= 0 else str(int(sys.degree_k_total))
    )
    print(f"Degree budget: K_total={deg_str} (k_tp+k_pp+k_dp<=K_total)")
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
        "Payloads at bf16 (per layer, per TP rank): "
        "mixed precision with fp32 for DP RS gradients\n"
        f"A_full={_fmt_bytes(a_full)} | A_shard={_fmt_bytes(a_shard)} | "
        f"P_layer_tp_bf16={_fmt_bytes(p_layer_tp_bf16)} | "
        f"P_layer_tp_fp32={_fmt_bytes(p_layer_tp_fp32)}"
    )
    print(
        f"Per-layer params: total={p_layer_total_params:,d} elems | "
        f"per TP rank={int(p_layer_total_params / par.tp):,d} elems | "
        f"LN grads (TP allreduce)={_fmt_bytes(ln_grad_bytes)}"
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
                ("Tree (gather+bcast)", "tree", "symmetric"),
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
                ("Tree (reduce+scatter)", "tree", "symmetric"),
                ("Recursive Halving (RS)", "rh", "symmetric"),
            ],
        ),
        (
            f"PP P2P boundary (per transfer, 1 hop) TP-sharded, {_fmt_mib_compact(pp_p2p_bytes)} "
            "send/recv per layer",
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
            f"DP/ZeRO-2 ReduceScatter(dW) (per layer, nodes=DP), send/recv {_fmt_gib(dp_sendrecv_bytes_fp32)} gradients",
            "dp",
            p_layer_tp_fp32,
            par.dp,
            "reducescatter",
            [
                ("Ring Asym", "ring", "asymmetric"),
                ("Ring Sym", "ring", "symmetric"),
                ("Tree (reduce+scatter)", "tree", "symmetric"),
                ("Recursive Halving (RS)", "rh", "symmetric"),
            ],
        ),
        (
            f"DP/ZeRO-2 AllGather(W) (per layer, nodes=DP), send/recv {_fmt_gib(dp_sendrecv_bytes_bf16)} parameters",
            "dp",
            p_layer_tp_bf16,
            par.dp,
            "allgather",
            [
                ("Ring Asym", "ring", "asymmetric"),
                ("Ring Sym", "ring", "symmetric"),
                ("Tree (gather+bcast)", "tree", "symmetric"),
                ("Recursive Doubling (AG)", "rd", "symmetric"),
            ],
        ),
        (
            "AllReduce (generic) (per call)",
            "dp",
            p_layer_tp_bf16,
            par.dp,
            "allreduce",
            [
                ("Ring Asym", "ring", "asymmetric"),
                ("Ring Sym", "ring", "symmetric"),
                ("Tree (reduce+bcast)", "tree", "symmetric"),
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
    # Megatron schedule: 2x AG + 2x RS in forward, and 2x AG + 2x RS in backward per layer.
    tp_calls_per_layer = {"allgather": 4, "reducescatter": 4}

    def _choices_for_profile(profile: str) -> Dict[str, Tuple[str, str]]:
        """Return algo/link_type choices for the planning schedule.

        Keys:
          - tp_allgather, tp_reducescatter, tp_allreduce,
            pp_p2p, dp_reducescatter, dp_allgather
        """
        prof = str(profile).strip().lower()
        if prof == "mixed":
            # Backwards-compatible default: ring/asym for TP+PP, hypercube (RH/RD) for DP.
            return {
                "tp_allgather": ("ring", "asymmetric"),
                "tp_reducescatter": ("ring", "asymmetric"),
                "tp_allreduce": ("ring", "asymmetric"),
                "pp_p2p": ("p2p", "asymmetric"),
                "dp_reducescatter": ("rh", "symmetric"),
                "dp_allgather": ("rd", "symmetric"),
            }
        if prof == "ring_asym":
            return {
                "tp_allgather": ("ring", "asymmetric"),
                "tp_reducescatter": ("ring", "asymmetric"),
                "tp_allreduce": ("ring", "asymmetric"),
                "pp_p2p": ("p2p", "asymmetric"),
                "dp_reducescatter": ("ring", "asymmetric"),
                "dp_allgather": ("ring", "asymmetric"),
            }
        if prof == "ring_sym":
            return {
                "tp_allgather": ("ring", "symmetric"),
                "tp_reducescatter": ("ring", "symmetric"),
                "tp_allreduce": ("ring", "symmetric"),
                "pp_p2p": ("p2p", "symmetric"),
                "dp_reducescatter": ("ring", "symmetric"),
                "dp_allgather": ("ring", "symmetric"),
            }
        if prof == "hypercube":
            # "1-peer-at-a-time" collectives modeled as eta=1.0 (see get_efficiency()).
            return {
                "tp_allgather": ("rd", "symmetric"),
                "tp_reducescatter": ("rh", "symmetric"),
                "tp_allreduce": ("recursive_doubling", "symmetric"),
                "pp_p2p": ("p2p", "symmetric"),
                "dp_reducescatter": ("rh", "symmetric"),
                "dp_allgather": ("rd", "symmetric"),
            }
        if prof == "tree":
            # Tree collectives: modeled as gather/bcast, reduce/scatter, and reduce/bcast.
            # We use symmetric links (tree doesn't rely on the ring-style Tx/Rx coupling trick).
            return {
                "tp_allgather": ("tree", "symmetric"),
                "tp_reducescatter": ("tree", "symmetric"),
                "tp_allreduce": ("tree", "symmetric"),
                "pp_p2p": ("p2p", "symmetric"),
                "dp_reducescatter": ("tree", "symmetric"),
                "dp_allgather": ("tree", "symmetric"),
            }
        raise ValueError(f"Unknown collective profile: {profile}")

    def _profiles_to_run() -> List[str]:
        req = [str(x).strip().lower() for x in (args.collective_profiles or ["mixed"])]
        if "all" in req:
            # Requested sweep over canonical profiles (excluding mixed unless explicitly specified).
            base = ["ring_asym", "ring_sym", "hypercube", "tree"]
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
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Return (tp_total_ms, pp_total_ms, dp_total_ms, t_tp_ag_ms, t_tp_rs_ms, t_tp_ar_ms)
        for one step under the Megatron per-layer schedule.
        """
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
        t_tp_ar = estimate_time_ms(
            ln_grad_bytes,
            par.tp,
            bw_tp,
            "allreduce",
            choices["tp_allreduce"][0],
            choices["tp_allreduce"][1],
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
            p_layer_tp_fp32,
            par.dp,
            bw_dp,
            "reducescatter",
            choices["dp_reducescatter"][0],
            choices["dp_reducescatter"][1],
            sys,
        )
        t_dp_ag = estimate_time_ms(
            p_layer_tp_bf16,
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
                + 1.0 * t_tp_ar
            )
        )
        # PP: recv+send in forward, recv+send in backward per layer.
        pp_total = par.microbatches_per_step * mod.layers * (4.0 * t_pp_one)
        # DP ZeRO-2 (no bucketization): per-layer RS + AG.
        dp_total = par.microbatches_per_step * mod.layers * (t_dp_rs + t_dp_ag)
        return tp_total, pp_total, dp_total, t_tp_ag, t_tp_rs, t_tp_ar

    # Per-step totals will be printed per collective profile later (so we can compare profiles).

    # ---- Pre-planned max utilization strategy ----
    #
    # - "When to reconfigure": global DP partitioning over the known comm-node sequence.
    # - "How much bw per domain": within each segment, choose bw shares that minimize the
    #   modeled collective completion time (estimate_time_ms), NOT traffic ratios.

    def _print_segments(
        tag: str,
        nodes: List[CommNode],
        segments: List[BWSegmentPlan],
        seg_sys: SystemConfig,
    ) -> None:
        print("\n" + "=" * 78)
        print(f"Pre-planned segmentation result ({tag}):")
        total_bw_rc = sum(float(s.exposed_bw_boundary_ms) for s in segments)
        total_link_rc = sum(float(s.exposed_link_boundaries_ms) for s in segments)
        total_internal = sum(float(s.internal_retune_ms) for s in segments)
        total_comm = sum(float(s.comm_time_ms) for s in segments)
        print(
            f"BW_segments={len(segments)} | "
            f"R_BW={total_bw_rc:.3f} ms | "
            f"R_link={total_link_rc:.3f} ms | "
            f"internal_link={total_internal:.3f} ms | "
            f"comm={total_comm:.3f} ms | "
            f"total={(total_bw_rc + total_link_rc + total_internal + total_comm):.3f} ms"
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
                f"  BWseg{k}: nodes[{s.start_idx + 1}..{s.end_idx + 1}] domains={doms}  "
                f"R_BW={s.exposed_bw_boundary_ms:.3f} ms  "
                f"R_link={s.exposed_link_boundaries_ms:.3f} ms  "
                f"internal_link={s.internal_retune_ms:.3f} ms  "
                f"comm={s.comm_time_ms:.3f} ms  "
                f"total={s.total_ms:.3f} ms  "
                f"bw: {bw_str}"
            )
            if seg_sys.degree_k_total != 0 and s.link_segments:
                for li, ls in enumerate(s.link_segments, start=1):
                    ks = ", ".join(
                        [
                            f"{d}={int(ls.degree_split.get(d, 0))}"
                            for d in ["tp", "pp", "dp"]
                        ]
                    )
                    print(
                        f"    L{li}: nodes[{ls.start_idx + 1}..{ls.end_idx + 1}]  "
                        f"R_L={ls.exposed_link_boundary_ms:.3f} ms  "
                        f"internal={ls.internal_retune_ms:.3f} ms  "
                        f"comm={ls.comm_time_ms:.3f} ms  "
                        f"total={ls.total_ms:.3f} ms  "
                        f"k: {ks}"
                    )

    def _build_llama_nodes(choices: Dict[str, Tuple[str, str]]) -> List[CommNode]:
        """Build a Llama 3 405B Megatron-style comm-node sequence (All-Forward, All-Backward)."""
        nodes: List[CommNode] = []

        # Use a small representative window (2 microbatches × 2 layers) to capture
        # boundary effects (first/last layer, inter-layer, and PP handoff) without full expansion.
        mbs_count = min(2, int(par.microbatches_per_step))
        layers_per_stage = min(2, int(math.ceil(mod.layers / par.pp)))

        # Sizes in bytes
        MiB = 1024**2
        # User: "rank activation=32MiB" (Shard)
        a_shard_bytes = 32 * MiB
        # User: "Full activation" = 256 MiB (derived from 32 * 8 TP)
        a_full_bytes = 256 * MiB

        # 1. Forward Pass 1..16
        for mb in range(mbs_count):
            # a. PP P2P Recv
            nodes.append(
                CommNode(
                    name=f"FWD[{mb}]:PP:Recv",
                    domain="pp",
                    payload_bytes=a_shard_bytes,
                    nodes=2,
                    op="p2p",
                    algo=str(choices["pp_p2p"][0]),
                    link_type=str(choices["pp_p2p"][1]),
                    count=1,
                    gap_before_ms=0.0,
                )
            )

            for lyr in range(layers_per_stage):
                # b. TP AG (QKV)
                nodes.append(
                    CommNode(
                        name=f"FWD[{mb}]:L{lyr}:TP:AG(QKV)",
                        domain="tp",
                        payload_bytes=a_full_bytes,
                        nodes=par.tp,
                        op="allgather",
                        algo=str(choices["tp_allgather"][0]),
                        link_type=str(choices["tp_allgather"][1]),
                        count=1,
                        gap_before_ms=0.0,
                    )
                )
                # d. TP RS (AttnOut) - User: "TP RS, 256->32"
                nodes.append(
                    CommNode(
                        name=f"FWD[{mb}]:L{lyr}:TP:RS(AttnOut)",
                        domain="tp",
                        payload_bytes=a_full_bytes,
                        nodes=par.tp,
                        op="reducescatter",
                        algo=str(choices["tp_reducescatter"][0]),
                        link_type=str(choices["tp_reducescatter"][1]),
                        count=1,
                        gap_before_ms=0.0,
                    )
                )
                # e. TP AG (MLP)
                nodes.append(
                    CommNode(
                        name=f"FWD[{mb}]:L{lyr}:TP:AG(MLP)",
                        domain="tp",
                        payload_bytes=a_full_bytes,
                        nodes=par.tp,
                        op="allgather",
                        algo=str(choices["tp_allgather"][0]),
                        link_type=str(choices["tp_allgather"][1]),
                        count=1,
                        gap_before_ms=0.0,
                    )
                )
                # g. TP RS (MLPOut)
                nodes.append(
                    CommNode(
                        name=f"FWD[{mb}]:L{lyr}:TP:RS(MLPOut)",
                        domain="tp",
                        payload_bytes=a_full_bytes,
                        nodes=par.tp,
                        op="reducescatter",
                        algo=str(choices["tp_reducescatter"][0]),
                        link_type=str(choices["tp_reducescatter"][1]),
                        count=1,
                        gap_before_ms=0.0,
                    )
                )

            # h. PP P2P Send
            nodes.append(
                CommNode(
                    name=f"FWD[{mb}]:PP:Send",
                    domain="pp",
                    payload_bytes=a_shard_bytes,
                    nodes=2,
                    op="p2p",
                    algo=str(choices["pp_p2p"][0]),
                    link_type=str(choices["pp_p2p"][1]),
                    count=1,
                    gap_before_ms=0.0,
                )
            )

        # 3. Backward Pass 1..16
        for mb in range(mbs_count):
            # a. PP Recv Grad
            nodes.append(
                CommNode(
                    name=f"BWD[{mb}]:PP:Recv",
                    domain="pp",
                    payload_bytes=a_shard_bytes,
                    nodes=2,
                    op="p2p",
                    algo=str(choices["pp_p2p"][0]),
                    link_type=str(choices["pp_p2p"][1]),
                    count=1,
                    gap_before_ms=0.0,
                )
            )

            for lyr in range(layers_per_stage):
                # b. TP AG (FC2 grad)
                nodes.append(
                    CommNode(
                        name=f"BWD[{mb}]:L{lyr}:TP:AG(FC2_g)",
                        domain="tp",
                        payload_bytes=a_full_bytes,
                        nodes=par.tp,
                        op="allgather",
                        algo=str(choices["tp_allgather"][0]),
                        link_type=str(choices["tp_allgather"][1]),
                        count=1,
                        gap_before_ms=0.0,
                    )
                )
                # c. TP AG (FC1 wgrad activation)
                nodes.append(
                    CommNode(
                        name=f"BWD[{mb}]:L{lyr}:TP:AG(FC1_in)",
                        domain="tp",
                        payload_bytes=a_full_bytes,
                        nodes=par.tp,
                        op="allgather",
                        algo=str(choices["tp_allgather"][0]),
                        link_type=str(choices["tp_allgather"][1]),
                        count=1,
                        gap_before_ms=0.0,
                    )
                )
                # d. TP RS (FC1 dgrad)
                nodes.append(
                    CommNode(
                        name=f"BWD[{mb}]:L{lyr}:TP:RS(FC1_dg)",
                        domain="tp",
                        payload_bytes=a_full_bytes,
                        nodes=par.tp,
                        op="reducescatter",
                        algo=str(choices["tp_reducescatter"][0]),
                        link_type=str(choices["tp_reducescatter"][1]),
                        count=1,
                        gap_before_ms=0.0,
                    )
                )
                # e. TP AG (Proj grad)
                nodes.append(
                    CommNode(
                        name=f"BWD[{mb}]:L{lyr}:TP:AG(Proj_g)",
                        domain="tp",
                        payload_bytes=a_full_bytes,
                        nodes=par.tp,
                        op="allgather",
                        algo=str(choices["tp_allgather"][0]),
                        link_type=str(choices["tp_allgather"][1]),
                        count=1,
                        gap_before_ms=0.0,
                    )
                )
                # f. TP AG (QKV wgrad activation)
                nodes.append(
                    CommNode(
                        name=f"BWD[{mb}]:L{lyr}:TP:AG(QKV_in)",
                        domain="tp",
                        payload_bytes=a_full_bytes,
                        nodes=par.tp,
                        op="allgather",
                        algo=str(choices["tp_allgather"][0]),
                        link_type=str(choices["tp_allgather"][1]),
                        count=1,
                        gap_before_ms=0.0,
                    )
                )
                # g. TP RS (QKV dgrad)
                nodes.append(
                    CommNode(
                        name=f"BWD[{mb}]:L{lyr}:TP:RS(QKV_dg)",
                        domain="tp",
                        payload_bytes=a_full_bytes,
                        nodes=par.tp,
                        op="reducescatter",
                        algo=str(choices["tp_reducescatter"][0]),
                        link_type=str(choices["tp_reducescatter"][1]),
                        count=1,
                        gap_before_ms=0.0,
                    )
                )

            # h. PP Send Grad
            nodes.append(
                CommNode(
                    name=f"BWD[{mb}]:PP:Send",
                    domain="pp",
                    payload_bytes=a_shard_bytes,
                    nodes=2,
                    op="p2p",
                    algo=str(choices["pp_p2p"][0]),
                    link_type=str(choices["pp_p2p"][1]),
                    count=1,
                    gap_before_ms=0.0,
                )
            )

        # 5. TP AllReduce LayerNorm (accumulated once per step or layer? User: <20MB)
        nodes.append(
            CommNode(
                name="OPT:TP:AR(LN)",
                domain="tp",
                payload_bytes=20 * MiB,
                nodes=par.tp,
                op="allreduce",
                algo=str(choices["tp_allreduce"][0]),
                link_type=str(choices["tp_allreduce"][1]),
                count=1,
                gap_before_ms=0.0,
            )
        )

        # 6. DP ZeRO-2 ReduceScatter
        # User: 760 MiB * layers_per_stage * 2 (fp32)
        dp_grad_bytes = 760 * MiB * layers_per_stage * 2
        nodes.append(
            CommNode(
                name="OPT:DP:RS(Grads)",
                domain="dp",
                payload_bytes=dp_grad_bytes,
                nodes=par.dp,
                op="reducescatter",
                algo=str(choices["dp_reducescatter"][0]),
                link_type=str(choices["dp_reducescatter"][1]),
                count=1,
                gap_before_ms=0.0,
            )
        )

        return nodes

    def _one_shot_total(
        nodes: List[CommNode], seg_sys: SystemConfig
    ) -> Tuple[Dict[str, float], BWSegmentPlan]:
        """Single BW segment over the whole schedule; inner DP may insert link-only boundaries."""
        bw, units, _comm_ms = solve_min_delay_bw_split(
            nodes, seg_sys, bw_grid_step=float(args.bw_grid_step)
        )
        bw_seg = solve_best_link_plan_for_bw_segment(
            nodes, 0, len(nodes) - 1, bw_share=bw, bw_units=units, sys=seg_sys
        )
        return bw, bw_seg

    def _static_total(
        nodes: List[CommNode],
        seg_sys: SystemConfig,
        bw: Dict[str, float],
        bw_units: Dict[str, int] | None,
    ) -> BWSegmentPlan:
        return solve_best_link_plan_for_bw_segment(
            nodes, 0, len(nodes) - 1, bw_share=bw, bw_units=bw_units, sys=seg_sys
        )

    if not args.no_preplanned:
        profiles = _profiles_to_run()
        # If we're sweeping multiple profiles, write ONE combined PNG with a shared time scale.
        _combine_png = (
            bool(args.emit_comm_trace)
            and (not args.comm_trace_no_png)
            and (len(profiles) > 1)
        )
        _combined_rows_by_strategy: Dict[str, List[Tuple[str, List[TraceEvent]]]] = {
            "preplanned": [],
            "one-shot": [],
            "static": [],
        }
        _combined_rows: List[Tuple[str, List[TraceEvent]]] = []
        _combined_params_one_shot_bw: Dict[str, Dict[str, float]] = {}
        for prof in profiles:
            choices = _choices_for_profile(prof)
            tag = f"llama3-step-megatron | profile={prof}"

            # Compute per-step totals for this profile under the current static bw_share.
            tp_total, pp_total, dp_total, t_tp_ag, t_tp_rs, t_tp_ar = _per_step_totals(
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
                f"(per-call AG {t_tp_ag:.3f} ms, RS {t_tp_rs:.3f} ms, AR {t_tp_ar:.3f} ms; "
                f"{mod.layers} layers, {par.microbatches_per_step} mbs)"
            )
            print(f"PP total: {pp_total / 1000:.3f} s  (per-layer fwd+bwd P2P)")
            print(
                f"DP total: {dp_total / 1000:.3f} s  (per-layer RS+AG, no bucketization)"
            )

            llama_nodes = _build_llama_nodes(choices=choices)
            segments = preplanned_dp_partition(
                llama_nodes, sys, bw_grid_step=float(args.bw_grid_step)
            )
            _print_segments(tag, llama_nodes, segments, sys)

            bw_all, one_shot_seg = _one_shot_total(llama_nodes, sys)
            static_seg = _static_total(llama_nodes, sys, bw_share, bw_share_units)

            print("\n" + "-" * 78)
            print(f"Pre-planned vs one-shot vs static(equal-share) [{tag}]:")
            pre_bw_rc = sum(float(s.exposed_bw_boundary_ms) for s in segments)
            pre_link_rc = sum(float(s.exposed_link_boundaries_ms) for s in segments)
            pre_internal = sum(float(s.internal_retune_ms) for s in segments)
            pre_comm = sum(float(s.comm_time_ms) for s in segments)
            pre_total = pre_bw_rc + pre_link_rc + pre_internal + pre_comm
            print(
                f"  preplanned: comm={pre_comm:.3f} ms  internal_link={pre_internal:.3f} ms  "
                f"R_link={pre_link_rc:.3f} ms  R_BW={pre_bw_rc:.3f} ms  total={pre_total:.3f} ms"
            )

            os_total = float(one_shot_seg.total_ms)
            print(
                f"  one-shot  : comm={one_shot_seg.comm_time_ms:.3f} ms  internal_link={one_shot_seg.internal_retune_ms:.3f} ms  "
                f"R_link={one_shot_seg.exposed_link_boundaries_ms:.3f} ms  R_BW={one_shot_seg.exposed_bw_boundary_ms:.3f} ms  "
                f"total={os_total:.3f} ms  "
                f"bw={', '.join([f'{d}={bw_all[d]:.3f}' for d in ['tp', 'pp', 'dp']])}"
            )

            st_total = float(static_seg.total_ms)
            print(
                f"  static    : comm={static_seg.comm_time_ms:.3f} ms  internal_link={static_seg.internal_retune_ms:.3f} ms  "
                f"R_link={static_seg.exposed_link_boundaries_ms:.3f} ms  R_BW={static_seg.exposed_bw_boundary_ms:.3f} ms  "
                f"total={st_total:.3f} ms  (bw_share tp/pp/dp all = {bw_share['tp']:.3f})"
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
                        _combined_rows_by_strategy["preplanned"].append(
                            (f"{prof} | Preplanned", preplanned_trace)
                        )
                        _combined_rows_by_strategy["one-shot"].append(
                            (f"{prof} | One-shot", one_shot_trace)
                        )
                        _combined_rows_by_strategy["static"].append(
                            (f"{prof} | Even share", static_trace)
                        )
                        _combined_params_one_shot_bw[str(prof)] = dict(one_shot_bw)
                    else:
                        # Build a concise parameter block for the figure.
                        params_lines: List[str] = []
                        params_lines.append(
                            f"profile={prof}  schedule=megatron  nodes: TP={par.tp}, PP={par.pp}, DP={par.dp}"
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

        if _combine_png:
            _combined_rows = (
                _combined_rows_by_strategy["preplanned"]
                + _combined_rows_by_strategy["one-shot"]
                + _combined_rows_by_strategy["static"]
            )

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
                f"profiles={','.join([str(p) for p in profiles])}  schedule=megatron  nodes: TP={par.tp}, PP={par.pp}, DP={par.dp}"
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
