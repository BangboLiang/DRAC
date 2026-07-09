import math

# ================= 1. System & Model Configuration =================


class SystemConfig:
    def __init__(self, bandwidth_GBps, latency_us):
        # 200 GB/s per direction (e.g., 1.6T NIC)
        self.bw_bytes_sec = bandwidth_GBps * 1e9
        self.latency_sec = latency_us * 1e-6


class ModelConfig:
    def __init__(
        self,
        layers=126,
        hidden=16384,
        seq=8192,
        total_params=405e9,
        bytes_per_act=2,  # BF16
        bytes_per_param=2,  # BF16
    ):
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
        tp=8,
        pp=16,
        dp=128,  # ZeRO-2/DP group size
        global_batch_seqs=2048,
        microbatch_seqs=1,
    ):
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


def llama3_405b_payloads(mod: ModelConfig, par: ParallelConfig):
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


def get_efficiency(pattern, link_type):
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


def estimate_time_ms(payload_bytes, nodes, b_d, op, algo, link_type, sys: SystemConfig):
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

    # Effective bandwidth (shared pool scaled by b_d, then pattern efficiency eta).
    bw_budget = b_d * sys.bw_bytes_sec
    pattern = "ring" if algo == "ring" else ("p2p" if op == "p2p" else "1peer")
    eta = get_efficiency(pattern, link_type)
    bw_eff = eta * bw_budget

    M = float(payload_bytes)

    if op == "p2p":
        steps = 1
        total_sent = M
        return (steps * sys.latency_sec + total_sent / bw_eff) * 1000

    if op == "allgather":
        if algo == "ring":
            steps = nodes - 1
            total_sent = M * (nodes - 1) / nodes
            return (steps * sys.latency_sec + total_sent / bw_eff) * 1000
        if algo == "rd":
            steps = _ceil_log2(nodes)
            # Bandwidth is the same order as ring: M(1-1/p); latency is better.
            total_sent = M * (nodes - 1) / nodes
            return (steps * sys.latency_sec + total_sent / bw_eff) * 1000
        raise ValueError(f"Unsupported allgather algo: {algo}")

    if op == "reducescatter":
        if algo == "ring":
            steps = nodes - 1
            total_sent = M * (nodes - 1) / nodes
            return (steps * sys.latency_sec + total_sent / bw_eff) * 1000
        if algo == "rh":
            steps = _ceil_log2(nodes)
            total_sent = M * (nodes - 1) / nodes
            return (steps * sys.latency_sec + total_sent / bw_eff) * 1000
        raise ValueError(f"Unsupported reducescatter algo: {algo}")

    if op == "allreduce":
        if algo == "ring":
            steps = 2 * (nodes - 1)
            # Ring allreduce ~= reduce-scatter + allgather: 2M(1-1/p).
            total_sent = 2 * M * (nodes - 1) / nodes
            return (steps * sys.latency_sec + total_sent / bw_eff) * 1000
        if algo == "rabenseifner":
            steps = 2 * _ceil_log2(nodes)
            total_sent = 2 * M * (nodes - 1) / nodes
            return (steps * sys.latency_sec + total_sent / bw_eff) * 1000
        if algo in ["recursive_doubling", "rd_allreduce"]:
            # Classic recursive-doubling allreduce: log2(p) rounds exchanging full M each round.
            steps = _ceil_log2(nodes)
            total_sent = steps * M
            return (steps * sys.latency_sec + total_sent / bw_eff) * 1000
        raise ValueError(f"Unsupported allreduce algo: {algo}")

    raise ValueError(f"Unsupported op: {op}")


# ================= 3. Execution & Comparison =================

# Setup: keep network configuration as-is.
# sys = SystemConfig(200, 2.0)  # 200 GB/s link, 2 us latency
sys = SystemConfig(240, 2.0)  # 200 GB/s link, 2 us latency

# Llama 3 405B 8K (CP=1) inputs.
mod = ModelConfig(layers=126, hidden=16384, seq=8192, total_params=405e9)
par = ParallelConfig(tp=8, pp=16, dp=128, global_batch_seqs=2048, microbatch_seqs=1)

# Bandwidth shares: portions from the same total BW pool.
# You can edit these if you want to bias one domain.
bw_share = {
    "tp": 1.0 / 3.0,
    "pp": 1.0 / 3.0,
    "dp": 1.0 / 3.0,
}

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
        speed = (t_base / t) if t > 0 else 0.0
        print(f"{name:<28} | {t:10.3f} | {speed:10.2f}x")

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
    * (tp_calls_per_layer["allgather"] * t_tp_ag + tp_calls_per_layer["reducescatter"] * t_tp_rs)
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
