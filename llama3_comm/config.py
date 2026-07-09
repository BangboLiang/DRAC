"""Configuration classes for system, model, and parallelism parameters."""

from __future__ import annotations


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
        bandwidth_tx_GBps: float | None = None,
        bandwidth_rx_GBps: float | None = None,
        ports: int = 1,
        objective_dp_gap_overlap: bool = False,
    ) -> None:
        """System/network parameters.

        :param bandwidth_GBps: legacy per-direction injection bandwidth (GB/s).
            When bandwidth_tx_GBps/bandwidth_rx_GBps are not provided, both Tx and
            Rx budgets default to bandwidth_GBps.
        :param bandwidth_tx_GBps: optional total transmit injection budget (GB/s).
        :param bandwidth_rx_GBps: optional total receive injection budget (GB/s).
        :param ports: number of ports per node (used by Level-2+ event models).
        :param objective_dp_gap_overlap: if True, the DP solver objective treats DP
            communication as partially hideable by per-node n.gap_before_ms.
            This is an approximation used by overlap-aware bucketing scripts.
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
        self.bandwidth_tx_GBps = (
            float(bandwidth_tx_GBps)
            if bandwidth_tx_GBps is not None
            else float(bandwidth_GBps)
        )
        self.bandwidth_rx_GBps = (
            float(bandwidth_rx_GBps)
            if bandwidth_rx_GBps is not None
            else float(bandwidth_GBps)
        )
        self.ports = int(ports)
        self.latency_us = float(latency_us)

        # Legacy single-budget field kept for backwards compatibility.
        self.bw_bytes_sec = self.bandwidth_GBps * 1e9
        # Explicit duplex budgets (used by event simulator Level 1+).
        self.bw_tx_bytes_sec = self.bandwidth_tx_GBps * 1e9
        self.bw_rx_bytes_sec = self.bandwidth_rx_GBps * 1e9
        self.latency_sec = self.latency_us * 1e-6

        self.unit_bw_GBps = float(unit_bw_GBps)
        self.asym_min_reverse_units = int(asym_min_reverse_units)

        self.reconfig_sec = float(reconfig_ms) * 1e-3
        self.link_batch_sec = float(link_batch_ms) * 1e-3
        self.degree_k_total = int(degree_k_total)

        self.max_peers_per_collective = int(max_peers_per_collective)
        self.peer_switch_sec = float(peer_switch_us) * 1e-6

        self.objective_dp_gap_overlap = bool(objective_dp_gap_overlap)

        # Precompute number of discrete units ("lanes") if enabled.
        # Note: units are derived from the legacy bandwidth_GBps for backwards
        # compatibility. Level-3+ models may want to add separate tx/rx units.
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
