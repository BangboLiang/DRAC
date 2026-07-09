#!/usr/bin/env python3
"""
Llama 3 405B communication model with bandwidth granularity support.

Unified entry point using the modular llama3_comm package.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

from llama3_comm import (
    BWSegmentPlan,
    CommNode,
    GiB,
    MiB,
    ModelConfig,
    ParallelConfig,
    SystemConfig,
    TraceEvent,
    effective_bandwidth_GBps,
    estimate_time_ms,
    fast_preplanned_partition,
    llama3_megatron_payloads,
    preplanned_dp_partition,
    quantize_share_dict,
    solve_best_link_plan_for_bw_segment,
    solve_min_delay_bw_split,
    try_plot_rows_png,
    try_plot_trace_png,
    write_trace_csv,
    write_trace_json,
)
from llama3_comm.execution import _trace_from_segments
from llama3_comm.solvers import _trace_one_shot, _trace_static


def _fmt_bytes(x: float) -> str:
    if x >= GiB:
        return f"{x / GiB:.2f} GiB"
    if x >= MiB:
        return f"{x / MiB:.2f} MiB"
    return f"{x:.0f} B"


def _fmt_mib_compact(x_bytes: float) -> str:
    return f"{x_bytes / MiB:.0f}MiB"


def _fmt_gib(x_bytes: float) -> str:
    return f"{x_bytes / GiB:.2f} GiB"


def _choices_for_profile(profile: str) -> Dict[str, Tuple[str, str]]:
    """Return algo/link_type choices for the planning schedule."""
    prof = str(profile).strip().lower()
    if prof == "mixed":
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
        return {
            "tp_allgather": ("rd", "symmetric"),
            "tp_reducescatter": ("rh", "symmetric"),
            "tp_allreduce": ("recursive_doubling", "symmetric"),
            "pp_p2p": ("p2p", "symmetric"),
            "dp_reducescatter": ("rh", "symmetric"),
            "dp_allgather": ("rd", "symmetric"),
        }
    if prof == "tree":
        return {
            "tp_allgather": ("tree", "symmetric"),
            "tp_reducescatter": ("tree", "symmetric"),
            "tp_allreduce": ("tree", "symmetric"),
            "pp_p2p": ("p2p", "symmetric"),
            "dp_reducescatter": ("tree", "symmetric"),
            "dp_allgather": ("tree", "symmetric"),
        }
    raise ValueError(f"Unknown collective profile: {profile}")


def _profiles_to_run(collective_profiles: List[str]) -> List[str]:
    req = [str(x).strip().lower() for x in (collective_profiles or ["mixed"])]
    if "all" in req:
        base = ["ring_asym", "ring_sym", "hypercube", "tree"]
        extras = [p for p in req if p not in ["all"]]
        out: List[str] = []
        for p in base + extras:
            if p and p not in out:
                out.append(p)
        return out
    out = []
    for p in req:
        if p and p not in out:
            out.append(p)
    return out or ["mixed"]


def _build_llama_nodes(
    choices: Dict[str, Tuple[str, str]],
    mod: ModelConfig,
    par: ParallelConfig,
    *,
    mbs_count: int = 2,
    layers_per_stage: int = 2,
    bwd_last_mb_extra_layers: int = 0,
    dp_bucket_bytes: int = 0,
    dp_bucket_gap_ms: float = 0.0,
) -> List[CommNode]:
    """Build a Llama 3 405B Megatron-style comm-node sequence.

    Notes:
    - This script historically truncates the schedule for speed (few microbatches, few layers).
    - For overlap/bucketing studies we optionally add extra bwd layers on the *last* microbatch
      to create more DP launch points without blowing up total node count.
    - DP bucketing here is a coarse approximation: bucket comm nodes are inserted at bwd layer
      boundaries with a configurable gap_before_ms.
    """
    nodes: List[CommNode] = []

    mbs_count = min(int(mbs_count), int(par.microbatches_per_step))
    layers_per_stage = min(
        int(layers_per_stage), int(math.ceil(mod.layers / 16))
    )  # Assume PP=16

    a_shard_bytes = 32 * MiB
    a_full_bytes = 256 * MiB

    # 1. Forward Pass
    for mb in range(mbs_count):
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

    # 2. Backward Pass
    # Optionally insert DP buckets during the last microbatch to model overlap.
    if dp_bucket_bytes > 0:
        (
            _,
            _,
            p_layer_tp_bf16,
            p_layer_tp_fp32,
            _,
        ) = llama3_megatron_payloads(mod, par)

        # ZeRO-2: gradients RS in FP32; weights AG in BF16.
        dp_bytes_rs = float(p_layer_tp_fp32)
        dp_bytes_ag = float(p_layer_tp_bf16)

        # We model buckets for the whole (truncated) step-end payload:
        # - RS roughly corresponds to all bwd layers we explicitly modeled.
        # - AG is modeled once after bwd.
        layers_fwd_modeled = layers_per_stage * mbs_count
        layers_bwd_modeled = layers_per_stage * (mbs_count - 1) + (
            layers_per_stage + int(max(0, bwd_last_mb_extra_layers))
        )
        layers_modeled = layers_fwd_modeled + layers_bwd_modeled

        dp_bytes_rs_total = dp_bytes_rs * float(layers_modeled)
        dp_bytes_ag_total = dp_bytes_ag * float(layers_modeled)

        dp_rs_buckets = max(
            1, int(math.ceil(dp_bytes_rs_total / float(dp_bucket_bytes)))
        )
        dp_ag_buckets = max(
            1, int(math.ceil(dp_bytes_ag_total / float(dp_bucket_bytes)))
        )

        # Track totals for building the residual last bucket sizes.
        dp_bytes_rs = float(dp_bytes_rs_total)
        dp_bytes_ag = float(dp_bytes_ag_total)
    else:
        dp_rs_buckets = 0
        dp_ag_buckets = 0
        dp_bytes_rs = 0.0
        dp_bytes_ag = 0.0

    for mb in range(mbs_count):
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

        bwd_layers = layers_per_stage
        if mb == (mbs_count - 1):
            bwd_layers += int(max(0, bwd_last_mb_extra_layers))

        for lyr in range(bwd_layers):
            # For the extra bwd layers (only used to create more DP launch points), we do not
            # add TP comm nodes to keep the schedule short; treat them as compute-only.
            is_extra_layer = bool(lyr >= layers_per_stage)

            if not is_extra_layer:
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

            # DP buckets launched during the last microbatch's backward.
            # We treat dp_bucket_gap_ms as the compute window between bucket launches.
            if dp_bucket_bytes > 0 and mb == (mbs_count - 1):
                # Split dp_rs_buckets across bwd layers, putting the residual bucket in the last layer.
                base = int(dp_rs_buckets // max(1, bwd_layers))
                rem = int(dp_rs_buckets % max(1, bwd_layers))
                layer_bucket_cnt = base + (1 if lyr < rem else 0)
                if layer_bucket_cnt <= 0:
                    continue

                # Reserve one bucket for the global residual size in the last layer that has buckets.
                residual_layer = max(0, int(min(bwd_layers - 1, bwd_layers - 1)))
                is_residual_layer = bool(lyr == residual_layer and layer_bucket_cnt > 0)

                full_cnt = (
                    int(layer_bucket_cnt - 1)
                    if is_residual_layer
                    else int(layer_bucket_cnt)
                )
                if full_cnt > 0:
                    nodes.append(
                        CommNode(
                            name=f"BWD[{mb}]:L{lyr}:DP:RS(dW_bucket)",
                            domain="dp",
                            payload_bytes=float(dp_bucket_bytes),
                            nodes=par.dp,
                            op="reducescatter",
                            algo=str(choices["dp_reducescatter"][0]),
                            link_type=str(choices["dp_reducescatter"][1]),
                            count=int(full_cnt),
                            gap_before_ms=float(dp_bucket_gap_ms),
                        )
                    )

                if is_residual_layer:
                    # Last bucket uses residual size to preserve total bytes.
                    last_rs = float(dp_bytes_rs) - float(dp_bucket_bytes) * float(
                        max(0, dp_rs_buckets - 1)
                    )
                    last_rs = max(1.0, last_rs)
                    nodes.append(
                        CommNode(
                            name=f"BWD[{mb}]:L{lyr}:DP:RS(dW_bucket_last)",
                            domain="dp",
                            payload_bytes=float(last_rs),
                            nodes=par.dp,
                            op="reducescatter",
                            algo=str(choices["dp_reducescatter"][0]),
                            link_type=str(choices["dp_reducescatter"][1]),
                            count=1,
                            gap_before_ms=float(dp_bucket_gap_ms),
                        )
                    )

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

    # If enabled, model the DP allgather buckets after backward (still with gaps).
    if dp_bucket_bytes > 0:
        # Use the exact residual for the last bucket to preserve total bytes.
        if dp_ag_buckets > 1:
            nodes.append(
                CommNode(
                    name="OPT:DP:AG(W_bucket)",
                    domain="dp",
                    payload_bytes=float(dp_bucket_bytes),
                    nodes=par.dp,
                    op="allgather",
                    algo=str(choices["dp_allgather"][0]),
                    link_type=str(choices["dp_allgather"][1]),
                    count=int(dp_ag_buckets - 1),
                    gap_before_ms=float(dp_bucket_gap_ms),
                )
            )
        last_ag = float(dp_bytes_ag) - float(dp_bucket_bytes) * float(
            max(0, dp_ag_buckets - 1)
        )
        last_ag = max(1.0, last_ag)
        nodes.append(
            CommNode(
                name="OPT:DP:AG(W_bucket_last)",
                domain="dp",
                payload_bytes=float(last_ag),
                nodes=par.dp,
                op="allgather",
                algo=str(choices["dp_allgather"][0]),
                link_type=str(choices["dp_allgather"][1]),
                count=1,
                gap_before_ms=float(dp_bucket_gap_ms),
            )
        )

    # 3. TP AllReduce LayerNorm
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

    # 4. DP ZeRO-2 ReduceScatter (legacy, non-bucketed step-end model)
    if dp_bucket_bytes <= 0:
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


def _per_step_totals(
    choices: Dict[str, Tuple[str, str]],
    bw_tp: float,
    bw_pp: float,
    bw_dp: float,
    sys: SystemConfig,
    mod: ModelConfig,
    par: ParallelConfig,
) -> Tuple[float, float, float, float, float, float]:
    """Return (tp_total_ms, pp_total_ms, dp_total_ms, t_tp_ag_ms, t_tp_rs_ms, t_tp_ar_ms)."""
    a_full = mod.seq * mod.hidden * mod.bytes_per_act
    a_shard = (mod.seq / par.tp) * mod.hidden * mod.bytes_per_act
    ln_grad_bytes = 2 * mod.hidden * mod.bytes_per_grad

    (
        _,
        _,
        p_layer_tp_bf16,
        p_layer_tp_fp32,
        _,
    ) = llama3_megatron_payloads(mod, par)

    tp_calls_per_layer = {"allgather": 4, "reducescatter": 4}

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
    pp_total = par.microbatches_per_step * mod.layers * (4.0 * t_pp_one)
    dp_total = par.microbatches_per_step * mod.layers * (t_dp_rs + t_dp_ag)
    return tp_total, pp_total, dp_total, t_tp_ag, t_tp_rs, t_tp_ar


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

    comm_label = (
        "obj_comm"
        if bool(getattr(seg_sys, "objective_dp_gap_overlap", False))
        else "comm"
    )

    print(
        f"BW_segments={len(segments)} | "
        f"R_BW={total_bw_rc:.3f} ms | "
        f"R_link={total_link_rc:.3f} ms | "
        f"internal_link={total_internal:.3f} ms | "
        f"{comm_label}={total_comm:.3f} ms | "
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
        comm_label = (
            "obj_comm"
            if bool(getattr(seg_sys, "objective_dp_gap_overlap", False))
            else "comm"
        )
        print(
            f"  BWseg{k}: nodes[{s.start_idx + 1}..{s.end_idx + 1}] domains={doms}  "
            f"R_BW={s.exposed_bw_boundary_ms:.3f} ms  "
            f"R_link={s.exposed_link_boundaries_ms:.3f} ms  "
            f"internal_link={s.internal_retune_ms:.3f} ms  "
            f"{comm_label}={s.comm_time_ms:.3f} ms  "
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
                comm_label = (
                    "obj_comm"
                    if bool(getattr(seg_sys, "objective_dp_gap_overlap", False))
                    else "comm"
                )
                print(
                    f"    L{li}: nodes[{ls.start_idx + 1}..{ls.end_idx + 1}]  "
                    f"R_L={ls.exposed_link_boundary_ms:.3f} ms  "
                    f"internal={ls.internal_retune_ms:.3f} ms  "
                    f"{comm_label}={ls.comm_time_ms:.3f} ms  "
                    f"total={ls.total_ms:.3f} ms  "
                    f"k: {ks}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Llama 3 405B communication model with bandwidth granularity support (v2)"
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
        help="Link-only boundary retune latency (ms). (default: 0.0)",
    )
    parser.add_argument(
        "--degree-k-total",
        type=int,
        default=0,
        help="Total degree budget K: maximum number of simultaneous bidirectional peer edges "
        "(0 => unlimited/ideal). (default: 0)",
    )
    parser.add_argument(
        "--bw-grid-step",
        type=float,
        default=0.01,
        help="Grid step for solving per-segment min-delay bw splits (continuous model only). "
        "(default: 0.01)",
    )
    parser.add_argument(
        "--objective-dp-gap-overlap",
        action="store_true",
        help="When planning BW/link segments, treat DP nodes as partially hideable by their "
        "gap_before_ms: objective uses max(0, dp_comm_ms - gap_before_ms). "
        "This is an additive approximation (no backlog state).",
    )
    parser.add_argument(
        "--no-preplanned",
        action="store_true",
        help="Skip computing/printing the pre-planned DP schedule for the Llama step.",
    )
    parser.add_argument(
        "--no-fast-preplanned",
        action="store_true",
        help="Skip computing/printing the fast approximate pre-planned schedule.",
    )
    parser.add_argument(
        "--emit-comm-trace",
        action="store_true",
        help="Emit a one-rank serialized communication timeline for (preplanned, one-shot, static).",
    )
    parser.add_argument(
        "--mbs-count",
        type=int,
        default=2,
        help="How many microbatches to explicitly model in the comm-node schedule (default: 2).",
    )
    parser.add_argument(
        "--layers-per-stage",
        type=int,
        default=2,
        help="How many layers per PP stage to explicitly model (default: 2; full is ~8 when PP=16).",
    )
    parser.add_argument(
        "--bwd-last-mb-extra-layers",
        type=int,
        default=0,
        help="Add this many extra layers to the last modeled microbatch's backward pass to create "
        "more DP launch points (default: 0).",
    )
    parser.add_argument(
        "--dp-bucket-bytes",
        type=float,
        default=0.0,
        help="If >0, model ZeRO-2 DP as bucketed RS/AG and insert DP bucket nodes during backward. "
        "Unit: bytes (default: 0 => disabled).",
    )
    parser.add_argument(
        "--dp-bucket-gap-ms",
        type=float,
        default=0.0,
        help="Compute window (ms) between DP bucket launches, used as CommNode.gap_before_ms for DP "
        "overlap objective (default: 0).",
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
        help="Label every Nth PP block in the plot (default: 4; 0 disables PP labels).",
    )
    parser.add_argument(
        "--comm-trace-plot-min-marker-ms",
        type=float,
        default=0.0,
        help="Disabled: short-event markers are no longer drawn.",
    )
    parser.add_argument(
        "--comm-trace-plot-ms-per-inch",
        type=float,
        default=400.0,
        help="Plot horizontal scale: milliseconds per inch. (default: 400.0).",
    )
    parser.add_argument(
        "--comm-trace-plot-max-width-inch",
        type=float,
        default=60.0,
        help="Cap the plot width (inches). (default: 60).",
    )
    parser.add_argument(
        "--comm-trace-plot-no-params",
        action="store_true",
        help="Do not print the run parameters block onto the PNG figure.",
    )
    parser.add_argument(
        "--static-include-initial-reconfig",
        action="store_true",
        help="DEPRECATED/IGNORED.",
    )
    parser.add_argument(
        "--collective-profiles",
        type=str,
        nargs="+",
        default=["all"],
        choices=["mixed", "ring_asym", "ring_sym", "hypercube", "tree", "all"],
        help="Which collective-algorithm profile(s) to use. (default: all)",
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
        help="How many distinct peers can be kept connected during a collective "
        "(0=unlimited/ideal, default: 0)",
    )
    parser.add_argument(
        "--peer-switch-us",
        type=float,
        default=0.0,
        help="Extra latency to switch to a new peer inside a collective (us)",
    )
    args = parser.parse_args()

    # Setup: network configuration
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
        objective_dp_gap_overlap=bool(args.objective_dp_gap_overlap),
    )

    # Turn off legacy intra-collective peer-switching penalty
    if sys.peer_switch_sec > 0 and sys.max_peers_per_collective > 0:
        print(
            "[warn] legacy intra-collective peer switch penalty is enabled, but this script "
            "now models link retunes explicitly; disabling legacy penalty to avoid double-counting."
        )
    sys.max_peers_per_collective = 0
    sys.peer_switch_sec = 0.0

    # Llama 3 405B 8K (CP=1) inputs
    mod = ModelConfig(layers=126, hidden=16384, seq=8192, total_params=405e9)
    par = ParallelConfig(tp=8, pp=16, dp=128, global_batch_seqs=2048, microbatch_seqs=1)

    # Bandwidth shares
    bw_share: Dict[str, float] = {
        "tp": 1.0 / 3.0,
        "pp": 1.0 / 3.0,
        "dp": 1.0 / 3.0,
    }
    bw_share_requested = dict(bw_share)
    bw_share_units: Dict[str, int] | None = None

    if sys.unit_bw_GBps > 0:
        if sys.total_bw_units is None:
            raise ValueError("unit_bw_GBps>0 but total_bw_units is None")
        total_units = sys.total_bw_units
        active_domains = [d for d, v in bw_share_requested.items() if v > 0]
        min_units = 1 if total_units >= len(active_domains) else 0
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

    tp_peer_shards = par.tp - 1
    tp_recv_bytes = tp_peer_shards * a_shard
    pp_p2p_bytes = a_shard
    dp_sendrecv_bytes_fp32 = (par.dp - 1) * p_layer_tp_fp32 / par.dp
    dp_sendrecv_bytes_bf16 = (par.dp - 1) * p_layer_tp_bf16 / par.dp

    print("=== Llama 3 405B (8K, CP=1) Communication Model (v2) ===")
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

    # Per-op comparisons
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
            f"PP P2P boundary (per transfer, 1 hop) TP-sharded, {_fmt_mib_compact(pp_p2p_bytes)} send/recv per layer",
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

    # Pre-planned max utilization strategy
    if not args.no_preplanned:
        profiles = _profiles_to_run(args.collective_profiles)
        _combine_png = (
            bool(args.emit_comm_trace)
            and (not args.comm_trace_no_png)
            and (len(profiles) > 1)
        )
        _combined_rows_by_strategy: Dict[str, List[Tuple[str, List[TraceEvent]]]] = {
            "preplanned": [],
            "fast-preplanned": [],
            "one-shot": [],
            "static": [],
        }
        _combined_rows: List[Tuple[str, List[TraceEvent]]] = []
        _combined_params_one_shot_bw: Dict[str, Dict[str, float]] = {}

        for prof in profiles:
            choices = _choices_for_profile(prof)
            tag = f"llama3-step-megatron | profile={prof}"

            tp_total, pp_total, dp_total, t_tp_ag, t_tp_rs, t_tp_ar = _per_step_totals(
                choices=choices,
                bw_tp=float(bw_share["tp"]),
                bw_pp=float(bw_share["pp"]),
                bw_dp=float(bw_share["dp"]),
                sys=sys,
                mod=mod,
                par=par,
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

            llama_nodes = _build_llama_nodes(
                choices=choices,
                mod=mod,
                par=par,
                mbs_count=int(args.mbs_count),
                layers_per_stage=int(args.layers_per_stage),
                bwd_last_mb_extra_layers=int(args.bwd_last_mb_extra_layers),
                dp_bucket_bytes=int(args.dp_bucket_bytes)
                if float(args.dp_bucket_bytes) > 0
                else 0,
                dp_bucket_gap_ms=float(args.dp_bucket_gap_ms),
            )
            segments = preplanned_dp_partition(
                llama_nodes, sys, bw_grid_step=float(args.bw_grid_step)
            )
            _print_segments(tag, llama_nodes, segments, sys)

            fast_segments: List[BWSegmentPlan] | None = None
            if not bool(args.no_fast_preplanned):
                fast_segments = fast_preplanned_partition(
                    llama_nodes,
                    sys,
                    bw_grid_step=float(args.bw_grid_step),
                )
                _print_segments(f"{tag} | fast", llama_nodes, fast_segments, sys)

            bw_all, one_shot_seg = (
                solve_min_delay_bw_split(
                    llama_nodes, sys, bw_grid_step=float(args.bw_grid_step)
                )[:2],
                solve_best_link_plan_for_bw_segment(
                    llama_nodes,
                    0,
                    len(llama_nodes) - 1,
                    *solve_min_delay_bw_split(
                        llama_nodes, sys, bw_grid_step=float(args.bw_grid_step)
                    )[:2],
                    sys=sys,
                ),
            )
            bw_all = solve_min_delay_bw_split(
                llama_nodes, sys, bw_grid_step=float(args.bw_grid_step)
            )[0]
            one_shot_seg = solve_best_link_plan_for_bw_segment(
                llama_nodes,
                0,
                len(llama_nodes) - 1,
                bw_share=bw_all,
                bw_units=solve_min_delay_bw_split(
                    llama_nodes, sys, bw_grid_step=float(args.bw_grid_step)
                )[1],
                sys=sys,
            )
            static_seg = solve_best_link_plan_for_bw_segment(
                llama_nodes, 0, len(llama_nodes) - 1, bw_share, bw_share_units, sys
            )

            print("\n" + "-" * 78)
            print(
                f"Pre-planned vs fast-preplanned vs one-shot vs static(equal-share) [{tag}]:"
            )
            pre_bw_rc = sum(float(s.exposed_bw_boundary_ms) for s in segments)
            pre_link_rc = sum(float(s.exposed_link_boundaries_ms) for s in segments)
            pre_internal = sum(float(s.internal_retune_ms) for s in segments)
            pre_comm = sum(float(s.comm_time_ms) for s in segments)
            pre_total = pre_bw_rc + pre_link_rc + pre_internal + pre_comm

            comm_label = (
                "obj_comm"
                if bool(getattr(sys, "objective_dp_gap_overlap", False))
                else "comm"
            )

            print(
                f"  preplanned: {comm_label}={pre_comm:.3f} ms  internal_link={pre_internal:.3f} ms  "
                f"R_link={pre_link_rc:.3f} ms  R_BW={pre_bw_rc:.3f} ms  total={pre_total:.3f} ms"
            )

            if fast_segments is not None:
                fast_bw_rc = sum(float(s.exposed_bw_boundary_ms) for s in fast_segments)
                fast_link_rc = sum(
                    float(s.exposed_link_boundaries_ms) for s in fast_segments
                )
                fast_internal = sum(float(s.internal_retune_ms) for s in fast_segments)
                fast_comm = sum(float(s.comm_time_ms) for s in fast_segments)
                fast_total = fast_bw_rc + fast_link_rc + fast_internal + fast_comm
                print(
                    f"  fast-plan : {comm_label}={fast_comm:.3f} ms  internal_link={fast_internal:.3f} ms  "
                    f"R_link={fast_link_rc:.3f} ms  R_BW={fast_bw_rc:.3f} ms  total={fast_total:.3f} ms"
                )

            os_total = float(one_shot_seg.total_ms)
            print(
                f"  one-shot  : {comm_label}={one_shot_seg.comm_time_ms:.3f} ms  internal_link={one_shot_seg.internal_retune_ms:.3f} ms  "
                f"R_link={one_shot_seg.exposed_link_boundaries_ms:.3f} ms  R_BW={one_shot_seg.exposed_bw_boundary_ms:.3f} ms  "
                f"total={os_total:.3f} ms  "
                f"bw={', '.join([f'{d}={bw_all[d]:.3f}' for d in ['tp', 'pp', 'dp']])}"
            )

            st_total = float(static_seg.total_ms)
            print(
                f"  static    : {comm_label}={static_seg.comm_time_ms:.3f} ms  internal_link={static_seg.internal_retune_ms:.3f} ms  "
                f"R_link={static_seg.exposed_link_boundaries_ms:.3f} ms  R_BW={static_seg.exposed_bw_boundary_ms:.3f} ms  "
                f"total={st_total:.3f} ms  (bw_share tp/pp/dp all = {bw_share['tp']:.3f})"
            )

            if args.emit_comm_trace:
                preplanned_trace = _trace_from_segments(
                    "preplanned", llama_nodes, sys, segments
                )
                fast_trace = (
                    _trace_from_segments(
                        "fast-preplanned", llama_nodes, sys, fast_segments
                    )
                    if fast_segments is not None
                    else []
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

                write_trace_json(out_dir / f"{pfx}.preplanned.json", preplanned_trace)
                if fast_segments is not None:
                    write_trace_json(
                        out_dir / f"{pfx}.fast_preplanned.json", fast_trace
                    )
                write_trace_json(out_dir / f"{pfx}.one_shot.json", one_shot_trace)
                write_trace_json(out_dir / f"{pfx}.static.json", static_trace)
                write_trace_csv(out_dir / f"{pfx}.preplanned.csv", preplanned_trace)
                if fast_segments is not None:
                    write_trace_csv(out_dir / f"{pfx}.fast_preplanned.csv", fast_trace)
                write_trace_csv(out_dir / f"{pfx}.one_shot.csv", one_shot_trace)
                write_trace_csv(out_dir / f"{pfx}.static.csv", static_trace)

                if not args.comm_trace_no_png:
                    if _combine_png:
                        _combined_rows_by_strategy["preplanned"].append(
                            (f"{prof} | Preplanned", preplanned_trace)
                        )
                        if fast_segments is not None:
                            _combined_rows_by_strategy["fast-preplanned"].append(
                                (f"{prof} | Fast preplanned", fast_trace)
                            )
                        _combined_rows_by_strategy["one-shot"].append(
                            (f"{prof} | One-shot", one_shot_trace)
                        )
                        _combined_rows_by_strategy["static"].append(
                            (f"{prof} | Even share", static_trace)
                        )
                        _combined_params_one_shot_bw[str(prof)] = dict(one_shot_bw)
                    else:
                        params_lines = []
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

                        wrote = try_plot_trace_png(
                            out_dir / f"{pfx}.png",
                            strategy_to_events={
                                "preplanned": preplanned_trace,
                                "fast-preplanned": fast_trace,
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

                bw_str = ", ".join(
                    [f"{d}={one_shot_bw.get(d, 0.0):.3f}" for d in ["tp", "pp", "dp"]]
                )
                print("\n" + "-" * 78)
                print(f"[comm-trace] wrote traces to: {out_dir.resolve()}")
                print(f"[comm-trace] profile={prof} one-shot bw: {bw_str}")

        if _combine_png:
            _combined_rows = (
                _combined_rows_by_strategy["preplanned"]
                + _combined_rows_by_strategy["fast-preplanned"]
                + _combined_rows_by_strategy["one-shot"]
                + _combined_rows_by_strategy["static"]
            )

        if (
            _combine_png
            and _combined_rows
            and args.emit_comm_trace
            and (not args.comm_trace_no_png)
        ):
            out_dir = Path(str(args.comm_trace_out_dir))
            prefix = str(args.comm_trace_prefix)
            combined_path = out_dir / f"{prefix}.png"

            params_lines = []
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

            wrote = try_plot_rows_png(
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
