#!/usr/bin/env python3
"""DP bucketing + overlap-aware planning entry script.

This script is a convenience wrapper around the modular Llama 3 comm model that:
- enables the overlap-aware solver objective by default
- enables DP bucketing by default (with reasonable defaults)

The overlap approximation is *additive* (no backlog state): for each DP comm node,
its objective contribution becomes:

    exposed_dp_ms = max(0, dp_comm_ms - gap_before_ms)

TP/PP are still treated as blocking/serialized in the objective.

Note: The printed/optimized "obj_comm" is the solver objective, not necessarily a
serialized wall-clock time.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import llama3_modular as base
from llama3_comm import (
    BWSegmentPlan,
    CommNode,
    fast_preplanned_partition,
    MiB,
    ModelConfig,
    ParallelConfig,
    SystemConfig,
    TraceEvent,
    preplanned_dp_partition,
    quantize_share_dict,
    solve_best_link_plan_for_bw_segment,
    solve_min_delay_bw_split,
    try_plot_trace_png,
    write_trace_csv,
    write_trace_json,
)
from llama3_comm.execution import _trace_from_segments
from llama3_comm.solvers import _trace_one_shot, _trace_static


def _default_dp_bucket_bytes(par: ParallelConfig) -> int:
    # Matches the legacy rule-of-thumb referenced in README:
    # max(40MB, 1MB * dp_world_size). We use MiB (binary) consistently.
    # Note: dp_world_size here is the DP group size (par.dp).
    return int(max(40 * MiB, int(par.dp) * MiB))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Llama 3 405B comm model with DP bucketing and overlap-aware planning (v2)"
        )
    )

    # --- Network/system knobs ---
    parser.add_argument("--bandwidth-gbps", type=float, default=240.0)
    parser.add_argument("--latency-us", type=float, default=2.0)
    parser.add_argument("--unit-bw-gbps", type=float, default=0.0)
    parser.add_argument("--asym-min-reverse-units", type=int, default=1)
    parser.add_argument("--reconfig-ms", type=float, default=0.0)
    parser.add_argument("--link-batch-ms", type=float, default=0.0)
    parser.add_argument("--degree-k-total", type=int, default=0)
    parser.add_argument("--bw-grid-step", type=float, default=0.05)

    # Keep these for compatibility; we still disable their effect inside the script
    # to avoid double-counting (same behavior as llama3_modular.py).
    parser.add_argument("--max-peers-per-collective", type=int, default=0)
    parser.add_argument("--peer-switch-us", type=float, default=0.0)

    # --- Workload/schedule truncation knobs ---
    parser.add_argument(
        "--mbs-count",
        type=int,
        default=2,
        help="How many microbatches to explicitly model (default: 2).",
    )
    parser.add_argument(
        "--layers-per-stage",
        type=int,
        default=2,
        help="How many layers per PP stage to explicitly model (default: 2).",
    )
    parser.add_argument(
        "--bwd-last-mb-extra-layers",
        type=int,
        default=6,
        help=(
            "Extra backward layers (last modeled microbatch only) to create more DP launch points "
            "without expanding TP comm nodes (default: 6)."
        ),
    )

    # --- DP bucketing + overlap objective ---
    parser.add_argument(
        "--dp-bucket-bytes",
        type=float,
        default=None,
        help=(
            "DP bucket size in bytes. Default: max(40MiB, 1MiB * dp_world_size). "
            "Set <=0 to disable bucketing."
        ),
    )
    parser.add_argument(
        "--dp-bucket-gap-ms",
        type=float,
        default=0.2,
        help=(
            "Compute window (ms) between DP bucket launches used as gap_before_ms for the overlap "
            "objective (default: 0.2ms)."
        ),
    )
    parser.add_argument(
        "--objective-dp-gap-overlap",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "Enable the overlap-aware solver objective for DP nodes (default: enabled). "
            "Use --no-objective-dp-gap-overlap to disable."
        ),
    )

    # --- Output/trace ---
    parser.add_argument(
        "--collective-profiles",
        type=str,
        nargs="+",
        default=["all"],
        choices=["mixed", "ring_asym", "ring_sym", "hypercube", "tree", "all"],
    )
    parser.add_argument(
        "--no-preplanned",
        action="store_true",
        help="Skip the preplanned DP schedule and only print per-op comparisons.",
    )
    parser.add_argument(
        "--emit-comm-trace",
        action="store_true",
        help="Write one-rank serialized comm traces (JSON/CSV, optional PNG).",
    )
    parser.add_argument("--comm-trace-out-dir", type=str, default="out")
    parser.add_argument(
        "--comm-trace-prefix",
        type=str,
        default="one_node_comm_sequence_iteration.dp_bucketing",
    )
    parser.add_argument(
        "--comm-trace-no-png",
        action="store_true",
        help="Do not attempt PNG generation.",
    )
    parser.add_argument(
        "--comm-trace-pp-label-every",
        type=int,
        default=4,
        help="Label every Nth PP block in the plot (default: 4; 0 disables).",
    )
    parser.add_argument(
        "--comm-trace-plot-ms-per-inch",
        type=float,
        default=400.0,
        help="Plot horizontal scale: milliseconds per inch.",
    )
    parser.add_argument(
        "--comm-trace-plot-max-width-inch",
        type=float,
        default=60.0,
        help="Cap the plot width (inches).",
    )

    args = parser.parse_args()

    # --- Base model configs (same as llama3_modular.py) ---
    mod = ModelConfig(layers=126, hidden=16384, seq=8192, total_params=405e9)
    par = ParallelConfig(tp=8, pp=16, dp=128, global_batch_seqs=2048, microbatch_seqs=1)

    # Defaults that depend on dp world size.
    dp_bucket_bytes: int
    if args.dp_bucket_bytes is None:
        dp_bucket_bytes = _default_dp_bucket_bytes(par)
    else:
        dp_bucket_bytes = int(args.dp_bucket_bytes)

    if dp_bucket_bytes <= 0:
        dp_bucket_bytes = 0

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

    # Match llama3_modular.py: turn off legacy intra-collective peer-switch penalty.
    sys.max_peers_per_collective = 0
    sys.peer_switch_sec = 0.0

    # Bandwidth shares (same defaults as llama3_modular.py)
    bw_share: Dict[str, float] = {"tp": 1.0 / 3.0, "pp": 1.0 / 3.0, "dp": 1.0 / 3.0}
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

    print("=== Llama 3 405B DP bucketing model (v2) ===")
    print(f"Network: {sys.bw_bytes_sec / 1e9:.0f} GB/s, {sys.latency_sec * 1e6:.1f} us")
    print(
        f"Boundaries: T_segment_reconfig={sys.reconfig_sec * 1e3:.3f} ms, "
        f"T_link_batch={sys.link_batch_sec * 1e3:.3f} ms"
    )
    deg_str = (
        "unlimited/ideal" if sys.degree_k_total <= 0 else str(int(sys.degree_k_total))
    )
    print(f"Degree budget: K_total={deg_str}")

    if dp_bucket_bytes > 0:
        print(
            "DP bucketing: enabled | "
            f"dp_bucket_bytes={dp_bucket_bytes / MiB:.1f} MiB | "
            f"dp_bucket_gap_ms={float(args.dp_bucket_gap_ms):.3f}"
        )
    else:
        print("DP bucketing: disabled")

    print(
        "Objective: "
        + (
            "DP gap overlap enabled (obj_comm)"
            if sys.objective_dp_gap_overlap
            else "serialized comm (comm)"
        )
    )

    # Run per-op comparisons and/or planning.
    profiles = base._profiles_to_run(args.collective_profiles)

    # Keep llama3_modular.py's per-op comparison tables.
    # (We reuse its main by calling the internal helpers it exposes.)
    # For bucketing experiments, the key result is the preplanned/one-shot/static comparison.

    if args.no_preplanned:
        # If the user only wants the per-op tables, just reuse llama3_modular.py.
        # We do this by directly invoking its main entry via subprocess-like behavior would
        # lose our defaulting, so instead we call through for one profile in a minimal way.
        print("[note] --no-preplanned set; this script focuses on planning outputs.")
        return

    for prof in profiles:
        choices = base._choices_for_profile(prof)
        tag = f"llama3-step-megatron | profile={prof}"

        llama_nodes = base._build_llama_nodes(
            choices=choices,
            mod=mod,
            par=par,
            mbs_count=int(args.mbs_count),
            layers_per_stage=int(args.layers_per_stage),
            bwd_last_mb_extra_layers=int(args.bwd_last_mb_extra_layers),
            dp_bucket_bytes=int(dp_bucket_bytes),
            dp_bucket_gap_ms=float(args.dp_bucket_gap_ms),
        )

        segments = preplanned_dp_partition(
            llama_nodes, sys, bw_grid_step=float(args.bw_grid_step)
        )
        base._print_segments(tag, llama_nodes, segments, sys)
        fast_segments = fast_preplanned_partition(
            llama_nodes,
            sys,
            bw_grid_step=float(args.bw_grid_step),
        )
        base._print_segments(f"{tag} | fast", llama_nodes, fast_segments, sys)

        bw_all, units_all, _ = solve_min_delay_bw_split(
            llama_nodes, sys, bw_grid_step=float(args.bw_grid_step)
        )
        one_shot_seg = solve_best_link_plan_for_bw_segment(
            llama_nodes,
            0,
            len(llama_nodes) - 1,
            bw_share=bw_all,
            bw_units=units_all,
            sys=sys,
        )
        static_seg = solve_best_link_plan_for_bw_segment(
            llama_nodes, 0, len(llama_nodes) - 1, bw_share, bw_share_units, sys
        )

        comm_label = "obj_comm" if sys.objective_dp_gap_overlap else "comm"

        print("\n" + "-" * 78)
        print(
            f"Pre-planned vs fast-preplanned vs one-shot vs static(equal-share) [{tag}]:"
        )
        pre_bw_rc = sum(float(s.exposed_bw_boundary_ms) for s in segments)
        pre_link_rc = sum(float(s.exposed_link_boundaries_ms) for s in segments)
        pre_internal = sum(float(s.internal_retune_ms) for s in segments)
        pre_comm = sum(float(s.comm_time_ms) for s in segments)
        pre_total = pre_bw_rc + pre_link_rc + pre_internal + pre_comm
        fast_bw_rc = sum(float(s.exposed_bw_boundary_ms) for s in fast_segments)
        fast_link_rc = sum(float(s.exposed_link_boundaries_ms) for s in fast_segments)
        fast_internal = sum(float(s.internal_retune_ms) for s in fast_segments)
        fast_comm = sum(float(s.comm_time_ms) for s in fast_segments)
        fast_total = fast_bw_rc + fast_link_rc + fast_internal + fast_comm

        print(
            f"  preplanned: {comm_label}={pre_comm:.3f} ms  internal_link={pre_internal:.3f} ms  "
            f"R_link={pre_link_rc:.3f} ms  R_BW={pre_bw_rc:.3f} ms  total={pre_total:.3f} ms"
        )
        print(
            f"  fast-plan : {comm_label}={fast_comm:.3f} ms  internal_link={fast_internal:.3f} ms  "
            f"R_link={fast_link_rc:.3f} ms  R_BW={fast_bw_rc:.3f} ms  total={fast_total:.3f} ms"
        )

        print(
            f"  one-shot  : {comm_label}={one_shot_seg.comm_time_ms:.3f} ms  internal_link={one_shot_seg.internal_retune_ms:.3f} ms  "
            f"R_link={one_shot_seg.exposed_link_boundaries_ms:.3f} ms  R_BW={one_shot_seg.exposed_bw_boundary_ms:.3f} ms  "
            f"total={float(one_shot_seg.total_ms):.3f} ms  "
            f"bw={', '.join([f'{d}={bw_all[d]:.3f}' for d in ['tp', 'pp', 'dp']])}"
        )

        print(
            f"  static    : {comm_label}={static_seg.comm_time_ms:.3f} ms  internal_link={static_seg.internal_retune_ms:.3f} ms  "
            f"R_link={static_seg.exposed_link_boundaries_ms:.3f} ms  R_BW={static_seg.exposed_bw_boundary_ms:.3f} ms  "
            f"total={float(static_seg.total_ms):.3f} ms  (bw_share tp/pp/dp all = {bw_share['tp']:.3f})"
        )

        if args.emit_comm_trace:
            preplanned_trace: List[TraceEvent] = _trace_from_segments(
                "preplanned", llama_nodes, sys, segments
            )
            fast_trace: List[TraceEvent] = _trace_from_segments(
                "fast-preplanned", llama_nodes, sys, fast_segments
            )
            one_shot_trace, _one_shot_bw, _one_shot_units = _trace_one_shot(
                "one-shot", llama_nodes, sys, bw_grid_step=float(args.bw_grid_step)
            )
            static_trace = _trace_static(
                "static",
                llama_nodes,
                sys,
                bw_share=bw_share,
                bw_units=bw_share_units,
                include_initial_reconfig=False,
            )

            out_dir = Path(str(args.comm_trace_out_dir))
            out_dir.mkdir(parents=True, exist_ok=True)
            prefix = str(args.comm_trace_prefix)
            pfx = f"{prefix}.{prof}"

            write_trace_json(out_dir / f"{pfx}.preplanned.json", preplanned_trace)
            write_trace_json(out_dir / f"{pfx}.fast_preplanned.json", fast_trace)
            write_trace_json(out_dir / f"{pfx}.one_shot.json", one_shot_trace)
            write_trace_json(out_dir / f"{pfx}.static.json", static_trace)
            write_trace_csv(out_dir / f"{pfx}.preplanned.csv", preplanned_trace)
            write_trace_csv(out_dir / f"{pfx}.fast_preplanned.csv", fast_trace)
            write_trace_csv(out_dir / f"{pfx}.one_shot.csv", one_shot_trace)
            write_trace_csv(out_dir / f"{pfx}.static.csv", static_trace)

            if not args.comm_trace_no_png:
                title = (
                    "One node's serialized communication sequence within one iteration (linearized schedule)\n"
                    "Grey = reconfiguration cost; Blue = TP comm; Pink = PP comm; Green = DP comm"
                )
                try_plot_trace_png(
                    out_dir / f"{pfx}.png",
                    strategy_to_events={
                        "preplanned": preplanned_trace,
                        "fast-preplanned": fast_trace,
                        "one-shot": one_shot_trace,
                        "static": static_trace,
                    },
                    title=title,
                    pp_label_every=int(args.comm_trace_pp_label_every),
                    min_marker_ms=0.0,
                    ms_per_inch=float(args.comm_trace_plot_ms_per_inch),
                    max_width_in=float(args.comm_trace_plot_max_width_inch),
                    params_text="",
                )

            print(f"[comm-trace] wrote traces to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
