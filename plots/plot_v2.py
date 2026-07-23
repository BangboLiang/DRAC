from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plots._common import read_rows, save_figure


COLORS = {
    "Static-Sym": "#7f7f7f",
    "Sym-OCS": "#1f77b4",
    "DRAC-v1": "#9467bd",
    "DRAC-SegmentOpt": "#ff7f0e",
    "DRAC-SegmentOpt+Fallback": "#d62728",
    "OneConfig": "#7f7f7f",
    "PerNodeReconfig": "#1f77b4",
    "Medoid-DynamicProgramming": "#9467bd",
    "SegmentOpt-DynamicProgramming": "#d62728",
    "ExhaustivePartitionOracle": "#2ca02c",
    "SymmetricFallbackSchedule": "#8c564b",
    "SegmentOpt+Fallback-Integer": "#ff7f0e",
    "FloorOnly": "#7f7f7f",
    "NearestRounding": "#9467bd",
    "FillAllResidual": "#1f77b4",
    "DRACSparse-FloorSeed": "#ff7f0e",
    "DRACSparse-CoverageSeed": "#8c564b",
    "DRACSparse-MultiSeed": "#d62728",
    "ExhaustiveOracle": "#2ca02c",
    "FullReservation": "#4c78a8",
    "DRAC-Sparse": "#d62728",
}


WORKLOAD_ORDER = ("DP", "PP", "DP+PP Mixed")


def plot_end_to_end_v2(input_csv: str, output_base: str) -> None:
    rows = read_rows(input_csv)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5), constrained_layout=True)
    schemes = ("Static-Sym", "Sym-OCS", "DRAC-v1", "DRAC-SegmentOpt", "DRAC-SegmentOpt+Fallback")
    for ax, workload in zip(axes, WORKLOAD_ORDER):
        for scheme in schemes:
            selected = sorted(
                (r for r in rows if r["workload"] == workload and r["scheme"] == scheme),
                key=lambda r: int(r["port_budget"]),
            )
            ax.plot(
                [int(r["port_budget"]) for r in selected],
                [float(r["total_cost_ms"]) for r in selected],
                marker="o",
                label=scheme,
                color=COLORS[scheme],
            )
        ax.set_title(workload)
        ax.set_xlabel("OCS channels per endpoint")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Total communication cost (ms)")
    axes[-1].legend(frameon=False, fontsize=7)
    save_figure(fig, output_base)


def plot_segmentation_v2(input_csv: str, workload: str, output_base: str) -> None:
    rows = [r for r in read_rows(input_csv) if r["workload"] == workload]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), constrained_layout=True)
    schemes = (
        "OneConfig",
        "PerNodeReconfig",
        "Medoid-DynamicProgramming",
        "SegmentOpt-DynamicProgramming",
        "ExhaustivePartitionOracle",
        "SymmetricFallbackSchedule",
        "SegmentOpt+Fallback-Integer",
    )
    for scheme in schemes:
        selected = sorted((r for r in rows if r["scheme"] == scheme), key=lambda r: float(r["delta_ms"]))
        x = [float(r["delta_ms"]) for r in selected]
        axes[0].plot(x, [float(r["total_cost_ms"]) for r in selected], marker="o", ms=3, label=scheme, color=COLORS[scheme])
        axes[1].plot(x, [int(r["segment_count"]) for r in selected], marker="o", ms=3, label=scheme, color=COLORS[scheme])
    fallback = sorted((r for r in rows if r["scheme"] == "SegmentOpt+Fallback-Integer"), key=lambda r: float(r["delta_ms"]))
    axes[2].plot(
        [float(r["delta_ms"]) for r in fallback],
        [float(r["fallback_fraction"]) for r in fallback],
        marker="o",
        color=COLORS["DRAC-SegmentOpt+Fallback"],
    )
    for ax in axes:
        ax.set_xscale("symlog", linthresh=0.05)
        ax.set_xticks([0.0, 0.1, 1.0, 10.0], ["0", "0.1", "1", "10"])
        ax.set_xlabel("Reconfiguration delay δ (ms)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Total cost (ms)")
    axes[1].set_ylabel("Selected segment count")
    axes[2].set_ylabel("Symmetric fallback fraction")
    axes[2].set_ylim(-0.05, 1.05)
    axes[0].set_title(f"{workload}: total cost")
    axes[1].set_title("Segmentation granularity")
    axes[2].set_title("Integer fallback usage")
    axes[0].legend(frameon=False, fontsize=6)
    save_figure(fig, output_base)


def plot_realization_v2(input_csv: str, output_base: str) -> None:
    rows = read_rows(input_csv)
    workloads = ("DP", "PP", "DP+PP Mixed", "Synthetic Hard")
    policies = (
        "FloorOnly",
        "NearestRounding",
        "FillAllResidual",
        "DRACSparse-FloorSeed",
        "DRACSparse-CoverageSeed",
        "DRACSparse-MultiSeed",
        "ExhaustiveOracle",
    )
    fig, axes = plt.subplots(2, 4, figsize=(14.8, 6.5), constrained_layout=True, sharex="col")
    for column, workload in enumerate(workloads):
        for policy in policies:
            selected = sorted(
                (
                    r
                    for r in rows
                    if r["workload"] == workload
                    and r["policy"] == policy
                    and r["tolerance_satisfied"].lower() == "true"
                ),
                key=lambda r: float(r["epsilon"]),
            )
            x = [float(r["epsilon"]) for r in selected]
            style = "--" if policy in {"FloorOnly", "NearestRounding", "FillAllResidual"} else "-"
            axes[0, column].plot(x, [float(r["realized_slowdown"]) for r in selected], marker="o", ms=2.5, ls=style, label=policy, color=COLORS[policy])
            axes[1, column].plot(x, [int(r["used_connection_units"]) for r in selected], marker="o", ms=2.5, ls=style, label=policy, color=COLORS[policy])
        axes[0, column].set_title(workload)
        axes[1, column].set_xlabel("Tolerance ε")
        axes[0, column].grid(alpha=0.25)
        axes[1, column].grid(alpha=0.25)
    axes[0, 0].set_ylabel("Slowdown vs. continuous target")
    axes[1, 0].set_ylabel("Used connection units")
    axes[0, -1].legend(frameon=False, fontsize=6, loc="upper left")
    save_figure(fig, output_base)


def plot_compaction_v2(compaction_csv: str, iso_csv: str, output_dir: str) -> None:
    rows = read_rows(compaction_csv)
    schemes = ("FullReservation", "Sym-OCS", "DRAC-v1", "DRAC-SegmentOpt", "DRAC-Sparse")
    x = np.arange(len(WORKLOAD_ORDER))
    width = 0.16
    fig, ax = plt.subplots(figsize=(8.0, 3.8), constrained_layout=True)
    for index, scheme in enumerate(schemes):
        selected = [next(r for r in rows if r["workload"] == workload and r["scheme"] == scheme) for workload in WORKLOAD_ORDER]
        ax.bar(x + (index - 2) * width, [float(r["total_stable_pool"]) for r in selected], width, label=scheme, color=COLORS.get(scheme, "#4c78a8"))
    ax.set_xticks(x, WORKLOAD_ORDER)
    ax.set_ylabel("Stable directional channel pool")
    ax.set_title("Fixed physical budget")
    ax.legend(frameon=False, fontsize=7, ncol=3)
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, Path(output_dir) / "schedule_compaction_v2")

    iso = read_rows(iso_csv)
    fig, ax = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)
    reached = [next(r for r in iso if r["workload"] == workload) for workload in WORKLOAD_ORDER]
    directional = [float(r["minimum_stable_directional_pool"]) if r["status"] == "reached" else np.nan for r in reached]
    bundles = [2 * float(r["minimum_stable_bundle_pool"]) if r["status"] == "reached" else np.nan for r in reached]
    ax.bar(x - 0.18, directional, 0.36, label="Independent Tx/Rx")
    ax.bar(x + 0.18, bundles, 0.36, label="Bidirectional bundles")
    ax.set_xticks(x, WORKLOAD_ORDER)
    ax.set_ylabel("Minimum stable directional channels")
    ax.set_title("Iso-performance minimum pool")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, Path(output_dir) / "iso_performance_pool_v2")


def plot_planning_runtime_v2(input_csv: str, output_base: str) -> None:
    rows = sorted(read_rows(input_csv), key=lambda r: int(r["node_count"]))
    stages = (
        ("candidate_segment_target_ms", "Candidate segment targets"),
        ("dynamic_programming_ms", "Dynamic Programming"),
        ("sparse_realization_ms", "Sparse realization"),
        ("schedule_compaction_ms", "Schedule compaction"),
    )
    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    x = [int(r["node_count"]) for r in rows]
    bottom = np.zeros(len(rows))
    for field, label in stages:
        values = np.asarray([float(r[field]) for r in rows])
        ax.bar(x, values, bottom=bottom, label=label)
        bottom += values
    ax.set_yscale("log")
    ax.set_xlabel("Communication nodes K")
    ax.set_ylabel("Offline planning time (ms, log scale)")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, output_base)


def plot_target_case_study_pp(timeline_json: str, output_base: str, delta: float = 1.0) -> None:
    data = json.loads(Path(timeline_json).read_text(encoding="utf-8"))
    pp = [row for row in data if row["workload"] == "PP"]
    selected = min(pp, key=lambda row: abs(float(row["delta_ms"]) - delta))
    allocations = [np.asarray(value, dtype=float) for value in selected["segment_allocations"]]
    boundaries = [tuple(value) for value in selected["segment_opt_boundaries"]]
    count = len(allocations)
    fig = plt.figure(figsize=(min(14.8, max(6.0, 2.5 * count)), 4.6), constrained_layout=True)
    grid = fig.add_gridspec(2, count, height_ratios=[1, 3])
    timeline = fig.add_subplot(grid[0, :])
    colors = ("#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2")
    for index, (start, end) in enumerate(boundaries):
        timeline.barh(0, end - start + 1, left=start, color=colors[index % len(colors)], edgecolor="white")
        timeline.text((start + end + 1) / 2, 0, f"S{index}\n{start}–{end}", ha="center", va="center", fontsize=8)
    timeline.set_xlim(0, len(selected["node_ids"]))
    timeline.set_yticks([])
    timeline.set_xlabel("Ordered PP communication-node index")
    timeline.set_title(f"PP segment targets at δ={selected['delta_ms']} ms")
    maximum = max((float(np.max(matrix)) for matrix in allocations), default=1.0)
    for index, matrix in enumerate(allocations):
        ax = fig.add_subplot(grid[1, index])
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        ax.set_title(f"S{index}: {selected['integer_fallback_types'][index]}", fontsize=8)
        ax.set_xlabel("Rx endpoint")
        if index == 0:
            ax.set_ylabel("Tx endpoint")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] > 0:
                    ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(image, ax=fig.axes[1:], shrink=0.72, label="Continuous connection units")
    save_figure(fig, output_base)
