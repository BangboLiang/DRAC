#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
from plots._common import COLORS, read_rows, save_figure, sorted_unique


def plot(input_csv: str, output_base: str) -> None:
    rows = read_rows(input_csv)
    available = set(sorted_unique(rows, "workload"))
    workloads = [name for name in ("DP", "PP", "DP+PP Mixed") if name in available]
    workloads.extend(sorted(available.difference(workloads)))
    fig, axes = plt.subplots(1, len(workloads), figsize=(4.0 * len(workloads), 3.4), constrained_layout=True, squeeze=False)
    for ax, workload in zip(axes[0], workloads):
        selected = [row for row in rows if row["workload"] == workload]
        for scheme in ("Static-Sym", "Sym-OCS", "DRAC"):
            items = sorted((row for row in selected if row["scheme"] == scheme), key=lambda row: int(row["port_budget"]))
            ax.plot([int(row["port_budget"]) for row in items], [float(row["total_communication_time_ms"]) for row in items], marker="o", label=scheme, color=COLORS[scheme])
        ax.set_title(workload)
        ax.set_xlabel("OCS channels per endpoint")
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("Communication time (ms)")
    axes[0][-1].legend(frameon=False, fontsize=8)
    save_figure(fig, output_base)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/evaluation_v1/processed/end_to_end_performance.csv")
    parser.add_argument("--output", default="results/evaluation_v1/figures/end_to_end_performance")
    args = parser.parse_args(); plot(args.input, args.output)
