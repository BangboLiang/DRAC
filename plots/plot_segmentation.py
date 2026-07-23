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
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), constrained_layout=True)
    for scheme in sorted_unique(rows, "scheme"):
        items = sorted((row for row in rows if row["scheme"] == scheme), key=lambda row: float(row["delta_ms"]))
        color = COLORS.get(scheme)
        axes[0].plot([float(row["delta_ms"]) for row in items], [float(row["total_cost_ms"]) for row in items], marker="o", label=scheme, color=color)
        axes[1].plot([float(row["delta_ms"]) for row in items], [int(row["segment_count"]) for row in items], marker="o", label=scheme, color=color)
    for ax in axes:
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_xlabel("Reconfiguration delay δ (ms)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Total cost (ms)")
    axes[1].set_ylabel("Selected segment count")
    axes[0].legend(frameon=False, fontsize=8)
    save_figure(fig, output_base)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--input",default="results/evaluation_v1/processed/segmentation.csv"); parser.add_argument("--output",default="results/evaluation_v1/figures/segmentation"); args=parser.parse_args(); plot(args.input,args.output)
