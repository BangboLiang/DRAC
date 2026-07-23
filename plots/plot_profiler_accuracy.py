#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from plots._common import read_rows, save_figure


def plot(input_csv: str, output_base: str) -> None:
    rows = read_rows(input_csv)
    complete = [row for row in rows if row.get("status") == "complete"]
    fig, ax = plt.subplots(figsize=(5.6, 4.2), constrained_layout=True)
    if not complete:
        ax.axis("off")
        ax.text(0.5, 0.58, "Profiler accuracy: measurement pending", ha="center", va="center", fontsize=12)
        ax.text(0.5, 0.42, "No measured NIC directional-counter input is available.\nNo simulated value is shown as measured.", ha="center", va="center", fontsize=9)
    else:
        for model, marker in (("PayloadOnly", "o"), ("Payload+Calibration", "s")):
            selected = [row for row in complete if row["model"] == model]
            ax.scatter([float(row["measured_bytes"]) for row in selected], [float(row["predicted_bytes"]) for row in selected], label=model, marker=marker, alpha=0.8)
        values = [float(row["measured_bytes"]) for row in complete] + [float(row["predicted_bytes"]) for row in complete]
        low, high = min(values), max(values)
        ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Measured directional bytes")
        ax.set_ylabel("Predicted directional bytes")
        ax.legend(frameon=False)
    save_figure(fig, output_base)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/evaluation_v1/processed/profiler_accuracy.csv")
    parser.add_argument("--output", default="results/evaluation_v1/figures/profiler_accuracy")
    args = parser.parse_args()
    plot(args.input, args.output)
