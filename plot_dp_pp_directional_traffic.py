#!/usr/bin/env python3
"""Plot DP/PP directional traffic from a precomputed summary CSV only."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np

WORKLOAD_ORDER = ("DP", "PP")
EXPECTED_ASSUMPTIONS = {
    "DP": "dp-zero2-rs-ag-v1",
    "PP": "pp-independent-tensors-v2-repo-model",
}
EXPECTED_RATIOS = {"DP": 141.029756, "PP": 1.0}


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_workload = {row["workload"]: row for row in rows}
    if set(by_workload) != set(WORKLOAD_ORDER):
        raise ValueError("input CSV must contain exactly one DP row and one PP row")
    ordered = [by_workload[name] for name in WORKLOAD_ORDER]
    for row in ordered:
        workload = row["workload"]
        if row.get("assumption_version") != EXPECTED_ASSUMPTIONS[workload]:
            raise ValueError(
                f"unexpected {workload} assumption_version: {row.get('assumption_version')!r}"
            )
        main = float(row["main_direction_bytes"])
        opposite = float(row["opposite_direction_bytes"])
        ratio = main / opposite
        if main < opposite or opposite <= 0:
            raise ValueError(f"invalid {workload} Main/Opposite values")
        if not math.isclose(ratio, EXPECTED_RATIOS[workload], rel_tol=1e-6):
            raise ValueError(
                f"unexpected {workload} ratio {ratio}; expected {EXPECTED_RATIOS[workload]}"
            )
    return ordered


def plot(csv_path: Path, output_path: Path, square_root_scale: bool) -> None:
    """Render one PDF/PNG; values remain original GB under the function scale."""

    rows = _load_rows(csv_path)
    main_gb = np.array([float(row["main_direction_bytes"]) / 1e9 for row in rows])
    opposite_gb = np.array([float(row["opposite_direction_bytes"]) / 1e9 for row in rows])
    if np.any(main_gb < opposite_gb) or np.any(opposite_gb < 0):
        raise ValueError("invalid Main/Opposite ordering or negative traffic")

    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.labelsize": 17,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    x = np.arange(len(WORKLOAD_ORDER))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.bar(x - width / 2, main_gb, width, label="Main direction", color="#F5A889")
    ax.bar(x + width / 2, opposite_gb, width, label="Opposite direction", color="#ACD6EC")
    if square_root_scale:
        ax.set_yscale("function", functions=(np.sqrt, np.square))
    ax.set_xticks(x, WORKLOAD_ORDER)
    ax.set_ylabel("Directional traffic volume\nper iteration (GB)", labelpad=6)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(False)
    ax.set_ylim(0, float(max(main_gb.max(), opposite_gb.max())) * 1.18)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"bbox_inches": "tight"}
    if output_path.suffix.lower() == ".png":
        save_kwargs["dpi"] = 200
    fig.savefig(output_path, **save_kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="results/dp_pp_directional_traffic/directional_traffic.csv",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--linear", action="store_true")
    args = parser.parse_args()
    plot(Path(args.input), Path(args.output), square_root_scale=not args.linear)
    print(args.output)


if __name__ == "__main__":
    main()
