from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "Static-Sym": "#7f7f7f",
    "Sym-OCS": "#1f77b4",
    "DRAC": "#d62728",
    "DRAC-DP": "#d62728",
    "OneConfig": "#7f7f7f",
    "PerNode-Reconfig": "#1f77b4",
    "SegmentOracle": "#2ca02c",
    "FloorOnly": "#7f7f7f",
    "NearestRounding": "#9467bd",
    "FillAllResidual": "#1f77b4",
    "DRACSparse": "#d62728",
    "ILPOracle": "#2ca02c",
}


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig: plt.Figure, output_base: str | Path) -> None:
    base = Path(output_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    width, height = fig.get_size_inches()
    if width > 16 or height > 12:
        raise ValueError(f"unreasonable Matplotlib canvas: {width}x{height}")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def sorted_unique(rows: Iterable[dict[str, str]], field: str) -> list[str]:
    return sorted({row[field] for row in rows})
