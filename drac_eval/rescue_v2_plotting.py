from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)


LEVELS = ["endpoint", "server", "tor", "aggregation"]


def _save(fig: object, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_level(rows: List[Dict[str, object]], value: str, ylabel: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for workload in sorted({str(r["workload"]) for r in rows}):
        values = []
        for level in LEVELS:
            selected = [float(r[value]) for r in rows if r["workload"] == workload and r["level"] == level and r["mapping"] == "contiguous" and np.isfinite(float(r[value]))]
            values.append(float(np.mean(selected)) if selected else np.nan)
        ax.plot(range(len(LEVELS)), values, marker="o", label=workload.upper())
    ax.set_xticks(range(len(LEVELS)), LEVELS)
    ax.set_xlabel("Aggregation level")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=4)
    _save(fig, path)


def plot_aggregation_v2(signal: List[Dict[str, object]], performance: List[Dict[str, object]], mapping: List[Dict[str, object]], out: Path) -> None:
    _plot_level(signal, "absolute_directionality_bytes", "Absolute directional signal A (bytes)", out / "aggregation_absolute_directionality.pdf")
    _plot_level(signal, "boundary_traffic_fraction", "Boundary traffic fraction", out / "aggregation_boundary_traffic_fraction.pdf")
    _plot_level(signal, "omega", "Omega", out / "aggregation_omega_by_level.pdf")
    resource = [r for r in performance if r["normalization_mode"] == "resource_equivalent"]
    _plot_level(resource, "speedup_drac_over_sym", "DRAC makespan-opt speedup over Sym-OCS", out / "aggregation_speedup_resource_equivalent.pdf")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    mappings = ["contiguous", "round_robin", "random"]
    for workload in sorted({str(r["workload"]) for r in mapping}):
        values = []
        for name in mappings:
            selected = [float(r["absolute_retention"]) for r in mapping if r["workload"] == workload and r["mapping"] == name and np.isfinite(float(r["absolute_retention"]))]
            values.append(float(np.mean(selected)) if selected else np.nan)
        ax.plot(range(len(mappings)), values, marker="o", label=workload.upper())
    ax.set_xticks(range(len(mappings)), mappings)
    ax.set_xlabel("Mapping")
    ax.set_ylabel("ToR absolute directionality retention")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=4)
    _save(fig, out / "aggregation_mapping_sensitivity.pdf")


def plot_collective_v2(summary: List[Dict[str, object]], out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 4.3), constrained_layout=True)
    models = ["unidirectional_executable_ring", "executable_bidirectional_ring"]
    x = np.arange(len(models))
    width = 0.18
    for idx, workload in enumerate(sorted({str(r["workload"]) for r in summary})):
        values = []
        for model in models:
            selected = [float(r["drac_gain"]) for r in summary if r["workload"] == workload and r["collective_model"] == model]
            values.append(float(np.mean(selected)) if selected else np.nan)
        ax.bar(x + (idx - 1.5) * width, values, width, label=workload.upper())
    ax.set_xticks(x, ["Unidirectional executable ring", "Bidirectional executable ring"])
    ax.set_ylabel("DRAC gain over Sym-OCS")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=4)
    _save(fig, out / "executable_ring_drac_gain.pdf")


def plot_makespan_v2(runtime: List[Dict[str, object]], out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 4.1), constrained_layout=True)
    rows = sorted(runtime, key=lambda r: int(r["rank_count"]))
    ax.plot([int(r["rank_count"]) for r in rows], [float(r["runtime_ms"]) for r in rows], marker="o")
    ax.set_xlabel("Abstract nodes")
    ax.set_ylabel("drac_makespan_opt runtime (ms)")
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.25)
    _save(fig, out / "makespan_runtime_scaling.pdf")
