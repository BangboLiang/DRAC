from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)


LEVELS = ["endpoint", "server", "tor", "aggregation"]


def _mean(rows: Sequence[Dict[str, object]], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r and np.isfinite(float(r[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def _save(fig: object, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _line_by_group(
    rows: List[Dict[str, object]],
    x_key: str,
    y_key: str,
    group_key: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    x_order: Iterable[object] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    groups = sorted({str(r[group_key]) for r in rows})
    for group in groups:
        selected = [r for r in rows if str(r[group_key]) == group]
        xs_raw = list(x_order) if x_order is not None else sorted({r[x_key] for r in selected})
        ys, xs = [], []
        for x in xs_raw:
            sub = [r for r in selected if r[x_key] == x]
            value = _mean(sub, y_key)
            if np.isfinite(value):
                xs.append(x)
                ys.append(value)
        if xs:
            plot_x = list(range(len(xs))) if x_order is not None else xs
            ax.plot(plot_x, ys, marker="o", linewidth=1.7, label=group.upper())
            if x_order is not None:
                ax.set_xticks(range(len(xs)), [str(v) for v in xs])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=8, ncol=min(4, len(handles)))
    _save(fig, path)


def plot_aggregation(segment: List[Dict[str, object]], summary: List[Dict[str, object]], performance: List[Dict[str, object]], out: Path) -> None:
    contiguous = [r for r in summary if r["mapping"] == "contiguous"]
    _line_by_group(contiguous, "level", "traffic_weighted_omega", "workload", "Aggregation level", "Traffic-weighted Omega", out / "aggregation_omega_by_level.pdf", LEVELS)
    _line_by_group(contiguous, "level", "retention_mean", "workload", "Aggregation level", "Omega retention", out / "aggregation_retention_by_level.pdf", LEVELS)
    drac = [r for r in performance if r["algorithm"] == "drac" and r["mapping"] == "contiguous"]
    _line_by_group(drac, "level", "speedup_vs_sym_ocs", "workload", "Aggregation level", "DRAC speedup over Sym-OCS", out / "aggregation_speedup_by_level.pdf", LEVELS)
    mapping_rows = [r for r in summary if r["level"] == "tor"]
    _line_by_group(mapping_rows, "mapping", "traffic_weighted_omega", "workload", "Rank mapping", "ToR traffic-weighted Omega", out / "aggregation_mapping_sensitivity.pdf")


def plot_collective(segment: List[Dict[str, object]], performance: List[Dict[str, object]], summary: List[Dict[str, object]], gating: List[Dict[str, object]], out: Path) -> None:
    _line_by_group(summary, "collective_model", "traffic_weighted_omega", "workload", "Collective traffic model", "Traffic-weighted Omega", out / "collective_balancing_omega.pdf")
    main = [r for r in performance if r["algorithm"] in {"sym_ocs", "drac"}]
    _line_by_group(main, "collective_model", "normalized_communication_time", "algorithm", "Collective traffic model", "Normalized communication time", out / "collective_balancing_comm_time.pdf")
    retention = [r for r in summary if r["collective_model"] != "original"]
    _line_by_group(retention, "collective_model", "benefit_retention", "workload", "Collective traffic model", "DRAC benefit retention", out / "collective_balancing_gain_retention.pdf")

    import matplotlib.pyplot as plt
    labels = []
    values = []
    for model in ["original", "bidirectional_balanced", "pairwise_balancing_oracle"]:
        for alg in ["sym_ocs", "drac"]:
            sub = [r for r in performance if r["collective_model"] == model and r["algorithm"] == alg]
            labels.append(("Oracle" if "oracle" in model else model.replace("_balanced", "").title()) + " + " + ("DRAC" if alg == "drac" else "Sym-OCS"))
            values.append(_mean(sub, "normalized_communication_time"))
    fig, ax = plt.subplots(figsize=(8.5, 4.3), constrained_layout=True)
    ax.bar(range(len(labels)), values, color=["#8da0cb", "#fc8d62"] * 3)
    ax.set_xticks(range(len(labels)), labels, rotation=24, ha="right")
    ax.set_ylabel("Normalized communication time")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out / "collective_balancing_cross_product.pdf")
    _line_by_group(gating, "omega_threshold", "normalized_communication_time", "workload", "Omega threshold", "Gated DRAC normalized time", out / "drac_gating_sensitivity.pdf")


def plot_makespan(comparison: List[Dict[str, object]], gaps: List[Dict[str, object]], runtimes: List[Dict[str, object]], out: Path) -> None:
    _line_by_group(comparison, "port_budget", "communication_makespan_ms", "method", "Per-node port budget", "Communication makespan (ms)", out / "makespan_objective_comparison.pdf")
    _line_by_group(gaps, "port_budget", "optimality_gap", "method", "Per-node port budget", "Optimality gap", out / "makespan_optimality_gap.pdf")
    _line_by_group(runtimes, "port_budget", "runtime_ms", "method", "Per-node port budget", "Solver runtime (ms)", out / "makespan_runtime.pdf")
