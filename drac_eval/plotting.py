from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


MAX_FIGURE_INCHES = 20.0
DEFAULT_DPI = 200


def _group(rows: List[Dict[str, str]], key: str) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        out.setdefault(str(row[key]), []).append(row)
    return out


def _validate_figure_size(fig: plt.Figure, figure_name: str) -> None:
    width, height = fig.get_size_inches()
    if width > MAX_FIGURE_INCHES or height > MAX_FIGURE_INCHES:
        raise ValueError(
            f"Figure '{figure_name}' is too large before save: "
            f"{width:.2f}x{height:.2f} inches. "
            f"Please reduce plotted categories or adjust aggregation."
        )


def _save(fig: plt.Figure, out_base: Path, formats: Iterable[str]) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    _validate_figure_size(fig, out_base.name)
    for fmt in formats:
        fig.savefig(out_base.with_suffix(f".{fmt}"), dpi=DEFAULT_DPI)
    plt.close(fig)


def _avg_rows(
    rows: Sequence[Dict[str, str]],
    group_keys: Sequence[str],
    value_keys: Sequence[str],
) -> List[Dict[str, str]]:
    grouped: Dict[Tuple[str, ...], List[Dict[str, str]]] = {}
    for row in rows:
        key = tuple(str(row[k]) for k in group_keys)
        grouped.setdefault(key, []).append(row)

    out: List[Dict[str, str]] = []
    for key, items in grouped.items():
        merged = {k: v for k, v in zip(group_keys, key)}
        for value_key in value_keys:
            values = [float(item[value_key]) for item in items]
            merged[value_key] = f"{sum(values) / len(values):.12g}"
        out.append(merged)
    return out


def _set_sparse_xticks(ax: plt.Axes, xs: Sequence[float], max_ticks: int = 8) -> None:
    uniq = sorted({float(x) for x in xs})
    if len(uniq) <= max_ticks:
        ax.set_xticks(uniq)
        return
    idx = np.linspace(0, len(uniq) - 1, num=max_ticks, dtype=int)
    chosen = [uniq[i] for i in sorted(set(idx.tolist()))]
    ax.set_xticks(chosen)


def plot_comm_time_vs_asymmetry(
    summary_rows: List[Dict[str, str]], out_dir: Path, formats: Iterable[str]
) -> None:
    agg_rows = _avg_rows(
        summary_rows,
        group_keys=("workload", "algorithm", "asymmetry_level"),
        value_keys=("total_time_ms",),
    )
    fig, axes = plt.subplots(2, 2, figsize=(8, 5), constrained_layout=True)
    workloads = ["tp", "dp", "mixed", "pp"]
    for ax, workload in zip(axes.flat, workloads):
        rows = [row for row in agg_rows if row["workload"] == workload]
        alg_groups = _group(rows, "algorithm")
        for algorithm, alg_rows in sorted(alg_groups.items()):
            alg_rows = sorted(alg_rows, key=lambda item: float(item["asymmetry_level"]))
            xs = [float(item["asymmetry_level"]) for item in alg_rows]
            ys = [float(item["total_time_ms"]) for item in alg_rows]
            ax.plot(xs, ys, marker="o", linewidth=1.6, markersize=4, label=algorithm)
            _set_sparse_xticks(ax, xs)
        ax.set_title(workload.upper())
        ax.set_xlabel("Asymmetry")
        ax.set_ylabel("Comm time (ms)")
        ax.grid(True, alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=8)
    fig.suptitle("Communication time sensitivity to asymmetry", fontsize=11)
    _save(fig, out_dir / "comm_time_vs_asymmetry", formats)


def plot_waste_vs_port_budget(
    summary_rows: List[Dict[str, str]], out_dir: Path, formats: Iterable[str]
) -> None:
    agg_rows = _avg_rows(
        summary_rows,
        group_keys=("algorithm", "port_budget"),
        value_keys=("mean_symmetric_waste_gbps",),
    )
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    alg_groups = _group(agg_rows, "algorithm")
    for algorithm, rows in sorted(alg_groups.items()):
        rows = sorted(rows, key=lambda item: float(item["port_budget"]))
        xs = [float(item["port_budget"]) for item in rows]
        ys = [float(item["mean_symmetric_waste_gbps"]) for item in rows]
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=4, label=algorithm)
        _set_sparse_xticks(ax, xs)
    ax.set_xlabel("Per-node OCS port budget")
    ax.set_ylabel("Symmetric waste (Gbps)")
    ax.set_title("Symmetric capacity waste vs OCS port budget")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    _save(fig, out_dir / "waste_vs_port_budget", formats)


def plot_exposure_breakdown(
    summary_rows: List[Dict[str, str]], out_dir: Path, formats: Iterable[str]
) -> None:
    drac_rows = [row for row in summary_rows if row["algorithm"] == "drac"]
    if not drac_rows:
        return
    agg_rows = _avg_rows(
        drac_rows,
        group_keys=("workload",),
        value_keys=("active_directional_ports", "releasable_directional_ports"),
    )
    agg_rows = sorted(agg_rows, key=lambda item: item["workload"])
    labels = [row["workload"].upper() for row in agg_rows]
    active = np.array([float(row["active_directional_ports"]) for row in agg_rows])
    releasable = np.array([float(row["releasable_directional_ports"]) for row in agg_rows])
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    ax.bar(x, active, width=0.62, label="Pact")
    ax.bar(x, releasable, width=0.62, bottom=active, label="Prel")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Directional resources")
    ax.set_title("DRAC physical resource exposure")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.2)
    _save(fig, out_dir / "drac_resource_exposure", formats)


def plot_matching_cdf(
    raw_rows: List[Dict[str, str]], out_dir: Path, formats: Iterable[str]
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    alg_groups = _group(raw_rows, "algorithm")
    for algorithm, rows in sorted(alg_groups.items()):
        values = np.array([float(row["matching_error_l1"]) for row in rows], dtype=float)
        if values.size == 0:
            continue
        xs = np.sort(values)
        ys = np.arange(1, xs.size + 1) / xs.size
        ax.plot(xs, ys, linewidth=1.8, label=algorithm)
    ax.set_xlabel("Bandwidth matching error (L1)")
    ax.set_ylabel("CDF")
    ax.set_title("Directional bandwidth matching error distribution")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    _save(fig, out_dir / "matching_error_cdf", formats)


def plot_skew_distribution(
    raw_rows: List[Dict[str, str]], out_dir: Path, formats: Iterable[str]
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    workload_groups = _group(raw_rows, "workload")
    for workload, rows in sorted(workload_groups.items()):
        values = np.array([float(row["skew_p95"]) for row in rows], dtype=float)
        if values.size == 0:
            continue
        xs = np.sort(values)
        ys = np.arange(1, xs.size + 1) / xs.size
        ax.plot(xs, ys, linewidth=1.8, label=workload.upper())
    ax.set_xlabel("Directional skew ratio")
    ax.set_ylabel("CDF")
    ax.set_title("Skew distribution across workloads")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, out_dir / "skew_distribution_cdf", formats)


def plot_representative_heatmaps(
    matrix_dir: Path, out_dir: Path, formats: Iterable[str]
) -> None:
    drac_candidates = sorted(matrix_dir.glob("*.workload-dp.*.algorithm-drac.json"))
    if not drac_candidates:
        return
    drac_path = drac_candidates[0]
    sym_path = Path(str(drac_path).replace("algorithm-drac", "algorithm-sym_ocs"))
    demand_path = Path(str(drac_path).replace("algorithm-drac", "algorithm-demand"))
    paths = {"demand": demand_path, "sym_ocs": sym_path, "drac": drac_path}
    if not all(path.exists() for path in paths.values()):
        return

    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.2), constrained_layout=True)
    for ax, (label, path) in zip(axes, paths.items()):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        mat = np.array(payload["matrix"], dtype=float)
        im = ax.imshow(mat, cmap="viridis", aspect="auto")
        ax.set_title(label)
        ax.set_xlabel("dst")
        ax.set_ylabel("src")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("Representative DP workload heatmaps", fontsize=11)
    _save(fig, out_dir / "representative_dp_heatmaps", formats)


def generate_all_figures(
    summary_rows: List[Dict[str, str]],
    raw_rows: List[Dict[str, str]],
    matrix_dir: Path,
    out_dir: Path,
    formats: Iterable[str],
) -> None:
    plot_comm_time_vs_asymmetry(summary_rows, out_dir, formats)
    plot_waste_vs_port_budget(summary_rows, out_dir, formats)
    plot_exposure_breakdown(summary_rows, out_dir, formats)
    plot_matching_cdf(raw_rows, out_dir, formats)
    plot_skew_distribution(raw_rows, out_dir, formats)
    plot_representative_heatmaps(matrix_dir, out_dir, formats)
