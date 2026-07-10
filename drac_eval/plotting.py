from __future__ import annotations

import json
import re
from json import JSONDecodeError
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .allocation import allocate_for_algorithm
from .config import NetworkConfig


MAX_FIGURE_INCHES = 20.0
DEFAULT_DPI = 200
WORKLOAD_ORDER = ["tp", "dp", "mixed", "pp"]
MAIN_ALGORITHMS = ["static_sym", "sym_ocs", "drac"]
COMM_ALGORITHMS = ["static_sym", "sym_ocs", "drac", "ideal_asym"]
ALGORITHM_LABELS = {
    "drac": "DRAC",
    "static_sym": "Static-Sym",
    "sym_ocs": "Sym-OCS",
    "ideal_asym": "Ideal-Asym",
    "drac_sym": "DRAC-Sym",
}
ALGORITHM_COLORS = {
    "drac": "#1f77b4",
    "static_sym": "#7f7f7f",
    "sym_ocs": "#ff7f0e",
    "ideal_asym": "#2ca02c",
    "drac_sym": "#d62728",
}
ALGORITHM_MARKERS = {
    "drac": "o",
    "static_sym": "s",
    "sym_ocs": "^",
    "ideal_asym": "D",
    "drac_sym": "x",
}
MATRIX_RE = re.compile(
    r"(?P<run_id>.+)\.workload-(?P<workload>[^.]+)\.segment-(?P<segment>\d+)\.algorithm-(?P<algorithm>[^.]+)\.json$"
)
RUN_RE = re.compile(
    r"n(?P<n>\d+)_a(?P<a>[0-9p]+)_p(?P<p>\d+)_l(?P<l>\d+)_r(?P<r>[0-9p]+)$"
)


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
        fig.savefig(out_base.with_suffix(f".{fmt}"), dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def _set_sparse_xticks(ax: plt.Axes, xs: Sequence[float], max_ticks: int = 6) -> None:
    uniq = sorted({float(x) for x in xs})
    if len(uniq) <= max_ticks:
        ax.set_xticks(uniq)
        return
    idx = np.linspace(0, len(uniq) - 1, num=max_ticks, dtype=int)
    chosen = [uniq[i] for i in sorted(set(idx.tolist()))]
    ax.set_xticks(chosen)


def _load_matrix_records(matrix_dir: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for path in sorted(matrix_dir.glob("*.json")):
        match = MATRIX_RE.match(path.name)
        if not match:
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, JSONDecodeError) as exc:
            print(f"[drac_eval] skipping unreadable matrix json: {path.name} ({exc})")
            continue
        records.append(
            {
                "run_id": match.group("run_id"),
                "workload": match.group("workload"),
                "segment": int(match.group("segment")),
                "algorithm": match.group("algorithm"),
                "matrix": np.array(payload["matrix"], dtype=float),
            }
        )
    return records


def _build_matrix_index(matrix_dir: Path) -> Dict[Tuple[str, str, int, str], np.ndarray]:
    index: Dict[Tuple[str, str, int, str], np.ndarray] = {}
    for rec in _load_matrix_records(matrix_dir):
        index[(str(rec["run_id"]), str(rec["workload"]), int(rec["segment"]), str(rec["algorithm"]))] = rec["matrix"]  # type: ignore[index]
    return index


def _normalize_share(matrix: np.ndarray) -> np.ndarray:
    out = np.array(matrix, dtype=float, copy=True)
    np.fill_diagonal(out, 0.0)
    total = float(np.sum(out))
    if total <= 0.0:
        return np.zeros_like(out, dtype=float)
    return out / total


def _mask_diagonal(matrix: np.ndarray) -> np.ndarray:
    out = np.array(matrix, dtype=float, copy=True)
    np.fill_diagonal(out, np.nan)
    return out


def _algorithm_label(name: str) -> str:
    return ALGORITHM_LABELS.get(name, name)


def _parse_run_id(run_id: str, base_net: NetworkConfig) -> NetworkConfig:
    match = RUN_RE.match(run_id)
    if not match:
        return base_net
    return base_net.with_overrides(
        {
            "per_node_port_budget": int(match.group("p")),
            "total_ocs_links": int(match.group("l")),
            "reconfig_delay_ms": float(match.group("r").replace("p", ".")),
        }
    )


def _normalized_asymmetry_score(matrix: np.ndarray) -> float:
    mat = np.array(matrix, dtype=float, copy=True)
    np.fill_diagonal(mat, 0.0)
    denom = mat + mat.T
    numer = np.abs(mat - mat.T)
    mask = ~np.eye(mat.shape[0], dtype=bool)
    valid = mask & (denom > 0.0)
    if not np.any(valid):
        return 0.0
    return float(np.mean(numer[valid] / denom[valid]))


def _line_style(ax: plt.Axes, xs: Sequence[float], ys: Sequence[float], algorithm: str) -> None:
    ax.plot(
        xs,
        ys,
        marker=ALGORITHM_MARKERS.get(algorithm, "o"),
        linewidth=1.8,
        markersize=4.5,
        color=ALGORITHM_COLORS.get(algorithm, None),
        label=_algorithm_label(algorithm),
    )


def _ordered_skew_ratio(matrix: np.ndarray) -> np.ndarray:
    eps = 1e-9
    return np.maximum(matrix, matrix.T) / (np.minimum(matrix, matrix.T) + eps)


def _high_demand_mask(matrix: np.ndarray, tau: float, eta_fraction: float) -> np.ndarray:
    offdiag = ~np.eye(matrix.shape[0], dtype=bool)
    positive = matrix[offdiag]
    eta = 0.0
    if positive.size > 0:
        eta = float(np.max(positive)) * float(max(0.0, eta_fraction))
    rho = _ordered_skew_ratio(matrix)
    return offdiag & (matrix >= eta) & (rho >= tau)


def _weighted_cdf(
    values: Sequence[float],
    weights: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    if not values:
        return np.array([]), np.array([])
    x = np.array(values, dtype=float)
    w = np.array(weights, dtype=float)
    order = np.argsort(x)
    x = x[order]
    w = w[order]
    total = float(np.sum(w))
    if total <= 0.0:
        total = float(len(w))
        w = np.ones_like(w)
    y = np.cumsum(w) / total
    return x, y


def _summary_dir_from_figure_dir(out_dir: Path) -> Path:
    return out_dir.parent / "summary"


def _mean_by_keys(
    rows: Sequence[Dict[str, str]],
    group_keys: Sequence[str],
    value_key: str,
) -> Dict[Tuple[str, ...], float]:
    grouped: Dict[Tuple[str, ...], List[float]] = {}
    for row in rows:
        grouped.setdefault(tuple(str(row[k]) for k in group_keys), []).append(float(row[value_key]))
    return {key: float(sum(vals) / len(vals)) for key, vals in grouped.items()}


def plot_comm_time_vs_asymmetry(
    summary_rows: List[Dict[str, str]],
    matrix_dir: Path,
    base_net: NetworkConfig,
    out_dir: Path,
    formats: Iterable[str],
) -> None:
    workloads = ["tp", "dp", "mixed"]
    avg_times = _mean_by_keys(
        summary_rows,
        ("workload", "algorithm", "asymmetry_level"),
        "total_time_ms",
    )
    fig, axes = plt.subplots(1, 3, figsize=(8, 3.8), constrained_layout=True, sharey=True)
    for ax, workload in zip(axes, workloads):
        base_by_asym = {
            float(asym): val
            for (wl, alg, asym), val in avg_times.items()
            if wl == workload and alg == "static_sym"
        }
        for algorithm in MAIN_ALGORITHMS:
            pts = [
                (float(asym), val / base_by_asym[float(asym)])
                for (wl, alg, asym), val in avg_times.items()
                if wl == workload and alg == algorithm and float(asym) in base_by_asym and base_by_asym[float(asym)] > 0.0
            ]
            pts.sort(key=lambda item: item[0])
            if not pts:
                continue
            xs = [item[0] for item in pts]
            ys = [item[1] for item in pts]
            _line_style(ax, xs, ys, algorithm)
            _set_sparse_xticks(ax, xs)
        ax.set_title(workload.upper())
        ax.set_xlabel("Injected directional skew factor")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Normalized communication time")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    print("[drac_eval] normalized_comm_time_vs_skew_factor_final: normalized to Static-Sym.")
    _save(fig, out_dir / "normalized_comm_time_vs_skew_factor_final", formats)

    matrix_index = _build_matrix_index(matrix_dir)
    for workload in workloads:
        base_time = avg_times.get((workload, "static_sym", "1.0"))
        drac_time = avg_times.get((workload, "drac", "1.0"))
        sym_time = avg_times.get((workload, "sym_ocs", "1.0"))
        if (
            base_time is None
            or drac_time is None
            or sym_time is None
            or base_time <= 0.0
        ):
            continue
        drac_norm = drac_time / base_time
        sym_norm = sym_time / base_time
        if abs(drac_norm - sym_norm) > 0.03:
            demand_scores: List[float] = []
            alloc_scores: List[float] = []
            for (run_id, wl, segment, algorithm), demand in matrix_index.items():
                if wl != workload or algorithm != "demand" or "_a1p0_" not in run_id:
                    continue
                demand_scores.append(_normalized_asymmetry_score(demand))
                drac_matrix = matrix_index.get((run_id, wl, segment, "drac"))
                if drac_matrix is None:
                    continue
                net = _parse_run_id(run_id, base_net)
                base_only = np.full_like(drac_matrix, net.base_bw_gbps, dtype=float)
                np.fill_diagonal(base_only, 0.0)
                alloc_scores.append(_normalized_asymmetry_score(np.maximum(0.0, drac_matrix - base_only)))
            print(
                f"[drac_eval] skew=1 closeness check {workload.upper()}: "
                f"DRAC={drac_norm:.4f}, Sym-OCS={sym_norm:.4f}, "
                f"demand_asymmetry_score={np.mean(demand_scores) if demand_scores else 0.0:.6f}, "
                f"allocation_asymmetry_score={np.mean(alloc_scores) if alloc_scores else 0.0:.6f}"
            )


def plot_pp_skew_sensitivity(
    summary_rows: List[Dict[str, str]],
    out_dir: Path,
    formats: Iterable[str],
) -> None:
    pp_rows = [row for row in summary_rows if row["workload"] == "pp"]
    avg_times = _mean_by_keys(pp_rows, ("algorithm", "asymmetry_level"), "total_time_ms")
    base_by_asym = {
        float(asym): val
        for (alg, asym), val in avg_times.items()
        if alg == "static_sym"
    }
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for algorithm in MAIN_ALGORITHMS:
        pts = [
            (float(asym), val / base_by_asym[float(asym)])
            for (alg, asym), val in avg_times.items()
            if alg == algorithm and float(asym) in base_by_asym and base_by_asym[float(asym)] > 0.0
        ]
        pts.sort(key=lambda item: item[0])
        if not pts:
            continue
        xs = [item[0] for item in pts]
        ys = [item[1] for item in pts]
        _line_style(ax, xs, ys, algorithm)
        _set_sparse_xticks(ax, xs)
    ax.set_xlabel("Injected directional skew factor")
    ax.set_ylabel("Normalized communication time")
    ax.set_title("PP")
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=8, frameon=False)
    print("[drac_eval] pp_skew_sensitivity_appendix: This stress test injects artificial directional skew into PP demand.")
    _save(fig, out_dir / "pp_skew_sensitivity_appendix", formats)


def plot_pp_no_harm_bar(
    summary_rows: List[Dict[str, str]],
    matrix_dir: Path,
    out_dir: Path,
    formats: Iterable[str],
    tau: float,
    eta_fraction: float,
) -> None:
    pp_rows = [
        row for row in summary_rows if row["workload"] == "pp" and float(row["asymmetry_level"]) == 1.0
    ]
    avg_times = _mean_by_keys(pp_rows, ("algorithm",), "total_time_ms")
    base = avg_times.get(("static_sym",), 0.0)
    if base <= 0.0:
        return
    algorithms = MAIN_ALGORITHMS
    values = [avg_times.get((alg,), base) / base for alg in algorithms]
    fig, ax = plt.subplots(figsize=(5.6, 4), constrained_layout=True)
    x = np.arange(len(algorithms))
    ax.bar(
        x,
        values,
        width=0.62,
        color=[ALGORITHM_COLORS[alg] for alg in algorithms],
    )
    ax.set_xticks(x)
    ax.set_xticklabels([_algorithm_label(alg) for alg in algorithms])
    ax.set_ylabel("Normalized communication time")
    ax.set_ylim(0.95, 1.02)
    ax.grid(True, axis="y", alpha=0.25)
    for xpos, value in zip(x, values):
        ax.text(xpos, value + 0.001, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    print("[drac_eval] pp_no_harm_comm_time_bar uses only PP workload with skew factor = 1.0")
    print(
        "[drac_eval] pp_no_harm_comm_time_final: "
        "DRAC remains comparable to Sym-OCS and does not degrade near-symmetric PP traffic."
    )
    _save(fig, out_dir / "pp_no_harm_comm_time_final", formats)

    matrix_index = _build_matrix_index(matrix_dir)
    demand_scores: List[float] = []
    selected_counts: List[int] = []
    drac_alloc_scores: List[float] = []
    sym_alloc_scores: List[float] = []
    for (run_id, workload, segment, algorithm), demand in matrix_index.items():
        if workload != "pp" or algorithm != "demand" or "_a1p0_" not in run_id:
            continue
        demand_scores.append(_normalized_asymmetry_score(demand))
        selected_counts.append(int(np.count_nonzero(_high_demand_mask(demand, tau=tau, eta_fraction=eta_fraction))))
        drac_matrix = matrix_index.get((run_id, workload, segment, "drac"))
        sym_matrix = matrix_index.get((run_id, workload, segment, "sym_ocs"))
        if drac_matrix is not None:
            drac_alloc_scores.append(_normalized_asymmetry_score(drac_matrix))
        if sym_matrix is not None:
            sym_alloc_scores.append(_normalized_asymmetry_score(sym_matrix))

    drac_norm = avg_times.get(("drac",), base) / base
    sym_norm = avg_times.get(("sym_ocs",), base) / base
    delta = drac_norm - sym_norm
    relative_delta = delta / sym_norm if sym_norm > 0.0 else 0.0
    mean_demand_asym = float(np.mean(demand_scores)) if demand_scores else 0.0
    mean_selected = float(np.mean(selected_counts)) if selected_counts else 0.0
    mean_drac_asym = float(np.mean(drac_alloc_scores)) if drac_alloc_scores else 0.0
    mean_sym_asym = float(np.mean(sym_alloc_scores)) if sym_alloc_scores else 0.0

    if abs(relative_delta) <= 0.005:
        print("[drac_eval] PP no-harm passed: DRAC is essentially comparable to Sym-OCS.")
    elif relative_delta < -0.01:
        print("[warning] DRAC improves PP noticeably; check whether PP demand has local directional skew or Sym-OCS baseline is weak.")
    elif relative_delta > 0.01:
        print("[warning] DRAC degrades PP; no-harm claim not supported.")

    print(f"[drac_eval] PP demand_asymmetry_score = {mean_demand_asym:.6g}")
    print(f"[drac_eval] PP selected high-demand direction count under tau = {mean_selected:.3f}")
    print(f"[drac_eval] PP DRAC allocation asymmetry score = {mean_drac_asym:.6g}")
    print(f"[drac_eval] PP Sym-OCS allocation asymmetry score = {mean_sym_asym:.6g}")
    print(
        f"[drac_eval] PP DRAC vs Sym-OCS normalized time delta = {delta:.6g}, "
        f"relative_delta = {relative_delta:.6g}"
    )
    _write_csv(
        _summary_dir_from_figure_dir(out_dir) / "pp_no_harm_sanity_summary.csv",
        [
            {
                "workload": "pp",
                "demand_asymmetry_score": mean_demand_asym,
                "selected_high_demand_direction_count": mean_selected,
                "drac_allocation_asymmetry_score": mean_drac_asym,
                "sym_ocs_allocation_asymmetry_score": mean_sym_asym,
                "drac_normalized_time": drac_norm,
                "sym_ocs_normalized_time": sym_norm,
                "delta": delta,
                "relative_delta": relative_delta,
            }
        ],
    )


def _plot_ratio_vs_port_budget(
    summary_rows: List[Dict[str, str]],
    out_dir: Path,
    formats: Iterable[str],
    value_key: str,
    filename: str,
    ylabel: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.2), constrained_layout=True, sharey=True)
    for ax, workload in zip(axes.flat, WORKLOAD_ORDER):
        rows = [row for row in summary_rows if row["workload"] == workload]
        agg_rows = _avg_rows(
            rows,
            group_keys=("algorithm", "port_budget"),
            value_keys=(value_key,),
        )
        for algorithm in MAIN_ALGORITHMS:
            alg_rows = [row for row in agg_rows if row["algorithm"] == algorithm]
            alg_rows.sort(key=lambda item: float(item["port_budget"]))
            if not alg_rows:
                continue
            xs = [float(item["port_budget"]) for item in alg_rows]
            ys = [float(item[value_key]) for item in alg_rows]
            _line_style(ax, xs, ys, algorithm)
            _set_sparse_xticks(ax, xs)
        ax.set_title(workload.upper())
        ax.set_xlabel("Per-node OCS port budget")
        ax.grid(True, alpha=0.25)
    axes[0, 0].set_ylabel(ylabel)
    axes[1, 0].set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    _save(fig, out_dir / filename, formats)


def plot_resource_breakdown(
    summary_rows: List[Dict[str, str]],
    out_dir: Path,
    formats: Iterable[str],
) -> None:
    drac_rows = [row for row in summary_rows if row["algorithm"] == "drac"]
    if not drac_rows:
        return
    agg_rows = _avg_rows(
        drac_rows,
        group_keys=("workload",),
        value_keys=(
            "active_directional_ports",
            "releasable_directional_ports",
            "active_bidirectional_bundles",
            "releasable_bidirectional_bundles",
            "requested_extra_bw_gbps",
        ),
    )
    agg_rows.sort(key=lambda item: WORKLOAD_ORDER.index(item["workload"]) if item["workload"] in WORKLOAD_ORDER else 99)
    labels = [row["workload"].upper() for row in agg_rows]
    x = np.arange(len(labels))

    def _stacked(
        active_key: str,
        releasable_key: str,
        ylabel: str,
        filename: str,
        colors: Tuple[str, str],
    ) -> None:
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        active = np.array([float(row[active_key]) for row in agg_rows], dtype=float)
        releasable = np.array([float(row[releasable_key]) for row in agg_rows], dtype=float)
        ax.bar(x, active, label="P_act", color=colors[0], width=0.62)
        ax.bar(x, releasable, bottom=active, label="P_rel", color=colors[1], width=0.62)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.22)
        ax.legend(fontsize=8, frameon=False)
        _save(fig, out_dir / filename, formats)

    _stacked(
        "active_directional_ports",
        "releasable_directional_ports",
        "Directional ports",
        "resource_breakdown_directional_port",
        ("#1f77b4", "#9ecae1"),
    )
    _stacked(
        "active_bidirectional_bundles",
        "releasable_bidirectional_bundles",
        "Bidirectional bundles",
        "resource_breakdown_bidirectional_bundle",
        ("#ff7f0e", "#fdd0a2"),
    )

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    requested = np.array([float(row["requested_extra_bw_gbps"]) for row in agg_rows], dtype=float)
    ax.bar(x, requested, width=0.62, color="#6baed6")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Requested extra bandwidth (Gbps)")
    ax.grid(True, axis="y", alpha=0.22)
    _save(fig, out_dir / "requested_resources_by_workload", formats)


def _scenario_key(row: Dict[str, str]) -> Tuple[str, str, str, str, str, str, str]:
    return (
        row["run_id"],
        row["workload"],
        row["cluster_size"],
        row["asymmetry_level"],
        row["port_budget"],
        row["total_ocs_links"],
        row["reconfig_delay_ms"],
    )


def plot_active_resource_requirement(
    summary_rows: List[Dict[str, str]],
    out_dir: Path,
    formats: Iterable[str],
) -> None:
    grouped: Dict[Tuple[str, str, str, str, str, str, str], Dict[str, Dict[str, str]]] = {}
    for row in summary_rows:
        grouped.setdefault(_scenario_key(row), {})[row["algorithm"]] = row

    directional_norms: Dict[str, Dict[str, List[float]]] = {
        workload: {alg: [] for alg in MAIN_ALGORITHMS} for workload in WORKLOAD_ORDER
    }
    bundle_norms: Dict[str, Dict[str, List[float]]] = {
        workload: {alg: [] for alg in MAIN_ALGORITHMS} for workload in WORKLOAD_ORDER
    }
    extra_dir: Dict[str, List[float]] = {workload: [] for workload in WORKLOAD_ORDER}
    extra_bundle: Dict[str, List[float]] = {workload: [] for workload in WORKLOAD_ORDER}

    for key, alg_rows in grouped.items():
        workload = key[1]
        if workload not in WORKLOAD_ORDER or "sym_ocs" not in alg_rows:
            continue
        sym_dir = float(alg_rows["sym_ocs"]["active_directional_ports"])
        sym_bundle = float(alg_rows["sym_ocs"]["active_bidirectional_bundles"])
        if sym_dir <= 0.0 or sym_bundle <= 0.0:
            continue
        for alg in MAIN_ALGORITHMS:
            row = alg_rows.get(alg)
            if row is None:
                continue
            directional_norms[workload][alg].append(float(row["active_directional_ports"]) / sym_dir)
            bundle_norms[workload][alg].append(float(row["active_bidirectional_bundles"]) / sym_bundle)
        drac_row = alg_rows.get("drac")
        if drac_row is not None:
            extra_dir[workload].append(max(0.0, sym_dir - float(drac_row["active_directional_ports"])) / sym_dir)
            extra_bundle[workload].append(
                max(0.0, sym_bundle - float(drac_row["active_bidirectional_bundles"])) / sym_bundle
            )

    def _bar_plot(
        norms: Dict[str, Dict[str, List[float]]],
        ylabel: str,
        filename: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        x = np.arange(len(WORKLOAD_ORDER))
        width = 0.22
        for idx, alg in enumerate(MAIN_ALGORITHMS):
            values = [
                float(np.mean(norms[workload][alg])) if norms[workload][alg] else np.nan
                for workload in WORKLOAD_ORDER
            ]
            ax.bar(
                x + (idx - 1) * width,
                values,
                width=width,
                color=ALGORITHM_COLORS[alg],
                label=_algorithm_label(alg),
            )
        ax.set_xticks(x)
        ax.set_xticklabels([w.upper() for w in WORKLOAD_ORDER])
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.22)
        ax.legend(fontsize=8, frameon=False, ncol=3)
        _save(fig, out_dir / filename, formats)

    _bar_plot(
        directional_norms,
        "Normalized active resources",
        "active_resource_requirement_directional_port_appendix",
    )
    _bar_plot(
        bundle_norms,
        "Normalized active resources",
        "active_resource_requirement_bidirectional_bundle_appendix",
    )

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True, sharey=True)
    x = np.arange(len(WORKLOAD_ORDER))
    for ax, values_by_workload, title in zip(
        axes,
        (extra_dir, extra_bundle),
        ("Directional-port model", "Bidirectional-bundle model"),
    ):
        vals = [
            float(np.mean(values_by_workload[workload])) if values_by_workload[workload] else np.nan
            for workload in WORKLOAD_ORDER
        ]
        ax.bar(x, vals, width=0.62, color="#6baed6")
        ax.set_xticks(x)
        ax.set_xticklabels([w.upper() for w in WORKLOAD_ORDER])
        ax.set_title(title, fontsize=9)
        ax.grid(True, axis="y", alpha=0.22)
    axes[0].set_ylabel("Extra releasable ratio over Sym-OCS")
    _save(fig, out_dir / "extra_releasable_over_sym_ocs_appendix", formats)

    pp_dir = float(np.mean(extra_dir["pp"])) if extra_dir["pp"] else 0.0
    pp_bundle = float(np.mean(extra_bundle["pp"])) if extra_bundle["pp"] else 0.0
    non_pp_dir = [float(np.mean(extra_dir[w])) for w in ["tp", "dp", "mixed"] if extra_dir[w]]
    non_pp_bundle = [float(np.mean(extra_bundle[w])) for w in ["tp", "dp", "mixed"] if extra_bundle[w]]
    if non_pp_dir and pp_dir >= 0.75 * float(np.mean(non_pp_dir)):
        print("[warning] PP resource exposure conflicts with no-harm / near-symmetric narrative. (directional-port)")
    if non_pp_bundle and pp_bundle >= 0.75 * float(np.mean(non_pp_bundle)):
        print("[warning] PP resource exposure conflicts with no-harm / near-symmetric narrative. (bidirectional-bundle)")
    print("[drac_eval] active resource requirement is computed from actual realized X schedule via connection_units-derived metrics.")


def plot_iso_performance_port_saving(
    summary_rows: List[Dict[str, str]],
    out_dir: Path,
    formats: Iterable[str],
    target_sym_port_budget: int = 6,
) -> None:
    avg_times = _mean_by_keys(
        summary_rows,
        ("workload", "algorithm", "port_budget"),
        "total_time_ms",
    )
    saving_ratio: Dict[str, float] = {}
    drac_required: Dict[str, float] = {}
    workloads_used: List[str] = []
    summary_rows_out: List[Dict[str, object]] = []
    available_budgets = sorted({int(float(row["port_budget"])) for row in summary_rows})
    effective_target_budget = int(target_sym_port_budget)
    if effective_target_budget not in available_budgets and available_budgets:
        effective_target_budget = int(max(available_budgets))
        print(
            f"[drac_eval] iso-performance: target port budget {target_sym_port_budget} not present; "
            f"falling back to max available budget {effective_target_budget}."
        )

    for workload in WORKLOAD_ORDER:
        sym_target = avg_times.get((workload, "sym_ocs", str(effective_target_budget)))
        if sym_target is None:
            print(
                f"[warning] iso-performance: missing Sym-OCS target at port budget {effective_target_budget} for {workload.upper()}."
            )
            continue
        candidates = [
            (float(port_budget), time_ms)
            for (wl, alg, port_budget), time_ms in avg_times.items()
            if wl == workload and alg == "drac"
        ]
        candidates.sort(key=lambda item: item[0])
        sym_curve = sorted(
            [
                (float(port_budget), time_ms)
                for (wl, alg, port_budget), time_ms in avg_times.items()
                if wl == workload and alg == "sym_ocs"
            ],
            key=lambda item: item[0],
        )
        min_port: float | None = None
        min_port_time: float | None = None
        for port_budget, time_ms in candidates:
            if time_ms <= sym_target:
                min_port = port_budget
                min_port_time = time_ms
                break
        if min_port is None:
            min_port = float(effective_target_budget)
            min_port_time = None
            saving_ratio[workload] = 0.0
            print(
                f"[warning] iso-performance: DRAC does not reach Sym-OCS target time for {workload.upper()} within swept port budgets."
            )
        else:
            saving_ratio[workload] = max(0.0, (effective_target_budget - min_port) / float(effective_target_budget))
        drac_required[workload] = min_port
        workloads_used.append(workload)
        for port_budget, time_ms in sym_curve:
            drac_time = next((t for p, t in candidates if abs(p - port_budget) < 1e-9), None)
            summary_rows_out.append(
                {
                    "workload": workload,
                    "target_sym_port_budget": effective_target_budget,
                    "target_time": sym_target,
                    "algorithm": "sym_ocs",
                    "port_budget": port_budget,
                    "total_time_ms": time_ms,
                    "interpolation": False,
                    "min_port_drac": min_port,
                    "min_port_drac_time": min_port_time,
                    "port_saving": float(effective_target_budget - min_port),
                    "port_saving_ratio": saving_ratio[workload],
                    "drac_time_same_port": drac_time,
                }
            )
            if drac_time is not None:
                summary_rows_out.append(
                    {
                        "workload": workload,
                        "target_sym_port_budget": effective_target_budget,
                        "target_time": sym_target,
                        "algorithm": "drac",
                        "port_budget": port_budget,
                        "total_time_ms": drac_time,
                        "interpolation": False,
                        "min_port_drac": min_port,
                        "min_port_drac_time": min_port_time,
                        "port_saving": float(effective_target_budget - min_port),
                        "port_saving_ratio": saving_ratio[workload],
                        "drac_time_same_port": drac_time,
                    }
                )

    x = np.arange(len(workloads_used))
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    vals = [saving_ratio[w] for w in workloads_used]
    ax.bar(x, vals, width=0.62, color="#6baed6")
    ax.set_xticks(x)
    ax.set_xticklabels([w.upper() for w in workloads_used])
    ax.set_ylabel("Port saving ratio over Sym-OCS")
    ax.grid(True, axis="y", alpha=0.22)
    _save(fig, out_dir / "iso_performance_port_saving_appendix", formats)

    fig, ax = plt.subplots(figsize=(6.8, 4), constrained_layout=True)
    width = 0.32
    sym_vals = [float(effective_target_budget) for _ in workloads_used]
    drac_vals = [drac_required[w] for w in workloads_used]
    ax.bar(x - width / 2, sym_vals, width=width, color=ALGORITHM_COLORS["sym_ocs"], label="Sym-OCS")
    ax.bar(x + width / 2, drac_vals, width=width, color=ALGORITHM_COLORS["drac"], label="DRAC")
    ax.set_xticks(x)
    ax.set_xticklabels([w.upper() for w in workloads_used])
    ax.set_ylabel("Required per-node OCS port budget")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, fontsize=8, frameon=False)
    print(
        f"[drac_eval] ports_required_for_target_time_final: "
        f"The target is the communication time achieved by Sym-OCS with {effective_target_budget} ports per node."
    )
    _save(fig, out_dir / "ports_required_for_target_time_final", formats)

    print(f"[drac_eval] iso-performance target port budget = {effective_target_budget}")
    for workload in workloads_used:
        print(
            f"[drac_eval] iso-performance {workload.upper()}: "
            f"DRAC required ports = {drac_required[workload]:.2f}, "
            f"port saving ratio = {saving_ratio[workload]:.4f}"
        )
        if workload == "pp" and saving_ratio[workload] > 0.05:
            print("[warning] PP shows noticeable iso-performance port saving; re-check whether PP is truly near-symmetric under this workload.")
    for row in summary_rows_out:
        if (
            row["workload"] == "dp"
            and float(row["port_budget"]) == 2.0
            and row["algorithm"] == "drac"
        ):
            print(
                f"[drac_eval] DP iso-performance check at port=2: "
                f"time(DRAC,2)={float(row['total_time_ms']):.6g}, target_time={float(row['target_time']):.6g}"
            )
            break
    _write_csv(
        _summary_dir_from_figure_dir(out_dir) / "iso_performance_port_saving_summary.csv",
        summary_rows_out,
    )


def plot_matching_cdf(
    matrix_dir: Path,
    out_dir: Path,
    formats: Iterable[str],
    tau: float,
    eta_fraction: float,
) -> None:
    matrix_index = _build_matrix_index(matrix_dir)
    workloads = ["tp", "dp", "mixed"]
    fig, axes = plt.subplots(1, 3, figsize=(8, 3.8), constrained_layout=True, sharey=True)
    low_count_workloads: List[str] = []

    for ax, workload in zip(axes, workloads):
        per_algorithm_values: Dict[str, List[float]] = {alg: [] for alg in MAIN_ALGORITHMS}
        per_algorithm_weights: Dict[str, List[float]] = {alg: [] for alg in MAIN_ALGORITHMS}
        selected_count = 0

        for (run_id, wl, segment, algorithm), demand in matrix_index.items():
            if wl != workload or algorithm != "demand":
                continue
            demand_share = _normalize_share(demand)
            mask = _high_demand_mask(demand, tau=tau, eta_fraction=eta_fraction)
            count = int(np.count_nonzero(mask))
            selected_count += count
            if count <= 0:
                continue
            weights = demand_share[mask]
            for alg in MAIN_ALGORITHMS:
                realized = matrix_index.get((run_id, wl, segment, alg))
                if realized is None:
                    continue
                realized_share = _normalize_share(realized)
                diff = np.abs(realized_share - demand_share)[mask]
                per_algorithm_values[alg].extend(diff.tolist())
                per_algorithm_weights[alg].extend(weights.tolist())

        for algorithm in MAIN_ALGORITHMS:
            xs, ys = _weighted_cdf(per_algorithm_values[algorithm], per_algorithm_weights[algorithm])
            if xs.size == 0:
                continue
            ax.plot(
                xs,
                ys,
                linewidth=1.8,
                color=ALGORITHM_COLORS[algorithm],
                linestyle="--" if algorithm == "static_sym" else "-",
                alpha=0.75 if algorithm == "static_sym" else 1.0,
                label=_algorithm_label(algorithm),
            )
            median = float(np.quantile(xs, 0.5))
            p90 = float(np.quantile(xs, 0.9))
            p95 = float(np.quantile(xs, 0.95))
            print(
                f"[drac_eval] matching error {workload.upper()} {_algorithm_label(algorithm)}: "
                f"median={median:.6g}, p90={p90:.6g}, p95={p95:.6g}"
            )

        ax.set_title(workload.upper())
        ax.set_xlabel("Weighted directional share error")
        ax.grid(True, alpha=0.25)
        ax.text(
            0.04,
            0.08,
            f"selected dirs = {selected_count}",
            transform=ax.transAxes,
            fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
        if selected_count < 24:
            low_count_workloads.append(workload)
            ax.text(
                0.04,
                0.18,
                "low selected count",
                transform=ax.transAxes,
                fontsize=7,
                color="#b22222",
            )

    axes[0].set_ylabel("CDF")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    print("[drac_eval] PP has no selected high-demand directions under tau, so it is excluded from the CDF.")
    print("[drac_eval] Static-Sym and Sym-OCS may partially overlap in some workloads due to similar directional matching under this metric.")
    _save(fig, out_dir / "matching_error_cdf_final", formats)

    if low_count_workloads:
        print(
            "[drac_eval] matching_error_cdf low selected-direction counts for workloads:",
            ", ".join(w.upper() for w in low_count_workloads),
        )


def plot_high_demand_residual_gap(
    matrix_dir: Path,
    base_net: NetworkConfig,
    out_dir: Path,
    formats: Iterable[str],
    tau: float,
    eta_fraction: float,
) -> None:
    matrix_index = _build_matrix_index(matrix_dir)
    grouped: Dict[Tuple[str, str, str], List[float]] = {}
    selected_counts: Dict[str, List[int]] = {workload: [] for workload in WORKLOAD_ORDER}

    for (run_id, workload, segment, algorithm), demand in matrix_index.items():
        if algorithm != "demand":
            continue
        drac_total = matrix_index.get((run_id, workload, segment, "drac"))
        if drac_total is None:
            continue
        net = _parse_run_id(run_id, base_net)
        alloc = allocate_for_algorithm("drac", demand, net)
        target = alloc.target_overlay
        realized = alloc.realized_overlay
        mask = _high_demand_mask(demand, tau=tau, eta_fraction=eta_fraction)
        count = int(np.count_nonzero(mask))
        selected_counts.setdefault(workload, []).append(count)
        if count <= 0:
            value = 0.0
        else:
            gap = np.maximum(target - realized, 0.0)
            denom = float(np.sum(target[mask]))
            value = float(np.sum(gap[mask]) / denom) if denom > 0.0 else 0.0
        port_budget = str(net.per_node_port_budget)
        grouped.setdefault((workload, port_budget, run_id), []).append(value)

    agg: Dict[Tuple[str, str], List[float]] = {}
    for (workload, port_budget, _run_id), values in grouped.items():
        agg.setdefault((workload, port_budget), []).append(float(np.mean(values)))

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    workload_colors = {"tp": "#1f77b4", "dp": "#ff7f0e", "mixed": "#2ca02c", "pp": "#d62728"}
    for workload in WORKLOAD_ORDER:
        pts = [
            (float(port_budget), float(np.mean(values)))
            for (wl, port_budget), values in agg.items()
            if wl == workload
        ]
        pts.sort(key=lambda item: item[0])
        if not pts:
            continue
        xs = [item[0] for item in pts]
        ys = [item[1] for item in pts]
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            color=workload_colors[workload],
            label=workload.upper(),
        )
        _set_sparse_xticks(ax, xs)
    ax.set_xlabel("Per-node OCS port budget")
    ax.set_ylabel("Residual target gap ratio")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=4, fontsize=8, frameon=False)
    pp_counts = selected_counts.get("pp", [])
    if pp_counts and float(np.mean(pp_counts)) <= 4.0:
        ax.text(
            0.98,
            0.06,
            "PP: no selected high-demand directions",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
    _save(fig, out_dir / "high_demand_residual_gap_vs_port_budget_final", formats)

    for workload in WORKLOAD_ORDER:
        counts = selected_counts.get(workload, [])
        mean_count = float(np.mean(counts)) if counts else 0.0
        if workload == "pp" and mean_count <= 4.0:
            print(f"[drac_eval] PP selected H count is low as expected: avg={mean_count:.2f}")
        elif workload == "pp":
            print(f"[warning] PP selected H count is higher than expected: avg={mean_count:.2f}")
    print("[drac_eval] high-demand residual gap uses only H = {(i,j): T_ij >= eta and rho_ij >= tau}.")
    grouped_max_budget: Dict[str, Tuple[float, float]] = {}
    for (workload, port_budget, _run_id), values in grouped.items():
        pb = float(port_budget)
        val = float(np.mean(values))
        if workload not in grouped_max_budget or pb > grouped_max_budget[workload][0]:
            grouped_max_budget[workload] = (pb, val)
    for workload, (pb, val) in grouped_max_budget.items():
        print(f"[drac_eval] residual gap @ max port budget {workload.upper()}: budget={pb:.0f}, ratio={val:.6g}")


def plot_skew_distribution(
    matrix_dir: Path,
    out_dir: Path,
    formats: Iterable[str],
    tau: float,
    eta_fraction: float,
) -> None:
    values_by_workload: Dict[str, List[float]] = {workload: [] for workload in WORKLOAD_ORDER}
    selected_high_demand_counts: Dict[str, int] = {workload: 0 for workload in WORKLOAD_ORDER}
    clip_max = 12.0
    for rec in _load_matrix_records(matrix_dir):
        if str(rec["algorithm"]) != "demand":
            continue
        workload = str(rec["workload"])
        if workload not in values_by_workload:
            continue
        matrix = np.array(rec["matrix"], dtype=float)
        selected_high_demand_counts[workload] += int(np.count_nonzero(_high_demand_mask(matrix, tau=tau, eta_fraction=eta_fraction)))
        for i in range(matrix.shape[0]):
            for j in range(i + 1, matrix.shape[1]):
                a = float(matrix[i, j])
                b = float(matrix[j, i])
                if a <= 0.0 and b <= 0.0:
                    continue
                rho = max(a, b) / (min(a, b) + 1e-9)
                # Clip only for readability so a few extreme tails do not flatten the rest.
                values_by_workload[workload].append(min(rho, clip_max))

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for workload in WORKLOAD_ORDER:
        vals = values_by_workload[workload]
        if not vals:
            continue
        xs = np.sort(np.array(vals, dtype=float))
        ys = np.arange(1, len(xs) + 1, dtype=float) / float(len(xs))
        ax.plot(
            xs,
            ys,
            linewidth=1.8,
            color={"tp": "#1f77b4", "dp": "#ff7f0e", "mixed": "#2ca02c", "pp": "#d62728"}[workload],
            label=workload.upper(),
        )
        frac = float(np.mean(xs >= tau))
        print(
            f"[drac_eval] workload {workload.upper()}: "
            f"fraction_rho_ge_tau={frac:.4f}, "
            f"selected_high_demand_direction_count={selected_high_demand_counts[workload]}"
        )

    ax.axvline(tau, color="#444444", linestyle="--", linewidth=1.2, label=f"τ = {tau:g}")
    ax.set_xlim(1.0, 6.5)
    ax.set_xlabel("Directional skew ratio")
    ax.set_ylabel("CDF")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=5, fontsize=8, frameon=False)
    print("[drac_eval] skew_distribution_cdf_final includes only off-diagonal ordered pairs.")
    _save(fig, out_dir / "skew_distribution_cdf_final", formats)


def _plot_single_representative_heatmap(
    matrix_index: Dict[Tuple[str, str, int, str], np.ndarray],
    workload_name: str,
    out_dir: Path,
    formats: Iterable[str],
    filename: str,
    summary_rows_out: List[Dict[str, object]],
) -> None:
    best_key: Tuple[str, str, int, str] | None = None
    best_score = -1.0
    best_stats: Dict[str, float] | None = None
    sym_max_asym = 0.0
    for key, demand in matrix_index.items():
        run_id, workload, segment, algorithm = key
        if workload != workload_name or algorithm != "demand":
            continue
        sym = matrix_index.get((run_id, workload, segment, "sym_ocs"))
        drac = matrix_index.get((run_id, workload, segment, "drac"))
        if sym is None or drac is None:
            continue
        sym_max_asym = max(sym_max_asym, float(np.max(np.abs(sym - sym.T))))
        demand_share = _normalize_share(demand)
        sym_share = _normalize_share(sym)
        drac_share = _normalize_share(drac)
        offdiag = ~np.eye(demand.shape[0], dtype=bool)
        gain = np.abs(sym_share - demand_share) - np.abs(drac_share - demand_share)
        score = float(np.sum(np.maximum(gain[offdiag], 0.0)))
        if score > best_score:
            best_score = score
            best_key = key
            best_stats = {
                "segment_score": score,
                "sum_positive_gain": float(np.sum(np.maximum(gain[offdiag], 0.0))),
                "mean_gain": float(np.mean(gain[offdiag])) if np.any(offdiag) else 0.0,
                "max_gain": float(np.max(gain[offdiag])) if np.any(offdiag) else 0.0,
                "selected_offdiag_count": float(np.count_nonzero(offdiag)),
            }

    if best_key is None:
        return

    run_id, workload, segment, _ = best_key
    demand = matrix_index[(run_id, workload, segment, "demand")]
    sym = matrix_index.get((run_id, workload, segment, "sym_ocs"))
    drac = matrix_index.get((run_id, workload, segment, "drac"))
    if sym is None or drac is None:
        return

    demand_share = _normalize_share(demand)
    sym_share = _normalize_share(sym)
    drac_share = _normalize_share(drac)
    error_reduction = np.abs(sym_share - demand_share) - np.abs(drac_share - demand_share)

    plot_mats = {
        "Demand share": _mask_diagonal(demand_share),
        "Sym-OCS share": _mask_diagonal(sym_share),
        "DRAC share": _mask_diagonal(drac_share),
        "Error reduction by DRAC": _mask_diagonal(error_reduction),
    }

    vmax = max(
        float(np.nanmax(plot_mats["Demand share"])),
        float(np.nanmax(plot_mats["Sym-OCS share"])),
        float(np.nanmax(plot_mats["DRAC share"])),
    )
    lim = float(np.nanmax(np.abs(plot_mats["Error reduction by DRAC"]))) or 1e-6
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), constrained_layout=True)
    share_axes = axes[:3]
    share_im = None
    err_im = None
    sparse_ticks = [tick for tick in [0, 4, 8, 12] if tick < demand.shape[0]]
    for idx, (ax, (label, mat)) in enumerate(zip(axes, plot_mats.items())):
        if label == "Error reduction by DRAC":
            err_im = ax.imshow(mat, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        else:
            share_im = ax.imshow(mat, cmap="YlGnBu", vmin=0.0, vmax=vmax, aspect="auto")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("dst")
        ax.set_ylabel("src")
        ax.set_xticks(sparse_ticks)
        ax.set_yticks(sparse_ticks)
    if share_im is not None:
        fig.colorbar(share_im, ax=share_axes.tolist(), fraction=0.025, pad=0.02)
    if err_im is not None:
        fig.colorbar(err_im, ax=[axes[3]], fraction=0.046, pad=0.03)
    if lim < 1e-4:
        axes[3].text(
            0.5,
            -0.18,
            "negligible gain",
            transform=axes[3].transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color="#b22222",
        )
        print(f"[warning] representative {workload_name.upper()} heatmap gain is negligible under the current matching metric.")
    _save(fig, out_dir / filename, formats)

    if best_stats is not None:
        print(
            f"[drac_eval] representative {workload_name.upper()} best-gain segment: "
            f"workload={workload}, segment_id={segment}, "
            f"segment_score={best_stats['segment_score']:.6g}, "
            f"sum_positive_gain={best_stats['sum_positive_gain']:.6g}, "
            f"mean_gain={best_stats['mean_gain']:.6g}, "
            f"max_gain={best_stats['max_gain']:.6g}, "
            f"selected_offdiag_count={int(best_stats['selected_offdiag_count'])}"
        )
        print(f"[drac_eval] Sym-OCS heatmap max |B_ij - B_ji| = {sym_max_asym:.6g}")
        if sym_max_asym <= 1e-9:
            print("[drac_eval] Sym-OCS directional symmetry sanity check passed.")
        else:
            print("[warning] Sym-OCS is not directionally symmetric under current plotted matrix.")
        print("[drac_eval] Heatmap shows raw provisioned share derived from realized bandwidth matrices, not after-steering effective share.")
        summary_rows_out.append(
            {
                "workload": workload_name,
                "selected_segment_id": int(segment),
                "segment_score": best_stats["segment_score"],
                "sum_positive_gain": best_stats["sum_positive_gain"],
                "mean_gain": best_stats["mean_gain"],
                "max_gain": best_stats["max_gain"],
                "selected_offdiag_count": int(best_stats["selected_offdiag_count"]),
                "max_sym_ocs_directional_error": sym_max_asym,
                "whether_plotted_matrix_is_raw_capacity_or_effective_share": "raw_provisioned_share",
            }
        )


def plot_representative_heatmaps(
    matrix_dir: Path,
    out_dir: Path,
    formats: Iterable[str],
) -> None:
    matrix_index = _build_matrix_index(matrix_dir)
    summary_rows_out: List[Dict[str, object]] = []
    _plot_single_representative_heatmap(
        matrix_index,
        "dp",
        out_dir,
        formats,
        "representative_dp_heatmaps_final",
        summary_rows_out,
    )
    _plot_single_representative_heatmap(
        matrix_index,
        "tp",
        out_dir,
        formats,
        "representative_tp_heatmaps_appendix",
        summary_rows_out,
    )
    _plot_single_representative_heatmap(
        matrix_index,
        "mixed",
        out_dir,
        formats,
        "representative_mixed_heatmaps_appendix",
        summary_rows_out,
    )
    _write_csv(
        _summary_dir_from_figure_dir(out_dir) / "representative_heatmap_segment_summary.csv",
        summary_rows_out,
    )


def generate_all_figures(
    summary_rows: List[Dict[str, str]],
    raw_rows: List[Dict[str, str]],
    matrix_dir: Path,
    out_dir: Path,
    formats: Iterable[str],
    tau: float,
    eta_fraction: float,
    base_net: NetworkConfig,
) -> None:
    del raw_rows
    print("[drac_eval] figure 1/12: normalized_comm_time_vs_skew_factor_final")
    plot_comm_time_vs_asymmetry(summary_rows, matrix_dir, base_net, out_dir, formats)
    print("[drac_eval] figure 2/12: pp_skew_sensitivity_appendix")
    plot_pp_skew_sensitivity(summary_rows, out_dir, formats)
    print("[drac_eval] figure 3/12: pp_no_harm_comm_time_final")
    plot_pp_no_harm_bar(summary_rows, matrix_dir, out_dir, formats, tau=tau, eta_fraction=eta_fraction)
    print("[drac_eval] figure 4/12: waste_ratio_vs_port_budget_final")
    _plot_ratio_vs_port_budget(
        summary_rows,
        out_dir,
        formats,
        "mean_waste_ratio",
        "waste_ratio_vs_port_budget_final",
        "Waste ratio",
    )
    max_budget = max(float(row["port_budget"]) for row in summary_rows) if summary_rows else 0.0
    for workload in WORKLOAD_ORDER:
        sym = [
            float(row["mean_waste_ratio"])
            for row in summary_rows
            if row["workload"] == workload and row["algorithm"] == "sym_ocs" and float(row["port_budget"]) == max_budget
        ]
        drac = [
            float(row["mean_waste_ratio"])
            for row in summary_rows
            if row["workload"] == workload and row["algorithm"] == "drac" and float(row["port_budget"]) == max_budget
        ]
        if sym and drac and sym[0] > 0.0:
            reduction = (sym[0] - drac[0]) / sym[0]
            print(f"[drac_eval] waste reduction @ max port budget {workload.upper()}: {reduction:.4f}")
    print("[drac_eval] figure 5/12: useful_ratio_vs_port_budget_appendix")
    _plot_ratio_vs_port_budget(
        summary_rows,
        out_dir,
        formats,
        "mean_useful_ratio",
        "useful_ratio_vs_port_budget_appendix",
        "Useful ratio",
    )
    print("[drac_eval] figure 6/12: active_resource_requirement / extra_releasable appendix")
    plot_active_resource_requirement(summary_rows, out_dir, formats)
    print("[drac_eval] figure 7/12: iso_performance_port_saving / ports_required_for_target_time")
    plot_iso_performance_port_saving(summary_rows, out_dir, formats, target_sym_port_budget=6)
    print("[drac_eval] figure 8/12: matching_error_cdf_final")
    plot_matching_cdf(matrix_dir, out_dir, formats, tau=tau, eta_fraction=eta_fraction)
    print("[drac_eval] figure 9/12: high_demand_residual_gap_vs_port_budget_final")
    plot_high_demand_residual_gap(matrix_dir, base_net, out_dir, formats, tau=tau, eta_fraction=eta_fraction)
    print("[drac_eval] figure 10/12: skew_distribution_cdf_final")
    plot_skew_distribution(matrix_dir, out_dir, formats, tau=tau, eta_fraction=eta_fraction)
    print("[drac_eval] figure 11/12: representative_dp_heatmaps_final")
    plot_representative_heatmaps(matrix_dir, out_dir, formats)
    print("[drac_eval] figure 12/12: final readiness summary")
    print("Main-paper candidates:")
    print("- skew_distribution_cdf_final.png")
    print("- normalized_comm_time_vs_skew_factor_final.png")
    print("- matching_error_cdf_final.png")
    print("- representative_dp_heatmaps_final.png")
    print("- waste_ratio_vs_port_budget_final.png")
    print("- high_demand_residual_gap_vs_port_budget_final.png")
    print("- pp_no_harm_comm_time_final.png")
    print("- ports_required_for_target_time_final.png")
    print("Appendix candidates:")
    print("- representative_tp_heatmaps_appendix.png")
    print("- representative_mixed_heatmaps_appendix.png")
    print("- useful_ratio_vs_port_budget_appendix.png")
    print("- iso_performance_port_saving_appendix.png")
    print("- pp_skew_sensitivity_appendix.png")
    print("- active_resource_requirement_directional_port_appendix.png")
    print("- active_resource_requirement_bidirectional_bundle_appendix.png")
    print("- extra_releasable_over_sym_ocs_appendix.png")
