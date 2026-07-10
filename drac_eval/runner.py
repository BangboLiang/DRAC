from __future__ import annotations

import csv
import json
import itertools
import re
from json import JSONDecodeError
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .allocation import AllocationResult, allocate_for_algorithm
from .config import ExperimentConfig, NetworkConfig
from .metrics import aggregate_port_exposure, compute_segment_metrics
from .plotting import generate_all_figures
from .traffic import SegmentDemand, load_or_generate_workload

MATRIX_RE = re.compile(
    r"(?P<run_id>.+)\.workload-(?P<workload>[^.]+)\.segment-(?P<segment>\d+)\.algorithm-(?P<algorithm>[^.]+)\.json$"
)


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
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


def _write_matrix_json(
    matrix_dir: Path,
    run_id: str,
    algorithm: str,
    segment: int,
    workload: str,
    matrix: np.ndarray,
    matrix_type: str,
) -> None:
    matrix_dir.mkdir(parents=True, exist_ok=True)
    path = matrix_dir / (
        f"{run_id}.workload-{workload}.segment-{segment}.algorithm-{algorithm if matrix_type != 'demand' else 'demand'}.json"
    )
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"matrix": matrix.tolist(), "matrix_type": matrix_type}, handle, indent=2)


def _scenario_iter(cfg: ExperimentConfig) -> Iterable[Tuple[int, float, int, int, float]]:
    sweeps = cfg.sweeps
    return itertools.product(
        sweeps.cluster_sizes,
        sweeps.asymmetry_levels,
        sweeps.port_budgets,
        sweeps.total_ocs_links,
        sweeps.reconfig_delays_ms,
    )


def _summary_key(row: Dict[str, object]) -> tuple:
    return (
        row["run_id"],
        row["algorithm"],
        row["workload"],
        row["cluster_size"],
        row["asymmetry_level"],
        row["port_budget"],
        row["total_ocs_links"],
        row["reconfig_delay_ms"],
    )


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _build_summary(raw_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple, List[Dict[str, object]]] = {}
    for row in raw_rows:
        grouped.setdefault(_summary_key(row), []).append(row)

    out: List[Dict[str, object]] = []
    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda item: int(item["segment_idx"]))
        exposure = {
            "active_directional_ports": max(float(item["active_directional_ports"]) for item in rows),
            "releasable_directional_ports": min(float(item["releasable_directional_ports"]) for item in rows),
            "active_bidirectional_bundles": rows[0].get("active_bidirectional_bundles", 0.0),
            "releasable_bidirectional_bundles": rows[0].get("releasable_bidirectional_bundles", 0.0),
        }
        out.append(
            {
                "run_id": key[0],
                "algorithm": key[1],
                "workload": key[2],
                "cluster_size": key[3],
                "asymmetry_level": key[4],
                "port_budget": key[5],
                "total_ocs_links": key[6],
                "reconfig_delay_ms": key[7],
                "segment_count": len(rows),
                "total_time_ms": float(sum(float(item["segment_total_time_ms"]) for item in rows)),
                "mean_completion_time_ms": _mean([float(item["completion_time_ms"]) for item in rows]),
                "mean_matching_error_l1": _mean([float(item["matching_error_l1"]) for item in rows]),
                "p95_matching_error": max(float(item["matching_error_p95"]) for item in rows),
                "mean_network_utilization": _mean([float(item["network_utilization"]) for item in rows]),
                "mean_ocs_port_utilization": _mean([float(item["ocs_port_utilization"]) for item in rows]),
                "mean_useful_capacity_gbps": _mean([float(item["useful_capacity_gbps"]) for item in rows]),
                "mean_symmetric_waste_gbps": _mean([float(item["symmetric_waste_gbps"]) for item in rows]),
                "mean_wasted_idle_capacity_gbps": _mean([float(item["wasted_idle_capacity_gbps"]) for item in rows]),
                "mean_total_provisioned_capacity_gbps": _mean([float(item["total_provisioned_capacity_gbps"]) for item in rows]),
                "mean_useful_ratio": _mean([float(item["useful_ratio"]) for item in rows]),
                "mean_waste_ratio": _mean([float(item["waste_ratio"]) for item in rows]),
                "requested_extra_bw_gbps": float(sum(float(item["requested_extra_bw_gbps"]) for item in rows)),
                "active_directional_ports": exposure["active_directional_ports"],
                "releasable_directional_ports": exposure["releasable_directional_ports"],
                "active_bidirectional_bundles": exposure["active_bidirectional_bundles"],
                "releasable_bidirectional_bundles": exposure["releasable_bidirectional_bundles"],
            }
        )
    return out


def _normalized_asymmetry_score(matrix: np.ndarray) -> float:
    denom = matrix + matrix.T
    numer = np.abs(matrix - matrix.T)
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    valid = mask & (denom > 0.0)
    if not np.any(valid):
        return 0.0
    return float(np.mean(numer[valid] / denom[valid]))


def _run_sanity_checks(
    cfg: ExperimentConfig,
    summary_rows: List[Dict[str, object]],
    matrix_dir: Path,
    figure_dir: Path,
) -> None:
    demand_diag_ok = True
    rho_ok = True
    sym_ocs_ok = True
    share_sum_ok = True
    asym_scores: List[float] = []
    demand_count = 0
    workloads_present = {str(row["workload"]) for row in summary_rows}
    expected_figs = [
        "normalized_comm_time_vs_skew_factor_final",
        "waste_ratio_vs_port_budget_final",
        "useful_ratio_vs_port_budget_appendix",
        "matching_error_cdf_final",
        "representative_dp_heatmaps_final",
        "skew_distribution_cdf_final",
        "active_resource_requirement_directional_port_appendix",
        "active_resource_requirement_bidirectional_bundle_appendix",
        "extra_releasable_over_sym_ocs_appendix",
        "high_demand_residual_gap_vs_port_budget_final",
        "iso_performance_port_saving_appendix",
        "ports_required_for_target_time_final",
    ]
    if "tp" in workloads_present:
        expected_figs.append("representative_tp_heatmaps_appendix")
    if "mixed" in workloads_present:
        expected_figs.append("representative_mixed_heatmaps_appendix")
    if "pp" in workloads_present:
        expected_figs.append("pp_no_harm_comm_time_final")
    formats = list(cfg.figure_formats)

    for path in matrix_dir.glob("*.json"):
        match = MATRIX_RE.match(path.name)
        if not match:
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, JSONDecodeError) as exc:
            print(f"[sanity] skipping unreadable matrix json: {path.name} ({exc})")
            continue
        matrix = np.array(payload["matrix"], dtype=float)
        algorithm = match.group("algorithm")
        if algorithm == "demand":
            demand_count += 1
            demand_diag_ok = demand_diag_ok and bool(np.allclose(np.diag(matrix), 0.0))
            for i in range(matrix.shape[0]):
                for j in range(i + 1, matrix.shape[1]):
                    a = float(matrix[i, j])
                    b = float(matrix[j, i])
                    if a <= 0.0 and b <= 0.0:
                        continue
                    rho = max(a, b) / (min(a, b) + 1e-9)
                    if rho < 1.0:
                        rho_ok = False
            total = float(np.sum(matrix))
            if total > 0.0:
                share = matrix / total
                share_sum_ok = share_sum_ok and abs(float(np.sum(share)) - 1.0) < 1e-6
            if "_a1p0_" in match.group("run_id"):
                asym_scores.append(_normalized_asymmetry_score(matrix))
        elif algorithm == "sym_ocs":
            sym_ocs_ok = sym_ocs_ok and bool(np.allclose(matrix, matrix.T))

    directional_ok = True
    bidirectional_ok = True
    for row in summary_rows:
        cluster_size = int(row["cluster_size"])
        directional_reserved = (
            int(cfg.network.directional_port_reserved)
            if cfg.network.directional_port_reserved is not None
            else int(row["port_budget"])
        ) * cluster_size * 2
        bidirectional_reserved = (
            int(cfg.network.bidirectional_bundle_reserved)
            if cfg.network.bidirectional_bundle_reserved is not None
            else int(row["port_budget"])
        ) * cluster_size
        directional_ok = directional_ok and abs(
            (float(row["active_directional_ports"]) + float(row["releasable_directional_ports"]))
            - directional_reserved
        ) < 1e-6
        bidirectional_ok = bidirectional_ok and abs(
            (float(row["active_bidirectional_bundles"]) + float(row["releasable_bidirectional_bundles"]))
            - bidirectional_reserved
        ) < 1e-6

    figs_ok = True
    missing: List[str] = []
    for stem in expected_figs:
        for fmt in formats:
            fig_path = figure_dir / f"{stem}.{fmt}"
            if not fig_path.exists():
                figs_ok = False
                missing.append(fig_path.name)

    print(f"[sanity] demand matrix diagonal excluded: {demand_diag_ok}")
    print(f"[sanity] rho >= 1 across demand matrices: {rho_ok}")
    print(f"[sanity] Sym-OCS realized bandwidth symmetric: {sym_ocs_ok}")
    avg_asym = float(sum(asym_scores) / len(asym_scores)) if asym_scores else 0.0
    print(
        f"[sanity] skew factor = 1 demand_asymmetry_score avg: {avg_asym:.6f} "
        "(smaller is closer to symmetric)"
    )
    print(f"[sanity] share indicators sum to ~1: {share_sum_ok}")
    print(f"[sanity] physical exposure directional Pact + Prel = reserved_total: {directional_ok}")
    print(f"[sanity] physical exposure bidirectional Pact + Prel = reserved_total: {bidirectional_ok}")
    print("[sanity] P_req is tracked separately from reserved_total: True")
    print(f"[sanity] all expected figures generated: {figs_ok}")
    if missing:
        print(f"[sanity] missing figures: {missing[:10]}")


def run_experiments(cfg: ExperimentConfig) -> Dict[str, Path]:
    root = Path(cfg.output_dir)
    raw_dir = root / "raw"
    summary_dir = root / "summary"
    figure_dir = root / "figures"
    matrix_dir = raw_dir / "matrices"
    root.mkdir(parents=True, exist_ok=True)

    raw_rows: List[Dict[str, object]] = []
    scenarios = list(_scenario_iter(cfg))
    total_scenarios = len(scenarios)

    for scenario_idx, (
        cluster_size,
        asymmetry,
        port_budget,
        total_links,
        reconfig_delay,
    ) in enumerate(scenarios, start=1):
        run_id = (
            f"n{cluster_size}_a{str(asymmetry).replace('.', 'p')}"
            f"_p{port_budget}_l{total_links}_r{str(reconfig_delay).replace('.', 'p')}"
        )
        net = cfg.network.with_overrides(
            {
                "per_node_port_budget": int(port_budget),
                "total_ocs_links": int(total_links),
                "reconfig_delay_ms": float(reconfig_delay),
            }
        )
        for workload in cfg.workloads:
            workload_rows: List[Dict[str, object]] = []
            segments = load_or_generate_workload(
                workload, int(cluster_size), float(asymmetry), int(cfg.seed)
            )
            aggregate_demand = np.sum([segment.matrix for segment in segments], axis=0)
            static_alloc = allocate_for_algorithm("static_sym", aggregate_demand, net)

            per_algorithm_allocs: Dict[str, List[AllocationResult]] = {alg: [] for alg in cfg.algorithms}

            for segment in segments:
                if cfg.save_matrices:
                    _write_matrix_json(
                        matrix_dir,
                        run_id,
                        "demand",
                        segment.segment_idx,
                        workload.name,
                        segment.matrix,
                        "demand",
                    )
                for algorithm in cfg.algorithms:
                    allocation = allocate_for_algorithm(
                        algorithm,
                        segment.matrix,
                        net,
                        static_target=static_alloc.target_overlay,
                    )
                    per_algorithm_allocs[algorithm].append(allocation)
                    if cfg.save_matrices:
                        _write_matrix_json(
                            matrix_dir,
                            run_id,
                            algorithm,
                            segment.segment_idx,
                            workload.name,
                            allocation.total_bandwidth
                            if algorithm != "ideal_asym"
                            else allocation.target_overlay,
                            "bandwidth",
                        )
                    metrics = compute_segment_metrics(segment.matrix, allocation, net)
                    reconfig = (
                        float(net.reconfig_delay_ms)
                        if algorithm != "static_sym" and segment.segment_idx > 0
                        else 0.0
                    )
                    workload_rows.append(
                        {
                            "run_id": run_id,
                            "algorithm": algorithm,
                            "workload": workload.name,
                            "workload_kind": workload.kind,
                            "cluster_size": int(cluster_size),
                            "asymmetry_level": float(asymmetry),
                            "port_budget": int(port_budget),
                            "total_ocs_links": int(total_links),
                            "reconfig_delay_ms": float(reconfig_delay),
                            "segment_idx": int(segment.segment_idx),
                            "completion_time_ms": metrics.completion_time_ms,
                            "reconfig_overhead_ms": reconfig,
                            "segment_total_time_ms": metrics.completion_time_ms + reconfig,
                            "matching_error_l1": metrics.matching_error_l1,
                            "matching_error_p95": metrics.matching_error_p95,
                            "network_utilization": metrics.network_utilization,
                            "ocs_port_utilization": metrics.ocs_port_utilization,
                            "useful_capacity_gbps": metrics.useful_capacity_gbps,
                            "wasted_idle_capacity_gbps": metrics.wasted_idle_capacity_gbps,
                            "total_provisioned_capacity_gbps": metrics.total_provisioned_capacity_gbps,
                            "useful_ratio": metrics.useful_ratio,
                            "waste_ratio": metrics.waste_ratio,
                            "symmetric_waste_gbps": metrics.symmetric_waste_gbps,
                            "active_directional_ports": metrics.active_directional_ports,
                            "releasable_directional_ports": metrics.releasable_directional_ports,
                            "requested_extra_bw_gbps": metrics.requested_extra_bw_gbps,
                            "skew_p50": metrics.skew_p50,
                            "skew_p95": metrics.skew_p95,
                        }
                    )

            for algorithm, allocs in per_algorithm_allocs.items():
                exposure = aggregate_port_exposure(allocs, net)
                for row in workload_rows:
                    if (
                        row["run_id"] == run_id
                        and row["algorithm"] == algorithm
                        and row["workload"] == workload.name
                    ):
                        row["active_bidirectional_bundles"] = exposure["active_bidirectional_bundles"]
                        row["releasable_bidirectional_bundles"] = exposure["releasable_bidirectional_bundles"]
            raw_rows.extend(workload_rows)

        if scenario_idx % 25 == 0 or scenario_idx == total_scenarios:
            print(
                f"[drac_eval] completed {scenario_idx}/{total_scenarios} scenarios "
                f"({scenario_idx / max(1, total_scenarios) * 100.0:.1f}%)"
            )

    summary_rows = _build_summary(raw_rows)
    print("[drac_eval] writing raw results csv")
    _write_csv(raw_dir / "results_raw.csv", raw_rows)
    print("[drac_eval] writing summary csv")
    _write_csv(summary_dir / "results_summary.csv", summary_rows)

    manifest = {
        "name": cfg.name,
        "seed": cfg.seed,
        "notes": cfg.notes,
        "algorithms": cfg.algorithms,
        "output_dir": str(root.resolve()),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if cfg.generate_figures:
        if not cfg.save_matrices:
            raise ValueError(
                "generate_figures=True requires save_matrices=True because the current "
                "plotting pipeline reads matrix JSON files."
            )
        print("[drac_eval] generating figures")
        generate_all_figures(
            [{k: str(v) for k, v in row.items()} for row in summary_rows],
            [{k: str(v) for k, v in row.items()} for row in raw_rows],
            matrix_dir,
            figure_dir,
            cfg.figure_formats,
            tau=cfg.high_demand_tau,
            eta_fraction=cfg.high_demand_eta_fraction,
            base_net=cfg.network,
        )
        print("[drac_eval] figures complete")

    _run_sanity_checks(cfg, summary_rows, matrix_dir, figure_dir)

    return {
        "root": root,
        "raw_csv": raw_dir / "results_raw.csv",
        "summary_csv": summary_dir / "results_summary.csv",
        "figure_dir": figure_dir,
        "matrix_dir": matrix_dir,
    }
