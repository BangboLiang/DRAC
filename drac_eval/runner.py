from __future__ import annotations

import csv
import json
import itertools
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .allocation import AllocationResult, allocate_for_algorithm
from .config import ExperimentConfig, NetworkConfig
from .metrics import aggregate_port_exposure, compute_segment_metrics
from .plotting import generate_all_figures
from .traffic import SegmentDemand, load_or_generate_workload


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
                "mean_symmetric_waste_gbps": _mean([float(item["symmetric_waste_gbps"]) for item in rows]),
                "mean_wasted_idle_capacity_gbps": _mean([float(item["wasted_idle_capacity_gbps"]) for item in rows]),
                "requested_extra_bw_gbps": float(sum(float(item["requested_extra_bw_gbps"]) for item in rows)),
                "active_directional_ports": exposure["active_directional_ports"],
                "releasable_directional_ports": exposure["releasable_directional_ports"],
                "active_bidirectional_bundles": exposure["active_bidirectional_bundles"],
                "releasable_bidirectional_bundles": exposure["releasable_bidirectional_bundles"],
            }
        )
    return out


def run_experiments(cfg: ExperimentConfig) -> Dict[str, Path]:
    root = Path(cfg.output_dir)
    raw_dir = root / "raw"
    summary_dir = root / "summary"
    figure_dir = root / "figures"
    matrix_dir = raw_dir / "matrices"
    root.mkdir(parents=True, exist_ok=True)

    raw_rows: List[Dict[str, object]] = []

    for cluster_size, asymmetry, port_budget, total_links, reconfig_delay in _scenario_iter(cfg):
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
            segments = load_or_generate_workload(
                workload, int(cluster_size), float(asymmetry), int(cfg.seed)
            )
            aggregate_demand = np.sum([segment.matrix for segment in segments], axis=0)
            static_alloc = allocate_for_algorithm("static_sym", aggregate_demand, net)

            per_algorithm_allocs: Dict[str, List[AllocationResult]] = {alg: [] for alg in cfg.algorithms}

            for segment in segments:
                _write_matrix_json(
                    matrix_dir, run_id, "demand", segment.segment_idx, workload.name, segment.matrix, "demand"
                )
                for algorithm in cfg.algorithms:
                    allocation = allocate_for_algorithm(
                        algorithm,
                        segment.matrix,
                        net,
                        static_target=static_alloc.target_overlay,
                    )
                    per_algorithm_allocs[algorithm].append(allocation)
                    _write_matrix_json(
                        matrix_dir,
                        run_id,
                        algorithm,
                        segment.segment_idx,
                        workload.name,
                        allocation.total_bandwidth if algorithm != "ideal_asym" else allocation.target_overlay,
                        "bandwidth",
                    )
                    metrics = compute_segment_metrics(segment.matrix, allocation, net)
                    reconfig = (
                        float(net.reconfig_delay_ms)
                        if algorithm != "static_sym" and segment.segment_idx > 0
                        else 0.0
                    )
                    raw_rows.append(
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
                            "wasted_idle_capacity_gbps": metrics.wasted_idle_capacity_gbps,
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
                for row in raw_rows:
                    if (
                        row["run_id"] == run_id
                        and row["algorithm"] == algorithm
                        and row["workload"] == workload.name
                    ):
                        row["active_bidirectional_bundles"] = exposure["active_bidirectional_bundles"]
                        row["releasable_bidirectional_bundles"] = exposure["releasable_bidirectional_bundles"]

    summary_rows = _build_summary(raw_rows)
    _write_csv(raw_dir / "results_raw.csv", raw_rows)
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
        generate_all_figures(
            [{k: str(v) for k, v in row.items()} for row in summary_rows],
            [{k: str(v) for k, v in row.items()} for row in raw_rows],
            matrix_dir,
            figure_dir,
            cfg.figure_formats,
        )

    return {
        "root": root,
        "raw_csv": raw_dir / "results_raw.csv",
        "summary_csv": summary_dir / "results_summary.csv",
        "figure_dir": figure_dir,
        "matrix_dir": matrix_dir,
    }
