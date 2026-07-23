"""Independent, reproducible runners for the revised DRAC Evaluation."""

from __future__ import annotations

import math
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Sequence

import numpy as np

from .demand_profiler import (
    CalibrationBin,
    TransportCalibration,
    load_directional_measurements,
)
from .directional_target import (
    completion_time,
    solve_continuous_target,
    solve_continuous_target_numerical,
)
from .evaluation_pipeline import evaluate_main_schemes, plan_reconfigurable_schedule
from .evaluation_workloads import build_evaluation_workload
from .experiment_io import result_paths, write_csv, write_json, write_manifest
from .resource_compaction import compact_schedule
from .sparse_realization import (
    OCSResources,
    exhaustive_realization_oracle,
    realize_drac_sparse,
    realize_fill_all_residual,
    realize_floor_only,
    realize_nearest_rounding,
)
from .target_segmentation import (
    build_service_cost_matrix,
    candidate_segment_costs,
    exhaustive_segmentation_oracle,
    segment_target_sequence,
)


def _bytes_per_ms(gbps: float) -> float:
    return float(gbps) * 1e9 / 8.0 / 1000.0


def _fixed_matrix(size: int, base_gbps: float) -> np.ndarray:
    fixed = np.full((size, size), _bytes_per_ms(base_gbps), dtype=float)
    np.fill_diagonal(fixed, 0.0)
    return fixed


def _resources(size: int, ports: int, total_units: int | None = None) -> OCSResources:
    return OCSResources(
        np.full(size, ports, dtype=int),
        np.full(size, ports, dtype=int),
        total_units=size * ports if total_units is None else int(total_units),
    )


def _calibration(config: dict[str, Any]) -> TransportCalibration | None:
    bins = tuple(CalibrationBin(**row) for row in config.get("calibration_bins", []))
    return TransportCalibration(bins=bins, environment=str(config.get("calibration_environment", ""))) if bins else None


def run_profiler_accuracy(config: dict[str, Any], output_root: str | Path | None = None) -> dict[str, Path]:
    output = output_root or config["output_dir"]
    paths = result_paths(output, "profiler")
    seed = int(config.get("seed", 7))
    calibration = _calibration(config)
    prediction_rows: list[dict[str, Any]] = []
    for message_bytes in config.get("message_sizes_bytes", [1024, 1048576]):
        for kind in ("dp", "pp"):
            workload = build_evaluation_workload(
                kind,
                endpoint_count=int(config.get("endpoint_count", 4)),
                message_bytes=float(message_bytes),
                repeats=1,
                calibration=calibration,
            )
            for profiled in workload.profiled:
                for src, dst in zip(*np.where(profiled.matrix > 0)):
                    prediction_rows.append(
                        {
                            "workload": workload.name,
                            "node_id": profiled.node.node_id,
                            "operation": profiled.node.operation,
                            "message_bytes": message_bytes,
                            "src_endpoint": int(src),
                            "dst_endpoint": int(dst),
                            "payload_only_bytes": float(profiled.payload_matrix[src, dst]),
                            "payload_calibrated_bytes": float(profiled.matrix[src, dst]),
                            "provenance": profiled.provenance,
                        }
                    )
    prediction_csv = write_csv(paths["raw"] / "profiler_predictions.csv", prediction_rows)
    measurement_path = config.get("measurement_csv")
    accuracy_rows: list[dict[str, Any]] = []
    if measurement_path and Path(measurement_path).exists():
        measured = load_directional_measurements(measurement_path)
        index = {
            (
                row["operation"],
                int(float(row["message_bytes"])),
                int(row["src_endpoint"]),
                int(row["dst_endpoint"]),
            ): float(row["directional_bytes"])
            for row in measured
        }
        for row in prediction_rows:
            key = (row["operation"], int(row["message_bytes"]), row["src_endpoint"], row["dst_endpoint"])
            if key not in index:
                continue
            actual = index[key]
            for model, field in (("PayloadOnly", "payload_only_bytes"), ("Payload+Calibration", "payload_calibrated_bytes")):
                predicted = float(row[field])
                accuracy_rows.append(
                    {
                        **{key_name: row[key_name] for key_name in ("workload", "node_id", "operation", "message_bytes", "src_endpoint", "dst_endpoint")},
                        "model": model,
                        "predicted_bytes": predicted,
                        "measured_bytes": actual,
                        "absolute_error": abs(predicted - actual),
                        "relative_error": abs(predicted - actual) / actual if actual > 0 else math.nan,
                        "status": "complete",
                    }
                )
        status, evidence = "complete", "MEASURED_NIC_OR_PACKET_COUNTERS"
    else:
        accuracy_rows.append(
            {
                "workload": "",
                "node_id": "",
                "operation": "",
                "message_bytes": "",
                "src_endpoint": "",
                "dst_endpoint": "",
                "model": "",
                "predicted_bytes": "",
                "measured_bytes": "",
                "absolute_error": "",
                "relative_error": "",
                "status": "MEASUREMENT_PENDING",
            }
        )
        status, evidence = "measurement_pending", "NO_MEASURED_DIRECTIONAL_COUNTER_INPUT"
    accuracy_csv = write_csv(paths["processed"] / "profiler_accuracy.csv", accuracy_rows)
    manifest = write_manifest(paths["raw"], experiment="profiler", config=config, seed=seed, status=status, outputs={"predictions": str(prediction_csv), "accuracy": str(accuracy_csv)}, evidence=evidence)
    return {"predictions": prediction_csv, "processed": accuracy_csv, "manifest": manifest, "figures": paths["figures"]}


def run_end_to_end(config: dict[str, Any], output_root: str | Path | None = None) -> dict[str, Path]:
    output = output_root or config["output_dir"]
    paths = result_paths(output, "end_to_end")
    seed = int(config.get("seed", 7))
    size = int(config.get("endpoint_count", 4))
    unit = _bytes_per_ms(float(config.get("unit_bandwidth_gbps", 100.0)))
    fixed = _fixed_matrix(size, float(config.get("base_bandwidth_gbps", 25.0)))
    delta = float(config.get("delta_ms", 0.5))
    epsilon = float(config.get("epsilon", 0.1))
    rows: list[dict[str, Any]] = []
    for kind in config.get("workloads", ["dp", "pp", "mixed"]):
        workload = build_evaluation_workload(kind, endpoint_count=size, message_bytes=float(config.get("message_bytes", 64 * 1024 * 1024)), repeats=int(config.get("repeats", 2)), calibration=_calibration(config))
        write_json(paths["raw"] / f"{kind}_node_demands.json", {"provenance": workload.provenance, "node_ids": [node.node_id for node in workload.nodes], "matrices": [matrix.tolist() for matrix in workload.demands]})
        for ports in config.get("port_budgets", [2, 4, 6]):
            schedules = evaluate_main_schemes(workload.demands, _resources(size, int(ports)), unit, delta, epsilon, fixed)
            static_time = schedules[0].total_cost
            for schedule in schedules:
                constrained = sum(result.resource_constrained for result in schedule.realizations)
                rows.append(
                    {
                        "workload": workload.name,
                        "scheme": schedule.scheme,
                        "port_budget": int(ports),
                        "delta_ms": delta,
                        "epsilon": epsilon,
                        "communication_only_ms": schedule.communication_cost,
                        "reconfiguration_overhead_ms": schedule.reconfiguration_cost,
                        "total_communication_time_ms": schedule.total_cost,
                        "normalized_speedup_vs_static": static_time / schedule.total_cost if schedule.total_cost > 0 else math.nan,
                        "segment_count": len(schedule.realizations),
                        "resource_constrained_segment_ratio": constrained / len(schedule.realizations),
                        "stable_reserved_tx": int(schedule.compaction.reserved_tx.sum()),
                        "stable_reserved_rx": int(schedule.compaction.reserved_rx.sum()),
                        "stable_exposed_tx": int(schedule.compaction.exposed_tx.sum()),
                        "stable_exposed_rx": int(schedule.compaction.exposed_rx.sum()),
                        "evidence": workload.provenance,
                    }
                )
    raw_csv = write_csv(paths["raw"] / "end_to_end_raw.csv", rows)
    processed_csv = write_csv(paths["processed"] / "end_to_end_performance.csv", rows)
    manifest = write_manifest(paths["raw"], experiment="end_to_end", config=config, seed=seed, status="complete", outputs={"raw": str(raw_csv), "processed": str(processed_csv)}, evidence="DETERMINISTIC_SIMULATOR")
    return {"raw": raw_csv, "processed": processed_csv, "manifest": manifest, "figures": paths["figures"]}


def run_segmentation(config: dict[str, Any], output_root: str | Path | None = None) -> dict[str, Path]:
    output = output_root or config["output_dir"]
    paths = result_paths(output, "segmentation")
    seed = int(config.get("seed", 7))
    size = int(config.get("endpoint_count", 4))
    unit = _bytes_per_ms(float(config.get("unit_bandwidth_gbps", 100.0)))
    fixed = _fixed_matrix(size, float(config.get("base_bandwidth_gbps", 25.0)))
    workload = build_evaluation_workload(config.get("workload", "mixed"), endpoint_count=size, message_bytes=float(config.get("message_bytes", 64 * 1024 * 1024)), repeats=int(config.get("repeats", 1)))
    resources = _resources(size, int(config.get("port_budget", 4)))
    targets = [solve_continuous_target(demand, resources.n_tx, resources.n_rx, unit, fixed) for demand in workload.demands]
    service = build_service_cost_matrix(workload.demands, targets, unit, fixed)
    candidate_cost, candidate_rep = candidate_segment_costs(service)
    rows: list[dict[str, Any]] = []
    for delta in config.get("delta_values_ms", [0.0, 0.1, 1.0, 10.0]):
        delta = float(delta)
        result = segment_target_sequence(workload.demands, targets, unit, delta, fixed)
        oracle_cost, oracle_segments = exhaustive_segmentation_oracle(service, delta) if len(workload.demands) <= 14 else (math.nan, ())
        one = float(candidate_cost[0, len(workload.demands) - 1])
        per_node_comm = float(np.trace(service))
        per_node = per_node_comm + delta * (len(workload.demands) - 1)
        for scheme, total, comm, segments, representatives in (
            ("OneConfig", one, one, 1, [int(candidate_rep[0, len(workload.demands) - 1])]),
            ("PerNode-Reconfig", per_node, per_node_comm, len(workload.demands), list(range(len(workload.demands)))),
            ("DRAC-DP", result.total_cost, result.communication_cost, len(result.segments), [segment.representative for segment in result.segments]),
            ("SegmentOracle", oracle_cost, oracle_cost - delta * (len(oracle_segments) - 1) if np.isfinite(oracle_cost) else math.nan, len(oracle_segments), [segment.representative for segment in oracle_segments]),
        ):
            rows.append({"workload": workload.name, "scheme": scheme, "delta_ms": delta, "total_cost_ms": total, "communication_cost_ms": comm, "reconfiguration_cost_ms": total - comm, "segment_count": segments, "oracle_gap": (total - oracle_cost) / oracle_cost if np.isfinite(oracle_cost) and oracle_cost > 0 else math.nan, "representative_indices": ";".join(map(str, representatives)), "evidence": workload.provenance})
    raw_csv = write_csv(paths["raw"] / "segmentation_raw.csv", rows)
    processed_csv = write_csv(paths["processed"] / "segmentation.csv", rows)
    manifest = write_manifest(paths["raw"], experiment="segmentation", config=config, seed=seed, status="complete", outputs={"raw": str(raw_csv), "processed": str(processed_csv)}, evidence="DETERMINISTIC_SIMULATOR")
    return {"raw": raw_csv, "processed": processed_csv, "manifest": manifest, "figures": paths["figures"]}


def run_realization(config: dict[str, Any], output_root: str | Path | None = None) -> dict[str, Path]:
    output = output_root or config["output_dir"]
    paths = result_paths(output, "realization")
    seed = int(config.get("seed", 7))
    size = int(config.get("endpoint_count", 3))
    unit = _bytes_per_ms(float(config.get("unit_bandwidth_gbps", 100.0)))
    fixed = _fixed_matrix(size, float(config.get("base_bandwidth_gbps", 25.0)))
    workload = build_evaluation_workload(config.get("workload", "dp"), endpoint_count=size, message_bytes=float(config.get("message_bytes", 32 * 1024 * 1024)), repeats=1)
    resources = _resources(size, int(config.get("port_budget", 4)))
    targets = [solve_continuous_target(demand, resources.n_tx, resources.n_rx, unit, fixed) for demand in workload.demands]
    segmentation = segment_target_sequence(workload.demands, targets, unit, float(config.get("delta_ms", 0.5)), fixed)
    policies: list[tuple[str, Callable[..., Any]]] = [
        ("FloorOnly", realize_floor_only),
        ("NearestRounding", realize_nearest_rounding),
        ("FillAllResidual", realize_fill_all_residual),
        ("DRACSparse", realize_drac_sparse),
    ]
    rows: list[dict[str, Any]] = []
    for epsilon_value in config.get("epsilon_values", [0.0, 0.1, 0.25, 0.5]):
        epsilon_value = float(epsilon_value)
        per_policy: dict[str, list[Any]] = {name: [] for name, _ in policies}
        for segment in segmentation.segments:
            segment_demands = workload.demands[segment.start : segment.end + 1]
            target = targets[segment.representative].allocation
            for name, policy in policies:
                per_policy[name].append(policy(target, segment_demands, segment.communication_cost, epsilon_value, resources, unit, fixed))
            if size <= 3:
                try:
                    oracle = exhaustive_realization_oracle(target, segment_demands, segment.communication_cost, epsilon_value, resources, unit, fixed)
                    per_policy.setdefault("ILPOracle", []).append(oracle)
                except ValueError:
                    pass
        for name, results in per_policy.items():
            if not results:
                continue
            compaction = compact_schedule([result.units for result in results], resources.n_tx, resources.n_rx, results)
            cost = sum(result.cost for result in results)
            logical = sum(result.logical_cost for result in results)
            oracle_units = sum(result.used_units for result in per_policy.get("ILPOracle", []))
            used = sum(result.used_units for result in results)
            rows.append({"workload": workload.name, "policy": name, "epsilon": epsilon_value, "realized_slowdown": cost / logical if logical > 0 else math.nan, "used_connection_units": used, "stable_reserved_channels": compaction.total_stable_directional_pool, "tolerance_satisfaction_rate": sum(result.tolerance_satisfied for result in results) / len(results), "resource_constrained_segment_ratio": sum(result.resource_constrained for result in results) / len(results), "oracle_unit_gap": used - oracle_units if oracle_units and name != "ILPOracle" else 0, "evidence": workload.provenance})
    raw_csv = write_csv(paths["raw"] / "realization_raw.csv", rows)
    processed_csv = write_csv(paths["processed"] / "realization_tradeoff.csv", rows)
    manifest = write_manifest(paths["raw"], experiment="realization", config=config, seed=seed, status="complete", outputs={"raw": str(raw_csv), "processed": str(processed_csv)}, evidence="DETERMINISTIC_SIMULATOR")
    return {"raw": raw_csv, "processed": processed_csv, "manifest": manifest, "figures": paths["figures"]}


def run_compaction(config: dict[str, Any], output_root: str | Path | None = None) -> dict[str, Path]:
    output = output_root or config["output_dir"]
    paths = result_paths(output, "compaction")
    seed = int(config.get("seed", 7))
    size = int(config.get("endpoint_count", 4))
    ports = int(config.get("port_budget", 6))
    unit = _bytes_per_ms(float(config.get("unit_bandwidth_gbps", 100.0)))
    fixed = _fixed_matrix(size, float(config.get("base_bandwidth_gbps", 25.0)))
    rows: list[dict[str, Any]] = []
    iso_rows: list[dict[str, Any]] = []
    for kind in config.get("workloads", ["dp", "pp", "mixed"]):
        workload = build_evaluation_workload(kind, endpoint_count=size, message_bytes=float(config.get("message_bytes", 64 * 1024 * 1024)), repeats=int(config.get("repeats", 2)))
        resources = _resources(size, ports)
        schedules = evaluate_main_schemes(workload.demands, resources, unit, float(config.get("delta_ms", 0.5)), float(config.get("epsilon", 0.1)), fixed)
        for schedule in schedules[1:]:
            comp = schedule.compaction
            rows.append({"workload": workload.name, "scheme": f"{schedule.scheme} schedule-wide peak", "reserved_tx": int(comp.reserved_tx.sum()), "reserved_rx": int(comp.reserved_rx.sum()), "exposed_tx": int(comp.exposed_tx.sum()), "exposed_rx": int(comp.exposed_rx.sum()), "total_stable_pool": comp.total_stable_directional_pool, "compaction_ratio": 1.0 - comp.total_stable_directional_pool / (2 * size * ports), "reserved_bundles": int(comp.reserved_bundles.sum()), "evidence": workload.provenance})
            no_compaction = sum(int(result.units.sum()) * 2 for result in schedule.realizations)
            rows.append({"workload": workload.name, "scheme": f"{schedule.scheme} without compaction", "reserved_tx": no_compaction // 2, "reserved_rx": no_compaction // 2, "exposed_tx": 0, "exposed_rx": 0, "total_stable_pool": no_compaction, "compaction_ratio": 0.0, "reserved_bundles": no_compaction // 2, "evidence": workload.provenance})
        rows.append({"workload": workload.name, "scheme": "FullReservation", "reserved_tx": size * ports, "reserved_rx": size * ports, "exposed_tx": 0, "exposed_rx": 0, "total_stable_pool": 2 * size * ports, "compaction_ratio": 0.0, "reserved_bundles": size * ports, "evidence": workload.provenance})
        reference_ports = int(config.get("iso_reference_port_budget", min(ports, 4)))
        reference_schedule = plan_reconfigurable_schedule(
            workload.demands,
            _resources(size, reference_ports),
            unit,
            float(config.get("delta_ms", 0.5)),
            float(config.get("epsilon", 0.1)),
            fixed_bandwidth=fixed,
            symmetric=True,
        )
        target_time = reference_schedule.total_cost
        reached = False
        for candidate_ports in range(1, ports + 1):
            trial = plan_reconfigurable_schedule(workload.demands, _resources(size, candidate_ports), unit, float(config.get("delta_ms", 0.5)), float(config.get("epsilon", 0.1)), fixed_bandwidth=fixed, symmetric=False)
            if trial.total_cost <= target_time * (1 + 1e-9):
                iso_rows.append({"workload": workload.name, "scheme": "DRAC", "reference": f"Sym-OCS-{reference_ports}-ports", "reference_time_ms": target_time, "minimum_port_budget": candidate_ports, "minimum_stable_directional_pool": trial.compaction.total_stable_directional_pool, "minimum_stable_bundle_pool": int(trial.compaction.reserved_bundles.sum()), "status": "reached", "evidence": workload.provenance})
                reached = True
                break
        if not reached:
            iso_rows.append({"workload": workload.name, "scheme": "DRAC", "reference": f"Sym-OCS-{reference_ports}-ports", "reference_time_ms": target_time, "minimum_port_budget": "", "minimum_stable_directional_pool": "", "minimum_stable_bundle_pool": "", "status": "not_reached", "evidence": workload.provenance})
    raw_csv = write_csv(paths["raw"] / "compaction_raw.csv", rows)
    processed_csv = write_csv(paths["processed"] / "schedule_compaction.csv", rows)
    iso_csv = write_csv(paths["processed"] / "iso_performance_pool.csv", iso_rows)
    manifest = write_manifest(paths["raw"], experiment="compaction", config=config, seed=seed, status="complete", outputs={"raw": str(raw_csv), "processed": str(processed_csv), "iso": str(iso_csv)}, evidence="DETERMINISTIC_SIMULATOR")
    return {"raw": raw_csv, "processed": processed_csv, "iso": iso_csv, "manifest": manifest, "figures": paths["figures"]}


def run_planning_overhead(config: dict[str, Any], output_root: str | Path | None = None) -> dict[str, Path]:
    output = output_root or config["output_dir"]
    paths = result_paths(output, "overhead")
    seed = int(config.get("seed", 7))
    rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for size in config.get("endpoint_counts", [3, 4, 6]):
        for requested_k in config.get("node_counts", [4, 8, 16]):
            unit = _bytes_per_ms(float(config.get("unit_bandwidth_gbps", 100.0)))
            fixed = _fixed_matrix(int(size), float(config.get("base_bandwidth_gbps", 25.0)))
            start = perf_counter()
            workload = build_evaluation_workload("mixed", endpoint_count=int(size), message_bytes=float(config.get("message_bytes", 16 * 1024 * 1024)), repeats=max(1, int(requested_k) // (2 * int(size))))
            demands = list(workload.demands[: int(requested_k)])
            profile_ms = (perf_counter() - start) * 1000.0
            resources = _resources(int(size), int(config.get("port_budget", 4)))
            start = perf_counter()
            targets = [solve_continuous_target(demand, resources.n_tx, resources.n_rx, unit, fixed) for demand in demands]
            target_ms = (perf_counter() - start) * 1000.0
            start = perf_counter()
            service = build_service_cost_matrix(demands, targets, unit, fixed)
            service_ms = (perf_counter() - start) * 1000.0
            start = perf_counter()
            candidate_segment_costs(service)
            candidate_ms = (perf_counter() - start) * 1000.0
            start = perf_counter()
            segmentation = segment_target_sequence(demands, targets, unit, float(config.get("delta_ms", 0.5)), fixed)
            dp_ms = (perf_counter() - start) * 1000.0
            start = perf_counter()
            realizations = [realize_drac_sparse(targets[s.representative].allocation, demands[s.start : s.end + 1], s.communication_cost, float(config.get("epsilon", 0.1)), resources, unit, fixed) for s in segmentation.segments]
            realization_ms = (perf_counter() - start) * 1000.0
            start = perf_counter()
            compact_schedule([result.units for result in realizations], resources.n_tx, resources.n_rx, realizations)
            compaction_ms = (perf_counter() - start) * 1000.0
            rows.append({"node_count": len(demands), "endpoint_count": int(size), "segment_count": len(segmentation.segments), "port_budget": int(config.get("port_budget", 4)), "graph_parsing_ms": 0.0, "ordered_demand_profiling_ms": profile_ms, "target_generation_ms": target_ms, "service_matrix_ms": service_ms, "candidate_segment_cost_ms": candidate_ms, "dynamic_programming_ms": dp_ms, "sparse_realization_ms": realization_ms, "schedule_compaction_ms": compaction_ms, "total_planning_ms": profile_ms + target_ms + service_ms + candidate_ms + dp_ms + realization_ms + compaction_ms, "evidence": workload.provenance})
            if demands:
                closed_start = perf_counter()
                closed = solve_continuous_target(demands[0], resources.n_tx, resources.n_rx, unit, fixed)
                closed_runtime = (perf_counter() - closed_start) * 1000.0
                numerical_start = perf_counter()
                numerical = solve_continuous_target_numerical(demands[0], resources.n_tx, resources.n_rx, unit, fixed)
                numerical_runtime = (perf_counter() - numerical_start) * 1000.0
                validation_rows.append({"node_count": len(demands), "endpoint_count": int(size), "objective_gap": (closed.theta - numerical.theta) / numerical.theta if numerical.theta else 0.0, "feasible": bool(np.all(closed.tx_usage <= resources.n_tx + 1e-9) and np.all(closed.rx_usage <= resources.n_rx + 1e-9)), "resource_usage": float(closed.allocation.sum()), "closed_form_runtime_ms": closed_runtime, "numerical_runtime_ms": numerical_runtime, "evidence": workload.provenance})
    raw_csv = write_csv(paths["raw"] / "planning_runtime_raw.csv", rows)
    processed_csv = write_csv(paths["processed"] / "planning_runtime.csv", rows)
    validation_csv = write_csv(paths["processed"] / "target_solver_validation.csv", validation_rows)
    table_path = paths["tables"] / "target_solver_validation.tex"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    mean_gap = float(np.mean([abs(float(row["objective_gap"])) for row in validation_rows])) if validation_rows else math.nan
    all_feasible = all(bool(row["feasible"]) for row in validation_rows)
    mean_closed = float(np.mean([float(row["closed_form_runtime_ms"]) for row in validation_rows])) if validation_rows else math.nan
    mean_numerical = float(np.mean([float(row["numerical_runtime_ms"]) for row in validation_rows])) if validation_rows else math.nan
    mean_planning = float(np.mean([float(row["total_planning_ms"]) for row in rows])) if rows else math.nan
    table_path.write_text(
        "\\begin{tabular}{lrrr}\n"
        "\\toprule\n"
        "Method & Objective gap & Feasible & Mean runtime (ms) \\\\\n"
        "\\midrule\n"
        f"Closed-form target & {mean_gap:.3e} & {'yes' if all_feasible else 'no'} & {mean_closed:.3f} \\\\\n"
        f"Numerical reference & 0 & yes & {mean_numerical:.3f} \\\\\n"
        f"Full offline planning & -- & yes & {mean_planning:.3f} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n",
        encoding="utf-8",
    )
    manifest = write_manifest(paths["raw"], experiment="overhead", config=config, seed=seed, status="complete", outputs={"raw": str(raw_csv), "processed": str(processed_csv), "validation": str(validation_csv), "table": str(table_path)}, evidence="DETERMINISTIC_SIMULATOR")
    return {"raw": raw_csv, "processed": processed_csv, "validation": validation_csv, "table": table_path, "manifest": manifest, "figures": paths["figures"]}
