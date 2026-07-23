"""Core v2 experiments for segment optimization and sparse realization."""

from __future__ import annotations

import math
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Sequence

import numpy as np

from .evaluation_experiments import _bytes_per_ms, _fixed_matrix, _resources
from .evaluation_pipeline import evaluate_main_schemes, plan_reconfigurable_schedule
from .evaluation_pipeline_v2 import PlannedScheduleV2, _realize_schedule, plan_schedule_candidates_v2
from .evaluation_workloads import EvaluationWorkload, build_evaluation_workload
from .experiment_io import result_paths, write_csv, write_json, write_manifest
from .resource_compaction import compact_schedule
from .segment_target import SegmentTarget, solve_segment_continuous_target
from .sparse_realization import (
    OCSResources,
    RealizationResult,
    exhaustive_realization_oracle,
    realize_drac_sparse_coverage_seed,
    realize_drac_sparse_floor_seed,
    realize_drac_sparse_multi_seed,
    realize_fill_all_residual,
    realize_floor_only,
    realize_nearest_rounding,
)
from .target_segmentation import (
    CandidateTargetTable,
    build_candidate_target_table,
    exhaustive_partition_oracle_v2,
    segment_continuous_sequence,
)


EVIDENCE = "DETERMINISTIC_SIMULATOR_INPUT"


def _workload(config: dict[str, Any], kind: str, *, repeats: int | None = None) -> EvaluationWorkload:
    return build_evaluation_workload(
        kind,
        endpoint_count=int(config.get("endpoint_count", 4)),
        message_bytes=float(config.get("message_bytes", 64 * 1024 * 1024)),
        repeats=int(config.get("repeats", 1) if repeats is None else repeats),
    )


def _schedule_row(
    workload: str,
    scheme: str,
    ports: int,
    delta: float,
    epsilon: float,
    schedule: Any,
    static_time: float,
) -> dict[str, Any]:
    realizations = schedule.realizations
    selected_types = tuple(getattr(schedule, "selected_target_types", ()))
    symmetric_segments = sum(value == "symmetric" for value in selected_types)
    fallback_usage = symmetric_segments / len(selected_types) if selected_types else 0.0
    return {
        "workload": workload,
        "scheme": scheme,
        "port_budget": ports,
        "delta_ms": delta,
        "epsilon": epsilon,
        "communication_only_ms": schedule.communication_cost,
        "reconfiguration_cost_ms": schedule.reconfiguration_cost,
        "total_cost_ms": schedule.total_cost,
        "normalized_speedup_vs_static": static_time / schedule.total_cost if schedule.total_cost > 0 else math.nan,
        "segment_count": len(realizations),
        "fallback_usage_fraction": fallback_usage,
        "selected_symmetric_segments": symmetric_segments,
        "selected_target_types": ";".join(selected_types),
        "resource_constrained_segment_ratio": sum(r.resource_constrained for r in realizations) / len(realizations),
        "stable_reserved_tx": int(schedule.compaction.reserved_tx.sum()),
        "stable_reserved_rx": int(schedule.compaction.reserved_rx.sum()),
        "stable_exposed_tx": int(schedule.compaction.exposed_tx.sum()),
        "stable_exposed_rx": int(schedule.compaction.exposed_rx.sum()),
        "selected_from": getattr(schedule, "selected_from", "v1 pipeline"),
        "evidence": EVIDENCE,
    }


def run_end_to_end_v2(config: dict[str, Any], output_root: str | Path | None = None) -> dict[str, Path]:
    output = Path(output_root or config["output_dir"])
    paths = result_paths(output, "end_to_end_v2")
    seed = int(config.get("seed", 7))
    size = int(config.get("endpoint_count", 4))
    unit = _bytes_per_ms(float(config.get("unit_bandwidth_gbps", 100.0)))
    fixed = _fixed_matrix(size, float(config.get("base_bandwidth_gbps", 25.0)))
    delta = float(config.get("delta_ms", 0.5))
    epsilon = float(config.get("epsilon", 0.1))
    rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for kind in config.get("workloads", ["dp", "pp", "mixed"]):
        workload = _workload(config, str(kind))
        write_json(
            paths["raw"] / f"{kind}_ordered_demands.json",
            {
                "workload": workload.name,
                "provenance": workload.provenance,
                "node_ids": [node.node_id for node in workload.nodes],
                "matrices": [matrix.tolist() for matrix in workload.demands],
            },
        )
        for ports_value in config.get("port_budgets", [2, 4, 8]):
            ports = int(ports_value)
            resources = _resources(size, ports)
            static, symmetric_v1, drac_v1 = evaluate_main_schemes(
                workload.demands, resources, unit, delta, epsilon, fixed
            )
            v2 = plan_schedule_candidates_v2(
                workload.demands, resources, unit, delta, epsilon, fixed
            )
            candidates = (
                ("Static-Sym", static),
                ("Sym-OCS", symmetric_v1),
                ("DRAC-v1", drac_v1),
                ("DRAC-SegmentOpt", v2.directional),
                ("DRAC-SegmentOpt+Fallback", v2.selected),
            )
            for name, schedule in candidates:
                rows.append(_schedule_row(workload.name, name, ports, delta, epsilon, schedule, static.total_cost))
            decisions.append(
                {
                    "workload": workload.name,
                    "port_budget": ports,
                    "selected_from": v2.selected.selected_from,
                    "selected_types": list(v2.selected.selected_target_types),
                    "fallback_reasons": list(v2.selected.fallback_reasons),
                    "directional_total_ms": v2.directional.total_cost,
                    "symmetric_total_ms": v2.symmetric.total_cost,
                    "segment_fallback_total_ms": v2.segment_fallback.total_cost,
                    "selected_total_ms": v2.selected.total_cost,
                }
            )
    raw_csv = write_csv(paths["raw"] / "end_to_end_v2_raw.csv", rows)
    processed = write_csv(paths["processed"] / "end_to_end_v2.csv", rows)
    decisions_json = write_json(paths["raw"] / "fallback_decisions.json", decisions)
    manifest = write_manifest(
        paths["raw"],
        experiment="end_to_end_v2",
        config=config,
        seed=seed,
        status="complete",
        outputs={"raw": str(raw_csv), "processed": str(processed), "decisions": str(decisions_json)},
        evidence=EVIDENCE,
    )
    return {"raw": raw_csv, "processed": processed, "decisions": decisions_json, "manifest": manifest}


def _segmentation_rows(
    workload: EvaluationWorkload,
    table: CandidateTargetTable,
    deltas: Sequence[float],
    resources: OCSResources,
    unit_bandwidth: float,
    fixed_bandwidth: np.ndarray,
    epsilon: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    count = len(workload.demands)
    rows: list[dict[str, Any]] = []
    timelines: list[dict[str, Any]] = []
    for delta in (float(value) for value in deltas):
        medoid = segment_continuous_sequence(
            workload.demands, np.ones(1), np.ones(1), 1.0, delta, method="medoid", candidate_targets=table
        )
        segment_opt = segment_continuous_sequence(
            workload.demands, np.ones(1), np.ones(1), 1.0, delta, method="directional", candidate_targets=table
        )
        symmetric = segment_continuous_sequence(
            workload.demands, np.ones(1), np.ones(1), 1.0, delta, method="symmetric", candidate_targets=table
        )
        oracle_cost, oracle_boundaries = exhaustive_partition_oracle_v2(table, delta, method="directional")
        fallback_integer = _realize_schedule(
            segment_opt,
            workload.demands,
            resources,
            unit_bandwidth,
            delta,
            epsilon,
            fixed_bandwidth,
            mode="fallback",
        )
        fallback_fraction = sum(
            target_type == "symmetric" for target_type in fallback_integer.selected_target_types
        ) / len(fallback_integer.selected_target_types)
        schemes = (
            ("OneConfig", float(table.directional_cost[0, count - 1]), 1, ((0, count - 1),)),
            ("PerNodeReconfig", float(np.trace(table.directional_cost)) + delta * (count - 1), count, tuple((i, i) for i in range(count))),
            ("Medoid-DynamicProgramming", medoid.total_cost, len(medoid.segments), tuple((s.start, s.end) for s in medoid.segments)),
            ("SegmentOpt-DynamicProgramming", segment_opt.total_cost, len(segment_opt.segments), tuple((s.start, s.end) for s in segment_opt.segments)),
            ("ExhaustivePartitionOracle", oracle_cost, len(oracle_boundaries), oracle_boundaries),
            ("SymmetricFallbackSchedule", symmetric.total_cost, len(symmetric.segments), tuple((s.start, s.end) for s in symmetric.segments)),
            ("SegmentOpt+Fallback-Integer", fallback_integer.total_cost, len(fallback_integer.realizations), tuple((s.start, s.end) for s in segment_opt.segments)),
        )
        for scheme, total, segment_count, boundaries in schemes:
            communication = total - delta * max(0, segment_count - 1)
            rows.append(
                {
                    "workload": workload.name,
                    "scheme": scheme,
                    "delta_ms": delta,
                    "total_cost_ms": total,
                    "communication_cost_ms": communication,
                    "reconfiguration_cost_ms": total - communication,
                    "segment_count": segment_count,
                    "fallback_fraction": fallback_fraction if scheme == "SegmentOpt+Fallback-Integer" else (1.0 if scheme == "SymmetricFallbackSchedule" else 0.0),
                    "oracle_gap": (total - oracle_cost) / oracle_cost if oracle_cost > 0 else 0.0,
                    "boundaries": ";".join(f"{start}-{end}" for start, end in boundaries),
                    "evidence": EVIDENCE,
                }
            )
        timelines.append(
            {
                "workload": workload.name,
                "delta_ms": delta,
                "node_ids": [node.node_id for node in workload.nodes],
                "segment_opt_boundaries": [[s.start, s.end] for s in segment_opt.segments],
                "segment_target_methods": [s.target.method for s in segment_opt.segments],
                "segment_allocations": [s.target.allocation.tolist() for s in segment_opt.segments],
                "integer_fallback_types": list(fallback_integer.selected_target_types),
                "integer_fallback_reasons": list(fallback_integer.fallback_reasons),
            }
        )
    return rows, timelines


def run_segmentation_v2(config: dict[str, Any], output_root: str | Path | None = None) -> dict[str, Path]:
    output = Path(output_root or config["output_dir"])
    paths = result_paths(output, "segmentation_v2")
    seed = int(config.get("seed", 7))
    size = int(config.get("endpoint_count", 4))
    unit = _bytes_per_ms(float(config.get("unit_bandwidth_gbps", 100.0)))
    fixed = _fixed_matrix(size, float(config.get("base_bandwidth_gbps", 25.0)))
    resources = _resources(size, int(config.get("port_budget", 6)))
    rows: list[dict[str, Any]] = []
    timelines: list[dict[str, Any]] = []
    for kind in config.get("workloads", ["dp", "pp", "mixed"]):
        workload = _workload(config, str(kind))
        table = build_candidate_target_table(
            workload.demands, resources.n_tx, resources.n_rx, unit, fixed
        )
        workload_rows, workload_timelines = _segmentation_rows(
            workload,
            table,
            config.get("delta_values_ms", [0.0, 0.1, 1.0, 10.0]),
            resources,
            unit,
            fixed,
            float(config.get("epsilon", 0.5)),
        )
        rows.extend(workload_rows)
        timelines.extend(workload_timelines)
    raw_csv = write_csv(paths["raw"] / "segmentation_v2_raw.csv", rows)
    processed = write_csv(paths["processed"] / "segmentation_v2.csv", rows)
    timeline_json = write_json(paths["raw"] / "segmentation_timelines.json", timelines)
    manifest = write_manifest(paths["raw"], experiment="segmentation_v2", config=config, seed=seed, status="complete", outputs={"raw": str(raw_csv), "processed": str(processed), "timelines": str(timeline_json)}, evidence=EVIDENCE)
    return {"raw": raw_csv, "processed": processed, "timelines": timeline_json, "manifest": manifest}


def _hard_case(scale: float = 1.0) -> tuple[str, tuple[np.ndarray, ...]]:
    d0 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 6.0], [0.0, 3.0, 0.0]])
    d1 = np.array([[0.0, 6.0, 0.0], [0.0, 0.0, 5.0], [0.0, 1.0, 0.0]])
    return "Synthetic Hard", (d0 * scale, d1 * scale)


def _realization_cases(config: dict[str, Any]) -> list[tuple[str, tuple[np.ndarray, ...]]]:
    cases: list[tuple[str, tuple[np.ndarray, ...]]] = []
    for kind in ("dp", "pp", "mixed"):
        workload = build_evaluation_workload(
            kind,
            endpoint_count=int(config.get("endpoint_count", 3)),
            message_bytes=float(config.get("message_bytes", 32 * 1024 * 1024)),
            repeats=1,
        )
        cases.append((workload.name, workload.demands))
    cases.append(_hard_case(float(config.get("message_bytes", 32 * 1024 * 1024)) / 6.0))
    return cases


def run_realization_v2(config: dict[str, Any], output_root: str | Path | None = None) -> dict[str, Path]:
    output = Path(output_root or config["output_dir"])
    paths = result_paths(output, "realization_v2")
    seed = int(config.get("seed", 7))
    size = int(config.get("endpoint_count", 3))
    unit = _bytes_per_ms(float(config.get("unit_bandwidth_gbps", 100.0)))
    fixed = _fixed_matrix(size, float(config.get("base_bandwidth_gbps", 25.0)))
    resources = _resources(size, int(config.get("port_budget", 4)))
    epsilons = sorted(float(value) for value in config.get("epsilon_values", [0.0, 0.25, 0.5, 1.0]))
    rows: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    tradeoff_observed = False
    for name, demands in _realization_cases(config):
        target = solve_segment_continuous_target(
            demands, resources.n_tx, resources.n_rx, unit, fixed, symmetric=False
        )
        history: list[np.ndarray] = []
        multiseed_counts: list[tuple[int, bool]] = []
        for epsilon in epsilons:
            policies: list[tuple[str, Callable[..., RealizationResult]]] = [
                ("FloorOnly", realize_floor_only),
                ("NearestRounding", realize_nearest_rounding),
                ("FillAllResidual", realize_fill_all_residual),
                ("DRACSparse-FloorSeed", realize_drac_sparse_floor_seed),
                ("DRACSparse-CoverageSeed", realize_drac_sparse_coverage_seed),
            ]
            results: list[tuple[str, RealizationResult]] = [
                (policy_name, policy(target.allocation, demands, target.cost, epsilon, resources, unit, fixed))
                for policy_name, policy in policies
            ]
            multiseed = realize_drac_sparse_multi_seed(
                target.allocation,
                demands,
                target.cost,
                epsilon,
                resources,
                unit,
                fixed,
                historical_units=history,
            )
            history.append(multiseed.units.copy())
            multiseed_counts.append((multiseed.used_units, multiseed.tolerance_satisfied))
            results.append(("DRACSparse-MultiSeed", multiseed))
            try:
                oracle = exhaustive_realization_oracle(
                    target.allocation, demands, target.cost, epsilon, resources, unit, fixed
                )
                results.append(("ExhaustiveOracle", oracle))
                oracle_units = oracle.used_units if oracle.tolerance_satisfied else math.nan
            except ValueError:
                oracle_units = math.nan
            for policy_name, result in results:
                compaction = compact_schedule([result.units], resources.n_tx, resources.n_rx, [result])
                rows.append(
                    {
                        "workload": name,
                        "policy": policy_name,
                        "epsilon": epsilon,
                        "realized_slowdown": result.cost / target.cost if target.cost > 0 else 1.0,
                        "used_connection_units": result.used_units,
                        "stable_reserved_channels": compaction.total_stable_directional_pool,
                        "tolerance_satisfied": result.tolerance_satisfied,
                        "resource_constrained": result.resource_constrained,
                        "oracle_unit_gap": result.used_units - oracle_units if np.isfinite(oracle_units) else math.nan,
                        "seed": result.seed,
                        "group_additions": result.group_additions,
                        "swap_count": result.swaps,
                        "swap_gain_ms": result.swap_gain,
                        "pruning_count": result.pruned,
                        "history_reused": result.reused_history,
                        "evidence": EVIDENCE,
                    }
                )
                logs.append({"workload": name, "policy": policy_name, "epsilon": epsilon, "events": list(result.events)})
        feasible_counts = [count for count, feasible in multiseed_counts if feasible]
        if feasible_counts != sorted(feasible_counts, reverse=True):
            raise AssertionError(f"feasible epsilon monotonicity failed for {name}: {multiseed_counts}")
        tradeoff_observed |= len(set(feasible_counts)) > 1
    if not tradeoff_observed:
        raise RuntimeError("DRACSparse-MultiSeed remained horizontal across every epsilon case")
    raw_csv = write_csv(paths["raw"] / "realization_v2_raw.csv", rows)
    processed = write_csv(paths["processed"] / "realization_tradeoff_v2.csv", rows)
    log_json = write_json(paths["raw"] / "realization_events.json", logs)
    manifest = write_manifest(paths["raw"], experiment="realization_v2", config=config, seed=seed, status="complete", outputs={"raw": str(raw_csv), "processed": str(processed), "events": str(log_json)}, evidence=EVIDENCE)
    return {"raw": raw_csv, "processed": processed, "events": log_json, "manifest": manifest}


def _compaction_row(workload: str, scheme: str, compaction: Any, size: int, ports: int) -> dict[str, Any]:
    full = 2 * size * ports
    return {
        "condition": "fixed_physical_budget",
        "workload": workload,
        "scheme": scheme,
        "physical_port_budget": ports,
        "stable_reserved_tx": int(compaction.reserved_tx.sum()),
        "stable_reserved_rx": int(compaction.reserved_rx.sum()),
        "stable_exposed_tx": int(compaction.exposed_tx.sum()),
        "stable_exposed_rx": int(compaction.exposed_rx.sum()),
        "total_stable_pool": compaction.total_stable_directional_pool,
        "reserved_bundle_pool": int(compaction.reserved_bundles.sum()),
        "compaction_ratio": 1.0 - compaction.total_stable_directional_pool / full,
        "evidence": EVIDENCE,
    }


def run_compaction_v2(config: dict[str, Any], output_root: str | Path | None = None) -> dict[str, Path]:
    output = Path(output_root or config["output_dir"])
    paths = result_paths(output, "compaction_v2")
    seed = int(config.get("seed", 7))
    size = int(config.get("endpoint_count", 4))
    ports = int(config.get("port_budget", 8))
    unit = _bytes_per_ms(float(config.get("unit_bandwidth_gbps", 100.0)))
    fixed = _fixed_matrix(size, float(config.get("base_bandwidth_gbps", 25.0)))
    delta = float(config.get("delta_ms", 0.5))
    epsilon = float(config.get("epsilon", 0.5))
    rows: list[dict[str, Any]] = []
    iso_rows: list[dict[str, Any]] = []
    for kind in config.get("workloads", ["dp", "pp", "mixed"]):
        workload = _workload(config, str(kind))
        resources = _resources(size, ports)
        _, sym_v1, drac_v1 = evaluate_main_schemes(
            workload.demands, resources, unit, delta, epsilon, fixed
        )
        v2 = plan_schedule_candidates_v2(
            workload.demands, resources, unit, delta, epsilon, fixed
        )
        full_units = np.full(size, ports, dtype=int)
        full_compaction = compact_schedule([], full_units, full_units)
        # Empty schedules expose everything, so construct the FullReservation
        # accounting row explicitly instead of misusing the compactor output.
        rows.append(
            {
                "condition": "fixed_physical_budget",
                "workload": workload.name,
                "scheme": "FullReservation",
                "physical_port_budget": ports,
                "stable_reserved_tx": size * ports,
                "stable_reserved_rx": size * ports,
                "stable_exposed_tx": 0,
                "stable_exposed_rx": 0,
                "total_stable_pool": 2 * size * ports,
                "reserved_bundle_pool": size * ports,
                "compaction_ratio": 0.0,
                "evidence": EVIDENCE,
            }
        )
        for scheme, schedule in (
            ("Sym-OCS", sym_v1),
            ("DRAC-v1", drac_v1),
            ("DRAC-SegmentOpt", v2.directional),
            ("DRAC-Sparse", v2.selected),
        ):
            rows.append(_compaction_row(workload.name, scheme, schedule.compaction, size, ports))

        reference_ports = int(config.get("iso_reference_port_budget", 4))
        reference = plan_reconfigurable_schedule(
            workload.demands,
            _resources(size, reference_ports),
            unit,
            delta,
            epsilon,
            fixed_bandwidth=fixed,
            symmetric=True,
        )
        reached = None
        for candidate_ports in range(1, ports + 1):
            trial = plan_schedule_candidates_v2(
                workload.demands,
                _resources(size, candidate_ports),
                unit,
                delta,
                epsilon,
                fixed,
            ).selected
            if trial.total_cost <= reference.total_cost * (1.0 + 1e-9):
                reached = (candidate_ports, trial)
                break
        iso_rows.append(
            {
                "condition": "iso_performance_search",
                "workload": workload.name,
                "scheme": "DRAC-Sparse",
                "reference_scheme": "Sym-OCS",
                "reference_port_budget": reference_ports,
                "reference_total_ms": reference.total_cost,
                "minimum_port_budget": reached[0] if reached else "",
                "minimum_stable_directional_pool": reached[1].compaction.total_stable_directional_pool if reached else "",
                "minimum_stable_bundle_pool": int(reached[1].compaction.reserved_bundles.sum()) if reached else "",
                "status": "reached" if reached else "not_reached",
                "evidence": EVIDENCE,
            }
        )
    raw_csv = write_csv(paths["raw"] / "schedule_compaction_v2_raw.csv", rows)
    processed = write_csv(paths["processed"] / "schedule_compaction_v2.csv", rows)
    iso = write_csv(paths["processed"] / "iso_performance_pool_v2.csv", iso_rows)
    manifest = write_manifest(paths["raw"], experiment="compaction_v2", config=config, seed=seed, status="complete", outputs={"raw": str(raw_csv), "processed": str(processed), "iso": str(iso)}, evidence=EVIDENCE)
    return {"raw": raw_csv, "processed": processed, "iso": iso, "manifest": manifest}


def run_planning_overhead_v2(config: dict[str, Any], output_root: str | Path | None = None) -> dict[str, Path]:
    output = Path(output_root or config["output_dir"])
    paths = result_paths(output, "overhead_v2")
    seed = int(config.get("seed", 7))
    rows: list[dict[str, Any]] = []
    for requested_k in config.get("node_counts", [4, 8, 16]):
        size = int(config.get("endpoint_count", 4))
        unit = _bytes_per_ms(float(config.get("unit_bandwidth_gbps", 100.0)))
        fixed = _fixed_matrix(size, float(config.get("base_bandwidth_gbps", 25.0)))
        resources = _resources(size, int(config.get("port_budget", 6)))
        workload = build_evaluation_workload("mixed", endpoint_count=size, message_bytes=float(config.get("message_bytes", 16 * 1024 * 1024)), repeats=max(1, int(requested_k) // max(1, 2 * size)))
        demands = workload.demands[: int(requested_k)]
        start = perf_counter()
        table = build_candidate_target_table(demands, resources.n_tx, resources.n_rx, unit, fixed)
        target_ms = (perf_counter() - start) * 1000.0
        start = perf_counter()
        segmentation = segment_continuous_sequence(demands, resources.n_tx, resources.n_rx, unit, float(config.get("delta_ms", 0.5)), fixed, candidate_targets=table)
        dynamic_ms = (perf_counter() - start) * 1000.0
        start = perf_counter()
        realizations = [realize_drac_sparse_multi_seed(segment.target.allocation, demands[segment.start:segment.end+1], segment.target.cost, float(config.get("epsilon", 0.5)), resources, unit, fixed) for segment in segmentation.segments]
        realization_ms = (perf_counter() - start) * 1000.0
        start = perf_counter()
        compact_schedule([r.units for r in realizations], resources.n_tx, resources.n_rx, realizations)
        compaction_ms = (perf_counter() - start) * 1000.0
        rows.append(
            {
                "node_count": len(demands),
                "endpoint_count": size,
                "segment_count": len(segmentation.segments),
                "port_budget": int(config.get("port_budget", 6)),
                "candidate_segment_target_ms": target_ms,
                "dynamic_programming_ms": dynamic_ms,
                "sparse_realization_ms": realization_ms,
                "schedule_compaction_ms": compaction_ms,
                "total_planning_ms": target_ms + dynamic_ms + realization_ms + compaction_ms,
                "evidence": EVIDENCE,
            }
        )
    raw_csv = write_csv(paths["raw"] / "planning_runtime_v2_raw.csv", rows)
    processed = write_csv(paths["processed"] / "planning_runtime_v2.csv", rows)
    manifest = write_manifest(paths["raw"], experiment="overhead_v2", config=config, seed=seed, status="complete", outputs={"raw": str(raw_csv), "processed": str(processed)}, evidence=EVIDENCE)
    return {"raw": raw_csv, "processed": processed, "manifest": manifest}
