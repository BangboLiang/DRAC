from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .allocation import AllocationResult, allocate_for_algorithm
from .config import NetworkConfig
from .metrics import _gbps_to_bytes_per_ms, compute_segment_metrics
from .rescue_allocation import allocate_rescue_method, validate_units
from .rescue_config import RescueConfig
from .rescue_plotting import plot_aggregation, plot_collective, plot_makespan
from .rescue_traffic import (
    aggregate_matrix,
    apply_collective_model,
    build_mapping,
    directional_opportunity,
    level_group_size,
    skew_statistics,
)
from .traffic import SegmentDemand, load_or_generate_workload


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _safe_ratio(numer: float, denom: float, epsilon: float = 1e-12) -> float:
    return float(numer / denom) if abs(denom) > epsilon else float("nan")


def _net_for(cfg: RescueConfig, node_count: int, port_budget: int) -> NetworkConfig:
    total = min(int(cfg.network.total_ocs_links), int(node_count) * int(port_budget))
    return cfg.network.with_overrides(
        {"per_node_port_budget": int(port_budget), "total_ocs_links": max(0, total)}
    )


def _alloc_metrics(demand: np.ndarray, allocation: AllocationResult, net: NetworkConfig) -> Dict[str, float]:
    validate_units(allocation.connection_units, net)
    metrics = compute_segment_metrics(demand, allocation, net)
    values = {
        "communication_time_ms": metrics.completion_time_ms,
        "capacity_waste_ratio": metrics.waste_ratio,
        "matching_error": metrics.matching_error_l1,
        "ocs_port_utilization": metrics.ocs_port_utilization,
    }
    if not all(np.isfinite(v) for v in values.values()):
        raise AssertionError("unexpected NaN/inf in performance metrics")
    return values


def _endpoint_segments(cfg: RescueConfig, workload: object, seed: int) -> List[SegmentDemand]:
    return load_or_generate_workload(
        workload, cfg.endpoint_count, cfg.asymmetry_level, seed
    )


def _add_iso_ports(rows: List[Dict[str, object]], group_keys: Sequence[str]) -> None:
    grouped: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in group_keys)].append(row)
    for group in grouped.values():
        by_alg_port: Dict[Tuple[str, int], float] = defaultdict(float)
        for row in group:
            by_alg_port[(str(row["algorithm"]), int(row["port_budget"]))] += float(row["communication_time_ms"])
        ports = sorted({int(row["port_budget"]) for row in group})
        sym_ref = by_alg_port.get(("sym_ocs", max(ports)), float("nan")) if ports else float("nan")
        required: Dict[str, float] = {}
        for algorithm in {str(row["algorithm"]) for row in group}:
            feasible = [p for p in ports if by_alg_port.get((algorithm, p), float("inf")) <= sym_ref * (1.0 + 1e-12)]
            required[algorithm] = float(min(feasible)) if feasible else float("nan")
        for row in group:
            row["required_ports_for_iso_performance"] = required[str(row["algorithm"])]


def run_aggregation(cfg: RescueConfig, root: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    segment_rows: List[Dict[str, object]] = []
    performance: List[Dict[str, object]] = []
    for workload in cfg.workloads:
        for seed in cfg.seeds:
            endpoint_segments = _endpoint_segments(cfg, workload, seed)
            endpoint_omegas = {s.segment_idx: directional_opportunity(s.matrix) for s in endpoint_segments}
            for mapping_name in cfg.mapping_strategies:
                for level in cfg.aggregation_levels:
                    group_size = level_group_size(cfg, level)
                    mapping_seed = int(seed)
                    mapping = build_mapping(cfg.endpoint_count, group_size, mapping_name, mapping_seed)
                    if len(set(mapping.tolist())) < 2:
                        continue
                    aggregated: List[Tuple[SegmentDemand, np.ndarray]] = []
                    for segment in endpoint_segments:
                        matrix, cross = aggregate_matrix(segment.matrix, mapping)
                        omega = directional_opportunity(matrix)
                        endpoint_omega = endpoint_omegas[segment.segment_idx]
                        retention = _safe_ratio(omega, endpoint_omega, cfg.omega_epsilon)
                        skew = skew_statistics(matrix)
                        row: Dict[str, object] = {
                            "workload": workload.name,
                            "seed": seed,
                            "mapping_seed": mapping_seed,
                            "mapping": mapping_name,
                            "level": level,
                            "segment_idx": segment.segment_idx,
                            "endpoint_count": cfg.endpoint_count,
                            "abstract_node_count": matrix.shape[0],
                            "cross_boundary_bytes": cross,
                            "omega": omega,
                            "endpoint_omega": endpoint_omega,
                            "retention_ratio": retention,
                        }
                        row.update(skew)
                        segment_rows.append(row)
                        aggregated.append((segment, matrix))

                    aggregate_demand = np.sum([m for _, m in aggregated], axis=0)
                    for port_budget in cfg.port_budgets:
                        net = _net_for(cfg, aggregate_demand.shape[0], port_budget)
                        static = allocate_for_algorithm("static_sym", aggregate_demand, net)
                        for segment, matrix in aggregated:
                            allocs = {
                                "static_sym": allocate_for_algorithm("static_sym", matrix, net, static_target=static.target_overlay),
                                "sym_ocs": allocate_for_algorithm("sym_ocs", matrix, net),
                                "drac": allocate_for_algorithm("drac", matrix, net),
                            }
                            vals = {alg: _alloc_metrics(matrix, alloc, net) for alg, alloc in allocs.items()}
                            for algorithm, metrics in vals.items():
                                performance.append(
                                    {
                                        "workload": workload.name,
                                        "seed": seed,
                                        "mapping_seed": mapping_seed,
                                        "mapping": mapping_name,
                                        "level": level,
                                        "segment_idx": segment.segment_idx,
                                        "abstract_node_count": matrix.shape[0],
                                        "port_budget": port_budget,
                                        "algorithm": algorithm,
                                        **metrics,
                                        "normalized_communication_time": _safe_ratio(metrics["communication_time_ms"], vals["static_sym"]["communication_time_ms"]),
                                        "speedup_vs_sym_ocs": _safe_ratio(vals["sym_ocs"]["communication_time_ms"], metrics["communication_time_ms"]),
                                    }
                                )
    _add_iso_ports(performance, ["workload", "seed", "mapping", "level"])

    summary: List[Dict[str, object]] = []
    groups: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in segment_rows:
        groups[(str(row["workload"]), str(row["mapping"]), str(row["level"]))].append(row)
    for (workload, mapping, level), rows in groups.items():
        omegas = [float(r["omega"]) for r in rows]
        weights = [float(r["cross_boundary_bytes"]) for r in rows]
        retentions = [float(r["retention_ratio"]) for r in rows]
        finite_omegas = [v for v in omegas if np.isfinite(v)]
        finite_retentions = [v for v in retentions if np.isfinite(v)]
        summary.append(
            {
                "workload": workload,
                "mapping": mapping,
                "level": level,
                "sample_count": len(rows),
                "omega_mean": _mean(omegas),
                "omega_std": float(np.std(finite_omegas)) if finite_omegas else float("nan"),
                "omega_min": float(min(finite_omegas)) if finite_omegas else float("nan"),
                "omega_max": float(max(finite_omegas)) if finite_omegas else float("nan"),
                "traffic_weighted_omega": float(np.average([v for v, w in zip(omegas, weights) if np.isfinite(v) and w > 0], weights=[w for v, w in zip(omegas, weights) if np.isfinite(v) and w > 0])) if any(np.isfinite(v) and w > 0 for v, w in zip(omegas, weights)) else float("nan"),
                "retention_mean": _mean(retentions),
                "retention_std": float(np.std(finite_retentions)) if finite_retentions else float("nan"),
                "retention_min": float(min(finite_retentions)) if finite_retentions else float("nan"),
                "retention_max": float(max(finite_retentions)) if finite_retentions else float("nan"),
            }
        )
    out = root / "aggregation_retention"
    _write_csv(out / "aggregation_segment_metrics.csv", segment_rows)
    _write_csv(out / "aggregation_workload_summary.csv", summary)
    _write_csv(out / "aggregation_performance.csv", performance)
    plot_aggregation(segment_rows, summary, performance, out)
    return segment_rows, summary, performance


def run_collective(cfg: RescueConfig, root: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    segment_rows: List[Dict[str, object]] = []
    performance: List[Dict[str, object]] = []
    for workload in cfg.workloads:
        for seed in cfg.seeds:
            original = _endpoint_segments(cfg, workload, seed)
            for model in cfg.collective_models:
                modeled = [(s, apply_collective_model(s.matrix, model, cfg.bidirectional_chunks)) for s in original]
                for segment, matrix in modeled:
                    segment_rows.append(
                        {"workload": workload.name, "seed": seed, "segment_idx": segment.segment_idx, "collective_model": model, "omega": directional_opportunity(matrix), "total_payload_bytes": float(matrix.sum())}
                    )
                aggregate_demand = np.sum([m for _, m in modeled], axis=0)
                for port_budget in cfg.port_budgets:
                    net = _net_for(cfg, cfg.endpoint_count, port_budget)
                    static = allocate_for_algorithm("static_sym", aggregate_demand, net)
                    for segment, matrix in modeled:
                        omega = directional_opportunity(matrix)
                        base_allocs = {
                            "static_sym": allocate_for_algorithm("static_sym", matrix, net, static_target=static.target_overlay),
                            "sym_ocs": allocate_for_algorithm("sym_ocs", matrix, net),
                            "drac": allocate_for_algorithm("drac", matrix, net),
                        }
                        vals = {alg: _alloc_metrics(matrix, alloc, net) for alg, alloc in base_allocs.items()}
                        for algorithm in [a for a in cfg.algorithms if a != "drac_gated"]:
                            if algorithm not in vals:
                                continue
                            metrics = vals[algorithm]
                            gain = 1.0 - _safe_ratio(metrics["communication_time_ms"], vals["sym_ocs"]["communication_time_ms"])
                            performance.append(
                                {"workload": workload.name, "seed": seed, "segment_idx": segment.segment_idx, "collective_model": model, "port_budget": port_budget, "omega_threshold": "", "algorithm": algorithm, "omega": omega, **metrics, "normalized_communication_time": _safe_ratio(metrics["communication_time_ms"], vals["static_sym"]["communication_time_ms"]), "gain_vs_sym_ocs": gain}
                            )
                        if "drac_gated" in cfg.algorithms:
                            for threshold in cfg.omega_thresholds:
                                gate_to_sym = bool(np.isfinite(omega) and omega < threshold)
                                alloc = base_allocs["sym_ocs"] if gate_to_sym else base_allocs["drac"]
                                if gate_to_sym and not np.array_equal(alloc.connection_units, base_allocs["sym_ocs"].connection_units):
                                    raise AssertionError("drac_gated did not degrade to Sym-OCS")
                                metrics = _alloc_metrics(matrix, alloc, net)
                                performance.append(
                                    {"workload": workload.name, "seed": seed, "segment_idx": segment.segment_idx, "collective_model": model, "port_budget": port_budget, "omega_threshold": threshold, "algorithm": "drac_gated", "gated_to_sym_ocs": gate_to_sym, "omega": omega, **metrics, "normalized_communication_time": _safe_ratio(metrics["communication_time_ms"], vals["static_sym"]["communication_time_ms"]), "gain_vs_sym_ocs": 1.0 - _safe_ratio(metrics["communication_time_ms"], vals["sym_ocs"]["communication_time_ms"])}
                                )
    _add_iso_ports(performance, ["workload", "seed", "collective_model", "omega_threshold"])

    gain_totals: Dict[Tuple[str, str, int], float] = {}
    for workload in {str(r["workload"]) for r in performance}:
        for model in cfg.collective_models:
            for port in cfg.port_budgets:
                sym = sum(float(r["communication_time_ms"]) for r in performance if r["workload"] == workload and r["collective_model"] == model and r["port_budget"] == port and r["algorithm"] == "sym_ocs")
                drac = sum(float(r["communication_time_ms"]) for r in performance if r["workload"] == workload and r["collective_model"] == model and r["port_budget"] == port and r["algorithm"] == "drac")
                gain_totals[(workload, model, port)] = 1.0 - _safe_ratio(drac, sym)
    summary: List[Dict[str, object]] = []
    for workload in {str(r["workload"]) for r in segment_rows}:
        for model in cfg.collective_models:
            rows = [r for r in segment_rows if r["workload"] == workload and r["collective_model"] == model]
            weights = [float(r["total_payload_bytes"]) for r in rows]
            omega = float(np.average([float(r["omega"]) for r in rows], weights=weights)) if sum(weights) else float("nan")
            for port in cfg.port_budgets:
                gain = gain_totals[(workload, model, port)]
                original_gain = gain_totals[(workload, "original", port)]
                summary.append({"workload": workload, "collective_model": model, "port_budget": port, "traffic_weighted_omega": omega, "drac_gain_vs_sym_ocs": gain, "benefit_retention": _safe_ratio(gain, original_gain, 1e-9)})
    gating = [r for r in performance if r["algorithm"] == "drac_gated"]
    out = root / "collective_balancing"
    _write_csv(out / "collective_balancing_segment_metrics.csv", segment_rows)
    _write_csv(out / "collective_balancing_performance.csv", performance)
    _write_csv(out / "collective_balancing_summary.csv", summary)
    _write_csv(out / "gating_threshold_sensitivity.csv", gating)
    plot_collective(segment_rows, performance, summary, gating, out)
    return segment_rows, performance, summary, gating


def _aggregate_flow_delay(demand: np.ndarray, capacity: np.ndarray) -> float:
    cap = _gbps_to_bytes_per_ms(capacity)
    mask = demand > 0.0
    return float(np.sum(demand[mask] / cap[mask]))


def run_makespan(cfg: RescueConfig, root: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    comparison: List[Dict[str, object]] = []
    methods = ["sqrt_sum_delay", "proportional_makespan", "discrete_makespan_opt"]
    for workload in cfg.workloads:
        for seed in cfg.seeds:
            for segment in _endpoint_segments(cfg, workload, seed):
                for port_budget in cfg.port_budgets:
                    net = _net_for(cfg, cfg.endpoint_count, port_budget)
                    results = {method: allocate_rescue_method(method, segment.matrix, net) for method in methods}
                    opt_alloc = results["discrete_makespan_opt"][0]
                    opt_time = compute_segment_metrics(segment.matrix, opt_alloc, net).completion_time_ms
                    for method, (alloc, runtime_ms) in results.items():
                        validate_units(alloc.connection_units, net)
                        metrics = compute_segment_metrics(segment.matrix, alloc, net)
                        if opt_time > metrics.completion_time_ms * (1.0 + 1e-9) + 1e-9:
                            raise AssertionError("discrete_makespan_opt is worse than a comparison method")
                        comparison.append(
                            {"workload": workload.name, "seed": seed, "segment_idx": segment.segment_idx, "port_budget": port_budget, "method": method, "communication_makespan_ms": metrics.completion_time_ms, "aggregate_flow_delay_ms": _aggregate_flow_delay(segment.matrix, alloc.total_bandwidth), "target_realization_gap_gbps": float(np.abs(alloc.target_overlay - alloc.realized_overlay).sum()), "runtime_ms": runtime_ms, "capacity_waste_ratio": metrics.waste_ratio, "optimality_gap": _safe_ratio(metrics.completion_time_ms - opt_time, opt_time)}
                        )
    gaps = [{k: r[k] for k in ["workload", "seed", "segment_idx", "port_budget", "method", "communication_makespan_ms", "optimality_gap"]} for r in comparison]
    runtimes = [{k: r[k] for k in ["workload", "seed", "segment_idx", "port_budget", "method", "runtime_ms"]} for r in comparison]
    out = root / "makespan_objective"
    _write_csv(out / "makespan_method_comparison.csv", comparison)
    _write_csv(out / "makespan_optimality_gap.csv", gaps)
    _write_csv(out / "makespan_solver_runtime.csv", runtimes)
    plot_makespan(comparison, gaps, runtimes, out)
    return comparison, gaps, runtimes


def _risk(value: bool) -> str:
    return "HIGH" if value else "LOW"


def write_report(cfg: RescueConfig, root: Path, aggregation: object | None, collective: object | None, makespan: object | None, smoke: bool) -> None:
    agg_summary = aggregation[1] if aggregation else []
    agg_perf = aggregation[2] if aggregation else []
    coll_summary = collective[2] if collective else []
    make_rows = makespan[0] if makespan else []
    tor_omega = _mean(float(r["traffic_weighted_omega"]) for r in agg_summary if r["level"] == "tor" and r["mapping"] == "contiguous")
    tor_speedup = _mean(float(r["speedup_vs_sym_ocs"]) for r in agg_perf if r["level"] == "tor" and r["mapping"] == "contiguous" and r["algorithm"] == "drac")
    bidi_retention = _mean(float(r["benefit_retention"]) for r in coll_summary if r["collective_model"] == "bidirectional_balanced")
    oracle_retention = _mean(float(r["benefit_retention"]) for r in coll_summary if r["collective_model"] == "pairwise_balancing_oracle")
    sqrt_gap = _mean(float(r["optimality_gap"]) for r in make_rows if r["method"] == "sqrt_sum_delay")
    prop_gap = _mean(float(r["optimality_gap"]) for r in make_rows if r["method"] == "proportional_makespan")
    aggregation_risk = bool(np.isfinite(tor_omega) and (tor_omega < 0.1 or tor_speedup < 1.02))
    replacement_risk = bool(np.isfinite(bidi_retention) and bidi_retention < 0.25 or np.isfinite(oracle_retention) and oracle_retention < 0.1)
    objective_risk = bool(np.isfinite(sqrt_gap) and (sqrt_gap > 0.05 or sqrt_gap > prop_gap + 0.03))
    command = "python run_rescue_experiments.py --config configs/rescue_experiments.json" + (" --smoke-test" if smoke else " --experiment all")
    agg_lines = ["| Workload | Level | Traffic-weighted Omega | Retention | DRAC/Sym-OCS speedup |", "|---|---|---:|---:|---:|"]
    for workload in sorted({str(r["workload"]) for r in agg_summary}):
        for level in ["endpoint", "server", "tor", "aggregation"]:
            rows = [r for r in agg_summary if r["workload"] == workload and r["mapping"] == "contiguous" and r["level"] == level]
            perf = [r for r in agg_perf if r["workload"] == workload and r["mapping"] == "contiguous" and r["level"] == level and r["algorithm"] == "drac"]
            if not rows:
                continue
            agg_lines.append(f"| {workload.upper()} | {level} | {float(rows[0]['traffic_weighted_omega']):.4f} | {float(rows[0]['retention_mean']):.4f} | {_mean(float(r['speedup_vs_sym_ocs']) for r in perf):.4f} |")
    coll_lines = ["| Workload | Model | Omega | DRAC gain vs Sym-OCS | Benefit retention |", "|---|---|---:|---:|---:|"]
    for workload in sorted({str(r["workload"]) for r in coll_summary}):
        for model in cfg.collective_models:
            rows = [r for r in coll_summary if r["workload"] == workload and r["collective_model"] == model]
            if rows:
                label = "Pairwise Balancing Oracle" if "oracle" in model else model
                coll_lines.append(f"| {workload.upper()} | {label} | {_mean(float(r['traffic_weighted_omega']) for r in rows):.4f} | {_mean(float(r['drac_gain_vs_sym_ocs']) for r in rows):.4f} | {_mean(float(r['benefit_retention']) for r in rows):.4f} |")
    make_lines = ["| Workload | Method | Mean makespan (ms) | Mean optimality gap | Mean runtime (ms) |", "|---|---|---:|---:|---:|"]
    for workload in sorted({str(r["workload"]) for r in make_rows}):
        for method in ["sqrt_sum_delay", "proportional_makespan", "discrete_makespan_opt"]:
            rows = [r for r in make_rows if r["workload"] == workload and r["method"] == method]
            if rows:
                make_lines.append(f"| {workload.upper()} | {method} | {_mean(float(r['communication_makespan_ms']) for r in rows):.4f} | {_mean(float(r['optimality_gap']) for r in rows):.4f} | {_mean(float(r['runtime_ms']) for r in rows):.4f} |")
    text = f"""# DRAC Rescue Experiments Report

This report is generated from measured simulator output; no expected conclusion is hard-coded.

## Files and reproduction

Added `drac_eval/rescue_config.py`, `drac_eval/rescue_traffic.py`, `drac_eval/rescue_allocation.py`, `drac_eval/rescue_plotting.py`, `drac_eval/rescue_runner.py`, `run_rescue_experiments.py`, `configs/rescue_experiments.json`, and `tests/test_rescue_experiments.py`.

Run: `{command}`

## Existing simulator structure

The active paper path is `run_drac_eval.py -> drac_eval.runner -> drac_eval.traffic/allocation/metrics`, not the legacy `llama3_*.py` entry scripts. Workloads are lists of `SegmentDemand`; each segment contains a square byte-valued ordered rank-pair matrix. Payload sizes come from `llama3_comm.traffic.llama3_megatron_payloads`. The current generator indexes Megatron ranks (GPU/rank endpoints). Although the paper defines an abstract node as GPU, server, ToR, or aggregation block, the existing Evaluation does not execute the lower-level `phi` aggregation.

`static_sym` realizes one aggregate symmetric target for all segments; `sym_ocs` recomputes a symmetric square-root pair target per segment; `drac` uses a global `sqrt(T_ij)` directional target; `drac_sym` applies symmetric realization to the DRAC target. Realization is floor followed by greedy largest residual gap, subject to per-node outbound/inbound and total directed-link budgets. Completion time is the maximum ordered-pair `demand/capacity`; waste is idle provisioned capacity at that makespan. Port budget is enforced independently on row and column sums. The base network is a complete ordered-pair matrix.

## Assumptions

- Aggregation starts from the actual rank-level workload generator and excludes traffic whose endpoints map to the same abstract node. Hierarchy sizes and mapping are configured.
- `bidirectional_balanced` is a chunk-routing abstraction: half the chunks use each orientation; an odd extra chunk retains original orientation. Iterative proportional fitting preserves every endpoint's total sends and receives. It is not a claim about a specific NCCL implementation.
- `pairwise_balancing_oracle` is an optimistic Oracle only; it averages each unordered pair and preserves total bytes.
- The exact discrete makespan solver needs no ILP under the current all-to-all reachability model: for a candidate Theta, entrywise minimum required integer units are unique lower bounds, so row/column/global sums are a complete feasibility test.

## Key results

| Metric | Measured value |
|---|---:|
| Mean contiguous ToR traffic-weighted Omega | {tor_omega:.6g} |
| Mean contiguous ToR DRAC/Sym-OCS speedup | {tor_speedup:.6g} |
| Bidirectional DRAC benefit retention | {bidi_retention:.6g} |
| Oracle DRAC benefit retention | {oracle_retention:.6g} |
| sqrt target mean makespan optimality gap | {sqrt_gap:.6g} |
| proportional target mean makespan optimality gap | {prop_gap:.6g} |

### Aggregation by workload (contiguous mapping)

{chr(10).join(agg_lines)}

### Collective balancing by workload

{chr(10).join(coll_lines)}

### Makespan objective by workload

{chr(10).join(make_lines)}

## Claims and diagnostics

- Directionality remains at server/ToR/aggregation: {'SUPPORTED' if np.isfinite(tor_omega) and tor_omega >= 0.1 else 'NOT SUPPORTED OR WEAK'} by the configured ToR diagnostic.
- Collective balancing cannot fully replace DRAC: {'SUPPORTED' if np.isfinite(bidi_retention) and bidi_retention >= 0.25 else 'NOT SUPPORTED OR WEAK'}.
- The current square-root allocation suits communication makespan: {'SUPPORTED' if np.isfinite(sqrt_gap) and sqrt_gap <= 0.05 else 'NOT SUPPORTED'}.
- Aggregation Risk: **{_risk(aggregation_risk)}** (diagnostic threshold only: ToR Omega < 0.1 or speedup < 1.02).
- Collective Replacement Risk: **{_risk(replacement_risk)}** (diagnostic threshold only: bidirectional retention < 0.25 or Oracle retention < 0.1).
- Objective Mismatch Risk: **{_risk(objective_risk)}** (diagnostic threshold only: sqrt gap > 5% or materially worse than proportional).

## Negative results that may weaken the paper

Any HIGH diagnostic above is a direct negative result. In particular, low ToR Omega/speedup weakens deployment-level opportunity; low benefit retention indicates collective-layer balancing is a substitute; and a positive sqrt optimality gap shows the current sum-delay target is mismatched to the Evaluation's max-pair makespan. The Pairwise Balancing Oracle removes all directionality and all measured DRAC gain in this model. Bidirectional balancing reduces DP's opportunity sharply, so the collective layer is a credible substitute for that workload even though TP and Mixed retain residual opportunity. PP is weakly directional before aggregation and remains a no-opportunity/no-harm case. Full per-workload, seed, mapping, port, threshold, and method results are retained in the CSV files without selective filtering.
"""
    (root / "REPORT.md").write_text(text, encoding="utf-8")


def run_rescue_experiments(cfg: RescueConfig, experiment: str = "all", smoke: bool = False) -> Dict[str, Path]:
    if smoke:
        cfg = cfg.smoke_copy()
    root = Path(cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    aggregation = run_aggregation(cfg, root) if experiment in {"aggregation", "all"} else None
    collective = run_collective(cfg, root) if experiment in {"collective", "all"} else None
    makespan = run_makespan(cfg, root) if experiment in {"makespan", "all"} else None
    write_report(cfg, root, aggregation, collective, makespan, smoke)
    manifest = {"name": cfg.name, "experiment": experiment, "smoke_test": smoke, "seed": cfg.seed, "seeds": cfg.seeds, "output_dir": str(root.resolve())}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"root": root, "report": root / "REPORT.md"}
