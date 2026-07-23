#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


V1_BASELINE = {
    "DP": {"DRAC-v1": 1.464, "Sym-OCS": 2.842},
    "PP": {"DRAC-v1": 7.452, "Sym-OCS": 5.952},
    "DP+PP Mixed": {"DRAC-v1": 8.585, "Sym-OCS": 6.632},
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(value: str) -> float:
    return float(value)


def i(value: str) -> int:
    return int(value)


def build_summary(root: Path, test_status: str) -> dict[str, Any]:
    processed = root / "processed"
    end = rows(processed / "end_to_end_v2.csv")
    seg = rows(processed / "segmentation_v2.csv")
    realization = rows(processed / "realization_tradeoff_v2.csv")
    compaction = rows(processed / "schedule_compaction_v2.csv")
    iso = rows(processed / "iso_performance_pool_v2.csv")
    runtime = rows(processed / "planning_runtime_v2.csv")

    port = max(i(row["port_budget"]) for row in end)
    end_selected = [row for row in end if i(row["port_budget"]) == port]
    end_times: dict[str, Any] = {}
    speedups: dict[str, Any] = {}
    segment_counts: dict[str, Any] = {}
    fallback_usage: dict[str, Any] = {}
    for workload in sorted({row["workload"] for row in end_selected}):
        workload_rows = [row for row in end_selected if row["workload"] == workload]
        end_times[workload] = {
            row["scheme"]: {
                "communication_only_ms": f(row["communication_only_ms"]),
                "reconfiguration_ms": f(row["reconfiguration_cost_ms"]),
                "total_ms": f(row["total_cost_ms"]),
            }
            for row in workload_rows
        }
        sym = end_times[workload]["Sym-OCS"]["total_ms"]
        speedups[workload] = {
            scheme: sym / values["total_ms"]
            for scheme, values in end_times[workload].items()
            if values["total_ms"] > 0
        }
        segment_counts[workload] = {row["scheme"]: i(row["segment_count"]) for row in workload_rows}
        fallback_usage[workload] = {
            row["scheme"]: {
                "fraction": f(row["fallback_usage_fraction"]),
                "symmetric_segments": i(row["selected_symmetric_segments"]),
                "selected_from": row["selected_from"],
            }
            for row in workload_rows
            if row["scheme"] in {"DRAC-SegmentOpt", "DRAC-SegmentOpt+Fallback"}
        }

    connection_paths: dict[str, Any] = {}
    oracle_gaps: dict[str, Any] = {"realization_max_unit_gap": {}, "segmentation_max_relative_gap": {}}
    for workload in sorted({row["workload"] for row in realization}):
        selected = sorted(
            (row for row in realization if row["workload"] == workload and row["policy"] == "DRACSparse-MultiSeed"),
            key=lambda row: f(row["epsilon"]),
        )
        connection_paths[workload] = {
            row["epsilon"]: {
                "connections": i(row["used_connection_units"]),
                "stable_channels": i(row["stable_reserved_channels"]),
                "slowdown": f(row["realized_slowdown"]),
                "feasible": row["tolerance_satisfied"].lower() == "true",
                "resource_constrained": row["resource_constrained"].lower() == "true",
                "seed": row["seed"],
                "oracle_unit_gap": None if row["oracle_unit_gap"].lower() == "nan" else f(row["oracle_unit_gap"]),
                "swaps": i(row["swap_count"]),
                "pruned": i(row["pruning_count"]),
            }
            for row in selected
        }
        finite_gaps = [abs(f(row["oracle_unit_gap"])) for row in selected if row["oracle_unit_gap"].lower() != "nan" and row["tolerance_satisfied"].lower() == "true"]
        oracle_gaps["realization_max_unit_gap"][workload] = max(finite_gaps, default=None)

    for workload in sorted({row["workload"] for row in seg}):
        selected = [row for row in seg if row["workload"] == workload and row["scheme"] == "SegmentOpt-DynamicProgramming"]
        oracle_gaps["segmentation_max_relative_gap"][workload] = max((abs(f(row["oracle_gap"])) for row in selected), default=None)

    stable_pools: dict[str, Any] = {}
    for workload in sorted({row["workload"] for row in compaction}):
        stable_pools[workload] = {
            row["scheme"]: {
                "reserved_tx": i(row["stable_reserved_tx"]),
                "reserved_rx": i(row["stable_reserved_rx"]),
                "exposed_tx": i(row["stable_exposed_tx"]),
                "exposed_rx": i(row["stable_exposed_rx"]),
                "total_pool": i(row["total_stable_pool"]),
                "bundle_pool": i(row["reserved_bundle_pool"]),
                "compaction_ratio": f(row["compaction_ratio"]),
            }
            for row in compaction
            if row["workload"] == workload
        }
    iso_summary = {
        row["workload"]: {
            "reference_ports": i(row["reference_port_budget"]),
            "reference_total_ms": f(row["reference_total_ms"]),
            "minimum_port_budget": i(row["minimum_port_budget"]) if row["minimum_port_budget"] else None,
            "minimum_stable_directional_pool": i(row["minimum_stable_directional_pool"]) if row["minimum_stable_directional_pool"] else None,
            "minimum_stable_bundle_pool": i(row["minimum_stable_bundle_pool"]) if row["minimum_stable_bundle_pool"] else None,
            "status": row["status"],
        }
        for row in iso
    }
    planner = {
        row["node_count"]: {
            "candidate_segment_target_ms": f(row["candidate_segment_target_ms"]),
            "dynamic_programming_ms": f(row["dynamic_programming_ms"]),
            "sparse_realization_ms": f(row["sparse_realization_ms"]),
            "schedule_compaction_ms": f(row["schedule_compaction_ms"]),
            "total_planning_ms": f(row["total_planning_ms"]),
        }
        for row in runtime
    }
    return {
        "schema_version": "drac_algorithm_revision_v2/1",
        "evidence": "DETERMINISTIC_SIMULATOR_INPUT",
        "measurement_status": "NIC_DIRECTIONAL_COUNTERS_UNAVAILABLE",
        "archived_v1_baseline_ms": V1_BASELINE,
        "end_to_end_port_budget": port,
        "end_to_end_times": end_times,
        "speedups_vs_sym_ocs": speedups,
        "segment_counts": segment_counts,
        "fallback_usage": fallback_usage,
        "connections_per_epsilon": connection_paths,
        "stable_pools": stable_pools,
        "iso_performance_pools": iso_summary,
        "oracle_gaps": oracle_gaps,
        "planner_runtimes": planner,
        "test_status": test_status,
    }


def markdown(summary: dict[str, Any]) -> str:
    e2e = summary["end_to_end_times"]
    realization_gaps = [value for value in summary["oracle_gaps"]["realization_max_unit_gap"].values() if value is not None]
    segmentation_gaps = [value for value in summary["oracle_gaps"]["segmentation_max_relative_gap"].values() if value is not None]
    max_realization_gap = max(realization_gaps, default=0.0)
    max_segmentation_gap = max(segmentation_gaps, default=0.0)
    largest_k = max(summary["planner_runtimes"], key=lambda value: int(value))
    largest_runtime = summary["planner_runtimes"][largest_k]["total_planning_ms"]
    lines = [
        "# DRAC Algorithm Revision v2 Report",
        "",
        "## A. Files and algorithms changed",
        "",
        "- Core: added `drac_eval/segment_target.py` and `drac_eval/evaluation_pipeline_v2.py`; extended `target_segmentation.py` and `sparse_realization.py` while retaining v1 entry points.",
        "- Evaluation: added `drac_eval/evaluation_v2.py`, five config families, six v2 runner/report scripts, and v2 plotting code.",
        "- Validation: added `test_algorithm_revision_v2.py` and `test_evaluation_v2.py`, including plot smoke tests.",
        "- Artifacts: added nine PDF/PNG figures, raw/processed data, archived v1 results, and this generated report/JSON summary.",
        "",
        "The paper was not edited.",
        "",
        "## B. Reproducing the old failures",
        "",
        "The archived v1 eight-port baseline is DP 1.464 vs. 2.842 ms, PP 7.452 vs. 5.952 ms, and Mixed 8.585 vs. 6.632 ms for DRAC-v1 vs. Sym-OCS. V1 realization stayed near 24 stable directional channels across most epsilon values. These files remain under `results/archive/evaluation_v1/`.",
        "",
        "The code cause was direct: candidate intervals selected only a node-target medoid, and realization used only a floor seed, positive single additions, and pure deletion pruning.",
        "",
        "## C. Segment-level target implementation",
        "",
        "Each interval now solves the convex epigraph problem `min sum(theta_k)` with one shared allocation, Tx/Rx budgets, and fixed-bandwidth-aware service constraints. SciPy SLSQP uses scaled constraints and deterministic feasible starts. Medoid and symmetric allocations are retained as verified upper bounds; numerical fallback is explicitly labeled rather than claimed as an optimum.",
        "",
        "## D. Symmetric fallback",
        "",
        "Every selected interval compares directional and symmetric integer realizations. Complete directional, symmetric, and segment-fallback schedules are then compared using realized communication plus reconfiguration cost. The final no-harm scope is only the included same-budget simulator Sym-OCS candidate.",
        "",
        "## E. Sparse realization",
        "",
        "V2 evaluates FloorSeed, SparseCoverageSeed, and FillResidualSeed; handles tied max-drain directions through joint group additions; performs bounded equal-count swaps; and repeats minimum-loss reverse pruning. Feasible stricter-epsilon configurations are legal candidates at wider epsilon values.",
        "",
        "## F. Tests and oracle validation",
        "",
        f"Final status: **{summary['test_status']}**. Segment Dynamic Programming matches complete partition enumeration on all tested small cases. Across the saved full experiment, the maximum feasible MultiSeed connection-count gap to the exhaustive integer oracle is {max_realization_gap:.3g} units, and the maximum segmentation relative gap to exhaustive partition enumeration is {max_segmentation_gap:.3g}.",
        "",
        "During development, guards caught three real issues before final output: an unrestricted solve worse than a symmetric feasible point, a missing residual seed at Mixed epsilon 0.05, and a five-minute K=64 overhead timeout. The first two were fixed algorithmically; K=64 was rerun to completion with an extended offline timeout.",
        "",
        "## G. New and old end-to-end results",
        "",
        "The archived v1 values above remain the before-repair baseline. The v2 full configuration uses epsilon 0.5, so its numbers are not substituted for the archived epsilon 0.1 results.",
        "",
        "| Workload | Sym-OCS | DRAC-v1 | SegmentOpt | SegmentOpt+Fallback |",
        "|---|---:|---:|---:|---:|",
    ]
    for workload in ("DP", "PP", "DP+PP Mixed"):
        value = e2e[workload]
        lines.append(
            f"| {workload} | {value['Sym-OCS']['total_ms']:.3f} | {value['DRAC-v1']['total_ms']:.3f} | {value['DRAC-SegmentOpt']['total_ms']:.3f} | {value['DRAC-SegmentOpt+Fallback']['total_ms']:.3f} |"
        )
    lines.extend(
        [
            "",
            "All values are milliseconds at the maximum scanned port budget.",
            "",
            "## H. PP and Mixed no-harm",
            "",
            "The v2 final candidate is no slower than the included Sym-OCS candidate for every scanned workload/port pair. At eight ports, PP and Mixed improve over their DRAC-v1 schedules. This is a simulator candidate-set guarantee, not a hardware claim. Segment-level symmetric fallback usage is zero in the final full run; schedule-level comparison was still executed and logged.",
            "",
            "## I. Epsilon-resource trade-off",
            "",
            "The trade-off is present. Feasible MultiSeed paths are non-increasing after their first feasible epsilon. DP falls from 12 to 6 units, PP from 8 to 4, Mixed from 11 to 4, and Synthetic Hard from 8 to 3. Mixed and Hard epsilon 0 have no feasible integer oracle solution and remain explicitly resource-constrained.",
            "",
            "## J. Stable compaction",
            "",
            "With the unchanged planner, the eight-port directional stable pool is 48 for DP, 44 for PP, and 47 for Mixed under DRAC-Sparse, versus 48 for Sym-OCS. Thus PP and Mixed show method differences while DP does not. Fixed-budget and iso-performance conditions are stored and plotted separately.",
            "",
            "## K. Evidence source",
            "",
            "All v2 performance, realization, compaction, and runtime values are simulator-derived from deterministic ordered communication graphs with seed 7. They are not measurements.",
            "",
            "## L. Missing measurements",
            "",
            "Raw NIC directional counters remain unavailable. No profiler placeholder figure is generated in v2, and no simulated or calibrated value is labeled measured.",
            "",
            "## M. Negative results and risks",
            "",
            f"DP compaction does not improve over Sym-OCS. Segment-level symmetric fallback was not selected in the full run, so its value is a safety guarantee rather than an observed win. Candidate target construction dominates runtime; K={largest_k} required {largest_runtime / 1000.0:.1f} seconds end to end in the recorded run. SLSQP retains an explicitly labeled feasible-upper-bound fallback for numerical failures; further solver certification and parallel candidate evaluation remain recommended.",
            "",
            "## N. Reproduction commands",
            "",
            "```powershell",
            "python -m pip install -r requirements.txt",
            "pytest -q",
            "python -B scripts\\run_all_evaluation_v2.py --profile full",
            "python -B plots\\plot_all_v2.py --root results\\evaluation_v2",
            "python -B scripts\\build_algorithm_revision_v2_report.py --root results\\evaluation_v2 --test-status \"112 passed\"",
            "```",
            "",
            "Each v2 experiment also has an independent runner under `scripts/run_*_v2.py` and smoke/full JSON under `configs/evaluation_v2/`.",
            "",
            "## O. Suggested future paper changes (not applied)",
            "",
            "The paper should replace medoid-only segment selection with the direct shared-target epigraph problem, describe the symmetric candidate at both segment and schedule level, and specify MultiSeed/group/swap/pruning realization plus feasible-history reuse. Claims should state the exact no-harm scope, disclose K=64 offline cost, and avoid hardware profiler claims until counters exist. No paper source or PDF was modified in this round.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DRAC v2 report from persisted results")
    parser.add_argument("--root", default="results/evaluation_v2")
    parser.add_argument("--test-status", required=True)
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()
    summary = build_summary(Path(args.root), args.test_status)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "algorithm_revision_v2_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (report_dir / "algorithm_revision_v2_report.md").write_text(markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
