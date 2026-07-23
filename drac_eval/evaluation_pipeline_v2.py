"""V2 schedule planning with segment reoptimization and explicit fallback.

Introduced for ``scripts/run_all_evaluation_v2.py``.  The v1 pipeline remains
unchanged so archived results and ablations stay reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .resource_compaction import CompactionResult, compact_schedule
from .sparse_realization import (
    OCSResources,
    RealizationResult,
    realize_drac_sparse_multi_seed,
    realize_sparse_symmetric,
)
from .target_segmentation import (
    CandidateTargetTable,
    SegmentationResultV2,
    build_candidate_target_table,
    segment_continuous_sequence,
)


@dataclass(frozen=True)
class PlannedScheduleV2:
    scheme: str
    segmentation: SegmentationResultV2
    realizations: tuple[RealizationResult, ...]
    selected_target_types: tuple[str, ...]
    fallback_reasons: tuple[str, ...]
    compaction: CompactionResult
    communication_cost: float
    reconfiguration_cost: float
    total_cost: float
    selected_from: str


@dataclass(frozen=True)
class ScheduleCandidatesV2:
    selected: PlannedScheduleV2
    directional: PlannedScheduleV2
    symmetric: PlannedScheduleV2
    segment_fallback: PlannedScheduleV2
    medoid: PlannedScheduleV2
    candidate_targets: CandidateTargetTable


def _peak_key(units: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    tx = units.sum(axis=1)
    rx = units.sum(axis=0)
    return int(np.max(tx, initial=0) + np.max(rx, initial=0)), int(units.sum()), tuple(int(v) for v in units.ravel())


def _choose_segment_realization(
    directional: RealizationResult,
    symmetric: RealizationResult,
) -> tuple[RealizationResult, str, str]:
    candidates = [(directional, "directional"), (symmetric, "symmetric")]
    feasible = [item for item in candidates if item[0].tolerance_satisfied]
    pool = feasible if feasible else candidates
    winner, target_type = min(
        pool,
        key=lambda item: (
            not item[0].tolerance_satisfied,
            item[0].cost,
            item[0].used_units,
            _peak_key(item[0].units),
            item[1],
        ),
    )
    other = symmetric if winner is directional else directional
    if winner.tolerance_satisfied and not other.tolerance_satisfied:
        reason = f"{target_type} selected: other candidate violated epsilon tolerance"
    elif winner.cost < other.cost - 1e-10:
        reason = f"{target_type} selected: lower realized segment cost"
    elif winner.used_units < other.used_units:
        reason = f"{target_type} selected: equal cost with fewer connection units"
    else:
        reason = f"{target_type} selected by deterministic footprint tie-break"
    return winner, target_type, reason


def _realize_schedule(
    segmentation: SegmentationResultV2,
    demands: Sequence[np.ndarray],
    resources: OCSResources,
    unit_bandwidth: float,
    delta: float,
    epsilon: float,
    fixed_bandwidth: np.ndarray,
    *,
    mode: str,
) -> PlannedScheduleV2:
    realizations: list[RealizationResult] = []
    selected_types: list[str] = []
    reasons: list[str] = []
    table = segmentation.candidate_targets
    for segment in segmentation.segments:
        interval = demands[segment.start : segment.end + 1]
        directional_target = table.directional[segment.start][segment.end]
        symmetric_target = table.symmetric[segment.start][segment.end]
        if directional_target is None or symmetric_target is None:
            raise RuntimeError("candidate target table is incomplete")
        if mode == "directional" or mode == "medoid":
            target = segment.target if mode == "medoid" else directional_target
            result = realize_drac_sparse_multi_seed(
                target.allocation,
                interval,
                target.cost,
                epsilon,
                resources,
                unit_bandwidth,
                fixed_bandwidth,
            )
            target_type = "medoid" if mode == "medoid" else "directional"
            reason = f"{target_type} schedule candidate"
        elif mode == "symmetric":
            result = realize_sparse_symmetric(
                symmetric_target.allocation,
                interval,
                symmetric_target.cost,
                epsilon,
                resources,
                unit_bandwidth,
                fixed_bandwidth,
            )
            target_type = "symmetric"
            reason = "symmetric schedule candidate"
        elif mode == "fallback":
            directional_result = realize_drac_sparse_multi_seed(
                directional_target.allocation,
                interval,
                directional_target.cost,
                epsilon,
                resources,
                unit_bandwidth,
                fixed_bandwidth,
            )
            symmetric_result = realize_sparse_symmetric(
                symmetric_target.allocation,
                interval,
                symmetric_target.cost,
                epsilon,
                resources,
                unit_bandwidth,
                fixed_bandwidth,
            )
            result, target_type, reason = _choose_segment_realization(
                directional_result, symmetric_result
            )
        else:
            raise ValueError(f"unknown realization mode: {mode}")
        realizations.append(result)
        selected_types.append(target_type)
        reasons.append(reason)
    compaction = compact_schedule(
        [result.units for result in realizations], resources.n_tx, resources.n_rx, realizations
    )
    communication = float(sum(result.cost for result in realizations))
    reconfiguration = float(max(0, len(realizations) - 1) * delta)
    labels = {
        "directional": "DRAC-SegmentOpt",
        "symmetric": "SymmetricFallbackSchedule",
        "fallback": "DRAC-SegmentOpt+Fallback",
        "medoid": "DRAC-v1-MedoidTarget",
    }
    return PlannedScheduleV2(
        labels[mode],
        segmentation,
        tuple(realizations),
        tuple(selected_types),
        tuple(reasons),
        compaction,
        communication,
        reconfiguration,
        communication + reconfiguration,
        mode,
    )


def _schedule_key(schedule: PlannedScheduleV2) -> tuple[object, ...]:
    return (
        schedule.total_cost,
        schedule.compaction.total_stable_directional_pool,
        sum(result.used_units for result in schedule.realizations),
        len(schedule.realizations),
        tuple(tuple(int(v) for v in result.units.ravel()) for result in schedule.realizations),
        schedule.scheme,
    )


def plan_schedule_candidates_v2(
    demands: Sequence[np.ndarray],
    resources: OCSResources,
    unit_bandwidth: float,
    delta: float,
    epsilon: float,
    fixed_bandwidth: np.ndarray | None = None,
) -> ScheduleCandidatesV2:
    if not demands:
        raise ValueError("at least one communication node is required")
    model = resources.normalized(demands[0].shape[0])
    fixed = np.zeros_like(demands[0], dtype=float) if fixed_bandwidth is None else np.asarray(fixed_bandwidth, dtype=float)
    table = build_candidate_target_table(
        demands, model.n_tx, model.n_rx, unit_bandwidth, fixed
    )
    directional_segmentation = segment_continuous_sequence(
        demands,
        model.n_tx,
        model.n_rx,
        unit_bandwidth,
        delta,
        fixed,
        method="directional",
        candidate_targets=table,
    )
    symmetric_segmentation = segment_continuous_sequence(
        demands,
        model.n_tx,
        model.n_rx,
        unit_bandwidth,
        delta,
        fixed,
        method="symmetric",
        candidate_targets=table,
    )
    medoid_segmentation = segment_continuous_sequence(
        demands,
        model.n_tx,
        model.n_rx,
        unit_bandwidth,
        delta,
        fixed,
        method="medoid",
        candidate_targets=table,
    )
    directional = _realize_schedule(
        directional_segmentation, demands, model, unit_bandwidth, delta, epsilon, fixed, mode="directional"
    )
    symmetric = _realize_schedule(
        symmetric_segmentation, demands, model, unit_bandwidth, delta, epsilon, fixed, mode="symmetric"
    )
    fallback = _realize_schedule(
        directional_segmentation, demands, model, unit_bandwidth, delta, epsilon, fixed, mode="fallback"
    )
    medoid = _realize_schedule(
        medoid_segmentation, demands, model, unit_bandwidth, delta, epsilon, fixed, mode="medoid"
    )
    selected = min((directional, symmetric, fallback), key=_schedule_key)
    selected = replace(
        selected,
        scheme="DRAC-SegmentOpt+Fallback",
        selected_from=f"schedule-level fallback selected {selected.selected_from}",
    )
    if selected.total_cost > symmetric.total_cost + 1e-8 * max(1.0, symmetric.total_cost):
        raise AssertionError("schedule-level fallback violated its Sym-OCS no-harm candidate bound")
    return ScheduleCandidatesV2(selected, directional, symmetric, fallback, medoid, table)

