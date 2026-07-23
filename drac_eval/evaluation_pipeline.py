"""End-to-end paper-aligned DRAC and fair symmetric baseline planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .directional_target import (
    ContinuousTarget,
    completion_time,
    solve_continuous_target,
    solve_symmetric_continuous_target,
)
from .resource_compaction import CompactionResult, compact_schedule
from .sparse_realization import (
    OCSResources,
    RealizationResult,
    realize_drac_sparse,
    realize_sparse_symmetric,
    segment_cost,
)
from .target_segmentation import SegmentationResult, SegmentChoice, segment_target_sequence


@dataclass(frozen=True)
class PlannedSchedule:
    scheme: str
    targets: tuple[ContinuousTarget, ...]
    segmentation: SegmentationResult
    realizations: tuple[RealizationResult, ...]
    compaction: CompactionResult
    communication_cost: float
    reconfiguration_cost: float
    total_cost: float


def _fixed(demands: Sequence[np.ndarray], fixed_bandwidth: np.ndarray | None) -> np.ndarray:
    return np.zeros_like(demands[0]) if fixed_bandwidth is None else np.asarray(fixed_bandwidth, dtype=float)


def plan_reconfigurable_schedule(
    demands: Sequence[np.ndarray],
    resources: OCSResources,
    unit_bandwidth: float,
    delta: float,
    epsilon: float,
    *,
    fixed_bandwidth: np.ndarray | None = None,
    symmetric: bool = False,
) -> PlannedSchedule:
    if not demands:
        raise ValueError("at least one communication node is required")
    model = resources.normalized(demands[0].shape[0])
    fixed = _fixed(demands, fixed_bandwidth)
    solver = solve_symmetric_continuous_target if symmetric else solve_continuous_target
    targets = tuple(
        solver(demand, model.n_tx, model.n_rx, unit_bandwidth, fixed)
        for demand in demands
    )
    segmentation = segment_target_sequence(demands, targets, unit_bandwidth, delta, fixed)
    realizations: list[RealizationResult] = []
    configurations: list[np.ndarray] = []
    for segment in segmentation.segments:
        segment_demands = demands[segment.start : segment.end + 1]
        target = targets[segment.representative].allocation
        realize = realize_sparse_symmetric if symmetric else realize_drac_sparse
        result = realize(
            target,
            segment_demands,
            segment.communication_cost,
            epsilon,
            model,
            unit_bandwidth,
            fixed,
        )
        realizations.append(result)
        configurations.append(result.units)
    compaction = compact_schedule(configurations, model.n_tx, model.n_rx, realizations)
    communication = float(sum(result.cost for result in realizations))
    reconfiguration = float(max(0, len(realizations) - 1) * delta)
    return PlannedSchedule(
        "Sym-OCS" if symmetric else "DRAC",
        targets,
        segmentation,
        tuple(realizations),
        compaction,
        communication,
        reconfiguration,
        communication + reconfiguration,
    )


def plan_static_symmetric(
    demands: Sequence[np.ndarray],
    resources: OCSResources,
    unit_bandwidth: float,
    epsilon: float,
    fixed_bandwidth: np.ndarray | None = None,
) -> PlannedSchedule:
    """Plan one fixed symmetric configuration for the entire recurring sequence."""

    if not demands:
        raise ValueError("at least one communication node is required")
    model = resources.normalized(demands[0].shape[0])
    fixed = _fixed(demands, fixed_bandwidth)
    aggregate = np.sum(np.stack(demands), axis=0)
    target = solve_symmetric_continuous_target(
        aggregate, model.n_tx, model.n_rx, unit_bandwidth, fixed
    )
    logical = float(sum(completion_time(demand, target.allocation, unit_bandwidth, fixed) for demand in demands))
    realization = realize_sparse_symmetric(
        target.allocation,
        demands,
        logical,
        epsilon,
        model,
        unit_bandwidth,
        fixed,
    )
    service = np.array([[logical]], dtype=float)
    segmentation = SegmentationResult(
        (SegmentChoice(0, len(demands) - 1, 0, logical),),
        logical,
        logical,
        0.0,
        service,
        service.copy(),
        np.array([[0]], dtype=int),
    )
    compaction = compact_schedule([realization.units], model.n_tx, model.n_rx, [realization])
    return PlannedSchedule(
        "Static-Sym",
        (target,),
        segmentation,
        (realization,),
        compaction,
        realization.cost,
        0.0,
        realization.cost,
    )


def evaluate_main_schemes(
    demands: Sequence[np.ndarray],
    resources: OCSResources,
    unit_bandwidth: float,
    delta: float,
    epsilon: float,
    fixed_bandwidth: np.ndarray | None = None,
) -> tuple[PlannedSchedule, PlannedSchedule, PlannedSchedule]:
    return (
        plan_static_symmetric(demands, resources, unit_bandwidth, epsilon, fixed_bandwidth),
        plan_reconfigurable_schedule(
            demands,
            resources,
            unit_bandwidth,
            delta,
            epsilon,
            fixed_bandwidth=fixed_bandwidth,
            symmetric=True,
        ),
        plan_reconfigurable_schedule(
            demands,
            resources,
            unit_bandwidth,
            delta,
            epsilon,
            fixed_bandwidth=fixed_bandwidth,
            symmetric=False,
        ),
    )
