"""Target-dictionary service costs and dynamic-programming segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np

from .directional_target import ContinuousTarget, completion_time
from .segment_target import SegmentTarget, medoid_segment_target, solve_segment_continuous_target


@dataclass(frozen=True)
class SegmentChoice:
    start: int
    end: int
    representative: int
    communication_cost: float


@dataclass(frozen=True)
class SegmentationResult:
    segments: tuple[SegmentChoice, ...]
    total_cost: float
    communication_cost: float
    reconfiguration_cost: float
    service_cost: np.ndarray
    candidate_cost: np.ndarray
    candidate_representative: np.ndarray


def build_service_cost_matrix(
    demands: Sequence[np.ndarray],
    targets: Sequence[ContinuousTarget | np.ndarray],
    unit_bandwidth: float,
    fixed_bandwidth: np.ndarray | None = None,
) -> np.ndarray:
    if len(demands) != len(targets):
        raise ValueError("demands and targets must have equal length")
    k_count = len(demands)
    output = np.zeros((k_count, k_count), dtype=float)
    for k, demand in enumerate(demands):
        fixed = np.zeros_like(demand) if fixed_bandwidth is None else fixed_bandwidth
        for h, target in enumerate(targets):
            allocation = target.allocation if isinstance(target, ContinuousTarget) else target
            output[k, h] = completion_time(demand, allocation, unit_bandwidth, fixed)
    return output


def candidate_segment_costs(service_cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if service_cost.ndim != 2 or service_cost.shape[0] != service_cost.shape[1]:
        raise ValueError("service_cost must be square")
    count = service_cost.shape[0]
    costs = np.full((count, count), np.inf, dtype=float)
    representatives = np.full((count, count), -1, dtype=int)
    for start in range(count):
        for end in range(start, count):
            best_cost = float("inf")
            best_h = -1
            for h in range(start, end + 1):
                cost = float(np.sum(service_cost[start : end + 1, h]))
                if cost < best_cost - 1e-12 or (abs(cost - best_cost) <= 1e-12 and h < best_h):
                    best_cost = cost
                    best_h = h
            costs[start, end] = best_cost
            representatives[start, end] = best_h
    return costs, representatives


def segment_target_sequence(
    demands: Sequence[np.ndarray],
    targets: Sequence[ContinuousTarget | np.ndarray],
    unit_bandwidth: float,
    delta: float,
    fixed_bandwidth: np.ndarray | None = None,
) -> SegmentationResult:
    if delta < 0:
        raise ValueError("delta must be non-negative")
    service = build_service_cost_matrix(demands, targets, unit_bandwidth, fixed_bandwidth)
    costs, reps = candidate_segment_costs(service)
    count = len(demands)
    opt = np.full(count + 1, np.inf, dtype=float)
    previous = np.full(count + 1, -1, dtype=int)
    opt[0] = 0.0
    for end_exclusive in range(1, count + 1):
        for q in range(end_exclusive):
            value = opt[q] + costs[q, end_exclusive - 1] + (delta if q > 0 else 0.0)
            if value < opt[end_exclusive] - 1e-12:
                opt[end_exclusive] = value
                previous[end_exclusive] = q
    segments: list[SegmentChoice] = []
    cursor = count
    while cursor > 0:
        q = int(previous[cursor])
        if q < 0:
            raise RuntimeError("segmentation backtracking failed")
        end = cursor - 1
        segments.append(SegmentChoice(q, end, int(reps[q, end]), float(costs[q, end])))
        cursor = q
    segments.reverse()
    communication = float(sum(segment.communication_cost for segment in segments))
    reconfiguration = float(max(0, len(segments) - 1) * delta)
    return SegmentationResult(
        tuple(segments),
        communication + reconfiguration,
        communication,
        reconfiguration,
        service,
        costs,
        reps,
    )


def exhaustive_segmentation_oracle(
    service_cost: np.ndarray, delta: float
) -> tuple[float, tuple[SegmentChoice, ...]]:
    """Enumerate every contiguous partition for small validation cases."""

    costs, reps = candidate_segment_costs(service_cost)
    count = service_cost.shape[0]
    best = float("inf")
    best_segments: tuple[SegmentChoice, ...] = ()
    for cut_count in range(count):
        for cuts in combinations(range(1, count), cut_count):
            boundaries = (0, *cuts, count)
            segments = tuple(
                SegmentChoice(
                    boundaries[idx],
                    boundaries[idx + 1] - 1,
                    int(reps[boundaries[idx], boundaries[idx + 1] - 1]),
                    float(costs[boundaries[idx], boundaries[idx + 1] - 1]),
                )
                for idx in range(len(boundaries) - 1)
            )
            value = sum(segment.communication_cost for segment in segments) + delta * (len(segments) - 1)
            if value < best - 1e-12:
                best = float(value)
                best_segments = segments
    return best, best_segments


@dataclass(frozen=True)
class CandidateTargetTable:
    """All v2 candidate-segment targets and verified costs."""

    directional: tuple[tuple[SegmentTarget | None, ...], ...]
    symmetric: tuple[tuple[SegmentTarget | None, ...], ...]
    medoid_allocation: tuple[tuple[np.ndarray | None, ...], ...]
    directional_cost: np.ndarray
    symmetric_cost: np.ndarray
    medoid_cost: np.ndarray
    medoid_index: np.ndarray


@dataclass(frozen=True)
class SegmentChoiceV2:
    start: int
    end: int
    target: SegmentTarget
    target_type: str
    communication_cost: float
    medoid_index: int
    selection_reason: str


@dataclass(frozen=True)
class SegmentationResultV2:
    segments: tuple[SegmentChoiceV2, ...]
    total_cost: float
    communication_cost: float
    reconfiguration_cost: float
    candidate_targets: CandidateTargetTable
    predecessor: np.ndarray
    method: str


def build_candidate_target_table(
    demands: Sequence[np.ndarray],
    n_tx: np.ndarray,
    n_rx: np.ndarray,
    unit_bandwidth: float,
    fixed_bandwidth: np.ndarray | None = None,
) -> CandidateTargetTable:
    """Solve every O(K^2) candidate interval directly and retain v1 medoids."""

    if not demands:
        raise ValueError("at least one communication node is required")
    count = len(demands)
    fixed = np.zeros_like(demands[0], dtype=float) if fixed_bandwidth is None else np.asarray(fixed_bandwidth, dtype=float)
    node_targets = tuple(
        solve_segment_continuous_target(
            [demand], n_tx, n_rx, unit_bandwidth, fixed, symmetric=False
        )
        for demand in demands
    )
    directional: list[list[SegmentTarget | None]] = [[None] * count for _ in range(count)]
    symmetric: list[list[SegmentTarget | None]] = [[None] * count for _ in range(count)]
    medoid_allocations: list[list[np.ndarray | None]] = [[None] * count for _ in range(count)]
    directional_cost = np.full((count, count), np.inf, dtype=float)
    symmetric_cost = np.full((count, count), np.inf, dtype=float)
    medoid_cost = np.full((count, count), np.inf, dtype=float)
    medoid_index = np.full((count, count), -1, dtype=int)
    for start in range(count):
        for end in range(start, count):
            interval = demands[start : end + 1]
            local_targets = node_targets[start : end + 1]
            allocation, m_cost, local_index = medoid_segment_target(
                interval, local_targets, unit_bandwidth, fixed
            )
            s_target = solve_segment_continuous_target(
                interval, n_tx, n_rx, unit_bandwidth, fixed, symmetric=True
            )
            d_target = solve_segment_continuous_target(
                interval,
                n_tx,
                n_rx,
                unit_bandwidth,
                fixed,
                symmetric=False,
                warm_start_allocation=s_target.allocation,
            )
            scale = max(1.0, abs(m_cost)) if np.isfinite(m_cost) else 1.0
            if np.isfinite(m_cost) and d_target.cost > m_cost + 1e-7 * scale:
                raise AssertionError("segment target is worse than its medoid feasible upper bound")
            if d_target.cost > s_target.cost + 1e-7 * max(1.0, abs(s_target.cost)):
                raise AssertionError("unrestricted target is worse than symmetric target")
            directional[start][end] = d_target
            symmetric[start][end] = s_target
            medoid_allocations[start][end] = allocation
            directional_cost[start, end] = d_target.cost
            symmetric_cost[start, end] = s_target.cost
            medoid_cost[start, end] = m_cost
            medoid_index[start, end] = start + local_index
    return CandidateTargetTable(
        tuple(tuple(row) for row in directional),
        tuple(tuple(row) for row in symmetric),
        tuple(tuple(row) for row in medoid_allocations),
        directional_cost,
        symmetric_cost,
        medoid_cost,
        medoid_index,
    )


def _dynamic_programming_from_costs(costs: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray]:
    if delta < 0:
        raise ValueError("delta must be non-negative")
    count = costs.shape[0]
    opt = np.full(count + 1, np.inf, dtype=float)
    previous = np.full(count + 1, -1, dtype=int)
    opt[0] = 0.0
    for end_exclusive in range(1, count + 1):
        for q in range(end_exclusive):
            value = opt[q] + costs[q, end_exclusive - 1] + (delta if q > 0 else 0.0)
            if not np.isfinite(value):
                continue
            if value < opt[end_exclusive] - 1e-10 or (
                np.isfinite(opt[end_exclusive])
                and abs(value - opt[end_exclusive]) <= 1e-10
                and q < previous[end_exclusive]
            ):
                opt[end_exclusive] = value
                previous[end_exclusive] = q
    return opt, previous


def segment_continuous_sequence(
    demands: Sequence[np.ndarray],
    n_tx: np.ndarray,
    n_rx: np.ndarray,
    unit_bandwidth: float,
    delta: float,
    fixed_bandwidth: np.ndarray | None = None,
    *,
    method: str = "directional",
    candidate_targets: CandidateTargetTable | None = None,
) -> SegmentationResultV2:
    """Segment a target sequence using direct segment costs or the medoid ablation."""

    table = candidate_targets or build_candidate_target_table(
        demands, n_tx, n_rx, unit_bandwidth, fixed_bandwidth
    )
    if method == "directional":
        costs = table.directional_cost
    elif method == "symmetric":
        costs = table.symmetric_cost
    elif method == "medoid":
        costs = table.medoid_cost
    else:
        raise ValueError(f"unknown segmentation method: {method}")
    opt, previous = _dynamic_programming_from_costs(costs, delta)
    segments: list[SegmentChoiceV2] = []
    cursor = len(demands)
    while cursor > 0:
        start = int(previous[cursor])
        if start < 0:
            raise RuntimeError("v2 segmentation backtracking failed")
        end = cursor - 1
        if method == "directional":
            target = table.directional[start][end]
            target_type = "directional"
        elif method == "symmetric":
            target = table.symmetric[start][end]
            target_type = "symmetric"
        else:
            allocation = table.medoid_allocation[start][end]
            if allocation is None:
                raise RuntimeError("missing medoid allocation")
            interval = demands[start : end + 1]
            times = np.asarray(
                [completion_time(d, allocation, unit_bandwidth, fixed_bandwidth) for d in interval]
            )
            target = SegmentTarget(
                allocation,
                times,
                float(np.sum(times)),
                False,
                True,
                "medoid_ablation",
                0,
                0.0,
                f"node-target-{table.medoid_index[start, end]}",
                "MedoidTarget ablation",
            )
            target_type = "medoid"
        if target is None:
            raise RuntimeError("missing candidate segment target")
        segments.append(
            SegmentChoiceV2(
                start,
                end,
                target,
                target_type,
                float(costs[start, end]),
                int(table.medoid_index[start, end]),
                f"{method} candidate selected by Dynamic Programming",
            )
        )
        cursor = start
    segments.reverse()
    communication = float(sum(segment.communication_cost for segment in segments))
    reconfiguration = float(max(0, len(segments) - 1) * delta)
    return SegmentationResultV2(
        tuple(segments),
        communication + reconfiguration,
        communication,
        reconfiguration,
        table,
        previous,
        method,
    )


def exhaustive_partition_oracle_v2(
    candidate_targets: CandidateTargetTable,
    delta: float,
    *,
    method: str = "directional",
) -> tuple[float, tuple[tuple[int, int], ...]]:
    """Enumerate partitions using complete segment-continuous candidate costs."""

    costs = {
        "directional": candidate_targets.directional_cost,
        "symmetric": candidate_targets.symmetric_cost,
        "medoid": candidate_targets.medoid_cost,
    }.get(method)
    if costs is None:
        raise ValueError(f"unknown oracle method: {method}")
    count = costs.shape[0]
    best = float("inf")
    best_boundaries: tuple[tuple[int, int], ...] = ()
    for cut_count in range(count):
        for cuts in combinations(range(1, count), cut_count):
            boundaries = (0, *cuts, count)
            intervals = tuple((boundaries[i], boundaries[i + 1] - 1) for i in range(len(boundaries) - 1))
            value = float(sum(costs[start, end] for start, end in intervals) + delta * (len(intervals) - 1))
            if value < best - 1e-10 or (abs(value - best) <= 1e-10 and intervals < best_boundaries):
                best = value
                best_boundaries = intervals
    return best, best_boundaries
