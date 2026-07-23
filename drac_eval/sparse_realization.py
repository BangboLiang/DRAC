"""Target-bounded sparse integer OCS realization and comparison policies."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np

from .directional_target import completion_time


EPS = 1e-12


def _within_tolerance(cost: float, tolerance: float) -> bool:
    numerical_slack = max(EPS, 1e-9 * max(1.0, abs(tolerance)))
    return bool(cost <= tolerance + numerical_slack)


@dataclass(frozen=True)
class OCSResources:
    n_tx: np.ndarray
    n_rx: np.ndarray
    total_units: int | None = None
    reachable: np.ndarray | None = None

    def normalized(self, size: int) -> "OCSResources":
        tx = np.asarray(self.n_tx, dtype=int)
        rx = np.asarray(self.n_rx, dtype=int)
        if tx.shape != (size,) or rx.shape != (size,) or np.any(tx < 0) or np.any(rx < 0):
            raise ValueError("invalid Tx/Rx inventory")
        reachable = (
            ~np.eye(size, dtype=bool)
            if self.reachable is None
            else np.asarray(self.reachable, dtype=bool)
        )
        if reachable.shape != (size, size):
            raise ValueError("reachability shape mismatch")
        reachable = reachable.copy()
        np.fill_diagonal(reachable, False)
        return OCSResources(tx, rx, self.total_units, reachable)


@dataclass(frozen=True)
class DirectionRequest:
    src: int
    dst: int
    marginal_gain: float
    aggregate_demand: float


@dataclass(frozen=True)
class RealizationResult:
    policy: str
    units: np.ndarray
    cost: float
    logical_cost: float
    tolerance_cost: float
    tolerance_satisfied: bool
    resource_constrained: bool
    used_units: int
    additions: int
    pruned: int
    requests: tuple[DirectionRequest, ...]
    seed: str = "legacy"
    group_additions: int = 0
    swaps: int = 0
    swap_gain: float = 0.0
    reused_history: bool = False
    events: tuple[str, ...] = ()


def validate_integer_configuration(units: np.ndarray, resources: OCSResources) -> None:
    units = np.asarray(units)
    model = resources.normalized(units.shape[0])
    if units.ndim != 2 or units.shape[0] != units.shape[1]:
        raise ValueError("integer configuration must be square")
    if not np.issubdtype(units.dtype, np.integer) or np.any(units < 0):
        raise ValueError("connection units must be non-negative integers")
    if np.any(units[~model.reachable] != 0):
        raise ValueError("configuration uses an unreachable direction")
    if np.any(units.sum(axis=1) > model.n_tx) or np.any(units.sum(axis=0) > model.n_rx):
        raise ValueError("configuration violates endpoint Tx/Rx inventory")
    if model.total_units is not None and int(units.sum()) > int(model.total_units):
        raise ValueError("configuration violates global unit inventory")


def segment_cost(
    demands: Sequence[np.ndarray],
    units: np.ndarray,
    unit_bandwidth: float,
    fixed_bandwidth: np.ndarray | None = None,
) -> float:
    return float(
        sum(
            completion_time(demand, units, unit_bandwidth, fixed_bandwidth)
            for demand in demands
        )
    )


def _can_add(units: np.ndarray, src: int, dst: int, resources: OCSResources) -> bool:
    return bool(
        resources.reachable[src, dst]
        and units[src].sum() < resources.n_tx[src]
        and units[:, dst].sum() < resources.n_rx[dst]
        and (resources.total_units is None or units.sum() < resources.total_units)
    )


def _positive_without_fixed(
    demands: Sequence[np.ndarray], fixed_bandwidth: np.ndarray
) -> np.ndarray:
    positive = np.any(np.stack([demand > 0 for demand in demands]), axis=0)
    return positive & (fixed_bandwidth <= 0)


def _marginal_gain(
    demands: Sequence[np.ndarray],
    units: np.ndarray,
    src: int,
    dst: int,
    unit_bandwidth: float,
    fixed_bandwidth: np.ndarray,
    current_cost: float | None = None,
) -> float:
    before = segment_cost(demands, units, unit_bandwidth, fixed_bandwidth) if current_cost is None else current_cost
    trial = units.copy()
    trial[src, dst] += 1
    after = segment_cost(demands, trial, unit_bandwidth, fixed_bandwidth)
    if np.isinf(before) and np.isfinite(after):
        return float("inf")
    if np.isinf(before) and np.isinf(after):
        return 0.0
    return float(before - after)


def _requests(
    demands: Sequence[np.ndarray],
    units: np.ndarray,
    resources: OCSResources,
    unit_bandwidth: float,
    fixed_bandwidth: np.ndarray,
) -> tuple[DirectionRequest, ...]:
    current = segment_cost(demands, units, unit_bandwidth, fixed_bandwidth)
    aggregate = np.sum(np.stack(demands), axis=0)
    rows: list[DirectionRequest] = []
    for src, dst in zip(*np.where(aggregate > 0)):
        if not resources.reachable[src, dst]:
            continue
        gain = _marginal_gain(
            demands, units, int(src), int(dst), unit_bandwidth, fixed_bandwidth, current
        )
        if gain > EPS:
            rows.append(DirectionRequest(int(src), int(dst), gain, float(aggregate[src, dst])))
    rows.sort(key=lambda item: (-item.marginal_gain, -item.aggregate_demand, item.src, item.dst))
    return tuple(rows)


def _floor_feasible(target: np.ndarray, resources: OCSResources) -> np.ndarray:
    units = np.floor(np.maximum(target, 0.0) + EPS).astype(int)
    units[~resources.reachable] = 0
    validate_integer_configuration(units, resources)
    return units


def _ensure_serviceable(
    demands: Sequence[np.ndarray],
    units: np.ndarray,
    resources: OCSResources,
    fixed_bandwidth: np.ndarray,
) -> tuple[int, bool]:
    required = _positive_without_fixed(demands, fixed_bandwidth)
    missing = [(int(i), int(j)) for i, j in zip(*np.where(required & (units == 0)))]
    aggregate = np.sum(np.stack(demands), axis=0)
    missing.sort(key=lambda pair: (-float(aggregate[pair]), pair))
    added = 0
    complete = True
    for src, dst in missing:
        if _can_add(units, src, dst, resources):
            units[src, dst] += 1
            added += 1
        else:
            complete = False
    return added, complete


def _is_serviceable(
    demands: Sequence[np.ndarray], units: np.ndarray, fixed_bandwidth: np.ndarray
) -> bool:
    required = _positive_without_fixed(demands, fixed_bandwidth)
    return bool(np.all(units[required] > 0))


def _coverage_seed(
    demands: Sequence[np.ndarray], resources: OCSResources, fixed_bandwidth: np.ndarray
) -> tuple[np.ndarray, int, bool]:
    units = np.zeros((len(resources.n_tx), len(resources.n_tx)), dtype=int)
    additions, complete = _ensure_serviceable(demands, units, resources, fixed_bandwidth)
    return units, additions, complete


def _peak_footprint(units: np.ndarray) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    tx = tuple(int(value) for value in units.sum(axis=1))
    rx = tuple(int(value) for value in units.sum(axis=0))
    return max(tx, default=0) + max(rx, default=0), tx, rx


def _trial_add_group(
    units: np.ndarray,
    group: tuple[tuple[int, int], ...],
    resources: OCSResources,
) -> np.ndarray | None:
    trial = units.copy()
    for src, dst in group:
        trial[src, dst] += 1
    try:
        validate_integer_configuration(trial, resources)
    except ValueError:
        return None
    return trial


def _bottleneck_groups(
    demands: Sequence[np.ndarray],
    units: np.ndarray,
    resources: OCSResources,
    unit_bandwidth: float,
    fixed_bandwidth: np.ndarray,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Return tied max-drain direction groups for bounded joint look-ahead."""

    capacity = fixed_bandwidth + unit_bandwidth * units
    groups: set[tuple[tuple[int, int], ...]] = set()
    for demand in demands:
        positive = demand > 0
        if not np.any(positive) or np.any(capacity[positive] <= 0):
            continue
        drain = np.zeros_like(demand, dtype=float)
        drain[positive] = demand[positive] / capacity[positive]
        maximum = float(np.max(drain))
        tolerance = max(1e-10, 1e-8 * max(1.0, maximum))
        tied = tuple(
            sorted(
                (int(src), int(dst))
                for src, dst in zip(*np.where(positive & (np.abs(drain - maximum) <= tolerance)))
            )
        )
        if len(tied) > 1 and _trial_add_group(units, tied, resources) is not None:
            groups.add(tied)
    return tuple(sorted(groups, key=lambda group: (len(group), group)))


def _best_addition_action(
    demands: Sequence[np.ndarray],
    units: np.ndarray,
    current: float,
    resources: OCSResources,
    unit_bandwidth: float,
    fixed_bandwidth: np.ndarray,
) -> tuple[np.ndarray, int, bool, float] | None:
    aggregate = np.sum(np.stack(demands), axis=0)
    candidates: list[tuple[float, float, int, tuple[tuple[int, int], ...], np.ndarray, bool]] = []
    for src, dst in zip(*np.where(aggregate > 0)):
        src_i, dst_i = int(src), int(dst)
        if not _can_add(units, src_i, dst_i, resources):
            continue
        group = ((src_i, dst_i),)
        trial = _trial_add_group(units, group, resources)
        assert trial is not None
        after = segment_cost(demands, trial, unit_bandwidth, fixed_bandwidth)
        gain = current - after
        if gain > EPS:
            candidates.append((gain, gain, 1, group, trial, False))
    for group in _bottleneck_groups(demands, units, resources, unit_bandwidth, fixed_bandwidth):
        trial = _trial_add_group(units, group, resources)
        if trial is None:
            continue
        after = segment_cost(demands, trial, unit_bandwidth, fixed_bandwidth)
        gain = current - after
        if gain > EPS:
            candidates.append((gain / len(group), gain, len(group), group, trial, True))
    if not candidates:
        return None
    _, gain, count, group, trial, is_group = max(
        candidates, key=lambda item: (item[0], item[1], -item[2], tuple((-i, -j) for i, j in item[3]))
    )
    return trial, count, is_group, float(gain)


def _reverse_prune(
    demands: Sequence[np.ndarray],
    units: np.ndarray,
    current: float,
    tolerance: float,
    resources: OCSResources,
    unit_bandwidth: float,
    fixed_bandwidth: np.ndarray,
) -> tuple[np.ndarray, float, int, list[str]]:
    pruned = 0
    events: list[str] = []
    while True:
        removals: list[tuple[float, int, int, float, np.ndarray]] = []
        for src, dst in zip(*np.where(units > 0)):
            trial = units.copy()
            trial[src, dst] -= 1
            if not _is_serviceable(demands, trial, fixed_bandwidth):
                continue
            trial_cost = segment_cost(demands, trial, unit_bandwidth, fixed_bandwidth)
            if _within_tolerance(trial_cost, tolerance):
                removals.append((trial_cost - current, int(src), int(dst), trial_cost, trial))
        if not removals:
            break
        loss, src, dst, current, units = min(removals, key=lambda item: (item[0], item[1], item[2]))
        pruned += 1
        events.append(f"prune {src}->{dst} loss={loss:.12g}")
    validate_integer_configuration(units, resources)
    return units, float(current), pruned, events


def _swap_local_search(
    demands: Sequence[np.ndarray],
    units: np.ndarray,
    current: float,
    tolerance: float,
    resources: OCSResources,
    unit_bandwidth: float,
    fixed_bandwidth: np.ndarray,
    *,
    max_swaps: int = 64,
) -> tuple[np.ndarray, float, int, float, list[str]]:
    aggregate = np.sum(np.stack(demands), axis=0)
    additions = tuple(sorted((int(i), int(j)) for i, j in zip(*np.where(aggregate > 0))))
    swaps = 0
    total_gain = 0.0
    events: list[str] = []
    while swaps < max_swaps:
        before_footprint = _peak_footprint(units)
        accepted = None
        for src, dst in sorted((int(i), int(j)) for i, j in zip(*np.where(units > 0))):
            for add_src, add_dst in additions:
                if (src, dst) == (add_src, add_dst):
                    continue
                trial = units.copy()
                trial[src, dst] -= 1
                trial[add_src, add_dst] += 1
                try:
                    validate_integer_configuration(trial, resources)
                except ValueError:
                    continue
                if not _is_serviceable(demands, trial, fixed_bandwidth):
                    continue
                trial_cost = segment_cost(demands, trial, unit_bandwidth, fixed_bandwidth)
                if not _within_tolerance(trial_cost, tolerance):
                    continue
                after_footprint = _peak_footprint(trial)
                footprint_improved = after_footprint < before_footprint
                cost_improved = trial_cost < current - EPS
                if footprint_improved or (after_footprint == before_footprint and cost_improved):
                    accepted = (trial, float(trial_cost), src, dst, add_src, add_dst, before_footprint, after_footprint)
                    break
            if accepted is not None:
                break
        if accepted is None:
            break
        trial, trial_cost, src, dst, add_src, add_dst, before_fp, after_fp = accepted
        gain = current - trial_cost
        units, current = trial, trial_cost
        swaps += 1
        total_gain += gain
        events.append(
            f"swap {src}->{dst} to {add_src}->{add_dst} gain={gain:.12g} footprint={before_fp[0]}->{after_fp[0]}"
        )
    return units, float(current), swaps, float(total_gain), events


def _realize_from_seed(
    policy: str,
    seed: str,
    initial_units: np.ndarray,
    serviceable: bool,
    initial_additions: int,
    demands: Sequence[np.ndarray],
    logical_cost: float,
    epsilon: float,
    resources: OCSResources,
    unit_bandwidth: float,
    fixed_bandwidth: np.ndarray,
) -> RealizationResult:
    units = initial_units.copy()
    tolerance = float((1.0 + epsilon) * logical_cost)
    current = segment_cost(demands, units, unit_bandwidth, fixed_bandwidth)
    additions = int(initial_additions)
    group_additions = 0
    events: list[str] = [f"seed={seed} units={int(units.sum())} cost={current:.12g}"]
    while serviceable and not _within_tolerance(current, tolerance):
        action = _best_addition_action(
            demands, units, current, resources, unit_bandwidth, fixed_bandwidth
        )
        if action is None:
            events.append("addition-stop=no-positive-single-or-group-gain")
            break
        units, count, is_group, gain = action
        additions += count
        group_additions += int(is_group)
        current = segment_cost(demands, units, unit_bandwidth, fixed_bandwidth)
        events.append(f"add {'group' if is_group else 'single'} size={count} gain={gain:.12g}")
    pruned = 0
    swaps = 0
    swap_gain = 0.0
    if serviceable and _within_tolerance(current, tolerance):
        units, current, first_pruned, prune_events = _reverse_prune(
            demands, units, current, tolerance, resources, unit_bandwidth, fixed_bandwidth
        )
        pruned += first_pruned
        events.extend(prune_events)
        units, current, swaps, swap_gain, swap_events = _swap_local_search(
            demands, units, current, tolerance, resources, unit_bandwidth, fixed_bandwidth
        )
        events.extend(swap_events)
        units, current, second_pruned, prune_events = _reverse_prune(
            demands, units, current, tolerance, resources, unit_bandwidth, fixed_bandwidth
        )
        pruned += second_pruned
        events.extend(prune_events)
    validate_integer_configuration(units, resources)
    satisfied = bool(serviceable and _is_serviceable(demands, units, fixed_bandwidth) and _within_tolerance(current, tolerance))
    requests = () if satisfied else _requests(demands, units, resources, unit_bandwidth, fixed_bandwidth)
    return RealizationResult(
        policy,
        units,
        float(current),
        float(logical_cost),
        tolerance,
        satisfied,
        not satisfied,
        int(units.sum()),
        additions,
        pruned,
        requests,
        seed,
        group_additions,
        swaps,
        swap_gain,
        False,
        tuple(events),
    )


def realize_drac_sparse_floor_seed(
    target: np.ndarray,
    demands: Sequence[np.ndarray],
    logical_cost: float,
    epsilon: float,
    resources: OCSResources,
    unit_bandwidth: float,
    fixed_bandwidth: np.ndarray | None = None,
) -> RealizationResult:
    if epsilon < 0 or logical_cost < 0:
        raise ValueError("epsilon and logical_cost must be non-negative")
    target = np.asarray(target, dtype=float)
    model = resources.normalized(target.shape[0])
    fixed = np.zeros_like(target) if fixed_bandwidth is None else np.asarray(fixed_bandwidth, dtype=float)
    units = _floor_feasible(target, model)
    additions, serviceable = _ensure_serviceable(demands, units, model, fixed)
    return _realize_from_seed(
        "DRACSparse-FloorSeed",
        "FloorSeed",
        units,
        serviceable,
        additions,
        demands,
        logical_cost,
        epsilon,
        model,
        unit_bandwidth,
        fixed,
    )


def realize_drac_sparse_coverage_seed(
    target: np.ndarray,
    demands: Sequence[np.ndarray],
    logical_cost: float,
    epsilon: float,
    resources: OCSResources,
    unit_bandwidth: float,
    fixed_bandwidth: np.ndarray | None = None,
) -> RealizationResult:
    if epsilon < 0 or logical_cost < 0:
        raise ValueError("epsilon and logical_cost must be non-negative")
    target = np.asarray(target, dtype=float)
    model = resources.normalized(target.shape[0])
    fixed = np.zeros_like(target) if fixed_bandwidth is None else np.asarray(fixed_bandwidth, dtype=float)
    units, additions, serviceable = _coverage_seed(demands, model, fixed)
    return _realize_from_seed(
        "DRACSparse-CoverageSeed",
        "SparseCoverageSeed",
        units,
        serviceable,
        additions,
        demands,
        logical_cost,
        epsilon,
        model,
        unit_bandwidth,
        fixed,
    )


def _reuse_historical_candidate(
    units: np.ndarray,
    demands: Sequence[np.ndarray],
    logical_cost: float,
    epsilon: float,
    resources: OCSResources,
    unit_bandwidth: float,
    fixed_bandwidth: np.ndarray,
) -> RealizationResult | None:
    candidate = np.asarray(units, dtype=int).copy()
    try:
        validate_integer_configuration(candidate, resources)
    except ValueError:
        return None
    cost = segment_cost(demands, candidate, unit_bandwidth, fixed_bandwidth)
    tolerance = float((1.0 + epsilon) * logical_cost)
    if not _is_serviceable(demands, candidate, fixed_bandwidth) or not _within_tolerance(cost, tolerance):
        return None
    candidate, cost, pruned, events = _reverse_prune(
        demands, candidate, cost, tolerance, resources, unit_bandwidth, fixed_bandwidth
    )
    return RealizationResult(
        "DRACSparse-History",
        candidate,
        cost,
        logical_cost,
        tolerance,
        True,
        False,
        int(candidate.sum()),
        0,
        pruned,
        (),
        "HistoricalCandidate",
        0,
        0,
        0.0,
        True,
        tuple(("reused stricter-epsilon configuration", *events)),
    )


def realize_drac_sparse_multi_seed(
    target: np.ndarray,
    demands: Sequence[np.ndarray],
    logical_cost: float,
    epsilon: float,
    resources: OCSResources,
    unit_bandwidth: float,
    fixed_bandwidth: np.ndarray | None = None,
    *,
    historical_units: Sequence[np.ndarray] = (),
) -> RealizationResult:
    target = np.asarray(target, dtype=float)
    model = resources.normalized(target.shape[0])
    fixed = np.zeros_like(target) if fixed_bandwidth is None else np.asarray(fixed_bandwidth, dtype=float)
    floor_candidate = realize_drac_sparse_floor_seed(
        target, demands, logical_cost, epsilon, model, unit_bandwidth, fixed
    )
    coverage_candidate = realize_drac_sparse_coverage_seed(
        target, demands, logical_cost, epsilon, model, unit_bandwidth, fixed
    )
    filled = realize_fill_all_residual(
        target, demands, logical_cost, epsilon, model, unit_bandwidth, fixed
    )
    residual_candidate = _realize_from_seed(
        "DRACSparse-ResidualSeed",
        "FillResidualSeed",
        filled.units,
        _is_serviceable(demands, filled.units, fixed),
        filled.additions,
        demands,
        logical_cost,
        epsilon,
        model,
        unit_bandwidth,
        fixed,
    )
    candidates = [floor_candidate, coverage_candidate, residual_candidate]
    for units in historical_units:
        reused = _reuse_historical_candidate(
            units, demands, logical_cost, epsilon, model, unit_bandwidth, fixed
        )
        if reused is not None:
            candidates.append(reused)
    feasible = [candidate for candidate in candidates if candidate.tolerance_satisfied]
    pool = feasible if feasible else candidates
    winner = min(
        pool,
        key=lambda candidate: (
            not candidate.tolerance_satisfied,
            candidate.used_units,
            _peak_footprint(candidate.units),
            candidate.cost,
            tuple(int(value) for value in candidate.units.ravel()),
            candidate.seed,
        ),
    )
    events = tuple((*winner.events, f"MultiSeed selected {winner.seed} from {len(candidates)} candidates"))
    return RealizationResult(
        "DRACSparse-MultiSeed",
        winner.units.copy(),
        winner.cost,
        winner.logical_cost,
        winner.tolerance_cost,
        winner.tolerance_satisfied,
        winner.resource_constrained,
        winner.used_units,
        winner.additions,
        winner.pruned,
        winner.requests,
        winner.seed,
        winner.group_additions,
        winner.swaps,
        winner.swap_gain,
        winner.reused_history,
        events,
    )


def realize_drac_sparse(
    target: np.ndarray,
    demands: Sequence[np.ndarray],
    logical_cost: float,
    epsilon: float,
    resources: OCSResources,
    unit_bandwidth: float,
    fixed_bandwidth: np.ndarray | None = None,
) -> RealizationResult:
    """V1 floor/add/prune implementation retained for baseline reproducibility."""

    if epsilon < 0 or logical_cost < 0:
        raise ValueError("epsilon and logical_cost must be non-negative")
    target = np.asarray(target, dtype=float)
    model = resources.normalized(target.shape[0])
    fixed = np.zeros_like(target) if fixed_bandwidth is None else np.asarray(fixed_bandwidth, dtype=float)
    units = _floor_feasible(target, model)
    additions, serviceable = _ensure_serviceable(demands, units, model, fixed)
    tolerance = float((1.0 + epsilon) * logical_cost)
    current = segment_cost(demands, units, unit_bandwidth, fixed)
    while not _within_tolerance(current, tolerance):
        best: tuple[float, int, int] | None = None
        aggregate = np.sum(np.stack(demands), axis=0)
        for src, dst in zip(*np.where(aggregate > 0)):
            src_i, dst_i = int(src), int(dst)
            if not _can_add(units, src_i, dst_i, model):
                continue
            gain = _marginal_gain(demands, units, src_i, dst_i, unit_bandwidth, fixed, current)
            candidate = (gain, src_i, dst_i)
            if best is None or candidate[0] > best[0] + EPS or (
                abs(candidate[0] - best[0]) <= EPS and candidate[1:] < best[1:]
            ):
                best = candidate
        if best is None or best[0] <= EPS:
            break
        units[best[1], best[2]] += 1
        additions += 1
        current = segment_cost(demands, units, unit_bandwidth, fixed)
    pruned = 0
    if _within_tolerance(current, tolerance):
        while True:
            candidates: list[tuple[float, int, int, float]] = []
            for src, dst in zip(*np.where(units > 0)):
                trial = units.copy()
                trial[src, dst] -= 1
                trial_cost = segment_cost(demands, trial, unit_bandwidth, fixed)
                if _within_tolerance(trial_cost, tolerance):
                    candidates.append((trial_cost - current, int(src), int(dst), trial_cost))
            if not candidates:
                break
            _, src, dst, current = min(candidates, key=lambda item: (item[0], item[1], item[2]))
            units[src, dst] -= 1
            pruned += 1
    validate_integer_configuration(units, model)
    satisfied = bool(serviceable and _within_tolerance(current, tolerance))
    return RealizationResult(
        "DRACSparse-v1",
        units,
        float(current),
        float(logical_cost),
        tolerance,
        satisfied,
        not satisfied,
        int(units.sum()),
        additions,
        pruned,
        () if satisfied else _requests(demands, units, model, unit_bandwidth, fixed),
        "FloorSeed-v1",
    )


def realize_floor_only(
    target: np.ndarray,
    demands: Sequence[np.ndarray],
    logical_cost: float,
    epsilon: float,
    resources: OCSResources,
    unit_bandwidth: float,
    fixed_bandwidth: np.ndarray | None = None,
) -> RealizationResult:
    model = resources.normalized(target.shape[0])
    fixed = np.zeros_like(target) if fixed_bandwidth is None else np.asarray(fixed_bandwidth, dtype=float)
    units = _floor_feasible(target, model)
    cost = segment_cost(demands, units, unit_bandwidth, fixed)
    tolerance = (1.0 + epsilon) * logical_cost
    satisfied = _within_tolerance(cost, tolerance)
    return RealizationResult("FloorOnly", units, cost, logical_cost, tolerance, satisfied, not satisfied, int(units.sum()), 0, 0, () if satisfied else _requests(demands, units, model, unit_bandwidth, fixed))


def realize_nearest_rounding(
    target: np.ndarray,
    demands: Sequence[np.ndarray],
    logical_cost: float,
    epsilon: float,
    resources: OCSResources,
    unit_bandwidth: float,
    fixed_bandwidth: np.ndarray | None = None,
) -> RealizationResult:
    model = resources.normalized(target.shape[0])
    fixed = np.zeros_like(target) if fixed_bandwidth is None else np.asarray(fixed_bandwidth, dtype=float)
    units = np.rint(np.maximum(target, 0.0)).astype(int)
    units[~model.reachable] = 0
    while (
        np.any(units.sum(axis=1) > model.n_tx)
        or np.any(units.sum(axis=0) > model.n_rx)
        or (model.total_units is not None and units.sum() > model.total_units)
    ):
        positive = [(float(target[i, j]), int(i), int(j)) for i, j in zip(*np.where(units > 0))]
        if not positive:
            break
        _, src, dst = min(positive)
        units[src, dst] -= 1
    validate_integer_configuration(units, model)
    cost = segment_cost(demands, units, unit_bandwidth, fixed)
    tolerance = (1.0 + epsilon) * logical_cost
    satisfied = _within_tolerance(cost, tolerance)
    return RealizationResult("NearestRounding", units, cost, logical_cost, tolerance, satisfied, not satisfied, int(units.sum()), 0, 0, () if satisfied else _requests(demands, units, model, unit_bandwidth, fixed))


def realize_fill_all_residual(
    target: np.ndarray,
    demands: Sequence[np.ndarray],
    logical_cost: float,
    epsilon: float,
    resources: OCSResources,
    unit_bandwidth: float,
    fixed_bandwidth: np.ndarray | None = None,
) -> RealizationResult:
    model = resources.normalized(target.shape[0])
    fixed = np.zeros_like(target) if fixed_bandwidth is None else np.asarray(fixed_bandwidth, dtype=float)
    units = _floor_feasible(target, model)
    additions, _ = _ensure_serviceable(demands, units, model, fixed)
    while True:
        candidates = []
        for src, dst in zip(*np.where(target - units > EPS)):
            if _can_add(units, int(src), int(dst), model):
                candidates.append((float(target[src, dst] - units[src, dst]), int(src), int(dst)))
        if not candidates:
            break
        _, src, dst = max(candidates, key=lambda item: (item[0], -item[1], -item[2]))
        units[src, dst] += 1
        additions += 1
    cost = segment_cost(demands, units, unit_bandwidth, fixed)
    tolerance = (1.0 + epsilon) * logical_cost
    satisfied = _within_tolerance(cost, tolerance)
    return RealizationResult("FillAllResidual", units, cost, logical_cost, tolerance, satisfied, not satisfied, int(units.sum()), additions, 0, () if satisfied else _requests(demands, units, model, unit_bandwidth, fixed))


def realize_sparse_symmetric(
    target: np.ndarray,
    demands: Sequence[np.ndarray],
    logical_cost: float,
    epsilon: float,
    resources: OCSResources,
    unit_bandwidth: float,
    fixed_bandwidth: np.ndarray | None = None,
) -> RealizationResult:
    model = resources.normalized(target.shape[0])
    fixed = np.zeros_like(target) if fixed_bandwidth is None else np.asarray(fixed_bandwidth, dtype=float)
    symmetric_target = np.maximum(target, target.T)
    units = np.floor(symmetric_target + EPS).astype(int)
    units = np.minimum(units, units.T)
    np.fill_diagonal(units, 0)
    validate_integer_configuration(units, model)
    tolerance = (1.0 + epsilon) * logical_cost
    aggregate = np.sum(np.stack(demands), axis=0)
    additions = 0

    def can_add_pair(i: int, j: int) -> bool:
        trial = units.copy()
        trial[i, j] += 1
        trial[j, i] += 1
        try:
            validate_integer_configuration(trial, model)
            return True
        except ValueError:
            return False

    for i in range(units.shape[0]):
        for j in range(i + 1, units.shape[0]):
            needs = (aggregate[i, j] > 0 and fixed[i, j] <= 0) or (aggregate[j, i] > 0 and fixed[j, i] <= 0)
            if needs and units[i, j] == 0 and can_add_pair(i, j):
                units[i, j] += 1
                units[j, i] += 1
                additions += 2
    current = segment_cost(demands, units, unit_bandwidth, fixed)
    while not _within_tolerance(current, tolerance):
        best = None
        for i in range(units.shape[0]):
            for j in range(i + 1, units.shape[0]):
                if not can_add_pair(i, j) or aggregate[i, j] + aggregate[j, i] <= 0:
                    continue
                trial = units.copy()
                trial[i, j] += 1
                trial[j, i] += 1
                after = segment_cost(demands, trial, unit_bandwidth, fixed)
                gain = float("inf") if np.isinf(current) and np.isfinite(after) else current - after
                candidate = (gain, i, j, after)
                if best is None or candidate[0] > best[0] + EPS:
                    best = candidate
        if best is None or best[0] <= EPS:
            break
        _, i, j, current = best
        units[i, j] += 1
        units[j, i] += 1
        additions += 2
    pruned = 0
    if _within_tolerance(current, tolerance):
        changed = True
        while changed:
            changed = False
            for i in range(units.shape[0]):
                for j in range(i + 1, units.shape[0]):
                    if units[i, j] <= 0:
                        continue
                    trial = units.copy()
                    trial[i, j] -= 1
                    trial[j, i] -= 1
                    trial_cost = segment_cost(demands, trial, unit_bandwidth, fixed)
                    if _within_tolerance(trial_cost, tolerance):
                        units, current, pruned, changed = trial, trial_cost, pruned + 2, True
                        break
                if changed:
                    break
    validate_integer_configuration(units, model)
    if not np.array_equal(units, units.T):
        raise AssertionError("symmetric realization lost pair symmetry")
    satisfied = _within_tolerance(current, tolerance)
    return RealizationResult("SymOCSSparse", units, current, logical_cost, tolerance, satisfied, not satisfied, int(units.sum()), additions, pruned, () if satisfied else _requests(demands, units, model, unit_bandwidth, fixed))


def exhaustive_realization_oracle(
    target: np.ndarray,
    demands: Sequence[np.ndarray],
    logical_cost: float,
    epsilon: float,
    resources: OCSResources,
    unit_bandwidth: float,
    fixed_bandwidth: np.ndarray | None = None,
) -> RealizationResult:
    """Small-instance enumeration oracle for the minimum-unit objective."""

    size = target.shape[0]
    model = resources.normalized(size)
    fixed = np.zeros_like(target) if fixed_bandwidth is None else np.asarray(fixed_bandwidth, dtype=float)
    active = [(i, j) for i in range(size) for j in range(size) if model.reachable[i, j] and any(d[i, j] > 0 for d in demands)]
    if len(active) > 8 or max([*model.n_tx, *model.n_rx], default=0) > 4:
        raise ValueError("exhaustive oracle is restricted to small validation instances")
    tolerance = (1.0 + epsilon) * logical_cost
    best_units = None
    best_key = None
    limits = [min(int(model.n_tx[i]), int(model.n_rx[j])) for i, j in active]
    for values in product(*[range(limit + 1) for limit in limits]):
        units = np.zeros((size, size), dtype=int)
        for (i, j), value in zip(active, values):
            units[i, j] = value
        try:
            validate_integer_configuration(units, model)
        except ValueError:
            continue
        cost = segment_cost(demands, units, unit_bandwidth, fixed)
        if not _within_tolerance(cost, tolerance):
            continue
        key = (int(units.sum()), float(cost), tuple(int(v) for v in units.ravel()))
        if best_key is None or key < best_key:
            best_key, best_units = key, units.copy()
    if best_units is None:
        best_units = np.zeros((size, size), dtype=int)
        cost = segment_cost(demands, best_units, unit_bandwidth, fixed)
        satisfied = False
    else:
        cost = segment_cost(demands, best_units, unit_bandwidth, fixed)
        satisfied = True
    return RealizationResult("EnumerationOracle", best_units, cost, logical_cost, tolerance, satisfied, not satisfied, int(best_units.sum()), 0, 0, () if satisfied else _requests(demands, best_units, model, unit_bandwidth, fixed))
