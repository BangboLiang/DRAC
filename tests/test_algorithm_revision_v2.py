from __future__ import annotations

import math

import numpy as np

from drac_eval.directional_target import solve_continuous_target
from drac_eval.evaluation_pipeline_v2 import plan_schedule_candidates_v2
from drac_eval.segment_target import (
    medoid_segment_target,
    solve_segment_continuous_target,
)
from drac_eval.sparse_realization import (
    OCSResources,
    exhaustive_realization_oracle,
    realize_drac_sparse_coverage_seed,
    realize_drac_sparse_floor_seed,
    realize_drac_sparse_multi_seed,
    realize_fill_all_residual,
    validate_integer_configuration,
)
from drac_eval.target_segmentation import (
    build_candidate_target_table,
    exhaustive_partition_oracle_v2,
    segment_continuous_sequence,
)


def _demand(size: int, src: int, dst: int, amount: float = 1.0) -> np.ndarray:
    matrix = np.zeros((size, size), dtype=float)
    matrix[src, dst] = amount
    return matrix


def _resources(size: int, ports: int) -> OCSResources:
    return OCSResources(np.full(size, ports, dtype=int), np.full(size, ports, dtype=int))


def test_same_direction_nodes_merge_and_match_medoid() -> None:
    demands = (_demand(2, 0, 1, 2.0), _demand(2, 0, 1, 4.0))
    table = build_candidate_target_table(demands, np.ones(2), np.ones(2), 1.0)
    result = segment_continuous_sequence(
        demands, np.ones(2), np.ones(2), 1.0, 1.0, candidate_targets=table
    )
    assert len(result.segments) == 1
    assert table.directional_cost[0, 1] <= table.medoid_cost[0, 1] + 1e-8
    assert math.isclose(table.directional_cost[0, 1], table.medoid_cost[0, 1], rel_tol=1e-7)


def test_pp_forward_backward_creates_new_near_symmetric_target() -> None:
    forward = _demand(2, 0, 1, 10.0)
    backward = _demand(2, 1, 0, 10.0)
    targets = [
        solve_continuous_target(d, np.ones(2), np.ones(2), 1.0)
        for d in (forward, backward)
    ]
    _, medoid_cost, _ = medoid_segment_target((forward, backward), targets, 1.0)
    segment = solve_segment_continuous_target(
        (forward, backward), np.ones(2), np.ones(2), 1.0
    )
    assert np.allclose(segment.allocation, segment.allocation.T, atol=1e-7)
    assert segment.allocation[0, 1] > 0 and segment.allocation[1, 0] > 0
    assert segment.cost < medoid_cost
    assert segment.method == "segment_epigraph_slsqp"


def test_direction_flip_low_delta_splits_high_delta_merges() -> None:
    demands = (_demand(3, 0, 1), _demand(3, 0, 2))
    tx = rx = np.ones(3)
    table = build_candidate_target_table(demands, tx, rx, 1.0)
    low = segment_continuous_sequence(demands, tx, rx, 1.0, 0.01, candidate_targets=table)
    high = segment_continuous_sequence(demands, tx, rx, 1.0, 10.0, candidate_targets=table)
    assert len(low.segments) == 2
    assert len(high.segments) == 1


def test_directional_segment_cost_not_above_symmetric() -> None:
    demands = (
        _demand(3, 0, 1, 2.0) + _demand(3, 2, 0, 0.5),
        _demand(3, 1, 0, 1.0) + _demand(3, 0, 2, 1.0),
    )
    table = build_candidate_target_table(demands, np.full(3, 2.0), np.full(3, 2.0), 1.0)
    assert table.directional_cost[0, 1] <= table.symmetric_cost[0, 1] + 1e-7


def test_segment_dynamic_programming_matches_complete_partition_oracle() -> None:
    demands = (
        _demand(3, 0, 1),
        _demand(3, 0, 2),
        _demand(3, 0, 1, 2.0),
        _demand(3, 0, 2, 2.0),
    )
    tx = rx = np.ones(3)
    table = build_candidate_target_table(demands, tx, rx, 1.0)
    for delta in (0.0, 0.2, 10.0):
        result = segment_continuous_sequence(
            demands, tx, rx, 1.0, delta, candidate_targets=table
        )
        oracle_cost, oracle_boundaries = exhaustive_partition_oracle_v2(table, delta)
        assert math.isclose(result.total_cost, oracle_cost, rel_tol=1e-8, abs_tol=1e-8)
        assert tuple((segment.start, segment.end) for segment in result.segments) == oracle_boundaries


def test_segment_solver_and_backtracking_are_deterministic() -> None:
    demands = (_demand(3, 0, 1), _demand(3, 0, 2))
    tx = rx = np.ones(3)
    first = segment_continuous_sequence(demands, tx, rx, 1.0, 0.3)
    second = segment_continuous_sequence(demands, tx, rx, 1.0, 0.3)
    assert first.total_cost == second.total_cost
    assert np.array_equal(first.predecessor, second.predecessor)
    assert [(s.start, s.end) for s in first.segments] == [(s.start, s.end) for s in second.segments]


def test_tied_bottleneck_group_addition_makes_progress() -> None:
    demand = _demand(3, 0, 1) + _demand(3, 0, 2)
    target = np.zeros((3, 3), dtype=float)
    target[0, 1] = target[0, 2] = 2.0
    result = realize_drac_sparse_coverage_seed(
        target, [demand], 0.5, 0.0, _resources(3, 4), 1.0
    )
    assert result.tolerance_satisfied
    assert result.group_additions >= 1
    assert result.units[0, 1] == 2 and result.units[0, 2] == 2


def test_swap_local_search_improves_constructed_floor_seed() -> None:
    demands = (_demand(3, 0, 1, 1.0), _demand(3, 0, 2, 3.0))
    target = np.zeros((3, 3), dtype=float)
    target[0, 1] = 2.1
    target[0, 2] = 1.1
    result = realize_drac_sparse_floor_seed(
        target, demands, 3.6, 0.0, _resources(3, 3), 1.0
    )
    assert result.tolerance_satisfied
    assert result.swaps >= 1
    assert result.swap_gain > 0
    assert result.units[0, 1] == 1 and result.units[0, 2] == 2


def test_multiseed_epsilon_path_is_monotone_and_beats_fill_density() -> None:
    demand = _demand(3, 0, 1) + _demand(3, 0, 2)
    target = np.zeros((3, 3), dtype=float)
    target[0, 1] = target[0, 2] = 3.0
    historical: list[np.ndarray] = []
    counts = []
    for epsilon in (0.0, 0.5, 1.0, 2.0):
        result = realize_drac_sparse_multi_seed(
            target,
            [demand],
            1.0 / 3.0,
            epsilon,
            _resources(3, 6),
            1.0,
            historical_units=historical,
        )
        assert result.tolerance_satisfied
        counts.append(result.used_units)
        historical.append(result.units.copy())
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] < counts[0]
    fill = realize_fill_all_residual(
        target, [demand], 1.0 / 3.0, 2.0, _resources(3, 6), 1.0
    )
    assert counts[-1] < fill.used_units


def test_sparse_coverage_seed_escapes_overdense_floor_basin() -> None:
    demand0 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 6.0], [0.0, 3.0, 0.0]])
    demand1 = np.array([[0.0, 6.0, 0.0], [0.0, 0.0, 5.0], [0.0, 1.0, 0.0]])
    target = np.array(
        [
            [0.0, 1.232583967428955, 0.0],
            [0.0, 0.0, 2.724980534682386],
            [0.0, 3.8507218990277674, 0.0],
        ]
    )
    resources = _resources(3, 4)
    logical = 7.069673115232875
    floor = realize_drac_sparse_floor_seed(
        target, [demand0, demand1], logical, 0.2, resources, 1.0
    )
    coverage = realize_drac_sparse_coverage_seed(
        target, [demand0, demand1], logical, 0.2, resources, 1.0
    )
    assert floor.tolerance_satisfied and coverage.tolerance_satisfied
    assert coverage.used_units < floor.used_units


def test_multiseed_matches_small_exhaustive_oracle() -> None:
    demand = _demand(3, 0, 1) + _demand(3, 0, 2)
    target = np.zeros((3, 3), dtype=float)
    target[0, 1] = target[0, 2] = 2.0
    resources = _resources(3, 4)
    result = realize_drac_sparse_multi_seed(target, [demand], 0.5, 1.0, resources, 1.0)
    oracle = exhaustive_realization_oracle(target, [demand], 0.5, 1.0, resources, 1.0)
    assert result.tolerance_satisfied and oracle.tolerance_satisfied
    assert result.used_units == oracle.used_units


def test_resource_constrained_coverage_is_explicit() -> None:
    demand = _demand(3, 0, 1) + _demand(3, 0, 2)
    resources = OCSResources(np.array([1, 1, 1]), np.ones(3, dtype=int))
    result = realize_drac_sparse_coverage_seed(
        np.zeros((3, 3)), [demand], 1.0, 1.0, resources, 1.0
    )
    validate_integer_configuration(result.units, resources)
    assert result.resource_constrained
    assert not result.tolerance_satisfied


def test_schedule_level_symmetric_fallback_has_explicit_no_harm_bound() -> None:
    demands = (_demand(2, 0, 1, 10.0), _demand(2, 1, 0, 10.0))
    candidates = plan_schedule_candidates_v2(
        demands, _resources(2, 2), 1.0, delta=0.5, epsilon=0.1
    )
    assert candidates.selected.total_cost <= candidates.symmetric.total_cost + 1e-8
    assert "schedule-level fallback selected" in candidates.selected.selected_from
    assert len(candidates.selected.fallback_reasons) == len(candidates.selected.realizations)
