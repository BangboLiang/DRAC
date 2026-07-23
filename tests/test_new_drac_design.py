from __future__ import annotations

import math

import numpy as np

from drac_eval.demand_profiler import (
    CalibrationBin,
    CommunicationNode,
    RankPlacement,
    TransportCalibration,
    profile_communication_nodes,
)
from drac_eval.directional_target import (
    completion_time,
    solve_continuous_target,
    solve_continuous_target_numerical,
)
from drac_eval.resource_compaction import compact_schedule, verify_compaction_sufficiency
from drac_eval.sparse_realization import (
    OCSResources,
    exhaustive_realization_oracle,
    realize_drac_sparse,
    validate_integer_configuration,
)
from drac_eval.target_segmentation import (
    build_service_cost_matrix,
    candidate_segment_costs,
    exhaustive_segmentation_oracle,
    segment_target_sequence,
)


def _placements(servers: tuple[str, ...]) -> dict[int, RankPlacement]:
    return {
        rank: RankPlacement(rank=rank, endpoint=rank, server=server)
        for rank, server in enumerate(servers)
    }


def test_ordered_profiler_ring_directions_and_calibration() -> None:
    node = CommunicationNode(
        "ar0", "allreduce", 400.0, ranks=(0, 1, 2, 3), algorithm="ring"
    )
    calibration = TransportCalibration(
        bins=(CalibrationBin(100, 10.0, 5.0, samples=10),), environment="fixture"
    )
    result = profile_communication_nodes(
        [node], _placements(("s0", "s1", "s2", "s3")), calibration
    )[0]
    # Two ring phases, three steps, 100 bytes per transfer on each ordered edge.
    assert result.payload_matrix[0, 1] == 600.0
    assert result.payload_matrix[1, 0] == 0.0
    assert result.control_matrix[0, 1] == 60.0
    assert result.control_matrix[1, 0] == 30.0
    assert result.matrix[0, 1] != result.matrix[1, 0]
    assert result.provenance == "payload+calibration"


def test_rank_to_endpoint_mapping_is_explicit() -> None:
    placements = {
        0: RankPlacement(0, 2, "server-a"),
        1: RankPlacement(1, 0, "server-b"),
    }
    node = CommunicationNode("pp-fwd", "p2p", 128.0, src_rank=0, dst_rank=1)
    result = profile_communication_nodes([node], placements)[0]
    assert result.endpoint_order == (0, 2)
    assert result.matrix[1, 0] == 128.0
    assert result.matrix[0, 1] == 0.0


def test_intra_server_traffic_is_excluded() -> None:
    node = CommunicationNode("pp-local", "p2p", 256.0, src_rank=0, dst_rank=1)
    result = profile_communication_nodes(
        [node], _placements(("same-server", "same-server"))
    )[0]
    assert result.matrix.sum() == 0.0
    assert result.excluded_intra_server_bytes == 256.0


def _target_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    demand = np.array(
        [[0.0, 8.0, 2.0], [1.0, 0.0, 5.0], [4.0, 3.0, 0.0]], dtype=float
    )
    return demand, np.array([3.0, 2.0, 4.0]), np.array([2.0, 4.0, 3.0])


def test_closed_form_target_feasibility_completion_and_numerical_optimum() -> None:
    demand, n_tx, n_rx = _target_case()
    closed = solve_continuous_target(demand, n_tx, n_rx, 2.0)
    numerical = solve_continuous_target_numerical(demand, n_tx, n_rx, 2.0)
    assert closed.method == "closed_form"
    assert np.all(closed.tx_usage <= n_tx + 1e-10)
    assert np.all(closed.rx_usage <= n_rx + 1e-10)
    assert math.isclose(closed.theta, numerical.theta, rel_tol=1e-9, abs_tol=1e-10)
    assert completion_time(demand, closed.allocation, 2.0) <= closed.theta * (1 + 1e-10)


def test_closed_form_target_is_entrywise_minimum_resource_at_optimum() -> None:
    demand, n_tx, n_rx = _target_case()
    result = solve_continuous_target(demand, n_tx, n_rx, 2.0)
    for src, dst in zip(*np.where(demand > 0)):
        smaller = result.allocation.copy()
        smaller[src, dst] *= 1.0 - 1e-5
        assert completion_time(demand, smaller, 2.0) > result.theta


def test_fixed_bandwidth_binary_search_solver() -> None:
    demand, n_tx, n_rx = _target_case()
    fixed = np.zeros_like(demand)
    fixed[0, 1] = 1.5
    result = solve_continuous_target(demand, n_tx, n_rx, 2.0, fixed)
    numerical = solve_continuous_target_numerical(demand, n_tx, n_rx, 2.0, fixed)
    assert result.method == "binary_search"
    assert np.all(result.tx_usage <= n_tx + 1e-9)
    assert np.all(result.rx_usage <= n_rx + 1e-9)
    assert math.isclose(result.theta, numerical.theta, rel_tol=1e-9, abs_tol=1e-9)
    assert completion_time(demand, result.allocation, 2.0, fixed) <= result.theta * (1 + 1e-9)


def _flipping_demands() -> list[np.ndarray]:
    return [
        np.array([[0.0, 9.0], [1.0, 0.0]]),
        np.array([[0.0, 8.0], [1.0, 0.0]]),
        np.array([[0.0, 1.0], [9.0, 0.0]]),
        np.array([[0.0, 1.0], [8.0, 0.0]]),
    ]


def test_service_cost_and_candidate_representative() -> None:
    demands = _flipping_demands()
    targets = [
        solve_continuous_target(demand, np.array([4, 4]), np.array([4, 4]), 1.0)
        for demand in demands
    ]
    service = build_service_cost_matrix(demands, targets, 1.0)
    assert math.isclose(service[0, 0], targets[0].theta)
    costs, representatives = candidate_segment_costs(service)
    direct = [sum(service[k, h] for k in range(2)) for h in range(2)]
    assert math.isclose(costs[0, 1], min(direct))
    assert representatives[0, 1] == int(np.argmin(direct))


def test_dp_segmentation_matches_exhaustive_and_backtracks() -> None:
    demands = _flipping_demands()
    targets = [
        solve_continuous_target(demand, np.array([4, 4]), np.array([4, 4]), 1.0)
        for demand in demands
    ]
    result = segment_target_sequence(demands, targets, 1.0, delta=0.3)
    oracle_cost, oracle_segments = exhaustive_segmentation_oracle(result.service_cost, 0.3)
    assert math.isclose(result.total_cost, oracle_cost, rel_tol=1e-12)
    assert [(s.start, s.end, s.representative) for s in result.segments] == [
        (s.start, s.end, s.representative) for s in oracle_segments
    ]
    assert result.segments[0].start == 0
    assert result.segments[-1].end == len(demands) - 1
    assert all(left.end + 1 == right.start for left, right in zip(result.segments, result.segments[1:]))


def test_segmentation_delta_boundaries() -> None:
    demands = _flipping_demands()
    targets = [
        solve_continuous_target(demand, np.array([4, 4]), np.array([4, 4]), 1.0)
        for demand in demands
    ]
    zero = segment_target_sequence(demands, targets, 1.0, delta=0.0)
    high = segment_target_sequence(demands, targets, 1.0, delta=1e9)
    assert zero.communication_cost <= high.communication_cost + 1e-12
    assert len(zero.segments) >= len(high.segments)
    assert len(high.segments) == 1


def test_sparse_realization_feasibility_tolerance_and_reverse_pruning() -> None:
    demand = np.array([[0.0, 10.0], [1.0, 0.0]])
    resources = OCSResources(np.array([4, 4]), np.array([4, 4]), total_units=8)
    target = solve_continuous_target(demand, resources.n_tx, resources.n_rx, 1.0)
    result = realize_drac_sparse(
        target.allocation, [demand], target.theta, 0.5, resources, 1.0
    )
    validate_integer_configuration(result.units, resources)
    assert result.tolerance_satisfied
    assert result.cost <= 1.5 * target.theta + 1e-12
    assert result.pruned >= 1
    for src, dst in zip(*np.where(result.units > 0)):
        trial = result.units.copy()
        trial[src, dst] -= 1
        assert completion_time(demand, trial, 1.0) > result.tolerance_cost


def test_sparse_realization_resource_constrained_request_map() -> None:
    demand = np.array([[0.0, 10.0], [1.0, 0.0]])
    resources = OCSResources(np.array([1, 1]), np.array([1, 1]), total_units=1)
    target = solve_continuous_target(
        demand, np.array([1, 1]), np.array([1, 1]), 1.0
    )
    result = realize_drac_sparse(
        target.allocation, [demand], target.theta, 0.0, resources, 1.0
    )
    assert result.resource_constrained
    assert not result.tolerance_satisfied
    assert result.requests


def test_sparse_realization_matches_small_enumeration_unit_objective() -> None:
    demand = np.array([[0.0, 6.0], [2.0, 0.0]])
    resources = OCSResources(np.array([3, 3]), np.array([3, 3]), total_units=6)
    target = solve_continuous_target(demand, resources.n_tx, resources.n_rx, 1.0)
    sparse = realize_drac_sparse(
        target.allocation, [demand], target.theta, 0.5, resources, 1.0
    )
    oracle = exhaustive_realization_oracle(
        target.allocation, [demand], target.theta, 0.5, resources, 1.0
    )
    assert sparse.tolerance_satisfied
    assert sparse.used_units == oracle.used_units


def test_schedule_wide_compaction_lower_bound_sufficiency_and_binding() -> None:
    first = np.array([[0, 2, 0], [0, 0, 1], [1, 0, 0]], dtype=int)
    second = np.array([[0, 0, 1], [2, 0, 0], [0, 1, 0]], dtype=int)
    inventory = np.array([4, 4, 4])
    result = compact_schedule([first, second], inventory, inventory)
    assert np.array_equal(result.reserved_tx, np.maximum(first.sum(axis=1), second.sum(axis=1)))
    assert np.array_equal(result.reserved_rx, np.maximum(first.sum(axis=0), second.sum(axis=0)))
    assert verify_compaction_sufficiency(result, [first, second])
    assert np.array_equal(result.reserved_bundles, np.maximum(result.reserved_tx, result.reserved_rx))
    assert [len(bindings) for bindings in result.bindings] == [int(first.sum()), int(second.sum())]
    for bindings in result.bindings:
        assert len({(item.src, item.tx_channel) for item in bindings}) == len(bindings)
        assert len({(item.dst, item.rx_channel) for item in bindings}) == len(bindings)


def test_profiler_is_deterministic() -> None:
    node = CommunicationNode(
        "rs", "reducescatter", 512.0, ranks=(0, 1, 2, 3), algorithm="ring"
    )
    placement = _placements(("a", "b", "c", "d"))
    first = profile_communication_nodes([node], placement)[0]
    second = profile_communication_nodes([node], placement)[0]
    assert np.array_equal(first.matrix, second.matrix)
    assert first.transfers == second.transfers
