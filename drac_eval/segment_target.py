"""Convex continuous target reoptimization for a contiguous node segment.

Introduced by the v2 algorithm revision requested for
``scripts/run_all_evaluation_v2.py``.  Unlike the v1 medoid dictionary, this
module optimizes one new allocation against every node in the segment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from .directional_target import (
    ContinuousTarget,
    completion_time,
    solve_continuous_target,
    solve_symmetric_continuous_target,
)


NUMERICAL_TOL = 1e-8


@dataclass(frozen=True)
class SegmentTarget:
    allocation: np.ndarray
    node_times: np.ndarray
    cost: float
    symmetric: bool
    success: bool
    method: str
    iterations: int
    max_constraint_violation: float
    warm_start: str
    optimizer_message: str


def segment_service_cost(
    demands: Sequence[np.ndarray],
    allocation: np.ndarray,
    unit_bandwidth: float,
    fixed_bandwidth: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    times = np.asarray(
        [completion_time(d, allocation, unit_bandwidth, fixed_bandwidth) for d in demands],
        dtype=float,
    )
    return float(np.sum(times)), times


def _validate_segment_inputs(
    demands: Sequence[np.ndarray], n_tx: np.ndarray, n_rx: np.ndarray
) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    if not demands:
        raise ValueError("a candidate segment must contain at least one demand")
    normalized = tuple(np.asarray(d, dtype=float) for d in demands)
    shape = normalized[0].shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("demand matrices must be square")
    if any(d.shape != shape or np.any(d < 0) for d in normalized):
        raise ValueError("segment demand matrices must share one non-negative shape")
    tx = np.asarray(n_tx, dtype=float)
    rx = np.asarray(n_rx, dtype=float)
    if tx.shape != (shape[0],) or rx.shape != (shape[0],) or np.any(tx < 0) or np.any(rx < 0):
        raise ValueError("invalid endpoint resource vectors")
    return normalized, tx, rx


def _single_node_targets(
    demands: Sequence[np.ndarray],
    n_tx: np.ndarray,
    n_rx: np.ndarray,
    unit_bandwidth: float,
    fixed: np.ndarray,
    symmetric: bool,
) -> tuple[ContinuousTarget, ...]:
    solver = solve_symmetric_continuous_target if symmetric else solve_continuous_target
    return tuple(solver(d, n_tx, n_rx, unit_bandwidth, fixed) for d in demands)


def medoid_segment_target(
    demands: Sequence[np.ndarray],
    node_targets: Sequence[ContinuousTarget | np.ndarray],
    unit_bandwidth: float,
    fixed_bandwidth: np.ndarray | None = None,
) -> tuple[np.ndarray, float, int]:
    """Return the v1 MedoidTarget ablation and its verified service cost."""

    if len(demands) != len(node_targets) or not demands:
        raise ValueError("demands and node_targets must be non-empty and equally sized")
    best: tuple[float, int, np.ndarray] | None = None
    for index, target in enumerate(node_targets):
        allocation = (
            np.asarray(target.allocation, dtype=float)
            if hasattr(target, "allocation")
            else np.asarray(target, dtype=float)
        )
        cost, _ = segment_service_cost(demands, allocation, unit_bandwidth, fixed_bandwidth)
        key = (cost, index, allocation)
        if best is None or cost < best[0] - 1e-10 or (abs(cost - best[0]) <= 1e-10 and index < best[1]):
            best = key
    assert best is not None
    return np.asarray(best[2], dtype=float).copy(), float(best[0]), int(best[1])


def solve_segment_continuous_target(
    demands: Sequence[np.ndarray],
    n_tx: np.ndarray,
    n_rx: np.ndarray,
    unit_bandwidth: float,
    fixed_bandwidth: np.ndarray | None = None,
    *,
    symmetric: bool = False,
    warm_start_allocation: np.ndarray | None = None,
    max_iterations: int = 2000,
) -> SegmentTarget:
    """Solve ``min_Y sum_k L_k(Y)`` with a convex epigraph formulation.

    SciPy SLSQP is used only as the numerical optimizer.  Feasibility and the
    objective are independently recomputed before a result is accepted.  The
    best single-node medoid remains a feasible upper-bound candidate.
    """

    normalized, tx, rx = _validate_segment_inputs(demands, n_tx, n_rx)
    size = normalized[0].shape[0]
    if unit_bandwidth <= 0:
        raise ValueError("unit_bandwidth must be positive")
    fixed = np.zeros((size, size), dtype=float) if fixed_bandwidth is None else np.asarray(fixed_bandwidth, dtype=float)
    if fixed.shape != (size, size) or np.any(fixed < 0):
        raise ValueError("invalid fixed bandwidth")

    node_targets = _single_node_targets(normalized, tx, rx, unit_bandwidth, fixed, symmetric)
    medoid_y, medoid_cost, medoid_index = medoid_segment_target(
        normalized, node_targets, unit_bandwidth, fixed
    )
    if len(normalized) == 1:
        cost, times = segment_service_cost(normalized, medoid_y, unit_bandwidth, fixed)
        return SegmentTarget(
            medoid_y,
            times,
            cost,
            symmetric,
            True,
            "single_node_exact",
            0,
            0.0,
            "single-node",
            "closed-form single-node target",
        )

    edges = [(i, j) for i in range(size) for j in range(size) if i != j]
    edge_index = {edge: index for index, edge in enumerate(edges)}
    edge_count = len(edges)
    node_count = len(normalized)

    average_y = np.mean(np.stack([t.allocation for t in node_targets]), axis=0)
    warm_label = "average-node-targets"
    if warm_start_allocation is not None:
        supplied = np.asarray(warm_start_allocation, dtype=float)
        if supplied.shape != (size, size) or np.any(supplied < 0):
            raise ValueError("invalid segment warm-start allocation")
        if np.any(supplied.sum(axis=1) > tx + NUMERICAL_TOL) or np.any(supplied.sum(axis=0) > rx + NUMERICAL_TOL):
            raise ValueError("segment warm start violates endpoint budgets")
        if symmetric and not np.allclose(supplied, supplied.T, atol=NUMERICAL_TOL):
            raise ValueError("symmetric solve requires a symmetric warm start")
        average_y = supplied.copy()
        warm_label = "supplied-feasible-allocation"
    np.fill_diagonal(average_y, 0.0)
    average_cost, average_times = segment_service_cost(normalized, average_y, unit_bandwidth, fixed)
    if not np.isfinite(average_cost):
        raise ValueError("could not construct a finite feasible segment warm start")

    y0 = np.asarray([average_y[i, j] for i, j in edges], dtype=float)
    theta0 = np.maximum(average_times * 1.01, 1e-12)
    x0 = np.concatenate([y0, theta0])

    def objective(x: np.ndarray) -> float:
        return float(np.sum(x[edge_count:]))

    def objective_jac(x: np.ndarray) -> np.ndarray:
        gradient = np.zeros_like(x)
        gradient[edge_count:] = 1.0
        return gradient

    def capacity_constraints(x: np.ndarray) -> np.ndarray:
        y = x[:edge_count]
        outgoing = np.zeros(size, dtype=float)
        incoming = np.zeros(size, dtype=float)
        for value, (src, dst) in zip(y, edges):
            outgoing[src] += value
            incoming[dst] += value
        return np.concatenate([tx - outgoing, rx - incoming])

    capacity_jacobian = np.zeros((2 * size, edge_count + node_count), dtype=float)
    for position, (src, dst) in enumerate(edges):
        capacity_jacobian[src, position] = -1.0
        capacity_jacobian[size + dst, position] = -1.0

    service_rows: list[tuple[int, int, float]] = []
    for node, demand in enumerate(normalized):
        for src, dst in zip(*np.where(demand > 0)):
            service_rows.append((node, edge_index[(int(src), int(dst))], float(demand[src, dst])))

    fixed_edges = np.asarray([fixed[i, j] for i, j in edges], dtype=float)

    def service_constraints(x: np.ndarray) -> np.ndarray:
        y = x[:edge_count]
        theta = x[edge_count:]
        return np.asarray(
            [(fixed_edges[edge] + unit_bandwidth * y[edge]) / demand - 1.0 / theta[node]
             for node, edge, demand in service_rows],
            dtype=float,
        )

    def service_jacobian(x: np.ndarray) -> np.ndarray:
        theta = x[edge_count:]
        jac = np.zeros((len(service_rows), edge_count + node_count), dtype=float)
        for row, (node, edge, demand) in enumerate(service_rows):
            jac[row, edge] = unit_bandwidth / demand
            jac[row, edge_count + node] = 1.0 / (theta[node] ** 2)
        return jac

    constraints: list[dict[str, object]] = [
        {"type": "ineq", "fun": capacity_constraints, "jac": lambda x: capacity_jacobian},
        {"type": "ineq", "fun": service_constraints, "jac": service_jacobian},
    ]
    if symmetric:
        pairs = [(edge_index[(i, j)], edge_index[(j, i)]) for i in range(size) for j in range(i + 1, size)]

        def symmetry_constraints(x: np.ndarray) -> np.ndarray:
            return np.asarray([x[left] - x[right] for left, right in pairs], dtype=float)

        symmetry_jacobian = np.zeros((len(pairs), edge_count + node_count), dtype=float)
        for row, (left, right) in enumerate(pairs):
            symmetry_jacobian[row, left] = 1.0
            symmetry_jacobian[row, right] = -1.0
        constraints.append({"type": "eq", "fun": symmetry_constraints, "jac": lambda x: symmetry_jacobian})

    bounds = []
    for src, dst in edges:
        bounds.append((0.0, float(min(tx[src], rx[dst]))))
    bounds.extend([(1e-12, None)] * node_count)

    result = minimize(
        objective,
        x0,
        jac=objective_jac,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-11, "maxiter": int(max_iterations), "disp": False},
    )

    candidate_y = np.zeros((size, size), dtype=float)
    for value, (src, dst) in zip(result.x[:edge_count], edges):
        candidate_y[src, dst] = max(0.0, float(value))
    candidate_cost, candidate_times = segment_service_cost(normalized, candidate_y, unit_bandwidth, fixed)
    cap_values = capacity_constraints(result.x)
    service_values = service_constraints(result.x)
    symmetry_values = np.zeros(1) if not symmetric else symmetry_constraints(result.x)
    max_violation = float(
        max(
            0.0,
            -float(np.min(cap_values, initial=0.0)),
            -float(np.min(service_values, initial=0.0)),
            float(np.max(np.abs(symmetry_values), initial=0.0)),
        )
    )
    candidate_valid = bool(
        np.isfinite(candidate_cost)
        and max_violation <= NUMERICAL_TOL
        and np.all(candidate_y.sum(axis=1) <= tx + NUMERICAL_TOL)
        and np.all(candidate_y.sum(axis=0) <= rx + NUMERICAL_TOL)
        and (not symmetric or np.allclose(candidate_y, candidate_y.T, atol=NUMERICAL_TOL))
    )

    upper_bounds: list[tuple[float, np.ndarray, str]] = [(medoid_cost, medoid_y, "medoid")]
    if warm_start_allocation is not None:
        warm_cost, _ = segment_service_cost(normalized, average_y, unit_bandwidth, fixed)
        upper_bounds.append((warm_cost, average_y, "supplied-warm-start"))
    upper_cost, upper_y, upper_name = min(upper_bounds, key=lambda item: (item[0], item[2]))

    # The medoid and supplied allocation are legitimate feasible upper bounds. Returning one is safer
    # than accepting a failed numerical solve, and the method field makes this
    # fallback explicit for the report and acceptance tests.
    if not candidate_valid or candidate_cost > upper_cost + 1e-7 * max(1.0, upper_cost):
        cost, times = segment_service_cost(normalized, upper_y, unit_bandwidth, fixed)
        return SegmentTarget(
            upper_y.copy(),
            times,
            cost,
            symmetric,
            True,
            "feasible_warm_start_numerical_fallback" if upper_name != "medoid" else "medoid_numerical_fallback",
            int(getattr(result, "nit", 0)),
            max_violation,
            f"{warm_label}; upper-bound={upper_name}; medoid={medoid_index}",
            str(result.message),
        )

    return SegmentTarget(
        candidate_y,
        candidate_times,
        candidate_cost,
        symmetric,
        bool(result.success or candidate_valid),
        "segment_epigraph_slsqp",
        int(getattr(result, "nit", 0)),
        max_violation,
        f"{warm_label}; medoid={medoid_index}",
        str(result.message),
    )
