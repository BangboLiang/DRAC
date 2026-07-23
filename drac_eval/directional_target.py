"""Per-communication-node continuous directional targets from DRAC Section IV-C."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TOL = 1e-10


@dataclass(frozen=True)
class ContinuousTarget:
    allocation: np.ndarray
    theta: float
    tx_usage: np.ndarray
    rx_usage: np.ndarray
    method: str


def _validate_inputs(
    demand: np.ndarray,
    n_tx: np.ndarray,
    n_rx: np.ndarray,
    unit_bandwidth: float,
    fixed_bandwidth: np.ndarray,
) -> None:
    if demand.ndim != 2 or demand.shape[0] != demand.shape[1]:
        raise ValueError("demand must be square")
    n = demand.shape[0]
    if n_tx.shape != (n,) or n_rx.shape != (n,) or fixed_bandwidth.shape != demand.shape:
        raise ValueError("resource and fixed-capacity shapes must match demand")
    if np.any(demand < 0) or np.any(fixed_bandwidth < 0):
        raise ValueError("demand and fixed capacity must be non-negative")
    if np.any(n_tx < 0) or np.any(n_rx < 0) or unit_bandwidth <= 0:
        raise ValueError("invalid channel inventory or unit bandwidth")


def completion_time(
    demand: np.ndarray,
    allocation: np.ndarray,
    unit_bandwidth: float,
    fixed_bandwidth: np.ndarray | None = None,
) -> float:
    fixed = np.zeros_like(demand, dtype=float) if fixed_bandwidth is None else np.asarray(fixed_bandwidth, dtype=float)
    capacity = fixed + float(unit_bandwidth) * np.asarray(allocation, dtype=float)
    positive = demand > 0
    if not np.any(positive):
        return 0.0
    if np.any(capacity[positive] <= 0):
        return float("inf")
    return float(np.max(demand[positive] / capacity[positive]))


def target_for_theta(
    demand: np.ndarray,
    theta: float,
    unit_bandwidth: float,
    fixed_bandwidth: np.ndarray,
) -> np.ndarray:
    if theta <= 0:
        return np.full_like(demand, float("inf"), dtype=float)
    target = np.maximum(demand / theta - fixed_bandwidth, 0.0) / unit_bandwidth
    target[demand <= 0] = 0.0
    np.fill_diagonal(target, 0.0)
    return target


def _feasible(target: np.ndarray, n_tx: np.ndarray, n_rx: np.ndarray) -> bool:
    return bool(
        np.all(target.sum(axis=1) <= n_tx + TOL)
        and np.all(target.sum(axis=0) <= n_rx + TOL)
    )


def solve_continuous_target(
    demand: np.ndarray,
    n_tx: np.ndarray,
    n_rx: np.ndarray,
    unit_bandwidth: float,
    fixed_bandwidth: np.ndarray | None = None,
    *,
    binary_iterations: int = 100,
) -> ContinuousTarget:
    demand = np.asarray(demand, dtype=float)
    n_tx = np.asarray(n_tx, dtype=float)
    n_rx = np.asarray(n_rx, dtype=float)
    fixed = np.zeros_like(demand) if fixed_bandwidth is None else np.asarray(fixed_bandwidth, dtype=float)
    _validate_inputs(demand, n_tx, n_rx, unit_bandwidth, fixed)
    if not np.any(demand > 0):
        zero = np.zeros_like(demand)
        return ContinuousTarget(zero, 0.0, zero.sum(axis=1), zero.sum(axis=0), "zero-demand")

    if np.allclose(fixed, 0.0):
        outgoing = demand.sum(axis=1)
        incoming = demand.sum(axis=0)
        if np.any((outgoing > 0) & (n_tx <= 0)) or np.any((incoming > 0) & (n_rx <= 0)):
            raise ValueError("positive demand touches an endpoint with zero channel inventory")
        tx_theta = np.divide(
            outgoing,
            unit_bandwidth * n_tx,
            out=np.zeros_like(outgoing),
            where=n_tx > 0,
        )
        rx_theta = np.divide(
            incoming,
            unit_bandwidth * n_rx,
            out=np.zeros_like(incoming),
            where=n_rx > 0,
        )
        theta = float(max(np.max(tx_theta), np.max(rx_theta)))
        target = demand / (unit_bandwidth * theta)
        target[demand <= 0] = 0.0
        np.fill_diagonal(target, 0.0)
        if not _feasible(target, n_tx, n_rx):
            raise AssertionError("closed-form target violates an endpoint constraint")
        return ContinuousTarget(target, theta, target.sum(axis=1), target.sum(axis=0), "closed_form")

    positive = demand > 0
    fixed_only = np.full_like(demand, np.inf, dtype=float)
    mask = positive & (fixed > 0)
    fixed_only[mask] = demand[mask] / fixed[mask]
    high = float(np.max(fixed_only[mask])) if np.any(mask) else 1.0
    high = max(high, 1e-12)
    while not _feasible(target_for_theta(demand, high, unit_bandwidth, fixed), n_tx, n_rx):
        high *= 2.0
        if not np.isfinite(high):
            raise ValueError("could not bracket a feasible completion time")
    low = 0.0
    for _ in range(binary_iterations):
        mid = (low + high) / 2.0
        if _feasible(target_for_theta(demand, mid, unit_bandwidth, fixed), n_tx, n_rx):
            high = mid
        else:
            low = mid
    target = target_for_theta(demand, high, unit_bandwidth, fixed)
    return ContinuousTarget(target, high, target.sum(axis=1), target.sum(axis=0), "binary_search")


def solve_continuous_target_numerical(
    demand: np.ndarray,
    n_tx: np.ndarray,
    n_rx: np.ndarray,
    unit_bandwidth: float,
    fixed_bandwidth: np.ndarray | None = None,
) -> ContinuousTarget:
    """Independent bisection path used as a numerical validation reference."""

    demand = np.asarray(demand, dtype=float)
    fixed = np.zeros_like(demand) if fixed_bandwidth is None else np.asarray(fixed_bandwidth, dtype=float)
    n_tx = np.asarray(n_tx, dtype=float)
    n_rx = np.asarray(n_rx, dtype=float)
    _validate_inputs(demand, n_tx, n_rx, unit_bandwidth, fixed)
    if not np.any(demand > 0):
        return solve_continuous_target(demand, n_tx, n_rx, unit_bandwidth, fixed)
    high = 1.0
    while not _feasible(target_for_theta(demand, high, unit_bandwidth, fixed), n_tx, n_rx):
        high *= 2.0
    low = 0.0
    for _ in range(160):
        mid = (low + high) / 2.0
        if _feasible(target_for_theta(demand, mid, unit_bandwidth, fixed), n_tx, n_rx):
            high = mid
        else:
            low = mid
    target = target_for_theta(demand, high, unit_bandwidth, fixed)
    return ContinuousTarget(target, high, target.sum(axis=1), target.sum(axis=0), "numerical_bisection")


def solve_symmetric_continuous_target(
    demand: np.ndarray,
    n_tx: np.ndarray,
    n_rx: np.ndarray,
    unit_bandwidth: float,
    fixed_bandwidth: np.ndarray | None = None,
) -> ContinuousTarget:
    """Minimum symmetric target for the fair Sym-OCS baseline."""

    demand = np.asarray(demand, dtype=float)
    fixed = np.zeros_like(demand) if fixed_bandwidth is None else np.asarray(fixed_bandwidth, dtype=float)
    n_tx = np.asarray(n_tx, dtype=float)
    n_rx = np.asarray(n_rx, dtype=float)
    _validate_inputs(demand, n_tx, n_rx, unit_bandwidth, fixed)

    def candidate(theta: float) -> np.ndarray:
        directed = target_for_theta(demand, theta, unit_bandwidth, fixed)
        symmetric = np.maximum(directed, directed.T)
        np.fill_diagonal(symmetric, 0.0)
        return symmetric

    if not np.any(demand > 0):
        zero = np.zeros_like(demand)
        return ContinuousTarget(zero, 0.0, zero.sum(axis=1), zero.sum(axis=0), "symmetric-zero")
    high = 1.0
    while not _feasible(candidate(high), n_tx, n_rx):
        high *= 2.0
    low = 0.0
    for _ in range(120):
        mid = (low + high) / 2.0
        if _feasible(candidate(mid), n_tx, n_rx):
            high = mid
        else:
            low = mid
    target = candidate(high)
    return ContinuousTarget(target, high, target.sum(axis=1), target.sum(axis=0), "symmetric_binary_search")
