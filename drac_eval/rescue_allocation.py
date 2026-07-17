from __future__ import annotations

from time import perf_counter
from typing import Dict, Tuple

import numpy as np

from .allocation import AllocationResult, _realize_asymmetric, allocate_for_algorithm
from .config import NetworkConfig
from .metrics import _completion_time_ms


def validate_units(units: np.ndarray, net: NetworkConfig, reachable: np.ndarray | None = None) -> None:
    if not np.issubdtype(units.dtype, np.integer) or np.any(units < 0):
        raise AssertionError("OCS connection units must be non-negative integers")
    if np.any(np.diag(units) != 0):
        raise AssertionError("self connections are not reachable")
    if np.any(units.sum(axis=1) > int(net.per_node_port_budget)):
        raise AssertionError("outbound port budget violated")
    if np.any(units.sum(axis=0) > int(net.per_node_port_budget)):
        raise AssertionError("inbound port budget violated")
    if int(units.sum()) > int(net.total_ocs_links):
        raise AssertionError("total OCS link budget violated")
    if reachable is not None and np.any(units[~reachable] != 0):
        raise AssertionError("reachability constraint violated")


def proportional_target(demand: np.ndarray, net: NetworkConfig) -> np.ndarray:
    target = np.maximum(demand, 0.0).astype(float)
    np.fill_diagonal(target, 0.0)
    node_cap = float(net.per_node_port_budget) * float(net.ocs_unit_bw_gbps)
    global_cap = float(net.total_ocs_links) * float(net.ocs_unit_bw_gbps)
    for _ in range(200):
        old = target.copy()
        rows = target.sum(axis=1)
        target *= np.minimum(1.0, np.divide(node_cap, rows, out=np.ones_like(rows), where=rows > 0))[:, None]
        cols = target.sum(axis=0)
        target *= np.minimum(1.0, np.divide(node_cap, cols, out=np.ones_like(cols), where=cols > 0))[None, :]
        total = float(target.sum())
        if total > global_cap > 0.0:
            target *= global_cap / total
        if np.allclose(old, target, rtol=1e-10, atol=1e-12):
            break
    return target


def allocate_proportional(demand: np.ndarray, net: NetworkConfig) -> AllocationResult:
    target = proportional_target(demand, net)
    units, overlay = _realize_asymmetric(target, net)
    validate_units(units, net)
    base = np.full_like(demand, float(net.base_bw_gbps), dtype=float)
    np.fill_diagonal(base, 0.0)
    return AllocationResult("proportional_makespan", target, overlay, base + overlay, units, {"used_ocs_links": int(units.sum())})


def allocate_discrete_makespan_opt(demand: np.ndarray, net: NetworkConfig) -> Tuple[AllocationResult, float]:
    start = perf_counter()
    n = demand.shape[0]
    base = float(net.base_bw_gbps)
    unit = float(net.ocs_unit_bw_gbps)
    max_units = min(int(net.per_node_port_budget), int(net.total_ocs_links))
    bytes_per_ms_per_gbps = 1e9 / 8.0 / 1000.0
    candidates = {0.0}
    for i in range(n):
        for j in range(n):
            if i == j or demand[i, j] <= 0.0:
                continue
            for k in range(max_units + 1):
                candidates.add(float(demand[i, j]) / ((base + unit * k) * bytes_per_ms_per_gbps))

    best_units = np.zeros((n, n), dtype=int)
    best_theta = max(candidates)
    for theta in sorted(candidates):
        required = np.zeros((n, n), dtype=int)
        if theta <= 0.0 and np.any(demand > 0.0):
            continue
        needed_bw = demand / max(theta * bytes_per_ms_per_gbps, 1e-300) - base
        required = np.ceil(np.maximum(needed_bw, 0.0) / unit - 1e-12).astype(int)
        np.fill_diagonal(required, 0)
        try:
            validate_units(required, net)
        except AssertionError:
            continue
        best_units, best_theta = required, theta
        break
    overlay = best_units.astype(float) * unit
    base_mat = np.full_like(demand, base, dtype=float)
    np.fill_diagonal(base_mat, 0.0)
    total = base_mat + overlay
    actual = _completion_time_ms(demand, total)
    if actual > best_theta * (1.0 + 1e-9) + 1e-9:
        raise AssertionError("discrete optimum feasibility calculation is inconsistent")
    result = AllocationResult(
        "discrete_makespan_opt",
        overlay.copy(),
        overlay,
        total,
        best_units,
        {"used_ocs_links": int(best_units.sum()), "theta_ms": actual},
    )
    return result, (perf_counter() - start) * 1000.0


def allocate_rescue_method(method: str, demand: np.ndarray, net: NetworkConfig) -> Tuple[AllocationResult, float]:
    start = perf_counter()
    if method == "sqrt_sum_delay":
        alloc = allocate_for_algorithm("drac", demand, net)
    elif method == "proportional_makespan":
        alloc = allocate_proportional(demand, net)
    elif method == "discrete_makespan_opt":
        return allocate_discrete_makespan_opt(demand, net)
    else:
        raise ValueError(method)
    validate_units(alloc.connection_units, net)
    return alloc, (perf_counter() - start) * 1000.0
