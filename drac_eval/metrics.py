from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .allocation import AllocationResult, EPS
from .config import NetworkConfig
from .traffic import directional_skew_values


@dataclass
class SegmentMetrics:
    completion_time_ms: float
    matching_error_l1: float
    matching_error_p95: float
    network_utilization: float
    ocs_port_utilization: float
    wasted_idle_capacity_gbps: float
    symmetric_waste_gbps: float
    active_directional_ports: int
    releasable_directional_ports: int
    requested_extra_bw_gbps: float
    skew_p50: float
    skew_p95: float


def _completion_time_ms(demand: np.ndarray, capacity: np.ndarray) -> float:
    ratios = np.zeros_like(demand, dtype=float)
    mask = demand > 0
    ratios[mask] = demand[mask] / np.maximum(capacity[mask], EPS)
    if not np.any(mask):
        return 0.0
    return float(np.max(ratios))


def _matching_errors(target: np.ndarray, realized: np.ndarray) -> tuple[float, float]:
    diff = np.abs(target - realized)
    values = diff[np.where(~np.eye(diff.shape[0], dtype=bool))]
    if values.size == 0:
        return 0.0, 0.0
    return float(np.sum(values)), float(np.percentile(values, 95))


def _network_utilization(demand: np.ndarray, capacity: np.ndarray, completion: float) -> float:
    if completion <= 0.0:
        return 0.0
    capacity_total = float(np.sum(capacity))
    if capacity_total <= 0.0:
        return 0.0
    return float(np.sum(demand) / (completion * capacity_total))


def _wasted_idle_capacity(demand: np.ndarray, capacity: np.ndarray, completion: float) -> float:
    if completion <= 0.0:
        return 0.0
    required = demand / max(completion, EPS)
    waste = np.maximum(0.0, capacity - required)
    np.fill_diagonal(waste, 0.0)
    return float(np.sum(waste))


def _symmetric_waste(demand: np.ndarray, realized_overlay: np.ndarray, unit_bw: float) -> float:
    waste = 0.0
    n = demand.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            a = float(demand[i, j])
            b = float(demand[j, i])
            if a <= EPS and b <= EPS:
                continue
            if abs(realized_overlay[i, j] - realized_overlay[j, i]) > EPS:
                low_dir = (i, j) if a < b else (j, i)
                low_demand = min(a, b)
                low_cap = float(realized_overlay[low_dir[0], low_dir[1]])
                high_cap = float(realized_overlay[low_dir[1], low_dir[0]])
                waste += max(0.0, min(low_cap, high_cap) - low_demand / max(unit_bw, 1.0))
            else:
                low_demand = min(a, b)
                waste += max(0.0, float(realized_overlay[i, j]) - low_demand)
    return float(waste)


def _port_utilization(units: np.ndarray, budget: int) -> tuple[float, int]:
    used_out = units.sum(axis=1)
    used_in = units.sum(axis=0)
    active = int(used_out.sum() + used_in.sum())
    capacity = max(1, int(budget) * units.shape[0] * 2)
    return float(active / capacity), active


def compute_segment_metrics(
    demand: np.ndarray,
    allocation: AllocationResult,
    net: NetworkConfig,
) -> SegmentMetrics:
    completion = _completion_time_ms(demand, allocation.total_bandwidth)
    match_l1, match_p95 = _matching_errors(
        allocation.target_overlay, allocation.realized_overlay
    )
    util = _network_utilization(demand, allocation.total_bandwidth, completion)
    port_util, active_ports = _port_utilization(
        allocation.connection_units, int(net.per_node_port_budget)
    )
    reserved_dir = int(
        net.directional_port_reserved
        if net.directional_port_reserved is not None
        else net.per_node_port_budget
    )
    total_reserved_dir = demand.shape[0] * reserved_dir * 2
    requested_extra = float(
        np.maximum(0.0, allocation.target_overlay - allocation.realized_overlay).sum()
    )
    skews = directional_skew_values(demand)
    return SegmentMetrics(
        completion_time_ms=completion,
        matching_error_l1=match_l1,
        matching_error_p95=match_p95,
        network_utilization=util,
        ocs_port_utilization=port_util,
        wasted_idle_capacity_gbps=_wasted_idle_capacity(
            demand, allocation.total_bandwidth, completion
        ),
        symmetric_waste_gbps=_symmetric_waste(
            demand, allocation.realized_overlay, float(net.ocs_unit_bw_gbps)
        ),
        active_directional_ports=active_ports,
        releasable_directional_ports=max(0, total_reserved_dir - active_ports),
        requested_extra_bw_gbps=requested_extra,
        skew_p50=float(np.percentile(skews, 50)) if skews.size else 1.0,
        skew_p95=float(np.percentile(skews, 95)) if skews.size else 1.0,
    )


def aggregate_port_exposure(
    allocations: List[AllocationResult], net: NetworkConfig
) -> Dict[str, float]:
    if not allocations:
        return {
            "active_directional_ports": 0.0,
            "releasable_directional_ports": 0.0,
            "active_bidirectional_bundles": 0.0,
            "releasable_bidirectional_bundles": 0.0,
        }
    max_out = np.max(np.stack([alloc.connection_units.sum(axis=1) for alloc in allocations]), axis=0)
    max_in = np.max(np.stack([alloc.connection_units.sum(axis=0) for alloc in allocations]), axis=0)
    active_dir = float(np.sum(max_out) + np.sum(max_in))
    reserved_dir = float(
        (net.directional_port_reserved or net.per_node_port_budget) * len(max_out) * 2
    )
    active_bundles = float(np.sum(np.maximum(max_out, max_in)))
    reserved_bundles = float(
        (net.bidirectional_bundle_reserved or net.per_node_port_budget) * len(max_out)
    )
    return {
        "active_directional_ports": active_dir,
        "releasable_directional_ports": max(0.0, reserved_dir - active_dir),
        "active_bidirectional_bundles": active_bundles,
        "releasable_bidirectional_bundles": max(0.0, reserved_bundles - active_bundles),
    }
