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
    useful_capacity_gbps: float
    wasted_idle_capacity_gbps: float
    total_provisioned_capacity_gbps: float
    useful_ratio: float
    waste_ratio: float
    symmetric_waste_gbps: float
    active_directional_ports: int
    releasable_directional_ports: int
    requested_extra_bw_gbps: float
    skew_p50: float
    skew_p95: float


def _gbps_to_bytes_per_ms(bandwidth_gbps: np.ndarray | float) -> np.ndarray | float:
    # Interpret config bandwidth in Gbps; convert to bytes/ms.
    return np.asarray(bandwidth_gbps, dtype=float) * (1e9 / 8.0 / 1000.0)


def _bytes_per_ms_to_gbps(rate_bytes_per_ms: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(rate_bytes_per_ms, dtype=float) * (8.0 * 1000.0 / 1e9)


def _completion_time_ms(demand: np.ndarray, capacity_gbps: np.ndarray) -> float:
    ratios = np.zeros_like(demand, dtype=float)
    mask = demand > 0
    capacity_bytes_per_ms = _gbps_to_bytes_per_ms(capacity_gbps)
    ratios[mask] = demand[mask] / np.maximum(capacity_bytes_per_ms[mask], EPS)
    if not np.any(mask):
        return 0.0
    return float(np.max(ratios))


def _matching_errors(target: np.ndarray, realized: np.ndarray) -> tuple[float, float]:
    diff = np.abs(target - realized)
    values = diff[np.where(~np.eye(diff.shape[0], dtype=bool))]
    if values.size == 0:
        return 0.0, 0.0
    return float(np.sum(values)), float(np.percentile(values, 95))


def _network_utilization(
    demand: np.ndarray, capacity_gbps: np.ndarray, completion: float
) -> float:
    if completion <= 0.0:
        return 0.0
    capacity_total = float(np.sum(_gbps_to_bytes_per_ms(capacity_gbps)))
    if capacity_total <= 0.0:
        return 0.0
    return float(np.sum(demand) / (completion * capacity_total))


def _wasted_idle_capacity(
    demand: np.ndarray, capacity_gbps: np.ndarray, completion: float
) -> float:
    if completion <= 0.0:
        return 0.0
    capacity_bytes_per_ms = _gbps_to_bytes_per_ms(capacity_gbps)
    required = demand / max(completion, EPS)
    waste = np.maximum(0.0, capacity_bytes_per_ms - required)
    np.fill_diagonal(waste, 0.0)
    return float(np.sum(_bytes_per_ms_to_gbps(waste)))


def _useful_capacity(
    demand: np.ndarray, capacity_gbps: np.ndarray, completion: float
) -> float:
    if completion <= 0.0:
        return 0.0
    capacity_bytes_per_ms = _gbps_to_bytes_per_ms(capacity_gbps)
    required = demand / max(completion, EPS)
    useful = np.minimum(capacity_bytes_per_ms, required)
    np.fill_diagonal(useful, 0.0)
    return float(np.sum(_bytes_per_ms_to_gbps(useful)))


def _symmetric_waste(
    demand: np.ndarray, realized_overlay_gbps: np.ndarray, completion_ms: float
) -> float:
    if completion_ms <= 0.0:
        return 0.0
    waste = 0.0
    n = demand.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            a = float(demand[i, j])
            b = float(demand[j, i])
            if a <= EPS and b <= EPS:
                continue
            if abs(realized_overlay_gbps[i, j] - realized_overlay_gbps[j, i]) > EPS:
                low_dir = (i, j) if a < b else (j, i)
                low_demand = min(a, b)
                low_cap = float(realized_overlay_gbps[low_dir[0], low_dir[1]])
                high_cap = float(realized_overlay_gbps[low_dir[1], low_dir[0]])
                required_gbps = float(
                    _bytes_per_ms_to_gbps(low_demand / max(completion_ms, EPS))
                )
                waste += max(0.0, min(low_cap, high_cap) - required_gbps)
            else:
                low_demand = min(a, b)
                required_gbps = float(
                    _bytes_per_ms_to_gbps(low_demand / max(completion_ms, EPS))
                )
                waste += max(0.0, float(realized_overlay_gbps[i, j]) - required_gbps)
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
    total_provisioned_capacity_gbps = float(np.sum(allocation.total_bandwidth))
    useful_capacity_gbps = _useful_capacity(demand, allocation.total_bandwidth, completion)
    wasted_idle_capacity_gbps = _wasted_idle_capacity(
        demand, allocation.total_bandwidth, completion
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
        useful_capacity_gbps=useful_capacity_gbps,
        wasted_idle_capacity_gbps=wasted_idle_capacity_gbps,
        total_provisioned_capacity_gbps=total_provisioned_capacity_gbps,
        useful_ratio=(
            useful_capacity_gbps / total_provisioned_capacity_gbps
            if total_provisioned_capacity_gbps > 0.0
            else 0.0
        ),
        waste_ratio=(
            wasted_idle_capacity_gbps / total_provisioned_capacity_gbps
            if total_provisioned_capacity_gbps > 0.0
            else 0.0
        ),
        symmetric_waste_gbps=_symmetric_waste(
            demand, allocation.realized_overlay, completion
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
