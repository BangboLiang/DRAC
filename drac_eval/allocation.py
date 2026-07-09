from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .config import NetworkConfig


EPS = 1e-12


@dataclass
class AllocationResult:
    algorithm: str
    target_overlay: np.ndarray
    realized_overlay: np.ndarray
    total_bandwidth: np.ndarray
    connection_units: np.ndarray
    metadata: Dict[str, float | int | str]


def _overlay_bw_budget(net: NetworkConfig) -> float:
    return float(net.total_ocs_links) * float(net.ocs_unit_bw_gbps)


def _base_bandwidth_matrix(n: int, net: NetworkConfig) -> np.ndarray:
    mat = np.full((n, n), float(net.base_bw_gbps), dtype=float)
    np.fill_diagonal(mat, 0.0)
    return mat


def _sqrt_share_matrix(demand: np.ndarray, budget: float) -> np.ndarray:
    weights = np.sqrt(np.maximum(demand, 0.0))
    total = float(weights.sum())
    if total <= 0.0 or budget <= 0.0:
        return np.zeros_like(demand, dtype=float)
    out = budget * weights / total
    np.fill_diagonal(out, 0.0)
    return out


def _symmetric_target(demand: np.ndarray, budget: float) -> np.ndarray:
    n = demand.shape[0]
    pair_weight = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            pair_weight[i, j] = np.sqrt(float(demand[i, j] + demand[j, i]))
    total = float(pair_weight.sum())
    out = np.zeros_like(demand, dtype=float)
    if total <= 0.0 or budget <= 0.0:
        return out
    pair_budget = budget / 2.0
    for i in range(n):
        for j in range(i + 1, n):
            bw = pair_budget * pair_weight[i, j] / total
            out[i, j] = bw
            out[j, i] = bw
    return out


def _remaining_ports(units: np.ndarray, net: NetworkConfig) -> Tuple[np.ndarray, np.ndarray]:
    used_out = units.sum(axis=1)
    used_in = units.sum(axis=0)
    remain_out = np.maximum(0, int(net.per_node_port_budget) - used_out).astype(int)
    remain_in = np.maximum(0, int(net.per_node_port_budget) - used_in).astype(int)
    return remain_out, remain_in


def _realize_asymmetric(target: np.ndarray, net: NetworkConfig) -> Tuple[np.ndarray, np.ndarray]:
    n = target.shape[0]
    unit_bw = float(net.ocs_unit_bw_gbps)
    units = np.zeros((n, n), dtype=int)
    realized = np.zeros((n, n), dtype=float)
    if unit_bw <= 0.0 or int(net.total_ocs_links) <= 0:
        return units, realized

    candidates: List[Tuple[int, int, float]] = []
    for i in range(n):
        for j in range(n):
            if i == j or target[i, j] <= 0.0:
                continue
            candidates.append((i, j, float(target[i, j])))
    candidates.sort(key=lambda item: item[2], reverse=True)

    remaining_links = int(net.total_ocs_links)
    for i, j, value in candidates:
        if remaining_links <= 0:
            break
        want = int(np.floor(value / unit_bw + EPS))
        if want <= 0:
            continue
        remain_out, remain_in = _remaining_ports(units, net)
        take = min(want, int(remain_out[i]), int(remain_in[j]), remaining_links)
        if take <= 0:
            continue
        units[i, j] += take
        remaining_links -= take

    while remaining_links > 0:
        remain_out, remain_in = _remaining_ports(units, net)
        best_pair: tuple[int, int] | None = None
        best_gap = 0.0
        for i, j, value in candidates:
            if remain_out[i] <= 0 or remain_in[j] <= 0:
                continue
            gap = value - float(units[i, j]) * unit_bw
            if gap > best_gap + EPS:
                best_gap = gap
                best_pair = (i, j)
        if best_pair is None or best_gap <= EPS:
            break
        units[best_pair[0], best_pair[1]] += 1
        remaining_links -= 1

    realized = units.astype(float) * unit_bw
    np.fill_diagonal(realized, 0.0)
    return units, realized


def _realize_symmetric(target: np.ndarray, net: NetworkConfig) -> Tuple[np.ndarray, np.ndarray]:
    n = target.shape[0]
    unit_bw = float(net.ocs_unit_bw_gbps)
    units = np.zeros((n, n), dtype=int)
    realized = np.zeros((n, n), dtype=float)
    if unit_bw <= 0.0 or int(net.total_ocs_links) <= 1:
        return units, realized

    candidates: List[Tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            score = float(target[i, j] + target[j, i])
            if score > 0.0:
                candidates.append((i, j, score))
    candidates.sort(key=lambda item: item[2], reverse=True)

    remaining_links = int(net.total_ocs_links)
    for i, j, score in candidates:
        if remaining_links < 2:
            break
        want = int(np.floor(max(target[i, j], target[j, i]) / unit_bw + EPS))
        if want <= 0:
            continue
        remain_out, remain_in = _remaining_ports(units, net)
        take = min(
            want,
            int(remain_out[i]),
            int(remain_in[i]),
            int(remain_out[j]),
            int(remain_in[j]),
            remaining_links // 2,
        )
        if take <= 0:
            continue
        units[i, j] += take
        units[j, i] += take
        remaining_links -= 2 * take

    while remaining_links >= 2:
        remain_out, remain_in = _remaining_ports(units, net)
        best_pair: tuple[int, int] | None = None
        best_gap = 0.0
        for i, j, _score in candidates:
            if (
                remain_out[i] <= 0
                or remain_in[i] <= 0
                or remain_out[j] <= 0
                or remain_in[j] <= 0
            ):
                continue
            gap = max(target[i, j], target[j, i]) - float(units[i, j]) * unit_bw
            if gap > best_gap + EPS:
                best_gap = gap
                best_pair = (i, j)
        if best_pair is None or best_gap <= EPS:
            break
        units[best_pair[0], best_pair[1]] += 1
        units[best_pair[1], best_pair[0]] += 1
        remaining_links -= 2

    realized = units.astype(float) * unit_bw
    np.fill_diagonal(realized, 0.0)
    return units, realized


def allocate_for_algorithm(
    algorithm: str,
    demand: np.ndarray,
    net: NetworkConfig,
    static_target: np.ndarray | None = None,
) -> AllocationResult:
    n = demand.shape[0]
    base = _base_bandwidth_matrix(n, net)
    overlay_budget = _overlay_bw_budget(net)

    if algorithm == "static_sym":
        target = static_target if static_target is not None else _symmetric_target(demand, overlay_budget)
        units, realized = _realize_symmetric(target, net)
    elif algorithm == "sym_ocs":
        target = _symmetric_target(demand, overlay_budget)
        units, realized = _realize_symmetric(target, net)
    elif algorithm == "drac":
        target = _sqrt_share_matrix(demand, overlay_budget)
        units, realized = _realize_asymmetric(target, net)
    elif algorithm == "ideal_asym":
        target = _sqrt_share_matrix(demand, overlay_budget)
        units = np.rint(target / max(net.ocs_unit_bw_gbps, EPS)).astype(int)
        realized = target.copy()
    elif algorithm == "drac_sym":
        target = _sqrt_share_matrix(demand, overlay_budget)
        units, realized = _realize_symmetric(target, net)
    else:
        raise ValueError(f"unknown algorithm: {algorithm}")

    total = base + realized
    np.fill_diagonal(total, 0.0)
    meta: Dict[str, float | int | str] = {
        "overlay_budget_gbps": overlay_budget,
        "base_bw_gbps": float(net.base_bw_gbps),
        "used_ocs_links": int(units.sum()),
    }
    return AllocationResult(
        algorithm=algorithm,
        target_overlay=target.copy(),
        realized_overlay=realized,
        total_bandwidth=total,
        connection_units=units,
        metadata=meta,
    )
