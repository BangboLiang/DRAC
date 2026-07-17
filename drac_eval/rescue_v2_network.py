from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from time import perf_counter
from typing import Iterable, Tuple

import numpy as np

from .allocation import AllocationResult, _sqrt_share_matrix
from .metrics import _completion_time_ms
from .rescue_v2_config import RescueV2Config


@dataclass
class AggregatedNetworkModel:
    base_capacity_gbps: np.ndarray
    out_port_budget: np.ndarray
    in_port_budget: np.ndarray
    reachable: np.ndarray
    total_ocs_links: int
    ocs_unit_bw_gbps: float
    normalization_mode: str
    level: str

    @property
    def node_count(self) -> int:
        return int(self.base_capacity_gbps.shape[0])


def endpoint_network_model(cfg: RescueV2Config, n: int, port_budget: int | None = None) -> AggregatedNetworkModel:
    base = np.full((n, n), cfg.endpoint_base_capacity_gbps, dtype=float)
    np.fill_diagonal(base, 0.0)
    reachable = ~np.eye(n, dtype=bool)
    out_budget = int(port_budget if port_budget is not None else cfg.endpoint_out_port_budget)
    in_budget = int(port_budget if port_budget is not None else cfg.endpoint_in_port_budget)
    return AggregatedNetworkModel(
        base, np.full(n, out_budget, dtype=int), np.full(n, in_budget, dtype=int),
        reachable, int(cfg.global_ocs_budget), cfg.ocs_unit_bw_gbps,
        "endpoint", "endpoint",
    )


def aggregate_network_model(
    endpoint_model: AggregatedNetworkModel,
    mapping: np.ndarray,
    mode: str,
    level: str,
    cfg: RescueV2Config,
) -> AggregatedNetworkModel:
    labels = sorted(set(int(v) for v in mapping))
    index = {v: i for i, v in enumerate(labels)}
    m = len(labels)
    if mode == "deployment_specific":
        if level not in cfg.deployment_specific:
            raise ValueError(f"deployment_specific parameters missing for level {level}")
        spec = cfg.deployment_specific[level]
        base = np.full((m, m), float(spec.base_bw_gbps), dtype=float)
        np.fill_diagonal(base, 0.0)
        return AggregatedNetworkModel(
            base,
            np.full(m, spec.out_port_budget, dtype=int),
            np.full(m, spec.in_port_budget, dtype=int),
            ~np.eye(m, dtype=bool),
            int(spec.total_ocs_links),
            cfg.ocs_unit_bw_gbps,
            mode,
            level,
        )
    if mode != "resource_equivalent":
        raise ValueError(mode)
    base = np.zeros((m, m), dtype=float)
    reachable = np.zeros((m, m), dtype=bool)
    out_budget = np.zeros(m, dtype=int)
    in_budget = np.zeros(m, dtype=int)
    for u, old_a in enumerate(mapping):
        a = index[int(old_a)]
        out_budget[a] += int(endpoint_model.out_port_budget[u])
        in_budget[a] += int(endpoint_model.in_port_budget[u])
        for v, old_b in enumerate(mapping):
            b = index[int(old_b)]
            if a == b:
                continue
            base[a, b] += float(endpoint_model.base_capacity_gbps[u, v])
            reachable[a, b] = reachable[a, b] or bool(endpoint_model.reachable[u, v])
    if not np.isclose(float(base.sum()), sum(float(endpoint_model.base_capacity_gbps[u, v]) for u in range(len(mapping)) for v in range(len(mapping)) if mapping[u] != mapping[v])):
        raise AssertionError("base capacity aggregation is not conservative")
    if int(out_budget.sum()) != int(endpoint_model.out_port_budget.sum()):
        raise AssertionError("outbound aggregate port budget is not conservative")
    if int(in_budget.sum()) != int(endpoint_model.in_port_budget.sum()):
        raise AssertionError("inbound aggregate port budget is not conservative")
    return AggregatedNetworkModel(
        base, out_budget, in_budget, reachable,
        endpoint_model.total_ocs_links, endpoint_model.ocs_unit_bw_gbps,
        mode, level,
    )


def validate_general_units(units: np.ndarray, model: AggregatedNetworkModel) -> None:
    if not np.issubdtype(units.dtype, np.integer) or np.any(units < 0):
        raise AssertionError("units must be non-negative integers")
    if np.any(units[~model.reachable] != 0):
        raise AssertionError("reachability violated")
    if np.any(units.sum(axis=1) > model.out_port_budget):
        raise AssertionError("outbound port budget violated")
    if np.any(units.sum(axis=0) > model.in_port_budget):
        raise AssertionError("inbound port budget violated")
    if int(units.sum()) > int(model.total_ocs_links):
        raise AssertionError("global OCS budget violated")


def _allocation(name: str, target: np.ndarray, units: np.ndarray, model: AggregatedNetworkModel) -> AllocationResult:
    validate_general_units(units, model)
    overlay = units.astype(float) * model.ocs_unit_bw_gbps
    total = model.base_capacity_gbps + overlay
    return AllocationResult(name, target, overlay, total, units, {"used_ocs_links": int(units.sum()), "normalization_mode": model.normalization_mode})


def allocate_drac_makespan_opt(demand: np.ndarray, model: AggregatedNetworkModel) -> Tuple[AllocationResult, float]:
    start = perf_counter()
    byte_rate = 1e9 / 8.0 / 1000.0
    candidates = {0.0}
    max_units = min(int(model.total_ocs_links), int(max(model.out_port_budget.max(initial=0), model.in_port_budget.max(initial=0))))
    for i, j in zip(*np.where(demand > 0.0)):
        if not model.reachable[i, j]:
            continue
        for k in range(max_units + 1):
            cap = float(model.base_capacity_gbps[i, j]) + k * model.ocs_unit_bw_gbps
            if cap > 0.0:
                candidates.add(float(demand[i, j]) / (cap * byte_rate))
    best: np.ndarray | None = None
    for theta in sorted(candidates):
        if theta <= 0.0 and np.any(demand > 0.0):
            continue
        needed = demand / max(theta * byte_rate, 1e-300) - model.base_capacity_gbps
        units = np.ceil(np.maximum(needed, 0.0) / model.ocs_unit_bw_gbps - 1e-12).astype(int)
        units[~model.reachable] = 0
        try:
            validate_general_units(units, model)
        except AssertionError:
            continue
        best = units
        break
    if best is None:
        raise RuntimeError("no feasible discrete makespan solution")
    return _allocation("drac_makespan_opt", best.astype(float) * model.ocs_unit_bw_gbps, best, model), (perf_counter() - start) * 1000.0


def _greedy_asymmetric(target: np.ndarray, model: AggregatedNetworkModel) -> np.ndarray:
    units = np.zeros_like(target, dtype=int)
    candidates = sorted([(i, j, float(target[i, j])) for i, j in zip(*np.where((target > 0) & model.reachable))], key=lambda x: x[2], reverse=True)
    remaining = model.total_ocs_links
    for i, j, value in candidates:
        want = int(np.floor(value / model.ocs_unit_bw_gbps + 1e-12))
        take = min(want, int(model.out_port_budget[i] - units[i].sum()), int(model.in_port_budget[j] - units[:, j].sum()), remaining)
        if take > 0:
            units[i, j] += take
            remaining -= take
    while remaining > 0:
        feasible = [(target[i, j] - units[i, j] * model.ocs_unit_bw_gbps, i, j) for i, j, _ in candidates if units[i].sum() < model.out_port_budget[i] and units[:, j].sum() < model.in_port_budget[j]]
        if not feasible:
            break
        gap, i, j = max(feasible)
        if gap <= 1e-12:
            break
        units[i, j] += 1
        remaining -= 1
    return units


def allocate_target_ablation(method: str, demand: np.ndarray, model: AggregatedNetworkModel) -> Tuple[AllocationResult, float]:
    start = perf_counter()
    budget = model.total_ocs_links * model.ocs_unit_bw_gbps
    if method == "sqrt_target_floor_greedy":
        target = _sqrt_share_matrix(demand, budget)
    elif method == "proportional_target_floor_greedy":
        target = budget * demand / float(demand.sum()) if demand.sum() > 0 else np.zeros_like(demand)
    else:
        raise ValueError(method)
    units = _greedy_asymmetric(target, model)
    return _allocation(method, target, units, model), (perf_counter() - start) * 1000.0


def allocate_sym_ocs(demand: np.ndarray, model: AggregatedNetworkModel) -> AllocationResult:
    pair_scores = []
    for i in range(model.node_count):
        for j in range(i + 1, model.node_count):
            if model.reachable[i, j] and model.reachable[j, i] and demand[i, j] + demand[j, i] > 0:
                pair_scores.append((float(np.sqrt(demand[i, j] + demand[j, i])), i, j))
    target = np.zeros_like(demand, dtype=float)
    score_sum = sum(score for score, _, _ in pair_scores)
    if score_sum > 0.0:
        directional_budget = model.total_ocs_links * model.ocs_unit_bw_gbps / 2.0
        for score, i, j in pair_scores:
            target[i, j] = target[j, i] = directional_budget * score / score_sum
    units = np.zeros_like(demand, dtype=int)
    remaining = model.total_ocs_links
    for _, i, j in sorted(pair_scores, reverse=True):
        if remaining < 2:
            break
        want = int(np.floor(target[i, j] / model.ocs_unit_bw_gbps + 1e-12))
        take = min(
            want,
            int(model.out_port_budget[i] - units[i].sum()),
            int(model.in_port_budget[i] - units[:, i].sum()),
            int(model.out_port_budget[j] - units[j].sum()),
            int(model.in_port_budget[j] - units[:, j].sum()),
            remaining // 2,
        )
        if take > 0:
            units[i, j] += take
            units[j, i] += take
            remaining -= 2 * take
    while remaining >= 2:
        feasible = []
        for _, i, j in pair_scores:
            if units[i].sum() < model.out_port_budget[i] and units[:, i].sum() < model.in_port_budget[i] and units[j].sum() < model.out_port_budget[j] and units[:, j].sum() < model.in_port_budget[j]:
                feasible.append((target[i, j] - units[i, j] * model.ocs_unit_bw_gbps, i, j))
        if not feasible:
            break
        gap, i, j = max(feasible)
        if gap <= 1e-12:
            break
        units[i, j] += 1
        units[j, i] += 1
        remaining -= 2
    return _allocation("sym_ocs", target, units, model)


def brute_force_makespan(demand: np.ndarray, model: AggregatedNetworkModel) -> Tuple[float, np.ndarray]:
    edges = [(i, j) for i, j in zip(*np.where(model.reachable)) if i != j]
    best_time = float("inf")
    best = np.zeros_like(demand, dtype=int)
    max_unit = min(model.total_ocs_links, int(max(model.out_port_budget.max(), model.in_port_budget.max())))
    for values in product(range(max_unit + 1), repeat=len(edges)):
        if sum(values) > model.total_ocs_links:
            continue
        units = np.zeros_like(demand, dtype=int)
        for (i, j), value in zip(edges, values):
            units[i, j] = value
        try:
            validate_general_units(units, model)
        except AssertionError:
            continue
        time = _completion_time_ms(demand, model.base_capacity_gbps + units * model.ocs_unit_bw_gbps)
        if time < best_time - 1e-12:
            best_time, best = time, units.copy()
    return best_time, best
