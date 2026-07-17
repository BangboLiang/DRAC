from __future__ import annotations

from math import ceil
from typing import Dict, Tuple

import numpy as np

from .rescue_config import RescueConfig
from .traffic import validate_demand_matrix


LEVEL_ORDER = {"endpoint": 0, "server": 1, "tor": 2, "aggregation": 3}


def level_group_size(cfg: RescueConfig, level: str) -> int:
    sizes = {
        "endpoint": 1,
        "server": cfg.endpoints_per_server,
        "tor": cfg.endpoints_per_server * cfg.servers_per_tor,
        "aggregation": cfg.endpoints_per_server * cfg.servers_per_tor * cfg.tors_per_aggregation,
    }
    if level not in sizes:
        raise ValueError(f"unknown aggregation level: {level}")
    return max(1, int(sizes[level]))


def build_mapping(n: int, group_size: int, strategy: str, seed: int) -> np.ndarray:
    groups = int(ceil(n / max(1, group_size)))
    if groups >= n:
        return np.arange(n, dtype=int)
    if strategy == "contiguous":
        return np.minimum(np.arange(n, dtype=int) // group_size, groups - 1)
    if strategy == "round_robin":
        return np.arange(n, dtype=int) % groups
    if strategy == "random":
        rng = np.random.default_rng(seed)
        labels = np.minimum(np.arange(n, dtype=int) // group_size, groups - 1)
        rng.shuffle(labels)
        return labels
    raise ValueError(f"unknown mapping strategy: {strategy}")


def aggregate_matrix(endpoint_matrix: np.ndarray, mapping: np.ndarray) -> Tuple[np.ndarray, float]:
    validate_demand_matrix(endpoint_matrix)
    if len(mapping) != endpoint_matrix.shape[0]:
        raise ValueError("mapping length does not match endpoint matrix")
    unique = sorted(set(int(v) for v in mapping))
    remap = {old: new for new, old in enumerate(unique)}
    out = np.zeros((len(unique), len(unique)), dtype=float)
    expected_cross = 0.0
    for u in range(endpoint_matrix.shape[0]):
        for v in range(endpoint_matrix.shape[1]):
            if u == v or mapping[u] == mapping[v]:
                continue
            value = float(endpoint_matrix[u, v])
            expected_cross += value
            out[remap[int(mapping[u])], remap[int(mapping[v])]] += value
    np.fill_diagonal(out, 0.0)
    if not np.isclose(float(out.sum()), expected_cross, rtol=1e-10, atol=1e-6):
        raise AssertionError("cross-boundary traffic was not conserved during aggregation")
    validate_demand_matrix(out)
    return out, expected_cross


def directional_opportunity(matrix: np.ndarray) -> float:
    validate_demand_matrix(matrix)
    numer = 0.0
    denom = 0.0
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            a, b = float(matrix[i, j]), float(matrix[j, i])
            numer += abs(a - b)
            denom += a + b
    if denom <= 0.0:
        return float("nan")
    value = numer / denom
    if value < -1e-12 or value > 1.0 + 1e-12:
        raise AssertionError(f"Omega outside [0,1]: {value}")
    return float(np.clip(value, 0.0, 1.0))


def skew_statistics(matrix: np.ndarray, epsilon: float = 1e-9) -> Dict[str, float]:
    values = []
    weights = []
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            a, b = float(matrix[i, j]), float(matrix[j, i])
            if a + b <= 0.0:
                continue
            values.append(max(a, b) / (min(a, b) + epsilon))
            weights.append(a + b)
    if not values:
        return {"rho_mean": float("nan"), "rho_weighted": float("nan"), "rho_max": float("nan")}
    return {
        "rho_mean": float(np.mean(values)),
        "rho_weighted": float(np.average(values, weights=weights)),
        "rho_max": float(np.max(values)),
    }


def pairwise_balancing_oracle(matrix: np.ndarray) -> np.ndarray:
    out = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(out, 0.0)
    if not np.isclose(float(out.sum()), float(matrix.sum()), rtol=1e-12, atol=1e-6):
        raise AssertionError("oracle changed total payload")
    return out


def _fit_margins(seed: np.ndarray, row_target: np.ndarray, col_target: np.ndarray) -> np.ndarray:
    out = np.maximum(seed, 0.0).copy()
    np.fill_diagonal(out, 0.0)
    positive = ~np.eye(out.shape[0], dtype=bool)
    out[positive] += max(float(out.sum()), 1.0) * 1e-15
    for _ in range(2000):
        row = out.sum(axis=1)
        factors = np.divide(row_target, row, out=np.ones_like(row), where=row > 0)
        out *= factors[:, None]
        col = out.sum(axis=0)
        factors = np.divide(col_target, col, out=np.ones_like(col), where=col > 0)
        out *= factors[None, :]
        np.fill_diagonal(out, 0.0)
        if np.allclose(out.sum(axis=1), row_target, rtol=1e-8, atol=1e-4) and np.allclose(out.sum(axis=0), col_target, rtol=1e-8, atol=1e-4):
            break
    return out


def bidirectional_balanced(matrix: np.ndarray, chunks: int) -> np.ndarray:
    """Chunk-routing abstraction introduced by run_rescue_experiments.py.

    floor(K/2) chunks use each orientation; for odd K the extra chunk keeps the
    original orientation. Iterative proportional fitting restores every rank's
    original total send and receive volume, leaving residual skew when those
    endpoint margins are incompatible with exact pairwise symmetry.
    """
    if chunks < 2:
        raise ValueError("bidirectional_chunks must be at least 2")
    balanced_fraction = 2.0 * float(chunks // 2) / float(chunks)
    seed = (1.0 - balanced_fraction) * matrix + balanced_fraction * pairwise_balancing_oracle(matrix)
    out = _fit_margins(seed, matrix.sum(axis=1), matrix.sum(axis=0))
    if not np.isclose(float(out.sum()), float(matrix.sum()), rtol=1e-8, atol=1e-4):
        raise AssertionError("bidirectional balancing changed total payload")
    if not np.allclose(out.sum(axis=1), matrix.sum(axis=1), rtol=1e-7, atol=1e-3):
        raise AssertionError("bidirectional balancing changed endpoint send totals")
    if not np.allclose(out.sum(axis=0), matrix.sum(axis=0), rtol=1e-7, atol=1e-3):
        raise AssertionError("bidirectional balancing changed endpoint receive totals")
    np.fill_diagonal(out, 0.0)
    return out


def apply_collective_model(matrix: np.ndarray, model: str, chunks: int) -> np.ndarray:
    if model == "original":
        return matrix.copy()
    if model == "bidirectional_balanced":
        return bidirectional_balanced(matrix, chunks)
    if model == "pairwise_balancing_oracle":
        return pairwise_balancing_oracle(matrix)
    raise ValueError(f"unknown collective model: {model}")
