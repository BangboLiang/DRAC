"""ATLAHS trace-derived directionality analysis introduced by V4."""
from __future__ import annotations

from collections import defaultdict
import csv
import math
from pathlib import Path
import sqlite3
from typing import Iterable

import numpy as np

TRACE = "ATLAHS_TRACE_DERIVED"
SIM = "SIMULATED_TIMELINE"
HYP = "HYPOTHETICAL_AGGREGATION"


def directional_metrics(pair_bytes: dict[tuple[int, int], float], node_count: int) -> dict[str, float]:
    a = v = weighted_skew_num = 0.0
    active = directional = 0
    max_skew = math.nan
    for left in range(node_count):
        for right in range(left + 1, node_count):
            forward = float(pair_bytes.get((left, right), 0.0)); reverse = float(pair_bytes.get((right, left), 0.0))
            total = forward + reverse
            if total <= 0: continue
            active += 1; diff = abs(forward - reverse); a += diff; v += total
            skew = diff / total; weighted_skew_num += total * skew
            max_skew = skew if math.isnan(max_skew) else max(max_skew, skew)
            if skew >= 0.1: directional += 1
    possible = node_count * (node_count - 1) / 2
    return {"A": a, "V": v, "Omega": a / v if v else math.nan,
            "pair_coverage": active / possible if possible else math.nan,
            "active_pair_count": active,
            "directional_pair_fraction_0_1": directional / active if active else math.nan,
            "dominant_direction_mass": a / v if v else math.nan,
            "max_pair_skew": max_skew,
            "traffic_weighted_pair_skew": weighted_skew_num / v if v else math.nan}


def load_sends(database: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    connection = sqlite3.connect(database)
    rows = connection.execute("SELECT rank,peer,bytes_or_ns,sequence,id FROM events WHERE operation_type='send' ORDER BY id").fetchall()
    connection.close()
    if not rows:
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty, empty, empty
    values = np.asarray(rows, dtype=np.int64)
    return tuple(values[:, index] for index in range(5))  # type: ignore[return-value]


def aggregate_pairs(src: np.ndarray, dst: np.ndarray, sizes: np.ndarray, mapping: dict[int, int] | None = None) -> dict[tuple[int, int], float]:
    result: dict[tuple[int, int], float] = defaultdict(float)
    if mapping is None:
        mapping = {int(rank): int(rank) for rank in np.unique(np.concatenate((src, dst)))}
    for a, b, size in zip(src, dst, sizes):
        left, right = mapping[int(a)], mapping[int(b)]
        if left != right: result[(left, right)] += float(size)
    return dict(result)


def mapping_for(nodes: int, group_size: int, strategy: str, seed: int) -> dict[int, int]:
    groups = math.ceil(nodes / group_size)
    if strategy == "contiguous": return {node: node // group_size for node in range(nodes)}
    if strategy == "round_robin": return {node: node % groups for node in range(nodes)}
    if strategy == "random":
        order = np.random.default_rng(seed).permutation(nodes)
        result = {}
        for position, node in enumerate(order): result[int(node)] = position // group_size
        return result
    raise ValueError(strategy)


def persistence(previous: dict[tuple[int, int], float], current: dict[tuple[int, int], float]) -> float:
    unordered = {tuple(sorted(pair)) for pair in previous} | {tuple(sorted(pair)) for pair in current}
    intersection = union = 0.0
    for left, right in unordered:
        p = previous.get((left, right), 0.0) - previous.get((right, left), 0.0)
        c = current.get((left, right), 0.0) - current.get((right, left), 0.0)
        wp, wc = abs(p), abs(c)
        union += max(wp, wc)
        if p * c > 0: intersection += min(wp, wc)
    return intersection / union if union else math.nan


def channel_cancellation(channels: dict[int, dict[tuple[int, int], float]], node_count: int) -> float:
    summed_a = sum(directional_metrics(pairs, node_count)["A"] for pairs in channels.values())
    all_pairs: dict[tuple[int, int], float] = defaultdict(float)
    for pairs in channels.values():
        for pair, value in pairs.items(): all_pairs[pair] += value
    combined = directional_metrics(dict(all_pairs), node_count)["A"]
    return 1.0 - combined / summed_a if summed_a else math.nan


def nonoverlap_windows(src: np.ndarray, dst: np.ndarray, sizes: np.ndarray, starts_ns: np.ndarray,
                       window_ns: int, node_count: int, trace_name: str, link_gbps: float,
                       assignment: str = "start_time", overlap_fraction: float = 0.0) -> list[dict[str, object]]:
    if len(src) == 0: return []
    end = int(starts_ns.max()) + 1
    result = []
    step = max(1, int(window_ns * (1.0 - overlap_fraction)))
    bins: dict[int, dict[tuple[int, int], float]] = defaultdict(lambda: defaultdict(float))
    totals: dict[int, float] = defaultdict(float)
    for a, b, size, start in zip(src, dst, sizes, starts_ns):
        latest = int(start) // step * step
        candidates = (latest,) if overlap_fraction == 0 else (latest, latest-step)
        for window_start in candidates:
            if window_start >= 0 and window_start <= int(start) < window_start + window_ns:
                bins[window_start][(int(a),int(b))] += float(size); totals[window_start] += float(size)
    possible = math.ceil(end / step)
    for window_start in sorted(bins):
        pairs = dict(bins[window_start])
        metrics = directional_metrics(pairs, node_count)
        result.append({"trace_name": trace_name, "evidence_label": SIM, "link_rate_gbps": link_gbps,
                       "window_ns": window_ns, "overlap_fraction": overlap_fraction, "assignment": assignment,
                       "window_start_ns": window_start, "window_end_ns": window_start + window_ns,
                       "cross_node_bytes": totals[window_start], "total_possible_windows": possible,
                       "active_window_coverage": len(bins)/possible if possible else math.nan,
                       **metrics, "pairs": pairs})
    return result


def simulated_starts(src: np.ndarray, sizes: np.ndarray, link_gbps: float) -> np.ndarray:
    """Conservative per-source serialized timeline; not a measured timestamp."""
    cursors: dict[int, float] = defaultdict(float); starts = np.zeros(len(src), dtype=np.int64)
    bits_per_ns = link_gbps
    for index, (rank, size) in enumerate(zip(src, sizes)):
        starts[index] = int(cursors[int(rank)])
        cursors[int(rank)] += float(size) * 8.0 / bits_per_ns
    return starts


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str] | None = None) -> None:
    rows = list(rows); path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def summarize_windows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(row["trace_name"], row["evidence_label"], row["link_rate_gbps"], row["window_ns"], row["overlap_fraction"],
                 row.get("guard_overhead_ns", "NA"), row.get("minimum_useful_payload_bytes", "NA"))].append(row)
    result = []
    for key, group in grouped.items():
        weights = np.array([float(row["V"]) for row in group]); omega = np.array([float(row["Omega"]) for row in group])
        valid = np.isfinite(omega); coverage = float(group[0].get("active_window_coverage", float(valid.mean()) if len(valid) else math.nan))
        vals = omega[valid]; w = weights[valid]
        result.append({"trace_name": key[0], "evidence_label": key[1], "link_rate_gbps": key[2], "window_ns": key[3],
                       "overlap_fraction": key[4], "window_count": len(group), "nonempty_window_coverage": coverage,
                       "guard_overhead_ns": key[5], "minimum_useful_payload_bytes": key[6],
                       "traffic_weighted_omega": float(np.sum(vals*w)/np.sum(w)) if np.sum(w)>0 else math.nan,
                       "median_omega": float(np.median(vals)) if len(vals) else math.nan,
                       "p90_omega": float(np.quantile(vals, .9)) if len(vals) else math.nan,
                       "total_directional_bytes": float(sum(float(row["A"]) for row in group)),
                       "total_cross_node_bytes": float(sum(float(row["V"]) for row in group))})
    return result
