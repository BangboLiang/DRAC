from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from .config import WorkloadConfig
from llama3_comm.config import ModelConfig, ParallelConfig
from llama3_comm.traffic import llama3_megatron_payloads


@dataclass
class SegmentDemand:
    workload: str
    segment_idx: int
    matrix: np.ndarray
    metadata: Dict[str, float | int | str]


def validate_demand_matrix(matrix: np.ndarray) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"demand matrix must be square, got shape {matrix.shape}")
    if np.any(matrix < 0):
        raise ValueError("demand matrix must be non-negative")
    if np.any(np.diag(matrix) != 0):
        raise ValueError("demand matrix diagonal must be zero")


def _dominant_pair_value(
    rng: np.random.Generator, base: float, asymmetry: float, noise: float
) -> tuple[float, float]:
    dominant = base * (1.0 + noise * rng.random())
    reverse = dominant / max(1.0, asymmetry)
    return dominant, reverse


def _component_asymmetry(
    kind: str, asymmetry_level: float, rng: np.random.Generator
) -> float:
    level = max(1.0, float(asymmetry_level))
    if kind == "tp":
        return 1.0 + 0.30 * (level - 1.0) * (1.0 + 0.08 * rng.random()) + 0.004 * rng.random()
    if kind == "dp":
        return 1.0 + 0.60 * (level - 1.0) * (1.0 + 0.12 * rng.random()) + 0.006 * rng.random()
    if kind == "pp":
        # PP stays close to balanced over longer windows and only weakly reacts to the sweep.
        return 1.0 + 0.04 * (level - 1.0) * (1.0 + 0.05 * rng.random()) + 0.002 * rng.random()
    return level


def _ensure_group_size(group_size: int, n: int) -> int:
    return max(2, min(int(group_size), int(n)))


def _build_model_and_parallel(
    workload: WorkloadConfig, cluster_size: int
) -> tuple[ModelConfig, ParallelConfig]:
    tp = max(2, int(workload.tp_group_size))
    dp = max(2, int(workload.dp_group_size))
    pp = max(1, min(int(workload.pp_stage_count), int(cluster_size)))
    microbatches = max(1, int(workload.microbatches))
    mod = ModelConfig(
        layers=int(workload.model_layers),
        hidden=int(workload.model_hidden),
        seq=int(workload.model_seq),
        head_dim=int(workload.model_head_dim),
        kv_dim=int(workload.model_kv_dim),
        ffn_hidden=int(workload.model_ffn_hidden),
        total_params=float(workload.model_total_params),
        bytes_per_act=int(workload.bytes_per_act),
        bytes_per_param=int(workload.bytes_per_param),
        bytes_per_grad=int(workload.bytes_per_grad),
    )
    par = ParallelConfig(
        tp=tp,
        pp=pp,
        dp=dp,
        global_batch_seqs=tp * pp * microbatches,
        microbatch_seqs=1,
    )
    return mod, par


def _layers_per_segment(workload: WorkloadConfig) -> int:
    return max(1, int(ceil(int(workload.model_layers) / max(1, int(workload.segment_count)))))


def _tp_matrix(
    n: int,
    asymmetry: float,
    scale: float,
    rng: np.random.Generator,
    group_size: int,
    workload: WorkloadConfig,
) -> np.ndarray:
    mat = np.zeros((n, n), dtype=float)
    group = _ensure_group_size(group_size, n)
    mod, par = _build_model_and_parallel(workload, n)
    a_full, _a_shard, _p_bf16, _p_fp32, _ = llama3_megatron_payloads(mod, par)
    layers_per_segment = _layers_per_segment(workload)
    segment_microbatches = max(1, int(workload.microbatches))
    ring_bytes_per_op = float(a_full) * float(group - 1) / float(group)
    base_bytes = (
        4.0
        * ring_bytes_per_op
        * float(layers_per_segment)
        * float(segment_microbatches)
        * float(scale)
    )
    for start in range(0, n, group):
        members = list(range(start, min(start + group, n)))
        for idx, src in enumerate(members):
            dst = members[(idx + 1) % len(members)]
            fwd, rev = _dominant_pair_value(
                rng, base_bytes, _component_asymmetry("tp", asymmetry, rng), 0.15
            )
            if rng.random() < 0.5:
                mat[src, dst] += fwd
                mat[dst, src] += rev
            else:
                mat[src, dst] += rev
                mat[dst, src] += fwd
    return mat


def _dp_matrix(
    n: int,
    asymmetry: float,
    scale: float,
    rng: np.random.Generator,
    group_size: int,
    workload: WorkloadConfig,
) -> np.ndarray:
    mat = np.zeros((n, n), dtype=float)
    group = _ensure_group_size(group_size, n)
    mod, par = _build_model_and_parallel(workload, n)
    _a_full, _a_shard, p_layer_tp_bf16, p_layer_tp_fp32, _ = llama3_megatron_payloads(
        mod, par
    )
    layers_per_segment = _layers_per_segment(workload)
    offset = max(1, group // 2)
    ring_rs = float(p_layer_tp_fp32) * float(group - 1) / float(group)
    ring_ag = float(p_layer_tp_bf16) * float(group - 1) / float(group)
    base_bytes = (ring_rs + ring_ag) * float(layers_per_segment) * float(scale)
    for src in range(n):
        dst = (src + offset) % n
        fwd, rev = _dominant_pair_value(
            rng, base_bytes, _component_asymmetry("dp", asymmetry, rng), 0.2
        )
        mat[src, dst] += fwd
        mat[dst, src] += rev
        for extra in range(1, min(3, n - 1)):
            peer = (src + extra * group) % n
            if peer == src:
                continue
            mat[src, peer] += 0.12 * fwd
            mat[peer, src] += 0.12 * rev
    return mat


def _pp_matrix(
    n: int,
    scale: float,
    asymmetry: float,
    segment_idx: int,
    segments: int,
    rng: np.random.Generator,
    workload: WorkloadConfig,
) -> np.ndarray:
    mat = np.zeros((n, n), dtype=float)
    phase = 1 if segment_idx < max(1, segments // 2) else -1
    mod, par = _build_model_and_parallel(workload, n)
    _a_full, a_shard, _p_bf16, _p_fp32, _ = llama3_megatron_payloads(mod, par)
    pp_transfer_bytes = float(a_shard) * float(max(1, int(workload.microbatches))) * float(scale)
    for src in range(n - 1):
        dst = src + 1
        phase_bias = 1.0 + (0.015 if phase > 0 else -0.015)
        major = (0.94 + 0.08 * rng.random()) * pp_transfer_bytes * phase_bias
        skew = _component_asymmetry("pp", asymmetry, rng)
        minor = major / skew
        if phase > 0:
            mat[src, dst] += major
            mat[dst, src] += minor
        else:
            mat[src, dst] += minor
            mat[dst, src] += major
    return mat


def _mixed_matrix(
    n: int,
    asymmetry: float,
    scale: float,
    rng: np.random.Generator,
    segment_idx: int,
    segments: int,
    weights: Dict[str, float],
    tp_group_size: int,
    dp_group_size: int,
    workload: WorkloadConfig,
) -> np.ndarray:
    tp_w = float(weights.get("tp", 0.5))
    dp_w = float(weights.get("dp", 0.5))
    pp_w = float(weights.get("pp", 0.0))
    mat = (
        tp_w * _tp_matrix(n, asymmetry, scale, rng, tp_group_size, workload)
        + dp_w * _dp_matrix(n, asymmetry, scale, rng, dp_group_size, workload)
        + pp_w * _pp_matrix(n, scale, asymmetry, segment_idx, segments, rng, workload)
    )
    return mat


def _load_single_matrix(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        matrix = np.load(path)
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            matrix = np.array(json.load(handle), dtype=float)
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            matrix = np.array([[float(cell) for cell in row] for row in reader], dtype=float)
    else:
        raise ValueError(f"unsupported matrix file: {path}")
    validate_demand_matrix(matrix)
    return matrix


def load_or_generate_workload(
    workload: WorkloadConfig, cluster_size: int, asymmetry: float, base_seed: int
) -> List[SegmentDemand]:
    segments = max(1, int(workload.segment_count))
    rng = np.random.default_rng(
        int(base_seed) + int(workload.seed_offset) + cluster_size * 17
    )
    effective_asymmetry = (
        float(workload.asymmetry) if workload.asymmetry is not None else float(asymmetry)
    )
    out: List[SegmentDemand] = []
    if workload.load_path:
        matrix = _load_single_matrix(Path(workload.load_path))
        if matrix.shape[0] != cluster_size:
            raise ValueError(
                f"loaded matrix shape {matrix.shape} does not match cluster size {cluster_size}"
            )
        for segment_idx in range(segments):
            out.append(
                SegmentDemand(
                    workload=workload.name,
                    segment_idx=segment_idx,
                    matrix=matrix.copy(),
                    metadata={"kind": workload.kind, "source": workload.load_path},
                )
            )
        return out

    for segment_idx in range(segments):
        if workload.kind == "tp":
            mat = _tp_matrix(
                cluster_size,
                effective_asymmetry,
                workload.scale,
                rng,
                workload.tp_group_size,
                workload,
            )
        elif workload.kind == "dp":
            mat = _dp_matrix(
                cluster_size,
                effective_asymmetry,
                workload.scale,
                rng,
                workload.dp_group_size,
                workload,
            )
        elif workload.kind == "pp":
            mat = _pp_matrix(
                cluster_size,
                workload.scale,
                effective_asymmetry,
                segment_idx,
                segments,
                rng,
                workload,
            )
        elif workload.kind == "mixed":
            mat = _mixed_matrix(
                cluster_size,
                effective_asymmetry,
                workload.scale,
                rng,
                segment_idx,
                segments,
                workload.mixed_weights,
                workload.tp_group_size,
                workload.dp_group_size,
                workload,
            )
        else:
            raise ValueError(f"unknown workload kind: {workload.kind}")
        np.fill_diagonal(mat, 0.0)
        validate_demand_matrix(mat)
        out.append(
            SegmentDemand(
                workload=workload.name,
                segment_idx=segment_idx,
                matrix=mat,
                metadata={
                    "kind": workload.kind,
                    "asymmetry": effective_asymmetry,
                    "units": "bytes",
                    "model_layers": int(workload.model_layers),
                    "microbatches": int(workload.microbatches),
                },
            )
        )
    return out


def directional_skew_values(matrix: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
    validate_demand_matrix(matrix)
    values: List[float] = []
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            a = float(matrix[i, j])
            b = float(matrix[j, i])
            if a <= 0.0 and b <= 0.0:
                continue
            values.append(max(a, b) / (min(a, b) + epsilon))
    return np.array(values, dtype=float)


def iter_nonzero_pairs(matrix: np.ndarray) -> Iterable[tuple[int, int, float]]:
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i == j:
                continue
            value = float(matrix[i, j])
            if value > 0.0:
                yield i, j, value
