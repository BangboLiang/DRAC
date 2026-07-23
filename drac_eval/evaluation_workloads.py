"""Deterministic DP, PP, and DP+PP communication graphs for Evaluation.

These are simulator inputs derived from explicit communication operations.  They
are not hardware measurements, and no random directional skew is injected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .demand_profiler import (
    CommunicationNode,
    ProfiledDemand,
    RankPlacement,
    TransportCalibration,
    profile_communication_nodes,
)


@dataclass(frozen=True)
class EvaluationWorkload:
    name: str
    nodes: tuple[CommunicationNode, ...]
    placements: dict[int, RankPlacement]
    profiled: tuple[ProfiledDemand, ...]
    provenance: str = "DETERMINISTIC_SIMULATOR_INPUT"

    @property
    def demands(self) -> tuple[np.ndarray, ...]:
        return tuple(item.matrix for item in self.profiled)


def one_rank_per_server(endpoint_count: int) -> dict[int, RankPlacement]:
    if endpoint_count < 2:
        raise ValueError("Evaluation workloads require at least two endpoints")
    return {
        rank: RankPlacement(
            rank=rank,
            endpoint=rank,
            server=f"server-{rank}",
            gpu=f"gpu-{rank}",
            nic=f"nic-{rank}",
        )
        for rank in range(endpoint_count)
    }


def _dp_nodes(endpoint_count: int, message_bytes: float, repeats: int) -> list[CommunicationNode]:
    ranks = tuple(range(endpoint_count))
    nodes: list[CommunicationNode] = []
    for repetition in range(repeats):
        nodes.extend(
            [
                CommunicationNode(
                    f"dp-rs-{repetition}",
                    "reduce_scatter",
                    message_bytes,
                    ranks=ranks,
                    algorithm="ring",
                    phase="backward",
                ),
                CommunicationNode(
                    f"dp-ag-{repetition}",
                    "all_gather",
                    message_bytes / 2.0,
                    ranks=ranks,
                    algorithm="ring",
                    phase="optimizer",
                ),
            ]
        )
    return nodes


def _pp_nodes(endpoint_count: int, message_bytes: float, microbatches: int) -> list[CommunicationNode]:
    nodes: list[CommunicationNode] = []
    for microbatch in range(microbatches):
        for stage in range(endpoint_count - 1):
            nodes.append(
                CommunicationNode(
                    f"pp-fwd-mb{microbatch}-s{stage}",
                    "p2p",
                    message_bytes,
                    src_rank=stage,
                    dst_rank=stage + 1,
                    phase="forward",
                )
            )
        for stage in reversed(range(endpoint_count - 1)):
            nodes.append(
                CommunicationNode(
                    f"pp-bwd-mb{microbatch}-s{stage}",
                    "p2p",
                    message_bytes,
                    src_rank=stage + 1,
                    dst_rank=stage,
                    phase="backward",
                )
            )
    return nodes


def build_evaluation_workload(
    kind: str,
    *,
    endpoint_count: int = 4,
    message_bytes: float = 64 * 1024 * 1024,
    repeats: int = 2,
    calibration: TransportCalibration | None = None,
) -> EvaluationWorkload:
    kind_norm = kind.lower().replace("_", "-")
    placements = one_rank_per_server(endpoint_count)
    if kind_norm in {"dp", "dp-only"}:
        nodes = _dp_nodes(endpoint_count, message_bytes, repeats)
        name = "DP"
    elif kind_norm in {"pp", "pp-only"}:
        nodes = _pp_nodes(endpoint_count, message_bytes / 4.0, repeats)
        name = "PP"
    elif kind_norm in {"mixed", "dp+pp", "dp-pp"}:
        dp_nodes = _dp_nodes(endpoint_count, message_bytes, repeats)
        pp_nodes = _pp_nodes(endpoint_count, message_bytes / 4.0, repeats)
        nodes = []
        for index in range(max(len(dp_nodes), len(pp_nodes))):
            if index < len(pp_nodes):
                nodes.append(pp_nodes[index])
            if index < len(dp_nodes):
                nodes.append(dp_nodes[index])
        name = "DP+PP Mixed"
    else:
        raise ValueError(f"unsupported main Evaluation workload: {kind}")
    profiled = profile_communication_nodes(nodes, placements, calibration)
    return EvaluationWorkload(name, tuple(nodes), placements, tuple(profiled))
