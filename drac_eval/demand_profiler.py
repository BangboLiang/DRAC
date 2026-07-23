"""Ordered communication-node demand profiling for the paper-aligned DRAC path.

The profiler consumes communication nodes, expands their selected collective
implementation into ordered transfers, maps ranks to GPU-NIC endpoints, applies
optional transport calibration, and removes intra-server traffic.  It never
segments the node sequence and never uses a skew threshold.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class RankPlacement:
    rank: int
    endpoint: int
    server: str
    gpu: str = ""
    nic: str = ""


@dataclass(frozen=True)
class CommunicationNode:
    node_id: str
    operation: str
    message_bytes: float
    ranks: tuple[int, ...] = ()
    src_rank: int | None = None
    dst_rank: int | None = None
    algorithm: str = "ring"
    phase: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderedTransfer:
    node_id: str
    src_rank: int
    dst_rank: int
    payload_bytes: float
    phase: str
    step: int
    chunk: int


@dataclass(frozen=True)
class CalibrationBin:
    message_bytes: int
    forward_overhead_bytes: float
    reverse_control_bytes: float
    samples: int = 0
    provenance: str = "calibrated"


@dataclass(frozen=True)
class TransportCalibration:
    bins: tuple[CalibrationBin, ...] = ()
    environment: str = ""

    def nearest(self, message_bytes: float) -> CalibrationBin | None:
        if not self.bins:
            return None
        return min(self.bins, key=lambda item: (abs(item.message_bytes - message_bytes), item.message_bytes))


@dataclass
class ProfiledDemand:
    node: CommunicationNode
    endpoint_order: tuple[int, ...]
    matrix: np.ndarray
    payload_matrix: np.ndarray
    control_matrix: np.ndarray
    transfers: tuple[OrderedTransfer, ...]
    excluded_intra_server_bytes: float
    provenance: str


def validate_placements(placements: Mapping[int, RankPlacement]) -> None:
    for rank, placement in placements.items():
        if int(rank) != int(placement.rank):
            raise ValueError("placement dictionary key disagrees with placement.rank")
        if placement.endpoint < 0 or not placement.server:
            raise ValueError("every rank needs a non-negative endpoint and non-empty server")


def _normalized_operation(operation: str) -> str:
    return operation.lower().replace("_", "").replace("-", "")


def expand_ordered_transfers(node: CommunicationNode) -> list[OrderedTransfer]:
    """Expand one communication node without aggregating it with neighboring nodes."""

    if not np.isfinite(node.message_bytes) or node.message_bytes < 0:
        raise ValueError("message_bytes must be finite and non-negative")
    op = _normalized_operation(node.operation)
    if op in {"p2p", "send", "recv", "pp", "pipeline"}:
        if node.src_rank is None or node.dst_rank is None or node.src_rank == node.dst_rank:
            raise ValueError("point-to-point node requires distinct src_rank and dst_rank")
        return [
            OrderedTransfer(
                node.node_id,
                int(node.src_rank),
                int(node.dst_rank),
                float(node.message_bytes),
                node.phase or "p2p",
                0,
                0,
            )
        ]

    if node.algorithm.lower() != "ring":
        raise NotImplementedError(
            f"ordered expansion for algorithm={node.algorithm!r} is not implemented; "
            "do not silently substitute a different collective schedule"
        )
    ranks = tuple(int(rank) for rank in node.ranks)
    if len(ranks) < 2 or len(set(ranks)) != len(ranks):
        raise ValueError("ring collective requires at least two unique ranks")
    phases = {
        "allreduce": ("reduce_scatter", "all_gather"),
        "reducescatter": ("reduce_scatter",),
        "allgather": ("all_gather",),
    }.get(op)
    if phases is None:
        raise ValueError(f"unsupported communication operation: {node.operation}")

    chunk_bytes = float(node.message_bytes) / len(ranks)
    transfers: list[OrderedTransfer] = []
    for phase_index, phase in enumerate(phases):
        for step in range(len(ranks) - 1):
            for chunk, src in enumerate(ranks):
                idx = ranks.index(src)
                dst = ranks[(idx + 1) % len(ranks)]
                transfers.append(
                    OrderedTransfer(
                        node.node_id,
                        src,
                        dst,
                        chunk_bytes,
                        phase,
                        phase_index * (len(ranks) - 1) + step,
                        chunk,
                    )
                )
    return transfers


def profile_communication_nodes(
    nodes: Sequence[CommunicationNode],
    placements: Mapping[int, RankPlacement],
    calibration: TransportCalibration | None = None,
) -> list[ProfiledDemand]:
    """Return one ordered endpoint demand matrix per input communication node."""

    validate_placements(placements)
    endpoint_order = tuple(sorted({placement.endpoint for placement in placements.values()}))
    endpoint_index = {endpoint: idx for idx, endpoint in enumerate(endpoint_order)}
    output: list[ProfiledDemand] = []
    for node in nodes:
        transfers = expand_ordered_transfers(node)
        n = len(endpoint_order)
        payload = np.zeros((n, n), dtype=float)
        control = np.zeros((n, n), dtype=float)
        excluded = 0.0
        for transfer in transfers:
            try:
                src = placements[transfer.src_rank]
                dst = placements[transfer.dst_rank]
            except KeyError as exc:
                raise ValueError(f"rank {exc.args[0]} is missing from placement") from exc
            if src.server == dst.server:
                excluded += transfer.payload_bytes
                continue
            i = endpoint_index[src.endpoint]
            j = endpoint_index[dst.endpoint]
            if i == j:
                excluded += transfer.payload_bytes
                continue
            payload[i, j] += transfer.payload_bytes
            selected = None if calibration is None else calibration.nearest(transfer.payload_bytes)
            if selected is not None:
                control[i, j] += selected.forward_overhead_bytes
                control[j, i] += selected.reverse_control_bytes
        matrix = payload + control
        np.fill_diagonal(matrix, 0.0)
        output.append(
            ProfiledDemand(
                node=node,
                endpoint_order=endpoint_order,
                matrix=matrix,
                payload_matrix=payload,
                control_matrix=control,
                transfers=tuple(transfers),
                excluded_intra_server_bytes=float(excluded),
                provenance="payload+calibration" if calibration and calibration.bins else "payload_only",
            )
        )
    return output


def load_calibration(path: str | Path) -> TransportCalibration:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    bins = tuple(CalibrationBin(**item) for item in raw.get("bins", []))
    return TransportCalibration(bins=bins, environment=str(raw.get("environment", "")))


MEASUREMENT_COLUMNS = {
    "operation",
    "message_bytes",
    "src_endpoint",
    "dst_endpoint",
    "directional_bytes",
}


def load_directional_measurements(path: str | Path) -> list[dict[str, str]]:
    """Load genuine measurement rows; provenance must explicitly say measured."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = MEASUREMENT_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"measurement CSV missing columns: {sorted(missing)}")
        rows = list(reader)
    for row in rows:
        if row.get("provenance", "").strip().lower() not in {
            "measured_nic_counter",
            "measured_packet_trace",
        }:
            raise ValueError("measurement rows require an explicit measured provenance")
        if float(row["directional_bytes"]) < 0:
            raise ValueError("measured directional bytes must be non-negative")
    return rows


def placements_from_rows(rows: Iterable[Mapping[str, Any]]) -> dict[int, RankPlacement]:
    return {
        int(row["rank"]): RankPlacement(
            rank=int(row["rank"]),
            endpoint=int(row["endpoint"]),
            server=str(row["server"]),
            gpu=str(row.get("gpu", "")),
            nic=str(row.get("nic", "")),
        )
        for row in rows
    }
