from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


PROVENANCE_VALUES = {
    "measured_packet_trace",
    "measured_runtime_log",
    "nccl_selected_schedule",
    "executable_reconstructed_schedule",
    "synthetic_sensitivity",
}


@dataclass(frozen=True)
class RankPlacement:
    rank: int
    host: str
    gpu: str = "unavailable"


@dataclass(frozen=True)
class ChannelTopology:
    channel_id: int
    topology_type: str
    rank_order: Tuple[int, ...] = ()
    tree_description: str = "unavailable"
    transport: str = "unavailable"


@dataclass(frozen=True)
class CollectiveEvent:
    execution_id: str
    collective_id: str
    operation_type: str
    algorithm: str
    protocol: str
    channel_id: int
    phase: str
    step: int
    chunk_id: int
    src_rank: int
    dst_rank: int
    src_host: str
    dst_host: str
    bytes: float
    start_time_us: float | None = None
    end_time_us: float | None = None
    dependency_ids: Tuple[str, ...] = ()
    provenance: str = "executable_reconstructed_schedule"
    event_id: str = ""


@dataclass
class CollectiveStep:
    phase: str
    step: int
    events: List[CollectiveEvent] = field(default_factory=list)


@dataclass
class CollectiveOperation:
    collective_id: str
    operation_type: str
    algorithm: str
    protocol: str
    message_bytes: int
    events: List[CollectiveEvent] = field(default_factory=list)
    status: str = "ok"


@dataclass
class CollectiveExecution:
    execution_id: str
    placements: Dict[int, RankPlacement]
    channels: List[ChannelTopology]
    operations: List[CollectiveOperation]
    provenance: str


EVENT_COLUMNS = [
    "event_id", "execution_id", "collective_id", "operation_type", "algorithm",
    "protocol", "channel_id", "phase", "step", "chunk_id", "src_rank",
    "dst_rank", "src_host", "dst_host", "bytes", "start_time_us",
    "end_time_us", "dependency_ids", "provenance",
]


def validate_event_schema(event: CollectiveEvent) -> None:
    if event.provenance not in PROVENANCE_VALUES:
        raise ValueError(f"invalid provenance: {event.provenance}")
    if event.src_rank < 0 or event.dst_rank < 0 or event.src_rank == event.dst_rank:
        raise ValueError("invalid ordered rank pair")
    if event.bytes < 0 or event.step < 0 or event.channel_id < 0:
        raise ValueError("negative event field")
    if event.start_time_us is not None and event.end_time_us is not None and event.end_time_us < event.start_time_us:
        raise ValueError("event ends before it starts")


def write_collective_events_csv(path: str | Path, events: Sequence[CollectiveEvent]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_COLUMNS)
        writer.writeheader()
        for event in events:
            validate_event_schema(event)
            row = {key: getattr(event, key) for key in EVENT_COLUMNS}
            row["dependency_ids"] = ";".join(event.dependency_ids)
            row["start_time_us"] = "" if event.start_time_us is None else event.start_time_us
            row["end_time_us"] = "" if event.end_time_us is None else event.end_time_us
            writer.writerow(row)


def load_collective_events_csv(path: str | Path) -> List[CollectiveEvent]:
    events = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(EVENT_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"collective event CSV missing columns: {sorted(missing)}")
        for row in reader:
            event = CollectiveEvent(
                event_id=row["event_id"], execution_id=row["execution_id"], collective_id=row["collective_id"],
                operation_type=row["operation_type"], algorithm=row["algorithm"], protocol=row["protocol"],
                channel_id=int(row["channel_id"]), phase=row["phase"], step=int(row["step"]),
                chunk_id=int(row["chunk_id"]), src_rank=int(row["src_rank"]), dst_rank=int(row["dst_rank"]),
                src_host=row["src_host"], dst_host=row["dst_host"], bytes=float(row["bytes"]),
                start_time_us=float(row["start_time_us"]) if row["start_time_us"] else None,
                end_time_us=float(row["end_time_us"]) if row["end_time_us"] else None,
                dependency_ids=tuple(v for v in row["dependency_ids"].split(";") if v), provenance=row["provenance"],
            )
            validate_event_schema(event)
            events.append(event)
    validate_dependencies(events)
    return events


def validate_dependencies(events: Sequence[CollectiveEvent]) -> None:
    ids = {event.event_id for event in events}
    if len(ids) != len(events) or "" in ids:
        raise ValueError("event ids must be unique and non-empty")
    deps = {event.event_id: set(event.dependency_ids) for event in events}
    if any(not values <= ids for values in deps.values()):
        raise ValueError("dependency references an unknown event")
    visiting: set[str] = set()
    complete: set[str] = set()
    def visit(event_id: str) -> None:
        if event_id in visiting:
            raise ValueError("dependency graph contains a cycle")
        if event_id in complete:
            return
        visiting.add(event_id)
        for dep in deps[event_id]:
            visit(dep)
        visiting.remove(event_id)
        complete.add(event_id)
    for event_id in ids:
        visit(event_id)


def validate_host_rank_mapping(events: Sequence[CollectiveEvent], placements: Dict[int, RankPlacement]) -> None:
    for event in events:
        if event.src_rank not in placements or event.dst_rank not in placements:
            raise ValueError("event rank missing from placement")
        if placements[event.src_rank].host != event.src_host or placements[event.dst_rank].host != event.dst_host:
            raise ValueError("event host does not match rank placement")


def validate_byte_conservation(events: Sequence[CollectiveEvent], expected_bytes: float, tolerance: float = 1e-9) -> None:
    actual = sum(event.bytes for event in events)
    if abs(actual - expected_bytes) > tolerance * max(1.0, expected_bytes):
        raise ValueError(f"byte conservation failed: actual={actual}, expected={expected_bytes}")


def aggregate_ordered_demand(events: Iterable[CollectiveEvent], rank_count: int) -> "object":
    import numpy as np
    matrix = np.zeros((rank_count, rank_count), dtype=float)
    for event in events:
        matrix[event.src_rank, event.dst_rank] += event.bytes
    return matrix


def validate_execution(execution: CollectiveExecution) -> None:
    if execution.provenance not in PROVENANCE_VALUES:
        raise ValueError("invalid execution provenance")
    events = [event for operation in execution.operations for event in operation.events]
    validate_dependencies(events)
    validate_host_rank_mapping(events, execution.placements)
    for event in events:
        validate_event_schema(event)
