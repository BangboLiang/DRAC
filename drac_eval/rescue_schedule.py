from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ScheduleEvent:
    event_id: str
    operation_id: str
    phase: str
    step: int
    chunk_id: int
    direction: str
    src_rank: int
    dst_rank: int
    bytes: float
    dependency_ids: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        row = asdict(self)
        row["dependency_ids"] = ";".join(self.dependency_ids)
        return row


@dataclass
class ExecutableRingSchedule:
    rank_count: int
    chunk_count: int
    operation_type: str
    bidirectional: bool
    events: List[ScheduleEvent]
    payload_bytes_per_rank: float

    @property
    def total_transmitted_bytes(self) -> float:
        return float(sum(event.bytes for event in self.events))


def _chunk_directions(chunk_count: int, bidirectional: bool, odd_rule: str) -> List[str]:
    if not bidirectional:
        return ["clockwise"] * chunk_count
    if odd_rule != "extra_clockwise":
        raise ValueError(f"unsupported odd chunk rule: {odd_rule}")
    clockwise = (chunk_count + 1) // 2
    return ["clockwise" if chunk < clockwise else "counter_clockwise" for chunk in range(chunk_count)]


def build_executable_ring_schedule(
    rank_count: int,
    payload_bytes_per_rank: float,
    operation_type: str,
    bidirectional: bool,
    chunk_count: int | None = None,
    odd_rule: str = "extra_clockwise",
    operation_id: str = "ring-0",
) -> ExecutableRingSchedule:
    if rank_count < 2:
        raise ValueError("ring requires at least two ranks")
    chunks = int(chunk_count or rank_count)
    if chunks < 1:
        raise ValueError("chunk_count must be positive")
    phases = {
        "allreduce": ("reduce_scatter", "all_gather"),
        "allgather": ("all_gather_only",),
        "reducescatter": ("reduce_scatter_only",),
    }.get(operation_type)
    if phases is None:
        raise ValueError(operation_type)
    directions = _chunk_directions(chunks, bidirectional, odd_rule)
    chunk_bytes = float(payload_bytes_per_rank) / float(chunks)
    events: List[ScheduleEvent] = []
    last_event: Dict[int, str] = {}
    holder: Dict[int, int] = {chunk: chunk % rank_count for chunk in range(chunks)}

    for phase in phases:
        if phase in {"reduce_scatter", "reduce_scatter_only"}:
            for step in range(rank_count - 1):
                updates: Dict[int, int] = {}
                for chunk in range(chunks):
                    direction = directions[chunk]
                    src = holder[chunk]
                    dst = (src + 1) % rank_count if direction == "clockwise" else (src - 1) % rank_count
                    event_id = f"{operation_id}:{phase}:s{step}:c{chunk}"
                    deps = (last_event[chunk],) if chunk in last_event else ()
                    events.append(ScheduleEvent(event_id, operation_id, phase, step, chunk, direction, src, dst, chunk_bytes, deps))
                    last_event[chunk] = event_id
                    updates[chunk] = dst
                holder.update(updates)
        else:
            # all-gather-only starts with each final chunk at a deterministic owner.
            if phase == "all_gather_only":
                holder = {chunk: chunk % rank_count for chunk in range(chunks)}
                last_event = {}
            for step in range(rank_count - 1):
                updates = {}
                for chunk in range(chunks):
                    direction = directions[chunk]
                    src = holder[chunk]
                    dst = (src + 1) % rank_count if direction == "clockwise" else (src - 1) % rank_count
                    event_id = f"{operation_id}:{phase}:s{step}:c{chunk}"
                    deps = (last_event[chunk],) if chunk in last_event else ()
                    events.append(ScheduleEvent(event_id, operation_id, phase, step, chunk, direction, src, dst, chunk_bytes, deps))
                    last_event[chunk] = event_id
                    updates[chunk] = dst
                holder.update(updates)
    schedule = ExecutableRingSchedule(rank_count, chunks, operation_type, bidirectional, events, float(payload_bytes_per_rank))
    validate_executable_ring_schedule(schedule)
    return schedule


def validate_executable_ring_schedule(schedule: ExecutableRingSchedule) -> Dict[str, object]:
    n = schedule.rank_count
    expected_phases = 2 if schedule.operation_type == "allreduce" else 1
    expected_per_chunk = expected_phases * (n - 1)
    by_chunk: Dict[int, List[ScheduleEvent]] = {chunk: [] for chunk in range(schedule.chunk_count)}
    known: Dict[int, set[int]] = {chunk: {chunk % n} for chunk in range(schedule.chunk_count)}
    reduced_holders: Dict[int, int] = {chunk: chunk % n for chunk in range(schedule.chunk_count)}
    completed_ids: set[str] = set()
    phases = ["reduce_scatter", "reduce_scatter_only", "all_gather", "all_gather_only"]
    for phase in phases:
        phase_events = [event for event in schedule.events if event.phase == phase]
        for step in sorted({event.step for event in phase_events}):
            events = [event for event in phase_events if event.step == step]
            sends: Dict[Tuple[int, str], int] = {}
            receives: Dict[Tuple[int, str], int] = {}
            for event in events:
                if any(dep not in completed_ids for dep in event.dependency_ids):
                    raise AssertionError("event dependency not completed")
                by_chunk[event.chunk_id].append(event)
                sends[(event.src_rank, event.direction)] = sends.get((event.src_rank, event.direction), 0) + 1
                receives[(event.dst_rank, event.direction)] = receives.get((event.dst_rank, event.direction), 0) + 1
                if phase in {"reduce_scatter", "reduce_scatter_only"}:
                    if reduced_holders[event.chunk_id] != event.src_rank:
                        raise AssertionError("source does not own the partial reduction")
                else:
                    if event.src_rank not in known[event.chunk_id]:
                        raise AssertionError("source sends an all-gather chunk it does not own")
            if any(count > 1 for count in sends.values()) or any(count > 1 for count in receives.values()):
                raise AssertionError("a rank sends/receives more than once per direction in one step")
            for event in events:
                if phase in {"reduce_scatter", "reduce_scatter_only"}:
                    reduced_holders[event.chunk_id] = event.dst_rank
                else:
                    known[event.chunk_id].add(event.dst_rank)
                completed_ids.add(event.event_id)
        if phase == "reduce_scatter":
            # All-gather may start only after every chunk has n contributions.
            known = {chunk: {holder} for chunk, holder in reduced_holders.items()}
    for chunk, events in by_chunk.items():
        if len(events) != expected_per_chunk:
            raise AssertionError(f"chunk {chunk} has {len(events)} transfers, expected {expected_per_chunk}")
    if schedule.operation_type in {"allreduce", "allgather"}:
        for chunk in range(schedule.chunk_count):
            if len(known[chunk]) != n:
                raise AssertionError("not every rank owns the final collective result")
    expected_bytes = schedule.payload_bytes_per_rank * (n - 1) * expected_phases
    if not np.isclose(schedule.total_transmitted_bytes, expected_bytes, rtol=1e-12, atol=1e-6):
        raise AssertionError("ring schedule changed total transmitted bytes")
    return {
        "dependency_valid": True,
        "semantic_valid": True,
        "expected_transfers_per_chunk": expected_per_chunk,
        "total_transmitted_bytes": schedule.total_transmitted_bytes,
    }


def schedule_step_matrices(schedule: ExecutableRingSchedule) -> List[Tuple[str, int, np.ndarray]]:
    out: List[Tuple[str, int, np.ndarray]] = []
    keys = []
    for event in schedule.events:
        key = (event.phase, event.step)
        if key not in keys:
            keys.append(key)
    for phase, step in keys:
        matrix = np.zeros((schedule.rank_count, schedule.rank_count), dtype=float)
        for event in schedule.events:
            if event.phase == phase and event.step == step:
                matrix[event.src_rank, event.dst_rank] += event.bytes
        out.append((phase, step, matrix))
    return out


def merge_schedules(schedules: Sequence[ExecutableRingSchedule]) -> ExecutableRingSchedule:
    if not schedules:
        raise ValueError("no schedules")
    n = schedules[0].rank_count
    if any(schedule.rank_count != n for schedule in schedules):
        raise ValueError("rank counts differ")
    events: List[ScheduleEvent] = []
    for schedule in schedules:
        events.extend(schedule.events)
    return ExecutableRingSchedule(
        n,
        sum(schedule.chunk_count for schedule in schedules),
        "composite",
        any(schedule.bidirectional for schedule in schedules),
        events,
        sum(schedule.payload_bytes_per_rank for schedule in schedules),
    )
