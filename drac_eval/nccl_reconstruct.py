from __future__ import annotations

from typing import Dict, List, Sequence

from .collective_trace import (
    CollectiveEvent, CollectiveExecution, CollectiveOperation, RankPlacement,
    validate_byte_conservation, validate_execution,
)
from .nccl_log import NCCLLogRecord


def reconstruct_ring_schedule(
    record: NCCLLogRecord,
    execution_id: str = "nccl-fixture",
    collective_id: str = "collective-0",
    direction_by_channel: Dict[int, str] | None = None,
) -> CollectiveExecution:
    if record.algorithm.lower() not in {"ring", "unavailable"}:
        operation = CollectiveOperation(collective_id, record.operation_type, record.algorithm, record.protocol, record.message_bytes or 0, [], "unsupported_schedule")
        return CollectiveExecution(execution_id, record.placements, record.channels, [operation], "nccl_selected_schedule")
    ring_channels = [channel for channel in record.channels if channel.rank_order]
    if not ring_channels or not record.message_bytes or record.operation_type not in {"allreduce", "allgather", "reducescatter"}:
        operation = CollectiveOperation(collective_id, record.operation_type, record.algorithm, record.protocol, record.message_bytes or 0, [], "unsupported_schedule")
        return CollectiveExecution(execution_id, record.placements, record.channels, [operation], "nccl_selected_schedule")
    nranks = record.rank_count or len(ring_channels[0].rank_order)
    placements = record.placements or {rank: RankPlacement(rank, f"host{rank}") for rank in range(nranks)}
    channel_count = len(ring_channels)
    # One chunk per rank/channel. Total message bytes are divided across channels
    # and rank chunks without changing theoretical ring traffic volume.
    chunk_bytes = float(record.message_bytes) / float(channel_count * nranks)
    phases = {"allreduce": ["reduce_scatter", "all_gather"], "allgather": ["all_gather"], "reducescatter": ["reduce_scatter"]}[record.operation_type]
    events: List[CollectiveEvent] = []
    last: Dict[tuple[int, int], str] = {}
    for channel in ring_channels:
        order = list(channel.rank_order)
        direction = (direction_by_channel or {}).get(channel.channel_id, "clockwise")
        if direction == "counter_clockwise": order = [order[0]] + list(reversed(order[1:]))
        position = {rank: idx for idx, rank in enumerate(order)}
        holder = {chunk: order[chunk % nranks] for chunk in range(nranks)}
        previous_phase_ids: List[str] = []
        for phase_index, phase in enumerate(phases):
            if phase == "all_gather" and record.operation_type == "allgather":
                holder = {chunk: order[chunk % nranks] for chunk in range(nranks)}
            previous_step_ids: List[str] = list(previous_phase_ids)
            for step in range(nranks - 1):
                updates = {}
                current_step_ids: List[str] = []
                for chunk in range(nranks):
                    src = holder[chunk]; dst = order[(position[src] + 1) % nranks]
                    event_id = f"{collective_id}:ch{channel.channel_id}:{phase}:s{step}:c{chunk}"
                    deps = set(previous_step_ids)
                    if (channel.channel_id, chunk) in last: deps.add(last[(channel.channel_id, chunk)])
                    dep = tuple(sorted(deps))
                    events.append(CollectiveEvent(execution_id, collective_id, record.operation_type, "Ring", record.protocol, channel.channel_id, phase, step + phase_index * (nranks - 1), chunk + channel.channel_id * nranks, src, dst, placements[src].host, placements[dst].host, chunk_bytes, dependency_ids=dep, provenance="nccl_selected_schedule", event_id=event_id))
                    last[(channel.channel_id, chunk)] = event_id; updates[chunk] = dst; current_step_ids.append(event_id)
                holder.update(updates)
                previous_step_ids = current_step_ids
            previous_phase_ids = previous_step_ids
    operation = CollectiveOperation(collective_id, record.operation_type, "Ring", record.protocol, record.message_bytes, events)
    execution = CollectiveExecution(execution_id, placements, record.channels, [operation], "nccl_selected_schedule")
    expected_factor = 2 if record.operation_type == "allreduce" else 1
    validate_byte_conservation(events, record.message_bytes * (nranks - 1) * expected_factor)
    validate_execution(execution)
    return execution


def fixed_half_bidirectional(record: NCCLLogRecord, **kwargs: object) -> CollectiveExecution:
    rings = [channel for channel in record.channels if channel.rank_order]
    directions = {channel.channel_id: ("clockwise" if idx < (len(rings)+1)//2 else "counter_clockwise") for idx, channel in enumerate(rings)}
    return reconstruct_ring_schedule(record, direction_by_channel=directions, **kwargs)
