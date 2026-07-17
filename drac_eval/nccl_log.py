from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from .collective_trace import ChannelTopology, RankPlacement


@dataclass
class NCCLLogRecord:
    source_path: str
    rank_count: int | None = None
    channel_count: int | None = None
    algorithm: str = "unavailable"
    protocol: str = "unavailable"
    operation_type: str = "unavailable"
    message_bytes: int | None = None
    runtime_us: float | None = None
    channels: List[ChannelTopology] = field(default_factory=list)
    placements: Dict[int, RankPlacement] = field(default_factory=dict)
    transports: List[str] = field(default_factory=list)
    unsupported_schedule: bool = False


RING_RE = re.compile(r"Channel\s+(\d+)/(\d+)\s*:\s*([0-9 ]+)$", re.I)
TREE_RE = re.compile(r"Trees?\s*\[(\d+)\]\s*(.*)$", re.I)
NRANKS_RE = re.compile(r"nranks[ =]+(\d+)", re.I)
NCHANNELS_RE = re.compile(r"nChannels[ =]+(\d+)", re.I)
RANK_HOST_RE = re.compile(r"(?:host|Host)\s+([^ ]+).*?rank[ =]+(\d+)", re.I)
COLL_RE = re.compile(r"\b(AllReduce|AllGather|ReduceScatter|Broadcast|SendRecv)\b", re.I)
BYTES_RE = re.compile(r"(?:bytes|nbytes|size)[ =:]+(\d+)", re.I)
RUNTIME_RE = re.compile(r"(?:time|runtime)[ =:]+([0-9.]+)\s*us", re.I)
ALGO_RE = re.compile(r"(?:algorithm|algo)[ =:]+(Ring|Tree|CollNet|NVLS|PAT)", re.I)
PROTO_RE = re.compile(r"(?:protocol|proto)[ =:]+(Simple|LL128|LL)", re.I)


def parse_nccl_log(path: str | Path) -> NCCLLogRecord:
    path = Path(path)
    record = NCCLLogRecord(str(path))
    channels: Dict[int, ChannelTopology] = {}
    tree_lines: Dict[int, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := RING_RE.search(line):
            channel_id, count, ranks = int(match.group(1)), int(match.group(2)), tuple(int(v) for v in match.group(3).split())
            channels[channel_id] = ChannelTopology(channel_id, "ring", ranks)
            record.channel_count = int(count)
            record.rank_count = len(ranks)
        if match := TREE_RE.search(line):
            tree_lines[int(match.group(1))] = match.group(2).strip()
        if match := NRANKS_RE.search(line): record.rank_count = int(match.group(1))
        if match := NCHANNELS_RE.search(line): record.channel_count = int(match.group(1))
        if match := RANK_HOST_RE.search(line):
            rank = int(match.group(2)); record.placements[rank] = RankPlacement(rank, match.group(1))
        if match := COLL_RE.search(line): record.operation_type = match.group(1).lower()
        if match := BYTES_RE.search(line): record.message_bytes = int(match.group(1))
        if match := RUNTIME_RE.search(line): record.runtime_us = float(match.group(1))
        if match := ALGO_RE.search(line): record.algorithm = match.group(1)
        if match := PROTO_RE.search(line): record.protocol = match.group(1)
        if "NET/" in line or "P2P/" in line:
            transport = "NET" if "NET/" in line else "P2P"
            if transport not in record.transports: record.transports.append(transport)
    for channel_id, description in tree_lines.items():
        if channel_id not in channels:
            channels[channel_id] = ChannelTopology(channel_id, "tree", (), description)
        else:
            ring = channels[channel_id]
            channels[channel_id] = ChannelTopology(ring.channel_id, ring.topology_type, ring.rank_order, description, ring.transport)
    record.channels = [channels[key] for key in sorted(channels)]
    if record.algorithm.lower() == "tree" and not any(channel.topology_type == "tree" for channel in record.channels):
        record.unsupported_schedule = True
    return record
