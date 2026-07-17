"""Official-text GOAL adapter introduced by V4 ATLAHS rescue experiments.

This parser follows the grammar accepted by ATLAHS' pinned LogGOPSim txt2bin
reader. It intentionally rejects compact .bin files; binary schedules must be
read by the official C++ Parser.hpp, never inferred here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import sqlite3
from typing import Iterator

EVIDENCE_TRACE_DERIVED = "ATLAHS_TRACE_DERIVED"
_HEADER = re.compile(r"^num_ranks\s+(\d+)\s*$")
_RANK = re.compile(r"^rank\s+(\d+)\s*\{\s*$")
_SEND = re.compile(r"^(\w+):\s+send\s+(\d+)b\s+to\s+(\d+)\s+tag\s+(\S+)(?:\s+cpu\s+(\d+))?(?:\s*nic\s+(\d+))?\s*$")
_RECV = re.compile(r"^(\w+):\s+recv\s+(\d+)b\s+from\s+(\d+)\s+tag\s+(\S+)(?:\s+cpu\s+(\d+))?(?:\s*nic\s+(\d+))?\s*$")
_CALC = re.compile(r"^(\w+):\s+calc\s+(\d+)(?:\s+cpu\s+(\d+))?(?:\s*nic\s+(\d+))?\s*$")
_DEP = re.compile(r"^(\w+)\s+(i?requires)\s+(\w+)\s*$")


@dataclass(frozen=True)
class GoalRank:
    rank: int
    server_id: int
    mapping_source: str


@dataclass(frozen=True)
class GoalOperation:
    operation_id: str
    rank: int
    operation_type: str
    bytes_or_ns: int
    peer: int | None = None
    tag: str | None = None
    cpu: int | None = None
    nic: int | None = None
    sequence: int = 0


@dataclass(frozen=True)
class GoalSendEvent(GoalOperation):
    evidence_label: str = EVIDENCE_TRACE_DERIVED


@dataclass(frozen=True)
class GoalRecvEvent(GoalOperation):
    evidence_label: str = EVIDENCE_TRACE_DERIVED


@dataclass(frozen=True)
class GoalComputeEvent(GoalOperation):
    evidence_label: str = EVIDENCE_TRACE_DERIVED


@dataclass(frozen=True)
class GoalDependency:
    rank: int
    operation_id: str
    requires_id: str
    start_dependency: bool


@dataclass
class AtlahsGoalExecution:
    trace_name: str
    source_file: str
    num_ranks: int
    model: str = "unknown"
    gpu_count: int | None = None
    node_count: int | None = None
    ranks: list[GoalRank] = field(default_factory=list)


def ensure_official_text_goal(path: Path) -> None:
    if path.suffix.lower() in {".bin", ".gz", ".bz2"}:
        raise ValueError("compact/unknown GOAL input is not parsed by inference; use ATLAHS sim/LogGOPSim/Parser.hpp")
    with path.open("rb") as handle:
        prefix = handle.read(128)
    if b"\x00" in prefix or not prefix.lstrip().startswith(b"num_ranks"):
        raise ValueError("input is not official GOAL text; refusing to guess a binary format")


def iter_goal(path: Path) -> Iterator[GoalOperation | GoalDependency | tuple[str, int]]:
    ensure_official_text_goal(path)
    rank: int | None = None
    sequence = 0
    with path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            if match := _HEADER.match(line):
                yield ("num_ranks", int(match.group(1)))
                continue
            if match := _RANK.match(line):
                rank = int(match.group(1)); sequence = 0
                yield ("rank", rank)
                continue
            if line == "}":
                rank = None
                continue
            if rank is None:
                raise ValueError(f"operation outside rank block at {path}:{line_number}")
            sequence += 1
            if match := _SEND.match(line):
                yield GoalSendEvent(match.group(1), rank, "send", int(match.group(2)), int(match.group(3)), match.group(4), _optint(match.group(5)), _optint(match.group(6)), sequence)
            elif match := _RECV.match(line):
                yield GoalRecvEvent(match.group(1), rank, "recv", int(match.group(2)), int(match.group(3)), match.group(4), _optint(match.group(5)), _optint(match.group(6)), sequence)
            elif match := _CALC.match(line):
                yield GoalComputeEvent(match.group(1), rank, "calc", int(match.group(2)), None, None, _optint(match.group(3)), _optint(match.group(4)), sequence)
            elif match := _DEP.match(line):
                yield GoalDependency(rank, match.group(1), match.group(3), match.group(2) == "irequires")
            else:
                raise ValueError(f"unsupported GOAL syntax at {path}:{line_number}: {line[:120]}")


def _optint(value: str | None) -> int | None:
    return None if value is None else int(value)


def parse_goal_to_sqlite(path: Path, database: Path, trace_name: str, batch_size: int = 100_000,
                         communication_only: bool | None = None) -> dict[str, int]:
    if communication_only is None:
        communication_only = path.stat().st_size > 1_000_000_000
    database.parent.mkdir(parents=True, exist_ok=True)
    if database.exists():
        database.unlink()
    connection = sqlite3.connect(database)
    connection.executescript("""
      PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;
      CREATE TABLE events(id INTEGER PRIMARY KEY, trace_name TEXT, rank INTEGER, operation_id TEXT,
        operation_type TEXT, peer INTEGER, bytes_or_ns INTEGER, tag TEXT, cpu INTEGER, nic INTEGER,
        sequence INTEGER, evidence_label TEXT);
      CREATE TABLE dependencies(rank INTEGER, operation_id TEXT, requires_id TEXT, start_dependency INTEGER);
      CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT);
      CREATE INDEX event_type_idx ON events(operation_type, rank, peer);
      CREATE INDEX event_tag_idx ON events(tag, rank, peer, bytes_or_ns);
    """)
    events: list[tuple] = []; deps: list[tuple] = []
    counts = {"num_ranks": 0, "send": 0, "recv": 0, "calc": 0, "dependencies": 0}
    for item in iter_goal(path):
        if isinstance(item, tuple):
            if item[0] == "num_ranks": counts["num_ranks"] = item[1]
            continue
        if isinstance(item, GoalDependency):
            if not communication_only:
                deps.append((item.rank, item.operation_id, item.requires_id, int(item.start_dependency)))
            counts["dependencies"] += 1
        else:
            if not communication_only or item.operation_type in {"send", "recv"}:
                events.append((trace_name, item.rank, item.operation_id, item.operation_type, item.peer,
                               item.bytes_or_ns, item.tag, item.cpu, item.nic, item.sequence, EVIDENCE_TRACE_DERIVED))
            counts[item.operation_type] += 1
        if len(events) >= batch_size:
            connection.executemany("INSERT INTO events(trace_name,rank,operation_id,operation_type,peer,bytes_or_ns,tag,cpu,nic,sequence,evidence_label) VALUES(?,?,?,?,?,?,?,?,?,?,?)", events); events.clear()
        if len(deps) >= batch_size:
            connection.executemany("INSERT INTO dependencies VALUES(?,?,?,?)", deps); deps.clear()
    if events:
        connection.executemany("INSERT INTO events(trace_name,rank,operation_id,operation_type,peer,bytes_or_ns,tag,cpu,nic,sequence,evidence_label) VALUES(?,?,?,?,?,?,?,?,?,?,?)", events)
    if deps:
        connection.executemany("INSERT INTO dependencies VALUES(?,?,?,?)", deps)
    connection.execute("INSERT INTO metadata VALUES('parse_complete','1')")
    connection.execute("INSERT INTO metadata VALUES('communication_only',?)", ("1" if communication_only else "0",))
    for key, value in counts.items():
        connection.execute("INSERT INTO metadata VALUES(?,?)", (f"count_{key}", str(value)))
    connection.commit(); connection.close()
    return counts


def sqlite_parse_complete(database: Path) -> bool:
    if not database.exists():
        return False
    connection = None
    try:
        connection = sqlite3.connect(database)
        value = connection.execute("SELECT value FROM metadata WHERE key='parse_complete'").fetchone()
        return value == ("1",)
    except sqlite3.Error:
        return False
    finally:
        if connection is not None:
            connection.close()


def integrity_from_sqlite(database: Path) -> dict[str, object]:
    connection = sqlite3.connect(database)
    scalar = lambda sql: connection.execute(sql).fetchone()[0]
    ranks = scalar("SELECT COUNT(DISTINCT rank) FROM events")
    send_bytes = scalar("SELECT COALESCE(SUM(bytes_or_ns),0) FROM events WHERE operation_type='send'")
    recv_bytes = scalar("SELECT COALESCE(SUM(bytes_or_ns),0) FROM events WHERE operation_type='recv'")
    invalid_peers = scalar("SELECT COUNT(*) FROM events WHERE operation_type IN ('send','recv') AND (peer<0 OR peer>=?)".replace("?", str(ranks)))
    unmatched = scalar("""SELECT COUNT(*) FROM (
      SELECT rank,peer,bytes_or_ns,tag,COUNT(*) c FROM events WHERE operation_type='send' GROUP BY rank,peer,bytes_or_ns,tag
      EXCEPT SELECT peer,rank,bytes_or_ns,tag,COUNT(*) c FROM events WHERE operation_type='recv' GROUP BY peer,rank,bytes_or_ns,tag)""")
    metadata = dict(connection.execute("SELECT key,value FROM metadata").fetchall()) if scalar("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='metadata'") else {}
    result = {"rank_count": ranks, "send_events": int(metadata.get("count_send", scalar("SELECT COUNT(*) FROM events WHERE operation_type='send'"))),
              "receive_events": scalar("SELECT COUNT(*) FROM events WHERE operation_type='recv'"),
              "compute_events": int(metadata.get("count_calc", scalar("SELECT COUNT(*) FROM events WHERE operation_type='calc'"))),
              "dependency_count": int(metadata.get("count_dependencies", scalar("SELECT COUNT(*) FROM dependencies"))), "total_send_bytes": send_bytes,
              "total_receive_bytes": recv_bytes, "byte_difference": send_bytes-recv_bytes,
              "invalid_peer_events": invalid_peers, "unmatched_send_groups": unmatched,
              "has_timestamps": False, "has_collective_labels": False, "has_channel_labels": False,
              "dependency_edges_materialized": metadata.get("communication_only","0") != "1",
              "evidence_label": EVIDENCE_TRACE_DERIVED}
    connection.close()
    return result
