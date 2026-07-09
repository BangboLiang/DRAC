"""Trace ingestion helpers for trace-driven reconfiguration planning.

Introduced by `trace_reconfig_plan.py`.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .trace_ir import TraceBundle, TraceOp, TraceRankGraph, role_key_for_rank


def _normalize_trace_event(event: Dict[str, Any], rank: int, raw_index: int) -> TraceOp:
    """Normalize one native trace event.

    Introduced by `trace_reconfig_plan.py`.
    """
    communicator = event.get("communicator") or {}
    et_node_id = int(raw_index + 1)
    return TraceOp(
        uid=str(event["id"]),
        et_node_id=et_node_id,
        name=str(event.get("op_name", event["id"])),
        event_type=str(event.get("event_type", "UNKNOWN")),
        phase=str(event.get("phase", "UNKNOWN")),
        rank=int(rank),
        coordinates=dict(event.get("coordinates") or {}),
        predecessors=tuple(str(dep) for dep in event.get("predecessors", [])),
        payload_bytes=int(event.get("payload_bytes", 0) or 0),
        flops=float(event.get("flops", 0.0) or 0.0),
        duration_us=(
            None
            if event.get("duration_us") is None
            else float(event.get("duration_us"))
        ),
        group_type=(
            None
            if communicator.get("group_type") is None
            else str(communicator["group_type"])
        ),
        group_ranks=tuple(int(x) for x in communicator.get("ranks", [])),
        peer_rank=(
            None if event.get("peer_rank") is None else int(event.get("peer_rank"))
        ),
        raw_index=int(raw_index),
    )


def _normalize_chakra_like_node(
    node: Dict[str, Any], rank: int, coordinates: Dict[str, int], raw_index: int
) -> TraceOp:
    """Normalize one Chakra-like JSON node.

    Introduced by `trace_reconfig_plan.py`.
    """
    et_node_id = int(raw_index + 1)
    return TraceOp(
        uid=str(node["id"]),
        et_node_id=et_node_id,
        name=str(node.get("name", node["id"])),
        event_type=str(node.get("type", "UNKNOWN")),
        phase=str(node.get("phase", "UNKNOWN")),
        rank=int(rank),
        coordinates=dict(coordinates),
        predecessors=tuple(str(dep) for dep in node.get("dependencies", [])),
        payload_bytes=int(node.get("comm_size", 0) or 0),
        flops=float(node.get("flops", 0.0) or 0.0),
        duration_us=(
            None
            if node.get("compute_cost") is None
            else float(node.get("compute_cost"))
        ),
        group_type=(
            None
            if node.get("communicator_type") is None
            else str(node.get("communicator_type"))
        ),
        group_ranks=tuple(int(x) for x in node.get("communicator_ranks", []) or []),
        peer_rank=None,
        raw_index=int(raw_index),
    )


def _load_native_trace_bundle(trace_dir: Path) -> TraceBundle:
    with (trace_dir / "trace_metadata.json").open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    num_ranks = int(metadata["summary"]["num_ranks"])
    ranks: Dict[int, TraceRankGraph] = {}
    for rank in range(num_ranks):
        with (trace_dir / f"trace_rank_{rank:03d}.json").open(
            "r", encoding="utf-8"
        ) as fh:
            trace = json.load(fh)
        ops = tuple(
            _normalize_trace_event(event, rank=int(trace["rank"]), raw_index=idx)
            for idx, event in enumerate(trace["events"])
        )
        ranks[int(trace["rank"])] = TraceRankGraph(
            rank=int(trace["rank"]),
            coordinates=dict(trace.get("coordinates") or {}),
            ops=ops,
            event_id_to_et_node_id={op.uid: op.et_node_id for op in ops},
        )
    return TraceBundle(metadata=dict(metadata), ranks=ranks)


def _load_chakra_like_bundle(trace_dir: Path) -> TraceBundle:
    metadata: Dict[str, Any] = {"schema_version": "chakra_like_bundle/v1"}
    ranks: Dict[int, TraceRankGraph] = {}
    for path in sorted(trace_dir.glob("chakra_like_rank_*.json")):
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        rank = int(data["rank"])
        coordinates = dict(
            (data.get("nodes") or [{}])[0].get("device", {}).get("coordinates", {})
        )
        ops = tuple(
            _normalize_chakra_like_node(
                node, rank=rank, coordinates=coordinates, raw_index=idx
            )
            for idx, node in enumerate(data.get("nodes", []))
        )
        ranks[rank] = TraceRankGraph(
            rank=rank,
            coordinates=coordinates,
            ops=ops,
            event_id_to_et_node_id={op.uid: op.et_node_id for op in ops},
        )
    metadata["summary"] = {"num_ranks": len(ranks)}
    return TraceBundle(metadata=metadata, ranks=ranks)


def load_trace_bundle(trace_dir: str | Path) -> TraceBundle:
    """Load a trace directory in native or Chakra-like JSON form.

    Introduced by `trace_reconfig_plan.py`.
    """
    trace_path = Path(trace_dir)
    if (trace_path / "trace_metadata.json").exists():
        return _load_native_trace_bundle(trace_path)
    if list(trace_path.glob("chakra_like_rank_*.json")):
        return _load_chakra_like_bundle(trace_path)
    raise FileNotFoundError(
        f"Could not find native trace JSON or chakra_like_rank_*.json in {trace_path}"
    )


def stable_topological_ops(rank_graph: TraceRankGraph) -> List[TraceOp]:
    """Return a stable topological order for one rank's DAG.

    Introduced by `trace_reconfig_plan.py`.
    """
    ops = list(rank_graph.ops)
    by_uid = {op.uid: op for op in ops}
    local_pred_count: Dict[str, int] = {}
    succs: Dict[str, List[str]] = defaultdict(list)
    for op in ops:
        local_preds = [dep for dep in op.predecessors if dep in by_uid]
        local_pred_count[op.uid] = len(local_preds)
        for dep in local_preds:
            succs[dep].append(op.uid)

    ready = deque(
        sorted(
            (op for op in ops if local_pred_count[op.uid] == 0),
            key=lambda x: x.raw_index,
        )
    )
    ordered: List[TraceOp] = []
    while ready:
        op = ready.popleft()
        ordered.append(op)
        for succ_uid in sorted(
            succs.get(op.uid, []), key=lambda uid: by_uid[uid].raw_index
        ):
            local_pred_count[succ_uid] -= 1
            if local_pred_count[succ_uid] == 0:
                ready.append(by_uid[succ_uid])

    if len(ordered) != len(ops):
        return sorted(ops, key=lambda op: op.raw_index)
    return ordered


def select_representative_rank(bundle: TraceBundle, policy: str = "middle") -> int:
    """Pick a canonical representative rank for planning.

    Introduced by `trace_reconfig_plan.py`.
    """
    policy_norm = str(policy).strip().lower()
    ordered = sorted(
        bundle.ranks,
        key=lambda rank: (
            int(bundle.ranks[rank].coordinates.get("pp_rank", 0)),
            int(bundle.ranks[rank].coordinates.get("dp_rank", 0)),
            int(bundle.ranks[rank].coordinates.get("tp_rank", 0)),
            rank,
        ),
    )
    if not ordered:
        raise ValueError("trace bundle has no ranks")
    if policy_norm == "rank0":
        return min(ordered)

    pp_size = bundle.pp_size
    preferred_role = {
        "first": "pp:first",
        "middle": "pp:middle",
        "last": "pp:last",
        "only": "pp:only",
    }.get(policy_norm)
    if preferred_role is None:
        raise ValueError(f"Unsupported representative policy: {policy}")

    candidates = [
        rank
        for rank in ordered
        if role_key_for_rank(bundle.ranks[rank], pp_size) == preferred_role
    ]
    if candidates:
        return candidates[0]

    fallback_order = ["pp:middle", "pp:first", "pp:last", "pp:only"]
    for role in fallback_order:
        candidates = [
            rank
            for rank in ordered
            if role_key_for_rank(bundle.ranks[rank], pp_size) == role
        ]
        if candidates:
            return candidates[0]
    return ordered[0]
