"""Emit Astra-sim mutable-topology configs from a planning result.

Introduced by `trace_reconfig_plan.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from .execution import BWSegmentPlan
from .trace_ir import CommTriggerRef

Edge = Tuple[int, int]


def _fmt_float(value: float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def render_mutable_network_yaml(
    *,
    npus_count: int,
    default_bandwidth_gbps: float,
    default_latency_ns: float,
    initial_state: Mapping[Edge, float],
) -> str:
    """Render a mutable-topology analytical network config.

    Introduced by `trace_reconfig_plan.py`.
    """
    lines = [
        "topology: [ Mutable ]",
        f"npus_count: [ {int(npus_count)} ]",
        f"bandwidth: [ {_fmt_float(default_bandwidth_gbps)} ]",
        f"latency: [ {_fmt_float(default_latency_ns)} ]",
        "",
        "links:",
    ]
    for (src, dst), bw in sorted(initial_state.items()):
        lines.append(
            "  - { src: %d, dest: %d, bandwidth: %s, latency: %s }"
            % (src, dst, _fmt_float(bw), _fmt_float(default_latency_ns))
        )
    if not initial_state:
        lines.append("  []")
    lines.extend(["", "reconfiguration-events:", "  []", ""])
    return "\n".join(lines)


def build_dag_reconfiguration_events(
    *,
    states: List[Mapping[Edge, float]],
    trigger_refs: List[CommTriggerRef],
    trigger_rank: int,
    trigger_phase: str = "start",
    latency_ns: float | None = None,
    bandwidth_eps: float = 1e-9,
) -> List[Dict[str, Any]]:
    """Build `dag-reconfiguration-events` for BW-segment transitions.

    Introduced by `trace_reconfig_plan.py`.
    """
    if len(states) <= 1:
        return []
    events: List[Dict[str, Any]] = []
    prior_existing = {
        edge for edge, bw in states[0].items() if float(bw) > bandwidth_eps
    }
    for seg_idx in range(1, len(states)):
        old = {
            edge: float(bw)
            for edge, bw in states[seg_idx - 1].items()
            if float(bw) > bandwidth_eps
        }
        new = {
            edge: float(bw)
            for edge, bw in states[seg_idx].items()
            if float(bw) > bandwidth_eps
        }
        trigger = trigger_refs[seg_idx]
        all_edges = sorted(set(old) | set(new))
        for edge in all_edges:
            old_bw = float(old.get(edge, 0.0))
            new_bw = float(new.get(edge, 0.0))
            action: Dict[str, Any] | None = None
            if old_bw <= bandwidth_eps and new_bw > bandwidth_eps:
                action = {
                    "type": "add-link",
                    "src": int(edge[0]),
                    "dest": int(edge[1]),
                    "bandwidth": float(new_bw),
                }
                if latency_ns is not None:
                    action["latency"] = float(latency_ns)
                prior_existing.add(edge)
            elif old_bw > bandwidth_eps and new_bw <= bandwidth_eps:
                action = {
                    "type": "remove-link",
                    "src": int(edge[0]),
                    "dest": int(edge[1]),
                }
            elif abs(old_bw - new_bw) > bandwidth_eps:
                action = {
                    "type": "set-bandwidth",
                    "src": int(edge[0]),
                    "dest": int(edge[1]),
                    "bandwidth": float(new_bw),
                }
            if action is None:
                continue
            events.append(
                {
                    "rank": int(trigger_rank),
                    "node-id": int(trigger.et_node_id),
                    "phase": str(trigger_phase),
                    "action": action,
                }
            )
    return events


def build_rank_scoped_dag_reconfiguration_events(
    *,
    states: List[Mapping[Edge, float]],
    rank_trigger_refs: Mapping[int, List[CommTriggerRef | None]],
    fallback_trigger_refs: List[CommTriggerRef],
    trigger_phase: str = "start",
    latency_ns: float | None = None,
    bandwidth_eps: float = 1e-9,
) -> List[Dict[str, Any]]:
    """Build reconfiguration events triggered by each edge source rank.

    Introduced by `trace_reconfig_plan.py`.
    """
    if len(states) <= 1:
        return []
    events: List[Dict[str, Any]] = []
    for seg_idx in range(1, len(states)):
        old = {
            edge: float(bw)
            for edge, bw in states[seg_idx - 1].items()
            if float(bw) > bandwidth_eps
        }
        new = {
            edge: float(bw)
            for edge, bw in states[seg_idx].items()
            if float(bw) > bandwidth_eps
        }
        all_edges = sorted(set(old) | set(new))
        for edge in all_edges:
            old_bw = float(old.get(edge, 0.0))
            new_bw = float(new.get(edge, 0.0))
            action: Dict[str, Any] | None = None
            if old_bw <= bandwidth_eps and new_bw > bandwidth_eps:
                action = {
                    "type": "add-link",
                    "src": int(edge[0]),
                    "dest": int(edge[1]),
                    "bandwidth": float(new_bw),
                }
                if latency_ns is not None:
                    action["latency"] = float(latency_ns)
            elif old_bw > bandwidth_eps and new_bw <= bandwidth_eps:
                action = {
                    "type": "remove-link",
                    "src": int(edge[0]),
                    "dest": int(edge[1]),
                }
            elif abs(old_bw - new_bw) > bandwidth_eps:
                action = {
                    "type": "set-bandwidth",
                    "src": int(edge[0]),
                    "dest": int(edge[1]),
                    "bandwidth": float(new_bw),
                }
            if action is None:
                continue
            trigger_rank = int(edge[0])
            rank_refs = rank_trigger_refs.get(trigger_rank, [])
            trigger = (
                rank_refs[seg_idx]
                if seg_idx < len(rank_refs) and rank_refs[seg_idx] is not None
                else fallback_trigger_refs[seg_idx]
            )
            events.append(
                {
                    "rank": int(trigger_rank),
                    "node-id": int(trigger.et_node_id),
                    "phase": str(trigger_phase),
                    "action": action,
                }
            )
    return events


def densify_edge_states(
    states: List[Mapping[Edge, float]],
    *,
    floor_bandwidth_gbps: float,
) -> List[Dict[Edge, float]]:
    """Ensure every state contains every known edge with a positive floor bandwidth.

    Introduced by `trace_reconfig_plan.py`.
    """
    universe: set[Edge] = set()
    for state in states:
        universe.update(state.keys())
    dense: List[Dict[Edge, float]] = []
    for state in states:
        dense_state: Dict[Edge, float] = {}
        for edge in universe:
            dense_state[edge] = float(state.get(edge, float(floor_bandwidth_gbps)))
            if dense_state[edge] <= 0.0:
                dense_state[edge] = float(floor_bandwidth_gbps)
        dense.append(dense_state)
    return dense


def write_astra_plan_bundle(
    *,
    output_dir: str | Path,
    network_yaml: str,
    system_json: Mapping[str, Any],
    summary_json: Mapping[str, Any],
) -> None:
    """Write the emitted Astra plan bundle to disk.

    Introduced by `trace_reconfig_plan.py`.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "network_cfg.yml").write_text(network_yaml, encoding="utf-8")
    with (out_dir / "system_cfg.json").open("w", encoding="utf-8") as fh:
        json.dump(system_json, fh, indent=2)
    with (out_dir / "plan_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary_json, fh, indent=2)
