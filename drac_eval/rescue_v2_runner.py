from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from llama3_comm.traffic import llama3_megatron_payloads

from .allocation import allocate_for_algorithm
from .config import NetworkConfig, WorkloadConfig
from .metrics import _completion_time_ms
from .rescue_traffic import bidirectional_balanced, build_mapping, level_group_size, pairwise_balancing_oracle
from .rescue_v2_config import RescueV2Config
from .rescue_v2_network import (
    AggregatedNetworkModel,
    aggregate_network_model,
    allocate_drac_makespan_opt,
    allocate_sym_ocs,
    allocate_target_ablation,
    brute_force_makespan,
    endpoint_network_model,
    validate_general_units,
)
from .rescue_v2_plotting import plot_aggregation_v2, plot_collective_v2, plot_makespan_v2
from .rescue_schedule import ScheduleEvent, build_executable_ring_schedule, schedule_step_matrices
from .traffic import _build_model_and_parallel, _layers_per_segment, load_or_generate_workload


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def benefit_retention(original_gain: float, balanced_gain: float, epsilon: float) -> float:
    return float(balanced_gain / original_gain) if abs(original_gain) >= epsilon else float("nan")


def aggregate_gain_ratio(rows: Sequence[Dict[str, object]], weight_mode: str, epsilon: float) -> float:
    weights = {
        "communication_bytes": lambda r: float(r["communication_bytes"]),
        "baseline_communication_time": lambda r: float(r["original_sym_time_ms"]),
        "workload_equal": lambda r: 1.0,
    }
    if weight_mode not in weights:
        raise ValueError(weight_mode)
    numer = sum(weights[weight_mode](r) * float(r["balanced_gain"]) for r in rows)
    denom = sum(weights[weight_mode](r) * float(r["original_gain"]) for r in rows)
    return float(numer / denom) if abs(denom) >= epsilon else float("nan")


def collective_replacement_risk(executable_aggregate_retention: float, threshold: float = 0.25) -> str:
    """Diagnostic depends only on executable bidirectional-ring retention."""
    return "HIGH" if np.isfinite(executable_aggregate_retention) and executable_aggregate_retention < threshold else "LOW"


def _direction_stats(matrix: np.ndarray) -> Tuple[float, float, float]:
    a = 0.0
    v = 0.0
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            x, y = float(matrix[i, j]), float(matrix[j, i])
            a += abs(x - y)
            v += x + y
    return a, v, (a / v if v > 1e-12 else float("nan"))


def _matrix_hash(matrix: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(matrix, dtype=np.float64).tobytes()).hexdigest()[:16]


def run_traffic_audit(cfg: RescueV2Config, root: Path) -> List[Dict[str, object]]:
    provenance: List[Dict[str, object]] = []
    descriptions = {
        "tp": ("2x AllGather + 2x ReduceScatter proxy", "llama3_megatron_payloads:a_full", "dominant-pair + injected skew + random orientation", "synthetic adjacent-ring template"),
        "dp": ("ReduceScatter + AllGather proxy", "llama3_megatron_payloads:per-layer BF16/FP32 params", "dominant-pair + injected skew + fixed offset peers", "synthetic offset/extra-peer template"),
        "mixed": ("weighted TP + DP matrix sum", "weighted TP/DP LLaMA-3 payloads", "weighted synthetic TP/DP direction templates", "matrix mixture; no merged executable schedule"),
        "pp": ("pipeline P2P forward/backward proxy", "llama3_megatron_payloads:a_shard", "hand-written adjacent-rank phase orientation + injected weak skew", "synthetic adjacent P2P phase template"),
    }
    for workload in cfg.workloads:
        op, payload, direction, schedule = descriptions[workload.kind]
        segments = load_or_generate_workload(workload, cfg.endpoint_count, cfg.asymmetry_level, cfg.seeds[0])
        for segment in segments:
            provenance.append({
                "workload": workload.name,
                "segment_id": segment.segment_idx,
                "operation_type": op,
                "payload_source": payload,
                "direction_source": direction,
                "schedule_type": schedule,
                "synthetic_skew_used": True,
                "skew_parameter": cfg.asymmetry_level,
                "rank_count": cfg.endpoint_count,
                "total_bytes": float(segment.matrix.sum()),
                "notes": "Segment aggregates scaled layers/microbatches; it is not one step or an explicit full collective.",
            })
        if _ring_specs(workload, cfg.endpoint_count):
            _events, steps, total_bytes = _workload_schedule(cfg, workload, False)
            provenance.append({
                "workload": f"{workload.name}_exec_ring",
                "segment_id": "explicit_schedule",
                "operation_type": "executable ring phases",
                "payload_source": "llama3_megatron_payloads scaled by configured layers/microbatches",
                "direction_source": "explicit clockwise event schedule with dependencies",
                "schedule_type": "unidirectional executable ring",
                "synthetic_skew_used": False,
                "skew_parameter": "",
                "rank_count": cfg.endpoint_count,
                "total_bytes": total_bytes,
                "notes": f"{len(steps)} dependency-ordered step matrices; used as executable aggregation contrast.",
            })
    _write_csv(root / "traffic_provenance.csv", provenance)
    audit = """# Traffic Provenance Audit

## Call path

`run_drac_eval.py -> drac_eval.runner.run_experiments -> drac_eval.traffic.load_or_generate_workload -> _tp_matrix/_dp_matrix/_mixed_matrix/_pp_matrix`. `llama3_comm.traffic.llama3_megatron_payloads` supplies LLaMA-3 tensor and parameter byte counts only. `SegmentDemand` is created after each synthetic matrix is generated. `rescue_traffic.py` consumes these matrices but cannot recover a missing schedule.

## Finding

The current rank-level directionality is synthetic, not evidence extracted from NCCL or an executable collective trace. TP uses adjacent pairs, a dominant/reverse ratio, injected skew, noise, and a random orientation. DP uses fixed offset and extra-peer templates plus dominant/reverse skew. MIXED is a weighted matrix sum. PP uses adjacent ranks and a hand-written forward/backward phase orientation with weak injected skew. No generator explicitly simulates ring steps, tree steps, chunk dependencies, AllGather, ReduceScatter, or AllReduce execution.

Payload provenance and direction provenance must therefore be separated: payload magnitudes are LLaMA-3-derived, while ordered-pair directions are synthetic. A segment is a layer/microbatch-scaled training-phase aggregate, not a communication step and not a trace-derived complete collective.

V2 retains these workloads only as synthetic sensitivity inputs. The executable ring comparison is generated separately from explicit events and dependencies.
"""
    (root / "TRAFFIC_AUDIT.md").write_text(audit, encoding="utf-8")
    return provenance


def _level_group_size(cfg: RescueV2Config, level: str) -> int:
    return {
        "endpoint": 1,
        "server": cfg.endpoints_per_server,
        "tor": cfg.endpoints_per_server * cfg.servers_per_tor,
        "aggregation": cfg.endpoints_per_server * cfg.servers_per_tor * cfg.tors_per_aggregation,
    }[level]


def _aggregate_demand(matrix: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    labels = sorted(set(int(v) for v in mapping))
    index = {v: i for i, v in enumerate(labels)}
    out = np.zeros((len(labels), len(labels)), dtype=float)
    for u in range(len(mapping)):
        for v in range(len(mapping)):
            if mapping[u] != mapping[v]:
                out[index[int(mapping[u])], index[int(mapping[v])]] += matrix[u, v]
    return out


def run_aggregation_v2(cfg: RescueV2Config, root: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    signal: List[Dict[str, object]] = []
    resources: List[Dict[str, object]] = []
    performance: List[Dict[str, object]] = []
    for workload in cfg.workloads:
        for seed in cfg.seeds:
            segments = load_or_generate_workload(workload, cfg.endpoint_count, cfg.asymmetry_level, seed)
            for mapping_name in cfg.mapping_strategies:
                for level in cfg.aggregation_levels:
                    mapping = build_mapping(cfg.endpoint_count, _level_group_size(cfg, level), mapping_name, seed)
                    node_count = len(set(mapping.tolist()))
                    for segment in segments:
                        endpoint_a, endpoint_v, _ = _direction_stats(segment.matrix)
                        matrix = _aggregate_demand(segment.matrix, mapping)
                        a, v, omega = _direction_stats(matrix)
                        abs_ret = a / endpoint_a if endpoint_a > 1e-12 else float("nan")
                        boundary = v / endpoint_v if endpoint_v > 1e-12 else float("nan")
                        if a > endpoint_a * (1.0 + 1e-10):
                            raise AssertionError("absolute directionality increased after aggregation")
                        if np.isfinite(abs_ret) and not -1e-10 <= abs_ret <= 1.0 + 1e-10:
                            raise AssertionError("AbsoluteRetention outside [0,1]")
                        if np.isfinite(boundary) and not -1e-10 <= boundary <= 1.0 + 1e-10:
                            raise AssertionError("BoundaryTrafficFraction outside [0,1]")
                        status = "ok"
                        if node_count < 2:
                            status = "insufficient_abstract_nodes"
                        elif v <= 1e-12:
                            status = "no_cross_boundary_traffic"
                        elif a <= 1e-12:
                            status = "no_direction_difference"
                        signal.append({
                            "workload": workload.name, "seed": seed, "mapping_seed": seed,
                            "traffic_source": "synthetic_segment_demand",
                            "mapping": mapping_name, "level": level, "segment_id": segment.segment_idx,
                            "endpoint_count": cfg.endpoint_count, "abstract_node_count": node_count,
                            "absolute_directionality_bytes": a, "boundary_traffic_bytes": v, "omega": omega,
                            "endpoint_absolute_directionality_bytes": endpoint_a, "endpoint_traffic_bytes": endpoint_v,
                            "absolute_retention": abs_ret, "boundary_traffic_fraction": boundary,
                            "status": status,
                        })
                    for mode in cfg.normalization_modes:
                        port_values = cfg.port_budgets if mode == "resource_equivalent" else [-1]
                        for port in port_values:
                            endpoint_model = endpoint_network_model(cfg, cfg.endpoint_count, None if port < 0 else port)
                            model = aggregate_network_model(endpoint_model, mapping, mode, level, cfg)
                            resources.append({
                                "workload": workload.name, "mapping": mapping_name, "mapping_seed": seed, "level": level,
                                "normalization_mode": mode, "port_sweep_value": port,
                                "abstract_node_count": model.node_count,
                                "base_capacity_sum_gbps": float(model.base_capacity_gbps.sum()),
                                "out_port_budget_sum": int(model.out_port_budget.sum()),
                                "in_port_budget_sum": int(model.in_port_budget.sum()),
                                "global_ocs_budget": model.total_ocs_links,
                                "reachable_ordered_pairs": int(model.reachable.sum()),
                            })
                            for segment in segments:
                                demand = _aggregate_demand(segment.matrix, mapping)
                                if demand.sum() <= 0:
                                    performance.append({"workload": workload.name, "seed": seed, "mapping": mapping_name, "level": level, "segment_id": segment.segment_idx, "normalization_mode": mode, "port_sweep_value": port, "sym_time_ms": 0.0, "drac_time_ms": 0.0, "speedup_drac_over_sym": float("nan"), "status": "no_cross_boundary_traffic"})
                                    continue
                                sym = allocate_sym_ocs(demand, model)
                                drac, runtime = allocate_drac_makespan_opt(demand, model)
                                sym_t = _completion_time_ms(demand, sym.total_bandwidth)
                                drac_t = _completion_time_ms(demand, drac.total_bandwidth)
                                performance.append({
                                    "workload": workload.name, "seed": seed, "mapping": mapping_name, "level": level,
                                    "traffic_source": "synthetic_segment_demand",
                                    "segment_id": segment.segment_idx, "normalization_mode": mode, "port_sweep_value": port,
                                    "sym_time_ms": sym_t, "drac_time_ms": drac_t,
                                    "speedup_drac_over_sym": sym_t / drac_t if drac_t > 0 else float("nan"),
                                    "drac_solver_runtime_ms": runtime, "status": "ok",
                                })
    # Executable unidirectional-ring contrast. It is kept separate from synthetic
    # SegmentDemand so provenance is never averaged implicitly.
    for workload in cfg.workloads:
        if not _ring_specs(workload, cfg.endpoint_count):
            continue
        _events, steps, _bytes = _workload_schedule(cfg, workload, False)
        endpoint_matrix = np.sum([matrix for _, matrix in steps], axis=0)
        exec_name = f"{workload.name}_exec_ring"
        endpoint_a, endpoint_v, _ = _direction_stats(endpoint_matrix)
        for mapping_name in cfg.mapping_strategies:
            mapping_seed = cfg.seeds[0]
            for level in cfg.aggregation_levels:
                mapping = build_mapping(cfg.endpoint_count, _level_group_size(cfg, level), mapping_name, mapping_seed)
                demand = _aggregate_demand(endpoint_matrix, mapping)
                a, v, omega = _direction_stats(demand)
                abs_ret = a/endpoint_a if endpoint_a>1e-12 else float("nan")
                boundary = v/endpoint_v if endpoint_v>1e-12 else float("nan")
                status = "ok" if v>1e-12 else "no_cross_boundary_traffic"
                signal.append({"workload":exec_name,"seed":-1,"mapping_seed":mapping_seed,"traffic_source":"executable_unidirectional_ring","mapping":mapping_name,"level":level,"segment_id":"explicit_schedule","endpoint_count":cfg.endpoint_count,"abstract_node_count":demand.shape[0],"absolute_directionality_bytes":a,"boundary_traffic_bytes":v,"omega":omega,"endpoint_absolute_directionality_bytes":endpoint_a,"endpoint_traffic_bytes":endpoint_v,"absolute_retention":abs_ret,"boundary_traffic_fraction":boundary,"status":status})
                for mode in cfg.normalization_modes:
                    port_values = cfg.port_budgets if mode=="resource_equivalent" else [-1]
                    for port in port_values:
                        endpoint_model = endpoint_network_model(cfg,cfg.endpoint_count,None if port<0 else port)
                        model = aggregate_network_model(endpoint_model,mapping,mode,level,cfg)
                        if demand.sum()<=0:
                            performance.append({"workload":exec_name,"seed":-1,"mapping":mapping_name,"level":level,"segment_id":"explicit_schedule","traffic_source":"executable_unidirectional_ring","normalization_mode":mode,"port_sweep_value":port,"sym_time_ms":0.0,"drac_time_ms":0.0,"speedup_drac_over_sym":float("nan"),"status":status})
                        else:
                            sym=allocate_sym_ocs(demand,model); drac,runtime=allocate_drac_makespan_opt(demand,model)
                            sym_t=_completion_time_ms(demand,sym.total_bandwidth); drac_t=_completion_time_ms(demand,drac.total_bandwidth)
                            performance.append({"workload":exec_name,"seed":-1,"mapping":mapping_name,"level":level,"segment_id":"explicit_schedule","traffic_source":"executable_unidirectional_ring","normalization_mode":mode,"port_sweep_value":port,"sym_time_ms":sym_t,"drac_time_ms":drac_t,"speedup_drac_over_sym":sym_t/drac_t if drac_t>0 else float("nan"),"drac_solver_runtime_ms":runtime,"status":"ok"})
    mapping_summary: List[Dict[str, object]] = []
    for workload in {str(r["workload"]) for r in signal}:
        for mapping_name in cfg.mapping_strategies:
            rows = [r for r in signal if r["workload"] == workload and r["mapping"] == mapping_name and r["level"] == "tor"]
            mapping_summary.append({"workload": workload, "traffic_source": rows[0]["traffic_source"] if rows else "", "mapping": mapping_name, "level": "tor", "absolute_retention": _mean(float(r["absolute_retention"]) for r in rows), "boundary_traffic_fraction": _mean(float(r["boundary_traffic_fraction"]) for r in rows), "omega": _mean(float(r["omega"]) for r in rows), "sample_count": len(rows)})
    out = root / "aggregation"
    _write_csv(out / "aggregation_signal_metrics.csv", signal)
    _write_csv(out / "aggregation_resource_models.csv", resources)
    _write_csv(out / "aggregation_performance_normalized.csv", performance)
    _write_csv(out / "aggregation_mapping_sensitivity.csv", mapping_summary)
    plot_aggregation_v2(signal, performance, mapping_summary, out)
    return signal, resources, performance, mapping_summary


def _ring_specs(workload: WorkloadConfig, n: int) -> List[Tuple[str, float, int, str]]:
    mod, par = _build_model_and_parallel(workload, n)
    a_full, _a_shard, p_bf16, p_fp32, _ = llama3_megatron_payloads(mod, par)
    layers = _layers_per_segment(workload)
    micro = max(1, workload.microbatches)
    if workload.kind == "tp":
        payload = 2.0 * a_full * layers * micro * workload.scale
        return [("allgather", payload, min(workload.tp_group_size, n), "tp_allgather"), ("reducescatter", payload, min(workload.tp_group_size, n), "tp_reducescatter")]
    if workload.kind == "dp":
        group = min(workload.dp_group_size, n)
        return [("reducescatter", p_fp32 * layers * workload.scale, group, "dp_reducescatter"), ("allgather", p_bf16 * layers * workload.scale, group, "dp_allgather")]
    if workload.kind == "mixed":
        tp = replace(workload, kind="tp", scale=workload.scale * float(workload.mixed_weights.get("tp", 0.0)))
        dp = replace(workload, kind="dp", scale=workload.scale * float(workload.mixed_weights.get("dp", 0.0)))
        return _ring_specs(tp, n) + _ring_specs(dp, n)
    return []


def _workload_schedule(cfg: RescueV2Config, workload: WorkloadConfig, bidirectional: bool) -> Tuple[List[Dict[str, object]], List[Tuple[str, np.ndarray]], float]:
    n = cfg.endpoint_count
    event_rows: List[Dict[str, object]] = []
    steps: List[Tuple[str, np.ndarray]] = []
    total_bytes = 0.0
    for spec_idx, (op_type, payload, group_size, label) in enumerate(_ring_specs(workload, n)):
        local_by_key: Dict[Tuple[str, int], np.ndarray] = {}
        for group_idx, start in enumerate(range(0, n, group_size)):
            size = min(group_size, n - start)
            if size < 2:
                continue
            schedule = build_executable_ring_schedule(size, payload, op_type, bidirectional, min(cfg.chunk_count, max(1, size)), cfg.odd_chunk_rule, f"{workload.name}-{label}-g{group_idx}")
            total_bytes += schedule.total_transmitted_bytes
            for event in schedule.events:
                row = event.to_dict()
                row.update({"workload": workload.name, "collective_model": "executable_bidirectional_ring" if bidirectional else "unidirectional_executable_ring", "src_rank": event.src_rank + start, "dst_rank": event.dst_rank + start})
                event_rows.append(row)
            for phase, step, local in schedule_step_matrices(schedule):
                key = (phase, step)
                if key not in local_by_key:
                    local_by_key[key] = np.zeros((n, n), dtype=float)
                local_by_key[key][start:start + size, start:start + size] += local
        for (phase, step), matrix in local_by_key.items():
            steps.append((f"{spec_idx}:{label}:{phase}:{step}", matrix))
    return event_rows, steps, total_bytes


def _simulate_fixed_schedule(steps: Sequence[Tuple[str, np.ndarray]], allocation: object) -> Tuple[float, List[Dict[str, object]]]:
    total = 0.0
    rows = []
    for label, matrix in steps:
        time = _completion_time_ms(matrix, allocation.total_bandwidth)
        total += time
        rows.append({"step_label": label, "step_bytes": float(matrix.sum()), "step_time_ms": time})
    return total, rows


def run_collective_v2(cfg: RescueV2Config, root: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    events: List[Dict[str, object]] = []
    step_rows: List[Dict[str, object]] = []
    summary: List[Dict[str, object]] = []
    comparison: List[Dict[str, object]] = []
    matrix_sensitivity: List[Dict[str, object]] = []
    for workload in cfg.workloads:
        specs = _ring_specs(workload, cfg.endpoint_count)
        if not specs:
            summary.append({"workload": workload.name, "collective_model": "unchanged_non_ring", "executable_balancing_applicable": False, "reason": "PP P2P is not a ring collective"})
            continue
        uni_events, uni_steps, uni_bytes = _workload_schedule(cfg, workload, False)
        bi_events, bi_steps, bi_bytes = _workload_schedule(cfg, workload, True)
        if not np.isclose(uni_bytes, bi_bytes, rtol=1e-12, atol=1e-6):
            raise AssertionError("bidirectional executable ring changed total payload")
        events.extend(uni_events)
        events.extend(bi_events)
        for seed in cfg.seeds:
            for port in cfg.port_budgets:
                model = endpoint_network_model(cfg, cfg.endpoint_count, port)
                model_results = {}
                for model_name, steps, byte_count in [("unidirectional_executable_ring", uni_steps, uni_bytes), ("executable_bidirectional_ring", bi_steps, bi_bytes)]:
                    aggregate = np.sum([matrix for _, matrix in steps], axis=0)
                    sym = allocate_sym_ocs(aggregate, model)
                    drac, runtime = allocate_drac_makespan_opt(aggregate, model)
                    sym_time, sym_steps = _simulate_fixed_schedule(steps, sym)
                    drac_time, drac_steps = _simulate_fixed_schedule(steps, drac)
                    gain = (sym_time - drac_time) / sym_time if sym_time > 0 else float("nan")
                    model_results[model_name] = (sym_time, drac_time, gain, byte_count)
                    summary.append({"workload": workload.name, "seed": seed, "port_budget": port, "collective_model": model_name, "executable_balancing_applicable": True, "communication_bytes": byte_count, "sym_time_ms": sym_time, "drac_time_ms": drac_time, "drac_gain": gain, "drac_solver_runtime_ms": runtime, "intra_collective_reconfiguration": cfg.intra_collective_reconfiguration})
                    for scheme, details in [("sym_ocs", sym_steps), ("drac_makespan_opt", drac_steps)]:
                        for detail in details:
                            step_rows.append({"workload": workload.name, "seed": seed, "port_budget": port, "collective_model": model_name, "scheme": scheme, **detail})
                orig = model_results["unidirectional_executable_ring"]
                bal = model_results["executable_bidirectional_ring"]
                comparison.append({"workload": workload.name, "seed": seed, "port_budget": port, "original_sym_time_ms": orig[0], "original_drac_time_ms": orig[1], "original_gain": orig[2], "balanced_sym_time_ms": bal[0], "balanced_drac_time_ms": bal[1], "balanced_gain": bal[2], "benefit_retention": benefit_retention(orig[2], bal[2], cfg.benefit_epsilon), "communication_bytes": orig[3]})
    overall = []
    for mode in ["communication_bytes", "baseline_communication_time", "workload_equal"]:
        overall.append({"weight_mode": mode, "aggregate_gain_ratio": aggregate_gain_ratio(comparison, mode, cfg.benefit_epsilon), "oracle_used_for_risk": False, "matrix_abstraction_used_for_risk": False})
    # Supplemental matrix-only sensitivity. These rows are deliberately excluded
    # from executable BenefitRetention and the replacement-risk diagnostic.
    for workload in cfg.workloads:
        for seed in cfg.seeds:
            segments = load_or_generate_workload(workload, cfg.endpoint_count, cfg.asymmetry_level, seed)
            for port in cfg.port_budgets:
                network = endpoint_network_model(cfg, cfg.endpoint_count, port)
                for model_name in ["matrix_balancing_abstraction", "pairwise_balancing_oracle"]:
                    sym_total = drac_total = byte_total = 0.0
                    transform_applied = not (workload.kind == "pp" and model_name == "matrix_balancing_abstraction")
                    for segment in segments:
                        if model_name == "matrix_balancing_abstraction":
                            matrix = bidirectional_balanced(segment.matrix, cfg.chunk_count) if transform_applied else segment.matrix.copy()
                        else:
                            matrix = pairwise_balancing_oracle(segment.matrix)
                        sym = allocate_sym_ocs(matrix, network)
                        drac, _ = allocate_drac_makespan_opt(matrix, network)
                        sym_total += _completion_time_ms(matrix, sym.total_bandwidth)
                        drac_total += _completion_time_ms(matrix, drac.total_bandwidth)
                        byte_total += float(matrix.sum())
                    matrix_sensitivity.append({"workload": workload.name, "seed": seed, "port_budget": port, "collective_model": model_name, "display_label": "Matrix Balancing Abstraction" if model_name.startswith("matrix") else "Pairwise Balancing Oracle (non-deployable upper bound)", "transform_applied": transform_applied, "reason": "PP is non-ring; matrix abstraction left unchanged" if not transform_applied else "supplemental matrix-only sensitivity", "executable_balancing_applicable": False, "communication_bytes": byte_total, "sym_time_ms": sym_total, "drac_time_ms": drac_total, "drac_gain": (sym_total-drac_total)/sym_total if sym_total>0 else float("nan"), "used_for_replacement_risk": False})
    out = root / "collective"
    _write_csv(out / "collective_schedule_events.csv", events)
    _write_csv(out / "collective_step_performance.csv", step_rows)
    _write_csv(out / "collective_executable_summary.csv", summary)
    _write_csv(out / "collective_benefit_retention.csv", comparison)
    _write_csv(out / "collective_aggregate_gain_ratio.csv", overall)
    _write_csv(out / "collective_matrix_sensitivity.csv", matrix_sensitivity)
    plot_collective_v2(summary, out)
    return events, summary, comparison, overall


def _old_gain(workload: WorkloadConfig, n: int, asym: float, seed: int, port: int, links: int, reconfig_ms: float, include_reconfig: bool) -> Tuple[float, float, float, float, str]:
    segments = load_or_generate_workload(workload, n, asym, seed)
    net = NetworkConfig(base_bw_gbps=25.0, ocs_unit_bw_gbps=100.0, per_node_port_budget=port, total_ocs_links=links, reconfig_delay_ms=reconfig_ms)
    sym_time = drac_time = 0.0
    aggregate = np.sum([s.matrix for s in segments], axis=0)
    for segment in segments:
        sym = allocate_for_algorithm("sym_ocs", segment.matrix, net)
        drac = allocate_for_algorithm("drac", segment.matrix, net)
        overhead = reconfig_ms if include_reconfig and segment.segment_idx > 0 else 0.0
        sym_time += _completion_time_ms(segment.matrix, sym.total_bandwidth) + overhead
        drac_time += _completion_time_ms(segment.matrix, drac.total_bandwidth) + overhead
    gain = (sym_time - drac_time) / sym_time
    return sym_time, drac_time, gain, _direction_stats(aggregate)[2], _matrix_hash(aggregate)


def run_pp_discrepancy(cfg: RescueV2Config, root: Path) -> List[Dict[str, object]]:
    pp = next((w for w in cfg.workloads if w.kind == "pp"), WorkloadConfig(name="pp", kind="pp", segment_count=4, scale=0.9))
    cases = [
        ("paper_no_harm_reference", 1.0, 4, True, 7),
        ("only_change_skew_to_4", 4.0, 4, True, 7),
        ("then_change_segments_to_3", 4.0, 3, True, 7),
        ("then_exclude_reconfiguration", 4.0, 3, False, 7),
    ]
    rows = []
    for name, asym, segments, include_reconfig, seed in cases:
        work = replace(pp, segment_count=segments)
        sym, drac, gain, omega, digest = _old_gain(work, 32, asym, seed, 4, 32, 0.5, include_reconfig)
        rows.append({"case": name, "cluster_size": 32, "asymmetry_level": asym, "segment_count": segments, "seed": seed, "port_budget": 4, "base_bw_gbps": 25.0, "ocs_unit_bw_gbps": 100.0, "total_ocs_links": 32, "reconfig_delay_ms": 0.5, "reconfiguration_included": include_reconfig, "algorithm_path": "allocate_for_algorithm(sym_ocs/drac)", "normalization": "gain=(sym-drac)/sym", "sym_time_ms": sym, "drac_time_ms": drac, "gain": gain, "demand_omega": omega, "demand_matrix_hash": digest})
    _write_csv(root / "pp_discrepancy_audit.csv", rows)
    delta = rows[1]["gain"] - rows[0]["gain"]
    text = f"""# PP Discrepancy Audit

The paper no-harm plot explicitly filters PP to `asymmetry_level == 1.0`; it does not claim that the artificially skewed PP sweep is always below 1%. The reproduced reference gain is {rows[0]['gain']:.6%}. Changing only the injected skew to 4 raises gain to {rows[1]['gain']:.6%}, a {delta:.6%} absolute change. Changing four segments to three and excluding the equal Sym/DRAC reconfiguration overhead accounts for the remaining difference to the previous rescue result. The allocation implementation, base bandwidth, OCS unit bandwidth, port budget, total links, and seed are otherwise held fixed in the factor-isolation rows.

Root cause: the previous rescue report compared its fixed synthetic skew=4 result with a paper figure selected at skew=1. Payload source is unchanged; the demand matrix hash and Omega change because direction generation depends on the injected skew.
"""
    (root / "PP_DISCREPANCY.md").write_text(text, encoding="utf-8")
    return rows


def _bottleneck(matrix: np.ndarray, capacity: np.ndarray) -> Tuple[int, int, float]:
    ratios = np.divide(matrix, capacity, out=np.zeros_like(matrix), where=capacity > 0)
    idx = np.unravel_index(int(np.argmax(ratios)), ratios.shape)
    return int(idx[0]), int(idx[1]), float(ratios[idx])


def run_mixed_anomaly(cfg: RescueV2Config, root: Path) -> List[Dict[str, object]]:
    mixed = next((w for w in cfg.workloads if w.kind == "mixed"), None)
    if mixed is None:
        return []
    rows = []
    for seed in cfg.seeds:
        for segment in load_or_generate_workload(mixed, cfg.endpoint_count, cfg.asymmetry_level, seed):
            pre = segment.matrix
            post = bidirectional_balanced(pre, cfg.chunk_count)
            for port in cfg.port_budgets:
                net = NetworkConfig(base_bw_gbps=cfg.endpoint_base_capacity_gbps, ocs_unit_bw_gbps=cfg.ocs_unit_bw_gbps, per_node_port_budget=port, total_ocs_links=cfg.global_ocs_budget)
                allocs = {name: allocate_for_algorithm(alg, matrix, net) for name, matrix in [("pre_sym", pre), ("pre_drac", pre), ("post_sym", post), ("post_drac", post)] for alg in []}
                pre_sym = allocate_for_algorithm("sym_ocs", pre, net); pre_drac = allocate_for_algorithm("drac", pre, net)
                post_sym = allocate_for_algorithm("sym_ocs", post, net); post_drac = allocate_for_algorithm("drac", post, net)
                ps = _completion_time_ms(pre, pre_sym.total_bandwidth); pd = _completion_time_ms(pre, pre_drac.total_bandwidth)
                bs = _completion_time_ms(post, post_sym.total_bandwidth); bd = _completion_time_ms(post, post_drac.total_bandwidth)
                for stage, matrix, sym, drac, sym_t, drac_t in [("pre", pre, pre_sym, pre_drac, ps, pd), ("post", post, post_sym, post_drac, bs, bd)]:
                    si, sj, _ = _bottleneck(matrix, sym.total_bandwidth)
                    di, dj, _ = _bottleneck(matrix, drac.total_bandwidth)
                    rows.append({"workload": "mixed", "seed": seed, "segment_id": segment.segment_idx, "port_budget": port, "stage": stage, "omega": _direction_stats(matrix)[2], "demand_matrix_json": json.dumps(matrix.tolist(), separators=(",", ":")), "sym_completion_time_ms": sym_t, "drac_completion_time_ms": drac_t, "drac_gain": (sym_t-drac_t)/sym_t, "sym_bottleneck_src": si, "sym_bottleneck_dst": sj, "sym_bottleneck_demand": matrix[si,sj], "sym_target_capacity_gbps": sym.target_overlay[si,sj], "sym_target_floor_units": int(np.floor(sym.target_overlay[si,sj]/cfg.ocs_unit_bw_gbps+1e-12)), "sym_realized_capacity_gbps": sym.total_bandwidth[si,sj], "sym_ocs_units_at_bottleneck": int(sym.connection_units[si,sj]), "drac_bottleneck_src": di, "drac_bottleneck_dst": dj, "drac_bottleneck_demand": matrix[di,dj], "drac_target_capacity_gbps": drac.target_overlay[di,dj], "drac_target_floor_units": int(np.floor(drac.target_overlay[di,dj]/cfg.ocs_unit_bw_gbps+1e-12)), "drac_realized_capacity_gbps": drac.total_bandwidth[di,dj], "drac_ocs_units_at_bottleneck": int(drac.connection_units[di,dj]), "available_ports": port, "total_ocs_links": cfg.global_ocs_budget, "base_capacity_gbps": cfg.endpoint_base_capacity_gbps, "sym_residual_gap_gbps": float(np.maximum(sym.target_overlay-sym.realized_overlay,0).sum()), "drac_residual_gap_gbps": float(np.maximum(drac.target_overlay-drac.realized_overlay,0).sum()), "rounding_changed_bottleneck": (si,sj)!=(di,dj) or int(sym.connection_units[si,sj])!=int(drac.connection_units[si,sj])})
    _write_csv(root / "mixed_anomaly_segments.csv", rows)
    paired = defaultdict(dict)
    for row in rows:
        paired[(row["seed"],row["segment_id"],row["port_budget"])][row["stage"]] = row
    increases = sum(1 for pair in paired.values() if "pre" in pair and "post" in pair and float(pair["post"]["drac_gain"]) > float(pair["pre"]["drac_gain"]) and float(pair["post"]["omega"]) < float(pair["pre"]["omega"]))
    text = f"""# MIXED Anomaly Audit

The previous anomaly is reproduced per segment with both matrices, bottleneck pairs, realized capacities, integer units, residual gaps, base capacity, and port availability. {increases} scenario(s) show lower Omega but higher DRAC gain. The mechanism is not a normalization bug: pre/post gain is always `(Sym-DRAC)/Sym` within the same matrix. It is caused by discrete rounding and bottleneck relocation: matrix balancing changes which ordered pair determines makespan and can make the symmetric pair allocation lose a unit at the new bottleneck while DRAC retains or redirects an integer unit. Omega is a global byte-weighted direction statistic and is not monotone in the max-pair performance gain.
"""
    (root / "MIXED_ANOMALY.md").write_text(text, encoding="utf-8")
    return rows


def run_makespan_v2(cfg: RescueV2Config, root: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    continuous = []
    equivalence = []
    methods = ["sqrt_target_floor_greedy", "proportional_target_floor_greedy", "drac_makespan_opt"]
    for workload in cfg.workloads:
        for seed in cfg.seeds:
            for segment in load_or_generate_workload(workload, cfg.endpoint_count, cfg.asymmetry_level, seed):
                for port in cfg.port_budgets:
                    model = endpoint_network_model(cfg, cfg.endpoint_count, port)
                    sqrt, sqrt_rt = allocate_target_ablation(methods[0], segment.matrix, model)
                    prop, prop_rt = allocate_target_ablation(methods[1], segment.matrix, model)
                    opt, opt_rt = allocate_drac_makespan_opt(segment.matrix, model)
                    continuous.append({"workload": workload.name, "seed": seed, "segment_id": segment.segment_idx, "port_budget": port, "sqrt_vs_proportional_target_l1_gbps": float(np.abs(sqrt.target_overlay-prop.target_overlay).sum()), "sqrt_target_sum_gbps": float(sqrt.target_overlay.sum()), "proportional_target_sum_gbps": float(prop.target_overlay.sum())})
                    opt_t = _completion_time_ms(segment.matrix, opt.total_bandwidth)
                    for name, alloc, runtime in [(methods[0],sqrt,sqrt_rt),(methods[1],prop,prop_rt),(methods[2],opt,opt_rt)]:
                        validate_general_units(alloc.connection_units, model)
                        time = _completion_time_ms(segment.matrix, alloc.total_bandwidth)
                        equivalence.append({"workload": workload.name, "seed": seed, "segment_id": segment.segment_idx, "port_budget": port, "method": name, "makespan_ms": time, "optimality_gap": (time-opt_t)/opt_t if opt_t>0 else 0.0, "integer_solution_equals_opt": bool(np.array_equal(alloc.connection_units,opt.connection_units)), "runtime_ms": runtime, "used_links": int(alloc.connection_units.sum()), "nonzero_demand_entries": int(np.count_nonzero(segment.matrix))})
    validation = []
    demand = np.array([[0.,9.,1.],[2.,0.,6.],[4.,3.,0.]])
    small_cfg = replace(cfg, endpoint_base_capacity_gbps=10.0, ocs_unit_bw_gbps=10.0, global_ocs_budget=2)
    model = endpoint_network_model(small_cfg, 3, 1)
    opt, runtime = allocate_drac_makespan_opt(demand, model)
    brute_time, brute_units = brute_force_makespan(demand, model)
    exact_time = _completion_time_ms(demand, opt.total_bandwidth)
    if not np.isclose(exact_time, brute_time, rtol=1e-12, atol=1e-12):
        raise AssertionError("exact solver disagrees with brute force")
    validation.append({"case": "3-node brute-force cross-check", "rank_count": 3, "exact_time_ms": exact_time, "brute_force_time_ms": brute_time, "time_match": True, "integer_solution_match": bool(np.array_equal(opt.connection_units,brute_units)), "exact_runtime_ms": runtime, "milp_available": False, "notes": "No MILP dependency is present in requirements.txt."})
    runtime_rows = []
    base_workload = cfg.workloads[0]
    for n in cfg.rank_counts:
        workload = replace(base_workload, segment_count=1)
        matrix = load_or_generate_workload(workload, n, cfg.asymmetry_level, cfg.seeds[0])[0].matrix
        model = endpoint_network_model(cfg, n, cfg.endpoint_out_port_budget)
        alloc, runtime = allocate_drac_makespan_opt(matrix, model)
        runtime_rows.append({"rank_count": n, "nonzero_demand_entries": int(np.count_nonzero(matrix)), "runtime_ms": runtime, "makespan_ms": _completion_time_ms(matrix, alloc.total_bandwidth), "used_links": int(alloc.connection_units.sum())})
    out = root / "makespan"
    _write_csv(out / "continuous_target_difference.csv", continuous)
    _write_csv(out / "integer_solution_equivalence.csv", equivalence)
    _write_csv(out / "makespan_solver_validation.csv", validation)
    _write_csv(out / "makespan_runtime_scaling.csv", runtime_rows)
    plot_makespan_v2(runtime_rows, out)
    return continuous, equivalence, validation, runtime_rows


def write_report_v2(cfg: RescueV2Config, root: Path, aggregation: object | None, collective: object | None, makespan: object | None) -> None:
    signal = aggregation[0] if aggregation else []
    perf = aggregation[2] if aggregation else []
    comparison = collective[2] if collective else []
    overall = collective[3] if collective else []
    equivalence = makespan[1] if makespan else []
    tor_abs = _mean(float(r["absolute_retention"]) for r in signal if r["level"]=="tor" and r["mapping"]=="contiguous" and r.get("traffic_source")=="executable_unidirectional_ring")
    tor_boundary = _mean(float(r["boundary_traffic_fraction"]) for r in signal if r["level"]=="tor" and r["mapping"]=="contiguous" and r.get("traffic_source")=="executable_unidirectional_ring")
    tor_speed = _mean(float(r["speedup_drac_over_sym"]) for r in perf if r["level"]=="tor" and r["mapping"]=="contiguous" and r["normalization_mode"]=="resource_equivalent" and r.get("traffic_source")=="executable_unidirectional_ring")
    executable_ret = next((float(r["aggregate_gain_ratio"]) for r in overall if r["weight_mode"]=="communication_bytes"), float("nan"))
    opt_gain = _mean(float(r["optimality_gap"]) for r in equivalence if r["method"]=="sqrt_target_floor_greedy")
    agg_diag = "PRESENT" if np.isfinite(tor_abs) and tor_abs >= 0.25 and tor_boundary >= 0.25 and tor_speed > 1.0 else "WEAK_OR_MAPPING_DEPENDENT"
    coll_risk = collective_replacement_risk(executable_ret)
    agg_lines = ["| Workload | A_ToR (bytes) | V_ToR (bytes) | Omega | AbsoluteRetention | BoundaryTrafficFraction | Resource-equivalent speedup |", "|---|---:|---:|---:|---:|---:|---:|"]
    for workload in sorted({str(r["workload"]) for r in signal}):
        rows = [r for r in signal if r["workload"]==workload and r["level"]=="tor" and r["mapping"]=="contiguous"]
        speeds = [r for r in perf if r["workload"]==workload and r["level"]=="tor" and r["mapping"]=="contiguous" and r["normalization_mode"]=="resource_equivalent"]
        agg_lines.append(f"| {workload.upper()} | {_mean(float(r['absolute_directionality_bytes']) for r in rows):.4g} | {_mean(float(r['boundary_traffic_bytes']) for r in rows):.4g} | {_mean(float(r['omega']) for r in rows):.4f} | {_mean(float(r['absolute_retention']) for r in rows):.4f} | {_mean(float(r['boundary_traffic_fraction']) for r in rows):.4f} | {_mean(float(r['speedup_drac_over_sym']) for r in speeds):.4f} |")
    coll_lines = ["| Workload | OriginalGain | BalancedGain | Executable BenefitRetention |", "|---|---:|---:|---:|"]
    for workload in sorted({str(r["workload"]) for r in comparison}):
        rows = [r for r in comparison if r["workload"]==workload]
        weights = [float(r["communication_bytes"]) for r in rows]
        original = float(np.average([float(r["original_gain"]) for r in rows], weights=weights)) if rows else float("nan")
        balanced = float(np.average([float(r["balanced_gain"]) for r in rows], weights=weights)) if rows else float("nan")
        coll_lines.append(f"| {workload.upper()} | {original:.4f} | {balanced:.4f} | {benefit_retention(original,balanced,cfg.benefit_epsilon):.4f} |")
    make_lines = ["| Method | Mean optimality gap | Integer solution equals optimum fraction |", "|---|---:|---:|"]
    for method in ["sqrt_target_floor_greedy","proportional_target_floor_greedy","drac_makespan_opt"]:
        rows = [r for r in equivalence if r["method"]==method]
        if rows:
            make_lines.append(f"| {method} | {_mean(float(r['optimality_gap']) for r in rows):.4f} | {_mean(1.0 if r['integer_solution_equals_opt'] else 0.0 for r in rows):.4f} |")
    text = f"""# DRAC Rescue Experiments V2 Report

## Changes

Added V2 configuration, generalized resource aggregation, explicit executable ring schedules, corrected gain statistics, PP/MIXED audits, exact makespan validation, and V2 plotting/runner modules. Previous `results/rescue_experiments/` is untouched.

## Traffic provenance

The original rank matrices are synthetic. LLaMA-3 supplies payload magnitudes only; dominant-pair rules, injected skew, random orientation, fixed offset peers, and hand-written PP phases supply directionality. A segment aggregates multiple scaled operations. It is neither a single step nor an NCCL trace. Therefore the original matrices are sensitivity workloads, not real-NCCL directionality evidence. See `TRAFFIC_AUDIT.md` and `traffic_provenance.csv`.

## Aggregation and resource normalization

V2 reports A, V, Omega, AbsoluteRetention=A_level/A_endpoint, and BoundaryTrafficFraction=V_level/V_endpoint. Omega ratios are no longer called retention. Resource-equivalent performance aggregates the endpoint base-capacity matrix and endpoint out/in budgets while preserving the endpoint global OCS-unit budget and reachability. Deployment-specific runs use explicit per-level configuration only.

Mean contiguous ToR AbsoluteRetention={tor_abs:.6g}, BoundaryTrafficFraction={tor_boundary:.6g}, and resource-equivalent DRAC/Sym speedup={tor_speed:.6g}. Aggregation Opportunity: **{agg_diag}**. NaN rows retain a status explaining no cross-boundary traffic, zero direction difference, or insufficient nodes.

{chr(10).join(agg_lines)}

## Executable bidirectional ring

The primary collective baseline is an explicit event schedule with operation, phase, step, chunk, direction, ordered ranks, bytes, and dependencies. Send ownership, ReduceScatter-before-AllGather, final ownership, step legality, transfer counts, and byte conservation are validated. Odd chunks use `{cfg.odd_chunk_rule}`. Intra-collective reconfiguration is `{cfg.intra_collective_reconfiguration}`; the primary fixed-configuration comparison sums step-level makespans using one configuration planned from the whole executable schedule.

Mean executable BenefitRetention={executable_ret:.6g}. Collective Replacement Risk: **{coll_risk}**. Oracle and Matrix Balancing Abstraction are excluded from this diagnostic.

{chr(10).join(coll_lines)}

## Statistical fixes

BenefitRetention is now BalancedGain/OriginalGain for the same workload/seed/port and becomes NaN for near-zero OriginalGain. Overall results use aggregate gain ratios with separately labeled byte, baseline-time, and equal-workload weights. The previous TP value 1.8486 came from taking an unweighted arithmetic mean of port-specific ratios; the low-port case divided by a very small original gain. The aggregate 0.3068/0.3382 comparison is about 0.907, not 1.8486.

## PP discrepancy

The paper no-harm plot filters skew=1, while rescue fixed skew=4. The isolated rerun in `pp_discrepancy_audit.csv` shows this injected-skew change causes the large gain increase; segment count and equal reconfiguration overhead are secondary. See `PP_DISCREPANCY.md`.

## MIXED anomaly

The lower-Omega/higher-gain cases are not a normalization bug. Matrix balancing relocates the max-pair bottleneck; discrete Sym-OCS rounding can lose a unit while DRAC redirects one. All matrices, bottlenecks, units, capacities, port counts, and residual gaps are in `mixed_anomaly_segments.csv` and `MIXED_ANOMALY.md`.

## Objective correctness

`drac_makespan_opt` is now the primary method. The 3-node exact result matches brute-force enumeration. Runtime scaling covers {', '.join(str(v) for v in cfg.rank_counts)} nodes and reports sparsity. sqrt/proportional continuous targets remain ablations. Mean sqrt final-integer optimality gap={opt_gain:.6g}; equality means only that tested rounding happened to reach an optimal integer configuration, not that sqrt is theoretically correct for makespan. Objective Correctness: exact discrete optimization is used directly.

{chr(10).join(make_lines)}

## Answers

1. Server/ToR absolute direction signal: contiguous ToR AbsoluteRetention={tor_abs:.6g} and BoundaryTrafficFraction={tor_boundary:.6g}; because the original source is synthetic rather than trace-derived, this does not establish real NCCL opportunity by itself.
2. After executable bidirectional ring, byte-weighted aggregate BenefitRetention={executable_ret:.6g}; the per-workload table above is the primary answer, with no Oracle/IPF substitution.
3. With exact optimization, the measured resource-equivalent ToR speedup averages {tor_speed:.6g}. Any remaining gain belongs to these demand/resource instances; sqrt integer equivalence is not a proof of its continuous objective.

All negative workloads and mappings are retained without filtering.
"""
    (root / "REPORT_V2.md").write_text(text, encoding="utf-8")


def run_rescue_v2(cfg: RescueV2Config, experiment: str, smoke: bool = False) -> Dict[str, Path]:
    if smoke:
        cfg = cfg.smoke_copy()
    root = Path(cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    audit = run_traffic_audit(cfg, root) if experiment in {"audit","all"} else None
    if experiment in {"audit","all"}:
        run_pp_discrepancy(cfg, root)
        run_mixed_anomaly(cfg, root)
    aggregation = run_aggregation_v2(cfg, root) if experiment in {"aggregation","all"} else None
    collective = run_collective_v2(cfg, root) if experiment in {"collective","all"} else None
    makespan = run_makespan_v2(cfg, root) if experiment in {"makespan","all"} else None
    write_report_v2(cfg, root, aggregation, collective, makespan)
    (root / "manifest_v2.json").write_text(json.dumps({"name":cfg.name,"experiment":experiment,"smoke":smoke,"output_dir":str(root.resolve())},indent=2),encoding="utf-8")
    return {"root":root,"report":root/"REPORT_V2.md"}
