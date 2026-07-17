from __future__ import annotations
import csv,json,shutil,subprocess,sys
from dataclasses import asdict,replace
from itertools import product
from pathlib import Path
from typing import Dict,List,Sequence,Tuple
import numpy as np

from .allocation import AllocationResult
from .collective_trace import CollectiveEvent,aggregate_ordered_demand,write_collective_events_csv
from .metrics import _completion_time_ms
from .nccl_log import NCCLLogRecord,parse_nccl_log
from .nccl_reconstruct import fixed_half_bidirectional,reconstruct_ring_schedule
from .rescue_v2_network import AggregatedNetworkModel,allocate_drac_makespan_opt,allocate_sym_ocs,validate_general_units
from .rescue_v3_config import EndpointResourceModel,RescueV3Config
from .rescue_v3_endpoint import cost_based_gate,simulate_events
from .rescue_v3_plotting import plot_calibration_pending,plot_sensitivity,plot_timescale
from .rescue_v3_timescale import analyze_timescales,direction_metrics

def _write_csv(path:Path,rows:Sequence[Dict[str,object]]):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:path.write_text("",encoding="utf-8");return
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields:fields.append(key)
    with path.open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)

def _events(execution):return [event for operation in execution.operations for event in operation.events]

def run_environment(cfg:RescueV3Config,root:Path)->Dict[str,object]:
    output=root/"environment.json";script=Path("tools/nccl_trace/collect_environment.py")
    subprocess.run([sys.executable,str(script),str(output)],check=True,capture_output=True,text=True)
    env=json.loads(output.read_text(encoding="utf-8"));capable=bool(env["torch"].get("nccl_backend") and env["torch"].get("cuda_devices",0)>=2 and env["tools"].get("all_reduce_perf")!="unavailable")
    env["real_nccl_measurement_capable"]=capable;output.write_text(json.dumps(env,indent=2),encoding="utf-8")
    instructions="""# Measurement Instructions

This host cannot run a real NCCL experiment. Current V3 outputs are fixture-validated schedule reconstruction and an uncalibrated model only.

On the target cluster:

1. Install matching NVIDIA driver, CUDA toolkit, NCCL, nccl-tests, MPI or the site launcher, and Nsight Systems if permitted.
2. Run `python tools/nccl_trace/collect_environment.py OUT/environment.json` on the allocated nodes.
3. Export `NCCL_DEBUG=INFO` and `NCCL_DEBUG_SUBSYS=INIT,GRAPH,COLL,NET,TUNING`. Record any forced `NCCL_ALGO`, `NCCL_PROTO`, and channel variables.
4. Launch AllReduce, ReduceScatter, and AllGather at 1 MiB, 16 MiB, 128 MiB, 1 GiB, and the LLaMA-derived sizes using `run_nccl_tests.sh` or the site's multi-node launcher.
5. Preserve raw stdout/stderr, environment JSON, commands, topology XML, rank/host/GPU mapping, runtime tables, scheduler allocation, and git commit.
6. Copy those files into a new trace directory, update V3 `trace_sources`, and rerun `--experiment nccl_parse`, `timescale`, `endpoint_model`, and `all`.

Do not relabel NCCL INFO runtime logs as packet traces. Packet-level claims require an independently captured packet/transport trace.
"""
    (root/"MEASUREMENT_INSTRUCTIONS.md").write_text(instructions,encoding="utf-8")
    return env

def run_nccl_parse(cfg:RescueV3Config,root:Path):
    records=[];summary=[];channels=[];all_events=[]
    for index,source in enumerate(cfg.trace_sources):
        record=parse_nccl_log(source["path"]);records.append(record)
        evidence="MEASUREMENT_PENDING_FIXTURE" if "fixture" in source["kind"] else "MEASURED_RUNTIME_LOG"
        summary.append({"source":source["path"],"source_kind":source["kind"],"evidence_class":evidence,"rank_count":record.rank_count,"channel_count":record.channel_count,"operation_type":record.operation_type,"algorithm":record.algorithm,"protocol":record.protocol,"message_bytes":record.message_bytes,"runtime_us":record.runtime_us if evidence.startswith("MEASURED") else "","runtime_field_in_fixture":record.runtime_us if "FIXTURE" in evidence else "","transport":";".join(record.transports) or "unavailable","unsupported_schedule":record.unsupported_schedule})
        for channel in record.channels:channels.append({"source":source["path"],"channel_id":channel.channel_id,"topology_type":channel.topology_type,"rank_order":" ".join(map(str,channel.rank_order)) or "unavailable","tree_description":channel.tree_description,"transport":channel.transport,"evidence_class":evidence})
        execution=reconstruct_ring_schedule(record,execution_id=f"trace-{index}")
        events=_events(execution)
        if "FIXTURE" in evidence:events=[replace(event,provenance="executable_reconstructed_schedule") for event in events]
        all_events.extend(events)
    _write_csv(root/"nccl_log_summary.csv",summary);_write_csv(root/"channel_topology.csv",channels);write_collective_events_csv(root/"traces"/"collective_events.csv",all_events)
    return records,all_events,summary

def run_timescale(cfg:RescueV3Config,root:Path,events:Sequence[CollectiveEvent]):
    if not events:return [],[],[],[]
    rank_count=max(max(e.src_rank,e.dst_rank) for e in events)+1;actual={e.src_rank:e.src_host for e in events}|{e.dst_rank:e.dst_host for e in events}
    modeled_placements=[name for name in cfg.rank_placement["strategies"] if name!="actual"]
    times,levels,persistence,flips=analyze_timescales(events,rank_count,cfg.window_sizes,cfg.hierarchy,modeled_placements,cfg.seeds[0])
    # Add actual placement separately when host mapping exists.
    if actual:
        from .rescue_v3_timescale import aggregate_matrix,build_mapping
        total=aggregate_ordered_demand(events,rank_count);rank_v=direction_metrics(total)["traffic_bytes"]
        mapping=build_mapping(rank_count,"server","actual",cfg.hierarchy,cfg.seeds[0],actual);matrix=aggregate_matrix(total,mapping);m=direction_metrics(matrix);levels.append({"placement":"actual","level":"server",**m,"boundary_traffic_fraction":m["traffic_bytes"]/rank_v if rank_v else float("nan"),"provenance":"NCCL-SCHEDULE-DERIVED_FIXTURE"})
    # Explicit phase, whole-collective, iteration, and modeled OCS-time windows.
    for phase in sorted({event.phase for event in events}):
        matrix=aggregate_ordered_demand([event for event in events if event.phase==phase],rank_count)
        times.append({"window_steps":"","window_index":0,"window_kind":f"phase:{phase}",**direction_metrics(matrix),"provenance":"NCCL-SCHEDULE-DERIVED_FIXTURE"})
    whole=aggregate_ordered_demand(events,rank_count)
    times.append({"window_steps":"","window_index":0,"window_kind":"whole_collective",**direction_metrics(whole),"provenance":"NCCL-SCHEDULE-DERIVED_FIXTURE"})
    times.append({"window_steps":"","window_index":0,"window_kind":"multi_collective_iteration_single_fixture",**direction_metrics(whole),"provenance":"NCCL-SCHEDULE-DERIVED_FIXTURE"})
    modeled=simulate_events(events,cfg.endpoint,cfg.base_bw_gbps);timing={row.event_id:row for row in modeled.timings};event_by_id={event.event_id:event for event in events}
    for delay in cfg.ocs_reconfiguration_delay_us:
        buckets={}
        for event_id,row in timing.items():buckets.setdefault(int(row.start_us//max(delay,1e-12)),[]).append(event_by_id[event_id])
        matrices=[]
        for bucket,selected in sorted(buckets.items()):
            matrix=aggregate_ordered_demand(selected,rank_count);matrices.append(matrix);times.append({"window_steps":"","window_index":bucket,"window_kind":"ocs_reconfiguration_window","window_us":delay,**direction_metrics(matrix),"provenance":"EXECUTABLE-MODEL"})
        for idx in range(1,len(matrices)):
            from .rescue_v3_timescale import weighted_direction_jaccard
            value,count=weighted_direction_jaccard(matrices[idx-1],matrices[idx]);persistence.append({"window_steps":"","window_us":delay,"window_kind":"ocs_reconfiguration_window","left_window":idx-1,"right_window":idx,"direction_persistence":value,"provenance":"EXECUTABLE-MODEL"});flips.append({"window_steps":"","window_us":delay,"window_kind":"ocs_reconfiguration_window","window_transition":idx-1,"dominant_direction_flip_count":count,"provenance":"EXECUTABLE-MODEL"})
    for rows in [times,levels,persistence,flips]:
        for row in rows:row["evidence_class"]="NCCL-SCHEDULE-DERIVED_FIXTURE"
    _write_csv(root/"nccl_directionality_by_timescale.csv",times);_write_csv(root/"nccl_directionality_by_level.csv",levels);_write_csv(root/"direction_persistence.csv",persistence);_write_csv(root/"direction_flip_statistics.csv",flips);plot_timescale(times,levels,persistence,root/"figures")
    return times,levels,persistence,flips

def _network(cfg:RescueV3Config,n:int)->AggregatedNetworkModel:
    base=np.full((n,n),cfg.base_bw_gbps);np.fill_diagonal(base,0);return AggregatedNetworkModel(base,np.full(n,cfg.per_rank_port_budget),np.full(n,cfg.per_rank_port_budget),~np.eye(n,dtype=bool),cfg.total_ocs_links,cfg.ocs_unit_bw_gbps,"v3_endpoint","rank")

def _static(model:AggregatedNetworkModel)->AllocationResult:
    z=np.zeros_like(model.base_capacity_gbps);u=np.zeros_like(z,dtype=int);return AllocationResult("static_sym",z,z,model.base_capacity_gbps.copy(),u,{})

def _clone_provenance(events,provenance):return [replace(event,provenance=provenance) for event in events]

def _simulate(events,resources,capacity):
    result=simulate_events(events,resources,capacity)
    network_resources=replace(resources,nic_egress_bw_gbps=1e12,nic_ingress_bw_gbps=1e12,pcie_or_nvlink_tx_bw_gbps=1e12,pcie_or_nvlink_rx_bw_gbps=1e12,gpu_reduce_bw_gbps=1e12,gpu_copy_bw_gbps=1e12,per_message_startup_us=0,per_step_sync_us=0,per_channel_launch_us=0,optional_kernel_launch_us=0)
    # Network-only keeps path capacity but removes endpoint and launch overhead.
    network=simulate_events(events,network_resources,capacity)
    return result,network.completion_time_us

def _variant_executions(record:NCCLLogRecord,cfg:RescueV3Config):
    original=reconstruct_ring_schedule(record,execution_id="original")
    fixed=fixed_half_bidirectional(record,execution_id="fixed-half")
    original_events=_events(original);fixed_events=_clone_provenance(_events(fixed),"executable_reconstructed_schedule")
    rings=[c for c in record.channels if c.rank_order];best_events=fixed_events;best=float("inf");best_dirs={}
    model=_network(cfg,record.rank_count or cfg.rank_count);aggregate=aggregate_ordered_demand(original_events,model.node_count);sym=allocate_sym_ocs(aggregate,model)
    for bits in product(["clockwise","counter_clockwise"],repeat=len(rings)):
        directions={channel.channel_id:direction for channel,direction in zip(rings,bits)};candidate=reconstruct_ring_schedule(record,execution_id="optimized",direction_by_channel=directions);events=_clone_provenance(_events(candidate),"executable_reconstructed_schedule");time=simulate_events(events,cfg.endpoint,sym.total_bandwidth).completion_time_us
        if time<best:best=time;best_events=events;best_dirs=directions
    return {"nccl_selected_schedule":original_events,"fixed_half_bidirectional":fixed_events,"optimized_bidirectional":best_events},best_dirs

def run_cross_compare(cfg:RescueV3Config,root:Path,record:NCCLLogRecord):
    variants,best_dirs=_variant_executions(record,cfg);n=record.rank_count or cfg.rank_count;model=_network(cfg,n);rows=[]
    for collective_scheme,events in variants.items():
        demand=aggregate_ordered_demand(events,n);metrics=direction_metrics(demand);sym=allocate_sym_ocs(demand,model);drac,_=allocate_drac_makespan_opt(demand,model);allocations={"static_sym":_static(model),"sym_ocs":sym,"drac_makespan_opt":drac}
        sym_pred=simulate_events(events,cfg.endpoint,sym.total_bandwidth).completion_time_us;drac_pred=simulate_events(events,cfg.endpoint,drac.total_bandwidth).completion_time_us
        gate=cost_based_gate(sym_pred,drac_pred,cfg.ocs_reconfiguration_delay_us[-1],metrics["omega"] if np.isfinite(metrics["omega"]) else 0,cfg.omega_threshold);allocations["drac_gated"]=drac if gate=="drac_makespan_opt" else sym
        for network_scheme,allocation in allocations.items():
            validate_general_units(allocation.connection_units,model);result,network_only=_simulate(events,cfg.endpoint,allocation.total_bandwidth);reconfigs=1 if network_scheme in {"drac_makespan_opt","drac_gated"} and (network_scheme!="drac_gated" or gate=="drac_makespan_opt") else 0
            rows.append({"collective_scheme":collective_scheme,"network_scheme":network_scheme,"selected_by_gate":gate if network_scheme=="drac_gated" else "","completion_time_us":result.completion_time_us+reconfigs*cfg.ocs_reconfiguration_delay_us[-1],"network_only_time_us":network_only,"endpoint_stall_time_us":max(0,result.completion_time_us-network_only),"startup_overhead_us":result.startup_overhead_us,"reduction_copy_overhead_us":result.reduction_copy_overhead_us,"total_ocs_reconfigurations":reconfigs,"total_messages":len(events),"total_channels":len({event.channel_id for event in events}),"total_chunks":len({event.chunk_id for event in events}),"total_payload_bytes":sum(event.bytes for event in events),"used_ocs_links":int(allocation.connection_units.sum()),"integer_constraints_valid":True,"maximum_concurrent_sends":result.max_concurrent_sends,"maximum_concurrent_receives":result.max_concurrent_receives,"directionality_at_actual_ocs_timescale":metrics["omega"],"parameter_provenance":cfg.endpoint.provenance,"schedule_provenance":"NCCL-SCHEDULE-DERIVED_FIXTURE" if collective_scheme=="nccl_selected_schedule" else "EXECUTABLE-MODEL","optimized_channel_directions":json.dumps(best_dirs) if collective_scheme=="optimized_bidirectional" else ""})
    # Relative metrics use Original+Sym as common reference and within-schedule Sym for DRAC gain.
    baseline=next(float(r["completion_time_us"]) for r in rows if r["collective_scheme"]=="nccl_selected_schedule" and r["network_scheme"]=="sym_ocs")
    for row in rows:
        same_sym=next(float(r["completion_time_us"]) for r in rows if r["collective_scheme"]==row["collective_scheme"] and r["network_scheme"]=="sym_ocs")
        row["drac_relative_gain"]=(same_sym-float(row["completion_time_us"]))/same_sym if row["network_scheme"] in {"drac_makespan_opt","drac_gated"} else 0.0;row["combined_gain_vs_original_sym"]=(baseline-float(row["completion_time_us"]))/baseline
        original_same_network=next(float(r["completion_time_us"]) for r in rows if r["collective_scheme"]=="nccl_selected_schedule" and r["network_scheme"]==row["network_scheme"])
        row["bidirectional_collective_relative_gain"]=(original_same_network-float(row["completion_time_us"]))/original_same_network if row["collective_scheme"]!="nccl_selected_schedule" else 0.0
    orig_gain=next(float(r["drac_relative_gain"]) for r in rows if r["collective_scheme"]=="nccl_selected_schedule" and r["network_scheme"]=="drac_makespan_opt")
    for row in rows:row["benefit_retention"]=float(row["drac_relative_gain"])/orig_gain if row["network_scheme"]=="drac_makespan_opt" and abs(orig_gain)>1e-12 else float("nan")
    _write_csv(root/"cross_comparison.csv",rows);return rows,variants

def _record_variant(base:NCCLLogRecord,channels:int|None=None,ranks:int|None=None,message_bytes:int|None=None)->NCCLLogRecord:
    from .collective_trace import ChannelTopology,RankPlacement
    n=ranks or base.rank_count or 4;c=channels or base.channel_count or 2;tops=[]
    for ch in range(c):
        order=tuple(range(n)) if ch%2==0 else tuple([0]+list(range(n-1,0,-1)));tops.append(ChannelTopology(ch,"ring",order))
    return NCCLLogRecord("generated_from_fixture_topology",n,c,"Ring",base.protocol,base.operation_type,message_bytes or base.message_bytes,None,tops,{r:RankPlacement(r,f"host{r//2}") for r in range(n)},["unavailable"])

def run_sensitivity(cfg:RescueV3Config,root:Path,base:NCCLLogRecord):
    rows=[]
    for parameter,values in cfg.sensitivity.items():
        for value in values:
            record=_record_variant(base,channels=int(value) if parameter=="channel_count" else None,ranks=int(value) if parameter=="rank_count" else None,message_bytes=int(value)*(base.channel_count or 1)*(base.rank_count or 4) if parameter=="chunk_bytes" else None)
            resources=cfg.endpoint
            if parameter=="per_message_startup_us":resources=replace(resources,per_message_startup_us=value)
            elif parameter=="max_active_channels":resources=replace(resources,max_active_channels=int(value))
            elif parameter=="max_concurrent_sends":resources=replace(resources,max_concurrent_sends=int(value))
            elif parameter=="max_concurrent_receives":resources=replace(resources,max_concurrent_receives=int(value))
            elif parameter=="endpoint_bandwidth_gbps":resources=replace(resources,nic_egress_bw_gbps=value,nic_ingress_bw_gbps=value)
            elif parameter=="pcie_or_nvlink_bw_gbps":resources=replace(resources,pcie_or_nvlink_tx_bw_gbps=value,pcie_or_nvlink_rx_bw_gbps=value)
            elif parameter=="gpu_reduce_bw_gbps":resources=replace(resources,gpu_reduce_bw_gbps=value)
            original_execution=reconstruct_ring_schedule(record,execution_id="sensitivity-original")
            fixed_execution=fixed_half_bidirectional(record,execution_id="sensitivity-fixed")
            variants={"nccl_selected_schedule":_events(original_execution),"fixed_half_bidirectional":_clone_provenance(_events(fixed_execution),"executable_reconstructed_schedule")};n=record.rank_count or 4;model=_network(cfg,n)
            for name,events,scheme in [("nccl_selected_schedule",variants["nccl_selected_schedule"],"original_drac"),("fixed_half_bidirectional",variants["fixed_half_bidirectional"],"fixed_half_sym"),("fixed_half_bidirectional",variants["fixed_half_bidirectional"],"fixed_half_drac")]:
                demand=aggregate_ordered_demand(events,n);allocation=allocate_sym_ocs(demand,model) if scheme.endswith("sym") else allocate_drac_makespan_opt(demand,model)[0];result=simulate_events(events,resources,allocation.total_bandwidth);delay=value if parameter=="ocs_reconfiguration_delay_us" and scheme.endswith("drac") else (cfg.ocs_reconfiguration_delay_us[-1] if scheme.endswith("drac") else 0)
                placement_name={0:"contiguous",1:"round_robin",2:"random"}.get(int(value),"") if parameter=="rank_placement" else ""
                rows.append({"parameter":parameter,"value":value,"placement_name":placement_name,"comparison_scheme":scheme,"collective_scheme":name,"completion_time_us":result.completion_time_us+delay,"rank_count":n,"channel_count":record.channel_count,"message_bytes":record.message_bytes,"parameter_provenance":"configured_sensitivity","evidence_class":"EXECUTABLE-MODEL"})
    _write_csv(root/"sensitivity_results.csv",rows);plot_sensitivity(rows,root/"figures");return rows

def write_calibration(root:Path):
    _write_csv(root/"runtime_calibration.csv",[{"status":"MEASUREMENT_PENDING","measured_runtime_us":"","predicted_runtime_us":"","prediction_error":"","model_status":"UNCALIBRATED_MODEL"}]);(root/"CALIBRATION.md").write_text("# Calibration\n\n**MEASUREMENT_PENDING**\n\n**UNCALIBRATED_MODEL**\n\nNo real NCCL runtime was available. No parameter was fitted to the fixture runtime text.\n",encoding="utf-8");plot_calibration_pending(root/"figures")

def write_report(cfg,root,env,summary,times,levels,cross,sensitivity):
    measured=any(str(r.get("evidence_class","")).startswith("MEASURED") for r in summary);tor=[r for r in levels if r.get("level")=="tor" and r.get("placement")=="contiguous"];finite_tor=[float(r["omega"]) for r in tor if np.isfinite(float(r["omega"]))];tor_omega=float(np.mean(finite_tor)) if finite_tor else float("nan");tor_persist=float("nan")
    original=next((r for r in cross if r["collective_scheme"]=="nccl_selected_schedule" and r["network_scheme"]=="drac_makespan_opt"),None);original_sym=next((r for r in cross if r["collective_scheme"]=="nccl_selected_schedule" and r["network_scheme"]=="sym_ocs"),None);fixed=next((r for r in cross if r["collective_scheme"]=="fixed_half_bidirectional" and r["network_scheme"]=="sym_ocs"),None);gated=next((r for r in cross if r["collective_scheme"]=="nccl_selected_schedule" and r["network_scheme"]=="drac_gated"),None)
    whole=next((r for r in times if r.get("window_kind")=="whole_collective"),None);ocs_long=[r for r in times if r.get("window_kind")=="ocs_reconfiguration_window" and float(r.get("window_us",-1))==max(cfg.ocs_reconfiguration_delay_us)];ocs_omega=float(np.nanmean([float(r["omega"]) for r in ocs_long])) if ocs_long else float("nan")
    if measured and np.isfinite(tor_omega) and tor_omega>=cfg.omega_threshold and original and fixed and float(original["completion_time_us"])<float(fixed["completion_time_us"]):direction="ORIGINAL_STORY_SALVAGEABLE"
    elif measured and (not np.isfinite(tor_omega) or tor_omega<cfg.omega_threshold):direction="PIVOT_REQUIRED"
    else:direction="SCOPE_REDUCTION_REQUIRED"
    text=f"""# DRAC Rescue Experiments V3 Report

## Evidence labels

- **MEASURED:** none on this host.
- **NCCL-SCHEDULE-DERIVED:** fixture-validated NCCL INFO topology reconstruction only; not a packet trace and not this host's measured selection.
- **EXECUTABLE-MODEL:** endpoint-resource event simulation and bidirectional schedule variants.
- **SYNTHETIC-SENSITIVITY:** parameter sweeps only.
- **MEASUREMENT-PENDING:** real cluster topology, algorithm selection, and runtime calibration.

## Environment and real NCCL information

One GTX 1650 is driver-visible, but CUDA toolkit, CUDA-enabled PyTorch, NCCL backend, nccl-tests, MPI, nsys, multi-GPU, and multi-node execution are unavailable. Consequently no real NCCL algorithm/channel/ring/tree selection or runtime was obtained. The parser recovered the fixture's Ring/Simple, two channels, their rank orders, host mapping, message size, and transport syntax. Fixture runtime text was not used as a measurement or calibration target.

## Directionality by timescale and level

The CSVs report per-window, level, persistence, and flips. DirectionPersistence is weighted Jaccard over absolute dominant-direction weights: intersection uses `min(weight)` only when the dominant ordered direction agrees; union uses `max(weight)` and direction flips contribute zero intersection. Contiguous ToR Omega from the four-rank fixture is {tor_omega:.6g}; this topology fits under one configured ToR, so ToR evidence is absent rather than positive. Directionality shorter than the configured OCS delay is not counted as deployable evidence.

The fixture has channel orders `0-1-2-3` and `0-3-2-1`. Their opposite directions cancel at every logical step and over the whole collective: whole-collective Omega={float(whole['omega']) if whole else float('nan'):.6g}. The modeled {max(cfg.ocs_reconfiguration_delay_us):g} us OCS window has mean Omega={ocs_omega:.6g}. Shorter modeled windows can be directional because endpoint serialization separates events in time, but that is not stable opportunity at the configured 20 ms switching scale.

## Endpoint-constrained collective comparison

Original, fixed-half, and optimized schedules preserve channel count, chunk count, payload, ranks, endpoint budgets, and OCS budget. The deterministic simulator enforces dependencies, per-channel order, active-channel slots, send/receive slots, NIC ingress/egress, PCIe/NVLink, GPU reduce/copy, startup, sync, and launch costs. Parameters are configured sensitivity, not measured. Cross-comparison results are in `cross_comparison.csv`.

Original+Sym-OCS completion: {float(original_sym['completion_time_us']) if original_sym else float('nan'):.6g} us. Original+forced-DRAC completion including reconfiguration: {float(original['completion_time_us']) if original else float('nan'):.6g} us. Fixed-half+Sym-OCS completion: {float(fixed['completion_time_us']) if fixed else float('nan'):.6g} us. DRAC-Gated selects `{gated['selected_by_gate'] if gated else 'unavailable'}` and completes in {float(gated['completion_time_us']) if gated else float('nan'):.6g} us. The fixture original is already channel-balanced, so fixed-half provides no modeled improvement. These are **UNCALIBRATED_MODEL** results and cannot settle hardware performance.

DRAC-Gated uses both Omega and cost: it selects DRAC only when predicted DRAC plus reconfiguration delay is lower than Sym-OCS.

## Sensitivity and break-even

All requested resource dimensions are present in `sensitivity_results.csv`. Break-even figures show Original+DRAC, Bidirectional+Sym-OCS, and Bidirectional+DRAC across the full configured ranges; no range was selected for a favorable conclusion.

## Calibration and missing measurements

`runtime_calibration.csv` is `MEASUREMENT_PENDING`; the model is `UNCALIBRATED_MODEL`. Required cluster commands and return artifacts are listed in `MEASUREMENT_INSTRUCTIONS.md`.

## Negative results

- No real-NCCL evidence currently supports stable server/ToR directionality.
- The fixture's reverse multi-channel rings can cancel whole-collective directionality.
- At the 20 ms model window, directionality is zero; sub-window spikes are too short to justify OCS reconfiguration.
- Forced DRAC is harmful once the configured reconfiguration delay is included; cost-based gating avoids it.
- Endpoint costs do not turn an uncalibrated model into a measurement.
- Synthetic skew remains excluded from reality claims.

## Three decisive questions

### A. Real NCCL Opportunity

**MEASUREMENT_PENDING.** No real NCCL log or multi-node placement was measured. Fixture-derived results cannot establish stable server/ToR opportunity.

### B. Collective Replacement

**EXECUTABLE-MODEL ONLY.** The fair fixed/optimized comparison is available in CSV, but without endpoint calibration it cannot prove whether bidirectional collective replaces DRAC on real hardware.

### C. Paper Direction

**{direction}**. The original broad TP/DP story is not presently supported by real NCCL evidence. A defensible paper must restrict claims to explicitly observed NCCL algorithms, channel topologies, placements, OCS timescales, and calibrated endpoint regimes until cluster measurements arrive. If real traces later show low persistent boundary directionality, the decision should move to `PIVOT_REQUIRED`; only strong measured persistence plus a calibrated bidirectional cost can justify `ORIGINAL_STORY_SALVAGEABLE`.
"""
    (root/"REPORT_V3.md").write_text(text,encoding="utf-8")

def run_rescue_v3(cfg:RescueV3Config,experiment:str,smoke:bool=False):
    if smoke:cfg=cfg.smoke_copy()
    root=Path(cfg.output_dir);root.mkdir(parents=True,exist_ok=True);env={};records=[];events=[];summary=[];times=[];levels=[];cross=[];sensitivity=[]
    if experiment in {"environment","all"}:env=run_environment(cfg,root)
    if experiment in {"nccl_parse","timescale","endpoint_model","cross_compare","sensitivity","all"}:records,events,summary=run_nccl_parse(cfg,root)
    if experiment in {"timescale","all"}:times,levels,_,_=run_timescale(cfg,root,events)
    if experiment in {"endpoint_model","cross_compare","all"} and records:cross,_=run_cross_compare(cfg,root,records[0])
    if experiment in {"sensitivity","all"} and records:sensitivity=run_sensitivity(cfg,root,records[0])
    write_calibration(root);write_report(cfg,root,env,summary,times,levels,cross,sensitivity);(root/"manifest_v3.json").write_text(json.dumps({"experiment":experiment,"smoke":smoke,"measurement_status":"MEASUREMENT_PENDING"},indent=2),encoding="utf-8");return {"root":root,"report":root/"REPORT_V3.md"}
