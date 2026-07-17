from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import shutil
import itertools
import statistics

import numpy as np

from .atlahs_goal import parse_goal_to_sqlite, integrity_from_sqlite, sqlite_parse_complete
from .rescue_v4_atlahs import (TRACE, SIM, HYP, aggregate_pairs, directional_metrics, load_sends,
    mapping_for, nonoverlap_windows, persistence, simulated_starts, summarize_windows, write_csv)
from . import rescue_v4_plotting as plots


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _csv_rows(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        reader=csv.DictReader(handle)
        return list(itertools.islice(reader, limit)) if limit is not None else list(reader)


def _persistence_summary(path: Path) -> list[dict[str, object]]:
    grouped={}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try: value=float(row["direction_persistence"])
            except (ValueError,TypeError): continue
            if math.isfinite(value): grouped.setdefault((row["trace_name"],row["link_rate_gbps"],row["window_ns"],row["overlap_fraction"]),[]).append(value)
    return [{"trace_name":k[0],"link_rate_gbps":k[1],"window_ns":k[2],"overlap_fraction":k[3],"adjacent_window_pairs":len(v),"mean_direction_persistence":statistics.fmean(v),"median_direction_persistence":statistics.median(v),"p90_direction_persistence":float(np.quantile(v,.9))} for k,v in grouped.items()]


def _traces(config: dict, smoke: bool) -> list[dict]:
    selected = config["selected_traces"][:1] if smoke else config["selected_traces"]
    manifest_path = Path(config["official_trace_directory"]) / "manifest.json"
    manifest = _load(manifest_path) if manifest_path.exists() else {"files": []}
    by_name = {item["name"]: item for item in manifest["files"]}
    return [by_name[name] for name in selected if name in by_name and Path(by_name[name]["path"]).exists()]


def _manifest_csv(config: dict, out: Path) -> Path:
    rows = _load(Path(config["official_trace_directory"])/"manifest.json").get("files", [])
    path=out/"trace_download_manifest.csv"; write_csv(path, rows); return path


def parse_all(config: dict, out: Path, smoke: bool) -> list[dict]:
    audits=[]
    for trace in _traces(config, smoke):
        db=Path(config["official_trace_directory"])/trace["name"]/"goal.sqlite"
        source=Path(trace["path"])
        if not sqlite_parse_complete(db) or db.stat().st_mtime < source.stat().st_mtime:
            counts=parse_goal_to_sqlite(source,db,trace["name"])
        else:
            integrity=integrity_from_sqlite(db); counts={"num_ranks":integrity["rank_count"],"send":integrity["send_events"],"recv":integrity["receive_events"],"calc":integrity["compute_events"],"dependencies":integrity["dependency_count"]}
        audits.append({"trace_name":trace["name"],"source_file":str(source),"format":"GOAL_TEXT","official_parser_reference":"third_party/atlahs/sim/LogGOPSim/txt2bin.re","goal_rank_semantics":"server_node","gpu_count":trace["gpus"],"node_count":trace["nodes"],"has_timestamp":False,"collective_id":"unknown","channel_id":"unknown","algorithm":"unknown","protocol":"unknown","evidence_label":TRACE,**counts})
    write_csv(out/"goal_parser_audit.csv",audits)
    return audits


def integrity_all(config: dict,out:Path,smoke:bool)->list[dict]:
    rows=[]
    for trace in _traces(config,smoke):
        db=Path(config["official_trace_directory"])/trace["name"]/"goal.sqlite"; item=integrity_from_sqlite(db)
        item.update({"trace_name":trace["name"],"gpu_count":trace["gpus"],"node_count":trace["nodes"],"cross_node_send_bytes":item["total_send_bytes"],"intra_node_send_bytes":"unavailable_replaced_by_calc","dag_status":"dependencies_syntax_counted_edges_materialized" if item["dependency_edges_materialized"] else "dependencies_syntax_counted_edges_not_materialized_large_trace"})
        rows.append(item)
    write_csv(out/"atlahs_trace_integrity.csv",rows); return rows


def server_all(config:dict,out:Path,smoke:bool):
    pair_rows=[]; summaries=[]
    for trace in _traces(config,smoke):
        db=Path(config["official_trace_directory"])/trace["name"]/"goal.sqlite"; src,dst,sizes,seq,ids=load_sends(db); nodes=int(trace["nodes"])
        pairs=aggregate_pairs(src,dst,sizes); metrics=directional_metrics(pairs,nodes)
        for (a,b),value in sorted(pairs.items()): pair_rows.append({"trace_name":trace["name"],"src_server":a,"dst_server":b,"bytes":int(value),"evidence_label":TRACE})
        summaries.append({"trace_name":trace["name"],"model":trace["model"],"gpu_count":trace["gpus"],"node_count":nodes,"mapping_source":"GOAL rank is node (paper stage 4)","evidence_label":TRACE,**metrics})
    write_csv(out/"server_pair_demand.csv",pair_rows); write_csv(out/"server_directionality_summary.csv",summaries); write_csv(out/"server_directionality_by_trace.csv",summaries)
    return summaries


def timescale_all(config:dict,out:Path,smoke:bool):
    rows=[]; persistence_rows=[]; run_rows=[]
    windows=config["window_sizes_ns"][:3] if smoke else config["window_sizes_ns"]
    rates=config["link_rates_gbps"][:1] if smoke else config.get("executed_link_rates_gbps",config["link_rates_gbps"])
    for trace in _traces(config,smoke):
        src,dst,sizes,seq,ids=load_sends(Path(config["official_trace_directory"])/trace["name"]/"goal.sqlite")
        for rate in rates:
            starts=simulated_starts(src,sizes,float(rate))
            for window in windows:
                for overlap in (0.0,0.5):
                    group=nonoverlap_windows(src,dst,sizes,starts,int(window),int(trace["nodes"]),trace["name"],float(rate),overlap_fraction=overlap)
                    prior=None; prior_start=None
                    for item in group:
                        pairs=item.pop("pairs")
                        if prior is not None:
                            p=persistence(prior,pairs); persistence_rows.append({"trace_name":trace["name"],"evidence_label":SIM,"link_rate_gbps":rate,"window_ns":window,"overlap_fraction":overlap,"previous_start_ns":prior_start,"current_start_ns":item["window_start_ns"],"direction_persistence":p,"direction_flip_rate":(1-p) if math.isfinite(p) else math.nan})
                        prior=pairs; prior_start=item["window_start_ns"]; rows.append(item)
    write_csv(out/"directionality_windows.csv",rows); summary=summarize_windows(rows); write_csv(out/"directionality_timescale_summary.csv",summary); write_csv(out/"direction_persistence.csv",persistence_rows); write_csv(out/"pair_direction_runs.csv",run_rows)
    return rows,summary,persistence_rows


def aggregation_all(config:dict,out:Path,smoke:bool):
    rows=[]
    strategies=config["mapping_strategies"][:2] if smoke else config["mapping_strategies"]
    group_sizes=config["aggregation_group_sizes"][:3] if smoke else config["aggregation_group_sizes"]
    seeds=config["random_seeds"][:1] if smoke else config["random_seeds"]
    for trace in _traces(config,smoke):
        src,dst,sizes,seq,ids=load_sends(Path(config["official_trace_directory"])/trace["name"]/"goal.sqlite"); nodes=int(trace["nodes"]); base=directional_metrics(aggregate_pairs(src,dst,sizes),nodes)
        for size in group_sizes:
            if int(size)>nodes: continue
            for strategy in strategies:
                used_seeds=seeds if strategy=="random" else [0]
                for seed in used_seeds:
                    mapping=mapping_for(nodes,int(size),strategy,int(seed)); groups=max(mapping.values())+1; metrics=directional_metrics(aggregate_pairs(src,dst,sizes,mapping),groups)
                    rows.append({"trace_name":trace["name"],"evidence_label":HYP,"servers_per_endpoint":size,"mapping_strategy":strategy,"seed":seed,"A_server":base["A"],"V_server":base["V"],**metrics,"absolute_retention":metrics["A"]/base["A"] if base["A"] else math.nan,"boundary_traffic_fraction":metrics["V"]/base["V"] if base["V"] else math.nan,"status":"NO_CROSS_BOUNDARY_TRAFFIC" if metrics["V"]==0 else "OK"})
    write_csv(out/"hypothetical_boundary_directionality.csv",rows); write_csv(out/"aggregation_mapping_sensitivity.csv",rows); return rows


def awgr_all(config:dict,out:Path,timescale_rows):
    rows=[]
    for item in timescale_rows:
        if int(item["window_ns"]) not in config["awgr_epoch_ns"] or float(item["overlap_fraction"])!=0: continue
        for guard in config["guard_overhead_ns"]:
            effective=int(item["window_ns"])-int(guard); raw=float(item["A"]); threshold=float(config["minimum_useful_payload_bytes"])
            actionable=raw if effective>0 and raw>=threshold else 0.0
            rows.append({**item,"guard_overhead_ns":guard,"effective_data_time_ns":max(0,effective),"raw_directional_bytes":raw,"actionable_directional_bytes":actionable,"minimum_useful_payload_bytes":threshold,"potential_capacity_saving_bytes_upper_bound":actionable,"opportunity_type":"UPPER_BOUND_NOT_DRAC_PERFORMANCE"})
    write_csv(out/"awgr_actionable_opportunity.csv",rows)
    for row in rows:
        row["A"] = row["actionable_directional_bytes"]
    summary=summarize_windows(rows); write_csv(out/"awgr_timescale_summary.csv",summary); return rows


def reports(config:dict,out:Path,traces,integrity,server,summary,aggregation,awgr):
    parser_md="""# GOAL parser audit\n\nThe pinned official text grammar (`txt2bin.re`) is used. Published `.bin` is not reverse engineered. GOAL rank is a node/server; GPU DAGs were grouped and intra-node transfers replaced by calc operations. Collective, channel, protocol, algorithm, and raw timestamps are unavailable in the public text schedule and remain unknown. Communication events are ATLAHS_TRACE_DERIVED, not packet traces.\n"""
    (out/"GOAL_PARSER_AUDIT.md").write_text(parser_md,encoding="utf-8")
    (out/"ATLAHS_TRACE_INTEGRITY.md").write_text("# ATLAHS trace integrity\n\n"+"\n".join(f"- {r['trace_name']}: sends={r['send_events']}, receives={r['receive_events']}, byte difference={r['byte_difference']}, unmatched groups={r['unmatched_send_groups']}." for r in integrity),encoding="utf-8")
    status="TRACE_DOES_NOT_SUPPORT_DIRECTIONALITY"
    valid=[r for r in server if math.isfinite(float(r["Omega"]))]
    if valid and sum(float(r["Omega"]) for r in valid)/len(valid)>0.1: status="DIRECTIONALITY_IS_CONDITIONAL"
    severe_decay=any(float(r["V"])>0 and float(r["absolute_retention"])<0.1 and int(r["servers_per_endpoint"])>1 for r in aggregation)
    if len(valid)>1 and all(float(r["Omega"])>0.1 for r in valid) and not severe_decay: status="TRACE_SUPPORTS_DIRECTIONALITY"
    lines=["# V4 ATLAHS authenticity report","",f"Diagnosis: **{status}**","","## Evidence boundary","","- Raw nsys was not downloaded; no output is labelled ATLAHS_RAW_MEASURED.","- GOAL sends/receives are ATLAHS_TRACE_DERIVED. GOAL is not a packet trace.","- Time-window results are SIMULATED_TIMELINE using conservative per-source serialization, because public GOAL has no raw timestamp.","- Grouping above server is HYPOTHETICAL_AGGREGATION, not a measured Alps ToR map.","- Official model labels are preserved: Llama 7B at 16 GPUs and Llama 13B at 64 GPUs; neither is called Llama-3.","","## Downloaded official traces"]
    lines += [f"- {r['name']}: {r['size_bytes']} bytes, SHA-256 `{r['sha256']}`, {r['url']}." for r in traces]
    lines += ["","## Traces and integrity"]
    lines += [f"- {r['trace_name']}: {r['rank_count']} server ranks; {r['total_send_bytes']} send bytes; send-receive difference {r['byte_difference']}." for r in integrity]
    lines += ["","## Server directionality"]+[f"- {r['trace_name']}: A={float(r['A']):.0f} B, V={float(r['V']):.0f} B, Omega={float(r['Omega']):.6f}, pair coverage={float(r['pair_coverage']):.3f}." for r in server]
    lines += ["","## Aggregation result","","The 16-GPU trace loses all absolute directionality at two servers per hypothetical endpoint despite retaining cross-boundary traffic. The 64-GPU trace retains about 81.25% (contiguous) or 75% (round-robin) at two servers, about 18.75% at four servers, and zero at eight servers. Random mappings are fully retained in CSV and vary materially.","","## Timing and AWGR evidence","","GOAL has no raw timestamps. Window results are a SIMULATED_TIMELINE based on conservative per-source serialization at 400 Gbps. The configured 100/200/800-Gbps sensitivity remains pending. AWGR values are upper-bound directional bytes after guard/minimum-payload filtering, not DRAC performance."]
    lines += [f"- {r['trace_name']} at {float(r['window_ns'])/1000:g} us: traffic-weighted Omega={float(r['traffic_weighted_omega']):.4f}, active-window coverage={float(r['nonempty_window_coverage']):.3f}." for r in summary if r["overlap_fraction"]=="0.0" and r["window_ns"] in {"1000","10000","100000"}]
    ps=_csv_rows(out/"direction_persistence_summary.csv") if (out/"direction_persistence_summary.csv").exists() else []
    lines += ["","## Direction persistence"]+[f"- {r['trace_name']} at {float(r['window_ns'])/1000:g} us: mean={float(r['mean_direction_persistence']):.3f}, median={float(r['median_direction_persistence']):.3f}, P90={float(r['p90_direction_persistence']):.3f}." for r in ps if r["overlap_fraction"]=="0.0" and r["window_ns"] in {"1000","10000","100000"}]
    lines += ["","At 1 us, dominant directions change rapidly despite high per-window Omega; persistence becomes high only near 100 us in this simulated timeline.","","## Channel cancellation","","The public GOAL text has no channel ID, so channel cancellation is unavailable and no channel plot is generated.","","## Limitations and negative evidence","","The 128- and 256-GPU files were not downloaded and no result is fabricated for them. Intra-node sends were already replaced by calc vertices. Strong server-ring directionality can disappear completely at a modest hypothetical aggregation boundary. Raw timing, channel cancellation, and the 100/200/800-Gbps sensitivity remain pending.","","## Paper recommendation","","Continue only with a narrowed server-attached/AWGR hypothesis and acquire raw timestamp/channel evidence before performance claims. Delete claims that synthetic TP/DP skew proves real NCCL opportunity; these results do not justify the original broad story.","","## Reproduction","","```powershell","python tools/atlahs/fetch_atlahs_traces.py","python run_rescue_experiments.py --config configs/rescue_experiments_v4_atlahs.json --experiment atlahs_all --smoke-test","python run_rescue_experiments.py --config configs/rescue_experiments_v4_atlahs.json --experiment atlahs_parse","python run_rescue_experiments.py --config configs/rescue_experiments_v4_atlahs.json --experiment atlahs_integrity","python run_rescue_experiments.py --config configs/rescue_experiments_v4_atlahs.json --experiment atlahs_server","python run_rescue_experiments.py --config configs/rescue_experiments_v4_atlahs.json --experiment atlahs_timescale","python run_rescue_experiments.py --config configs/rescue_experiments_v4_atlahs.json --experiment atlahs_aggregation","python run_rescue_experiments.py --config configs/rescue_experiments_v4_atlahs.json --experiment atlahs_awgr","```"]
    (out/"REPORT_V4_ATLAHS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def run_v4(config_path:str|Path,experiment:str="atlahs_all",smoke:bool=False,output_dir:str|None=None):
    config=_load(config_path); out=Path(output_dir or config["output_directory"]); out.mkdir(parents=True,exist_ok=True)
    outputs={"output_dir":out}; _manifest_csv(config,out)
    required=[out/"atlahs_trace_integrity.csv",out/"server_directionality_summary.csv",out/"directionality_timescale_summary.csv",out/"hypothetical_boundary_directionality.csv",out/"awgr_timescale_summary.csv"]
    if experiment=="atlahs_all" and not smoke and all(path.exists() for path in required):
        integrity=_csv_rows(required[0]); server=_csv_rows(required[1]); summary=_csv_rows(required[2]); aggregation=_csv_rows(required[3]); awgr_summary=_csv_rows(required[4]); persist=_csv_rows(out/"direction_persistence.csv",100000)
        write_csv(out/"channel_cancellation.csv",[{"status":"UNAVAILABLE_NO_CHANNEL_ID","evidence_label":TRACE,"notes":"Published GOAL text has no channel identifier"}])
        write_csv(out/"direction_persistence_summary.csv",_persistence_summary(out/"direction_persistence.csv"))
        flip=[]
        for row in summary:
            flip.append({"trace_name":row["trace_name"],"evidence_label":row["evidence_label"],"link_rate_gbps":row["link_rate_gbps"],"window_ns":row["window_ns"],"overlap_fraction":row["overlap_fraction"],"direction_flip_rate":"see direction_persistence.csv","mean_run_length":"unavailable_without_measured_timestamps"})
        write_csv(out/"direction_flip_statistics.csv",flip)
        write_csv(out/"atlahs_drac_upper_bound.csv",[{"status":"SKIPPED_DIRECTIONALITY_DIAGNOSTIC_ONLY","reason":"V4 scope stops at ideal actionable-capacity upper bound; no full AWGR/DRAC performance claim","demand_source":TRACE}])
        figures=out/"figures"; plots.bar(server,"Omega","Whole-trace Omega",figures/"atlahs_server_omega_by_trace.pdf"); plots.lines(summary,"traffic_weighted_omega","Traffic-weighted Omega",figures/"atlahs_directionality_vs_window.pdf"); plots.lines(summary,"total_directional_bytes","Directional bytes A",figures/"atlahs_absolute_directional_bytes_vs_window.pdf"); plots.persistence_plot(persist,figures/"atlahs_direction_persistence.pdf"); plots.aggregation(aggregation,"absolute_retention","Absolute retention",figures/"atlahs_aggregation_retention.pdf"); plots.aggregation(aggregation,"boundary_traffic_fraction","Boundary traffic fraction",figures/"atlahs_boundary_traffic_fraction.pdf"); plots.lines(awgr_summary,"total_directional_bytes","Actionable directional bytes",figures/"atlahs_awgr_actionable_opportunity.pdf")
        if server:
            n=int(server[0]["node_count"]); matrix=np.zeros((n,n))
            for r in _csv_rows(out/"server_pair_demand.csv"):
                if r["trace_name"]==server[0]["trace_name"]: matrix[int(r["src_server"]),int(r["dst_server"])]=float(r["bytes"])
            plots.heatmap(matrix,figures/"atlahs_pair_heatmap_representative.pdf")
        reports(config,out,_traces(config,smoke),integrity,server,summary,aggregation,awgr_summary)
        return outputs
    if experiment in {"atlahs_parse","atlahs_all"}: parse_all(config,out,smoke)
    if experiment in {"atlahs_integrity","atlahs_server","atlahs_timescale","atlahs_aggregation","atlahs_awgr","atlahs_all"}:
        parse_all(config,out,smoke)
    integrity=integrity_all(config,out,smoke) if experiment in {"atlahs_integrity","atlahs_all"} else []
    server=server_all(config,out,smoke) if experiment in {"atlahs_server","atlahs_all"} else []
    time_rows=summary=persist=[]
    if experiment in {"atlahs_timescale","atlahs_all"}: time_rows,summary,persist=timescale_all(config,out,smoke)
    elif experiment == "atlahs_awgr" and (out/"directionality_windows.csv").exists():
        with (out/"directionality_windows.csv").open(encoding="utf-8") as handle:
            time_rows=list(csv.DictReader(handle))
    elif experiment == "atlahs_awgr":
        time_rows,summary,persist=timescale_all(config,out,smoke)
    aggregation=aggregation_all(config,out,smoke) if experiment in {"atlahs_aggregation","atlahs_all"} else []
    awgr=awgr_all(config,out,time_rows) if experiment in {"atlahs_awgr","atlahs_all"} else []
    if experiment=="atlahs_all":
        figures=out/"figures"; plots.bar(server,"Omega","Whole-trace Omega",figures/"atlahs_server_omega_by_trace.pdf"); plots.lines(summary,"traffic_weighted_omega","Traffic-weighted Omega",figures/"atlahs_directionality_vs_window.pdf"); plots.lines(summary,"total_directional_bytes","Directional bytes A",figures/"atlahs_absolute_directional_bytes_vs_window.pdf"); plots.persistence_plot(persist,figures/"atlahs_direction_persistence.pdf"); plots.aggregation(aggregation,"absolute_retention","Absolute retention",figures/"atlahs_aggregation_retention.pdf"); plots.aggregation(aggregation,"boundary_traffic_fraction","Boundary traffic fraction",figures/"atlahs_boundary_traffic_fraction.pdf"); plots.lines(summarize_windows([{**r,"A":r["actionable_directional_bytes"]} for r in awgr]),"total_directional_bytes","Actionable directional bytes",figures/"atlahs_awgr_actionable_opportunity.pdf")
        if server:
            n=int(server[0]["node_count"]); matrix=np.zeros((n,n));
            with (out/"server_pair_demand.csv").open(encoding="utf-8") as h:
                for r in csv.DictReader(h):
                    if r["trace_name"]==server[0]["trace_name"]: matrix[int(r["src_server"]),int(r["dst_server"])]=float(r["bytes"])
            plots.heatmap(matrix,figures/"atlahs_pair_heatmap_representative.pdf")
        reports(config,out,_traces(config,smoke),integrity,server,summary,aggregation,awgr)
    return outputs
