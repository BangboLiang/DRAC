from __future__ import annotations
import matplotlib
matplotlib.use("Agg",force=True)
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict,List
import numpy as np

def _finite_mean(values):
    finite=[float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(finite)) if finite else float("nan")

def _save(fig,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=160,bbox_inches="tight");plt.close(fig)

def plot_timescale(times:List[Dict[str,object]],levels:List[Dict[str,object]],persistence:List[Dict[str,object]],out:Path):
    fig,ax=plt.subplots(figsize=(6.8,4.1),constrained_layout=True)
    grouped={}
    for r in times:
        if str(r.get("window_steps",""))!="":grouped.setdefault(int(r["window_steps"]),[]).append(float(r["omega"]))
    xs=sorted(grouped);ax.plot(xs,[_finite_mean(grouped[x]) for x in xs],marker="o");ax.set_xscale("log",base=2);ax.set_xlabel("Window (steps)");ax.set_ylabel("Omega");ax.grid(True,alpha=.25);_save(fig,out/"nccl_omega_vs_window.pdf")
    for key,name,ylabel in [("absolute_directionality_bytes","nccl_absolute_directionality_vs_level.pdf","A (bytes)"),("boundary_traffic_fraction","nccl_boundary_traffic_fraction.pdf","Boundary traffic fraction")]:
        fig,ax=plt.subplots(figsize=(7,4.1),constrained_layout=True);order=["rank","server","tor","aggregation"]
        for placement in sorted({str(r["placement"]) for r in levels}):
            vals=[_finite_mean([float(r[key]) for r in levels if r["placement"]==placement and r["level"]==level]) for level in order]
            ax.plot(range(4),vals,marker="o",label=placement)
        ax.set_xticks(range(4),order);ax.set_ylabel(ylabel);ax.grid(True,alpha=.25);ax.legend(fontsize=8);_save(fig,out/name)
    fig,ax=plt.subplots(figsize=(6.8,4.1),constrained_layout=True);grouped={}
    for r in persistence:
        if str(r.get("window_steps",""))!="":grouped.setdefault(int(r["window_steps"]),[]).append(float(r["direction_persistence"]))
    xs=sorted(grouped);ax.plot(xs,[_finite_mean(grouped[x]) for x in xs],marker="o");ax.set_xscale("log",base=2);ax.set_xlabel("Window (steps)");ax.set_ylabel("Weighted direction Jaccard");ax.grid(True,alpha=.25);_save(fig,out/"nccl_direction_persistence.pdf")

def plot_sensitivity(rows:List[Dict[str,object]],out:Path):
    specs=[("per_message_startup_us","break_even_startup_latency.pdf"),("channel_count","break_even_channel_count.pdf"),("endpoint_bandwidth_gbps","break_even_endpoint_bandwidth.pdf"),("ocs_reconfiguration_delay_us","break_even_reconfiguration_delay.pdf"),("chunk_bytes","break_even_chunk_size.pdf")]
    labels={"original_drac":"Original + DRAC","fixed_half_sym":"Bidirectional + Sym-OCS","fixed_half_drac":"Bidirectional + DRAC"}
    for parameter,filename in specs:
        selected=[r for r in rows if r["parameter"]==parameter];fig,ax=plt.subplots(figsize=(7,4.1),constrained_layout=True)
        for scheme,label in labels.items():
            data=sorted([(float(r["value"]),float(r["completion_time_us"])) for r in selected if r["comparison_scheme"]==scheme])
            if data:ax.plot([x for x,_ in data],[y for _,y in data],marker="o",label=label)
        ax.set_xlabel(parameter);ax.set_ylabel("Completion time (us)");ax.grid(True,alpha=.25);ax.legend(fontsize=8);_save(fig,out/filename)

def plot_calibration_pending(out:Path):
    fig,ax=plt.subplots(figsize=(5.8,3.5),constrained_layout=True);ax.axis("off");ax.text(.5,.5,"MEASUREMENT_PENDING\nUNCALIBRATED_MODEL",ha="center",va="center",fontsize=14);_save(fig,out/"calibration_error.pdf")
