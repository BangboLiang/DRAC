#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import numpy as np
from plots._common import read_rows, save_figure


def plot(compaction_csv: str, iso_csv: str, output_dir: str) -> None:
    rows=read_rows(compaction_csv); selected=[r for r in rows if r["scheme"] in {"FullReservation","Sym-OCS schedule-wide peak","DRAC schedule-wide peak"}]
    labels=[f"{r['workload']}\n{r['scheme'].replace(' schedule-wide peak','')}" for r in selected]; x=np.arange(len(selected)); reserved=[float(r["reserved_tx"])+float(r["reserved_rx"]) for r in selected]; exposed=[float(r["exposed_tx"])+float(r["exposed_rx"]) for r in selected]
    fig,ax=plt.subplots(figsize=(9.0,3.8),constrained_layout=True);ax.bar(x,reserved,label="Stable reserved",color="#4c78a8");ax.bar(x,exposed,bottom=reserved,label="Stable exposed",color="#b8d6ea");ax.set_xticks(x,labels,rotation=25,ha="right");ax.set_ylabel("Directional channels");ax.set_ylim(0, max((a+b for a,b in zip(reserved,exposed)), default=1)*1.12);ax.legend(frameon=False);save_figure(fig,f"{output_dir}/schedule_compaction")
    iso=read_rows(iso_csv); fig,ax=plt.subplots(figsize=(5.8,3.6),constrained_layout=True)
    reached=[r for r in iso if r.get("status")=="reached"]
    labels=[r["workload"] for r in reached]; x=np.arange(len(reached))
    if reached:
        ax.bar(x-.18,[float(r["minimum_stable_directional_pool"]) for r in reached],.36,label="Independent Tx/Rx")
        ax.bar(x+.18,[2*float(r["minimum_stable_bundle_pool"]) for r in reached],.36,label="Bidirectional bundles")
        ax.set_xticks(x,labels);ax.legend(frameon=False)
    missing=[r["workload"] for r in iso if r.get("status")!="reached"]
    if missing:
        ax.text(.98,.97,"Not reached: "+", ".join(missing),transform=ax.transAxes,ha="right",va="top",fontsize=8)
    ax.set_ylabel("Minimum stable directional channels");save_figure(fig,f"{output_dir}/iso_performance_pool")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--input",default="results/evaluation_v1/processed/schedule_compaction.csv");p.add_argument("--iso-input",default="results/evaluation_v1/processed/iso_performance_pool.csv");p.add_argument("--output-dir",default="results/evaluation_v1/figures");a=p.parse_args();plot(a.input,a.iso_input,a.output_dir)
