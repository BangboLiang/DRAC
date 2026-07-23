#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
from plots._common import read_rows, save_figure


STAGES=[("ordered_demand_profiling_ms","Profiling"),("target_generation_ms","Targets"),("service_matrix_ms","D matrix"),("candidate_segment_cost_ms","Segment costs"),("dynamic_programming_ms","DP"),("sparse_realization_ms","Realization"),("schedule_compaction_ms","Compaction")]


def plot(input_csv:str,output_base:str)->None:
    rows=read_rows(input_csv); endpoints=sorted({int(r["endpoint_count"]) for r in rows});fig,axes=plt.subplots(1,len(endpoints),figsize=(3.7*len(endpoints),3.6),constrained_layout=True,squeeze=False)
    for ax,n in zip(axes[0],endpoints):
        items=sorted((r for r in rows if int(r["endpoint_count"])==n),key=lambda r:int(r["node_count"])); xs=[int(r["node_count"]) for r in items]; bottom=[0.0]*len(items)
        for field,label in STAGES:
            ys=[float(r[field]) for r in items];ax.bar(xs,ys,bottom=bottom,label=label);bottom=[a+b for a,b in zip(bottom,ys)]
        ax.set_title(f"|U|={n}");ax.set_xlabel("Communication nodes K")
    axes[0][0].set_ylabel("Offline planning time (ms)");axes[0][-1].legend(frameon=False,fontsize=7,bbox_to_anchor=(1.02,1),loc="upper left");save_figure(fig,output_base)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--input",default="results/evaluation_v1/processed/planning_runtime.csv");p.add_argument("--output",default="results/evaluation_v1/figures/planning_runtime_breakdown");a=p.parse_args();plot(a.input,a.output)
