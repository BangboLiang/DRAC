from __future__ import annotations
from pathlib import Path
import math
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def bar(rows, key, ylabel, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.2)); labels=[r["trace_name"] for r in rows]; vals=[float(r[key]) for r in rows]
    ax.bar(range(len(vals)), vals); ax.set_xticks(range(len(vals)), [x.replace("Llama", "L") for x in labels], rotation=20, ha="right"); ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=.25); _save(fig,path)


def lines(rows, key, ylabel, path):
    fig, ax = plt.subplots(figsize=(7.2,4.2)); groups={}
    for row in rows:
        guard=row.get("guard_overhead_ns","NA")
        label=row["trace_name"] if guard in {"NA",None,""} else f"{row['trace_name']} guard={guard}ns"
        groups.setdefault(label,[]).append(row)
    for name, group in groups.items():
        group=sorted(group,key=lambda x:float(x["window_ns"])); ax.plot([float(x["window_ns"])/1000 for x in group],[float(x[key]) for x in group],marker="o",label=name)
    ax.set_xscale("log"); ax.set_xlabel("Window (us), simulated timeline"); ax.set_ylabel(ylabel); ax.grid(alpha=.25); ax.legend(fontsize=7); _save(fig,path)


def aggregation(rows, key, ylabel, path):
    fig,ax=plt.subplots(figsize=(7.2,4.2)); groups={}
    for r in rows:
        if r["mapping_strategy"]=="contiguous" and int(r["seed"])==0: groups.setdefault(r["trace_name"],[]).append(r)
    for n,g in groups.items():
        g=sorted(g,key=lambda x:int(x["servers_per_endpoint"])); ax.plot([int(x["servers_per_endpoint"]) for x in g],[float(x[key]) for x in g],marker="o",label=n)
    ax.set_xscale("log",base=2); ax.set_xlabel("Servers per hypothetical OCS endpoint"); ax.set_ylabel(ylabel); ax.grid(alpha=.25); ax.legend(fontsize=7); _save(fig,path)


def persistence_plot(rows,path):
    fig,ax=plt.subplots(figsize=(7.2,4.2)); vals=[float(r["direction_persistence"]) for r in rows if math.isfinite(float(r["direction_persistence"]))]; ax.hist(vals,bins=20); ax.set_xlabel("Traffic-weighted dominant-direction Jaccard"); ax.set_ylabel("Adjacent-window count"); _save(fig,path)


def heatmap(matrix: np.ndarray, path: Path):
    fig,ax=plt.subplots(figsize=(5.2,4.6)); im=ax.imshow(matrix,aspect="auto",cmap="magma"); ax.set_xlabel("Destination server"); ax.set_ylabel("Source server"); fig.colorbar(im,ax=ax,label="Bytes"); _save(fig,path)
