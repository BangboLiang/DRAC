#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
from plots._common import COLORS, read_rows, save_figure, sorted_unique


def plot(input_csv: str, output_base: str) -> None:
    rows=read_rows(input_csv); fig,axes=plt.subplots(1,2,figsize=(8.2,3.4),constrained_layout=True)
    for policy in sorted_unique(rows,"policy"):
        items=sorted((r for r in rows if r["policy"]==policy),key=lambda r:float(r["epsilon"])); color=COLORS.get(policy)
        axes[0].plot([float(r["epsilon"]) for r in items],[float(r["realized_slowdown"]) for r in items],marker="o",label=policy,color=color)
        axes[1].plot([float(r["epsilon"]) for r in items],[float(r["stable_reserved_channels"]) for r in items],marker="o",label=policy,color=color)
    axes[0].axhline(1.0,color="black",linestyle="--",linewidth=1); axes[0].set_ylabel("Slowdown vs. continuous target"); axes[1].set_ylabel("Stable reserved channels")
    for ax in axes: ax.set_xlabel("Tolerance ε"); ax.grid(alpha=.25)
    axes[0].legend(frameon=False,fontsize=7); save_figure(fig,output_base)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--input",default="results/evaluation_v1/processed/realization_tradeoff.csv");p.add_argument("--output",default="results/evaluation_v1/figures/realization_tradeoff");a=p.parse_args();plot(a.input,a.output)
