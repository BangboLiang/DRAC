#!/usr/bin/env python3
"""Memory-bounded rebuild of V4 window summaries from validated detail CSVs."""
from __future__ import annotations
import argparse, csv, math
from collections import defaultdict
from pathlib import Path
import numpy as np


def rebuild(source: Path, destination: Path, actionable: bool = False) -> None:
    groups = defaultdict(lambda: {"omega": [], "weight": [], "a": 0.0, "v": 0.0, "coverage": math.nan, "count": 0})
    with source.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key=(row["trace_name"],row["evidence_label"],row["link_rate_gbps"],row["window_ns"],row["overlap_fraction"],row.get("guard_overhead_ns","NA"),row.get("minimum_useful_payload_bytes","NA"))
            state=groups[key]; omega=float(row["Omega"]); weight=float(row["V"])
            if math.isfinite(omega): state["omega"].append(omega); state["weight"].append(weight)
            state["a"] += float(row["actionable_directional_bytes"] if actionable else row["A"]); state["v"] += weight; state["count"] += 1
            if math.isnan(state["coverage"]): state["coverage"]=float(row.get("active_window_coverage","nan"))
    fields=["trace_name","evidence_label","link_rate_gbps","window_ns","overlap_fraction","guard_overhead_ns","minimum_useful_payload_bytes","window_count","nonempty_window_coverage","traffic_weighted_omega","median_omega","p90_omega","total_directional_bytes","total_cross_node_bytes"]
    with destination.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        for key,state in groups.items():
            values=np.asarray(state["omega"]); weights=np.asarray(state["weight"])
            writer.writerow(dict(zip(fields[:7],key),window_count=state["count"],nonempty_window_coverage=state["coverage"],traffic_weighted_omega=float(np.sum(values*weights)/np.sum(weights)) if np.sum(weights)>0 else math.nan,median_omega=float(np.median(values)) if len(values) else math.nan,p90_omega=float(np.quantile(values,.9)) if len(values) else math.nan,total_directional_bytes=state["a"],total_cross_node_bytes=state["v"]))


def main():
    p=argparse.ArgumentParser(); p.add_argument("source"); p.add_argument("destination"); p.add_argument("--actionable",action="store_true"); a=p.parse_args(); rebuild(Path(a.source),Path(a.destination),a.actionable)


if __name__=="__main__": main()
