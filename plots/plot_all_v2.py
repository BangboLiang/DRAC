#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plots.plot_v2 import (
    plot_compaction_v2,
    plot_end_to_end_v2,
    plot_planning_runtime_v2,
    plot_realization_v2,
    plot_segmentation_v2,
    plot_target_case_study_pp,
)

def main(root: str) -> None:
    base = Path(root)
    processed = base / "processed"
    figures = base / "figures"
    plot_end_to_end_v2(str(processed / "end_to_end_v2.csv"), str(figures / "end_to_end_v2"))
    for workload, suffix in (("DP", "dp"), ("PP", "pp"), ("DP+PP Mixed", "mixed")):
        plot_segmentation_v2(str(processed / "segmentation_v2.csv"), workload, str(figures / f"segmentation_v2_{suffix}"))
    plot_realization_v2(str(processed / "realization_tradeoff_v2.csv"), str(figures / "realization_tradeoff_v2"))
    plot_compaction_v2(str(processed / "schedule_compaction_v2.csv"), str(processed / "iso_performance_pool_v2.csv"), str(figures))
    plot_planning_runtime_v2(str(processed / "planning_runtime_v2.csv"), str(figures / "planning_runtime_v2"))
    plot_target_case_study_pp(str(base / "raw" / "segmentation_v2" / "segmentation_timelines.json"), str(figures / "target_case_study_pp"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate all DRAC v2 figures")
    parser.add_argument("--root", default="results/evaluation_v2")
    args = parser.parse_args()
    main(args.root)
