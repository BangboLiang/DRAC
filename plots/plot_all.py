#!/usr/bin/env python3
import argparse
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plots.plot_compaction import plot as plot_compaction
from plots.plot_end_to_end import plot as plot_end_to_end
from plots.plot_planning_overhead import plot as plot_overhead
from plots.plot_profiler_accuracy import plot as plot_profiler
from plots.plot_realization import plot as plot_realization
from plots.plot_segmentation import plot as plot_segmentation


def main(root: str = "results/evaluation_v1") -> None:
    plot_profiler(f"{root}/processed/profiler_accuracy.csv",f"{root}/figures/profiler_accuracy")
    plot_end_to_end(f"{root}/processed/end_to_end_performance.csv",f"{root}/figures/end_to_end_performance")
    plot_segmentation(f"{root}/processed/segmentation.csv",f"{root}/figures/segmentation")
    plot_realization(f"{root}/processed/realization_tradeoff.csv",f"{root}/figures/realization_tradeoff")
    plot_compaction(f"{root}/processed/schedule_compaction.csv",f"{root}/processed/iso_performance_pool.csv",f"{root}/figures")
    plot_overhead(f"{root}/processed/planning_runtime.csv",f"{root}/figures/planning_runtime_breakdown")
    aliases = {
        "profiler_accuracy": "figure_6_profiler_accuracy",
        "end_to_end_performance": "figure_7_end_to_end_performance",
        "segmentation": "figure_8_segmentation",
        "realization_tradeoff": "figure_9_realization_tradeoff",
        "schedule_compaction": "figure_10_schedule_compaction",
        "iso_performance_pool": "figure_11_iso_performance_pool",
    }
    for source, destination in aliases.items():
        for suffix in (".pdf", ".png"):
            shutil.copyfile(
                Path(root) / "figures" / f"{source}{suffix}",
                Path(root) / "figures" / f"{destination}{suffix}",
            )


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Generate every revised DRAC Evaluation figure")
    parser.add_argument("--root", default="results/evaluation_v1")
    args = parser.parse_args()
    main(args.root)
