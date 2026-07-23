#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drac_eval.evaluation_experiments import (
    run_compaction,
    run_end_to_end,
    run_planning_overhead,
    run_profiler_accuracy,
    run_realization,
    run_segmentation,
)
from drac_eval.experiment_io import load_json


RUNNERS = {
    "profiler": run_profiler_accuracy,
    "end_to_end": run_end_to_end,
    "segmentation": run_segmentation,
    "realization": run_realization,
    "compaction": run_compaction,
    "overhead": run_planning_overhead,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run every revised DRAC Evaluation experiment")
    parser.add_argument("--profile", choices=["smoke", "full"], default="full")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    outputs = {}
    for name, runner in RUNNERS.items():
        config = load_json(Path("configs/evaluation") / name / f"{args.profile}.json")
        outputs[name] = {key: str(value) for key, value in runner(config, args.output_dir).items()}
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
