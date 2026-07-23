#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drac_eval.evaluation_v2 import (
    run_compaction_v2,
    run_end_to_end_v2,
    run_planning_overhead_v2,
    run_realization_v2,
    run_segmentation_v2,
)
from drac_eval.experiment_io import load_json

RUNNERS = {
    "end_to_end": run_end_to_end_v2,
    "segmentation": run_segmentation_v2,
    "realization": run_realization_v2,
    "compaction": run_compaction_v2,
    "overhead": run_planning_overhead_v2,
}

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete DRAC v2 evaluation")
    parser.add_argument("--profile", choices=["smoke", "full"], default="full")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    outputs = {}
    for name, runner in RUNNERS.items():
        config = load_json(Path("configs/evaluation_v2") / name / f"{args.profile}.json")
        outputs[name] = {key: str(value) for key, value in runner(config, args.output_dir).items()}
    print(json.dumps(outputs, indent=2))

if __name__ == "__main__":
    main()
