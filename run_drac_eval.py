#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from drac_eval import load_experiment_config, run_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DRAC evaluation simulations.")
    parser.add_argument(
        "--config",
        default="configs/drac_eval_smoke.json",
        help="Path to a JSON or YAML experiment config.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional override for the config output_dir.",
    )
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    if args.output_dir:
        cfg.output_dir = str(args.output_dir)
    outputs = run_experiments(cfg)
    print(
        json.dumps(
            {
                "config": args.config,
                "raw_csv": str(outputs["raw_csv"]),
                "summary_csv": str(outputs["summary_csv"]),
                "figure_dir": str(outputs["figure_dir"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
