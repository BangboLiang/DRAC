#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from drac_eval.rescue_config import load_rescue_config
from drac_eval.rescue_runner import run_rescue_experiments
from drac_eval.rescue_v2_config import load_rescue_v2_config
from drac_eval.rescue_v2_runner import run_rescue_v2
from drac_eval.rescue_v3_config import load_rescue_v3_config
from drac_eval.rescue_v3_runner import run_rescue_v3
from drac_eval.rescue_v4_runner import run_v4


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DRAC paper rescue experiments.")
    parser.add_argument("--config", default="configs/rescue_experiments.json")
    parser.add_argument(
        "--experiment",
        choices=["audit", "aggregation", "collective", "makespan", "environment", "nccl_parse", "timescale", "endpoint_model", "cross_compare", "sensitivity", "all",
                 "atlahs_download", "atlahs_parse", "atlahs_integrity", "atlahs_server", "atlahs_timescale", "atlahs_aggregation", "atlahs_awgr", "atlahs_all"],
        default="all",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        version = int(json.load(handle).get("version", 1))
    if version >= 4:
        if args.experiment == "atlahs_download":
            from tools.atlahs.fetch_atlahs_traces import fetch
            raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
            selected = raw["selected_traces"][:1] if args.smoke_test else raw["selected_traces"]
            data_dir = Path(raw["official_trace_directory"])
            records = fetch(selected, data_dir, data_dir / "manifest.json")
            outputs = {"manifest": data_dir / "manifest.json", "files": len(records)}
        else:
            outputs = run_v4(args.config, experiment=args.experiment, smoke=args.smoke_test, output_dir=args.output_dir)
    elif version >= 3:
        cfg = load_rescue_v3_config(args.config)
        if args.output_dir:
            cfg.output_dir = args.output_dir
        outputs = run_rescue_v3(cfg, experiment=args.experiment, smoke=args.smoke_test)
    elif version >= 2:
        cfg = load_rescue_v2_config(args.config)
        if args.output_dir:
            cfg.output_dir = args.output_dir
        outputs = run_rescue_v2(cfg, experiment=args.experiment, smoke=args.smoke_test)
    else:
        if args.experiment == "audit":
            parser.error("--experiment audit requires a version 2 config")
        cfg = load_rescue_config(args.config)
        if args.output_dir:
            cfg.output_dir = args.output_dir
        outputs = run_rescue_experiments(cfg, experiment=args.experiment, smoke=args.smoke_test)
    print(json.dumps({k: str(v) for k, v in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
