#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from drac_eval.directional_traffic import write_directional_traffic_csvs
from drac_eval.symmetric_provisioning import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the symmetric-provisioning motivating micro-experiment.")
    parser.add_argument("--input-csv", default="results/dp_pp_directional_traffic/directional_traffic.csv")
    parser.add_argument("--output-dir", default="results/symmetric_provisioning_cost")
    parser.add_argument("--figure-dir", default="figures")
    parser.add_argument("--total-channels", type=int, default=8)
    parser.add_argument("--channel-bandwidth-gbps", type=float, default=50.0)
    parser.add_argument("--minimum-channels-per-active-direction", type=int, default=1)
    args = parser.parse_args()
    input_path = Path(args.input_csv)
    if not input_path.exists():
        write_directional_traffic_csvs(input_path.parent)
    outputs = run_experiment(
        input_csv=input_path,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        total_channels=args.total_channels,
        channel_bandwidth_gbps=args.channel_bandwidth_gbps,
        minimum_channels_per_active_direction=args.minimum_channels_per_active_direction,
    )
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
