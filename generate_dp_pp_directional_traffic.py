#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from drac_eval.directional_traffic import write_directional_traffic_csvs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate canonical DP/PP directional-traffic CSVs.")
    parser.add_argument("--output-dir", default="results/dp_pp_directional_traffic")
    args = parser.parse_args()
    outputs = write_directional_traffic_csvs(args.output_dir)
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
