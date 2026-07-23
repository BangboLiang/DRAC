#!/usr/bin/env python3
"""Generate provisional, auditable DP/PP directional-traffic CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

from drac_eval.directional_traffic import build_outputs, load_derivation_config, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/dp_pp_directional_traffic.json",
        help="Derivation configuration JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/dp_pp_directional_traffic",
        help="Directory for directional_traffic.csv and traffic_components.csv.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    config = load_derivation_config(config_path)
    rows, components = build_outputs(config, str(config_path))
    summary_path = output_dir / "directional_traffic.csv"
    component_path = output_dir / "traffic_components.csv"
    preview_path = output_dir / "directional_traffic_preview.csv"
    write_csv(summary_path, rows)
    write_csv(preview_path, rows)
    write_csv(component_path, components)
    print(f"status={config['status']}")
    print(summary_path)
    print(preview_path)
    print(component_path)
    for row in rows:
        print(
            f"{row['workload']}: main={float(row['main_direction_bytes']):.6f} bytes, "
            f"opposite={float(row['opposite_direction_bytes']):.6f} bytes"
        )


if __name__ == "__main__":
    main()
