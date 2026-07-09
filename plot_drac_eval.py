#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from drac_eval.plotting import generate_all_figures


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-generate DRAC evaluation figures.")
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--raw-csv", required=True)
    parser.add_argument("--matrix-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--format", action="append", default=["png"])
    args = parser.parse_args()

    generate_all_figures(
        _read_csv(Path(args.summary_csv)),
        _read_csv(Path(args.raw_csv)),
        Path(args.matrix_dir),
        Path(args.out_dir),
        args.format,
    )


if __name__ == "__main__":
    main()
