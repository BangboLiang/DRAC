"""Canonical directional-traffic data used by small paper examples.

The raw counters below are the physical per-port byte counters already used by
``figures/rx_tx_port_traffic_total_only_8layer_sqrt.py``.  Keeping them here
lets data-generation and plotting code share one source instead of embedding
another LLaMA communication model.

Functions in this module were introduced by
``generate_dp_pp_directional_traffic.py``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List


RAW_PHYSICAL_COUNTERS: Dict[str, Dict[str, int]] = {
    "TP": {
        "opposite_direction_bytes_raw": 361_013_422,
        "main_direction_bytes_raw": 50_154_823_250,
        "iterations": 200,
        "representative_layer_multiplier": 1,
    },
    "DP": {
        "opposite_direction_bytes_raw": 598_526_166,
        "main_direction_bytes_raw": 84_409_999_178,
        "iterations": 100,
        "representative_layer_multiplier": 24,
    },
}


def directional_traffic_rows() -> List[Dict[str, object]]:
    """Return representative ordered-direction demands in bytes and decimal GB."""
    rows: List[Dict[str, object]] = []
    for workload, counters in RAW_PHYSICAL_COUNTERS.items():
        iterations = counters["iterations"]
        multiplier = counters["representative_layer_multiplier"]
        main_bytes = counters["main_direction_bytes_raw"] / iterations * multiplier
        opposite_bytes = counters["opposite_direction_bytes_raw"] / iterations * multiplier
        rows.append(
            {
                "workload": workload,
                "main_direction_bytes": f"{main_bytes:.6f}",
                "opposite_direction_bytes": f"{opposite_bytes:.6f}",
                "main_direction_gb": f"{main_bytes / 1e9:.12f}",
                "opposite_direction_gb": f"{opposite_bytes / 1e9:.12f}",
                "gb_definition": "1 GB = 1e9 bytes",
                "source": "physical per-port counters from figures/rx_tx_port_traffic_total_only_8layer_sqrt.py",
                "raw_iterations": iterations,
                "representative_layer_multiplier": multiplier,
            }
        )
    return rows


def write_directional_traffic_csvs(output_dir: str | Path) -> Dict[str, Path]:
    """Write the full and compact preview directional-traffic CSV files."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = directional_traffic_rows()
    full_path = output / "directional_traffic.csv"
    preview_path = output / "directional_traffic_preview.csv"
    _write_rows(full_path, rows)
    preview_fields = [
        "workload",
        "main_direction_bytes",
        "opposite_direction_bytes",
        "main_direction_gb",
        "opposite_direction_gb",
        "gb_definition",
    ]
    _write_rows(preview_path, [{key: row[key] for key in preview_fields} for row in rows])
    return {"directional_traffic": full_path, "preview": preview_path}


def _write_rows(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError("directional traffic output must contain at least one row")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def load_dp_directional_demand(path: str | Path) -> tuple[float, float]:
    """Load DP main/opposite demands from a generated CSV, returning decimal GB."""
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row.get("workload", "").strip().upper() == "DP"]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one DP row in {csv_path}, found {len(matches)}")
    row = matches[0]
    if row.get("gb_definition") != "1 GB = 1e9 bytes":
        raise ValueError("directional traffic CSV does not declare decimal-GB units")
    main_bytes = float(row["main_direction_bytes"])
    opposite_bytes = float(row["opposite_direction_bytes"])
    if main_bytes <= 0 or opposite_bytes <= 0:
        raise ValueError("both DP ordered directions must have positive demand")
    return main_bytes / 1e9, opposite_bytes / 1e9
