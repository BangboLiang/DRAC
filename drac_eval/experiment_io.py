"""Reproducible result I/O shared by the new Evaluation experiments."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], fields: list[str] | None = None) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    materialized = [dict(row) for row in rows]
    fieldnames = fields or (list(materialized[0]) if materialized else [])
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(materialized)
    return output


def write_json(path: str | Path, value: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    return output


def git_state() -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", True


def write_manifest(
    experiment_dir: str | Path,
    *,
    experiment: str,
    config: Mapping[str, Any],
    seed: int,
    status: str,
    outputs: Mapping[str, str],
    evidence: str,
) -> Path:
    commit, dirty = git_state()
    manifest = {
        "schema_version": "drac_evaluation/v1",
        "experiment": experiment,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "status": status,
        "evidence": evidence,
        "git_commit": commit,
        "git_dirty": dirty,
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "config": dict(config),
        "outputs": dict(outputs),
    }
    return write_json(Path(experiment_dir) / "manifest.json", manifest)


def result_paths(output_root: str | Path, experiment: str) -> dict[str, Path]:
    root = Path(output_root)
    return {
        "root": root,
        "raw": root / "raw" / experiment,
        "processed": root / "processed",
        "figures": root / "figures",
        "tables": root / "tables",
    }
