#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Tuple


DEFAULT_UNIT_BW = list(range(0, 33))
DEFAULT_RECONFIG = [10, 20, 30, 50, 100, 25, 1.5, 3, 0.01, 0.03, 0.1, 0.5]
DEFAULT_DEGREE_K = list(range(3, 9))
DEFAULT_BANDWIDTH = list(range(50, 801, 25))
DEFAULT_LATENCY = [0.25, 1, 0.1, 0.3]


def _parse_list(value: str, cast: type) -> List:
    if value is None:
        return []
    parts = [p for p in value.replace(",", " ").split() if p]
    return [cast(p) for p in parts]


def _format_tag(value: float | int) -> str:
    text = f"{value}"
    text = text.replace("-", "m").replace(".", "p")
    return text


def _ensure_nonempty(values: List, default: Iterable) -> List:
    return values if values else list(default)


def _iter_runs(
    unit_bw: List[float],
    reconfig_ms: List[float],
    degree_k: List[int],
    bandwidth: List[float],
    latency: List[float],
) -> Iterable[dict]:
    for unit, rc, k, bw, lat in itertools.product(
        unit_bw, reconfig_ms, degree_k, bandwidth, latency
    ):
        yield {
            "unit_bw_gbps": float(unit),
            "reconfig_ms": float(rc),
            "degree_k_total": int(k),
            "bandwidth_gbps": float(bw),
            "latency_us": float(lat),
        }


def _command_for_run(
    python_bin: str,
    script_path: Path,
    run: dict,
    out_dir: Path,
    prefix: str,
    extra_args: List[str],
) -> List[str]:
    return [
        python_bin,
        str(script_path),
        "--emit-comm-trace",
        "--unit-bw-gbps",
        str(run["unit_bw_gbps"]),
        "--reconfig-ms",
        str(run["reconfig_ms"]),
        "--link-batch-ms",
        str(run["reconfig_ms"]),
        "--degree-k-total",
        str(run["degree_k_total"]),
        "--bandwidth-gbps",
        str(run["bandwidth_gbps"]),
        "--latency-us",
        str(run["latency_us"]),
        "--comm-trace-out-dir",
        str(out_dir),
        "--comm-trace-prefix",
        prefix,
        *extra_args,
    ]


def _run_command(cmd: List[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    return result.returncode


def _build_run_id(run: dict) -> str:
    return (
        f"unit{_format_tag(run['unit_bw_gbps'])}"
        f"_rc{_format_tag(run['reconfig_ms'])}"
        f"_k{run['degree_k_total']}"
        f"_bw{_format_tag(run['bandwidth_gbps'])}"
        f"_lat{_format_tag(run['latency_us'])}"
    )


def _execute_run(
    index: int,
    run: dict,
    python_bin: str,
    script_path: Path,
    out_root: Path,
    extra_args: List[str],
    dry_run: bool,
) -> Tuple[dict, str]:
    run_id = _build_run_id(run)
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = run_id

    cmd = _command_for_run(
        python_bin,
        script_path,
        run,
        out_dir,
        prefix,
        extra_args,
    )
    cmd_display = " ".join(shlex.quote(c) for c in cmd)

    returncode = 0
    if not dry_run:
        log_path = out_dir / "run.log"
        returncode = _run_command(cmd, log_path)

    record = {
        "index": index,
        "run_id": run_id,
        "params": run,
        "out_dir": str(out_dir),
        "cmd": cmd,
        "cmd_display": cmd_display,
        "returncode": returncode,
    }
    return record, run_id


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep llama3_refine_comm.py over parameter combinations."
    )
    parser.add_argument(
        "--llama-script",
        type=Path,
        default=Path("llama3_refine_comm.py"),
        help="Path to llama3_refine_comm.py (default: llama3_refine_comm.py).",
    )
    parser.add_argument(
        "--python",
        dest="python_bin",
        default=sys.executable,
        help="Python executable to use (default: current interpreter).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("out/llama3_refine_grid"),
        help="Root directory for run outputs.",
    )
    parser.add_argument(
        "--unit-bw-values",
        type=str,
        default=None,
        help="Comma/space separated unit_bw values (default: 0..32).",
    )
    parser.add_argument(
        "--reconfig-values",
        type=str,
        default=None,
        help="Comma/space separated reconfig/link batch values (default list).",
    )
    parser.add_argument(
        "--degree-k-values",
        type=str,
        default=None,
        help="Comma/space separated degree-k-total values (default: 3..8).",
    )
    parser.add_argument(
        "--bandwidth-values",
        type=str,
        default=None,
        help="Comma/space separated bandwidth values (default: 50..800 step 25).",
    )
    parser.add_argument(
        "--latency-values",
        type=str,
        default=None,
        help="Comma/space separated latency values (default list).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of runs to execute (0 = no limit).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel worker count (default: half of CPU cores).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Number of runs to schedule per batch (default: 200).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args passed to llama3_refine_comm.py after '--'.",
    )
    args = parser.parse_args()

    unit_bw = _ensure_nonempty(_parse_list(args.unit_bw_values, float), DEFAULT_UNIT_BW)
    reconfig_ms = _ensure_nonempty(
        _parse_list(args.reconfig_values, float), DEFAULT_RECONFIG
    )
    degree_k = _ensure_nonempty(
        _parse_list(args.degree_k_values, int), DEFAULT_DEGREE_K
    )
    bandwidth = _ensure_nonempty(
        _parse_list(args.bandwidth_values, float), DEFAULT_BANDWIDTH
    )
    latency = _ensure_nonempty(_parse_list(args.latency_values, float), DEFAULT_LATENCY)

    script_path = args.llama_script
    if not script_path.exists():
        raise FileNotFoundError(f"llama script not found: {script_path}")

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "run_manifest.jsonl"

    runs = list(_iter_runs(unit_bw, reconfig_ms, degree_k, bandwidth, latency))
    total = len(runs)
    limit = int(args.limit)
    if limit > 0:
        runs = runs[:limit]

    cpu_count = os.cpu_count() or 1
    workers = int(args.workers) if int(args.workers) > 0 else max(1, cpu_count // 2)
    batch_size = int(args.batch_size) if int(args.batch_size) > 0 else len(runs)

    print(f"Planned runs: {len(runs)} (of {total})")
    print(f"Workers: {workers} | Batch size: {batch_size}")

    run_entries = list(enumerate(runs, start=1))
    total_runs = len(run_entries)

    if args.dry_run:
        for index, run in run_entries:
            record, run_id = _execute_run(
                index,
                run,
                args.python_bin,
                script_path,
                out_root,
                list(args.extra_args),
                dry_run=True,
            )
            print(f"[{index}/{total_runs}] {run_id}")
            print(record["cmd_display"])
            with manifest_path.open("a", encoding="utf-8") as manifest:
                manifest.write(json.dumps(record) + "\n")
        return 0

    batches = [
        run_entries[i : i + batch_size] for i in range(0, total_runs, batch_size)
    ]
    for batch_index, batch in enumerate(batches, start=1):
        start_run = batch[0][0]
        end_run = batch[-1][0]
        print(f"Batch {batch_index}/{len(batches)} (runs {start_run}-{end_run})")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _execute_run,
                    index,
                    run,
                    args.python_bin,
                    script_path,
                    out_root,
                    list(args.extra_args),
                    False,
                )
                for index, run in batch
            ]

            for future in as_completed(futures):
                record, run_id = future.result()
                index = record["index"]
                print(f"[{index}/{total_runs}] {run_id}")
                print(record["cmd_display"])
                with manifest_path.open("a", encoding="utf-8") as manifest:
                    manifest.write(json.dumps(record) + "\n")
                if record["returncode"] != 0:
                    print(f"[warn] run failed: {run_id} (code {record['returncode']})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
