#!/usr/bin/env python3
"""Fetch official ATLAHS traces with resume and provenance metadata."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request

OFFICIAL_ROOT = "http://storage2.spcl.ethz.ch/traces/ai/llama/"
TRACE_CATALOG = [
    {"name": "Llama7B_N4_GPU16_TP1_PP1_DP16_BS32", "model": "Llama 7B", "gpus": 16, "nodes": 4},
    {"name": "Llama13B_N16_GPU64_TP4_PP2_DP8_VPP5_BS32", "model": "Llama 13B", "gpus": 64, "nodes": 16},
    {"name": "Llama7B_N32_GPU128_PP1_DP128_7B_BS128", "model": "Llama 7B", "gpus": 128, "nodes": 32},
    {"name": "Llama70B_N64_GPU256_TP1_PP8_DP32_70B_BS32", "model": "Llama 70B", "gpus": 256, "nodes": 64},
]


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def _headers(url: str) -> dict[str, str]:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        return {k.lower(): v for k, v in response.headers.items()}


def download(url: str, destination: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    remote = _headers(url)
    expected = int(remote.get("content-length", 0))
    offset = destination.stat().st_size if destination.exists() else 0
    if offset > expected > 0:
        raise RuntimeError(f"partial file is larger than remote object: {destination}")
    if offset < expected or expected == 0:
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                status = getattr(response, "status", 200)
                mode = "ab" if offset and status == 206 else "wb"
                if offset and status != 206:
                    offset = 0
                with destination.open(mode) as output:
                    while data := response.read(8 << 20):
                        output.write(data)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"official ATLAHS download failed for {url}; partial data is kept for resume: {exc}"
            ) from exc
    size = destination.stat().st_size
    if expected and size != expected:
        raise RuntimeError(f"incomplete download for {url}: {size} of {expected} bytes")
    return {
        "url": url,
        "filename": destination.name,
        "path": destination.as_posix(),
        "size_bytes": size,
        "etag": remote.get("etag"),
        "last_modified": remote.get("last-modified"),
        "downloaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sha256": sha256_file(destination),
        "source": "ATLAHS_OFFICIAL_STORAGE",
    }


def fetch(selected: list[str], data_dir: Path, manifest_path: Path, binary: bool = False) -> list[dict[str, object]]:
    by_name = {item["name"]: item for item in TRACE_CATALOG}
    records: list[dict[str, object]] = []
    if manifest_path.exists():
        records = json.loads(manifest_path.read_text(encoding="utf-8")).get("files", [])
    keyed = {record["url"]: record for record in records}
    filename = "llama.bin" if binary else "llama.goal"
    for name in selected:
        if name not in by_name:
            raise ValueError(f"unknown official trace name: {name}")
        url = f"{OFFICIAL_ROOT}{name}/{filename}"
        record = download(url, data_dir / name / filename)
        record.update(by_name[name])
        keyed[url] = record
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps({"schema_version": 1, "files": list(keyed.values())}, indent=2), encoding="utf-8"
        )
    return list(keyed.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/atlahs")
    parser.add_argument("--manifest", default="data/atlahs/manifest.json")
    parser.add_argument("--trace", action="append", dest="traces")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--binary", action="store_true", help="download official .bin instead of preferred text GOAL")
    args = parser.parse_args()
    selected = [item["name"] for item in TRACE_CATALOG] if args.all else (args.traces or [TRACE_CATALOG[0]["name"]])
    print(json.dumps(fetch(selected, Path(args.data_dir), Path(args.manifest), args.binary), indent=2))


if __name__ == "__main__":
    main()
