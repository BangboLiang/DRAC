#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="${1:-nccl_trace_output}"
mkdir -p "$OUT_DIR"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,GRAPH,COLL,NET,TUNING}"
export NCCL_ALGO="${NCCL_ALGO:-Ring,Tree}"
export NCCL_PROTO="${NCCL_PROTO:-Simple,LL,LL128}"
SIZES=(1048576 16777216 134217728 1073741824)
for tool in all_reduce_perf all_gather_perf reduce_scatter_perf; do
  command -v "$tool" >/dev/null || { echo "missing $tool" >&2; exit 2; }
  for size in "${SIZES[@]}"; do
    log="$OUT_DIR/${tool}_${size}.log"
    echo "command=$tool -b $size -e $size -f 2 -g 1" > "$log"
    echo "timestamp=$(date --iso-8601=seconds)" >> "$log"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)" >> "$log"
    "$tool" -b "$size" -e "$size" -f 2 -g 1 >> "$log" 2>&1
  done
done
