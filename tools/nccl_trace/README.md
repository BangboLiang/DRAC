# NCCL trace collection

This directory collects NCCL-selected topology and measured runtime metadata. NCCL INFO logs are runtime logs, not packet traces, and may omit per-message events.

1. Build/install CUDA, NCCL, nccl-tests, and MPI on every node.
2. Run `python tools/nccl_trace/collect_environment.py OUT/environment.json`.
3. Run `bash tools/nccl_trace/run_nccl_tests.sh OUT` for single-process fixtures or wrap each nccl-tests command with the cluster's `mpirun`/scheduler for multi-rank runs.
4. For PyTorch, launch `torchrun --nproc-per-node ... tools/nccl_trace/run_torch_collective.py --collective allreduce --bytes 16777216` with `NCCL_DEBUG=INFO` and capture stdout/stderr.
5. Preserve the environment JSON, raw logs, rank/host mapping, exact command, scheduler allocation, and any NCCL topology XML. Copy them into a new directory and point V3 `trace_sources` at the raw logs.

The included `fixtures/` log is labeled fixture-only and must never be reported as measured data.
