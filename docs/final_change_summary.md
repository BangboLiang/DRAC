# Final Change Summary

## Algorithms changed

- Added an ordered-demand profiler that consumes communication nodes, maps ranks
  to GPU--NIC endpoints, expands ordered transfers, preserves opposite
  directions, excludes intra-server traffic, and applies explicit calibration.
- Replaced aggregate/square-root target allocation in the revised path with the
  per-node minimum-resource continuous target.  The fixed-bandwidth case uses a
  monotone feasibility search.
- Added the paper's service-cost matrix, medoid representative target for every
  candidate interval, and dynamic-programming segmentation with backtracking.
- Added sparse integer realization: floor initialization, positive-demand
  coverage, marginal-cost additions, tolerance stopping, reverse pruning, and
  resource-constrained direction reporting.
- Added schedule-wide Tx/Rx peak compaction, bidirectional-bundle reservation,
  stable exposed-channel accounting, and per-segment physical bindings.

Legacy `llama3_*` behavior remains available.  The revised Evaluation calls the
new modules directly, so old entry points are not silently reinterpreted.

## Evaluation migration

Removed from the main Evaluation storyline:

- weighted directional share-error CDF;
- residual target-gap main figure;
- capacity-waste ratio as a primary result;
- algorithmic use of skew threshold $\tau/\eta$;
- TP as a default cross-server main workload.

Directional-skew characterization, injected-skew sensitivity, and explicit
cross-server TP scenarios are Appendix-only candidates and require recomputation
with the revised implementation.  No legacy result is accepted as revised DRAC
evidence.

Added or rebuilt:

- DP, PP, and DP+PP ordered workloads;
- Static-Sym, Sym-OCS, and DRAC end-to-end comparison;
- target solver numerical/exhaustive validation;
- segmentation delay scan and SegmentOracle comparison;
- epsilon-driven sparse-realization trade-off with exhaustive oracle;
- schedule-wide compaction and iso-performance stable pool;
- stage-by-stage offline planning runtime;
- measurement-aware profiler accuracy framework.

## Evidence status

- **Hardware-measured:** none.  The repository has no raw NIC directional
  counters.  Figure 6 is explicitly marked `MEASUREMENT_PENDING`.
- **Simulator-derived:** Figures 7--11 and Table II, from checked-in deterministic
  graph generators/configurations and the revised DRAC implementation.
- **Not completed:** real profiler accuracy, cluster end-to-end validation, and a
  successful PP no-harm result.

The full simulator reveals a meaningful negative result.  At eight endpoint
channels, DRAC is faster for DP (1.464 ms versus 2.842 ms for Sym-OCS), but slower
for PP (7.452 versus 5.952 ms) and DP+PP Mixed (8.585 versus 6.632 ms), primarily
because of additional reconfiguration overhead.  The CSVs and plot retain this
result unchanged.

## Reproduction

```powershell
pytest -q
python -B scripts\run_all_evaluation.py --profile full
python -B plots\plot_all.py --root results\evaluation_v1
```

See `evaluation_reproduction.md` for independent experiment commands, evidence
labels, output layout, and profiler measurement input requirements.

## Paper status

The paper's LaTeX project is not present.  The existing `paper.pdf` was not
overwritten.  A source-ready replacement Evaluation is provided in
`paper_evaluation_revision.tex`; compilation, label checking, and final
Abstract/Introduction/Conclusion editing remain blocked until the actual source
project is supplied.  See `paper_update_blocker.md`.

## Remaining risks and TODOs

1. Collect immutable directional NIC counters for DP Ring
   AllReduce/ReduceScatter/AllGather and PP forward/backward message sizes.
2. Diagnose and improve PP/mixed segmentation or realization so that the desired
   no-harm property is demonstrated rather than assumed.
3. Validate the deterministic communication model against a real cluster.
4. Run larger integer-oracle instances with a production ILP solver; the current
   oracle deliberately uses exhaustive small cases.
5. Restore the latest paper LaTeX project, integrate the generated assets, compile,
   and visually verify every page and cross-reference.

