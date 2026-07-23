# DRAC Algorithm Revision v2 Report

## A. Files and algorithms changed

- Core: added `drac_eval/segment_target.py` and `drac_eval/evaluation_pipeline_v2.py`; extended `target_segmentation.py` and `sparse_realization.py` while retaining v1 entry points.
- Evaluation: added `drac_eval/evaluation_v2.py`, five config families, six v2 runner/report scripts, and v2 plotting code.
- Validation: added `test_algorithm_revision_v2.py` and `test_evaluation_v2.py`, including plot smoke tests.
- Artifacts: added nine PDF/PNG figures, raw/processed data, archived v1 results, and this generated report/JSON summary.

The paper was not edited.

## B. Reproducing the old failures

The archived v1 eight-port baseline is DP 1.464 vs. 2.842 ms, PP 7.452 vs. 5.952 ms, and Mixed 8.585 vs. 6.632 ms for DRAC-v1 vs. Sym-OCS. V1 realization stayed near 24 stable directional channels across most epsilon values. These files remain under `results/archive/evaluation_v1/`.

The code cause was direct: candidate intervals selected only a node-target medoid, and realization used only a floor seed, positive single additions, and pure deletion pruning.

## C. Segment-level target implementation

Each interval now solves the convex epigraph problem `min sum(theta_k)` with one shared allocation, Tx/Rx budgets, and fixed-bandwidth-aware service constraints. SciPy SLSQP uses scaled constraints and deterministic feasible starts. Medoid and symmetric allocations are retained as verified upper bounds; numerical fallback is explicitly labeled rather than claimed as an optimum.

## D. Symmetric fallback

Every selected interval compares directional and symmetric integer realizations. Complete directional, symmetric, and segment-fallback schedules are then compared using realized communication plus reconfiguration cost. The final no-harm scope is only the included same-budget simulator Sym-OCS candidate.

## E. Sparse realization

V2 evaluates FloorSeed, SparseCoverageSeed, and FillResidualSeed; handles tied max-drain directions through joint group additions; performs bounded equal-count swaps; and repeats minimum-loss reverse pruning. Feasible stricter-epsilon configurations are legal candidates at wider epsilon values.

## F. Tests and oracle validation

Final status: **112 passed in 72.31s**. Segment Dynamic Programming matches complete partition enumeration on all tested small cases. Across the saved full experiment, the maximum feasible MultiSeed connection-count gap to the exhaustive integer oracle is 0 units, and the maximum segmentation relative gap to exhaustive partition enumeration is 1.6e-11.

During development, guards caught three real issues before final output: an unrestricted solve worse than a symmetric feasible point, a missing residual seed at Mixed epsilon 0.05, and a five-minute K=64 overhead timeout. The first two were fixed algorithmically; K=64 was rerun to completion with an extended offline timeout.

## G. New and old end-to-end results

The archived v1 values above remain the before-repair baseline. The v2 full configuration uses epsilon 0.5, so its numbers are not substituted for the archived epsilon 0.1 results.

| Workload | Sym-OCS | DRAC-v1 | SegmentOpt | SegmentOpt+Fallback |
|---|---:|---:|---:|---:|
| DP | 3.717 | 1.933 | 1.933 | 1.933 |
| PP | 6.577 | 8.077 | 4.404 | 4.404 |
| DP+PP Mixed | 9.936 | 9.816 | 7.846 | 7.846 |

All values are milliseconds at the maximum scanned port budget.

## H. PP and Mixed no-harm

The v2 final candidate is no slower than the included Sym-OCS candidate for every scanned workload/port pair. At eight ports, PP and Mixed improve over their DRAC-v1 schedules. This is a simulator candidate-set guarantee, not a hardware claim. Segment-level symmetric fallback usage is zero in the final full run; schedule-level comparison was still executed and logged.

## I. Epsilon-resource trade-off

The trade-off is present. Feasible MultiSeed paths are non-increasing after their first feasible epsilon. DP falls from 12 to 6 units, PP from 8 to 4, Mixed from 11 to 4, and Synthetic Hard from 8 to 3. Mixed and Hard epsilon 0 have no feasible integer oracle solution and remain explicitly resource-constrained.

## J. Stable compaction

With the unchanged planner, the eight-port directional stable pool is 48 for DP, 44 for PP, and 47 for Mixed under DRAC-Sparse, versus 48 for Sym-OCS. Thus PP and Mixed show method differences while DP does not. Fixed-budget and iso-performance conditions are stored and plotted separately.

## K. Evidence source

All v2 performance, realization, compaction, and runtime values are simulator-derived from deterministic ordered communication graphs with seed 7. They are not measurements.

## L. Missing measurements

Raw NIC directional counters remain unavailable. No profiler placeholder figure is generated in v2, and no simulated or calibrated value is labeled measured.

## M. Negative results and risks

DP compaction does not improve over Sym-OCS. Segment-level symmetric fallback was not selected in the full run, so its value is a safety guarantee rather than an observed win. Candidate target construction dominates runtime; K=64 required 302.8 seconds end to end in the recorded run. SLSQP retains an explicitly labeled feasible-upper-bound fallback for numerical failures; further solver certification and parallel candidate evaluation remain recommended.

## N. Reproduction commands

```powershell
python -m pip install -r requirements.txt
pytest -q
python -B scripts\run_all_evaluation_v2.py --profile full
python -B plots\plot_all_v2.py --root results\evaluation_v2
python -B scripts\build_algorithm_revision_v2_report.py --root results\evaluation_v2 --test-status "112 passed"
```

Each v2 experiment also has an independent runner under `scripts/run_*_v2.py` and smoke/full JSON under `configs/evaluation_v2/`.

## O. Suggested future paper changes (not applied)

The paper should replace medoid-only segment selection with the direct shared-target epigraph problem, describe the symmetric candidate at both segment and schedule level, and specify MultiSeed/group/swap/pruning realization plus feasible-history reuse. Claims should state the exact no-harm scope, disclose K=64 offline cost, and avoid hardware profiler claims until counters exist. No paper source or PDF was modified in this round.
