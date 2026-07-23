# Evaluation Migration Plan

## 1. Migration principles

This plan maps the current paper and code-generated figures to the Evaluation required by the latest DRAC Design.

Rules applied throughout:

- New main workloads are DP-only, PP-only, and DP+PP Mixed.
- TP is excluded from the default main-paper system boundary because it is intra-server/NVSwitch traffic. Cross-server TP may appear only as an explicitly labeled Appendix extension.
- Every system comparison consumes the same ordered communication-node sequence, placement, resource budget, baseline capacity, and reconfiguration-delay model.
- Figures are generated only from saved result files; no manual number transfer is allowed.
- Simulator results and trace-derived results are labeled as such. Missing hardware data stays `MEASUREMENT_PENDING`.
- Old outputs are archived; new runs use a separate versioned result root.

## 2. Current figure migration table

| Current paper figure | Current implementation/data | Current metric/story | Action | Recompute? | Replacement | Required new data |
|---|---|---|---|---|---|---|
| Fig. 2 | `generate_dp_pp_directional_traffic.py`, `directional_traffic.py`, legacy embedded counters | representative DP/PP main vs opposite bytes | Keep as Background only after conflict resolution and provenance correction; do not call provisional derived values measured | Yes if data derivation changes | Background characterization; optional expanded Appendix provenance table | raw counter artifact or explicit legacy/provisional status |
| Fig. 3 | legacy embedded component counters | DP traffic decomposition | Keep only if raw measurement provenance can be restored; otherwise mark provisional/TODO | Yes for publication-quality claim | profiler calibration motivation | raw directional counter components and calibration metadata |
| Fig. 5 | `symmetric_provisioning.py` | motivating channel-time example and skew sensitivity | Keep as analytic Background example; separate from Evaluation | No for formula, yes if source demand changes | unchanged motivating example; sensitivity to Appendix | exact configured demands and enumerated allocation CSV |
| Fig. 6 | `plot_skew_distribution` | directional skew CDF; `tau=1.5`; TP/DP/MIXED/PP | Remove from main; delete algorithm-selection claim; TP only in Appendix extension | Yes if retained | New Fig. 6: Profiler predicted vs measured | profiler prediction rows and genuine measured directional counters |
| Fig. 7 | `plot_comm_time_vs_asymmetry` | normalized time under injected skew; TP/DP/MIXED | Move to Sensitivity/Appendix and rerun new DRAC; replace TP with DP/PP/DP+PP | Yes | New Fig. 7: end-to-end DP/PP/DP+PP communication performance | common node sequences, baseline results, delta, epsilon, budgets |
| Fig. 8 | `plot_matching_cdf` | weighted directional share error CDF on threshold-selected entries | Delete | No | New Fig. 8: segmentation cost and segment count vs delta | service-cost matrix, boundaries, representative indices, oracle rows |
| Fig. 9 | `plot_representative_heatmaps` | demand/capacity shares and share-error reduction | Rebuild as End-to-End DRAC Case Study; remove share error | Yes | case-study panels in Fig. 8 or Appendix | node demands, per-node targets, segments, representatives, integer matrices, stable pools |
| Fig. 10 | `_plot_ratio_vs_port_budget` | capacity waste ratio | Remove from main; optional Appendix characterization | Yes if retained | New Fig. 10: schedule-wide reserved/exposed channels | independent/bundled stable pool and exposed counts |
| Fig. 11 | `plot_high_demand_residual_gap` | residual target gap over selected directions | Delete | No | New Fig. 9: slowdown–resource trade-off vs epsilon | realization ablations, slowdown, used units, stable channels, tolerance status |
| Fig. 12 | `plot_iso_performance_port_saving` | ports to match Sym-OCS six-port time | Keep idea, redefine using minimum stable schedule-wide pool | Yes | New Fig. 11: iso-performance minimum stable channel pool | schedule-wide peak pools for DP/PP/DP+PP and all baselines |
| Fig. 13 | `plot_pp_no_harm_bar` | PP no-harm at skew=1 on fixed synthetic segments | Keep, rerun full PP graph with target segmentation and sparse realization | Yes | subpanel/sensitivity or Appendix; may support Fig. 7 | complete PP node sequence, common stable budget, delta and epsilon |
| Fig. 14 | `symmetric_provisioning.py` dense enumeration | injected two-direction skew benefit | Keep in Appendix as analytic sensitivity | Only if inputs change | Appendix skew sensitivity | deterministic enumeration CSV |

## 3. Current metric disposition

| Metric | Disposition | Reason/replacement |
|---|---|---|
| directional skew ratio/CDF | Appendix characterization only | `tau` is descriptive, never an optimization gate |
| weighted directional share error | Delete | new realization intentionally permits residual target mismatch |
| residual target gap | Delete as quality metric | replace with achieved slowdown, tolerance satisfaction, used/stable units |
| capacity waste ratio | Background or Appendix only | replace main resource claim with stable reserved/exposed channels and compaction ratio |
| instantaneous active/releasable ports | Delete/rename | unsafe across schedule; replace with per-endpoint schedule-wide peaks |
| total communication time | Keep and recompute | main performance metric |
| communication-only time | Add | separates service from switching overhead |
| reconfiguration overhead | Keep with exact boundary count | identical delay model for fair comparisons |
| normalized speedup | Keep and recompute | normalize to an explicitly named baseline/scenario |
| segment count | Add | validates delta response |
| resource-constrained segment ratio | Add | exposes discrete feasibility limits |
| stable reserved Tx/Rx/bundles | Add | direct output of physical compaction |
| stable exposed channels | Add | safe resource exposure claim |
| compaction ratio | Add | schedule-wide resource efficiency |
| profiler weighted absolute/relative error | Add, measurement-gated | requires real directional counters |
| target objective gap/feasibility/runtime | Add | validates the continuous solver |
| realization oracle gap | Add for small cases | validates heuristic quality |

## 4. New main-paper experiment map

### Figure 6 — Ordered Demand Profiler accuracy

Comparisons:

- PayloadOnly
- Payload+Calibration
- Measured Directional Bytes

Operations:

- Ring AllReduce
- Ring ReduceScatter
- Ring AllGather
- PP forward activation
- PP backward gradient
- multiple message sizes

Output files:

- `results/<version>/raw/profiler/*.json`
- `results/<version>/processed/profiler_accuracy.csv`
- `results/<version>/figures/profiler_accuracy.{pdf,png}`

Current blocker: no raw measured NIC directional-counter dataset exists in the repository. The runner and loader can be completed, but the main figure must remain TODO/measurement-pending unless such data is supplied.

### Figure 7 — End-to-end communication performance

Main schemes:

- Static-Sym
- Sym-OCS
- DRAC

Workloads:

- DP-only
- PP-only
- DP+PP Mixed

Metrics:

- total communication time
- communication-only time
- reconfiguration overhead
- normalized speedup
- segment count
- resource-constrained segment ratio

All schemes use the same node sequence, stable physical budget, and delay model. Sym-OCS may segment but must impose `X_uv=X_vu`.

### Figure 8 — Target-sequence segmentation

Schemes:

- OneConfig
- PerNode-Reconfig
- DRAC-DP
- SegmentOracle for small sequences

Sweep: `delta`.

Metrics:

- total, communication, and reconfiguration cost
- selected segment count
- oracle gap
- representative indices

Required behavioral checks:

- near-zero `delta`: approach PerNode-Reconfig;
- increasing `delta`: monotone tendency toward fewer segments;
- sufficiently high `delta`: approach OneConfig.

### Figure 9 — Sparse realization trade-off

Schemes:

- FloorOnly
- NearestRounding
- FillAllResidual
- DRACSparse
- small ILP/enumeration oracle

Sweep: `epsilon`.

Metrics:

- slowdown relative to selected continuous target
- used connection units
- stable reserved channels
- tolerance satisfaction rate
- resource-constrained segment ratio
- oracle gap

Use two readable panels rather than a crowded dual y-axis.

### Figure 10 — Schedule-wide compaction

Schemes:

- FullReservation
- Sym-OCS schedule-wide peak
- DRAC without compaction
- DRAC with compaction
- bidirectional-bundle reservation

Metrics:

- stable reserved Tx and Rx channels
- stable exposed Tx and Rx channels
- total stable pool
- compaction ratio
- independent versus bundled reservation

### Figure 11 — Iso-performance minimum stable pool

For DP, PP, and DP+PP Mixed, find the minimum stable schedule-wide pool that meets an explicitly selected reference performance. Report Tx/Rx and bundled policies separately when applicable. Do not use a single phase's used ports.

### Table II — Solver correctness and offline planning runtime

Combine:

- closed-form/binary-search target feasibility and objective gap versus numerical/exhaustive references;
- resource usage and solver runtime;
- offline stage runtime breakdown for graph parsing, profiling, target generation, `D_kh`, candidate costs, DP, realization, and compaction.

Every table cell must come from generated CSV/JSON. Missing measured inputs appear as `N/A` or `MEASUREMENT_PENDING`, never fabricated numbers.

## 5. Ablation and oracle placement

Main text should emphasize module validation rather than many full-system baselines.

- OneConfig and PerNode-Reconfig: segmentation section
- FloorOnly, NearestRounding, FillAllResidual: realization section
- NoCompaction: compaction section
- SegmentOracle: small segmentation cases
- ILP/enumeration realization oracle: small realization cases
- cross-server TP: Appendix only
- injected skew: Sensitivity or Appendix
- directional skew CDF: Appendix characterization
- full matrices/heatmaps: Appendix

## 6. New directory and result contract

Planned source layout:

```text
configs/evaluation/
  profiler/
  end_to_end/
  segmentation/
  realization/
  compaction/
  overhead/
scripts/
  run_profiler_accuracy.py
  run_end_to_end.py
  run_segmentation.py
  run_realization.py
  run_compaction.py
  run_planning_overhead.py
plots/
  plot_profiler_accuracy.py
  plot_end_to_end.py
  plot_segmentation.py
  plot_realization.py
  plot_compaction.py
  plot_planning_overhead.py
results/<version>/
  raw/
  processed/
  figures/
  tables/
  manifest.json
```

Each experiment manifest must contain:

- schema/version and experiment name
- command line and UTC timestamp
- explicit seed
- git commit and dirty-state marker
- complete resolved config snapshot
- input file hashes and provenance class
- Python/dependency/environment information
- status (`complete`, `measurement_pending`, `failed`)
- raw and processed output paths

## 7. Legacy result handling

Do not overwrite:

- `results/drac_eval_paper/`
- `results/drac_eval_smoke*/`
- `results/rescue_experiments*/`
- `results/symmetric_provisioning_cost/`
- `results/dp_pp_directional_traffic/`

The new reproduction documentation will label these as `legacy_pre_design_update`. The 96,768 old matrix JSON files are especially unsuitable as inputs because their manifest lacks a config snapshot, commit hash, and evidence provenance, and their demand matrices come from the superseded synthetic segment generator.

## 8. Paper migration checklist

When LaTeX source becomes available and experiments are complete:

1. Replace Methodology workload and baseline definitions.
2. Remove `tau/eta` as algorithmic selectors.
3. Remove weighted share error and residual target gap sections.
4. Replace current Figures 6–13 with the six-figure plan above; move qualified extras to Appendix.
5. Rewrite Evaluation questions around profiler accuracy, performance, target/segmentation correctness, sparse realization, compaction, and overhead.
6. Update captions from generated metadata.
7. Replace old TP+DP wording with DP+PP Mixed in main scope.
8. Update Abstract, Introduction, Summary, and Conclusion only with completed outputs.
9. Leave explicit TODOs for measurement-pending profiler results.
10. Verify all algorithm environments, labels, references, figure numbers, and table numbers.
11. Rebuild the PDF and visually inspect every changed page.

## 9. Migration exit criteria

Evaluation migration is complete only when:

- the core implementation tests pass;
- all six experiment smoke tests and all plot smoke tests pass;
- full simulator runs have immutable raw/processed outputs and manifests;
- Figure 6 is either supported by real measurements or explicitly omitted/TODO;
- Figures 7–11 and Table II can be regenerated without manual edits;
- paper claims match result files exactly;
- simulator, trace-derived, analytic, and measured evidence are clearly separated.
