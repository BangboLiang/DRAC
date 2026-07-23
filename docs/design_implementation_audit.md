# DRAC Design–Implementation Audit

## 1. Audit scope and source of truth

This audit was performed before changing the DRAC implementation. The source of truth is the newest paper artifact in the repository:

- File: `paper.pdf`
- Modified: 2026-07-23 08:20:35 +08:00
- Size: 1,629,007 bytes
- Pages: 17
- SHA-256: `61B1E9C9480F9C1CF84515811D5BEB0CCB1929F5284E68D4ABA7447E17E1F2E6`
- Repository revision at audit time: `c715afba5ec1939d5c62f1a27962a9328cd1b6ab`

No LaTeX source is present in this checkout (`rg --files -g '*.tex'` returned no files). Therefore the PDF is the only available paper source. Paper-source synchronization and PDF rebuilding cannot be completed until the current LaTeX project is supplied or restored.

The following paper sections were read in full and visually checked after rendering their pages:

- Problem Formulation: pages 5–6
- DRAC Design: pages 6–9
- Current Evaluation: pages 9–12
- Appendix algorithms and proofs: pages 14–17

Repository areas inspected:

- `README.md`, `AGENTS.md`, requirements and test configuration
- all root `llama3_*.py` entry scripts and the README evolution history
- `llama3_comm/` traffic, trace ingestion, trace-to-communication conversion, solvers, peer/rank lifting, execution, and Astra emission paths
- the main Evaluation path under `drac_eval/`
- rescue/audit paths `rescue_v2`, `rescue_v3`, and `rescue_v4_atlahs`
- workload/config files, tests, plotting code, cached results, manifests, raw traces, and reports

## 2. Actual entry points and call paths

Names alone are misleading because the repository contains several generations of logic. The actual call paths are:

### 2.1 Latest general Llama communication simulator

`llama3_modular.py` is the latest general simulator identified by `README.md`.

Its active path is:

`llama3_modular.py::main`
→ `llama3_modular.py::_build_llama_nodes`
→ `llama3_comm.solvers.preplanned_dp_partition` or `fast_preplanned_partition`
→ `llama3_comm.solvers.solve_min_delay_bw_split`
→ `llama3_comm.solvers.solve_best_link_plan_for_bw_segment`
→ `llama3_comm.execution._trace_from_segments`.

This path optimizes scalar shares among TP/PP/DP domains and peer-degree batches. It is a legacy ACTINA-style domain allocation model, not the paper's ordered endpoint-pair DRAC implementation.

### 2.2 Trace-driven Astra planning

`trace_reconfig_plan.py::main`
→ `llama3_comm.trace_ingest.load_trace_bundle`
→ `llama3_comm.trace_to_comm.build_comm_nodes_from_rank_trace`
→ `llama3_comm.role_planning.plan_segments`
→ legacy domain-share segmentation in `llama3_comm.solvers`
→ `llama3_comm.peer_plan.build_abstract_peer_plan`
→ `instantiate_concrete_edge_state`
→ Astra mutable-topology output.

This path preserves an ordered communication-node stream and can lift logical collective edges, but it does not produce one calibrated ordered demand matrix per node, solve the new continuous target, or run the new integer realization.

### 2.3 Current main paper Evaluation

`run_drac_eval.py::main`
→ `drac_eval.config.load_experiment_config`
→ `drac_eval.runner.run_experiments`
→ `drac_eval.traffic.load_or_generate_workload`
→ `drac_eval.allocation.allocate_for_algorithm`
→ `drac_eval.metrics.compute_segment_metrics`
→ `drac_eval.plotting.generate_all_figures`.

This is the code that generated `results/drac_eval_paper/` and the old Evaluation plots. It is the primary implementation that must be replaced for the new Design.

### 2.4 Rescue/audit paths

- `run_rescue_experiments.py` dispatches to `rescue_runner`, `rescue_v2_runner`, `rescue_v3_runner`, or `rescue_v4_runner`.
- `rescue_v2` established that the original rank matrices are synthetic sensitivity inputs and that square-root allocation is not theoretically correct for the makespan objective.
- `rescue_v3` contains a deterministic endpoint-resource event simulator and NCCL INFO fixture reconstruction, but explicitly reports no real measurement on this host.
- `rescue_v4_atlahs` parses official GOAL traces. Those traces are trace-derived communication records without packet-level counters or raw timestamps; its time windows are simulated and its larger-boundary maps are hypothetical.

These rescue artifacts are useful provenance and negative evidence. They are not substitutes for the new end-to-end DRAC implementation.

## 3. Paper-defined module contracts

### 3.1 Ordered Demand Profiler

Paper input:

- static training computation graph or communication subgraph
- ordered communication nodes `P = (p_1, ..., p_K)`
- rank placement and GPU–NIC endpoint mapping
- selected collective implementation and its deterministic schedule
- transport/control calibration indexed by environment and message-size bin

Paper output:

- one ordered inter-server demand matrix `T^(k)` for every communication node
- payload and calibrated overhead/control contributions retained with provenance
- `T_uv^(k)` and `T_vu^(k)` treated independently
- all intra-server demand removed

Core algorithm:

1. Extract communication nodes without pre-segmenting them.
2. Expand every collective into ordered point-to-point transfers.
3. Map transfer ranks to GPU–NIC endpoints.
4. Add same-direction payload/overhead and reverse-direction control calibration.
5. Exclude transfers for which source and destination servers are equal.

The profiler does not use `tau`, `eta`, skew selection, workload-level aggregation, or a TP/DP/PP scalar domain as the optimization object.

### 3.2 Per-node continuous directional target

Paper input:

- a single node demand matrix `T^(k)`
- per-endpoint `n_tx` and `n_rx`
- connection-unit bandwidth `c`
- optional fixed capacity matrix `B_fix`

Paper output:

- optimal node completion time `Theta*_k`
- minimum-resource optimal target `Y*^(k)`

For `B_fix = 0`:

`Theta*_k = max(max_u sum_v T_uv/(c n_tx_u), max_v sum_u T_uv/(c n_rx_v))`

and

`Y*^(k)_uv = T_uv/(c Theta*_k)`.

For `B_fix > 0`, feasibility is monotone in `Theta` and the minimum augmentation is:

`Y_uv(Theta) = max(T_uv/Theta - B_fix_uv, 0)/c`.

Binary search finds the smallest feasible `Theta`. The result does not exhaust slack endpoint resources.

### 3.3 Target-sequence segmentation

Paper input:

- complete node-target sequence `{Y*^(k)}`
- original demands `{T^(k)}`
- reconfiguration delay `delta`

Paper output:

- contiguous boundaries `0=q_0<...<q_M=K`
- one representative node-target index `h_m` per segment
- logical segment costs and total DP objective

Core algorithm:

1. Build `D_kh = L_k(Y*^(h))`.
2. For every candidate interval `[s,e]`, select the in-interval target minimizing the sum of service costs.
3. Run the recurrence in Equation 10.
4. Backtrack exact boundaries and representative indices.

The candidate segment must not be re-solved as an aggregate-demand continuous problem.

### 3.4 Sparse integer OCS realization

Paper input:

- selected segment representative target `Y*^(m)`
- all demands in the segment and logical cost `C_tilde_m`
- `epsilon`, endpoint Tx/Rx budgets, reachability, `B_fix`, and unit bandwidth `c`

Paper output:

- feasible integer matrix `X^(m)`
- achieved cost, used units, tolerance status
- resource-constrained flag and remaining high-gain ordered directions

Objective:

minimize `||X||_1` subject to `X in F_OCS` and `C_m(X) <= (1+epsilon) C_tilde_m`.

Required heuristic:

1. start from `floor(Y*)`;
2. ensure positive-demand directions without fixed capacity are serviceable;
3. add the feasible unit with largest marginal segment-cost gain;
4. stop immediately when the performance target is reached;
5. reverse-prune removable units;
6. report inability to meet the target without fabricating feasibility.

### 3.5 Schedule-wide resource compaction

Paper input:

- the complete integer schedule `{X^(m)}`
- physical Tx/Rx channel inventory

Paper output:

- minimum stable independent Tx and Rx pools
- optional bidirectional bundle pool
- physical channel binding for every segment
- stable exposed resources
- resource-constrained direction request map

Closed form:

- `nbar_tx_u = max_m sum_v X_uv^(m)`
- `nbar_rx_u = max_m sum_v X_vu^(m)`
- `nbar_bi_u = max(nbar_tx_u, nbar_rx_u)` for bundled resources.

Per-segment idle channels are not releasable unless they fall outside these schedule-wide peaks.

## 4. Implementation-by-module findings

| Paper module | Current implementation location | What is already usable | Material mismatch or gap | Verdict |
|---|---|---|---|---|
| Communication-node extraction | `llama3_comm.trace_ingest`, `trace_to_comm.build_comm_nodes_from_rank_trace` | Preserves a serialized communication-node order; reads normalized trace DAGs; distinguishes PP send/recv and collective domains | Uses one representative rank, converts nodes to scalar `CommNode`, forces missing payload to at least 1 byte, defaults to include TP, and does not emit all-rank ordered transfers or endpoint matrices | Partial building block |
| Collective expansion | `drac_eval.rescue_schedule.build_executable_ring_schedule`; `nccl_reconstruct.reconstruct_ring_schedule`; `rank_lift` | Explicit ring events and byte-conservation tests exist; PP directed edges can be lifted | Not integrated with the main profiler; only ring reconstruction is supported reliably; fixture-derived NCCL topology is not a measured schedule; reverse transport traffic is absent | Partial building block |
| Rank placement | `collective_trace.RankPlacement`, `trace_ir`, `rank_lift`, `rescue_v3_timescale.build_mapping` | Rank/host records and several mappings exist | Main Evaluation ignores server placement and models matrix indices directly as endpoints; no explicit GPU–NIC endpoint object; intra-server exclusion is absent in `drac_eval.traffic` | Missing in main path |
| Calibration | `directional_traffic.py`, `rescue_v3` measurement instructions | A provisional legacy overhead-ratio model and honest `MEASUREMENT_PENDING` status exist | `directional_traffic.py` has committed conflict markers and cannot import; legacy aggregate ratios are not message-bin calibration; no measured directional-counter loader for requested accuracy evaluation | Broken/incomplete |
| Per-node target | `drac_eval.allocation._sqrt_share_matrix`; `rescue_allocation.proportional_target` | Endpoint row/column budgets are represented by the realization constraints | Main DRAC uses `sqrt(demand)` over a global budget. Rescue proportional scaling iterates toward fully scaled capacity. Neither implements Equation 8 or `B_fix` binary search | Old algorithm |
| Segmentation | `llama3_comm.solvers.preplanned_dp_partition`; main runner's fixed `segment_count` | Legacy DP recurrence exists for domain-share segments | Candidate segments are re-solved with `solve_min_delay_bw_split`; main Evaluation does no segmentation at all and charges reconfiguration on every pre-generated matrix; no `D_kh`, representative target, or new backtracking | Old/missing |
| Integer realization | `drac_eval.allocation._realize_asymmetric`, `_realize_symmetric`; `rescue_allocation.allocate_discrete_makespan_opt` | Feasible row, column, and global integer budgets; an exact minimum-makespan reference exists for small/current cases | `_realize_asymmetric` fills positive residual target gaps until no positive residual/port remains; no epsilon bound, minimum-unit objective, reverse pruning, constrained request map, or required ablation policies | Old algorithm |
| Symmetric baselines | `_symmetric_target`, `_realize_symmetric` | Enforces pair symmetry | Uses square-root aggregate pair weights and pre-fixed segments; not the same target-sequence segmentation/delay model as new DRAC | Must be rebuilt for fairness |
| Simulator | `drac_eval.metrics._completion_time_ms`; `rescue_v3_endpoint.simulate_events`; legacy `llama3_comm.traffic.estimate_time_ms` | Matrix bottleneck cost matches Equation 4 at a simple level; V3 simulator models endpoints/dependencies deterministically | Main runner uses only matrix makespan with an all-to-all fixed base bandwidth matrix; does not model node dependency/overlap or segment schedule consistently; V3 is uncalibrated and not wired to new configurations | Partial, uncalibrated |
| Compaction | `drac_eval.metrics.aggregate_port_exposure` | Computes elementwise schedule peak outbound/inbound and bundled max, which matches the core lower-bound formula | Main report also exposes per-segment idle values; no stable physical pool identities, per-segment binding, per-endpoint result table, independent-vs-bundle experiment, or request map | Formula aligned, implementation incomplete |
| Results/provenance | `drac_eval.runner`, rescue reports | Seed is explicit; raw CSV and matrices are saved; rescue reports label evidence honestly | Main manifest lacks git hash, full config, dependency/environment, timestamps, input hashes, provenance, and calibration status; 96,768 matrix JSONs are generated with no schema version; output structure is legacy | Must be replaced |
| Plotting | `drac_eval.plotting` | Explicit `figsize`, size validation, closing figures, PDF/PNG support | Generates the obsolete Evaluation: skew CDF, weighted share error, capacity waste, residual gap, and old port saving | Replace main plots |

## 5. Detailed incompatibilities with the new Design

### 5.1 Ordered profiler

`drac_eval.traffic.load_or_generate_workload` takes a configured `segment_count` and directly returns `SegmentDemand` matrices. The matrices are already aggregated over modeled layers/microbatches. Directionality is created by `_dominant_pair_value`, random orientation, fixed-offset peer rules, and a user-supplied asymmetry factor. `rescue_v2` already documents that these are synthetic sensitivity matrices, not NCCL traces.

Specific violations:

- optimization starts from pre-segmented matrices rather than communication nodes;
- TP/DP/PP are treated as synthetic workload domains;
- main Evaluation uses TP and mixed TP+DP;
- no server map is applied and same-server demand is not removed;
- collective expansions are not called by the main runner;
- calibration is not message-size indexed;
- current PP nodes combine forward and reverse values in each phase matrix instead of representing actual forward and backward communication nodes;
- no provenance differentiates payload-derived, trace-derived, measured, calibrated, or synthetic bytes.

### 5.2 Continuous target

`allocate_for_algorithm('drac', ...)` calls `_sqrt_share_matrix`. This implements:

`target = global_budget * sqrt(demand) / sum(sqrt(demand))`.

It is exactly the superseded square-root/global-share target. It ignores distinct `n_tx` and `n_rx`, uses a single global link budget as the normalizer, has no `B_fix` solver, and forces the target sum to the whole global overlay budget even when non-bottleneck endpoints have slack.

The rescue path's `proportional_target` rescales a whole aggregate demand under row/column/global caps, but it is not the closed form because it seeks a fully scaled feasible matrix and includes a global cap absent from the paper's node formula.

### 5.3 Segmentation

The main runner never invokes a segmenter. It loops over matrices already produced by `segment_count` and charges `delta` for every non-static matrix after the first.

The legacy `preplanned_dp_partition` does use dynamic programming, but for every candidate interval it calls `solve_min_delay_bw_split` again and optimizes TP/PP/DP shares. Thus its recurrence and state object differ from Algorithm 1 even though both are named DP segmentation.

### 5.4 Sparse realization

`_realize_asymmetric` floors requested units and then repeatedly adds to the largest positive target residual. It stops only when no positive residual remains or resources are exhausted. This is FillAllResidual behavior, not DRACSparse. It does not evaluate segment cost during additions and cannot stop at `(1+epsilon) C_tilde`. No pruning is performed.

`requested_extra_bw_gbps` is just aggregate target residual. Treating it as a resource request conflicts with the new paper, which requires marginal-gain directions only for a resource-constrained segment.

### 5.5 Compaction

`aggregate_port_exposure` correctly computes max per-endpoint row and column usage over supplied allocations, then sums them. This can be retained as a mathematical core after making results per endpoint and adding binding. However, `compute_segment_metrics` also reports `releasable_directional_ports` from instantaneous segment usage. Those values are unsafe to interpret as releasable resources. The summary partly masks this by taking max active/min releasable, but the raw CSV remains semantically wrong.

## 6. Evaluation and paper consistency findings

The latest PDF contains new Problem Formulation, Design, and Appendix material, but its Evaluation is still the old version. Confirmed contradictions include:

- Design says no skew threshold; Evaluation says DRAC selects high-demand directions using `tau` and `eta`.
- Design targets per-node completion time and sparse resources; Evaluation uses weighted directional share error and residual target gap.
- System boundary contains cross-server DP/PP and intra-server TP; Evaluation and Conclusion center TP and mixed TP+DP.
- Design segments the node-target sequence; Methodology says it constructs workload demand matrices and the code uses preselected segment counts.
- Design exposes schedule-wide stable resources; the current port-saving result is based on old port budgets/aggregate summaries and not a minimum stable channel pool experiment.
- Abstract and Introduction contain `xx` placeholders, while Conclusion states unsupported old TP/DP/mixed findings.

Figure 6 in the current PDF is the old directional-skew CDF. Figures 7–13 are likewise produced by the old `drac_eval.plotting` storyline. None can be used as a validation result for the new Design.

## 7. Data and measurement evidence

Available evidence:

- `data/atlahs/` contains official GOAL traces with checksums and parsed SQLite caches.
- These traces can support trace-derived ordered send/receive demand at their documented server granularity.
- `tools/nccl_trace/fixtures/` supports parser and deterministic schedule tests.
- legacy physical per-port aggregate counters are referenced in the directional-traffic work, but the source file is embedded code/data rather than a preserved raw measurement artifact.

Unavailable evidence:

- no raw NIC directional-counter dataset for the requested profiler accuracy experiment;
- no packet trace;
- no real multi-node NCCL run on this host;
- no measured per-message-size calibration bins;
- no calibrated endpoint runtime model;
- GOAL traces have no raw timestamps or channel IDs.

Consequences:

- `profiler_accuracy.csv/pdf` may include schema validation and any legitimately trace-derived comparison, but must mark measured metrics as `MEASUREMENT_PENDING` until raw counter data arrives.
- PayloadOnly vs Payload+Calibration cannot be claimed against measured bytes using current data.
- Simulator results must be labeled `SIMULATOR`, `TRACE_DERIVED_SIMULATOR`, or similar, never `measured`.
- Existing paper performance numbers must not be copied into the revised Abstract/Introduction/Conclusion.

## 8. Test and repository integrity baseline

Audit command:

```powershell
python -B -m pytest -q -p no:cacheprovider
```

Result: collection failed after 45.61 seconds.

Cause:

- `drac_eval/directional_traffic.py` contains committed `<<<<<<<`, `=======`, and `>>>>>>>` markers.
- `generate_dp_pp_directional_traffic.py` contains the same unresolved merge conflict.
- `tests/test_directional_traffic.py` and `tests/test_symmetric_provisioning_cost.py` cannot import the module.

The working tree was otherwise clean at audit start. The conflict markers are part of repository revision `c715afb`, not uncommitted user work.

Existing tests cover several useful legacy invariants, including matrix shape, symmetric realization, budget feasibility, ring byte conservation, trace parsing, reproducible mapping, and some compaction-related aggregate formulas. They do not cover the 16 required new Design tests.

## 9. Files to add or modify

The safest approach is to keep legacy `llama3_*` behavior immutable and introduce a new Design-specific pipeline under `drac_eval/`. Expected changes:

Core modules:

- add `drac_eval/demand_profiler.py`
- add `drac_eval/directional_target.py`
- add `drac_eval/target_segmentation.py`
- add `drac_eval/sparse_realization.py`
- add `drac_eval/resource_compaction.py`
- add `drac_eval/evaluation_pipeline.py`
- extend or replace the main use of `drac_eval/config.py`, `runner.py`, and `metrics.py`
- resolve `drac_eval/directional_traffic.py` and `generate_dp_pp_directional_traffic.py` without losing either intended API
- reuse trace primitives from `llama3_comm` only where their behavior matches the new contracts

Tests:

- add focused test files for profiler, target solver, segmentation, realization, compaction, reproducibility, experiment smoke, and plotting smoke
- keep legacy tests running to enforce backward compatibility

Evaluation:

- add `configs/evaluation/{profiler,end_to_end,segmentation,realization,compaction,overhead}/`
- add six independent runners under `scripts/`
- add six independent plotting modules under `plots/`
- add a top-level all-experiments runner and smoke configuration
- write results under a versioned new root; never overwrite old results

Documentation/paper:

- add `docs/evaluation_reproduction.md`
- add `docs/final_change_summary.md`
- update README with the new canonical entry path and evidence labels
- update the LaTeX project after it is supplied; then rebuild and visually verify `paper.pdf`

## 10. Recommended implementation order

1. Preserve a baseline snapshot and resolve committed conflict markers.
2. Define typed communication nodes, endpoints, placements, transfers, calibration records, and provenance.
3. Implement ordered expansion and same-server filtering.
4. Implement and validate the per-node target solver for both `B_fix` cases.
5. Implement service-cost construction and target-dictionary DP segmentation.
6. Implement realization ablations, DRACSparse, pruning, and the small oracle.
7. Implement stable compaction, physical binding, and request maps.
8. Integrate fair Static-Sym/Sym-OCS/DRAC pipelines on identical node sequences.
9. Add tests before building Evaluation runners.
10. Run smoke suites; only then run full simulator experiments.
11. Generate figures/tables only from stored processed results.
12. Update the paper only with completed results and explicit TODOs for missing measurements.

## 11. Old outputs that must be rerun or retired

All artifacts under `results/drac_eval_paper/` must be treated as legacy and cannot support the new paper. This includes its raw CSV, summary CSV, 96,768 matrix JSON files, and all generated figures.

Must be recomputed after core changes:

- all communication-time comparisons;
- PP no-harm;
- injected-skew sensitivity if retained;
- iso-performance channel saving;
- representative heatmaps/case study;
- any resource exposure or utilization result;
- any offline runtime result.

Must not be used as new main metrics:

- weighted directional share error;
- residual target gap;
- instantaneous releasable ports;
- capacity waste ratio as an Evaluation headline;
- threshold-selected high-demand direction counts.

May be retained only as clearly labeled Appendix/background evidence:

- symmetric-provisioning motivating micro-example;
- directional-skew characterization with `tau` used only as a descriptive threshold;
- injected-skew sensitivity rerun through the new algorithm;
- extended cross-server TP scenarios, explicitly outside the default system boundary;
- rescue negative-evidence reports and trace provenance audits.

The old result directories should be archived or referenced in a legacy index. They must not be silently deleted or overwritten.

## 12. Phase-1 conclusion

The mathematical Design in the latest paper is implementable with existing repository primitives, but the current main DRAC path does not implement it. The main Evaluation is tied to the superseded algorithm and synthetic pre-segmented matrices. The only core component already close to the new Design is the schedule-wide peak calculation inside `aggregate_port_exposure`; even that needs stable-pool identities, binding, and corrected reporting semantics.

No old performance figure is valid evidence for the new Design. The next phase must begin with a new ordered-node pipeline and exact per-node target solver, not with plot edits.
