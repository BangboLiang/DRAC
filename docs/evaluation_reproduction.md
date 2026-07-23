# Revised DRAC Evaluation Reproduction

## Scope and evidence labels

The revised Evaluation is deterministic and uses seed `7` in every checked-in
configuration.  Every result row carries an evidence label:

- `DETERMINISTIC_SIMULATOR_INPUT`: derived from the checked-in communication-graph
  generator and DRAC simulator, not from hardware measurements.
- `MEASUREMENT_PENDING`: the measured NIC directional-counter input is absent.

The profiler experiment deliberately emits a pending record instead of creating
synthetic "measured" bytes.  Supply a real measurement file to the loader before
claiming profiler accuracy.

## Environment

Run from the repository root with Python 3 and the dependencies in
`requirements.txt`.  The recorded source revision is written into each raw
experiment manifest; a dirty-worktree flag is recorded separately.

## Smoke tests

```powershell
pytest -q
python -B scripts\run_all_evaluation.py --profile smoke --output-dir results\evaluation_smoke
python -B plots\plot_all.py --root results\evaluation_smoke
```

Each experiment can also be run independently:

```powershell
python -B scripts\run_profiler_accuracy.py --config configs\evaluation\profiler\smoke.json
python -B scripts\run_end_to_end.py --config configs\evaluation\end_to_end\smoke.json
python -B scripts\run_segmentation.py --config configs\evaluation\segmentation\smoke.json
python -B scripts\run_realization.py --config configs\evaluation\realization\smoke.json
python -B scripts\run_compaction.py --config configs\evaluation\compaction\smoke.json
python -B scripts\run_planning_overhead.py --config configs\evaluation\overhead\smoke.json
```

## Full simulator evaluation

```powershell
python -B scripts\run_all_evaluation.py --profile full
python -B plots\plot_all.py --root results\evaluation_v1
```

The full configurations are under `configs/evaluation/*/full.json`.  Important
fixed parameters are: seed 7, 25 Gb/s fixed bandwidth, 100 Gb/s per OCS unit,
`delta=0.5 ms` and `epsilon=0.1` for the end-to-end scan.  The compaction scan
uses `epsilon=0.5`; the realization and segmentation experiments explicitly scan
epsilon and delta, respectively.

## Outputs and provenance

- `results/evaluation_v1/raw/`: per-experiment configurations, manifests, and
  ordered node-demand JSON files.
- `results/evaluation_v1/processed/`: plotting inputs in CSV form.
- `results/evaluation_v1/figures/`: vector PDF and 200-dpi PNG outputs.
- `results/evaluation_v1/tables/target_solver_validation.tex`: Table II generated
  from the validation CSV.

The canonical Figure 6--11 files are prefixed `figure_6_` through `figure_11_`.
No number is manually copied into a plotting script.

## Providing profiler measurements

The measurement loader expects directional rows identifying operation/message
size/source/destination and measured bytes.  Use a copy of the profiler config
with its measurement path set to the immutable raw counter export.  Do not place
the only copy in `results/`: generated output directories may be regenerated.
The run fails on malformed or ambiguous directional rows rather than silently
falling back to simulated measurements.

## Interpretation caveat

The deterministic end-to-end run is a simulator validation, not a cluster
measurement.  At eight channels per endpoint, DRAC improves the DP case over
Sym-OCS, but the current PP and mixed schedules incur more reconfiguration cost
than Sym-OCS.  These negative results are retained in the CSV and Figure 7; they
must not be replaced by legacy numbers.  Hardware validation and further PP
policy work remain open.

