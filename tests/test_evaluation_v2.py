from __future__ import annotations

import csv
from pathlib import Path

import pytest

from drac_eval.evaluation_v2 import (
    run_compaction_v2,
    run_end_to_end_v2,
    run_planning_overhead_v2,
    run_realization_v2,
    run_segmentation_v2,
)
from drac_eval.experiment_io import load_json
from plots.plot_all_v2 import main as plot_all_v2


RUNNERS = {
    "end_to_end": run_end_to_end_v2,
    "segmentation": run_segmentation_v2,
    "realization": run_realization_v2,
    "compaction": run_compaction_v2,
    "overhead": run_planning_overhead_v2,
}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("name", tuple(RUNNERS))
def test_each_v2_experiment_smoke(name: str, tmp_path: Path) -> None:
    config = load_json(Path("configs/evaluation_v2") / name / "smoke.json")
    outputs = RUNNERS[name](config, tmp_path / name)
    assert Path(outputs["processed"]).is_file()
    assert Path(outputs["manifest"]).is_file()
    assert _rows(Path(outputs["processed"]))


def test_v2_end_to_end_fallback_never_exceeds_symmetric_candidate(tmp_path: Path) -> None:
    config = load_json("configs/evaluation_v2/end_to_end/smoke.json")
    rows = _rows(Path(run_end_to_end_v2(config, tmp_path)["processed"]))
    for workload in {row["workload"] for row in rows}:
        selected = next(float(row["total_cost_ms"]) for row in rows if row["workload"] == workload and row["scheme"] == "DRAC-SegmentOpt+Fallback")
        symmetric = next(float(row["total_cost_ms"]) for row in rows if row["workload"] == workload and row["scheme"] == "Sym-OCS")
        assert selected <= symmetric + 1e-8


def test_v2_realization_tradeoff_is_monotone_and_nonhorizontal(tmp_path: Path) -> None:
    config = load_json("configs/evaluation_v2/realization/smoke.json")
    rows = _rows(Path(run_realization_v2(config, tmp_path)["processed"]))
    changed = False
    for workload in {row["workload"] for row in rows}:
        selected = sorted(
            (row for row in rows if row["workload"] == workload and row["policy"] == "DRACSparse-MultiSeed"),
            key=lambda row: float(row["epsilon"]),
        )
        counts = [int(row["used_connection_units"]) for row in selected]
        assert counts == sorted(counts, reverse=True)
        changed |= len(set(counts)) > 1
    assert changed


def test_v2_segmentation_dynamic_programming_matches_oracle(tmp_path: Path) -> None:
    config = load_json("configs/evaluation_v2/segmentation/smoke.json")
    rows = _rows(Path(run_segmentation_v2(config, tmp_path)["processed"]))
    selected = [row for row in rows if row["scheme"] == "SegmentOpt-DynamicProgramming"]
    assert selected
    assert all(abs(float(row["oracle_gap"])) <= 1e-8 for row in selected)


def test_all_v2_plots_smoke(tmp_path: Path) -> None:
    root = tmp_path / "evaluation_v2"
    for name, runner in RUNNERS.items():
        config = load_json(Path("configs/evaluation_v2") / name / "smoke.json")
        runner(config, root)
    plot_all_v2(str(root))
    expected = (
        "segmentation_v2_dp",
        "segmentation_v2_pp",
        "segmentation_v2_mixed",
        "end_to_end_v2",
        "realization_tradeoff_v2",
        "schedule_compaction_v2",
        "iso_performance_pool_v2",
        "planning_runtime_v2",
        "target_case_study_pp",
    )
    for stem in expected:
        assert (root / "figures" / f"{stem}.pdf").stat().st_size > 0
        assert (root / "figures" / f"{stem}.png").stat().st_size > 0
