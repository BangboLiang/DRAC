from __future__ import annotations

import json
from pathlib import Path

import pytest

from drac_eval.evaluation_experiments import (
    run_compaction,
    run_end_to_end,
    run_planning_overhead,
    run_profiler_accuracy,
    run_realization,
    run_segmentation,
)
from drac_eval.experiment_io import load_json
from plots.plot_compaction import plot as plot_compaction
from plots.plot_end_to_end import plot as plot_end_to_end
from plots.plot_planning_overhead import plot as plot_overhead
from plots.plot_profiler_accuracy import plot as plot_profiler
from plots.plot_realization import plot as plot_realization
from plots.plot_segmentation import plot as plot_segmentation


CASES = [
    ("profiler", run_profiler_accuracy),
    ("end_to_end", run_end_to_end),
    ("segmentation", run_segmentation),
    ("realization", run_realization),
    ("compaction", run_compaction),
    ("overhead", run_planning_overhead),
]


@pytest.mark.parametrize(("name", "runner"), CASES)
def test_each_experiment_smoke(name, runner, tmp_path: Path) -> None:
    config = load_json(Path("configs/evaluation") / name / "smoke.json")
    outputs = runner(config, tmp_path / "evaluation")
    assert outputs["processed"].exists()
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "drac_evaluation/v1"
    assert manifest["seed"] == 7
    assert manifest["git_commit"]
    assert manifest["config"] == config


def test_end_to_end_fixed_seed_is_byte_reproducible(tmp_path: Path) -> None:
    config = load_json("configs/evaluation/end_to_end/smoke.json")
    first = run_end_to_end(config, tmp_path / "first")["processed"]
    second = run_end_to_end(config, tmp_path / "second")["processed"]
    assert first.read_bytes() == second.read_bytes()


def test_all_plot_generators_smoke(tmp_path: Path) -> None:
    root = tmp_path / "evaluation"
    outputs = {}
    for name, runner in CASES:
        config = load_json(Path("configs/evaluation") / name / "smoke.json")
        outputs[name] = runner(config, root)

    plot_profiler(str(outputs["profiler"]["processed"]), str(root / "figures" / "profiler_accuracy"))
    plot_end_to_end(str(outputs["end_to_end"]["processed"]), str(root / "figures" / "end_to_end_performance"))
    plot_segmentation(str(outputs["segmentation"]["processed"]), str(root / "figures" / "segmentation"))
    plot_realization(str(outputs["realization"]["processed"]), str(root / "figures" / "realization_tradeoff"))
    plot_compaction(str(outputs["compaction"]["processed"]), str(outputs["compaction"]["iso"]), str(root / "figures"))
    plot_overhead(str(outputs["overhead"]["processed"]), str(root / "figures" / "planning_runtime_breakdown"))

    expected = {
        "profiler_accuracy",
        "end_to_end_performance",
        "segmentation",
        "realization_tradeoff",
        "schedule_compaction",
        "iso_performance_pool",
        "planning_runtime_breakdown",
    }
    for stem in expected:
        assert (root / "figures" / f"{stem}.pdf").exists()
        assert (root / "figures" / f"{stem}.png").exists()
