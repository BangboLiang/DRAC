#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drac_eval.evaluation_experiments import run_realization
from scripts._runner import run_cli

if __name__ == "__main__":
    run_cli("Run sparse integer realization Evaluation", "configs/evaluation/realization/full.json", run_realization)
