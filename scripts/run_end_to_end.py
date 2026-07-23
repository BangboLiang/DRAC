#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drac_eval.evaluation_experiments import run_end_to_end
from scripts._runner import run_cli

if __name__ == "__main__":
    run_cli("Run end-to-end DRAC Evaluation", "configs/evaluation/end_to_end/full.json", run_end_to_end)
