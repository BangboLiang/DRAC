#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drac_eval.evaluation_v2 import run_planning_overhead_v2
from scripts._runner import run_cli
if __name__ == "__main__":
    run_cli("Run DRAC v2 planning-overhead evaluation", "configs/evaluation_v2/overhead/full.json", run_planning_overhead_v2)
