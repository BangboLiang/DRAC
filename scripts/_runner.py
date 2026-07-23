from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from drac_eval.experiment_io import load_json


def run_cli(description: str, default_config: str, runner: Callable) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    config = load_json(args.config)
    outputs = runner(config, args.output_dir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
