from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class NetworkConfig:
    base_bw_gbps: float = 25.0
    ocs_unit_bw_gbps: float = 25.0
    per_node_port_budget: int = 4
    total_ocs_links: int = 16
    reconfig_delay_ms: float = 0.5
    directional_port_reserved: int | None = None
    bidirectional_bundle_reserved: int | None = None

    def with_overrides(self, overrides: Dict[str, Any]) -> "NetworkConfig":
        data = {
            "base_bw_gbps": self.base_bw_gbps,
            "ocs_unit_bw_gbps": self.ocs_unit_bw_gbps,
            "per_node_port_budget": self.per_node_port_budget,
            "total_ocs_links": self.total_ocs_links,
            "reconfig_delay_ms": self.reconfig_delay_ms,
            "directional_port_reserved": self.directional_port_reserved,
            "bidirectional_bundle_reserved": self.bidirectional_bundle_reserved,
        }
        data.update(overrides)
        return NetworkConfig(**data)


@dataclass
class WorkloadConfig:
    name: str
    kind: str
    segment_count: int = 4
    scale: float = 1.0
    asymmetry: float | None = None
    seed_offset: int = 0
    load_path: str | None = None
    tp_group_size: int = 8
    dp_group_size: int = 8
    pp_stage_count: int = 4
    microbatches: int = 4
    model_layers: int = 126
    model_hidden: int = 16384
    model_seq: int = 8192
    model_head_dim: int = 128
    model_kv_dim: int = 1024
    model_ffn_hidden: int = 53248
    model_total_params: float = 405e9
    bytes_per_act: int = 2
    bytes_per_param: int = 2
    bytes_per_grad: int = 4
    mixed_weights: Dict[str, float] = field(
        default_factory=lambda: {"tp": 0.5, "dp": 0.5}
    )


@dataclass
class SweepConfig:
    cluster_sizes: List[int] = field(default_factory=lambda: [8])
    asymmetry_levels: List[float] = field(default_factory=lambda: [1.0, 2.0, 4.0])
    port_budgets: List[int] = field(default_factory=lambda: [2, 4, 8])
    total_ocs_links: List[int] = field(default_factory=lambda: [16])
    reconfig_delays_ms: List[float] = field(default_factory=lambda: [0.5])


@dataclass
class ExperimentConfig:
    name: str = "drac_eval"
    seed: int = 7
    output_dir: str = "results/drac_eval"
    algorithms: List[str] = field(
        default_factory=lambda: [
            "static_sym",
            "sym_ocs",
            "drac",
            "ideal_asym",
            "drac_sym",
        ]
    )
    network: NetworkConfig = field(default_factory=NetworkConfig)
    workloads: List[WorkloadConfig] = field(default_factory=list)
    sweeps: SweepConfig = field(default_factory=SweepConfig)
    generate_figures: bool = True
    save_matrices: bool = True
    high_demand_tau: float = 1.5
    high_demand_eta_fraction: float = 0.05
    figure_formats: List[str] = field(default_factory=lambda: ["png"])
    notes: str = ""


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "YAML config requested but PyYAML is not installed. "
            "Install dependencies from requirements.txt or use JSON config."
        ) from exc
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_raw_config(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        return _load_yaml(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    cfg_path = Path(path)
    raw = _load_raw_config(cfg_path)
    network = NetworkConfig(**raw.get("network", {}))
    workloads = [WorkloadConfig(**item) for item in raw.get("workloads", [])]
    sweeps = SweepConfig(**raw.get("sweeps", {}))
    return ExperimentConfig(
        name=raw.get("name", cfg_path.stem),
        seed=int(raw.get("seed", 7)),
        output_dir=str(raw.get("output_dir", "results/drac_eval")),
        algorithms=list(raw.get("algorithms", []))
        or [
            "static_sym",
            "sym_ocs",
            "drac",
            "ideal_asym",
            "drac_sym",
        ],
        network=network,
        workloads=workloads,
        sweeps=sweeps,
        generate_figures=bool(raw.get("generate_figures", True)),
        save_matrices=bool(raw.get("save_matrices", True)),
        high_demand_tau=float(raw.get("high_demand_tau", 1.5)),
        high_demand_eta_fraction=float(raw.get("high_demand_eta_fraction", 0.05)),
        figure_formats=list(raw.get("figure_formats", ["png"])),
        notes=str(raw.get("notes", "")),
    )
