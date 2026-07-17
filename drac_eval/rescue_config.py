from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .config import NetworkConfig, WorkloadConfig


@dataclass
class RescueConfig:
    name: str = "rescue_experiments"
    seed: int = 7
    seeds: List[int] = field(default_factory=lambda: [7, 19, 31])
    output_dir: str = "results/rescue_experiments"
    endpoint_count: int = 32
    asymmetry_level: float = 4.0
    endpoints_per_server: int = 2
    servers_per_tor: int = 2
    tors_per_aggregation: int = 2
    aggregation_levels: List[str] = field(
        default_factory=lambda: ["endpoint", "server", "tor", "aggregation"]
    )
    mapping_strategies: List[str] = field(
        default_factory=lambda: ["contiguous", "round_robin", "random"]
    )
    collective_models: List[str] = field(
        default_factory=lambda: [
            "original",
            "bidirectional_balanced",
            "pairwise_balancing_oracle",
        ]
    )
    algorithms: List[str] = field(
        default_factory=lambda: ["static_sym", "sym_ocs", "drac", "drac_gated"]
    )
    port_budgets: List[int] = field(default_factory=lambda: [1, 2, 4, 6])
    omega_thresholds: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.2])
    bidirectional_chunks: int = 7
    omega_epsilon: float = 1e-12
    network: NetworkConfig = field(default_factory=NetworkConfig)
    workloads: List[WorkloadConfig] = field(default_factory=list)

    def smoke_copy(self) -> "RescueConfig":
        data = dict(self.__dict__)
        data.update(
            endpoint_count=min(8, self.endpoint_count),
            seeds=[self.seeds[0] if self.seeds else self.seed],
            mapping_strategies=["contiguous"],
            port_budgets=[min(self.port_budgets) if self.port_budgets else 1],
            omega_thresholds=[0.1],
            workloads=[self.workloads[0]] if self.workloads else [],
            output_dir=str(Path(self.output_dir) / "smoke"),
        )
        return RescueConfig(**data)


def load_rescue_config(path: str | Path) -> RescueConfig:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as handle:
        raw: Dict[str, Any] = json.load(handle)
    hierarchy = raw.get("hierarchy", {})
    return RescueConfig(
        name=str(raw.get("name", cfg_path.stem)),
        seed=int(raw.get("seed", 7)),
        seeds=[int(v) for v in raw.get("seeds", [7, 19, 31])],
        output_dir=str(raw.get("output_dir", "results/rescue_experiments")),
        endpoint_count=int(raw.get("endpoint_count", 32)),
        asymmetry_level=float(raw.get("asymmetry_level", 4.0)),
        endpoints_per_server=int(hierarchy.get("endpoints_per_server", 2)),
        servers_per_tor=int(hierarchy.get("servers_per_tor", 2)),
        tors_per_aggregation=int(hierarchy.get("tors_per_aggregation", 2)),
        aggregation_levels=list(raw.get("aggregation_levels", ["endpoint", "server", "tor", "aggregation"])),
        mapping_strategies=list(raw.get("mapping_strategies", ["contiguous", "round_robin", "random"])),
        collective_models=list(raw.get("collective_models", ["original", "bidirectional_balanced", "pairwise_balancing_oracle"])),
        algorithms=list(raw.get("algorithms", ["static_sym", "sym_ocs", "drac", "drac_gated"])),
        port_budgets=[int(v) for v in raw.get("port_budgets", [1, 2, 4, 6])],
        omega_thresholds=[float(v) for v in raw.get("omega_thresholds", [0.0, 0.05, 0.1, 0.2])],
        bidirectional_chunks=int(raw.get("bidirectional_chunks", 7)),
        omega_epsilon=float(raw.get("omega_epsilon", 1e-12)),
        network=NetworkConfig(**raw.get("network", {})),
        workloads=[WorkloadConfig(**item) for item in raw.get("workloads", [])],
    )
