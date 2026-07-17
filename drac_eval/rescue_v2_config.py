from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .config import WorkloadConfig


@dataclass
class DeploymentLevelConfig:
    base_bw_gbps: float
    out_port_budget: int
    in_port_budget: int
    total_ocs_links: int


@dataclass
class RescueV2Config:
    name: str
    output_dir: str
    seeds: List[int]
    rank_counts: List[int]
    endpoint_count: int
    asymmetry_level: float
    workloads: List[WorkloadConfig]
    aggregation_levels: List[str]
    mapping_strategies: List[str]
    endpoints_per_server: int
    servers_per_tor: int
    tors_per_aggregation: int
    normalization_modes: List[str]
    endpoint_base_capacity_gbps: float
    endpoint_out_port_budget: int
    endpoint_in_port_budget: int
    ocs_unit_bw_gbps: float
    global_ocs_budget: int
    deployment_specific: Dict[str, DeploymentLevelConfig]
    collective_schedule_models: List[str]
    chunk_count: int
    odd_chunk_rule: str
    intra_collective_reconfiguration: bool
    reconfig_delay_ms: float
    algorithms: List[str]
    port_budgets: List[int]
    omega_thresholds: List[float]
    solver_tolerance: float
    benefit_epsilon: float
    paper_config: str

    def smoke_copy(self) -> "RescueV2Config":
        data = dict(self.__dict__)
        data.update(
            output_dir=str(Path(self.output_dir) / "smoke"),
            seeds=[self.seeds[0]],
            rank_counts=[8],
            endpoint_count=8,
            mapping_strategies=["contiguous"],
            normalization_modes=["resource_equivalent"],
            port_budgets=[min(self.port_budgets)],
            workloads=[self.workloads[0]],
            chunk_count=min(8, self.chunk_count),
        )
        return RescueV2Config(**data)


def load_rescue_v2_config(path: str | Path) -> RescueV2Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: Dict[str, Any] = json.load(handle)
    hierarchy = raw["aggregation_hierarchy"]
    endpoint = raw["endpoint_network"]
    deployment = {
        level: DeploymentLevelConfig(**values)
        for level, values in raw.get("deployment_specific", {}).items()
    }
    return RescueV2Config(
        name=str(raw.get("name", "rescue_experiments_v2")),
        output_dir=str(raw.get("output_dir", "results/rescue_experiments_v2")),
        seeds=[int(v) for v in raw.get("seeds", [7, 19, 31])],
        rank_counts=[int(v) for v in raw.get("rank_counts", [8, 16, 32, 64, 128])],
        endpoint_count=int(raw.get("endpoint_count", 32)),
        asymmetry_level=float(raw.get("asymmetry_level", 4.0)),
        workloads=[WorkloadConfig(**v) for v in raw.get("workloads", [])],
        aggregation_levels=list(raw.get("aggregation_levels", ["endpoint", "server", "tor", "aggregation"])),
        mapping_strategies=list(raw.get("mapping_strategies", ["contiguous", "round_robin", "random"])),
        endpoints_per_server=int(hierarchy["endpoints_per_server"]),
        servers_per_tor=int(hierarchy["servers_per_tor"]),
        tors_per_aggregation=int(hierarchy["tors_per_aggregation"]),
        normalization_modes=list(raw.get("normalization_modes", ["resource_equivalent", "deployment_specific"])),
        endpoint_base_capacity_gbps=float(endpoint["base_capacity_gbps"]),
        endpoint_out_port_budget=int(endpoint["out_port_budget"]),
        endpoint_in_port_budget=int(endpoint["in_port_budget"]),
        ocs_unit_bw_gbps=float(endpoint["ocs_unit_bw_gbps"]),
        global_ocs_budget=int(endpoint["global_ocs_budget"]),
        deployment_specific=deployment,
        collective_schedule_models=list(raw.get("collective_schedule_models", [])),
        chunk_count=int(raw.get("chunk_count", 8)),
        odd_chunk_rule=str(raw.get("odd_chunk_rule", "extra_clockwise")),
        intra_collective_reconfiguration=bool(raw.get("intra_collective_reconfiguration", False)),
        reconfig_delay_ms=float(raw.get("reconfig_delay_ms", 0.5)),
        algorithms=list(raw.get("algorithms", ["sym_ocs", "drac_makespan_opt"])),
        port_budgets=[int(v) for v in raw.get("port_budgets", [1, 2, 4, 6])],
        omega_thresholds=[float(v) for v in raw.get("omega_thresholds", [0.0, 0.05, 0.1, 0.2])],
        solver_tolerance=float(raw.get("solver_tolerance", 1e-10)),
        benefit_epsilon=float(raw.get("benefit_epsilon", 1e-8)),
        paper_config=str(raw.get("paper_config", "configs/drac_eval_paper.json")),
    )
