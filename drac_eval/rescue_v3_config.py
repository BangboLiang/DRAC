from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class EndpointResourceModel:
    nic_egress_bw_gbps: float
    nic_ingress_bw_gbps: float
    pcie_or_nvlink_tx_bw_gbps: float
    pcie_or_nvlink_rx_bw_gbps: float
    gpu_reduce_bw_gbps: float
    gpu_copy_bw_gbps: float
    max_concurrent_sends: int
    max_concurrent_receives: int
    max_active_channels: int
    per_message_startup_us: float
    per_step_sync_us: float
    per_channel_launch_us: float
    optional_kernel_launch_us: float
    full_duplex_enabled: bool
    provenance: str


@dataclass
class RescueV3Config:
    output_dir: str
    trace_sources: List[Dict[str, str]]
    collective_types: List[str]
    algorithms: List[str]
    protocols: List[str]
    message_sizes: List[int]
    channel_counts: List[int]
    rank_count: int
    rank_placement: Dict[str, object]
    hierarchy: Dict[str, int]
    window_sizes: List[int]
    ocs_reconfiguration_delay_us: List[float]
    endpoint: EndpointResourceModel
    base_bw_gbps: float
    ocs_unit_bw_gbps: float
    per_rank_port_budget: int
    total_ocs_links: int
    network_schemes: List[str]
    collective_schemes: List[str]
    seeds: List[int]
    solver_tolerance: float
    omega_threshold: float
    sensitivity: Dict[str, List[float]]

    def smoke_copy(self) -> "RescueV3Config":
        data = dict(self.__dict__)
        data.update(
            output_dir=str(Path(self.output_dir) / "smoke"),
            collective_types=["allreduce"], message_sizes=[1048576],
            channel_counts=[2], window_sizes=[1,2,4], seeds=[self.seeds[0]],
            sensitivity={key: values[:2] for key, values in self.sensitivity.items()},
        )
        return RescueV3Config(**data)


def load_rescue_v3_config(path: str | Path) -> RescueV3Config:
    with Path(path).open("r", encoding="utf-8") as handle: raw = json.load(handle)
    return RescueV3Config(
        output_dir=raw["output_dir"], trace_sources=list(raw["trace_sources"]),
        collective_types=list(raw["collective_types"]), algorithms=list(raw["algorithms"]),
        protocols=list(raw["protocols"]), message_sizes=[int(v) for v in raw["message_sizes"]],
        channel_counts=[int(v) for v in raw["channel_counts"]], rank_count=int(raw["rank_count"]),
        rank_placement=dict(raw["rank_placement"]), hierarchy={k:int(v) for k,v in raw["aggregation_hierarchy"].items()},
        window_sizes=[int(v) for v in raw["window_sizes"]],
        ocs_reconfiguration_delay_us=[float(v) for v in raw["ocs_reconfiguration_delay_us"]],
        endpoint=EndpointResourceModel(**raw["endpoint_resource_model"]),
        base_bw_gbps=float(raw["network_model"]["base_bw_gbps"]),
        ocs_unit_bw_gbps=float(raw["network_model"]["ocs_unit_bw_gbps"]),
        per_rank_port_budget=int(raw["network_model"]["per_rank_port_budget"]),
        total_ocs_links=int(raw["network_model"]["total_ocs_links"]),
        network_schemes=list(raw["network_schemes"]), collective_schemes=list(raw["collective_schemes"]),
        seeds=[int(v) for v in raw["seeds"]], solver_tolerance=float(raw["solver_tolerance"]),
        omega_threshold=float(raw["omega_threshold"]), sensitivity={k:[float(x) for x in v] for k,v in raw["sensitivity"].items()},
    )
