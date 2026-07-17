from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .collective_trace import CollectiveEvent, validate_dependencies
from .rescue_v3_config import EndpointResourceModel


@dataclass
class EventTiming:
    event_id: str
    start_us: float
    end_us: float
    network_us: float
    startup_us: float
    endpoint_us: float


@dataclass
class SimulationResult:
    completion_time_us: float
    network_only_us: float
    endpoint_stall_us: float
    startup_overhead_us: float
    reduction_copy_overhead_us: float
    max_concurrent_sends: int
    max_concurrent_receives: int
    timings: List[EventTiming]
    parameter_provenance: str


def _slot_time(slots: List[float], ready: float) -> Tuple[int, float]:
    index = min(range(len(slots)), key=lambda i: slots[i])
    return index, max(ready, slots[index])


def simulate_events(events: Sequence[CollectiveEvent], resources: EndpointResourceModel, network_bw_gbps: object) -> SimulationResult:
    validate_dependencies(events)
    ranks = sorted({event.src_rank for event in events} | {event.dst_rank for event in events})
    send_slots = {rank:[0.0]*max(1,resources.max_concurrent_sends) for rank in ranks}
    recv_slots = {rank:[0.0]*max(1,resources.max_concurrent_receives) for rank in ranks}
    channels: Dict[int,float] = {}
    channel_slots = [0.0] * max(1, resources.max_active_channels)
    launched_channels: set[int] = set()
    completed: Dict[str,float] = {}
    pending = {event.event_id:event for event in events}
    timings: List[EventTiming] = []
    startup_total = reduction_total = network_total = 0.0
    while pending:
        ready_events = [event for event in pending.values() if all(dep in completed for dep in event.dependency_ids)]
        if not ready_events:
            raise ValueError("event graph is cyclic or has missing dependencies")
        ready_events.sort(key=lambda e:(max([completed[d] for d in e.dependency_ids] or [0.0]),e.step,e.channel_id,e.event_id))
        event = ready_events[0]
        dependency_ready = max([completed[d] for d in event.dependency_ids] or [0.0])
        send_index, send_ready = _slot_time(send_slots[event.src_rank], dependency_ready)
        recv_index, recv_ready = _slot_time(recv_slots[event.dst_rank], dependency_ready)
        channel_slot = min(range(len(channel_slots)), key=lambda i: channel_slots[i])
        start = max(dependency_ready, send_ready, recv_ready, channels.get(event.channel_id,0.0), channel_slots[channel_slot])
        if not resources.full_duplex_enabled:
            start = max(start, min(send_slots[event.dst_rank]), min(recv_slots[event.src_rank]))
        # Conservative sharing: one active slot receives 1/max_concurrency of each endpoint budget.
        path_bw = float(network_bw_gbps[event.src_rank,event.dst_rank]) if hasattr(network_bw_gbps,"shape") else float(network_bw_gbps)
        effective_gbps = min(
            path_bw,
            resources.nic_egress_bw_gbps/max(1,resources.max_concurrent_sends),
            resources.nic_ingress_bw_gbps/max(1,resources.max_concurrent_receives),
            resources.pcie_or_nvlink_tx_bw_gbps/max(1,resources.max_concurrent_sends),
            resources.pcie_or_nvlink_rx_bw_gbps/max(1,resources.max_concurrent_receives),
        )
        network_us = event.bytes * 8.0 / max(effective_gbps*1e3,1e-12)
        endpoint_bw = resources.gpu_reduce_bw_gbps if "reduce" in event.phase else resources.gpu_copy_bw_gbps
        endpoint_us = event.bytes * 8.0 / max(endpoint_bw*1e3,1e-12)
        startup = resources.per_message_startup_us + resources.optional_kernel_launch_us
        if event.channel_id not in launched_channels:
            startup += resources.per_channel_launch_us
            launched_channels.add(event.channel_id)
        duration = network_us + endpoint_us + startup + resources.per_step_sync_us
        end = start + duration
        send_slots[event.src_rank][send_index] = end; recv_slots[event.dst_rank][recv_index] = end
        channels[event.channel_id] = end; channel_slots[channel_slot] = end; completed[event.event_id] = end
        timings.append(EventTiming(event.event_id,start,end,network_us,startup,endpoint_us))
        network_total += network_us; startup_total += startup; reduction_total += endpoint_us
        del pending[event.event_id]
    completion = max(completed.values(),default=0.0)
    critical_network = max((timing.network_us for timing in timings),default=0.0)
    event_by_id={event.event_id:event for event in events}
    def maximum_overlap(rank:int, source:bool)->int:
        intervals=[(t.start_us,t.end_us) for t in timings if (event_by_id[t.event_id].src_rank if source else event_by_id[t.event_id].dst_rank)==rank]
        points=sorted([(start,1) for start,_ in intervals]+[(end,-1) for _,end in intervals],key=lambda x:(x[0],x[1]))
        active=maximum=0
        for _,delta in points: active+=delta; maximum=max(maximum,active)
        return maximum
    max_s=max([maximum_overlap(rank,True) for rank in ranks] or [0]); max_r=max([maximum_overlap(rank,False) for rank in ranks] or [0])
    return SimulationResult(completion,critical_network,max(0.0,completion-critical_network),startup_total,reduction_total,max_s,max_r,timings,resources.provenance)


def cost_based_gate(predicted_sym_us: float, predicted_drac_us: float, reconfiguration_delay_us: float, omega: float, threshold: float) -> str:
    if omega < threshold or predicted_drac_us + reconfiguration_delay_us >= predicted_sym_us:
        return "sym_ocs"
    return "drac_makespan_opt"
