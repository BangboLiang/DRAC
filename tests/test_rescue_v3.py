from __future__ import annotations
import tempfile
from dataclasses import replace
from pathlib import Path
import numpy as np

from drac_eval.collective_trace import CollectiveEvent,load_collective_events_csv,write_collective_events_csv,validate_dependencies
from drac_eval.nccl_log import parse_nccl_log
from drac_eval.nccl_reconstruct import fixed_half_bidirectional,reconstruct_ring_schedule
from drac_eval.rescue_v2_network import allocate_drac_makespan_opt,validate_general_units
from drac_eval.rescue_v3_config import load_rescue_v3_config
from drac_eval.rescue_v3_endpoint import cost_based_gate,simulate_events
from drac_eval.rescue_v3_runner import _events,_network,run_environment
from drac_eval.rescue_v3_timescale import analyze_timescales,direction_metrics

FIXTURE="tools/nccl_trace/fixtures/nccl_info_ring_4ranks_2channels.log"

def _cfg():return load_rescue_v3_config("configs/rescue_experiments_v3.json")
def _record():return parse_nccl_log(FIXTURE)

def test_nccl_ring_and_multichannel_parser():
    record=_record();assert record.rank_count==4;assert record.channel_count==2
    assert record.channels[0].rank_order==(0,1,2,3);assert record.channels[1].rank_order==(0,3,2,1)
    assert record.algorithm=="Ring" and record.protocol=="Simple" and record.transports==["NET"]

def test_schedule_semantics_bytes_channels_chunks_and_ownership():
    record=_record();original=reconstruct_ring_schedule(record);balanced=fixed_half_bidirectional(record)
    oe=_events(original);be=_events(balanced);assert len(original.channels)==len(balanced.channels)==2
    assert len(oe)==len(be)==48;assert sum(e.bytes for e in oe)==sum(e.bytes for e in be)==record.message_bytes*6
    assert len({e.chunk_id for e in oe})==8
    for channel in range(2):
        for chunk in range(channel*4,(channel+1)*4):
            seq=sorted([e for e in oe if e.channel_id==channel and e.chunk_id==chunk],key=lambda e:e.step)
            assert len(seq)==6
            for left,right in zip(seq,seq[1:]):assert left.dst_rank==right.src_rank and left.event_id in right.dependency_ids

def test_event_csv_schema_dependency_and_provenance():
    events=_events(reconstruct_ring_schedule(_record()));validate_dependencies(events)
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/"events.csv";write_collective_events_csv(path,events);loaded=load_collective_events_csv(path)
        assert len(loaded)==len(events);assert all(e.provenance=="nccl_selected_schedule" for e in loaded)
        assert not any(e.provenance=="measured_packet_trace" for e in loaded)

def _manual_events():
    return [CollectiveEvent("x","c","allreduce","Ring","Simple",i,"reduce_scatter",0,i,0,i+1,"h0",f"h{i+1}",1000,event_id=f"e{i}") for i in range(2)]

def test_endpoint_concurrency_nic_pcie_gpu_limits_and_startup_once():
    cfg=_cfg();r=replace(cfg.endpoint,nic_egress_bw_gbps=8,nic_ingress_bw_gbps=8,pcie_or_nvlink_tx_bw_gbps=8,pcie_or_nvlink_rx_bw_gbps=8,gpu_reduce_bw_gbps=8,max_concurrent_sends=1,max_concurrent_receives=1,max_active_channels=2,per_message_startup_us=2,per_channel_launch_us=3,optional_kernel_launch_us=0,per_step_sync_us=0)
    result=simulate_events(_manual_events(),r,1000.0)
    assert result.max_concurrent_sends<=1 and result.max_concurrent_receives<=1
    assert result.startup_overhead_us==2*2+2*3
    assert all(t.network_us>=1.0 for t in result.timings) and all(t.endpoint_us>=1.0 for t in result.timings)
    assert result.completion_time_us>=12.0

def test_event_simulator_small_critical_path():
    cfg=_cfg();r=replace(cfg.endpoint,nic_egress_bw_gbps=8,nic_ingress_bw_gbps=8,pcie_or_nvlink_tx_bw_gbps=8,pcie_or_nvlink_rx_bw_gbps=8,gpu_reduce_bw_gbps=1e12,gpu_copy_bw_gbps=1e12,max_concurrent_sends=1,max_concurrent_receives=1,max_active_channels=1,per_message_startup_us=0,per_channel_launch_us=0,optional_kernel_launch_us=0,per_step_sync_us=0)
    first=CollectiveEvent("x","c","allgather","Ring","Simple",0,"all_gather",0,0,0,1,"h0","h1",1000,event_id="a")
    second=replace(first,event_id="b",step=1,src_rank=1,dst_rank=2,src_host="h1",dst_host="h2",dependency_ids=("a",))
    result=simulate_events([first,second],r,1000.0);assert np.isclose(result.completion_time_us,2.0,rtol=1e-5)

def test_exact_drac_ports_and_cost_gate():
    cfg=_cfg();model=_network(cfg,4);demand=np.array([[0.,9.,0.,0.],[0.,0.,8.,0.],[0.,0.,0.,7.],[6.,0.,0.,0.]])
    allocation,_=allocate_drac_makespan_opt(demand,model);validate_general_units(allocation.connection_units,model)
    assert cost_based_gate(100,95,10,.5,.05)=="sym_ocs";assert cost_based_gate(100,80,10,.5,.05)=="drac_makespan_opt";assert cost_based_gate(100,50,0,.01,.05)=="sym_ocs"

def test_window_conservation_metrics_and_persistence():
    events=_events(reconstruct_ring_schedule(_record()));times,levels,persistence,flips=analyze_timescales(events,4,[1,2,4],{"gpus_per_server":2,"servers_per_tor":2,"tors_per_aggregation":2},["contiguous"],7)
    total=sum(e.bytes for e in events);assert np.isclose(sum(float(r["traffic_bytes"]) for r in times if r["window_steps"]==1),total)
    assert all(0<=float(r["omega"])<=1 for r in times if np.isfinite(float(r["omega"])))
    assert all(0<=float(r["direction_persistence"])<=1 for r in persistence if np.isfinite(float(r["direction_persistence"])))
    matrix=np.array([[0.,3.],[1.,0.]]);m=direction_metrics(matrix);assert m["absolute_directionality_bytes"]==2 and m["traffic_bytes"]==4 and m["omega"]==.5

def test_no_hardware_does_not_emit_fake_measurement():
    cfg=_cfg()
    with tempfile.TemporaryDirectory() as tmp:
        env=run_environment(replace(cfg,output_dir=tmp),Path(tmp));assert env["real_nccl_measurement_capable"] is False
        assert "MEASUREMENT_PENDING" in (Path(tmp)/"MEASUREMENT_INSTRUCTIONS.md").read_text(encoding="utf-8") or "cannot run" in (Path(tmp)/"MEASUREMENT_INSTRUCTIONS.md").read_text(encoding="utf-8")

def test_v3_plot_closes(monkeypatch):
    import matplotlib.pyplot as plt
    from drac_eval.rescue_v3_plotting import plot_calibration_pending
    calls=[];real=plt.close;monkeypatch.setattr(plt,"close",lambda fig=None:(calls.append(fig),real(fig))[1])
    with tempfile.TemporaryDirectory() as tmp:plot_calibration_pending(Path(tmp));assert calls and (Path(tmp)/"calibration_error.pdf").exists()
