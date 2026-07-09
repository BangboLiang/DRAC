"""
llama3_comm - Modular package for Llama 3 communication modeling.

This package provides tools for calculating/simulating communication time for different
collective algorithms and reconfiguration plans during LLM training.
"""

from .config import ModelConfig, ParallelConfig, SystemConfig
from .degree import (
    PeerSet,
    build_domain_stream,
    calc_batches,
    critical_degrees,
    exposed_boundary_ms,
    op_peer_stream,
)
from .execution import (
    BWSegmentPlan,
    CommNode,
    LinkSegmentPlan,
    SegmentPlan,
    TraceEvent,
    exposed_reconfig_ms_for_segment_start,
)
from .plotting import (
    try_plot_rows_png,
    try_plot_trace_png,
    write_trace_csv,
    write_trace_json,
)
from .solvers import (
    fast_preplanned_partition,
    preplanned_dp_partition,
    solve_best_link_only_plan,
    solve_best_link_plan_for_bw_segment,
    solve_min_delay_bw_split,
)
from .traffic import (
    GiB,
    MiB,
    effective_bandwidth_GBps,
    estimate_time_ms,
    get_efficiency,
    llama3_405b_payloads,
    llama3_megatron_payloads,
    quantize_share_dict,
)
from .trace_ir import (
    CommTriggerRef,
    TraceBundle,
    TraceOp,
    TraceRankGraph,
    role_key_for_rank,
)
from .trace_ingest import (
    load_trace_bundle,
    select_representative_rank,
    stable_topological_ops,
)
from .trace_to_comm import (
    build_comm_nodes_from_rank_trace,
    collective_profile_choices,
    infer_collective_op,
)
from .rank_lift import collect_domain_edge_templates, edge_bandwidth_state_for_share
from .astra_emit import (
    build_dag_reconfiguration_events,
    build_rank_scoped_dag_reconfiguration_events,
    densify_edge_states,
    render_mutable_network_yaml,
    write_astra_plan_bundle,
)
from .peer_plan import (
    AbstractPeerPlanEvent,
    abstract_peer_edges_for_trace_op,
    build_abstract_peer_plan,
    instantiate_concrete_edge_state,
)
from .role_planning import (
    RoleTemplatePlan,
    build_dynamic_events,
    build_role_templates,
    plan_segments,
    segments_to_summary,
)


__all__ = [
    # Config
    "SystemConfig",
    "ModelConfig",
    "ParallelConfig",
    # Traffic
    "MiB",
    "GiB",
    "quantize_share_dict",
    "llama3_405b_payloads",
    "llama3_megatron_payloads",
    "get_efficiency",
    "estimate_time_ms",
    "effective_bandwidth_GBps",
    # Degree
    "PeerSet",
    "op_peer_stream",
    "build_domain_stream",
    "calc_batches",
    "critical_degrees",
    "exposed_boundary_ms",
    # Execution
    "CommNode",
    "SegmentPlan",
    "LinkSegmentPlan",
    "BWSegmentPlan",
    "TraceEvent",
    "exposed_reconfig_ms_for_segment_start",
    # Solvers
    "fast_preplanned_partition",
    "solve_min_delay_bw_split",
    "solve_best_link_only_plan",
    "solve_best_link_plan_for_bw_segment",
    "preplanned_dp_partition",
    # Plotting
    "write_trace_json",
    "write_trace_csv",
    "try_plot_rows_png",
    "try_plot_trace_png",
    # Trace-driven planning
    "TraceOp",
    "TraceRankGraph",
    "TraceBundle",
    "CommTriggerRef",
    "role_key_for_rank",
    "load_trace_bundle",
    "select_representative_rank",
    "stable_topological_ops",
    "collective_profile_choices",
    "infer_collective_op",
    "build_comm_nodes_from_rank_trace",
    "collect_domain_edge_templates",
    "edge_bandwidth_state_for_share",
    "render_mutable_network_yaml",
    "build_dag_reconfiguration_events",
    "build_rank_scoped_dag_reconfiguration_events",
    "densify_edge_states",
    "write_astra_plan_bundle",
    "AbstractPeerPlanEvent",
    "abstract_peer_edges_for_trace_op",
    "build_abstract_peer_plan",
    "instantiate_concrete_edge_state",
    "RoleTemplatePlan",
    "plan_segments",
    "segments_to_summary",
    "build_role_templates",
    "build_dynamic_events",
    # Discrete event simulator
    "ComputeTask",
    "CommTask",
    "SimConfig",
    "SimTraceEvent",
    "SimResult",
    "simulate",
    # Task graph helpers
    "StepTaskGraphParams",
    "build_llama3_step_tasks",
    "default_dp_bucket_bytes",
]
