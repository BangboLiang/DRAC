from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from drac_eval.rescue_schedule import (
    build_executable_ring_schedule,
    schedule_step_matrices,
    validate_executable_ring_schedule,
)
from drac_eval.rescue_v2_config import load_rescue_v2_config
from drac_eval.rescue_v2_network import (
    aggregate_network_model,
    allocate_drac_makespan_opt,
    brute_force_makespan,
    endpoint_network_model,
    validate_general_units,
)
from drac_eval.rescue_v2_runner import (
    _aggregate_demand,
    _direction_stats,
    aggregate_gain_ratio,
    benefit_retention,
    collective_replacement_risk,
    run_pp_discrepancy,
)
from drac_eval.rescue_traffic import build_mapping


def _cfg():
    return load_rescue_v2_config("configs/rescue_experiments_v2.json")


def test_demand_and_resource_aggregation_conservation():
    cfg = _cfg()
    n = 8
    matrix = np.arange(n * n, dtype=float).reshape(n, n)
    np.fill_diagonal(matrix, 0.0)
    mapping = build_mapping(n, 2, "contiguous", 7)
    aggregate = _aggregate_demand(matrix, mapping)
    expected = sum(matrix[u, v] for u in range(n) for v in range(n) if mapping[u] != mapping[v])
    assert np.isclose(aggregate.sum(), expected)
    endpoint = endpoint_network_model(cfg, n, 3)
    model = aggregate_network_model(endpoint, mapping, "resource_equivalent", "server", cfg)
    expected_base = sum(endpoint.base_capacity_gbps[u, v] for u in range(n) for v in range(n) if mapping[u] != mapping[v])
    assert np.isclose(model.base_capacity_gbps.sum(), expected_base)
    assert model.out_port_budget.sum() == endpoint.out_port_budget.sum()
    assert model.in_port_budget.sum() == endpoint.in_port_budget.sum()
    assert model.total_ocs_links == endpoint.total_ocs_links


def test_absolute_retention_and_boundary_fraction_bounds():
    matrix = np.array([[0., 9., 1., 0.], [2., 0., 3., 1.], [4., 1., 0., 8.], [0., 2., 1., 0.]])
    mapping = build_mapping(4, 2, "contiguous", 7)
    aggregate = _aggregate_demand(matrix, mapping)
    a0, v0, _ = _direction_stats(matrix)
    a, v, _ = _direction_stats(aggregate)
    assert a <= a0 + 1e-12
    assert 0 <= a / a0 <= 1
    assert 0 <= v / v0 <= 1


def test_executable_ring_4_and_8_rank_semantics_and_bytes():
    for n in [4, 8]:
        uni = build_executable_ring_schedule(n, 4096, "allreduce", False, n)
        bi = build_executable_ring_schedule(n, 4096, "allreduce", True, n)
        assert validate_executable_ring_schedule(uni)["semantic_valid"]
        assert validate_executable_ring_schedule(bi)["dependency_valid"]
        assert np.isclose(uni.total_transmitted_bytes, bi.total_transmitted_bytes)
        assert len(schedule_step_matrices(uni)) == 2 * (n - 1)
        assert len(schedule_step_matrices(bi)) == 2 * (n - 1)


def test_benefit_retention_formula_and_near_zero():
    assert np.isclose(benefit_retention(0.3382, 0.3068, 1e-8), 0.3068 / 0.3382)
    assert np.isnan(benefit_retention(1e-12, 0.1, 1e-8))
    rows = [{"communication_bytes": 10., "original_sym_time_ms": 5., "original_gain": .2, "balanced_gain": .1}, {"communication_bytes": 20., "original_sym_time_ms": 10., "original_gain": .4, "balanced_gain": .3}]
    assert np.isclose(aggregate_gain_ratio(rows, "communication_bytes", 1e-8), 7 / 10)


def test_oracle_not_part_of_replacement_risk():
    assert collective_replacement_risk(0.1) == "HIGH"
    assert collective_replacement_risk(0.9) == "LOW"
    # The API accepts only executable retention; no Oracle argument exists.
    assert collective_replacement_risk.__code__.co_argcount == 2


def test_exact_solver_matches_brute_force_and_constraints():
    cfg = _cfg()
    cfg.endpoint_base_capacity_gbps = 10.0
    cfg.ocs_unit_bw_gbps = 10.0
    cfg.global_ocs_budget = 2
    demand = np.array([[0., 9., 1.], [2., 0., 6.], [4., 3., 0.]])
    model = endpoint_network_model(cfg, 3, 1)
    exact, _ = allocate_drac_makespan_opt(demand, model)
    brute_time, _ = brute_force_makespan(demand, model)
    validate_general_units(exact.connection_units, model)
    from drac_eval.metrics import _completion_time_ms
    assert np.isclose(_completion_time_ms(demand, exact.total_bandwidth), brute_time)


def test_pp_discrepancy_reproduces_paper_and_rescue_regimes():
    cfg = _cfg()
    with tempfile.TemporaryDirectory() as tmp:
        rows = run_pp_discrepancy(cfg, Path(tmp))
        assert float(rows[0]["gain"]) < 0.01
        assert float(rows[1]["gain"]) > 0.05
        assert (Path(tmp) / "PP_DISCREPANCY.md").exists()


def test_all_plot_saves_close_figures(monkeypatch):
    import matplotlib.pyplot as plt
    from drac_eval.rescue_v2_plotting import plot_makespan_v2

    calls = []
    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda fig=None: (calls.append(fig), real_close(fig))[1])
    with tempfile.TemporaryDirectory() as tmp:
        plot_makespan_v2([{"rank_count": 8, "runtime_ms": 1.0}, {"rank_count": 16, "runtime_ms": 2.0}], Path(tmp))
        assert calls
        assert (Path(tmp) / "makespan_runtime_scaling.pdf").exists()
