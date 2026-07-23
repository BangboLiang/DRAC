from __future__ import annotations

import csv
import pytest

from drac_eval.directional_traffic import load_dp_directional_demand, write_directional_traffic_csvs
from drac_eval.symmetric_provisioning import (
    dense_sensitivity_rows,
    detect_allocation_regions,
    direction_aware_provisioning,
    evaluate_allocation,
    feasible_allocations,
    gbps_to_gb_per_second,
    symmetric_provisioning,
)


EXPECTED_MAIN_GB = 20.258608051
EXPECTED_OPPOSITE_GB = 0.143647756


def test_default_dp_case() -> None:
    sym = symmetric_provisioning(EXPECTED_MAIN_GB, EXPECTED_OPPOSITE_GB)
    aware = direction_aware_provisioning(EXPECTED_MAIN_GB, EXPECTED_OPPOSITE_GB)
    assert (sym.main_channels, sym.opposite_channels) == (4, 4)
    assert (aware.main_channels, aware.opposite_channels) == (7, 1)
    assert sym.completion_time_ms == pytest.approx(810.344, abs=0.1)
    assert aware.completion_time_ms == pytest.approx(463.054, abs=0.1)
    assert 1.0 - aware.completion_time_ms / sym.completion_time_ms == pytest.approx(0.428571, abs=1e-6)
    assert sym.total_idle_fraction == pytest.approx(0.496455, abs=1e-6)
    assert aware.total_idle_fraction == pytest.approx(0.118796, abs=1e-6)


def test_equal_demands_degenerate_to_symmetric() -> None:
    sym = symmetric_provisioning(10.0, 10.0)
    aware = direction_aware_provisioning(10.0, 10.0)
    assert (aware.main_channels, aware.opposite_channels) == (4, 4)
    assert aware.completion_time_ms == pytest.approx(sym.completion_time_ms)
    assert 1.0 - aware.completion_time_ms / sym.completion_time_ms == pytest.approx(0.0)


def test_high_skew_uses_seven_one_and_reaches_budget_plateau() -> None:
    share = 0.875
    sym = symmetric_provisioning(share, 1.0 - share)
    aware = direction_aware_provisioning(share, 1.0 - share)
    reduction = 1.0 - aware.completion_time_ms / sym.completion_time_ms
    assert (aware.main_channels, aware.opposite_channels) == (7, 1)
    assert reduction == pytest.approx(3.0 / 7.0)


def test_nonzero_opposite_demand_never_gets_zero_channels() -> None:
    aware = direction_aware_provisioning(100.0, 1e-9)
    assert aware.opposite_channels >= 1


def test_all_enumerated_allocations_obey_integer_budget() -> None:
    allocations = list(feasible_allocations(7, 1))
    assert allocations
    assert all(isinstance(a, int) and isinstance(b, int) and a + b <= 7 for a, b in allocations)


def test_gbps_unit_conversion() -> None:
    assert gbps_to_gb_per_second(50.0) == pytest.approx(6.25)


def test_solver_is_exhaustively_optimal_for_small_budget() -> None:
    main_gb, opposite_gb, budget = 3.7, 1.2, 5
    result = direction_aware_provisioning(main_gb, opposite_gb, total_channels=budget)
    brute = [
        evaluate_allocation(main_gb, opposite_gb, budget, a, b, 50.0, "brute")
        for a, b in feasible_allocations(budget, 1)
    ]
    assert result.completion_time_ms == pytest.approx(min(item.completion_time_ms for item in brute))


def test_csv_input_loads_dp_main_and_opposite(tmp_path) -> None:
    paths = write_directional_traffic_csvs(tmp_path)
    main_gb, opposite_gb = load_dp_directional_demand(paths["directional_traffic"])
    assert main_gb == pytest.approx(20.2584, rel=2e-5)
    assert opposite_gb == pytest.approx(0.143646, rel=2e-5)
    with paths["directional_traffic"].open(newline="", encoding="utf-8") as handle:
        dp = next(row for row in csv.DictReader(handle) if row["workload"] == "DP")
    assert float(dp["main_direction_bytes"]) / 1e9 == pytest.approx(main_gb)
    assert float(dp["opposite_direction_bytes"]) / 1e9 == pytest.approx(opposite_gb)


def test_dense_scan_is_feasible_and_never_harms_completion() -> None:
    rows = dense_sensitivity_rows(20.0, samples=500)
    assert len(rows) == 500
    for row in rows:
        main_channels = int(row["direction_aware_main_channels"])
        opposite_channels = int(row["direction_aware_opposite_channels"])
        assert 1 <= main_channels <= 7
        assert main_channels + opposite_channels == 8
        assert float(row["completion_time_reduction"]) >= -1e-10
        assert float(row["direction_aware_completion_time_ms"]) <= float(row["symmetric_completion_time_ms"]) + 1e-10


def test_dense_scan_detects_all_expected_allocation_regions() -> None:
    rows = dense_sensitivity_rows(20.0, samples=500)
    regions = detect_allocation_regions(rows)
    assert [region["allocation_region"] for region in regions] == ["4/4", "5/3", "6/2", "7/1"]
