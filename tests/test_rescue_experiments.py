from __future__ import annotations

import tempfile
import unittest
import csv
from pathlib import Path

import numpy as np

from drac_eval.config import NetworkConfig, WorkloadConfig
from drac_eval.metrics import compute_segment_metrics
from drac_eval.rescue_allocation import allocate_rescue_method, validate_units
from drac_eval.rescue_config import RescueConfig
from drac_eval.rescue_runner import run_rescue_experiments
from drac_eval.rescue_traffic import (
    aggregate_matrix,
    apply_collective_model,
    build_mapping,
    directional_opportunity,
)


class RescueExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = np.array(
            [[0.0, 9.0, 2.0, 0.0], [1.0, 0.0, 4.0, 1.0], [3.0, 2.0, 0.0, 8.0], [0.0, 2.0, 1.0, 0.0]]
        )

    def test_aggregation_and_omega(self) -> None:
        mapping = build_mapping(4, 2, "contiguous", 7)
        aggregate, cross = aggregate_matrix(self.matrix, mapping)
        self.assertAlmostEqual(float(aggregate.sum()), cross)
        omega = directional_opportunity(aggregate)
        self.assertGreaterEqual(omega, 0.0)
        self.assertLessEqual(omega, 1.0)

    def test_balancing_conservation(self) -> None:
        balanced = apply_collective_model(self.matrix, "bidirectional_balanced", 7)
        oracle = apply_collective_model(self.matrix, "pairwise_balancing_oracle", 7)
        self.assertTrue(np.allclose(balanced.sum(axis=1), self.matrix.sum(axis=1)))
        self.assertTrue(np.allclose(balanced.sum(axis=0), self.matrix.sum(axis=0)))
        self.assertAlmostEqual(float(oracle.sum()), float(self.matrix.sum()))
        self.assertTrue(np.allclose(oracle, oracle.T))

    def test_discrete_optimum_and_constraints(self) -> None:
        net = NetworkConfig(base_bw_gbps=10.0, ocs_unit_bw_gbps=10.0, per_node_port_budget=2, total_ocs_links=6)
        allocs = {method: allocate_rescue_method(method, self.matrix, net)[0] for method in ["sqrt_sum_delay", "proportional_makespan", "discrete_makespan_opt"]}
        for alloc in allocs.values():
            validate_units(alloc.connection_units, net)
        times = {method: compute_segment_metrics(self.matrix, alloc, net).completion_time_ms for method, alloc in allocs.items()}
        self.assertLessEqual(times["discrete_makespan_opt"], min(times["sqrt_sum_delay"], times["proportional_makespan"]) + 1e-9)

    def test_smoke_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = RescueConfig(
                output_dir=str(Path(tmp) / "rescue"),
                endpoint_count=8,
                seeds=[7],
                mapping_strategies=["contiguous"],
                port_budgets=[2],
                omega_thresholds=[0.1],
                network=NetworkConfig(base_bw_gbps=20.0, ocs_unit_bw_gbps=20.0, per_node_port_budget=2, total_ocs_links=8),
                workloads=[WorkloadConfig(name="tp", kind="tp", segment_count=1, tp_group_size=4)],
            )
            outputs = run_rescue_experiments(cfg, smoke=True)
            root = outputs["root"]
            self.assertTrue((root / "aggregation_retention" / "aggregation_segment_metrics.csv").exists())
            self.assertTrue((root / "collective_balancing" / "collective_balancing_performance.csv").exists())
            self.assertTrue((root / "makespan_objective" / "makespan_method_comparison.csv").exists())
            self.assertTrue(any(root.rglob("*.pdf")))
            self.assertTrue(outputs["report"].exists())
            with (root / "collective_balancing" / "collective_balancing_performance.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            gated = [r for r in rows if r["algorithm"] == "drac_gated" and r.get("gated_to_sym_ocs") == "True"]
            for row in gated:
                sym = next(r for r in rows if r["algorithm"] == "sym_ocs" and r["workload"] == row["workload"] and r["seed"] == row["seed"] and r["segment_idx"] == row["segment_idx"] and r["collective_model"] == row["collective_model"] and r["port_budget"] == row["port_budget"])
                self.assertAlmostEqual(float(row["communication_time_ms"]), float(sym["communication_time_ms"]))


if __name__ == "__main__":
    unittest.main()
