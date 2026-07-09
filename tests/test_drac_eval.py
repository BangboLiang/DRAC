from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from drac_eval import run_experiments
from drac_eval.allocation import allocate_for_algorithm
from drac_eval.config import ExperimentConfig, NetworkConfig, SweepConfig, WorkloadConfig
from drac_eval.traffic import load_or_generate_workload, validate_demand_matrix


class DracEvalTests(unittest.TestCase):
    def test_validate_demand_matrix(self) -> None:
        validate_demand_matrix(np.zeros((4, 4), dtype=float))
        with self.assertRaises(ValueError):
            validate_demand_matrix(np.zeros((4, 3), dtype=float))

    def test_budget_constraints(self) -> None:
        demand = np.array(
            [
                [0.0, 4.0, 0.0, 0.0],
                [1.0, 0.0, 3.0, 0.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        net = NetworkConfig(
            base_bw_gbps=10.0,
            ocs_unit_bw_gbps=5.0,
            per_node_port_budget=2,
            total_ocs_links=6,
            reconfig_delay_ms=0.1,
        )
        alloc = allocate_for_algorithm("drac", demand, net)
        self.assertTrue(np.all(alloc.connection_units.sum(axis=1) <= 2))
        self.assertTrue(np.all(alloc.connection_units.sum(axis=0) <= 2))
        self.assertLessEqual(int(alloc.connection_units.sum()), 6)

    def test_symmetric_baseline_is_symmetric(self) -> None:
        demand = np.array(
            [[0.0, 6.0, 1.0], [2.0, 0.0, 3.0], [4.0, 1.0, 0.0]], dtype=float
        )
        net = NetworkConfig(total_ocs_links=8, per_node_port_budget=3, ocs_unit_bw_gbps=5.0)
        alloc = allocate_for_algorithm("sym_ocs", demand, net)
        self.assertTrue(np.allclose(alloc.realized_overlay, alloc.realized_overlay.T))

    def test_drac_can_realize_asymmetry(self) -> None:
        demand = np.array(
            [[0.0, 9.0, 0.0], [1.0, 0.0, 2.0], [0.0, 1.0, 0.0]], dtype=float
        )
        net = NetworkConfig(total_ocs_links=8, per_node_port_budget=4, ocs_unit_bw_gbps=5.0)
        alloc = allocate_for_algorithm("drac", demand, net)
        self.assertGreater(float(alloc.realized_overlay[0, 1]), float(alloc.realized_overlay[1, 0]))

    def test_smoke_runner_outputs_csv_and_figure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ExperimentConfig(
                name="smoke",
                output_dir=str(Path(tmp) / "results"),
                workloads=[
                    WorkloadConfig(name="tp", kind="tp", segment_count=2),
                    WorkloadConfig(name="dp", kind="dp", segment_count=2),
                ],
                sweeps=SweepConfig(
                    cluster_sizes=[8],
                    asymmetry_levels=[4.0],
                    port_budgets=[2],
                    total_ocs_links=[8],
                    reconfig_delays_ms=[0.2],
                ),
            )
            outputs = run_experiments(cfg)
            self.assertTrue(outputs["raw_csv"].exists())
            self.assertTrue(outputs["summary_csv"].exists())
            self.assertTrue(any(outputs["figure_dir"].glob("*.png")))

    def test_workload_generator_matches_cluster_size(self) -> None:
        workload = WorkloadConfig(name="mixed", kind="mixed", segment_count=3)
        segments = load_or_generate_workload(workload, 8, 4.0, 7)
        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0].matrix.shape, (8, 8))


if __name__ == "__main__":
    unittest.main()
