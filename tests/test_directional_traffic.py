from __future__ import annotations

import math
import unittest
from copy import deepcopy
from pathlib import Path

from drac_eval.directional_traffic import (
    build_outputs,
    derive_pp,
    layers_for_pipeline_stage,
    load_derivation_config,
    protocol_ratios,
)


CONFIG = Path("configs/dp_pp_directional_traffic.json")


class DirectionalTrafficTest(unittest.TestCase):
    def test_layer_distribution_and_microbatch_window(self) -> None:
        self.assertEqual(
            [layers_for_pipeline_stage(126, 16, i) for i in range(16)],
            [8] * 14 + [7] * 2,
        )

    def test_legacy_ratio_recovers_old_dp_141x(self) -> None:
        config = load_derivation_config(CONFIG)
        ratios = protocol_ratios(config)
        recovered = (1.0 + ratios.payload_direction_overhead) / ratios.reverse_control
        self.assertTrue(math.isclose(recovered, 141.02975604578663, rel_tol=1e-12))

    def test_directional_outputs_are_conservative_and_ordered(self) -> None:
        config = load_derivation_config(CONFIG)
        rows, components = build_outputs(config, str(CONFIG))
        self.assertEqual([row["workload"] for row in rows], ["DP", "PP"])
        self.assertTrue(components)
        for row in rows:
            self.assertGreaterEqual(
                float(row["main_direction_bytes"]), float(row["opposite_direction_bytes"])
            )
            self.assertGreaterEqual(float(row["opposite_direction_bytes"]), 0)
            self.assertTrue(
                math.isclose(
                    float(row["main_direction_bytes"]),
                    float(row["main_payload_bytes"]) + float(row["main_control_bytes"]),
                    rel_tol=1e-12,
                )
            )
            self.assertEqual(
                row["main_source_endpoint"], row["opposite_destination_endpoint"]
            )
            self.assertEqual(
                row["main_destination_endpoint"], row["opposite_source_endpoint"]
            )
            expected_ratio = float(row["main_direction_bytes"]) / float(
                row["opposite_direction_bytes"]
            )
            self.assertTrue(math.isclose(float(row["ratio"]), expected_ratio, rel_tol=1e-12))
        pp = rows[1]
        self.assertIs(pp["directions_tied"], True)
        self.assertTrue(
            math.isclose(
                float(pp["main_direction_bytes"]),
                float(pp["opposite_direction_bytes"]),
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        )
        self.assertTrue(all(float(component["bytes"]) >= 0 for component in components))

        for row in rows:
            for prefix in ("main", "opposite"):
                source = row[f"{prefix}_source_endpoint"]
                destination = row[f"{prefix}_destination_endpoint"]
                component_sum = sum(
                    float(component["bytes"])
                    for component in components
                    if component["workload"] == row["workload"]
                    and component["source_endpoint"] == source
                    and component["destination_endpoint"] == destination
                )
                self.assertTrue(
                    math.isclose(
                        component_sum,
                        float(row[f"{prefix}_direction_bytes"]),
                        rel_tol=1e-12,
                    )
                )

    def test_pp_forward_and_backward_are_independently_derived(self) -> None:
        config = load_derivation_config(CONFIG)
        changed = deepcopy(config)
        changed["pp_tensors"]["backward"]["element_bytes"] = 4
        directions, _components, metadata = derive_pp(changed, protocol_ratios(changed))
        self.assertEqual(metadata["backward_bytes"], 2 * metadata["forward_bytes"])
        self.assertGreater(directions[1].total_bytes, directions[0].total_bytes)


if __name__ == "__main__":
    unittest.main()
