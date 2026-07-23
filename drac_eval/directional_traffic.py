<<<<<<< HEAD
"""Canonical directional-traffic data used by small paper examples.

The raw counters below are the physical per-port byte counters already used by
``figures/rx_tx_port_traffic_total_only_8layer_sqrt.py``.  Keeping them here
lets data-generation and plotting code share one source instead of embedding
another LLaMA communication model.

Functions in this module were introduced by
``generate_dp_pp_directional_traffic.py``.
=======
"""Auditable DP/PP ordered-direction traffic derivation.

This module supports ``generate_dp_pp_directional_traffic.py``.  It deliberately
keeps workload-byte derivation separate from plotting and labels legacy embedded
counter ratios as provisional when their raw measurement artifact is unavailable.
>>>>>>> 6839052da73682436a4eeed00ae6ac55603f3e49
"""

from __future__ import annotations

import csv
<<<<<<< HEAD
from pathlib import Path
from typing import Dict, Iterable, List


RAW_PHYSICAL_COUNTERS: Dict[str, Dict[str, int]] = {
    "TP": {
        "opposite_direction_bytes_raw": 361_013_422,
        "main_direction_bytes_raw": 50_154_823_250,
        "iterations": 200,
        "representative_layer_multiplier": 1,
    },
    "DP": {
        "opposite_direction_bytes_raw": 598_526_166,
        "main_direction_bytes_raw": 84_409_999_178,
        "iterations": 100,
        "representative_layer_multiplier": 24,
    },
}


def directional_traffic_rows() -> List[Dict[str, object]]:
    """Return representative ordered-direction demands in bytes and decimal GB."""
    rows: List[Dict[str, object]] = []
    for workload, counters in RAW_PHYSICAL_COUNTERS.items():
        iterations = counters["iterations"]
        multiplier = counters["representative_layer_multiplier"]
        main_bytes = counters["main_direction_bytes_raw"] / iterations * multiplier
        opposite_bytes = counters["opposite_direction_bytes_raw"] / iterations * multiplier
        rows.append(
            {
                "workload": workload,
                "main_direction_bytes": f"{main_bytes:.6f}",
                "opposite_direction_bytes": f"{opposite_bytes:.6f}",
                "main_direction_gb": f"{main_bytes / 1e9:.12f}",
                "opposite_direction_gb": f"{opposite_bytes / 1e9:.12f}",
                "gb_definition": "1 GB = 1e9 bytes",
                "source": "physical per-port counters from figures/rx_tx_port_traffic_total_only_8layer_sqrt.py",
                "raw_iterations": iterations,
                "representative_layer_multiplier": multiplier,
            }
        )
    return rows


def write_directional_traffic_csvs(output_dir: str | Path) -> Dict[str, Path]:
    """Write the full and compact preview directional-traffic CSV files."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = directional_traffic_rows()
    full_path = output / "directional_traffic.csv"
    preview_path = output / "directional_traffic_preview.csv"
    _write_rows(full_path, rows)
    preview_fields = [
        "workload",
        "main_direction_bytes",
        "opposite_direction_bytes",
        "main_direction_gb",
        "opposite_direction_gb",
        "gb_definition",
    ]
    _write_rows(preview_path, [{key: row[key] for key in preview_fields} for row in rows])
    return {"directional_traffic": full_path, "preview": preview_path}


def _write_rows(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError("directional traffic output must contain at least one row")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def load_dp_directional_demand(path: str | Path) -> tuple[float, float]:
    """Load DP main/opposite demands from a generated CSV, returning decimal GB."""
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row.get("workload", "").strip().upper() == "DP"]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one DP row in {csv_path}, found {len(matches)}")
    row = matches[0]
    if row.get("gb_definition") != "1 GB = 1e9 bytes":
        raise ValueError("directional traffic CSV does not declare decimal-GB units")
    main_bytes = float(row["main_direction_bytes"])
    opposite_bytes = float(row["opposite_direction_bytes"])
    if main_bytes <= 0 or opposite_bytes <= 0:
        raise ValueError("both DP ordered directions must have positive demand")
    return main_bytes / 1e9, opposite_bytes / 1e9
=======
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from llama3_comm.config import ModelConfig, ParallelConfig
from llama3_comm.traffic import llama3_megatron_payloads


@dataclass(frozen=True)
class ProtocolRatios:
    """Per-payload ratios recovered from the legacy embedded aggregate counters."""

    rocev2: float
    phy: float
    cts: float
    reverse_other: float
    reverse_unclassified: float

    @property
    def payload_direction_overhead(self) -> float:
        return self.rocev2 + self.phy

    @property
    def reverse_control(self) -> float:
        return self.cts + self.reverse_other + self.reverse_unclassified


@dataclass(frozen=True)
class DirectionTotal:
    """One concrete ordered direction and its payload/control byte totals."""

    name: str
    source: str
    destination: str
    payload_bytes: float
    control_bytes: float

    @property
    def total_bytes(self) -> float:
        return self.payload_bytes + self.control_bytes


def load_derivation_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate the standalone derivation configuration."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    for key in ("model", "parallelism", "schedule", "legacy_measurement"):
        if key not in raw:
            raise ValueError(f"missing required config object: {key}")
    if not str(raw.get("derivation_version", "")).strip():
        raise ValueError("derivation_version must be non-empty")
    return raw


def protocol_ratios(config: dict[str, Any]) -> ProtocolRatios:
    """Convert legacy aggregate component bytes into unit-payload ratios."""

    measurement = config["legacy_measurement"]
    forward = measurement["payload_direction"]
    reverse = measurement["reverse_direction"]
    payload = float(forward["payload_bytes"])
    if payload <= 0:
        raise ValueError("legacy payload_bytes must be positive")
    return ProtocolRatios(
        rocev2=float(forward["rocev2_overhead_bytes"]) / payload,
        phy=float(forward["phy_overhead_bytes"]) / payload,
        cts=float(reverse["cts_bytes"]) / payload,
        reverse_other=float(reverse["other_rocev2_control_bytes"]) / payload,
        reverse_unclassified=float(reverse["unclassified_control_bytes"]) / payload,
    )


def _model_and_parallel(config: dict[str, Any]) -> tuple[ModelConfig, ParallelConfig]:
    model = config["model"]
    parallel = config["parallelism"]
    mod = ModelConfig(
        layers=int(model["layers"]),
        hidden=int(model["hidden_size"]),
        seq=int(model["sequence_length"]),
        head_dim=int(model["head_dim"]),
        kv_dim=int(model["kv_dim"]),
        ffn_hidden=int(model["ffn_hidden"]),
        total_params=float(model["parameter_count"]),
        bytes_per_act=int(model["activation_bytes"]),
        bytes_per_param=int(model["parameter_bytes"]),
        bytes_per_grad=int(model["gradient_bytes"]),
    )
    par = ParallelConfig(
        tp=int(parallel["tp_group_size"]),
        pp=int(parallel["pipeline_stage_count"]),
        dp=int(parallel["dp_group_size"]),
        global_batch_seqs=int(parallel["global_batch_sequences"]),
        microbatch_seqs=int(parallel["microbatch_sequences"]),
    )
    configured_microbatches = int(parallel["microbatch_count"])
    if par.microbatches_per_step != configured_microbatches:
        raise ValueError(
            "microbatch_count disagrees with global_batch/(TP*PP*microbatch_sequences): "
            f"{configured_microbatches} != {par.microbatches_per_step}"
        )
    return mod, par


def layers_for_pipeline_stage(total_layers: int, stage_count: int, stage_index: int) -> int:
    """Return balanced contiguous layer ownership for a representative PP stage."""

    if total_layers <= 0 or stage_count <= 0:
        raise ValueError("total_layers and stage_count must be positive")
    if not 0 <= stage_index < stage_count:
        raise ValueError("representative pipeline stage index is out of range")
    base, remainder = divmod(total_layers, stage_count)
    return base + (1 if stage_index < remainder else 0)


def _ordered_pair(raw_pair: Any, field: str) -> tuple[str, str]:
    if not isinstance(raw_pair, list) or len(raw_pair) != 2:
        raise ValueError(f"{field} must contain exactly two endpoint names")
    source, destination = str(raw_pair[0]), str(raw_pair[1])
    if not source or not destination or source == destination:
        raise ValueError(f"{field} endpoints must be distinct and non-empty")
    return source, destination


def derive_dp(
    config: dict[str, Any], ratios: ProtocolRatios
) -> tuple[list[DirectionTotal], list[dict[str, Any]], dict[str, Any]]:
    """Derive one representative ring-neighbor pair over a full training iteration."""

    mod, par = _model_and_parallel(config)
    parallel = config["parallelism"]
    stage_index = int(parallel["representative_pipeline_stage_index"])
    stage_layers = layers_for_pipeline_stage(mod.layers, par.pp, stage_index)
    _, _, layer_param_bf16, layer_grad_fp32, architecture_layer_params = (
        llama3_megatron_payloads(mod, par)
    )
    ring_fraction = float(par.dp - 1) / float(par.dp)
    rs_payload = float(layer_grad_fp32) * stage_layers * ring_fraction
    ag_payload = float(layer_param_bf16) * stage_layers * ring_fraction
    payload = rs_payload + ag_payload

    source, destination = _ordered_pair(
        config["schedule"]["dp_ring_ordered_pair"], "dp_ring_ordered_pair"
    )
    main_control = payload * ratios.payload_direction_overhead
    opposite_control = payload * ratios.reverse_control
    directions = [
        DirectionTotal(
            "ring payload direction (RS + AG)", source, destination, payload, main_control
        ),
        DirectionTotal(
            "reverse ring control direction", destination, source, 0.0, opposite_control
        ),
    ]
    components = [
        _component("DP", source, destination, "reduce_scatter_gradient_payload_fp32", rs_payload, "schedule-derived"),
        _component("DP", source, destination, "all_gather_parameter_payload_bf16", ag_payload, "schedule-derived"),
        _component("DP", source, destination, "rocev2_overhead", payload * ratios.rocev2, "legacy-measurement-ratio-scaled"),
        _component("DP", source, destination, "phy_overhead", payload * ratios.phy, "legacy-measurement-ratio-scaled"),
        _component("DP", destination, source, "cts", payload * ratios.cts, "legacy-measurement-ratio-scaled"),
        _component("DP", destination, source, "other_rocev2_control", payload * ratios.reverse_other, "legacy-measurement-ratio-scaled"),
        _component("DP", destination, source, "unclassified_reverse_control", payload * ratios.reverse_unclassified, "legacy-measurement-ratio-scaled"),
    ]
    metadata = {
        "stage_layers": stage_layers,
        "architecture_layer_parameter_count": architecture_layer_params,
        "ring_fraction": ring_fraction,
        "interval_definition": (
            f"one full training iteration; representative PP stage {stage_index} owns "
            f"{stage_layers} layers; Ring RS + AG over DP={par.dp}"
        ),
    }
    return directions, components, metadata


def derive_pp(
    config: dict[str, Any], ratios: ProtocolRatios
) -> tuple[list[DirectionTotal], list[dict[str, Any]], dict[str, Any]]:
    """Derive an adjacent PP pair from independent forward/backward tensor specs."""

    mod, par = _model_and_parallel(config)
    microbatches = int(config["parallelism"]["microbatch_count"])
    tensor_specs = config.get("pp_tensors", {})
    if set(tensor_specs) != {"forward", "backward"}:
        raise ValueError("pp_tensors must independently define forward and backward")
    tensor_elements = (mod.seq / par.tp) * mod.hidden
    if not float(tensor_elements).is_integer():
        raise ValueError("PP boundary tensor element count must be integral")
    forward_element_bytes = int(tensor_specs["forward"]["element_bytes"])
    backward_element_bytes = int(tensor_specs["backward"]["element_bytes"])
    if forward_element_bytes <= 0 or backward_element_bytes <= 0:
        raise ValueError("PP tensor element widths must be positive")
    forward_per_microbatch = float(tensor_elements) * forward_element_bytes
    backward_per_microbatch = float(tensor_elements) * backward_element_bytes
    forward_payload = forward_per_microbatch * microbatches
    backward_payload = backward_per_microbatch * microbatches
    source, destination = _ordered_pair(
        config["schedule"]["pp_ordered_pair"], "pp_ordered_pair"
    )

    # Each ordered direction carries its own tensor and same-direction framing overhead.
    # Reverse controls generated by the other tensor are added to their actual direction.
    forward_direction_control = (
        forward_payload * ratios.payload_direction_overhead
        + backward_payload * ratios.reverse_control
    )
    backward_direction_control = (
        backward_payload * ratios.payload_direction_overhead
        + forward_payload * ratios.reverse_control
    )
    directions = [
        DirectionTotal(
            "forward activation direction",
            source,
            destination,
            forward_payload,
            forward_direction_control,
        ),
        DirectionTotal(
            "backward activation-gradient direction",
            destination,
            source,
            backward_payload,
            backward_direction_control,
        ),
    ]
    components = [
        _component("PP", source, destination, "forward_activation_payload", forward_payload, "workload-derived"),
        _component("PP", source, destination, "rocev2_overhead_for_forward_payload", forward_payload * ratios.rocev2, "legacy-measurement-ratio-scaled"),
        _component("PP", source, destination, "phy_overhead_for_forward_payload", forward_payload * ratios.phy, "legacy-measurement-ratio-scaled"),
        _component("PP", source, destination, "cts_for_backward_payload", backward_payload * ratios.cts, "legacy-measurement-ratio-scaled"),
        _component("PP", source, destination, "other_rocev2_control_for_backward_payload", backward_payload * ratios.reverse_other, "legacy-measurement-ratio-scaled"),
        _component("PP", source, destination, "unclassified_control_for_backward_payload", backward_payload * ratios.reverse_unclassified, "legacy-measurement-ratio-scaled"),
        _component("PP", destination, source, "backward_activation_gradient_payload", backward_payload, "workload-derived"),
        _component("PP", destination, source, "rocev2_overhead_for_backward_payload", backward_payload * ratios.rocev2, "legacy-measurement-ratio-scaled"),
        _component("PP", destination, source, "phy_overhead_for_backward_payload", backward_payload * ratios.phy, "legacy-measurement-ratio-scaled"),
        _component("PP", destination, source, "cts_for_forward_payload", forward_payload * ratios.cts, "legacy-measurement-ratio-scaled"),
        _component("PP", destination, source, "other_rocev2_control_for_forward_payload", forward_payload * ratios.reverse_other, "legacy-measurement-ratio-scaled"),
        _component("PP", destination, source, "unclassified_control_for_forward_payload", forward_payload * ratios.reverse_unclassified, "legacy-measurement-ratio-scaled"),
    ]
    metadata = {
        "forward_bytes_per_microbatch": forward_per_microbatch,
        "backward_bytes_per_microbatch": backward_per_microbatch,
        "forward_bytes": forward_payload,
        "backward_bytes": backward_payload,
        "stage_pair": f"{source}<->{destination}",
        "interval_definition": (
            f"one full training iteration ({microbatches} microbatches); adjacent PP boundary; "
            "one forward activation and one backward activation-gradient transfer per microbatch"
        ),
    }
    return directions, components, metadata


def _component(
    workload: str,
    source: str,
    destination: str,
    component: str,
    byte_count: float,
    provenance_class: str,
) -> dict[str, Any]:
    if byte_count < 0 or not math.isfinite(byte_count):
        raise ValueError(f"invalid component byte count for {component}: {byte_count}")
    return {
        "workload": workload,
        "source_endpoint": source,
        "destination_endpoint": destination,
        "component": component,
        "bytes": byte_count,
        "provenance_class": provenance_class,
    }


def select_main_and_opposite(
    directions: Iterable[DirectionTotal],
) -> tuple[DirectionTotal, DirectionTotal, bool]:
    """Sort two real directions by volume; retain configured order for an exact tie."""

    items = list(directions)
    if len(items) != 2:
        raise ValueError("exactly two ordered directions are required")
    tied = math.isclose(items[0].total_bytes, items[1].total_bytes, rel_tol=0.0, abs_tol=1e-6)
    if items[1].total_bytes > items[0].total_bytes:
        return items[1], items[0], tied
    return items[0], items[1], tied


def build_outputs(
    config: dict[str, Any], source_config_file: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the summary and component tables without writing or plotting."""

    ratios = protocol_ratios(config)
    model = config["model"]
    parallel = config["parallelism"]
    measurement = config["legacy_measurement"]
    rows: list[dict[str, Any]] = []
    all_components: list[dict[str, Any]] = []
    for workload, derivation in (("DP", derive_dp), ("PP", derive_pp)):
        directions, components, metadata = derivation(config, ratios)
        main, opposite, tied = select_main_and_opposite(directions)
        if main.total_bytes + 1e-6 < opposite.total_bytes:
            raise AssertionError("main direction must not be smaller than opposite")
        ratio = main.total_bytes / opposite.total_bytes if opposite.total_bytes > 0 else math.inf
        if workload == "DP":
            model_source = "llama3_comm.traffic.llama3_megatron_payloads + legacy overhead ratios"
            schedule_source = "llama3_comm.traffic ring (p-1)/p; llama3_modular ZeRO-2 RS+AG"
        else:
            model_source = "independent pp_tensors in configs/dp_pp_directional_traffic.json"
            schedule_source = "modeling.md full iteration; llama3_modular PP P2P nodes"
        rows.append({
            "workload": workload,
            "interval_definition": metadata["interval_definition"],
            "main_direction_bytes": main.total_bytes,
            "opposite_direction_bytes": opposite.total_bytes,
            "ratio": ratio,
            "main_direction_name": main.name,
            "opposite_direction_name": opposite.name,
            "main_source_endpoint": main.source,
            "main_destination_endpoint": main.destination,
            "opposite_source_endpoint": opposite.source,
            "opposite_destination_endpoint": opposite.destination,
            "directions_tied": tied,
            "model_source": model_source,
            "schedule_source": schedule_source,
            "assumption_version": config["assumption_versions"][workload.lower()],
            "forward_bytes": metadata.get("forward_bytes", ""),
            "backward_bytes": metadata.get("backward_bytes", ""),
            "stage_pair": metadata.get("stage_pair", ""),
            "payload_bytes": (main.payload_bytes + opposite.payload_bytes) if workload == "DP" else "",
            "control_bytes": (main.control_bytes + opposite.control_bytes) if workload == "DP" else "",
            "measurement_source": measurement["source_measurement_file"],
            "main_payload_bytes": main.payload_bytes,
            "main_control_bytes": main.control_bytes,
            "opposite_payload_bytes": opposite.payload_bytes,
            "opposite_control_bytes": opposite.control_bytes,
            "model_name": model["name"],
            "model_parameter_count": model["parameter_count"],
            "data_type": (
                "FP32 gradient RS + BF16 parameter AG"
                if workload == "DP"
                else (
                    f"forward={config['pp_tensors']['forward']['precision']}; "
                    f"backward={config['pp_tensors']['backward']['precision']}"
                )
            ),
            "dp_group_size": parallel["dp_group_size"] if workload == "DP" else "",
            "pipeline_stage_count": parallel["pipeline_stage_count"],
            "microbatch_count": parallel["microbatch_count"] if workload == "PP" else "",
            "sequence_length": model["sequence_length"],
            "hidden_size": model["hidden_size"],
            "source_config_file": source_config_file,
            "derivation_version": config["derivation_version"],
            "result_status": config["status"],
        })
        for component in components:
            component.update(
                {
                    "interval_definition": metadata["interval_definition"],
                    "source_measurement_file": measurement["source_measurement_file"],
                    "source_config_file": source_config_file,
                    "derivation_version": config["derivation_version"],
                    "result_status": config["status"],
                }
            )
        all_components.extend(components)
    return rows, all_components


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write a non-empty, consistently keyed list of dictionaries as CSV."""

    if not rows:
        raise ValueError("cannot write an empty CSV")
    keys = list(rows[0].keys())
    if any(set(row) != set(keys) for row in rows):
        raise ValueError("CSV rows have inconsistent fields")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
>>>>>>> 6839052da73682436a4eeed00ae6ac55603f3e49
