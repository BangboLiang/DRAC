"""Micro-model for the cost of symmetric directional provisioning.

This module deliberately excludes the DRAC solver, topology constraints, and
OCS reconfiguration delay.  Functions were introduced by
``run_symmetric_provisioning_cost.py``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .directional_traffic import load_dp_directional_demand


SYMMETRIC_COLOR = "#ff7f0e"
DIRECTION_AWARE_COLOR = "#1f77b4"
MAIN_COLOR = "#4E79A7"
OPPOSITE_COLOR = "#F28E2B"
IDLE_COLOR = "#E6E6E6"
DEFAULT_DPI = 300


@dataclass(frozen=True)
class ProvisioningResult:
    scheme: str
    main_demand_gb: float
    opposite_demand_gb: float
    total_channels: int
    main_channels: int
    opposite_channels: int
    channel_bandwidth_gbps: float
    main_capacity_gbps: float
    opposite_capacity_gbps: float
    main_time_ms: float
    opposite_time_ms: float
    completion_time_ms: float
    useful_main_gb: float
    useful_opposite_gb: float
    main_idle_time_ms: float
    opposite_idle_time_ms: float
    main_channel_idle_time_ms: float
    opposite_channel_idle_time_ms: float
    total_channel_time_ms: float
    total_idle_fraction: float
    opposite_idle_fraction: float

    @property
    def active_time_main_ms(self) -> float:
        return self.main_time_ms

    @property
    def active_time_opposite_ms(self) -> float:
        return self.opposite_time_ms

    @property
    def total_idle_channel_time_ms(self) -> float:
        return self.main_channel_idle_time_ms + self.opposite_channel_idle_time_ms


def gbps_to_gb_per_second(channel_bandwidth_gbps: float) -> float:
    if channel_bandwidth_gbps <= 0:
        raise ValueError("channel bandwidth must be positive")
    return float(channel_bandwidth_gbps) / 8.0


def evaluate_allocation(
    main_demand_gb: float,
    opposite_demand_gb: float,
    total_channels: int,
    main_channels: int,
    opposite_channels: int,
    channel_bandwidth_gbps: float,
    scheme: str,
) -> ProvisioningResult:
    if main_demand_gb <= 0 or opposite_demand_gb <= 0:
        raise ValueError("both ordered-direction demands must be positive")
    if min(main_channels, opposite_channels) < 1:
        raise ValueError("each active direction requires at least one channel")
    if main_channels + opposite_channels > total_channels:
        raise ValueError("allocation exceeds the total channel budget")
    unit_gbps = float(channel_bandwidth_gbps)
    unit_gb_s = gbps_to_gb_per_second(unit_gbps)
    main_time_s = main_demand_gb / (main_channels * unit_gb_s)
    opposite_time_s = opposite_demand_gb / (opposite_channels * unit_gb_s)
    theta_s = max(main_time_s, opposite_time_s)
    main_idle_s = max(0.0, theta_s - main_time_s)
    opposite_idle_s = max(0.0, theta_s - opposite_time_s)
    allocated = main_channels + opposite_channels
    channel_time = allocated * theta_s
    total_idle = (main_channels * main_idle_s + opposite_channels * opposite_idle_s) / channel_time
    opposite_idle = opposite_channels * opposite_idle_s / channel_time
    return ProvisioningResult(
        scheme=scheme,
        main_demand_gb=main_demand_gb,
        opposite_demand_gb=opposite_demand_gb,
        total_channels=total_channels,
        main_channels=main_channels,
        opposite_channels=opposite_channels,
        channel_bandwidth_gbps=unit_gbps,
        main_capacity_gbps=main_channels * unit_gbps,
        opposite_capacity_gbps=opposite_channels * unit_gbps,
        main_time_ms=main_time_s * 1000.0,
        opposite_time_ms=opposite_time_s * 1000.0,
        completion_time_ms=theta_s * 1000.0,
        useful_main_gb=min(main_demand_gb, main_channels * unit_gb_s * theta_s),
        useful_opposite_gb=min(opposite_demand_gb, opposite_channels * unit_gb_s * theta_s),
        main_idle_time_ms=main_idle_s * 1000.0,
        opposite_idle_time_ms=opposite_idle_s * 1000.0,
        main_channel_idle_time_ms=main_channels * main_idle_s * 1000.0,
        opposite_channel_idle_time_ms=opposite_channels * opposite_idle_s * 1000.0,
        total_channel_time_ms=channel_time * 1000.0,
        total_idle_fraction=total_idle,
        opposite_idle_fraction=opposite_idle,
    )


def symmetric_provisioning(
    main_demand_gb: float,
    opposite_demand_gb: float,
    total_channels: int = 8,
    channel_bandwidth_gbps: float = 50.0,
    minimum_channels_per_active_direction: int = 1,
) -> ProvisioningResult:
    if total_channels < 2 * minimum_channels_per_active_direction:
        raise ValueError("channel budget cannot satisfy both active directions")
    channels_each = total_channels // 2
    if channels_each < minimum_channels_per_active_direction:
        raise ValueError("symmetric allocation violates the per-direction minimum")
    return evaluate_allocation(
        main_demand_gb,
        opposite_demand_gb,
        total_channels,
        channels_each,
        channels_each,
        channel_bandwidth_gbps,
        "Symmetric",
    )


def feasible_allocations(total_channels: int, minimum_channels: int = 1) -> Iterable[tuple[int, int]]:
    if minimum_channels < 1:
        raise ValueError("minimum_channels must be at least one for active directions")
    for main_channels in range(minimum_channels, total_channels + 1):
        for opposite_channels in range(minimum_channels, total_channels + 1):
            if main_channels + opposite_channels <= total_channels:
                yield main_channels, opposite_channels


def direction_aware_provisioning(
    main_demand_gb: float,
    opposite_demand_gb: float,
    total_channels: int = 8,
    channel_bandwidth_gbps: float = 50.0,
    minimum_channels_per_active_direction: int = 1,
) -> ProvisioningResult:
    demand_share = main_demand_gb / (main_demand_gb + opposite_demand_gb)
    candidates: List[ProvisioningResult] = []
    for main_channels, opposite_channels in feasible_allocations(
        total_channels, minimum_channels_per_active_direction
    ):
        candidates.append(
            evaluate_allocation(
                main_demand_gb,
                opposite_demand_gb,
                total_channels,
                main_channels,
                opposite_channels,
                channel_bandwidth_gbps,
                "Direction-aware",
            )
        )
    if not candidates:
        raise ValueError("no feasible allocation under the channel budget")
    best_time = min(item.completion_time_ms for item in candidates)
    tolerance = max(1e-12, abs(best_time) * 1e-12)
    tied = [item for item in candidates if abs(item.completion_time_ms - best_time) <= tolerance]
    return min(
        tied,
        key=lambda item: (
            item.main_channels + item.opposite_channels,
            abs(item.main_channels / (item.main_channels + item.opposite_channels) - demand_share),
            item.main_channels,
        ),
    )


def _row(result: ProvisioningResult, normalizer_ms: float, speedup: float) -> Dict[str, object]:
    row = asdict(result)
    row["active_time_main_ms"] = result.active_time_main_ms
    row["active_time_opposite_ms"] = result.active_time_opposite_ms
    row["total_idle_channel_time_ms"] = result.total_idle_channel_time_ms
    row["normalized_completion_time"] = result.completion_time_ms / normalizer_ms
    row["speedup_vs_symmetric"] = speedup
    return row


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(fig: object, base_path: Path) -> None:
    import matplotlib.pyplot as plt

    base_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = fig.get_size_inches()
    if width > 20 or height > 20:
        raise ValueError(f"abnormal figure size: {width}x{height} inches")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".png"), dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_representative_legacy(sym: ProvisioningResult, aware: ProvisioningResult, output_base: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), constrained_layout=True)
    labels = ["Symmetric", "Direction-aware"]
    x = np.arange(2)
    main = [sym.main_channels, aware.main_channels]
    opposite = [sym.opposite_channels, aware.opposite_channels]
    axes[0].bar(x, main, width=0.62, color=MAIN_COLOR, label="Main direction")
    axes[0].bar(x, opposite, width=0.62, bottom=main, color=OPPOSITE_COLOR, label="Opposite direction")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Allocated optical channels")
    axes[0].set_ylim(0, max(item.total_channels for item in (sym, aware)) * 1.12)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8, frameon=False, loc="upper center")
    axes[0].text(-0.12, 1.03, "(a)", transform=axes[0].transAxes, fontsize=9, fontweight="bold")

    normalized = [1.0, aware.completion_time_ms / sym.completion_time_ms]
    axes[1].bar(x, normalized, width=0.62, color=[SYMMETRIC_COLOR, DIRECTION_AWARE_COLOR])
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Normalized DP completion time")
    axes[1].set_ylim(0, 1.14)
    axes[1].grid(axis="y", alpha=0.25)
    for xpos, value in zip(x, normalized):
        axes[1].text(xpos, value + 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    axes[1].text(-0.12, 1.03, "(b)", transform=axes[1].transAxes, fontsize=9, fontweight="bold")
    _save_figure(fig, output_base)


def plot_sensitivity_legacy(rows: Sequence[Dict[str, object]], output_base: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 3.7), constrained_layout=True)
    for scheme, color, marker in [
        ("Symmetric", SYMMETRIC_COLOR, "s"),
        ("Direction-aware", DIRECTION_AWARE_COLOR, "o"),
    ]:
        selected = sorted((row for row in rows if row["scheme"] == scheme), key=lambda row: float(row["main_share"]))
        ax.plot(
            [float(row["main_share"]) for row in selected],
            [float(row["normalized_time"]) for row in selected],
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=4.5,
            label=scheme,
        )
    ax.set_xlabel("Main-direction traffic share")
    ax.set_ylabel("Normalized completion time")
    ax.set_xticks([0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99])
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    _save_figure(fig, output_base)


def _draw_channel_time_band(ax: object, result: ProvisioningResult, x_limit: float) -> None:
    """Draw one exact channel-time allocation without distorting short flows."""
    from matplotlib.patches import Rectangle

    main_y = 0.0
    opposite_y = float(result.main_channels)
    main_height = float(result.main_channels)
    opposite_height = float(result.opposite_channels)

    ax.add_patch(Rectangle((0, main_y), result.main_time_ms, main_height, facecolor=MAIN_COLOR, edgecolor="none"))
    if result.main_idle_time_ms > 0:
        ax.add_patch(
            Rectangle(
                (result.main_time_ms, main_y),
                result.main_idle_time_ms,
                main_height,
                facecolor=IDLE_COLOR,
                edgecolor="#999999",
                hatch="///",
                linewidth=0.5,
            )
        )
    ax.add_patch(
        Rectangle(
            (0, opposite_y),
            result.opposite_time_ms,
            opposite_height,
            facecolor=OPPOSITE_COLOR,
            edgecolor="none",
        )
    )
    if result.opposite_idle_time_ms > 0:
        ax.add_patch(
            Rectangle(
                (result.opposite_time_ms, opposite_y),
                result.opposite_idle_time_ms,
                opposite_height,
                facecolor=IDLE_COLOR,
                edgecolor="#999999",
                hatch="///",
                linewidth=0.5,
            )
        )
    ax.add_patch(
        Rectangle(
            (0, 0),
            result.completion_time_ms,
            result.main_channels + result.opposite_channels,
            fill=False,
            edgecolor="#333333",
            linewidth=0.9,
        )
    )

    ax.axvline(result.opposite_time_ms, color=OPPOSITE_COLOR, linestyle="--", linewidth=1.0)
    ax.axvline(result.completion_time_ms, color="#333333", linestyle="--", linewidth=1.0)
    ax.text(
        result.main_time_ms * 0.50,
        main_y + main_height * 0.50,
        "Main-direction traffic",
        ha="center",
        va="center",
        fontsize=8,
        color="white",
        fontweight="semibold",
    )
    if result.opposite_idle_time_ms > 0 and result.opposite_channels >= 2:
        ax.text(
            result.opposite_time_ms + result.opposite_idle_time_ms * 0.50,
            opposite_y + opposite_height * 0.50,
            "Idle capacity",
            ha="center",
            va="center",
            fontsize=8,
            color="#444444",
        )
    ax.annotate(
        f"Opposite completes: {result.opposite_time_ms:.1f} ms",
        xy=(result.opposite_time_ms, opposite_y + opposite_height * 0.72),
        xytext=(max(55.0, 0.12 * result.completion_time_ms), opposite_y + opposite_height * 0.72),
        textcoords="data",
        ha="left",
        va="center",
        fontsize=7.5,
        color="#7A4311",
        arrowprops={"arrowstyle": "-", "color": OPPOSITE_COLOR, "linewidth": 0.8},
    )
    ax.text(
        result.completion_time_ms,
        0.12,
        f"Completion: {result.completion_time_ms:.1f} ms",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="white",
        fontweight="semibold",
    )
    idle_label_outside = result.completion_time_ms < 0.75 * x_limit
    ax.text(
        result.completion_time_ms + 15.0 if idle_label_outside else max(0.0, result.completion_time_ms - 5.0),
        result.main_channels + result.opposite_channels - 0.25,
        f"{100 * result.total_idle_fraction:.1f}% idle channel-time",
        ha="left" if idle_label_outside else "right",
        va="top",
        fontsize=8,
    )
    ax.set_title(
        f"{result.scheme} ({result.main_channels}/{result.opposite_channels})",
        loc="left",
        fontsize=9,
        fontweight="semibold",
        pad=5,
    )
    ax.set_ylim(0, result.main_channels + result.opposite_channels)
    ticks = sorted({0, result.main_channels, result.main_channels + result.opposite_channels})
    ax.set_yticks(ticks)
    ax.grid(axis="x", alpha=0.18, linewidth=0.7)
    ax.set_axisbelow(True)


def plot_channel_time_utilization(
    sym: ProvisioningResult, aware: ProvisioningResult, output_base: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    completion_max = max(sym.completion_time_ms, aware.completion_time_ms)
    x_limit = float(int(completion_max / 50.0 + 1.0) * 50)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.35), sharex=True)
    for ax, result in zip(axes, (sym, aware)):
        _draw_channel_time_band(ax, result, x_limit)
        ax.set_xlim(0, x_limit)
    axes[-1].set_xlabel("Communication time (ms)")
    fig.text(0.018, 0.5, "Allocated directional channels", rotation=90, va="center", fontsize=10)
    legend_handles = [
        Patch(facecolor=MAIN_COLOR, edgecolor="none", label="Main-direction traffic"),
        Patch(facecolor=OPPOSITE_COLOR, edgecolor="none", label="Opposite-direction traffic"),
        Patch(facecolor=IDLE_COLOR, edgecolor="#999999", hatch="///", label="Idle capacity"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.12, top=0.86, hspace=0.34)
    _save_figure(fig, output_base)


def dense_sensitivity_rows(
    total_demand_gb: float,
    total_channels: int = 8,
    channel_bandwidth_gbps: float = 50.0,
    minimum_channels_per_active_direction: int = 1,
    samples: int = 500,
) -> List[Dict[str, object]]:
    """Compute a dense inclusive scan from main share 0.500 to 0.999."""
    if samples < 2:
        raise ValueError("dense sensitivity requires at least two samples")
    rows: List[Dict[str, object]] = []
    for index in range(samples):
        share = 0.5 + (0.999 - 0.5) * index / (samples - 1)
        main_gb = total_demand_gb * share
        opposite_gb = total_demand_gb * (1.0 - share)
        sym = symmetric_provisioning(
            main_gb,
            opposite_gb,
            total_channels,
            channel_bandwidth_gbps,
            minimum_channels_per_active_direction,
        )
        aware = direction_aware_provisioning(
            main_gb,
            opposite_gb,
            total_channels,
            channel_bandwidth_gbps,
            minimum_channels_per_active_direction,
        )
        normalized = aware.completion_time_ms / sym.completion_time_ms
        allocation = f"{aware.main_channels}/{aware.opposite_channels}"
        rows.append(
            {
                "main_share": share,
                "opposite_share": 1.0 - share,
                "skew_ratio": share / (1.0 - share),
                "symmetric_main_channels": sym.main_channels,
                "symmetric_opposite_channels": sym.opposite_channels,
                "direction_aware_main_channels": aware.main_channels,
                "direction_aware_opposite_channels": aware.opposite_channels,
                "symmetric_completion_time_ms": sym.completion_time_ms,
                "direction_aware_completion_time_ms": aware.completion_time_ms,
                "normalized_direction_aware_time": normalized,
                "completion_time_reduction": 100.0 * (1.0 - normalized),
                "total_idle_fraction_symmetric": sym.total_idle_fraction,
                "total_idle_fraction_direction_aware": aware.total_idle_fraction,
                "allocation_region": allocation,
            }
        )
    return rows


def detect_allocation_regions(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Detect contiguous integer-allocation regions from computed scan rows."""
    if not rows:
        return []
    regions: List[Dict[str, object]] = []
    start = 0
    for index in range(1, len(rows) + 1):
        changed = index == len(rows) or rows[index]["allocation_region"] != rows[start]["allocation_region"]
        if changed:
            regions.append(
                {
                    "allocation_region": rows[start]["allocation_region"],
                    "start_main_share": float(rows[start]["main_share"]),
                    "end_main_share": float(rows[index - 1]["main_share"]),
                }
            )
            start = index
    return regions


def plot_skew_reduction(rows: Sequence[Dict[str, object]], output_base: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    regions = detect_allocation_regions(rows)
    fig, ax = plt.subplots(figsize=(6.4, 3.75), constrained_layout=True)
    for index, region in enumerate(regions):
        start = 100.0 * float(region["start_main_share"])
        end = 100.0 * float(region["end_main_share"])
        ax.axvspan(start, end, color="#F4F4F4" if index % 2 == 0 else "#FAFAFA", zorder=0)
        if index > 0:
            ax.axvline(start, color="#888888", linestyle="--", linewidth=0.8, alpha=0.8)
        ax.text(
            (start + end) / 2.0,
            0.965,
            str(region["allocation_region"]),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color="#444444",
        )
    ax.plot(
        [100.0 * float(row["main_share"]) for row in rows],
        [float(row["completion_time_reduction"]) for row in rows],
        color=DIRECTION_AWARE_COLOR,
        linewidth=1.8,
    )
    ax.set_xlabel("Main-direction traffic share (%)")
    ax.set_ylabel("Completion-time reduction (%)")
    ax.set_xlim(50.0, 100.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xticks([50, 60, 70, 80, 90, 100])
    ax.grid(axis="y", alpha=0.22, linewidth=0.7)
    ax.set_axisbelow(True)
    _save_figure(fig, output_base)


def _tex_percent(value: float) -> str:
    return f"{value:.1f}\\%"


def run_experiment(
    input_csv: str | Path = "results/dp_pp_directional_traffic/directional_traffic.csv",
    output_dir: str | Path = "results/symmetric_provisioning_cost",
    figure_dir: str | Path = "figures",
    total_channels: int = 8,
    channel_bandwidth_gbps: float = 50.0,
    minimum_channels_per_active_direction: int = 1,
    main_shares: Sequence[float] = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
    dense_samples: int = 500,
) -> Dict[str, Path]:
    input_path = Path(input_csv)
    output = Path(output_dir)
    figures = Path(figure_dir)
    output.mkdir(parents=True, exist_ok=True)
    main_gb, opposite_gb = load_dp_directional_demand(input_path)
    sym = symmetric_provisioning(main_gb, opposite_gb, total_channels, channel_bandwidth_gbps, minimum_channels_per_active_direction)
    aware = direction_aware_provisioning(main_gb, opposite_gb, total_channels, channel_bandwidth_gbps, minimum_channels_per_active_direction)
    speedup = sym.completion_time_ms / aware.completion_time_ms
    reduction = 1.0 - aware.completion_time_ms / sym.completion_time_ms
    representative_rows = [_row(sym, sym.completion_time_ms, 1.0), _row(aware, sym.completion_time_ms, speedup)]
    representative_csv = output / "representative_dp_case.csv"
    _write_csv(representative_csv, representative_rows)

    total_gb = main_gb + opposite_gb
    sensitivity_rows: List[Dict[str, object]] = []
    allocations: List[str] = []
    for share in main_shares:
        scan_main = total_gb * share
        scan_opposite = total_gb * (1.0 - share)
        scan_sym = symmetric_provisioning(scan_main, scan_opposite, total_channels, channel_bandwidth_gbps, minimum_channels_per_active_direction)
        scan_aware = direction_aware_provisioning(scan_main, scan_opposite, total_channels, channel_bandwidth_gbps, minimum_channels_per_active_direction)
        scan_speedup = scan_sym.completion_time_ms / scan_aware.completion_time_ms
        allocations.append(f"{share:.2f}: {scan_aware.main_channels}/{scan_aware.opposite_channels}")
        for item in (scan_sym, scan_aware):
            sensitivity_rows.append(
                {
                    "main_share": share,
                    "skew_ratio": share / (1.0 - share),
                    "scheme": item.scheme,
                    "main_channels": item.main_channels,
                    "opposite_channels": item.opposite_channels,
                    "completion_time_ms": item.completion_time_ms,
                    "normalized_time": item.completion_time_ms / scan_sym.completion_time_ms,
                    "speedup": scan_speedup,
                    "total_idle_fraction": item.total_idle_fraction,
                }
            )
    sensitivity_csv = output / "skew_sensitivity.csv"
    _write_csv(sensitivity_csv, sensitivity_rows)
    dense_rows = dense_sensitivity_rows(
        total_gb,
        total_channels,
        channel_bandwidth_gbps,
        minimum_channels_per_active_direction,
        samples=dense_samples,
    )
    dense_csv = output / "skew_sensitivity_dense.csv"
    _write_csv(dense_csv, dense_rows)
    allocation_regions = detect_allocation_regions(dense_rows)

    config = {
        "input_data_path": str(input_path),
        "dp_data_fields": {"main": "main_direction_bytes", "opposite": "opposite_direction_bytes"},
        "input_units": "bytes; converted to decimal GB using 1 GB = 1e9 bytes",
        "total_channels": total_channels,
        "channel_bandwidth_gbps": channel_bandwidth_gbps,
        "minimum_channels_per_active_direction": minimum_channels_per_active_direction,
        "minimum_opposite_channels": minimum_channels_per_active_direction,
        "symmetric_odd_budget_policy": "use the largest even number not exceeding total_channels",
        "main_share_scan": list(main_shares),
        "dense_main_share_scan": {"start": 0.5, "stop": 0.999, "samples": dense_samples},
        "detected_allocation_regions": allocation_regions,
        "model": "two ordered directions transfer concurrently; stage time is their maximum",
    }
    config_path = output / "experiment_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    main_figure = figures / "symmetric_provisioning_cost"
    legacy_main_figure = figures / "symmetric_provisioning_cost_legacy"
    legacy_sensitivity_figure = figures / "symmetric_provisioning_skew_sensitivity_legacy"
    skew_figure = figures / "symmetric_provisioning_skew_reduction"
    plot_representative_legacy(sym, aware, legacy_main_figure)
    plot_sensitivity_legacy(sensitivity_rows, legacy_sensitivity_figure)
    plot_channel_time_utilization(sym, aware, main_figure)
    plot_skew_reduction(dense_rows, skew_figure)

    caption = (
        "\\caption{\n"
        "Cost of symmetric provisioning under a representative directional DP demand. "
        f"Both configurations use the same budget of {total_channels} equal-rate directional optical connection units. "
        f"Under symmetric provisioning, the {sym.opposite_channels} opposite-direction units finish their control-dominated traffic early and remain idle while the main direction continues transmitting. "
        f"Direction-aware provisioning reallocates {aware.main_channels - sym.main_channels} of these units to the payload-dominant direction, reducing idle channel-time from {_tex_percent(100 * sym.total_idle_fraction)} to {_tex_percent(100 * aware.total_idle_fraction)} and communication completion time from {sym.completion_time_ms:.1f}~ms to {aware.completion_time_ms:.1f}~ms. "
        "The improvement is achieved without increasing the total optical-resource budget.\n"
        "}\n\\label{fig:symmetric_provisioning_cost}\n"
    )
    caption_path = output / "caption.tex"
    caption_path.write_text(caption, encoding="utf-8")

    opposite_idle_stage_fraction = sym.opposite_idle_time_ms / sym.completion_time_ms
    background = (
        "\\subsection{Motivating Example: The Cost of Symmetric Provisioning}\n"
        "We use the representative ordered DP demand reported in Figure~\\ref{fig:dp_pp_directionality}. "
        f"The main direction carries {main_gb:.3f}~GB, whereas the opposite direction carries {opposite_gb:.3f}~GB. "
        f"Both provisioning schemes use the same budget of {total_channels} directional optical connection units, each operating at {channel_bandwidth_gbps:g}~Gbps.\n\n"
        f"Under the {total_channels}-channel budget, the best feasible symmetric configuration assigns {sym.main_channels} units to each direction. "
        f"The main direction requires {sym.main_time_ms:.1f}~ms to complete, while the opposite direction requires only {sym.opposite_time_ms:.1f}~ms. "
        f"Consequently, the {sym.opposite_channels} opposite-direction units remain idle for {100 * opposite_idle_stage_fraction:.1f}\\% of the communication interval while the main direction continues transmitting. "
        f"As visualized by the channel-time area in Figure~\\ref{{fig:symmetric_provisioning_cost}}, this idle capacity accounts for {_tex_percent(100 * sym.total_idle_fraction)} of the entire directional channel-time budget.\n\n"
        f"Enumerating feasible direction-aware integer allocations instead selects {aware.main_channels}/{aware.opposite_channels}. "
        f"The main direction then completes in {aware.main_time_ms:.1f}~ms, and the opposite direction completes in {aware.opposite_time_ms:.1f}~ms. "
        f"The total idle channel-time fraction falls to {_tex_percent(100 * aware.total_idle_fraction)}, and the communication completion time falls by {_tex_percent(100 * reduction)} to {aware.completion_time_ms:.1f}~ms.\n\n"
        "The gain comes entirely from redistributing the same directional connection units: neither the total channel count nor the per-channel rate is increased. "
        f"The {_tex_percent(100 * reduction)} reduction is a representative result for this discrete channel budget, not a universal performance bound for DP workloads. "
        "When the two ordered-direction demands approach balance, the enumerated direction-aware allocation naturally degenerates to the symmetric configuration.\n"
    )
    background_path = output / "background_text.tex"
    background_path.write_text(background, encoding="utf-8")

    max_reduction = max(float(row["completion_time_reduction"]) for row in dense_rows)
    skew_caption = (
        "\\caption{\n"
        "Sensitivity of the completion-time benefit to ordered-direction traffic skew. "
        f"We keep the total communication volume and the budget of {total_channels} equal-rate directional connection units fixed while varying the fraction of traffic in the main direction. "
        f"Symmetric provisioning uses a {sym.main_channels}-and-{sym.opposite_channels} allocation, whereas direction-aware provisioning selects the completion-time-minimizing integer allocation while retaining at least one unit for each active direction. "
        f"The benefit is zero for balanced demand, increases as the optimal allocation changes through {', '.join(str(region['allocation_region']) for region in allocation_regions)}, and saturates at {_tex_percent(max_reduction)} once the payload-dominant direction receives {total_channels - minimum_channels_per_active_direction} units.\n"
        "}\n\\label{fig:symmetric_provisioning_skew}\n"
    )
    skew_caption_path = output / "skew_caption.tex"
    skew_caption_path.write_text(skew_caption, encoding="utf-8")

    opposite_idle_channel_ms = sym.opposite_channel_idle_time_ms
    report = f"""# Cost of Symmetric Provisioning

## Scope and input

This is a micro-simulation based on a representative DP demand. It isolates the effect of the symmetric provisioning constraint; it is not a complete training-performance result or a real-cluster measurement. The DP demand is loaded from `{input_path}` using `main_direction_bytes` and `opposite_direction_bytes`, interpreted as bytes and converted with 1 GB = 1e9 bytes.

- Main-direction demand: {main_gb:.12f} GB
- Opposite-direction demand: {opposite_gb:.12f} GB
- Resource budget: {total_channels} directional channels at {channel_bandwidth_gbps:g} Gbps each
- Total directional line-rate budget: {total_channels * channel_bandwidth_gbps:g} Gbps for both schemes

## Representative result

| Scheme | Allocation (main/opposite) | Main time | Opposite time | Stage completion |
|---|---:|---:|---:|---:|
| Symmetric | {sym.main_channels}/{sym.opposite_channels} | {sym.main_time_ms:.3f} ms | {sym.opposite_time_ms:.3f} ms | {sym.completion_time_ms:.3f} ms |
| Direction-aware | {aware.main_channels}/{aware.opposite_channels} | {aware.main_time_ms:.3f} ms | {aware.opposite_time_ms:.3f} ms | {aware.completion_time_ms:.3f} ms |

The direction-aware integer optimum lowers completion time by **{100 * reduction:.3f}%** (speedup {speedup:.3f}x) without adding channels or changing the per-channel bandwidth. Under symmetric provisioning, opposite-direction channels contribute {opposite_idle_channel_ms:.3f} channel-ms of idle time after their traffic completes. The total idle channel-time fraction falls from {sym.total_idle_fraction:.6f} under Symmetric to {aware.total_idle_fraction:.6f} under Direction-aware provisioning.

## Skew sensitivity

The dense scan evaluates {dense_samples} points from main share 0.500 through 0.999. Detected allocation regions are: {', '.join(f"{region['allocation_region']} from {100 * float(region['start_main_share']):.1f}% to {100 * float(region['end_main_share']):.1f}%" for region in allocation_regions)}. At main share 0.50 the optimum is 4/4 and completion-time reduction is zero. The 7/1 region reaches a {max_reduction:.3f}% plateau because the main direction can receive at most seven of the eight channels when one unit must remain assigned to the active opposite direction. This plateau is specific to the configured discrete budget, not a general upper bound.

## Generated artifacts

- `representative_dp_case.csv`
- `skew_sensitivity.csv`
- `skew_sensitivity_dense.csv`
- `experiment_config.json`
- `{main_figure}.pdf` and `{main_figure}.png`
- `{skew_figure}.pdf` and `{skew_figure}.png`
- `{legacy_main_figure}.pdf` and `{legacy_main_figure}.png` (legacy comparison)
- `{legacy_sensitivity_figure}.pdf` and `{legacy_sensitivity_figure}.png` (legacy comparison)
- `caption.tex`
- `skew_caption.tex`
- `background_text.tex`

## LaTeX caption

```tex
{caption.rstrip()}
```

## Background text

```tex
{background.rstrip()}
```

## Skew-sensitivity caption

```tex
{skew_caption.rstrip()}
```

## Reproduction

```powershell
python generate_dp_pp_directional_traffic.py
python run_symmetric_provisioning_cost.py
pytest -q tests/test_symmetric_provisioning_cost.py
```

## Limitations

- The model assumes the two ordered directions transfer concurrently and stage completion is determined by the slower direction.
- It does not include OCS reconfiguration latency.
- It does not include contention among multiple endpoints.
- It is not a complete DRAC solver comparison and does not model full-cluster training performance.
- Complete cluster-level results belong in the Evaluation section.
"""
    report_path = output / "REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    return {
        "representative_csv": representative_csv,
        "sensitivity_csv": sensitivity_csv,
        "dense_sensitivity_csv": dense_csv,
        "config": config_path,
        "main_figure_pdf": main_figure.with_suffix(".pdf"),
        "main_figure_png": main_figure.with_suffix(".png"),
        "skew_figure_pdf": skew_figure.with_suffix(".pdf"),
        "skew_figure_png": skew_figure.with_suffix(".png"),
        "legacy_main_figure_pdf": legacy_main_figure.with_suffix(".pdf"),
        "legacy_main_figure_png": legacy_main_figure.with_suffix(".png"),
        "legacy_sensitivity_figure_pdf": legacy_sensitivity_figure.with_suffix(".pdf"),
        "legacy_sensitivity_figure_png": legacy_sensitivity_figure.with_suffix(".png"),
        "caption": caption_path,
        "skew_caption": skew_caption_path,
        "background": background_path,
        "report": report_path,
    }
