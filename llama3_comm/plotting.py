"""Trace output functions: JSON, CSV, and PNG plotting."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

from .execution import TraceEvent


def write_trace_json(path: Path, events: List[TraceEvent]) -> None:
    """Write trace events to a JSON file."""
    data = [
        {
            "strategy": e.strategy,
            "kind": e.kind,
            "label": e.label,
            "domain": e.domain,
            "start_ms": e.start_ms,
            "duration_ms": e.duration_ms,
            "end_ms": e.end_ms,
            "bw_share": e.bw_share,
            "bw_units": e.bw_units,
            "degree_split": e.degree_split,
        }
        for e in events
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )


def write_trace_csv(path: Path, events: List[TraceEvent]) -> None:
    """Write trace events to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "strategy",
                "kind",
                "domain",
                "label",
                "start_ms",
                "duration_ms",
                "end_ms",
                "bw_tp",
                "bw_pp",
                "bw_dp",
                "k_tp",
                "k_pp",
                "k_dp",
            ]
        )
        for e in events:
            bw = e.bw_share or {}
            k = e.degree_split or {}
            w.writerow(
                [
                    e.strategy,
                    e.kind,
                    e.domain,
                    e.label,
                    f"{e.start_ms:.6f}",
                    f"{e.duration_ms:.6f}",
                    f"{e.end_ms:.6f}",
                    f"{float(bw.get('tp', 0.0)):.6f}",
                    f"{float(bw.get('pp', 0.0)):.6f}",
                    f"{float(bw.get('dp', 0.0)):.6f}",
                    str(int(k.get("tp", 0))) if "tp" in k else "0",
                    str(int(k.get("pp", 0))) if "pp" in k else "0",
                    str(int(k.get("dp", 0))) if "dp" in k else "0",
                ]
            )


def try_plot_rows_png(
    out_path: Path,
    rows: List[Tuple[str, List[TraceEvent]]],
    title: str,
    pp_label_every: int,
    min_marker_ms: float,
    ms_per_inch: float,
    max_width_in: float,
    params_text: str,
) -> bool:
    """Plot an arbitrary list of labeled trace rows with a shared time scale.

    :param rows: list of (row_label, events) pairs. Events should already have absolute start_ms.
    :return: True if plot was written; False if matplotlib unavailable.
    """
    try:
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
    except Exception:
        return False

    COL_TP = "#4DA3D9"
    COL_PP = "#E56B8A"
    COL_DP = "#4FB06E"
    COL_BW_RECFG = "#9AA0A6"
    COL_LINK_RECFG = "#B0B5BB"

    def _lighten(hex_color: str, t: float = 0.55) -> str:
        """Blend a hex color toward white by factor t in [0,1]."""
        t = max(0.0, min(1.0, float(t)))
        c = hex_color.lstrip("#")
        r = int(c[0:2], 16)
        g = int(c[2:4], 16)
        b = int(c[4:6], 16)
        r2 = int(round(r + (255 - r) * t))
        g2 = int(round(g + (255 - g) * t))
        b2 = int(round(b + (255 - b) * t))
        return f"#{r2:02x}{g2:02x}{b2:02x}"

    COL_TP_L = _lighten(COL_TP, 0.62)
    COL_PP_L = _lighten(COL_PP, 0.62)
    COL_DP_L = _lighten(COL_DP, 0.62)

    total_ms = 0.0
    for _lbl, evs in rows:
        if evs:
            total_ms = max(total_ms, max(e.end_ms for e in evs))
    # Padding space on the right for per-row completion-time annotations.
    pad_ms = max(5.0, 0.02 * total_ms) if total_ms > 0 else 5.0

    def _bw_key(e: TraceEvent) -> Tuple[Tuple[str, float], ...]:
        bw = e.bw_share or {}
        return tuple((k, float(bw.get(k, 0.0))) for k in ["tp", "pp", "dp"])

    def _coalesce_for_plot(evs: List[TraceEvent]) -> List[TraceEvent]:
        """Reduce clutter by coalescing consecutive comm events with same domain and bw split."""
        out: List[TraceEvent] = []
        i = 0
        while i < len(evs):
            e = evs[i]
            if e.kind != "comm":
                out.append(e)
                i += 1
                continue

            # Merge runs of consecutive comm events with same domain + bw split.
            j = i + 1
            dur = float(e.duration_ms)
            while (
                j < len(evs)
                and evs[j].kind == "comm"
                and evs[j].domain == e.domain
                and _bw_key(evs[j]) == _bw_key(e)
            ):
                dur += float(evs[j].duration_ms)
                j += 1

            # Compact label: "MBk\nTP"/"MBk\nPP", DP => "DP\n(RS+AG)" when merging multiple.
            label = e.label
            if e.domain in ["tp", "pp"] and e.label.startswith("MB"):
                # e.label like "MB{n}:TP:AllGather"
                mb = e.label.split(":")[0] if ":" in e.label else e.label.split()[0]
                dom = "TP" if e.domain == "tp" else "PP"
                label = f"{mb}\n{dom}"
            elif e.domain == "dp":
                label = "DP\n(RS+AG)" if (j - i) >= 2 else "DP"

            out.append(
                TraceEvent(
                    strategy=e.strategy,
                    kind=e.kind,
                    label=label,
                    domain=e.domain,
                    start_ms=e.start_ms,
                    duration_ms=dur,
                    bw_share=e.bw_share,
                    bw_units=e.bw_units,
                )
            )
            i = j
        return out

    # Coalesce each row trace for plotting readability.
    plot_rows: List[Tuple[str, List[TraceEvent]]] = [
        (lbl, _coalesce_for_plot(evs)) for lbl, evs in rows
    ]

    # Auto-size width so small PP/reconfig blocks become visible, without wrapping the timeline.
    ms_per_in = max(1.0, float(ms_per_inch))
    w_in = total_ms / ms_per_in
    w_in = max(22.0, float(w_in))
    w_in = min(float(max_width_in), float(w_in))

    # Height scales with number of rows.
    nrows = max(1, len(plot_rows))
    h = 0.75
    row_step = 1.0
    fig_h_in = max(4.2, 1.0 + nrows * 0.85)
    fig, ax = plt.subplots(figsize=(w_in, fig_h_in))

    def _col(domain: str, active: bool) -> str:
        if domain == "tp":
            return COL_TP if active else COL_TP_L
        if domain == "pp":
            return COL_PP if active else COL_PP_L
        if domain == "dp":
            return COL_DP if active else COL_DP_L
        return COL_BW_RECFG

    dom_stack = ["tp", "pp", "dp"]  # bottom -> top

    def _active_domain_from_bw(bw: Dict[str, float]) -> str:
        # Prefer the highest share; break ties by dom_stack order.
        best = dom_stack[0]
        best_v = float(bw.get(best, 0.0))
        for d in dom_stack[1:]:
            v = float(bw.get(d, 0.0))
            if v > best_v:
                best = d
                best_v = v
        return best

    def _domain_y_span(
        y_base: float, h_total: float, bw: Dict[str, float], domain: str
    ) -> Tuple[float, float]:
        """Return (y0,y1) vertical span for a domain slice within a stacked BW bar."""
        s = (
            float(bw.get("tp", 0.0))
            + float(bw.get("pp", 0.0))
            + float(bw.get("dp", 0.0))
        )
        if s <= 0:
            s = 1.0
        cum = 0.0
        for d in dom_stack:
            frac = float(bw.get(d, 0.0)) / s
            if d == domain:
                y0 = y_base + h_total * cum
                y1 = y_base + h_total * (cum + frac)
                return y0, y1
            cum += frac
        return y_base, y_base + h_total

    for row_i, (row_label, evs) in enumerate(plot_rows):
        # Top-to-bottom order: first row at the top.
        y = float((nrows - 1 - row_i) * row_step)
        # row label
        ax.text(
            -0.01 * max(1.0, total_ms),
            y + h / 2,
            row_label,
            ha="right",
            va="center",
            fontsize=10,
            color="#333333",
        )
        # Completion-time annotation (total = makespan; show comm vs reconfig breakdown).
        if evs:
            makespan = max(float(e.end_ms) for e in evs)
            comm_ms = sum(float(e.duration_ms) for e in evs if e.kind == "comm")
            rc_ms = sum(float(e.duration_ms) for e in evs if e.kind != "comm")
            ax.text(
                total_ms + pad_ms,
                y + h / 2,
                f"total={makespan:.1f}ms  (comm {comm_ms:.1f} + R {rc_ms:.1f})",
                ha="left",
                va="center",
                fontsize=9,
                color="#222222",
            )
        for idx, e in enumerate(evs):
            if e.kind == "link_internal":
                # Internal retune: full-height hatched block (not domain-colored).
                rect = patches.Rectangle(
                    (e.start_ms, y),
                    e.duration_ms,
                    h,
                    linewidth=0.8,
                    edgecolor="white",
                    facecolor=COL_LINK_RECFG,
                    hatch="///",
                    alpha=0.9,
                )
                ax.add_patch(rect)
            elif e.kind != "comm":
                # Keep reconfiguration cost as a full-height block (separate semantic from BW usage).
                reconfig_color = (
                    COL_BW_RECFG if e.kind == "bw_reconfig" else COL_LINK_RECFG
                )
                rect = patches.Rectangle(
                    (e.start_ms, y),
                    e.duration_ms,
                    h,
                    linewidth=0.8,
                    edgecolor="white",
                    facecolor=reconfig_color,
                )
                ax.add_patch(rect)
            else:
                # Bandwidth-sliced view: stack TP/PP/DP vertically by bw share.
                bw = e.bw_share or {}
                # Normalize defensively (should already sum to 1.0).
                s = (
                    float(bw.get("tp", 0.0))
                    + float(bw.get("pp", 0.0))
                    + float(bw.get("dp", 0.0))
                )
                if s <= 0:
                    s = 1.0
                y_off = 0.0
                for d in dom_stack:
                    frac = float(bw.get(d, 0.0)) / s
                    hh = h * max(0.0, frac)
                    if hh <= 0:
                        continue
                    rect = patches.Rectangle(
                        (e.start_ms, y + y_off),
                        e.duration_ms,
                        hh,
                        linewidth=0.8,
                        edgecolor="white",
                        facecolor=_col(d, active=(d == e.domain)),
                    )
                    ax.add_patch(rect)
                    y_off += hh

            # Keep labels sparse to avoid clutter.
            if e.kind != "comm":
                if e.duration_ms >= 0.6:
                    if e.kind == "bw_reconfig":
                        label = "R"
                    elif e.kind == "link_reconfig":
                        label = "L"
                    else:
                        label = "L*"
                    ax.text(
                        e.start_ms + e.duration_ms / 2,
                        y + h / 2,
                        label,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white",
                    )
                continue

            lbl = e.label
            if e.domain in ["tp", "dp"]:
                if e.duration_ms >= 5.0:
                    # Place label in the center of the active domain's vertical slice.
                    bw = e.bw_share or {}
                    y0, y1 = _domain_y_span(y, h, bw, e.domain)
                    y_mid = (y0 + y1) / 2.0
                    ax.text(
                        e.start_ms + e.duration_ms / 2,
                        y_mid,
                        lbl,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white",
                    )
            else:
                # PP: label every Nth PP block (after coalescing).
                if (
                    pp_label_every > 0
                    and ((idx + 1) % pp_label_every == 0)
                    and e.duration_ms >= 2.0
                ):
                    bw = e.bw_share or {}
                    y0, y1 = _domain_y_span(y, h, bw, e.domain)
                    y_mid = (y0 + y1) / 2.0
                    ax.text(
                        e.start_ms + e.duration_ms / 2,
                        y_mid,
                        lbl,
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white",
                    )

    ax.set_xlim(0, max(1e-6, total_ms + 3.0 * pad_ms))
    ax.set_ylim(-0.4, float((nrows - 1) * row_step) + h + 0.35)
    ax.set_yticks([])
    ax.set_xlabel("Time (ms)")
    for spine in ax.spines.values():
        spine.set_visible(False)

    note = ""
    if float(min_marker_ms) > 0:
        note = f"; short events (<{float(min_marker_ms):g}ms) marked with thick tick"
    fig.suptitle(title + note, fontsize=12, y=0.985)

    if params_text.strip():
        # Parameter block below the title (outside the plot area).
        fig.text(
            0.01,
            0.945,
            params_text,
            ha="left",
            va="top",
            fontsize=8,
            family="monospace",
            color="#222222",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="#dddddd",
                alpha=0.9,
            ),
        )
    legend = [
        patches.Patch(color=COL_BW_RECFG, label="BW reconfig (R)"),
        patches.Patch(color=COL_LINK_RECFG, label="Link reconfig (L)"),
        patches.Patch(
            facecolor="none",
            edgecolor=COL_LINK_RECFG,
            hatch="///",
            label="Internal retune (overlay)",
        ),
        patches.Patch(color=COL_TP, label="TP BW slice (dark=active, light=idle)"),
        patches.Patch(color=COL_PP, label="PP BW slice (dark=active, light=idle)"),
        patches.Patch(color=COL_DP, label="DP BW slice (dark=active, light=idle)"),
    ]
    ax.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        frameon=False,
        fontsize=9,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if params_text.strip():
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    else:
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def try_plot_trace_png(
    out_path: Path,
    strategy_to_events: Dict[str, List[TraceEvent]],
    title: str,
    pp_label_every: int,
    min_marker_ms: float,
    ms_per_inch: float,
    max_width_in: float,
    params_text: str,
) -> bool:
    """Return True if plot was written; False if matplotlib unavailable."""
    row_order = ["preplanned", "fast-preplanned", "one-shot", "static"]
    row_names = {
        "preplanned": "Preplanned",
        "fast-preplanned": "Fast preplanned",
        "one-shot": "One-shot",
        "static": "Even share",
    }
    rows: List[Tuple[str, List[TraceEvent]]] = [
        (row_names[k], strategy_to_events.get(k, [])) for k in row_order
    ]
    return try_plot_rows_png(
        out_path=out_path,
        rows=rows,
        title=title,
        pp_label_every=pp_label_every,
        min_marker_ms=min_marker_ms,
        ms_per_inch=ms_per_inch,
        max_width_in=max_width_in,
        params_text=params_text,
    )
