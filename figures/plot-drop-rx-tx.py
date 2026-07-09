import matplotlib.pyplot as plt
from pathlib import Path

# Per-iteration decomposition
MiB = 1024**2
tx_payload = 79_069_184_000 / 100 / MiB
tx_roce = 5_015_313_872 / 100 / MiB
tx_phy = 325_501_306 / 100 / MiB

rx_cts = 4_832_000 / 100 / MiB
rx_other = 593_685_400 / 100 / MiB

tx_total = tx_payload + tx_roce + tx_phy
rx_total = rx_cts + rx_other
ratio = tx_total / rx_total

plt.rcParams.update({
    "figure.dpi": 180,
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
})

fig = plt.figure(figsize=(11.2, 4.8))
gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 2.4], wspace=0.05)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)

y_tx, y_rx = 1, 0
h = 0.48

# left panel: show full Rx + beginning of Tx for context
ax1.barh(y_tx, tx_payload, height=h, edgecolor="black", label="Payload")
ax1.barh(y_tx, tx_roce, left=tx_payload, height=h, edgecolor="black", label="RoCEv2 overhead")
ax1.barh(y_tx, tx_phy, left=tx_payload + tx_roce, height=h, edgecolor="black", label="PHY-only overhead")

ax1.barh(y_rx, rx_cts, height=h, edgecolor="black", label="CTS message")
ax1.barh(y_rx, rx_other, left=rx_cts, height=h, edgecolor="black", label="Other RoCEv2 control")

# right panel: only the high Tx range
ax2.barh(y_tx, tx_payload, height=h, edgecolor="black")
ax2.barh(y_tx, tx_roce, left=tx_payload, height=h, edgecolor="black")
ax2.barh(y_tx, tx_phy, left=tx_payload + tx_roce, height=h, edgecolor="black")
ax2.barh(y_rx, rx_cts, height=h, edgecolor="black")
ax2.barh(y_rx, rx_other, left=rx_cts, height=h, edgecolor="black")

# limits
ax1.set_xlim(0, 12)
ax2.set_xlim(735, 840)

# clean broken axis look
ax1.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax2.tick_params(left=False, labelleft=False, right=False, labelright=False)
ax1.tick_params(right=False)

d = 0.015
kwargs = dict(transform=ax1.transAxes, color="black", clip_on=False, linewidth=1.2)
ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
kwargs = dict(transform=ax2.transAxes, color="black", clip_on=False, linewidth=1.2)
ax2.plot((-d, +d), (-d, +d), **kwargs)
ax2.plot((-d, +d), (1-d, 1+d), **kwargs)

# labels
ax1.set_yticks([y_tx, y_rx])
ax1.set_yticklabels(["Tx", "Rx"])
ax1.set_xlabel("Traffic per iteration (MiB)")
ax2.set_xlabel("Traffic per iteration (MiB)")
fig.suptitle("Gradient sync ReduceScatter port traffic decomposition per iteration", y=0.98)

# annotations: place cleanly away from bars/legend
ax1.text(rx_total + 0.18, y_rx, f"{rx_total:.2f} MiB", va="center", ha="left", fontsize=12)
ax2.text(tx_total + 1.8, y_tx, f"{tx_total:.2f} MiB", va="center", ha="left", fontsize=12)

# Rx callouts
ax1.annotate(
    f"CTS: {rx_cts:.3f} MiB",
    xy=(rx_cts / 2, y_rx),
    xytext=(1.0, -0.42),
    textcoords="data",
    arrowprops=dict(arrowstyle="-", lw=1),
    ha="left",
    va="center",
)
ax1.annotate(
    f"Other RoCEv2 control: {rx_other:.2f} MiB",
    xy=(rx_cts + rx_other * 0.55, y_rx),
    xytext=(2.1, 0.35),
    textcoords="data",
    arrowprops=dict(arrowstyle="-", lw=1),
    ha="left",
    va="center",
)

# Tx segment labels on right panel
ax2.text(744, y_tx + 0.33, f"Payload: {tx_payload:.2f} MiB", ha="left", va="bottom", fontsize=10)
ax2.text(775, y_tx + 0.18, f"RoCEv2 overhead: {tx_roce:.2f} MiB", ha="left", va="bottom", fontsize=10)
ax2.text(798, y_tx - 0.32, f"PHY-only: {tx_phy:.2f} MiB", ha="left", va="top", fontsize=10)

# legend above, not overlapping title
handles, labels = ax1.get_legend_handles_labels()
seen = {}
for hnd, lab in zip(handles, labels):
    if lab not in seen:
        seen[lab] = hnd
fig.legend(
    seen.values(), seen.keys(),
    ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.93),
    frameon=True
)

# ratio note below
fig.text(0.5, 0.05, f"Tx/Rx ≈ {ratio:.1f}× per iteration", ha="center", fontsize=12)

# more room for title/legend and bottom note
fig.subplots_adjust(top=0.80, bottom=0.16, left=0.08, right=0.98)

out = Path("./dp_grad_sync_decomposition_broken_axis_clean.png")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)

print(out)
