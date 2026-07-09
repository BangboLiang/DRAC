import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

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
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
})

c_payload = "#4E79A7"
c_roce    = "#F28E2B"
c_phy     = "#E15759"
c_cts     = "#59A14F"
c_other   = "#B07AA1"

fig, ax = plt.subplots(figsize=(11.2, 5.2))

y_tx = 1
y_rx = 0
h = 0.52

ax.barh(y_tx, tx_payload, height=h, color=c_payload, edgecolor="none", label="Payload")
ax.barh(y_tx, tx_roce, left=tx_payload, height=h, color=c_roce, edgecolor="none", label="RoCEv2 overhead")
ax.barh(y_tx, tx_phy, left=tx_payload + tx_roce, height=h, color=c_phy, edgecolor="none", label="PHY-only overhead")

ax.barh(y_rx, rx_cts, height=h, color=c_cts, edgecolor="none", label="CTS message")
ax.barh(y_rx, rx_other, left=rx_cts, height=h, color=c_other, edgecolor="none", label="Other RoCEv2 control")

ax.set_yticks([y_tx, y_rx])
ax.set_yticklabels(["Tx composition", "Rx composition"])
ax.set_xlabel("Traffic per iteration (MiB, sqrt scale)")
ax.set_title("Traffic decomposition in DP gradient synchronization")
ax.set_xscale("function", functions=(np.sqrt, lambda x: x**2))
ax.grid(axis="x", linestyle="--", alpha=0.35)

ax.set_xlim(0, tx_total * 1.28)

ax.text(tx_total + tx_total * 0.015, y_tx, f"{tx_total:.2f} MiB",
        va="center", ha="left", fontsize=12)
ax.text(rx_total + tx_total * 0.015, y_rx, f"{rx_total:.2f} MiB",
        va="center", ha="left", fontsize=12)

ax.text(
    0.98, 0.08,
    f"Tx/Rx ≈ {ratio:.1f}×",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7")
)

handles, labels = ax.get_legend_handles_labels()
seen = {}
for hnd, lab in zip(handles, labels):
    if lab not in seen:
        seen[lab] = hnd

ax.legend(
    seen.values(),
    seen.keys(),
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    frameon=True
)

fig.subplots_adjust(bottom=0.28, top=0.88)

out = Path("./dp_grad_sync_decomposition_horizontal.png")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)

print(out)