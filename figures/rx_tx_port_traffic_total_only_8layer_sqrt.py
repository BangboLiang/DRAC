import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from drac_eval.directional_traffic import RAW_PHYSICAL_COUNTERS

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 19,
    "axes.labelsize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
})

groups = [
    "Tensor Parallelism",
    "Data Parallelism",
]

physical_rx = [RAW_PHYSICAL_COUNTERS[key]["opposite_direction_bytes_raw"] for key in ("TP", "DP")]
physical_tx = [RAW_PHYSICAL_COUNTERS[key]["main_direction_bytes_raw"] for key in ("TP", "DP")]

iterations = [RAW_PHYSICAL_COUNTERS[key]["iterations"] for key in ("TP", "DP")]
layer_multiplier = [1, 8]  # keep DP at 8 layers

# Normalize using total physical traffic only
norm_rx = [p / it * m for p, it, m in zip(physical_rx, iterations, layer_multiplier)]
norm_tx = [p / it * m for p, it, m in zip(physical_tx, iterations, layer_multiplier)]

MiB = 1024**2
rx_mib = np.array([x / MiB for x in norm_rx], dtype=float)
tx_mib = np.array([x / MiB for x in norm_tx], dtype=float)

x = np.arange(len(groups))
width = 0.36

fig, ax = plt.subplots(figsize=(10.8, 6.4))

ax.bar(x - width/2, rx_mib, width, label="Rx", color="#ACD6EC")
ax.bar(x + width/2, tx_mib, width, label="Tx", color="#F5A889")

# Sqrt scale to preserve visibility while keeping bars short
ax.set_yscale("function", functions=(np.sqrt, lambda y: y**2))
ax.grid(False)

ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylabel("Traffic per iteration (MiB, sqrt scale)")
ax.set_title("Rx/Tx port traffic per iteration\n(total traffic, DP counted for 8 layers)")
ax.legend(loc="upper left", frameon=True)

for i in range(len(groups)):
    ax.text(x[i] - width/2, rx_mib[i] * 1.05, f"{rx_mib[i]:.2f}", ha="center", va="bottom", fontsize=14)
    ax.text(x[i] + width/2, tx_mib[i] * 1.05, f"{tx_mib[i]:.2f}", ha="center", va="bottom", fontsize=14)

ax.set_ylim(0, tx_mib.max() * 1.25)

fig.tight_layout()

out = Path("./rx_tx_port_traffic_total_only_8layer_sqrt.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)

print(out)
print("Per-iteration totals (MiB):")
for i, g in enumerate(groups):
    print(g)
    print(f"  Rx total: {rx_mib[i]:.6f}")
    print(f"  Tx total: {tx_mib[i]:.6f}")
