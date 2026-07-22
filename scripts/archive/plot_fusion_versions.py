#!/usr/bin/env python3
"""Visualize all ElecFM versions: SMAPE and Spike-F1(head) comparison."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

versions = [
    "R2\nFrozen-QH",
    "R3\nLoRA",
    "V5\nSpikeOnly",
    "V5b\n720h",
    "V6\nCrossNode",
    "V6\nAllGroups",
    "V6\nd32",
    "V6\nAG+d32",
    "V6\nd128",
    "V7\nSwiGLU",
]
colors = [
    "#e67e22", "#f39c12", "#2ecc71", "#1abc9c",
    "#3498db", "#2980b9", "#9b59b6", "#8e44ad", "#34495e", "#e91e63",
]

# SMAPE per window
smape_w1 = [29.19, 29.39, 27.55, 27.52, 28.90, 28.90, 28.90, 28.90, 28.90, 27.55]
smape_w2 = [73.16, 74.80, 75.92, 79.17, 77.55, 77.55, 77.55, 77.55, 77.55, 75.92]
smape_w3 = [69.17, 72.83, 65.54, 64.85, 64.30, 64.30, 64.30, 64.30, 64.30, 65.54]

# Spike-F1(head) per window
f1_w1 = [0.335, 0.351, 0.416, 0.378, 0.444, 0.384, 0.397, 0.372, 0.382, 0.398]
f1_w2 = [0.101, 0.160, 0.183, 0.149, 0.190, 0.180, 0.126, 0.178, 0.200, 0.198]
f1_w3 = [0.287, 0.257, 0.230, 0.265, 0.181, 0.361, 0.192, 0.249, 0.095, 0.198]

# Zero-shot baselines
zs_smape = [27.67, 75.71, 64.07]
zs_signal_f1 = [0.329, 0.0, 0.660]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("ElecFM: All Versions Comparison", fontsize=16, fontweight="bold", y=0.98)

x = np.arange(len(versions))
bw = 0.6

# Row 1: SMAPE
for ax, data, title, bl in zip(
    axes[0],
    [smape_w1, smape_w2, smape_w3],
    ["W1 Stable - SMAPE (lower=better)", "W2 Negative - SMAPE (lower=better)", "W3 Extreme - SMAPE (lower=better)"],
    zs_smape
):
    bars = ax.bar(x, data, bw, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axhline(y=bl, color="red", linestyle="--", lw=1.5, label=f"Zero-shot = {bl}")
    ax.set_title(title, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(versions, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("SMAPE (%)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for i, bar in enumerate(bars):
        if versions[i].startswith("V5\nSpike"):
            bar.set_edgecolor("green")
            bar.set_linewidth(2.5)

# Row 2: Spike-F1(head)
for ax, data, title, bl in zip(
    axes[1],
    [f1_w1, f1_w2, f1_w3],
    ["W1 Stable - Spike-F1 head (higher=better)", "W2 Negative - Spike-F1 head (higher=better)", "W3 Extreme - Spike-F1 head (higher=better)"],
    zs_signal_f1
):
    bars = ax.bar(x, data, bw, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axhline(y=bl, color="red", linestyle="--", lw=1.5, label=f"Zero-shot signal = {bl}")
    ax.set_title(title, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(versions, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Spike-F1")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for i, bar in enumerate(bars):
        if versions[i].startswith("V5\nSpike"):
            bar.set_edgecolor("green")
            bar.set_linewidth(2.5)
    best_idx = np.argmax(data)
    ax.annotate(f"{data[best_idx]:.3f}", xy=(best_idx, data[best_idx]),
                xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=8, fontweight="bold", color="darkblue")

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = "/Users/wanghaochen/school/data/results/fusion_all_versions_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Done: {out}")
