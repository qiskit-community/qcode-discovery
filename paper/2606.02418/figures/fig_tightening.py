#!/usr/bin/env python3
"""Generate distance tightening histogram for the paper.

Single-panel figure: histogram of distance revision (d_evolution - d_150k)
across all 154 Campaigns 1-3 codes.
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent

# --- Data: all 154 codes (evolution d_est vs 150k verified d) ---
with open(root / "results" / "soak_test_publication.json") as f:
    flash_codes = json.load(f)

with open(root / "results" / "ensemble_verification_150k.json") as f:
    ensemble_codes = json.load(f)

all_codes = flash_codes + ensemble_codes

revisions = []
for c in all_codes:
    delta = c["claimed_d"] - c["verified_d"]
    revisions.append(delta)

revisions = np.array(revisions)
n_confirmed = np.sum(revisions == 0)
n_tightened = np.sum(revisions > 0)
n_loosened = np.sum(revisions < 0)
mean_revision = np.mean(revisions[revisions > 0]) if n_tightened > 0 else 0

print(f"Total codes: {len(revisions)}")
print(f"Tightened: {n_tightened} ({100*n_tightened/len(revisions):.1f}%)")
print(f"Confirmed: {n_confirmed} ({100*n_confirmed/len(revisions):.1f}%)")
print(f"Loosened: {n_loosened} ({100*n_loosened/len(revisions):.1f}%)")
print(f"Mean tightening (where >0): {mean_revision:.1f} points")

# --- Color palette ---
CLR_DIFF = "#882255"       # wine
CLR_HIST = "#56B4E9"       # sky blue
CLR_HIST_ZERO = "#E69F00"  # amber for confirmed

# --- Plot ---
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(3.4, 2.6))

max_rev = int(max(revisions)) + 1
bins = np.arange(-4, max_rev + 1) - 0.5

counts, bin_edges, patches = ax.hist(
    revisions, bins=bins, color=CLR_HIST, alpha=0.8,
    edgecolor="white", linewidth=0.5, zorder=2,
)

# Color the zero/negative bins differently
for patch, left_edge in zip(patches, bin_edges[:-1]):
    center = left_edge + 0.5
    if center <= 0:
        patch.set_facecolor(CLR_HIST_ZERO)
        patch.set_alpha(0.8)

ax.set_xlabel("Distance revision ($d_{\\mathrm{evo}} - d_{\\mathrm{150k}}$)", fontsize=8)
ax.set_ylabel("Number of codes", fontsize=8)
ax.tick_params(labelsize=7)
ax.set_title(f"All 154 codes: {n_tightened} tightened, {n_confirmed+n_loosened} confirmed",
             fontsize=7.5, pad=4)
ax.grid(True, axis="y", alpha=0.15, linewidth=0.5, color="#888888")
ax.set_axisbelow(True)

ax.annotate(
    f"{100*n_tightened/len(revisions):.1f}% tightened\nmean \u0394d = {mean_revision:.1f}",
    xy=(0.97, 0.95), xycoords="axes fraction",
    ha="right", va="top", fontsize=7,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=CLR_DIFF, alpha=0.9, lw=0.6),
)

fig.tight_layout(pad=0.5)
fig.savefig("fig_tightening.pdf", bbox_inches="tight", dpi=300)
fig.savefig("fig_tightening.png", bbox_inches="tight", dpi=300)
print("Saved fig_tightening.pdf and .png")
