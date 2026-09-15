#!/usr/bin/env python3
"""Generate block LER vs physical error rate curves for the paper.

Plots 10 codes from CSS MILP, non-CSS, missing simulation data, and
the top trusted PBB FOM follow-up run.
Single-column width, log-scale y-axis, dashed LER=p diagonal.
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent

# --- Load data ---
with open(root / "results" / "threshold_simulation_milp.json") as f:
    milp_data = json.load(f)

with open(root / "results" / "threshold_simulation_noncss.json") as f:
    noncss_data = json.load(f)

with open(root / "results" / "threshold_simulation_missing.json") as f:
    missing_data = json.load(f)

# Top TRUSTED PBB at (30,6) with FOM <= 19.2; curve from
# tests/threshold_simulation_360_12_24.py.
with open(root / "results" / "threshold_simulation_360_12_24.json") as f:
    extra_data = [json.load(f)]

# Build lookup by label
by_label = {}
for d in milp_data + noncss_data + missing_data + extra_data:
    by_label[d["label"]] = d

# --- Select codes to plot ---
codes_to_plot = [
    # CSS codes (solid lines)
    ("[[144,12,12]] gross code (control)", "Gross [[144,12,12]]", "#000000", "-", "o"),
    ("[[288,24,12]] MILP d=12, FOM=12.0", "[[288,24,12]]", "#0072B2", "-", "s"),
    ("[[288,16,12]] MILP d=12, FOM=8.0", "[[288,16,12]]", "#009E73", "-", "^"),
    ("[[288,50,8]] MILP d=8, FOM=11.1 (cross-factored)", "[[288,50,8]]", "#E69F00", "-", "X"),
    ("[[144,8,12]] MILP d=12, FOM=8.0 (mixed-monomial)", "[[144,8,12]]", "#882255", "-", "d"),
    ("[[360,20,<=14]] MILP d<=14, FOM<=10.9 (C4)", r"[[360,20,$\leq$14]]", "#CC79A7", "-", "D"),
    # Non-CSS PBB codes (dashed lines)
    ("[[144,12,12]] PBB (MILP exact, x/y-swap base)", "PBB [[144,12,12]]", "#D55E00", "--", "v"),
    ("[[72,4,8]] PBB (MILP exact, mixed base)", "PBB [[72,4,8]]", "#56B4E9", "--", "P"),
    ("[[360,12,<=20]] PBB (best PBB FOM)", r"PBB [[360,12,$\leq$20]]", "#44AA99", "--", "h"),
    ("[[360,12,<=24]] PBB (top TRUSTED FOM)", r"PBB [[360,12,$\leq$24]]", "#117733", "--", "*"),
]

# --- Plot ---
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(3.4, 3.2))

for label, display, color, ls, marker in codes_to_plot:
    d = by_label[label]
    ps = np.array([e["p"] for e in d["error_rates"]])
    lers = np.array([e["logical_error_rate"] for e in d["error_rates"]])
    # Filter out zero LER points for log scale
    mask = lers > 0
    ax.plot(ps[mask], lers[mask], ls, color=color, marker=marker,
            markersize=3.5, linewidth=1.2, label=display, zorder=3)

# Dashed diagonal LER = p
p_range = np.linspace(0.002, 0.08, 100)
ax.plot(p_range, p_range, "k--", linewidth=0.8, alpha=0.4, label="LER $= p$", zorder=1)

ax.set_yscale("log")
ax.set_xlabel("Physical error rate $p$", fontsize=8)
ax.set_ylabel("Block logical error rate", fontsize=8)
ax.tick_params(labelsize=7)
ax.set_xlim(0, 0.085)
ax.set_ylim(1e-5, 1)
ax.legend(fontsize=4.5, loc="lower right", framealpha=0.95, edgecolor="#cccccc",
          handlelength=2.0, ncol=1)
ax.grid(True, alpha=0.15, linewidth=0.5, color="#888888")
ax.set_axisbelow(True)

fig.tight_layout(pad=0.5)
fig.savefig("fig_ler_curves.pdf", bbox_inches="tight", dpi=300)
fig.savefig("fig_ler_curves.png", bbox_inches="tight", dpi=300)
print("Saved fig_ler_curves.pdf and .png")
