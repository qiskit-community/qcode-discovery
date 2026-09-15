#!/usr/bin/env python3
"""Generate 3-panel fitness trajectory figure for the paper.

Plots per-iteration combined_score and running best across all three
evolution campaigns:
  - Campaign 1: Gemini Flash (100 iterations) — hardcoded data
  - Campaign 2: Ensemble, 251 iterations — parsed from log
  - Campaign 3: Ensemble, 500 iterations — parsed from log
"""

import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- Campaign 1: Gemini Flash (hardcoded from original script) ---
campaign1_data = [
    (0, 193.97), (1, 236.81), (2, 212.93), (3, 186.96), (4, 174.57),
    (5, 213.96), (6, 237.50), (7, 199.53), (8, 230.58), (9, 228.37),
    (10, 227.05), (11, 165.43), (12, 200.79), (13, 234.32), (14, 198.63),
    (15, 216.82), (16, 172.61), (17, 196.49), (18, 273.42), (19, 224.63),
    (20, 168.47), (21, 202.09), (22, 162.66), (23, 247.19), (24, 0.00),
    (25, 205.17), (26, 189.22), (27, 263.13), (28, 176.39), (29, 139.84),
    (30, 0.00), (31, 206.70), (32, 229.73), (33, 222.81), (34, 225.42),
    (35, 233.65), (36, 225.91), (37, 244.71), (38, 204.62), (39, 212.10),
    (40, 248.33), (41, 308.70), (42, 242.09), (43, 224.93), (44, 254.88),
    (45, 229.91), (46, 247.78), (47, 221.56), (48, 277.35), (49, 280.21),
    (50, 240.79), (51, 0.00), (52, 271.45), (53, 291.25), (54, 256.07),
    (55, 183.46), (56, 252.19), (57, 296.44), (58, 245.88), (59, 311.66),
    (60, 179.82), (61, 206.73), (62, 279.65), (63, 318.48), (64, 274.96),
    (65, 240.19), (66, 255.70), (67, 201.77), (68, 299.48), (69, 297.98),
    (70, 234.18), (71, 285.22), (72, 215.01), (73, 212.35), (74, 309.33),
    (75, 302.37), (76, 225.84), (77, 254.55), (78, 299.32), (79, 226.33),
    (80, 234.10), (81, 263.87), (82, 295.28), (83, 326.99), (84, 295.76),
    (85, 297.80), (86, 255.98), (87, 244.38), (88, 166.89), (89, 297.53),
    (90, 303.00), (91, 295.24), (92, 269.90), (93, 300.93), (94, 256.19),
    (95, 203.66), (96, 318.48), (97, 224.22), (98, 290.65), (99, 292.56),
    (100, 309.75),
]


def parse_log_scores(log_path):
    """Extract per-iteration combined_score from an OpenEvolve log."""
    pattern = re.compile(r"Metrics: combined_score=([\d.]+)")
    scores = []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                scores.append(float(m.group(1)))
    return scores


def compute_running_best(scores):
    """Compute running best, ignoring zeros (failed evaluations)."""
    running = np.zeros(len(scores))
    best = 0.0
    for i, s in enumerate(scores):
        if s > best:
            best = s
        running[i] = best
    return running


# --- Parse ensemble logs ---
root = Path(__file__).resolve().parent.parent.parent
log2 = root / "results/evolution/run_20260219_203003/logs/openevolve_20260219_203006.log"
log3 = root / "results/evolution/run_20260220_060158/logs/openevolve_20260220_060202.log"

campaign2_scores = parse_log_scores(log2)
campaign3_scores = parse_log_scores(log3)

# --- Prepare data ---
campaigns = [
    {
        "title": "Campaign 1: Gemini Flash\n(100 iterations, pop=100)",
        "iterations": np.array([d[0] for d in campaign1_data]),
        "scores": np.array([d[1] for d in campaign1_data]),
    },
    {
        "title": "Campaign 2: 3-Model Ensemble\n(251 iterations, pop=100)",
        "iterations": np.arange(len(campaign2_scores)),
        "scores": np.array(campaign2_scores),
    },
    {
        "title": "Campaign 3: 3-Model Ensemble\n(500 iterations, pop=1000)",
        "iterations": np.arange(len(campaign3_scores)),
        "scores": np.array(campaign3_scores),
    },
]

# --- Color palette (colorblind-friendly, Okabe-Ito inspired) ---
CLR_SCATTER = "#56B4E9"   # sky blue
CLR_BEST    = "#D55E00"   # vermillion
CLR_ANNOT   = "#404040"   # dark gray

# --- Plot ---
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8), sharey=True)

for ax, camp in zip(axes, campaigns):
    iters = camp["iterations"]
    scores = camp["scores"]
    running = compute_running_best(scores)

    # Individual scores (exclude zeros)
    valid = scores > 0
    ax.scatter(
        iters[valid], scores[valid],
        s=8, alpha=0.30, color=CLR_SCATTER, zorder=2,
        edgecolors="none",
    )

    # Running best
    ax.step(iters, running, where="post", color=CLR_BEST, linewidth=1.5, zorder=3)

    ax.set_title(camp["title"], fontsize=7, pad=4)
    ax.set_xlabel("Iteration", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.15, linewidth=0.5, color="#888888")
    ax.set_axisbelow(True)

    # Annotate final best
    final_best = running[-1]
    ax.annotate(
        f"{final_best:.0f}",
        xy=(iters[-1], final_best),
        xytext=(-15, 8),
        textcoords="offset points",
        fontsize=6.5, color=CLR_BEST, fontweight="bold",
    )

axes[0].set_ylabel("Combined score", fontsize=8)
axes[0].set_ylim(100, 380)

fig.tight_layout(pad=0.4, w_pad=0.8)
fig.savefig("fig_fitness_trajectory.pdf", bbox_inches="tight", dpi=300)
fig.savefig("fig_fitness_trajectory.png", bbox_inches="tight", dpi=300)
print("Saved fig_fitness_trajectory.pdf and .png")
