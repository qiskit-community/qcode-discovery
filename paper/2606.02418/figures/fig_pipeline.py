#!/usr/bin/env python3
"""Generate the evolution pipeline diagram for the paper.

Uses matplotlib patches and arrows to create a flowchart.
Single-column width (~3.4 in).

Key distinction: Campaigns 1-3 use BP-OSD only in-loop, with MILP post-hoc.
Campaigns 4-5 integrate MILP into the evaluation cascade itself.
Post-campaign verification applies to all campaigns.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MPath

fig, ax = plt.subplots(figsize=(3.4, 6.0))
ax.set_xlim(0, 10)
ax.set_ylim(0, 16)
ax.axis("off")

# --- Colors ---
CLR_LLM = "#4477AA"      # blue
CLR_EVAL = "#228833"      # green
CLR_ARCHIVE = "#EE6677"   # red/coral
CLR_MILP = "#CCBB44"      # yellow
CLR_POST = "#AA3377"       # purple — post-campaign verification
CLR_TEXT = "#333333"
CLR_LIGHT = "#F0F0F0"
CLR_ANNOT = "#666666"

def add_box(ax, x, y, w, h, text, color, fontsize=6.5, textcolor="white"):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.15", linewidth=0.8,
                         edgecolor=color, facecolor=color, alpha=0.85,
                         zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=textcolor, fontweight="bold", zorder=4)

def add_arrow(ax, x1, y1, x2, y2, color="#555555", style="-|>", lw=1.0):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=style, mutation_scale=10,
                            linewidth=lw, color=color, zorder=2)
    ax.add_patch(arrow)

# --- Vertical positions ---
Y_SEED = 15.2
Y_LLM = 13.8
Y_MUT = 12.4
Y_S1 = 10.8
Y_S2 = 9.2
Y_MILP_INLINE = 7.6
Y_ARCHIVE = 6.0
Y_POST = 3.6
Y_VERIFIED = 1.6

# --- Main flow ---
add_box(ax, 5, Y_SEED, 3.6, 0.7, "Seed ansatz $G(\\ell, m)$", CLR_LLM)
add_box(ax, 5, Y_LLM, 3.6, 0.7, "LLM proposes diff", CLR_LLM)
add_box(ax, 5, Y_MUT, 3.6, 0.7, "Mutant ansatz $G'(\\ell, m)$", CLR_LLM)

add_box(ax, 5, Y_S1, 4.4, 0.9,
        "Stage 1: $k$-only screening\n(6,6), (12,6)  [$\\sim$2 s]",
        CLR_EVAL, fontsize=6)

add_box(ax, 5, Y_S2, 4.4, 0.9,
        "Stage 2: BP-OSD top-10\n8 lattices  [$\\sim$30\u201360 s]",
        CLR_EVAL, fontsize=6)

add_box(ax, 5, Y_MILP_INLINE, 4.4, 0.9,
        "Stage 3: MILP exact distance\nCampaigns 4\u20135 only",
        CLR_MILP, fontsize=6, textcolor=CLR_TEXT)

add_box(ax, 5, Y_ARCHIVE, 4.0, 0.7, "MAP-Elites archive", CLR_ARCHIVE)

# Main flow arrows
add_arrow(ax, 5, Y_SEED - 0.35, 5, Y_LLM + 0.35)
add_arrow(ax, 5, Y_LLM - 0.35, 5, Y_MUT + 0.35)
add_arrow(ax, 5, Y_MUT - 0.35, 5, Y_S1 + 0.45)
add_arrow(ax, 5, Y_S1 - 0.45, 5, Y_S2 + 0.45)
add_arrow(ax, 5, Y_S2 - 0.45, 5, Y_MILP_INLINE + 0.45)
add_arrow(ax, 5, Y_MILP_INLINE - 0.45, 5, Y_ARCHIVE + 0.35)

# Bypass arrow: Campaigns 1-3 skip MILP stage (right side)
verts_bypass = [(7.5, Y_S2 - 0.45), (9.2, Y_S2 - 0.8),
                (9.2, Y_ARCHIVE + 0.6), (7.4, Y_ARCHIVE + 0.35)]
codes_bypass = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4]
arrow_bypass = FancyArrowPatch(path=MPath(verts_bypass, codes_bypass),
                               arrowstyle="-|>", mutation_scale=10,
                               linewidth=1.0, color=CLR_ANNOT,
                               linestyle="--", zorder=2)
ax.add_patch(arrow_bypass)
ax.text(9.8, Y_MILP_INLINE, "C1\u20133", ha="center", va="center",
        fontsize=5.5, color=CLR_ANNOT, fontstyle="italic")

# Feedback loop: archive -> LLM (left side)
verts = [(2.5, Y_ARCHIVE), (0.5, Y_ARCHIVE), (0.5, Y_LLM), (2.8, Y_LLM)]
codes_path = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4]
arrow_fb = FancyArrowPatch(path=MPath(verts, codes_path),
                           arrowstyle="-|>", mutation_scale=10,
                           linewidth=1.2, color=CLR_ARCHIVE, zorder=2)
ax.add_patch(arrow_fb)
ax.text(0.05, (Y_ARCHIVE + Y_LLM) / 2, "feedback", ha="center", va="center",
        fontsize=5.5, color=CLR_ARCHIVE, fontstyle="italic", rotation=90)

# Filter annotation on Stage 1
ax.text(7.8, Y_S1 + 0.35, "$\\sim$30%\nfiltered", ha="center", va="center",
        fontsize=5, color=CLR_ANNOT, fontstyle="italic")

# --- Post-campaign verification block ---
add_box(ax, 5, Y_POST, 4.8, 2.0,
        "Post-campaign verification\n"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "MILP exact distance (all camps.)\n"
        "BLISS deduplication\n"
        "Tanner-graph decomposability\n"
        "LC equivalence (PBB)",
        CLR_POST, fontsize=5.5, textcolor="white")

add_arrow(ax, 5, Y_ARCHIVE - 0.35, 5, Y_POST + 1.0,
          color=CLR_POST, lw=1.2)

# --- Verified catalog output ---
add_box(ax, 5, Y_VERIFIED, 4.0, 0.7,
        "Verified code catalog",
        "#555555", fontsize=6.5, textcolor="white")
add_arrow(ax, 5, Y_POST - 1.0, 5, Y_VERIFIED + 0.35,
          color="#555555", lw=1.0)

# Campaign label
ax.text(5, 0.5, "CSS: Campaigns 1\u20134  |  PBB: Campaign 5",
        ha="center", va="center", fontsize=6, color=CLR_ANNOT,
        fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.3", fc=CLR_LIGHT, ec="#cccccc", lw=0.5))

fig.tight_layout(pad=0.3)
fig.savefig("fig_pipeline.pdf", bbox_inches="tight", dpi=300)
fig.savefig("fig_pipeline.png", bbox_inches="tight", dpi=300)
print("Saved fig_pipeline.pdf and .png")
