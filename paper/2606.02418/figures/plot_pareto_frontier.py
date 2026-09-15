#!/usr/bin/env python3
"""Generate the MILP Pareto frontier figure for CSS + non-CSS codes.

Plots k vs d and FOM vs n for both CSS BB codes and non-CSS PBB codes,
with exact/trusted distinction for non-CSS. Requested by R2 and R3.

Usage::
    uv run python paper/2606.02418/figures/plot_pareto_frontier.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent


def classify_structure(A_terms, B_terms):
    """Classify a code's structural family from its polynomial terms."""
    A_x = [a for a, b in A_terms if b == 0 and a != 0]
    A_y = [b for a, b in A_terms if a == 0 and b != 0]
    A_const = [(a, b) for a, b in A_terms if a == 0 and b == 0]
    B_x = [a for a, b in B_terms if b == 0 and a != 0]
    B_y = [b for a, b in B_terms if a == 0 and b != 0]
    B_const = [(a, b) for a, b in B_terms if a == 0 and b == 0]

    # Check for mixed monomials (cross terms x^a * y^b with a>0 and b>0)
    A_mixed = [(a, b) for a, b in A_terms if a > 0 and b > 0]
    B_mixed = [(a, b) for a, b in B_terms if a > 0 and b > 0]
    if A_mixed or B_mixed:
        return "mixed"

    # Check if univariate (one poly only x-terms, other only y-terms)
    # Or constant-monomial (has constant term 1)
    if A_const or B_const:
        return "constant"

    # x/y-swap: A has (x,0) + (0,y) terms, B has (0,y) + (x,0) terms
    if len(A_x) >= 1 and len(A_y) >= 1 and len(B_x) >= 1 and len(B_y) >= 1:
        return "xy-swap"

    # Univariate: A in one variable, B in the other
    if (len(A_x) == len(A_terms) or len(A_y) == len(A_terms)) and \
       (len(B_x) == len(B_terms) or len(B_y) == len(B_terms)):
        return "univariate"

    return "other"


def load_css_codes():
    """Load MILP-verified CSS codes from ilp_catalog.json + campaign6 (C4)."""
    codes = []

    # Campaigns 1-3: trinomial codes from ilp_catalog.json
    # Deduplicate to unique (n, k, d) triples.
    seen = set()
    with open(ROOT / "results" / "ilp_catalog.json") as f:
        data = json.load(f)
    for nkey in ["n=144", "n=288", "n=360"]:
        for c in data[nkey]:
            n = 2 * c["ell"] * c["m"]
            d = c["ilp_d"]
            k = c["k"]
            if d > 0 and k > 0:
                key = (n, k, d)
                if key in seen:
                    continue
                seen.add(key)
                family = classify_structure(c["A"], c["B"])
                codes.append({
                    "n": n, "k": k, "d": d,
                    "fom": k * d * d / n,
                    "family": family,
                })

    # Campaign 4: mixed-monomial codes from campaign6_milp_verified.jsonl
    # Only include codes with MILP-proven exact distances (d_is_exact=True).
    # Deduplicate to unique (n, k, d) triples (many polynomial representations
    # map to the same code parameters via BLISS Tanner-graph isomorphism).
    c4_path = ROOT / "results" / "campaign6_milp_verified.jsonl"
    if c4_path.exists():
        c4_seen = set()
        with open(c4_path) as f:
            for line in f:
                c = json.loads(line)
                if not c.get("d_is_exact", False):
                    continue
                k = c["k"]
                d = c["d"]
                n = c["n"]
                if d > 0 and k > 0:
                    key = (n, k, d)
                    if key in c4_seen:
                        continue
                    c4_seen.add(key)
                    family = classify_structure(c["A_terms"], c["B_terms"])
                    codes.append({
                        "n": n, "k": k, "d": d,
                        "fom": k * d * d / n,
                        "family": family,
                    })

    return codes


def load_noncss_codes():
    """Load non-CSS PBB codes from campaign7_publication_merged.jsonl."""
    codes = []
    with open(ROOT / "results" / "campaign7_publication_merged.jsonl") as f:
        for line in f:
            c = json.loads(line)
            codes.append({
                "n": c["n"], "k": c["k"], "d": c["d"],
                "fom": c.get("fom", c["k"] * c["d"]**2 / c["n"]),
                "exact": c.get("d_is_exact", c.get("trust_level") == "EXACT"),
            })
    return codes


def plot_kd_panel(ax, codes, bravyi, title, is_css=True):
    """Plot k vs d scatter with Pareto frontier."""
    if is_css:
        family_config = {
            "xy-swap": {"c": "#2196F3", "m": "o", "label": "x/y-swap"},
            "constant": {"c": "#FF9800", "m": "s", "label": "Constant-monomial"},
            "mixed": {"c": "#9C27B0", "m": "D", "label": "Mixed monomial"},
            "univariate": {"c": "#4CAF50", "m": "^", "label": "Univariate"},
            "other": {"c": "#757575", "m": "v", "label": "Other"},
        }
        for family, cfg in family_config.items():
            fam_codes = [c for c in codes if c["family"] == family]
            if not fam_codes:
                continue
            ax.scatter([c["d"] for c in fam_codes],
                       [c["k"] for c in fam_codes],
                       c=cfg["c"], marker=cfg["m"],
                       label=cfg["label"], alpha=0.6, s=40,
                       edgecolors="k", linewidths=0.3, zorder=3)
    else:
        # Non-CSS: color by exact vs trusted
        exact = [c for c in codes if c["exact"]]
        upper = [c for c in codes if not c["exact"]]
        if exact:
            ax.scatter([c["d"] for c in exact],
                       [c["k"] for c in exact],
                       c="#2196F3", marker="o",
                       label=f"PBB exact ({len(exact)})", alpha=0.6, s=40,
                       edgecolors="k", linewidths=0.3, zorder=3)
        if upper:
            ax.scatter([c["d"] for c in upper],
                       [c["k"] for c in upper],
                       c="#FF9800", marker="s",
                       label=f"PBB upper bound ({len(upper)})", alpha=0.45, s=35,
                       edgecolors="k", linewidths=0.3, zorder=2)

    # Bravyi baselines
    ax.scatter([b["d"] for b in bravyi], [b["k"] for b in bravyi],
               c="red", marker="P", s=100,
               label="Bravyi et al.", zorder=5, edgecolors="k", linewidths=0.5)
    for b in bravyi:
        ax.annotate(b["label"], (b["d"], b["k"]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=6, color="red")

    # Pareto frontier
    pareto_points = {}
    for c in codes:
        d = c["d"]
        if d not in pareto_points or c["k"] > pareto_points[d]:
            pareto_points[d] = c["k"]
    if pareto_points:
        frontier_d, frontier_k = [], []
        max_k_seen = 0
        for d in sorted(pareto_points.keys(), reverse=True):
            k = pareto_points[d]
            if k >= max_k_seen:
                max_k_seen = k
                frontier_d.append(d)
                frontier_k.append(k)
        frontier_d.reverse()
        frontier_k.reverse()
        ax.step(frontier_d, frontier_k, where="post", color="gray",
                linestyle="--", alpha=0.5, linewidth=1.5, label="Pareto frontier")

    ax.set_xlabel("MILP distance $d$", fontsize=10)
    ax.set_ylabel("Encoding dimension $k$", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)


def plot_fom_panel(ax, codes, bravyi, title, is_css=True, d_min=4):
    """Plot FOM vs n scatter."""
    if is_css:
        family_config = {
            "xy-swap": {"c": "#2196F3", "m": "o", "label": "x/y-swap"},
            "constant": {"c": "#FF9800", "m": "s", "label": "Constant-monomial"},
            "mixed": {"c": "#9C27B0", "m": "D", "label": "Mixed monomial"},
            "univariate": {"c": "#4CAF50", "m": "^", "label": "Univariate"},
            "other": {"c": "#757575", "m": "v", "label": "Other"},
        }
        for family, cfg in family_config.items():
            fam_codes = [c for c in codes
                         if c["family"] == family and c["d"] >= d_min]
            if not fam_codes:
                continue
            ax.scatter([c["n"] for c in fam_codes],
                       [c["fom"] for c in fam_codes],
                       c=cfg["c"], marker=cfg["m"],
                       label=cfg["label"], alpha=0.6, s=40,
                       edgecolors="k", linewidths=0.3, zorder=3)
    else:
        exact = [c for c in codes if c["exact"] and c["d"] >= d_min]
        upper = [c for c in codes if not c["exact"] and c["d"] >= d_min]
        if exact:
            ax.scatter([c["n"] for c in exact],
                       [c["fom"] for c in exact],
                       c="#2196F3", marker="o",
                       label=f"PBB exact ({len(exact)})", alpha=0.6, s=40,
                       edgecolors="k", linewidths=0.3, zorder=3)
        if upper:
            ax.scatter([c["n"] for c in upper],
                       [c["fom"] for c in upper],
                       c="#FF9800", marker="s",
                       label=f"PBB upper bound ({len(upper)})", alpha=0.45, s=35,
                       edgecolors="k", linewidths=0.3, zorder=2)

    # Bravyi baselines
    ax.scatter([b["n"] for b in bravyi], [b["fom"] for b in bravyi],
               c="red", marker="P", s=100,
               label="Bravyi et al.", zorder=5, edgecolors="k", linewidths=0.5)
    for b in bravyi:
        ax.annotate(b["label"], (b["n"], b["fom"]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=6, color="red")

    # FOM=12 reference line
    ax.axhline(y=12.0, color="red", linestyle=":", alpha=0.4,
               label="FOM = 12 (gross code)")

    ax.set_xlabel("Block length $n$", fontsize=10)
    ax.set_ylabel("Figure of merit $kd^2/n$", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)


def main():
    css_codes = load_css_codes()
    noncss_codes = load_noncss_codes()

    bravyi = [
        {"n": 72, "k": 12, "d": 6, "fom": 6.0, "label": "[[72,12,6]]"},
        {"n": 144, "k": 12, "d": 12, "fom": 12.0, "label": "[[144,12,12]]"},
        {"n": 288, "k": 12, "d": 18, "fom": 13.5, "label": "[[288,12,18]]"},
        {"n": 360, "k": 12, "d": 24, "fom": 19.2, "label": "[[360,12,≤24]]"},
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top row: CSS
    plot_kd_panel(axes[0, 0], css_codes, bravyi,
                  "(a) Rate–distance tradeoff (CSS codes, weight-6 and weight-8)", is_css=True)
    plot_fom_panel(axes[0, 1], css_codes, bravyi,
                   "(b) FOM vs block length (CSS, weight-6 and weight-8, $d \\geq 4$)", is_css=True)

    # Bottom row: non-CSS
    plot_kd_panel(axes[1, 0], noncss_codes, bravyi,
                  "(c) Rate–distance tradeoff (non-CSS PBB codes)", is_css=False)
    plot_fom_panel(axes[1, 1], noncss_codes, bravyi,
                   "(d) FOM vs block length (non-CSS, $d \\geq 6$)",
                   is_css=False, d_min=6)

    plt.tight_layout()
    outpath = ROOT / "paper" / "figures" / "fig_pareto_frontier.pdf"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved to {outpath}")

    fig.savefig(outpath.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved preview to {outpath.with_suffix('.png')}")

    # Summary stats
    print(f"\nCSS codes: {len(css_codes)}")
    print(f"  d>=4: {sum(1 for c in css_codes if c['d'] >= 4)}")
    print(f"  FOM>=6: {sum(1 for c in css_codes if c['fom'] >= 6)}")
    print(f"  FOM>=12: {sum(1 for c in css_codes if c['fom'] >= 12)}")
    print(f"\nNon-CSS codes: {len(noncss_codes)}")
    n_exact = sum(1 for c in noncss_codes if c["exact"])
    n_upper = sum(1 for c in noncss_codes if not c["exact"])
    print(f"  exact: {n_exact}, upper-bound: {n_upper}")
    print(f"  FOM>=6: {sum(1 for c in noncss_codes if c['fom'] >= 6)}")
    print(f"  FOM>=12: {sum(1 for c in noncss_codes if c['fom'] >= 12)}")


if __name__ == "__main__":
    main()
