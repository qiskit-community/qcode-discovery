#!/usr/bin/env python3
"""Generate per-batch distance distributions for headline codes.

Runs the 150k-trial multi-decoder protocol (3 decoders × 10 batches × 5000 trials)
on headline codes + Bravyi control, storing all 30 individual batch distances.
Produces a boxplot figure for the paper and a JSON data file.

Addresses reviewer request: "report per-batch distributions (not just minima)."

Usage::

    uv run python tests/per_batch_distributions.py
    uv run python tests/per_batch_distributions.py --trials 2000  # faster (lower quality)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.soak_test import run_soak_test, DECODER_CONFIGS


# Codes to verify -- headline codes from Table II + Bravyi control
CODES_FOR_DISTRIBUTION = [
    {
        "label": "[[144,12,12]] Bravyi",
        "ell": 12, "m": 6,
        "A_terms": [(3, 0), (0, 1), (0, 2)],
        "B_terms": [(0, 3), (1, 0), (2, 0)],
        "claimed_d": 12, "claimed_fom": 12.0,
        "notes": "Bravyi et al. 2024 gross code (control)",
    },
    {
        "label": "[[360,40,≤24]]",
        "ell": 15, "m": 12,
        "A_terms": [(0, 0), (0, 10), (0, 11)],
        "B_terms": [(0, 0), (5, 0), (10, 0)],
        "claimed_d": 24, "claimed_fom": 64.0,
        "notes": "headline code 1, constant-monomial",
    },
    {
        "label": "[[288,32,≤22]]",
        "ell": 24, "m": 6,
        "A_terms": [(12, 0), (0, 2), (0, 4)],
        "B_terms": [(0, 3), (10, 0), (20, 0)],
        "claimed_d": 22, "claimed_fom": 53.8,
        "notes": "headline code 2, x/y-swap",
    },
    {
        "label": "[[144,24,≤12]]",
        "ell": 12, "m": 6,
        "A_terms": [(6, 0), (0, 1), (0, 2)],
        "B_terms": [(0, 3), (2, 0), (4, 0)],
        "claimed_d": 12, "claimed_fom": 24.0,
        "notes": "best at n=144, x/y-swap",
    },
    {
        "label": "[[288,24,≤18]]",
        "ell": 12, "m": 12,
        "A_terms": [(6, 0), (0, 1), (0, 2)],
        "B_terms": [(0, 3), (2, 0), (4, 0)],
        "claimed_d": 18, "claimed_fom": 27.0,
        "notes": "best FOM code (MILP d=12, FOM=12.0), x/y-swap",
    },
]


def generate_boxplot(results: list[dict], output_path: str):
    """Generate a boxplot figure showing per-batch distance distributions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(7, 4))

    # Prepare data: one box per code, with all 30 batch values
    labels = []
    all_data = []
    colors = []

    # Color scheme: control (green), headline codes (blue shades)
    color_map = {
        "[[144,12,12]] Bravyi": "#2ca02c",   # green for control
        "[[360,40,≤24]]": "#1f77b4",         # blue
        "[[288,32,≤22]]": "#ff7f0e",         # orange
        "[[144,24,≤12]]": "#d62728",         # red
        "[[288,24,≤18]]": "#9467bd",         # purple for best-FOM code
    }

    for r in results:
        label = r["label"]
        n, k = r["n"], r["k"]
        d_verified = r["verified_d"]

        # Collect all batch values across decoders
        batch_values = []
        for decoder_name, batches in r["per_decoder_batches"].items():
            batch_values.extend(batches)

        short_label = f"[[{n},{k}]]"
        if "Bravyi" in label:
            short_label += "\n(Bravyi)"
        labels.append(short_label)
        all_data.append(batch_values)
        colors.append(color_map.get(label, "#333333"))

    # Create boxplot
    bp = ax.boxplot(
        all_data,
        labels=labels,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker='o', markerfacecolor='red', markersize=5, alpha=0.7),
        medianprops=dict(color='black', linewidth=1.5),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Add individual batch points (jittered)
    for i, data in enumerate(all_data):
        x = np.random.normal(i + 1, 0.06, size=len(data))
        ax.scatter(x, data, alpha=0.3, s=10, color=colors[i], zorder=3)

    # Add horizontal lines for verified d (global min)
    for i, r in enumerate(results):
        d_verified = r["verified_d"]
        ax.hlines(d_verified, i + 0.7, i + 1.3, colors='red', linestyles='dashed',
                  linewidth=1, alpha=0.7)

    ax.set_ylabel("Distance upper bound per batch", fontsize=11)
    ax.set_title("Per-batch distance distributions (150k-trial protocol)", fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # Add annotation for the protocol
    ax.text(0.98, 0.02,
            "3 decoders × 10 batches × 5,000 trials\nRed dashed: global minimum",
            transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
            style='italic', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Figure saved to {output_path}")
    plt.close()


def generate_per_decoder_boxplot(results: list[dict], output_path: str):
    """Generate a detailed per-decoder boxplot showing distributions by decoder config."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n_codes = len(results)
    fig, axes = plt.subplots(1, n_codes, figsize=(3.5 * n_codes, 4), sharey=False)
    if n_codes == 1:
        axes = [axes]

    decoder_colors = {
        "OSD_0/product_sum": "#1f77b4",
        "OSD_CS10/product_sum": "#ff7f0e",
        "OSD_CS10/minimum_sum": "#2ca02c",
    }
    decoder_short = {
        "OSD_0/product_sum": "OSD₀/PS",
        "OSD_CS10/product_sum": "CS10/PS",
        "OSD_CS10/minimum_sum": "CS10/MS",
    }

    for idx, (r, ax) in enumerate(zip(results, axes)):
        label = r["label"]
        n, k = r["n"], r["k"]
        d_verified = r["verified_d"]

        decoder_data = []
        decoder_labels = []
        box_colors = []

        for decoder_name in ["OSD_0/product_sum", "OSD_CS10/product_sum",
                             "OSD_CS10/minimum_sum"]:
            batches = r["per_decoder_batches"].get(decoder_name, [])
            if batches:
                decoder_data.append(batches)
                decoder_labels.append(decoder_short[decoder_name])
                box_colors.append(decoder_colors[decoder_name])

        bp = ax.boxplot(
            decoder_data,
            labels=decoder_labels,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker='o', markersize=4, alpha=0.7),
            medianprops=dict(color='black', linewidth=1.5),
            widths=0.6,
        )

        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        # Add jittered points
        for i, (data, color) in enumerate(zip(decoder_data, box_colors)):
            x = np.random.normal(i + 1, 0.05, size=len(data))
            ax.scatter(x, data, alpha=0.5, s=12, color=color, zorder=3)

        # Mark global min
        ax.axhline(d_verified, color='red', linestyle='--', linewidth=1, alpha=0.7)

        if "Bravyi" in label:
            ax.set_title(f"[[{n},{k},{d_verified}]]\n(Bravyi)", fontsize=10)
        else:
            ax.set_title(f"[[{n},{k},≤{d_verified}]]", fontsize=10)

        if idx == 0:
            ax.set_ylabel("Batch distance", fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', labelsize=8)

    fig.suptitle("Per-decoder batch distributions (150k-trial protocol)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Detailed figure saved to {output_path}")
    plt.close()


def print_distribution_table(results: list[dict]):
    """Print a text table of per-batch statistics for each code and decoder."""
    print(f"\n{'='*90}")
    print(f"  PER-BATCH DISTANCE DISTRIBUTIONS")
    print(f"{'='*90}")

    for r in results:
        label = r["label"]
        n, k = r["n"], r["k"]
        d_verified = r["verified_d"]

        print(f"\n  {label}  (n={n}, k={k}, d_verified≤{d_verified})")
        print(f"  {'Decoder':<30s}  {'Batches':>7s}  {'Min':>4s}  "
              f"{'Q1':>4s}  {'Med':>4s}  {'Q3':>4s}  {'Max':>4s}  {'Mean':>6s}  {'Std':>5s}")
        print(f"  {'-'*30}  {'-'*7}  {'-'*4}  "
              f"{'-'*4}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*6}  {'-'*5}")

        import numpy as np
        all_batches = []
        for decoder_name, batches in r["per_decoder_batches"].items():
            arr = np.array(batches)
            all_batches.extend(batches)
            q1 = int(np.percentile(arr, 25))
            med = int(np.median(arr))
            q3 = int(np.percentile(arr, 75))
            print(f"  {decoder_name:<30s}  {len(batches):>7d}  {arr.min():>4d}  "
                  f"{q1:>4d}  {med:>4d}  {q3:>4d}  {arr.max():>4d}  "
                  f"{arr.mean():>6.1f}  {arr.std():>5.1f}")

        # Combined across all decoders
        all_arr = np.array(all_batches)
        q1 = int(np.percentile(all_arr, 25))
        med = int(np.median(all_arr))
        q3 = int(np.percentile(all_arr, 75))
        print(f"  {'ALL DECODERS':<30s}  {len(all_batches):>7d}  {all_arr.min():>4d}  "
              f"{q1:>4d}  {med:>4d}  {q3:>4d}  {all_arr.max():>4d}  "
              f"{all_arr.mean():>6.1f}  {all_arr.std():>5.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-batch distance distributions for headline codes"
    )
    parser.add_argument("--batches", type=int, default=10,
                        help="Batches per decoder (default: 10)")
    parser.add_argument("--trials", type=int, default=5000,
                        help="Trials per batch (default: 5000)")
    parser.add_argument("--output", type=str,
                        default="results/per_batch_distributions.json",
                        help="Output JSON file")
    parser.add_argument("--figure", type=str,
                        default="paper/2606.02418/figures/per_batch_distributions.pdf",
                        help="Output figure file")
    parser.add_argument("--detailed-figure", type=str,
                        default="paper/2606.02418/figures/per_batch_by_decoder.pdf",
                        help="Detailed per-decoder figure file")
    parser.add_argument("--from-json", type=str, default=None,
                        help="Load results from existing JSON instead of re-running")
    args = parser.parse_args()

    if args.from_json:
        print(f"Loading results from {args.from_json}")
        with open(args.from_json) as f:
            all_results = json.load(f)
    else:
        print(f"Per-batch distribution generation")
        print(f"Protocol: 3 decoders × {args.batches} batches × {args.trials} trials")
        print(f"Codes: {len(CODES_FOR_DISTRIBUTION)}")

        all_results = []
        t_start = time.time()

        for code_spec in CODES_FOR_DISTRIBUTION:
            result = run_soak_test(
                code_spec,
                num_batches=args.batches,
                trials_per_batch=args.trials,
                decoder_configs=DECODER_CONFIGS,
            )
            all_results.append(result)

        total_time = time.time() - t_start
        print(f"\nTotal time: {total_time/60:.1f} minutes")

    # Print distribution table
    print_distribution_table(all_results)

    # Save JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nData saved to {output_path}")

    # Generate figures
    fig_dir = Path(args.figure).parent
    fig_dir.mkdir(parents=True, exist_ok=True)

    generate_boxplot(all_results, args.figure)
    generate_per_decoder_boxplot(all_results, args.detailed_figure)


if __name__ == "__main__":
    main()
