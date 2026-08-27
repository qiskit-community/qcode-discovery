#!/usr/bin/env python3
"""MILP-verify the weight5_css_large_v1 candidates that hit exact_timeout.

These codes had high FOM but the in-loop cascade's exact stage
(``compute_distance_exact`` / qldpc's brute-force ``get_distance_exact()``,
300s SIGALRM) never finished. MILP (``evaluate_milp_parallel`` /
``compute_distance_milp``) is a separate, more scalable exact-distance path
that was never tried on these candidates -- this script runs it with
extended timeouts, following the same pattern as
scripts/verify_weight5_css_small_v1_milp.py (which already ran on the
small campaign and successfully certified a d_lower=14 result).

Usage:
    uv run python scripts/verify_weight5_css_large_v1_milp.py [--workers N]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluator import evaluate_milp_parallel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DISCOVERY_EVENTS = "results/evolution/weight5_css_large_v1/discovery_events.jsonl"
SAVE_PATH = "results/weight5_css_large_v1_milp_verified.jsonl"

TIMEOUT_PER_LOGICAL = 1800   # 30 min per individual ILP solve
TOTAL_TIMEOUT = 14400        # 4h total per code
EARLY_STOP = 4


def load_exact_timeout_candidates(path: str) -> list[dict]:
    """Load unique exact_timeout-stage CSS candidates, best FOM per code."""
    best_by_key = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("stage") != "exact_timeout":
                continue
            if r.get("C_terms") is not None or r.get("D_terms") is not None:
                continue  # CSS only
            key = (
                r["ell"], r["m"],
                tuple(sorted(tuple(t) for t in r["A_terms"])),
                tuple(sorted(tuple(t) for t in r["B_terms"])),
            )
            if key not in best_by_key or (r.get("fom") or 0) > (best_by_key[key].get("fom") or 0):
                best_by_key[key] = r
    results = list(best_by_key.values())
    results.sort(key=lambda x: -(x.get("fom") or 0))
    return results


def main():
    parser = argparse.ArgumentParser(description="MILP-verify weight5_css_large_v1 exact_timeout candidates")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count - 2)")
    args = parser.parse_args()

    logger.info("Loading exact_timeout candidates from %s", DISCOVERY_EVENTS)
    codes = load_exact_timeout_candidates(DISCOVERY_EVENTS)
    logger.info("Found %d unique exact_timeout CSS candidates to MILP-verify", len(codes))

    if not codes:
        logger.error("No candidates found!")
        return

    for i, c in enumerate(codes):
        logger.info(
            "  %2d. [[%d, %d, %d(uncert.)]] FOM=%.2f  A=%s B=%s",
            i + 1, c["n"], c["k"], c["d"], c.get("fom") or 0,
            c["A_terms"], c["B_terms"],
        )

    tasks = [
        (c["ell"], c["m"], c["A_terms"], c["B_terms"])
        for c in codes
    ]

    logger.info(
        "Starting MILP verification: %d codes, timeout=%ds/logical, %ds total/code",
        len(tasks), TIMEOUT_PER_LOGICAL, TOTAL_TIMEOUT,
    )

    results = evaluate_milp_parallel(
        tasks,
        milp_timeout_per_logical=TIMEOUT_PER_LOGICAL,
        milp_total_timeout=TOTAL_TIMEOUT,
        milp_early_stop=EARLY_STOP,
        max_workers=args.workers,
        save_path=SAVE_PATH,
    )

    logger.info("=" * 60)
    logger.info("MILP Verification Results")
    logger.info("=" * 60)

    results.sort(key=lambda x: -(x.get("fom") or 0))
    for i, r in enumerate(results):
        exact_flag = "EXACT" if r.get("d_is_exact") else "UB/incumbent"
        stage = r.get("stage", "?")
        logger.info(
            "  %2d. [[%d, %d, %d]] FOM=%.2f [%s] stage=%s  A=%s B=%s",
            i + 1, r["n"], r["k"], r["d"], r.get("fom") or 0, exact_flag, stage,
            r["A_terms"], r["B_terms"],
        )

    logger.info("\nResults saved to %s", SAVE_PATH)


if __name__ == "__main__":
    main()
