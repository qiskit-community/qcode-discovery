"""Build the MILP-certification target set for weight5_pbb_large_v1.

The large campaign's own worker (evolve/_noncss_distance_worker.py) certifies
most gate-passed candidates exactly via the hash-based low-weight-logical
search, but for n>216 codes it only searches up to weight 4 -- anything with
a genuinely larger distance falls through to `d_method="milp+bposd"`, an
MILP *incumbent* upper bound combined with a BP-OSD estimate, with
`milp_exact=false`. Those codes have NO exact certification at all; MILP is
the primary path to one (unlike the small campaign, where MILP was only a
cross-check against an already hash-exact d).

This script reads the campaign's raw per-iteration log, deduplicates to one
row per distinct code (keeping the best-fom row for each), and selects the
milp_exact==false subset as the certification target set, sorted by fom
descending so the most valuable codes verify first.

Usage:
    uv run python scripts/prepare_weight5_pbb_large_winners.py \\
        --input results/evolution/weight5_pbb_large_v1/all_codes_weight5_pbb.jsonl \\
        --output results/weight5_pbb_large_v1_winners.jsonl
"""
from __future__ import annotations

import argparse
import json


def canonical_key(r: dict) -> tuple:
    return (
        str(r["A_terms"]), str(r["B_terms"]),
        str(r["C_terms"]), str(r["D_terms"]),
        r["ell"], r["m"],
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Loaded {len(rows)} rows from {args.input}")

    best_by_key = {}
    for r in rows:
        key = canonical_key(r)
        cur = best_by_key.get(key)
        if cur is None or r["fom"] > cur["fom"]:
            best_by_key[key] = r
    distinct = list(best_by_key.values())
    print(f"Deduplicated to {len(distinct)} distinct codes")

    # NOTE: filter on d_method, not the `milp_exact` field -- the latter only
    # means "MILP was the method that proved exactness", so hash-exact codes
    # (d_method in exact_w2/w3/w4) also show milp_exact=False despite already
    # being exactly certified. The genuinely-unproven codes are exactly the
    # ones the worker fell through to MILP+BP-OSD for.
    targets = [r for r in distinct if r["d_method"] == "milp+bposd"]
    already_exact = len(distinct) - len(targets)
    print(f"Already exactly certified (hash or milp): {already_exact}")
    print(f"Certification targets (d_method=milp+bposd): {len(targets)}")

    targets.sort(key=lambda r: r["fom"], reverse=True)

    total_logicals = sum(2 * r["k"] for r in targets)
    by_n = {}
    for r in targets:
        by_n[r["n"]] = by_n.get(r["n"], 0) + 1
    print(f"Total logical solves (sum of 2k): {total_logicals}")
    print(f"By n: {dict(sorted(by_n.items()))}")

    with open(args.output, "w") as f:
        for r in targets:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(targets)} records to {args.output}")


if __name__ == "__main__":
    main()
