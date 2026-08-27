"""Deep-budget follow-up: push the remaining unproven logicals from
verify_weight5_pbb_small_milp.py to full ILP optimality.

Reads results/weight5_pbb_small_v1_deep_milp_verify.jsonl, finds every
logical operator that did not reach proven optimality within the first
pass's 180s timeout, and re-solves only those with a much longer
per-logical timeout. Updates each record's per_logical/milp_exact/
agrees_with_hash fields in place once every logical is resolved.

Usage:
    uv run python scripts/verify_weight5_pbb_small_milp_deep.py \\
        --input results/weight5_pbb_small_v1_deep_milp_verify.jsonl \\
        --timeout 3600 --workers 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.distance_milp import ilp_min_weight_symplectic
from evaluation.pbb_code import build_pbb_code, get_symplectic_logicals


def solve_single_logical(args):
    stab_matrix, logical_vec, timeout, rec_idx, logical_idx = args
    t0 = time.monotonic()
    try:
        w, optimal = ilp_min_weight_symplectic(stab_matrix, logical_vec, timeout=timeout)
        elapsed = time.monotonic() - t0
        return {"rec_idx": rec_idx, "logical_idx": logical_idx,
                "weight": w, "optimal": optimal,
                "time_s": round(elapsed, 1), "error": None}
    except Exception as e:
        elapsed = time.monotonic() - t0
        return {"rec_idx": rec_idx, "logical_idx": logical_idx,
                "weight": None, "optimal": False,
                "time_s": round(elapsed, 1), "error": str(e)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", required=True)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=3600,
                     help="Deep per-logical timeout in seconds (default 3600)")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.input) if l.strip()]
    print(f"Loaded {len(records)} records from {args.input}")

    all_tasks = []
    code_objs = {}
    for ri, r in enumerate(records):
        unsolved = [i for i, pl in enumerate(r["per_logical"]) if not pl["opt"]]
        if not unsolved:
            continue
        ell, m = r["ell"], r["m"]
        A = [tuple(t) for t in r["A_terms"]]
        B = [tuple(t) for t in r["B_terms"]]
        C = [tuple(t) for t in r.get("C_terms") or []]
        D = [tuple(t) for t in r.get("D_terms") or []]
        code = build_pbb_code(ell, m, A, B, C, D)
        stab_matrix = np.array(code.matrix, dtype=int) % 2
        logicals = get_symplectic_logicals(code)
        code_objs[ri] = True
        for li in unsolved:
            all_tasks.append((stab_matrix, logicals[li], args.timeout, ri, li))

    print(f"Records with unsolved logicals: {len(code_objs)}")
    print(f"Total logicals to re-solve at deep budget "
          f"({args.timeout}s/logical): {len(all_tasks)}\n")
    if not all_tasks:
        print("Nothing to do -- every logical already proven optimal.")
        return

    results_by_rec = {ri: {} for ri in code_objs}
    t0_global = time.monotonic()
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(solve_single_logical, t): (t[3], t[4]) for t in all_tasks}
        for future in as_completed(futures):
            ri, li = futures[future]
            try:
                result = future.result()
            except Exception as e:
                result = {"weight": None, "optimal": False, "time_s": 0, "error": str(e)}
            results_by_rec[ri][li] = result
            completed += 1
            elapsed = time.monotonic() - t0_global
            r = records[ri]
            print(f"[{completed}/{len(all_tasks)}] ({r['ell']},{r['m']}) "
                  f"logical#{li}: weight={result['weight']} "
                  f"optimal={result['optimal']} time={result['time_s']}s "
                  f"(elapsed {elapsed:.0f}s)", flush=True)

    # Merge results back into records
    pushed_to_exact = 0
    for ri, per_li in results_by_rec.items():
        r = records[ri]
        for li, res in per_li.items():
            r["per_logical"][li] = {"w": res["weight"], "opt": res["optimal"],
                                     "t": res["time_s"]}
        all_optimal = all(pl["opt"] for pl in r["per_logical"])
        any_found = any(pl["w"] is not None for pl in r["per_logical"])
        weights = [pl["w"] for pl in r["per_logical"] if pl["w"] is not None]
        milp_d = min(weights) if weights else None
        exact = all_optimal and any_found
        r["logicals_optimal"] = sum(1 for pl in r["per_logical"] if pl["opt"])
        r["milp_d"] = milp_d
        r["milp_exact"] = exact
        r["agrees_with_hash"] = exact and milp_d == r["hash_d"]
        if exact:
            pushed_to_exact += 1

    tmp_path = args.input + ".tmp"
    with open(tmp_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp_path, args.input)

    total_elapsed = time.monotonic() - t0_global
    print(f"\nDone in {total_elapsed/60:.1f} min.")
    print(f"Records fully pushed to proven-exact this pass: {pushed_to_exact}")
    now_exact = sum(1 for r in records if r["milp_exact"])
    mismatches = [r for r in records if r["milp_exact"] and r["milp_d"] != r["hash_d"]]
    print(f"Total records now fully MILP-exact: {now_exact}/{len(records)}")
    print(f"Mismatches: {len(mismatches)}")
    for r in mismatches:
        print(f"  MISMATCH: ({r['ell']},{r['m']}) hash_d={r['hash_d']} milp_d={r['milp_d']}")


if __name__ == "__main__":
    main()
