"""Independent MILP cross-check for weight5_pbb_small_v1's winning codes.

The evolutionary campaign's own worker (evolve/_noncss_distance_worker.py)
certifies these codes as exact via a Stage-1 exhaustive hash-based
low-weight-logical search (has_low_weight_logical, max_weight=6) -- every
gate-passed candidate in this campaign bottomed out there, so MILP was
never actually invoked during the run.

This script re-derives the distance from scratch using a completely
different algorithm: the symplectic ILP of Landahl/Anderson/Rice
(evaluation.distance_milp.ilp_min_weight_symplectic), solving every
logical operator to proven optimality with no early stopping. Agreement
between the two independent methods is the cross-check; disagreement
would indicate a bug in one of them.

Usage:
    uv run python scripts/verify_weight5_pbb_small_milp.py \\
        --input results/weight5_pbb_small_v1_winners.jsonl \\
        --output results/weight5_pbb_small_v1_deep_milp_verify.jsonl \\
        --workers 60 --timeout 120
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.distance_milp import ilp_min_weight_symplectic
from evaluation.pbb_code import build_pbb_code, get_symplectic_logicals


def solve_single_logical(args):
    stab_matrix, logical_vec, timeout, code_idx, logical_idx = args
    t0 = time.monotonic()
    try:
        w, optimal = ilp_min_weight_symplectic(stab_matrix, logical_vec, timeout=timeout)
        elapsed = time.monotonic() - t0
        return {"code_idx": code_idx, "logical_idx": logical_idx,
                "weight": w, "optimal": optimal,
                "time_s": round(elapsed, 1), "error": None}
    except Exception as e:
        elapsed = time.monotonic() - t0
        return {"code_idx": code_idx, "logical_idx": logical_idx,
                "weight": None, "optimal": False,
                "time_s": round(elapsed, 1), "error": str(e)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=120,
                     help="Timeout per logical ILP solve (seconds)")
    args = ap.parse_args()

    codes = [json.loads(l) for l in open(args.input) if l.strip()]
    print(f"Loaded {len(codes)} codes from {args.input}")

    done_keys = set()
    if os.path.exists(args.output):
        for line in open(args.output):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            done_keys.add((str(r["A_terms"]), str(r["B_terms"]),
                            str(r["C_terms"]), str(r["D_terms"]),
                            r["ell"], r["m"]))
        print(f"Resuming: {len(done_keys)} already verified")

    todo = [c for c in codes
            if (str(c["A_terms"]), str(c["B_terms"]), str(c["C_terms"]),
                str(c["D_terms"]), c["ell"], c["m"]) not in done_keys]
    print(f"Verifying {len(todo)} codes via direct MILP (bypassing hash shortcut)")
    if not todo:
        print("nothing to do")
        return

    code_metadata = []
    all_tasks = []
    for ci, c in enumerate(todo):
        ell, m = c["ell"], c["m"]
        A = [tuple(t) for t in c["A_terms"]]
        B = [tuple(t) for t in c["B_terms"]]
        C = [tuple(t) for t in c.get("C_terms") or []]
        D = [tuple(t) for t in c.get("D_terms") or []]

        code = build_pbb_code(ell, m, A, B, C, D)
        n = code.num_qudits
        k = code.dimension
        stab_matrix = np.array(code.matrix, dtype=int) % 2
        logicals = get_symplectic_logicals(code)
        num_logicals = logicals.shape[0]

        code_metadata.append({"entry": c, "n": n, "k": k,
                               "num_logicals": num_logicals})
        for li in range(num_logicals):
            all_tasks.append((stab_matrix, logicals[li], args.timeout, ci, li))

    print(f"Total independent ILP solves: {len(all_tasks)}  "
          f"(workers={args.workers}, timeout/logical={args.timeout}s)\n")

    header = (f"{'#':>3} {'lattice':>8} {'n':>4} {'k':>3} {'hash_d':>6} "
              f"{'milp_d':>6} {'agree':>6} {'solved':>8} {'time':>7}")
    print(header)
    print("-" * 60)
    sys.stdout.flush()

    code_results = defaultdict(dict)
    code_start = {}
    codes_done = set()
    mismatches = []
    output_order = 0
    t_global = time.monotonic()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for task in all_tasks:
            ci = task[3]
            code_start.setdefault(ci, time.monotonic())
            f = pool.submit(solve_single_logical, task)
            futures[f] = (task[3], task[4])

        with open(args.output, "a") as fout:
            for future in as_completed(futures):
                ci, li = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {"code_idx": ci, "logical_idx": li,
                              "weight": None, "optimal": False,
                              "time_s": 0, "error": str(e)}
                code_results[ci][li] = result

                meta = code_metadata[ci]
                if ci in codes_done or len(code_results[ci]) != meta["num_logicals"]:
                    continue
                codes_done.add(ci)
                elapsed = time.monotonic() - code_start[ci]

                c = meta["entry"]
                n, k, num_log = meta["n"], meta["k"], meta["num_logicals"]
                hash_d = c["d"]

                d_best = n
                all_optimal = True
                any_found = False
                logicals_optimal = 0
                per_logical_detail = []
                for idx in range(num_log):
                    r = code_results[ci].get(idx)
                    w = r["weight"] if r else None
                    opt = r["optimal"] if r else False
                    per_logical_detail.append({"w": w, "opt": opt,
                                                "t": r.get("time_s") if r else None})
                    if w is not None:
                        d_best = min(d_best, w)
                        any_found = True
                        if opt:
                            logicals_optimal += 1
                        else:
                            all_optimal = False
                    else:
                        all_optimal = False

                exact = all_optimal and any_found
                milp_d = d_best if any_found else None
                agree = (exact and milp_d == hash_d)
                if exact and milp_d != hash_d:
                    mismatches.append({"code": c, "hash_d": hash_d, "milp_d": milp_d})

                output_order += 1
                print(f"{output_order:3d} ({c['ell']},{c['m']}) {n:4d} {k:3d} "
                      f"{hash_d:6d} {str(milp_d):>6} {str(agree):>6} "
                      f"{logicals_optimal}/{num_log:>3} {elapsed:6.0f}s")
                sys.stdout.flush()

                out_record = {
                    **{kk: v for kk, v in c.items()},
                    "hash_d": hash_d,
                    "milp_d": milp_d,
                    "milp_exact": exact,
                    "agrees_with_hash": agree,
                    "logicals_optimal": logicals_optimal,
                    "total_logicals": num_log,
                    "per_logical": per_logical_detail,
                    "deep_milp_time_s": round(elapsed, 1),
                    "timeout_per_logical": args.timeout,
                    "verified_at": datetime.now(timezone.utc).isoformat(),
                }
                fout.write(json.dumps(out_record) + "\n")
                fout.flush()

    total_elapsed = time.monotonic() - t_global
    print(f"\nDone in {total_elapsed/60:.1f} min.")
    print(f"Verified: {len(codes_done)}/{len(todo)}")
    print(f"Mismatches (MILP-exact but disagrees with hash-based d): {len(mismatches)}")
    for m in mismatches:
        print(f"  MISMATCH: ({m['code']['ell']},{m['code']['m']}) "
              f"hash_d={m['hash_d']} milp_d={m['milp_d']}")


if __name__ == "__main__":
    main()
