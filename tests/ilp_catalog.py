"""ILP exact distance for all CSS catalog codes.

Parses polynomial definitions from paper/2606.02418/css_catalog_tables.tex and runs ILP
distance computation on each code. Saves results to results/ilp_catalog.json.
"""

import os
import re
import sys
import time
import json
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

sys.path.insert(0, ".")
from evaluation.bb_code import build_bb_code
from qldpc.objects import Pauli


# ── ILP solver ──────────────────────────────────────────────────────

def ilp_min_weight(check_matrix, logical_op, timeout=120):
    m, n = check_matrix.shape
    num_vars = n + m + 1
    c = np.zeros(num_vars)
    c[:n] = 1.0

    rows = []
    for r in range(m):
        row = np.zeros(num_vars)
        row[:n] = check_matrix[r]
        row[n + r] = -2
        rows.append(row)
    row = np.zeros(num_vars)
    row[:n] = logical_op
    row[n + m] = -2
    rows.append(row)

    A_mat = np.array(rows)
    b_eq = np.zeros(m + 1)
    b_eq[m] = 1
    constraints = LinearConstraint(A_mat, b_eq, b_eq)

    lb = np.zeros(num_vars)
    ub = np.ones(num_vars)
    for r in range(m):
        ub[n + r] = np.ceil(np.sum(check_matrix[r]) / 2)
    ub[n + m] = np.ceil(np.sum(logical_op) / 2)

    result = milp(
        c=c, constraints=constraints,
        integrality=np.ones(num_vars),
        bounds=Bounds(lb, ub),
        options={"time_limit": timeout, "presolve": True},
    )
    if result.success and result.x is not None:
        return int(round(result.fun))
    return None


def compute_distance(code, timeout_per=120):
    n, k = code.num_qudits, code.dimension
    if k == 0:
        return n, n, n

    hx = np.array(code.matrix_x, dtype=int) % 2
    hz = np.array(code.matrix_z, dtype=int) % 2
    lx = np.array(code.get_logical_ops(Pauli.X), dtype=int) % 2
    lz = np.array(code.get_logical_ops(Pauli.Z), dtype=int) % 2

    d_z = n
    for i in range(k):
        w = ilp_min_weight(hx, lx[i], timeout=timeout_per)
        if w is not None:
            d_z = min(d_z, w)
        if d_z <= 2:
            break

    d_x = n
    for i in range(k):
        w = ilp_min_weight(hz, lz[i], timeout=timeout_per)
        if w is not None:
            d_x = min(d_x, w)
        if d_x <= 2:
            break

    return min(d_x, d_z), d_x, d_z


# ── LaTeX polynomial parser ────────────────────────────────────────

def parse_poly(latex_str):
    """Parse LaTeX polynomial like '1{+}y^{2}{+}x^{4}' into exponent tuples."""
    s = latex_str.strip()

    terms = s.split('{+}')
    result = []
    for term in terms:
        term = term.strip()
        x_exp = 0
        y_exp = 0

        if term == '1':
            result.append((0, 0))
            continue

        # Match x^{N} or bare x
        xm = re.search(r'x\^\{(\d+)\}', term)
        if xm:
            x_exp = int(xm.group(1))
        elif 'x' in term and not re.search(r'x\^\{', term):
            x_exp = 1

        # Match y^{N} or bare y
        ym = re.search(r'y\^\{(\d+)\}', term)
        if ym:
            y_exp = int(ym.group(1))
        elif 'y' in term and not re.search(r'y\^\{', term):
            y_exp = 1

        result.append((x_exp, y_exp))

    return result


def parse_table_row(line):
    """Parse a LaTeX table row into code parameters."""
    line = line.replace('\\\\', '').strip()
    if not line or line.startswith('%') or 'midrule' in line or 'toprule' in line or 'bottomrule' in line:
        return None
    parts = line.split('&')
    if len(parts) < 5:
        return None

    # (ell, m)
    lm_match = re.search(r'\((\d+),(\d+)\)', parts[0])
    if not lm_match:
        return None
    ell, m = int(lm_match.group(1)), int(lm_match.group(2))

    # A polynomial - extract between $ signs
    a_match = re.search(r'\$(.+?)\$', parts[1])
    if not a_match:
        return None
    A = parse_poly(a_match.group(1))

    # B polynomial
    b_match = re.search(r'\$(.+?)\$', parts[2])
    if not b_match:
        return None
    B = parse_poly(b_match.group(1))

    # k
    k_str = parts[3].strip()
    k = int(k_str)

    # d (may have * suffix)
    d_str = parts[4].strip().replace('*', '')
    bp_d = int(d_str)

    return ell, m, A, B, k, bp_d


def extract_table(tex_lines, label):
    """Extract all data rows from a table with given label."""
    codes = []
    in_table = False
    found_label = False

    for i, line in enumerate(tex_lines):
        if f'\\label{{{label}}}' in line:
            found_label = True
        if found_label and '\\midrule' in line:
            in_table = True
            continue
        if in_table and '\\bottomrule' in line:
            break
        if in_table:
            parsed = parse_table_row(line)
            if parsed:
                codes.append(parsed)

    return codes


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", choices=["144", "288", "360", "all"], default="all")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout per logical (seconds)")
    parser.add_argument("--index", type=int, default=None, help="Run only code at this index within table")
    parser.add_argument("--start", type=int, default=0, help="Start from this index (skip earlier codes)")
    parser.add_argument("--outdir", type=str, default=None, help="Write per-index results to this directory (for parallel runs)")
    args = parser.parse_args()

    # Read supplemental CSS catalog tables.
    with open("paper/2606.02418/css_catalog_tables.tex") as f:
        tex_lines = f.readlines()

    tables = {}
    if args.table in ("144", "all"):
        tables["n=144"] = extract_table(tex_lines, "tab:cat144")
    if args.table in ("288", "all"):
        tables["n=288"] = extract_table(tex_lines, "tab:cat288")
    if args.table in ("360", "all"):
        tables["n=360"] = extract_table(tex_lines, "tab:cat360")

    for tname, codes in tables.items():
        print(f"\n{'#'*60}")
        print(f"# {tname}: {len(codes)} codes")
        print(f"{'#'*60}")

    def load_existing_results(n_val):
        """Load existing results from per-table JSON, keyed by (ell, m, tuple(A), tuple(B))."""
        per_file = f"results/ilp_catalog_{n_val}.json"
        existing = {}
        try:
            with open(per_file) as f:
                data = json.load(f)
            for key, results in data.items():
                if key.startswith("n=") and isinstance(results, list):
                    for r in results:
                        code_key = (r["ell"], r["m"], str(r["A"]), str(r["B"]))
                        existing[code_key] = r
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        return existing

    def save_table_results(tname, results_list):
        """Save results for one table, merging with existing."""
        n_val = tname.replace("n=", "")
        if args.outdir and args.index is not None:
            os.makedirs(args.outdir, exist_ok=True)
            outfile = f"{args.outdir}/ilp_{n_val}_idx{args.index}.json"
        else:
            outfile = f"results/ilp_catalog_{n_val}.json"
        with open(outfile, "w") as f:
            json.dump({tname: results_list}, f, indent=2)

    all_results = {}
    for tname, codes in tables.items():
        n_val = tname.replace("n=", "")
        print(f"\n{'='*60}")
        print(f"TABLE {tname} ({len(codes)} codes)")
        print(f"{'='*60}\n")

        # Load existing results to avoid re-computing
        existing = load_existing_results(n_val)
        print(f"  Loaded {len(existing)} existing results from results/ilp_catalog_{n_val}.json")

        # Build full results list, preserving existing
        table_results = []
        if args.index is not None:
            subset = [(args.index, codes[args.index])]
        else:
            subset = list(enumerate(codes))[args.start:]

        # First, add all existing results for codes before --start
        if args.start > 0 or args.index is not None:
            for idx in range(len(codes)):
                if args.index is not None and idx != args.index:
                    ell, m, A, B, k, bp_d = codes[idx]
                    code_key = (ell, m, str(A), str(B))
                    if code_key in existing:
                        table_results.append(existing[code_key])
                elif args.index is None and idx < args.start:
                    ell, m, A, B, k, bp_d = codes[idx]
                    code_key = (ell, m, str(A), str(B))
                    if code_key in existing:
                        table_results.append(existing[code_key])

        for real_idx, (ell, m, A, B, k, bp_d) in subset:
            n_expected = 2 * ell * m
            label = f"[[{n_expected},{k},<={bp_d}]]"
            code_key = (ell, m, str(A), str(B))

            # Skip if already computed
            if code_key in existing and "ilp_d" in existing[code_key]:
                r = existing[code_key]
                print(f"[{real_idx}/{len(codes)}] {label} CACHED d={r['ilp_d']} FOM={r['true_fom']}", flush=True)
                table_results.append(r)
                continue

            print(f"[{real_idx}/{len(codes)}] {label} ({ell},{m}) A={A} B={B}", flush=True)

            try:
                code = build_bb_code(ell, m, A, B)
                n, k_actual = code.num_qudits, code.dimension
                assert k_actual == k, f"k mismatch: expected {k}, got {k_actual}"

                t0 = time.time()
                d, d_x, d_z = compute_distance(code, timeout_per=args.timeout)
                elapsed = time.time() - t0

                fom = k * d**2 / n
                status = "CONFIRMED" if d == bp_d else ("TIGHTER" if d < bp_d else "LOOSER")
                if d == n:
                    status = "ALL_TIMEOUT"

                print(f"  -> d={d} (d_X={d_x}, d_Z={d_z}) FOM={fom:.1f} [{elapsed:.1f}s] {status}", flush=True)

                result = {
                    "label": label, "ell": ell, "m": m,
                    "A": A, "B": B, "k": k, "bp_osd_d": bp_d,
                    "ilp_d": d, "ilp_d_x": d_x, "ilp_d_z": d_z,
                    "true_fom": round(fom, 1), "time_s": round(elapsed, 1),
                    "status": status,
                }
                table_results.append(result)

            except Exception as e:
                print(f"  -> ERROR: {e}", flush=True)
                result = {
                    "label": label, "ell": ell, "m": m,
                    "A": A, "B": B, "k": k, "bp_osd_d": bp_d,
                    "status": f"ERROR: {e}",
                }
                table_results.append(result)

            # Incremental save after each code
            save_table_results(tname, table_results)

        all_results[tname] = table_results

        # Final save for this table
        save_table_results(tname, table_results)

        # Summary for this table
        solved = [r for r in table_results if "ilp_d" in r]
        confirmed = [r for r in solved if r["status"] == "CONFIRMED"]
        tighter = [r for r in solved if r["status"] == "TIGHTER"]
        print(f"\n--- {tname} Summary ---")
        print(f"  Total: {len(table_results)}, Solved: {len(solved)}")
        print(f"  Confirmed BP-OSD: {len(confirmed)}")
        print(f"  Tighter than BP-OSD: {len(tighter)}")
        if tighter:
            worst = max(tighter, key=lambda r: r["bp_osd_d"] - r["ilp_d"])
            print(f"  Worst overestimate: {worst['label']} BP-OSD={worst['bp_osd_d']} -> ILP={worst['ilp_d']}")
        if solved:
            best = max(solved, key=lambda r: r["true_fom"])
            print(f"  Best FOM: {best['label']} FOM={best['true_fom']}")

    # Merge into combined file
    combined = {}
    for n_val in ["144", "288", "360"]:
        per_file = f"results/ilp_catalog_{n_val}.json"
        try:
            with open(per_file) as f:
                data = json.load(f)
            for key, val in data.items():
                if key.startswith("n="):
                    combined[key] = val
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    combined_file = "results/ilp_catalog.json"
    with open(combined_file, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Merged all results to {combined_file}")
