#!/usr/bin/env python3
r"""Enumerate k values for the constant-monomial BB code family and check
the k >= 8 * gcd(3, ell) conjecture empirically.

Background
----------
A "constant-monomial" BB code in this codebase has univariate
generators of the form

    A(x, y) = 1 + y^a + y^b      (depends only on y)
    B(x, y) = 1 + x^c + x^{2c}   (depends only on x)

i.e., A and B each live in a single variable.  Such codes are
hypergraph-product (HGP) codes of two cyclic codes (Tillich & Zemor
2014; Eberhardt et al. 2412.04181), so the parameters
$[[2 \ell m, k, d]]$ follow standard HGP formulas in terms of the
chosen weight-3 cyclic codes; see ``paper/2606.02418/review_monomial_family.md``
for the prior-art map.  The terminology "constant-monomial" is local
to this codebase -- the standard names elsewhere are "HGP of cyclic
codes" or "BB codes associated with univariate polynomials".

Why exhaustive enumeration (not evolutionary search)?
This is a tightly parametrised family: a single integer $c$ (with
$0 < c < \ell$, $2c \not\equiv 0, c \pmod{\ell}$ so the three
monomials of $B$ are distinct) determines $B$ once $\ell, m$ are
fixed.  At $\ell m \le 250$ the search space has a few thousand
$(\ell, m, c)$ triples -- small enough that exhaustive sweep is
faster and more reliable than running an LLM-guided evolutionary
search.  Evolutionary search is reserved for unstructured families
(mixed-monomial Campaign 4, non-CSS Campaign 5).

Two variants enumerated here
----------------------------
- **Primary family**: $A = 1 + y + y^2$ (the canonical weight-3
  cyclic generator).  This is the family the conjecture below was
  originally observed on.

- **Frobenius variant**: $A = 1 + y^2 + y^4$.  Over $\mathrm{GF}(2)$
  the Frobenius endomorphism is $u \mapsto u^2$, and applied to the
  primary generator it gives $(1 + y + y^2)^2 = 1 + y^2 + y^4$ (cross
  terms vanish in characteristic 2).  Squaring is an automorphism of
  $\mathrm{GF}(2)[y]/(y^m{-}1)$, so the Frobenius variant is a
  natural sanity check on the primary family -- its enumeration tests
  whether the primary family's behaviour is preserved under this
  algebraic transformation.

Conjecture (checked at runtime)
-------------------------------
For the primary family with $3 \mid m$ and $c \mid \ell$,

    k >= 8 * gcd(3, ell).

The script reports HOLDS or VIOLATED across all enumerated lattices
and lists any violations.  When the conjecture holds across the
sweep, this is empirical evidence for it; it is not a proof.

The output is also formatted as a LaTeX table for the paper (primary
family, $3 \mid m$, best $c$ per lattice).

Usage:
    uv run python tests/enumerate_constant_monomial.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.bb_code import build_bb_code, get_code_params_fast


def valid_B_c_values(ell: int) -> list[int]:
    """Return valid c values for B = 1 + x^c + x^{2c} at given ℓ.

    Requires: c > 0, 2c % ℓ ≠ 0, 2c % ℓ ≠ c (3 distinct monomials).
    """
    valid = []
    for c in range(1, ell):
        two_c = (2 * c) % ell
        if two_c == 0:
            continue  # x^{2c} = x^0 = 1, duplicate with constant term
        if two_c == c:
            continue  # x^{2c} = x^c, duplicate
        valid.append(c)
    return valid


def enumerate_family(a_exponents: tuple[int, int], label: str,
                     max_product: int = 500) -> list[dict]:
    """Enumerate k values for A = 1 + y^a + y^b across lattices.

    Args:
        a_exponents: (a, b) for A = 1 + y^a + y^b
        label: family name for output
        max_product: max ℓ*m to consider

    Returns:
        List of dicts with ℓ, m, n, c, k, family.
    """
    a, b = a_exponents
    results = []

    for m in range(max(b + 1, 3), max_product + 1):
        # Check A has 3 distinct monomials: y^0, y^a, y^b must be distinct mod m
        a_mod = a % m
        b_mod = b % m
        if a_mod == 0 or b_mod == 0 or a_mod == b_mod:
            continue

        for ell in range(3, max_product // m + 1):
            if ell * m > max_product:
                break

            for c in valid_B_c_values(ell):
                A_terms = [(0, 0), (0, a_mod), (0, b_mod)]
                two_c = (2 * c) % ell
                B_terms = [(0, 0), (c, 0), (two_c, 0)]

                try:
                    code = build_bb_code(ell, m, A_terms, B_terms)
                    n, k = get_code_params_fast(code)
                except Exception:
                    continue

                results.append({
                    "family": label,
                    "ell": ell,
                    "m": m,
                    "n": n,
                    "c": c,
                    "k": k,
                    "A": f"1+y^{a}+y^{b}",
                    "B": f"1+x^{c}+x^{two_c}",
                })

    return results


def main():
    print("=" * 70)
    print("  CONSTANT-MONOMIAL FAMILY: k ENUMERATION")
    print("  A = 1 + y^a + y^b,  B = 1 + x^c + x^{2c}")
    print("  Lattices: all (ℓ, m) with ℓ*m ≤ 250")
    print("=" * 70)

    # --- Primary family: A = 1 + y + y² ---
    primary = enumerate_family((1, 2), "1+y+y²", max_product=250)
    print(f"\nPrimary family (A = 1+y+y²): {len(primary)} codes")

    # --- Frobenius square: A = 1 + y² + y⁴ ---
    frobenius = enumerate_family((2, 4), "1+y²+y⁴", max_product=250)
    print(f"Frobenius family (A = 1+y²+y⁴): {len(frobenius)} codes")

    all_results = primary + frobenius

    # Save raw results
    output_path = Path("results/constant_monomial_enumeration.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} results to {output_path}")

    # --- Summary tables ---
    # Group by (family, ℓ, m) and show best k and all c values
    print("\n" + "=" * 70)
    print("  PRIMARY FAMILY: A = 1 + y + y²")
    print("=" * 70)
    print(f"{'ℓ':>4} {'m':>4} {'n':>5} | {'c':>4} {'k':>4} | {'c':>4} {'k':>4} | {'c':>4} {'k':>4} | {'c':>4} {'k':>4} | best_k  3|m  c|ℓ  8·gcd(3,ℓ)")

    # Group primary by (ℓ, m)
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in primary:
        grouped[(r["ell"], r["m"])].append(r)

    # Check conjecture: k ≥ 8·gcd(3, ℓ) when 3|m and c|ℓ
    conjecture_holds = True
    conjecture_violations = []

    for (ell, m), codes in sorted(grouped.items()):
        best_k = max(r["k"] for r in codes)
        divides_3_m = (m % 3 == 0)
        bound = 8 * gcd_val(3, ell)

        # Show first few c values
        entries = []
        for r in sorted(codes, key=lambda x: x["c"]):
            c_divides_ell = (ell % r["c"] == 0)
            entries.append(f"{r['c']:>4} {r['k']:>4}")

            # Check conjecture for codes where 3|m and c|ℓ
            if divides_3_m and c_divides_ell and r["k"] < bound:
                conjecture_holds = False
                conjecture_violations.append(
                    f"  VIOLATION: ({ell},{m}) c={r['c']}: k={r['k']} < {bound}"
                )

        row = f"{ell:>4} {m:>4} {2*ell*m:>5} | " + " | ".join(entries[:4])
        if len(entries) > 4:
            row += f" | +{len(entries)-4} more"
        row += f"  | {best_k:>5}  {'Y' if divides_3_m else 'n':>3}  {bound:>10}"
        print(row)

    print(f"\nConjecture k ≥ 8·gcd(3,ℓ) (when 3|m, c|ℓ): "
          f"{'HOLDS' if conjecture_holds else 'VIOLATED'}")
    if conjecture_violations:
        for v in conjecture_violations:
            print(v)

    # --- Frobenius summary ---
    print("\n" + "=" * 70)
    print("  FROBENIUS FAMILY: A = 1 + y² + y⁴")
    print("=" * 70)

    grouped_frob = defaultdict(list)
    for r in frobenius:
        grouped_frob[(r["ell"], r["m"])].append(r)

    for (ell, m), codes in sorted(grouped_frob.items()):
        best_k = max(r["k"] for r in codes)
        entries = []
        for r in sorted(codes, key=lambda x: x["c"]):
            entries.append(f"c={r['c']}→k={r['k']}")
        print(f"  ({ell:>3},{m:>3}) n={2*ell*m:>4}: best_k={best_k:>3}  "
              f"{', '.join(entries[:6])}"
              f"{f', +{len(entries)-6} more' if len(entries) > 6 else ''}")

    # --- LaTeX table for paper (primary family, 3|m only, best k per lattice) ---
    print("\n" + "=" * 70)
    print("  LATEX TABLE (primary family, 3|m, unique (ℓ,m) with best k)")
    print("=" * 70)
    print(r"\begin{tabular}{cccccc}")
    print(r"\toprule")
    print(r"$\ell$ & $m$ & $n$ & $c$ & $k$ & $8\gcd(3,\ell)$ \\")
    print(r"\midrule")

    for (ell, m), codes in sorted(grouped.items()):
        if m % 3 != 0:
            continue
        # Find the c that gives best k among those where c|ℓ
        divisor_codes = [r for r in codes if ell % r["c"] == 0]
        if not divisor_codes:
            continue
        best = max(divisor_codes, key=lambda r: r["k"])
        bound = 8 * gcd_val(3, ell)
        print(f"{ell} & {m} & {2*ell*m} & {best['c']} & {best['k']} & {bound} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")


def gcd_val(a: int, b: int) -> int:
    """Compute gcd."""
    while b:
        a, b = b, a % b
    return a


if __name__ == "__main__":
    main()
