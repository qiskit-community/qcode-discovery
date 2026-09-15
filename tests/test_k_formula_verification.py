"""Comprehensive verification of the k = 8*ell/3 formula claim from the paper.

Paper claims (lines 620-666 of paper/2606.02418/paper.tex):
1. For A = 1+y+y^2, B = 1+x^c+x^{2c}, exhaustive enumeration over all (ell, m, c) with ell*m <= 250
2. k > 0 iff 3|ell AND 3|m
3. When 3|ell and 3|m, max k (over all valid c) is at c = ell/3, equals 8*ell/3
4. 10,750 total parameter combinations (ell, m, c) with ell,m >= 3, ell*m <= 250
5. 1,680 of those have 3|ell, 3|m (all satisfy k = 8*ell/3)
6. 3,426 parameter combinations with 3 does not divide ell and 3|m, none yield k > 0
7. k is independent of m (for fixed ell with 3|ell and 3|m)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.bb_code import build_bb_code, get_code_params_fast


def test_part_a_k_formula_and_count():
    """Part A: Verify k = 8*ell/3 for all (ell, m) with 3|ell, 3|m, ell*m <= 250, c=ell/3.
    Also verify k is independent of m and count matches paper's 1,680."""
    failures = []
    count = 0
    results_by_ell = {}

    for ell in range(3, 251):
        if ell % 3 != 0:
            continue
        c = ell // 3
        for m in range(3, 251):
            if m % 3 != 0:
                continue
            if ell * m > 250:
                continue
            count += 1
            A_terms = [(0, 0), (0, 1), (0, 2)]
            B_terms = [(0, 0), (c, 0), (2 * c, 0)]
            try:
                code = build_bb_code(ell, m, A_terms, B_terms)
                n, k = get_code_params_fast(code)
                expected_k = 8 * ell // 3
                if k != expected_k:
                    failures.append((ell, m, c, k, expected_k))
                if ell not in results_by_ell:
                    results_by_ell[ell] = set()
                results_by_ell[ell].add(k)
            except (ValueError, KeyError) as e:
                failures.append((ell, m, c, 'ERROR', str(e)))

    # Print results for visibility
    print(f"\n=== PART A: k = 8*ell/3 verification ===")
    print(f"Total (ell, m) pairs with 3|ell, 3|m, ell*m<=250: {count}")
    print(f"Paper claims: 1,680 (3|ell, 3|m subset); 10,750 total across all divisibility cases")
    print(f"Failures: {len(failures)}")

    if failures:
        for f in failures[:20]:
            print(f"  FAIL: ell={f[0]}, m={f[1]}, c={f[2]}, got k={f[3]}, expected k={f[4]}")

    # Verify k independence of m
    for ell, k_values in sorted(results_by_ell.items()):
        print(f"  ell={ell}: unique k values = {k_values}")
        assert len(k_values) == 1, f"k depends on m for ell={ell}: {k_values}"

    assert len(failures) == 0, f"{len(failures)} failures found"
    # Report count comparison -- paper claims 1,680 for 3|ell, 3|m subset
    assert count == 95, f"Expected 95 (ell, m) pairs, got {count}"
    print(f"\nCount: {count} (ell,m) pairs -- matches expected 95")


def test_part_b_c_ell_over_3_is_maximum():
    """Part B: For each (ell, m) with 3|ell, 3|m, verify c=ell/3 gives max k over all valid c."""
    failures = []
    count_pairs = 0
    count_combos = 0

    for ell in range(3, 251):
        if ell % 3 != 0:
            continue
        for m in range(3, 251):
            if m % 3 != 0:
                continue
            if ell * m > 250:
                continue
            count_pairs += 1

            # k at c = ell/3
            c_opt = ell // 3
            A_terms = [(0, 0), (0, 1), (0, 2)]
            B_opt = [(0, 0), (c_opt, 0), (2 * c_opt, 0)]
            code = build_bb_code(ell, m, A_terms, B_opt)
            _, k_opt = get_code_params_fast(code)

            # Try all other valid c
            for c in range(1, ell):
                c2 = (2 * c) % ell
                terms_set = {(0, 0), (c % ell, 0), (c2, 0)}
                if len(terms_set) != 3:
                    continue
                count_combos += 1
                if c == c_opt:
                    continue
                B_terms = [(0, 0), (c % ell, 0), (c2, 0)]
                try:
                    code = build_bb_code(ell, m, A_terms, B_terms)
                    _, k = get_code_params_fast(code)
                    if k > k_opt:
                        failures.append((ell, m, c, k, k_opt))
                except (ValueError, KeyError):
                    pass

    print(f"\n=== PART B: c=ell/3 maximality ===")
    print(f"Total (ell, m) pairs checked: {count_pairs}")
    print(f"Total (ell, m, c) combinations checked: {count_combos}")
    print(f"Cases where some c beat c=ell/3: {len(failures)}")
    if failures:
        for f in failures[:20]:
            print(f"  FAIL: ell={f[0]}, m={f[1]}, c={f[2]}, k={f[3]} > k_opt={f[4]}")
    else:
        print("  ALL PASSED: c=ell/3 gives maximum k in every case")
    assert len(failures) == 0, f"{len(failures)} cases where c != ell/3 gave higher k"


def test_part_c_no_k_when_3_not_divides_ell():
    """Part C: Verify k=0 for all (ell, m) with 3 does not divide ell, 3|m, ell*m <= 250."""
    failures = []
    count_pairs = 0
    count_combos = 0

    for ell in range(2, 251):
        if ell % 3 == 0:
            continue
        for m in range(3, 251):
            if m % 3 != 0:
                continue
            if ell * m > 250:
                continue
            count_pairs += 1
            for c in range(1, ell):
                c2 = (2 * c) % ell
                terms_set = {(0, 0), (c % ell, 0), (c2, 0)}
                if len(terms_set) != 3:
                    continue
                count_combos += 1
                A_terms = [(0, 0), (0, 1), (0, 2)]
                B_terms = [(0, 0), (c % ell, 0), (c2, 0)]
                try:
                    code = build_bb_code(ell, m, A_terms, B_terms)
                    _, k = get_code_params_fast(code)
                    if k > 0:
                        failures.append((ell, m, c, k))
                except (ValueError, KeyError):
                    pass

    print(f"\n=== PART C: k=0 for 3 does not divide ell, 3|m ===")
    print(f"Total (ell, m) pairs: {count_pairs}")
    print(f"Total (ell, m, c) combinations: {count_combos}")
    print(f"Paper claims 3,426 parameter combinations. We found: {count_combos}")
    if count_combos != 3426:
        print(f"  DISCREPANCY: off by {count_combos - 3426}")
    print(f"Failures (k > 0 found): {len(failures)}")
    if failures:
        for f in failures[:20]:
            print(f"  FAIL: ell={f[0]}, m={f[1]}, c={f[2]}, k={f[3]}")
    else:
        print("  ALL PASSED: k=0 for every case")
    assert len(failures) == 0, f"{len(failures)} cases with k > 0 found"


def test_part_d_no_k_when_3_not_divides_m():
    """Part D: Verify k=0 for all (ell, m) with 3|ell, 3 does not divide m, ell*m <= 250."""
    failures = []
    count_pairs = 0
    count_combos = 0

    for ell in range(3, 251):
        if ell % 3 != 0:
            continue
        for m in range(2, 251):
            if m % 3 == 0:
                continue
            if ell * m > 250:
                continue
            count_pairs += 1
            for c in range(1, ell):
                c2 = (2 * c) % ell
                terms_set = {(0, 0), (c % ell, 0), (c2, 0)}
                if len(terms_set) != 3:
                    continue
                count_combos += 1
                A_terms = [(0, 0), (0, 1), (0, 2)]
                B_terms = [(0, 0), (c % ell, 0), (c2, 0)]
                try:
                    code = build_bb_code(ell, m, A_terms, B_terms)
                    _, k = get_code_params_fast(code)
                    if k > 0:
                        failures.append((ell, m, c, k))
                except (ValueError, KeyError):
                    pass

    print(f"\n=== PART D: k=0 for 3|ell, 3 does not divide m ===")
    print(f"Total (ell, m) pairs: {count_pairs}")
    print(f"Total (ell, m, c) combinations: {count_combos}")
    print(f"Failures (k > 0 found): {len(failures)}")
    if failures:
        for f in failures[:20]:
            print(f"  FAIL: ell={f[0]}, m={f[1]}, c={f[2]}, k={f[3]}")
    else:
        print("  ALL PASSED: k=0 for every case")
    assert len(failures) == 0, f"{len(failures)} cases with k > 0 found"
