"""Tests for Clifford equivalence checking."""

import json
import numpy as np
import pytest
from qldpc.codes import QuditCode

from evaluation.clifford_equivalence import (
    exact_local_clifford_css_search,
    is_equivalently_css,
    is_lc_equivalent_css,
    is_lc_equivalent_css_group,
    verify_lc_bruteforce,
    check_lc_pattern_feasibility,
    verify_uniform_reduction_exact,
    verify_not_lc_css,
    _apply_clifford_to_block,
    _is_css_group,
)
from evaluation.bb_code import build_bb_code
from evaluation.mirror_code import build_mirror_code
from evaluation.pbb_code import (
    build_pbb_code, check_commutativity, _poly_to_matrix,
)


class TestCliffordEquivalence:
    """Test the Clifford equivalence checker."""

    def test_css_code_is_equivalently_css(self):
        """A CSS BB code should be identified as equivalently CSS."""
        bb = build_bb_code(6, 6, [(3, 0), (0, 1), (0, 2)],
                           [(0, 3), (1, 0), (2, 0)])
        stab = np.array(bb.matrix, dtype=int) % 2
        code = QuditCode(stab)
        result = is_equivalently_css(code)
        assert result["is_css"] is True
        assert result["num_y_qubits"] == 0

    def test_mirror_code_with_y_support(self):
        """Mirror codes typically have Y support → not equivalently CSS."""
        code = build_mirror_code(
            (3, 3),
            [(0, 0), (1, 0), (2, 0)],
            [(0, 0), (0, 1), (0, 2)],
        )
        result = is_equivalently_css(code)
        # Mirror codes with overlapping A, B have Y support
        assert result["num_y_qubits"] > 0

    def test_noncss_ptb_code(self):
        """A non-CSS PBB code should be identified as non-CSS."""
        from evaluation.bb_code import terms_to_poly
        from sympy.abc import x, y
        from qldpc import codes

        ell, m = 6, 3
        A_terms = [(0, 0), (0, 1), (0, 2)]
        B_terms = [(1, 0), (0, 0), (0, 1)]
        poly_a = terms_to_poly(A_terms)
        poly_b = terms_to_poly(B_terms)
        bb = codes.BBCode({x: ell, y: m}, poly_a, poly_b)
        mat_A = _poly_to_matrix(bb, A_terms)
        mat_B = _poly_to_matrix(bb, B_terms)

        rng = np.random.default_rng(42)
        for _ in range(10000):
            num_c = int(rng.integers(2, 5))
            num_d = int(rng.integers(2, 5))
            c_exps = set()
            while len(c_exps) < num_c:
                c_exps.add((int(rng.integers(0, ell)), int(rng.integers(0, m))))
            d_exps = set()
            while len(d_exps) < num_d:
                d_exps.add((int(rng.integers(0, ell)), int(rng.integers(0, m))))

            C_terms = list(c_exps)
            D_terms = list(d_exps)
            mat_C = _poly_to_matrix(bb, C_terms)
            mat_D = _poly_to_matrix(bb, D_terms)

            if check_commutativity(mat_A, mat_B, mat_C, mat_D):
                ptb = build_pbb_code(ell, m, A_terms, B_terms, C_terms, D_terms)
                if ptb.dimension > 0:
                    result = is_equivalently_css(ptb)
                    assert result["is_css"] is False
                    return

        pytest.skip("Could not find valid non-CSS PBB code")

    def test_coloring_returned_for_css(self):
        """CSS codes should return a valid coloring."""
        bb = build_bb_code(6, 3, [(0, 0), (0, 1), (0, 2)],
                           [(1, 0), (0, 0), (0, 1)])
        stab = np.array(bb.matrix, dtype=int) % 2
        code = QuditCode(stab)
        result = is_equivalently_css(code)
        assert result["is_css"], "CSS BB code should be identified as CSS"
        assert result["coloring"] is not None
        assert len(result["coloring"]) == stab.shape[0]
        assert all(c in (0, 1) for c in result["coloring"])


# --- Polynomials used in LC tests ---
_LC_ELL, _LC_M = 6, 3
_LC_A = [(0, 0), (0, 1), (0, 2)]   # 1 + y + y^2
_LC_B = [(1, 0), (0, 0), (0, 1)]   # x + 1 + y
# A real PBB code from the catalog (first entry of campaign7_dedup.jsonl),
# expressed in the paper convention z-part = [C | D].
_REAL_ELL, _REAL_M = 3, 6
_REAL_A = [(2, 0), (1, 3), (1, 5)]
_REAL_B = [(2, 3), (1, 2), (2, 2)]
_REAL_C = [(0, 4)]
_REAL_D = [(0, 1)]
# Code #38: [[36,4,6]], the only PBB code that is group-level LC-equivalent to CSS
_CODE38_ELL, _CODE38_M = 6, 3
_CODE38_A = [(0, 1), (3, 2), (4, 1)]
_CODE38_B = [(3, 1), (4, 0), (4, 1)]
_CODE38_C = [(0, 1), (3, 1)]
_CODE38_D = [(1, 1), (4, 1)]


class TestCliffordTransformations:
    """Verify the Clifford gate transformations used in the LC proof."""

    def test_gate_partition_on_zero_x(self):
        """The 6 single-qubit Cliffords partition into 3 classes on (0,z)."""
        z = np.array([[1, 0, 1, 1, 0]], dtype=int)  # arbitrary nonzero
        x = np.zeros_like(z)

        # {I, S}: preserve (0, z) -> (0, *)
        for gate in ("I", "S"):
            new_x, new_z = _apply_clifford_to_block(x, z, gate)
            assert not np.any(new_x), f"{gate} should keep X=0"

        # {H, HS}: convert (0, z) -> (z, 0) i.e. pure-X
        for gate in ("H", "HS"):
            new_x, new_z = _apply_clifford_to_block(x, z, gate)
            assert not np.any(new_z), f"{gate} should zero Z"
            assert np.array_equal(new_x, z), f"{gate} should move Z to X"

        # {SH, HSH}: produce Y-support (both X and Z nonzero)
        for gate in ("SH", "HSH"):
            new_x, new_z = _apply_clifford_to_block(x, z, gate)
            assert np.any(new_x), f"{gate} should have nonzero X"
            assert np.any(new_z), f"{gate} should have nonzero Z"

    def test_s_gate_action(self):
        """S: (x, z) -> (x, x+z)."""
        x = np.array([[1, 0, 1]], dtype=int)
        z = np.array([[0, 1, 1]], dtype=int)
        new_x, new_z = _apply_clifford_to_block(x, z, "S")
        assert np.array_equal(new_x, x)
        assert np.array_equal(new_z, (x + z) % 2)

    def test_hs_gate_action(self):
        """HS: (x, z) -> (x+z, x)."""
        x = np.array([[1, 0, 1]], dtype=int)
        z = np.array([[0, 1, 1]], dtype=int)
        new_x, new_z = _apply_clifford_to_block(x, z, "HS")
        assert np.array_equal(new_x, (x + z) % 2)
        assert np.array_equal(new_z, x)

    def test_sh_gate_action(self):
        """SH: (x, z) -> (z, x+z)."""
        x = np.array([[1, 0, 1]], dtype=int)
        z = np.array([[0, 1, 1]], dtype=int)
        new_x, new_z = _apply_clifford_to_block(x, z, "SH")
        assert np.array_equal(new_x, z)
        assert np.array_equal(new_z, (x + z) % 2)

    def test_hsh_gate_action(self):
        """HSH: (x, z) -> (x+z, z)."""
        x = np.array([[1, 0, 1]], dtype=int)
        z = np.array([[0, 1, 1]], dtype=int)
        new_x, new_z = _apply_clifford_to_block(x, z, "HSH")
        assert np.array_equal(new_x, (x + z) % 2)
        assert np.array_equal(new_z, z)


class TestExactLocalCliffordCssSearch:
    """Check the complete small-code LC search against direct enumeration."""

    @staticmethod
    def _brute_force(stab):
        from itertools import product

        n = stab.shape[1] // 2
        gates = ("I", "S", "H", "HS", "SH", "HSH")
        for assignment in product(gates, repeat=n):
            transformed = stab.copy() % 2
            transformed_x = transformed[:, :n]
            transformed_z = transformed[:, n:]
            for qubit, gate in enumerate(assignment):
                new_x, new_z = _apply_clifford_to_block(
                    transformed_x[:, qubit : qubit + 1],
                    transformed_z[:, qubit : qubit + 1],
                    gate,
                )
                transformed_x[:, qubit] = new_x[:, 0]
                transformed_z[:, qubit] = new_z[:, 0]
            if _is_css_group(transformed):
                return True
        return False

    def test_matches_direct_enumeration_on_small_matrices(self):
        rng = np.random.default_rng(20260903)
        for n in (2, 3, 4):
            for _ in range(4):
                stab = rng.integers(0, 2, size=(n, 2 * n), dtype=np.uint8)
                expected = self._brute_force(stab)
                result = exact_local_clifford_css_search(stab)
                assert result["complete"]
                assert result["is_lc_css"] is expected

    def test_invariant_under_row_operations(self):
        stab = np.array(
            [
                [1, 0, 1, 0, 1, 1],
                [0, 1, 1, 1, 0, 1],
                [1, 1, 0, 1, 1, 0],
            ],
            dtype=np.uint8,
        )
        changed_basis = stab.copy()
        changed_basis[2] ^= changed_basis[0]
        assert (
            exact_local_clifford_css_search(stab)["is_lc_css"]
            == exact_local_clifford_css_search(changed_basis)["is_lc_css"]
        )

    def test_five_qubit_code_is_not_lc_equivalent_to_css(self):
        """Exercise a deterministic negative case against direct enumeration."""
        stab = np.array(
            [
                [1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
                [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
                [0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
            ],
            dtype=np.uint8,
        )
        assert self._brute_force(stab) is False
        result = exact_local_clifford_css_search(stab)
        assert result["complete"]
        assert result["is_lc_css"] is False


class TestLCAlgebraic:
    """Test the algebraic LC equivalence check (is_lc_equivalent_css)."""

    def test_css_code(self):
        """CSS code (C=D=None) -> LC-equivalent to CSS."""
        result = is_lc_equivalent_css(_LC_A, _LC_B, None, None)
        assert result["is_lc_css"] is True
        assert "CSS" in result["condition"]

    def test_condition_3_synthetic(self):
        """Synthetic code with C=A, D=B -> condition 3 satisfied."""
        result = is_lc_equivalent_css(_LC_A, _LC_B,
                                      C_terms=_LC_A, D_terms=_LC_B)
        assert result["is_lc_css"] is True
        assert result["condition"] == "C=A, D=B"
        assert "both blocks" in result["gate_assignment"]

    def test_condition_1_synthetic(self):
        """Synthetic code with C=0, D=B -> condition 1 satisfied."""
        result = is_lc_equivalent_css(_LC_A, _LC_B,
                                      C_terms=None, D_terms=_LC_B)
        assert result["is_lc_css"] is True
        assert result["condition"] == "C=0, D=B"

    def test_condition_2_synthetic(self):
        """Synthetic code with D=0, C=A -> condition 2 satisfied."""
        result = is_lc_equivalent_css(_LC_A, _LC_B,
                                      C_terms=_LC_A, D_terms=None)
        assert result["is_lc_css"] is True
        assert result["condition"] == "D=0, C=A"

    def test_real_code_not_lc_css(self):
        """A real PBB code should NOT be LC-equivalent to CSS."""
        result = is_lc_equivalent_css(_REAL_A, _REAL_B, _REAL_C, _REAL_D)
        assert result["is_lc_css"] is False

    def test_distinct_perturbation_not_lc_css(self):
        """Perturbation polynomials distinct from A,B -> not LC-CSS."""
        result = is_lc_equivalent_css(
            _LC_A, _LC_B,
            C_terms=[(1, 1), (2, 2)],
            D_terms=[(0, 1), (1, 0)],
        )
        assert result["is_lc_css"] is False


class TestLCAlgebraicGroup:
    """Test the generalized group-level LC equivalence check."""

    def test_css_code(self):
        """CSS code -> group-level LC-equivalent."""
        result = is_lc_equivalent_css_group(
            _LC_ELL, _LC_M, _LC_A, _LC_B, None, None)
        assert result["is_lc_css"] is True
        assert result["s1"] == 0 and result["s2"] == 0

    def test_p0_conditions_detected(self):
        """p=0 cases (C=A, D=B) detected by group check too."""
        result = is_lc_equivalent_css_group(
            _LC_ELL, _LC_M, _LC_A, _LC_B,
            C_terms=_LC_A, D_terms=_LC_B)
        assert result["is_lc_css"] is True
        assert result["s1"] == 1 and result["s2"] == 1

    def test_code38_group_lc_css(self):
        """Code #38 ([[36,4,6]]): p=xy^2, s1=s2=1, group-level LC-CSS."""
        result = is_lc_equivalent_css_group(
            _CODE38_ELL, _CODE38_M,
            _CODE38_A, _CODE38_B, _CODE38_C, _CODE38_D)
        assert result["is_lc_css"] is True
        assert result["s1"] == 1 and result["s2"] == 1

    def test_code38_not_generator_lc_css(self):
        """Code #38 is NOT generator-level LC-CSS (old check misses it)."""
        result = is_lc_equivalent_css(
            _CODE38_A, _CODE38_B, _CODE38_C, _CODE38_D)
        assert result["is_lc_css"] is False

    def test_real_code_not_group_lc_css(self):
        """A genuinely non-CSS PBB code fails the group check too."""
        result = is_lc_equivalent_css_group(
            _REAL_ELL, _REAL_M, _REAL_A, _REAL_B, _REAL_C, _REAL_D)
        assert result["is_lc_css"] is False


class TestLCBruteforce:
    """Test brute-force verification of all 36 uniform Clifford assignments."""

    def test_css_code_bruteforce(self):
        """CSS code should have I,I and H,H as working assignments."""
        result = verify_lc_bruteforce(
            _LC_ELL, _LC_M, _LC_A, _LC_B, None, None)
        assert result["consistent"]
        assert "I,I" in result["css_assignments"]
        assert "H,H" in result["css_assignments"]

    def test_condition_3_bruteforce(self):
        """Synthetic C=A, D=B code: S,S and HS,HS should work."""
        result = verify_lc_bruteforce(
            _LC_ELL, _LC_M, _LC_A, _LC_B,
            C_terms=_LC_A, D_terms=_LC_B)
        assert result["consistent"]
        assert result["algebraic_result"]["is_lc_css"]
        assert "S,S" in result["css_assignments"]
        assert "HS,HS" in result["css_assignments"]

    def test_condition_1_bruteforce(self):
        """Synthetic C=0, D=B code: I,S and H,HS should work."""
        result = verify_lc_bruteforce(
            _LC_ELL, _LC_M, _LC_A, _LC_B,
            C_terms=None, D_terms=_LC_B)
        assert result["consistent"]
        assert "I,S" in result["css_assignments"]
        assert "H,HS" in result["css_assignments"]

    def test_condition_2_bruteforce(self):
        """Synthetic D=0, C=A code: S,I and HS,H should work."""
        result = verify_lc_bruteforce(
            _LC_ELL, _LC_M, _LC_A, _LC_B,
            C_terms=_LC_A, D_terms=None)
        assert result["consistent"]
        assert "S,I" in result["css_assignments"]
        assert "HS,H" in result["css_assignments"]

    def test_real_code_no_assignment_works(self):
        """A real PBB code: no uniform assignment should produce CSS."""
        result = verify_lc_bruteforce(
            _REAL_ELL, _REAL_M, _REAL_A, _REAL_B, _REAL_C, _REAL_D)
        assert result["consistent"]
        assert len(result["css_assignments"]) == 0
        assert len(result["group_css_assignments"]) == 0
        assert len(result["mixed_macro_group_css"]) == 0
        assert not result["algebraic_result"]["is_lc_css"]
        assert not result["group_result"]["is_lc_css"]

    def test_code38_group_css_bruteforce(self):
        """Code #38: generator check fails, but group check finds S,S and HS,HS."""
        result = verify_lc_bruteforce(
            _CODE38_ELL, _CODE38_M,
            _CODE38_A, _CODE38_B, _CODE38_C, _CODE38_D)
        assert result["consistent"]
        assert len(result["css_assignments"]) == 0
        assert "S,S" in result["group_css_assignments"]
        assert "HS,HS" in result["group_css_assignments"]
        assert len(result["mixed_macro_group_css"]) == 0
        assert result["group_result"]["is_lc_css"]

    def test_each_condition_has_exactly_two_assignments(self):
        """Each LC condition yields exactly 2 working assignments (one per
        macro-class), confirming the proof's exhaustive case analysis."""
        for C, D, expected in [
            (None, None, {"I,I", "H,H"}),
            (None, _LC_B, {"I,S", "H,HS"}),
            (_LC_A, None, {"S,I", "HS,H"}),
            (_LC_A, _LC_B, {"S,S", "HS,HS"}),
        ]:
            result = verify_lc_bruteforce(
                _LC_ELL, _LC_M, _LC_A, _LC_B, C, D)
            assert set(result["css_assignments"]) == expected, (
                f"C={C}, D={D}: got {result['css_assignments']}, "
                f"expected {expected}")


class TestLCPatternFeasibility:
    """Test the per-qubit S-pattern feasibility solver."""

    def test_real_code_infeasible(self):
        """For a real PBB code, no per-qubit S-pattern makes it CSS."""
        result = check_lc_pattern_feasibility(
            _REAL_ELL, _REAL_M, _REAL_A, _REAL_B, _REAL_C, _REAL_D)
        assert not result["any_feasible"]
        assert result["all_patterns_uniform"]

    def test_condition_3_feasible_uniform(self):
        """For C=A, D=B: pattern is feasible and uniform (all 1s)."""
        result = check_lc_pattern_feasibility(
            _LC_ELL, _LC_M, _LC_A, _LC_B,
            C_terms=_LC_A, D_terms=_LC_B)
        assert result["any_feasible"]
        assert result["all_patterns_uniform"]
        is_cls = result["IS_class"]
        assert is_cls["feasible"]
        assert is_cls["block1_uniform"]
        assert is_cls["block2_uniform"]

    def test_css_code_feasible_uniform(self):
        """For CSS code: pattern is feasible and uniform (all 0s)."""
        result = check_lc_pattern_feasibility(
            _LC_ELL, _LC_M, _LC_A, _LC_B, None, None)
        assert result["any_feasible"]
        assert result["all_patterns_uniform"]

    def test_matrix_matches_build_pbb_code(self):
        """Verify bruteforce's independent matrix matches build_pbb_code."""
        from evaluation.bb_code import terms_to_poly
        from sympy.abc import x, y
        from qldpc import codes

        ell, m = _REAL_ELL, _REAL_M
        dim = ell * m

        ptb = build_pbb_code(ell, m, _REAL_A, _REAL_B, _REAL_C, _REAL_D)
        ptb_matrix = np.array(ptb.matrix, dtype=int) % 2

        poly_a = terms_to_poly(_REAL_A)
        poly_b = terms_to_poly(_REAL_B)
        bb = codes.BBCode({x: ell, y: m}, poly_a, poly_b)
        mat_A = _poly_to_matrix(bb, _REAL_A)
        mat_B = _poly_to_matrix(bb, _REAL_B)
        mat_C = _poly_to_matrix(bb, _REAL_C)
        mat_D = _poly_to_matrix(bb, _REAL_D)
        zero = np.zeros((dim, dim), dtype=int)

        block1_x = np.hstack([mat_A, mat_B])
        block1_z = np.hstack([mat_C, mat_D])
        block2_x = np.hstack([zero, zero])
        block2_z = np.hstack([mat_B.T % 2, mat_A.T % 2])
        our_matrix = np.vstack([
            np.hstack([block1_x, block1_z]),
            np.hstack([block2_x, block2_z]),
        ]) % 2

        assert np.array_equal(our_matrix, ptb_matrix)


class TestLCCatalog:
    """Verify LC equivalence computationally against the full PBB code catalog."""

    @pytest.fixture
    def catalog(self):
        import pathlib
        path = pathlib.Path("results/campaign7_publication_merged.jsonl")
        if not path.exists():
            pytest.skip("Catalog file not found")
        codes = []
        with open(path) as f:
            for line in f:
                codes.append(json.loads(line))
        return codes

    def test_all_368_not_generator_lc_css(self, catalog):
        """None of the 368 PBB codes satisfy generator-level LC conditions (p=0)."""
        assert len(catalog) == 368
        for i, code in enumerate(catalog):
            result = is_lc_equivalent_css(
                code["A_terms"], code["B_terms"],
                code["C_terms"], code["D_terms"],
            )
            assert not result["is_lc_css"], (
                f"Code {i} ({code['n']},{code['k']},{code['d']}) "
                f"unexpectedly generator-LC-CSS: {result['condition']}")

    def test_exactly_one_group_lc_css(self, catalog):
        """Exactly 1 of 368 PBB codes is group-level LC-equivalent to CSS.

        The (36,4,6) at (6,3) with C=y+x^3y, D=xy+x^4y; (s1=1, s2=1, p=xy^2).
        """
        assert len(catalog) == 368
        lc_css_codes = []
        for i, code in enumerate(catalog):
            result = is_lc_equivalent_css_group(
                code["ell"], code["m"],
                code["A_terms"], code["B_terms"],
                code["C_terms"], code["D_terms"],
            )
            if result["is_lc_css"]:
                lc_css_codes.append((i, code["n"], code["k"], code["d"],
                                     result["s1"], result["s2"]))
        assert len(lc_css_codes) == 1, (
            f"Expected 1 group-LC-CSS code, found {len(lc_css_codes)}: "
            f"{lc_css_codes}")
        idx, n, k, d, s1, s2 = lc_css_codes[0]
        assert (n, k, d) == (36, 4, 6)
        assert s1 == 1 and s2 == 1

    def test_bruteforce_sample(self, catalog):
        """Brute-force verify a sample of catalog codes (first, last, highest FOM)."""
        sorted_by_fom = sorted(catalog, key=lambda c: c.get("fom", 0),
                                reverse=True)
        sample = [catalog[0], catalog[-1], sorted_by_fom[0]]
        for code in sample:
            result = verify_lc_bruteforce(
                code["ell"], code["m"],
                code["A_terms"], code["B_terms"],
                code["C_terms"], code["D_terms"],
            )
            assert result["consistent"], (
                f"Group-algebraic/bruteforce mismatch for "
                f"({code['n']},{code['k']},{code['d']})")
            assert len(result["css_assignments"]) == 0
            assert len(result["mixed_macro_group_css"]) == 0, (
                f"Mixed-macro group-CSS hit for "
                f"({code['n']},{code['k']},{code['d']})")

    def test_pattern_feasibility_sample(self, catalog):
        """S-pattern feasibility check on sample: infeasible, always uniform."""
        for code in catalog[:5]:
            result = check_lc_pattern_feasibility(
                code["ell"], code["m"],
                code["A_terms"], code["B_terms"],
                code["C_terms"], code["D_terms"],
            )
            assert not result["any_feasible"]
            assert result["all_patterns_uniform"]

    def test_all_368_not_lc_css(self, catalog):
        """Comprehensive LC-CSS verification: 36 uniform + non-uniform exact.

        For each of the 368 PBB codes, verify_not_lc_css combines all 36
        uniform brute-force assignments with the exact GF(2) non-uniform
        system solve.  Exactly one code (the [[36,4,6]]) is detected as LC-CSS
        by these two paths; the additional Hadamard-2-coloring path (run via
        scripts/run_lc_analysis.py, which calls is_equivalently_css) catches
        10 more for a total of 11 LC-CSS, but verify_not_lc_css itself does
        not include the Hadamard check, so this assertion remains at 1.
        """
        assert len(catalog) == 368
        lc_css_indices = []
        for i, code in enumerate(catalog):
            result = verify_not_lc_css(
                code["ell"], code["m"],
                code["A_terms"], code["B_terms"],
                code["C_terms"], code["D_terms"],
            )
            if result["is_lc_css"]:
                lc_css_indices.append(i)
        assert len(lc_css_indices) == 1, (
            f"Expected exactly 1 LC-CSS code, found {len(lc_css_indices)}: "
            f"indices {lc_css_indices}")
        lc_code = catalog[lc_css_indices[0]]
        assert (lc_code["n"], lc_code["k"], lc_code["d"]) == (36, 4, 6)


class TestUniformReduction:
    """Verify that non-uniform S-patterns cannot create LC-CSS equivalences
    beyond what uniform patterns achieve.

    Uses exact GF(2) linear algebra with the correct right-multiplication
    S-gate action (A * diag(s1), not diag(s1) * A).
    """

    def test_css_code_all_uniform_work(self):
        """CSS code (C=D=0): all 4 uniform patterns should work."""
        result = verify_uniform_reduction_exact(
            _LC_ELL, _LC_M, _LC_A, _LC_B, None, None)
        assert result["reduction_holds"]
        assert result["consistent"]
        assert (0, 0) in result["uniform_solutions"]

    def test_da_cb_synthetic(self):
        """Synthetic C=A, D=B: uniform (1,1) should work."""
        result = verify_uniform_reduction_exact(
            _LC_ELL, _LC_M, _LC_A, _LC_B,
            C_terms=_LC_A, D_terms=_LC_B)
        assert result["reduction_holds"]
        assert result["consistent"]
        assert (1, 1) in result["uniform_solutions"]

    def test_d0_cb_synthetic(self):
        """Synthetic C=0, D=B: uniform (0,1) should work."""
        result = verify_uniform_reduction_exact(
            _LC_ELL, _LC_M, _LC_A, _LC_B,
            C_terms=None, D_terms=_LC_B)
        assert result["reduction_holds"]
        assert result["consistent"]
        assert (0, 1) in result["uniform_solutions"]

    def test_da_c0_synthetic(self):
        """Synthetic D=0, C=A: uniform (1,0) should work."""
        result = verify_uniform_reduction_exact(
            _LC_ELL, _LC_M, _LC_A, _LC_B,
            C_terms=_LC_A, D_terms=None)
        assert result["reduction_holds"]
        assert result["consistent"]
        assert (1, 0) in result["uniform_solutions"]

    def test_real_code_inconsistent(self):
        """A genuinely non-CSS PBB code: system should be inconsistent."""
        result = verify_uniform_reduction_exact(
            _REAL_ELL, _REAL_M, _REAL_A, _REAL_B, _REAL_C, _REAL_D)
        assert result["reduction_holds"]
        assert not result["consistent"]
        assert len(result["uniform_solutions"]) == 0

    def test_code38_group_lc_css(self):
        """Code #38 ([[36,4,6]]): group-level LC-CSS with p=xy^2, (s1,s2)=(1,1)."""
        result = verify_uniform_reduction_exact(
            _CODE38_ELL, _CODE38_M,
            _CODE38_A, _CODE38_B, _CODE38_C, _CODE38_D)
        assert result["reduction_holds"]
        assert result["consistent"]
        assert (1, 1) in result["uniform_solutions"]

    def test_agrees_with_group_check(self):
        """Uniform solutions from exact system must match is_lc_equivalent_css_group."""
        for ell, m, A, B, C, D in [
            (_LC_ELL, _LC_M, _LC_A, _LC_B, None, None),
            (_LC_ELL, _LC_M, _LC_A, _LC_B, _LC_A, _LC_B),
            (_REAL_ELL, _REAL_M, _REAL_A, _REAL_B, _REAL_C, _REAL_D),
            (_CODE38_ELL, _CODE38_M, _CODE38_A, _CODE38_B,
             _CODE38_C, _CODE38_D),
        ]:
            exact = verify_uniform_reduction_exact(ell, m, A, B, C, D)
            group = is_lc_equivalent_css_group(ell, m, A, B, C, D)
            assert group["is_lc_css"] == (len(exact["uniform_solutions"]) > 0), (
                f"Mismatch for ({ell},{m}): group={group['is_lc_css']}, "
                f"uniform_solutions={exact['uniform_solutions']}")

    @pytest.fixture
    def catalog(self):
        import pathlib
        path = pathlib.Path("results/campaign7_publication_merged.jsonl")
        if not path.exists():
            pytest.skip("Catalog file not found")
        codes_list = []
        with open(path) as f:
            for line in f:
                codes_list.append(json.loads(line))
        return codes_list

    def test_all_368_uniform_reduction_holds(self, catalog):
        """For every PBB code, if any S-pattern makes it CSS, a uniform one does.

        Exact computational verification using correct right-multiplication
        S-gate physics.  The constraint system has 2*ell*m variables and
        O(dim(L_perp) * ell*m) constraints; solved via GF(2) rank comparison.
        """
        assert len(catalog) == 368
        for i, code in enumerate(catalog):
            result = verify_uniform_reduction_exact(
                code["ell"], code["m"],
                code["A_terms"], code["B_terms"],
                code["C_terms"], code["D_terms"],
            )
            assert result["reduction_holds"], (
                f"COUNTEREXAMPLE: Code {i} ({code['n']},{code['k']},{code['d']}) "
                f"has non-uniform LC-CSS solution with no uniform equivalent! "
                f"consistent={result['consistent']}, "
                f"uniform_solutions={result['uniform_solutions']}")


class TestVerifyNotLcCss:
    """Test the comprehensive verify_not_lc_css function."""

    def test_css_code_is_lc_css(self):
        """CSS code (C=D=None) should be detected as LC-CSS."""
        result = verify_not_lc_css(
            _LC_ELL, _LC_M, _LC_A, _LC_B, None, None)
        assert result["is_lc_css"]

    def test_real_code_not_lc_css(self):
        """A genuinely non-CSS PBB code should fail all checks."""
        result = verify_not_lc_css(
            _REAL_ELL, _REAL_M, _REAL_A, _REAL_B, _REAL_C, _REAL_D)
        assert not result["is_lc_css"]

    def test_code38_is_lc_css(self):
        """Code #38 ([[36,4,6]]) is group-level LC-CSS."""
        result = verify_not_lc_css(
            _CODE38_ELL, _CODE38_M,
            _CODE38_A, _CODE38_B, _CODE38_C, _CODE38_D)
        assert result["is_lc_css"]

    def test_synthetic_da_cb(self):
        """Synthetic C=A, D=B should be LC-CSS."""
        result = verify_not_lc_css(
            _LC_ELL, _LC_M, _LC_A, _LC_B,
            C_terms=_LC_A, D_terms=_LC_B)
        assert result["is_lc_css"]
