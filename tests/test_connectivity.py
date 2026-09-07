"""Tests for stabilizer connectivity and direct-sum decomposition."""

import numpy as np

from evaluation.bb_code import build_bb_code
from evaluation.pbb_code import build_pbb_code
from evaluation.connectivity import (
    bicycle_translation_component_count,
    bicycle_translation_is_connected,
    tanner_components,
    stabilizer_components,
    decompose,
    components_isomorphic,
)
from evaluation.tanner_equivalence import canonical_hash, canonical_hash_noncss


class TestBicycleTranslationGate:
    """Cheap future-campaign gate, cross-checked against full decomposition."""

    def test_known_connected_css_presentation(self):
        A = [(0, 0), (5, 1), (2, 2)]
        B = [(0, 0), (3, 3)]
        assert bicycle_translation_component_count(10, 9, A, B) == 1
        assert bicycle_translation_is_connected(10, 9, A, B) is True
        assert decompose(build_bb_code(10, 9, A, B)).is_connected is True

    def test_known_css_direct_sum_reports_exact_multiplicity(self):
        A = [(0, 0), (0, 13), (12, 11)]
        B = [(0, 0), (3, 7)]
        assert bicycle_translation_component_count(15, 14, A, B) == 3
        assert bicycle_translation_is_connected(15, 14, A, B) is False
        assert decompose(build_bb_code(15, 14, A, B)).n_components == 3

    def test_known_pbb_direct_sum_uses_all_four_supports(self):
        A = [(0, 0), (4, 2)]
        B = [(0, 0), (2, 0), (1, 6)]
        C = []
        D = [(0, 0), (2, 0)]
        assert bicycle_translation_component_count(
            18, 12, A, B, C, D
        ) == 2
        code = build_pbb_code(18, 12, A, B, C, D)
        assert decompose(code).n_components == 2

    def test_pbb_perturbation_support_can_join_backbone_components(self):
        # This generalized support test is independent of the weight-five
        # campaign's stricter C subset A / D subset B constraint.
        A = [(0, 0), (2, 0)]
        B = [(0, 0), (0, 2)]
        assert bicycle_translation_component_count(4, 4, A, B) == 4
        assert bicycle_translation_component_count(
            4, 4, A, B, [(1, 0)], [(0, 1)]
        ) == 1

    def test_invalid_exponent_fails_closed(self):
        with np.testing.assert_raises_regex(ValueError, "outside"):
            bicycle_translation_is_connected(
                6, 6, [(0, 0), (6, 0)], [(0, 0), (1, 0)]
            )

    def test_empty_a_family_fails_closed(self):
        # An empty A/C family severs the only edges linking the L and R
        # qubit sectors; the subgroup-index formula can't see that, so it
        # must be rejected rather than silently under-reporting components.
        with np.testing.assert_raises_regex(ValueError, "nonempty"):
            bicycle_translation_component_count(6, 6, [], [(0, 0), (1, 0)])

    def test_empty_b_and_d_family_fails_closed(self):
        with np.testing.assert_raises_regex(ValueError, "nonempty"):
            bicycle_translation_component_count(
                6, 6, [(0, 0), (1, 0)], [], [], []
            )

    def test_duplicate_monomial_fails_closed(self):
        # Over GF(2) a repeated monomial cancels rather than coinciding, so
        # it must be rejected rather than silently deduplicated.
        with np.testing.assert_raises_regex(ValueError, "repeated monomial"):
            bicycle_translation_component_count(
                6, 6, [(0, 0), (0, 0)], [(0, 0), (1, 0)]
            )


class _CSSMatrixCode:
    def __init__(self, matrix_x, matrix_z, *, dimension):
        self.matrix_x = np.asarray(matrix_x, dtype=int)
        self.matrix_z = np.asarray(matrix_z, dtype=int)
        self.num_qudits = self.matrix_x.shape[1]
        self.dimension = dimension


class _NonCSSMatrixCode:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=int)
        self.num_qudits = self.matrix.shape[1] // 2


class TestConnectivityCSS:
    """Cross-checked against the known [[288,24,12]] = 2 x [[144,12,12]] gross-code
    decomposition (tests/verify_decomposition_288_24_12.py) and against codes whose
    connectivity was independently confirmed via ad hoc BFS during the weight-5
    disconnected-code audit."""

    def test_known_288_24_12_decomposition(self):
        code = build_bb_code(24, 6, [(6, 0), (0, 1), (0, 2)], [(0, 3), (2, 0), (4, 0)])
        assert code.num_qudits == 288 and code.dimension == 24

        result = decompose(code, d=12)
        assert result.n_components == 2
        assert result.component_sizes == [144, 144]
        assert result.homogeneous is True
        assert result.component_k == [12, 12]
        assert result.k_additivity_ok is True
        assert (result.base_n, result.base_k, result.base_d) == (144, 12, 12)
        assert components_isomorphic(code) is True

    def test_180_4_14_is_connected(self):
        code = build_bb_code(10, 9, [(0, 0), (5, 1), (2, 2)], [(0, 0), (3, 3)])
        assert code.num_qudits == 180 and code.dimension == 4

        result = decompose(code, d=14)
        assert result.n_components == 1
        assert result.is_connected is True
        assert result.base_n is None

    def test_120_4_10_is_connected(self):
        code = build_bb_code(6, 10, [(0, 0), (1, 1), (5, 5)], [(0, 0), (0, 1)])
        assert code.num_qudits == 120 and code.dimension == 4

        result = decompose(code, d=10)
        assert result.n_components == 1
        assert result.is_connected is True

    def test_432_16_9_decomposes_into_four_isomorphic_components(self):
        code = build_bb_code(18, 12, [(0, 0), (5, 4)], [(0, 0), (5, 0), (17, 8)])
        assert code.num_qudits == 432 and code.dimension == 16

        result = decompose(code, d=9)
        assert result.n_components == 4
        assert result.component_sizes == [108, 108, 108, 108]
        assert result.k_additivity_ok is True
        assert (result.base_n, result.base_k, result.base_d) == (108, 4, 9)
        assert components_isomorphic(code) is True

    def test_420_18_10_decomposes_into_three_isomorphic_components(self):
        # Real tuple from results/pareto_front_v2.json exact_front (the
        # abstract's other "larger-k, distance-ten" headline claim).
        code = build_bb_code(15, 14, [(0, 0), (0, 13), (12, 11)], [(0, 0), (3, 7)])
        assert code.num_qudits == 420 and code.dimension == 18

        result = decompose(code, d=10)
        assert result.n_components == 3
        assert result.component_sizes == [140, 140, 140]
        assert (result.base_n, result.base_k, result.base_d) == (140, 6, 10)
        assert components_isomorphic(code) is True

    def test_tanner_components_partitions_all_qubits(self):
        code = build_bb_code(24, 6, [(6, 0), (0, 1), (0, 2)], [(0, 3), (2, 0), (4, 0)])
        components = tanner_components(code)
        all_qubits = sorted(q for c in components for q in c)
        assert all_qubits == list(range(code.num_qudits))

    def test_row_space_decomposition_is_generator_basis_invariant(self):
        # <X1 X2, X2> is the same stabilizer group as <X1, X2>.  Its supplied
        # Tanner presentation is connected, but the code itself decomposes.
        code = _CSSMatrixCode(
            [[1, 1], [0, 1]],
            np.zeros((0, 2), dtype=int),
            dimension=0,
        )
        assert tanner_components(code) == [[0, 1]]
        assert stabilizer_components(code) == [[0], [1]]
        result = decompose(code)
        assert result.n_components == 2
        assert result.component_k == [0, 0]
        assert result.presentation_agrees is False
        assert components_isomorphic(code) is True

    def test_component_isomorphism_preserves_css_check_type(self):
        # One component is X-stabilized and one is Z-stabilized. A hash
        # mismatch is never treated as proof of non-isomorphism (see
        # components_isomorphic's docstring), so this stays inconclusive;
        # what the X/Z coloring buys us is checked directly below, via
        # canonical_hash on the two single-qubit presentations.
        code = _CSSMatrixCode([[1, 0]], [[0, 1]], dimension=0)
        assert stabilizer_components(code) == [[0], [1]]
        assert components_isomorphic(code) is None

        x_only = _CSSMatrixCode([[1]], np.zeros((0, 1), dtype=int), dimension=0)
        z_only = _CSSMatrixCode(np.zeros((0, 1), dtype=int), [[1]], dimension=0)
        assert canonical_hash(x_only) != canonical_hash(z_only)

    def test_hidden_decomposition_can_have_unresolved_isomorphism(self):
        # These two four-qubit row spaces are related by a qubit permutation,
        # but their RREF generator presentations are not related by row and
        # qubit permutations alone.  Mixing one generator across the blocks
        # hides the split from the supplied Tanner presentation.  The fallback
        # RREF hash is therefore allowed to certify equality, but inequality is
        # inconclusive rather than proof that the component codes differ.
        left = np.array([[1, 1, 0, 0], [1, 0, 1, 1]], dtype=int)
        right = left[:, [0, 3, 1, 2]]
        matrix_x = np.block(
            [[left, np.zeros_like(left)], [np.zeros_like(right), right]]
        )
        matrix_x[0] ^= matrix_x[2]
        code = _CSSMatrixCode(
            matrix_x,
            np.zeros((0, 8), dtype=int),
            dimension=4,
        )

        assert tanner_components(code) == [list(range(8))]
        assert stabilizer_components(code) == [list(range(4)), list(range(4, 8))]
        assert decompose(code).presentation_agrees is False
        assert components_isomorphic(code) is None

    def test_equal_component_sizes_do_not_imply_a_shared_dimension(self):
        # Two four-qubit components have stabilizer ranks three and one, so
        # their dimensions are one and three despite equal block sizes.
        code = _CSSMatrixCode(
            [
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1],
            ],
            np.zeros((0, 8), dtype=int),
            dimension=4,
        )
        result = decompose(code, d=7)
        assert result.component_sizes == [4, 4]
        assert result.component_k == [1, 3]
        assert result.base_n == 4
        assert result.base_k is None
        assert result.base_d is None
        # Different dimensions rule out isomorphism, but components_isomorphic
        # doesn't compute per-component k itself and a hash mismatch alone is
        # never a definitive negative -- so it stays conservative here too.
        assert components_isomorphic(code) is None


class TestConnectivityNonCSS:
    """Non-CSS (PBB) path, cross-checked against the weight-5 PBB campaign's
    known disconnected headline codes."""

    def test_432_8_10_pbb_decomposes_into_two_isomorphic_components(self):
        # Real tuple from results/weight5_pbb_large_v1_deep_milp_verify.jsonl
        # (one of the two FOM=1.85 headline [[432,8,10]] specifications).
        code = build_pbb_code(
            18, 12,
            [(0, 0), (4, 2)],
            [(0, 0), (2, 0), (1, 6)],
            None,
            [(0, 0), (2, 0)],
        )
        assert code.num_qudits == 432 and code.dimension == 8

        result = decompose(code, d=10)
        assert result.n_components == 2
        assert result.component_sizes == [216, 216]
        assert result.k_additivity_ok is True
        assert (result.base_n, result.base_k, result.base_d) == (216, 4, 10)
        assert components_isomorphic(code) is True

    def test_pbb_tanner_components_partitions_all_qubits(self):
        # Real tuple from results/weight5_pbb_large_v1_deep_milp_verify.jsonl
        # (same [[432,8,10]] record as the isomorphism test above).
        code = build_pbb_code(
            18, 12,
            [(0, 0), (4, 2)],
            [(0, 0), (2, 0), (1, 6)],
            None,
            [(0, 0), (2, 0)],
        )
        components = tanner_components(code)
        all_qubits = sorted(q for c in components for q in c)
        assert all_qubits == list(range(code.num_qudits))

    def test_360_4_11_pbb_decomposes_into_two_isomorphic_components(self):
        # Real tuple from results/weight5_pbb_large_v1_deep_milp_verify.jsonl
        # (the "largest exact distance 11" PBB headline claim).
        code = build_pbb_code(
            15, 12,
            [(0, 0), (2, 4)],
            [(0, 0), (4, 0), (2, 6)],
            [(0, 0), (2, 4)],
            [(0, 0), (4, 0)],
        )
        assert code.num_qudits == 360 and code.dimension == 4

        result = decompose(code, d=11)
        assert result.n_components == 2
        assert result.component_sizes == [180, 180]
        assert result.k_additivity_ok is True
        assert (result.base_n, result.base_k, result.base_d) == (180, 2, 11)
        assert components_isomorphic(code) is True

    def test_noncss_canonical_hash_retains_x_z_colors(self):
        x_only = _NonCSSMatrixCode([[1, 0]])
        z_only = _NonCSSMatrixCode([[0, 1]])
        assert canonical_hash_noncss(x_only) != canonical_hash_noncss(z_only)
