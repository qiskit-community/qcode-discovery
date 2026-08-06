"""Tests for evaluation/weight_enforcement.py (Phase A2).

Covers the plan's "Weight and construction" test bullets in
plans/direction2_weight5_campaigns.md: CSS backbone weight/split
enforcement, PBB symbolic and matrix-level support containment, PBB row
weight, and the CSS-fast-path ((C,D) both empty) case.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.weight_enforcement import (
    css_row_weight,
    css_weight_ok,
    matrix_support_contained,
    pbb_row_weight,
    pbb_support_contained,
    pbb_weight_ok,
)


class TestCssWeightOk:
    def test_canonical_2_3_split_accepted(self):
        A = [(0, 0), (1, 0)]
        B = [(0, 1), (1, 1), (2, 1)]
        assert css_weight_ok(A, B) is True

    def test_mirrored_3_2_split_also_accepted(self):
        A = [(0, 0), (1, 0), (2, 0)]
        B = [(0, 1), (1, 1)]
        assert css_weight_ok(A, B) is True

    def test_total_weight_4_rejected(self):
        A = [(0, 0), (1, 0)]
        B = [(0, 1), (1, 1)]
        assert css_weight_ok(A, B) is False

    def test_total_weight_6_rejected(self):
        A = [(0, 0), (1, 0), (2, 0)]
        B = [(0, 1), (1, 1), (2, 1)]
        assert css_weight_ok(A, B) is False

    def test_weight_5_but_wrong_split_1_4_rejected(self):
        A = [(0, 0)]
        B = [(0, 1), (1, 1), (2, 1), (3, 1)]
        assert css_weight_ok(A, B) is False

    def test_canonical_split_none_accepts_any_5_way_split(self):
        A = [(0, 0)]
        B = [(0, 1), (1, 1), (2, 1), (3, 1)]
        assert css_weight_ok(A, B, canonical_split=None) is True

    def test_duplicate_terms_reduced_before_counting(self):
        # Raw list lengths (2, 3) sum to 5 with a canonical split, but the
        # reduced (deduplicated) support of A is only 1 element -- the
        # check must count reduced support, not raw list length.
        A = [(0, 0), (0, 0)]
        B = [(0, 1), (1, 1), (2, 1)]
        assert css_weight_ok(A, B) is False

    def test_css_row_weight_matches_total(self):
        A = [(0, 0), (1, 0)]
        B = [(0, 1), (1, 1), (2, 1)]
        assert css_row_weight(A, B) == 5

    def test_empty_terms_do_not_crash(self):
        assert css_weight_ok([], []) is False
        assert css_row_weight(None, None) == 0


class TestPbbSupportContained:
    def test_subset_accepted(self):
        base = [(0, 0), (1, 0)]
        perturb = [(0, 0)]
        assert pbb_support_contained(base, perturb) is True

    def test_equal_sets_accepted(self):
        base = [(0, 0), (1, 0)]
        perturb = [(1, 0), (0, 0)]
        assert pbb_support_contained(base, perturb) is True

    def test_element_outside_base_rejected(self):
        base = [(0, 0), (1, 0)]
        perturb = [(2, 0)]
        assert pbb_support_contained(base, perturb) is False

    def test_empty_perturb_always_contained(self):
        assert pbb_support_contained([(0, 0)], []) is True
        assert pbb_support_contained([(0, 0)], None) is True

    def test_empty_base_only_contains_empty_perturb(self):
        assert pbb_support_contained([], []) is True
        assert pbb_support_contained([], [(0, 0)]) is False


class TestPbbRowWeight:
    def test_no_perturbation_equals_css_weight(self):
        A = [(0, 0), (1, 0)]
        B = [(0, 1), (1, 1), (2, 1)]
        assert pbb_row_weight(A, B) == css_row_weight(A, B) == 5

    def test_contained_perturbation_does_not_raise_row_weight(self):
        A = [(0, 0), (1, 0)]
        B = [(0, 1), (1, 1), (2, 1)]
        C = [(0, 0)]  # subset of supp(A)
        D = []
        assert pbb_row_weight(A, B, C, D) == 5

    def test_perturbation_outside_support_raises_row_weight(self):
        A = [(0, 0), (1, 0)]
        B = [(0, 1), (1, 1), (2, 1)]
        C = [(2, 0)]  # NOT in supp(A) -- union grows
        assert pbb_row_weight(A, B, C, None) == 6


class TestPbbWeightOk:
    A = [(0, 0), (1, 0)]
    B = [(0, 1), (1, 1), (2, 1)]

    def test_both_perturbations_contained_accepted(self):
        C = [(0, 0)]
        D = [(1, 1)]
        assert pbb_weight_ok(self.A, self.B, C, D) is True

    def test_exactly_one_empty_accepted(self):
        assert pbb_weight_ok(self.A, self.B, [(0, 0)], None) is True
        assert pbb_weight_ok(self.A, self.B, None, [(1, 1)]) is True

    def test_both_empty_accepted_as_css_calibration_case(self):
        # pbb_weight_ok itself doesn't special-case the CSS fast path --
        # callers (e.g. the weight5 PBB evaluator's _build_and_check) are
        # responsible for routing (C=D=[]) to the CSS-calibration branch
        # before this containment check ever runs. At the containment
        # level, "nothing to contain" is trivially satisfied.
        assert pbb_weight_ok(self.A, self.B, [], []) is True

    def test_c_outside_supp_a_rejected(self):
        C = [(5, 5)]
        assert pbb_weight_ok(self.A, self.B, C, None) is False

    def test_d_outside_supp_b_rejected(self):
        D = [(5, 5)]
        assert pbb_weight_ok(self.A, self.B, None, D) is False

    def test_bad_css_backbone_rejected_before_containment_checked(self):
        bad_A = [(0, 0)]
        bad_B = [(0, 1)]
        # Backbone weight is 2, not 5 -- fails regardless of C/D.
        assert pbb_weight_ok(bad_A, bad_B, [], []) is False


class TestMatrixSupportContained:
    def test_subset_matrix_accepted(self):
        base = np.array([[1, 0], [0, 1]])
        perturb = np.array([[1, 0], [0, 0]])
        assert matrix_support_contained(base, perturb) is True

    def test_extra_nonzero_outside_base_rejected(self):
        base = np.array([[1, 0], [0, 0]])
        perturb = np.array([[1, 0], [0, 1]])
        assert matrix_support_contained(base, perturb) is False

    def test_mod_2_reduction_applied(self):
        # A "2" entry is 0 mod 2, so it must not count as support.
        base = np.array([[0, 0], [0, 1]])
        perturb = np.array([[2, 0], [0, 1]])
        assert matrix_support_contained(base, perturb) is True

    def test_identical_matrices_accepted(self):
        mat = np.array([[1, 1], [0, 1]])
        assert matrix_support_contained(mat, mat) is True

    def test_all_zero_perturb_always_accepted(self):
        base = np.zeros((3, 3), dtype=int)
        perturb = np.zeros((3, 3), dtype=int)
        assert matrix_support_contained(base, perturb) is True
