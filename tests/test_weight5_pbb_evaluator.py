"""Tests for evolve/openevolve_evaluator_weight5_pbb.py's _build_and_check.

Covers the codex Commit-3 finding that PBB backbone enforcement neither
validated exponent ranges nor guarded against degenerate aliased exponents
(e.g. an exponent literally equal to the lattice modulus, which collapses
to the same circulant shift as 0 at the matrix level but is symbolically
distinct as a raw (int, int) tuple) before running the symbolic weight/
containment checks. The CSS sibling evaluator already guarded this via
``evaluation.bb_code.validate_terms``; this file locks in the PBB
evaluator's parallel fix.
"""

from __future__ import annotations

from evaluation import weight_enforcement
from evolve.openevolve_evaluator_weight5_pbb import _build_and_check


class TestExponentRangeValidation:
    def test_out_of_range_exponent_aliasing_with_zero_rejected(self):
        # x_exp=6 is out of range for ell=6 (valid range is [0, ell)) and
        # aliases to the same circulant shift as the existing (0, 0) term
        # -- a bug-class collapse the naive symbolic weight check below
        # can't see, since (0, 0) != (6, 0) as raw tuples.
        A_terms = [(0, 0), (6, 0)]
        B_terms = [(0, 0), (2, 0), (4, 1)]
        info, reason = _build_and_check(6, 6, A_terms, B_terms, [], [])
        assert info is None
        assert reason == "exponent_range_or_duplicate"

    def test_the_same_aliased_terms_pass_the_naive_symbolic_check(self):
        # Documents exactly why _build_and_check needs its own
        # validate_terms call rather than trusting css_weight_ok alone:
        # the raw-tuple-based symbolic check sees two distinct elements
        # and reports the backbone as weight-ok.
        A_terms = [(0, 0), (6, 0)]
        B_terms = [(0, 0), (2, 0), (4, 1)]
        assert weight_enforcement.css_weight_ok(A_terms, B_terms) is True

    def test_out_of_range_c_term_rejected(self):
        A_terms = [(0, 0), (1, 1)]
        B_terms = [(0, 0), (2, 0), (1, 3)]
        C_terms = [(0, 0), (6, 6)]  # both exponents out of range for (6, 6)
        info, reason = _build_and_check(6, 6, A_terms, B_terms, C_terms, [])
        assert info is None
        assert reason == "exponent_range_or_duplicate"

    def test_empty_c_and_d_do_not_trigger_range_validation_errors(self):
        # C_terms/D_terms are legitimately empty when a block has no
        # perturbation -- validate_terms must accept min_terms=0 for them
        # rather than rejecting on term count.
        A_terms = [(0, 0), (1, 1)]
        B_terms = [(0, 0), (2, 0), (1, 3)]
        info, reason = _build_and_check(6, 6, A_terms, B_terms, [], [])
        assert reason != "exponent_range_or_duplicate"


class TestBuildAndCheckHappyPath:
    def test_valid_non_safety_net_candidate_builds_successfully(self):
        A_terms = [(0, 0), (1, 1)]
        B_terms = [(0, 0), (2, 0), (1, 3)]
        C_terms = [(0, 0), (1, 1)]
        D_terms = [(1, 3)]
        info, reason = _build_and_check(6, 6, A_terms, B_terms, C_terms, D_terms)
        assert reason is None
        assert info is not None
        assert info["n"] == 72
        assert info["k"] > 0

    def test_both_perturbations_empty_is_classified_css_calibration(self):
        A_terms = [(0, 0), (1, 1)]
        B_terms = [(0, 0), (2, 0), (1, 3)]
        info, reason = _build_and_check(6, 6, A_terms, B_terms, [], [])
        assert info is None
        assert reason == "css_calibration"
