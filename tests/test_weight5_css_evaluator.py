"""Tests for evolve/openevolve_evaluator_weight5_css.py's ``_run_evaluation``.

Covers four of the five codex round-2 findings from the out-of-band review
of commit 49fe15a that are specific to this evaluator (the fifth,
malformed-candidate arity safety, is shared with the PBB sibling and
covered generically by ``tests/test_candidate_selection.py``'s
``TestHasArity``).
"""

from __future__ import annotations

import pytest

from evaluation import gates
import evolve.openevolve_evaluator_weight5_css as css_eval
from evolve.openevolve_evaluator_weight5_css import (
    _filter_connected,
    _filter_weight5,
)
from evolve.seed_solution_weight5_css import generate_candidates as css_generate


class TestFilterWeight5ArityGuard:
    """Round-2 finding #5: ``_filter_weight5`` used to call ``len(cand) !=
    2`` directly, raising ``TypeError`` (and aborting the whole lattice)
    on a malformed, non-sized candidate instead of just rejecting it."""

    def test_malformed_candidate_rejected_not_raised(self):
        malformed = 42
        valid_cand = ([(0, 0), (1, 1)], [(0, 0), (2, 0), (1, 3)])
        valid, rejected = _filter_weight5(6, 6, [malformed, valid_cand])
        assert rejected == 1
        assert valid == [valid_cand]


class TestConnectivityGate:
    CONNECTED = (
        [(0, 0), (3, 1)],
        [(0, 0), (1, 0), (2, 1)],
    )
    DISCONNECTED = (
        [(0, 0), (0, 13), (12, 11)],
        [(0, 0), (3, 7)],
    )

    def test_default_rejects_disconnected_candidate(self, monkeypatch):
        monkeypatch.delenv("QCODE_WEIGHT5_CONNECTIVITY_POLICY", raising=False)
        kept, rejected = _filter_connected(
            15, 14, [self.CONNECTED, self.DISCONNECTED]
        )
        assert kept == [self.CONNECTED]
        assert rejected == 1

    def test_explicit_allow_admits_disconnected_candidate(self, monkeypatch):
        monkeypatch.setenv("QCODE_WEIGHT5_CONNECTIVITY_POLICY", "allow")
        candidates = [self.CONNECTED, self.DISCONNECTED]
        kept, rejected = _filter_connected(15, 14, candidates)
        assert kept == candidates
        assert rejected == 0

    def test_invalid_policy_is_observable_and_fail_closed(self, monkeypatch):
        monkeypatch.setenv("QCODE_WEIGHT5_CONNECTIVITY_POLICY", "disabled")
        with pytest.raises(ValueError, match="QCODE_WEIGHT5_CONNECTIVITY_POLICY"):
            _filter_connected(15, 14, [self.CONNECTED])

    def test_disconnected_safety_net_is_not_reinserted(self, monkeypatch):
        # The real safety net is itself translation-connected (that's the
        # whole point of the fix that made it so -- see
        # seed_solution_weight5_css.py's _SAFETY_NET_RAW comment), so this
        # exercises the reinsertion path with a synthetic disconnected
        # stand-in patched in place of the real seed helper: if the
        # generator regresses to producing exactly the (disconnected)
        # "safety net" it's given, the reinsertion logic must not force
        # those candidates back in past the gate.
        #
        # This must be weight-5-valid and in-range at (6, 6) -- unlike
        # `self.DISCONNECTED` (only valid at (15, 14)) -- so it is rejected
        # for being disconnected specifically, not merely out of range: it
        # is exactly the old univariate safety net retired in favor of the
        # current connected one (see _SAFETY_NET_RAW's comment), which is
        # translation-disconnected (4 components) at every campaign lattice
        # including (6, 6).
        monkeypatch.delenv("QCODE_WEIGHT5_CONNECTIVITY_POLICY", raising=False)
        ell, m = 6, 6
        disconnected_safety_net = [
            ([(0, 0), (0, 2)], [(0, 0), (2, 0), (4, 0)]),
        ]
        monkeypatch.setattr(
            css_eval,
            "_css_safety_net_candidates",
            lambda _ell, _m: list(disconnected_safety_net),
        )
        captured = {}

        def fake_evaluate_batch(ell, m, candidates, **kwargs):
            captured["candidates"] = list(candidates)
            return []

        monkeypatch.setattr(css_eval, "evaluate_batch", fake_evaluate_batch)
        metrics = css_eval._run_evaluation(
            lambda _ell, _m: list(disconnected_safety_net),
            [(ell, m)],
            quick=True,
        )

        assert captured["candidates"] == []
        assert metrics["total_connectivity_rejected"] == len(disconnected_safety_net)
        assert any("rejected by connectivity" in msg for msg in metrics["errors"])


class TestSafetyNetSurvivesPrebuildCap:
    """Round-2 finding #3: the pre-build cap's per-strategy stratification
    (keyed by ``_classify_pattern``) has no dedicated bucket for the
    safety net, so it could be hash-sampled out of a crowded stratum like
    any other candidate. ``_run_evaluation`` now re-inserts any of the
    seed's own ``_safety_net_candidates`` that didn't survive selection,
    unconditionally. Forcing ``max_build_candidates_css`` down to 1 makes
    survival-by-chance negligible if the fix regressed."""

    def test_both_safety_net_entries_present_in_all_results_despite_tiny_cap(
        self, monkeypatch
    ):
        # This regression test intentionally exercises pre-audit legacy
        # behavior; the default future-run policy rejects these direct sums.
        monkeypatch.setenv("QCODE_WEIGHT5_CONNECTIVITY_POLICY", "allow")
        monkeypatch.setattr(gates, "max_build_candidates_css", lambda: 1)
        ell, m = 6, 6
        safety_net = css_eval._css_safety_net_candidates(ell, m)
        assert safety_net

        metrics = css_eval._run_evaluation(css_generate, [(ell, m)], quick=True)

        result_keys = {
            (
                tuple(map(tuple, r["A_terms"])),
                tuple(map(tuple, r["B_terms"])),
            )
            for r in metrics["all_results"]
        }
        for A, B in safety_net:
            key = (tuple(map(tuple, A)), tuple(map(tuple, B)))
            assert key in result_keys, (ell, m, A, B)


class TestSafetyNetReservedWithinCap:
    """Round-3 finding: the safety net used to be appended AFTER the
    pre-build cap already selected up to ``max_build`` candidates, so the
    final build count could exceed the configured hard cap by up to
    ``len(safety_net)``. ``_run_evaluation`` now reserves room for the
    safety net WITHIN ``max_build`` before selection runs, so with
    ``max_build=1`` and 2 safety-net entries the reserved budget
    (``max(0, 1-2)=0``) selects nothing on its own -- the final build
    count is exactly ``len(safety_net)``, not ``max_build + len(safety_net)``."""

    def test_build_count_not_inflated_beyond_safety_net_size(self, monkeypatch):
        monkeypatch.setenv("QCODE_WEIGHT5_CONNECTIVITY_POLICY", "allow")
        monkeypatch.setattr(gates, "max_build_candidates_css", lambda: 1)
        ell, m = 6, 6
        safety_net = css_eval._css_safety_net_candidates(ell, m)
        assert len(safety_net) == 2
        captured = {}

        def fake_evaluate_batch(ell, m, candidates, **kwargs):
            captured["count"] = len(candidates)
            return []

        monkeypatch.setattr(css_eval, "evaluate_batch", fake_evaluate_batch)
        css_eval._run_evaluation(css_generate, [(ell, m)], quick=True)

        assert captured["count"] == len(safety_net)


class TestIsCredibleHelper:
    """Round-3 findings: Stage1's own ``valid`` computation and Stage2's
    ``per_lattice_best`` loop each had their own independent ``k > 0``
    check that leaked stage="k_low" results through, in addition to the
    ``promising`` prefilter (see TestPromisingPrefilterKeepsLowKBands
    below) and the aggregate ``valid`` filter (fixed in round 2). All four
    sites now share this single predicate."""

    def test_k_low_stage_is_not_credible(self):
        assert css_eval._is_credible({"k": 3, "stage": "k_low"}) is False

    def test_k_zero_is_not_credible(self):
        assert css_eval._is_credible({"k": 0, "stage": "quick_k_only"}) is False

    def test_positive_k_non_rejected_stage_is_credible(self):
        assert css_eval._is_credible({"k": 4, "stage": "quick_k_only"}) is True
        assert css_eval._is_credible({"k": 4}) is True


class TestPromisingPrefilterKeepsLowKBands:
    """Round-2 finding #1: the two-pass (quick=False) path used to
    prefilter ``quick_results`` down to ``k >= 8`` BEFORE calling
    ``select_postbuild_distance_candidates``, which itself already
    stratifies by k-band (k_2_3/k_4_5/k_ge_6) -- so the k_2_3/k_4_5 bands
    never received any candidates to select from regardless of what the
    generator produced. The prefilter is now just ``k > 0``, letting the
    k-band stratification do the real work."""

    def test_low_k_results_reach_the_distance_selector(self, monkeypatch):
        # Force every quick-screened result to report a low, but nonzero,
        # k -- if the old `k >= 8` prefilter were still in place, `top`
        # would always end up empty here.
        ell, m = 6, 6
        captured = {}

        def fake_select_postbuild(promising, **kwargs):
            captured["promising_ks"] = [r.get("k", 0) for r in promising]
            return [], {}

        monkeypatch.setattr(
            css_eval.candidate_selection,
            "select_postbuild_distance_candidates",
            fake_select_postbuild,
        )

        def fake_evaluate_batch(ell, m, candidates, **kwargs):
            return [
                {
                    "A_terms": A, "B_terms": B,
                    "ell": ell, "m": m, "k": 2, "n": 2 * ell * m,
                    "d": 0, "fom": 0.0, "stage": "quick_k_only",
                }
                for A, B in candidates
            ]

        monkeypatch.setattr(css_eval, "evaluate_batch", fake_evaluate_batch)

        css_eval._run_evaluation(css_generate, [(ell, m)], quick=False)

        assert captured["promising_ks"], "no candidates reached the distance selector"
        assert all(k == 2 for k in captured["promising_ks"])

    def test_k_low_stage_excluded_from_promising(self, monkeypatch):
        # Round-3 finding: k_low results (nonzero k, but a cascade
        # rejection) used to still satisfy the bare `k > 0` prefilter and
        # consume k_2_3 selection quota despite being guaranteed to fail
        # refinement. They must never reach the distance selector.
        ell, m = 6, 6
        captured = {}

        def fake_select_postbuild(promising, **kwargs):
            captured["promising"] = list(promising)
            return [], {}

        monkeypatch.setattr(
            css_eval.candidate_selection,
            "select_postbuild_distance_candidates",
            fake_select_postbuild,
        )

        def fake_evaluate_batch(ell, m, candidates, **kwargs):
            return [
                {
                    "A_terms": A, "B_terms": B,
                    "ell": ell, "m": m, "k": 1, "n": 2 * ell * m,
                    "d": 0, "fom": 0.0, "stage": "k_low",
                }
                for A, B in candidates
            ]

        monkeypatch.setattr(css_eval, "evaluate_batch", fake_evaluate_batch)

        css_eval._run_evaluation(css_generate, [(ell, m)], quick=False)

        assert captured["promising"] == []


class TestKLowExcludedFromAggregateValid:
    """Round-2 finding #4: the aggregate ``valid`` filter counted any
    result with ``k > 0`` as valid, including ``stage="k_low"`` results
    (k nonzero but below ``gates.min_k_threshold()`` -- a cascade
    rejection, per ``evaluation/evaluator.py``). Now excluded explicitly."""

    def test_k_low_stage_not_counted_as_valid(self, monkeypatch):
        ell, m = 6, 6

        def fake_evaluate_batch(ell, m, candidates, **kwargs):
            return [
                {
                    "A_terms": A, "B_terms": B, "ell": ell, "m": m,
                    "k": 1, "n": 2 * ell * m, "d": 0, "fom": 0.0,
                    "stage": "k_low",
                }
                for A, B in candidates
            ]

        monkeypatch.setattr(css_eval, "evaluate_batch", fake_evaluate_batch)

        metrics = css_eval._run_evaluation(css_generate, [(ell, m)], quick=True)

        assert metrics["all_results"], "expected some results to exercise the filter"
        assert all(r.get("stage") == "k_low" for r in metrics["all_results"])
        assert metrics["num_valid"] == 0
