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

import json

import pytest

from evaluation import gates, weight_enforcement
import evolve.openevolve_evaluator_weight5_pbb as pbb_eval
from evolve.openevolve_evaluator_weight5_pbb import (
    _build_and_check,
    _classify_pbb_strategy,
)
from evolve.seed_solution_weight5_pbb import generate_candidates as pbb_generate


class TestPbbObservationProvenance:
    def test_stage1_captures_identity_before_program_import(
        self, monkeypatch
    ):
        """Import-time context clearing must not erase run provenance.

        OpenEvolve may clear its process-global attribution context while an
        evaluator thread is still active.  Capturing the identity before
        loading arbitrary evolved code keeps that race outside the import
        path as well as outside the distance-evaluation path.
        """
        context = {"program_id": "program-before-import", "model": "model-a"}
        monkeypatch.setattr(
            pbb_eval, "_peek_program_id", lambda: context["program_id"]
        )
        monkeypatch.setattr(pbb_eval, "_peek_model", lambda: context["model"])

        generate_fn = lambda ell, m: []

        def load_and_clear(_program_path):
            context["program_id"] = None
            context["model"] = None
            return generate_fn

        captured = {}

        def fake_run(_generate_fn, _lattices, **kwargs):
            captured.update(kwargs)
            return {"total_candidates": 0}

        monkeypatch.setattr(pbb_eval, "_load_generate_candidates", load_and_clear)
        monkeypatch.setattr(pbb_eval, "_run_evaluation", fake_run)

        pbb_eval.evaluate_stage1("unused.py")

        assert captured["program_id"] == "program-before-import"
        assert captured["generating_model"] == "model-a"

    def test_stage2_captures_identity_before_program_import(
        self, monkeypatch
    ):
        context = {"program_id": "program-before-import", "model": "model-a"}
        monkeypatch.setattr(
            pbb_eval, "_peek_program_id", lambda: context["program_id"]
        )
        monkeypatch.setattr(pbb_eval, "_peek_model", lambda: context["model"])

        def load_and_clear(_program_path):
            context["program_id"] = None
            context["model"] = None
            return lambda ell, m: []

        captured = {}

        class StopAfterCapture(Exception):
            pass

        def fake_run(_generate_fn, _lattices, **kwargs):
            captured.update(kwargs)
            raise StopAfterCapture

        monkeypatch.setattr(pbb_eval, "_load_generate_candidates", load_and_clear)
        monkeypatch.setattr(pbb_eval, "_run_evaluation", fake_run)
        monkeypatch.setattr(pbb_eval, "file_content_hash", lambda _path: "hash")

        with pytest.raises(StopAfterCapture):
            pbb_eval.evaluate_stage2("unused.py")

        assert captured["program_id"] == "program-before-import"
        assert captured["generating_model"] == "model-a"

    def test_raw_log_retains_program_and_model_identity(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pbb_eval, "_PROJECT_ROOT", str(tmp_path))
        result = {
            "ell": 6,
            "m": 6,
            "A_terms": [(0, 0), (1, 1)],
            "B_terms": [(0, 0), (2, 0), (1, 3)],
            "C_terms": [(0, 0)],
            "D_terms": [],
            "n": 72,
            "k": 4,
            "d": 3,
            "fom": 0.5,
            "run_id": "pbb-test",
            "program_id": "program-1",
            "generating_model": "model-a",
            "model_alias": "model-a",
            "source_hash": "source-hash",
            "code_key": "canonical-key",
        }

        pbb_eval._log_code_jsonl(result, run_name="pbb-test")

        path = tmp_path / "results/evolution/pbb-test/all_codes_weight5_pbb.jsonl"
        record = json.loads(path.read_text(encoding="utf-8"))
        for field in (
            "run_id",
            "program_id",
            "generating_model",
            "model_alias",
            "source_hash",
            "code_key",
        ):
            assert record[field] == result[field]

    def test_subthreshold_positive_observation_still_emits_event(self, monkeypatch):
        logged = []
        emitted = []
        monkeypatch.setattr(
            pbb_eval,
            "_log_code_jsonl",
            lambda result, run_name=None: logged.append((dict(result), run_name)),
        )
        monkeypatch.setattr(
            pbb_eval,
            "_emit_discovery_events",
            lambda results: emitted.extend(dict(result) for result in results),
        )

        positive = {
            "ell": 6,
            "m": 6,
            "A_terms": [(0, 0), (1, 1)],
            "B_terms": [(0, 0), (2, 0), (1, 3)],
            "C_terms": [(0, 0)],
            "D_terms": [],
            "n": 72,
            "k": 4,
            "d": 3,
            "fom": 0.5,
        }
        zero_distance = {**positive, "d": 0}

        pbb_eval._persist_observed_results(
            [positive, zero_distance],
            source_hash="source-hash",
            run_name="pbb-test",
            program_id="program-1",
            generating_model="model-a",
        )

        assert len(logged) == 1
        assert len(emitted) == 1
        assert emitted[0]["d"] == 3
        assert emitted[0]["program_id"] == "program-1"
        assert emitted[0]["model_alias"] == "model-a"
        assert emitted[0]["source_hash"] == "source-hash"
        assert emitted[0]["code_key"]


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
        B_terms = [(0, 0), (1, 0), (1, 2)]
        C_terms = []
        D_terms = [(0, 0), (1, 0), (1, 2)]
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


class TestConnectivityGate:
    # PS-106b8669: a valid [[72,12,*]] PBB presentation with two
    # translation components.
    A = [(0, 0), (3, 3)]
    B = [(0, 0), (1, 3), (2, 0)]
    C = [(0, 0), (3, 3)]
    D = [(0, 0), (2, 0)]

    def test_default_rejects_before_construction(self, monkeypatch):
        monkeypatch.delenv("QCODE_WEIGHT5_CONNECTIVITY_POLICY", raising=False)

        def must_not_build(*_args, **_kwargs):
            raise AssertionError("disconnected candidate reached construction")

        monkeypatch.setattr(pbb_eval, "build_pbb_code", must_not_build)
        info, reason = _build_and_check(
            6, 6, self.A, self.B, self.C, self.D
        )
        assert info is None
        assert reason == "disconnected"

    def test_explicit_allow_admits_disconnected_candidate(self, monkeypatch):
        monkeypatch.setenv("QCODE_WEIGHT5_CONNECTIVITY_POLICY", "allow")
        info, reason = _build_and_check(
            6, 6, self.A, self.B, self.C, self.D
        )
        assert reason is None
        assert info is not None
        assert (info["n"], info["k"]) == (72, 12)

    def test_invalid_policy_is_observable_and_fail_closed(self, monkeypatch):
        monkeypatch.setenv("QCODE_WEIGHT5_CONNECTIVITY_POLICY", "disabled")

        def must_not_build(*_args, **_kwargs):
            raise AssertionError("invalid policy reached construction")

        monkeypatch.setattr(pbb_eval, "build_pbb_code", must_not_build)
        info, reason = _build_and_check(
            6, 6, self.A, self.B, self.C, self.D
        )
        assert info is None
        assert reason == "connectivity_policy_error"

    def test_run_metrics_expose_disconnected_rejection(self, monkeypatch):
        monkeypatch.delenv("QCODE_WEIGHT5_CONNECTIVITY_POLICY", raising=False)
        monkeypatch.setattr(pbb_eval, "_pbb_safety_net_candidates", lambda *_: [])
        monkeypatch.setattr(
            pbb_eval.noncss_gate,
            "passes_noncss_gate",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("disconnected candidate reached non-CSS gate")
            ),
        )

        candidate = (self.A, self.B, self.C, self.D)
        metrics = pbb_eval._run_evaluation(
            lambda _ell, _m: [candidate], [(6, 6)], quick=True
        )
        assert metrics["all_results"] == []
        assert metrics["reject_counts"] == {"disconnected": 1}
        assert metrics["total_connectivity_rejected"] == 1

    def test_prefilter_preserves_more_specific_weight_rejection(
        self, monkeypatch
    ):
        monkeypatch.delenv("QCODE_WEIGHT5_CONNECTIVITY_POLICY", raising=False)
        invalid_weight = (
            [(0, 0), (3, 0)],
            [(0, 0), (0, 3)],
            [(0, 0)],
            [],
        )
        kept, rejected = pbb_eval._filter_connected(
            6, 6, [invalid_weight]
        )
        assert kept == [invalid_weight]
        assert rejected == 0
        info, reason = _build_and_check(6, 6, *invalid_weight)
        assert info is None
        assert reason == "backbone_weight"


class TestClassifyPbbStrategy:
    """Regression coverage for round-2 finding #2 (part B): the old
    ``"safety_net_full"`` branch pattern-matched the OLD (C=full A, D=full
    B) safety-net shape, which the CURRENT ``_SAFETY_NET_RAW`` (a
    partial-subset shape, see ``seed_solution_weight5_pbb.py``) never
    matched -- so real safety-net candidates silently fell into "both" and
    were never treated specially by this classifier. The branch was
    removed entirely rather than re-encoding the new shape a second time;
    survival is now guaranteed elsewhere (``_run_evaluation``'s explicit
    re-insertion). This just pins the simplified classifier's behavior."""

    def test_both_empty_is_css_calibration(self):
        assert _classify_pbb_strategy([(0, 0)], [(0, 1)], [], []) == "css_calibration"

    def test_c_only_is_c_only(self):
        assert _classify_pbb_strategy([(0, 0)], [(0, 1)], [(0, 0)], []) == "c_only"

    def test_d_only_is_d_only(self):
        assert _classify_pbb_strategy([(0, 0)], [(0, 1)], [], [(0, 1)]) == "d_only"

    def test_both_nonempty_is_both_even_for_the_actual_safety_net(self):
        from evolve.seed_solution_weight5_pbb import _safety_net_candidates

        for A, B, C, D in _safety_net_candidates(6, 6):
            assert _classify_pbb_strategy(A, B, C, D) == "both"


class TestSafetyNetSurvivesPrebuildCap:
    """Regression coverage for round-2 finding #2 (part C): the pre-build
    cap used to rely on ``_classify_pbb_strategy`` giving the safety net a
    dedicated, protected stratum -- which never worked (see
    TestClassifyPbbStrategy above) -- so the safety net could be
    hash-sampled out like any other "both" candidate whenever the cap was
    small relative to the population. ``_run_evaluation`` now re-inserts
    any of the seed's own ``_safety_net_candidates`` that the cap dropped,
    unconditionally. Forcing ``max_build_candidates_noncss`` down to 1
    makes the cap keep at most one non-safety-net survivor per lattice --
    if the fix regressed, the odds of both safety-net entries surviving
    hash-sampling by chance are negligible."""

    def test_both_safety_net_entries_present_in_all_results_despite_tiny_cap(
        self, monkeypatch
    ):
        monkeypatch.setattr(gates, "max_build_candidates_noncss", lambda: 1)
        ell, m = 6, 6
        safety_net = pbb_eval._pbb_safety_net_candidates(ell, m)
        assert len(safety_net) == 2

        metrics = pbb_eval._run_evaluation(pbb_generate, [(ell, m)], quick=True)

        result_keys = {
            (
                tuple(map(tuple, r["A_terms"])),
                tuple(map(tuple, r["B_terms"])),
                tuple(map(tuple, r["C_terms"])),
                tuple(map(tuple, r["D_terms"])),
            )
            for r in metrics["all_results"]
        }
        for A, B, C, D in safety_net:
            key = (
                tuple(map(tuple, A)),
                tuple(map(tuple, B)),
                tuple(map(tuple, C)),
                tuple(map(tuple, D)),
            )
            assert key in result_keys, (ell, m, A, B, C, D)


class TestSafetyNetReservedWithinCap:
    """Round-3 finding: the safety net used to be appended AFTER the
    pre-build cap already selected up to ``max_build`` candidates,
    exceeding the configured hard cap by up to ``len(safety_net)``.
    ``_run_evaluation`` now reserves room for the safety net WITHIN
    ``max_build`` before selection runs, so with ``max_build=1`` and 2
    safety-net entries, the reserved budget (``max(0, 1-2)=0``) selects
    nothing on its own -- the final build count is exactly
    ``len(safety_net)``, not ``max_build + len(safety_net)``."""

    def test_build_count_not_inflated_beyond_safety_net_size(self, monkeypatch):
        monkeypatch.setattr(gates, "max_build_candidates_noncss", lambda: 1)
        ell, m = 6, 6
        safety_net = pbb_eval._pbb_safety_net_candidates(ell, m)
        assert len(safety_net) == 2
        calls = []
        real_build_and_check = pbb_eval._build_and_check

        def counting_build_and_check(ell, m, A, B, C, D):
            calls.append((A, B, C, D))
            return real_build_and_check(ell, m, A, B, C, D)

        monkeypatch.setattr(pbb_eval, "_build_and_check", counting_build_and_check)
        pbb_eval._run_evaluation(pbb_generate, [(ell, m)], quick=True)

        assert len(calls) == len(safety_net)
