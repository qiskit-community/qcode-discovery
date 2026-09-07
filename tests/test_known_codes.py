"""Tests that verify the evaluation pipeline reproduces known BB code parameters.

This module contains 38 tests across 11 test classes covering every layer
of the evaluation pipeline:

``TestTermsToPoly``
    Verifies sympy polynomial construction from exponent tuples.
``TestValidation``
    Verifies input validation: correct trinomials pass, wrong count /
    out-of-range / duplicate monomials raise ``ValueError``.
``TestBuildBBCode``
    Verifies ``build_bb_code`` produces correct ``(n, k)`` for the
    [[144,12,12]] gross code and the [[72,12,6]] code.
``TestKnownCodeParams``
    Parametrized test: verifies ``(n, k)`` for all 14 entries in
    ``KNOWN_CODES`` that have expected ``k``.
``TestFOM``
    Verifies ``compute_fom`` arithmetic and edge cases (``k=0``, ``d=0``).
``TestDistanceEstimate`` (marked ``slow``)
    Verifies BP-OSD upper bounds are reasonable for two benchmark codes.
``TestEvaluator``
    Verifies cascade stages: ``quick_k_only``, ``invalid``, ``k_zero``,
    ``k_low``, and full evaluation.
``TestEvaluateBatch``
    Verifies ``evaluate_batch`` and ``evaluate_lattices`` work end-to-end
    with ``generate_candidates`` output.
``TestResultsPersistence``
    Verifies JSON save/load round-trip, deduplication, and Pareto front
    dominance filtering with incremental merging.
``TestSeedSolution``
    Verifies ``generate_candidates`` returns valid trinomials and includes
    all ``KNOWN_CODES`` for their respective lattices.
``TestTracking``
    Verifies the ``RunTracker`` lifecycle: start_run, log_evaluation,
    end_generation, end_run; file creation and loader functions.
"""

import tempfile
from pathlib import Path

import pytest
from evaluation.bb_code import build_bb_code, validate_terms, terms_to_poly, get_code_params_fast
from evaluation.evaluator import compute_fom, evaluate_candidate, evaluate_batch, evaluate_lattices
from evaluation.results import save_code, load_codes, update_pareto_front
from evaluation.tracking import RunTracker, load_run_generations, load_run_meta
from evolve.seed_solution import KNOWN_CODES, generate_candidates


# Filter to codes with fully known expected params for parametrized tests
CODES_WITH_KNOWN_K = [c for c in KNOWN_CODES if c["expected"][1] is not None]


class TestTermsToPoly:
    def test_gross_code_poly_a(self):
        from sympy.abc import x, y
        poly = terms_to_poly([(3, 0), (0, 1), (0, 2)])
        # x^3 + y + y^2 -- evaluating at (1,1) over integers gives 3 (1 mod 2)
        assert poly.subs([(x, 1), (y, 1)]) == 3

    def test_identity_monomial(self):
        from sympy.abc import x, y
        poly = terms_to_poly([(0, 0), (1, 0), (0, 1)])
        # 1 + x + y -- evaluating at (1,1) over integers gives 3
        assert poly.subs([(x, 1), (y, 1)]) == 3


class TestValidation:
    def test_valid_trinomial(self):
        validate_terms(12, 6, [(3, 0), (1, 0), (0, 1)], "A")

    def test_too_few_terms_validation(self):
        with pytest.raises(ValueError, match="2-6 terms"):
            validate_terms(12, 6, [(3, 0)], "A")

    def test_too_many_terms_validation(self):
        with pytest.raises(ValueError, match="2-6 terms"):
            validate_terms(
                12, 6,
                [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (0, 1)],
                "A",
            )

    def test_valid_2_terms(self):
        validate_terms(12, 6, [(3, 0), (1, 0)], "A")

    def test_valid_4_terms(self):
        validate_terms(12, 6, [(0, 0), (3, 0), (0, 1), (0, 2)], "A")

    def test_valid_5_terms(self):
        validate_terms(12, 6, [(3, 0), (0, 1), (0, 2), (1, 1), (2, 3)], "A")

    def test_x_out_of_range(self):
        with pytest.raises(ValueError, match="x-exponent"):
            validate_terms(12, 6, [(12, 0), (1, 0), (0, 1)], "A")

    def test_y_out_of_range(self):
        with pytest.raises(ValueError, match="y-exponent"):
            validate_terms(12, 6, [(3, 0), (1, 0), (0, 6)], "A")

    def test_duplicate_monomials(self):
        with pytest.raises(ValueError, match="duplicate"):
            validate_terms(12, 6, [(3, 0), (3, 0), (0, 1)], "A")

    def test_non_integer_exponent_rejected(self):
        # A float exponent (e.g. from a buggy evolved generator) is in
        # range and reduces fine under `%`, so it would otherwise pass
        # silently instead of being caught here.
        with pytest.raises(ValueError, match="must be integers"):
            validate_terms(12, 6, [(3.0, 0), (1, 0), (0, 1)], "A")


class TestBuildBBCode:
    def test_gross_code_construction(self):
        """The [[144,12,12]] gross code should have n=144, k=12."""
        # A = x^3 + y + y^2, B = y^3 + x + x^2
        code = build_bb_code(12, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
        n, k = get_code_params_fast(code)
        assert n == 144
        assert k == 12

    def test_72_code_construction(self):
        """The [[72,12,6]] code should have n=72, k=12."""
        code = build_bb_code(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
        n, k = get_code_params_fast(code)
        assert n == 72
        assert k == 12


class TestKnownCodeParams:
    """Verify n and k for all known benchmark codes."""

    @pytest.mark.parametrize(
        "code_spec", CODES_WITH_KNOWN_K,
        ids=[c["name"] for c in CODES_WITH_KNOWN_K],
    )
    def test_n_k(self, code_spec):
        code = build_bb_code(
            code_spec["ell"], code_spec["m"],
            code_spec["A_terms"], code_spec["B_terms"],
        )
        n, k = get_code_params_fast(code)
        expected_n, expected_k = code_spec["expected"][0], code_spec["expected"][1]
        assert n == expected_n, f"Expected n={expected_n}, got {n}"
        assert k == expected_k, f"Expected k={expected_k}, got {k}"


class TestFOM:
    def test_gross_code_fom(self):
        assert compute_fom(144, 12, 12) == 12.0

    def test_zero_k(self):
        assert compute_fom(144, 0, 12) == 0.0

    def test_zero_d(self):
        assert compute_fom(144, 12, 0) == 0.0


class TestDistanceEstimate:
    """Test distance estimation -- these are slower (seconds per test)."""

    @pytest.mark.slow
    def test_gross_code_distance_bound(self):
        """Distance upper bound for [[144,12,12]] should be reasonable."""
        code = build_bb_code(12, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
        from evaluation.distance import estimate_distance
        d_upper = estimate_distance(code, num_trials=100)
        assert d_upper > 0
        # BP-OSD with only 100 trials may not find a tight bound;
        # exact distance is 12 but the stochastic upper bound can be higher
        assert d_upper <= 30

    @pytest.mark.slow
    def test_72_code_distance_bound(self):
        """Distance upper bound for [[72,12,6]] should be reasonable."""
        code = build_bb_code(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
        from evaluation.distance import estimate_distance
        d_upper = estimate_distance(code, num_trials=100)
        assert d_upper > 0
        assert d_upper <= 12  # loose bound; exact is 6


class TestEvaluator:
    def test_gross_code_quick(self):
        """Quick evaluation should return correct n, k."""
        result = evaluate_candidate(
            12, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)],
            quick=True,
        )
        assert result["n"] == 144
        assert result["k"] == 12
        assert result["stage"] == "quick_k_only"

    def test_invalid_terms(self):
        """Invalid terms should be rejected."""
        result = evaluate_candidate(
            12, 6, [(3, 0), (3, 0), (0, 1)], [(3, 0), (0, 1), (0, 2)],
        )
        assert result["stage"] == "invalid"
        assert result["score"] == float("-inf")

    def test_one_term_rejected(self):
        """Single-term polynomial should be rejected."""
        result = evaluate_candidate(
            12, 6, [(3, 0)], [(3, 0), (0, 1), (0, 2)],
        )
        assert result["stage"] == "invalid"

    def test_k_zero_stage(self):
        """Code with k=0 should be rejected at k_zero stage."""
        # A = 1 + x + y, B = 1 + x + y at (2,2) gives k=0
        result = evaluate_candidate(
            2, 2, [(0, 0), (1, 0), (0, 1)], [(0, 0), (1, 0), (0, 1)],
        )
        assert result["k"] == 0, f"Expected k=0 at (2,2) with A=B=1+x+y, got k={result['k']}"
        assert result["stage"] == "k_zero"
        assert result["score"] == float("-inf")

    def test_k_low_stage(self):
        """Code with 0 < k < 4 should get self_dual_d2 or k_low stage."""
        # A=B produces self-dual code; evaluator catches A=B before k_low
        result = evaluate_candidate(
            3, 3, [(1, 0), (0, 1), (0, 2)], [(1, 0), (0, 1), (0, 2)],
        )
        assert result["k"] > 0, f"Expected k>0 at (3,3), got k={result['k']}"
        # A=B codes are caught by the self-dual gate
        assert result["stage"] in ("self_dual_d2", "k_low")

    @pytest.mark.slow
    def test_gross_code_full_evaluation(self):
        """Full evaluation of the gross code."""
        result = evaluate_candidate(
            12, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)],
            quick_trials=50,
            fom_threshold_refine=100.0,  # skip refine for speed
            fom_threshold_exact=100.0,   # skip exact for speed
        )
        assert result["n"] == 144
        assert result["k"] == 12
        assert result["d"] > 0
        assert result["fom"] > 0


class TestEvaluateBatch:
    def test_batch_with_generate_candidates(self):
        """evaluate_batch should accept generate_candidates output."""
        candidates = generate_candidates(6, 6)
        assert len(candidates) > 0
        # Quick evaluation only -- just test the interface works
        results = evaluate_batch(6, 6, candidates[:3], quick=True)
        assert len(results) == 3
        for r in results:
            assert "n" in r
            assert "k" in r
            assert r["n"] == 72

    def test_evaluate_lattices(self):
        """evaluate_lattices should run across multiple lattices."""
        results = evaluate_lattices(
            [(6, 6)],
            lambda ell, m: generate_candidates(ell, m)[:2],
            quick=True,
        )
        assert len(results) == 2


class TestResultsPersistence:
    def test_save_and_load(self):
        """Save and load codes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "codes.json"
            result = {
                "ell": 12, "m": 6,
                "A_terms": [(3, 0), (0, 1), (0, 2)],
                "B_terms": [(0, 3), (1, 0), (2, 0)],
                "n": 144, "k": 12, "d": 12, "fom": 12.0,
            }
            save_code(result, filepath)
            loaded = load_codes(filepath)
            assert len(loaded) == 1
            assert loaded[0]["fom"] == 12.0

            # Saving same code again should not duplicate
            save_code(result, filepath)
            loaded = load_codes(filepath)
            assert len(loaded) == 1

    def test_pareto_front(self):
        """Pareto front should keep non-dominated codes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "pareto.json"
            results = [
                {"n": 144, "k": 12, "d": 12, "fom": 12.0,
                 "ell": 12, "m": 6,
                 "A_terms": [(3, 0), (0, 1), (0, 2)],
                 "B_terms": [(0, 3), (1, 0), (2, 0)]},
                {"n": 144, "k": 12, "d": 14, "fom": 16.3,
                 "ell": 12, "m": 6,
                 "A_terms": [(3, 0), (0, 2), (0, 3)],
                 "B_terms": [(0, 3), (1, 0), (2, 0)]},
                {"n": 72, "k": 12, "d": 6, "fom": 6.0,
                 "ell": 6, "m": 6,
                 "A_terms": [(3, 0), (0, 1), (0, 2)],
                 "B_terms": [(0, 3), (1, 0), (2, 0)]},
            ]
            front = update_pareto_front(results, filepath)
            # Code with d=14 dominates d=12 (same n, same k, better d)
            assert len(front) == 2
            foms = {r["fom"] for r in front}
            assert 16.3 in foms
            assert 6.0 in foms  # different trade-off (smaller n)

    def test_pareto_front_merges_with_existing(self):
        """Pareto front should merge with existing front on disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "pareto.json"
            # First run
            results1 = [
                {"n": 144, "k": 12, "d": 12, "fom": 12.0,
                 "ell": 12, "m": 6,
                 "A_terms": [(3, 0), (0, 1), (0, 2)],
                 "B_terms": [(0, 3), (1, 0), (2, 0)]},
            ]
            front1 = update_pareto_front(results1, filepath)
            assert len(front1) == 1

            # Second run with a different non-dominated code
            results2 = [
                {"n": 72, "k": 12, "d": 6, "fom": 6.0,
                 "ell": 6, "m": 6,
                 "A_terms": [(3, 0), (0, 1), (0, 2)],
                 "B_terms": [(0, 3), (1, 0), (2, 0)]},
            ]
            front2 = update_pareto_front(results2, filepath)
            # Should have both: old (n=144,d=12) and new (n=72,d=6)
            assert len(front2) == 2
            foms = {r["fom"] for r in front2}
            assert 12.0 in foms
            assert 6.0 in foms


class TestSeedSolution:
    def test_generate_candidates_returns_list(self):
        cands = generate_candidates(12, 6)
        assert isinstance(cands, list)
        assert len(cands) > 0

    def test_candidates_are_valid_trinomials(self):
        """All candidates should pass validation."""
        cands = generate_candidates(6, 6)
        for A, B in cands:
            assert len(A) == 3
            assert len(B) == 3
            assert len(set(map(tuple, A))) == 3
            assert len(set(map(tuple, B))) == 3
            for ax, ay in A:
                assert 0 <= ax < 6
                assert 0 <= ay < 6
            for bx, by in B:
                assert 0 <= bx < 6
                assert 0 <= by < 6

    def test_known_codes_in_candidates(self):
        """Known good codes should appear in generate_candidates output."""
        for spec in KNOWN_CODES:
            ell, m = spec["ell"], spec["m"]
            cands = generate_candidates(ell, m)
            target_key = (
                tuple(sorted(map(tuple, spec["A_terms"]))),
                tuple(sorted(map(tuple, spec["B_terms"]))),
            )
            cand_keys = {
                (tuple(sorted(map(tuple, A))), tuple(sorted(map(tuple, B))))
                for A, B in cands
            }
            assert target_key in cand_keys, (
                f"{spec['name']} not found in candidates for ({ell},{m})"
            )


class TestTracking:
    def test_run_tracker_lifecycle(self):
        """Tracker should create files and record metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = RunTracker(base_dir=tmpdir)
            tracker.start_run(run_id="test-001", config={"quick": True})

            tracker.start_generation(0)
            result = evaluate_candidate(
                6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)],
                quick=True,
            )
            tracker.log_evaluation(result)
            summary = tracker.end_generation(0, [result])

            assert summary["total_candidates"] == 1
            assert summary["valid_candidates"] == 1

            meta = tracker.end_run()
            assert meta["run_id"] == "test-001"
            assert meta["status"] == "completed"
            assert meta["total_evaluations"] == 1

            # Files should exist
            run_dir = Path(tmpdir) / "test-001"
            assert (run_dir / "run_meta.json").exists()
            assert (run_dir / "evaluations.jsonl").exists()
            assert (run_dir / "generations.jsonl").exists()

            # Load helpers
            gens = load_run_generations(run_dir)
            assert len(gens) == 1
            loaded_meta = load_run_meta(run_dir)
            assert loaded_meta["run_id"] == "test-001"
