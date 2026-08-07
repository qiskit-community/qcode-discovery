"""Tests for scripts/run_weight5_stage0.py (Stage-0 driver, Phase B2/C2).

All tests run at smoke scale (tiny synthetic lattices like (3,3)/(6,6), tiny
CPU-hour ceilings, tiny audit caps) -- this file verifies the driver's
*mechanics* (enumeration counts, backbone-cache correctness, checkpoint/
resume, the budget gate, the statistics helpers), not the real Stage-1
census at full CPU-hour scale. Running the actual small/large-profile
audits is deliberately out of scope here; see the plan's Rollout section.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import scripts.run_weight5_stage0 as stage0
from evaluation.pbb_code import build_pbb_code


class TestBackbonePopulation:
    def test_matches_plan_stated_counts(self):
        # Verbatim counts from plans/direction2_weight5_campaigns.md's B1
        # bounded-sweep description at the three Stage-1 small lattices.
        assert stage0.backbone_population(6, 6) == 3125
        assert stage0.backbone_population(6, 9) == 4500
        assert stage0.backbone_population(9, 8) == 7776

    def test_bounded_backbones_generator_length_matches_population(self):
        assert len(list(stage0.bounded_backbones(6, 6))) == stage0.backbone_population(6, 6)

    def test_every_backbone_is_weight5_canonical_2_3(self):
        for A, B in stage0.bounded_backbones(3, 3):
            assert stage0.css_weight_ok(A, B)

    def test_pbb_pair_population_is_31x_backbone_population(self):
        assert stage0.pbb_backbone_pair_population(6, 6) == 3125 * 31


class TestNontrivialSubsetPairs:
    A = [(0, 0), (1, 1)]
    B = [(0, 0), (1, 0), (2, 2)]

    def test_exactly_31_nontrivial_pairs(self):
        pairs = list(stage0.nontrivial_subset_pairs(self.A, self.B))
        assert len(pairs) == 31

    def test_excludes_empty_empty_pair(self):
        pairs = list(stage0.nontrivial_subset_pairs(self.A, self.B))
        assert ([], []) not in pairs

    def test_every_pair_is_contained_in_backbone(self):
        for C, D in stage0.nontrivial_subset_pairs(self.A, self.B):
            assert set(C).issubset(set(self.A))
            assert set(D).issubset(set(self.B))

    def test_no_duplicate_pairs(self):
        pairs = [(tuple(C), tuple(D)) for C, D in stage0.nontrivial_subset_pairs(self.A, self.B)]
        assert len(pairs) == len(set(pairs))


class TestPbbBackboneCache:
    def test_cache_matches_build_pbb_code_across_backbones_and_subsets(self):
        ell, m = 6, 6
        checked = 0
        for A, B in list(stage0.bounded_backbones(ell, m))[:20]:
            cache = stage0._PbbBackboneCache(ell, m, A, B)
            for C, D in stage0.nontrivial_subset_pairs(A, B):
                checked += 1
                try:
                    cached_code = cache.build(C, D)
                except ValueError as e:
                    with pytest.raises(ValueError) as exc_info:
                        build_pbb_code(ell, m, A, B, C, D)
                    assert str(e) == str(exc_info.value)
                    continue
                real_code = build_pbb_code(ell, m, A, B, C, D)
                assert np.array_equal(np.array(cached_code.matrix), np.array(real_code.matrix))
        assert checked == 20 * 31

    def test_cache_builds_each_term_matrix_at_most_once(self):
        ell, m = 6, 6
        A, B = [(0, 0), (1, 1)], [(0, 0), (2, 0), (3, 3)]
        cache = stage0._PbbBackboneCache(ell, m, A, B)
        for C, D in stage0.nontrivial_subset_pairs(A, B):
            try:
                cache.build(C, D)
            except ValueError:
                pass
        all_terms = {tuple(t) for t in A} | {tuple(t) for t in B}
        assert set(cache._term_cache.keys()) == all_terms


class TestWilsonInterval:
    def test_zero_trials_gives_full_width_interval(self):
        lo, hi = stage0.wilson_interval(0, 0)
        assert lo == 0.0 and hi == 1.0

    def test_all_successes_interval_near_one(self):
        lo, hi = stage0.wilson_interval(100, 100)
        assert 0.9 < lo <= 1.0 and hi == 1.0

    def test_all_failures_interval_near_zero(self):
        lo, hi = stage0.wilson_interval(0, 100)
        assert lo < 1e-9 and hi < 0.1

    def test_interval_contains_point_estimate(self):
        lo, hi = stage0.wilson_interval(50, 100)
        assert lo < 0.5 < hi

    def test_narrows_with_more_trials_same_rate(self):
        lo_small, hi_small = stage0.wilson_interval(5, 10)
        lo_large, hi_large = stage0.wilson_interval(500, 1000)
        assert (hi_large - lo_large) < (hi_small - lo_small)


class TestStratifiedRateEstimate:
    def test_empty_strata_gives_zero(self):
        result = stage0.stratified_rate_estimate([])
        assert result["point_estimate"] == 0.0

    def test_single_stratum_matches_simple_proportion(self):
        result = stage0.stratified_rate_estimate(
            [{"population": 100, "sample_n": 10, "successes": 5}]
        )
        assert result["point_estimate"] == pytest.approx(0.5)

    def test_weighted_estimate_differs_from_naive_pooled_when_quotas_differ(self):
        # Stratum A: huge population, tiny sample, all-failure sample.
        # Stratum B: tiny population, huge sample, all-success sample.
        # A naive pooled rate (sum successes / sum n) is dominated by
        # stratum B's oversized sample; the population-weighted estimate
        # must instead be dominated by stratum A's oversized population --
        # this is exactly the distinction the plan calls out (never an
        # unweighted pooled rate when quotas differ across strata).
        strata = [
            {"population": 1_000_000, "sample_n": 10, "successes": 0},
            {"population": 10, "sample_n": 1000, "successes": 1000},
        ]
        weighted = stage0.stratified_rate_estimate(strata)
        naive_pooled = sum(s["successes"] for s in strata) / sum(s["sample_n"] for s in strata)
        assert weighted["point_estimate"] < 0.01
        assert naive_pooled > 0.9
        assert abs(weighted["point_estimate"] - naive_pooled) > 0.5

    def test_zero_sample_stratum_excluded_without_dividing_by_zero(self):
        result = stage0.stratified_rate_estimate(
            [{"population": 100, "sample_n": 0, "successes": 0},
             {"population": 100, "sample_n": 10, "successes": 10}]
        )
        assert result["point_estimate"] == pytest.approx(0.5)


class TestCssCheapCensusCheckpointResume:
    def test_resume_matches_uninterrupted_run(self, tmp_path):
        d_full = tmp_path / "full"
        d_resumed = tmp_path / "resumed"

        full = stage0.run_css_cheap_census(6, 6, d_full, checkpoint_interval=1000)

        d_resumed.mkdir()
        backbones = list(stage0.bounded_backbones(6, 6))
        counts = stage0._new_counts()
        per_stratum: dict = {}
        surv_path = stage0._survivors_path(d_resumed, "css", 6, 6)
        with open(surv_path, "a", encoding="utf-8") as f:
            for A, B in backbones[:10]:
                counts["total_enumerated"] += 1
                code = stage0.build_bb_code(6, 6, A, B)
                n, k = stage0.get_code_params_fast(code)
                if k == 0:
                    counts["k_zero"] += 1
                else:
                    counts["k_positive"] += 1
                    band = stage0.k_band_for(k)
                    if band:
                        per_stratum[band] = per_stratum.get(band, 0) + 1
                        f.write(json.dumps(
                            {"A_terms": A, "B_terms": B, "n": n, "k": k, "band": band}
                        ) + "\n")

        config_hash = stage0._config_hash(
            {"family": "css", "ell": 6, "m": 6, "config_version": stage0._CONFIG_VERSION}
        )
        stage0._atomic_write_json(
            stage0._checkpoint_path(d_resumed, "css", 6, 6),
            {"config_hash": config_hash, "cursor": 10, "counts": counts,
             "per_stratum_population": per_stratum, "elapsed_seconds": 0.0,
             "status": "in_progress"},
        )

        resumed = stage0.run_css_cheap_census(6, 6, d_resumed, checkpoint_interval=1000)
        assert resumed["counts"] == full["counts"]
        assert resumed["per_stratum_population"] == full["per_stratum_population"]

        def _norm(survs):
            return sorted(
                (tuple(map(tuple, r["A_terms"])), tuple(map(tuple, r["B_terms"])))
                for r in survs
            )

        assert _norm(stage0._read_survivors(stage0._survivors_path(d_full, "css", 6, 6))) == \
            _norm(stage0._read_survivors(surv_path))

    def test_config_hash_mismatch_forces_restart_not_garbage_resume(self, tmp_path):
        d = tmp_path / "mismatch"
        d.mkdir()
        stage0._atomic_write_json(
            stage0._checkpoint_path(d, "css", 6, 6),
            {"config_hash": "not-the-real-hash", "cursor": 999,
             "counts": stage0._new_counts(), "per_stratum_population": {},
             "elapsed_seconds": 0.0, "status": "in_progress"},
        )
        result = stage0.run_css_cheap_census(6, 6, d, checkpoint_interval=1000)
        assert result["counts"]["total_enumerated"] == 3125

    def test_complete_checkpoint_is_the_cached_path_and_matches_uncached_result(
        self, tmp_path, monkeypatch,
    ):
        d = tmp_path / "already_complete"
        uncached = stage0.run_css_cheap_census(6, 6, d, checkpoint_interval=1000)
        assert uncached["status"] == "complete"

        def _boom(*a, **kw):
            raise AssertionError("cached path must not recompute a completed census")

        monkeypatch.setattr(stage0, "build_bb_code", _boom)
        cached = stage0.run_css_cheap_census(6, 6, d, checkpoint_interval=1000)
        assert cached == uncached

    def test_force_flag_ignores_existing_checkpoint(self, tmp_path):
        d = tmp_path / "forced"
        first = stage0.run_css_cheap_census(6, 6, d, checkpoint_interval=1000)
        second = stage0.run_css_cheap_census(6, 6, d, checkpoint_interval=1000, force=True)
        assert second["counts"] == first["counts"]


class TestPbbCheapCensusCheckpointResume:
    def test_resume_mid_backbone_matches_uninterrupted_run(self, tmp_path):
        ell, m = 3, 3
        d_full = tmp_path / "full"
        d_resumed = tmp_path / "resumed"

        full = stage0.run_pbb_cheap_census(ell, m, d_full, checkpoint_interval=10000)

        d_resumed.mkdir()
        backbones = list(stage0.bounded_backbones(ell, m))
        cursor_target = 5  # mid-way through the first backbone's 31 subset pairs
        counts = stage0._new_counts()
        per_stratum: dict = {}
        surv_path = stage0._survivors_path(d_resumed, "pbb", ell, m)
        A, B = backbones[0]
        cache = stage0._PbbBackboneCache(ell, m, A, B)
        pairs = list(stage0.nontrivial_subset_pairs(A, B))
        with open(surv_path, "a", encoding="utf-8") as f:
            for s_idx in range(cursor_target):
                C, D = pairs[s_idx]
                counts["total_enumerated"] += 1
                try:
                    code = cache.build(C, D)
                except ValueError:
                    counts["commutativity_rejected"] += 1
                    continue
                n, k = stage0.get_pbb_params_fast(code)
                if k == 0:
                    counts["k_zero"] += 1
                    continue
                counts["k_positive"] += 1
                lc = stage0.verify_not_lc_css(ell, m, A, B, C, D)
                if lc["is_lc_css"]:
                    counts["lc_css_rejected"] += 1
                    continue
                counts["true_survivors"] += 1
                band = stage0.k_band_for(k)
                if band:
                    per_stratum[band] = per_stratum.get(band, 0) + 1
                    f.write(json.dumps({
                        "A_terms": A, "B_terms": B, "C_terms": C, "D_terms": D,
                        "n": n, "k": k, "band": band,
                    }) + "\n")

        config_hash = stage0._config_hash({
            "family": "pbb", "ell": ell, "m": m, "config_version": stage0._CONFIG_VERSION,
            "skip_lc_css_census": False,
        })
        stage0._atomic_write_json(
            stage0._checkpoint_path(d_resumed, "pbb", ell, m),
            {"config_hash": config_hash, "cursor": cursor_target, "counts": counts,
             "per_stratum_population": per_stratum, "elapsed_seconds": 0.0,
             "status": "in_progress"},
        )

        resumed = stage0.run_pbb_cheap_census(ell, m, d_resumed, checkpoint_interval=10000)
        assert resumed["counts"] == full["counts"]
        assert resumed["per_stratum_population"] == full["per_stratum_population"]

    def test_skip_lc_css_census_changes_config_hash_and_survivor_counts(self, tmp_path):
        ell, m = 3, 3
        with_lc = stage0.run_pbb_cheap_census(
            ell, m, tmp_path / "with_lc", checkpoint_interval=10000, skip_lc_css_census=False,
        )
        without_lc = stage0.run_pbb_cheap_census(
            ell, m, tmp_path / "without_lc", checkpoint_interval=10000, skip_lc_css_census=True,
        )
        assert with_lc["config_hash"] != without_lc["config_hash"]
        # Skipping the LC-CSS pass means every k>0 candidate is a "true
        # survivor" by definition (no lc_css_rejected accounting).
        assert without_lc["counts"]["true_survivors"] == without_lc["counts"]["k_positive"]
        assert without_lc["counts"]["lc_css_rejected"] == 0
        assert with_lc["counts"]["true_survivors"] <= with_lc["counts"]["k_positive"]


class TestBudgetGate:
    def test_benchmark_reports_within_budget_for_generous_ceiling(self):
        bench = stage0.benchmark_and_check_budget("css", 6, 6, ceiling_hours=1000.0, slice_size=20)
        assert bench["within_budget"]
        assert bench["population"] == 3125

    def test_benchmark_reports_over_budget_for_tiny_ceiling(self):
        bench = stage0.benchmark_and_check_budget("css", 6, 6, ceiling_hours=1e-12, slice_size=20)
        assert not bench["within_budget"]

    def test_zero_population_lattice_is_trivially_within_budget(self):
        bench = stage0.benchmark_and_check_budget("css", 1, 6, ceiling_hours=1e-12, slice_size=20)
        assert bench["population"] == 0
        assert bench["within_budget"]

    def test_cli_main_writes_budget_exceeded_checkpoint_and_never_runs_census(
        self, tmp_path, monkeypatch,
    ):
        called = []
        monkeypatch.setattr(
            stage0, "run_css_cheap_census", lambda *a, **kw: called.append(1)
        )
        argv = [
            "--family", "css", "--lattices", "6,6",
            "--checkpoint-dir", str(tmp_path),
            "--cheap-cpu-hours-per-lattice", "1e-12",
            "--benchmark-slice", "10",
        ]
        rc = stage0.main(argv)
        assert rc == 0
        assert not called, "census must not run when the budget gate fails"

        report = json.loads((tmp_path / "stage0_report.json").read_text())
        assert report["lattices"][0]["status"] == "budget_exceeded"
        ckpt = json.loads(stage0._checkpoint_path(tmp_path, "css", 6, 6).read_text())
        assert ckpt["status"] == "budget_exceeded"

        md = (tmp_path / "stage0_report.md").read_text()
        assert "BUDGET EXCEEDED" in md

    def test_pbb_benchmark_includes_lc_css_cost_by_default(self, monkeypatch):
        # The real census (run_pbb_cheap_census) runs verify_not_lc_css on
        # every k>0 survivor unless --skip-lc-css-census. If the benchmark
        # slice skips that work, projected_hours systematically
        # underestimates the true per-item cost and the budget gate can't
        # do its job.
        calls = []
        real_verify = stage0.verify_not_lc_css

        def counting_verify(*a, **kw):
            calls.append(1)
            return real_verify(*a, **kw)

        monkeypatch.setattr(stage0, "verify_not_lc_css", counting_verify)
        # slice_size=200 covers enough backbones (~7) that at least one of
        # (3,3)'s 24 real k>0 survivors (out of 992 candidates total) is
        # almost certain to land in the sampled slice.
        stage0.benchmark_and_check_budget("pbb", 3, 3, ceiling_hours=1000.0, slice_size=200)
        assert calls, "benchmark must exercise verify_not_lc_css for the default census"

    def test_pbb_benchmark_skips_lc_css_cost_when_requested(self, monkeypatch):
        calls = []
        real_verify = stage0.verify_not_lc_css

        def counting_verify(*a, **kw):
            calls.append(1)
            return real_verify(*a, **kw)

        monkeypatch.setattr(stage0, "verify_not_lc_css", counting_verify)
        stage0.benchmark_and_check_budget(
            "pbb", 3, 3, ceiling_hours=1000.0, slice_size=20, skip_lc_css_census=True,
        )
        assert not calls, "skip_lc_css_census=True must not run the LC-CSS check"


class TestPbbSampledAuditMaxWeight:
    """Regression coverage for the codex finding that the max_weight cutoff
    compared ``ell * m`` to 216, but the actual code length governing
    has_low_weight_logical's exhaustive search cost is ``n = 2 * ell * m``.
    Lattices like (15,12) have ell*m=180<=216 (old code: max_weight=6) but
    n=360>216 (correct: max_weight=4)."""

    @staticmethod
    def _one_survivor(ell, m):
        return [{
            "A_terms": [[0, 0], [1, 1]], "B_terms": [[0, 0], [2, 0], [1, 3]],
            "C_terms": [[0, 0]], "D_terms": [[1, 3]], "band": "k_2_3",
            "n": 2 * ell * m, "k": 2,
        }]

    def _capture_max_weight(self, monkeypatch):
        monkeypatch.setattr(stage0, "build_pbb_code", lambda *a, **kw: object())
        monkeypatch.setattr(stage0, "passes_noncss_gate", lambda *a, **kw: {"passes": True})
        captured = {}

        def fake_has_low_weight_logical(code, max_weight):
            captured["max_weight"] = max_weight
            return False, 0

        monkeypatch.setattr(stage0, "has_low_weight_logical", fake_has_low_weight_logical)
        return captured

    def test_large_lattice_with_ell_times_m_under_cutoff_still_gets_weight4(
        self, monkeypatch,
    ):
        ell, m = 15, 12  # ell*m=180 <= 216, but n=2*ell*m=360 > 216
        captured = self._capture_max_weight(monkeypatch)
        stage0.run_pbb_sampled_audit(
            ell, m, self._one_survivor(ell, m), run_seed=0, max_audit_candidates=10,
        )
        assert captured["max_weight"] == 4

    def test_small_lattice_still_gets_weight6(self, monkeypatch):
        ell, m = 6, 6  # n=72 <= 216
        captured = self._capture_max_weight(monkeypatch)
        stage0.run_pbb_sampled_audit(
            ell, m, self._one_survivor(ell, m), run_seed=0, max_audit_candidates=10,
        )
        assert captured["max_weight"] == 6


class TestCliEndToEnd:
    def test_css_end_to_end_smoke(self, tmp_path):
        argv = [
            "--family", "css", "--lattices", "3,3",
            "--checkpoint-dir", str(tmp_path),
            "--max-audit-candidates", "5", "--quick-trials", "3",
            "--benchmark-slice", "8", "--cheap-cpu-hours-per-lattice", "1",
        ]
        rc = stage0.main(argv)
        assert rc == 0
        report = json.loads((tmp_path / "stage0_report.json").read_text())
        lr = report["lattices"][0]
        assert lr["status"] == "complete"
        assert lr["census"]["counts"]["total_enumerated"] == stage0.backbone_population(3, 3)
        assert "audit" in lr
        assert "stratified_overall_rate" in lr["audit"]
        assert (tmp_path / "stage0_report.md").exists()

    def test_pbb_end_to_end_smoke_with_skip_lc_css(self, tmp_path):
        argv = [
            "--family", "pbb", "--lattices", "3,3",
            "--checkpoint-dir", str(tmp_path),
            "--max-audit-candidates", "5",
            "--benchmark-slice", "8", "--cheap-cpu-hours-per-lattice", "1",
            "--skip-lc-css-census",
        ]
        rc = stage0.main(argv)
        assert rc == 0
        report = json.loads((tmp_path / "stage0_report.json").read_text())
        lr = report["lattices"][0]
        assert lr["status"] == "complete"
        assert lr["census"]["counts"]["total_enumerated"] == stage0.pbb_backbone_pair_population(3, 3)
        assert lr["audit"]["lc_css_census_skipped"] is True

        md = (tmp_path / "stage0_report.md").read_text()
        assert "skipped" in md.lower()

    def test_explicit_lattices_override_profile_default(self, tmp_path):
        argv = [
            "--family", "css", "--profile", "small", "--lattices", "3,3",
            "--checkpoint-dir", str(tmp_path),
            "--max-audit-candidates", "1", "--quick-trials", "1",
            "--benchmark-slice", "8", "--cheap-cpu-hours-per-lattice", "1",
        ]
        rc = stage0.main(argv)
        assert rc == 0
        report = json.loads((tmp_path / "stage0_report.json").read_text())
        assert len(report["lattices"]) == 1
        assert (report["lattices"][0]["ell"], report["lattices"][0]["m"]) == (3, 3)
