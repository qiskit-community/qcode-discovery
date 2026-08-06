"""Tests for the schema-v2 distance certification module and archive.

Covers, at minimum, every bullet in plans/direction2_weight5_campaigns.md's
Phase D "Distance" test list (one test per bullet, named after its
content), plus two additional cases the plan explicitly calls out:
a legacy record with a pre-existing ``d_is_exact=True`` flag migrating as
truly exact, and idempotency of ``update_pareto_front_v2``.
"""

import json
import tempfile
from pathlib import Path

import pytest

from evaluation import distance_schema, pareto_v2
from evaluation.distance_schema import (
    DistanceBoundConflict,
    aggregate_component_bounds,
    build_v2_record,
    certified_fom,
    empty_component_bounds,
    exploratory_fom,
    is_exact,
    merge_bounds,
    merge_component_bound,
    migrate_legacy_record,
    webster_status_effect,
)
from evaluation.pareto_v2 import load_pareto_v2, update_pareto_front_v2


def _v1_result(**overrides) -> dict:
    """A minimal v1-shaped evaluator result dict, with sensible defaults."""
    base = {
        "ell": 6, "m": 6,
        "A_terms": [(3, 0), (0, 1), (0, 2)],
        "B_terms": [(0, 3), (1, 0), (2, 0)],
        "n": 72, "k": 12, "d": 0,
        "d_is_exact": False, "distance_trusted": False,
        "fom": 0.0, "encoding_rate": 12 / 72,
        "score": 0.0, "stage": "rejected",
    }
    base.update(overrides)
    return base


class TestHeuristicFailureNeverRaisesDLower:
    """heuristic failure never raises d_lower."""

    def test_large_heuristic_d_gives_zero_lower(self):
        result = _v1_result(
            stage="quick_estimate", d=99, distance_trusted=False,
            fom=13.6, score=13.6,
        )
        v2 = build_v2_record(result)
        assert v2["d_lower"] == 0
        assert v2["d_upper"] == 99
        assert v2["bound_sources"] == ["bp_osd_heuristic"]
        assert v2["d_is_exact"] is False

    def test_refined_estimate_large_d_gives_zero_lower(self):
        result = _v1_result(stage="refined_estimate", d=40, distance_trusted=True)
        v2 = build_v2_record(result)
        assert v2["d_lower"] == 0
        assert v2["d_upper"] == 40


class TestExhaustiveNoLogicalRaisesLowerToWPlusOne:
    """exhaustive no-logical-through-w raises it to w+1."""

    def test_merge_bounds_exhaustive_raises_lower(self):
        # Enumeration searched through weight w=5 and found nothing, so
        # the certified lower bound may rise to w+1=6.
        lower, upper = merge_bounds(
            0, None, 6, None, new_source_kind="exhaustive",
        )
        assert lower == 6
        assert upper is None

    def test_merge_bounds_exhaustive_respects_max(self):
        # An existing higher certified lower bound is not lowered.
        lower, upper = merge_bounds(
            8, None, 6, None, new_source_kind="exhaustive",
        )
        assert lower == 8


class TestVerifiedWitnessTightensOnlyUpper:
    """a verified witness tightens only d_upper."""

    def test_heuristic_witness_lowers_upper_leaves_lower(self):
        # old: some certified lower bound of 4, unknown upper.
        # new: a heuristic/witness estimate of upper=10 (kind "heuristic",
        # not a proof source) -- must not touch lower.
        lower, upper = merge_bounds(
            4, None, 0, 10, new_source_kind="heuristic",
        )
        assert lower == 4
        assert upper == 10

    def test_witness_tightens_existing_upper(self):
        lower, upper = merge_bounds(
            4, 20, 0, 10, new_source_kind="heuristic",
        )
        assert lower == 4
        assert upper == 10  # min(20, 10)


class TestMergeConflictsAreQuarantined:
    """merge conflicts are quarantined."""

    def test_merge_bounds_raises_conflict(self):
        # old upper=3 (some prior witness), new lower=6 via a real proof
        # source -- 6 > 3 is a genuine contradiction.
        with pytest.raises(DistanceBoundConflict):
            merge_bounds(0, 3, 6, None, new_source_kind="solver_proof")

    def test_merge_component_bound_raises_conflict(self):
        cb = empty_component_bounds()
        cb = merge_component_bound(
            cb, "X", 0, 3, source_kind="heuristic", source_label="witness1",
        )
        with pytest.raises(DistanceBoundConflict):
            merge_component_bound(
                cb, "X", 6, None,
                source_kind="solver_proof", source_label="milp_proof",
            )

    def test_update_pareto_front_v2_quarantines_instead_of_crashing(self, tmp_path):
        filepath = tmp_path / "pareto_v2.json"

        old_result = _v1_result(
            ell=6, m=6, A_terms=[(1, 0)], B_terms=[(2, 0)],
            n=72, k=12, d=3,
            stage="milp_infeasible_lower_seed",
        )
        old_v2 = build_v2_record(old_result)
        # Manually stamp a proof-backed lower bound of 6 with an unverified
        # heuristic upper bound of 3 already recorded (a pre-existing
        # inconsistency loaded from disk), to set up a genuine conflict.
        old_v2["d_lower"] = 6
        old_v2["d_upper"] = 3
        old_v2["bound_sources"] = ["milp_infeasible_early_stop"]

        first = update_pareto_front_v2([old_v2], filepath=filepath)
        assert first["quarantined"] == []  # nothing to conflict with yet

        # Now merge in a new heuristic upper-bound witness of 2, which
        # contradicts the stored lower bound of 6.
        new_result = _v1_result(
            ell=6, m=6, A_terms=[(1, 0)], B_terms=[(2, 0)],
            n=72, k=12, d=2, distance_trusted=True,
        )
        new_v2 = build_v2_record(new_result)

        second = update_pareto_front_v2([new_v2], filepath=filepath)
        assert len(second["quarantined"]) == 1
        assert "old" in second["quarantined"][0]
        assert "new" in second["quarantined"][0]
        # The whole call did not crash and returned a valid archive shape.
        assert "exact_front" in second
        assert "lower_bound_front" in second
        assert "exploratory_witnesses" in second


class TestCssComponentAggregationHandlesNullUpperBounds:
    """CSS component aggregation handles null upper bounds."""

    def test_both_uppers_null(self):
        cb = empty_component_bounds()
        cb["X"]["lower"] = 2
        cb["Z"]["lower"] = 3
        lower, upper = aggregate_component_bounds(cb)
        assert lower == 2
        assert upper is None

    def test_one_upper_null_one_set(self):
        cb = empty_component_bounds()
        cb["X"]["lower"], cb["X"]["upper"] = 2, 8
        cb["Z"]["lower"], cb["Z"]["upper"] = 3, None
        lower, upper = aggregate_component_bounds(cb)
        assert lower == 2
        assert upper == 8

    def test_neither_upper_null(self):
        cb = empty_component_bounds()
        cb["X"]["lower"], cb["X"]["upper"] = 2, 8
        cb["Z"]["lower"], cb["Z"]["upper"] = 3, 6
        lower, upper = aggregate_component_bounds(cb)
        assert lower == 2
        assert upper == 6


class TestExactResultsBypassHeuristicGates:
    """exact results bypass heuristic gates."""

    def test_larger_heuristic_upper_does_not_widen_exact_upper(self):
        # old: exact d=6 (lower==upper==6). new: a heuristic estimate with
        # a LARGER upper bound of 20 -- min() must ignore it.
        lower, upper = merge_bounds(6, 6, 0, 20, new_source_kind="heuristic")
        assert lower == 6
        assert upper == 6

    def test_heuristic_source_cannot_raise_lower_past_exact_value(self):
        lower, upper = merge_bounds(6, 6, 50, None, new_source_kind="heuristic")
        assert lower == 6
        assert upper == 6

    def test_exact_record_merge_stays_exact_and_unchanged(self):
        exact_result = _v1_result(d=6, d_is_exact=True, stage="exact")
        exact_v2 = build_v2_record(exact_result)
        assert exact_v2["d_lower"] == exact_v2["d_upper"] == 6
        assert exact_v2["d_is_exact"] is True

        heuristic_result = _v1_result(d=99, distance_trusted=False, stage="quick_estimate")
        heuristic_v2 = build_v2_record(heuristic_result)

        merged = pareto_v2._merge_records(exact_v2, heuristic_v2)
        assert merged["d_lower"] == 6
        assert merged["d_upper"] == 6
        assert merged["d_is_exact"] is True


class TestLegacyBareDRecordsMigrateAsExploratory:
    """legacy bare-d records migrate as exploratory."""

    def test_bare_d_no_flags_migrates_exploratory(self):
        legacy = _v1_result(d=8)
        del legacy["d_is_exact"]
        legacy["stage"] = "some_old_unrecognized_stage"
        v2 = migrate_legacy_record(legacy)
        assert v2["d_lower"] == 0
        assert v2["d_upper"] == 8
        assert v2["d_is_exact"] is False
        assert "legacy_migrated" in v2["bound_sources"]


class TestLegacyExactFlagMigratesAsTrulyExact:
    """A legacy record that already sets d_is_exact=True migrates as truly
    exact (d_lower=d_upper=d), not as exploratory -- the plan's explicit
    D4 exception ("unless an existing exact/proof flag is recognized")."""

    def test_pre_existing_exact_flag_recognized(self):
        legacy = _v1_result(d=8, d_is_exact=True)
        legacy["stage"] = "some_old_unrecognized_stage"
        v2 = migrate_legacy_record(legacy)
        assert v2["d_lower"] == 8
        assert v2["d_upper"] == 8
        assert v2["d_is_exact"] is True
        assert "exact_milp" in v2["bound_sources"]
        assert "legacy_migrated" in v2["bound_sources"]

    def test_pre_existing_exact_flag_with_stage_exact_unaffected(self):
        # Sanity check: when stage=="exact" already matches build_v2_record's
        # own rule, the migration-specific override branch is a no-op.
        legacy = _v1_result(d=10, d_is_exact=True, stage="exact")
        v2 = migrate_legacy_record(legacy)
        assert v2["d_lower"] == 10
        assert v2["d_upper"] == 10
        assert v2["d_is_exact"] is True
        assert v2["bound_sources"].count("exact_milp") == 1


class TestExploratoryEntriesCannotEvictCertifiedFronts:
    """exploratory entries cannot evict certified fronts."""

    def test_exploratory_does_not_touch_exact_front(self, tmp_path):
        filepath = tmp_path / "pareto_v2.json"

        exact_result = _v1_result(
            ell=6, m=6, A_terms=[(1, 0)], B_terms=[(2, 0)],
            n=72, k=12, d=6, d_is_exact=True, stage="exact", fom=6.0,
        )
        exact_v2 = build_v2_record(exact_result)

        # A different code, with a much better exploratory (heuristic) d_upper.
        exploratory_result = _v1_result(
            ell=12, m=12, A_terms=[(1, 0)], B_terms=[(3, 0)],
            n=288, k=24, d=24, distance_trusted=False,
            stage="quick_estimate", fom=48.0,
        )
        exploratory_v2 = build_v2_record(exploratory_result)
        assert exploratory_v2["d_lower"] == 0
        assert exploratory_v2["d_upper"] == 24

        result = update_pareto_front_v2(
            [exact_v2, exploratory_v2], filepath=filepath,
        )

        exact_keys = {(r["ell"], r["m"]) for r in result["exact_front"]}
        exploratory_keys = {(r["ell"], r["m"]) for r in result["exploratory_witnesses"]}

        assert (6, 6) in exact_keys
        assert (6, 6) not in exploratory_keys
        assert (12, 12) in exploratory_keys
        assert (12, 12) not in exact_keys
        assert result["lower_bound_front"] == []

    def test_same_code_heuristic_update_does_not_evict_from_exact_front(self, tmp_path):
        filepath = tmp_path / "pareto_v2.json"

        exact_result = _v1_result(
            ell=6, m=6, A_terms=[(1, 0)], B_terms=[(2, 0)],
            n=72, k=12, d=6, d_is_exact=True, stage="exact", fom=6.0,
        )
        exact_v2 = build_v2_record(exact_result)
        first = update_pareto_front_v2([exact_v2], filepath=filepath)
        assert len(first["exact_front"]) == 1

        # A later heuristic re-estimate of the SAME code (e.g. a stray
        # BP-OSD re-run) must not move it out of exact_front.
        heuristic_witness = _v1_result(
            ell=6, m=6, A_terms=[(1, 0)], B_terms=[(2, 0)],
            n=72, k=12, d=6, distance_trusted=True, stage="quick_estimate",
        )
        heuristic_v2 = build_v2_record(heuristic_witness)

        second = update_pareto_front_v2([heuristic_v2], filepath=filepath)
        assert len(second["exact_front"]) == 1
        assert second["exact_front"][0]["d_is_exact"] is True
        assert second["exploratory_witnesses"] == []
        assert second["lower_bound_front"] == []


class TestUpdateParetoFrontV2Idempotency:
    """Running update_pareto_front_v2 twice with the same input produces
    the same output (no duplicate growth)."""

    def test_idempotent_on_repeated_input(self, tmp_path):
        filepath = tmp_path / "pareto_v2.json"

        records = [
            build_v2_record(_v1_result(
                ell=6, m=6, A_terms=[(1, 0)], B_terms=[(2, 0)],
                n=72, k=12, d=6, d_is_exact=True, stage="exact", fom=6.0,
            )),
            build_v2_record(_v1_result(
                ell=12, m=6, A_terms=[(1, 0)], B_terms=[(3, 0)],
                n=144, k=8, d=10, distance_trusted=True, stage="quick_estimate",
                fom=10.0,
            )),
        ]

        first = update_pareto_front_v2(records, filepath=filepath)
        second = update_pareto_front_v2(records, filepath=filepath)

        assert first["exact_front"] == second["exact_front"]
        assert first["lower_bound_front"] == second["lower_bound_front"]
        assert first["exploratory_witnesses"] == second["exploratory_witnesses"]
        assert first["quarantined"] == second["quarantined"] == []
        assert len(second["exact_front"]) == 1
        assert len(second["exploratory_witnesses"]) == 1


class TestDominatedRecordsResurfaceAfterDominatorUpgrade:
    """A record dominated in one update round must remain available (via
    ``all_records``) and resurface in a later round once the record that
    was dominating it is promoted out of that tier -- see pareto_v2's
    module docstring for why a fronts-only persistence would lose this
    record permanently.
    """

    def test_dominated_record_reappears_once_dominator_promoted_to_exact(self, tmp_path):
        filepath = tmp_path / "pareto_v2.json"

        record_a = build_v2_record(_v1_result(
            ell=6, m=6, A_terms=[(1, 0)], B_terms=[(2, 0)],
            n=72, k=12, d=0, stage="milp_promising_timeout", d_lower_bound=4,
        ))
        record_b = build_v2_record(_v1_result(
            ell=8, m=8, A_terms=[(5, 0)], B_terms=[(3, 0)],
            n=60, k=12, d=0, stage="milp_promising_timeout", d_lower_bound=5,
        ))

        def _norm(terms):
            return [tuple(t) for t in terms]

        def _is_a(r):
            return _norm(r["A_terms"]) == [(1, 0)] and _norm(r["B_terms"]) == [(2, 0)]

        def _is_b(r):
            return _norm(r["A_terms"]) == [(5, 0)] and _norm(r["B_terms"]) == [(3, 0)]

        # B strictly dominates A on every axis (higher rate, higher d_lower,
        # smaller n), so only B appears in round 1's lower_bound_front.
        round1 = update_pareto_front_v2([record_a, record_b], filepath=filepath)

        assert any(_is_b(r) for r in round1["lower_bound_front"])
        assert not any(_is_a(r) for r in round1["lower_bound_front"])
        # A must still be retained in the full record set even though it
        # lost the round-1 dominance comparison.
        assert any(_is_a(r) for r in round1["all_records"])

        # Round 2: B gets upgraded to a proven-exact result (same code_key,
        # d_lower==d_upper==5), promoting it into exact_front and removing
        # it from lower_bound_front's comparison pool entirely.
        record_b_upgraded = build_v2_record(_v1_result(
            ell=8, m=8, A_terms=[(5, 0)], B_terms=[(3, 0)],
            n=60, k=12, d=5, d_is_exact=True, stage="exact",
        ))
        round2 = update_pareto_front_v2([record_b_upgraded], filepath=filepath)

        assert any(_is_b(r) for r in round2["exact_front"])
        # A resurfaces into lower_bound_front now that its former dominator
        # has moved to a different tier.
        assert any(_is_a(r) for r in round2["lower_bound_front"])


# ── Additional coverage: build_v2_record derivation rules ─────────


class TestBuildV2RecordDerivationRules:
    def test_self_dual_d2(self):
        result = _v1_result(
            d=2, d_is_exact=True, distance_trusted=True, stage="self_dual_d2",
        )
        v2 = build_v2_record(result)
        assert v2["d_lower"] == 2
        assert v2["d_upper"] == 2
        assert v2["bound_sources"] == ["self_dual_proof"]
        assert v2["d_is_exact"] is True

    def test_exact_stage_from_evaluate_candidate(self):
        result = _v1_result(d=8, d_is_exact=True, distance_trusted=True, stage="exact")
        v2 = build_v2_record(result)
        assert v2["d_lower"] == v2["d_upper"] == 8
        assert v2["bound_sources"] == ["exact_milp"]
        assert v2["d_is_exact"] is True

    def test_milp_details_exact_flag(self):
        result = _v1_result(
            d=4, d_is_exact=True, distance_trusted=True, stage="milp_low_d",
            milp_details={"exact": True, "time_s": 1.2},
        )
        v2 = build_v2_record(result)
        assert v2["d_lower"] == v2["d_upper"] == 4
        assert v2["bound_sources"] == ["exact_milp"]
        assert v2["d_is_exact"] is True

    def test_milp_infeasible_early_stop(self):
        result = _v1_result(
            d=0, d_is_exact=False, distance_trusted=False,
            stage="milp_promising_timeout", d_lower_bound=5,
        )
        v2 = build_v2_record(result)
        assert v2["d_lower"] == 5
        assert v2["d_upper"] is None
        assert v2["bound_sources"] == ["milp_infeasible_early_stop"]
        assert v2["d_is_exact"] is False

    def test_d_le_2_trusted_is_conservative_not_exact(self):
        result = _v1_result(d=2, distance_trusted=True, stage="trivial_distance")
        v2 = build_v2_record(result)
        assert v2["d_lower"] == 0
        assert v2["d_upper"] == 2
        assert v2["bound_sources"] == ["bp_osd_heuristic"]
        assert v2["d_is_exact"] is False

    def test_symplectic_low_d_proof_is_exact_not_heuristic(self):
        # Contrast with test_d_le_2_trusted_is_conservative_not_exact above:
        # the evaluator's Gaussian-elimination proof (evaluator.py sets
        # d_is_exact=True together with stage="symplectic_low_d") must be
        # trusted as a genuine proof, not downgraded to the same conservative
        # bp_osd_heuristic bucket as an untrusted/heuristic d<=2 result.
        result = _v1_result(
            d=2, d_is_exact=True, distance_trusted=True, stage="symplectic_low_d",
        )
        v2 = build_v2_record(result)
        assert v2["d_lower"] == 2
        assert v2["d_upper"] == 2
        assert v2["bound_sources"] == ["symplectic_low_d_proof"]
        assert v2["d_is_exact"] is True

    def test_symplectic_low_d_without_exact_flag_falls_back_to_heuristic(self):
        # stage alone is not sufficient -- only when the evaluator also set
        # d_is_exact=True is this treated as a proof.
        result = _v1_result(
            d=2, d_is_exact=False, distance_trusted=True, stage="symplectic_low_d",
        )
        v2 = build_v2_record(result)
        assert v2["d_lower"] == 0
        assert v2["d_upper"] == 2
        assert v2["bound_sources"] == ["bp_osd_heuristic"]
        assert v2["d_is_exact"] is False

    def test_rejected_stage_has_no_bound_sources(self):
        for stage in ("rejected", "invalid", "construction_error", "k_zero", "k_low"):
            result = _v1_result(d=0, stage=stage)
            v2 = build_v2_record(result)
            assert v2["d_lower"] == 0
            assert v2["d_upper"] is None
            assert v2["bound_sources"] == []
            assert "certified_fom" not in v2
            assert "exploratory_fom" not in v2

    def test_certified_and_exploratory_fom_presence(self):
        exact_v2 = build_v2_record(
            _v1_result(d=6, d_is_exact=True, stage="exact", n=72, k=12)
        )
        assert exact_v2["certified_fom"] == pytest.approx(12 * 6 * 6 / 72)
        assert exact_v2["exploratory_fom"] == pytest.approx(12 * 6 * 6 / 72)

        heuristic_v2 = build_v2_record(
            _v1_result(d=10, distance_trusted=False, stage="quick_estimate", n=144, k=8)
        )
        assert "certified_fom" not in heuristic_v2
        assert heuristic_v2["exploratory_fom"] == pytest.approx(8 * 10 * 10 / 144)

        rejected_v2 = build_v2_record(_v1_result(d=0, stage="rejected", n=0, k=0))
        assert "certified_fom" not in rejected_v2
        assert "exploratory_fom" not in rejected_v2

    def test_extra_bound_sources_kwarg_appended(self):
        result = _v1_result(d=6, d_is_exact=True, stage="exact")
        v2 = build_v2_record(result, bound_sources=["webster_proven_exact"])
        assert v2["bound_sources"] == ["exact_milp", "webster_proven_exact"]

    def test_component_bounds_passthrough(self):
        cb = empty_component_bounds()
        result = _v1_result(d=6, d_is_exact=True, stage="exact")
        v2 = build_v2_record(result, component_bounds=cb)
        assert v2["component_bounds"] == cb

        v2_none = build_v2_record(_v1_result(d=6, d_is_exact=True, stage="exact"))
        assert v2_none["component_bounds"] is None


class TestBuildV2RecordNonCssWorkerVocabulary:
    """evolve/_noncss_distance_worker.py never sets stage/milp_details -- it
    sets a flat d_is_exact/d_is_upper_bound/milp_exact triple instead. These
    results must still raise d_lower per D2 (solver-proven optimality /
    completed exhaustive enumeration are lower-bound-raising events
    regardless of which module's field vocabulary reports them).
    """

    def test_hash_based_exact_result_raises_d_lower(self):
        result = _v1_result(
            d=3, d_is_exact=True, stage=None, d_method="exact_w3",
            d_is_upper_bound=False, trust_level="EXACT",
        )
        del result["distance_trusted"]
        v2 = build_v2_record(result)
        assert v2["d_lower"] == v2["d_upper"] == 3
        assert v2["bound_sources"] == ["exhaustive_hash"]
        assert v2["d_is_exact"] is True

    def test_milp_exact_result_raises_d_lower(self):
        result = _v1_result(
            d=7, d_is_exact=True, stage=None, d_method="milp_exact",
            milp_exact=True, d_is_upper_bound=False, trust_level="EXACT",
        )
        del result["distance_trusted"]
        v2 = build_v2_record(result)
        assert v2["d_lower"] == v2["d_upper"] == 7
        assert v2["bound_sources"] == ["exact_milp"]
        assert v2["d_is_exact"] is True

    def test_bposd_heuristic_result_stays_exploratory(self):
        result = _v1_result(
            d=9, d_is_exact=False, stage=None, d_method="bposd",
            milp_exact=False, d_is_upper_bound=True, trust_level="PARTIAL",
        )
        del result["distance_trusted"]
        v2 = build_v2_record(result)
        assert v2["d_lower"] == 0
        assert v2["d_upper"] == 9
        assert v2["bound_sources"] == ["bp_osd_heuristic"]
        assert v2["d_is_exact"] is False

    def test_css_cascade_results_unaffected_by_new_branch(self):
        # d_is_upper_bound is never set by the CSS cascade, so a CSS
        # d_is_exact=True result without stage=="exact" (e.g. a malformed
        # or unrelated stage) must NOT be caught by the new branch.
        result = _v1_result(d=5, d_is_exact=True, stage="quick_k_only")
        v2 = build_v2_record(result)
        assert v2["d_lower"] == 0
        assert v2["bound_sources"] == ["bp_osd_heuristic"]
        assert v2["d_is_exact"] is False


class TestIsExactInvariant:
    def test_equality_alone_is_not_sufficient(self):
        # Two independent heuristic estimates happen to coincide at d=8 --
        # this must NOT be treated as exact.
        assert is_exact(8, 8, ["bp_osd_heuristic"]) is False
        assert is_exact(8, 8, []) is False

    def test_allowlisted_source_with_equal_bounds_is_exact(self):
        assert is_exact(8, 8, ["exact_milp"]) is True
        assert is_exact(2, 2, ["self_dual_proof"]) is True

    def test_null_upper_is_never_exact(self):
        assert is_exact(5, None, ["exact_milp"]) is False


class TestFomHelpers:
    def test_certified_fom_zero_lower(self):
        assert certified_fom(12, 72, 0) is None

    def test_certified_fom_positive(self):
        assert certified_fom(12, 72, 6) == pytest.approx(12 * 36 / 72)

    def test_exploratory_fom_null_upper(self):
        assert exploratory_fom(12, 72, None) is None

    def test_exploratory_fom_positive(self):
        assert exploratory_fom(12, 72, 6) == pytest.approx(12 * 36 / 72)


class TestWebsterStatusEffect:
    @pytest.mark.parametrize("status,expected", [
        ("proven_exact", "raises_lower_after_verification"),
        ("incumbent", "upper_only"),
        ("heuristic", "upper_only"),
        ("timeout", "log_only"),
        ("failed", "log_only"),
    ])
    def test_known_statuses(self, status, expected):
        assert webster_status_effect(status) == expected

    def test_unknown_status_raises(self):
        with pytest.raises(ValueError):
            webster_status_effect("not_a_real_status")


class TestLoadParetoV2:
    def test_missing_file_returns_default_shape(self, tmp_path):
        filepath = tmp_path / "does_not_exist.json"
        archive = load_pareto_v2(filepath)
        assert archive["schema_version"] == 2
        assert archive["exact_front"] == []
        assert archive["lower_bound_front"] == []
        assert archive["exploratory_witnesses"] == []
        assert archive["generated_at"] is None

    def test_corrupted_file_returns_default_shape(self, tmp_path):
        filepath = tmp_path / "corrupt.json"
        filepath.write_text("{not valid json")
        archive = load_pareto_v2(filepath)
        assert archive["exact_front"] == []

    def test_roundtrip_via_update(self, tmp_path):
        filepath = tmp_path / "pareto_v2.json"
        v2 = build_v2_record(
            _v1_result(d=6, d_is_exact=True, stage="exact", n=72, k=12)
        )
        update_pareto_front_v2([v2], filepath=filepath)
        loaded = load_pareto_v2(filepath)
        assert len(loaded["exact_front"]) == 1
        assert loaded["exact_front"][0]["n"] == 72


class TestMigrateLegacyParetoToV2:
    def test_migrates_discovered_codes_file(self, tmp_path):
        legacy_codes_path = tmp_path / "discovered_codes.json"
        legacy_records = [
            _v1_result(
                ell=6, m=6, A_terms=[(1, 0)], B_terms=[(2, 0)],
                n=72, k=12, d=6, d_is_exact=True, stage="exact", fom=6.0,
            ),
            _v1_result(
                ell=12, m=6, A_terms=[(1, 0)], B_terms=[(3, 0)],
                n=144, k=8, d=10, distance_trusted=True,
                stage="quick_estimate", fom=10.0,
            ),
        ]
        with open(legacy_codes_path, "w") as f:
            json.dump(legacy_records, f)

        v2_filepath = tmp_path / "pareto_v2.json"
        result = pareto_v2.migrate_legacy_pareto_to_v2(
            legacy_codes_path=legacy_codes_path, v2_filepath=v2_filepath,
        )

        assert len(result["exact_front"]) == 1
        assert len(result["exploratory_witnesses"]) == 1
        assert result["exact_front"][0]["bound_sources"] == ["exact_milp", "legacy_migrated"]

        # Legacy file must remain untouched (read-only) -- compare against
        # a JSON round-trip of the original since tuples serialize to lists.
        with open(legacy_codes_path) as f:
            untouched = json.load(f)
        assert untouched == json.loads(json.dumps(legacy_records))

    def test_migration_is_idempotent(self, tmp_path):
        legacy_codes_path = tmp_path / "discovered_codes.json"
        legacy_records = [
            _v1_result(
                ell=6, m=6, A_terms=[(1, 0)], B_terms=[(2, 0)],
                n=72, k=12, d=6, d_is_exact=True, stage="exact", fom=6.0,
            ),
        ]
        with open(legacy_codes_path, "w") as f:
            json.dump(legacy_records, f)

        v2_filepath = tmp_path / "pareto_v2.json"
        first = pareto_v2.migrate_legacy_pareto_to_v2(
            legacy_codes_path=legacy_codes_path, v2_filepath=v2_filepath,
        )
        second = pareto_v2.migrate_legacy_pareto_to_v2(
            legacy_codes_path=legacy_codes_path, v2_filepath=v2_filepath,
        )
        assert first["exact_front"] == second["exact_front"]
        assert first["quarantined"] == second["quarantined"] == []
