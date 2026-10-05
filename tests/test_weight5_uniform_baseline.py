"""Focused tests for the matched-budget weight-five uniform control."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from scripts import run_weight5_uniform_baseline as baseline


def test_six_identity_anchored_universes_match_referee_counts():
    expected = {
        (6, 10): 100_949,
        (9, 9): 252_800,
        (10, 9): 348_524,
        (16, 12): 3_465_695,
        (15, 14): 4_542_824,
        (18, 12): 4_946_075,
    }
    assert {
        lattice: baseline.universe_size(*lattice) for lattice in expected
    } == expected
    assert sum(expected.values()) == 13_656_867


def test_unrank_normalized_pair_is_bijective_on_tiny_lattice():
    ell, m = 2, 2
    population = baseline.universe_size(ell, m)
    assert population == (ell * m - 1) * math.comb(ell * m - 1, 2) == 9

    pairs = []
    for index in range(population):
        A, B = baseline.unrank_normalized_pair(ell, m, index)
        assert A[0] == (0, 0)
        assert B[0] == (0, 0)
        assert len(A) == len(set(A)) == 2
        assert len(B) == len(set(B)) == 3
        assert all(0 <= x < ell and 0 <= y < m for x, y in A + B)
        pairs.append((tuple(A), tuple(B)))

    assert len(set(pairs)) == population
    with pytest.raises(IndexError):
        baseline.unrank_normalized_pair(ell, m, population)


def test_complete_css_builder_handles_valid_one_variable_supports():
    ell, m = 10, 9
    A = [(0, 0), (0, 6)]
    B = [(0, 0), (0, 1), (0, 5)]
    code = baseline._build_complete_css_code(ell, m, A, B)
    n, k = baseline._compute_k(ell, m, A, B)
    assert (n, k) == (180, 40)
    matrix_x = np.asarray(code.matrix_x, dtype=np.uint8)
    matrix_z = np.asarray(code.matrix_z, dtype=np.uint8)
    assert matrix_x.shape == matrix_z.shape == (90, 180)
    assert not np.any((matrix_x @ matrix_z.T) % 2)


def test_lazy_fisher_yates_is_seeded_unique_and_resumable_by_replay():
    first = list(baseline.LazyFisherYates(100, 1234))
    second = list(baseline.LazyFisherYates(100, 1234))
    other = list(baseline.LazyFisherYates(100, 1235))
    assert first == second
    assert first != other
    assert sorted(first) == list(range(100))

    resumed = baseline.LazyFisherYates(100, 1234)
    resumed.advance(37)
    assert list(resumed) == first[37:]


def test_translation_key_ignores_independent_shifts_term_order_and_sector_swap():
    ell, m = 6, 10
    A = [(0, 0), (2, 7)]
    B = [(0, 0), (1, 4), (5, 8)]

    def shift(terms, dx, dy):
        return [((x + dx) % ell, (y + dy) % m) for x, y in terms]

    expected = baseline.translation_candidate_key(ell, m, A, B)
    shifted = baseline.translation_candidate_key(
        ell,
        m,
        list(reversed(shift(A, 3, 2))),
        list(reversed(shift(B, 4, 9))),
    )
    swapped = baseline.translation_candidate_key(ell, m, B, A)
    assert shifted == expected
    assert swapped == expected


def _tiny_config() -> baseline.RunConfig:
    return baseline.RunConfig(
        run_seed=17,
        ell=6,
        m=10,
        quotas={"k_4_5": 2, "k_ge_6": 1},
        pilot=True,
        quick_trials=1,
        refine_trials=1,
        skip_exact=True,
        exact_timeout=1,
        milp_timeout_per_logical=1,
        milp_total_timeout=1,
    )


def test_selection_records_every_draw_and_resumes_without_duplicates(tmp_path):
    config = _tiny_config()

    def fake_k(_ell, m, A, _B):
        group_index = A[1][0] * m + A[1][1]
        return 120, 4 if group_index % 2 else 8

    first = baseline.run_selection(
        tmp_path, config, max_new_draws=1, k_function=fake_k, flush_every=1
    )
    assert first["status"] == "in_progress"
    second = baseline.run_selection(
        tmp_path, config, k_function=fake_k, flush_every=1
    )
    assert second["status"] == "complete"

    path = tmp_path / "seed_17" / "6x10" / "draws.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["draw_number_zero_based"] for row in rows] == list(range(len(rows)))
    assert len({row["universe_index"] for row in rows}) == len(rows)
    assert sum(row["selected"] for row in rows) == 3
    assert {
        band: sum(row["selected"] and row["band"] == band for row in rows)
        for band in config.quotas
    } == config.quotas

    third = baseline.run_selection(
        tmp_path, config, max_new_draws=50, k_function=fake_k
    )
    assert third["new_draws_this_invocation"] == 0
    assert path.read_text().splitlines() == [
        json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows
    ]


def test_config_hash_rejects_protocol_drift(tmp_path):
    config = _tiny_config()
    baseline.run_selection(
        tmp_path,
        config,
        max_new_draws=0,
        k_function=lambda *_: (120, 4),
    )
    changed = baseline.RunConfig(
        **{**config.__dict__, "quick_trials": config.quick_trials + 1}
    )
    with pytest.raises(ValueError, match="configuration mismatch"):
        baseline.run_selection(
            tmp_path,
            changed,
            max_new_draws=0,
            k_function=lambda *_: (120, 4),
        )


def test_checked_historical_budgets_equal_log_counts():
    assert baseline.historical_budget_from_logs() == baseline.HISTORICAL_DISTANCE_QUOTAS
    assert sum(
        sum(quota.values()) for quota in baseline.HISTORICAL_DISTANCE_QUOTAS.values()
    ) == 6_444


def test_historical_evaluator_and_runtime_sources_are_pinned():
    evidence = baseline.verify_historical_sources()
    assert evidence["evaluator"]["sha256"] == baseline.HISTORICAL_EVALUATOR_SHA256
    assert evidence["evaluator"]["git_object_verified"]
    assert set(evidence["runtime_components"]) == set(
        baseline.HISTORICAL_RUNTIME_SHA256
    )
    assert all(
        item["local_verified"] and item["git_object_verified"]
        for item in evidence["runtime_components"].values()
    )
    # evaluation/bb_code.py (commit 1b2ebca added exponent-type validation) and
    # evaluation/evaluator.py (the MILP all-timeout branch stopped reporting an
    # uncertified lower bound) no longer match their historical pins
    # byte-for-byte; they are accepted only via the documented, reviewed
    # inert-revision fallback, not a silent re-pin.
    revised = ("evaluation/bb_code.py", "evaluation/evaluator.py")
    for path in revised:
        item = evidence["runtime_components"][path]
        assert item["local_verified_via"] == "reviewed_inert_revision"
        assert (
            item["local_sha256"]
            == baseline.RUNTIME_VERIFIED_INERT_REVISIONS[path]["sha256"]
        )
    other_components = {
        path: item
        for path, item in evidence["runtime_components"].items()
        if path not in revised
    }
    assert all(
        item["local_verified_via"] == "historical_pin"
        for item in other_components.values()
    )


def test_historical_side_by_side_uses_explicit_call_and_unique_denominators():
    result = baseline.historical_llm_comparison()
    assert result["distance_observations"] == 6_444
    assert result["unique_positive_normalized_tuples"] == 936
    assert result["retained_d_gt_2_unique_tuples"] == 912
    assert result["retained_tanner_classes"] == 535
    assert result["connected_retained_tanner_classes"] == 329
    assert result["certified_d_ge_5_observations"] == 5_295
    assert result["certified_d_ge_5_unique_tuples"] == 816
    assert result["connected_certified_d_ge_5_tanner_classes"] == 311
    assert result["best_exact"]["fom"] == pytest.approx(4.355555555555555)
    assert result["best_certified_lower_bound"] == result["best_exact"]
    assert result["mapping_audit"] == {
        "positive_observations_mapped_to_retained_catalogue": 6_093,
        "positive_observations_not_retained_after_d_le_2_filter": 351,
        "unique_positive_tuples_not_retained_after_d_le_2_filter": 24,
    }


def test_pilot_seed_comparison_exposes_replicate_endpoints_and_validation():
    root = baseline.ROOT / "results" / "weight5_uniform_baseline_pilot"
    result = baseline.uniform_sampler_comparison(root, run_seed=42)

    assert result["run_seed"] == 42
    assert result["lattice_seed_runs"] == 6
    assert result["planned_distance_observations"] == 24
    assert result["completed_distance_observations"] == 24
    assert result["rank_only_draws"] == 267
    assert result["unique_normalized_tuples"] == 24
    assert result["retained_d_gt_2_unique_tuples"] == 19
    assert result["retained_tanner_classes"] == 19
    assert result["certified_d_ge_5_observations"] == 17
    assert result["connected_certified_d_ge_5_tanner_classes"] == 13
    assert result["best_certified_lower_bound"] is not None
    assert result["best_exact"] is not None
    assert all(
        result["stage_completeness"][stage]
        for stage in ("selection", "distance", "certification")
    )
    assert result["primary_complete"]
    assert not result["stage_completeness"]["milp"]
    assert not result["complete"]
    assert all(count == 0 for count in result["errors"].values())
    assert set(result["overlap"]) == {
        "selected_literal_in_historical_positive_log",
        "selected_translation_class_in_historical_positive_log",
        "selected_literal_in_retained_catalogue",
        "selected_translation_class_in_retained_catalogue",
        "certified_d_ge_5_tanner_classes_in_retained_catalogue",
        "connected_certified_d_ge_5_tanner_classes_in_retained_catalogue",
    }
    assert result["catalogue_class_exact_evidence"]["tanner_classes"] == 1
    assert (
        result["catalogue_class_exact_evidence"]["best_exact"]["evidence_source"]
        == "retained_catalogue_tanner_class_transfer"
    )
    assert (
        result["unresolved_after_catalogue_class_transfer"]
        ["certified_d_ge_5_tanner_classes"]
        == 16
    )


def test_replicate_statistics_do_not_hide_missing_exact_endpoint():
    complete = baseline._replicate_statistic([1, 2, 3])
    assert complete == {
        "values": [1, 2, 3],
        "all_values_present": True,
        "mean": 2.0,
        "sample_standard_deviation": 1.0,
        "min": 1.0,
        "max": 3.0,
    }
    missing = baseline._replicate_statistic([1.0, None, 3.0])
    assert missing["values"] == [1.0, None, 3.0]
    assert not missing["all_values_present"]
    assert (
        missing["mean"]
        is missing["sample_standard_deviation"]
        is missing["min"]
        is missing["max"]
        is None
    )


def test_skipped_exact_branch_is_not_reported_as_a_timeout():
    row = {"stage": "exact_timeout", "exact_search_skipped": True}
    assert baseline._effective_distance_stage(row) == "exact_skipped"
    assert (
        baseline._effective_distance_stage(
            {"stage": "exact_timeout", "exact_search_skipped": False}
        )
        == "exact_timeout"
    )


def test_aggregate_summary_contains_per_seed_statistics_and_validation(monkeypatch):
    root = baseline.ROOT / "results" / "weight5_uniform_baseline_pilot"
    monkeypatch.setattr(baseline, "_atomic_write_json", lambda *_args: None)
    result = baseline.aggregate_summaries(root)

    seed = result["per_seed"]["42"]
    assert seed["completed_distance_observations"] == 24
    assert seed["certified_d_ge_5_observations"] == 17
    assert result["replicate_statistics"]["certified_d_ge_5_observations"] == {
        "values": [17],
        "all_values_present": True,
        "mean": 17.0,
        "sample_standard_deviation": None,
        "min": 17.0,
        "max": 17.0,
    }
    assert not result["validation"]["all_seed_replicates_have_6444_planned_calls"]
    assert not result["validation"]["all_stage_records_complete"]
    assert result["validation"]["all_primary_stage_records_complete"]
    assert not result["validation"]["expected_three_seed_six_lattice_design_present"]
    assert not result["validation"]["exactly_three_runs_per_lattice"]
    assert not result["validation"]["ready_for_primary_comparison"]
    assert result["validation"]["all_error_counts_zero"]
    integrity = result["validation"]["record_integrity"]
    assert integrity["all_record_relationship_checks_pass"]
    assert integrity["stage_set_equality"]["distance_equals_selected"]
    assert integrity["stage_set_equality"]["certification_equals_targets"]
    assert not integrity["stage_set_equality"]["milp_equals_finalists"]
    assert all(integrity["primary_stage_set_equality"].values())
    assert not integrity["all_configurations_match_prespecified_protocol"]


def test_full_control_rank_repair_ledger_reconciles_with_active_records():
    root = baseline.ROOT / "results" / "weight5_uniform_baseline"
    audit = baseline._validate_one_variable_rank_repair(root)
    assert audit["status"] == "verified"
    assert audit["streams"] == 18
    assert audit["repaired_rank_rows"] == 631
    assert audit["selected_indices_added"] == audit["selected_indices_removed"] == 16
    assert audit["active_draw_errors"] == 0


def test_full_design_requires_all_three_seeds_even_if_other_state_is_complete():
    summaries = [
        {"run_seed": 42, "lattice": [ell, m]}
        for ell, m in baseline.PROFILE_LATTICES["all"]
    ]
    per_lattice = {
        f"{ell}x{m}": {"runs": 1, "seeds": [42]}
        for ell, m in baseline.PROFILE_LATTICES["all"]
    }
    complete, three_runs = baseline._has_expected_full_design(
        summaries,
        summary_path_count=6,
        config_path_count=6,
        per_lattice=per_lattice,
        integrity_configurations={
            (42, ell, m) for ell, m in baseline.PROFILE_LATTICES["all"]
        },
    )
    assert not complete
    assert not three_runs


def test_checked_pilot_manifest_recomputes_byte_for_byte():
    root = baseline.ROOT / "results" / "weight5_uniform_baseline_pilot"
    checked = json.loads((root / "artifact_manifest.json").read_text())
    assert checked == baseline._uniform_artifact_manifest(root)


def test_checked_pilot_artifact_is_complete_and_explicitly_not_matched_budget():
    root = baseline.ROOT / "results" / "weight5_uniform_baseline_pilot"
    aggregate = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert aggregate["lattice_seed_runs"] == 6

    total_draws = 0
    total_selected = 0
    total_estimated_ge5 = 0
    total_certified_ge5 = 0
    total_connected_classes = 0
    for lattice_dir in sorted((root / "seed_42").iterdir()):
        if not lattice_dir.is_dir():
            continue
        config = json.loads((lattice_dir / "config.json").read_text())
        summary = json.loads((lattice_dir / "summary.json").read_text())
        draws = [json.loads(line) for line in (lattice_dir / "draws.jsonl").read_text().splitlines()]
        assert config["config"]["pilot"]
        assert not config["config"]["matched_budget"]
        assert all(summary["completion"][stage] for stage in ("selection", "distance", "certification"))
        assert len({row["universe_index"] for row in draws}) == len(draws)
        total_draws += len(draws)
        total_selected += sum(row["selected"] for row in draws)
        total_estimated_ge5 += summary["yield"]["estimated_d_ge_5"]
        total_certified_ge5 += summary["yield"]["certified_d_ge_5"]
        total_connected_classes += summary["yield"]["connected_certified_d_ge_5_classes"]

    assert (total_draws, total_selected) == (267, 24)
    assert (total_estimated_ge5, total_certified_ge5, total_connected_classes) == (
        19,
        17,
        13,
    )


def test_checked_full_control_summary_and_manifest_recompute_byte_for_byte():
    root = baseline.ROOT / "results" / "weight5_uniform_baseline"
    aggregate = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert aggregate["validation"]["ready_for_primary_comparison"]
    assert not aggregate["validation"]["optional_milp_followup_complete"]
    pooled = aggregate["side_by_side"]["uniform_sampler_pooled_coverage"]
    assert pooled["rank_only_draws"] == 287_489
    assert pooled["completed_distance_observations"] == 19_332
    assert pooled["certified_d_ge_5_observations"] == 14_215
    assert pooled["connected_certified_d_ge_5_tanner_classes"] == 985
    assert pooled["optional_exact_followup"]["finalist_tanner_classes"] == 385
    assert (
        pooled["optional_exact_followup"]
        ["classes_with_transferred_exact_catalogue_evidence"]
        == 26
    )
    assert pooled["best_exact"]["parameters"] == [180, 4, 14]
    checked = json.loads((root / "artifact_manifest.json").read_text())
    assert checked == baseline._uniform_artifact_manifest(root)
