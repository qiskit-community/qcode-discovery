"""Tests for evaluation/candidate_selection.py (Phase A5).

Covers the plan's "Selection and Stage 0" test bullets that are properly
scoped to this module: hash-stratified (not prefix) sampling above the
5000/3000 caps, generator-loop-order invariance, identical seed+input
reproducibility, every nonempty k-band getting its quota, and the
k-band/dedup logic in select_prebuild_candidates/
select_postbuild_distance_candidates. (Stage-0 census/checkpoint/Wilson-
interval bullets belong to scripts/run_weight5_stage0.py, not this module.)
"""

from __future__ import annotations

import random

from evaluation.candidate_selection import (
    K_BANDS,
    _proportional_quotas,
    canonical_candidate_key,
    dedup_candidates,
    has_arity,
    k_band_for,
    select_postbuild_distance_candidates,
    select_prebuild_candidates,
    stratified_select,
)


class TestKBandFor:
    def test_k_2_3_band(self):
        assert k_band_for(2) == "k_2_3"
        assert k_band_for(3) == "k_2_3"

    def test_k_4_5_band(self):
        assert k_band_for(4) == "k_4_5"
        assert k_band_for(5) == "k_4_5"

    def test_k_ge_6_band(self):
        assert k_band_for(6) == "k_ge_6"
        assert k_band_for(1000) == "k_ge_6"

    def test_k_below_2_excluded(self):
        assert k_band_for(1) is None
        assert k_band_for(0) is None

    def test_every_band_name_unique(self):
        names = [name for name, _ in K_BANDS]
        assert len(names) == len(set(names))


class TestCanonicalCandidateKey:
    def test_order_independent_within_term_list(self):
        cand_a = ([(0, 0), (1, 0)], [(0, 1)])
        cand_b = ([(1, 0), (0, 0)], [(0, 1)])
        assert canonical_candidate_key(cand_a) == canonical_candidate_key(cand_b)

    def test_different_content_gives_different_key(self):
        cand_a = ([(0, 0), (1, 0)], [(0, 1)])
        cand_b = ([(0, 0), (2, 0)], [(0, 1)])
        assert canonical_candidate_key(cand_a) != canonical_candidate_key(cand_b)

    def test_handles_none_term_lists_like_empty(self):
        cand_a = ([(0, 0)], None, None, None)
        cand_b = ([(0, 0)], [], [], [])
        assert canonical_candidate_key(cand_a) == canonical_candidate_key(cand_b)

    def test_4_tuple_pbb_key_distinguishes_c_d(self):
        base = ([(0, 0)], [(0, 1)], [], [])
        with_c = ([(0, 0)], [(0, 1)], [(0, 0)], [])
        assert canonical_candidate_key(base) != canonical_candidate_key(with_c)


class TestHasArity:
    """Regression coverage for the codex round-2 finding that both weight-5
    evaluators' own ``len(cand) != N`` arity checks (in
    ``_filter_weight5``/the PBB build loop) raised ``TypeError`` and
    aborted the whole lattice whenever a malformed candidate (e.g. a bare
    int, matching ``TestDedupCandidates``'s ``malformed = 42`` convention)
    reached them -- instead of just rejecting that one candidate, the way
    ``dedup_candidates``'s own ``key_of`` guard already did."""

    def test_correct_arity_returns_true(self):
        assert has_arity(([(0, 0)], [(0, 1)]), 2) is True
        assert has_arity(([(0, 0)], [(0, 1)], [], []), 4) is True

    def test_wrong_arity_returns_false(self):
        assert has_arity(([(0, 0)], [(0, 1)]), 4) is False
        assert has_arity(([(0, 0)],), 2) is False

    def test_non_sized_object_returns_false_without_raising(self):
        malformed = 42  # not iterable/sized -- len() raises TypeError
        assert has_arity(malformed, 2) is False
        assert has_arity(None, 2) is False


class TestDedupCandidates:
    """Regression coverage for the codex Commit-3 finding that
    ``_run_evaluation`` in both weight-5 evaluators only deduplicated
    candidates as a side effect of ``select_prebuild_candidates``, which
    never runs below the pre-build cap -- so a generator that emits the
    same code twice (e.g. an evolved mutation whose strategies no longer
    dedup against each other) would have it built/scored twice whenever
    the raw count stayed under the cap. ``dedup_candidates`` must be called
    unconditionally, independent of any cap."""

    def test_exact_duplicate_removed(self):
        cand = ([(0, 0), (1, 0)], [(0, 1)])
        deduped, num_removed = dedup_candidates([cand, cand, cand])
        assert len(deduped) == 1
        assert num_removed == 2

    def test_reordered_term_lists_are_still_duplicates(self):
        cand_a = ([(0, 0), (1, 0)], [(0, 1)])
        cand_b = ([(1, 0), (0, 0)], [(0, 1)])
        deduped, num_removed = dedup_candidates([cand_a, cand_b])
        assert len(deduped) == 1
        assert num_removed == 1

    def test_distinct_candidates_all_kept_first_seen_order(self):
        cand_a = ([(0, 0), (1, 0)], [(0, 1)])
        cand_b = ([(0, 0), (2, 0)], [(0, 1)])
        deduped, num_removed = dedup_candidates([cand_a, cand_b])
        assert deduped == [cand_a, cand_b]
        assert num_removed == 0

    def test_pbb_4_tuples_deduped_by_all_four_terms(self):
        base = ([(0, 0)], [(0, 1)], [], [])
        with_c = ([(0, 0)], [(0, 1)], [(0, 0)], [])
        deduped, num_removed = dedup_candidates([base, with_c, base])
        assert len(deduped) == 2
        assert num_removed == 1

    def test_malformed_candidate_passes_through_without_crashing(self):
        malformed = 42  # not iterable -- canonical_candidate_key raises TypeError
        cand = ([(0, 0), (1, 0)], [(0, 1)])
        deduped, num_removed = dedup_candidates([malformed, cand])
        assert malformed in deduped
        assert cand in deduped
        assert num_removed == 0

    def test_empty_list_is_a_no_op(self):
        deduped, num_removed = dedup_candidates([])
        assert deduped == []
        assert num_removed == 0


class TestProportionalQuotas:
    def test_sum_never_exceeds_budget(self):
        populations = [10, 20, 0, 5]
        quotas = _proportional_quotas(populations, min_quota=1, budget=8)
        assert sum(quotas) <= 8

    def test_no_stratum_exceeds_its_own_population(self):
        populations = [2, 1, 7]
        quotas = _proportional_quotas(populations, min_quota=1, budget=100)
        for q, p in zip(quotas, populations):
            assert q <= p

    def test_empty_stratum_gets_no_quota(self):
        populations = [0, 10]
        quotas = _proportional_quotas(populations, min_quota=1, budget=5)
        assert quotas[0] == 0

    def test_every_nonempty_stratum_gets_at_least_min_quota_when_budget_allows(self):
        populations = [5, 5, 5]
        quotas = _proportional_quotas(populations, min_quota=1, budget=3)
        assert all(q >= 1 for q in quotas)

    def test_deterministic_across_repeated_calls(self):
        populations = [7, 13, 3, 29]
        q1 = _proportional_quotas(populations, min_quota=2, budget=17)
        q2 = _proportional_quotas(populations, min_quota=2, budget=17)
        assert q1 == q2


class TestStratifiedSelect:
    def _items(self, n_a, n_b):
        items = [("a", i) for i in range(n_a)] + [("b", i) for i in range(n_b)]
        return items

    def test_all_items_selected_when_budget_exceeds_population(self):
        items = self._items(3, 4)
        selected, metrics = stratified_select(
            items,
            stratum_of=lambda it: it[0],
            key_of=lambda it: repr(it),
            run_seed="seed1", ell=6, m=6, max_total=100,
        )
        assert len(selected) == 7
        assert metrics["rejected_count"] == 0

    def test_selection_not_a_naive_prefix_of_input_order(self):
        # 200 items, cap to 10 -- if selection were `items[:10]` it would be
        # exactly the first 10 "a" items (since "a" is listed first).
        # Hash-stratified selection must not reduce to that.
        items = self._items(100, 100)
        selected, _ = stratified_select(
            items,
            stratum_of=lambda it: it[0],
            key_of=lambda it: repr(it),
            run_seed="seed1", ell=6, m=6, max_total=10, min_quota=1,
        )
        naive_prefix = items[:10]
        assert selected != naive_prefix

    def test_generator_loop_order_does_not_affect_selection(self):
        items = self._items(50, 50)
        shuffled = list(items)
        random.Random(42).shuffle(shuffled)

        selected_1, _ = stratified_select(
            items,
            stratum_of=lambda it: it[0], key_of=lambda it: repr(it),
            run_seed="seedX", ell=6, m=6, max_total=15,
        )
        selected_2, _ = stratified_select(
            shuffled,
            stratum_of=lambda it: it[0], key_of=lambda it: repr(it),
            run_seed="seedX", ell=6, m=6, max_total=15,
        )
        assert set(selected_1) == set(selected_2)

    def test_identical_seed_and_input_reproduces_identical_sample(self):
        items = self._items(40, 60)
        s1, m1 = stratified_select(
            items, stratum_of=lambda it: it[0], key_of=lambda it: repr(it),
            run_seed="fixed-seed", ell=9, m=8, max_total=12,
        )
        s2, m2 = stratified_select(
            items, stratum_of=lambda it: it[0], key_of=lambda it: repr(it),
            run_seed="fixed-seed", ell=9, m=8, max_total=12,
        )
        assert s1 == s2
        assert m1 == m2

    def test_different_seed_can_change_sample(self):
        items = self._items(40, 60)
        s1, _ = stratified_select(
            items, stratum_of=lambda it: it[0], key_of=lambda it: repr(it),
            run_seed="seed-a", ell=9, m=8, max_total=12,
        )
        s2, _ = stratified_select(
            items, stratum_of=lambda it: it[0], key_of=lambda it: repr(it),
            run_seed="seed-b", ell=9, m=8, max_total=12,
        )
        assert s1 != s2

    def test_every_nonempty_stratum_gets_quota_in_output(self):
        items = self._items(30, 30)
        selected, metrics = stratified_select(
            items, stratum_of=lambda it: it[0], key_of=lambda it: repr(it),
            run_seed="seedY", ell=6, m=6, max_total=6, min_quota=1,
        )
        assert metrics["per_stratum"]["a"]["selected"] >= 1
        assert metrics["per_stratum"]["b"]["selected"] >= 1

    def test_items_returning_none_stratum_are_excluded(self):
        items = [("a", 0), ("a", 1), ("skip", 0)]
        selected, metrics = stratified_select(
            items,
            stratum_of=lambda it: None if it[0] == "skip" else it[0],
            key_of=lambda it: repr(it),
            run_seed="s", ell=6, m=6, max_total=100,
        )
        assert ("skip", 0) not in selected
        assert metrics["excluded_count"] == 1


class TestSelectPrebuildCandidates:
    def test_dedup_across_strategies_first_seen_wins(self):
        cand = ([(0, 0)], [(0, 1)])
        by_strategy = {
            "strategy_1": [cand],
            "strategy_2": [cand],
        }
        selected, metrics = select_prebuild_candidates(
            by_strategy, run_seed="s", ell=6, m=6, max_total=100,
        )
        assert len(selected) == 1
        assert metrics["raw_count"] == 2
        assert metrics["deduped_count"] == 1

    def test_min_quota_per_strategy_respected_under_cap(self):
        by_strategy = {
            "strategy_1": [([(0, i)], [(0, 1)]) for i in range(5)],
            "strategy_2": [([(1, i)], [(1, 1)]) for i in range(5)],
        }
        selected, metrics = select_prebuild_candidates(
            by_strategy, run_seed="s", ell=9, m=9, max_total=4,
            min_quota_per_strategy=2,
        )
        assert metrics["per_stratum"]["strategy_1"]["selected"] >= 2
        assert metrics["per_stratum"]["strategy_2"]["selected"] >= 2


class TestSelectPostbuildDistanceCandidates:
    def test_k_below_2_excluded_from_selection(self):
        results = [
            {"k": 1, "A_terms": [], "B_terms": []},
            {"k": 3, "A_terms": [(0, 0)], "B_terms": []},
            {"k": 6, "A_terms": [(1, 0)], "B_terms": []},
        ]
        selected, metrics = select_postbuild_distance_candidates(
            results, run_seed="s", ell=6, m=6, max_total=100,
        )
        assert all(r["k"] != 1 for r in selected)
        assert metrics["excluded_count"] == 1

    def test_multiple_k_bands_each_get_quota_under_cap(self):
        results = (
            [{"k": 2, "A_terms": [(0, i)], "B_terms": []} for i in range(5)]
            + [{"k": 4, "A_terms": [(1, i)], "B_terms": []} for i in range(5)]
            + [{"k": 8, "A_terms": [(2, i)], "B_terms": []} for i in range(5)]
        )
        selected, metrics = select_postbuild_distance_candidates(
            results, run_seed="s", ell=9, m=8, max_total=3, min_quota_per_band=1,
        )
        bands_selected = {k_band_for(r["k"]) for r in selected}
        assert bands_selected == {"k_2_3", "k_4_5", "k_ge_6"}
