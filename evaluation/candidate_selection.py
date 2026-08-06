"""Deterministic, hash-stratified candidate selection (Phase A5).

Replaces positional truncation (``candidates[:N]``) at two distinct points
in the evaluation pipeline:

1. **Pre-build candidate cap** -- reducing a raw, per-strategy candidate
   list down to ``QCODE_MAX_BUILD_CANDIDATES_CSS``/``_NONCSS`` entries
   before any code is constructed.
2. **Post-build distance selection** -- reducing a list of already-built
   ``(k>0)`` results down to the handful that receive expensive distance
   estimation, stratified by ``k`` band instead of sorted by raw ``k``.

Both cases share the same underlying mechanism: partition into strata,
guarantee every nonempty stratum a minimum quota, distribute the remaining
budget proportionally (largest-remainder method, ties broken by stratum
order for determinism), and select within a stratum by ascending
``SHA256(run_seed || lattice || stratum || canonical_candidate_key)``. This
selection is invariant to generator/result loop order and reproducible
given the same run seed and input population.
"""

from __future__ import annotations

import hashlib
from typing import Callable, Hashable, Sequence, TypeVar

T = TypeVar("T")

# Post-build k-bands, applied in order; a k not matching any band is excluded
# from distance selection entirely (falls back to the k-only aggregate count).
K_BANDS: list[tuple[str, Callable[[int], bool]]] = [
    ("k_2_3", lambda k: 2 <= k <= 3),
    ("k_4_5", lambda k: 4 <= k <= 5),
    ("k_ge_6", lambda k: k >= 6),
]


def k_band_for(k: int) -> str | None:
    """Return the A5 k-band name for ``k``, or ``None`` if ``k`` falls
    outside every band (e.g. k < 2)."""
    for name, pred in K_BANDS:
        if pred(k):
            return name
    return None


def canonical_candidate_key(terms: Sequence[Sequence[tuple]]) -> str:
    """Canonical, order-independent string key for a candidate.

    ``terms`` is a tuple/list of term-lists, e.g. ``(A_terms, B_terms)`` for
    CSS or ``(A_terms, B_terms, C_terms, D_terms)`` for PBB. Each term list
    is sorted internally so key equality does not depend on generator
    emission order.
    """

    def _norm(t):
        return tuple(sorted(tuple(term) for term in (t or [])))

    return repr(tuple(_norm(t) for t in terms))


def _stratum_hash(run_seed: object, ell: int, m: int, stratum: str, candidate_key: str) -> str:
    payload = f"{run_seed}|{ell}|{m}|{stratum}|{candidate_key}".encode()
    return hashlib.sha256(payload).hexdigest()


def _proportional_quotas(
    populations: list[int], min_quota: int, budget: int
) -> list[int]:
    """Split ``budget`` slots across strata sized by ``populations``.

    Every nonempty stratum (population > 0) first receives ``min_quota``
    slots (capped at its own population). Remaining slots are distributed
    proportionally to each stratum's remaining population using the
    largest-remainder method, so the total selected never exceeds
    ``min(budget, sum(populations))`` and rounding ties are broken by
    stratum order (deterministic, independent of population magnitude).
    """
    n = len(populations)
    quotas = [0] * n
    remaining_budget = budget

    # Guarantee minimum quota to every nonempty stratum, in stratum order,
    # so that if the budget is too small to cover every minimum, earlier
    # strata are favored deterministically rather than arbitrarily.
    for i, pop in enumerate(populations):
        if pop <= 0 or remaining_budget <= 0:
            continue
        grant = min(min_quota, pop, remaining_budget)
        quotas[i] = grant
        remaining_budget -= grant

    if remaining_budget <= 0:
        return quotas

    remaining_pop = [max(0, populations[i] - quotas[i]) for i in range(n)]
    total_remaining_pop = sum(remaining_pop)
    if total_remaining_pop <= 0:
        return quotas

    # Largest-remainder (Hamilton) method for proportional allocation.
    raw_shares = [
        remaining_budget * remaining_pop[i] / total_remaining_pop for i in range(n)
    ]
    base = [min(int(s), remaining_pop[i]) for i, s in enumerate(raw_shares)]
    for i in range(n):
        quotas[i] += base[i]
    leftover = remaining_budget - sum(base)

    fractions = sorted(
        range(n),
        key=lambda i: (raw_shares[i] - base[i], -i),
        reverse=True,
    )
    for i in fractions:
        if leftover <= 0:
            break
        if quotas[i] < populations[i]:
            quotas[i] += 1
            leftover -= 1

    return quotas


def stratified_select(
    items: Sequence[T],
    *,
    stratum_of: Callable[[T], str | None],
    key_of: Callable[[T], str],
    run_seed: object,
    ell: int,
    m: int,
    max_total: int,
    min_quota: int = 1,
) -> tuple[list[T], dict]:
    """Deterministically select up to ``max_total`` items from ``items``.

    ``stratum_of(item)`` assigns each item to a stratum name, or ``None`` to
    exclude it from selection entirely (it still counts in ``raw_count`` but
    not in any stratum population). ``key_of(item)`` returns the canonical
    candidate key used for hashing.

    Returns ``(selected, metrics)``. ``metrics`` contains ``raw_count``,
    ``excluded_count``, ``per_stratum`` (population/quota/selected per
    stratum, in first-seen order), ``selected_count``, and ``rejected_count``.
    Selection order and content are independent of the order of ``items``.
    """
    raw_count = len(items)
    strata: dict[str, list[T]] = {}
    excluded_count = 0

    for item in items:
        stratum = stratum_of(item)
        if stratum is None:
            excluded_count += 1
            continue
        strata.setdefault(stratum, []).append(item)

    stratum_names = list(strata.keys())
    populations = [len(strata[name]) for name in stratum_names]
    total_population = sum(populations)

    if max_total >= total_population:
        selected = [item for name in stratum_names for item in strata[name]]
        per_stratum = {
            name: {
                "population": len(strata[name]),
                "quota": len(strata[name]),
                "selected": len(strata[name]),
            }
            for name in stratum_names
        }
        return selected, {
            "raw_count": raw_count,
            "excluded_count": excluded_count,
            "per_stratum": per_stratum,
            "selected_count": len(selected),
            "rejected_count": 0,
        }

    quotas = _proportional_quotas(populations, min_quota, max_total)

    selected: list[T] = []
    per_stratum = {}
    for name, pop, quota in zip(stratum_names, populations, quotas):
        bucket = strata[name]
        ranked = sorted(
            bucket,
            key=lambda item: _stratum_hash(run_seed, ell, m, name, key_of(item)),
        )
        chosen = ranked[:quota]
        selected.extend(chosen)
        per_stratum[name] = {
            "population": pop,
            "quota": quota,
            "selected": len(chosen),
        }

    return selected, {
        "raw_count": raw_count,
        "excluded_count": excluded_count,
        "per_stratum": per_stratum,
        "selected_count": len(selected),
        "rejected_count": total_population - len(selected),
    }


def has_arity(cand: object, n: int) -> bool:
    """Return whether ``cand`` is a sized object of length ``n``, without
    raising on malformed input (e.g. a bare int or ``None`` instead of a
    tuple/list). Same defensive posture as :func:`dedup_candidates`'s
    ``key_of`` guard below, for the evaluators' own arity-check call sites
    (``len(cand) != N`` is not safe on its own -- a candidate lacking
    ``__len__`` raises ``TypeError`` there, which propagates out of the
    per-lattice loop and aborts the whole lattice instead of rejecting
    just that one malformed candidate).
    """
    try:
        return len(cand) == n
    except TypeError:
        return False


def dedup_candidates(
    candidates: list,
    *,
    key_of: Callable[[object], str] | None = None,
) -> tuple[list, int]:
    """Deduplicate a flat candidate list by canonical key, first-seen wins.

    Unlike :func:`select_prebuild_candidates`'s internal dedup (which only
    ever runs once the pre-build cap is actually triggered), this is meant
    to be called unconditionally on every raw ``generate_fn`` output --
    otherwise a generator that emits the same code twice (e.g. an evolved
    mutation whose strategies no longer dedup against each other the way
    the seed's own ``seen``-set strategies do) has its duplicate built and
    scored more than once whenever the raw count stays under the cap,
    inflating aggregate fitness metrics for no reason other than emission
    order.

    Returns ``(deduped, num_duplicates_removed)``.
    """
    if key_of is None:
        key_of = canonical_candidate_key

    seen_keys: set[str] = set()
    deduped = []
    for cand in candidates:
        try:
            key = key_of(cand)
        except (TypeError, ValueError):
            # Malformed shape -- pass through unchanged; downstream
            # validation (arity/weight checks) rejects it on its own
            # merits, so silently dropping it here would just hide a
            # different bug behind this one.
            deduped.append(cand)
            continue
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(cand)
    return deduped, len(candidates) - len(deduped)


def select_prebuild_candidates(
    candidates_by_strategy: dict[str, list],
    *,
    run_seed: object,
    ell: int,
    m: int,
    max_total: int,
    min_quota_per_strategy: int = 1,
    key_of: Callable[[object], str] | None = None,
) -> tuple[list, dict]:
    """Pre-build candidate cap (A5), stratified by generator strategy.

    ``candidates_by_strategy`` maps a strategy name to the raw candidate
    tuples it produced, in the order the generator emitted them (order does
    not affect the result). Candidates are canonicalized and deduplicated
    globally (first-seen strategy wins, iterating strategies in the order
    given) before stratification, so a candidate reachable from more than
    one strategy is only counted once.
    """
    if key_of is None:
        key_of = canonical_candidate_key

    seen_keys: set[str] = set()
    deduped_by_strategy: dict[str, list] = {}
    raw_count = 0
    for strategy, candidates in candidates_by_strategy.items():
        raw_count += len(candidates)
        bucket = deduped_by_strategy.setdefault(strategy, [])
        for cand in candidates:
            key = key_of(cand)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            bucket.append(cand)

    tagged = [
        (strategy, cand)
        for strategy, bucket in deduped_by_strategy.items()
        for cand in bucket
    ]

    selected_tagged, metrics = stratified_select(
        tagged,
        stratum_of=lambda item: item[0],
        key_of=lambda item: key_of(item[1]),
        run_seed=run_seed,
        ell=ell,
        m=m,
        max_total=max_total,
        min_quota=min_quota_per_strategy,
    )
    metrics["raw_count"] = raw_count
    metrics["deduped_count"] = len(tagged)
    return [cand for _, cand in selected_tagged], metrics


def select_postbuild_distance_candidates(
    results: Sequence[dict],
    *,
    run_seed: object,
    ell: int,
    m: int,
    max_total: int,
    min_quota_per_band: int = 1,
    key_of: Callable[[dict], str] | None = None,
) -> tuple[list[dict], dict]:
    """Post-build distance selection (A5), stratified by k-band.

    ``results`` are already-built candidate result dicts with a ``k`` key
    (and, conventionally, ``A_terms``/``B_terms``[/``C_terms``/``D_terms``]).
    Results whose ``k`` falls outside every :data:`K_BANDS` band (k < 2) are
    excluded from distance selection.
    """
    if key_of is None:
        def key_of(r):
            return canonical_candidate_key(
                (
                    r.get("A_terms", []),
                    r.get("B_terms", []),
                    r.get("C_terms", []),
                    r.get("D_terms", []),
                )
            )

    return stratified_select(
        list(results),
        stratum_of=lambda r: k_band_for(r.get("k", 0)),
        key_of=key_of,
        run_seed=run_seed,
        ell=ell,
        m=m,
        max_total=max_total,
        min_quota=min_quota_per_band,
    )
