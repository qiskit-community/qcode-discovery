"""Multi-stage evaluation cascade for bivariate bicycle code candidates.

This is the central evaluation module.  Every candidate polynomial pair
is assessed through a five-stage cascade that progressively invests more
compute in more promising candidates:

1. **Validate** (microseconds) -- Checks polynomial structure: 2-6
   distinct terms with exponents in range for the target lattice.
   See :func:`evaluation.bb_code.validate_terms`.

2. **Build & compute k** (milliseconds) -- Constructs a
   :class:`qldpc.codes.BBCode` and reads ``(n, k)`` via GF(2) rank.
   Rejects codes with ``k = 0`` (stage ``k_zero``) or ``k < 4``
   (stage ``k_low``, score ``-1000 + k``).

3. **Quick distance estimate** (seconds) -- Runs
   :func:`evaluation.distance.estimate_distance` with ``quick_trials``
   BP-OSD trials (default 100).  Codes with ``d <= 2`` are tagged
   ``trivial_distance`` and exit early.  The result carries a
   ``distance_trusted`` flag based on ``d / sqrt(n)`` vs
   :data:`DISTANCE_TRUST_RATIO`.

4. **Refined distance estimate** (seconds) -- Triggered when preliminary
   ``FOM >= fom_threshold_refine`` (default 6.0).  Runs 3 independent
   BP-OSD batches of ``refine_trials`` each and keeps the minimum.
   An additional OSD-CS order-10 verification pass
   (:func:`evaluation.distance.estimate_distance_osd_cs`) fires when
   ``FOM >= fom_threshold_exact``.

5. **Exact distance** (minutes) -- Triggered when refined
   ``FOM >= fom_threshold_exact`` (default 8.0).  Calls
   :func:`evaluation.distance.compute_distance_exact` with a configurable
   timeout.  Returns ``None`` on timeout (stage ``exact_timeout``).

The cascade is orchestrated by :func:`evaluate_candidate`, which returns
a dict with keys ``n, k, d, d_is_exact, distance_trusted, fom,
encoding_rate, ell, m, A_terms, B_terms, score, stage``.

:func:`evaluate_batch` and :func:`evaluate_lattices` provide batch
wrappers for evaluating lists of candidates across one or many lattices.

Constants
---------
SCORE_REJECTED : float
    Score assigned to all rejected candidates (``-inf``).
DISTANCE_TRUST_RATIO : float
    ``d / sqrt(n)`` threshold below which BP-OSD distance is fully trusted
    (1.3, just above the highest verified ratio of 1.26).
DISTANCE_UNTRUST_RATIO : float
    ``d / sqrt(n)`` threshold above which BP-OSD distance is discarded
    entirely (2.0, well below the lowest observed degenerate ratio of 2.5).
    Used by :mod:`evolve.openevolve_evaluator` to compute credible FOM
    with linear interpolation in the gap between the two thresholds.
"""

from __future__ import annotations

import logging
import math

from evaluation import gates
from evaluation.bb_code import build_bb_code, validate_terms, get_code_params_fast
from evaluation.distance import estimate_distance, estimate_distance_osd_cs, compute_distance_exact
from evaluation.distance_milp import compute_distance_milp, symplectic_weight_bound

logger = logging.getLogger(__name__)

SCORE_REJECTED = float("-inf")

# Trust boundaries for BP-OSD distance estimates (d/sqrt(n) ratio).
#
# Empirical observations on BB codes:
#   Known good codes:  d/√n = 0.71 ([[72,12,6]]), 1.0 ([[144,12,12]]),
#                      1.06 ([[288,12,18]]), 1.26 ([[360,12,24]])
#   Degenerate codes:  d/√n = 2.5-4.0 (BP-OSD wildly overestimates)
#   Gap:               No observed code has d/√n in (1.26, 2.5)
#
# DISTANCE_TRUST_RATIO: below this, BP-OSD d is fully trusted.
#   Set to 1.3, just above the highest verified d/√n (1.26), so all known
#   good codes get full FOM while leaving room for near-miss discoveries.
#
# DISTANCE_UNTRUST_RATIO: above this, BP-OSD d is discarded (use k/n only).
#   Set to 2.0, well below the lowest observed degenerate ratio (2.5). The
#   wide 1.3-2.0 decay zone provides a smooth gradient for borderline cases.
#   Tightening to e.g. √2 ≈ 1.41 would penalize plausible discoveries in
#   the uncharted 1.3-1.5 regime without empirical justification.
# Kept as literals for external consumers (ablation/verification scripts)
# that import these constants directly. Live gates inside this module read
# evaluation.gates fresh per call instead, so QCODE_SAVE_TRUST_RATIO /
# QCODE_MIN_K_THRESHOLD overrides take effect without re-importing.
DISTANCE_TRUST_RATIO = 1.3
DISTANCE_UNTRUST_RATIO = 2.0


MIN_K_THRESHOLD = 4
SCORE_K_LOW_PENALTY = -1000.0


def compute_fom(n: int, k: int, d: int) -> float:
    """Compute figure of merit kd²/n."""
    if n == 0 or k == 0 or d == 0:
        return 0.0
    return k * d * d / n


def _make_result_template(
    ell: int, m: int, A_terms: list, B_terms: list
) -> dict:
    """Create a default result dict for a candidate."""
    return {
        "ell": ell,
        "m": m,
        "A_terms": A_terms,
        "B_terms": B_terms,
        "n": 2 * ell * m,
        "k": 0,
        "d": 0,
        "d_is_exact": False,
        "distance_trusted": False,
        "fom": 0.0,
        "encoding_rate": 0.0,
        "score": SCORE_REJECTED,
        "stage": "rejected",
    }


def _validate_and_build(
    ell: int, m: int, A_terms: list, B_terms: list, result: dict
) -> tuple | None:
    """Validate terms, build code, compute k, apply early-exit rules.

    Returns (code, n, k) on success, or None if the candidate was rejected
    (result dict is updated in place with the rejection reason).
    """
    # Stage 1: Validate inputs
    try:
        validate_terms(ell, m, A_terms, "A")
        validate_terms(ell, m, B_terms, "B")
    except ValueError as e:
        logger.debug("Validation failed: %s", e)
        result["stage"] = "invalid"
        return None

    # Stage 2: Build code, compute k
    try:
        code = build_bb_code(ell, m, A_terms, B_terms)
        n, k = get_code_params_fast(code)
    except Exception as e:
        logger.debug("Construction failed: %s", e)
        result["stage"] = "construction_error"
        return None

    result["n"] = n
    result["k"] = k
    result["encoding_rate"] = k / n if n > 0 else 0.0

    if k == 0:
        result["stage"] = "k_zero"
        return None
    if k < gates.min_k_threshold():
        result["stage"] = "k_low"
        result["score"] = SCORE_K_LOW_PENALTY + k
        return None

    # Self-dual gate: BB codes with A=B always have d=2 (proven).
    # BP-OSD misses this in 29/30 batches, so we hard-code it.
    if sorted(tuple(t) for t in A_terms) == sorted(tuple(t) for t in B_terms):
        result["d"] = 2
        result["d_is_exact"] = True
        result["distance_trusted"] = True
        result["fom"] = k * 4 / n
        result["score"] = result["fom"]
        result["stage"] = "self_dual_d2"
        return None

    return code, n, k


def evaluate_candidate(
    ell: int,
    m: int,
    A_terms: list[tuple[int, int]],
    B_terms: list[tuple[int, int]],
    *,
    quick: bool = False,
    fom_threshold_refine: float = 6.0,
    fom_threshold_exact: float = 8.0,
    quick_trials: int = 100,
    refine_trials: int = 500,
    exact_timeout: int = 300,
) -> dict:
    """Evaluate a BB code candidate through the full cascade.

    Args:
        ell: Cyclic group order for x.
        m: Cyclic group order for y.
        A_terms: 3 exponent pairs for polynomial A.
        B_terms: 3 exponent pairs for polynomial B.
        quick: If True, only compute k (skip distance).
        fom_threshold_refine: FOM threshold to trigger refined distance estimation.
        fom_threshold_exact: FOM threshold to trigger exact distance computation.
        quick_trials: Number of BP-OSD trials for initial estimate.
        refine_trials: Number of BP-OSD trials for refined estimate.
        exact_timeout: Timeout in seconds for exact distance computation.

    Returns:
        Dict with keys: n, k, d, d_is_exact, fom, encoding_rate,
        ell, m, A_terms, B_terms, score, stage.
    """
    result = _make_result_template(ell, m, A_terms, B_terms)

    built = _validate_and_build(ell, m, A_terms, B_terms, result)
    if built is None:
        return result
    code, n, k = built

    if quick:
        result["stage"] = "quick_k_only"
        result["score"] = k / n  # use encoding rate as proxy
        return result

    # Stage 3: Quick distance estimate
    d_upper = estimate_distance(code, num_trials=quick_trials)
    if d_upper <= 2:
        result["d"] = d_upper
        result["distance_trusted"] = True  # d≤2 is always reliable
        result["fom"] = compute_fom(n, k, d_upper)
        result["score"] = result["fom"]
        result["stage"] = "trivial_distance"
        return result

    result["d"] = d_upper
    result["distance_trusted"] = d_upper <= gates.save_trust_ratio() * math.sqrt(n)
    fom = compute_fom(n, k, d_upper)
    result["fom"] = fom
    result["score"] = fom
    result["stage"] = "quick_estimate"

    # Stage 4: Refined distance estimate for promising candidates.
    # Run 3 independent BP-OSD batches (OSD_0) and take the minimum to reduce
    # variance. BP-OSD is an upper bound, so min of multiple runs is
    # tighter and more stable.
    if fom >= fom_threshold_refine:
        for _ in range(3):
            d_refined = estimate_distance(code, num_trials=refine_trials)
            d_upper = min(d_upper, d_refined)
        result["d"] = d_upper
        result["distance_trusted"] = d_upper <= gates.save_trust_ratio() * math.sqrt(n)
        fom = compute_fom(n, k, d_upper)
        result["fom"] = fom
        result["score"] = fom
        result["stage"] = "refined_estimate"

    # Stage 4b: OSD-CS verification for top candidates.
    # OSD-CS order=10 finds tighter bounds than OSD_0 for 7 of 9 tested
    # codes (4-12 point improvements on high-k codes). One batch of 200
    # trials catches the worst overestimates before they pollute fitness.
    if fom >= fom_threshold_exact:
        d_cs = estimate_distance_osd_cs(code, num_trials=200)
        if d_cs < d_upper:
            d_upper = d_cs
            result["d"] = d_upper
            result["distance_trusted"] = d_upper <= gates.save_trust_ratio() * math.sqrt(n)
            fom = compute_fom(n, k, d_upper)
            result["fom"] = fom
            result["score"] = fom
            result["stage"] = "osd_cs_verified"

    # Stage 5: Exact distance for top candidates
    if fom >= fom_threshold_exact:
        d_exact = compute_distance_exact(code, timeout_seconds=exact_timeout)
        if d_exact is not None:
            result["d"] = d_exact
            result["d_is_exact"] = True
            result["distance_trusted"] = True
            result["fom"] = compute_fom(n, k, d_exact)
            result["score"] = result["fom"]
            result["stage"] = "exact"
        else:
            result["stage"] = "exact_timeout"

    return result


def evaluate_batch(
    ell: int,
    m: int,
    candidates: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]],
    **kwargs,
) -> list[dict]:
    """Evaluate a batch of (A_terms, B_terms) candidates for a given lattice.

    Args:
        ell: Cyclic group order for x.
        m: Cyclic group order for y.
        candidates: List of (A_terms, B_terms) tuples, as returned by
            generate_candidates(ell, m).
        **kwargs: Passed to evaluate_candidate.

    Returns:
        List of result dicts, sorted by score descending.
    """
    results = []
    for A_terms, B_terms in candidates:
        result = evaluate_candidate(ell, m, A_terms, B_terms, **kwargs)
        results.append(result)
        if result["score"] > 0:
            logger.info(
                "[[%d, %d, %d]] FOM=%.2f (stage=%s)",
                result["n"], result["k"], result["d"],
                result["fom"], result["stage"],
            )
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def evaluate_lattices(
    lattices: list[tuple[int, int]],
    generate_fn,
    **kwargs,
) -> list[dict]:
    """Run evaluation across multiple lattice dimensions.

    Args:
        lattices: List of (ell, m) pairs to search.
        generate_fn: A function (ell, m) -> list of (A_terms, B_terms).
        **kwargs: Passed to evaluate_candidate.

    Returns:
        All results across all lattices, sorted by score descending.
    """
    all_results = []
    for ell, m in lattices:
        candidates = generate_fn(ell, m)
        logger.info(
            "Evaluating %d candidates for lattice (%d, %d), n=%d",
            len(candidates), ell, m, 2 * ell * m,
        )
        results = evaluate_batch(ell, m, candidates, **kwargs)
        all_results.extend(results)
    all_results.sort(key=lambda r: r["score"], reverse=True)
    return all_results


# ── MILP-based evaluation (Campaign 4+) ─────────────────────────


def evaluate_candidate_milp(
    ell: int,
    m: int,
    A_terms: list[tuple[int, int]],
    B_terms: list[tuple[int, int]],
    *,
    quick: bool = False,
    milp_timeout_per_logical: int = 30,
    milp_total_timeout: int = 120,
    milp_early_stop: int = 4,
) -> dict:
    """Evaluate a BB code candidate using MILP for distance bounds.

    Simplified 3-stage cascade (vs 5-stage BP-OSD cascade):
      1. Validate + build + compute k  (microseconds)
      2. Quick k-only return if quick=True  (microseconds)
      3. MILP distance  (sub-second for d≤4, seconds to minutes for d≥6)

    Every MILP incumbent is a valid upper bound (``distance_trusted=True``
    unconditionally -- unlike the BP-OSD cascade, no trust-ratio filtering
    is needed for that part), but ``d_is_exact`` is only True when every
    logical objective was solved to proven optimality; check
    ``result["d_is_exact"]`` (equivalently ``milp_details["exact"]``), not
    the ``stage`` name, to tell an exact result from an upper bound. A
    timeout with no incumbent at all (``stage=="milp_promising_timeout"``)
    yields no distance whatsoever -- see the all-timeout branch below.

    Args:
        ell: Cyclic group order for x.
        m: Cyclic group order for y.
        A_terms: 3 exponent pairs for polynomial A.
        B_terms: 3 exponent pairs for polynomial B.
        quick: If True, only compute k (skip distance).
        milp_timeout_per_logical: Timeout per individual ILP solve.
        milp_total_timeout: Total timeout for all logicals combined.
        milp_early_stop: Stop immediately when d ≤ this value.

    Returns:
        Dict with keys: n, k, d, d_is_exact, distance_trusted, fom,
        encoding_rate, ell, m, A_terms, B_terms, score, stage, milp_details.
    """
    result = _make_result_template(ell, m, A_terms, B_terms)

    built = _validate_and_build(ell, m, A_terms, B_terms, result)
    if built is None:
        return result
    code, n, k = built

    # Symplectic weight: instant upper bound on d from Gaussian elimination
    d_symp, _, _ = symplectic_weight_bound(code)
    result["d_symplectic"] = d_symp

    if quick:
        result["stage"] = "quick_k_only"
        result["score"] = k / n
        return result

    # Pre-filter using symplectic weight bound (instant, no MILP needed).
    # d_symp is an upper bound on d from Gaussian elimination.
    # - d_symp ≤ 2: provably exact (BB codes with k>0 have d ≥ 2)
    # - d_symp ≤ early_stop: MILP would solve in <1s anyway (d ≤ d_symp ≤ 4),
    #   but we can report d_symp directly as a valid upper bound and skip MILP.
    #   This saves hundreds of MILP calls per iteration.
    if d_symp <= milp_early_stop:
        result["d"] = d_symp
        result["d_is_exact"] = d_symp <= 2  # Only d≤2 is provably exact
        result["distance_trusted"] = True  # Valid upper bound
        result["fom"] = compute_fom(n, k, d_symp)
        result["score"] = result["fom"]
        result["stage"] = "symplectic_low_d"
        return result

    # Stage 3: MILP distance
    d, details = compute_distance_milp(
        code,
        timeout_per_logical=milp_timeout_per_logical,
        total_timeout=milp_total_timeout,
        early_stop=milp_early_stop,
    )

    result["milp_details"] = details

    if details.get("all_timeout"):
        # No feasible solution at all -- solver couldn't even find an
        # incumbent.  A timeout certifies neither an upper nor a lower
        # bound on d; it is not even weak evidence that d > early_stop,
        # since a timeout can happen for reasons unrelated to distance
        # (e.g. a hard search landscape).  The value below is a search
        # scheduling hint only -- how long this candidate resisted a quick
        # refutation, used to prioritize follow-up, not a bound on d -- and
        # is recorded under a name that downstream certification code must
        # not treat as proof.  Don't report phantom FOM; it would inflate
        # combined_score with fictitious values.
        result["d"] = 0
        result["distance_screening_threshold"] = milp_early_stop
        result["d_is_exact"] = False
        result["distance_trusted"] = False
        result["fom"] = 0.0
        result["score"] = 0.01  # Tiny positive: scheduling hint, not a bound on d
        result["stage"] = "milp_promising_timeout"
    else:
        result["d"] = d
        result["d_is_exact"] = details["exact"]
        result["distance_trusted"] = True  # Incumbent or optimal -- valid upper bound
        result["fom"] = compute_fom(n, k, d)
        result["score"] = result["fom"]
        if d <= milp_early_stop:
            result["stage"] = "milp_low_d"
        elif details["exact"]:
            result["stage"] = "milp_exact"
        else:
            # Incumbents found but not all proven optimal -- d is an upper
            # bound on true distance, so FOM is an upper bound too.
            result["stage"] = "milp_incumbent"

    return result


def evaluate_batch_milp(
    ell: int,
    m: int,
    candidates: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]],
    **kwargs,
) -> list[dict]:
    """Evaluate a batch of candidates using MILP distance.

    Args:
        ell: Cyclic group order for x.
        m: Cyclic group order for y.
        candidates: List of (A_terms, B_terms) tuples.
        **kwargs: Passed to evaluate_candidate_milp.

    Returns:
        List of result dicts, sorted by score descending.
    """
    results = []
    for A_terms, B_terms in candidates:
        result = evaluate_candidate_milp(ell, m, A_terms, B_terms, **kwargs)
        results.append(result)
        if result["score"] > 0:
            logger.info(
                "MILP [[%d, %d, %d]] FOM=%.2f (stage=%s, %.1fs)",
                result["n"], result["k"], result["d"],
                result["fom"], result["stage"],
                result.get("milp_details", {}).get("time_s", 0),
            )
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# ── Parallel k-screening ─────────────────────────────────────────


def evaluate_batch_milp_parallel(
    tasks: list[tuple[int, int, list, list]],
    *,
    max_workers: int | None = None,
    **kwargs,
) -> list[dict]:
    """Evaluate multiple candidates in parallel using ProcessPoolExecutor.

    Designed for the k-only screening phase where each candidate takes
    ~20-70ms (depending on lattice size) and there are 100k+ candidates.
    With 10 workers, this gives ~10x speedup over sequential evaluation.

    Args:
        tasks: List of (ell, m, A_terms, B_terms) tuples.
        max_workers: Number of parallel processes.  Defaults to
            min(cpu_count - 2, 10).
        **kwargs: Passed to evaluate_candidate_milp (e.g. quick=True).

    Returns:
        List of result dicts (same order as tasks).
    """
    import os
    from concurrent.futures import ProcessPoolExecutor

    if not tasks:
        return []

    if max_workers is None:
        max_workers = min((os.cpu_count() or 4) - 2, 10)
    max_workers = max(1, max_workers)

    worker_args = [(ell, m, A, B, kwargs) for ell, m, A, B in tasks]

    # chunksize=200 reduces IPC overhead: each worker processes 200
    # candidates per round-trip instead of 1.
    logger.info(
        "Parallel k-screening: %d tasks across %d workers",
        len(tasks), max_workers,
    )
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_milp_worker, worker_args, chunksize=200))

    n_valid = sum(1 for r in results if r.get("k", 0) > 0)
    logger.info(
        "Parallel k-screening done: %d/%d with k>0", n_valid, len(results),
    )
    return results


# ── Parallel MILP evaluation ─────────────────────────────────────


def _milp_worker(args):
    """Worker function for parallel evaluation (k-screening or MILP).

    Runs in a separate process via ProcessPoolExecutor.
    Args is a tuple: (ell, m, A_terms, B_terms, kwargs).
    """
    ell, m, A_terms, B_terms, kwargs = args
    return evaluate_candidate_milp(ell, m, A_terms, B_terms, **kwargs)


def _milp_cache_key(ell: int, m: int, A_terms, B_terms) -> tuple:
    """Canonical cache key for a BB code (lattice + sorted polynomial terms)."""
    a = tuple(sorted(tuple(t) for t in A_terms))
    b = tuple(sorted(tuple(t) for t in B_terms))
    return (ell, m, a, b)


def _load_milp_cache(path: str | None) -> dict[tuple, dict]:
    """Load MILP results cache from a JSONL file.

    Returns a dict mapping cache keys to result dicts.  Only caches
    results with d > 0 (successful solves).  For duplicate keys, keeps
    the best result: exact beats non-exact; among same exactness, lower
    d (tighter upper bound) wins.
    """
    import json
    from pathlib import Path

    cache: dict[tuple, dict] = {}
    if not path:
        return cache
    p = Path(path)
    if not p.exists():
        return cache
    try:
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # Skip corrupted lines, don't abort
            if r.get("d", 0) <= 0:
                continue
            key = _milp_cache_key(r["ell"], r["m"], r["A_terms"], r["B_terms"])
            existing = cache.get(key)
            if existing is None:
                cache[key] = r
            elif r.get("d_is_exact") and not existing.get("d_is_exact"):
                # Exact result always beats non-exact
                cache[key] = r
            elif not r.get("d_is_exact") and existing.get("d_is_exact"):
                pass  # Keep the exact result
            elif r.get("d", 0) < existing.get("d", 0):
                # Same exactness: lower d is tighter (better upper bound)
                cache[key] = r
    except OSError as e:
        logger.warning("MILP cache load error: %s", e)
    return cache


def evaluate_milp_parallel(
    tasks: list[tuple[int, int, list, list]],
    *,
    milp_timeout_per_logical: int = 300,
    milp_total_timeout: int = 7200,
    milp_early_stop: int = 4,
    max_workers: int | None = None,
    save_path: str | None = None,
) -> list[dict]:
    """Evaluate multiple (ell, m, A_terms, B_terms) tasks in parallel using MILP.

    Uses ProcessPoolExecutor for true parallelism (HiGHS is single-threaded,
    so each worker gets one core).  Results are saved incrementally to
    ``save_path`` (JSONL format) as each completes -- no code loss even if
    the parent process is killed.

    Includes a disk-based cache: results from previous iterations (loaded from
    ``save_path``) are reused without re-solving.  This avoids the dominant
    bottleneck where known seed codes are MILP-verified every iteration.

    Args:
        tasks: List of (ell, m, A_terms, B_terms) tuples.
        milp_timeout_per_logical: Per-logical ILP solve timeout (seconds).
        milp_total_timeout: Total timeout per code (seconds, 0=unlimited).
        milp_early_stop: Stop when d ≤ this value.
        max_workers: Number of parallel processes.  Defaults to
            min(cpu_count - 2, 10).
        save_path: Path to JSONL file for incremental persistence.
            Each result is appended as one JSON line immediately after solving.

    Returns:
        List of result dicts for all tasks, sorted by score descending.
    """
    import json
    import os
    import time as _time
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from pathlib import Path

    if not tasks:
        return []

    if max_workers is None:
        max_workers = min((os.cpu_count() or 4) - 2, 10)
    max_workers = max(1, max_workers)

    # ── Load MILP cache from previous results ──────────────────────
    cache = _load_milp_cache(save_path)

    # Separate cached vs uncached tasks
    cached_results = []
    uncached_tasks = []
    for ell, m, A_terms, B_terms in tasks:
        key = _milp_cache_key(ell, m, A_terms, B_terms)
        cached = cache.get(key)
        if cached is not None:
            cached_results.append(cached)
        else:
            uncached_tasks.append((ell, m, A_terms, B_terms))

    if cached_results:
        logger.info(
            "MILP cache: %d/%d tasks cached, %d to solve",
            len(cached_results), len(tasks), len(uncached_tasks),
        )

    kwargs = {
        "milp_timeout_per_logical": milp_timeout_per_logical,
        "milp_total_timeout": milp_total_timeout,
        "milp_early_stop": milp_early_stop,
    }

    # Prepare worker arguments for uncached tasks only
    worker_args = [
        (ell, m, A_terms, B_terms, kwargs)
        for ell, m, A_terms, B_terms in uncached_tasks
    ]

    results = list(cached_results)
    save_file = Path(save_path) if save_path else None
    if save_file:
        save_file.parent.mkdir(parents=True, exist_ok=True)

    if not worker_args:
        logger.info(
            "MILP cache: all %d tasks cached, skipping solver",
            len(tasks),
        )
        results.sort(key=lambda r: r.get("score", float("-inf")), reverse=True)
        return results

    t0 = _time.monotonic()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_milp_worker, args): i
            for i, args in enumerate(worker_args)
        }

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
            except Exception as e:
                ell, m, A_terms, B_terms, _ = worker_args[idx]
                logger.error(
                    "MILP worker error for (%d,%d) A=%s: %s",
                    ell, m, A_terms, e,
                )
                result = {
                    "ell": ell, "m": m,
                    "A_terms": A_terms, "B_terms": B_terms,
                    "n": 2 * ell * m, "k": 0, "d": 0,
                    "d_is_exact": False, "fom": 0.0,
                    "score": float("-inf"),
                    "stage": "worker_error",
                    "error": str(e),
                }

            results.append(result)

            # Incremental save: append to JSONL immediately
            if save_file and result.get("d", 0) > 0:
                result["_saved_at"] = _time.time()
                try:
                    with open(save_file, "a") as f:
                        f.write(json.dumps(result, default=str) + "\n")
                except OSError:
                    pass  # Don't fail evaluation over persistence

            # Log progress
            d = result.get("d", 0)
            if d > 0:
                logger.info(
                    "MILP [%d/%d] [[%d,%d,%d]] FOM=%.2f (%s, %.1fs)",
                    len(results) - len(cached_results), len(uncached_tasks),
                    result["n"], result["k"], d,
                    result.get("fom", 0), result.get("stage", "?"),
                    result.get("milp_details", {}).get("time_s", 0),
                )

    elapsed = _time.monotonic() - t0
    logger.info(
        "Parallel MILP: %d new + %d cached = %d total, %d workers, %.1fs",
        len(uncached_tasks), len(cached_results), len(results),
        max_workers, elapsed,
    )

    results.sort(key=lambda r: r.get("score", float("-inf")), reverse=True)
    return results
