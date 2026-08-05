"""OpenEvolve evaluator adapter for BB code discovery.

This module bridges OpenEvolve's ``evaluate(program_path)`` interface with
the project's multi-stage evaluation cascade.  OpenEvolve calls
``evaluate(program_path)`` (or ``evaluate_stage1`` / ``evaluate_stage2``
when cascade evaluation is enabled) with a temporary ``.py`` file containing
the evolved ``generate_candidates`` function.  This adapter dynamically
loads that function, runs it across target lattices, and returns a score
dict compatible with OpenEvolve's MAP-Elites population database.

Two-stage cascade
-----------------
**Stage 1** -- Quick k-only screening (~2 s)
    Evaluates the evolved program on 2 small lattices (``(6,6)`` and
    ``(12,6)``).  Programs must produce valid codes (``k > 0``) at
    **all** stage-1 lattices to advance past the cascade threshold.
    Score: ``0.1 base + best_encoding_rate + log1p(num_high_k) / 10``.

**Stage 2** -- Full evaluation with distance estimation (~30-60 s)
    Evaluates on 8 lattices with BP-OSD distance (1000 OSD_0 trials +
    200 OSD-CS order-10 trials for top candidates).  The primary fitness
    metric is ``combined_score`` -- the sum of the best *credible* FOM per
    lattice, where credibility is determined by a trust filter on
    ``d / sqrt(n)``:

    * ``d / sqrt(n) <= 1.3`` -- full trust: use raw FOM.
    * ``d / sqrt(n) >= 2.0`` -- no trust: use encoding rate (``k/n``) only.
    * Between -- linear interpolation (smooth decay, no cliff).

    Credible codes are persisted to ``results/discovered_codes.json`` and
    the Pareto front.  Metrics are written to a shared JSONL file for
    W&B background sync (since ``wandb.run`` is ``None`` in subprocess
    workers).

MAP-Elites feature dimensions
------------------------------
* ``lattices_with_high_k`` -- number of distinct lattices with at least one
  code having ``k >= 8``.
* ``num_high_k`` -- total count of codes with ``k >= 8``.

These features encourage behavioral diversity in the population: programs
that find high-k codes at many lattices occupy different niches from
programs that find a single exceptional code.

Constants
---------
STAGE1_LATTICES : list[tuple[int, int]]
    ``[(6, 6), (12, 6)]`` -- quick screening lattices.
STAGE2_LATTICES : list[tuple[int, int]]
    8 lattices for full evaluation.  Excludes ``(9,8)``, ``(10,10)``,
    ``(18,10)`` which produce zero ``k > 0`` codes.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import sys
from pathlib import Path

# Ensure the project root is on sys.path so we can import evaluation.*
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from evaluation.evaluator import (
    evaluate_batch,
    evaluate_batch_milp,
    evaluate_batch_milp_parallel,
    evaluate_milp_parallel,
    DISTANCE_TRUST_RATIO,
    DISTANCE_UNTRUST_RATIO,
)
from evaluation.results import save_code, update_pareto_front

# Per-model attribution: install here so it is active in each spawn worker
# (OpenEvolve imports this evaluator before generating). Defensive no-op if
# unavailable.
try:
    from evolve.model_attribution import install_tagging as _install_tagging
    _install_tagging()
except Exception:
    pass

logger = logging.getLogger(__name__)

# Lattice subsets for staged evaluation
# Stage 1: small/fast lattices for quick screening
STAGE1_LATTICES = [(6, 6), (12, 6)]
# Stage 2: full set
STAGE2_LATTICES = [
    (12, 6), (6, 12),
    (12, 12), (24, 6),
    (15, 12), (30, 6),
    (16, 9), (18, 8),
]
# Stage 2 for MILP: drop (16,9) and (18,8) which produce 0 valid x/y-swap codes
STAGE2_LATTICES_MILP = [
    (12, 6), (6, 12),
    (12, 12), (24, 6),
    (15, 12), (30, 6),
]


def _classify_pattern(A_terms, B_terms) -> float:
    """Classify polynomial structure for MAP-Elites.

    0.0 = univariate (A=f(y), B=g(x) or vice versa, incl. constant-monomial)
    1.0 = x/y-swap (pure terms, no constant, each poly mixes x+y axes)
    2.0 = self-dual (A=B)
    3.0 = mixed monomials (has x^a*y^b with both a,b > 0)
    4.0 = multi-term pure (4+ terms, all pure)
    5.0 = hybrid/non-standard pure (has constant term, not univariate --
          e.g. 1+x+y type, or constant-mono A + x/y-swap B)
    """
    a_set = sorted(tuple(t) for t in A_terms)
    b_set = sorted(tuple(t) for t in B_terms)
    if a_set == b_set:
        return 2.0
    has_mixed = any(x > 0 and y > 0 for x, y in A_terms) or \
                any(x > 0 and y > 0 for x, y in B_terms)
    if has_mixed:
        return 3.0
    a_y_only = all(x == 0 for x, y in A_terms)
    a_x_only = all(y == 0 for x, y in A_terms)
    b_y_only = all(x == 0 for x, y in B_terms)
    b_x_only = all(y == 0 for x, y in B_terms)
    if (a_y_only and b_x_only) or (a_x_only and b_y_only):
        return 0.0
    # 4+ term pure polynomials get their own niche
    if len(A_terms) >= 4 or len(B_terms) >= 4:
        return 4.0
    # Hybrid / non-standard pure: has a constant term (0,0) but isn't
    # univariate.  Separates novel structures (1+x+y, hybrid cross-family)
    # from classic x/y-swap (which never has a constant term).
    a_has_const = any(x == 0 and y == 0 for x, y in A_terms)
    b_has_const = any(x == 0 and y == 0 for x, y in B_terms)
    if a_has_const or b_has_const:
        return 5.0
    return 1.0


def _count_terms(A_terms, B_terms) -> float:
    """Return max term count across A and B (for MAP-Elites feature)."""
    return float(max(len(A_terms), len(B_terms)))


def _structural_feedback(result: dict) -> str:
    """Generate structural feedback string for a code with d >= 4.

    Reports mixed vs pure term counts, shift direction vectors,
    and axis coupling -- simple properties that help the LLM reason
    about WHY a code worked.
    """
    A = result.get("A_terms", [])
    B = result.get("B_terms", [])

    def _classify_terms(terms, name):
        pure_x = sum(1 for x, y in terms if x > 0 and y == 0)
        pure_y = sum(1 for x, y in terms if x == 0 and y > 0)
        const = sum(1 for x, y in terms if x == 0 and y == 0)
        mixed = sum(1 for x, y in terms if x > 0 and y > 0)
        return f"{name}: {pure_x} pure-x, {pure_y} pure-y, {const} const, {mixed} mixed"

    lines = [
        _classify_terms(A, "A"),
        _classify_terms(B, "B"),
        f"A shifts: {A}",
        f"B shifts: {B}",
        f"Terms: |A|={len(A)}, |B|={len(B)}",
    ]

    a_has_mixed = any(x > 0 and y > 0 for x, y in A)
    b_has_mixed = any(x > 0 and y > 0 for x, y in B)
    if a_has_mixed and b_has_mixed:
        lines.append("Axis coupling: both A and B have mixed terms")
    elif a_has_mixed:
        lines.append("Axis coupling: only A has mixed terms")
    elif b_has_mixed:
        lines.append("Axis coupling: only B has mixed terms")
    else:
        lines.append("Axis coupling: none (pure-term code)")

    return "\n  ".join(lines)


def _log_code_jsonl(result: dict, run_name: str | None = None) -> None:
    """Append a code result to the run-specific all_codes.jsonl file.

    Logs ALL codes with d > 0 (not just high-FOM codes), providing the
    full dataset for Phase D pattern extraction.  Each line is ~200 bytes.

    The run_name is resolved from (in priority order):
    1. Explicit ``run_name`` argument
    2. ``QCODE_RUN_NAME`` environment variable (set by run_evolution.py)
    3. Fallback to ``results/evolution/all_codes.jsonl``
    """
    import json
    import os
    import time

    d = result.get("d", 0)
    if d <= 0:
        return

    # Resolve run_name from argument or environment
    if not run_name:
        run_name = os.environ.get("QCODE_RUN_NAME")

    record = {
        "ell": result.get("ell"),
        "m": result.get("m"),
        "A_terms": result.get("A_terms"),
        "B_terms": result.get("B_terms"),
        "n": result.get("n"),
        "k": result.get("k"),
        "d": d,
        "fom": result.get("fom", 0.0),
        "stage": result.get("stage", ""),
        "pattern_type": _classify_pattern(
            result.get("A_terms", []), result.get("B_terms", [])
        ),
        "term_count": _count_terms(
            result.get("A_terms", []), result.get("B_terms", [])
        ),
        "timestamp": time.time(),
    }

    if run_name:
        log_dir = Path(_PROJECT_ROOT) / "results" / "evolution" / run_name
    else:
        log_dir = Path(_PROJECT_ROOT) / "results" / "evolution"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "all_codes.jsonl"

    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError:
        pass  # Don't fail evaluation over logging


def _error_result(error: str) -> dict:
    """Return an error result that includes all feature dimensions.

    MAP-Elites requires the feature dimensions to be present in every
    result, including errors.
    """
    return {
        "combined_score": 0.0,
        "error": error,
        "lattices_with_high_k": 0.0,
        "num_high_k": 0.0,
        "term_count": 0.0,
        "pattern_type": 0.0,
    }


def _load_generate_candidates(program_path: str):
    """Load generate_candidates from an evolved program file."""
    if not Path(program_path).exists():
        raise FileNotFoundError(f"Evolved program not found: {program_path}")
    spec = importlib.util.spec_from_file_location("evolved_program", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "generate_candidates"):
        raise AttributeError("Evolved program missing generate_candidates function")

    return module.generate_candidates


def _run_evaluation(
    generate_fn,
    lattices: list[tuple[int, int]],
    quick: bool = False,
    quick_trials: int = 100,
    refine_trials: int = 500,
    max_distance_per_lattice: int = 10,
    use_milp: bool = False,
    milp_timeout_per_logical: int = 30,
    milp_total_timeout: int = 120,
    milp_early_stop: int = 4,
    run_name: str | None = None,
) -> dict:
    """Run evaluation across lattices and compute aggregate metrics.

    When quick=False, uses a two-pass approach per lattice:
    1. Quick k-only screen of all candidates
    2. Distance estimation for the top `max_distance_per_lattice` candidates,
       using either BP-OSD (default) or MILP (when use_milp=True).
    """
    all_results = []
    total_candidates = 0
    errors = []

    for ell, m in lattices:
        try:
            candidates = generate_fn(ell, m)
            if not isinstance(candidates, list):
                errors.append(f"({ell},{m}): generate_candidates returned {type(candidates)}, not list")
                continue

            total_candidates += len(candidates)

            # Cap candidates per lattice to avoid runaway generation
            if len(candidates) > 5000:
                errors.append(f"({ell},{m}): {len(candidates)} candidates, capped to 5000")
                candidates = candidates[:5000]

            if quick:
                results = evaluate_batch(
                    ell, m, candidates,
                    quick=True,
                    quick_trials=quick_trials,
                    fom_threshold_refine=6.0,
                    fom_threshold_exact=8.0,
                )
            else:
                # Two-pass: quick screen, then distance on top candidates.
                # MILP path uses evaluate_batch_milp(quick=True) to get
                # symplectic weight bounds for smarter top-k ranking.
                if use_milp:
                    quick_results = evaluate_batch_milp(
                        ell, m, candidates, quick=True,
                    )
                else:
                    quick_results = evaluate_batch(
                        ell, m, candidates, quick=True,
                    )
                # Select diverse candidates for distance estimation.
                # Diversify on BOTH k value AND polynomial A -- prevents
                # wasting MILP budget on near-duplicate codes (e.g. 5 codes
                # with same A at (15,12) all giving k=16).
                # When symplectic weight is available (MILP path), rank by
                # approximate FOM = k * d_symp^2 / n instead of k alone.
                promising = [
                    r for r in quick_results if r.get("k", 0) >= 8
                ]
                n_code = 2 * ell * m
                def _rank_key(r):
                    d_s = r.get("d_symplectic", 0)
                    if d_s > 0 and use_milp:
                        return r["k"] * d_s * d_s / n_code
                    return r["k"]
                promising.sort(key=_rank_key, reverse=True)

                # Pass 1: one per distinct k value
                seen_k: set[int] = set()
                top: list[dict] = []
                for r in promising:
                    if r["k"] not in seen_k and len(top) < max_distance_per_lattice:
                        seen_k.add(r["k"])
                        top.append(r)
                # Pass 2: one per distinct A polynomial (among same-k codes)
                seen_a: set[tuple] = {tuple(map(tuple, r["A_terms"])) for r in top}
                for r in promising:
                    if len(top) >= max_distance_per_lattice:
                        break
                    a_key = tuple(map(tuple, r["A_terms"]))
                    if a_key not in seen_a and r not in top:
                        seen_a.add(a_key)
                        top.append(r)
                # Pass 3: fill remaining slots
                for r in promising:
                    if len(top) >= max_distance_per_lattice:
                        break
                    if r not in top:
                        top.append(r)

                top_candidates = [
                    (r["A_terms"], r["B_terms"]) for r in top
                ]

                if use_milp:
                    # MILP: scale timeout with n.  The adaptive per-logical
                    # timeout in distance_milp.py uses max(8s, total/2k),
                    # so the total budget directly determines coverage.
                    # k=24 at n=288 with 240s: 8s/logical × 30 logicals
                    # (63% of 48).  Most codes solve instantly (d=2-4),
                    # so only 1-2 codes per lattice use the full budget.
                    n_code = 2 * ell * m
                    if n_code <= 200:
                        lat_timeout = min(milp_total_timeout, 120)
                        lat_per_log = min(milp_timeout_per_logical, 20)
                    elif n_code <= 300:
                        lat_timeout = min(milp_total_timeout, 240)
                        lat_per_log = min(milp_timeout_per_logical, 45)
                    else:
                        lat_timeout = milp_total_timeout
                        lat_per_log = milp_timeout_per_logical
                    results = evaluate_batch_milp(
                        ell, m, top_candidates,
                        milp_timeout_per_logical=lat_per_log,
                        milp_total_timeout=lat_timeout,
                        milp_early_stop=milp_early_stop,
                    )
                else:
                    # BP-OSD: use refine_trials for tighter upper bounds.
                    # Skip exact distance (stage 5) -- requires SIGALRM which
                    # isn't available in OpenEvolve's worker threads.
                    results = evaluate_batch(
                        ell, m, top_candidates,
                        quick=False,
                        quick_trials=refine_trials,
                        fom_threshold_refine=6.0,
                        fom_threshold_exact=float("inf"),
                    )
                # Include quick-only results for aggregate counting
                quick_only = [
                    r for r in quick_results if r.get("k", 0) > 0
                    and r not in top
                ]
                results.extend(quick_only)

            all_results.extend(results)

            # Log ALL codes with d > 0 to JSONL for Phase D pattern extraction
            for r in results:
                _log_code_jsonl(r, run_name=run_name)
        except Exception as e:
            errors.append(f"({ell},{m}): {type(e).__name__}: {e}")

    # Compute aggregate metrics using encoding rate (exact) and FOM (approximate)
    valid = [r for r in all_results if r.get("k", 0) > 0]
    foms = [r.get("fom", 0.0) for r in valid if r.get("fom", 0.0) > 0]
    encoding_rates = [r.get("encoding_rate", 0.0) for r in valid]

    best_fom = max(foms) if foms else 0.0
    mean_fom = sum(foms) / len(foms) if foms else 0.0
    num_above_6 = sum(1 for f in foms if f >= 6.0)
    num_above_12 = sum(1 for f in foms if f >= 12.0)
    best_encoding_rate = max(encoding_rates) if encoding_rates else 0.0

    # Count high-k codes (k >= 8) -- exact metric, not affected by BP-OSD
    high_k_codes = [r for r in valid if r.get("k", 0) >= 8]
    # Count lattices with at least one high-k code (breadth across lattices)
    lattices_with_high_k = len(set(
        (r["ell"], r["m"]) for r in high_k_codes
    ))

    # Find the best code for reporting
    best_code = None
    if all_results:
        best_result = max(all_results, key=lambda r: r.get("fom", 0.0))
        if best_result.get("fom", 0.0) > 0:
            best_code = best_result

    return {
        "best_fom": best_fom,
        "mean_fom": mean_fom,
        "num_valid": len(valid),
        "num_above_6": num_above_6,
        "num_above_12": num_above_12,
        "total_candidates": total_candidates,
        "best_encoding_rate": best_encoding_rate,
        "num_high_k": len(high_k_codes),
        "lattices_with_high_k": lattices_with_high_k,
        "best_code": best_code,
        "all_results": all_results,
        "errors": errors,
    }


def evaluate_stage1(program_path: str) -> dict:
    """Stage 1: Quick screening on small lattices (k-only, ~2s).

    Programs must produce valid codes (k > 0) at ALL stage-1 lattices to
    advance.  (6,6) and (12,6) are the most forgiving lattices -- any
    reasonable x/y-swap program produces k>0 at both.  Failing either
    signals a fundamentally broken strategy, not worth ~5 min of stage-2
    evaluation.

    Score components (all lattices covered):
    - 0.1 base: guarantees passing cascade threshold (0.01)
    - best_rate: peak encoding rate (up to ~0.17 for k=12 at n=72)
    - high_k_quality: log-scaled count of k≥8 codes (diminishing returns,
      prevents gaming by sparse generators that score high on ratios)
    Total range: ~0.11 (barely alive) to ~0.8 (excellent), giving
    MAP-Elites ~7x differentiation vs the prior 15% spread.
    """
    try:
        generate_fn = _load_generate_candidates(program_path)
    except Exception as e:
        return _error_result(str(e))

    metrics = _run_evaluation(generate_fn, STAGE1_LATTICES, quick=True)

    if metrics["total_candidates"] == 0:
        return {
            "combined_score": 0.0,
            "num_valid": 0.0,
            "total_candidates": 0.0,
            "lattices_with_high_k": 0.0,
            "num_high_k": 0.0,
            "term_count": 0.0,
            "pattern_type": 0.0,
        }

    valid = [r for r in metrics.get("all_results", []) if r.get("k", 0) > 0]

    # Require valid codes at ALL stage-1 lattices
    lattices_with_valid = set(
        (r["ell"], r["m"]) for r in valid
    )
    lattice_coverage = len(lattices_with_valid) / len(STAGE1_LATTICES)

    if lattice_coverage < 1.0:
        # Partial coverage: below cascade threshold (0.01)
        score = len(lattices_with_valid) * 0.001
    else:
        # All lattices covered. Score by quality.
        best_rate = max(r.get("encoding_rate", 0.0) for r in valid)
        # Log-scaled count: rewards breadth without letting sparse
        # generators game the score (2 codes → 0.11, 259 codes → 0.56)
        high_k_quality = math.log1p(metrics["num_high_k"]) / 10.0
        score = 0.1 + best_rate + high_k_quality

    # MAP-Elites features: compute from best valid code
    best_valid = max(valid, key=lambda r: r.get("k", 0)) if valid else None
    if best_valid:
        best_tc = _count_terms(best_valid["A_terms"], best_valid["B_terms"])
        best_pattern = _classify_pattern(best_valid["A_terms"], best_valid["B_terms"])
    else:
        best_tc = 0.0
        best_pattern = 0.0

    return {
        "combined_score": score,
        "num_valid": float(metrics["num_valid"]),
        "total_candidates": float(metrics["total_candidates"]),
        "lattices_with_high_k": float(metrics["lattices_with_high_k"]),
        "num_high_k": float(metrics["num_high_k"]),
        "term_count": best_tc,
        "pattern_type": best_pattern,
    }


def evaluate_stage2(program_path: str) -> dict:
    """Stage 2: Full evaluation with distance estimation (~30-60s).

    Runs across all target lattices with distance estimation.
    """
    try:
        generate_fn = _load_generate_candidates(program_path)
    except Exception as e:
        return _error_result(str(e))

    metrics = _run_evaluation(
        generate_fn, STAGE2_LATTICES,
        quick=False, refine_trials=1000,
    )

    # --- Combined score ---
    # Sum of best *credible* FOM per lattice.
    #
    # BP-OSD with 1000 trials gives exact d for well-structured codes
    # (verified on [[72,12,6]], [[144,12,12]], [[288,12,18]]). But for
    # degenerate high-k codes it wildly overestimates d.
    #
    # We FILTER rather than cap: only trust BP-OSD estimates where
    # d ≤ TRUST_FULL * sqrt(n). Known best BB codes have d/sqrt(n) ≤ 1.26.
    # Degenerate high-k codes have d/sqrt(n) ≥ 2.5, leaving a wide gap.
    # Codes failing the filter get a small encoding-rate bonus (k/n)
    # instead, so they're not completely invisible but can't dominate.

    # Trust boundaries (imported from evaluation.evaluator -- single source of truth).
    TRUST_FULL = DISTANCE_TRUST_RATIO    # d/sqrt(n) ≤ 1.3: fully trust FOM
    TRUST_NONE = DISTANCE_UNTRUST_RATIO  # d/sqrt(n) ≥ 2.0: discard FOM, use k/n only
    # Between TRUST_FULL and TRUST_NONE: linear interpolation (soft decay, no cliff)
    best_fom = metrics["best_fom"]

    per_lattice_best: dict[tuple[int, int], float] = {}
    for r in metrics["all_results"]:
        d_raw = r.get("d", 0)
        k = r.get("k", 0)
        n = r.get("n", 0)
        if k <= 0 or n <= 0:
            continue
        key = (r["ell"], r["m"])
        fallback = k / n  # encoding rate, always available
        if d_raw <= 0:
            credible_fom = fallback
        else:
            ratio = d_raw / math.sqrt(n)
            raw_fom = k * d_raw * d_raw / n
            if ratio <= TRUST_FULL:
                credible_fom = raw_fom
            elif ratio >= TRUST_NONE:
                credible_fom = fallback
            else:
                # Linear decay: 100% FOM at TRUST_FULL, 0% at TRUST_NONE
                alpha = (TRUST_NONE - ratio) / (TRUST_NONE - TRUST_FULL)
                credible_fom = alpha * raw_fom + (1 - alpha) * fallback
        per_lattice_best[key] = max(per_lattice_best.get(key, 0.0), credible_fom)

    combined = sum(per_lattice_best.values())

    # Build artifacts for LLM feedback.
    # Show the best code by CREDIBLE FOM (d-filtered), not raw BP-OSD FOM,
    # so the LLM learns from genuine patterns, not degenerate codes.
    artifacts = {}
    credible_codes = []
    for r in metrics["all_results"]:
        d_val = r.get("d", 0)
        n_val = r.get("n", 0)
        if d_val > 0 and n_val > 0 and d_val <= TRUST_FULL * math.sqrt(n_val):
            credible_codes.append(r)
    if credible_codes:
        bc = max(credible_codes, key=lambda r: r.get("fom", 0.0))
        artifacts["best_code"] = (
            f"[[{bc['n']},{bc['k']},{bc['d']}]] FOM={bc['fom']:.2f} "
            f"at ({bc['ell']},{bc['m']})\n"
            f"  A={bc['A_terms']}\n"
            f"  B={bc['B_terms']}"
        )
    elif metrics["best_code"]:
        bc = metrics["best_code"]
        artifacts["best_code"] = (
            f"[[{bc['n']},{bc['k']},{bc.get('d', '?')}]] "
            f"(d estimate unreliable) at ({bc['ell']},{bc['m']})\n"
            f"  A={bc['A_terms']}\n"
            f"  B={bc['B_terms']}"
        )

    if metrics["errors"]:
        artifacts["errors"] = "\n".join(metrics["errors"][:5])

    # Report top 5 codes by credible FOM (reusing list from above)
    top5 = sorted(credible_codes, key=lambda r: r.get("fom", 0), reverse=True)[:5]
    if top5:
        top5_lines = []
        for r in top5:
            top5_lines.append(
                f"  [[{r['n']},{r['k']},{r['d']}]] FOM={r['fom']:.1f} "
                f"rate={r.get('encoding_rate', 0):.3f} ({r['ell']},{r['m']})"
            )
        artifacts["top_codes"] = "\n".join(top5_lines)

    # Structural feedback for codes with d >= 4 (helps LLM reason about patterns)
    d4_codes = [r for r in credible_codes if r.get("d", 0) >= 4]
    if d4_codes:
        struct_lines = []
        for r in sorted(d4_codes, key=lambda r: r.get("fom", 0), reverse=True)[:3]:
            struct_lines.append(
                f"  [[{r['n']},{r['k']},{r['d']}]] FOM={r['fom']:.1f}:\n"
                f"    {_structural_feedback(r)}"
            )
        artifacts["structural_analysis"] = (
            "Structural analysis of top codes with d>=4:\n" + "\n".join(struct_lines)
        )

    # Per-lattice breakdown for the LLM
    lattice_lines = []
    for key in sorted(per_lattice_best.keys()):
        lattice_lines.append(
            f"  ({key[0]},{key[1]}): credible FOM={per_lattice_best[key]:.1f}"
        )

    artifacts["summary"] = (
        f"Evaluated {metrics['total_candidates']} candidates across "
        f"{len(STAGE2_LATTICES)} lattices.\n"
        f"Valid codes (k>0): {metrics['num_valid']}\n"
        f"High-k codes (k>=8): {metrics['num_high_k']}\n"
        f"Lattices with high-k: {metrics['lattices_with_high_k']}/{len(STAGE2_LATTICES)}\n"
        f"Best raw FOM (BP-OSD, 1000 trials + OSD-CS check): {best_fom:.2f}\n"
        f"Combined score: {combined:.1f} = sum of best credible FOM per lattice "
        f"(full trust d/sqrt(n) <= {TRUST_FULL}, soft decay to {TRUST_NONE})\n"
        f"Per-lattice breakdown:\n" + "\n".join(lattice_lines)
    )

    # Save only codes with trusted distances (d in fully-trusted zone)
    credible_to_save = [
        r for r in metrics["all_results"]
        if r.get("fom", 0) > 0
        and r.get("d", 0) > 0
        and r.get("n", 0) > 0
        and r["d"] <= TRUST_FULL * math.sqrt(r["n"])
    ]
    if credible_to_save:
        best_credible = max(credible_to_save, key=lambda r: r["fom"])
        if best_credible["fom"] > 6.0:
            try:
                save_code(best_credible)
                update_pareto_front(credible_to_save)
            except Exception:
                pass  # Don't fail evaluation over persistence

    # Write metrics to shared JSONL file for W&B sync from main process.
    # (wandb.run is None in subprocess workers, so direct wandb.log doesn't work.)
    _write_metrics_jsonl(metrics)

    # MAP-Elites features from best credible code
    best_credible_code = max(credible_codes, key=lambda r: r.get("fom", 0)) if credible_codes else None
    if best_credible_code:
        s2_tc = _count_terms(best_credible_code["A_terms"], best_credible_code["B_terms"])
        s2_pattern = _classify_pattern(best_credible_code["A_terms"], best_credible_code["B_terms"])
    else:
        s2_tc = 0.0
        s2_pattern = 0.0

    result = {
        "combined_score": combined,
        "best_fom": best_fom,
        "mean_fom": metrics["mean_fom"],
        "num_valid": float(metrics["num_valid"]),
        "num_high_k": float(metrics["num_high_k"]),
        "lattices_with_high_k": float(metrics["lattices_with_high_k"]),
        "best_encoding_rate": metrics["best_encoding_rate"],
        "num_above_6": float(metrics["num_above_6"]),
        "num_above_12": float(metrics["num_above_12"]),
        "total_candidates": float(metrics["total_candidates"]),
        "term_count": s2_tc,
        "pattern_type": s2_pattern,
    }

    try:
        from openevolve.evaluation_result import EvaluationResult
        return EvaluationResult(metrics=result, artifacts=artifacts)
    except ImportError:
        return result


def evaluate_stage2_milp(program_path: str) -> dict:
    """Stage 2 with parallel MILP distance verification.

    Two-phase approach:
    1. Quick k-only screening on all lattices (sequential, ~3s)
    2. Parallel MILP verification for top candidates across all lattices

    Key improvements over Campaign 4:
    - Parallel MILP evaluation using ProcessPoolExecutor (10 workers)
    - 300s per-logical timeout (proven sufficient for n≤288 exact)
    - 7200s total timeout per code (covers k=12-24 fully)
    - Incremental code persistence to JSONL after each solve
    - Scoring: only codes with d≥6 contribute FOM (d≤4 are irrelevant)
    - Symplectic weight pre-filter: codes with d_symp≤4 skip MILP entirely
    - Better LLM feedback with clear signal about what works vs doesn't
    """
    try:
        generate_fn = _load_generate_candidates(program_path)
    except Exception as e:
        return _error_result(str(e))

    # MILP budget parameters
    MILP_TIMEOUT_PER_LOGICAL = 300   # 300s proven sufficient (milp_optimality_audit.json)
    MILP_TOTAL_TIMEOUT = 7200        # 2h per code -- covers k≤24 fully
    MILP_EARLY_STOP = 4              # Exit immediately if d≤4
    MAX_DISTANCE_PER_LATTICE = 5     # Top-5 per lattice → ~30 MILP tasks
    MIN_RELEVANT_D = 6               # Only d≥6 codes contribute to score

    # Save path for incremental MILP persistence
    codes_jsonl = str(Path(_PROJECT_ROOT) / "results" / "evolution_codes.jsonl")

    # ── Phase 1: k-only screening (parallel) ───────────────────────
    # Collect all candidates across lattices, then evaluate in parallel
    # using ProcessPoolExecutor.  ~10x faster than sequential (~3 min
    # vs ~30 min for 120k candidates).
    all_quick_tasks = []  # (ell, m, A_terms, B_terms)
    total_candidates = 0
    errors = []

    for ell, m in STAGE2_LATTICES_MILP:
        try:
            candidates = generate_fn(ell, m)
            if not isinstance(candidates, list):
                errors.append(f"({ell},{m}): returned {type(candidates)}, not list")
                continue
            total_candidates += len(candidates)
            # Cap at 20000 per lattice.
            # Priority candidates (perturbations, simultaneous perturbations)
            # are generated first in the seed, so they survive the cap.
            # Strategy 4's exhaustive search fills remaining slots.
            if len(candidates) > 20000:
                errors.append(f"({ell},{m}): {len(candidates)} candidates, capped to 20000")
                candidates = candidates[:20000]

            for A_terms, B_terms in candidates:
                all_quick_tasks.append((ell, m, A_terms, B_terms))
        except Exception as e:
            errors.append(f"({ell},{m}): {type(e).__name__}: {e}")

    all_quick_results = evaluate_batch_milp_parallel(
        all_quick_tasks, quick=True,
    )

    # ── Phase 2: Select top candidates for MILP ────────────────────
    # Use d_symplectic to filter and rank -- only codes with d_symp ≥ 5
    # are worth MILP verification (d_symp ≤ 4 already resolved by
    # evaluate_candidate_milp's pre-filter).  Among qualifying codes,
    # rank by approximate FOM = k * d_symp² / n.
    milp_tasks = []  # (ell, m, A_terms, B_terms)
    milp_skipped_low_d_symp = 0

    for ell, m in STAGE2_LATTICES_MILP:
        lattice_results = [
            r for r in all_quick_results
            if r.get("ell") == ell and r.get("m") == m
            and r.get("k", 0) >= 4
        ]

        # Filter: only consider codes with d_symplectic high enough to
        # potentially have d ≥ MIN_RELEVANT_D.  This is the key optimization
        # from the symplectic basis pre-filter.
        n_code = 2 * ell * m
        promising = []
        for r in lattice_results:
            d_s = r.get("d_symplectic", 0)
            if d_s > MILP_EARLY_STOP:
                promising.append(r)
            else:
                milp_skipped_low_d_symp += 1

        # Rank by approximate FOM using symplectic weight
        def _rank_key(r):
            d_s = r.get("d_symplectic", 0)
            if d_s > 0:
                return r["k"] * d_s * d_s / n_code
            return r["k"]
        promising.sort(key=_rank_key, reverse=True)

        # Diversify: pick top candidates by k, A-polynomial diversity
        seen_k: set[int] = set()
        top: list[dict] = []
        for r in promising:
            if r["k"] not in seen_k and len(top) < MAX_DISTANCE_PER_LATTICE:
                seen_k.add(r["k"])
                top.append(r)
        seen_a: set[tuple] = {tuple(map(tuple, r["A_terms"])) for r in top}
        for r in promising:
            if len(top) >= MAX_DISTANCE_PER_LATTICE:
                break
            a_key = tuple(map(tuple, r["A_terms"]))
            if a_key not in seen_a and r not in top:
                seen_a.add(a_key)
                top.append(r)
        for r in promising:
            if len(top) >= MAX_DISTANCE_PER_LATTICE:
                break
            if r not in top:
                top.append(r)

        for r in top:
            milp_tasks.append((ell, m, r["A_terms"], r["B_terms"]))

    # ── Phase 3: Parallel MILP verification ────────────────────────
    # Run all MILP tasks in parallel using ProcessPoolExecutor.
    # Each result is saved to JSONL immediately after solving.
    milp_results = []
    if milp_tasks:
        milp_results = evaluate_milp_parallel(
            milp_tasks,
            milp_timeout_per_logical=MILP_TIMEOUT_PER_LOGICAL,
            milp_total_timeout=MILP_TOTAL_TIMEOUT,
            milp_early_stop=MILP_EARLY_STOP,
            save_path=codes_jsonl,
        )

    # Combine: MILP results + k-only results (for codes that didn't get MILP)
    all_results = list(milp_results)
    milp_keys = {
        (r["ell"], r["m"], tuple(map(tuple, r["A_terms"])), tuple(map(tuple, r["B_terms"])))
        for r in milp_results
    }
    for r in all_quick_results:
        key = (r["ell"], r["m"], tuple(map(tuple, r["A_terms"])), tuple(map(tuple, r["B_terms"])))
        if key not in milp_keys and r.get("k", 0) > 0:
            all_results.append(r)

    # ── Phase 4: Scoring ───────────────────────────────────────────
    # Only codes with d ≥ MIN_RELEVANT_D contribute FOM to combined_score.
    # This sends a clear signal: d≤4 codes are worthless.
    per_lattice_best: dict[tuple[int, int], float] = {}
    for r in all_results:
        d = r.get("d", 0)
        fom = r.get("fom", 0.0)
        k = r.get("k", 0)
        if k <= 0:
            continue
        key = (r["ell"], r["m"])
        if d >= MIN_RELEVANT_D and fom > 0:
            per_lattice_best[key] = max(per_lattice_best.get(key, 0.0), fom)
        elif d > 0 and d < MIN_RELEVANT_D:
            # Low-d codes: tiny contribution (like encoding rate)
            per_lattice_best[key] = max(per_lattice_best.get(key, 0.0), 0.01)

    combined = sum(per_lattice_best.values())

    # Aggregate metrics
    valid = [r for r in all_results if r.get("k", 0) > 0]
    foms = [r.get("fom", 0.0) for r in valid if r.get("fom", 0.0) > 0]
    best_fom = max(foms) if foms else 0.0
    mean_fom = sum(foms) / len(foms) if foms else 0.0
    high_k_codes = [r for r in valid if r.get("k", 0) >= 8]
    lattices_with_high_k = len(set(
        (r["ell"], r["m"]) for r in high_k_codes
    ))
    num_above_6 = sum(1 for f in foms if f >= 6.0)
    num_above_12 = sum(1 for f in foms if f >= 12.0)

    # ── Phase 5: LLM feedback artifacts ────────────────────────────
    artifacts = {}

    def _stage_tag(r):
        """Human-readable label for MILP result quality."""
        stage = r.get("stage", "")
        if stage == "milp_exact":
            return "exact"
        elif stage == "milp_incumbent":
            return "d\u2264" + str(r["d"])
        elif stage == "milp_promising_timeout":
            return "d>" + str(r.get("milp_details", {}).get("early_stop", 4))
        elif stage in ("milp_low_d", "symplectic_low_d"):
            return "exact" if r.get("d", 0) <= 2 else f"d\u2264{r['d']}"
        return ""

    # Reference codes for perturbation gradient analysis
    _REFERENCE_CODES = [
        {"ell": 12, "m": 6, "A": [(3,0),(0,1),(0,2)], "B": [(0,3),(1,0),(2,0)],
         "d": 12, "k": 12, "name": "[[144,12,12]] gross"},
        {"ell": 12, "m": 12, "A": [(3,0),(0,2),(0,7)], "B": [(0,3),(1,0),(2,0)],
         "d": 18, "k": 12, "name": "[[288,12,18]] bravyi"},
        {"ell": 12, "m": 12, "A": [(6,0),(0,1),(0,2)], "B": [(0,3),(2,0),(4,0)],
         "d": 12, "k": 24, "name": "[[288,24,12]]"},
        {"ell": 12, "m": 12, "A": [(3,0),(0,1),(0,2)], "B": [(0,3),(1,0),(2,0)],
         "d": 12, "k": 16, "name": "[[288,16,12]] gross-scaled"},
        {"ell": 15, "m": 12, "A": [(3,0),(0,2),(0,4)], "B": [(0,6),(2,0),(4,0)],
         "d": 14, "k": 16, "name": "[[360,16,14]]"},
        {"ell": 30, "m": 6, "A": [(9,0),(0,1),(0,2)], "B": [(0,3),(25,0),(26,0)],
         "d": 24, "k": 12, "name": "[[360,12,24]] bravyi"},
    ]

    def _exponent_diff(terms1, terms2):
        """Count changed exponent positions between sorted term lists."""
        s1 = sorted(tuple(t) for t in terms1)
        s2 = sorted(tuple(t) for t in terms2)
        return sum(1 for a, b in zip(s1, s2) if a != b)

    def _describe_diff(terms1, terms2, poly_name):
        """Describe exponent changes between two polynomials."""
        s1 = sorted(tuple(t) for t in terms1)
        s2 = sorted(tuple(t) for t in terms2)
        changes = []
        for a, b in zip(s1, s2):
            if a != b:
                changes.append(f"{poly_name}: {a}->{b}")
        return changes

    # Codes with d reported (MILP-verified)
    codes_with_d = [
        r for r in all_results
        if r.get("d", 0) > 0 and r.get("fom", 0) > 0
    ]

    # Best code
    if codes_with_d:
        bc = max(codes_with_d, key=lambda r: r["fom"])
        tag = _stage_tag(bc)
        # Check if it beats any Bravyi baseline
        bravyi_foms = {144: 12.0, 288: 13.5, 360: 19.2}
        beat_bravyi = bc["fom"] > bravyi_foms.get(bc["n"], float("inf"))
        beat_tag = "  ← BEATS BRAVYI!" if beat_bravyi else ""
        artifacts["best_code"] = (
            f"[[{bc['n']},{bc['k']},{bc['d']}]] FOM={bc['fom']:.2f} "
            f"({tag}) at ({bc['ell']},{bc['m']}){beat_tag}\n"
            f"  A={bc['A_terms']}\n"
            f"  B={bc['B_terms']}"
        )

    if errors:
        artifacts["errors"] = "\n".join(errors[:5])

    # Top codes with d ≥ MIN_RELEVANT_D (the useful ones)
    relevant_codes = [r for r in codes_with_d if r.get("d", 0) >= MIN_RELEVANT_D]
    top5 = sorted(relevant_codes, key=lambda r: r.get("fom", 0), reverse=True)[:5]
    if top5:
        top5_lines = []
        for r in top5:
            tag = _stage_tag(r)
            tag_str = f" ({tag})" if tag else ""
            top5_lines.append(
                f"  [[{r['n']},{r['k']},{r['d']}]] FOM={r['fom']:.1f}"
                f"{tag_str} ({r['ell']},{r['m']})"
            )
        artifacts["top_codes"] = "Top codes with d>=" + str(MIN_RELEVANT_D) + ":\n" + "\n".join(top5_lines)

    # Low-d summary (to teach the LLM what's bad)
    low_d_count = sum(1 for r in codes_with_d if r.get("d", 0) < MIN_RELEVANT_D)
    if low_d_count > 0:
        max_k_low_d = max(
            (r.get("k", 0) for r in codes_with_d if r.get("d", 0) < MIN_RELEVANT_D),
            default=0,
        )
        artifacts["low_d_warning"] = (
            f"{low_d_count} codes with d<{MIN_RELEVANT_D} (max k={max_k_low_d}) -- "
            f"these DON'T count toward score. "
            f"Avoid univariate (A=f(y),B=g(x)) and self-dual (A=B)."
        )

    # Distance gradient analysis: compare MILP results to reference codes
    # to show the LLM how exponent changes affect d.
    gradient_lines = []
    for ref in _REFERENCE_CODES:
        ref_lattice = (ref["ell"], ref["m"])
        sorted(tuple(t) for t in ref["A"])
        sorted(tuple(t) for t in ref["B"])
        # Find MILP results at the same lattice that are close perturbations
        perturbations = []
        for r in codes_with_d:
            if (r["ell"], r["m"]) != ref_lattice:
                continue
            diff_a = _exponent_diff(r["A_terms"], ref["A"])
            diff_b = _exponent_diff(r["B_terms"], ref["B"])
            total_diff = diff_a + diff_b
            if 0 < total_diff <= 3:  # close perturbation, not identical
                changes = _describe_diff(ref["A"], r["A_terms"], "A")
                changes += _describe_diff(ref["B"], r["B_terms"], "B")
                d_delta = r["d"] - ref["d"]
                sign = "+" if d_delta > 0 else ""
                tag = _stage_tag(r)
                perturbations.append((
                    r["d"], d_delta, r.get("k", 0),
                    f"    d={r['d']} ({tag}) k={r['k']} [{', '.join(changes)}] "
                    f"d_change={sign}{d_delta}"
                ))
        if perturbations:
            # Sort: best d first
            perturbations.sort(key=lambda x: -x[0])
            gradient_lines.append(f"  Perturbations of {ref['name']} (d={ref['d']}, k={ref['k']}):")
            for _, _, _, line in perturbations[:5]:  # top 5
                gradient_lines.append(line)

    if gradient_lines:
        artifacts["distance_gradients"] = (
            "Distance gradients (how exponent changes affect d):\n"
            + "\n".join(gradient_lines)
        )

    # Per-lattice breakdown
    lattice_lines = []
    for key in sorted(per_lattice_best.keys()):
        lattice_lines.append(
            f"  ({key[0]},{key[1]}): best FOM={per_lattice_best[key]:.1f}"
        )

    artifacts["summary"] = (
        f"Evaluated {total_candidates} candidates across "
        f"{len(STAGE2_LATTICES_MILP)} lattices.\n"
        f"MILP verified: {len(milp_tasks)} codes ({milp_skipped_low_d_symp} "
        f"skipped by symplectic pre-filter d_symp<={MILP_EARLY_STOP}).\n"
        f"Valid codes (k>0): {len(valid)}\n"
        f"Codes with d>={MIN_RELEVANT_D}: {len(relevant_codes)} (these count toward score)\n"
        f"Best FOM: {best_fom:.2f}\n"
        f"Combined score: {combined:.1f} = sum of best FOM per lattice (d>={MIN_RELEVANT_D} only)\n"
        f"Per-lattice breakdown:\n" + "\n".join(lattice_lines)
    )

    # ── Phase 6: Persistence ───────────────────────────────────────
    # Individual codes are already saved to JSONL by evaluate_milp_parallel.
    # Also save to discovered_codes.json and pareto_front.json for compat.
    to_save = [
        r for r in all_results
        if r.get("fom", 0) > 6.0 and r.get("d", 0) > 0
    ]
    if to_save:
        best_to_save = max(to_save, key=lambda r: r["fom"])
        try:
            save_code(best_to_save)
            update_pareto_front(to_save)
        except Exception:
            pass

    _write_metrics_jsonl({
        "best_fom": best_fom,
        "mean_fom": mean_fom,
        "num_valid": len(valid),
        "num_high_k": len(high_k_codes),
        "lattices_with_high_k": lattices_with_high_k,
        "best_encoding_rate": max((r.get("encoding_rate", 0) for r in valid), default=0),
        "num_above_6": num_above_6,
        "num_above_12": num_above_12,
        "total_candidates": total_candidates,
        "all_results": all_results,
    })

    # MAP-Elites features from best MILP-verified code
    milp_best = max(
        (r for r in all_results if r.get("d", 0) > 0),
        key=lambda r: r.get("fom", 0), default=None,
    )
    if milp_best:
        milp_tc = _count_terms(milp_best["A_terms"], milp_best["B_terms"])
        milp_pattern = _classify_pattern(milp_best["A_terms"], milp_best["B_terms"])
    else:
        milp_tc = 0.0
        milp_pattern = 0.0

    result = {
        "combined_score": combined,
        "best_fom": best_fom,
        "mean_fom": mean_fom,
        "num_valid": float(len(valid)),
        "num_high_k": float(len(high_k_codes)),
        "lattices_with_high_k": float(lattices_with_high_k),
        "best_encoding_rate": max((r.get("encoding_rate", 0) for r in valid), default=0),
        "num_above_6": float(num_above_6),
        "num_above_12": float(num_above_12),
        "total_candidates": float(total_candidates),
        "term_count": milp_tc,
        "pattern_type": milp_pattern,
    }

    try:
        from openevolve.evaluation_result import EvaluationResult
        return EvaluationResult(metrics=result, artifacts=artifacts)
    except ImportError:
        return result


def evaluate(program_path: str) -> dict:
    """Full evaluation (fallback when cascade is disabled)."""
    return evaluate_stage2(program_path)


def _write_metrics_jsonl(metrics: dict) -> None:
    """Append evaluation metrics to a shared JSONL file.

    This is called from subprocess workers where wandb.run is None.
    The main process reads this file and syncs to W&B.
    """
    import json
    import time

    metrics_dir = Path(_PROJECT_ROOT) / "results"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = metrics_dir / "evolution_metrics.jsonl"

    # Build per-lattice best FOM for detailed tracking
    per_lattice: dict[tuple[int, int], float] = {}
    for r in metrics.get("all_results", []):
        d_raw = r.get("d", 0)
        k = r.get("k", 0)
        n = r.get("n", 0)
        if k > 0 and n > 0 and d_raw > 0:
            fom = k * d_raw * d_raw / n
            key = (r["ell"], r["m"])
            per_lattice[key] = max(per_lattice.get(key, 0.0), fom)

    record = {
        "timestamp": time.time(),
        "best_fom": metrics.get("best_fom", 0),
        "mean_fom": metrics.get("mean_fom", 0),
        "num_valid": metrics.get("num_valid", 0),
        "num_high_k": metrics.get("num_high_k", 0),
        "lattices_with_high_k": metrics.get("lattices_with_high_k", 0),
        "best_encoding_rate": metrics.get("best_encoding_rate", 0),
        "num_above_6": metrics.get("num_above_6", 0),
        "num_above_12": metrics.get("num_above_12", 0),
        "total_candidates": metrics.get("total_candidates", 0),
        "per_lattice_best_fom": {
            f"{k[0]}x{k[1]}": v for k, v in per_lattice.items()
        },
    }

    try:
        with open(metrics_file, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass  # Don't fail evaluation over metrics logging
