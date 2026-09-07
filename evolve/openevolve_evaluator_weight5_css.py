"""OpenEvolve evaluator adapter for the weight-5 CSS campaign (Phase B2).

Structurally mirrors ``evolve/openevolve_evaluator.py`` (the generic CSS
evaluator) -- same two-stage cascade, same ``gates.*()``/
``candidate_selection.*()`` wiring, same built-in MAP-Elites metric keys
(``combined_score``, ``best_fom``, ``num_valid``, ``lattices_with_high_k``,
``num_high_k``, ``term_count``, ``pattern_type``, ...).  Two differences:

1. **Weight-5 enforcement** (Phase A2) -- every candidate is checked with
   ``evaluation.weight_enforcement.css_weight_ok`` (the "2+3" canonical
   split) *and* ``evaluation.bb_code.validate_terms`` (exponent range, no
   duplicate monomials) BEFORE construction.  This is enforced by the
   evaluator itself, independent of the seed: an evolved/mutated
   ``generate_candidates`` could emit non-weight-5 candidates, and they are
   silently rejected here exactly like an invalid-terms rejection, never
   trusted from the seed alone.

2. **Selectable lattice profile** (Phase B2) -- ``QCODE_LATTICE_PROFILE``
   (``"small"`` default or ``"large"``) selects between two lattice sets.
   Read fresh via :func:`_lattice_profile` on every call (never cached at
   import time), so :func:`evaluate_stage1`/:func:`evaluate_stage2` always
   see the live environment.  A third, optional ``BOUNDARY_LATTICES`` set
   is defined for ad hoc exploration but not wired into either stage by
   default.

3. **Connectivity gate** -- direct-sum replications are rejected before
   construction using the exact translation-subgroup test in
   :mod:`evaluation.connectivity`.  The fail-closed default is ``reject``;
   ``QCODE_WEIGHT5_CONNECTIVITY_POLICY=allow`` is reserved for deliberate
   reproduction of campaigns that predated the connectivity audit.

Two-stage cascade
------------------
**Stage 1** -- Quick k-only screening on 3 small lattices per profile.
Programs must produce valid weight-5 codes (k > 0) at ALL stage-1 lattices
to advance past the cascade threshold.

**Stage 2** -- Full evaluation with BP-OSD distance estimation on 3
lattices per profile.  Combined score is the sum of the best *credible*
FOM per lattice (same d/sqrt(n) trust filter as the generic evaluator).
"""

from __future__ import annotations

import importlib.util
import logging
import math
import multiprocessing
import os
import queue
import sys
from pathlib import Path

# Ensure the project root is on sys.path so we can import evaluation.*
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from evaluation import candidate_selection, gates
from evaluation.bb_code import validate_terms
from evaluation.connectivity import bicycle_translation_is_connected
from evaluation.distance_schema import build_v2_record
from evaluation.evaluator import (
    evaluate_batch,
    DISTANCE_UNTRUST_RATIO,
)
from evaluation.pareto_v2 import update_pareto_front_v2
from evaluation.results import save_code, update_pareto_front
from evaluation.weight_enforcement import css_weight_ok

# Reuse the generic CSS evaluator's structural-pattern classifier,
# term-count feature, and Phase E1/E2 provenance helpers exactly, rather
# than reimplementing them -- Phase B3 calls for the SAME built-in
# MAP-Elites dimensions as the generic evaluator, not new custom ones, and
# the provenance-stamping logic (Phase E) is campaign-agnostic (operates
# purely on result-dict keys), so it should be shared, not duplicated a
# third time.
from evolve.openevolve_evaluator import (
    _classify_pattern,
    _count_terms,
    _emit_discovery_events,
    _stamp_provenance,
    file_content_hash,
)
from evolve.seed_solution_weight5_css import (
    _safety_net_candidates as _css_safety_net_candidates,
)
from evolve._weight5_css_generate_worker import generate_candidates_worker

# Per-model attribution: install here so it is active in each spawn worker
# (OpenEvolve imports this evaluator before generating). Defensive no-op if
# unavailable.
try:
    from evolve.model_attribution import install_tagging as _install_tagging
    _install_tagging()
except Exception:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lattice profiles (Phase B2). n = 2*ell*m always.
# ---------------------------------------------------------------------------
_STAGE1_SMALL = [(6, 6), (6, 9), (9, 8)]
_STAGE2_SMALL = [(6, 10), (9, 9), (10, 9)]
_STAGE1_LARGE = [(12, 9), (12, 12), (15, 12)]
_STAGE2_LARGE = [(16, 12), (15, 14), (18, 12)]

# Optional third lattice set for boundary/extension exploration. Not wired
# into evaluate_stage1/evaluate_stage2 by default -- available for ad hoc
# scripts or a future stage that opts in explicitly.
BOUNDARY_LATTICES = [(20, 10), (21, 11), (17, 14), (22, 11), (19, 13)]


def _lattice_profile() -> str:
    """Read ``QCODE_LATTICE_PROFILE`` fresh on every call (never cached at
    import time). ``"small"`` (default) or ``"large"``."""
    return os.environ.get("QCODE_LATTICE_PROFILE", "small")


def _stage1_lattices() -> list[tuple[int, int]]:
    return _STAGE1_LARGE if _lattice_profile() == "large" else _STAGE1_SMALL


def _stage2_lattices() -> list[tuple[int, int]]:
    return _STAGE2_LARGE if _lattice_profile() == "large" else _STAGE2_SMALL


def _generate_timeout_seconds() -> int:
    """Read ``QCODE_GENERATE_TIMEOUT_SECONDS`` fresh on every call (never
    cached at import time). Hard wall-clock bound, enforced by killing a
    subprocess (see ``_generate_candidates_bounded``), on a single evolved
    ``generate_candidates(ell, m)`` call."""
    return int(os.environ.get("QCODE_GENERATE_TIMEOUT_SECONDS", "60"))


# ---------------------------------------------------------------------------
# Phase A2: weight-5 enforcement, independent of the seed.
# ---------------------------------------------------------------------------
def _weight5_valid(ell: int, m: int, A_terms, B_terms) -> bool:
    """Reject a candidate BEFORE construction unless both term lists are
    individually valid for this lattice (exponent range, no duplicate
    monomials -- the same checks ``evaluate_candidate`` performs internally
    via ``evaluation.bb_code.validate_terms``) AND the pair satisfies the
    campaign's weight-5 "2+3" canonical split
    (``evaluation.weight_enforcement.css_weight_ok``).
    """
    try:
        validate_terms(ell, m, A_terms, "A")
        validate_terms(ell, m, B_terms, "B")
    except ValueError:
        return False
    return css_weight_ok(A_terms, B_terms)


def _filter_weight5(ell: int, m: int, candidates: list) -> tuple[list, int]:
    """Filter ``candidates`` (list of (A_terms, B_terms)-shaped items) down
    to weight-5-valid pairs. Returns (filtered, rejected_count). Malformed
    entries (not a 2-tuple) count as rejected, same as an invalid-terms
    rejection."""
    valid = []
    rejected = 0
    for cand in candidates:
        if not candidate_selection.has_arity(cand, 2):
            rejected += 1
            continue
        A_terms, B_terms = cand
        if _weight5_valid(ell, m, A_terms, B_terms):
            valid.append((A_terms, B_terms))
        else:
            rejected += 1
    return valid, rejected


def _filter_connected(ell: int, m: int, candidates: list) -> tuple[list, int]:
    """Apply the future-run direct-sum gate to validated CSS candidates.

    The caller runs :func:`_filter_weight5` first, so malformed exponents
    have already been rejected.  ``allow`` is an explicit legacy
    reproduction escape hatch; the default ``reject`` policy is fail-closed.
    """
    if gates.weight5_connectivity_policy() == "allow":
        return list(candidates), 0

    connected = []
    rejected = 0
    for A_terms, B_terms in candidates:
        if bicycle_translation_is_connected(
            ell, m, A_terms, B_terms
        ):
            connected.append((A_terms, B_terms))
        else:
            rejected += 1
    return connected, rejected


def _is_credible(result: dict) -> bool:
    """Whether ``result`` is a real, non-rejected candidate: k > 0 AND not
    stage="k_low" (the cascade's rejected-but-k>0 stage; see
    evaluation/evaluator.py). Used everywhere a bare ``k > 0`` check would
    otherwise let cascade-rejected candidates leak into promising-candidate
    selection or aggregate/Stage1/Stage2 metrics (round-2 finding #4 and
    round-3's follow-on leaks of the same distinction)."""
    return result.get("k", 0) > 0 and result.get("stage") != "k_low"


def _structural_feedback(result: dict) -> str:
    """Structural feedback string for a code with d >= 4 (same shape as
    the generic evaluator's, since the underlying result dict has the
    same keys)."""
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
    return "\n  ".join(lines)


def _log_code_jsonl(result: dict, run_name: str | None = None) -> None:
    """Append a code result to the run-specific all_codes jsonl file.

    Separate filename from the generic/ansatz campaigns
    (``all_codes_weight5_css.jsonl``) so concurrent campaigns sharing a
    run-name/output directory don't interleave records.
    """
    import json
    import time

    d = result.get("d", 0)
    if d <= 0:
        return

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
    log_file = log_dir / "all_codes_weight5_css.jsonl"

    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError:
        pass  # Don't fail evaluation over logging


def _error_result(error: str) -> dict:
    """Error result including every MAP-Elites feature dimension."""
    return {
        "combined_score": 0.0,
        "error": error,
        "lattices_with_high_k": 0.0,
        "num_high_k": 0.0,
        "term_count": 0.0,
        "pattern_type": 0.0,
        "connectivity_rejected": 0.0,
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


def _generate_candidates_bounded(
    program_path: str, ell: int, m: int, timeout_seconds: int
) -> tuple[list | None, str | None]:
    """Call ``generate_candidates(ell, m)`` in an isolated subprocess with a
    hard wall-clock deadline.

    An evolved ``generate_fn`` can loop indefinitely or blow up
    combinatorially. OpenEvolve's own per-call timeout
    (``asyncio.wait_for`` around ``run_in_executor``) only stops *waiting*
    for such a call -- the underlying thread keeps running forever, leaking
    into the (reused) worker process and causing cumulative GIL contention
    across every later call in that process, which is what actually
    happened in the weight5_css_large_v1 run: throughput degraded steadily
    and the worker processes became unresponsive even to SIGTERM. A
    subprocess boundary is required because ``.terminate()``/``.kill()`` on
    a genuinely separate OS process is unconditional, unlike a thread, which
    cannot be forcibly stopped. "spawn" (not the ambient default context) is
    used deliberately: the calling process may already have background
    threads from ``run_in_executor``, and forking a multi-threaded process
    is unsafe/deadlock-prone.

    Returns ``(candidates, None)`` on success or ``(None, error_message)``.

    Drains ``result_queue`` (via ``get(timeout=...)``) *before* joining the
    process, not after: a child that ``put()``s a payload larger than the
    pipe's OS buffer blocks inside its feeder thread until the parent reads,
    so joining first would deadlock forever on a large-but-otherwise-healthy
    candidate list instead of merely timing out on a genuinely stuck one.
    """
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=generate_candidates_worker,
        args=(program_path, ell, m, result_queue),
    )
    proc.start()
    try:
        status, payload = result_queue.get(timeout=timeout_seconds)
    except queue.Empty:
        status, payload = None, None
    finally:
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)

    if status is None:
        return None, f"generate_candidates timed out after {timeout_seconds}s"
    if status == "error":
        return None, payload
    return payload, None


def _run_evaluation(
    generate_fn,
    lattices: list[tuple[int, int]],
    quick: bool = False,
    quick_trials: int = 100,
    refine_trials: int = 500,
    max_distance_per_lattice: int = 10,
    run_name: str | None = None,
    source_hash: str | None = None,
    program_path: str | None = None,
) -> dict:
    """Run evaluation across lattices and compute aggregate metrics.

    Mirrors the generic CSS evaluator's ``_run_evaluation``: quick=True
    does a single k-only pass; quick=False does a two-pass
    quick-screen-then-distance-on-top-candidates pass. The only addition
    is the Phase A2 weight-5 filter and the translation-connectivity gate
    applied to every lattice's raw candidate list before the Phase A5
    pre-build cap.

    ``source_hash`` (Phase E1): passed through to :func:`_stamp_provenance`
    below so every result in ``all_results`` carries it -- see the generic
    CSS evaluator's ``_run_evaluation`` docstring for the full rationale.

    ``program_path`` (optional): when given, each per-lattice
    ``generate_fn(ell, m)`` call is isolated in a fresh subprocess with a
    hard wall-clock deadline (see :func:`_generate_candidates_bounded`)
    instead of being called in-process. ``evaluate_stage1``/
    ``evaluate_stage2`` always pass this, since their ``generate_fn`` is an
    untrusted, evolved program that can loop indefinitely or blow up
    combinatorially -- a thread-based timeout can only stop *waiting* for
    such a call, not the call itself, which then leaks into and eventually
    stalls the worker process. Callers that pass a trusted, in-repo
    ``generate_fn`` directly (e.g. tests) omit it and get the original,
    unisolated in-process call.
    """
    all_results = []
    total_candidates = 0
    total_weight_rejected = 0
    total_connectivity_rejected = 0
    errors = []
    selection_metrics: list[dict] = []

    for ell, m in lattices:
        try:
            if program_path is not None:
                candidates, gen_error = _generate_candidates_bounded(
                    program_path, ell, m, _generate_timeout_seconds()
                )
                if gen_error is not None:
                    errors.append(f"({ell},{m}): {gen_error}")
                    continue
            else:
                candidates = generate_fn(ell, m)
            if not isinstance(candidates, list):
                errors.append(f"({ell},{m}): generate_candidates returned {type(candidates)}, not list")
                continue

            raw_count = len(candidates)
            total_candidates += raw_count

            # De-duplicate before any downstream counting/capping -- an
            # evolved generate_fn is not guaranteed to dedup internally the
            # way the seed's own `seen`-set strategies do, and duplicates
            # would otherwise be built/scored multiple times whenever
            # raw_count stays under the pre-build cap (the cap's own
            # internal dedup never runs in that case).
            candidates, _ = candidate_selection.dedup_candidates(candidates)

            # Phase A2: weight-5 enforcement BEFORE construction, done by
            # the evaluator itself (not merely trusted from the seed).
            candidates, weight_rejected = _filter_weight5(ell, m, candidates)
            total_weight_rejected += weight_rejected
            weight_ok_count = len(candidates)

            # Reviewer-M1 guard for future runs: a disconnected translation
            # presentation is a replicated direct sum, so it must not consume
            # construction/distance budget or contribute fitness.  This runs
            # before the cap because the subgroup test is construction-free.
            candidates, connectivity_rejected = _filter_connected(
                ell, m, candidates
            )
            total_connectivity_rejected += connectivity_rejected
            gate_ok_count = len(candidates)

            # Phase A5: pre-build candidate cap via deterministic
            # hash-stratified selection instead of positional truncation.
            # generate_fn returns a flat list with no strategy
            # attribution, so _classify_pattern's structural niche
            # doubles as the stratification key (same proxy the generic
            # CSS evaluator uses).
            max_build = gates.max_build_candidates_css()
            # The fixed historical safety net is subject to exactly the same
            # policy; exempting it would reintroduce disconnected candidates
            # through the post-cap reinsertion path below. Also run it
            # through _filter_weight5 first, same as `candidates` above:
            # _filter_connected here (unlike the PBB sibling's) does not
            # validate exponents itself and trusts its caller to have done
            # so, so an out-of-range or malformed safety-net entry would
            # otherwise reach bicycle_translation_is_connected directly and
            # raise instead of being rejected gracefully.
            safety_net_valid, _ = _filter_weight5(
                ell, m, _css_safety_net_candidates(ell, m)
            )
            safety_net, _ = _filter_connected(ell, m, safety_net_valid)
            sel_metrics = None
            if gate_ok_count > max_build:
                # Round-3 finding: reserve room for the safety net WITHIN
                # max_build rather than appending it after selection, which
                # could exceed the configured hard cap by up to
                # len(safety_net) and made sel_metrics's selected_count
                # inaccurate. Reserving the full len(safety_net) is
                # deliberately conservative (some entries may already
                # survive selection on their own) but keeps the final count
                # bounded by max_build unconditionally.
                run_seed = run_name or os.environ.get("QCODE_RUN_NAME", "default")
                by_strategy: dict[str, list] = {}
                for cand in candidates:
                    strategy = str(_classify_pattern(cand[0], cand[1]))
                    by_strategy.setdefault(strategy, []).append(cand)
                build_budget = max(0, max_build - len(safety_net))
                candidates, sel_metrics = candidate_selection.select_prebuild_candidates(
                    by_strategy,
                    run_seed=run_seed,
                    ell=ell, m=m,
                    max_total=build_budget,
                )
                errors.append(
                    f"({ell},{m}): {raw_count} raw, {weight_rejected} rejected by "
                    f"weight-5 enforcement, {connectivity_rejected} rejected by "
                    f"connectivity, {gate_ok_count} gate-passed, capped to "
                    f"{len(candidates)} via hash-stratified selection"
                )
            else:
                errors.append(
                    f"({ell},{m}): {raw_count} raw, {weight_rejected} rejected by "
                    f"weight-5 enforcement, {connectivity_rejected} rejected by "
                    f"connectivity, {gate_ok_count}/{weight_ok_count} weight-ok "
                    f"passed through uncapped"
                )

            # Finding round-2 #3: the pre-build cap's per-strategy
            # stratification has no dedicated bucket for the eligible
            # (connectivity-gate-passed) safety net
            # (it falls into whatever ordinary structural stratum
            # _classify_pattern assigns it), so it can be hash-sampled out
            # like any other candidate in a crowded stratum -- breaking
            # the "Stage 1 always has something to score" guarantee.
            # Re-insert any eligible seed safety-net candidates that didn't
            # survive selection, rather than keeping a second,
            # duplicated notion of "what the safety net looks like" in
            # sync with the seed (that duplication is exactly what caused
            # the PBB sibling's analogous bug).
            if safety_net:
                existing_keys = {
                    candidate_selection.canonical_candidate_key(
                        (cand[0], cand[1])
                    )
                    for cand in candidates
                    if candidate_selection.has_arity(cand, 2)
                }
                for A, B in safety_net:
                    key = candidate_selection.canonical_candidate_key((A, B))
                    if key not in existing_keys:
                        candidates.append((A, B))
                        existing_keys.add(key)

            if sel_metrics is not None:
                selection_metrics.append({"ell": ell, "m": m, **sel_metrics})

            if quick:
                results = evaluate_batch(
                    ell, m, candidates,
                    quick=True,
                    quick_trials=quick_trials,
                    fom_threshold_refine=gates.fom_threshold_refine(),
                    fom_threshold_exact=gates.fom_threshold_exact(),
                )
            else:
                # Two-pass: quick screen, then distance on top candidates,
                # selected via deterministic k-band-stratified hashing
                # (Phase A5) rather than sorting by an unverified FOM.
                quick_results = evaluate_batch(ell, m, candidates, quick=True)

                # Pass every credible (k>0, non-rejected) candidate through
                # -- select_postbuild_distance_candidates itself excludes
                # k<2 (outside every K_BANDS band) and stratifies
                # k_2_3/k_4_5/k_ge_6. A k>=8 prefilter here defeated that
                # stratification by never giving the k_2_3/k_4_5 bands
                # anything to select from (round-2 finding #1). Excluding
                # stage="k_low" too (round-3 finding) keeps cascade-rejected
                # candidates from consuming scarce k_2_3 quota only to fail
                # refinement immediately.
                promising = [r for r in quick_results if _is_credible(r)]
                run_seed = run_name or os.environ.get("QCODE_RUN_NAME", "default")
                top, _ = candidate_selection.select_postbuild_distance_candidates(
                    promising,
                    run_seed=run_seed,
                    ell=ell, m=m,
                    max_total=max_distance_per_lattice,
                )

                top_candidates = [
                    (r["A_terms"], r["B_terms"]) for r in top
                ]

                # BP-OSD refine. Stage 5 (exact distance) requires
                # SIGALRM, unavailable in OpenEvolve's worker
                # threads/processes, so it stays disabled by default via
                # fom_threshold_exact_or_disabled(); QCODE_FOM_THRESHOLD_EXACT
                # lets a campaign opt in from a worker environment known
                # to support it.
                results = evaluate_batch(
                    ell, m, top_candidates,
                    quick=False,
                    quick_trials=refine_trials,
                    fom_threshold_refine=gates.fom_threshold_refine(),
                    fom_threshold_exact=gates.fom_threshold_exact_or_disabled(),
                )
                # Include quick-only results for aggregate counting
                quick_only = [
                    r for r in quick_results if _is_credible(r)
                    and r not in top
                ]
                results.extend(quick_only)

            all_results.extend(results)

            for r in results:
                _log_code_jsonl(r, run_name=run_name)
        except Exception as e:
            errors.append(f"({ell},{m}): {type(e).__name__}: {e}")

    # Compute aggregate metrics using encoding rate (exact) and FOM (approximate).
    # _is_credible excludes stage="k_low": the cascade (evaluation/evaluator.py)
    # sets this when k is nonzero but below gates.min_k_threshold() -- a
    # rejected candidate, not a valid one, that a bare `k > 0` check would
    # otherwise count toward Stage-1 coverage and encoding-rate fallback
    # metrics (round-2 finding #4).
    valid = [r for r in all_results if _is_credible(r)]
    foms = [r.get("fom", 0.0) for r in valid if r.get("fom", 0.0) > 0]
    encoding_rates = [r.get("encoding_rate", 0.0) for r in valid]

    best_fom = max(foms) if foms else 0.0
    mean_fom = sum(foms) / len(foms) if foms else 0.0
    num_above_6 = sum(1 for f in foms if f >= 6.0)
    num_above_12 = sum(1 for f in foms if f >= 12.0)
    best_encoding_rate = max(encoding_rates) if encoding_rates else 0.0

    high_k_codes = [r for r in valid if r.get("k", 0) >= 8]
    lattices_with_high_k = len(set(
        (r["ell"], r["m"]) for r in high_k_codes
    ))

    best_code = None
    if all_results:
        best_result = max(all_results, key=lambda r: r.get("fom", 0.0))
        if best_result.get("fom", 0.0) > 0:
            best_code = best_result

    # Phase E1: stamp provenance onto every per-code result once, here.
    _stamp_provenance(all_results, source_hash, run_name=run_name)

    return {
        "best_fom": best_fom,
        "mean_fom": mean_fom,
        "num_valid": len(valid),
        "num_above_6": num_above_6,
        "num_above_12": num_above_12,
        "total_candidates": total_candidates,
        "total_weight_rejected": total_weight_rejected,
        "total_connectivity_rejected": total_connectivity_rejected,
        "best_encoding_rate": best_encoding_rate,
        "num_high_k": len(high_k_codes),
        "lattices_with_high_k": lattices_with_high_k,
        "best_code": best_code,
        "all_results": all_results,
        "errors": errors,
        "selection_metrics": selection_metrics,
    }


def evaluate_stage1(program_path: str) -> dict:
    """Stage 1: Quick screening on 3 small lattices (k-only, weight-5
    enforced, ~2-5s).

    Programs must produce valid weight-5 codes (k > 0) at ALL stage-1
    lattices (profile-selected via QCODE_LATTICE_PROFILE, read fresh) to
    advance past the cascade threshold.
    """
    try:
        generate_fn = _load_generate_candidates(program_path)
    except Exception as e:
        return _error_result(str(e))

    stage1_lattices = _stage1_lattices()
    metrics = _run_evaluation(
        generate_fn, stage1_lattices, quick=True, program_path=program_path
    )

    if metrics["total_candidates"] == 0:
        return {
            "combined_score": 0.0,
            "num_valid": 0.0,
            "total_candidates": 0.0,
            "lattices_with_high_k": 0.0,
            "num_high_k": 0.0,
            "term_count": 0.0,
            "pattern_type": 0.0,
            "connectivity_rejected": 0.0,
        }

    valid = [r for r in metrics.get("all_results", []) if _is_credible(r)]

    lattices_with_valid = set((r["ell"], r["m"]) for r in valid)
    lattice_coverage = len(lattices_with_valid) / len(stage1_lattices)

    if lattice_coverage < 1.0:
        score = len(lattices_with_valid) * 0.001
    else:
        best_rate = max(r.get("encoding_rate", 0.0) for r in valid)
        high_k_quality = math.log1p(metrics["num_high_k"]) / 10.0
        score = 0.1 + best_rate + high_k_quality

    best_valid = max(valid, key=lambda r: r.get("k", 0)) if valid else None
    if best_valid:
        best_tc = _count_terms(best_valid["A_terms"], best_valid["B_terms"])
        best_pattern = _classify_pattern(best_valid["A_terms"], best_valid["B_terms"])
    else:
        best_tc = 0.0
        best_pattern = 0.0

    if metrics["errors"]:
        logger.debug("Stage1 selection/errors: %s", metrics["errors"][:5])

    return {
        "combined_score": score,
        "num_valid": float(metrics["num_valid"]),
        "total_candidates": float(metrics["total_candidates"]),
        "lattices_with_high_k": float(metrics["lattices_with_high_k"]),
        "num_high_k": float(metrics["num_high_k"]),
        "term_count": best_tc,
        "pattern_type": best_pattern,
        "connectivity_rejected": float(metrics["total_connectivity_rejected"]),
    }


def evaluate_stage2(program_path: str) -> dict:
    """Stage 2: Full evaluation with BP-OSD distance estimation on 3
    lattices (profile-selected via QCODE_LATTICE_PROFILE, read fresh),
    weight-5 enforced (~30-60s)."""
    try:
        generate_fn = _load_generate_candidates(program_path)
    except Exception as e:
        return _error_result(str(e))

    source_hash = file_content_hash(program_path) if file_content_hash else None
    stage2_lattices = _stage2_lattices()
    metrics = _run_evaluation(
        generate_fn, stage2_lattices,
        quick=False, refine_trials=1000,
        source_hash=source_hash,
        program_path=program_path,
    )

    # --- Combined score --- (identical trust filter to the generic evaluator)
    TRUST_FULL = gates.save_trust_ratio()
    TRUST_NONE = DISTANCE_UNTRUST_RATIO
    best_fom = metrics["best_fom"]

    per_lattice_best: dict[tuple[int, int], float] = {}
    for r in metrics["all_results"]:
        d_raw = r.get("d", 0)
        k = r.get("k", 0)
        n = r.get("n", 0)
        if n <= 0 or not _is_credible(r):
            continue
        key = (r["ell"], r["m"])
        fallback = k / n
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
                alpha = (TRUST_NONE - ratio) / (TRUST_NONE - TRUST_FULL)
                credible_fom = alpha * raw_fom + (1 - alpha) * fallback
        per_lattice_best[key] = max(per_lattice_best.get(key, 0.0), credible_fom)

    combined = sum(per_lattice_best.values())

    # Build artifacts for LLM feedback.
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

    # Phase A2/A5 bookkeeping surfaced to the LLM/log, in addition to the
    # generic per-lattice build/distance errors.
    gate_summary = (
        f"Weight-5 enforcement rejected {metrics['total_weight_rejected']} "
        f"candidates and the connectivity gate rejected "
        f"{metrics['total_connectivity_rejected']} across "
        f"{len(stage2_lattices)} lattices (profile="
        f"{_lattice_profile()})."
    )
    if metrics["selection_metrics"]:
        sel_lines = [gate_summary]
        for sm in metrics["selection_metrics"]:
            sel_lines.append(
                f"  ({sm['ell']},{sm['m']}): raw={sm['raw_count']} "
                f"deduped={sm['deduped_count']} selected={sm['selected_count']} "
                f"rejected={sm['rejected_count']} per_stratum={sm['per_stratum']}"
            )
        artifacts["selection_metrics"] = "\n".join(sel_lines)
    else:
        artifacts.setdefault("errors", "")
        errors_combined = ([gate_summary] + metrics["errors"])[:6]
        artifacts["errors"] = "\n".join(errors_combined)
    if metrics["errors"] and "errors" not in artifacts:
        artifacts["errors"] = "\n".join(metrics["errors"][:5])

    top5 = sorted(credible_codes, key=lambda r: r.get("fom", 0), reverse=True)[:5]
    if top5:
        top5_lines = []
        for r in top5:
            top5_lines.append(
                f"  [[{r['n']},{r['k']},{r['d']}]] FOM={r['fom']:.1f} "
                f"rate={r.get('encoding_rate', 0):.3f} ({r['ell']},{r['m']})"
            )
        artifacts["top_codes"] = "\n".join(top5_lines)

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

    lattice_lines = []
    for key in sorted(per_lattice_best.keys()):
        lattice_lines.append(
            f"  ({key[0]},{key[1]}): credible FOM={per_lattice_best[key]:.1f}"
        )

    artifacts["summary"] = (
        f"Weight-5 CSS campaign (lattice profile={_lattice_profile()}).\n"
        f"Evaluated {metrics['total_candidates']} raw candidates "
        f"({metrics['total_weight_rejected']} rejected by weight-5 enforcement; "
        f"{metrics['total_connectivity_rejected']} rejected as disconnected) "
        f"across {len(stage2_lattices)} lattices.\n"
        f"Valid codes (k>0): {metrics['num_valid']}\n"
        f"High-k codes (k>=8): {metrics['num_high_k']}\n"
        f"Lattices with high-k: {metrics['lattices_with_high_k']}/{len(stage2_lattices)}\n"
        f"Best raw FOM (BP-OSD, 1000 trials): {best_fom:.2f}\n"
        f"Combined score: {combined:.1f} = sum of best credible FOM per lattice "
        f"(full trust d/sqrt(n) <= {TRUST_FULL}, soft decay to {TRUST_NONE})\n"
        f"Per-lattice breakdown:\n" + "\n".join(lattice_lines)
    )

    # Save only codes with trusted distances.
    credible_to_save = [
        r for r in metrics["all_results"]
        if r.get("fom", 0) > 0
        and r.get("d", 0) > 0
        and r.get("n", 0) > 0
        and r["d"] <= TRUST_FULL * math.sqrt(r["n"])
    ]
    if credible_to_save:
        best_credible = max(credible_to_save, key=lambda r: r["fom"])
        if best_credible["fom"] > gates.save_fom_threshold_css():
            try:
                save_code(best_credible)
                update_pareto_front(credible_to_save)
            except Exception:
                pass  # Don't fail evaluation over persistence
            # Phase D: schema-v2 records + three-tier Pareto archive,
            # additive alongside the legacy v1 save above (never replaces
            # it -- see evaluation/distance_schema.py module docstring).
            try:
                v2_records = [build_v2_record(r) for r in credible_to_save]
                update_pareto_front_v2(v2_records)
            except Exception:
                pass  # Don't fail evaluation over persistence
            # Phase E2: one discovery event per already-stamped result.
            _emit_discovery_events(credible_to_save)

    _write_metrics_jsonl(metrics)

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
        "connectivity_rejected": float(metrics["total_connectivity_rejected"]),
        "term_count": s2_tc,
        "pattern_type": s2_pattern,
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
    """Append evaluation metrics to the SAME shared JSONL file the generic
    and non-CSS evaluators use (``results/evolution_metrics.jsonl``), so
    ``run_evolution.py``'s ``WandbSyncer`` (which watches a single,
    campaign-agnostic path) picks up weight-5 records too."""
    import json
    import time

    metrics_dir = Path(_PROJECT_ROOT) / "results"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = metrics_dir / "evolution_metrics.jsonl"

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
        "total_weight_rejected": metrics.get("total_weight_rejected", 0),
        "total_connectivity_rejected": metrics.get(
            "total_connectivity_rejected", 0
        ),
        "lattice_profile": _lattice_profile(),
        "per_lattice_best_fom": {
            f"{k[0]}x{k[1]}": v for k, v in per_lattice.items()
        },
    }

    try:
        with open(metrics_file, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass  # Don't fail evaluation over metrics logging
