"""OpenEvolve evaluator adapter for the weight-5 non-CSS PBB campaign (Phase C).

Mirrors ``openevolve_evaluator_noncss.py``'s two-stage cascade structure
(``evaluate_stage1`` quick k-only screening, ``evaluate_stage2`` full
adaptive-distance evaluation), but adds the weight-5 campaign's own
independent enforcement layer on top of every candidate, regardless of
what the (possibly LLM-mutated) seed's ``generate_candidates`` produces:

1. **Backbone weight-5 "2+3" check** -- ``evaluation.weight_enforcement.
   css_weight_ok(A_terms, B_terms)``.
2. **CSS-calibration exclusion** -- ``C_terms == D_terms == []`` reduces to
   plain CSS; classified as ``"css_calibration"`` and excluded from the
   PBB survivor pool (not scored, not crashed on).
3. **Symbolic containment** -- ``evaluation.weight_enforcement.
   pbb_weight_ok`` (``supp(C) subseteq supp(A)``, ``supp(D) subseteq
   supp(B)``, row weight == 5) on the term lists.
4. **Build** -- ``evaluation.pbb_code.build_pbb_code``, catching the
   "Commutativity violated" ``ValueError`` (and any other build
   exception) as a rejected-not-crashed candidate.
5. **Matrix-level containment** -- re-derives A/B/C/D's circulant
   sub-blocks directly from the constructed code's symplectic matrix
   (``_extract_block1_matrices``) and cross-checks containment there too,
   independent of the symbolic check in (3). This guards against the
   ``_poly_to_matrix``/bare-sympy-expression divergence bug class
   documented in ``CLAUDE.md`` -- a bug in the term-list bookkeeping could
   pass step (3) while the actual matrices qLDPC built disagree.
6. **Connectivity gate** -- rejects translation-disconnected direct-sum
   replications using :mod:`evaluation.connectivity`.  The fail-closed
   default is ``reject``; ``QCODE_WEIGHT5_CONNECTIVITY_POLICY=allow`` is
   reserved for deliberate legacy-run reproduction.
7. **k-threshold filter** -- ``evaluation.gates.min_k_threshold_noncss()``.
8. **Deferred non-CSS gate** -- ``evaluation.noncss_gate.
   passes_noncss_gate``, run ONLY on candidates that already passed every
   filter above (Phase A4: "defer full pair until after weight/
   commutativity/k filters"). ``passes_noncss_gate`` itself runs the
   cheaper check first: ``verify_not_lc_css`` is a purely algebraic
   brute-force + GF(2) affine solve over term lists (no code construction
   needed beyond what already happened in step 4); ``is_equivalently_css``
   needs the constructed code object and does a parity/union-find style
   solve over its stabilizers -- strictly more expensive, and only run if
   ``verify_not_lc_css`` didn't already short-circuit the conjunction to
   False. Results are cached by canonical ``(ell, m, A, B, C, D)`` key in a
   module-level dict (plain dict -- fresh per process, no cross-run
   persistence needed).

Lattice profiles (``QCODE_LATTICE_PROFILE`` env var, read live on every
call -- never cached at import time, matching ``gates.py``'s "read live"
principle): "small" (default), "large", or "boundary". Same lattice
tables as the sibling weight-5 CSS campaign
(``openevolve_evaluator_weight5_css.py``, re-derived independently here):

    small  Stage 1: (6,6), (6,9), (9,8)
           Stage 2 adds: (6,10), (9,9), (10,9)
    large  Stage 1: (12,9), (12,12), (15,12)
           Stage 2 adds: (16,12), (15,14), (18,12)
    boundary (optional, flat list, not split into stage1/stage2):
           (20,10), (21,11), (17,14), (22,11), (19,13)
"""

from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from evaluation import candidate_selection, gates, noncss_gate, weight_enforcement
from evaluation.bb_code import validate_terms
from evaluation.connectivity import bicycle_translation_is_connected
from evaluation.distance_schema import build_v2_record
from evaluation.pareto_v2 import update_pareto_front_v2
from evaluation.pbb_code import build_pbb_code, get_pbb_params_fast
from evolve._noncss_distance_worker import distance_worker as _distance_worker
from evolve.seed_solution_weight5_pbb import (
    _safety_net_candidates as _pbb_safety_net_candidates,
)

# Per-model attribution: install here so it is active in each spawn worker
# (OpenEvolve imports this evaluator before generating). Defensive no-op if
# unavailable.
try:
    from evolve.model_attribution import (
        install_tagging as _install_tagging,
        peek_model as _peek_model,
        peek_program_id as _peek_program_id,
    )
    _install_tagging()
except Exception:
    _peek_model = lambda: None
    _peek_program_id = lambda: None

# Reuse the generic non-CSS evaluator's Phase E1/E2 provenance helpers
# exactly, rather than reimplementing them a third time -- this logic is
# campaign-agnostic (operates purely on result-dict keys).
from evolve.openevolve_evaluator_noncss import (
    _emit_discovery_events,
    _stamp_provenance,
    file_content_hash,
)

logger = logging.getLogger(__name__)

# Parallelism for distance estimation -- see openevolve_evaluator_noncss.py
# for the sizing rationale (all single-threaded, embarrassingly parallel).
_NUM_DISTANCE_WORKERS = max(1, min((os.cpu_count() or 4) // 2, 16))

# Max candidates to run distance estimation on, per lattice.
MAX_DISTANCE_PER_LATTICE = 10

# Module-level non-CSS gate cache: fresh per process, shared across all
# candidates/lattices/calls within a single evaluation run.
_GATE_CACHE: dict = {}

# ---------------------------------------------------------------------------
# Lattice profiles (Phase A1/C tables), independent of the CSS sibling.
# ---------------------------------------------------------------------------
_SMALL_STAGE1 = [(6, 6), (6, 9), (9, 8)]
_SMALL_STAGE2_EXTRA = [(6, 10), (9, 9), (10, 9)]
_LARGE_STAGE1 = [(12, 9), (12, 12), (15, 12)]
_LARGE_STAGE2_EXTRA = [(16, 12), (15, 14), (18, 12)]
_BOUNDARY = [(20, 10), (21, 11), (17, 14), (22, 11), (19, 13)]


def _lattice_profile() -> str:
    """Read QCODE_LATTICE_PROFILE live (never cached at import time)."""
    return os.environ.get("QCODE_LATTICE_PROFILE", "small").strip().lower()


def _stage1_lattices() -> list[tuple[int, int]]:
    profile = _lattice_profile()
    if profile == "large":
        return list(_LARGE_STAGE1)
    if profile == "boundary":
        return list(_BOUNDARY)
    return list(_SMALL_STAGE1)


def _stage2_lattices() -> list[tuple[int, int]]:
    profile = _lattice_profile()
    if profile == "large":
        return list(_LARGE_STAGE1) + list(_LARGE_STAGE2_EXTRA)
    if profile == "boundary":
        return list(_BOUNDARY)
    return list(_SMALL_STAGE1) + list(_SMALL_STAGE2_EXTRA)


def _classify_pbb_strategy(A_terms, B_terms, C_terms, D_terms) -> str:
    """Coarse generator-strategy proxy for pre-build hash stratification.

    Distinct from ``openevolve_evaluator_noncss.py``'s version: the
    (C, D)=([], []) case is labeled ``"css_calibration"`` rather than
    ``"css"``, matching the Phase C2 census methodology's terminology for
    the excluded CSS-equivalent case.

    This used to carry a dedicated ``"safety_net_full"`` stratum that
    pattern-matched the OLD (C == full A, D == full B) safety net shape.
    The current safety net (``seed_solution_weight5_pbb.py``'s
    ``_SAFETY_NET_RAW``) uses a different, partial-subset shape that this
    detection never matched (round-2 finding #2) -- so the safety net
    silently fell into "both" and could still be starved out by the
    pre-build cap, exactly the failure this stratum existed to prevent.
    Rather than re-encode the seed's exact current shape here a second
    time (the duplication that caused the drift), safety-net survival is
    now guaranteed directly in ``_run_evaluation`` by re-inserting any of
    the seed's own ``_safety_net_candidates(ell, m)`` that the cap
    dropped, instead of relying on stratification alone.
    """
    c_empty = not C_terms
    d_empty = not D_terms
    if c_empty and d_empty:
        return "css_calibration"
    if c_empty:
        return "d_only"
    if d_empty:
        return "c_only"
    return "both"


def _filter_connected(ell: int, m: int, candidates: list) -> tuple[list, int]:
    """Remove provably translation-disconnected PBB candidates pre-cap.

    Malformed candidates are preserved for :func:`_build_and_check` to assign
    its established validation reason.  Well-formed direct sums are removed
    before hash-stratified sampling so they cannot crowd connected candidates
    out of the finite construction budget.
    """
    if gates.weight5_connectivity_policy() == "allow":
        return list(candidates), 0

    connected_or_unvalidated = []
    rejected = 0
    for cand in candidates:
        if not candidate_selection.has_arity(cand, 4):
            connected_or_unvalidated.append(cand)
            continue
        A_terms, B_terms, C_terms, D_terms = cand
        try:
            validate_terms(ell, m, A_terms, "A")
            validate_terms(ell, m, B_terms, "B")
            validate_terms(ell, m, C_terms, "C", min_terms=0)
            validate_terms(ell, m, D_terms, "D", min_terms=0)
        except (TypeError, ValueError):
            # Preserve the downstream evaluator's more specific validation
            # category for malformed candidates.
            connected_or_unvalidated.append(cand)
            continue
        if (
            not weight_enforcement.css_weight_ok(A_terms, B_terms)
            or (not C_terms and not D_terms)
            or not weight_enforcement.pbb_weight_ok(
                A_terms, B_terms, C_terms, D_terms
            )
        ):
            # Preserve backbone/containment/CSS-calibration reject reasons.
            connected_or_unvalidated.append(cand)
            continue
        is_connected = bicycle_translation_is_connected(
            ell, m, A_terms, B_terms, C_terms, D_terms
        )
        if is_connected:
            connected_or_unvalidated.append(cand)
        else:
            rejected += 1
    return connected_or_unvalidated, rejected


def _trust_level_for_result(result: dict) -> str:
    """Return a publication-style trust level for a distance result."""
    if result.get("d_is_exact") or result.get("milp_exact"):
        return "EXACT"
    trust = result.get("trust_level")
    if trust:
        return trust
    d = result.get("d", 0)
    n = result.get("n", 0)
    ratio = d / math.sqrt(n) if d > 0 and n > 0 else 0.0
    if ratio < 1.5:
        return "TRUSTED"
    if ratio < gates.save_trust_ratio_noncss():
        return "PARTIAL"
    return "UNTRUSTED"


def _trust_multiplier(result: dict) -> float:
    """Discount speculative distance bounds during evolution."""
    trust = _trust_level_for_result(result)
    if trust in {"EXACT", "TRUSTED"}:
        return 1.0
    if trust == "PARTIAL":
        return 0.25
    return 0.0


def _scored_fom(result: dict) -> float:
    return result.get("fom", 0.0) * _trust_multiplier(result)


def _error_result(error: str) -> dict:
    """Return an error result with all required MAP-Elites features."""
    return {
        "combined_score": 0.0,
        "error": error,
        "lattices_with_high_k": 0.0,
        "num_high_k": 0.0,
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


def _extract_block1_matrices(
    code, ell: int, m: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Slice (mat_A, mat_B, mat_C, mat_D) directly out of the constructed
    code's symplectic matrix, matching ``build_pbb_code``'s layout::

        block1_x = hstack([mat_A, mat_B])   # rows [0:dim), cols [0:2*dim)
        block1_z = hstack([mat_C, mat_D])   # rows [0:dim), cols [2*dim:4*dim)
        top      = hstack([block1_x, block1_z])
        matrix   = vstack([top, bottom])

    Avoids calling the private ``_poly_to_matrix``/``bb.eval()`` path a
    second time -- this reads the ACTUAL matrix qLDPC built for this code,
    which is the whole point of the matrix-level containment cross-check
    (guards against the ``_poly_to_matrix`` bare-sympy-expression bug class
    in CLAUDE.md: a divergence there would show up here, not in the
    symbolic term-list check).
    """
    dim = ell * m
    mat = np.asarray(code.matrix, dtype=int) % 2
    mat_A = mat[:dim, :dim]
    mat_B = mat[:dim, dim:2 * dim]
    mat_C = mat[:dim, 2 * dim:3 * dim]
    mat_D = mat[:dim, 3 * dim:4 * dim]
    return mat_A, mat_B, mat_C, mat_D


def _build_and_check(
    ell: int, m: int,
    A_terms: list, B_terms: list,
    C_terms: list, D_terms: list,
) -> tuple[dict | None, str | None]:
    """Full weight-5 PBB enforcement pipeline for one candidate.

    Returns ``(info, reject_reason)``: ``info`` is a result dict on
    success, ``None`` on rejection; ``reject_reason`` is a short label
    (``None`` on success) for diagnostics -- never raises.
    """
    # Validate exponent ranges (and reject degenerate same-lattice-shift
    # duplicates, e.g. (0,0) alongside (ell,0)) BEFORE any symbolic weight
    # check. css_weight_ok/pbb_weight_ok compare raw (int,int) tuples with
    # no modular reduction, so an out-of-range or aliased exponent can pass
    # them symbolically while collapsing to a lower-weight matrix -- the
    # CSS evaluator's analogous _weight5_valid already guards this via the
    # same validate_terms call; this mirrors it here (see CLAUDE.md's
    # `_poly_to_matrix` bug-class note for the underlying symbolic/matrix
    # divergence risk).
    try:
        validate_terms(ell, m, A_terms, "A")
        validate_terms(ell, m, B_terms, "B")
        validate_terms(ell, m, C_terms, "C", min_terms=0)
        validate_terms(ell, m, D_terms, "D", min_terms=0)
    except ValueError:
        return None, "exponent_range_or_duplicate"

    if not weight_enforcement.css_weight_ok(A_terms, B_terms):
        return None, "backbone_weight"

    if not C_terms and not D_terms:
        # Plain CSS: a calibration case (Phase C2), not a PBB candidate.
        return None, "css_calibration"

    if not weight_enforcement.pbb_weight_ok(A_terms, B_terms, C_terms, D_terms):
        return None, "containment_symbolic"

    # Reviewer-M1 guard for future runs.  The translation-subgroup test is
    # exact for the supplied BB/PBB presentation and cheap enough to run
    # before constructing a qLDPC object or computing its rank.
    try:
        connectivity_policy = gates.weight5_connectivity_policy()
    except ValueError as exc:
        logger.error("Invalid weight-five connectivity policy: %s", exc)
        return None, "connectivity_policy_error"
    if (
        connectivity_policy == "reject"
        and not bicycle_translation_is_connected(
            ell, m, A_terms, B_terms, C_terms, D_terms
        )
    ):
        return None, "disconnected"

    try:
        code = build_pbb_code(ell, m, A_terms, B_terms, C_terms, D_terms)
    except Exception as e:
        logger.debug(f"Build failed ({ell},{m}): {e}")
        return None, f"build_error:{type(e).__name__}"

    try:
        mat_A, mat_B, mat_C, mat_D = _extract_block1_matrices(code, ell, m)
    except Exception as e:
        logger.debug(f"Matrix extraction failed ({ell},{m}): {e}")
        return None, f"matrix_extract_error:{type(e).__name__}"

    if C_terms and not weight_enforcement.matrix_support_contained(mat_A, mat_C):
        return None, "containment_matrix_C"
    if D_terms and not weight_enforcement.matrix_support_contained(mat_B, mat_D):
        return None, "containment_matrix_D"

    if weight_enforcement.pbb_row_weight(A_terms, B_terms, C_terms, D_terms) != 5:
        return None, "row_weight"

    n, k = get_pbb_params_fast(code)
    info = {
        "code": code,
        "ell": ell, "m": m, "n": n, "k": k,
        "A_terms": A_terms, "B_terms": B_terms,
        "C_terms": C_terms, "D_terms": D_terms,
        "encoding_rate": k / n if n > 0 else 0.0,
    }
    return info, None


def _log_code_jsonl(result: dict, run_name: str | None = None) -> None:
    """Append a code result to the run-specific weight5-PBB JSONL file.

    Distinct filename (all_codes_weight5_pbb.jsonl) so this campaign's log
    never collides with the existing non-CSS campaign's
    all_codes_noncss.jsonl.
    """
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
        "C_terms": result.get("C_terms"),
        "D_terms": result.get("D_terms"),
        "n": result.get("n"),
        "k": result.get("k"),
        "d": d,
        "fom": result.get("fom", 0.0),
        "d_method": result.get("d_method"),
        "d_milp": result.get("d_milp"),
        "d_bposd": result.get("d_bposd"),
        "milp_exact": result.get("milp_exact"),
        "run_id": result.get("run_id"),
        "program_id": result.get("program_id"),
        "generating_model": result.get("generating_model"),
        "model_alias": result.get("model_alias"),
        "source_hash": result.get("source_hash"),
        "code_key": result.get("code_key"),
        "timestamp": time.time(),
    }

    if run_name:
        log_dir = Path(_PROJECT_ROOT) / "results" / "evolution" / run_name
    else:
        log_dir = Path(_PROJECT_ROOT) / "results" / "evolution"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "all_codes_weight5_pbb.jsonl"

    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError:
        pass


def _persist_observed_results(
    results: list[dict],
    *,
    source_hash: str | None,
    run_name: str | None,
    program_id: str | None,
    generating_model: str | None,
) -> None:
    """Stamp and persist every positive-distance observation.

    ``program_id`` and ``generating_model`` are captured when evaluation
    starts.  OpenEvolve can time out while the evaluator thread continues,
    clearing its process-global attribution context before distance workers
    finish; seeding those captured values here preserves the correct identity
    in that case.  Discovery logging is intentionally independent of Pareto
    archive admission.
    """
    resolved_run_id = run_name or os.environ.get("QCODE_RUN_NAME")
    for result in results:
        result.setdefault("run_id", resolved_run_id)
        result.setdefault("program_id", program_id)
        result.setdefault("generating_model", generating_model)
        result.setdefault("model_alias", generating_model)

    _stamp_provenance(results, source_hash, run_name=resolved_run_id)

    observed = [result for result in results if result.get("d", 0) > 0]
    for result in observed:
        _log_code_jsonl(result, run_name=resolved_run_id)
    _emit_discovery_events(observed)


def _write_metrics_jsonl(metrics: dict) -> None:
    """Write evaluation metrics to the shared JSONL for W&B sync.

    Reuses the same file as the CSS and non-CSS evaluators
    (evolution_metrics.jsonl), matching their convention, so a single
    WandbSyncer picks up all three campaigns' records.
    """
    metrics_file = Path(_PROJECT_ROOT) / "results" / "evolution_metrics.jsonl"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "best_fom": metrics.get("best_fom", 0),
        "mean_fom": metrics.get("mean_fom", 0),
        "num_valid": metrics.get("num_valid", 0),
        "num_high_k": metrics.get("num_high_k", 0),
        "lattices_with_high_k": metrics.get("lattices_with_high_k", 0),
        "best_encoding_rate": metrics.get("best_encoding_rate", 0),
        "total_candidates": metrics.get("total_candidates", 0),
        "total_connectivity_rejected": metrics.get(
            "total_connectivity_rejected", 0
        ),
        "timestamp": time.time(),
    }

    try:
        with open(metrics_file, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass


def _run_evaluation(
    generate_fn,
    lattices: list[tuple[int, int]],
    quick: bool = False,
    num_trials: int = 500,
    max_distance_per_lattice: int = MAX_DISTANCE_PER_LATTICE,
    run_name: str | None = None,
    source_hash: str | None = None,
    program_id: str | None = None,
    generating_model: str | None = None,
) -> dict:
    """Run evaluation across lattices and compute aggregate metrics.

    Pipeline per lattice: generate -> translation-connectivity gate ->
    pre-build hash-stratified cap (Phase A5) -> full weight-5/containment/
    connectivity/build/k filter
    (``_build_and_check``) -> deferred non-CSS gate (only on k-filtered
    survivors) -> post-build k-band-stratified distance selection (Phase
    A5) -> parallel distance estimation across all lattices.

    ``source_hash`` (Phase E1): passed through to :func:`_stamp_provenance`
    below so every result in ``all_results`` carries it.
    """
    # Capture identity before any long-running work.  The OpenEvolve timeout
    # path can clear the process-global attribution context while this worker
    # thread continues to finish its distance calculations.
    if program_id is None:
        program_id = _peek_program_id()
    if generating_model is None:
        generating_model = _peek_model()

    all_results = []
    total_candidates = 0
    errors = []
    reject_counts: dict[str, int] = {}

    per_lattice_top: list[tuple[tuple[int, int], list[dict]]] = []
    per_lattice_rest: list[tuple[tuple[int, int], list[dict]]] = []

    for ell, m in lattices:
        try:
            candidates = generate_fn(ell, m)
            if not isinstance(candidates, list):
                errors.append(f"({ell},{m}): returned {type(candidates)}, not list")
                continue

            total_candidates += len(candidates)

            # De-duplicate before any downstream counting/capping -- an
            # evolved generate_fn is not guaranteed to dedup internally the
            # way the seed's own `seen`-set strategies do, and duplicates
            # would otherwise be built/scored multiple times whenever
            # raw_count stays under the pre-build cap (the cap's own
            # internal dedup never runs in that case).
            candidates, _ = candidate_selection.dedup_candidates(candidates)

            # Apply the O(w^2) translation-subgroup gate before the sampling
            # cap.  Otherwise a large population of easy direct sums can
            # crowd connected presentations out of the construction budget.
            candidates, connectivity_rejected = _filter_connected(
                ell, m, candidates
            )
            if connectivity_rejected:
                reject_counts["disconnected"] = (
                    reject_counts.get("disconnected", 0)
                    + connectivity_rejected
                )

            max_build = gates.max_build_candidates_noncss()
            # Apply the same policy to injected fallback candidates so the
            # post-cap reinsertion path cannot bypass the gate.
            safety_net, _ = _filter_connected(
                ell, m, _pbb_safety_net_candidates(ell, m)
            )
            gate_ok_count = len(candidates)
            if gate_ok_count > max_build:
                # Round-3 finding: reserve room for the safety net WITHIN
                # max_build rather than appending it after selection, which
                # could exceed the configured hard cap by up to
                # len(safety_net). Deliberately conservative (reserves the
                # full count even though some entries may survive selection
                # on their own) but keeps the final count bounded by
                # max_build unconditionally.
                run_seed = run_name or os.environ.get("QCODE_RUN_NAME", "default")
                by_strategy: dict[str, list] = {}
                for cand in candidates:
                    if not candidate_selection.has_arity(cand, 4):
                        continue
                    strategy = _classify_pbb_strategy(*cand)
                    by_strategy.setdefault(strategy, []).append(cand)
                build_budget = max(0, max_build - len(safety_net))
                candidates, _ = candidate_selection.select_prebuild_candidates(
                    by_strategy,
                    run_seed=run_seed,
                    ell=ell, m=m,
                    max_total=build_budget,
                    # 2, not the default 1: a slightly larger floor for
                    # every populated stratum before falling back to
                    # proportional allocation. Safety-net survival itself
                    # is guaranteed explicitly below, not via this quota.
                    min_quota_per_strategy=2,
                )
                errors.append(
                    f"({ell},{m}): {gate_ok_count} connectivity-gate-passed "
                    f"candidates, capped to {max_build} via hash-stratified "
                    f"selection ({connectivity_rejected} disconnected rejected)"
                )

            # Round-2 finding #2: the pre-build cap's stratification had
            # no reliable way to detect the eligible (connectivity-gate-
            # passed) safety net's current (C, D)
            # shape (see _classify_pbb_strategy docstring), so it could be
            # hash-sampled out like any other candidate -- breaking the
            # "Stage 1 always has something to score" guarantee. Re-insert
            # any eligible seed safety-net candidates that didn't survive
            # selection, independent of strategy classification.
            if safety_net:
                existing_keys = {
                    candidate_selection.canonical_candidate_key(c)
                    for c in candidates
                    if candidate_selection.has_arity(c, 4)
                }
                for cand in safety_net:
                    key = candidate_selection.canonical_candidate_key(cand)
                    if key not in existing_keys:
                        candidates.append(cand)
                        existing_keys.add(key)

            # Build + defensive weight-5/containment/connectivity/k filter.
            built = []
            min_k = gates.min_k_threshold_noncss()
            for cand in candidates:
                if not candidate_selection.has_arity(cand, 4):
                    continue
                A_terms, B_terms, C_terms, D_terms = cand
                info, reason = _build_and_check(ell, m, A_terms, B_terms, C_terms, D_terms)
                if info is None:
                    reject_counts[reason] = reject_counts.get(reason, 0) + 1
                    continue
                if info["k"] < min_k:
                    reject_counts["k_below_threshold"] = (
                        reject_counts.get("k_below_threshold", 0) + 1
                    )
                    continue
                built.append(info)

            # Deferred non-CSS gate -- only run on candidates that already
            # passed weight/commutativity/k filters (Phase A4).
            gate_passed = []
            for info in built:
                try:
                    gate_result = noncss_gate.passes_noncss_gate(
                        ell, m,
                        info["A_terms"], info["B_terms"],
                        info["C_terms"], info["D_terms"],
                        info["code"],
                        cache=_GATE_CACHE,
                    )
                except Exception as e:
                    logger.debug(f"non-CSS gate failed ({ell},{m}): {e}")
                    reject_counts["gate_error"] = reject_counts.get("gate_error", 0) + 1
                    continue
                if gate_result["passes"]:
                    info["is_lc_css"] = gate_result["is_lc_css"]
                    info["is_css"] = gate_result["is_css"]
                    gate_passed.append(info)
                else:
                    reject_counts["gate_css_equivalent"] = (
                        reject_counts.get("gate_css_equivalent", 0) + 1
                    )
            built = gate_passed

            if quick:
                for info in built:
                    result = dict(info)
                    del result["code"]
                    result["d"] = 0
                    result["fom"] = 0.0
                    all_results.append(result)
            else:
                run_seed = run_name or os.environ.get("QCODE_RUN_NAME", "default")
                top, _ = candidate_selection.select_postbuild_distance_candidates(
                    built,
                    run_seed=run_seed,
                    ell=ell, m=m,
                    max_total=max_distance_per_lattice,
                )

                per_lattice_top.append(((ell, m), top))

                top_set = {id(x) for x in top}
                rest = []
                for info in built:
                    if id(info) not in top_set:
                        result = dict(info)
                        del result["code"]
                        result["d"] = 0
                        result["fom"] = 0.0
                        rest.append(result)
                per_lattice_rest.append(((ell, m), rest))

        except Exception as e:
            errors.append(f"({ell},{m}): {type(e).__name__}: {e}")

    # Phase 2: Parallel distance estimation across ALL lattices.
    if not quick and per_lattice_top:
        distance_tasks = []
        for (ell, m), top in per_lattice_top:
            for info in top:
                task = (
                    info["ell"], info["m"],
                    info["A_terms"], info["B_terms"],
                    info["C_terms"], info["D_terms"],
                    num_trials,
                )
                distance_tasks.append(task)

        num_workers = min(_NUM_DISTANCE_WORKERS, len(distance_tasks))
        logger.info(
            f"Running {len(distance_tasks)} distance tasks "
            f"with {num_workers} workers"
        )

        try:
            import multiprocessing as _mp
            _ctx = _mp.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=_ctx,
            ) as pool:
                futures = {
                    pool.submit(_distance_worker, task): i
                    for i, task in enumerate(distance_tasks)
                }
                distance_results = []
                for future in concurrent.futures.as_completed(futures, timeout=1200):
                    try:
                        distance_results.append(future.result())
                    except Exception as e_task:
                        idx = futures[future]
                        task = distance_tasks[idx]
                        logger.debug(
                            f"Distance worker crashed for ({task[0]},{task[1]}): {e_task}"
                        )
        except Exception as e:
            logger.warning(f"Parallel distance failed ({e}), falling back to sequential")
            distance_results = []
            for task in distance_tasks:
                try:
                    distance_results.append(_distance_worker(task))
                except Exception as e2:
                    logger.debug(f"Distance worker failed: {e2}")

        for result in distance_results:
            all_results.append(result)

        for (ell, m), rest in per_lattice_rest:
            all_results.extend(rest)

    valid = [r for r in all_results if r.get("k", 0) > 0]
    foms = [_scored_fom(r) for r in valid if _scored_fom(r) > 0]
    encoding_rates = [r.get("encoding_rate", 0.0) for r in valid]

    best_fom = max(foms) if foms else 0.0
    mean_fom = sum(foms) / len(foms) if foms else 0.0
    best_encoding_rate = max(encoding_rates) if encoding_rates else 0.0

    high_k_codes = [r for r in valid if r.get("k", 0) >= 8]
    lattices_with_high_k = len(set((r["ell"], r["m"]) for r in high_k_codes))

    best_code = None
    if all_results:
        best_result = max(all_results, key=_scored_fom)
        if _scored_fom(best_result) > 0:
            best_code = best_result

    # Phase E1/E2: provenance and discovery records are observation-level,
    # not conditional on crossing the archive FOM threshold.
    _persist_observed_results(
        all_results,
        source_hash=source_hash,
        run_name=run_name,
        program_id=program_id,
        generating_model=generating_model,
    )

    return {
        "best_fom": best_fom,
        "mean_fom": mean_fom,
        "num_valid": len(valid),
        "total_candidates": total_candidates,
        "best_encoding_rate": best_encoding_rate,
        "num_high_k": len(high_k_codes),
        "lattices_with_high_k": lattices_with_high_k,
        "best_code": best_code,
        "all_results": all_results,
        "errors": errors,
        "reject_counts": reject_counts,
        "total_connectivity_rejected": reject_counts.get("disconnected", 0),
    }


def evaluate_stage1(program_path: str) -> dict:
    """Stage 1: Quick screening on small lattices (k-only, ~5-30s).

    Programs must produce valid, non-CSS-gate-passing codes (k > 0) at
    ALL Stage-1 lattices for the current QCODE_LATTICE_PROFILE to advance
    past the cascade threshold.
    """
    program_id = _peek_program_id()
    generating_model = _peek_model()
    try:
        generate_fn = _load_generate_candidates(program_path)
    except Exception as e:
        return _error_result(str(e))

    stage1_lattices = _stage1_lattices()
    metrics = _run_evaluation(
        generate_fn,
        stage1_lattices,
        quick=True,
        program_id=program_id,
        generating_model=generating_model,
    )

    if metrics["total_candidates"] == 0:
        return {
            "combined_score": 0.0,
            "num_valid": 0.0,
            "total_candidates": 0.0,
            "lattices_with_high_k": 0.0,
            "num_high_k": 0.0,
            "connectivity_rejected": 0.0,
        }

    valid = [r for r in metrics.get("all_results", []) if r.get("k", 0) > 0]

    lattices_with_valid = set((r["ell"], r["m"]) for r in valid)
    lattice_coverage = len(lattices_with_valid) / len(stage1_lattices)

    if lattice_coverage < 1.0:
        score = len(lattices_with_valid) * 0.001
    else:
        best_rate = max(r.get("encoding_rate", 0.0) for r in valid)
        high_k_quality = math.log1p(metrics["num_high_k"]) / 10.0
        score = 0.1 + best_rate + high_k_quality

    return {
        "combined_score": score,
        "num_valid": float(metrics["num_valid"]),
        "total_candidates": float(metrics["total_candidates"]),
        "lattices_with_high_k": float(metrics["lattices_with_high_k"]),
        "num_high_k": float(metrics["num_high_k"]),
        "connectivity_rejected": float(
            metrics["total_connectivity_rejected"]
        ),
    }


def evaluate_stage2(program_path: str) -> dict:
    """Stage 2: Full evaluation with adaptive distance (~120-1200s).

    Runs across the Stage 1 + Stage 2 lattices for the current
    QCODE_LATTICE_PROFILE, with the same adaptive distance pipeline as the
    non-CSS campaign (hash-based exact, MILP symplectic, BP-OSD fallback).
    """
    program_id = _peek_program_id()
    generating_model = _peek_model()
    try:
        generate_fn = _load_generate_candidates(program_path)
    except Exception as e:
        return _error_result(str(e))

    source_hash = file_content_hash(program_path) if file_content_hash else None
    stage2_lattices = _stage2_lattices()
    metrics = _run_evaluation(
        generate_fn, stage2_lattices,
        quick=False, num_trials=1000,
        source_hash=source_hash,
        program_id=program_id,
        generating_model=generating_model,
    )

    min_relevant_d = gates.min_relevant_d_noncss()

    per_lattice_best: dict[tuple[int, int], float] = {}
    for r in metrics["all_results"]:
        d_raw = r.get("d", 0)
        k = r.get("k", 0)
        n = r.get("n", 0)
        if k <= 0 or n <= 0:
            continue
        key = (r["ell"], r["m"])

        if d_raw < min_relevant_d:
            fom = k / n * 0.1
        else:
            fom = _scored_fom(r)

        per_lattice_best[key] = max(per_lattice_best.get(key, 0.0), fom)

    combined = sum(per_lattice_best.values())
    best_fom = metrics["best_fom"]

    artifacts = {}
    credible_codes = []
    for r in metrics["all_results"]:
        d_val = r.get("d", 0)
        if d_val >= min_relevant_d and _trust_multiplier(r) > 0:
            credible_codes.append(r)

    if credible_codes:
        bc = max(credible_codes, key=lambda r: r.get("fom", 0.0))
        artifacts["best_code"] = (
            f"[[{bc['n']},{bc['k']},<={bc['d']}]] FOM={bc['fom']:.2f} "
            f"trust={_trust_level_for_result(bc)} "
            f"at ({bc['ell']},{bc['m']})\n"
            f"  A={bc['A_terms']} B={bc['B_terms']}\n"
            f"  C={bc['C_terms']} D={bc['D_terms']}"
        )
    elif metrics["best_code"]:
        bc = metrics["best_code"]
        artifacts["best_code"] = (
            f"[[{bc['n']},{bc['k']},<={bc.get('d', '?')}]] "
            f"at ({bc['ell']},{bc['m']})\n"
            f"  A={bc['A_terms']} B={bc['B_terms']}\n"
            f"  C={bc['C_terms']} D={bc['D_terms']}"
        )

    if metrics["errors"]:
        artifacts["errors"] = "\n".join(metrics["errors"][:5])
    if metrics["reject_counts"]:
        artifacts["reject_counts"] = "\n".join(
            f"  {reason}: {count}"
            for reason, count in sorted(
                metrics["reject_counts"].items(), key=lambda kv: -kv[1]
            )
        )

    top5 = sorted(credible_codes, key=_scored_fom, reverse=True)[:5]
    if top5:
        lines = []
        for r in top5:
            c_len = len(r.get("C_terms", []))
            d_len = len(r.get("D_terms", []))
            lines.append(
                f"  [[{r['n']},{r['k']},<={r['d']}]] FOM={r['fom']:.1f} ({r['ell']},{r['m']}) "
                f"trust={_trust_level_for_result(r)} |C|={c_len} |D|={d_len}"
            )
        artifacts["top_codes"] = "\n".join(lines)

    lattice_lines = []
    for key in sorted(per_lattice_best.keys()):
        lattice_lines.append(
            f"  ({key[0]},{key[1]}): credible FOM={per_lattice_best[key]:.1f}"
        )

    artifacts["summary"] = (
        f"Evaluated {metrics['total_candidates']} candidates across "
        f"{len(stage2_lattices)} lattices (profile={_lattice_profile()}).\n"
        f"Connectivity gate rejected "
        f"{metrics['total_connectivity_rejected']} disconnected candidates.\n"
        f"Valid non-CSS codes (k>0, gate-passed): {metrics['num_valid']}\n"
        f"High-k codes (k>=8): {metrics['num_high_k']}\n"
        f"Best FOM: {best_fom:.2f}\n"
        f"Combined score: {combined:.1f}\n"
        f"Per-lattice breakdown:\n" + "\n".join(lattice_lines)
    )

    credible_to_save = [
        r for r in metrics["all_results"]
        if r.get("fom", 0) > 0
        and r.get("d", 0) >= min_relevant_d
        and _trust_multiplier(r) > 0
    ]
    if credible_to_save:
        best_credible = max(credible_to_save, key=_scored_fom)
        if _scored_fom(best_credible) > gates.save_fom_threshold_noncss():
            try:
                from evaluation.results import save_code, update_pareto_front
                save_code(best_credible)
                update_pareto_front(credible_to_save)
            except Exception:
                pass  # Don't fail evaluation over persistence
            # Phase D: schema-v2 records + three-tier Pareto archive,
            # additive alongside the legacy v1 save above (never replaces
            # it -- see evaluation/distance_schema.py module docstring).
            # component_bounds stays None: it is CSS X/Z-only (CLAUDE.md),
            # not meaningful for a non-CSS PBB result.
            try:
                v2_records = [build_v2_record(r) for r in credible_to_save]
                update_pareto_front_v2(v2_records)
            except Exception:
                pass  # Don't fail evaluation over persistence

    _write_metrics_jsonl(metrics)

    result = {
        "combined_score": combined,
        "best_fom": best_fom,
        "mean_fom": metrics["mean_fom"],
        "num_valid": float(metrics["num_valid"]),
        "num_high_k": float(metrics["num_high_k"]),
        "lattices_with_high_k": float(metrics["lattices_with_high_k"]),
        "best_encoding_rate": metrics["best_encoding_rate"],
        "total_candidates": float(metrics["total_candidates"]),
        "connectivity_rejected": float(
            metrics["total_connectivity_rejected"]
        ),
    }

    try:
        from openevolve.evaluation_result import EvaluationResult
        return EvaluationResult(metrics=result, artifacts=artifacts)
    except ImportError:
        return result


# For direct OpenEvolve use (non-cascade mode)
def evaluate(program_path: str) -> dict:
    """Full evaluation (non-cascade mode)."""
    return evaluate_stage2(program_path)
