"""OpenEvolve evaluator adapter for non-CSS PBB code discovery.

Bridges OpenEvolve's ``evaluate(program_path)`` interface with the PBB
code construction and multi-tier distance estimation pipeline.

Two-stage cascade
-----------------
**Stage 1** -- Quick k-only screening (~5 s)
    Builds PBB codes from evolved candidates at 1 small lattice.
    Programs must produce valid codes (k > 0) at all stage-1 lattices.

**Stage 2** -- Full evaluation with adaptive distance (~120-600 s)
    Evaluates on 7 lattices (n=36 to n=360) with adaptive distance:
      1. Hash-based exact check: d≤6 at n≤216, d≤4 at n>216
      2. MILP symplectic with adaptive timeouts (15-60s/logical)
      3. BP-OSD fallback only if MILP yields nothing

Scoring
-------
FOM = k * d^2 / n.  Codes with d≤4 get fom=0 (rejected by worker).
Distances are reliable via hash + MILP -- raw FOM used directly.

TODO: add an LC-CSS filter using ``verify_not_lc_css`` from
``evaluation.clifford_equivalence`` to flag or penalize codes that are
merely LC-equivalent to CSS (and thus not genuinely non-CSS).
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

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from evaluation.pbb_code import build_pbb_code, get_pbb_params_fast
from evolve._noncss_distance_worker import distance_worker as _distance_worker

# Per-model attribution: install here so it is active in each spawn worker
# (OpenEvolve imports this evaluator before generating). Defensive no-op if
# unavailable.
try:
    from evolve.model_attribution import install_tagging as _install_tagging
    _install_tagging()
except Exception:
    pass

logger = logging.getLogger(__name__)

# Parallelism for distance estimation.
# Each worker runs hash check + potentially MILP + BP-OSD.
# At n≤216: hash ~5-135s, MILP ~180s budget.  At n>216: hash <1s, MILP ~360s.
# All single-threaded, embarrassingly parallel.  Capped at 16 workers.
_NUM_DISTANCE_WORKERS = max(1, min((os.cpu_count() or 4) // 2, 16))

# Lattice subsets -- PBB codes at (6,6) n=72 are the primary target,
# with larger lattices up to n=360 for exploring higher-FOM codes.
#
# Stage 1: single lattice -- avoids fragile multi-lattice gate.
# (6,3) produces only 20 valid codes with k<8 from the seed, so
# requiring it would block the LLM whenever it adjusts strategies.
STAGE1_LATTICES = [(6, 6)]
# Stage 2: primary + medium + large + exploratory lattices
# Distance pipeline is adaptive: d≤6 exact at n≤216, d≤4+MILP at n>216.
STAGE2_LATTICES = [
    (6, 6),    # n=72  -- primary target, best known FOM=6.0
    (9, 6),    # n=108 -- medium, d≤6 exact
    (12, 6),   # n=144 -- large, d≤6 exact
    (15, 6),   # n=180 -- large, d≤6 exact (borderline timing ~80s)
    (30, 6),   # n=360 -- big, d≤4 exact + MILP/BPOSD for d≥5
    (6, 3),    # n=36  -- fast validation
    (3, 6),    # n=36  -- transposed group structure
]

# Minimum relevant distance for scoring.
# d≤4 codes get fom=0 from the worker (rejected), so this only affects
# the scoring filter here (codes with d=4 from old runs).
MIN_RELEVANT_D = 5
# Max candidates to evaluate distance for, per lattice.
# Reduced: more lattices (7) × MILP per code = more total work.
MAX_DISTANCE_PER_LATTICE = 10


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
    if ratio < 2.5:
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


def _build_and_check(
    ell: int, m: int,
    A_terms: list, B_terms: list,
    C_terms: list, D_terms: list,
) -> dict | None:
    """Build a PBB code and return basic params, or None on failure."""
    try:
        code = build_pbb_code(ell, m, A_terms, B_terms, C_terms, D_terms)
        n, k = get_pbb_params_fast(code)
        return {
            "code": code,
            "ell": ell, "m": m, "n": n, "k": k,
            "A_terms": A_terms, "B_terms": B_terms,
            "C_terms": C_terms, "D_terms": D_terms,
            "encoding_rate": k / n if n > 0 else 0.0,
        }
    except (ValueError, Exception) as e:
        logger.debug(f"Build failed ({ell},{m}): {e}")
        return None



def _log_code_jsonl(result: dict, run_name: str | None = None) -> None:
    """Append a code result to the run-specific all_codes.jsonl file."""
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
        "timestamp": time.time(),
    }

    if run_name:
        log_dir = Path(_PROJECT_ROOT) / "results" / "evolution" / run_name
    else:
        log_dir = Path(_PROJECT_ROOT) / "results" / "evolution"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "all_codes_noncss.jsonl"

    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError:
        pass


def _write_metrics_jsonl(metrics: dict) -> None:
    """Write evaluation metrics to shared JSONL for W&B sync.

    Uses the same file as the CSS evaluator (evolution_metrics.jsonl)
    so the WandbSyncer in run_evolution.py picks up the records.
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
) -> dict:
    """Run evaluation across lattices and compute aggregate metrics.

    When quick=False, distance estimation runs in parallel across all
    lattices using ProcessPoolExecutor.  Each BP-OSD call is single-
    threaded, so tasks are embarrassingly parallel.
    """
    all_results = []
    total_candidates = 0
    errors = []

    # Phase 1: Build all codes, select top candidates (sequential -- fast)
    per_lattice_top: list[tuple[tuple[int, int], list[dict]]] = []
    per_lattice_rest: list[tuple[tuple[int, int], list[dict]]] = []

    for ell, m in lattices:
        try:
            candidates = generate_fn(ell, m)
            if not isinstance(candidates, list):
                errors.append(f"({ell},{m}): returned {type(candidates)}, not list")
                continue

            total_candidates += len(candidates)

            # Cap candidates per lattice
            if len(candidates) > 3000:
                errors.append(f"({ell},{m}): {len(candidates)} candidates, capped to 3000")
                candidates = candidates[:3000]

            # Build all codes and check k
            built = []
            for cand in candidates:
                if len(cand) != 4:
                    continue
                A_terms, B_terms, C_terms, D_terms = cand
                info = _build_and_check(ell, m, A_terms, B_terms, C_terms, D_terms)
                if info and info["k"] > 0:
                    built.append(info)

            if quick:
                # Stage 1: just report k values
                for info in built:
                    result = dict(info)
                    del result["code"]
                    result["d"] = 0
                    result["fom"] = 0.0
                    all_results.append(result)
            else:
                # Stage 2: select diverse top candidates for distance
                built.sort(key=lambda x: x["k"], reverse=True)

                seen_k: set[int] = set()
                top: list[dict] = []
                for info in built:
                    if info["k"] not in seen_k and len(top) < max_distance_per_lattice:
                        seen_k.add(info["k"])
                        top.append(info)
                for info in built:
                    if len(top) >= max_distance_per_lattice:
                        break
                    if info not in top:
                        top.append(info)

                per_lattice_top.append(((ell, m), top))

                # Rest: k-only results for non-top candidates
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

    # Phase 2: Parallel distance estimation across ALL lattices
    if not quick and per_lattice_top:
        # Collect tasks: (ell, m, A, B, C, D, num_trials)
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
            # Use "spawn" context to avoid segfaults in ldpc's C extensions
            # when forking processes that have already imported the decoder.
            # Submit individual futures so one worker segfault doesn't kill
            # the entire batch (BpOsdDecoder can segfault on certain non-CSS
            # matrices with degenerate structure).
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
            # Fallback to sequential on total pool failure
            logger.warning(f"Parallel distance failed ({e}), falling back to sequential")
            distance_results = []
            for task in distance_tasks:
                try:
                    distance_results.append(_distance_worker(task))
                except Exception as e2:
                    logger.debug(f"Distance worker failed: {e2}")

        for result in distance_results:
            all_results.append(result)
            _log_code_jsonl(result, run_name=run_name)

        # Append k-only results
        for (ell, m), rest in per_lattice_rest:
            all_results.extend(rest)

    # Aggregate metrics
    valid = [r for r in all_results if r.get("k", 0) > 0]
    foms = [_scored_fom(r) for r in valid if _scored_fom(r) > 0]
    encoding_rates = [r.get("encoding_rate", 0.0) for r in valid]

    best_fom = max(foms) if foms else 0.0
    mean_fom = sum(foms) / len(foms) if foms else 0.0
    best_encoding_rate = max(encoding_rates) if encoding_rates else 0.0

    high_k_codes = [r for r in valid if r.get("k", 0) >= 8]
    lattices_with_high_k = len(set(
        (r["ell"], r["m"]) for r in high_k_codes
    ))

    best_code = None
    if all_results:
        best_result = max(all_results, key=_scored_fom)
        if _scored_fom(best_result) > 0:
            best_code = best_result

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
    }


def evaluate_stage1(program_path: str) -> dict:
    """Stage 1: Quick screening on small lattices (k-only, ~5s).

    Programs must produce valid non-CSS codes (k > 0) at all stage-1
    lattices to advance past the cascade threshold.
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
        }

    valid = [r for r in metrics.get("all_results", []) if r.get("k", 0) > 0]

    # Require valid codes at ALL stage-1 lattices
    lattices_with_valid = set((r["ell"], r["m"]) for r in valid)
    lattice_coverage = len(lattices_with_valid) / len(STAGE1_LATTICES)

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
    }


def evaluate_stage2(program_path: str) -> dict:
    """Stage 2: Full evaluation with adaptive distance (~120-600s).

    Runs across 7 lattices (n=36 to n=360) with adaptive distance:
    hash-based exact (d≤6 at n≤216, d≤4 at n>216), MILP symplectic
    with adaptive timeouts, and BP-OSD fallback.
    """
    try:
        generate_fn = _load_generate_candidates(program_path)
    except Exception as e:
        return _error_result(str(e))

    metrics = _run_evaluation(
        generate_fn, STAGE2_LATTICES,
        quick=False, num_trials=1000,
    )

    # Combined score: sum of best trust-adjusted FOM per lattice.
    # Exact and trusted bounds score at full weight; partial high-d/sqrt(n)
    # bounds are discounted so speculative BP-OSD/MILP incumbents do not
    # dominate the evolution loop.
    per_lattice_best: dict[tuple[int, int], float] = {}
    for r in metrics["all_results"]:
        d_raw = r.get("d", 0)
        k = r.get("k", 0)
        n = r.get("n", 0)
        if k <= 0 or n <= 0:
            continue
        key = (r["ell"], r["m"])

        if d_raw < MIN_RELEVANT_D:
            # d≤4 codes are rejected (fom=0 from worker), but give a tiny
            # encoding-rate contribution so the LLM sees partial progress.
            fom = k / n * 0.1
        else:
            fom = _scored_fom(r)

        per_lattice_best[key] = max(per_lattice_best.get(key, 0.0), fom)

    combined = sum(per_lattice_best.values())
    best_fom = metrics["best_fom"]

    # Build artifacts for LLM feedback
    artifacts = {}
    # Show codes with d >= MIN_RELEVANT_D and nonzero trust-adjusted score.
    credible_codes = []
    for r in metrics["all_results"]:
        d_val = r.get("d", 0)
        if d_val >= MIN_RELEVANT_D and _trust_multiplier(r) > 0:
            credible_codes.append(r)

    if credible_codes:
        bc = max(credible_codes, key=lambda r: r.get("fom", 0.0))
        artifacts["best_code"] = (
            f"[[{bc['n']},{bc['k']},≤{bc['d']}]] FOM={bc['fom']:.2f} "
            f"trust={_trust_level_for_result(bc)} "
            f"at ({bc['ell']},{bc['m']})\n"
            f"  A={bc['A_terms']} B={bc['B_terms']}\n"
            f"  C={bc['C_terms']} D={bc['D_terms']}"
        )
    elif metrics["best_code"]:
        bc = metrics["best_code"]
        artifacts["best_code"] = (
            f"[[{bc['n']},{bc['k']},≤{bc.get('d', '?')}]] "
            f"at ({bc['ell']},{bc['m']})\n"
            f"  A={bc['A_terms']} B={bc['B_terms']}\n"
            f"  C={bc['C_terms']} D={bc['D_terms']}"
        )

    if metrics["errors"]:
        artifacts["errors"] = "\n".join(metrics["errors"][:5])

    # Top 5 codes with structural analysis
    top5 = sorted(credible_codes, key=_scored_fom, reverse=True)[:5]
    if top5:
        lines = []
        for r in top5:
            c_len = len(r.get("C_terms", []))
            d_len = len(r.get("D_terms", []))
            lines.append(
                f"  [[{r['n']},{r['k']},≤{r['d']}]] FOM={r['fom']:.1f} ({r['ell']},{r['m']}) "
                f"trust={_trust_level_for_result(r)} |C|={c_len} |D|={d_len}"
            )
        artifacts["top_codes"] = "\n".join(lines)

    # Structural feedback: analyze C,D patterns in top codes
    if top5:
        struct_lines = []
        for r in top5[:3]:
            C = r.get("C_terms", [])
            D = r.get("D_terms", [])
            struct_lines.append(
                f"  [[{r['n']},{r['k']},≤{r['d']}]] FOM={r['fom']:.1f} "
                f"trust={_trust_level_for_result(r)}:\n"
                f"    A={r['A_terms']} B={r['B_terms']}\n"
                f"    C={C} ({len(C)} terms)\n"
                f"    D={D} ({len(D)} terms)"
            )
        artifacts["structural_analysis"] = (
            "Structural analysis of top codes:\n" + "\n".join(struct_lines)
        )

    # Per-lattice breakdown
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
        f"Best FOM: {best_fom:.2f}\n"
        f"Combined score: {combined:.1f}\n"
        f"Per-lattice breakdown:\n" + "\n".join(lattice_lines)
    )

    # Save credible codes
    credible_to_save = [
        r for r in metrics["all_results"]
        if r.get("fom", 0) > 0
        and r.get("d", 0) >= MIN_RELEVANT_D
        and _trust_multiplier(r) > 0
    ]
    if credible_to_save:
        best_credible = max(credible_to_save, key=_scored_fom)
        if _scored_fom(best_credible) > 4.0:
            try:
                from evaluation.results import save_code, update_pareto_front
                save_code(best_credible)
                update_pareto_front(credible_to_save)
            except Exception:
                pass

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
