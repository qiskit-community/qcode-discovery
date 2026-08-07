#!/usr/bin/env python3
"""Stage-0 driver for the weight-5 CSS/PBB campaigns (plan Phase B2/C2).

Two distinct outputs per lattice, never conflated in the report:

1. **Exhaustive cheap census** -- for every bounded backbone (the B1/C1
   Strategy-1 sweep: ``A=[(0,0),(a1,a2)]``, ``B=[(0,0),(b1,0),(b2,b3)]``)
   at a Stage-1 lattice: canonicalize, construct, compute exact ``(n,k)``,
   and (PBB only) run the cheap algebraic LC-CSS check on every ``k>0``
   survivor. No BP-OSD, MILP, or the expensive ``is_equivalently_css``
   check runs over the full population.

2. **Deterministic sampled distance/gate audit** -- at most
   ``--max-audit-candidates`` (default 500) ``k>0`` survivors per lattice,
   selected by :func:`evaluation.candidate_selection.stratified_select`
   (k-band-stratified, seeded, reproducible). CSS runs the quick BP-OSD
   distance cascade on the sample; PBB runs the combined non-CSS gate
   (:func:`evaluation.noncss_gate.passes_noncss_gate`) and bounded
   exhaustive-hash distance work (:func:`evaluation.distance_bposd_noncss.
   has_low_weight_logical`) on gate survivors. Both report per-stratum
   rates with 95% Wilson intervals plus a population-size-weighted
   stratified overall estimate -- never an unweighted pooled rate.

This is a feasibility screen, not a distance proof: a lattice/audit
result here should inform which lattices proceed to the real evolutionary
campaign, not stand in for the campaign's own certified distance work.

Before running the census at a lattice, a deterministic slice of the
population is benchmarked and extrapolated against
``--cheap-cpu-hours-per-lattice``; exceeding the ceiling halts that
lattice with a resumable, honestly-labeled ``budget_exceeded`` checkpoint
rather than silently downgrading to a sample.

PBB backbone-level caching (plan's "Stage-0 limits" bullet): the BBCode
and each individual term's circulant matrix are built once per backbone
and reused to derive all 31 subset-pair matrices, instead of rebuilding
the same qLDPC object 31 times (see :class:`_PbbBackboneCache`).

Checkpoints are written atomically (temp file + fsync + os.replace, same
pattern as ``evolve/run_manifest.py``) and are only resumed when the
stored ``config_hash`` (family/lattice/config-version/flags) matches --
otherwise the census restarts from scratch for that lattice.

Usage::

    uv run python scripts/run_weight5_stage0.py \\
      --family css --profile small --seed 42 \\
      --max-audit-candidates 500 --cheap-cpu-hours-per-lattice 4 \\
      --checkpoint-dir results/stage0/weight5_css_small
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from qldpc import codes
from qldpc.codes import QuditCode
from sympy.abc import x, y

from evaluation.bb_code import build_bb_code, get_code_params_fast, terms_to_poly
from evaluation.candidate_selection import canonical_candidate_key, k_band_for, stratified_select
from evaluation.clifford_equivalence import verify_not_lc_css
from evaluation.distance import estimate_distance
from evaluation.distance_bposd_noncss import has_low_weight_logical
from evaluation.gates import min_relevant_d, min_relevant_d_noncss
from evaluation.noncss_gate import passes_noncss_gate
from evaluation.pbb_code import (
    _poly_to_matrix,
    build_pbb_code,
    check_commutativity,
    get_pbb_params_fast,
)
from evaluation.weight_enforcement import css_weight_ok

Z_95 = 1.959963984540054  # two-sided 95% normal quantile

SMALL_STAGE1 = [(6, 6), (6, 9), (9, 8)]
LARGE_STAGE1 = [(12, 9), (12, 12), (15, 12)]
PROFILE_LATTICES = {"small": SMALL_STAGE1, "large": LARGE_STAGE1}
DEFAULT_CEILING_HOURS = {"small": 4.0, "large": 8.0}

_CONFIG_VERSION = 1  # bump if the backbone/subset enumeration logic changes


# ---------------------------------------------------------------------------
# Deterministic backbone / subset-pair enumeration (B1/C1 Strategy-1 sweep)
# ---------------------------------------------------------------------------
def backbone_bounds(ell: int, m: int) -> tuple[int, int]:
    return min(ell - 1, 6), min(m - 1, 6)


def backbone_population(ell: int, m: int) -> int:
    Nx, Ny = backbone_bounds(ell, m)
    if Nx < 1 or Ny < 1:
        return 0
    return Nx * Nx * Nx * Ny * Ny  # Nx^3 * Ny^2


def bounded_backbones(ell: int, m: int):
    """Yield every (A_terms, B_terms) backbone in canonical, deterministic
    order. ``A=[(0,0),(a1,a2)]``, ``B=[(0,0),(b1,0),(b2,b3)]`` -- all terms
    within each polynomial are distinct by construction (a1,b1,b2,b3 >= 1),
    so every backbone has weight exactly 5 with a canonical 2+3 split."""
    Nx, Ny = backbone_bounds(ell, m)
    if Nx < 1 or Ny < 1:
        return
    for a1 in range(1, Nx + 1):
        for a2 in range(1, Ny + 1):
            A = [(0, 0), (a1, a2)]
            for b1 in range(1, Nx + 1):
                for b2 in range(1, Nx + 1):
                    for b3 in range(1, Ny + 1):
                        yield (A, [(0, 0), (b1, 0), (b2, b3)])


def _subsets(terms: list[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]]:
    out = [()]
    for r in range(1, len(terms) + 1):
        out.extend(itertools.combinations(terms, r))
    return out


def nontrivial_subset_pairs(A_terms, B_terms):
    """Yield all 31 nontrivial (C subseteq A, D subseteq B) pairs -- the 32
    combinations of subset(A) x subset(B) minus (empty, empty), which is
    excluded as the CSS calibration case (plan C2)."""
    a_subsets = _subsets(A_terms)
    b_subsets = _subsets(B_terms)
    for C in a_subsets:
        for D in b_subsets:
            if not C and not D:
                continue
            yield (list(C), list(D))


def pbb_backbone_pair_population(ell: int, m: int) -> int:
    return backbone_population(ell, m) * 31


class _PbbBackboneCache:
    """Backbone-level cache: build the BBCode and per-term circulant
    matrices once, then derive every subset's matrix by summing cached
    term matrices mod 2 -- avoids rebuilding the qLDPC object 31 times
    per backbone (plan's Stage-0-limits caching bullet).

    :meth:`build` mirrors ``evaluation.pbb_code.build_pbb_code``'s
    non-CSS tail exactly (same commutativity check, same error message,
    same symplectic assembly) -- ``tests/test_run_weight5_stage0.py``
    pins that the two paths agree on every candidate.
    """

    def __init__(self, ell: int, m: int, A_terms, B_terms):
        poly_a = terms_to_poly(A_terms)
        poly_b = terms_to_poly(B_terms)
        self._bb = codes.BBCode({x: ell, y: m}, poly_a, poly_b)
        self._dim = ell * m
        self._term_cache: dict[tuple[int, int], np.ndarray] = {}
        self.mat_A = self._subset_matrix(A_terms)
        self.mat_B = self._subset_matrix(B_terms)

    def _term_matrix(self, term: tuple[int, int]) -> np.ndarray:
        key = tuple(term)
        mat = self._term_cache.get(key)
        if mat is None:
            mat = _poly_to_matrix(self._bb, [term])
            self._term_cache[key] = mat
        return mat

    def _subset_matrix(self, terms) -> np.ndarray:
        out = np.zeros((self._dim, self._dim), dtype=int)
        for t in terms or []:
            out = (out + self._term_matrix(t)) % 2
        return out

    def build(self, C_terms, D_terms) -> QuditCode:
        mat_C = self._subset_matrix(C_terms)
        mat_D = self._subset_matrix(D_terms)
        if not check_commutativity(self.mat_A, self.mat_B, mat_C, mat_D):
            raise ValueError(
                "Commutativity violated: (A @ C^T + B @ D^T) % 2 is not symmetric"
            )
        zero = np.zeros((self._dim, self._dim), dtype=int)
        block1_x = np.hstack([self.mat_A, self.mat_B])
        block1_z = np.hstack([mat_C, mat_D])
        block2_x = np.hstack([zero, zero])
        block2_z = np.hstack([self.mat_B.T % 2, self.mat_A.T % 2])
        top = np.hstack([block1_x, block1_z])
        bottom = np.hstack([block2_x, block2_z])
        symplectic = np.vstack([top, bottom]) % 2
        return QuditCode(symplectic)


# ---------------------------------------------------------------------------
# Statistics helpers -- Wilson interval + weighted stratified estimate
# ---------------------------------------------------------------------------
def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """95% Wilson score interval for a single stratum's proportion."""
    if n == 0:
        return (0.0, 1.0)
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def stratified_rate_estimate(strata: list[dict]) -> dict:
    """Population-size-weighted stratified proportion estimate.

    ``strata`` is a list of ``{"population": N_h, "sample_n": n_h,
    "successes": x_h}`` dicts. Never use an unweighted pooled rate when
    quotas differ across strata (plan C2) -- this computes
    ``p_hat = sum(W_h * p_h)`` with ``W_h = N_h / N`` and the matching
    stratified variance, rather than ``sum(x_h) / sum(n_h)``.
    """
    total_population = sum(s["population"] for s in strata)
    if total_population == 0:
        return {"point_estimate": 0.0, "variance": 0.0, "ci": (0.0, 0.0)}
    point = 0.0
    variance = 0.0
    for s in strata:
        if s["sample_n"] == 0:
            continue
        w = s["population"] / total_population
        p_h = s["successes"] / s["sample_n"]
        point += w * p_h
        variance += (w * w) * p_h * (1 - p_h) / s["sample_n"]
    half = Z_95 * math.sqrt(variance)
    return {
        "point_estimate": point,
        "variance": variance,
        "ci": (max(0.0, point - half), min(1.0, point + half)),
    }


# ---------------------------------------------------------------------------
# Atomic checkpoint I/O (same temp-file + fsync + os.replace pattern as
# evolve/run_manifest.py)
# ---------------------------------------------------------------------------
def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".stage0.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _peak_memory_bytes() -> int:
    try:
        import resource

        # ru_maxrss is KB on Linux, bytes on macOS -- best-effort, diagnostic
        # only (not a budget-gate input), so platform variance is acceptable.
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except Exception:
        return 0


def _checkpoint_path(checkpoint_dir: Path, family: str, ell: int, m: int) -> Path:
    return checkpoint_dir / f"{family}_{ell}x{m}.checkpoint.json"


def _survivors_path(checkpoint_dir: Path, family: str, ell: int, m: int) -> Path:
    return checkpoint_dir / f"{family}_{ell}x{m}.survivors.jsonl"


def _load_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _new_counts() -> dict:
    return {
        "total_enumerated": 0,
        "weight_violation": 0,
        "commutativity_rejected": 0,
        "k_zero": 0,
        "k_positive": 0,
        "lc_css_rejected": 0,
        "true_survivors": 0,
    }


def _read_survivors(survivors_path: Path) -> list[dict]:
    if not survivors_path.exists():
        return []
    items = []
    with open(survivors_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ---------------------------------------------------------------------------
# Budget gate: benchmark a deterministic slice, extrapolate, and refuse to
# silently downgrade an over-budget exhaustive census into a mislabeled
# sample (plan's Compute Budget and Decision Gates section).
# ---------------------------------------------------------------------------
def benchmark_and_check_budget(
    family: str, ell: int, m: int, ceiling_hours: float, slice_size: int = 100,
    skip_lc_css_census: bool = False,
) -> dict:
    population = (
        backbone_population(ell, m) if family == "css"
        else pbb_backbone_pair_population(ell, m)
    )
    if population == 0:
        return {
            "population": 0, "slice_measured": 0, "seconds_per_item": 0.0,
            "projected_hours": 0.0, "peak_memory_bytes": _peak_memory_bytes(),
            "within_budget": True,
        }

    start = time.monotonic()
    measured = 0
    if family == "css":
        for A, B in itertools.islice(bounded_backbones(ell, m), slice_size):
            code = build_bb_code(ell, m, A, B)
            get_code_params_fast(code)
            measured += 1
    else:
        for A, B in itertools.islice(bounded_backbones(ell, m), max(1, slice_size // 31 + 1)):
            cache = _PbbBackboneCache(ell, m, A, B)
            for C, D in nontrivial_subset_pairs(A, B):
                try:
                    code = cache.build(C, D)
                except ValueError:
                    pass
                else:
                    _, k = get_pbb_params_fast(code)
                    # Mirror run_pbb_cheap_census's dominant per-survivor
                    # cost. Omitting this systematically underestimates
                    # projected_hours for the default (non-skip) census,
                    # so the budget gate would pass lattices that actually
                    # blow past the ceiling once the real census runs.
                    if k > 0 and not skip_lc_css_census:
                        verify_not_lc_css(ell, m, A, B, C, D)
                measured += 1
                if measured >= slice_size:
                    break
            if measured >= slice_size:
                break
    elapsed = time.monotonic() - start
    seconds_per_item = elapsed / measured if measured else 0.0
    projected_hours = seconds_per_item * population / 3600.0
    return {
        "population": population,
        "slice_measured": measured,
        "seconds_per_item": seconds_per_item,
        "projected_hours": projected_hours,
        "peak_memory_bytes": _peak_memory_bytes(),
        "within_budget": projected_hours <= ceiling_hours,
    }


# ---------------------------------------------------------------------------
# CSS exhaustive cheap census
# ---------------------------------------------------------------------------
def run_css_cheap_census(
    ell: int, m: int, checkpoint_dir: Path, *,
    checkpoint_interval: int = 500, force: bool = False,
) -> dict:
    config_hash = _config_hash(
        {"family": "css", "ell": ell, "m": m, "config_version": _CONFIG_VERSION}
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = _checkpoint_path(checkpoint_dir, "css", ell, m)
    survivors_path = _survivors_path(checkpoint_dir, "css", ell, m)

    checkpoint = None if force else _load_checkpoint(ckpt_path)
    if checkpoint and checkpoint.get("config_hash") != config_hash:
        checkpoint = None
    if checkpoint is not None and checkpoint.get("status") == "complete":
        return checkpoint

    if checkpoint is None:
        cursor = 0
        counts = _new_counts()
        per_stratum_population: dict[str, int] = {}
        elapsed_seconds = 0.0
        if survivors_path.exists():
            survivors_path.unlink()
    else:
        cursor = checkpoint["cursor"]
        counts = checkpoint["counts"]
        per_stratum_population = checkpoint["per_stratum_population"]
        elapsed_seconds = checkpoint["elapsed_seconds"]

    start = time.monotonic()
    with open(survivors_path, "a", encoding="utf-8") as survivors_f:
        for idx, (A, B) in enumerate(
            itertools.islice(bounded_backbones(ell, m), cursor, None), start=cursor,
        ):
            counts["total_enumerated"] += 1
            if css_weight_ok(A, B):
                code = build_bb_code(ell, m, A, B)
                n, k = get_code_params_fast(code)
                if k == 0:
                    counts["k_zero"] += 1
                else:
                    counts["k_positive"] += 1
                    band = k_band_for(k)
                    if band is not None:
                        per_stratum_population[band] = per_stratum_population.get(band, 0) + 1
                        survivors_f.write(json.dumps(
                            {"A_terms": A, "B_terms": B, "n": n, "k": k, "band": band}
                        ) + "\n")
            else:
                counts["weight_violation"] += 1

            cursor = idx + 1
            if cursor % checkpoint_interval == 0:
                survivors_f.flush()
                _atomic_write_json(ckpt_path, {
                    "config_hash": config_hash, "cursor": cursor, "counts": counts,
                    "per_stratum_population": per_stratum_population,
                    "elapsed_seconds": elapsed_seconds + (time.monotonic() - start),
                    "status": "in_progress",
                })

    final = {
        "config_hash": config_hash, "cursor": cursor, "counts": counts,
        "per_stratum_population": per_stratum_population,
        "elapsed_seconds": elapsed_seconds + (time.monotonic() - start),
        "status": "complete",
    }
    _atomic_write_json(ckpt_path, final)
    return final


# ---------------------------------------------------------------------------
# PBB exhaustive cheap census
# ---------------------------------------------------------------------------
def run_pbb_cheap_census(
    ell: int, m: int, checkpoint_dir: Path, *,
    checkpoint_interval: int = 500, force: bool = False,
    skip_lc_css_census: bool = False,
) -> dict:
    config_hash = _config_hash({
        "family": "pbb", "ell": ell, "m": m, "config_version": _CONFIG_VERSION,
        "skip_lc_css_census": skip_lc_css_census,
    })
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = _checkpoint_path(checkpoint_dir, "pbb", ell, m)
    survivors_path = _survivors_path(checkpoint_dir, "pbb", ell, m)

    checkpoint = None if force else _load_checkpoint(ckpt_path)
    if checkpoint and checkpoint.get("config_hash") != config_hash:
        checkpoint = None
    if checkpoint is not None and checkpoint.get("status") == "complete":
        return checkpoint

    if checkpoint is None:
        cursor = 0
        counts = _new_counts()
        per_stratum_population: dict[str, int] = {}
        elapsed_seconds = 0.0
        if survivors_path.exists():
            survivors_path.unlink()
    else:
        cursor = checkpoint["cursor"]
        counts = checkpoint["counts"]
        per_stratum_population = checkpoint["per_stratum_population"]
        elapsed_seconds = checkpoint["elapsed_seconds"]

    backbone_start, subset_start = divmod(cursor, 31)
    start = time.monotonic()

    with open(survivors_path, "a", encoding="utf-8") as survivors_f:
        for b_offset, (A, B) in enumerate(
            itertools.islice(bounded_backbones(ell, m), backbone_start, None)
        ):
            backbone_idx = backbone_start + b_offset
            cache = _PbbBackboneCache(ell, m, A, B)
            pairs = list(nontrivial_subset_pairs(A, B))
            subset_offset0 = subset_start if b_offset == 0 else 0

            for s_idx in range(subset_offset0, len(pairs)):
                C, D = pairs[s_idx]
                counts["total_enumerated"] += 1
                try:
                    code = cache.build(C, D)
                except ValueError:
                    counts["commutativity_rejected"] += 1
                else:
                    n, k = get_pbb_params_fast(code)
                    if k == 0:
                        counts["k_zero"] += 1
                    else:
                        counts["k_positive"] += 1
                        is_survivor = True
                        if not skip_lc_css_census:
                            lc = verify_not_lc_css(ell, m, A, B, C, D)
                            if lc["is_lc_css"]:
                                counts["lc_css_rejected"] += 1
                                is_survivor = False
                        if is_survivor:
                            counts["true_survivors"] += 1
                            band = k_band_for(k)
                            if band is not None:
                                per_stratum_population[band] = (
                                    per_stratum_population.get(band, 0) + 1
                                )
                                survivors_f.write(json.dumps({
                                    "A_terms": A, "B_terms": B,
                                    "C_terms": C, "D_terms": D,
                                    "n": n, "k": k, "band": band,
                                }) + "\n")

                cursor = backbone_idx * 31 + s_idx + 1
                if cursor % checkpoint_interval == 0:
                    survivors_f.flush()
                    _atomic_write_json(ckpt_path, {
                        "config_hash": config_hash, "cursor": cursor, "counts": counts,
                        "per_stratum_population": per_stratum_population,
                        "elapsed_seconds": elapsed_seconds + (time.monotonic() - start),
                        "status": "in_progress",
                    })

    final = {
        "config_hash": config_hash, "cursor": cursor, "counts": counts,
        "per_stratum_population": per_stratum_population,
        "elapsed_seconds": elapsed_seconds + (time.monotonic() - start),
        "status": "complete",
    }
    _atomic_write_json(ckpt_path, final)
    return final


# ---------------------------------------------------------------------------
# Sampled distance/gate audits (reuse evaluation.candidate_selection's
# seeded, k-band-stratified protocol -- never a bespoke sampler)
# ---------------------------------------------------------------------------
def _stratified_report(selected_metrics: dict, per_stratum_results: dict) -> tuple[dict, dict]:
    strata_for_estimate = []
    per_stratum_report = {}
    for band, pop_info in selected_metrics["per_stratum"].items():
        bucket = per_stratum_results.get(band, {"n": 0, "successes": 0})
        lo, hi = wilson_interval(bucket["successes"], bucket["n"])
        per_stratum_report[band] = {
            "population": pop_info["population"],
            "sample_n": bucket["n"],
            "successes": bucket["successes"],
            "wilson_ci_95": [lo, hi],
        }
        strata_for_estimate.append({
            "population": pop_info["population"],
            "sample_n": bucket["n"],
            "successes": bucket["successes"],
        })
    overall = stratified_rate_estimate(strata_for_estimate)
    return per_stratum_report, overall


def run_css_sampled_audit(
    ell: int, m: int, survivors: list[dict], *,
    run_seed, max_audit_candidates: int, quick_trials: int,
) -> dict:
    selected, metrics = stratified_select(
        survivors,
        stratum_of=lambda r: r.get("band"),
        key_of=lambda r: canonical_candidate_key((r["A_terms"], r["B_terms"])),
        run_seed=run_seed, ell=ell, m=m, max_total=max_audit_candidates,
    )
    threshold = min_relevant_d()
    per_stratum_results: dict[str, dict] = {}
    audited = []
    for r in selected:
        code = build_bb_code(ell, m, r["A_terms"], r["B_terms"])
        d_upper = estimate_distance(code, num_trials=quick_trials)
        meets = d_upper >= threshold
        audited.append({**r, "d_upper_estimate": d_upper, "meets_min_relevant_d": meets})
        bucket = per_stratum_results.setdefault(r.get("band"), {"n": 0, "successes": 0})
        bucket["n"] += 1
        bucket["successes"] += 1 if meets else 0

    per_stratum_report, overall = _stratified_report(metrics, per_stratum_results)
    return {
        "selection_metrics": metrics,
        "min_relevant_d_threshold": threshold,
        "quick_trials": quick_trials,
        "per_stratum": per_stratum_report,
        "stratified_overall_rate": overall,
        "audited_candidates": audited,
    }


def run_pbb_sampled_audit(
    ell: int, m: int, survivors: list[dict], *,
    run_seed, max_audit_candidates: int,
) -> dict:
    selected, metrics = stratified_select(
        survivors,
        stratum_of=lambda r: r.get("band"),
        key_of=lambda r: canonical_candidate_key(
            (r["A_terms"], r["B_terms"], r["C_terms"], r["D_terms"])
        ),
        run_seed=run_seed, ell=ell, m=m, max_total=max_audit_candidates,
    )
    threshold = min_relevant_d_noncss()
    n = 2 * ell * m
    max_weight = 6 if n <= 216 else 4
    gate_cache: dict = {}
    per_stratum_results: dict[str, dict] = {}
    audited = []
    for r in selected:
        code = build_pbb_code(ell, m, r["A_terms"], r["B_terms"], r["C_terms"], r["D_terms"])
        gate = passes_noncss_gate(
            ell, m, r["A_terms"], r["B_terms"], r["C_terms"], r["D_terms"], code,
            cache=gate_cache,
        )
        entry = {**r, "gate_passes": gate["passes"]}
        meets = False
        if gate["passes"]:
            found, min_weight = has_low_weight_logical(code, max_weight=max_weight)
            meets = (not found) or (min_weight >= threshold)
            entry["low_weight_logical_found"] = found
            entry["min_weight_found"] = min_weight
            entry["searched_to_weight"] = max_weight
        entry["meets_min_relevant_d"] = meets
        audited.append(entry)
        bucket = per_stratum_results.setdefault(r.get("band"), {"n": 0, "successes": 0})
        bucket["n"] += 1
        bucket["successes"] += 1 if meets else 0

    per_stratum_report, overall = _stratified_report(metrics, per_stratum_results)
    return {
        "selection_metrics": metrics,
        "min_relevant_d_noncss_threshold": threshold,
        "max_weight_searched": max_weight,
        "per_stratum": per_stratum_report,
        "stratified_overall_rate": overall,
        "audited_candidates": audited,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _write_markdown_summary(report: dict, path: Path) -> None:
    lines = [
        f"# Stage-0 {report['family'].upper()} report ({report['profile']} profile)",
        "",
        f"Seed: {report['seed']}  |  CPU-hour ceiling/lattice: "
        f"{report['cheap_cpu_hours_per_lattice_ceiling']}  |  Max audit "
        f"candidates/lattice: {report['max_audit_candidates']}",
        "",
        "**This is a feasibility screen, not a distance proof.** Exhaustive-"
        "census counts below cover every bounded backbone (CSS) or backbone "
        "x nontrivial-subset-pair (PBB) at each lattice. Audit rates are "
        "estimated from a deterministic, seeded, k-band-stratified sample of "
        "the k>0 survivor pool -- never the full population -- and are "
        "reported with per-stratum Wilson 95% intervals plus a population-"
        "weighted stratified overall estimate. A pass on a sampled candidate "
        "is a single quick-estimate/bounded-search result, not a certified "
        "distance; final lattice go/no-go decisions should follow the plan's "
        "Rollout review, not this script alone.",
        "",
    ]
    for lr in report["lattices"]:
        lines.append(f"## Lattice ({lr['ell']}, {lr['m']})")
        lines.append("")
        if lr["status"] == "budget_exceeded":
            b = lr["benchmark"]
            lines.append(
                f"**BUDGET EXCEEDED** -- projected {b['projected_hours']:.2f}h from a "
                f"{b['slice_measured']}-item benchmark slice exceeds the "
                f"{report['cheap_cpu_hours_per_lattice_ceiling']}h ceiling "
                f"(population {b['population']}). Census not run; no sample was "
                f"substituted for the exhaustive count. Options: optimize, narrow "
                f"the exponent space, or record an explicit budget increase and rerun."
            )
            lines.append("")
            continue

        counts = lr["census"]["counts"]
        lines.append("**Exhaustive cheap census** (counts, not a sample):")
        for key, val in counts.items():
            lines.append(f"- {key}: {val}")
        lines.append("")

        audit = lr["audit"]
        if audit.get("lc_css_census_skipped"):
            lines.append(
                "_LC-CSS exhaustive pass over survivors was skipped "
                "(--skip-lc-css-census); `lc_css_rejected`/`true_survivors` "
                "counts above are not meaningful._"
            )
            lines.append("")

        lines.append(
            f"**Sampled audit** ({audit['selection_metrics']['selected_count']} of "
            f"{audit['selection_metrics']['raw_count']} k>0 survivors, "
            f"seed={report['seed']}):"
        )
        lines.append("")
        lines.append("| k-band | population | sampled | passes | Wilson 95% CI |")
        lines.append("|---|---|---|---|---|")
        for band, stats in audit["per_stratum"].items():
            lo, hi = stats["wilson_ci_95"]
            lines.append(
                f"| {band} | {stats['population']} | {stats['sample_n']} | "
                f"{stats['successes']} | [{lo:.3f}, {hi:.3f}] |"
            )
        overall = audit["stratified_overall_rate"]
        lo, hi = overall["ci"]
        lines.append("")
        lines.append(
            f"Stratified overall pass rate (population-weighted, NOT pooled): "
            f"{overall['point_estimate']:.3f}  95% CI [{lo:.3f}, {hi:.3f}]"
        )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_lattice(s: str) -> tuple[int, int]:
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected 'ell,m', got {s!r}")
    return int(parts[0]), int(parts[1])


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--family", choices=["css", "pbb"], required=True)
    p.add_argument("--profile", choices=["small", "large"], default="small")
    p.add_argument(
        "--lattices", type=_parse_lattice, nargs="+", default=None,
        help="Override the profile's Stage-1 lattice list, e.g. --lattices 6,6 6,9",
    )
    p.add_argument("--seed", type=int, default=0, help="Run seed for stratified_select")
    p.add_argument("--max-audit-candidates", type=int, default=500)
    p.add_argument(
        "--cheap-cpu-hours-per-lattice", type=float, default=None,
        help="Defaults to 4.0 (small profile) / 8.0 (large profile)",
    )
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument(
        "--quick-trials", type=int, default=100,
        help="CSS BP-OSD trials for the sampled audit",
    )
    p.add_argument("--benchmark-slice", type=int, default=100)
    p.add_argument("--checkpoint-interval", type=int, default=500)
    p.add_argument(
        "--force", action="store_true",
        help="Ignore any existing checkpoint and restart the census",
    )
    p.add_argument(
        "--skip-lc-css-census", action="store_true",
        help="PBB only: skip the exhaustive verify_not_lc_css pass over k>0 "
             "survivors (unbounded-cost algebraic check); report is labeled "
             "lc_css_census_skipped",
    )
    p.add_argument(
        "--report-path", type=Path, default=None,
        help="Defaults to <checkpoint-dir>/stage0_report.json",
    )
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    lattices = args.lattices if args.lattices else PROFILE_LATTICES[args.profile]
    ceiling_hours = (
        args.cheap_cpu_hours_per_lattice
        if args.cheap_cpu_hours_per_lattice is not None
        else DEFAULT_CEILING_HOURS[args.profile]
    )
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_path or (args.checkpoint_dir / "stage0_report.json")

    lattice_reports = []
    for ell, m in lattices:
        print(f"[stage0] {args.family} lattice ({ell},{m}): benchmarking...", file=sys.stderr)
        bench = benchmark_and_check_budget(
            args.family, ell, m, ceiling_hours, slice_size=args.benchmark_slice,
            skip_lc_css_census=args.skip_lc_css_census,
        )
        if not bench["within_budget"]:
            print(
                f"[stage0] ({ell},{m}) projected {bench['projected_hours']:.2f}h exceeds "
                f"ceiling {ceiling_hours}h -- stopping this lattice, no sample-and-"
                f"mislabel fallback",
                file=sys.stderr,
            )
            config_hash = _config_hash({
                "family": args.family, "ell": ell, "m": m,
                "config_version": _CONFIG_VERSION,
                **({"skip_lc_css_census": args.skip_lc_css_census}
                   if args.family == "pbb" else {}),
            })
            _atomic_write_json(
                _checkpoint_path(args.checkpoint_dir, args.family, ell, m),
                {
                    "config_hash": config_hash, "status": "budget_exceeded",
                    "benchmark": bench, "cursor": 0, "counts": _new_counts(),
                    "per_stratum_population": {}, "elapsed_seconds": 0.0,
                },
            )
            lattice_reports.append({
                "ell": ell, "m": m, "status": "budget_exceeded", "benchmark": bench,
            })
            continue

        print(
            f"[stage0] ({ell},{m}) within budget "
            f"({bench['projected_hours']:.2f}h projected) -- running exhaustive census",
            file=sys.stderr,
        )
        if args.family == "css":
            census = run_css_cheap_census(
                ell, m, args.checkpoint_dir,
                checkpoint_interval=args.checkpoint_interval, force=args.force,
            )
        else:
            census = run_pbb_cheap_census(
                ell, m, args.checkpoint_dir,
                checkpoint_interval=args.checkpoint_interval, force=args.force,
                skip_lc_css_census=args.skip_lc_css_census,
            )

        survivors_path = _survivors_path(args.checkpoint_dir, args.family, ell, m)
        survivors = _read_survivors(survivors_path)
        print(
            f"[stage0] ({ell},{m}) census complete: {census['counts']} -- "
            f"running sampled audit on {len(survivors)} survivors",
            file=sys.stderr,
        )

        if args.family == "css":
            audit = run_css_sampled_audit(
                ell, m, survivors, run_seed=args.seed,
                max_audit_candidates=args.max_audit_candidates,
                quick_trials=args.quick_trials,
            )
        else:
            audit = run_pbb_sampled_audit(
                ell, m, survivors, run_seed=args.seed,
                max_audit_candidates=args.max_audit_candidates,
            )
            if args.skip_lc_css_census:
                audit["lc_css_census_skipped"] = True

        lattice_reports.append({
            "ell": ell, "m": m, "status": "complete",
            "benchmark": bench, "census": census, "audit": audit,
        })

    report = {
        "family": args.family, "profile": args.profile, "seed": args.seed,
        "cheap_cpu_hours_per_lattice_ceiling": ceiling_hours,
        "max_audit_candidates": args.max_audit_candidates,
        "lattices": lattice_reports,
    }
    _atomic_write_json(report_path, report)
    _write_markdown_summary(report, report_path.with_suffix(".md"))
    print(f"[stage0] report written to {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
