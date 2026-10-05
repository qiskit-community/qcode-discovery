#!/usr/bin/env python3
"""Matched-budget uniform control for the weight-five CSS campaign.

The control samples the finite identity-anchored ``2+3`` CSS universe

    A = {1, a},  B = {1, b, c}

uniformly without replacement on each of the six historical Stage-2
lattices.  It computes ``k`` for every draw and retains the first candidates
that fill the *observed* historical distance-stage quotas in the ``k=4`` and
``k>=6`` bands.  Thus the scarce distance budget is matched per lattice and
per band rather than merely in aggregate.

The pipeline is deliberately split into resumable stages:

``select``
    Draw without replacement, compute exact k, and record every draw.
``distance``
    Replay the historical BP--OSD cascade (1000 initial trials, three
    500-trial refinements at FOM >= 3, OSD-CS at FOM >= 5, and the historical
    exact-search attempt unless ``--skip-exact`` is used).
``certify``
    Canonicalize and run the same exhaustive weight-<=4 CSS audit on every
    retained result with decoder upper bound above two.  This either finds
    d<=4 exactly or certifies d>=5, with explicit witnesses when found.
``milp``
    Run the historical extended CSS MILP settings on finalists: estimated
    FOM >= 5 and a certified d>=5 lower bound.
``summarize``
    Report yields, connected stored-generator Tanner classes, diversity, and
    exact/class overlap with the historical campaign.

All records are append-only JSONL keyed by the sampled universe index.  A
stage can be interrupted and rerun; completed records are skipped.  Decoder
randomness is derived from the run seed and candidate key.  The historical
campaign did not persist its decoder RNG state, so this is a reproducibility
improvement, not a claim of bit-for-bit replay of its stochastic outputs.

Full matched-budget example (three deterministic seeds are the default)::

    .venv/bin/python scripts/run_weight5_uniform_baseline.py \
        --stage select --profile all
    .venv/bin/python scripts/run_weight5_uniform_baseline.py \
        --stage distance --profile all --workers 8
    .venv/bin/python scripts/run_weight5_uniform_baseline.py \
        --stage certify --profile all --workers 8
    .venv/bin/python scripts/run_weight5_uniform_baseline.py \
        --stage milp --profile all --workers 8
    .venv/bin/python scripts/run_weight5_uniform_baseline.py \
        --stage summarize --profile all

Fast end-to-end smoke protocol (explicitly *not* matched-budget)::

    .venv/bin/python scripts/run_weight5_uniform_baseline.py \
        --pilot --stage all --profile all --seeds 42 --workers 2 \
        --output-dir results/weight5_uniform_baseline_pilot
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Iterator, Sequence


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA_VERSION = 1
SAMPLER_VERSION = "lazy_fisher_yates_identity_anchored_v1"
HISTORICAL_EVALUATOR_COMMIT = "f86c1bccf1215c211f21c06bf97c7c2653b3147e"
HISTORICAL_EVALUATOR_PATH = "evolve/openevolve_evaluator_weight5_css.py"
HISTORICAL_EVALUATOR_SHA256 = (
    "c0971dee16048a3617cc7e9c5e45752e315ab255bd47669d43a94bda853fdf44"
)
HISTORICAL_RUNTIME_SHA256 = {
    # These are the modules called directly by this control.  They are
    # byte-identical in the working tree and at the pinned evaluator commit.
    "evaluation/evaluator.py": "99bd225a2fe6fa754bf19c851ad8e28daae02475827e8609b7309b3b53e5623d",
    "evaluation/distance.py": "3808b9bcd780de4a04cd971e724e43790e4e6eb2c4b252cba4da5f0fad12d3b1",
    "evaluation/bb_code.py": "42ca66564844ef0d6b72652738fc055c1f597c8bf89d041c2740661fa1cbfbfa",
}
# A runtime dependency above may legitimately stop matching its historical
# pin after a later, reviewed change that is behaviorally inert for this
# run's actual inputs -- e.g. one that only tightens validation on an input
# shape this run never produces. Each entry names the resulting local hash
# and why it is inert; verify_historical_sources() accepts it as a fallback
# only when the primary historical pin no longer matches the working tree,
# and records which case applied so the drift stays visible rather than
# silently re-pinned.
RUNTIME_VERIFIED_INERT_REVISIONS: dict[str, dict[str, str]] = {
    "evaluation/bb_code.py": {
        "sha256": "07b47a77e230cca4720b82d325efa7d4790408b08b977f096de0bda68a13ed1a",
        "reason": (
            "commit 1b2ebca added operator.index() validation rejecting "
            "non-integer exponent tuples; every exponent this historical "
            "run ever constructs is a plain Python int, so the added "
            "rejection path is never exercised and outputs are unaffected"
        ),
    },
    "evaluation/evaluator.py": {
        "sha256": "d07a47484f4633dc613884aa231c6dc8b83797748e02e0c22eafc8317affd047",
        "reason": (
            "evaluate_candidate_milp's all-timeout branch renamed its "
            "uncertified d_lower_bound field to distance_screening_threshold, "
            "dropped the stale +1 from its value (now milp_early_stop itself), "
            "and expanded the comment explaining it is a search-scheduling "
            "hint, not weak evidence of a bound on d; the docstring and a "
            "stage comment were also reworded from 'exact distance' to "
            "'distance bounds'/'MILP distance' to stop claiming every MILP "
            "result is exact. The selection/distance stages never call that "
            "function, this control's MILP stage produced no records, and no "
            "summary reads the field or these comments, so outputs are "
            "unaffected"
        ),
    },
}

SMALL_LATTICES = ((6, 10), (9, 9), (10, 9))
LARGE_LATTICES = ((16, 12), (15, 14), (18, 12))
PROFILE_LATTICES = {
    "small": SMALL_LATTICES,
    "large": LARGE_LATTICES,
    "all": SMALL_LATTICES + LARGE_LATTICES,
}

# Recovered directly from the historical positive-observation JSONL files.
# k is even for BB codes, so k_4_5 contains k=4 only here.
HISTORICAL_DISTANCE_QUOTAS: dict[tuple[int, int], dict[str, int]] = {
    (6, 10): {"k_4_5": 798, "k_ge_6": 1072},
    (9, 9): {"k_4_5": 1324, "k_ge_6": 546},
    (10, 9): {"k_4_5": 993, "k_ge_6": 877},
    (16, 12): {"k_4_5": 152, "k_ge_6": 122},
    (15, 14): {"k_4_5": 166, "k_ge_6": 114},
    (18, 12): {"k_4_5": 150, "k_ge_6": 130},
}

DEFAULT_SEEDS = (42, 314159, 271828)
HISTORICAL_GATES = {
    "min_k_threshold": 2,
    "fom_threshold_refine": 3.0,
    "fom_threshold_exact": 5.0,
    "save_trust_ratio": 2.0,
}
HISTORICAL_DECODER = {
    "quick_trials": 1000,
    "refine_trials_per_batch": 500,
    "refine_batches": 3,
    "osd_cs_trials": 200,
    "exact_timeout_seconds": 300,
}
HISTORICAL_MILP = {
    "timeout_per_logical_seconds": 1800,
    "total_timeout_seconds": 14400,
    "early_stop": 4,
}

UNIFORM_CONTROL_SOURCE_PATHS = (
    "scripts/run_weight5_uniform_baseline.py",
    "scripts/audit_weight5_css_low_weight.py",
    "evaluation/connectivity.py",
    "evaluation/tanner_equivalence.py",
    "evaluation/bb_code.py",
    "evaluation/evaluator.py",
    "evaluation/distance.py",
    "evaluation/distance_milp.py",
    "evaluation/gates.py",
)
UNIFORM_CONTROL_DEPENDENCY_PATHS = (
    "results/weight5_publication_catalogue.jsonl",
    "results/weight5_publication_manifest.json",
    "results/evolution/weight5_css_small_v1/all_codes_weight5_css.jsonl",
    "results/evolution/weight5_css_large_v1/all_codes_weight5_css.jsonl",
    "results/evolution/weight5_css_small_v1/run_manifest.json",
    "results/evolution/weight5_css_large_v1/run_manifest.json",
)
# The full-file hash loaded by the completed selection/distance processes.
# Subsequent edits in this revision only corrected derived reporting,
# validation, provenance, and the future-facing skipped-stage label.
EXECUTED_SELECTION_DISTANCE_SOURCE_SHA256 = (
    "0743d1d460821a54aa09b28fd7ada83ae194f87dee3e5b30711291366671f2b9"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    materialized = list(rows)
    if not materialized:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(_canonical_json(row) + b"\n" for row in materialized)
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return len(materialized)


def _atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(_canonical_json(row) + b"\n" for row in materialized)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSON at {path}:{line_number}")
            rows.append(row)
    return rows


def universe_size(ell: int, m: int) -> int:
    """Number of identity-anchored (binomial, trinomial) pairs."""
    group_order = ell * m
    return (group_order - 1) * math.comb(group_order - 1, 2)


def _unrank_two_subset(population: int, rank: int) -> tuple[int, int]:
    """Lexicographically unrank a 2-subset of ``range(population)``."""
    total = math.comb(population, 2)
    if not 0 <= rank < total:
        raise IndexError(f"combination rank {rank} outside [0,{total})")
    for left in range(population - 1):
        bucket = population - left - 1
        if rank < bucket:
            return left, left + 1 + rank
        rank -= bucket
    raise AssertionError("unreachable two-subset rank")


def _group_term(group_index: int, m: int) -> tuple[int, int]:
    return divmod(group_index, m)


def unrank_normalized_pair(
    ell: int, m: int, universe_index: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Map a universe index to one identity-anchored 2+3 support pair."""
    group_order = ell * m
    pair_population = math.comb(group_order - 1, 2)
    total = (group_order - 1) * pair_population
    if not 0 <= universe_index < total:
        raise IndexError(f"universe index {universe_index} outside [0,{total})")

    binomial_rank, trinomial_rank = divmod(universe_index, pair_population)
    left, right = _unrank_two_subset(group_order - 1, trinomial_rank)
    binomial_element = binomial_rank + 1
    trinomial_elements = (left + 1, right + 1)
    A_terms = [(0, 0), _group_term(binomial_element, m)]
    B_terms = [
        (0, 0),
        _group_term(trinomial_elements[0], m),
        _group_term(trinomial_elements[1], m),
    ]
    return A_terms, B_terms


class LazyFisherYates:
    """Seeded, lazy Fisher--Yates permutation using O(draws) memory."""

    def __init__(self, population: int, seed: int):
        if population < 0:
            raise ValueError("population must be nonnegative")
        self.population = population
        self.position = 0
        self._rng = random.Random(seed)
        self._swaps: dict[int, int] = {}

    def __iter__(self) -> "LazyFisherYates":
        return self

    def __next__(self) -> int:
        if self.position >= self.population:
            raise StopIteration
        position = self.position
        swap_position = self._rng.randrange(position, self.population)
        selected = self._swaps.get(swap_position, swap_position)
        replacement = self._swaps.get(position, position)
        self._swaps[swap_position] = replacement
        self._swaps.pop(position, None)
        self.position += 1
        return selected

    def advance(self, count: int) -> None:
        if count < 0 or self.position + count > self.population:
            raise ValueError("invalid permutation advance")
        for _ in range(count):
            next(self)


def _derived_seed(run_seed: int, ell: int, m: int, purpose: str) -> int:
    payload = f"{run_seed}|{ell}|{m}|{purpose}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _decoder_seed(run_seed: int, ell: int, m: int, universe_index: int) -> int:
    payload = f"{run_seed}|{ell}|{m}|{universe_index}|decoder-v1".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def k_band(k: int) -> str | None:
    if 4 <= k <= 5:
        return "k_4_5"
    if k >= 6:
        return "k_ge_6"
    return None


def _sorted_terms(terms: Sequence[Sequence[int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(term[0]), int(term[1])) for term in terms))


def literal_candidate_key(
    ell: int, m: int, A_terms: Sequence[Sequence[int]], B_terms: Sequence[Sequence[int]]
) -> tuple[Any, ...]:
    """Term-order and sector-swap invariant literal support key."""
    A = _sorted_terms(A_terms)
    B = _sorted_terms(B_terms)
    if len(A) == 3 and len(B) == 2:
        A, B = B, A
    return ell, m, A, B


def _translation_canonical_support(
    terms: Sequence[Sequence[int]], ell: int, m: int
) -> tuple[tuple[int, int], ...]:
    support = _sorted_terms(terms)
    translated = []
    for anchor_x, anchor_y in support:
        translated.append(
            tuple(
                sorted(
                    ((x - anchor_x) % ell, (y - anchor_y) % m)
                    for x, y in support
                )
            )
        )
    return min(translated)


def translation_candidate_key(
    ell: int, m: int, A_terms: Sequence[Sequence[int]], B_terms: Sequence[Sequence[int]]
) -> tuple[Any, ...]:
    """Independent-translation, term-order, and sector-swap invariant key."""
    A = _translation_canonical_support(A_terms, ell, m)
    B = _translation_canonical_support(B_terms, ell, m)
    if len(A) == 3 and len(B) == 2:
        A, B = B, A
    return ell, m, A, B


def _jsonable_key(key: tuple[Any, ...]) -> list[Any]:
    ell, m, A, B = key
    return [ell, m, [list(term) for term in A], [list(term) for term in B]]


def historical_budget_from_logs(root: Path = ROOT) -> dict[tuple[int, int], dict[str, int]]:
    """Recover the observed distance-stage budgets from pinned raw logs."""
    counts: dict[tuple[int, int], Counter[str]] = defaultdict(Counter)
    for profile in ("small", "large"):
        path = (
            root
            / "results"
            / "evolution"
            / f"weight5_css_{profile}_v1"
            / "all_codes_weight5_css.jsonl"
        )
        for row in _read_jsonl(path):
            band = k_band(int(row["k"]))
            if band is None:
                raise ValueError(f"historical positive observation has unsupported k: {row}")
            counts[(int(row["ell"]), int(row["m"]))][band] += 1
    return {lattice: dict(counter) for lattice, counter in counts.items()}


def _git_object_bytes(commit: str, path: str, root: Path = ROOT) -> bytes | None:
    try:
        result = subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout


def verify_historical_sources(root: Path = ROOT) -> dict[str, Any]:
    """Fail closed if the pinned evaluator/manifests/log-derived quotas drift."""
    manifests = {}
    for profile in ("small", "large"):
        path = (
            root
            / "results"
            / "evolution"
            / f"weight5_css_{profile}_v1"
            / "run_manifest.json"
        )
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("evaluator_hash") != HISTORICAL_EVALUATOR_SHA256:
            raise ValueError(f"unexpected evaluator hash in {path}")
        for name, expected in HISTORICAL_GATES.items():
            if manifest.get("effective_gates", {}).get(name) != expected:
                raise ValueError(f"unexpected {name} in {path}")
        manifests[profile] = {
            "path": str(path.relative_to(root)),
            "sha256": _sha256_path(path),
            "run_id": manifest.get("run_id"),
            "rng_seed": manifest.get("rng_seed"),
            "git_commit": manifest.get("git_commit"),
            "git_dirty": manifest.get("git_dirty"),
            "effective_gates": manifest.get("effective_gates"),
        }

    recovered = historical_budget_from_logs(root)
    if recovered != HISTORICAL_DISTANCE_QUOTAS:
        raise ValueError(
            f"historical observation counts drifted: {recovered!r} != "
            f"{HISTORICAL_DISTANCE_QUOTAS!r}"
        )

    evaluator_bytes = _git_object_bytes(
        HISTORICAL_EVALUATOR_COMMIT, HISTORICAL_EVALUATOR_PATH, root
    )
    evaluator_git_verified = evaluator_bytes is not None
    if evaluator_bytes is not None:
        actual = _sha256_bytes(evaluator_bytes)
        if actual != HISTORICAL_EVALUATOR_SHA256:
            raise ValueError(
                f"pinned git evaluator hashes to {actual}, expected "
                f"{HISTORICAL_EVALUATOR_SHA256}"
            )

    runtime_components = {}
    for relative_path, expected_sha256 in HISTORICAL_RUNTIME_SHA256.items():
        local_path = root / relative_path
        local_sha256 = _sha256_path(local_path)
        inert_revision = RUNTIME_VERIFIED_INERT_REVISIONS.get(relative_path)
        if local_sha256 == expected_sha256:
            local_verified_via = "historical_pin"
        elif inert_revision is not None and local_sha256 == inert_revision["sha256"]:
            local_verified_via = "reviewed_inert_revision"
        else:
            raise ValueError(
                f"runtime dependency {relative_path} hashes to {local_sha256}, "
                f"expected historical {expected_sha256}"
            )
        # The historical git blob is always checked against the original
        # historical pin: a reviewed inert revision explains why the local
        # working tree differs today, not what actually ran historically.
        git_bytes = _git_object_bytes(
            HISTORICAL_EVALUATOR_COMMIT, relative_path, root
        )
        if git_bytes is not None and _sha256_bytes(git_bytes) != expected_sha256:
            raise ValueError(f"historical git dependency drift for {relative_path}")
        component = {
            "sha256": expected_sha256,
            "local_verified": True,
            "local_verified_via": local_verified_via,
            "git_object_verified": git_bytes is not None,
        }
        if local_verified_via == "reviewed_inert_revision":
            component["local_sha256"] = local_sha256
            component["reviewed_inert_reason"] = inert_revision["reason"]
        runtime_components[relative_path] = component

    raw_logs = {}
    for profile in ("small", "large"):
        path = (
            root
            / "results"
            / "evolution"
            / f"weight5_css_{profile}_v1"
            / "all_codes_weight5_css.jsonl"
        )
        raw_logs[profile] = {
            "path": str(path.relative_to(root)),
            "sha256": _sha256_path(path),
            "records": sum(1 for _ in path.open(encoding="utf-8")),
        }

    return {
        "evaluator": {
            "path": HISTORICAL_EVALUATOR_PATH,
            "commit": HISTORICAL_EVALUATOR_COMMIT,
            "sha256": HISTORICAL_EVALUATOR_SHA256,
            "git_object_verified": evaluator_git_verified,
        },
        "runtime_components": runtime_components,
        "manifests": manifests,
        "raw_positive_observation_logs": raw_logs,
        "recovered_distance_quotas": {
            f"{ell}x{m}": quota
            for (ell, m), quota in sorted(recovered.items())
        },
    }


def protocol_manifest(root: Path = ROOT, verify_sources: bool = True) -> dict[str, Any]:
    source_evidence = verify_historical_sources(root) if verify_sources else None
    versions = {}
    for package in ("qldpc", "ldpc", "numpy", "scipy", "galois", "sympy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": "weight5_css_matched_budget_uniform_control_v1",
        "sampler": {
            "algorithm": SAMPLER_VERSION,
            "universe": "identity-anchored A binomial, B trinomial",
            "without_replacement": True,
            "conditional_selection": "first draws filling each exact-k band quota",
            "universe_sizes": {
                f"{ell}x{m}": universe_size(ell, m)
                for ell, m in PROFILE_LATTICES["all"]
            },
            "universe_size_total": sum(
                universe_size(ell, m) for ell, m in PROFILE_LATTICES["all"]
            ),
        },
        "distance_quotas": {
            f"{ell}x{m}": quota
            for (ell, m), quota in HISTORICAL_DISTANCE_QUOTAS.items()
        },
        "historical_gates": HISTORICAL_GATES,
        "historical_decoder": HISTORICAL_DECODER,
        "historical_milp": HISTORICAL_MILP,
        "software_versions_at_protocol_write": versions,
        "source_evidence": source_evidence,
        "known_replay_differences": [
            "The control samples the normalized universe directly, so generator Stage 1, "
            "the pre-build cap, and the fixed safety net do not apply.",
            "Historical decoder RNG state was not persisted; the control derives a stable "
            "per-candidate decoder seed.",
            "Historical quotas count repeated distance observations; the control uses the "
            "same counts but samples normalized pairs without replacement.",
        ],
    }


@dataclass(frozen=True)
class RunConfig:
    run_seed: int
    ell: int
    m: int
    quotas: dict[str, int]
    pilot: bool
    quick_trials: int
    refine_trials: int
    skip_exact: bool
    exact_timeout: int
    milp_timeout_per_logical: int
    milp_total_timeout: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "sampler_version": SAMPLER_VERSION,
            "run_seed": self.run_seed,
            "lattice": [self.ell, self.m],
            "universe_size": universe_size(self.ell, self.m),
            "quotas": self.quotas,
            "pilot": self.pilot,
            "matched_budget": self.quotas
            == HISTORICAL_DISTANCE_QUOTAS[(self.ell, self.m)],
            "historical_evaluator_sha256": HISTORICAL_EVALUATOR_SHA256,
            "distance": {
                "quick_trials": self.quick_trials,
                "refine_trials_per_batch": self.refine_trials,
                "refine_batches": 3,
                "fom_threshold_refine": 3.0,
                "fom_threshold_exact": 5.0,
                "osd_cs_trials": 200,
                "skip_exact": self.skip_exact,
                "exact_timeout_seconds": self.exact_timeout,
                "decoder_rng": "sha256-derived per-candidate numpy Generator",
            },
            "certification": {
                "algorithm": "css_weight_le_4_syndrome_quotient_mitm_v1",
                # The d>=5 subset is the certification target.  Results with
                # decoder d=3/4 are also passed through the same fast audit so
                # the retained-class denominator is fully canonicalized.
                "selection": "decoder upper bound d>=5",
            },
            "milp": {
                "selection": "estimated FOM>=5 and certified d>=5",
                "timeout_per_logical_seconds": self.milp_timeout_per_logical,
                "total_timeout_seconds": self.milp_total_timeout,
                "early_stop": 4,
            },
        }

    @property
    def digest(self) -> str:
        return _sha256_bytes(_canonical_json(self.as_dict()))


def _lattice_dir(output_dir: Path, config: RunConfig) -> Path:
    return output_dir / f"seed_{config.run_seed}" / f"{config.ell}x{config.m}"


def _ensure_run_config(path: Path, config: RunConfig) -> None:
    expected = {"config_sha256": config.digest, "config": config.as_dict()}
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(
                f"run configuration mismatch at {path}; use another output directory"
            )
        return
    _atomic_write_json(path, expected)


def _selection_state(draws: Sequence[dict[str, Any]]) -> tuple[Counter[str], set[int]]:
    counts: Counter[str] = Counter()
    indices: set[int] = set()
    for expected_draw, row in enumerate(draws):
        if int(row["draw_number_zero_based"]) != expected_draw:
            raise ValueError("draw log is not contiguous")
        index = int(row["universe_index"])
        if index in indices:
            raise ValueError("draw log contains a repeated universe index")
        indices.add(index)
        if row.get("selected"):
            counts[str(row["band"])] += 1
    return counts, indices


def _compute_k(ell: int, m: int, A_terms, B_terms) -> tuple[int, int]:
    from evaluation.bb_code import get_code_params_fast

    code = _build_complete_css_code(ell, m, A_terms, B_terms)
    return get_code_params_fast(code)


def _build_complete_css_code(ell: int, m: int, A_terms, B_terms):
    """Build every valid 2+3 presentation, including one-variable supports.

    qLDPC's ``BBCode`` constructor infers its variables from the polynomial
    expressions and rejects a valid presentation when both supports happen to
    use only one of ``x`` and ``y``.  The generic ``CSSCode`` fallback below
    constructs exactly the same regular-representation check matrices without
    relying on that symbol inference.
    """
    from evaluation.bb_code import build_bb_code, validate_terms

    validate_terms(ell, m, A_terms, "A")
    validate_terms(ell, m, B_terms, "B")
    uses_x = any(int(dx) != 0 for terms in (A_terms, B_terms) for dx, _ in terms)
    uses_y = any(int(dy) != 0 for terms in (A_terms, B_terms) for _, dy in terms)
    if uses_x and uses_y:
        return build_bb_code(ell, m, A_terms, B_terms)

    import numpy as np
    from qldpc.codes import CSSCode

    group_order = ell * m
    matrix_x = np.zeros((group_order, 2 * group_order), dtype=np.uint8)
    matrix_z = np.zeros((group_order, 2 * group_order), dtype=np.uint8)

    def shifted(index: int, dx: int, dy: int) -> int:
        x_coord, y_coord = divmod(index, m)
        return ((x_coord + dx) % ell) * m + (y_coord + dy) % m

    for row in range(group_order):
        for dx, dy in A_terms:
            matrix_x[row, shifted(row, int(dx), int(dy))] ^= 1
            matrix_z[row, group_order + shifted(row, -int(dx), -int(dy))] ^= 1
        for dx, dy in B_terms:
            matrix_x[row, group_order + shifted(row, int(dx), int(dy))] ^= 1
            matrix_z[row, shifted(row, -int(dx), -int(dy))] ^= 1
    return CSSCode(matrix_x, matrix_z, promise_equal_distance_xz=True)


def _selection_k_worker(payload: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        n, k = _compute_k(
            payload["ell"],
            payload["m"],
            payload["A_terms"],
            payload["B_terms"],
        )
        error = None
    except Exception as exc:
        n, k = 2 * payload["ell"] * payload["m"], 0
        error = f"{type(exc).__name__}: {exc}"
    return {**payload, "n": n, "k": k, "error": error}


def run_selection(
    output_dir: Path,
    config: RunConfig,
    *,
    max_new_draws: int | None = None,
    k_function: Callable[[int, int, Any, Any], tuple[int, int]] = _compute_k,
    flush_every: int = 25,
    selection_workers: int = 1,
) -> dict[str, Any]:
    """Resume selection for one seed/lattice and return its checkpoint."""
    lattice_dir = _lattice_dir(output_dir, config)
    _ensure_run_config(lattice_dir / "config.json", config)
    draws_path = lattice_dir / "draws.jsonl"
    checkpoint_path = lattice_dir / "selection_checkpoint.json"
    draws = _read_jsonl(draws_path)
    selected_counts, prior_indices = _selection_state(draws)

    permutation = LazyFisherYates(
        universe_size(config.ell, config.m),
        _derived_seed(config.run_seed, config.ell, config.m, "selection-v1"),
    )
    permutation.advance(len(draws))
    # Detect a draw log produced by a different sampler even if a config file
    # was manually copied: replay the final prefix element.
    if draws:
        replay = LazyFisherYates(
            universe_size(config.ell, config.m),
            _derived_seed(config.run_seed, config.ell, config.m, "selection-v1"),
        )
        last = None
        for _ in range(len(draws)):
            last = next(replay)
        if last != int(draws[-1]["universe_index"]):
            raise ValueError("draw log does not match the configured seeded permutation")

    started = time.monotonic()
    new_draws = 0
    buffer: list[dict[str, Any]] = []

    def complete() -> bool:
        return all(
            selected_counts[band] >= target for band, target in config.quotas.items()
        )

    from evaluation.connectivity import bicycle_translation_component_count

    pool = (
        ProcessPoolExecutor(max_workers=selection_workers)
        if selection_workers > 1 and k_function is _compute_k
        else None
    )
    try:
        while not complete():
            if max_new_draws is not None and new_draws >= max_new_draws:
                break
            remaining_limit = (
                max_new_draws - new_draws if max_new_draws is not None else None
            )
            batch_size = 1 if pool is None else max(64, selection_workers * 8)
            if remaining_limit is not None:
                batch_size = min(batch_size, remaining_limit)
            batch_size = min(batch_size, permutation.population - permutation.position)
            if batch_size <= 0:
                break

            payloads = []
            for _ in range(batch_size):
                universe_index = next(permutation)
                if universe_index in prior_indices:
                    raise AssertionError("seeded permutation repeated an existing draw")
                A_terms, B_terms = unrank_normalized_pair(
                    config.ell, config.m, universe_index
                )
                payloads.append(
                    {
                        "ell": config.ell,
                        "m": config.m,
                        "universe_index": universe_index,
                        "A_terms": A_terms,
                        "B_terms": B_terms,
                    }
                )

            if pool is None:
                computed = []
                for payload in payloads:
                    try:
                        n, k = k_function(
                            config.ell,
                            config.m,
                            payload["A_terms"],
                            payload["B_terms"],
                        )
                        error = None
                    except Exception as exc:
                        n, k = 2 * config.ell * config.m, 0
                        error = f"{type(exc).__name__}: {exc}"
                    computed.append({**payload, "n": n, "k": k, "error": error})
            else:
                computed = list(
                    pool.map(_selection_k_worker, payloads, chunksize=8)
                )

            for item in computed:
                if complete():
                    break
                universe_index = int(item["universe_index"])
                A_terms, B_terms = item["A_terms"], item["B_terms"]
                n, k, error = int(item["n"]), int(item["k"]), item["error"]
                band = k_band(k)
                selected = bool(
                    band in config.quotas
                    and selected_counts[band] < config.quotas[band]
                )
                if selected:
                    selected_counts[band] += 1
                component_count = (
                    bicycle_translation_component_count(
                        config.ell, config.m, A_terms, B_terms
                    )
                    if selected
                    else None
                )
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "record_type": "uniform_control_draw",
                    "config_sha256": config.digest,
                    "run_seed": config.run_seed,
                    "ell": config.ell,
                    "m": config.m,
                    "draw_number_zero_based": len(draws) + new_draws,
                    "universe_index": universe_index,
                    "A_terms": [list(term) for term in A_terms],
                    "B_terms": [list(term) for term in B_terms],
                    "n": n,
                    "k": k,
                    "band": band,
                    "selected": selected,
                    "translation_component_count": component_count,
                    "translation_connected": component_count == 1 if selected else None,
                    "error": error,
                }
                buffer.append(row)
                prior_indices.add(universe_index)
                new_draws += 1
                if len(buffer) >= flush_every:
                    _append_jsonl(draws_path, buffer)
                    buffer.clear()
    finally:
        if pool is not None:
            pool.shutdown()
    _append_jsonl(draws_path, buffer)

    status = "complete" if complete() else (
        "universe_exhausted" if permutation.position == permutation.population else "in_progress"
    )
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "config_sha256": config.digest,
        "status": status,
        "draws": len(draws) + new_draws,
        "new_draws_this_invocation": new_draws,
        "selected_by_band": dict(selected_counts),
        "quotas": config.quotas,
        "acceptance_rate": (
            sum(selected_counts.values()) / (len(draws) + new_draws)
            if draws or new_draws
            else 0.0
        ),
        "elapsed_seconds_this_invocation": time.monotonic() - started,
    }
    _atomic_write_json(checkpoint_path, checkpoint)
    return checkpoint


_ONE_VARIABLE_CONSTRUCTOR_ERROR = (
    "BBCodes should have exactly two cyclic group orders and two symbols"
)


def repair_one_variable_rank_records(
    output_dir: Path, config: RunConfig
) -> dict[str, Any]:
    """Repair the historical constructor's one-variable symbol-inference gap.

    The seeded universe and rank-band stopping rule are replayed exactly.  Raw
    records superseded by the repair are retained under ``repair/``; numerical
    distance/certification rows for still-selected indices are reused.
    """
    lattice_dir = _lattice_dir(output_dir, config)
    _ensure_run_config(lattice_dir / "config.json", config)
    draws_path = lattice_dir / "draws.jsonl"
    original_draws = _read_jsonl(draws_path)
    error_rows = [row for row in original_draws if row.get("error")]
    if not error_rows:
        return {
            "status": "already_clean",
            "run_seed": config.run_seed,
            "lattice": [config.ell, config.m],
            "repaired_rank_rows": 0,
        }
    if any(
        _ONE_VARIABLE_CONSTRUCTOR_ERROR not in str(row.get("error"))
        for row in error_rows
    ):
        raise ValueError(f"unexpected selection error in {draws_path}")
    if any(row.get("selected") for row in error_rows):
        raise ValueError(f"errored draw was selected in {draws_path}")

    repaired_by_index: dict[int, dict[str, Any]] = {}
    for original in error_rows:
        row = dict(original)
        n, k = _compute_k(config.ell, config.m, row["A_terms"], row["B_terms"])
        row.update(n=int(n), k=int(k), band=k_band(int(k)), error=None)
        repaired_by_index[int(row["universe_index"])] = row

    selected_counts: Counter[str] = Counter()
    repaired_draws: list[dict[str, Any]] = []
    superseded_draws: list[dict[str, Any]] = []
    from evaluation.connectivity import bicycle_translation_component_count

    for original in original_draws:
        index = int(original["universe_index"])
        row = dict(repaired_by_index.get(index, original))
        band = str(row.get("band"))
        selected = bool(
            band in config.quotas
            and selected_counts[band] < int(config.quotas[band])
        )
        if selected:
            selected_counts[band] += 1
            component_count = bicycle_translation_component_count(
                config.ell, config.m, row["A_terms"], row["B_terms"]
            )
        else:
            component_count = None
        row.update(
            draw_number_zero_based=len(repaired_draws),
            selected=selected,
            translation_component_count=component_count,
            translation_connected=(component_count == 1 if selected else None),
        )
        if row != original:
            superseded_draws.append(original)
        repaired_draws.append(row)
        if all(
            selected_counts[band_name] == int(target)
            for band_name, target in config.quotas.items()
        ):
            break
    if dict(selected_counts) != config.quotas:
        raise ValueError(f"repaired draw prefix does not fill quotas: {lattice_dir}")
    superseded_draws.extend(original_draws[len(repaired_draws) :])

    old_selected = {
        int(row["universe_index"]) for row in original_draws if row.get("selected")
    }
    new_selected = {
        int(row["universe_index"]) for row in repaired_draws if row.get("selected")
    }
    added = sorted(new_selected - old_selected)
    removed = sorted(old_selected - new_selected)

    stage_paths = {
        "distance": lattice_dir / "distance.jsonl",
        "certification": lattice_dir / "certification.jsonl",
        "milp": lattice_dir / "milp.jsonl",
    }
    retained_stage_rows: dict[str, list[dict[str, Any]]] = {}
    superseded_stage_rows: dict[str, list[dict[str, Any]]] = {}
    before_sha256 = {"draws": _sha256_path(draws_path)}
    for stage, path in stage_paths.items():
        rows = _read_jsonl(path)
        before_sha256[stage] = _sha256_path(path) if path.exists() else None
        retained_stage_rows[stage] = [
            row for row in rows if int(row["universe_index"]) in new_selected
        ]
        superseded_stage_rows[stage] = [
            row for row in rows if int(row["universe_index"]) not in new_selected
        ]

    repair_dir = lattice_dir / "repair" / "one_variable_rank_v1"
    _atomic_write_jsonl(repair_dir / "superseded_draws.jsonl", superseded_draws)
    for stage, rows in superseded_stage_rows.items():
        _atomic_write_jsonl(repair_dir / f"superseded_{stage}.jsonl", rows)

    _atomic_write_jsonl(draws_path, repaired_draws)
    for stage, path in stage_paths.items():
        _atomic_write_jsonl(path, retained_stage_rows[stage])

    distance_count = len(retained_stage_rows["distance"])
    certification_count = len(retained_stage_rows["certification"])
    milp_count = len(retained_stage_rows["milp"])
    _atomic_write_json(
        lattice_dir / "selection_checkpoint.json",
        {
            "schema_version": SCHEMA_VERSION,
            "config_sha256": config.digest,
            "status": "complete",
            "draws": len(repaired_draws),
            "selected_by_band": dict(selected_counts),
            "quotas": config.quotas,
            "acceptance_rate": len(new_selected) / len(repaired_draws),
            "repair": "one_variable_rank_v1",
        },
    )
    _atomic_write_json(
        lattice_dir / "distance_checkpoint.json",
        {
            "schema_version": SCHEMA_VERSION,
            "config_sha256": config.digest,
            "status": "complete" if distance_count == len(new_selected) else "in_progress",
            "distance_records": distance_count,
            "distance_budget": len(new_selected),
            "repair": "one_variable_rank_v1",
        },
    )
    _atomic_write_json(
        lattice_dir / "certification_checkpoint.json",
        {
            "schema_version": SCHEMA_VERSION,
            "config_sha256": config.digest,
            "status": "in_progress",
            "certification_records": certification_count,
            "repair": "one_variable_rank_v1",
        },
    )
    _atomic_write_json(
        lattice_dir / "milp_checkpoint.json",
        {
            "schema_version": SCHEMA_VERSION,
            "config_sha256": config.digest,
            "status": "in_progress",
            "milp_records": milp_count,
            "repair": "one_variable_rank_v1",
        },
    )

    result = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "uniform_control_one_variable_rank_repair",
        "status": "repaired",
        "run_seed": config.run_seed,
        "lattice": [config.ell, config.m],
        "repaired_rank_rows": len(error_rows),
        "original_draws": len(original_draws),
        "repaired_draws": len(repaired_draws),
        "selected_added": added,
        "selected_removed": removed,
        "superseded_stage_records": {
            stage: len(rows) for stage, rows in superseded_stage_rows.items()
        },
        "before_sha256": before_sha256,
        "after_sha256": {
            "draws": _sha256_path(draws_path),
            **{
                stage: _sha256_path(path)
                for stage, path in stage_paths.items()
                if path.exists()
            },
        },
    }
    _atomic_write_json(repair_dir / "manifest.json", result)
    return result


@contextmanager
def _deterministic_qldpc_random(seed: int) -> Iterator[None]:
    """Give qLDPC's randomized logical selection an explicit RNG stream."""
    import numpy as np
    from qldpc.codes import CSSCode

    method_globals = CSSCode.get_distance_bound_with_decoder.__globals__
    original = method_globals["get_random_array"]
    rng = np.random.default_rng(seed)

    def deterministic_random_array(field, shape, *, satisfy=lambda _: True, seed=None):
        local_rng = rng if seed is None else np.random.default_rng(seed)
        while not satisfy(array := field.Random(shape, seed=local_rng)):
            pass
        return array

    method_globals["get_random_array"] = deterministic_random_array
    try:
        yield
    finally:
        method_globals["get_random_array"] = original


def _distance_worker(payload: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    os.environ["QCODE_MIN_K_THRESHOLD"] = "2"
    os.environ["QCODE_SAVE_TRUST_RATIO"] = "2.0"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import evaluation.evaluator as historical_cascade

    started = time.monotonic()
    # The historical evaluator uses one threshold for both OSD-CS and exact
    # enumeration.  For a bounded pilot we still want the historical OSD-CS
    # pass, so disable only the exact call rather than raising that shared
    # threshold to infinity.
    original_exact = historical_cascade.compute_distance_exact
    original_builder = historical_cascade.build_bb_code
    historical_cascade.build_bb_code = _build_complete_css_code
    if payload["skip_exact"]:
        historical_cascade.compute_distance_exact = lambda *_args, **_kwargs: None
    try:
        with _deterministic_qldpc_random(payload["decoder_seed"]):
            result = historical_cascade.evaluate_candidate(
                payload["ell"],
                payload["m"],
                payload["A_terms"],
                payload["B_terms"],
                quick=False,
                fom_threshold_refine=3.0,
                fom_threshold_exact=5.0,
                quick_trials=payload["quick_trials"],
                refine_trials=payload["refine_trials"],
                exact_timeout=payload["exact_timeout"],
            )
        if payload["skip_exact"] and result.get("stage") == "exact_timeout":
            # The historical evaluator uses this label whenever its exact
            # branch returns ``None``.  In the control that branch is
            # deliberately disabled, so record what actually happened.
            result = {**result, "stage": "exact_skipped"}
        error = None
    except Exception as exc:
        result = {
            "ell": payload["ell"],
            "m": payload["m"],
            "A_terms": payload["A_terms"],
            "B_terms": payload["B_terms"],
            "n": 2 * payload["ell"] * payload["m"],
            "k": payload["k"],
            "d": 0,
            "d_is_exact": False,
            "fom": 0.0,
            "stage": "worker_error",
        }
        error = f"{type(exc).__name__}: {exc}"
    finally:
        historical_cascade.compute_distance_exact = original_exact
        historical_cascade.build_bb_code = original_builder
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "uniform_control_distance",
        "config_sha256": payload["config_sha256"],
        "run_seed": payload["run_seed"],
        "universe_index": payload["universe_index"],
        "decoder_seed": payload["decoder_seed"],
        "decoder_rng_controlled": True,
        "exact_search_skipped": bool(payload["skip_exact"]),
        "elapsed_seconds": time.monotonic() - started,
        "error": error,
        **result,
    }


def _records_by_index(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        index = int(row["universe_index"])
        if index in records:
            raise ValueError(f"duplicate universe index {index} in {path}")
        records[index] = row
    return records


def _run_parallel_stage(
    *,
    tasks: Sequence[dict[str, Any]],
    worker: Callable[[dict[str, Any]], dict[str, Any]],
    output_path: Path,
    workers: int,
) -> int:
    if not tasks:
        return 0
    completed = 0
    if workers <= 1:
        for task in tasks:
            _append_jsonl(output_path, [worker(task)])
            completed += 1
        return completed
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "record_type": "uniform_control_worker_failure",
                    "config_sha256": task["config_sha256"],
                    "run_seed": task["run_seed"],
                    "ell": task["ell"],
                    "m": task["m"],
                    "universe_index": task["universe_index"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            _append_jsonl(output_path, [row])
            completed += 1
    return completed


def run_distance(
    output_dir: Path,
    config: RunConfig,
    *,
    workers: int,
    max_new: int | None = None,
) -> dict[str, Any]:
    lattice_dir = _lattice_dir(output_dir, config)
    _ensure_run_config(lattice_dir / "config.json", config)
    selected = [row for row in _read_jsonl(lattice_dir / "draws.jsonl") if row["selected"]]
    if len(selected) != sum(config.quotas.values()):
        raise RuntimeError(
            f"selection incomplete for {config.ell}x{config.m}: "
            f"{len(selected)}/{sum(config.quotas.values())}"
        )
    output_path = lattice_dir / "distance.jsonl"
    existing = _records_by_index(output_path)
    pending = [row for row in selected if int(row["universe_index"]) not in existing]
    if max_new is not None:
        pending = pending[:max_new]
    tasks = [
        {
            "config_sha256": config.digest,
            "run_seed": config.run_seed,
            "ell": config.ell,
            "m": config.m,
            "universe_index": int(row["universe_index"]),
            "A_terms": row["A_terms"],
            "B_terms": row["B_terms"],
            "k": int(row["k"]),
            "decoder_seed": _decoder_seed(
                config.run_seed, config.ell, config.m, int(row["universe_index"])
            ),
            "quick_trials": config.quick_trials,
            "refine_trials": config.refine_trials,
            "skip_exact": config.skip_exact,
            "exact_timeout": config.exact_timeout,
        }
        for row in pending
    ]
    started = time.monotonic()
    completed = _run_parallel_stage(
        tasks=tasks, worker=_distance_worker, output_path=output_path, workers=workers
    )
    total = len(existing) + completed
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "config_sha256": config.digest,
        "status": "complete" if total == len(selected) else "in_progress",
        "distance_records": total,
        "distance_budget": len(selected),
        "new_records_this_invocation": completed,
        "elapsed_seconds_this_invocation": time.monotonic() - started,
    }
    _atomic_write_json(lattice_dir / "distance_checkpoint.json", checkpoint)
    return checkpoint


def _certification_worker(payload: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    started = time.monotonic()
    try:
        from scripts.audit_weight5_css_low_weight import (
            audit_css_rows,
            build_bb_check_rows,
        )
        from evaluation.tanner_equivalence import canonical_digest

        rows_x, rows_z = build_bb_check_rows(
            payload["ell"], payload["m"], payload["A_terms"], payload["B_terms"]
        )
        audit = audit_css_rows(rows_x, rows_z, 2 * payload["ell"] * payload["m"])
        code = _build_complete_css_code(
            payload["ell"], payload["m"], payload["A_terms"], payload["B_terms"]
        )
        digest = canonical_digest(code)
        error = None
    except Exception as exc:
        audit = {
            "result": "worker_error",
            "exact_distance": None,
            "certified_lower_bound": None,
        }
        digest = None
        error = f"{type(exc).__name__}: {exc}"
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "uniform_control_low_weight_certification",
        "config_sha256": payload["config_sha256"],
        "run_seed": payload["run_seed"],
        "ell": payload["ell"],
        "m": payload["m"],
        "universe_index": payload["universe_index"],
        "A_terms": payload["A_terms"],
        "B_terms": payload["B_terms"],
        "n": payload["n"],
        "k": payload["k"],
        "estimated_d": payload["d"],
        "estimated_fom": payload["fom"],
        "canonical_digest_sha256": digest,
        "elapsed_seconds": time.monotonic() - started,
        "error": error,
        **audit,
    }


def run_certification(
    output_dir: Path,
    config: RunConfig,
    *,
    workers: int,
    max_new: int | None = None,
) -> dict[str, Any]:
    lattice_dir = _lattice_dir(output_dir, config)
    _ensure_run_config(lattice_dir / "config.json", config)
    distances = list(_records_by_index(lattice_dir / "distance.jsonl").values())
    if len(distances) != sum(config.quotas.values()):
        raise RuntimeError(f"distance stage incomplete for {config.ell}x{config.m}")
    targets = [row for row in distances if int(row.get("d", 0)) > 2]
    estimated_ge5_targets = sum(int(row.get("d", 0)) >= 5 for row in targets)
    output_path = lattice_dir / "certification.jsonl"
    existing = _records_by_index(output_path)
    pending = [row for row in targets if int(row["universe_index"]) not in existing]
    if max_new is not None:
        pending = pending[:max_new]
    tasks = [
        {
            "config_sha256": config.digest,
            "run_seed": config.run_seed,
            "ell": config.ell,
            "m": config.m,
            "universe_index": int(row["universe_index"]),
            "A_terms": row["A_terms"],
            "B_terms": row["B_terms"],
            "n": int(row["n"]),
            "k": int(row["k"]),
            "d": int(row["d"]),
            "fom": float(row["fom"]),
        }
        for row in pending
    ]
    started = time.monotonic()
    completed = _run_parallel_stage(
        tasks=tasks,
        worker=_certification_worker,
        output_path=output_path,
        workers=workers,
    )
    total = len(existing) + completed
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "config_sha256": config.digest,
        "status": "complete" if total == len(targets) else "in_progress",
        "certification_records": total,
        "retained_d_gt_2_targets": len(targets),
        "estimated_d_ge_5_targets": estimated_ge5_targets,
        "new_records_this_invocation": completed,
        "elapsed_seconds_this_invocation": time.monotonic() - started,
    }
    _atomic_write_json(lattice_dir / "certification_checkpoint.json", checkpoint)
    return checkpoint


def _milp_worker(payload: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    os.environ["QCODE_MIN_K_THRESHOLD"] = "2"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import evaluation.evaluator as evaluator

    started = time.monotonic()
    original_builder = evaluator.build_bb_code
    evaluator.build_bb_code = _build_complete_css_code
    try:
        result = evaluator.evaluate_candidate_milp(
            payload["ell"],
            payload["m"],
            payload["A_terms"],
            payload["B_terms"],
            milp_timeout_per_logical=payload["timeout_per_logical"],
            milp_total_timeout=payload["total_timeout"],
            milp_early_stop=4,
        )
        error = None
    except Exception as exc:
        result = {
            "ell": payload["ell"],
            "m": payload["m"],
            "A_terms": payload["A_terms"],
            "B_terms": payload["B_terms"],
            "n": 2 * payload["ell"] * payload["m"],
            "k": payload["k"],
            "d": 0,
            "d_is_exact": False,
            "fom": 0.0,
            "stage": "worker_error",
        }
        error = f"{type(exc).__name__}: {exc}"
    finally:
        evaluator.build_bb_code = original_builder
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "uniform_control_milp",
        "config_sha256": payload["config_sha256"],
        "run_seed": payload["run_seed"],
        "universe_index": payload["universe_index"],
        "elapsed_seconds": time.monotonic() - started,
        "error": error,
        **result,
    }


def run_milp(
    output_dir: Path,
    config: RunConfig,
    *,
    workers: int,
    max_new: int | None = None,
) -> dict[str, Any]:
    lattice_dir = _lattice_dir(output_dir, config)
    _ensure_run_config(lattice_dir / "config.json", config)
    distances = _records_by_index(lattice_dir / "distance.jsonl")
    certifications = _records_by_index(lattice_dir / "certification.jsonl")
    targets = []
    for index, certification in certifications.items():
        distance = distances[index]
        if (
            float(distance.get("fom", 0.0)) >= 5.0
            and int(certification.get("certified_lower_bound") or 0) >= 5
        ):
            targets.append(distance)
    targets.sort(key=lambda row: (-float(row.get("fom", 0)), int(row["universe_index"])))
    output_path = lattice_dir / "milp.jsonl"
    existing = _records_by_index(output_path)
    pending = [row for row in targets if int(row["universe_index"]) not in existing]
    if max_new is not None:
        pending = pending[:max_new]
    tasks = [
        {
            "config_sha256": config.digest,
            "run_seed": config.run_seed,
            "ell": config.ell,
            "m": config.m,
            "universe_index": int(row["universe_index"]),
            "A_terms": row["A_terms"],
            "B_terms": row["B_terms"],
            "k": int(row["k"]),
            "timeout_per_logical": config.milp_timeout_per_logical,
            "total_timeout": config.milp_total_timeout,
        }
        for row in pending
    ]
    started = time.monotonic()
    completed = _run_parallel_stage(
        tasks=tasks, worker=_milp_worker, output_path=output_path, workers=workers
    )
    total = len(existing) + completed
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "config_sha256": config.digest,
        "status": "complete" if total == len(targets) else "in_progress",
        "milp_records": total,
        "finalists": len(targets),
        "new_records_this_invocation": completed,
        "elapsed_seconds_this_invocation": time.monotonic() - started,
    }
    _atomic_write_json(lattice_dir / "milp_checkpoint.json", checkpoint)
    return checkpoint


def _historical_overlap_sets(root: Path = ROOT) -> dict[str, set[Any]]:
    positive_literal: set[Any] = set()
    positive_translation: set[Any] = set()
    for profile in ("small", "large"):
        path = (
            root
            / "results"
            / "evolution"
            / f"weight5_css_{profile}_v1"
            / "all_codes_weight5_css.jsonl"
        )
        for row in _read_jsonl(path):
            literal = literal_candidate_key(
                int(row["ell"]), int(row["m"]), row["A_terms"], row["B_terms"]
            )
            positive_literal.add(literal)
            positive_translation.add(
                translation_candidate_key(
                    int(row["ell"]), int(row["m"]), row["A_terms"], row["B_terms"]
                )
            )

    retained_literal: set[Any] = set()
    retained_translation: set[Any] = set()
    retained_classes: set[str] = set()
    catalogue = root / "results" / "weight5_publication_catalogue.jsonl"
    for row in _read_jsonl(catalogue):
        if row.get("family") != "CSS":
            continue
        params = row["parameters"]
        generators = row["generators"]
        literal = literal_candidate_key(
            int(params["ell"]),
            int(params["m"]),
            generators["A_terms"],
            generators["B_terms"],
        )
        retained_literal.add(literal)
        retained_translation.add(
            translation_candidate_key(
                int(params["ell"]),
                int(params["m"]),
                generators["A_terms"],
                generators["B_terms"],
            )
        )
        digest = row.get("equivalence", {}).get("canonical_digest_sha256")
        if digest:
            retained_classes.add(str(digest))
    return {
        "positive_literal": positive_literal,
        "positive_translation": positive_translation,
        "retained_literal": retained_literal,
        "retained_translation": retained_translation,
        "retained_classes": retained_classes,
    }


def _catalogue_exact_class_evidence(
    root: Path = ROOT,
) -> dict[str, dict[str, Any]]:
    """Return exact CSS evidence indexed by presentation-class digest.

    A colored-Tanner presentation isomorphism is a qubit/check permutation, so
    exact distance transfers within a digest.  Keeping this evidence separate
    from control-arm MILP output makes its provenance explicit.
    """
    evidence: dict[str, dict[str, Any]] = {}
    catalogue = root / "results" / "weight5_publication_catalogue.jsonl"
    for row in _read_jsonl(catalogue):
        if row.get("family") != "CSS" or not row["distance_class"].get("is_exact"):
            continue
        digest = str(row["equivalence"]["canonical_digest_sha256"])
        item = {
            "canonical_digest_sha256": digest,
            "class_id": row["equivalence"].get("class_id"),
            "parameters": [
                int(row["parameters"]["n"]),
                int(row["parameters"]["k"]),
                int(row["distance_class"]["lower"]),
            ],
            "presentation_id": row["presentation_id"],
        }
        previous = evidence.setdefault(digest, item)
        if previous["parameters"] != item["parameters"]:
            raise ValueError(f"inconsistent exact catalogue class {digest}")
    return evidence


def _best_row(
    rows: Sequence[dict[str, Any]],
    field: str = "fom",
    *,
    exact_search_disabled: bool | None = None,
) -> dict[str, Any] | None:
    if not rows:
        return None
    row = max(rows, key=lambda item: float(item.get(field, 0.0)))
    return {
        "universe_index": int(row["universe_index"]),
        "parameters": [int(row["n"]), int(row["k"]), int(row.get("d", 0))],
        "fom": float(row.get(field, 0.0)),
        "A_terms": row["A_terms"],
        "B_terms": row["B_terms"],
        "stage": _effective_distance_stage(row, exact_search_disabled),
    }


def _effective_distance_stage(
    row: dict[str, Any], exact_search_disabled: bool | None = None
) -> str | None:
    """Normalize the legacy evaluator label when exact search was disabled."""
    stage = row.get("stage")
    disabled = bool(
        row.get("exact_search_skipped")
        if "exact_search_skipped" in row
        else exact_search_disabled
    )
    if disabled and stage == "exact_timeout":
        return "exact_skipped"
    return str(stage) if stage is not None else None


def summarize_lattice(
    output_dir: Path, config: RunConfig, overlap: dict[str, set[Any]] | None = None
) -> dict[str, Any]:
    lattice_dir = _lattice_dir(output_dir, config)
    _ensure_run_config(lattice_dir / "config.json", config)
    draws = _read_jsonl(lattice_dir / "draws.jsonl")
    selected = [row for row in draws if row.get("selected")]
    distances = _records_by_index(lattice_dir / "distance.jsonl")
    certifications = _records_by_index(lattice_dir / "certification.jsonl")
    milp = _records_by_index(lattice_dir / "milp.jsonl")
    overlap = overlap or _historical_overlap_sets(ROOT)

    selected_keys_literal = {
        literal_candidate_key(config.ell, config.m, row["A_terms"], row["B_terms"])
        for row in selected
    }
    selected_keys_translation = {
        translation_candidate_key(config.ell, config.m, row["A_terms"], row["B_terms"])
        for row in selected
    }
    estimated_ge5 = [row for row in distances.values() if int(row.get("d", 0)) >= 5]
    certification_targets = [
        row for row in distances.values() if int(row.get("d", 0)) > 2
    ]
    retained_certifications = [
        row
        for row in certifications.values()
        if int(row.get("certified_lower_bound") or 0) > 2
    ]
    certified_ge5 = [
        row
        for row in certifications.values()
        if int(row.get("certified_lower_bound") or 0) >= 5
    ]
    selected_by_index = {int(row["universe_index"]): row for row in selected}
    connected_certified = [
        row
        for row in certified_ge5
        if selected_by_index[int(row["universe_index"])].get("translation_connected")
    ]
    connected_retained = [
        row
        for row in retained_certifications
        if selected_by_index[int(row["universe_index"])].get("translation_connected")
    ]
    retained_class_digests = {
        row["canonical_digest_sha256"]
        for row in retained_certifications
        if row.get("canonical_digest_sha256")
    }
    connected_retained_class_digests = {
        row["canonical_digest_sha256"]
        for row in connected_retained
        if row.get("canonical_digest_sha256")
    }
    connected_class_digests = {
        row["canonical_digest_sha256"]
        for row in connected_certified
        if row.get("canonical_digest_sha256")
    }
    all_certified_class_digests = {
        row["canonical_digest_sha256"]
        for row in certified_ge5
        if row.get("canonical_digest_sha256")
    }

    exact_milp = [row for row in milp.values() if row.get("d_is_exact")]
    certified_lower_rows = []
    for row in certified_ge5:
        lower = int(row["certified_lower_bound"])
        certified_lower_rows.append(
            {
                **row,
                "d": lower,
                "fom_lower": int(row["k"]) * lower * lower / int(row["n"]),
            }
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "uniform_control_lattice_summary",
        "config_sha256": config.digest,
        "run_seed": config.run_seed,
        "lattice": [config.ell, config.m],
        "pilot": config.pilot,
        "matched_budget": config.as_dict()["matched_budget"],
        "completion": {
            "selection": len(selected) == sum(config.quotas.values()),
            "distance": len(distances) == len(selected),
            "certification": len(certifications) == len(certification_targets),
            "milp": len(milp)
            == sum(
                1
                for row in certified_ge5
                if float(distances[int(row["universe_index"])].get("fom", 0)) >= 5
            ),
        },
        "sampling": {
            "universe_size": universe_size(config.ell, config.m),
            "draws": len(draws),
            "selected": len(selected),
            "quota": sum(config.quotas.values()),
            "selected_by_band": dict(Counter(str(row["band"]) for row in selected)),
            "rank_draw_k_histogram": {
                str(k): count
                for k, count in sorted(
                    Counter(int(row.get("k", 0)) for row in draws).items()
                )
            },
            "selected_k_histogram": {
                str(k): count
                for k, count in sorted(
                    Counter(int(row.get("k", 0)) for row in selected).items()
                )
            },
            "k_positive_draws": sum(int(row.get("k", 0)) > 0 for row in draws),
            "acceptance_rate": len(selected) / len(draws) if draws else 0.0,
        },
        "yield": {
            "retained_d_gt_2": len(retained_certifications),
            "connected_retained_presentations": len(connected_retained),
            "connected_retained_tanner_classes": len(
                connected_retained_class_digests
            ),
            "connected_retained_class_yield_per_distance_evaluation": (
                len(connected_retained_class_digests) / len(distances)
                if distances
                else 0.0
            ),
            "estimated_d_ge_5": len(estimated_ge5),
            "estimated_d_ge_5_rate_per_distance_evaluation": (
                len(estimated_ge5) / len(distances) if distances else 0.0
            ),
            "certified_d_ge_5": len(certified_ge5),
            "certified_d_ge_5_rate_per_distance_evaluation": (
                len(certified_ge5) / len(distances) if distances else 0.0
            ),
            "connected_certified_d_ge_5_presentations": len(connected_certified),
            "connected_certified_d_ge_5_classes": len(connected_class_digests),
            "connected_class_yield_per_distance_evaluation": (
                len(connected_class_digests) / len(distances) if distances else 0.0
            ),
        },
        "diversity": {
            "unique_literal_selected_pairs": len(selected_keys_literal),
            "unique_translation_classes_selected": len(selected_keys_translation),
            "distinct_k_values_selected": sorted({int(row["k"]) for row in selected}),
            "retained_d_gt_2_tanner_classes": len(retained_class_digests),
            "connected_retained_d_gt_2_tanner_classes": len(
                connected_retained_class_digests
            ),
            "certified_d_ge_5_tanner_classes": len(all_certified_class_digests),
            "connected_certified_d_ge_5_tanner_classes": len(connected_class_digests),
        },
        "overlap": {
            "selected_literal_in_historical_positive_log": len(
                selected_keys_literal & overlap["positive_literal"]
            ),
            "selected_translation_class_in_historical_positive_log": len(
                selected_keys_translation & overlap["positive_translation"]
            ),
            "selected_literal_in_retained_catalogue": len(
                selected_keys_literal & overlap["retained_literal"]
            ),
            "selected_translation_class_in_retained_catalogue": len(
                selected_keys_translation & overlap["retained_translation"]
            ),
            "certified_d_ge_5_tanner_classes_in_retained_catalogue": len(
                all_certified_class_digests & overlap["retained_classes"]
            ),
            "connected_certified_d_ge_5_tanner_classes_in_retained_catalogue": len(
                connected_class_digests & overlap["retained_classes"]
            ),
        },
        "best": {
            "estimated": _best_row(
                list(distances.values()), exact_search_disabled=config.skip_exact
            ),
            "certified_lower_bound": _best_row(certified_lower_rows, "fom_lower"),
            "milp_exact": _best_row(exact_milp),
        },
        "stage_counts": dict(
            Counter(
                str(_effective_distance_stage(row, config.skip_exact))
                for row in distances.values()
            )
        ),
        "errors": {
            "draw": sum(bool(row.get("error")) for row in draws),
            "distance": sum(bool(row.get("error")) for row in distances.values()),
            "certification": sum(
                bool(row.get("error")) for row in certifications.values()
            ),
            "milp": sum(bool(row.get("error")) for row in milp.values()),
        },
    }
    _atomic_write_json(lattice_dir / "summary.json", summary)
    return summary


def _campaign_tuple_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Orientation-preserving normalized tuple key used for the 936 count."""
    return (
        int(row["ell"]),
        int(row["m"]),
        _sorted_terms(row["A_terms"]),
        _sorted_terms(row["B_terms"]),
    )


def _best_distance_claim(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    best = max(rows, key=lambda row: float(row["fom"]))
    result = {
        "parameters": [int(best["n"]), int(best["k"]), int(best["d"])],
        "fom": float(best["fom"]),
        "presentation_id": best.get("presentation_id"),
        "run_seed": best.get("run_seed"),
        "lattice": [int(best["ell"]), int(best["m"])],
        "universe_index": best.get("universe_index"),
    }
    for field in (
        "canonical_digest_sha256",
        "class_id",
        "evidence_source",
        "bound_type",
    ):
        if best.get(field) is not None:
            result[field] = best[field]
    return result


def historical_llm_comparison(root: Path = ROOT) -> dict[str, Any]:
    """Reconstruct the defensible LLM-arm denominators from pinned artifacts."""
    observations: list[dict[str, Any]] = []
    for profile in ("small", "large"):
        path = (
            root
            / "results"
            / "evolution"
            / f"weight5_css_{profile}_v1"
            / "all_codes_weight5_css.jsonl"
        )
        observations.extend(_read_jsonl(path))
    unique_positive = {_campaign_tuple_key(row) for row in observations}
    observation_k_histogram = Counter(int(row["k"]) for row in observations)
    observation_k_histogram_by_lattice: dict[str, Counter[int]] = defaultdict(Counter)
    for row in observations:
        observation_k_histogram_by_lattice[f"{int(row['ell'])}x{int(row['m'])}"][
            int(row["k"])
        ] += 1

    catalogue_rows = []
    catalogue_by_key = {}
    for row in _read_jsonl(root / "results" / "weight5_publication_catalogue.jsonl"):
        if row.get("family") != "CSS":
            continue
        params = row["parameters"]
        generators = row["generators"]
        keyed = {
            "ell": params["ell"],
            "m": params["m"],
            "A_terms": generators["A_terms"],
            "B_terms": generators["B_terms"],
        }
        catalogue_by_key[_campaign_tuple_key(keyed)] = row
        catalogue_rows.append(row)

    mapped_observations = []
    for observation in observations:
        catalogue_row = catalogue_by_key.get(_campaign_tuple_key(observation))
        if catalogue_row is not None:
            mapped_observations.append(catalogue_row)

    def class_digests(predicate: Callable[[dict[str, Any]], bool]) -> set[str]:
        return {
            str(row["equivalence"]["canonical_digest_sha256"])
            for row in catalogue_rows
            if predicate(row)
        }

    connected_classes = class_digests(
        lambda row: row["connectivity"]["state"] == "connected"
    )
    certified_classes = class_digests(
        lambda row: int(row["distance_class"].get("lower") or 0) >= 5
    )
    connected_certified_classes = class_digests(
        lambda row: row["connectivity"]["state"] == "connected"
        and int(row["distance_class"].get("lower") or 0) >= 5
    )
    certified_unique = [
        row
        for row in catalogue_rows
        if int(row["distance_class"].get("lower") or 0) >= 5
    ]
    certified_observations = [
        row
        for row in mapped_observations
        if int(row["distance_class"].get("lower") or 0) >= 5
    ]

    exact_claims = []
    certified_claims = []
    for row in catalogue_rows:
        lower = row["distance_class"].get("lower")
        if lower is None:
            continue
        n = int(row["parameters"]["n"])
        k = int(row["parameters"]["k"])
        claim = {
            "ell": int(row["parameters"]["ell"]),
            "m": int(row["parameters"]["m"]),
            "n": n,
            "k": k,
            "d": int(lower),
            "fom": k * int(lower) ** 2 / n,
            "presentation_id": row["presentation_id"],
            "connected": row["connectivity"]["state"] == "connected",
        }
        certified_claims.append(claim)
        if row["distance_class"].get("is_exact"):
            exact_claims.append(claim)

    calls = len(observations)
    unique_count = len(unique_positive)
    return {
        "arm": "historical_llm_guided_campaign",
        "distance_observations": calls,
        "distance_observation_k_histogram": {
            str(k): count for k, count in sorted(observation_k_histogram.items())
        },
        "distance_observation_k_histogram_by_lattice": {
            lattice: {
                str(k): count for k, count in sorted(histogram.items())
            }
            for lattice, histogram in sorted(
                observation_k_histogram_by_lattice.items()
            )
        },
        "unique_positive_normalized_tuples": unique_count,
        "retained_d_gt_2_unique_tuples": len(catalogue_rows),
        "retained_tanner_classes": len(class_digests(lambda _row: True)),
        "connected_retained_presentations": sum(
            row["connectivity"]["state"] == "connected" for row in catalogue_rows
        ),
        "connected_retained_tanner_classes": len(connected_classes),
        "certified_d_ge_5_observations": len(certified_observations),
        "certified_d_ge_5_unique_tuples": len(certified_unique),
        "certified_d_ge_5_tanner_classes": len(certified_classes),
        "connected_certified_d_ge_5_tanner_classes": len(
            connected_certified_classes
        ),
        "efficiency_per_observation": {
            "unique_positive_tuple": unique_count / calls,
            "connected_retained_tanner_class": len(connected_classes) / calls,
            "certified_d_ge_5_observation": len(certified_observations) / calls,
            "connected_certified_d_ge_5_tanner_class": len(
                connected_certified_classes
            )
            / calls,
        },
        "efficiency_per_unique_positive_tuple": {
            "connected_retained_tanner_class": len(connected_classes) / unique_count,
            "certified_d_ge_5_unique_tuple": len(certified_unique) / unique_count,
            "connected_certified_d_ge_5_tanner_class": len(
                connected_certified_classes
            )
            / unique_count,
        },
        "best_exact": _best_distance_claim(exact_claims),
        "best_certified_lower_bound": _best_distance_claim(certified_claims),
        "best_connected_exact": _best_distance_claim(
            [row for row in exact_claims if row["connected"]]
        ),
        "best_connected_certified_lower_bound": _best_distance_claim(
            [row for row in certified_claims if row["connected"]]
        ),
        "mapping_audit": {
            "positive_observations_mapped_to_retained_catalogue": len(
                mapped_observations
            ),
            "positive_observations_not_retained_after_d_le_2_filter": calls
            - len(mapped_observations),
            "unique_positive_tuples_not_retained_after_d_le_2_filter": unique_count
            - len(catalogue_rows),
        },
        "not_reconstructable_from_historical_logs": [
            "the number of distinct tuples sent to distance evaluation but producing no "
            "positive observation",
            "distinct coverage among all generated candidate instances",
            "decoder RNG states",
            "total CPU-hours",
        ],
    }


def uniform_sampler_comparison(
    output_dir: Path, *, run_seed: int | None = None
) -> dict[str, Any]:
    """Aggregate the sampler arm, optionally for one full-seed replicate."""
    selected_keys: set[tuple[Any, ...]] = set()
    selected_literal_keys: set[tuple[Any, ...]] = set()
    selected_translation_keys: set[tuple[Any, ...]] = set()
    retained_keys: set[tuple[Any, ...]] = set()
    connected_retained_keys: set[tuple[Any, ...]] = set()
    certified_keys: set[tuple[Any, ...]] = set()
    retained_classes: set[str] = set()
    connected_retained_classes: set[str] = set()
    certified_classes: set[str] = set()
    connected_certified_classes: set[str] = set()
    catalogue_exact_classes: set[str] = set()
    connected_catalogue_exact_classes: set[str] = set()
    catalogue_exact_observations = 0
    catalogue_exact_class_observations: Counter[str] = Counter()
    unresolved_certified_classes: set[str] = set()
    exact_followup_classes: set[str] = set()
    exact_followup_catalogue_classes: set[str] = set()
    exact_followup_unresolved_classes: set[str] = set()
    exact_followup_catalogue_observations: Counter[str] = Counter()
    distance_observations = 0
    planned_observations = 0
    certified_observations = 0
    exact_claims: list[dict[str, Any]] = []
    certified_claims: list[dict[str, Any]] = []
    unresolved_decoder_estimates: list[dict[str, Any]] = []
    rank_draw_k_histogram: Counter[int] = Counter()
    selected_k_histogram: Counter[int] = Counter()
    effective_distance_stage_histogram: Counter[str] = Counter()
    rank_only_draws = 0
    exact_search_disabled_records = 0
    exact_search_skipped = 0
    exact_search_attempted = 0
    completed_milp_records = 0
    planned_milp_finalists = 0
    distance_worker_seconds = 0.0
    certification_worker_seconds = 0.0
    milp_worker_seconds = 0.0
    errors: Counter[str] = Counter()
    completion: Counter[str] = Counter()
    lattice_seed_runs = 0
    overlap = _historical_overlap_sets(ROOT)
    exact_catalogue = _catalogue_exact_class_evidence(ROOT)

    for config_path in sorted(output_dir.glob("seed_*/*x*/config.json")):
        stored = json.loads(config_path.read_text(encoding="utf-8"))["config"]
        stored_seed = int(stored["run_seed"])
        if run_seed is not None and stored_seed != run_seed:
            continue
        lattice_seed_runs += 1
        config_exact_search_disabled = bool(stored["distance"]["skip_exact"])
        lattice_dir = config_path.parent
        all_draws = _read_jsonl(lattice_dir / "draws.jsonl")
        rank_only_draws += len(all_draws)
        rank_draw_k_histogram.update(int(row.get("k", 0)) for row in all_draws)
        draws = {
            int(row["universe_index"]): row
            for row in all_draws
            if row.get("selected")
        }
        selected_k_histogram.update(int(row.get("k", 0)) for row in draws.values())
        distances = _records_by_index(lattice_dir / "distance.jsonl")
        certifications = _records_by_index(lattice_dir / "certification.jsonl")
        milp = _records_by_index(lattice_dir / "milp.jsonl")
        distance_worker_seconds += sum(
            float(row.get("elapsed_seconds") or 0.0) for row in distances.values()
        )
        certification_worker_seconds += sum(
            float(row.get("elapsed_seconds") or 0.0)
            for row in certifications.values()
        )
        milp_worker_seconds += sum(
            float(row.get("elapsed_seconds") or 0.0) for row in milp.values()
        )
        completed_milp_records += len(milp)
        exact_search_disabled_records += sum(
            bool(
                row.get("exact_search_skipped")
                if "exact_search_skipped" in row
                else config_exact_search_disabled
            )
            for row in distances.values()
        )
        exact_search_skipped += sum(
            bool(
                row.get("exact_search_skipped")
                if "exact_search_skipped" in row
                else config_exact_search_disabled
            )
            and str(row.get("stage")) in {"exact_timeout", "exact_skipped"}
            for row in distances.values()
        )
        exact_search_attempted += sum(
            not bool(
                row.get("exact_search_skipped")
                if "exact_search_skipped" in row
                else config_exact_search_disabled
            )
            and str(row.get("stage")) in {"exact", "exact_timeout"}
            for row in distances.values()
        )
        effective_distance_stage_histogram.update(
            str(_effective_distance_stage(row, config_exact_search_disabled))
            for row in distances.values()
        )
        quota = sum(int(value) for value in stored["quotas"].values())
        planned_observations += quota
        distance_observations += len(distances)
        certification_targets = sum(
            int(row.get("d", 0)) > 2 for row in distances.values()
        )
        milp_targets = sum(
            float(distances[index].get("fom", 0.0)) >= 5.0
            and int(row.get("certified_lower_bound") or 0) >= 5
            for index, row in certifications.items()
        )
        planned_milp_finalists += milp_targets
        completion["selection_complete"] += len(draws) == quota
        completion["distance_complete"] += len(distances) == quota
        completion["certification_complete"] += (
            len(distances) == quota and len(certifications) == certification_targets
        )
        completion["milp_complete"] += (
            len(distances) == quota
            and len(certifications) == certification_targets
            and len(milp) == milp_targets
        )
        errors["draw"] += sum(bool(row.get("error")) for row in all_draws)
        errors["distance"] += sum(
            bool(row.get("error")) for row in distances.values()
        )
        errors["certification"] += sum(
            bool(row.get("error")) for row in certifications.values()
        )
        errors["milp"] += sum(bool(row.get("error")) for row in milp.values())

        for draw in draws.values():
            selected_keys.add(_campaign_tuple_key(draw))
            selected_literal_keys.add(
                literal_candidate_key(
                    int(draw["ell"]),
                    int(draw["m"]),
                    draw["A_terms"],
                    draw["B_terms"],
                )
            )
            selected_translation_keys.add(
                translation_candidate_key(
                    int(draw["ell"]),
                    int(draw["m"]),
                    draw["A_terms"],
                    draw["B_terms"],
                )
            )

        for index, certification in certifications.items():
            key = _campaign_tuple_key(certification)
            digest = certification.get("canonical_digest_sha256")
            connected = bool(draws[index].get("translation_connected"))
            lower = int(certification.get("certified_lower_bound") or 0)
            if lower > 2:
                retained_keys.add(key)
                if digest:
                    retained_classes.add(str(digest))
                if connected:
                    connected_retained_keys.add(key)
                    if digest:
                        connected_retained_classes.add(str(digest))
            if lower >= 5:
                certified_observations += 1
                certified_keys.add(key)
                if digest:
                    digest = str(digest)
                    certified_classes.add(digest)
                    if connected:
                        connected_certified_classes.add(digest)
            if lower > 0:
                n = int(certification["n"])
                k = int(certification["k"])
                certified_claims.append(
                    {
                        "ell": int(certification["ell"]),
                        "m": int(certification["m"]),
                        "n": n,
                        "k": k,
                        "d": lower,
                        "fom": k * lower * lower / n,
                        "run_seed": stored_seed,
                        "universe_index": index,
                        "connected": connected,
                    }
                )
                if certification.get("exact_distance") is not None:
                    exact_claims.append(
                        {
                            **certified_claims[-1],
                            "evidence_source": "control_low_weight_audit",
                            "canonical_digest_sha256": digest,
                        }
                    )
            if lower >= 5 and digest and digest in exact_catalogue:
                catalogue = exact_catalogue[digest]
                n = int(certification["n"])
                k = int(certification["k"])
                exact_n, exact_k, exact_d = map(int, catalogue["parameters"])
                if (n, k) != (exact_n, exact_k) or exact_d < lower:
                    raise ValueError(
                        f"inconsistent transferred exact evidence for class {digest}"
                    )
                catalogue_exact_observations += 1
                catalogue_exact_class_observations[digest] += 1
                catalogue_exact_classes.add(digest)
                if connected:
                    connected_catalogue_exact_classes.add(digest)
                transferred_claim = {
                    "ell": int(certification["ell"]),
                    "m": int(certification["m"]),
                    "n": n,
                    "k": k,
                    "d": exact_d,
                    "fom": k * exact_d * exact_d / n,
                    "presentation_id": catalogue["presentation_id"],
                    "run_seed": stored_seed,
                    "universe_index": index,
                    "connected": connected,
                    "canonical_digest_sha256": digest,
                    "class_id": catalogue["class_id"],
                    "evidence_source": "retained_catalogue_tanner_class_transfer",
                }
                exact_claims.append(transferred_claim)
                certified_claims.append(transferred_claim)
            elif lower >= 5 and digest:
                unresolved_certified_classes.add(digest)
                upper = int(distances[index].get("d") or 0)
                if upper >= lower:
                    unresolved_decoder_estimates.append(
                        {
                            "ell": int(certification["ell"]),
                            "m": int(certification["m"]),
                            "n": int(certification["n"]),
                            "k": int(certification["k"]),
                            "d": upper,
                            "fom": int(certification["k"]) * upper * upper
                            / int(certification["n"]),
                            "run_seed": stored_seed,
                            "universe_index": index,
                            "connected": connected,
                            "canonical_digest_sha256": digest,
                            "evidence_source": "bp_osd_distance_estimate",
                            "bound_type": "unverified_estimate",
                        }
                    )
            if (
                lower >= 5
                and float(distances[index].get("fom", 0.0)) >= 5.0
                and digest
            ):
                exact_followup_classes.add(digest)
                if digest in exact_catalogue:
                    exact_followup_catalogue_classes.add(digest)
                    exact_followup_catalogue_observations[digest] += 1
                else:
                    exact_followup_unresolved_classes.add(digest)

        for index, row in milp.items():
            if not row.get("d_is_exact"):
                continue
            exact = {
                "ell": int(row["ell"]),
                "m": int(row["m"]),
                "n": int(row["n"]),
                "k": int(row["k"]),
                "d": int(row["d"]),
                "fom": float(row["fom"]),
                "run_seed": stored_seed,
                "universe_index": index,
                "connected": bool(draws[index].get("translation_connected")),
            }
            exact_claims.append(exact)
            certified_claims.append(exact)

    unique_count = len(selected_keys)
    calls = distance_observations
    expected_lattice_seed_runs = (
        len(PROFILE_LATTICES["all"])
        if run_seed is not None
        else len(DEFAULT_SEEDS) * len(PROFILE_LATTICES["all"])
    )
    stage_completeness = {
        stage: int(completion[f"{stage}_complete"]) == lattice_seed_runs
        for stage in ("selection", "distance", "certification", "milp")
    }

    def divide(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    return {
        "arm": "matched_budget_uniform_sampler",
        "run_seed": run_seed,
        "lattice_seed_runs": lattice_seed_runs,
        "planned_distance_observations": planned_observations,
        "completed_distance_observations": calls,
        "rank_only_draws": rank_only_draws,
        "rank_draw_k_histogram": {
            str(k): count for k, count in sorted(rank_draw_k_histogram.items())
        },
        "selected_k_histogram": {
            str(k): count for k, count in sorted(selected_k_histogram.items())
        },
        "distance_stage_histogram": dict(
            sorted(effective_distance_stage_histogram.items())
        ),
        "distance_records_with_exact_search_disabled": exact_search_disabled_records,
        "exact_search_branches_skipped": exact_search_skipped,
        "exact_searches_attempted": exact_search_attempted,
        "planned_milp_finalists": planned_milp_finalists,
        "completed_milp_records": completed_milp_records,
        "summed_worker_time": {
            "distance_seconds": distance_worker_seconds,
            "distance_hours": distance_worker_seconds / 3600.0,
            "certification_seconds": certification_worker_seconds,
            "certification_hours": certification_worker_seconds / 3600.0,
            "milp_seconds": milp_worker_seconds,
            "milp_hours": milp_worker_seconds / 3600.0,
        },
        "unique_normalized_tuples": unique_count,
        "retained_d_gt_2_unique_tuples": len(retained_keys),
        "retained_tanner_classes": len(retained_classes),
        "connected_retained_presentations": len(connected_retained_keys),
        "connected_retained_tanner_classes": len(connected_retained_classes),
        "certified_d_ge_5_observations": certified_observations,
        "certified_d_ge_5_unique_tuples": len(certified_keys),
        "certified_d_ge_5_tanner_classes": len(certified_classes),
        "connected_certified_d_ge_5_tanner_classes": len(
            connected_certified_classes
        ),
        "efficiency_per_completed_observation": {
            "unique_tuple": divide(unique_count, calls),
            "connected_retained_tanner_class": divide(
                len(connected_retained_classes), calls
            ),
            "certified_d_ge_5_observation": divide(certified_observations, calls),
            "connected_certified_d_ge_5_tanner_class": divide(
                len(connected_certified_classes), calls
            ),
        },
        "efficiency_per_unique_tuple": {
            "connected_retained_tanner_class": divide(
                len(connected_retained_classes), unique_count
            ),
            "certified_d_ge_5_unique_tuple": divide(
                len(certified_keys), unique_count
            ),
            "connected_certified_d_ge_5_tanner_class": divide(
                len(connected_certified_classes), unique_count
            ),
        },
        "best_exact": _best_distance_claim(exact_claims),
        "catalogue_class_exact_evidence": {
            "observations": catalogue_exact_observations,
            "tanner_classes": len(catalogue_exact_classes),
            "connected_tanner_classes": len(connected_catalogue_exact_classes),
            "best_exact": _best_distance_claim(
                [
                    row
                    for row in exact_claims
                    if row.get("evidence_source")
                    == "retained_catalogue_tanner_class_transfer"
                ]
            ),
            "transferred_exact_classes": [
                {
                    **exact_catalogue[digest],
                    "sampled_observations": catalogue_exact_class_observations[
                        digest
                    ],
                }
                for digest in sorted(catalogue_exact_classes)
            ],
            "provenance": (
                "Exact distances are transferred only across identical "
                "colored-Tanner presentation-class digests in the released catalogue."
            ),
        },
        "unresolved_after_catalogue_class_transfer": {
            "certified_d_ge_5_tanner_classes": len(unresolved_certified_classes),
            "best_observed_decoder_estimate": _best_distance_claim(
                unresolved_decoder_estimates
            ),
            "note": (
                "The BP-OSD output stores a distance estimate but no explicit "
                "logical support, so it is not reported as an independently "
                "verified upper bound."
            ),
        },
        "optional_exact_followup": {
            "selection": "estimated FOM>=5 and certified d>=5",
            "finalist_observations": planned_milp_finalists,
            "finalist_tanner_classes": len(exact_followup_classes),
            "classes_with_transferred_exact_catalogue_evidence": len(
                exact_followup_catalogue_classes
            ),
            "classes_without_exact_catalogue_evidence": len(
                exact_followup_unresolved_classes
            ),
            "transferred_exact_classes": [
                {
                    **exact_catalogue[digest],
                    "sampled_observations": exact_followup_catalogue_observations[
                        digest
                    ],
                }
                for digest in sorted(exact_followup_catalogue_classes)
            ],
            "milp_records_completed": completed_milp_records,
            "status": "not_run; outside the primary matched-stage comparison",
        },
        "best_certified_lower_bound": _best_distance_claim(certified_claims),
        "best_connected_exact": _best_distance_claim(
            [row for row in exact_claims if row["connected"]]
        ),
        "best_connected_certified_lower_bound": _best_distance_claim(
            [row for row in certified_claims if row["connected"]]
        ),
        "overlap": {
            "selected_literal_in_historical_positive_log": len(
                selected_literal_keys & overlap["positive_literal"]
            ),
            "selected_translation_class_in_historical_positive_log": len(
                selected_translation_keys & overlap["positive_translation"]
            ),
            "selected_literal_in_retained_catalogue": len(
                selected_literal_keys & overlap["retained_literal"]
            ),
            "selected_translation_class_in_retained_catalogue": len(
                selected_translation_keys & overlap["retained_translation"]
            ),
            "certified_d_ge_5_tanner_classes_in_retained_catalogue": len(
                certified_classes & overlap["retained_classes"]
            ),
            "connected_certified_d_ge_5_tanner_classes_in_retained_catalogue": len(
                connected_certified_classes & overlap["retained_classes"]
            ),
        },
        "errors": {
            stage: int(errors[stage])
            for stage in ("draw", "distance", "certification", "milp")
        },
        "stage_completeness": stage_completeness,
        "primary_stage_completeness": {
            stage: stage_completeness[stage]
            for stage in ("selection", "distance", "certification")
        },
        "expected_lattice_seed_runs": expected_lattice_seed_runs,
        "primary_complete": (
            lattice_seed_runs == expected_lattice_seed_runs
            and calls == planned_observations
            and all(
                stage_completeness[stage]
                for stage in ("selection", "distance", "certification")
            )
            and all(
                int(errors[stage]) == 0
                for stage in ("draw", "distance", "certification")
            )
        ),
        "complete": (
            lattice_seed_runs == expected_lattice_seed_runs
            and calls == planned_observations
            and all(stage_completeness.values())
            and all(int(errors[stage]) == 0 for stage in errors)
        ),
    }


def _replicate_statistic(values: Sequence[int | float | None]) -> dict[str, Any]:
    """Summarize replicate values without silently dropping missing endpoints."""
    complete = bool(values) and all(value is not None for value in values)
    numeric = [float(value) for value in values if value is not None]
    return {
        "values": list(values),
        "all_values_present": complete,
        "mean": statistics.fmean(numeric) if complete else None,
        "sample_standard_deviation": (
            statistics.stdev(numeric) if complete and len(numeric) > 1 else None
        ),
        "min": min(numeric) if complete else None,
        "max": max(numeric) if complete else None,
    }


def _validate_one_variable_rank_repair(output_dir: Path) -> dict[str, Any]:
    """Reconcile the repair ledger with active and superseded records."""
    summary_path = output_dir / "one_variable_rank_repair.json"
    if not summary_path.exists():
        return {
            "performed": False,
            "status": "not_applicable",
            "repaired_rank_rows": 0,
            "selected_indices_added": 0,
            "selected_indices_removed": 0,
        }

    document = json.loads(summary_path.read_text(encoding="utf-8"))
    streams = document.get("streams", [])
    by_key: dict[tuple[int, int, int], dict[str, Any]] = {}
    repaired_rank_rows = 0
    selected_added_total = 0
    selected_removed_total = 0
    repaired_streams = 0
    intermediate_hashes_verified = 0
    superseded_files_verified = 0
    for stream in streams:
        seed = int(stream["run_seed"])
        ell, m = map(int, stream["lattice"])
        key = (seed, ell, m)
        if key in by_key:
            raise ValueError(f"duplicate rank-repair stream {key}")
        by_key[key] = stream
        repaired_rank_rows += int(stream.get("repaired_rank_rows") or 0)
        added = {int(index) for index in stream.get("selected_added") or []}
        removed = {int(index) for index in stream.get("selected_removed") or []}
        selected_added_total += len(added)
        selected_removed_total += len(removed)
        if stream.get("status") != "repaired":
            continue
        repaired_streams += 1
        lattice_dir = output_dir / f"seed_{seed}" / f"{ell}x{m}"
        repair_dir = lattice_dir / "repair" / "one_variable_rank_v1"
        manifest = json.loads(
            (repair_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest != stream:
            raise ValueError(f"rank-repair manifest mismatch: {lattice_dir}")

        draws_path = lattice_dir / "draws.jsonl"
        if _sha256_path(draws_path) != stream["after_sha256"]["draws"]:
            raise ValueError(f"repaired draw hash mismatch: {lattice_dir}")
        draws = _read_jsonl(draws_path)
        selected = {
            int(row["universe_index"]) for row in draws if bool(row.get("selected"))
        }
        if not added <= selected or removed & selected:
            raise ValueError(f"rank-repair selected-index mismatch: {lattice_dir}")
        if any(row.get("error") for row in draws):
            raise ValueError(f"active repaired draw still has an error: {lattice_dir}")

        for stage in ("distance", "certification", "milp"):
            active_rows = _read_jsonl(lattice_dir / f"{stage}.jsonl")
            intermediate_rows = [
                row
                for row in active_rows
                if int(row["universe_index"]) not in added
            ]
            intermediate_payload = b"".join(
                _canonical_json(row) + b"\n" for row in intermediate_rows
            )
            if (
                _sha256_bytes(intermediate_payload)
                != stream["after_sha256"][stage]
            ):
                raise ValueError(
                    f"post-removal repair hash mismatch for {stage}: {lattice_dir}"
                )
            intermediate_hashes_verified += 1
            superseded_path = repair_dir / f"superseded_{stage}.jsonl"
            superseded = _read_jsonl(superseded_path)
            if len(superseded) != int(
                stream["superseded_stage_records"][stage]
            ):
                raise ValueError(
                    f"superseded repair count mismatch for {stage}: {lattice_dir}"
                )
            if any(int(row["universe_index"]) not in removed for row in superseded):
                raise ValueError(
                    f"superseded repair index mismatch for {stage}: {lattice_dir}"
                )
            superseded_files_verified += 1

    active_configurations = {
        (
            int(document["config"]["run_seed"]),
            int(document["config"]["lattice"][0]),
            int(document["config"]["lattice"][1]),
        )
        for path in output_dir.glob("seed_*/*x*/config.json")
        for document in [json.loads(path.read_text(encoding="utf-8"))]
    }
    if set(by_key) != active_configurations:
        raise ValueError("rank-repair stream set does not match active configurations")
    if selected_added_total != selected_removed_total:
        raise ValueError("rank-repair selection swaps are unbalanced")

    return {
        "performed": True,
        "status": "verified",
        "streams": len(streams),
        "repaired_streams": repaired_streams,
        "repaired_rank_rows": repaired_rank_rows,
        "selected_indices_added": selected_added_total,
        "selected_indices_removed": selected_removed_total,
        "stream_manifests_match_summary": True,
        "active_draw_hashes_verified": repaired_streams,
        "post_removal_stage_hashes_verified": intermediate_hashes_verified,
        "superseded_stage_files_verified": superseded_files_verified,
        "active_draw_errors": 0,
        "note": (
            "Stage hashes in each repair manifest identify the post-removal, "
            "pre-replacement state; replayed replacement records are removed "
            "before those hashes are recomputed."
        ),
    }


def validate_uniform_control_records(output_dir: Path) -> dict[str, Any]:
    """Independently cross-check every stage boundary and seeded draw prefix."""
    configurations: set[tuple[int, int, int]] = set()
    totals: Counter[str] = Counter()
    stage_set_equalities: dict[str, list[bool]] = defaultdict(list)
    prescribed_protocol_matches: list[bool] = []
    for config_path in sorted(output_dir.glob("seed_*/*x*/config.json")):
        document = json.loads(config_path.read_text(encoding="utf-8"))
        stored = document["config"]
        config_sha256 = str(document["config_sha256"])
        if config_sha256 != _sha256_bytes(_canonical_json(stored)):
            raise ValueError(f"configuration digest mismatch: {config_path}")
        seed = int(stored["run_seed"])
        ell, m = map(int, stored["lattice"])
        if config_path.parent.name != f"{ell}x{m}" or config_path.parent.parent.name != f"seed_{seed}":
            raise ValueError(f"configuration path mismatch: {config_path}")
        key = (seed, ell, m)
        if key in configurations:
            raise ValueError(f"duplicate seed/lattice configuration: {key}")
        configurations.add(key)

        lattice_dir = config_path.parent
        draws = _read_jsonl(lattice_dir / "draws.jsonl")
        selected_counts, _ = _selection_state(draws)
        quotas = {str(name): int(value) for name, value in stored["quotas"].items()}
        prescribed_protocol_matches.append(
            not bool(stored["pilot"])
            and bool(stored["matched_budget"])
            and quotas == HISTORICAL_DISTANCE_QUOTAS[(ell, m)]
            and stored["sampler_version"] == SAMPLER_VERSION
            and stored["historical_evaluator_sha256"]
            == HISTORICAL_EVALUATOR_SHA256
            and stored["distance"]
            == {
                "quick_trials": 1000,
                "refine_trials_per_batch": 500,
                "refine_batches": 3,
                "fom_threshold_refine": 3.0,
                "fom_threshold_exact": 5.0,
                "osd_cs_trials": 200,
                "skip_exact": True,
                "exact_timeout_seconds": 300,
                "decoder_rng": "sha256-derived per-candidate numpy Generator",
            }
            and stored["certification"]
            == {
                "algorithm": "css_weight_le_4_syndrome_quotient_mitm_v1",
                "selection": "decoder upper bound d>=5",
            }
            and stored["milp"]
            == {
                "selection": "estimated FOM>=5 and certified d>=5",
                "timeout_per_logical_seconds": 1800,
                "total_timeout_seconds": 14400,
                "early_stop": 4,
            }
        )
        if dict(selected_counts) != quotas:
            raise ValueError(f"selected quota mismatch: {lattice_dir}")
        selected = {
            int(row["universe_index"]): row for row in draws if row.get("selected")
        }
        if len(selected) != sum(quotas.values()):
            raise ValueError(f"selected-index count mismatch: {lattice_dir}")

        replay = LazyFisherYates(
            universe_size(ell, m), _derived_seed(seed, ell, m, "selection-v1")
        )
        replay_selected_counts: Counter[str] = Counter()
        for row, expected_index in zip(draws, replay):
            actual_index = int(row["universe_index"])
            if actual_index != expected_index:
                raise ValueError(f"seeded draw mismatch: {lattice_dir}")
            expected_a, expected_b = unrank_normalized_pair(ell, m, actual_index)
            if row["A_terms"] != [list(term) for term in expected_a] or row[
                "B_terms"
            ] != [list(term) for term in expected_b]:
                raise ValueError(f"unranked support mismatch: {lattice_dir}")
            if str(row.get("band")) != str(k_band(int(row.get("k", 0)))):
                raise ValueError(f"stored k-band mismatch: {lattice_dir}")
            band = str(row.get("band"))
            expected_selected = (
                band in quotas and replay_selected_counts[band] < quotas[band]
            )
            if bool(row.get("selected")) != expected_selected:
                raise ValueError(f"first-in-band selection mismatch: {lattice_dir}")
            if expected_selected:
                replay_selected_counts[band] += 1

        distances = _records_by_index(lattice_dir / "distance.jsonl")
        certifications = _records_by_index(lattice_dir / "certification.jsonl")
        milp = _records_by_index(lattice_dir / "milp.jsonl")
        if not set(distances) <= set(selected):
            raise ValueError(f"distance index outside selection: {lattice_dir}")
        stage_set_equalities["distance_equals_selected"].append(
            set(distances) == set(selected)
        )
        expected_certifications = {
            index for index, row in distances.items() if int(row.get("d", 0)) > 2
        }
        if not set(certifications) <= expected_certifications:
            raise ValueError(f"certification index outside targets: {lattice_dir}")
        stage_set_equalities["certification_equals_targets"].append(
            set(certifications) == expected_certifications
        )
        expected_milp = {
            index
            for index, row in certifications.items()
            if float(distances[index].get("fom", 0.0)) >= 5.0
            and int(row.get("certified_lower_bound") or 0) >= 5
        }
        if not set(milp) <= expected_milp:
            raise ValueError(f"MILP index outside finalists: {lattice_dir}")
        stage_set_equalities["milp_equals_finalists"].append(set(milp) == expected_milp)

        for stage, rows in (
            ("draw", draws),
            ("distance", distances.values()),
            ("certification", certifications.values()),
            ("milp", milp.values()),
        ):
            for row in rows:
                if row.get("config_sha256") != config_sha256:
                    raise ValueError(f"{stage} configuration digest mismatch: {lattice_dir}")
                if row.get("error"):
                    raise ValueError(f"{stage} worker error in {lattice_dir}: {row['error']}")

        summary_path = lattice_dir / "summary.json"
        if not summary_path.exists():
            raise ValueError(f"missing lattice summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("config_sha256") != config_sha256
            or int(summary.get("run_seed")) != seed
            or list(map(int, summary.get("lattice", []))) != [ell, m]
        ):
            raise ValueError(f"summary configuration mismatch: {lattice_dir}")
        retained = [
            row
            for row in certifications.values()
            if int(row.get("certified_lower_bound") or 0) > 2
        ]
        certified_ge5 = [
            row
            for row in certifications.values()
            if int(row.get("certified_lower_bound") or 0) >= 5
        ]
        retained_classes = {
            row["canonical_digest_sha256"]
            for row in retained
            if row.get("canonical_digest_sha256")
        }
        if summary["yield"]["retained_d_gt_2"] != len(retained):
            raise ValueError(f"retained summary mismatch: {lattice_dir}")
        if summary["diversity"]["retained_d_gt_2_tanner_classes"] != len(
            retained_classes
        ):
            raise ValueError(f"retained-class summary mismatch: {lattice_dir}")
        if summary["yield"]["certified_d_ge_5"] != len(certified_ge5):
            raise ValueError(f"certified-yield summary mismatch: {lattice_dir}")

        totals.update(
            configurations=1,
            rank_draws=len(draws),
            selected=len(selected),
            distance=len(distances),
            certification=len(certifications),
            retained_after_audit=len(retained),
            certified_d_ge_5=len(certified_ge5),
            milp=len(milp),
        )

    primary_stage_set_equality = {
        name: bool(stage_set_equalities[name]) and all(stage_set_equalities[name])
        for name in ("distance_equals_selected", "certification_equals_targets")
    }
    return {
        "all_record_relationship_checks_pass": True,
        "stage_set_equality": {
            name: bool(values) and all(values)
            for name, values in sorted(stage_set_equalities.items())
        },
        "primary_stage_set_equality": primary_stage_set_equality,
        "all_configurations_match_prespecified_protocol": (
            bool(prescribed_protocol_matches)
            and all(prescribed_protocol_matches)
        ),
        "one_variable_rank_repair": _validate_one_variable_rank_repair(output_dir),
        "configurations": [list(item) for item in sorted(configurations)],
        "totals": dict(totals),
    }


def _has_expected_full_design(
    summaries: Sequence[dict[str, Any]],
    *,
    summary_path_count: int,
    config_path_count: int,
    per_lattice: dict[str, dict[str, Any]],
    integrity_configurations: set[tuple[int, int, int]],
) -> tuple[bool, bool]:
    expected_configurations = {
        (seed, ell, m)
        for seed in DEFAULT_SEEDS
        for ell, m in PROFILE_LATTICES["all"]
    }
    observed_configurations = {
        (int(row["run_seed"]), int(row["lattice"][0]), int(row["lattice"][1]))
        for row in summaries
    }
    expected_lattice_labels = {
        f"{ell}x{m}" for ell, m in PROFILE_LATTICES["all"]
    }
    three_runs_per_lattice = (
        set(per_lattice) == expected_lattice_labels
        and all(
            row["runs"] == len(DEFAULT_SEEDS)
            and set(map(int, row["seeds"])) == set(DEFAULT_SEEDS)
            for row in per_lattice.values()
        )
    )
    complete = (
        summary_path_count == len(expected_configurations)
        and config_path_count == len(expected_configurations)
        and len(summaries) == len(observed_configurations)
        and observed_configurations == expected_configurations
        and integrity_configurations == expected_configurations
        and three_runs_per_lattice
    )
    return complete, three_runs_per_lattice


def aggregate_summaries(output_dir: Path) -> dict[str, Any]:
    summaries = []
    summary_paths = sorted(output_dir.glob("seed_*/*x*/summary.json"))
    for path in summary_paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        seed = int(row["run_seed"])
        ell, m = map(int, row["lattice"])
        if path.parent.name != f"{ell}x{m}" or path.parent.parent.name != f"seed_{seed}":
            raise ValueError(f"summary path/identity mismatch: {path}")
        if not (path.parent / "config.json").exists():
            raise ValueError(f"summary has no matching configuration: {path}")
        summaries.append(row)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        grouped[f"{row['lattice'][0]}x{row['lattice'][1]}"].append(row)

    per_lattice = {}
    metric_paths = {
        "estimated_d_ge_5_rate": ("yield", "estimated_d_ge_5_rate_per_distance_evaluation"),
        "certified_d_ge_5_rate": ("yield", "certified_d_ge_5_rate_per_distance_evaluation"),
        "connected_class_yield": ("yield", "connected_class_yield_per_distance_evaluation"),
    }
    for lattice, rows in sorted(grouped.items()):
        metrics = {}
        for name, path in metric_paths.items():
            values = [float(row[path[0]][path[1]]) for row in rows]
            metrics[name] = {
                "values": values,
                "mean": statistics.fmean(values),
                "sample_standard_deviation": statistics.stdev(values)
                if len(values) > 1
                else None,
                "min": min(values),
                "max": max(values),
            }
        metrics["best_estimated_fom"] = {
            "values": [
                row["best"]["estimated"]["fom"]
                if row["best"]["estimated"] is not None
                else None
                for row in rows
            ]
        }
        per_lattice[lattice] = {
            "seeds": [row["run_seed"] for row in rows],
            "runs": len(rows),
            "all_matched_budget": all(row["matched_budget"] for row in rows),
            "all_required_non_milp_stages_complete": all(
                row["completion"]["selection"]
                and row["completion"]["distance"]
                and row["completion"]["certification"]
                for row in rows
            ),
            "rank_draw_k_histogram_by_seed": {
                str(row["run_seed"]): row["sampling"]["rank_draw_k_histogram"]
                for row in rows
            },
            "selected_k_histogram_by_seed": {
                str(row["run_seed"]): row["sampling"]["selected_k_histogram"]
                for row in rows
            },
            "metrics": metrics,
        }

    seed_ids = sorted({int(row["run_seed"]) for row in summaries})
    per_seed = {
        str(seed): uniform_sampler_comparison(output_dir, run_seed=seed)
        for seed in seed_ids
    }

    def seed_values(field: str) -> list[int | float | None]:
        return [per_seed[str(seed)][field] for seed in seed_ids]

    def best_fom(field: str) -> list[float | None]:
        return [
            (
                float(per_seed[str(seed)][field]["fom"])
                if per_seed[str(seed)][field] is not None
                else None
            )
            for seed in seed_ids
        ]

    replicate_statistics = {
        "rank_only_draws": _replicate_statistic(seed_values("rank_only_draws")),
        "distance_records_with_exact_search_disabled": _replicate_statistic(
            seed_values("distance_records_with_exact_search_disabled")
        ),
        "exact_search_branches_skipped": _replicate_statistic(
            seed_values("exact_search_branches_skipped")
        ),
        "exact_searches_attempted": _replicate_statistic(
            seed_values("exact_searches_attempted")
        ),
        "planned_milp_finalists": _replicate_statistic(
            seed_values("planned_milp_finalists")
        ),
        "completed_milp_records": _replicate_statistic(
            seed_values("completed_milp_records")
        ),
        "distance_worker_hours": _replicate_statistic(
            [
                per_seed[str(seed)]["summed_worker_time"]["distance_hours"]
                for seed in seed_ids
            ]
        ),
        "certification_worker_hours": _replicate_statistic(
            [
                per_seed[str(seed)]["summed_worker_time"]["certification_hours"]
                for seed in seed_ids
            ]
        ),
        "completed_distance_observations": _replicate_statistic(
            seed_values("completed_distance_observations")
        ),
        "unique_normalized_tuples": _replicate_statistic(
            seed_values("unique_normalized_tuples")
        ),
        "retained_d_gt_2_unique_tuples": _replicate_statistic(
            seed_values("retained_d_gt_2_unique_tuples")
        ),
        "connected_retained_tanner_classes": _replicate_statistic(
            seed_values("connected_retained_tanner_classes")
        ),
        "certified_d_ge_5_observations": _replicate_statistic(
            seed_values("certified_d_ge_5_observations")
        ),
        "certified_d_ge_5_unique_tuples": _replicate_statistic(
            seed_values("certified_d_ge_5_unique_tuples")
        ),
        "certified_d_ge_5_tanner_classes": _replicate_statistic(
            seed_values("certified_d_ge_5_tanner_classes")
        ),
        "connected_certified_d_ge_5_tanner_classes": _replicate_statistic(
            seed_values("connected_certified_d_ge_5_tanner_classes")
        ),
        "catalogue_exact_evidence_observations": _replicate_statistic(
            [
                per_seed[str(seed)]["catalogue_class_exact_evidence"]["observations"]
                for seed in seed_ids
            ]
        ),
        "catalogue_exact_evidence_tanner_classes": _replicate_statistic(
            [
                per_seed[str(seed)]["catalogue_class_exact_evidence"]["tanner_classes"]
                for seed in seed_ids
            ]
        ),
        "unresolved_certified_d_ge_5_tanner_classes": _replicate_statistic(
            [
                per_seed[str(seed)]["unresolved_after_catalogue_class_transfer"][
                    "certified_d_ge_5_tanner_classes"
                ]
                for seed in seed_ids
            ]
        ),
        "exact_followup_finalist_tanner_classes": _replicate_statistic(
            [
                per_seed[str(seed)]["optional_exact_followup"][
                    "finalist_tanner_classes"
                ]
                for seed in seed_ids
            ]
        ),
        "exact_followup_classes_with_transferred_exact_evidence": (
            _replicate_statistic(
                [
                    per_seed[str(seed)]["optional_exact_followup"][
                        "classes_with_transferred_exact_catalogue_evidence"
                    ]
                    for seed in seed_ids
                ]
            )
        ),
        "best_certified_lower_bound_fom": _replicate_statistic(
            best_fom("best_certified_lower_bound")
        ),
        "best_exact_fom": _replicate_statistic(best_fom("best_exact")),
        "certified_d_ge_5_rate_per_call": _replicate_statistic(
            [
                per_seed[str(seed)]["efficiency_per_completed_observation"][
                    "certified_d_ge_5_observation"
                ]
                for seed in seed_ids
            ]
        ),
        "connected_certified_class_rate_per_call": _replicate_statistic(
            [
                per_seed[str(seed)]["efficiency_per_completed_observation"][
                    "connected_certified_d_ge_5_tanner_class"
                ]
                for seed in seed_ids
            ]
        ),
    }
    overlap_names = (
        "selected_literal_in_historical_positive_log",
        "selected_translation_class_in_historical_positive_log",
        "selected_literal_in_retained_catalogue",
        "selected_translation_class_in_retained_catalogue",
        "certified_d_ge_5_tanner_classes_in_retained_catalogue",
        "connected_certified_d_ge_5_tanner_classes_in_retained_catalogue",
    )
    for name in overlap_names:
        replicate_statistics[f"overlap_{name}"] = _replicate_statistic(
            [int(per_seed[str(seed)]["overlap"][name]) for seed in seed_ids]
        )

    uniform_all = uniform_sampler_comparison(output_dir)
    integrity = validate_uniform_control_records(output_dir)
    config_paths = sorted(output_dir.glob("seed_*/*x*/config.json"))
    integrity_configurations = {
        tuple(map(int, item)) for item in integrity["configurations"]
    }
    full_design_present, three_runs_per_lattice = _has_expected_full_design(
        summaries,
        summary_path_count=len(summary_paths),
        config_path_count=len(config_paths),
        per_lattice=per_lattice,
        integrity_configurations=integrity_configurations,
    )
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "uniform_control_aggregate_summary",
        "lattice_seed_runs": len(summaries),
        "per_lattice": per_lattice,
        "per_seed": per_seed,
        "replicate_statistics": replicate_statistics,
        "validation": {
            "record_integrity": integrity,
            "expected_three_seed_six_lattice_design_present": full_design_present,
            "exactly_three_runs_per_lattice": three_runs_per_lattice,
            "all_seed_replicates_have_6444_planned_calls": all(
                row["planned_distance_observations"] == 6_444
                for row in per_seed.values()
            ),
            "all_stage_records_complete": all(
                all(row["stage_completeness"].values())
                for row in per_seed.values()
            ),
            "all_primary_stage_records_complete": all(
                all(row["primary_stage_completeness"].values())
                for row in per_seed.values()
            ),
            "all_error_counts_zero": all(
                all(count == 0 for count in row["errors"].values())
                for row in per_seed.values()
            ),
            "all_primary_error_counts_zero": all(
                all(
                    row["errors"][stage] == 0
                    for stage in ("draw", "distance", "certification")
                )
                for row in per_seed.values()
            ),
            "optional_milp_followup_complete": all(
                row["stage_completeness"]["milp"] for row in per_seed.values()
            ),
            "ready_for_primary_comparison": (
                full_design_present
                and all(integrity["primary_stage_set_equality"].values())
                and integrity["all_configurations_match_prespecified_protocol"]
                and all(
                    row["planned_distance_observations"] == 6_444
                    for row in per_seed.values()
                )
                and all(
                    all(row["primary_stage_completeness"].values())
                    for row in per_seed.values()
                )
                and all(
                    all(
                        row["errors"][stage] == 0
                        for stage in ("draw", "distance", "certification")
                    )
                    for row in per_seed.values()
                )
            ),
        },
        "side_by_side": {
            "scope": (
                "CSS positive distance observations on the six Stage-2 lattices; "
                "class metrics use the final retained d>2 catalogue"
            ),
            "comparison_unit": (
                "The primary control values are the three independent 6,444-call "
                "replicates. Pooled uniform counts cover 19,332 calls and are "
                "reported only as secondary coverage totals."
            ),
            "exact_evidence_scope": (
                "Primary readiness requires selection, distance, and deterministic "
                "weight-four certification, not the optional MILP follow-up. Exact "
                "control values are transferred only for sampled colored-Tanner "
                "classes already exact in the released catalogue; unresolved "
                "decoder estimates are reported separately."
            ),
            "historical_llm": historical_llm_comparison(ROOT),
            "uniform_sampler_replicates": per_seed,
            "uniform_sampler_pooled_coverage": uniform_all,
        },
    }
    _atomic_write_json(output_dir / "summary.json", aggregate)
    _atomic_write_json(
        output_dir / "artifact_manifest.json",
        _uniform_artifact_manifest(output_dir),
    )
    return aggregate


def _uniform_artifact_manifest(output_dir: Path) -> dict[str, Any]:
    """Pin the code and completed records used for the control analysis."""
    def metadata(path: Path) -> dict[str, Any]:
        result: dict[str, Any] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256_path(path),
        }
        if path.suffix == ".jsonl":
            result["records"] = sum(1 for _ in path.open(encoding="utf-8"))
        return result

    source_files = {}
    for relative in UNIFORM_CONTROL_SOURCE_PATHS:
        path = ROOT / relative
        source_files[relative] = metadata(path)
    dependency_files = {}
    for relative in UNIFORM_CONTROL_DEPENDENCY_PATHS:
        dependency_files[relative] = metadata(ROOT / relative)
    data_files = {}
    manifest_path = output_dir / "artifact_manifest.json"
    for path in sorted(candidate for candidate in output_dir.rglob("*") if candidate.is_file()):
        if path == manifest_path:
            continue
        relative = str(path.relative_to(output_dir))
        data_files[relative] = metadata(path)
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "uniform_control_artifact_manifest",
        "initial_selection_distance_full_source_sha256": (
            EXECUTED_SELECTION_DISTANCE_SOURCE_SHA256
        ),
        "executed_source_note": (
            "This hash identifies the untracked sampler source loaded by the initial "
            "run; those exact source bytes were not separately archived. The released "
            "current source records the protocol and implements its deterministic "
            "replay. A documented one-variable rank repair replayed the same seeded "
            "stopping rule, replaced 16 of 19,332 selected indices, and evaluated only "
            "those replacements with the current source. Superseded records and "
            "before/after hashes are retained under each stream's repair directory."
        ),
        "source_files": source_files,
        "dependency_files": dependency_files,
        "software_versions_at_manifest_write": {
            package: (
                importlib.metadata.version(package)
                if _package_is_installed(package)
                else None
            )
            for package in ("igraph", "numpy", "scipy", "qldpc", "ldpc")
        },
        "data_files": data_files,
    }


def _package_is_installed(package: str) -> bool:
    try:
        importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be distinct")
    return seeds


def _parse_lattices(value: str) -> tuple[tuple[int, int], ...]:
    lattices = []
    for item in value.split(","):
        try:
            ell_text, m_text = item.lower().split("x", 1)
            lattice = (int(ell_text), int(m_text))
        except (ValueError, TypeError) as exc:
            raise argparse.ArgumentTypeError(f"invalid lattice {item!r}") from exc
        if lattice not in HISTORICAL_DISTANCE_QUOTAS:
            raise argparse.ArgumentTypeError(f"unsupported lattice {lattice}")
        lattices.append(lattice)
    return tuple(lattices)


def _configs(args: argparse.Namespace) -> list[RunConfig]:
    lattices = args.lattices or PROFILE_LATTICES[args.profile]
    configs = []
    for seed in args.seeds:
        for ell, m in lattices:
            historical = HISTORICAL_DISTANCE_QUOTAS[(ell, m)]
            quotas = (
                {band: min(2, count) for band, count in historical.items()}
                if args.pilot
                else dict(historical)
            )
            configs.append(
                RunConfig(
                    run_seed=seed,
                    ell=ell,
                    m=m,
                    quotas=quotas,
                    pilot=args.pilot,
                    quick_trials=20 if args.pilot else args.quick_trials,
                    refine_trials=10 if args.pilot else args.refine_trials,
                    skip_exact=True if args.pilot else args.skip_exact,
                    exact_timeout=args.exact_timeout,
                    milp_timeout_per_logical=args.milp_timeout_per_logical,
                    milp_total_timeout=args.milp_total_timeout,
                )
            )
    return configs


def _print_checkpoint(stage: str, config: RunConfig, checkpoint: dict[str, Any]) -> None:
    print(
        f"[{stage}] seed={config.run_seed} lattice={config.ell}x{config.m} "
        f"status={checkpoint.get('status', 'written')}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stage",
        choices=(
            "select",
            "repair-univariate",
            "distance",
            "certify",
            "milp",
            "summarize",
            "all",
        ),
        default="select",
    )
    parser.add_argument("--profile", choices=("small", "large", "all"), default="all")
    parser.add_argument("--lattices", type=_parse_lattices)
    parser.add_argument("--seeds", type=_parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "weight5_uniform_baseline"
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--selection-workers-per-stream",
        type=int,
        default=1,
        help=(
            "parallel exact-k workers inside each seed/lattice stream; when >1, "
            "streams run sequentially to avoid nested oversubscription"
        ),
    )
    parser.add_argument("--max-new-draws", type=int)
    parser.add_argument("--max-new-distances", type=int)
    parser.add_argument("--max-new-certifications", type=int)
    parser.add_argument("--max-new-milp", type=int)
    parser.add_argument("--quick-trials", type=int, default=1000)
    parser.add_argument("--refine-trials", type=int, default=500)
    parser.add_argument("--exact-timeout", type=int, default=300)
    parser.add_argument("--skip-exact", action="store_true")
    parser.add_argument("--milp-timeout-per-logical", type=int, default=1800)
    parser.add_argument("--milp-total-timeout", type=int, default=14400)
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Use two candidates per k band, 20/10 decoder trials, and no exact search",
    )
    parser.add_argument(
        "--skip-source-check",
        action="store_true",
        help="Write provenance without verifying pinned local manifests/logs/git object",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.selection_workers_per_stream < 1:
        parser.error("--selection-workers-per-stream must be positive")
    for name in (
        "max_new_draws",
        "max_new_distances",
        "max_new_certifications",
        "max_new_milp",
    ):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    if args.stage == "all" and not args.pilot:
        parser.error(
            "full matched-budget runs must be launched stage by stage; --stage all is "
            "reserved for the bounded --pilot smoke protocol"
        )

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "protocol.json"
    manifest = protocol_manifest(ROOT, verify_sources=not args.skip_source_check)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Software versions and the availability of .git are diagnostic, so
        # do not make later resumes fail merely because the runtime moved.
        if existing["protocol"] != manifest["protocol"]:
            raise ValueError(f"incompatible protocol at {manifest_path}")
        for field in (
            "distance_quotas",
            "historical_gates",
            "historical_decoder",
            "historical_milp",
            "sampler",
            "known_replay_differences",
            "source_evidence",
        ):
            if existing.get(field) != manifest.get(field):
                raise ValueError(
                    f"protocol provenance field {field!r} drifted at {manifest_path}"
                )
    else:
        _atomic_write_json(manifest_path, manifest)

    configs = _configs(args)
    if (
        args.stage == "select"
        and args.workers > 1
        and len(configs) > 1
        and args.selection_workers_per_stream == 1
    ):
        # Seed/lattice streams are independent and write disjoint files.  Run
        # them concurrently so the exact-k screen can use the same multicore
        # host as the distance stage without complicating within-stream order
        # or resume semantics.
        with ProcessPoolExecutor(max_workers=min(args.workers, len(configs))) as executor:
            futures = {
                executor.submit(
                    run_selection,
                    args.output_dir,
                    config,
                    max_new_draws=args.max_new_draws,
                ): config
                for config in configs
            }
            for future in as_completed(futures):
                config = futures[future]
                result = future.result()
                _print_checkpoint("select", config, result)
        return 0

    overlap = None
    repair_results = []
    for config in configs:
        if args.stage == "repair-univariate":
            result = repair_one_variable_rank_records(args.output_dir, config)
            repair_results.append(result)
            _print_checkpoint("repair-univariate", config, result)
            continue
        if args.stage in ("select", "all"):
            result = run_selection(
                args.output_dir,
                config,
                max_new_draws=args.max_new_draws,
                selection_workers=args.selection_workers_per_stream,
            )
            _print_checkpoint("select", config, result)
            if result["status"] != "complete" and args.stage == "all":
                continue
        if args.stage in ("distance", "all"):
            result = run_distance(
                args.output_dir,
                config,
                workers=args.workers,
                max_new=args.max_new_distances,
            )
            _print_checkpoint("distance", config, result)
            if result["status"] != "complete" and args.stage == "all":
                continue
        if args.stage in ("certify", "all"):
            result = run_certification(
                args.output_dir,
                config,
                workers=args.workers,
                max_new=args.max_new_certifications,
            )
            _print_checkpoint("certify", config, result)
            if result["status"] != "complete" and args.stage == "all":
                continue
        if args.stage == "milp":
            result = run_milp(
                args.output_dir,
                config,
                workers=args.workers,
                max_new=args.max_new_milp,
            )
            _print_checkpoint("milp", config, result)
        if args.stage in ("summarize", "all"):
            if overlap is None:
                overlap = _historical_overlap_sets(ROOT)
            summary = summarize_lattice(args.output_dir, config, overlap)
            _print_checkpoint("summarize", config, {"status": "written"})
            print(
                f"  estimated d>=5: {summary['yield']['estimated_d_ge_5']}; "
                f"certified d>=5: {summary['yield']['certified_d_ge_5']}; "
                f"connected classes: "
                f"{summary['yield']['connected_certified_d_ge_5_classes']}"
            )
    if args.stage in ("summarize", "all"):
        aggregate_summaries(args.output_dir)
    if args.stage == "repair-univariate":
        _atomic_write_json(
            args.output_dir / "one_variable_rank_repair.json",
            {
                "schema_version": SCHEMA_VERSION,
                "record_type": "uniform_control_one_variable_rank_repair_summary",
                "reason": (
                    "qLDPC BBCode symbol inference rejected valid supports that use "
                    "only one lattice variable; exact ranks were recomputed from "
                    "the full CSS check matrices and the seeded stopping rule replayed"
                ),
                "streams": repair_results,
                "totals": {
                    "repaired_rank_rows": sum(
                        int(row.get("repaired_rank_rows", 0)) for row in repair_results
                    ),
                    "selected_added": sum(
                        len(row.get("selected_added", [])) for row in repair_results
                    ),
                    "selected_removed": sum(
                        len(row.get("selected_removed", [])) for row in repair_results
                    ),
                },
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
