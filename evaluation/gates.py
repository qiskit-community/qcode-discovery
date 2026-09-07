"""Environment-overridable gate thresholds shared by all evaluators.

Existing evaluators hard-code thresholds such as ``MIN_K_THRESHOLD`` or the
BP-OSD trust ratio as module-level literals. This module centralizes those
values behind environment-variable lookups so a campaign (e.g. the weight-5
campaigns) can override them without touching the literals that existing
campaigns depend on.

Every getter reads ``os.environ`` fresh on each call (not cached at import
time) so tests can monkeypatch the environment and see an immediate effect,
and so campaign scripts can set the variable right before invoking a worker
without needing a fresh process/import.

Every getter's Python default matches the "existing default" column of the
Phase A3 gate table in ``plans/direction2_weight5_campaigns.md`` -- when no
environment variable is set, behavior for existing (non-weight-5) campaigns
is unchanged.
"""

from __future__ import annotations

import os

# Name -> existing default, matching plans/direction2_weight5_campaigns.md Phase A3.
_INT_DEFAULTS = {
    "QCODE_MIN_K_THRESHOLD": 4,
    "QCODE_MIN_K_THRESHOLD_NONCSS": 1,
    "QCODE_MIN_RELEVANT_D": 6,
    "QCODE_MIN_RELEVANT_D_NONCSS": 5,
    "QCODE_MAX_BUILD_CANDIDATES_CSS": 5000,
    "QCODE_MAX_BUILD_CANDIDATES_NONCSS": 3000,
}

_FLOAT_DEFAULTS = {
    "QCODE_FOM_THRESHOLD_REFINE": 6.0,
    "QCODE_FOM_THRESHOLD_EXACT": 8.0,
    "QCODE_SAVE_FOM_THRESHOLD_CSS": 6.0,
    "QCODE_SAVE_FOM_THRESHOLD_NONCSS": 4.0,
    "QCODE_SAVE_TRUST_RATIO": 1.3,
    "QCODE_SAVE_TRUST_RATIO_NONCSS": 2.5,
}

_STRING_DEFAULTS = {
    # Future weight-five searches reject direct-sum replications by default.
    # ``allow`` exists only for intentional legacy-run reproduction or a
    # study whose objective explicitly includes disconnected presentations.
    "QCODE_WEIGHT5_CONNECTIVITY_POLICY": "reject",
}


def _env_int(name: str) -> int:
    val = os.environ.get(name)
    return int(val) if val is not None else _INT_DEFAULTS[name]


def _env_float(name: str) -> float:
    val = os.environ.get(name)
    return float(val) if val is not None else _FLOAT_DEFAULTS[name]


def _env_choice(name: str, choices: set[str]) -> str:
    value = os.environ.get(name, _STRING_DEFAULTS[name]).strip().lower()
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of {{{allowed}}}, got {value!r}")
    return value


def min_k_threshold() -> int:
    """CSS: minimum k to avoid the ``k_low`` rejection stage."""
    return _env_int("QCODE_MIN_K_THRESHOLD")


def min_k_threshold_noncss() -> int:
    """PBB: minimum k for a built code to be considered valid (existing
    behavior only required k > 0, i.e. k >= 1)."""
    return _env_int("QCODE_MIN_K_THRESHOLD_NONCSS")


def min_relevant_d() -> int:
    """CSS: minimum distance for a result to contribute FOM in the MILP
    cascade's combined_score."""
    return _env_int("QCODE_MIN_RELEVANT_D")


def min_relevant_d_noncss() -> int:
    """PBB: minimum distance for a result to contribute FOM."""
    return _env_int("QCODE_MIN_RELEVANT_D_NONCSS")


def fom_threshold_refine() -> float:
    """Preliminary FOM that triggers the refined (Stage 4) BP-OSD pass."""
    return _env_float("QCODE_FOM_THRESHOLD_REFINE")


def fom_threshold_exact() -> float:
    """Refined FOM that triggers the exact (Stage 5) distance computation."""
    return _env_float("QCODE_FOM_THRESHOLD_EXACT")


def fom_threshold_exact_or_disabled() -> float:
    """FOM threshold for Stage-5 exact distance inside OpenEvolve worker
    processes.

    Stage 5 uses a SIGALRM-based timeout that is not available in
    OpenEvolve's worker threads/processes, so the non-quick evaluation path
    disables it (``float("inf")``) by default. Only when a campaign
    explicitly sets ``QCODE_FOM_THRESHOLD_EXACT`` does this return a finite
    value, letting that campaign opt in (and accept responsibility for
    verifying Stage 5 is safe in its own worker environment).
    """
    val = os.environ.get("QCODE_FOM_THRESHOLD_EXACT")
    return float(val) if val is not None else float("inf")


def save_fom_threshold_css() -> float:
    """CSS: minimum FOM for a credible result to be persisted."""
    return _env_float("QCODE_SAVE_FOM_THRESHOLD_CSS")


def save_fom_threshold_noncss() -> float:
    """PBB: minimum trust-adjusted FOM for a credible result to be persisted."""
    return _env_float("QCODE_SAVE_FOM_THRESHOLD_NONCSS")


def save_trust_ratio() -> float:
    """CSS: d/sqrt(n) ratio at or below which a BP-OSD distance estimate is
    fully trusted for persistence/reporting purposes (the "credible" filter,
    distinct from the untrust ceiling used for smooth scoring interpolation)."""
    return _env_float("QCODE_SAVE_TRUST_RATIO")


def save_trust_ratio_noncss() -> float:
    """PBB: d/sqrt(n) ratio boundary between PARTIAL and UNTRUSTED trust."""
    return _env_float("QCODE_SAVE_TRUST_RATIO_NONCSS")


def max_build_candidates_css() -> int:
    """CSS: pre-build candidate cap per lattice (replaces positional
    ``candidates[:5000]`` truncation with hash-stratified selection)."""
    return _env_int("QCODE_MAX_BUILD_CANDIDATES_CSS")


def max_build_candidates_noncss() -> int:
    """PBB: pre-build candidate cap per lattice (replaces positional
    ``candidates[:3000]`` truncation with hash-stratified selection)."""
    return _env_int("QCODE_MAX_BUILD_CANDIDATES_NONCSS")


def weight5_connectivity_policy() -> str:
    """Policy for translation-disconnected weight-five candidates.

    ``"reject"`` is the fail-closed default for new campaign evaluations.
    Set ``QCODE_WEIGHT5_CONNECTIVITY_POLICY=allow`` only when deliberately
    reproducing a historical run that predated the connectivity audit.
    Invalid values raise rather than silently disabling the gate.
    """
    return _env_choice(
        "QCODE_WEIGHT5_CONNECTIVITY_POLICY", {"allow", "reject"}
    )
