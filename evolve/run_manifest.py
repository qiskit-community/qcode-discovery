"""Atomic run-launch manifest for OpenEvolve runs (Phase E3).

:func:`write_run_manifest` is called ONCE, at launch, before evolution
starts, from the run's own results directory
(``results/evolution/<run_id>/run_manifest.json``). It captures everything
needed to answer "what exactly produced this run's results" after the fact:
identity (run id, campaign name), code provenance (git commit + dirty flag),
content hashes of the exact config/seed/evaluator files used (so a later
diff against the live files in the repo shows whether they've since
changed), the model ensemble's aliases and weights, the *effective* gate
environment at launch (see :func:`_effective_gates` below), and the
evolution RNG seed.

Security constraint (do not weaken)
------------------------------------
This file NEVER records API keys, authorization headers, or any other
secret material. "Model aliases" means only the alias strings and their
ensemble weights as configured -- never a resolved API key, base URL with an
embedded token, or any value pulled from an environment variable whose name
suggests a secret. :func:`_scrub_secrets` is a defensive last line: even if a
caller accidentally passes something secret-shaped in ``extra``, this module
redacts it before writing rather than trusting every call site to have been
careful. See the module-level ``_SECRET_KEY_PATTERN``/``_looks_like_secret``
below.

Atomicity
---------
Written via temp-file-in-the-same-directory + ``os.replace`` (POSIX atomic
rename), so a reader never observes a partially-written manifest, and a
crash mid-write leaves the old file (or no file), never a corrupt one.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from evolve.discovery_events import file_content_hash, git_commit_hash, git_is_dirty

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_ROOT = _PROJECT_ROOT / "results" / "evolution"

# Defensive secret scrubbing (see module docstring). Matches common
# key/token/secret/password-shaped dict keys, case-insensitively.
_SECRET_KEY_PATTERN = re.compile(
    r"(api[_-]?key|secret|token|authorization|auth[_-]?header|password|passwd|"
    r"bearer|access[_-]?key|private[_-]?key)",
    re.IGNORECASE,
)
_REDACTED = "<redacted>"


def _looks_like_secret_key(key: str) -> bool:
    return bool(_SECRET_KEY_PATTERN.search(str(key)))


def _scrub_secrets(obj: Any) -> Any:
    """Recursively strip anything under a secret-shaped key.

    Applied to the whole manifest payload right before writing, as a
    defense-in-depth measure -- the manifest is built from known-safe inputs
    (config-declared alias strings/weights, gate getters, file hashes), but
    this scrub means a future caller accidentally passing something
    secret-shaped via ``extra`` still can't leak it into the manifest.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if _looks_like_secret_key(k):
                out[k] = _REDACTED
            else:
                out[k] = _scrub_secrets(v)
        return out
    if isinstance(obj, list):
        return [_scrub_secrets(v) for v in obj]
    return obj


def _effective_gates() -> dict[str, Any]:
    """Snapshot every public getter in ``evaluation.gates`` at call time.

    Calls the actual getters (not raw env var reads) so the recorded values
    are exactly what the gate module would hand back to evaluator code at
    this moment -- including defaults when no override env var is set.
    """
    from evaluation import gates

    snapshot = {}
    for name, fn in inspect.getmembers(gates, inspect.isfunction):
        if name.startswith("_"):
            continue
        try:
            snapshot[name] = fn()
        except Exception as exc:  # pragma: no cover - defensive only
            snapshot[name] = f"<error: {exc}>"
    return snapshot


def _manifest_path(run_id: str) -> Path:
    return _RESULTS_ROOT / run_id / "run_manifest.json"


def _json_safe(obj: Any) -> Any:
    """Recursively replace non-finite floats (inf/-inf/nan) with string
    sentinels so the manifest round-trips through strict JSON parsers.

    Gate getters like ``evaluation.gates.fom_threshold_exact_or_disabled``
    default to ``float("inf")`` when disabled; Python's ``json.dump`` emits
    that as the bare (non-standard) token ``Infinity`` unless caught here,
    which ``JSON.parse``/``jq`` and other strict readers reject outright.
    """
    if isinstance(obj, float):
        if obj != obj:  # nan
            return "nan"
        if obj == float("inf"):
            return "inf"
        if obj == float("-inf"):
            return "-inf"
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".run_manifest.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, default=str, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)  # atomic on POSIX within the same filesystem
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def write_run_manifest(
    run_id: str,
    *,
    campaign_name: str,
    config_path: str | os.PathLike | None,
    seed_path: str | os.PathLike | None,
    evaluator_path: str | os.PathLike | None,
    model_aliases: dict[str, float] | list[dict[str, Any]] | None,
    rng_seed: Any,
    extra: dict[str, Any] | None = None,
    manifest_path: str | os.PathLike | None = None,
) -> Path:
    """Write ``results/evolution/<run_id>/run_manifest.json`` atomically.

    ``model_aliases`` should be exactly the alias -> weight mapping (or a
    list of ``{"alias": ..., "weight": ...}`` records) taken from the
    ensemble config -- never anything resolved from it (no API keys, no
    resolved base URLs with embedded credentials). This function scrubs
    secret-shaped keys defensively (see :func:`_scrub_secrets`), but callers
    must not rely on that as the only safeguard.

    ``extra`` is an open slot for anything else worth recording (e.g. lattice
    list, iteration budget); it also gets scrubbed. The "approved budget
    deviations" escape hatch lives at ``manifest["approved_budget_deviations"]``,
    defaulting to an empty list; a human can hand-edit the written JSON file
    later to populate it -- this function only ever writes the empty default.

    ``manifest_path`` overrides the default ``results/evolution/<run_id>/
    run_manifest.json`` location -- e.g. so a caller whose actual output
    directory diverges from that fixed convention (``run_evolution.py``'s
    ``--output`` flag) writes the manifest next to its other run artifacts
    instead of into the fixed ``_RESULTS_ROOT`` tree.
    """
    config_path = str(config_path) if config_path else None
    seed_path = str(seed_path) if seed_path else None
    evaluator_path = str(evaluator_path) if evaluator_path else None

    manifest = {
        "run_id": run_id,
        "campaign_name": campaign_name,
        "git_commit": git_commit_hash(),
        "git_dirty": git_is_dirty(),
        "config_path": config_path,
        "config_hash": file_content_hash(config_path),
        "seed_path": seed_path,
        "seed_hash": file_content_hash(seed_path),
        "evaluator_path": evaluator_path,
        "evaluator_hash": file_content_hash(evaluator_path),
        "model_aliases": model_aliases,
        "effective_gates": _effective_gates(),
        "rng_seed": rng_seed,
        # Escape hatch: empty by default, meant for a human to populate by
        # hand-editing the written file if a deliberate deviation from the
        # nominal budget/schedule needs to be recorded and justified.
        "approved_budget_deviations": [],
    }
    if extra:
        manifest["extra"] = extra

    manifest = _scrub_secrets(manifest)
    manifest = _json_safe(manifest)

    path = Path(manifest_path) if manifest_path is not None else _manifest_path(run_id)
    _atomic_write_json(path, manifest)
    return path
