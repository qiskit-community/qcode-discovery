"""Authoritative, append-only discovery-event log for OpenEvolve runs
(Phase E2).

Each accepted/saved code result gets ONE JSONL line appended to
``results/evolution/<run_id>/discovery_events.jsonl``. Multiple OS worker
processes (OpenEvolve's real concurrency primitive — see
``evolve/model_attribution.py``'s module docstring and its Phase E1 section
for the traced evidence that our evaluator code runs inside
``ProcessPoolExecutor`` workers) may append concurrently; each append is
performed under an exclusive ``fcntl.flock`` on the file, released
immediately after, so writes from different processes are serialized and
never interleave/corrupt a line.

Event identity and dedup strategy
----------------------------------
``event_id = SHA256(run_id, program_id, code_key)`` (see
:func:`compute_event_id`) is deterministic: retrying the same program
re-evaluating the same code (same ``program_id`` + same ``code_key``)
produces the SAME id, while a genuinely different program independently
discovering the same code gets its own id (different ``program_id``) and is
retained as a distinct rediscovery.

Of the two dedupe strategies the design allows (a fresh read-scan of the
existing file under lock before appending, vs. accept harmless duplicate
lines and dedupe at reduce time), this module does the **read-scan-under-lock**
before appending: it is simplest to reason about (no separate index file to
keep consistent, no dependence on the Phase E4 reducer having run), pays for
itself only on the rare "OpenEvolve internal retry" path, and event logs are
modest in size (at most thousands of lines per run) so an O(n) scan per write
is cheap. The scan and the append happen inside the SAME held lock, so no
other writer can interleave a duplicate between the scan and the write.

Fields recorded per event (schema-v1, deliberately NOT dependent on any
schema-v2 fields such as ``d_lower``/``d_upper`` that a separate,
concurrently in-progress piece of work may add to result dicts): identity
(``event_id``, ``run_id``, ``program_id``, ``model_alias``,
``generating_model``, ``source_hash``, ``code_key``), the code's defining
parameters (``ell``, ``m``, ``A_terms``, ``B_terms``, ``C_terms``/``D_terms``
— null for CSS-only runs), distance/quality fields (``d``, ``d_is_exact``,
``k``, ``n``, ``fom``), provenance (``stage``, ``timestamp``, ``config_hash``,
``seed_hash``, ``git_commit``), and the two fields that are genuinely
unavailable at write time (``parent_id``, ``iteration`` — both ``null``; see
``evolve/provenance_reducer.py`` for why, and how they get backfilled
post-hoc). Any OTHER keys present on the result dict are passed through
verbatim under ``extra`` rather than dropped, so this module doesn't need to
track every possible evaluator-specific field name.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Iterator

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_ROOT = _PROJECT_ROOT / "results" / "evolution"

# Fields promoted to top-level keys on every event; anything else on the
# result dict is passed through under "extra" (see module docstring).
_TOP_LEVEL_RESULT_FIELDS = (
    "ell", "m", "A_terms", "B_terms", "C_terms", "D_terms",
    "n", "k", "d", "d_is_exact", "fom", "stage",
)
# Keys never copied into "extra" even if present on the result dict --
# e.g. non-JSON-serializable / huge objects some pipelines attach transiently
# (none currently reach evaluate_stage*'s all_results, but this is a cheap
# defense against a future one doing so; json.dumps(..., default=str) below
# is the second line of defense).
_EXCLUDE_FROM_EXTRA = {"code"}


def _events_path(run_id: str) -> Path:
    return _RESULTS_ROOT / run_id / "discovery_events.jsonl"


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------

def compute_event_id(run_id: Any, program_id: Any, code_key: Any) -> str:
    """Deterministic SHA-256 hex digest of ``(run_id, program_id, code_key)``.

    ``code_key`` may be any JSON-serializable value (this module always
    passes the JSON-string form produced in :func:`append_discovery_event`,
    but the function itself doesn't require that).
    """
    payload = json.dumps([str(run_id), str(program_id), code_key], default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_commit_hash(cwd: str | os.PathLike | None = None) -> str | None:
    """Current git HEAD commit hash, or None if unavailable (e.g. not a repo)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd or _PROJECT_ROOT),
            capture_output=True, text=True, timeout=10, check=True,
        )
        commit = out.stdout.strip()
        return commit or None
    except Exception:
        return None


def git_is_dirty(cwd: str | os.PathLike | None = None) -> bool | None:
    """True if ``git status --porcelain`` is non-empty, None if unavailable."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(cwd or _PROJECT_ROOT),
            capture_output=True, text=True, timeout=10, check=True,
        )
        return bool(out.stdout.strip())
    except Exception:
        return None


def file_content_hash(path: str | os.PathLike | None) -> str | None:
    """SHA-256 hex digest of a file's bytes, or None if unreadable/None."""
    if not path:
        return None
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Writer / reader
# ---------------------------------------------------------------------------

def append_discovery_event(
    run_id: str,
    result: dict,
    *,
    program_id: str,
    source_hash: str | None,
    config_hash: str | None = None,
    seed_hash: str | None = None,
    git_commit: str | None = None,
    model_alias: str | None = None,
    generating_model: str | None = None,
    stage: str | None = None,
) -> str:
    """Append one discovery event for ``result``, deduped by ``event_id``.

    Returns the ``event_id`` in all cases (whether newly appended or a
    duplicate that was skipped) -- callers that want to know which happened
    can compare against a prior call, but for logging purposes the id alone
    is normally what's needed.

    ``config_hash``/``seed_hash`` fall back to the ``QCODE_RUN_CONFIG_HASH``/
    ``QCODE_RUN_SEED_HASH`` env vars (mirroring the existing ``QCODE_RUN_NAME``
    convention in ``evolve/run_evolution.py``) so a launcher can export them
    once and have every worker process's events pick them up without
    threading them through the evaluator call chain. ``git_commit`` defaults
    to :func:`git_commit_hash` (computed fresh here) if not supplied.
    """
    from evaluation.results import _code_key

    code_key_str = json.dumps(_code_key(result))
    event_id = compute_event_id(run_id, program_id, code_key_str)

    config_hash = config_hash if config_hash is not None else os.environ.get("QCODE_RUN_CONFIG_HASH")
    seed_hash = seed_hash if seed_hash is not None else os.environ.get("QCODE_RUN_SEED_HASH")
    git_commit = git_commit if git_commit is not None else git_commit_hash()

    event = {
        "event_id": event_id,
        "run_id": run_id,
        "program_id": program_id,
        "model_alias": model_alias,
        "generating_model": generating_model,
        "source_hash": source_hash,
        "code_key": code_key_str,
        "stage": stage if stage is not None else result.get("stage"),
        "timestamp": time.time(),
        "config_hash": config_hash,
        "seed_hash": seed_hash,
        "git_commit": git_commit,
        # Genuinely unavailable at write time -- see module docstring and
        # evolve/provenance_reducer.py.
        "parent_id": None,
        "iteration": None,
    }
    for key in _TOP_LEVEL_RESULT_FIELDS:
        if key == "stage":
            # Already set above with the caller-supplied `stage` taking
            # precedence over result["stage"] -- do not clobber that choice.
            continue
        event[key] = result.get(key)
    event["extra"] = {
        k: v for k, v in result.items()
        if k not in _TOP_LEVEL_RESULT_FIELDS and k not in _EXCLUDE_FROM_EXTRA
    }

    path = _events_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    # a+ : create if missing, don't truncate, allow both read (for the dedup
    # scan) and append. All access to the file (scan + write) happens while
    # holding the exclusive lock so no other cooperating writer can interleave
    # a duplicate append between the scan and this write.
    with open(path, "a+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if existing.get("event_id") == event_id:
                    return event_id  # duplicate -- already recorded, skip
            f.seek(0, os.SEEK_END)
            f.write(json.dumps(event, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    return event_id


def iter_discovery_events(run_id: str) -> Iterator[dict]:
    """Yield every recorded event for ``run_id``, in file order.

    Read-only. Takes a shared lock for the duration of the read so it never
    observes a write from :func:`append_discovery_event` half-completed
    (advisory and cooperative -- only effective against other flock users,
    which is exactly the other function in this module).
    """
    path = _events_path(run_id)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        except OSError:
            pass
        try:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
        finally:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
