"""Post-run join/reducer that backfills ``parent_id``/``iteration`` onto
discovery events (Phase E4).

Why this exists: ``evolve/discovery_events.py`` writes each event from
*inside* ``Evaluator.evaluate_program`` (see that module and
``evolve/model_attribution.py``'s Phase E1 docstring section for the traced
call chain), which happens strictly *before* OpenEvolve constructs the child
``Program(id=child_id, parent_id=parent.id, iteration_found=iteration, ...)``
object in ``_run_iteration_worker`` (``openevolve/process_parallel.py``).
So ``parent_id``/``iteration`` are genuinely unknown at event-write time --
confirmed by inspecting that exact ordering in the installed package, not
merely asserted by the plan this implements.

``evolve/model_attribution.py``'s ``install_attribution_log`` records exactly
those two missing fields (plus ``generating_model``/``combined_score``) once
the ``Program`` *is* built and accepted into the database, keyed by
``program_id``, in ``<run_dir>/model_attribution.jsonl``. This module joins
the two logs by ``program_id`` after the fact.

This is a **read-mostly, single-pass, post-run** operation -- unlike
``discovery_events.append_discovery_event``, which must tolerate many
concurrent OS-process writers *during* a run, :func:`reduce_run` is expected
to be invoked after (or well outside the hot path of) a run, so a single
exclusive lock held for the whole reduce+write is simple and sufficient; it
does not need the finer-grained read-scan-then-append dance the event log
uses.

Output: ``results/evolution/<run_id>/discoveries_enriched.json``, one entry
per DISTINCT code (grouped by ``code_key``), each carrying a ``discovered_by``
list of entries keyed by ``(run_id, program_id)`` -- so the same program
joining twice contributes only one entry, while two different programs that
independently discovered the same code both get their own entry. The
authoritative ``discovery_events.jsonl`` itself is never modified or deleted;
this module only ever *reads* it and writes a *derived* artifact.

A code entry's top-level quality fields (``n``, ``k``, ``d``, ``d_is_exact``,
``fom``) are seeded from the FIRST event seen for that code_key but upgraded
in place whenever a LATER event for the same code carries strictly better
evidence -- see :func:`_is_better_quality`. Without this, a rediscovery that
happened to prove exactness (or land a tighter heuristic estimate) after an
earlier, weaker measurement would have its evidence silently ignored at the
top level (it would still be visible per-entry inside ``discovered_by``, but
the aggregate fields callers actually rank/filter on would stay stale).

Idempotency: re-running :func:`reduce_run` on an unchanged event log and
attribution log produces byte-identical output (a fresh reduction from
scratch each time, not an incremental merge against the previous enriched
file), so calling it any number of times is safe and never accumulates
duplicates.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from evolve.discovery_events import iter_discovery_events

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_ROOT = _PROJECT_ROOT / "results" / "evolution"


def _run_dir(run_id: str) -> Path:
    return _RESULTS_ROOT / run_id


def _attribution_path(run_id: str) -> Path:
    return _run_dir(run_id) / "model_attribution.jsonl"


def _enriched_path(run_id: str) -> Path:
    return _run_dir(run_id) / "discoveries_enriched.json"


def _lock_path(run_id: str) -> Path:
    return _run_dir(run_id) / ".discoveries_enriched.lock"


def _load_attribution_index(run_id: str) -> dict[str, dict[str, Any]]:
    """Map ``program_id -> {"iteration", "parent_id", "generating_model",
    "combined_score"}`` from ``<run_dir>/model_attribution.jsonl``.

    Missing/unreadable file yields an empty index (every event will then be
    ``join_status: "unjoined"``, which is a valid, retained outcome -- never
    an error).
    """
    path = _attribution_path(run_id)
    index: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return index
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = record.get("program_id")
            if pid is None:
                continue
            # Last write wins if a program_id appears more than once (should
            # not normally happen -- ProgramDatabase.add is called once per
            # accepted program -- but favor the most recent record rather
            # than erroring).
            index[pid] = {
                "iteration": record.get("iteration"),
                "parent_id": record.get("parent_id"),
                "generating_model": record.get("generating_model"),
                "combined_score": record.get("combined_score"),
            }
    return index


def _join_event(event: dict, attribution_index: dict[str, dict[str, Any]]) -> dict:
    """Return a copy of ``event`` with ``parent_id``/``iteration`` filled in
    from ``attribution_index`` when the join succeeds, plus a ``join_status``
    field (``"joined"`` or ``"unjoined"``). The event is returned either way
    -- an unsuccessful join never causes the event to be dropped.
    """
    joined = dict(event)
    attrib = attribution_index.get(event.get("program_id"))
    if attrib is not None:
        joined["parent_id"] = attrib.get("parent_id")
        joined["iteration"] = attrib.get("iteration")
        # generating_model from attribution is a cross-check against the
        # event's own model_alias/generating_model captured at eval time;
        # keep both rather than overwriting -- they should agree, and
        # disagreement itself is diagnostic information worth keeping.
        joined["generating_model_attrib"] = attrib.get("generating_model")
        joined["combined_score"] = attrib.get("combined_score")
        joined["join_status"] = "joined"
    else:
        joined.setdefault("parent_id", None)
        joined.setdefault("iteration", None)
        joined["join_status"] = "unjoined"
    return joined


def merge_discovered_by(existing: list[dict], new_entry: dict) -> list[dict]:
    """Add ``new_entry`` to ``existing`` (a code's ``discovered_by`` list),
    deduped by ``(run_id, program_id)``.

    Returns a NEW list (does not mutate ``existing``) with ``new_entry``
    appended unless an entry with the same ``(run_id, program_id)`` is
    already present, in which case ``existing`` is returned unchanged
    (as a shallow copy) -- the same program joining twice contributes only
    one ``discovered_by`` entry.
    """
    key = (new_entry.get("run_id"), new_entry.get("program_id"))
    for entry in existing:
        if (entry.get("run_id"), entry.get("program_id")) == key:
            return list(existing)
    return list(existing) + [new_entry]


def _new_code_entry(event: dict) -> dict:
    """Seed a fresh per-code aggregate entry from ``event``'s code-defining
    and quality fields."""
    return {
        "code_key": event.get("code_key"),
        "ell": event.get("ell"),
        "m": event.get("m"),
        "A_terms": event.get("A_terms"),
        "B_terms": event.get("B_terms"),
        "C_terms": event.get("C_terms"),
        "D_terms": event.get("D_terms"),
        "n": event.get("n"),
        "k": event.get("k"),
        "d": event.get("d"),
        "d_is_exact": event.get("d_is_exact"),
        "fom": event.get("fom"),
        "discovered_by": [],
    }


def _quality_fields_from_event(event: dict) -> dict:
    return {
        "n": event.get("n"),
        "k": event.get("k"),
        "d": event.get("d"),
        "d_is_exact": event.get("d_is_exact"),
        "fom": event.get("fom"),
    }


def _is_better_quality(candidate: dict, current: dict) -> bool:
    """True if ``candidate`` (a new event's quality fields) is strictly
    better evidence than ``current`` (a code entry's stored quality fields)
    for the same code.

    Two events sharing a code key are two independent measurements of the
    SAME code -- unlike ``distance_schema``'s v2 records, these bare
    discovery-event quality fields carry only a point estimate ``d`` (no
    ``d_lower``/``d_upper``), so the ordering this function encodes is
    intentionally conservative:

    * A proof (``d_is_exact=True``) always beats a heuristic
      (``d_is_exact`` falsy) -- an exact result is strictly more informative
      regardless of the numeric ``d`` values involved.
    * Between two proofs, or between two heuristics, neither is a priori
      "better" without bound-tracking, EXCEPT that among two heuristic
      (non-exact) point estimates a smaller ``d`` is a tighter upper bound
      (mirroring ``distance_schema.merge_bounds``'s ``upper = min(...)``
      convention) and is therefore preferred.
    """
    candidate_exact = bool(candidate.get("d_is_exact"))
    current_exact = bool(current.get("d_is_exact"))
    if candidate_exact and not current_exact:
        return True
    if current_exact and not candidate_exact:
        return False
    if not candidate_exact and not current_exact:
        candidate_d = candidate.get("d")
        current_d = current.get("d")
        if candidate_d is not None and current_d is not None:
            return candidate_d < current_d
    return False


def _do_reduce(run_id: str) -> dict:
    """Pure(ish) reduction: read the two input logs, join, group by
    code_key. No locking, no writing -- see :func:`reduce_run` for that.
    """
    attribution_index = _load_attribution_index(run_id)

    codes: dict[str, dict] = {}
    total_events = 0
    joined_count = 0
    for event in iter_discovery_events(run_id):
        total_events += 1
        joined = _join_event(event, attribution_index)
        if joined["join_status"] == "joined":
            joined_count += 1

        code_key = joined.get("code_key")
        entry = codes.get(code_key)
        if entry is None:
            entry = _new_code_entry(joined)
            codes[code_key] = entry
        elif _is_better_quality(_quality_fields_from_event(joined), entry):
            entry.update(_quality_fields_from_event(joined))

        discovered_by_entry = {
            "run_id": joined.get("run_id"),
            "program_id": joined.get("program_id"),
            "event_id": joined.get("event_id"),
            "model_alias": joined.get("model_alias"),
            "generating_model": joined.get("generating_model"),
            "generating_model_attrib": joined.get("generating_model_attrib"),
            "parent_id": joined.get("parent_id"),
            "iteration": joined.get("iteration"),
            "combined_score": joined.get("combined_score"),
            "join_status": joined.get("join_status"),
            "stage": joined.get("stage"),
            "timestamp": joined.get("timestamp"),
            "config_hash": joined.get("config_hash"),
            "seed_hash": joined.get("seed_hash"),
            "git_commit": joined.get("git_commit"),
            "source_hash": joined.get("source_hash"),
        }
        entry["discovered_by"] = merge_discovered_by(entry["discovered_by"], discovered_by_entry)

    return {
        "run_id": run_id,
        "total_events": total_events,
        "joined_events": joined_count,
        "unjoined_events": total_events - joined_count,
        "distinct_codes": len(codes),
        "codes": list(codes.values()),
    }


def _atomic_publish(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".discoveries_enriched.", suffix=".tmp", dir=str(path.parent))
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


def reduce_run(run_id: str) -> dict:
    """Join ``discovery_events.jsonl`` with ``model_attribution.jsonl`` for
    ``run_id`` and atomically (re)write ``discoveries_enriched.json``.

    A single exclusive ``fcntl.flock`` on a dedicated lock file
    (``.discoveries_enriched.lock``, in the same run directory) serializes
    concurrent invocations of this function against each other; the actual
    write is additionally atomic via temp-file + ``os.replace`` so a reader
    of ``discoveries_enriched.json`` never observes a partial write even
    without taking the lock itself. Never touches ``discovery_events.jsonl``
    (read-only access via :func:`evolve.discovery_events.iter_discovery_events`).

    Returns the payload that was written (also useful directly in tests
    without re-reading the file from disk).
    """
    run_dir = _run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = _lock_path(run_id)

    with open(lock_path, "a+") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            payload = _do_reduce(run_id)
            _atomic_publish(_enriched_path(run_id), payload)
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    return payload


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(f"usage: python -m evolve.provenance_reducer <run_id>", file=sys.stderr)
        raise SystemExit(2)
    result = reduce_run(sys.argv[1])
    print(json.dumps(
        {k: v for k, v in result.items() if k != "codes"} | {"distinct_codes": result["distinct_codes"]},
        indent=2,
    ))
