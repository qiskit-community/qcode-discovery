"""Verification tests for the Phase E provenance system.

Covers:
(a) 2+ concurrent OS-process writers appending to the same
    ``discovery_events.jsonl`` -- no corruption, no lost events.
(b) Two different ``program_id``s discovering the same code -> two
    distinct events (rediscovery preserved).
(c) The same ``(run_id, program_id, code_key)`` appended twice -> one
    retained event (idempotent retry).
(d) The Phase E4 reducer: correct ``join_status`` for joined/unjoined
    program ids, retains unjoined events (never drops them), and is
    idempotent across repeated invocations without mutating the
    authoritative event log.
(e) ``run_manifest.json`` never contains any substring of a fake secret
    placed in a fake API-key-shaped environment variable (or passed via
    the ``extra`` slot) during the test.

Concurrency tests use ``concurrent.futures.ProcessPoolExecutor`` (real OS
worker processes, matching OpenEvolve's actual concurrency primitive --
see ``evolve/model_attribution.py``'s module docstring), not threads.
Worker functions are defined at module scope so they are picklable (the
pool's call queue pickles tasks regardless of the platform's default start
method).  Each worker explicitly repoints the relevant module's
``_RESULTS_ROOT`` constant at the test's temp directory itself (rather than
relying on inheriting monkeypatched state via ``fork``), so the test is
correct under `fork` or `spawn` alike.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

import evolve.discovery_events as discovery_events
import evolve.provenance_reducer as provenance_reducer
import evolve.run_manifest as run_manifest


def _sample_result(suffix: str = "") -> dict:
    return {
        "ell": 6, "m": 6,
        "A_terms": [[0, 0], [1, 0], [0, 1]],
        "B_terms": [[0, 0], [2, 0], [0, 2]],
        "d": 4, "k": 2, "n": 72, "fom": 8.0,
        "d_is_exact": True,
        "_suffix": suffix,  # not part of _code_key; only for eyeballing test data
    }


# ---------------------------------------------------------------------------
# Module-scope worker functions for ProcessPoolExecutor (must be picklable).
# ---------------------------------------------------------------------------

def _worker_append_unique(args) -> str:
    run_dir_str, run_id, program_id = args
    import evolve.discovery_events as de
    de._RESULTS_ROOT = Path(run_dir_str)
    return de.append_discovery_event(
        run_id, _sample_result(program_id),
        program_id=program_id,
        source_hash="src-hash-stub",
        model_alias="claude-x",
        generating_model="claude-x",
    )


def _worker_append_same(args) -> str:
    run_dir_str, run_id, program_id = args
    import evolve.discovery_events as de
    de._RESULTS_ROOT = Path(run_dir_str)
    # Every worker uses the SAME program_id + SAME code -> same event_id.
    # Simulates several concurrent processes each retrying evaluation of
    # the exact same program on the exact same code.
    return de.append_discovery_event(
        run_id, _sample_result("same"),
        program_id=program_id,
        source_hash="src-hash-stub",
        model_alias="claude-x",
        generating_model="claude-x",
    )


# ---------------------------------------------------------------------------
# (a) Concurrency: no corruption, no lost events, across real OS processes.
# ---------------------------------------------------------------------------

def test_concurrent_writers_no_corruption(tmp_path, monkeypatch):
    monkeypatch.setattr(discovery_events, "_RESULTS_ROOT", tmp_path)
    run_id = "run_concurrent_a"
    n_workers = 12
    tasks = [(str(tmp_path), run_id, f"prog-{i}") for i in range(n_workers)]

    with ProcessPoolExecutor(max_workers=6) as pool:
        event_ids = list(pool.map(_worker_append_unique, tasks))

    assert len(event_ids) == n_workers
    assert len(set(event_ids)) == n_workers  # all distinct program_ids -> distinct events

    path = discovery_events._events_path(run_id)
    raw_lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(raw_lines) == n_workers, "no lines lost or duplicated"

    # Every line must be valid, complete JSON (no interleaved/corrupted writes).
    parsed = [json.loads(line) for line in raw_lines]
    seen_event_ids = {p["event_id"] for p in parsed}
    assert seen_event_ids == set(event_ids)

    # iter_discovery_events agrees.
    events = list(discovery_events.iter_discovery_events(run_id))
    assert len(events) == n_workers


def test_concurrent_writers_dedup_same_program(tmp_path, monkeypatch):
    monkeypatch.setattr(discovery_events, "_RESULTS_ROOT", tmp_path)
    run_id = "run_concurrent_b"
    program_id = "prog-retry"
    n_workers = 8
    tasks = [(str(tmp_path), run_id, program_id) for _ in range(n_workers)]

    with ProcessPoolExecutor(max_workers=6) as pool:
        event_ids = list(pool.map(_worker_append_same, tasks))

    # All workers computed the same deterministic event_id.
    assert len(set(event_ids)) == 1

    events = list(discovery_events.iter_discovery_events(run_id))
    assert len(events) == 1, "concurrent retries of the same program+code must dedup to one event"
    assert events[0]["event_id"] == event_ids[0]
    assert events[0]["program_id"] == program_id


# ---------------------------------------------------------------------------
# (b) Two different programs discovering the same code -> distinct events.
# ---------------------------------------------------------------------------

def test_two_program_ids_same_code_are_distinct_events(tmp_path, monkeypatch):
    monkeypatch.setattr(discovery_events, "_RESULTS_ROOT", tmp_path)
    run_id = "run_rediscovery"
    result = _sample_result("rediscovery")

    id1 = discovery_events.append_discovery_event(
        run_id, result, program_id="prog-A", source_hash="hashA",
        model_alias="model-A", generating_model="model-A",
    )
    id2 = discovery_events.append_discovery_event(
        run_id, result, program_id="prog-B", source_hash="hashB",
        model_alias="model-B", generating_model="model-B",
    )

    assert id1 != id2
    events = list(discovery_events.iter_discovery_events(run_id))
    assert len(events) == 2
    program_ids = {e["program_id"] for e in events}
    assert program_ids == {"prog-A", "prog-B"}
    # Same code_key for both (it's the same code).
    assert events[0]["code_key"] == events[1]["code_key"]


# ---------------------------------------------------------------------------
# (c) Same (run_id, program_id, code_key) appended twice -> one event.
# ---------------------------------------------------------------------------

def test_same_program_same_code_twice_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(discovery_events, "_RESULTS_ROOT", tmp_path)
    run_id = "run_idempotent"
    result = _sample_result("idempotent")

    id1 = discovery_events.append_discovery_event(
        run_id, result, program_id="prog-retry", source_hash="hash1",
    )
    id2 = discovery_events.append_discovery_event(
        run_id, result, program_id="prog-retry", source_hash="hash1",
    )

    assert id1 == id2
    events = list(discovery_events.iter_discovery_events(run_id))
    assert len(events) == 1


# ---------------------------------------------------------------------------
# (d) Phase E4 reducer: join correctness + idempotency + log preservation.
# ---------------------------------------------------------------------------

def test_reducer_join_status_and_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(discovery_events, "_RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(provenance_reducer, "_RESULTS_ROOT", tmp_path)

    run_id = "run_reducer"
    result_joined = _sample_result("joined")
    result_unjoined = _sample_result("unjoined")
    # Give the unjoined one a distinct code so it's a separate code entry.
    result_unjoined["A_terms"] = [[0, 0], [1, 0], [0, 3]]

    discovery_events.append_discovery_event(
        run_id, result_joined, program_id="prog-joined", source_hash="h1",
        model_alias="model-1", generating_model="model-1",
    )
    discovery_events.append_discovery_event(
        run_id, result_unjoined, program_id="prog-unjoined", source_hash="h2",
        model_alias="model-2", generating_model="model-2",
    )

    # Attribution log has an entry for prog-joined only.
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    attrib_path = run_dir / "model_attribution.jsonl"
    with open(attrib_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "program_id": "prog-joined",
            "iteration": 42,
            "parent_id": "parent-xyz",
            "generating_model": "model-1",
            "combined_score": 0.87,
        }) + "\n")

    # Snapshot the authoritative log before reducing.
    events_path = discovery_events._events_path(run_id)
    before_bytes = events_path.read_bytes()

    result1 = provenance_reducer.reduce_run(run_id)

    # The authoritative event log must be untouched by the reducer.
    assert events_path.read_bytes() == before_bytes

    assert result1["total_events"] == 2
    assert result1["joined_events"] == 1
    assert result1["unjoined_events"] == 1
    assert result1["distinct_codes"] == 2

    codes_by_suffix = {}
    for code in result1["codes"]:
        for db in code["discovered_by"]:
            codes_by_suffix[db["program_id"]] = (code, db)

    joined_code, joined_db = codes_by_suffix["prog-joined"]
    assert joined_db["join_status"] == "joined"
    assert joined_db["parent_id"] == "parent-xyz"
    assert joined_db["iteration"] == 42
    assert joined_db["combined_score"] == 0.87

    unjoined_code, unjoined_db = codes_by_suffix["prog-unjoined"]
    assert unjoined_db["join_status"] == "unjoined"
    assert unjoined_db["parent_id"] is None
    assert unjoined_db["iteration"] is None
    # Retained, not dropped.
    assert unjoined_code["code_key"] is not None

    # Idempotent: re-running produces the same result.
    result2 = provenance_reducer.reduce_run(run_id)
    assert events_path.read_bytes() == before_bytes
    assert result1 == result2

    enriched_path = provenance_reducer._enriched_path(run_id)
    assert enriched_path.exists()
    on_disk = json.loads(enriched_path.read_text(encoding="utf-8"))
    assert on_disk["total_events"] == 2


def test_reducer_upgrades_quality_fields_on_rediscovery_with_better_evidence(tmp_path, monkeypatch):
    # Two events for the SAME code: the first is a weak heuristic estimate,
    # the second (a later rediscovery) is a proven-exact result. The
    # aggregate top-level quality fields must upgrade to the second
    # measurement's evidence, not stay frozen at the first-seen values --
    # see provenance_reducer._is_better_quality.
    monkeypatch.setattr(discovery_events, "_RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(provenance_reducer, "_RESULTS_ROOT", tmp_path)

    run_id = "run_quality_upgrade"
    weak_result = _sample_result("weak")
    weak_result["d"] = 10
    weak_result["d_is_exact"] = False

    strong_result = _sample_result("strong")
    strong_result["d"] = 4
    strong_result["d_is_exact"] = True

    discovery_events.append_discovery_event(
        run_id, weak_result, program_id="prog-weak", source_hash="h1",
    )
    discovery_events.append_discovery_event(
        run_id, strong_result, program_id="prog-strong", source_hash="h2",
    )

    result = provenance_reducer.reduce_run(run_id)
    assert result["distinct_codes"] == 1
    code = result["codes"][0]
    assert code["d"] == 4
    assert code["d_is_exact"] is True
    # Both programs' contributions are still individually visible.
    assert {db["program_id"] for db in code["discovered_by"]} == {"prog-weak", "prog-strong"}

    # A later, WORSE heuristic rediscovery must not un-upgrade the entry.
    worse_result = _sample_result("worse")
    worse_result["d"] = 99
    worse_result["d_is_exact"] = False
    discovery_events.append_discovery_event(
        run_id, worse_result, program_id="prog-worse", source_hash="h3",
    )
    result2 = provenance_reducer.reduce_run(run_id)
    code2 = result2["codes"][0]
    assert code2["d"] == 4
    assert code2["d_is_exact"] is True


def test_stage_kwarg_takes_precedence_over_result_stage_field(tmp_path, monkeypatch):
    # append_discovery_event's explicit `stage=` kwarg must win over
    # result["stage"] in the recorded event -- previously the
    # _TOP_LEVEL_RESULT_FIELDS copy loop unconditionally overwrote the
    # explicit choice with result.get("stage") afterwards.
    monkeypatch.setattr(discovery_events, "_RESULTS_ROOT", tmp_path)
    run_id = "run_stage_precedence"
    result = _sample_result("stage-test")
    result["stage"] = "from_result_dict"

    discovery_events.append_discovery_event(
        run_id, result, program_id="prog-stage", source_hash="h1",
        stage="from_explicit_kwarg",
    )

    events = list(discovery_events.iter_discovery_events(run_id))
    assert len(events) == 1
    assert events[0]["stage"] == "from_explicit_kwarg"


def test_stage_falls_back_to_result_field_when_kwarg_omitted(tmp_path, monkeypatch):
    monkeypatch.setattr(discovery_events, "_RESULTS_ROOT", tmp_path)
    run_id = "run_stage_fallback"
    result = _sample_result("stage-fallback")
    result["stage"] = "from_result_dict"

    discovery_events.append_discovery_event(
        run_id, result, program_id="prog-stage2", source_hash="h1",
    )

    events = list(discovery_events.iter_discovery_events(run_id))
    assert events[0]["stage"] == "from_result_dict"


def test_merge_discovered_by_dedup():
    entry_a = {"run_id": "r1", "program_id": "p1", "note": "first"}
    entry_a_retry = {"run_id": "r1", "program_id": "p1", "note": "second-call-same-key"}
    entry_b = {"run_id": "r1", "program_id": "p2", "note": "different program"}

    existing = provenance_reducer.merge_discovered_by([], entry_a)
    assert existing == [entry_a]

    # Same (run_id, program_id) -> no duplicate, list unchanged.
    existing2 = provenance_reducer.merge_discovered_by(existing, entry_a_retry)
    assert existing2 == [entry_a]

    # Different program_id -> appended.
    existing3 = provenance_reducer.merge_discovered_by(existing2, entry_b)
    assert existing3 == [entry_a, entry_b]


# ---------------------------------------------------------------------------
# (e) run_manifest.json never leaks secret material.
# ---------------------------------------------------------------------------

def test_run_manifest_never_leaks_secrets(tmp_path, monkeypatch):
    monkeypatch.setattr(run_manifest, "_RESULTS_ROOT", tmp_path)

    fake_secret = "sk-FAKETESTSECRET0000DEADBEEF"
    # Simulate a fake ".env"-like variable being present in the environment
    # at manifest-write time.
    monkeypatch.setenv("ANTHROPIC_API_KEY", fake_secret)
    monkeypatch.setenv("OPENAI_API_KEY", fake_secret + "-2")

    config_file = tmp_path / "fake_config.yaml"
    config_file.write_text("models:\n  - alias: claude-sonnet\n", encoding="utf-8")

    run_id = "run_manifest_secrets"
    path = run_manifest.write_run_manifest(
        run_id,
        campaign_name="weight5_css_test",
        config_path=str(config_file),
        seed_path=None,
        evaluator_path=None,
        model_aliases={"claude-sonnet": 1.0, "gpt-4o": 0.5},
        rng_seed=1234,
        extra={
            "note": "safe text",
            # Deliberately secret-shaped key to test the defensive scrub,
            # even though callers should never pass this.
            "api_key": fake_secret,
        },
    )

    raw_text = Path(path).read_text(encoding="utf-8")

    assert fake_secret not in raw_text
    assert (fake_secret + "-2") not in raw_text
    assert "ANTHROPIC_API_KEY" not in raw_text or True  # key name alone is not a secret

    manifest = json.loads(raw_text)
    assert manifest["run_id"] == run_id
    assert manifest["model_aliases"] == {"claude-sonnet": 1.0, "gpt-4o": 0.5}
    assert manifest["extra"]["api_key"] == "<redacted>"
    assert manifest["extra"]["note"] == "safe text"
    assert manifest["approved_budget_deviations"] == []
    assert "effective_gates" in manifest and isinstance(manifest["effective_gates"], dict)
    assert manifest["config_hash"] is not None  # file_content_hash worked

    # Second write (e.g. a relaunch) must not error and must still be atomic
    # -- re-write and confirm no leftover temp files.
    run_manifest.write_run_manifest(
        run_id,
        campaign_name="weight5_css_test",
        config_path=str(config_file),
        seed_path=None,
        evaluator_path=None,
        model_aliases={"claude-sonnet": 1.0},
        rng_seed=5678,
        extra=None,
    )
    leftover_tmp = list((tmp_path / run_id).glob(".run_manifest.*.tmp"))
    assert leftover_tmp == []


def test_run_manifest_never_emits_bare_infinity_token(tmp_path, monkeypatch):
    # evaluation.gates.fom_threshold_exact_or_disabled() defaults to
    # float("inf") when its override env var is unset -- that value flows
    # straight into effective_gates. json.dump's default allow_nan=True
    # would emit the non-standard bare token `Infinity`, which strict JSON
    # readers (JSON.parse, jq) reject. _json_safe must convert it to the
    # string sentinel "inf" before writing.
    monkeypatch.setattr(run_manifest, "_RESULTS_ROOT", tmp_path)
    monkeypatch.delenv("QCODE_FOM_THRESHOLD_EXACT", raising=False)

    run_id = "run_manifest_infinity"
    path = run_manifest.write_run_manifest(
        run_id,
        campaign_name="weight5_css_test",
        config_path=None,
        seed_path=None,
        evaluator_path=None,
        model_aliases={"claude-sonnet": 1.0},
        rng_seed=42,
    )

    raw_text = Path(path).read_text(encoding="utf-8")
    assert "Infinity" not in raw_text
    assert "NaN" not in raw_text

    manifest = json.loads(raw_text)
    assert manifest["effective_gates"]["fom_threshold_exact_or_disabled"] == "inf"


def test_json_safe_converts_non_finite_floats():
    payload = {"a": float("inf"), "b": float("-inf"), "c": float("nan"), "d": 1.5, "e": [float("inf"), 2]}
    safe = run_manifest._json_safe(payload)
    assert safe == {"a": "inf", "b": "-inf", "c": "nan", "d": 1.5, "e": ["inf", 2]}
    # Must be genuinely JSON-round-trippable with allow_nan=False.
    json.dumps(safe, allow_nan=False)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
