#!/usr/bin/env python3
"""Recover model attribution for the historical weight-five PBB logs.

The original PBB evaluator appended candidate rows before it stamped the
program/model provenance.  This script reconstructs that missing join without
editing the append-only campaign logs:

* output rows are grouped into evaluation batches by their sub-second write
  times;
* completed evaluations are joined to their immediately preceding batch;
* delayed writes from timed-out large-profile evaluations are joined by
  replaying the archived generator on all six campaign lattices, requiring an
  exact ``total_candidates`` match and containment of every emitted candidate;
* model labels are read only from the campaign's ``model_attribution.jsonl``.

The sidecars contain one row per order-normalized polynomial/lattice
specification.  ``model_aliases`` are confirmed associations.
``possible_additional_model_aliases`` is nonempty only when the archived
evidence cannot exclude another association, in which case
``attribution_status`` is ``confirmed_minimum`` rather than ``exact``.

Usage:
    python scripts/recover_weight5_pbb_attribution.py
    python scripts/recover_weight5_pbb_attribution.py --check
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = ROOT / "results" / "evolution_metrics.jsonl"
LARGE_LATTICES = ((12, 9), (12, 12), (15, 12), (16, 12), (15, 14), (18, 12))
BATCH_GAP_SECONDS = 1.0
DIRECT_LOG_LAG_SECONDS = 1.0


@dataclass(frozen=True)
class Campaign:
    short: str
    run_id: str
    expected_rows: int
    expected_batches: int
    expected_specs: int
    expected_retained: int

    @property
    def directory(self) -> Path:
        return ROOT / "results" / "evolution" / self.run_id

    @property
    def raw_path(self) -> Path:
        return self.directory / "all_codes_weight5_pbb.jsonl"

    @property
    def attribution_path(self) -> Path:
        return self.directory / "model_attribution.jsonl"

    @property
    def program_dir(self) -> Path:
        return self.directory / "checkpoints" / "checkpoint_300" / "programs"

    @property
    def output_path(self) -> Path:
        return self.directory / "historical_attribution.jsonl"


CAMPAIGNS = (
    Campaign("small", "weight5_pbb_small_v1", 4890, 162, 257, 151),
    Campaign("large", "weight5_pbb_large_v1", 2589, 129, 104, 79),
)

# Seed copies created when runs were started or resumed.  A null model is
# legitimate only for these explicitly audited IDs.
NULL_SEED_PROGRAM_IDS = {
    "weight5_pbb_small_v1": {
        "564d1eaa-58be-48a8-8266-6b0b08a70811",
    },
    "weight5_pbb_large_v1": {
        "e7047855-d5ec-41bc-b2b1-932c526d8f73",
        "875dc701-a679-4f2d-b5f1-ef2d1e5567f5",
        "c2d55ea7-34a5-41d3-af62-39135006cb7e",
        "ae16aa11-9b8e-42e6-8688-50c738ffbda5",
        "7c9d1e19-480f-4ec7-b1b2-e1e7db0013cf",
        "4d67febd-2670-48a4-aaea-9a63e1a6864d",
    },
}
GPT = "azure/gpt-5.6-sol"
OPUS = "aws/claude-opus-5"
EXPECTED_RETAINED_MODEL_SETS = {
    "weight5_pbb_small_v1": {
        (GPT,): 90,
        (GPT, OPUS): 51,
        (GPT, None): 2,
        (GPT, OPUS, None): 8,
    },
    "weight5_pbb_large_v1": {
        (GPT,): 8,
        (GPT, OPUS): 56,
        (GPT, OPUS, None): 11,
        (OPUS,): 4,
    },
}


@dataclass(frozen=True)
class Evaluation:
    program_id: str
    timestamp: float
    started_at: float
    metrics: dict[str, float | bool]

    @property
    def completed_stage2(self) -> bool:
        return "best_fom" in self.metrics and not self.metrics.get("timeout", False)

    @property
    def timed_out_stage2(self) -> bool:
        return bool(self.metrics.get("stage2_passed") == 0 and self.metrics.get("timeout"))


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected object in {path}:{line_number}")
            rows.append(row)
    return rows


def normalized_terms(value: object) -> tuple[tuple[int, int], ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"polynomial terms are not a sequence: {value!r}")
    terms = []
    for term in value:
        if not isinstance(term, (list, tuple)) or len(term) != 2:
            raise ValueError(f"invalid polynomial term: {term!r}")
        x, y = term
        if not isinstance(x, int) or not isinstance(y, int):
            raise ValueError(f"non-integral polynomial term: {term!r}")
        terms.append((x, y))
    return tuple(sorted(terms))


def spec_key(row: dict) -> tuple:
    return (
        int(row["ell"]),
        int(row["m"]),
        normalized_terms(row.get("A_terms")),
        normalized_terms(row.get("B_terms")),
        normalized_terms(row.get("C_terms")),
        normalized_terms(row.get("D_terms")),
    )


def candidate_key(ell: int, m: int, candidate: object) -> tuple | None:
    if not isinstance(candidate, (list, tuple)) or len(candidate) != 4:
        return None
    try:
        a, b, c, d = candidate
        return (
            ell,
            m,
            normalized_terms(a),
            normalized_terms(b),
            normalized_terms(c),
            normalized_terms(d),
        )
    except (TypeError, ValueError):
        return None


def batch_rows(rows: list[dict]) -> list[list[dict]]:
    batches: list[list[dict]] = []
    for row in rows:
        timestamp = float(row["timestamp"])
        if not batches or timestamp - float(batches[-1][-1]["timestamp"]) > BATCH_GAP_SECONDS:
            batches.append([])
        batches[-1].append(row)
    return batches


LOG_PATTERN = re.compile(
    r"^(?P<date>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(?P<millis>\d{3})"
    r" - openevolve\.evaluator - INFO - Evaluated program "
    r"(?P<program>[0-9a-f-]+) in (?P<elapsed>[0-9.]+)s: "
    r"(?P<metrics>.*)$"
)
COMPLETION_PATTERN = re.compile(
    r"^(?P<date>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(?P<millis>\d{3})"
    r" - openevolve\.process_parallel - INFO - Iteration \d+: Program "
    r"(?P<program>[0-9a-f-]+).* completed in [0-9.]+s$"
)


def parse_metric_values(text: str) -> dict[str, float | bool | str]:
    values: dict[str, float | bool | str] = {}
    for field in text.split(", "):
        if "=" not in field:
            continue
        key, raw_value = field.split("=", 1)
        if raw_value in {"True", "False"}:
            values[key] = raw_value == "True"
        elif re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", raw_value):
            values[key] = float(raw_value)
        else:
            values[key] = raw_value
    return values


def load_evaluations(campaign: Campaign) -> list[Evaluation]:
    evaluations = []
    for path in sorted((campaign.directory / "logs").glob("*.log")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                match = LOG_PATTERN.match(line)
                if match is None:
                    continue
                timestamp = dt.datetime.strptime(
                    match.group("date"), "%Y-%m-%d %H:%M:%S"
                ).replace(tzinfo=dt.timezone.utc).timestamp()
                timestamp += int(match.group("millis")) / 1000
                elapsed = float(match.group("elapsed"))
                evaluations.append(
                    Evaluation(
                        program_id=match.group("program"),
                        timestamp=timestamp,
                        started_at=timestamp - elapsed,
                        metrics=parse_metric_values(match.group("metrics")),
                    )
                )
    evaluations.sort(key=lambda event: event.timestamp)
    return evaluations


def load_completion_events(campaign: Campaign) -> list[tuple[float, str]]:
    events = []
    for path in sorted((campaign.directory / "logs").glob("*.log")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                match = COMPLETION_PATTERN.match(line.rstrip())
                if match is None:
                    continue
                timestamp = dt.datetime.strptime(
                    match.group("date"), "%Y-%m-%d %H:%M:%S"
                ).replace(tzinfo=dt.timezone.utc).timestamp()
                timestamp += int(match.group("millis")) / 1000
                events.append((timestamp, match.group("program")))
    return sorted(events)


def load_models(campaign: Campaign) -> dict[str, str | None]:
    models: dict[str, str | None] = {}
    for row in load_jsonl(campaign.attribution_path):
        program_id = str(row["program_id"])
        model = row.get("generating_model")
        if program_id in models and models[program_id] != model:
            raise ValueError(f"conflicting model labels for {program_id}")
        if model is None and program_id not in NULL_SEED_PROGRAM_IDS[campaign.run_id]:
            raise ValueError(f"unexpected null model label for {program_id}")
        models[program_id] = model
    return models


def load_programs(campaign: Campaign) -> dict[str, dict]:
    programs = {}
    for path in campaign.program_dir.glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("id") != path.stem:
            raise ValueError(f"program ID/path mismatch: {path}")
        programs[path.stem] = row
    return programs


def recovery_source_digest(campaign: Campaign) -> str:
    """Hash every immutable input used by this campaign's recovery."""
    paths = [
        campaign.raw_path,
        campaign.attribution_path,
        campaign.directory / "run_manifest.json",
        *sorted((campaign.directory / "logs").glob("*.log")),
    ]
    if campaign.short == "large":
        paths.extend(sorted(campaign.program_dir.glob("*.json")))
        paths.append(METRICS_PATH)
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item.relative_to(ROOT))):
        relative = str(path.relative_to(ROOT)).encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        contents = path.read_bytes()
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def direct_batch_map(
    batches: list[list[dict]], evaluations: Iterable[Evaluation]
) -> dict[int, set[str]]:
    """Map completed evaluations to the batch flushed just before the log."""
    mapping: dict[int, set[str]] = {}
    for event in evaluations:
        if not event.completed_stage2:
            continue
        candidates = []
        for index, batch in enumerate(batches):
            lag = event.timestamp - float(batch[-1]["timestamp"])
            if 0 <= lag < DIRECT_LOG_LAG_SECONDS:
                candidates.append(index)
        if len(candidates) != 1:
            raise ValueError(
                f"{event.program_id}: expected one immediately preceding batch, "
                f"found {candidates}"
            )
        index = candidates[0]
        if index in mapping:
            raise ValueError(f"batch {index} maps to multiple completed evaluations")
        mapping[index] = {event.program_id}
    return mapping


def replay_program(
    program: dict, wanted: set[tuple], evaluator_safety_net: set[tuple]
) -> tuple[int, frozenset[tuple]]:
    """Return the six-lattice raw count and relevant generated specifications."""
    namespace = {"__name__": "_historical_pbb_generator_replay_"}
    code = str(program["code"])
    exec(compile(code, f"<archived program {program['id']}>", "exec"), namespace)
    generate = namespace.get("generate_candidates")
    if not callable(generate):
        raise ValueError(f"{program['id']}: no callable generate_candidates")

    total = 0
    # The evaluator independently reinserted the campaign seed's safety net
    # after calling an evolved generator.  It is intentionally not included
    # in ``total_candidates`` but must be included in the membership test.
    emitted: set[tuple] = set(evaluator_safety_net)
    for ell, m in LARGE_LATTICES:
        candidates = generate(ell, m)
        if not isinstance(candidates, list):
            raise ValueError(f"{program['id']}: generator did not return a list")
        total += len(candidates)
        for candidate in candidates:
            key = candidate_key(ell, m, candidate)
            if key in wanted:
                emitted.add(key)
    return total, frozenset(emitted)


def load_historical_safety_net(campaign: Campaign) -> set[tuple]:
    manifest = json.loads(
        (campaign.directory / "run_manifest.json").read_text(encoding="utf-8")
    )
    seed_path = ROOT / str(manifest["seed_path"])
    seed_bytes = seed_path.read_bytes()
    if hashlib.sha256(seed_bytes).hexdigest() != manifest["seed_hash"]:
        raise ValueError(f"campaign seed no longer matches {campaign.run_id} manifest")
    namespace = {"__name__": "_historical_pbb_seed_"}
    exec(compile(seed_bytes, str(seed_path), "exec"), namespace)
    safety = namespace.get("_safety_net_candidates")
    if not callable(safety):
        raise ValueError(f"{seed_path}: missing _safety_net_candidates")
    return {
        key
        for ell, m in LARGE_LATTICES
        for candidate in safety(ell, m)
        if (key := candidate_key(ell, m, candidate)) is not None
    }


def perfect_matching(adjacency: dict[int, set[str]]) -> dict[int, str] | None:
    """Return a complete deterministic bipartite matching, if one exists."""
    right_to_left: dict[str, int] = {}

    def augment(left: int, visited: set[str]) -> bool:
        for right in sorted(adjacency[left]):
            if right in visited:
                continue
            visited.add(right)
            incumbent = right_to_left.get(right)
            if incumbent is None or augment(incumbent, visited):
                right_to_left[right] = left
                return True
        return False

    for left in sorted(adjacency, key=lambda item: (len(adjacency[item]), item)):
        if not augment(left, set()):
            return None
    return {left: right for right, left in right_to_left.items()}


def edges_in_perfect_matchings(adjacency: dict[int, set[str]]) -> dict[int, set[str]]:
    """Discard locally plausible edges incompatible with any global join."""
    if perfect_matching(adjacency) is None:
        raise ValueError("delayed large-PBB join has no perfect matching")
    allowed: dict[int, set[str]] = {left: set() for left in adjacency}
    for fixed_left in sorted(adjacency):
        for fixed_right in sorted(adjacency[fixed_left]):
            remainder = {
                left: rights - {fixed_right}
                for left, rights in adjacency.items()
                if left != fixed_left
            }
            if all(remainder.values()) and perfect_matching(remainder) is not None:
                allowed[fixed_left].add(fixed_right)
        if not allowed[fixed_left]:
            raise ValueError(f"large batch {fixed_left}: no globally consistent join")
    return allowed


def metric_batches(batches: list[list[dict]]) -> dict[int, dict]:
    """Join each post-restart large batch to its metrics row by timestamp."""
    metrics = load_jsonl(METRICS_PATH)
    mapping: dict[int, dict] = {}
    for row in metrics:
        timestamp = float(row["timestamp"])
        matches = [
            index
            for index, batch in enumerate(batches)
            if 0 <= timestamp - float(batch[-1]["timestamp"]) < 1
        ]
        if len(matches) != 1:
            raise ValueError(
                f"metrics row at {timestamp}: expected one preceding batch, found {matches}"
            )
        index = matches[0]
        if index in mapping:
            raise ValueError(f"large batch {index}: multiple metrics rows")
        mapping[index] = row
    if len(mapping) != len(metrics):
        raise ValueError(
            f"matched {len(mapping)} large batches to {len(metrics)} metrics rows"
        )
    return mapping


def large_possible_programs(
    campaign: Campaign,
    batches: list[list[dict]],
    evaluations: list[Evaluation],
    direct: dict[int, str],
    programs: dict[str, dict],
) -> dict[int, set[str]]:
    """Recover delayed stage-two writes from timed-out worker threads."""
    metrics = metric_batches(batches)
    result = {index: set(program_ids) for index, program_ids in direct.items()}
    # For a timed-out worker, persistence normally finished in the worker
    # just before the process-pool completion callback.  This supplies an
    # equally direct timestamp join for delayed stage-two evaluations.
    for timestamp, program_id in load_completion_events(campaign):
        matches = [
            index
            for index, batch in enumerate(batches)
            if 0 <= timestamp - float(batch[-1]["timestamp"]) < DIRECT_LOG_LAG_SECONDS
        ]
        if len(matches) != 1:
            continue
        index = matches[0]
        if index in result and result[index] != {program_id}:
            raise ValueError(f"conflicting direct joins for large batch {index}")
        result[index] = {program_id}
    unmatched_batches = set(range(len(batches))) - set(result)

    directly_joined_ids = {
        program_id for program_ids in result.values() for program_id in program_ids
    }
    timed_out = [
        event
        for event in evaluations
        if event.timed_out_stage2 and event.program_id not in directly_joined_ids
    ]
    if len(timed_out) != len(unmatched_batches):
        raise ValueError(
            f"{len(timed_out)} stage-two timeouts for {len(unmatched_batches)} delayed batches"
        )

    # The first delayed batch came from the pre-restart seed.  Its archived
    # program record was pruned, but its timeout log and null attribution
    # remain and it is chronologically isolated by more than eight hours.
    first = min(unmatched_batches)
    missing_programs = [event for event in timed_out if event.program_id not in programs]
    if first != 0 or len(missing_programs) != 1:
        raise ValueError("unexpected missing-program layout in large campaign")
    result[first] = {missing_programs[0].program_id}
    unmatched_batches.remove(first)
    timed_out = [event for event in timed_out if event.program_id in programs]

    wanted = {spec_key(row) for batch in batches for row in batch}
    evaluator_safety_net = load_historical_safety_net(campaign)
    replay_cache: dict[str, tuple[int, frozenset[tuple]]] = {}
    program_replay: dict[str, tuple[int, frozenset[tuple]]] = {}
    for number, event in enumerate(timed_out, 1):
        program = programs[event.program_id]
        digest = hashlib.sha256(str(program["code"]).encode()).hexdigest()
        if digest not in replay_cache:
            replay_cache[digest] = replay_program(program, wanted, evaluator_safety_net)
        program_replay[event.program_id] = replay_cache[digest]
        if number % 10 == 0:
            print(f"replayed {number}/{len(timed_out)} delayed large-PBB programs", file=sys.stderr)

    # A program can only have produced a batch after its evaluation began.
    # The exact raw generator count and full emitted-row containment are the
    # primary join keys.  Keep all surviving alternatives for the audit.
    event_by_id = {event.program_id: event for event in timed_out}
    possible: dict[int, set[str]] = {}
    for index in sorted(unmatched_batches):
        batch = batches[index]
        expected_total = int(metrics[index]["total_candidates"])
        keys = {spec_key(row) for row in batch}
        completed_at = float(batch[-1]["timestamp"])
        matches = set()
        for program_id, (total, emitted) in program_replay.items():
            if total != expected_total or not keys.issubset(emitted):
                continue
            event = event_by_id[program_id]
            # A delayed write is the continuation of a stage-two call after
            # OpenEvolve's timeout was reported.  It must therefore occur
            # after that program's timeout (and, trivially, after its start).
            if event.started_at > completed_at or event.timestamp > completed_at:
                continue
            matches.add(program_id)
        if not matches:
            raise ValueError(
                f"large batch {index}: no replay matches total={expected_total}"
            )
        possible[index] = matches

    # Enforce the campaign-wide one-evaluation/one-batch relation.  This is
    # stronger than independent local filtering and leaves an edge only when
    # it occurs in at least one complete assignment of all delayed batches.
    result.update(edges_in_perfect_matchings(possible))
    return result


def retained_keys(rows: list[dict]) -> set[tuple]:
    """Apply the supplement's d-upper-endpoint >= 3 filter to raw evidence."""
    upper: dict[tuple, int] = {}
    for row in rows:
        key = spec_key(row)
        distance = int(row["d"])
        upper[key] = min(upper.get(key, distance), distance)
    return {key for key, distance in upper.items() if distance >= 3}


def model_sort_key(model: str | None) -> tuple[int, str]:
    order = {"azure/gpt-5.6-sol": 0, "aws/claude-opus-5": 1, None: 2}
    return order.get(model, 3), "" if model is None else model


def build_sidecar(campaign: Campaign) -> tuple[list[dict], Counter]:
    rows = load_jsonl(campaign.raw_path)
    if len(rows) != campaign.expected_rows:
        raise ValueError(f"{campaign.run_id}: expected {campaign.expected_rows} rows, got {len(rows)}")
    batches = batch_rows(rows)
    if len(batches) != campaign.expected_batches:
        raise ValueError(
            f"{campaign.run_id}: expected {campaign.expected_batches} batches, got {len(batches)}"
        )
    evaluations = load_evaluations(campaign)
    models = load_models(campaign)
    programs = load_programs(campaign)
    source_digest = recovery_source_digest(campaign)
    direct = direct_batch_map(batches, evaluations)

    if campaign.short == "small":
        completed = [event for event in evaluations if event.completed_stage2]
        if len(completed) != len(batches):
            raise ValueError("small campaign does not have one completed evaluation per batch")
        batch_programs = direct
    else:
        batch_programs = large_possible_programs(
            campaign, batches, evaluations, direct, programs
        )

    if set(batch_programs) != set(range(len(batches))):
        missing = sorted(set(range(len(batches))) - set(batch_programs))
        raise ValueError(f"{campaign.run_id}: unattributed batches {missing}")

    # Never silently reinterpret a missing journal row as a seed.  The
    # allowlist handles only explicitly audited restart-seed IDs.
    for ids in batch_programs.values():
        for program_id in ids:
            if program_id not in models:
                if program_id not in NULL_SEED_PROGRAM_IDS[campaign.run_id]:
                    raise ValueError(f"missing model-attribution row for {program_id}")
                models[program_id] = None

    observations: dict[tuple, list[tuple[int, dict]]] = defaultdict(list)
    for batch_index, batch in enumerate(batches):
        for row in batch:
            observations[spec_key(row)].append((batch_index, row))
    if len(observations) != campaign.expected_specs:
        raise ValueError(
            f"{campaign.run_id}: expected {campaign.expected_specs} specs, got {len(observations)}"
        )

    sidecar = []
    status_counts: Counter = Counter()
    for key in sorted(observations):
        batch_indices = sorted({index for index, _ in observations[key]})
        batch_model_sets = [
            {models[program_id] for program_id in batch_programs[index]}
            for index in batch_indices
        ]
        confirmed: set[str | None] = set()
        possible: set[str | None] = set()
        confirmed_programs: set[str] = set()
        possible_programs: set[str] = set()
        for index, model_set in zip(batch_indices, batch_model_sets):
            ids = batch_programs[index]
            possible.update(model_set)
            possible_programs.update(ids)
            if len(model_set) == 1:
                confirmed.update(model_set)
            if len(ids) == 1:
                confirmed_programs.update(ids)
        additional = possible - confirmed
        status = "exact" if not additional else "confirmed_minimum"
        status_counts[status] += 1
        ell, m, a, b, c, d = key
        example = observations[key][0][1]
        sidecar.append(
            {
                "run_id": campaign.run_id,
                "code_key": json.dumps(
                    [ell, m, a, b, c, d], separators=(",", ":")
                ),
                "ell": ell,
                "m": m,
                "A_terms": a,
                "B_terms": b,
                "C_terms": c,
                "D_terms": d,
                "n": int(example["n"]),
                "k": int(example["k"]),
                "observation_count": len(observations[key]),
                "batch_indices": batch_indices,
                "model_aliases": sorted(confirmed, key=model_sort_key),
                "possible_additional_model_aliases": sorted(
                    additional, key=model_sort_key
                ),
                "program_ids": sorted(confirmed_programs),
                "possible_program_ids": sorted(possible_programs - confirmed_programs),
                "attribution_status": status,
                "recovery_source_sha256": source_digest,
                "recovery_method": (
                    "ordered_timestamp_join"
                    if campaign.short == "small"
                    else "timestamp_or_generator_replay_join"
                ),
            }
        )

    retained = retained_keys(rows)
    if len(retained) != campaign.expected_retained:
        raise ValueError(
            f"{campaign.run_id}: expected {campaign.expected_retained} retained specs, "
            f"got {len(retained)}"
        )
    retained_rows = [row for row in sidecar if spec_key(row) in retained]
    if len(retained_rows) != campaign.expected_retained:
        raise ValueError(f"{campaign.run_id}: retained sidecar coverage failed")
    status_counts["retained"] = len(retained_rows)
    status_counts["retained_exact"] = sum(
        row["attribution_status"] == "exact" for row in retained_rows
    )
    status_counts["retained_minimum"] = sum(
        row["attribution_status"] == "confirmed_minimum" for row in retained_rows
    )
    for row in retained_rows:
        uniquely_backed_models = {
            models[program_id] for program_id in row["program_ids"]
        }
        if set(row["model_aliases"]) - uniquely_backed_models:
            raise ValueError(
                f"{campaign.run_id}: retained {row['code_key']} has an alias "
                "without a uniquely joined program"
            )
        labels = tuple(row["model_aliases"])
        status_counts[("retained_labels", labels)] += 1
    actual_distribution = {
        labels: status_counts[("retained_labels", labels)]
        for labels in EXPECTED_RETAINED_MODEL_SETS[campaign.run_id]
    }
    if actual_distribution != EXPECTED_RETAINED_MODEL_SETS[campaign.run_id]:
        raise ValueError(
            f"{campaign.run_id}: retained model-set distribution changed: "
            f"{actual_distribution}"
        )
    return sidecar, status_counts


def render_jsonl(rows: Iterable[dict]) -> str:
    return "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="fail if either sidecar is stale"
    )
    args = parser.parse_args()

    stale = False
    for campaign in CAMPAIGNS:
        rows, counts = build_sidecar(campaign)
        rendered = render_jsonl(rows)
        if args.check:
            if not campaign.output_path.exists() or campaign.output_path.read_text(
                encoding="utf-8"
            ) != rendered:
                print(f"stale: {campaign.output_path.relative_to(ROOT)}", file=sys.stderr)
                stale = True
            else:
                print(f"up to date: {campaign.output_path.relative_to(ROOT)}")
        else:
            campaign.output_path.write_text(rendered, encoding="utf-8")
            print(f"wrote {campaign.output_path.relative_to(ROOT)}")

        label_counts = {}
        for key, count in counts.items():
            if not isinstance(key, tuple) or key[0] != "retained_labels":
                continue
            labels = key[1]
            label = "/".join("N" if value is None else value for value in labels)
            label_counts[label] = count
        print(
            f"{campaign.run_id}: {len(rows)} specs; {counts['retained']} retained; "
            f"{counts['retained_exact']} exact attribution, "
            f"{counts['retained_minimum']} confirmed-minimum; labels={label_counts}; "
            f"sources={rows[0]['recovery_source_sha256']}"
        )
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
