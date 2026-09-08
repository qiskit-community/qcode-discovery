"""Tests for the deterministic CSS weight-four audit and its artifact."""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_weight5_css_low_weight.py"
ARTIFACT = ROOT / "results" / "weight5_css_low_weight_audit.jsonl"
CATALOGUE = ROOT / "results" / "weight5_publication_catalogue.jsonl"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "audit_weight5_css_low_weight", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rowspace(rows: list[int]) -> set[int]:
    values = {0}
    for row in rows:
        values |= {value ^ row for value in tuple(values)}
    return values


def _brute_sector_distance(
    check_rows: list[int], stabilizer_rows: list[int], num_qubits: int
) -> int | None:
    stabilizers = _rowspace(stabilizer_rows)
    for weight in range(1, 5):
        for support in itertools.combinations(range(num_qubits), weight):
            vector = sum(1 << index for index in support)
            if vector in stabilizers:
                continue
            if all((row & vector).bit_count() % 2 == 0 for row in check_rows):
                return weight
    return None


def test_sector_hash_search_matches_independent_brute_force():
    module = _load_script()
    rng = random.Random(20260903)
    num_qubits = 8
    for _ in range(30):
        stabilizers = [rng.randrange(1 << num_qubits) for _ in range(3)]
        orthogonal = [
            vector
            for vector in range(1 << num_qubits)
            if all((row & vector).bit_count() % 2 == 0 for row in stabilizers)
        ]
        check_rows = [orthogonal[rng.randrange(len(orthogonal))] for _ in range(3)]
        expected = _brute_sector_distance(
            check_rows, stabilizers, num_qubits
        )
        observed = module.audit_css_sector(
            check_rows, stabilizers, num_qubits
        )
        assert observed["exact_distance"] == expected
        assert observed["certified_lower_bound"] == (
            5 if expected is None else expected
        )


def test_bb_check_builder_and_two_sector_audit():
    module = _load_script()
    rows_x, rows_z = module.build_bb_check_rows(
        2,
        3,
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1), (0, 2)],
    )
    assert len(rows_x) == len(rows_z) == 6
    assert {row.bit_count() for row in rows_x + rows_z} == {5}
    result = module.audit_css_rows(rows_x, rows_z, 12)
    assert result["result"] == "exact_low_weight"
    assert result["exact_distance"] == 2


def test_checked_in_artifact_covers_every_direct_u_css_presentation():
    module = _load_script()
    rows = [
        json.loads(line)
        for line in ARTIFACT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest, audits = rows[0], rows[1:]
    assert manifest["record_type"] == "artifact_manifest"
    assert manifest["artifact_schema"] == module.ARTIFACT_SCHEMA
    assert len(audits) == 492
    assert len({row["class_identity"]["class_id"] for row in audits}) == 492

    target_ids = {
        presentation_id
        for row in audits
        for presentation_id in row["class_identity"]["target_direct_u_member_ids"]
    }
    retained_audit_targets = {
        row["presentation_id"]
        for line in CATALOGUE.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
        if row["css_low_weight_audit"] is not None
    }
    excluded_targets = {
        presentation_id
        for row in audits
        if row["audit_evidence"]["exact_distance"] in {1, 2}
        for presentation_id in row["class_identity"]["target_direct_u_member_ids"]
    }
    assert target_ids == retained_audit_targets | excluded_targets
    assert len(target_ids) == 858
    assert len(retained_audit_targets) == 856
    assert excluded_targets == {"CL-666ca92e", "CS-611d21a8"}
    assert manifest["counts"]["low_weight_exact_presentations_by_distance"] == {
        "2": 2,
        "3": 42,
        "4": 50,
    }
    assert manifest["counts"]["post_audit_status_by_target_presentation"] == {
        "C": 735,
        "E": 123,
    }
    for descriptor in manifest["source_artifacts"].values():
        path = ROOT / descriptor["path"]
        assert descriptor["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    for row in audits:
        assert row["representative"]["checks_commute"] is True
        assert row["representative"]["k_rebuilt"] == row["representative"][
            "k_catalogue"
        ]
        for sector in (row["search"]["X"], row["search"]["Z"]):
            witness = sector["witness"]
            if witness is None:
                assert sector["certified_lower_bound"] == 5
            else:
                assert len(witness["qubit_indices_zero_based"]) == sector[
                    "exact_distance"
                ]


def test_post_audit_distance_does_not_close_on_decoder_estimate():
    module = _load_script()
    audit = {"exact_distance": None, "certified_lower_bound": 5}
    source = {
        "lower": None,
        "upper": 5,
        "is_exact": False,
        "upper_is_supported": False,
    }

    observed = module._post_audit_distance(source, audit)

    assert observed == {
        "status_code": "C",
        "status": "certified_lower_with_estimated_upper",
        "lower": 5,
        "upper": 5,
        "is_exact": False,
        "upper_is_supported": False,
        "new_evidence_method": "L4",
    }


def test_post_audit_distance_closes_on_rigorous_upper_bound():
    module = _load_script()
    audit = {"exact_distance": None, "certified_lower_bound": 5}
    source = {
        "lower": None,
        "upper": 5,
        "is_exact": False,
        "upper_is_supported": True,
    }

    observed = module._post_audit_distance(source, audit)

    assert observed["status_code"] == "E"
    assert observed["status"] == "exact"
    assert observed["is_exact"] is True
