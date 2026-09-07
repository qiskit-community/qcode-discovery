"""Integrity checks for the normalized weight-five publication archive."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from evaluation.connectivity import bicycle_translation_component_count


ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = ROOT / "results" / "weight5_publication_catalogue.jsonl"
MANIFEST = ROOT / "results" / "weight5_publication_manifest.json"


@pytest.fixture(scope="module")
def records() -> list[dict]:
    with CATALOGUE.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_publication_catalogue_accounting_and_distance_semantics(records, manifest):
    assert len(records) == 1_142
    assert len({row["presentation_id"] for row in records}) == 1_142
    assert {row["schema_version"] for row in records} == {1}
    assert {row["record_type"] for row in records} == {"weight5_presentation"}
    assert Counter(row["family"] for row in records) == {"CSS": 912, "PBB": 230}

    direct = Counter(row["distance_direct"]["status_code"] for row in records)
    assert direct == {"E": 398, "C": 744}
    assert all(row["distance_direct"]["upper"] >= 3 for row in records)

    for row in records:
        distance = row["distance_direct"]
        if distance["status_code"] == "E":
            assert distance["is_exact"]
            assert distance["lower"] == distance["upper"]
        elif distance["status_code"] == "C":
            assert not distance["is_exact"]
            assert 0 < distance["lower"] < distance["upper"]
        else:
            assert distance["status_code"] == "U"
            assert distance["lower"] is None
            assert not distance["lower_is_certified"]
            assert distance["fom_lower"] is None

    pbb_intervals = [
        row
        for row in records
        if row["family"] == "PBB" and row["distance_direct"]["status_code"] == "C"
    ]
    assert len(pbb_intervals) == 25
    assert {row["distance_direct"]["lower"] for row in pbb_intervals} == {5}

    assert manifest["counts"]["retained_presentations"] == 1_142
    assert manifest["counts"]["excluded_presentations"] == 155
    assert manifest["counts"]["source_verification_rows"] == 177
    assert manifest["counts"]["retained_presentations_with_verification"] == 175
    assert manifest["counts"]["retained_verified_presentations_by_campaign"] == {
        "css-large": 78,
        "css-small": 16,
        "pbb-large": 51,
        "pbb-small": 30,
    }
    assert manifest["counts"]["presentations_by_direct_status"] == {
        "C": 744,
        "E": 398,
    }
    assert sum(manifest["counts"]["excluded_by_direct_status"].values()) == 155
    assert set(manifest["counts"]["excluded_by_upper_endpoint"]) <= {"1", "2"}


def test_publication_class_accounting_and_membership(records, manifest):
    by_class = defaultdict(list)
    for row in records:
        by_class[row["equivalence"]["class_id"]].append(row)
    assert len(by_class) == 630

    class_status = Counter()
    all_ids = {row["presentation_id"] for row in records}
    for class_id, members in by_class.items():
        reference = members[0]
        declared_ids = set(reference["equivalence"]["member_ids"])
        assert declared_ids == {row["presentation_id"] for row in members}
        assert declared_ids <= all_ids
        assert reference["equivalence"]["class_size"] == len(members)
        assert reference["equivalence"]["canonicalizer"]["name"] == (
            "python-igraph/BLISS"
        )
        assert all(row["equivalence"]["class_id"] == class_id for row in members)
        shared_distance = {
            key: value
            for key, value in reference["distance_class"].items()
            if key != "inherited_or_tightened"
        }
        assert all(
            {
                key: value
                for key, value in row["distance_class"].items()
                if key != "inherited_or_tightened"
            }
            == shared_distance
            for row in members
        )
        assert shared_distance["evidence"]["decisive_upper_bounds"]
        if shared_distance["status_code"] != "U":
            assert shared_distance["evidence"]["decisive_lower_bounds"]
        class_status[reference["distance_class"]["status_code"]] += 1

    assert class_status == {"E": 227, "C": 403}
    assert manifest["counts"]["retained_classes"] == 630
    assert manifest["counts"]["classes_by_status"] == {
        "C": 403,
        "E": 227,
    }


def test_translation_gate_matches_full_catalogue_connectivity_audit(records):
    for row in records:
        parameters = row["parameters"]
        generators = row["generators"]
        count = bicycle_translation_component_count(
            parameters["ell"],
            parameters["m"],
            generators["A_terms"],
            generators["B_terms"],
            generators["C_terms"],
            generators["D_terms"],
        )
        assert count == row["connectivity"]["n_components"], row[
            "presentation_id"
        ]


def test_component_certificates_are_materialized_and_d2_is_filtered(
    records, manifest
):
    expected_exact = {
        "CL-4b943aa2": 3,
        "CL-929c2a70": 3,
        "CL-b9275593": 3,
        "CL-437b6e5f": 5,
        "CL-d9f65a7e": 3,
        "CL-a25b2baf": 6,
    }
    by_id = {row["presentation_id"]: row for row in records}
    assert "CL-5cd6065f" not in by_id
    assert manifest["component_certification"]["excluded_presentation_ids"] == [
        "CL-5cd6065f"
    ]
    assert set(manifest["component_certification"]["retained_presentation_ids"]) == set(
        expected_exact
    )

    for presentation_id, distance in expected_exact.items():
        row = by_id[presentation_id]
        assert row["distance_direct"]["status_code"] == "E"
        assert row["distance_direct"]["lower"] == distance
        assert row["distance_direct"]["upper"] == distance
        assert "Hc" in row["distance_direct"]["evidence"]["exact_methods"]
        assert "Hc" in row["distance_class"]["evidence"]["exact_methods"]
        certificate = row["component_certification"]
        assert certificate["component_certificate"]["d_exact"] == distance
        assert certificate["distance_transfer"]["parent_d_exact"] == distance
        assert row["connectivity"]["state"] == "disconnected"
        assert row["connectivity"]["component_code"]["distance_equals_parent"]

    explicit_components = {
        "CS-d473da9b": ("W5BB-90-4-9", 90, 4, 9),
        "CL-885dfa58": ("W5BB-96-4-10", 96, 4, 10),
        "CL-f8ef7ae2": ("W5BB-140-6-10", 140, 6, 10),
    }
    for parent_id, expected in explicit_components.items():
        matches = by_id[parent_id]["explicit_bb_component_matches"]
        assert len(matches) == 1
        match = matches[0]
        component = match["explicit_bb_component"]
        assert (
            match["match_id"],
            component["n"],
            component["k"],
            component["d_exact"],
        ) == expected
        assert component["connected"]
        assert match["match"]["bliss_isomorphic"]
        assert match["match"]["distance_transfer_valid"]


def test_css_low_weight_audit_is_materialized_and_d2_rows_are_filtered(
    records, manifest
):
    by_id = {row["presentation_id"]: row for row in records}
    excluded = {"CL-666ca92e", "CS-611d21a8"}
    assert excluded.isdisjoint(by_id)
    audit_manifest = manifest["css_low_weight_audit"]
    assert set(audit_manifest["excluded_presentation_ids"]) == excluded
    assert len(audit_manifest["retained_presentation_ids"]) == 856
    assert manifest["counts"]["css_low_weight_audit_targets"] == 858
    assert manifest["counts"]["css_low_weight_audit_targets_retained"] == 856
    assert (
        manifest["counts"]["css_low_weight_audit_targets_excluded_d_le_2"]
        == 2
    )

    audited = [row for row in records if row["css_low_weight_audit"] is not None]
    assert len(audited) == 856
    assert {row["family"] for row in audited} == {"CSS"}
    assert all(
        "css_low_weight_audit" in row["source_artifacts"] for row in audited
    )
    assert all(
        row["distance_direct"]["lower"] is not None
        for row in records
        if row["family"] == "CSS"
    )


def test_origin_and_missing_attribution_are_explicit(records):
    expected_missing = {
        "CL-929c2a70",
        "CL-d9f65a7e",
        "CL-a25b2baf",
        "CL-4b943aa2",
        "CL-437b6e5f",
        "CL-b9275593",
    }
    missing = {
        row["presentation_id"]
        for row in records
        if row["provenance"]["missing_attribution_reason"]
        == "excluded_by_credibility_event_gate"
    }
    assert missing == expected_missing

    safety_net = [
        row for row in records if row["provenance"]["origin"]["fixed_safety_net"]
    ]
    assert len(safety_net) == 20
    assert {row["family"] for row in safety_net} == {"PBB"}
    assert all(
        "fixed_safety_net_definition" in row["source_artifacts"]
        for row in safety_net
    )

    for row in records:
        provenance = row["provenance"]
        recorded_times = provenance["verification_recorded_at"]
        if provenance["verification_row_count"]:
            assert recorded_times
            assert len(recorded_times) == provenance["verification_row_count"]
            assert provenance["verification_recorded_at_latest"] == max(
                recorded_times
            )
            assert all(timestamp.endswith("+00:00") for timestamp in recorded_times)
        else:
            assert recorded_times == []
            assert provenance["verification_recorded_at_latest"] is None


def test_manifest_hashes_pin_every_archive_input_and_the_catalogue(manifest):
    output = manifest["output_artifacts"]["catalogue"]
    assert output["path"] == "results/weight5_publication_catalogue.jsonl"
    assert output["records"] == 1_142
    assert output["bytes"] == CATALOGUE.stat().st_size
    assert output["sha256"] == _sha256(CATALOGUE)

    descriptors = list(manifest["source_artifacts"].values())
    descriptors.extend(manifest["software_artifacts"].values())
    assert len(manifest["source_artifacts"]) == 14
    assert set(manifest["software_artifacts"]) == {
        "bb_constructor",
        "component_certificate_generator",
        "css_low_weight_audit_generator",
        "connectivity_audit",
        "dependency_lock",
        "dependency_manifest",
        "fixed_pbb_safety_net",
        "pbb_constructor",
        "publication_catalogue_generator",
        "source_normalizer",
        "tanner_canonicalizer",
    }
    for descriptor in descriptors:
        path = ROOT / descriptor["path"]
        assert path.is_file()
        assert descriptor["bytes"] == path.stat().st_size
        assert descriptor["sha256"] == _sha256(path)

    component_sources = manifest["component_certification"]["artifact_manifest"][
        "source_files"
    ]
    pinned_by_path = {
        descriptor["path"]: descriptor["sha256"]
        for descriptor in manifest["source_artifacts"].values()
    }
    assert all(pinned_by_path[path] == digest for path, digest in component_sources.items())

    recorded = manifest["verification_recorded_time_ranges"]
    assert sum(item["records"] for item in recorded.values()) == 177
    assert set(recorded) == {"css-small", "css-large", "pbb-small", "pbb-large"}
    assert all(item["earliest"] <= item["latest"] for item in recorded.values())
    assert (
        recorded["pbb-large"]["semantics"]
        == "legacy_first_pass_not_continuation_completion"
    )
