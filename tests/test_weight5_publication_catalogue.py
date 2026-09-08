"""Integrity checks for the normalized weight-five publication archive."""

from __future__ import annotations

import hashlib
import json
import math
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


def _witness_artifact_rows_with_current_sources() -> list[dict]:
    artifact = ROOT / "results" / "weight5_css_upper_bound_witnesses.jsonl"
    rows = [
        json.loads(line)
        for line in artifact.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    component_path = ROOT / "results" / "weight5_component_certifications.jsonl"
    descriptor = {
        "path": component_path.relative_to(ROOT).as_posix(),
        "sha256": _sha256(component_path),
    }
    catalogue_inputs = rows[0]["source_artifacts"]["catalogue_inputs"]
    if descriptor["path"] not in {item["path"] for item in catalogue_inputs}:
        catalogue_inputs.append(descriptor)
        catalogue_inputs.sort(key=lambda item: item["path"])
    return rows


@pytest.fixture(scope="module")
def pre_upper_bound_witness_specs():
    from scripts import generate_weight5_supplement as supplement

    campaign_data = supplement.build_catalogue(
        include_css_upper_bound_witnesses=False
    )
    return supplement, {
        spec.spec_id: spec
        for _, specs, _, _ in campaign_data
        for spec in specs
    }


def test_publication_catalogue_accounting_and_distance_semantics(records, manifest):
    assert len(records) == 1_142
    assert len({row["presentation_id"] for row in records}) == 1_142
    assert {row["schema_version"] for row in records} == {2}
    assert {row["record_type"] for row in records} == {"weight5_presentation"}
    assert Counter(row["family"] for row in records) == {"CSS": 912, "PBB": 230}

    direct = Counter(row["distance_direct"]["status_code"] for row in records)
    assert direct == {"E": 398, "C": 744}
    assert all(row["distance_direct"]["upper"] >= 3 for row in records)

    for row in records:
        distance = row["distance_direct"]
        endpoint_fom = row["parameters"]["k"] * distance["upper"] ** 2 / row[
            "parameters"
        ]["n"]
        if distance["upper_is_supported"]:
            assert distance["estimated_upper"] is None
            assert distance["fom_upper"] == endpoint_fom
            assert distance["fom_estimate"] is None
        else:
            assert distance["estimated_upper"] == distance["upper"]
            assert distance["fom_upper"] is None
            assert distance["fom_estimate"] == endpoint_fom
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


def test_estimated_endpoints_are_not_serialized_as_rigorous_fom_bounds():
    from scripts import build_weight5_publication_catalogue as catalogue

    estimated = catalogue._distance_payload(
        n=100,
        k=4,
        lower=5,
        upper=9,
        exact=False,
        lower_evidence=[(5, "L4")],
        upper_evidence=[(9, "B")],
    )
    assert estimated["upper"] == estimated["estimated_upper"] == 9
    assert estimated["fom_upper"] is None
    assert estimated["fom_estimate"] == 3.24

    supported = catalogue._distance_payload(
        n=100,
        k=4,
        lower=5,
        upper=9,
        exact=False,
        lower_evidence=[(5, "L4")],
        upper_evidence=[(9, "I"), (10, "B")],
    )
    assert supported["upper"] == 9
    assert supported["estimated_upper"] is None
    assert supported["fom_upper"] == 3.24
    assert supported["fom_estimate"] is None


def test_class_distance_change_includes_evidence_upgrades():
    from scripts import build_weight5_publication_catalogue as catalogue

    direct = {
        "lower": 5,
        "upper": 9,
        "is_exact": False,
        "upper_is_supported": False,
    }
    unchanged = dict(direct)
    upgraded = {**direct, "upper_is_supported": True}

    assert not catalogue._class_distance_differs(direct, unchanged)
    assert catalogue._class_distance_differs(direct, upgraded)


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


def test_css_upper_bound_witnesses_are_materialized(records, manifest):
    expected = {
        "CL-1a333605": ("B", 22, 14),
        "CL-23177205": ("B", 22, 14),
        "CL-40a1f698": ("B", 23, 12),
        "CL-4a95cc39": ("B", 14, 13),
        "CL-54d0129e": ("B", 18, 13),
        "CL-85245770": ("B", 17, 14),
        "CL-d192c3d1": ("B", 14, 13),
        "CL-daf9d6b0": ("B", 22, 16),
        "CL-f5754da6": ("I", 24, 23),
        "CS-22b9dba0": ("B", 14, 9),
    }
    by_id = {row["presentation_id"]: row for row in records}
    correction_manifest = manifest["css_upper_bound_witness_correction"]
    assert set(correction_manifest["retained_presentation_ids"]) == set(expected)
    assert correction_manifest["excluded_presentation_ids"] == []
    assert manifest["counts"]["css_upper_bound_witness_corrections"] == 10

    for presentation_id, (prior_method, prior_value, new_value) in expected.items():
        row = by_id[presentation_id]
        correction = row["css_upper_bound_witness_correction"]
        assert correction["historical_upper_bound"]["method"] == prior_method
        assert correction["historical_upper_bound"]["value"] == prior_value
        assert correction["corrected_upper_bound"] == {
            "method": "W",
            "value": new_value,
        }
        assert set(correction["sector_weights"].values()) == {new_value}
        assert {"bound": new_value, "method": "W"} in row["distance_direct"][
            "evidence"
        ]["upper_bounds"]
        assert "css_upper_bound_witnesses" in row["source_artifacts"]

    f575 = by_id["CL-f5754da6"]
    assert {
        (item["bound"], item["method"])
        for item in f575["distance_direct"]["evidence"]["upper_bounds"]
    } == {(23, "W"), (24, "I"), (29, "B")}
    assert (f575["distance_class"]["lower"], f575["distance_class"]["upper"]) == (
        5,
        23,
    )

    leading_by_class = {}
    for row in records:
        if (
            row["family"] == "CSS"
            and row["connectivity"]["is_connected"]
            and row["distance_class"]["status_code"] == "C"
            and row["distance_class"]["upper_is_supported"]
        ):
            leading_by_class.setdefault(row["equivalence"]["class_id"], row)
    ranked = sorted(
        leading_by_class.values(),
        key=lambda row: -row["distance_class"]["fom_upper"],
    )
    assert ranked[0]["presentation_id"] == "CL-f5754da6"
    assert math.isclose(ranked[0]["distance_class"]["fom_upper"], 4 * 23**2 / 432)
    assert {
        row["presentation_id"] for row in ranked[1:3]
    } == {"CL-9fbce668", "CL-c271b632"}
    assert all(
        math.isclose(row["distance_class"]["fom_upper"], 4 * 22**2 / 420)
        for row in ranked[1:3]
    )


def test_css_upper_bound_witness_manifest_identity_is_strict(tmp_path):
    from scripts import generate_weight5_supplement as supplement

    mutations = {
        "missing schema version": ("schema_version", None),
        "wrong generator": ("generator", "scripts/not_the_generator.py"),
    }
    for label, (field, value) in mutations.items():
        rows = _witness_artifact_rows_with_current_sources()
        if value is None:
            rows[0].pop(field)
        else:
            rows[0][field] = value
        artifact = tmp_path / f"{label.replace(' ', '_')}.jsonl"
        artifact.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="malformed.*manifest"):
            supplement.load_and_merge_css_upper_bound_witnesses({}, artifact)


def test_css_upper_bound_witness_manifest_requires_every_source(tmp_path):
    from scripts import generate_weight5_supplement as supplement

    rows = _witness_artifact_rows_with_current_sources()
    rows[0]["source_artifacts"]["catalogue_inputs"] = [
        descriptor
        for descriptor in rows[0]["source_artifacts"]["catalogue_inputs"]
        if descriptor["path"]
        != "results/weight5_component_certifications.jsonl"
    ]
    artifact = tmp_path / "missing_source.jsonl"
    artifact.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source coverage mismatch"):
        supplement.load_and_merge_css_upper_bound_witnesses({}, artifact)


def test_css_upper_bound_witness_supports_are_recomputed(
    tmp_path, pre_upper_bound_witness_specs
):
    supplement, specs_by_id = pre_upper_bound_witness_specs
    rows = _witness_artifact_rows_with_current_sources()
    row = next(
        item for item in rows if item.get("presentation_id") == "CL-1a333605"
    )
    spec = specs_by_id[row["presentation_id"]]
    code = supplement.build_bb_code(
        spec.ell, spec.m, list(spec.A), list(spec.B)
    )
    rows_z = supplement._css_matrix_rows_as_int(code.matrix_z, spec.n)
    rows_x = supplement._css_matrix_rows_as_int(code.matrix_x, spec.n)
    basis_x, pivots_x = supplement._css_rowspace_basis(rows_x, spec.n)

    sector = row["replacement_explicit_witnesses"]["X"]
    weight = sector["weight"]
    invalid_indices = None
    for start in range(spec.n - weight + 1):
        candidate = list(range(start, start + weight))
        support = sum(1 << index for index in candidate)
        if not supplement._css_support_is_logical(
            support, rows_z, basis_x, pivots_x
        ):
            invalid_indices = candidate
            break
    assert invalid_indices is not None
    sector["qubit_indices_zero_based"] = invalid_indices
    sector["qubits"] = supplement._css_qubit_labels(
        invalid_indices, spec.ell, spec.m
    )

    artifact = tmp_path / "fabricated_support.jsonl"
    artifact.write_text(
        "".join(json.dumps(item) + "\n" for item in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid X logical witness"):
        supplement.load_and_merge_css_upper_bound_witnesses(
            specs_by_id, artifact
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
    assert len(manifest["source_artifacts"]) == 16
    assert set(manifest["software_artifacts"]) == {
        "bb_constructor",
        "component_certificate_generator",
        "css_upper_bound_witness_generator",
        "css_low_weight_audit_generator",
        "css_weight5_witness_generator",
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
