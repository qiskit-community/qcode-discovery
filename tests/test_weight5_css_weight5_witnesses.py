"""Checks for the explicit weight-five CSS witness artifact."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_weight5_css_weight5_witnesses.py"
ARTIFACT = ROOT / "results" / "weight5_css_weight5_witnesses.jsonl"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "audit_weight5_css_weight5_witnesses", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _artifact_rows() -> list[dict]:
    return [
        json.loads(line)
        for line in ARTIFACT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="module")
def pre_witness_specs():
    from scripts import generate_weight5_supplement as supplement

    campaign_data = supplement.build_catalogue(
        include_css_low_weight_audit=False,
        include_css_weight5_witnesses=False,
        include_css_upper_bound_witnesses=False,
    )
    specs_by_id = {
        spec.spec_id: spec
        for _, specs, _, _ in campaign_data
        for spec in specs
    }
    supplement.load_and_merge_css_low_weight_audit(specs_by_id)
    return supplement, specs_by_id


def test_checked_in_weight_five_witnesses_validate_directly():
    module = _load_script()
    records = _artifact_rows()
    manifest, audits = records[0], records[1:]
    assert manifest["artifact_schema"] == module.ARTIFACT_SCHEMA
    assert manifest["counts"] == {
        "all_x_witnesses_valid": True,
        "all_z_witnesses_valid": True,
        "explicit_witnesses": 90,
        "presentation_classes": 32,
        "presentations": 45,
    }

    for row in audits:
        params = row["parameters"]
        generators = row["generators"]
        rows_x, rows_z = module.build_bb_check_rows(
            params["ell"],
            params["m"],
            generators["A_terms"],
            generators["B_terms"],
        )
        for name, check_rows, stabilizer_rows in (
            ("X", rows_z, rows_x),
            ("Z", rows_x, rows_z),
        ):
            witness = row["replacement_explicit_witnesses"][name]
            indices = witness["qubit_indices_zero_based"]
            assert len(indices) == len(set(indices)) == 5
            support = sum(1 << index for index in indices)
            check_columns = module._columns_from_rows(
                check_rows, params["n"]
            )
            remainder_columns, _ = module._rowspace_remainder_columns(
                stabilizer_rows, params["n"]
            )
            module._verified_witness(
                support, check_columns, remainder_columns
            )


def test_weight_five_witness_artifact_has_no_catalogue_dependency():
    manifest = _artifact_rows()[0]
    assert set(manifest["source_artifacts"]) == {
        "catalogue_inputs",
        "low_weight_audit",
    }
    assert all(
        item["path"] != "results/weight5_publication_catalogue.jsonl"
        for item in manifest["source_artifacts"]["catalogue_inputs"]
    )
    assert "weight5_publication_catalogue.jsonl" not in SCRIPT.read_text(
        encoding="utf-8"
    )


def test_weight_five_witness_manifest_identity_is_strict(
    tmp_path, pre_witness_specs
):
    supplement, specs_by_id = pre_witness_specs
    rows = _artifact_rows()
    rows[0]["generator"] = "scripts/not_the_generator.py"
    artifact = tmp_path / "bad_manifest.jsonl"
    artifact.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="malformed CSS weight-five witness manifest"):
        supplement.load_and_merge_css_weight5_witnesses(specs_by_id, artifact)


def test_weight_five_witness_support_tampering_is_rejected(
    tmp_path, pre_witness_specs
):
    supplement, specs_by_id = pre_witness_specs
    rows = _artifact_rows()
    audit = next(
        row
        for row in rows
        if row.get("record_type") == "css_weight5_witness_audit"
    )
    sector = audit["replacement_explicit_witnesses"]["X"]
    sector["qubit_indices_zero_based"][1] = sector[
        "qubit_indices_zero_based"
    ][0]
    artifact = tmp_path / "tampered_support.jsonl"
    artifact.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="malformed X witness support"):
        supplement.load_and_merge_css_weight5_witnesses(specs_by_id, artifact)
