"""Checks for the explicit weight-five CSS witness artifact."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


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


def test_checked_in_weight_five_witnesses_validate_directly():
    module = _load_script()
    records = [
        json.loads(line)
        for line in ARTIFACT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
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
