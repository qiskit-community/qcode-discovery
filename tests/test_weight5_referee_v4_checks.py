"""Regression tests for the focused v4 referee evidence artifact."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_weight5_referee_v4.py"
ARTIFACT = ROOT / "results" / "weight5_referee_v4_checks.json"
CHALLENGE_ROOT = Path("/tmp/qldpc-challenge-src")


def _load_script():
    spec = importlib.util.spec_from_file_location("audit_weight5_referee_v4", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_sector_agreement_and_milp_retention_counts():
    artifact = _artifact()
    sectors = artifact["css_sector_agreement"]
    assert sectors["low_weight_audit"] == {
        **sectors["low_weight_audit"],
        "classes_checked": 492,
        "x_z_agreements": 492,
        "outcomes": {
            "both_certified_at_least_5": 428,
            "exact_2": 2,
            "exact_3": 25,
            "exact_4": 37,
        },
    }
    assert sectors["exact_css_milp"]["exact_records"] == 50
    assert sectors["exact_css_milp"]["x_z_agreements"] == 50
    assert sectors["replacement_weight_five_witnesses"] == {
        **sectors["replacement_weight_five_witnesses"],
        "presentations": 45,
        "presentation_classes": 32,
        "explicit_logicals": 90,
        "all_directly_validated": True,
    }

    milp = artifact["milp_provenance_and_retention"]
    assert milp["catalogue_direct_M_exact"]["total"] == 106
    assert milp["catalogue_direct_M_exact"]["by_family"] == {
        "CSS": 50,
        "PBB": 56,
    }
    assert milp["catalogue_direct_M_exact"]["by_campaign"] == {
        "css-large": 38,
        "css-small": 12,
        "pbb-large": 27,
        "pbb-small": 29,
    }
    assert milp["catalogue_direct_M_exact"]["pbb_source_reconciliation"][
        "pbb-large"
    ] == {
        "catalogue_direct_M_exact_presentations": 27,
        "in_search_exact_unique_tuples": 2,
        "overlap": 1,
        "post_campaign_all_logical_exact_records": 26,
    }
    assert milp["retention"]["native_highs_logs_retained"] is False
    assert milp["retention"]["formal_optimality_certificates_retained"] is False
    assert milp["implementation"]["release_environment"] == {
        "backend_used": (
            "the HiGHS core bundled inside SciPy and invoked by "
            "scipy.optimize.milp"
        ),
        "scipy": "1.17.0",
        "scipy_bundled_highs_core": "1.8.0",
        "standalone_highspy": "1.13.1",
        "standalone_highspy_used_by_this_path": False,
    }


def test_absent_pbb_lattices_are_stage2_only_and_evidentially_bounded():
    coverage = _artifact()["pbb_lattice_coverage"]
    assert coverage["zero_positive_observation_lattices"] == [[10, 9], [16, 12]]
    zero_rows = [
        row
        for profile in coverage["profiles"].values()
        for row in profile["lattices"]
        if row["positive_distance_observations"] == 0
    ]
    assert len(zero_rows) == 2
    assert all(row["stage_role"] == "stage_2_addition" for row in zero_rows)
    assert all(row["retained_catalogue_presentations"] == 0 for row in zero_rows)
    assert "do not distinguish" in coverage["finding"]


def test_embedded_edge_colourings_are_complete_and_optimal():
    module = _load_script()
    schedules = _artifact()["ideal_measurement_schedules"]
    by_slug = {row["slug"]: row for row in schedules["codes"]}
    expected_degrees = {
        "96-4-10": 5,
        "96-4-12": 6,
        "140-6-10": 5,
        "140-6-14": 6,
    }
    for slug, degree in expected_degrees.items():
        row = by_slug[slug]
        validation = module.validate_schedule(
            row["source_rows"], row["layers"], row["parameters"]["n"]
        )
        assert validation == row["validation"]
        assert validation["regular_degree"] == degree
        assert validation["schedule_layers"] == degree
        assert validation["edge_chromatic_number"] == degree

    for slug, expected in module.AUTHORS_CODES.items():
        source_rows, source_layers = module.bb_schedule_rows_and_layers(
            expected["ell"],
            expected["m"],
            expected["A_terms"],
            expected["B_terms"],
        )
        assert source_rows == by_slug[slug]["source_rows"]
        assert source_layers == by_slug[slug]["layers"]

    assert schedules["length_matched_comparisons"] == [
        {
            "this_work": "96-4-10",
            "comparator": "96-4-12",
            "this_work_layers": 5,
            "comparator_layers": 6,
            "ideal_entangling_layer_reduction": 1,
        },
        {
            "this_work": "140-6-10",
            "comparator": "140-6-14",
            "this_work_layers": 5,
            "comparator_layers": 6,
            "ideal_entangling_layer_reduction": 1,
        },
    ]


def test_pinned_challenge_inputs_when_checkout_is_available():
    if not (CHALLENGE_ROOT / ".git").exists():
        pytest.skip("pinned QLDPC Challenge checkout is not present")
    artifact = _artifact()
    challenge_records = {
        row["slug"]: row
        for row in artifact["ideal_measurement_schedules"]["codes"]
        if row["source_kind"] == "qldpc_challenge_length_matched_comparator"
    }
    for slug, row in challenge_records.items():
        path = CHALLENGE_ROOT / row["source"]["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["source"]["sha256"]
        source = json.loads(path.read_text(encoding="utf-8"))
        source_rows = source["checks"]["X"] + source["checks"]["Z"]
        assert source_rows == row["source_rows"]


def test_checked_in_artifact_matches_fresh_audit_when_checkout_is_available():
    if not (CHALLENGE_ROOT / ".git").exists():
        pytest.skip("pinned QLDPC Challenge checkout is not present")
    module = _load_script()
    assert module._render(module.build_artifact(CHALLENGE_ROOT)) == ARTIFACT.read_text(
        encoding="utf-8"
    )
