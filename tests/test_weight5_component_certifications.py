"""Regression tests for the weight-five component-distance artifact."""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts/certify_weight5_components.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "certify_weight5_components", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve postponed annotations through sys.modules.
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_component_certificates_and_explicit_bb_matches_are_reproducible():
    module = _load_script()
    rows = module.build_artifact_rows()

    assert rows[0]["record_type"] == "artifact_manifest"
    certificates = {
        row["source_identity"]["presentation_id"]: row
        for row in rows
        if row["record_type"] == "component_distance_certificate"
    }
    expected = {
        "CL-5cd6065f": (18, 24, 4, 2),
        "CL-4b943aa2": (10, 42, 6, 3),
        "CL-929c2a70": (16, 24, 4, 3),
        "CL-b9275593": (12, 36, 4, 3),
        "CL-437b6e5f": (10, 42, 4, 5),
        "CL-d9f65a7e": (8, 48, 4, 3),
        "CL-a25b2baf": (8, 48, 4, 6),
    }
    assert set(certificates) == set(expected)
    for presentation_id, (count, n, k, distance) in expected.items():
        row = certificates[presentation_id]
        assert row["decomposition"]["n_components"] == count
        assert row["decomposition"]["component_sizes"] == [n] * count
        assert row["decomposition"]["component_dimensions"] == [k] * count
        assert row["decomposition"]["components_bliss_isomorphic"] is True
        assert row["decomposition"][
            "rowspace_partition_equals_presentation_partition"
        ] is True
        assert row["decomposition"][
            "restricted_presentation_spans_induced_rowspace"
        ] is True
        assert row["component_certificate"]["d_exact"] == distance
        assert row["parent_spec"]["d_exact"] == distance
        assert row["distance_transfer"]["valid"] is True

    matches = [
        row for row in rows if row["record_type"] == "explicit_bb_component_match"
    ]
    assert [row["explicit_bb_component"]["n"] for row in matches] == [90, 96, 140]
    assert [row["explicit_bb_component"]["k"] for row in matches] == [4, 4, 6]
    assert [row["explicit_bb_component"]["d_exact"] for row in matches] == [9, 10, 10]
    assert all(row["explicit_bb_component"]["connected"] is True for row in matches)
    assert all(row["explicit_bb_component"]["stabilizer_weight"] == 5 for row in matches)
    assert all(row["match"]["bliss_isomorphic"] is True for row in matches)

    checked_in = module.DEFAULT_OUTPUT.read_text(encoding="utf-8")
    assert checked_in == module.render_artifact(rows)
