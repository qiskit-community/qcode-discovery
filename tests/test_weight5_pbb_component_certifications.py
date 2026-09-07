"""Regression tests for explicit connected weight-five PBB components."""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts/certify_weight5_pbb_components.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "certify_weight5_pbb_components", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve postponed annotations through sys.modules.
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_explicit_pbb_component_matches_are_reproducible():
    module = _load_script()
    rows = module.build_artifact_rows()

    manifest = rows[0]
    assert manifest["record_type"] == "artifact_manifest"
    assert manifest["record_counts"] == {"explicit_pbb_component_match": 4}

    matches = {
        row["parent_source_identity"]["presentation_id"]: row
        for row in rows[1:]
    }
    expected = {
        "PL-8ee754a5": (180, 2, 11, 2),
        "PL-ae4cc6ad": (216, 4, 10, 2),
        "PL-36c5ce3b": (216, 4, 10, 2),
        "PL-aefa6773": (144, 4, 8, 3),
    }
    assert set(matches) == set(expected)

    for parent_id, (n, k, distance, count) in expected.items():
        row = matches[parent_id]
        component = row["explicit_pbb_component"]
        decomposition = row["decomposition"]
        exact_source = row["parent_exact_distance_source"]
        lc_screen = row["local_clifford_css_screen"]
        match = row["match"]

        assert row["record_type"] == "explicit_pbb_component_match"
        assert (component["n"], component["k"], component["d_exact"]) == (
            n,
            k,
            distance,
        )
        assert component["connected"] is True
        assert component["stabilizer_weight"] == 5
        assert decomposition["n_components"] == count
        assert decomposition["component_sizes"] == [n] * count
        assert decomposition["component_dimensions"] == [k] * count
        assert decomposition["components_bliss_isomorphic"] is True
        assert decomposition[
            "rowspace_partition_equals_presentation_partition"
        ] is True
        assert decomposition[
            "restricted_presentation_spans_induced_rowspace"
        ] is True
        assert exact_source["milp_d"] == distance
        assert exact_source["logicals_optimal"] == exact_source["total_logicals"]
        assert lc_screen["passes_implemented_checks"] is True
        assert lc_screen["css_equivalent_under_tested_classes"] is False
        assert lc_screen["uniform_blockwise"] == {
            "assignments_tested": 36,
            "group_css_assignments": [],
        }
        assert lc_screen["nonuniform_IS_or_HHS"] == {
            "affine_system_consistent": False,
            "constraint_rank": n,
            "num_variables": n,
        }
        assert lc_screen["nonuniform_hadamard_on_stored_generators"][
            "is_css"
        ] is False
        assert lc_screen["scope"]["complete_local_clifford_orbit_test"] is False
        assert "{SH,HSH}" in lc_screen["scope"]["not_covered"]
        assert match["bliss_isomorphic"] is True
        assert match["distance_transfer_valid"] is True

    first_216 = matches["PL-ae4cc6ad"]["match"][
        "canonical_digest_sha256"
    ]
    second_216 = matches["PL-36c5ce3b"]["match"][
        "canonical_digest_sha256"
    ]
    assert first_216 != second_216

    checked_in = module.DEFAULT_OUTPUT.read_text(encoding="utf-8")
    assert checked_in == module.render_artifact(rows)
