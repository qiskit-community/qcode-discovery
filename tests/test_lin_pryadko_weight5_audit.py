"""Regression tests for the Lin--Pryadko weight-five prior-art audit."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "audit_lin_pryadko_weight5.py"
ARTIFACT = ROOT / "results" / "weight5_lin_pryadko_audit.json"


def _load_script():
    spec = importlib.util.spec_from_file_location("audit_lin_pryadko_weight5", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_bootstraps_repo_imports_outside_repo(tmp_path):
    probe = """
import importlib.util
import sys
from pathlib import Path

script = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("audit_lin_pryadko_weight5", script)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert str(script.parent.parent) == sys.path[0]
from evaluation.distance_milp import get_code_matrices, ilp_min_weight
"""
    subprocess.run(
        [sys.executable, "-I", "-c", probe, str(SCRIPT)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


def test_checked_in_audit_is_reproducible_and_structurally_complete():
    module = _load_script()
    rebuilt = module.build_audit()

    assert ARTIFACT.read_text(encoding="utf-8") == module.render_audit(rebuilt)
    assert rebuilt["archive"]["commit"] == (
        "403d194c3f98f0cadc236aecbc4a8b6139ccf23c"
    )
    assert "upper bound" in rebuilt["archive"]["distance_method"]["interpretation"]

    rows = {row["label"]: row for row in rebuilt["rows"]}
    expected = {
        "LP-48-4-7": (48, 4, 7, (0, 9), (0, 16, 11)),
        "LP-60-4-8": (60, 4, 8, (0, 21), (0, 10, 26)),
        "LP-66-4-8": (66, 4, 8, (0, 12), (0, 22, 26)),
        "LP-78-4-9": (78, 4, 9, (0, 27), (0, 13, 5)),
        "LP-84-4-9": (84, 4, 9, (0, 15), (0, 1, 38)),
        "LP-96-4-10": (96, 4, 10, (0, 3), (0, 16, 2)),
    }
    assert set(rows) == set(expected)

    for label, (n, k, distance, support_a, support_b) in expected.items():
        row = rows[label]
        assert row["cyclic_decode"]["a_support_exponents"] == list(support_a)
        assert row["cyclic_decode"]["b_support_exponents"] == list(support_b)
        assert row["reconstructed_code"]["n"] == n
        assert row["reconstructed_code"]["k"] == k
        assert row["reconstructed_code"]["stabilizer_weight"] == 5
        assert row["reconstructed_code"]["connected"] is True
        assert row["reconstructed_code"]["translation_gcd"] == 1
        assert row["exact_milp"]["d"] == distance
        assert row["exact_milp"]["d_x"] == distance
        assert row["exact_milp"]["d_z"] == distance
        assert row["exact_milp"]["exact"] is True
        assert row["exact_milp"]["all_objectives_optimal"] is True
        assert row["exact_milp"]["logical_objectives_optimal"] == 2 * k
        assert row["exact_milp"]["logical_objectives_total"] == 2 * k
        assert len(row["exact_milp"]["x_objectives"]) == k
        assert len(row["exact_milp"]["z_objectives"]) == k


def test_campaign_96_crt_map_is_a_bijection_and_matches_archive_after_swap():
    module = _load_script()
    audit = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    equivalence = audit["campaign_96_equivalence"]

    images = {
        module.branch_to_cyclic_exponent(x_exponent, y_exponent)
        for x_exponent in range(16)
        for y_exponent in range(3)
    }
    assert images == set(range(48))
    assert equivalence["map"]["mapped_branch_a"] == [0, 16, 2]
    assert equivalence["map"]["mapped_branch_b"] == [0, 3]
    assert equivalence["archive_cyclic_supports"]["a"] == [0, 3]
    assert equivalence["archive_cyclic_supports"]["b"] == [0, 16, 2]
    assert equivalence["matrix_conjugation_verified"] is True
    assert equivalence["result"] == "exact_presentation_equivalence"


def test_nonabelian_132_is_exact_quasi_abelian_over_c11():
    module = _load_script()
    audit = module.build_audit()
    assert audit["archive"]["nonabelian_zip_sha256"] == (
        "5962039bbfa3729a80f5001ff44dbefd57dac689d605fbaae7f247ff34fb5258"
    )

    rows = {row["label"]: row for row in audit["nonabelian_frontier_rows"]}
    assert set(rows) == {
        "LP-132-4-12",
        "LP-168-4-14",
        "LP-180-4-15",
        "LP-192-4-16",
    }
    row = rows["LP-132-4-12"]

    assert row["archive"]["row"] == "66 1 4 12 [5] [14,39]"
    assert row["group_decode"]["structure"] == "C11 x S3"
    assert row["group_decode"]["index_to_coordinate"] == {
        "5": [1, 0, 1],
        "14": [2, 1, 0],
        "39": [5, 2, 0],
    }
    assert row["group_decode"]["relation"] == "g39 = g14^8"
    assert row["group_decode"]["support_generator_overlap_relation"] == (
        "g5^2 = g14^12 = c^2"
    )

    support = row["support_subgroups"]
    assert support["G_a"] == {"generator": "g5", "structure": "C22", "order": 22}
    assert support["G_b"] == {
        "generator": "g14",
        "structure": "C33",
        "order": 33,
    }
    assert support["intersection"]["structure"] == "C11"
    assert support["intersection"]["abelian"] is True
    assert support["intersection"]["central_in_ambient_group"] is True
    assert support["intersection"]["normal_in_G_a"] is True
    assert support["intersection"]["normal_in_G_b"] is True
    assert support["double_coset_order"] == 66
    assert support["double_coset_covers_ambient_group"] is True
    assert support["quotient_by_intersection"] == {
        "ambient_group": "S3",
        "G_a_over_N": "C2=<s>",
        "G_b_over_N": "C3=<r>",
        "a_support": "1+s",
        "b_support": "1+r+r^2",
    }

    code = row["reconstructed_code"]
    assert (code["n"], code["k"], code["stabilizer_weight"]) == (132, 4, 5)
    assert code["connected"] is True
    assert code["ranks"] == {"A": 63, "B": 62, "AB": 61, "H_X": 64, "H_Z": 64}
    assert code["block_erasure_subsystem_k"] == 2
    assert code["rank_defects"] == {"delta_X": 0, "delta_Z": 0}

    family = row["family_classification"]
    assert family["ambient_group_nonabelian"] is True
    assert family["lin_pryadko_quasi_abelian_lifted_product"] is True
    assert family["equivalent_hypergraph_product_base_ring"] == "F2[C11]"
    assert family["essentially_nonabelian_in_rank_defect_sense"] is False
    assert family["equivalence_to_abelian_2BGA_established"] is False

    milp = row["exact_milp"]
    assert (milp["d"], milp["d_x"], milp["d_z"]) == (12, 12, 12)
    assert milp["exact"] is True
    assert milp["all_objectives_optimal"] is True
    assert len(milp["x_objectives"]) == 4
    assert len(milp["z_objectives"]) == 4


def test_larger_nonabelian_attempts_are_not_misreported_as_exact():
    module = _load_script()
    audit = module.build_audit()
    rows = {row["label"]: row for row in audit["nonabelian_frontier_rows"]}
    expected = {
        "LP-168-4-14": {
            "archive_row": "84 3 4 14 [6] [11,60]",
            "n": 168,
            "upper_bound": 14,
            "ranks": {"A": 81, "B": 74, "AB": 71, "H_X": 82, "H_Z": 82},
            "rank_defects": {"delta_X": 2, "delta_Z": 2},
            "optimal": 1,
            "timeouts": 7,
        },
        "LP-180-4-15": {
            "archive_row": "90 6 4 15 [10] [6,38]",
            "n": 180,
            "upper_bound": 15,
            "ranks": {"A": 84, "B": 78, "AB": 74, "H_X": 88, "H_Z": 88},
            "rank_defects": {"delta_X": 0, "delta_Z": 0},
            "optimal": 0,
            "timeouts": 8,
        },
        "LP-192-4-16": {
            "archive_row": "96 4 4 16 [3] [8,46]",
            "n": 192,
            "upper_bound": 16,
            "ranks": {"A": 90, "B": 92, "AB": 86, "H_X": 94, "H_Z": 94},
            "rank_defects": {"delta_X": 2, "delta_Z": 2},
            "optimal": 1,
            "timeouts": 7,
        },
    }

    for label, values in expected.items():
        row = rows[label]
        code = row["reconstructed_code"]
        attempt = row["milp_attempt"]
        assert row["archive"]["row"] == values["archive_row"]
        assert (code["n"], code["k"], code["stabilizer_weight"]) == (
            values["n"],
            4,
            5,
        )
        assert code["connected"] is True
        assert code["ranks"] == values["ranks"]
        assert code["rank_defects"] == values["rank_defects"]
        assert attempt["d"] is None
        assert attempt["d_upper_bound"] == values["upper_bound"]
        assert attempt["exact"] is False
        assert attempt["all_objectives_optimal"] is False
        assert attempt["logical_objectives_optimal"] == values["optimal"]
        assert attempt["timed_out_with_incumbent"] == values["timeouts"]
        assert len(attempt["x_objectives"]) == 4
        assert len(attempt["z_objectives"]) == 4
        assert all(
            objective["weight"] == values["upper_bound"]
            for objective in attempt["x_objectives"] + attempt["z_objectives"]
        )

    for label in ("LP-168-4-14", "LP-192-4-16"):
        family = rows[label]["family_classification"]
        assert family["nonzero_rank_defects_verified"] is True
        assert family["essentially_nonabelian_in_rank_defect_sense"] is True
        assert family["lin_pryadko_statement_as_written_counterexample"] is True
        assert rows[label]["support_subgroups"]["extension_splitting"][
            "G_a_over_N_splits"
        ] is False

    family_180 = rows["LP-180-4-15"]["family_classification"]
    assert family_180["lin_pryadko_quasi_abelian_lifted_product"] is True
    assert family_180["equivalent_hypergraph_product_base_ring"] == "F2[C15]"
    assert family_180["essentially_nonabelian_in_rank_defect_sense"] is False

    assert audit["summary"]["nonabelian_rows_exactly_certified"] == [
        "LP-132-4-12"
    ]
    assert audit["summary"]["nonabelian_rows_with_incomplete_milp_attempts"] == [
        "LP-168-4-14",
        "LP-180-4-15",
        "LP-192-4-16",
    ]
