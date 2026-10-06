"""Regression checks for the pinned QEC Challenge connectivity audit."""

from __future__ import annotations

from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "results" / "weight5_challenge_connectivity_audit.jsonl"
_PARETO_SCRIPT = ROOT / "paper" / "2610.06623" / "figures" / "plot_weight5_pareto_comparison.py"

_spec = importlib.util.spec_from_file_location(
    "plot_weight5_pareto_comparison", _PARETO_SCRIPT
)
assert _spec is not None and _spec.loader is not None
_pareto = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _pareto
_spec.loader.exec_module(_pareto)

aggregate_records = _pareto.aggregate_records
audited_challenge_fields = _pareto.audited_challenge_fields
base_record = _pareto.base_record
load_challenge_connectivity_audit = _pareto.load_challenge_connectivity_audit
mark_figure_selection = _pareto.mark_figure_selection


def _artifact_rows() -> tuple[dict, list[dict]]:
    rows = [
        json.loads(line)
        for line in ARTIFACT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifests = [row for row in rows if row["record_type"] == "artifact_manifest"]
    records = [
        row
        for row in rows
        if row["record_type"] == "qldpc_challenge_connectivity"
    ]
    assert len(manifests) == 1
    return manifests[0], records


def test_challenge_audit_is_complete_and_validated() -> None:
    manifest, records = _artifact_rows()
    assert manifest["upstream"]["commit"] == (
        "3b50503b058a0c09500c1017b54c3d547e1101cc"
    )
    assert manifest["counts"] == {
        "connected": 81,
        "disconnected": 21,
        "disconnected_by_weight": {"4": 4, "6": 14, "8": 3},
        "records": 102,
        "reduction_methods": {
            "connected": 81,
            "homogeneous_isomorphic": 12,
            "unique_positive_k": 9,
        },
    }
    assert len(records) == len({row["slug"] for row in records}) == 102
    assert Counter(row["snapshot"] for row in records) == {
        "actual_weight_five": 5,
        "bb": 97,
    }
    assert all(row["matrix_validation"]["commutes"] for row in records)
    assert all(
        row["matrix_validation"]["computed_k"] == row["parameters"]["k"]
        for row in records
    )
    assert all(row["connectivity"]["k_additivity_verified"] for row in records)
    assert all(row["reduction"]["distance_transfer_validated"] for row in records)
    assert all(
        witness["validated"]
        for row in records
        for witness in row["witnesses"].values()
    )


def test_weight_four_challenge_reductions() -> None:
    _, records = _artifact_rows()
    by_slug = {row["slug"]: row for row in records}
    expected = {
        "96-12-4-mvb": (6, 16, 2),
        "192-24-4": (12, 16, 2),
    }
    for slug, (multiplicity, n, k) in expected.items():
        row = by_slug[slug]
        assert row["connectivity"]["n_components"] == multiplicity
        assert row["connectivity"]["components_bliss_isomorphic"] is True
        assert row["reduction"]["method"] == "homogeneous_isomorphic"
        assert (row["reduction"]["n"], row["reduction"]["k"]) == (n, k)
    assert (
        by_slug["96-12-4-mvb"]["connectivity"][
            "component_canonical_digest_sha256"
        ]
        == by_slug["192-24-4"]["connectivity"][
            "component_canonical_digest_sha256"
        ]
    )


def test_unique_logical_component_reductions_are_conservative() -> None:
    _, records = _artifact_rows()
    unique = [
        row for row in records if row["reduction"]["method"] == "unique_positive_k"
    ]
    assert len(unique) == 9
    for row in unique:
        positive = row["connectivity"]["positive_k_component_indices"]
        assert positive == [row["reduction"]["component_index"]]
        assert all(
            witness["component_index"] == positive[0]
            for witness in row["witnesses"].values()
        )
        assert sum(
            component["k"] for component in row["connectivity"]["components"]
        ) == row["parameters"]["k"]


def test_comparison_uses_audited_challenge_components() -> None:
    # This also validates the artifact against both compact source snapshots.
    audit = load_challenge_connectivity_audit()
    assert len(audit) == 102
    snapshot = json.loads(
        (
            ROOT
            / "paper"
            / "2610.06623"
            / "figures"
            / "qldpc_challenge_bb_snapshot.json"
        ).read_text(encoding="utf-8")
    )
    records = []
    for row in snapshot["records"]:
        if row["max_check_weight"] != 4:
            continue
        reduced = audited_challenge_fields(row, audit)
        records.append(
            base_record(
                cohort="qec_challenge_w4",
                generation="challenge",
                structure="BB",
                weight=4,
                n=reduced["n"],
                k=reduced["k"],
                d=row["d"],
                status=row["distance_status"],
                evidence="test",
                source="test",
                connectivity_status=reduced["connectivity_status"],
                decomposition=reduced["decomposition"],
                presentation_agrees=reduced["presentation_agrees"],
                construction_role=reduced["construction_role"],
                parent_presentation_id=reduced["parent_presentation_id"],
                component_digest=reduced["component_digest"],
            )
        )
    assert all(row["connectivity_status"] != "not_audited" for row in records)

    aggregated = aggregate_records(records)
    mark_figure_selection(aggregated)
    rate_front = {
        (row["n"], row["k"], row["d"])
        for row in aggregated
        if row["max_check_weight"] == 4 and row["rate_distance_front"]
    }
    length_front = {
        (row["n"], row["k"], row["d"])
        for row in aggregated
        if row["max_check_weight"] == 4 and row["length_fom_front"]
    }
    assert rate_front == {(16, 2, 4), (84, 2, 9)}
    assert length_front == {(16, 2, 4)}
