"""Regression tests for the complete LC audit of frontier PBB components."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_weight5_pbb_component_lc import build_records, render


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results" / "weight5_pbb_component_lc_audit.jsonl"


def test_frontier_component_lc_audit_is_complete_and_negative():
    records = build_records()
    manifest, *audits = records
    assert manifest["counts"] == {
        "component_classes": 4,
        "lc_css_equivalent": 0,
        "certified_not_lc_css_equivalent": 4,
    }
    assert [row["component_class_label"] for row in audits] == [
        "PC001",
        "PC002",
        "PC003",
        "PC004",
    ]
    assert all(row["commutation_verified"] for row in audits)
    assert all(row["parameters"] == {"n": 12, "k": 2, "d": 3} for row in audits)
    assert all(row["search"]["complete"] for row in audits)
    assert all(not row["search"]["is_lc_css"] for row in audits)
    assert all(row["search"]["assignments_covered"] == "6^12" for row in audits)


def test_frontier_component_lc_artifact_is_fresh():
    assert ARTIFACT.read_text(encoding="utf-8") == render(build_records())
