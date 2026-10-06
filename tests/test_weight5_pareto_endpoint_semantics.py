from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "paper" / "2610.06623" / "figures" / "plot_weight5_pareto_comparison.py"

spec = importlib.util.spec_from_file_location("plot_weight5_pareto_comparison", SCRIPT)
assert spec is not None and spec.loader is not None
pareto = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pareto
spec.loader.exec_module(pareto)


def _record(*, supported: bool) -> dict[str, object]:
    return pareto.base_record(
        cohort="current_css_w5",
        generation="current",
        structure="CSS",
        weight=5,
        n=100,
        k=4,
        d=9,
        d_lower=5,
        status="interval",
        evidence="test evidence",
        source="test",
        upper_is_supported=supported,
    )


def test_decoder_estimate_is_not_an_upper_bound_or_envelope_input() -> None:
    estimated = pareto.aggregate_records([_record(supported=False)])[0]
    supported = pareto.aggregate_records([_record(supported=True)])[0]

    assert estimated["d_upper"] is None
    assert estimated["d_estimate"] == 9
    assert estimated["fom_upper"] is None
    assert estimated["fom_estimate"] == 3.24
    assert estimated["upper_bound_envelope"] is False
    assert r"\widehat d=9" in estimated["label"]

    assert supported["d_upper"] == 9
    assert supported["d_estimate"] is None
    assert supported["fom_upper"] == 3.24
    assert supported["fom_estimate"] is None
    assert supported["upper_bound_envelope"] is True


def test_csv_has_separate_supported_and_estimated_fields(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "comparison.csv"
    monkeypatch.setattr(pareto, "CSV_PATH", output)
    rows = pareto.aggregate_records([_record(supported=False)])

    pareto.write_csv(rows)

    with output.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["d_upper"] == ""
    assert row["d_estimate"] == "9"
    assert row["upper_is_supported"] == "False"
    assert row["fom_upper"] == ""
    assert row["fom_estimate"] == "3.24"


def test_archive_evidence_text_separates_b_from_rigorous_methods() -> None:
    text = pareto.archive_evidence_text(
        {
            "upper_is_supported": True,
            "evidence": {
                "exact_methods": [],
                "lower_bounds": [{"bound": 5, "method": "L4"}],
                "upper_bounds": [
                    {"bound": 9, "method": "I"},
                    {"bound": 12, "method": "B"},
                ],
            },
        }
    )

    assert "supported upper 9 (I)" in text
    assert "estimated endpoint 12 (B)" in text
