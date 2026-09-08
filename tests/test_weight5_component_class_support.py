from __future__ import annotations

from scripts.audit_weight5_component_classes import _component_classes


def _reduction(
    parent_id: str,
    *,
    upper: int,
    supported: bool,
    lower: int | None = 5,
    exact: bool = False,
) -> dict[str, object]:
    return {
        "family": "CSS",
        "parent_class_id": parent_id,
        "parent_class_label": parent_id,
        "parent_n": 100,
        "parent_k": 4,
        "parent_presentation_ids": [parent_id],
        "representative_parent_id": parent_id,
        "representative_component_qubits": list(range(100)),
        "multiplicity": 1,
        "component_n": 100,
        "component_k": 4,
        "component_digest": "a" * 64,
        "distance_lower": lower,
        "distance_upper": upper,
        "distance_exact": exact,
        "distance_upper_is_supported": supported,
    }


def test_component_class_keeps_b_only_endpoint_as_estimate() -> None:
    row = _component_classes(
        [
            _reduction("TC-estimate", upper=7, supported=False),
            _reduction("TC-bound", upper=9, supported=True),
        ]
    )[0]

    assert row["schema_version"] == 2
    assert row["distance"] == {
        "status": "certified_lower_with_estimated_upper",
        "status_code": "C",
        "lower": 5,
        "upper": 7,
        "estimated_upper": 7,
        "is_exact": False,
        "upper_is_supported": False,
        "fom_lower": 1.0,
        "fom_upper": None,
        "fom_estimate": 1.96,
    }


def test_component_class_support_follows_evidence_at_decisive_endpoint() -> None:
    row = _component_classes(
        [
            _reduction("TC-estimate", upper=7, supported=False),
            _reduction("TC-bound", upper=7, supported=True),
        ]
    )[0]

    assert row["distance"]["status"] == "certified_interval"
    assert row["distance"]["upper_is_supported"] is True
    assert row["distance"]["estimated_upper"] is None
    assert row["distance"]["fom_upper"] == 1.96
    assert row["distance"]["fom_estimate"] is None


def test_matching_lower_and_decoder_estimate_do_not_make_exact_distance() -> None:
    row = _component_classes(
        [_reduction("TC-estimate", upper=5, supported=False, lower=5)]
    )[0]

    assert row["distance"]["status_code"] == "C"
    assert row["distance"]["status"] == "certified_lower_with_estimated_upper"
    assert row["distance"]["is_exact"] is False
    assert row["distance"]["upper_is_supported"] is False
    assert row["distance"]["estimated_upper"] == 5


def test_matching_lower_and_supported_upper_close_component_distance() -> None:
    row = _component_classes(
        [_reduction("TC-bound", upper=5, supported=True, lower=5)]
    )[0]

    assert row["distance"]["status_code"] == "E"
    assert row["distance"]["status"] == "exact"
    assert row["distance"]["is_exact"] is True
    assert row["distance"]["upper_is_supported"] is True
    assert row["distance"]["estimated_upper"] is None
