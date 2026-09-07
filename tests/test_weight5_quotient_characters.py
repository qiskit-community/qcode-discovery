"""Regression tests for the weight-five algebraic-family audit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_weight5_quotient_characters import (
    DEFAULT_OUTPUT,
    build_analysis,
    cyclic_common_factor_details,
    cyclic_extension_splits,
    primitive_powers,
    summarize,
    summarize_cyclic_gcd_multiplicities,
    summarize_lattice_strata,
    support_subgroup_metrics,
    summarize_support_strata,
    two_block_rank_defects,
)


ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = ROOT / "results" / "weight5_publication_catalogue.jsonl"


@pytest.fixture(scope="module")
def records() -> list[dict]:
    with CATALOGUE.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.fixture(scope="module")
def analysis(records) -> dict:
    return build_analysis(records, CATALOGUE.resolve())


def test_noncyclic_two_direction_support_stratum_counts(records):
    summary = summarize_support_strata(records)
    assert summary["connected_css_classes"] == 329
    assert summary["classes_with_cyclic_ambient_presentation"] == 215
    assert summary["classes_with_only_noncyclic_ambient_presentations"] == 114
    assert summary["noncyclic_only_two_direction_cyclic_support_classes"] == 28
    assert summary["noncyclic_only_split_cyclic_support_classes"] == 18
    assert summary["noncyclic_only_nonsplit_cyclic_support_classes"] == 10
    assert summary["noncyclic_only_two_direction_zero_rank_defect_classes"] == 20
    assert summary["noncyclic_only_two_direction_nonzero_rank_defect_classes"] == 8
    assert summary["noncyclic_only_nonsplit_zero_rank_defect_classes"] == 2
    assert summary["noncyclic_only_nonsplit_nonzero_rank_defect_classes"] == 8
    assert summary["noncyclic_only_rank_two_support_classes"] == 86
    # Two high-distance classes plus five resolved by the uniform low-weight audit.
    assert (
        summary["exact_noncyclic_only_two_direction_cyclic_support_classes"] == 7
    )
    assert summary["common_kernel_vs_k_S_for_two_direction_classes"] == {
        "k_S_is_a_lower_bound": True,
        "classes_with_equality": 20,
        "classes_with_strict_inequality": 8,
        "class_histogram": {
            "k_S=0,common_kernel=2": 8,
            "k_S=2,common_kernel=2": 20,
        },
    }


def test_small_cyclotomic_quotients_exhaust_connected_css_classes(records):
    summary = summarize(
        records,
        lambda row: row["connectivity"]["is_connected"],
    )
    assert summary["classes"] == 329
    assert summary["order_3_classes"] == 298
    assert summary["order_5_classes"] == 0
    assert summary["order_5_presentations"] == 0
    assert summary["order_7_classes"] == 21
    assert summary["order_15_classes"] == 12
    assert summary["order_3_and_7_classes"] == 2
    assert summary["order_3_or_7_classes"] == 317
    assert summary["order_3_or_7_or_15_classes"] == 329
    assert summary["order_3_or_5_or_7_or_15_classes"] == 329


def test_order_five_field_element_and_lattice_coverage(records):
    assert len(set(primitive_powers(5))) == 5
    summary = summarize_lattice_strata(records)
    assert set(summary) == {"6x10", "9x9", "10x9", "16x12", "15x14", "18x12"}
    assert {
        lattice
        for lattice, item in summary.items()
        if 5 in item["cyclic_quotient_orders_available"]
    } == {"6x10", "10x9", "15x14"}
    assert all(
        3 in item["cyclic_quotient_orders_available"] for item in summary.values()
    )
    assert {
        lattice
        for lattice, item in summary.items()
        if 7 in item["cyclic_quotient_orders_available"]
    } == {"15x14"}
    assert {
        lattice
        for lattice, item in summary.items()
        if 15 in item["cyclic_quotient_orders_available"]
    } == {"6x10", "10x9", "15x14"}
    assert all(
        item["quotient_stratum_class_counts"]["5"] == 0
        for item in summary.values()
    )
    assert summary["10x9"]["quotient_stratum_class_counts"]["15"] == 5
    assert summary["15x14"]["quotient_stratum_class_counts"]["15"] == 7
    assert summary["6x10"]["quotient_stratum_class_counts"]["15"] == 0
    assert {
        lattice: (
            item["connected_css_classes"],
            item["exact_connected_css_classes"],
            item["quotient_stratum_class_counts"],
        )
        for lattice, item in summary.items()
    } == {
        "6x10": (20, 5, {"3": 20, "5": 0, "7": 0, "15": 0}),
        "9x9": (7, 2, {"3": 7, "5": 0, "7": 0, "15": 0}),
        "10x9": (83, 16, {"3": 78, "5": 0, "7": 0, "15": 5}),
        "16x12": (36, 7, {"3": 36, "5": 0, "7": 0, "15": 0}),
        "15x14": (132, 6, {"3": 106, "5": 0, "7": 21, "15": 7}),
        "18x12": (51, 2, {"3": 51, "5": 0, "7": 0, "15": 0}),
    }


def test_cyclic_gcd_multiplicities_are_explicit_and_squarefree(records):
    summary = summarize_cyclic_gcd_multiplicities(
        records, lambda row: row["connectivity"]["is_connected"]
    )
    assert summary["cyclic_classes"] == 215
    assert summary["classes_by_maximum_factor_multiplicity"] == {"1": 215}
    assert summary["classes_with_repeated_irreducible_factor"] == 0
    assert summary["classes_with_unclassified_factor"] == 0
    assert summary["dimension_formula_mismatch_class_ids"] == []
    assert summary["factor_occurrences_by_order_and_multiplicity"] == {
        "order_3_multiplicity_1": 184,
        "order_7_multiplicity_1": 21,
        "order_15_multiplicity_1": 12,
    }
    assert sum(item["classes"] for item in summary["factorization_profiles"]) == 215
    assert all(item["squarefree"] for item in summary["factorization_profiles"])


def test_selected_even_order_cyclic_gcd_keeps_multiplicity(records):
    row = next(
        row
        for row in records
        if row["presentation_id"] == "CL-885dfa58"
    )
    # The parent is disconnected, so use the explicit [[96,4,10]] realization
    # encoded by its (16,3) component rather than the parent row itself.
    component_row = {
        "parameters": {"ell": 16, "m": 3},
        "generators": {
            "A_terms": [[0, 0], [0, 1], [2, 2]],
            "B_terms": [[0, 0], [3, 0]],
        },
    }
    assert row["parameters"]["n"] == 384
    details = cyclic_common_factor_details(component_row)
    assert details == {
        "polynomial": "z^2+z+1",
        "degree_with_multiplicity": 2,
        "factors": [
            {
                "order": 3,
                "polynomial": "z^2+z+1",
                "degree": 2,
                "multiplicity": 1,
            }
        ],
        "unclassified_factor": None,
        "maximum_factor_multiplicity": 1,
        "squarefree": True,
    }


def test_cyclic_gcd_records_repeated_factor_multiplicity():
    synthetic_row = {
        "parameters": {"ell": 2, "m": 3},
        "generators": {
            # Under the CRT map these are both 1+z^2+z^4
            # = (z^2+z+1)^2 in F_2[z]/(z^6-1).
            "A_terms": [[0, 0], [0, 1], [0, 2]],
            "B_terms": [[0, 0], [0, 1], [0, 2]],
        },
    }
    details = cyclic_common_factor_details(synthetic_row)
    assert details["polynomial"] == "z^4+z^2+1"
    assert details["degree_with_multiplicity"] == 4
    assert details["maximum_factor_multiplicity"] == 2
    assert details["factors"][0]["multiplicity"] == 2
    assert not details["squarefree"]


def test_noncyclic_support_stratum_distance_control(records):
    control = summarize_support_strata(records)["distance_control"]
    patterned = control["two_direction_cyclic_support"]
    other = control["other_noncyclic_support"]
    assert patterned["classes"] == 28
    assert patterned["exact_classes"] == 7
    assert patterned["exact_distance_histogram"] == {"4": 5, "10": 2}
    assert patterned["classes_with_certified_lower_bound_at_least_5"] == 23
    assert patterned["logical_dimension_histogram"] == {"4": 28}
    assert other["classes"] == 86
    assert other["exact_classes"] == 9
    assert other["exact_distance_histogram"] == {"3": 2, "4": 4, "5": 3}
    assert other["classes_with_certified_lower_bound_at_least_5"] == 80
    assert other["exact_classes_with_distance_at_least_10"] == 0


def test_generated_analysis_artifact_is_current(analysis):
    assert json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8")) == analysis


@pytest.mark.parametrize(
    (
        "key",
        "ambient",
        "orders",
        "intersection",
        "shared_power",
        "extension_splits",
        "subsystem_k",
        "rank_defects",
    ),
    [
        (
            "120_4_10",
            [2, 30],
            (30, 10),
            5,
            (6, 6, 5),
            {"A": True, "B": True, "both": True},
            2,
            {"delta_X": 0, "delta_Z": 0},
        ),
        (
            "432_4_10",
            [6, 36],
            (36, 36),
            6,
            (6, 6, 6),
            {"A": False, "B": False, "both": False},
            0,
            {"delta_X": 2, "delta_Z": 2},
        ),
    ],
)
def test_exact_noncyclic_support_invariants(
    records,
    key,
    ambient,
    orders,
    intersection,
    shared_power,
    extension_splits,
    subsystem_k,
    rank_defects,
):
    example = summarize_support_strata(records)["exact_examples"][key]
    row = next(
        item for item in records if item["presentation_id"] == example["presentation_id"]
    )
    metrics = support_subgroup_metrics(row)
    assert metrics["ambient_invariant_factors"] == ambient
    assert (metrics["A"]["order"], metrics["B"]["order"]) == orders
    assert metrics["A"]["is_cyclic"]
    assert metrics["B"]["is_cyclic"]
    assert metrics["intersection_order"] == intersection
    assert metrics["cyclic_support_extensions_split"] == extension_splits
    assert metrics["generated_together_order"] == metrics["ambient_order"]
    assert metrics["generates_ambient_group"]
    assert (
        metrics["first_shared_power"]["A_exponent"],
        metrics["first_shared_power"]["B_exponent"],
        metrics["first_shared_power"]["element_order"],
    ) == shared_power
    rank_structure = two_block_rank_defects(row)
    assert rank_structure["block_erasure_subsystem_k"] == subsystem_k
    assert rank_structure["rank_defects"] == rank_defects
    assert rank_structure["logical_dimension"] == 4
    assert rank_structure["common_right_kernel_dimension"] == 2
    assert rank_structure["logical_dimension_from_common_kernel"] == 4
    assert rank_structure["block_erasure_is_common_kernel_lower_bound"]
    assert rank_structure["common_kernel_excess_over_k_S"] == 2 - subsystem_k
    assert rank_structure["kernel_sum_equality"] == (subsystem_k == 2)


def test_cyclic_extension_splitting_criterion():
    assert cyclic_extension_splits(30, 5)
    assert cyclic_extension_splits(10, 5)
    assert not cyclic_extension_splits(36, 6)
