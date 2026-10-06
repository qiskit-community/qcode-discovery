"""Integrity checks for the persisted weight-five Tanner-class audit."""

import json
import re
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SIDECAR = ROOT / "results" / "weight5_presentation_classes.jsonl"
CATALOGUE = ROOT / "results" / "weight5_publication_catalogue.jsonl"
SUPPLEMENT_TABLES = ROOT / "paper" / "2610.06623" / "weight5_supplemental_tables.tex"


def _test_css_spec(campaign, *, lower: int | None, upper: int):
    from scripts import generate_weight5_supplement as supplement

    A = ((0, 0), (1, 0))
    B = ((0, 0), (0, 1), (1, 1))
    code = supplement.build_bb_code(2, 2, list(A), list(B))
    spec = supplement.Spec(
        campaign=campaign,
        ell=2,
        m=2,
        n=int(code.num_qudits),
        k=int(code.dimension),
        A=A,
        B=B,
        C=(),
        D=(),
    )
    if lower is not None:
        spec.add_lower(lower, "L4")
    spec.add_upper(upper, "B")
    return spec


def test_decoder_scalar_cannot_close_a_direct_exact_distance():
    from scripts import generate_weight5_supplement as supplement

    spec = _test_css_spec(supplement.CAMPAIGNS[0], lower=5, upper=5)
    with pytest.raises(ValueError, match="unsupported exact-distance closure"):
        spec.final_bounds()


def test_decoder_scalar_cannot_close_a_cross_member_class_distance():
    from scripts import generate_weight5_supplement as supplement

    lower_member = _test_css_spec(
        supplement.CAMPAIGNS[0], lower=5, upper=6
    )
    upper_member = _test_css_spec(
        supplement.CAMPAIGNS[1], lower=None, upper=5
    )
    with pytest.raises(ValueError, match="B-only endpoint cannot close"):
        supplement.build_presentation_classes([lower_member, upper_member])


def load_classes() -> list[dict]:
    with SIDECAR.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_catalogue() -> list[dict]:
    with CATALOGUE.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_weight5_class_sidecar_accounting_and_schema():
    rows = load_classes()
    assert len(rows) == 630
    assert Counter(row["family"] for row in rows) == {"CSS": 535, "PBB": 95}
    assert len({row["class_id"] for row in rows}) == 630
    assert len({row["catalogue_label"] for row in rows}) == 630

    members = [member for row in rows for member in row["member_ids"]]
    assert len(members) == len(set(members)) == 1142
    for row in rows:
        assert row["schema_version"] == 2
        assert (
            row["equivalence_relation"]
            == "colored_stored_generator_tanner_isomorphism_v1"
        )
        assert row["canonicalizer"]["name"] == "python-igraph/BLISS"
        assert row["representative_id"] == row["member_ids"][0]
        assert row["class_id"].split("-", 1)[1] == row[
            "canonical_digest_sha256"
        ][:10]
        assert row["class_size"] == len(row["member_ids"])
        assert sum(row["member_direct_status_counts"].values()) == row["class_size"]
        assert row["class_upper_is_supported"] == any(
            method in {"H", "Hc", "I", "M", "S", "W"}
            for method in row["class_upper_methods"]
        )
        assert row["class_distance_exact"] is False or row[
            "class_upper_is_supported"
        ] is True


def test_reviewer_duplicate_examples_have_expected_classes():
    rows = load_classes()
    class_of = {
        member: row["class_id"] for row in rows for member in row["member_ids"]
    }
    sector_swap_pairs = (
        ("PS-bb41fc22", "PS-dd65d6ed"),
        ("PS-dcbd4997", "PS-02124a53"),
        ("PS-9828cd17", "PS-af58ae1f"),
        ("PS-b9bd7fa4", "PS-22083bd2"),
        ("PS-aacbee20", "PS-514d488f"),
        ("PL-ba29a9df", "PL-39082ddc"),
        ("PL-ce4ae900", "PL-7c740754"),
        ("PL-7bcb9ee6", "PL-71c11b15"),
    )
    for left, right in sector_swap_pairs:
        assert class_of[left] == class_of[right]

    exact_162_12_6 = {
        "CS-5aa2af5d",
        "CS-ed9ea89a",
        "CS-0e4d0ef4",
        "CS-0c92ab37",
    }
    assert len({class_of[member] for member in exact_162_12_6}) == 3
    assert class_of["CS-5aa2af5d"] == class_of["CS-ed9ea89a"]


def test_class_evidence_transfer_accounting():
    rows = load_classes()
    exact_classes = [row for row in rows if row["class_distance_exact"]]
    interval_classes = [
        row
        for row in rows
        if not row["class_distance_exact"] and row["class_distance_lower"] > 0
    ]
    assert len(exact_classes) == 227
    assert len(interval_classes) == 403
    assert sum(row["class_size"] for row in exact_classes) == 436
    direct = Counter()
    for row in exact_classes:
        direct.update(row["member_direct_status_counts"])
    assert direct == {"E": 398, "C": 38, "U": 0}


def test_compact_supplement_has_every_exact_class_member_and_one_per_nonexact_class():
    text = SUPPLEMENT_TABLES.read_text(encoding="utf-8")
    # Match catalogue rows specifically.  Representative presentation IDs are
    # also cited in the class-level parameter summary, so counting every ID in
    # the include would incorrectly treat those cross-references as duplicates.
    presentation_ids = re.findall(
        r"^\\texttt\{((?:CS|CL|PS|PL)-[0-9a-f]{8})/[CP][0-9]{3}\} &",
        text,
        flags=re.MULTILINE,
    )
    catalogue = load_catalogue()
    by_class = {}
    for row in catalogue:
        by_class.setdefault(row["equivalence"]["class_id"], []).append(row)

    def representative_key(row: dict) -> tuple:
        direct = row["distance_direct"]
        merged = row["distance_class"]
        strengthened = not direct["is_exact"] and (
            merged["lower"] > direct["lower"]
            or merged["upper"] < direct["upper"]
        )
        changes_display = strengthened or (
            merged["upper_is_supported"] and not direct["upper_is_supported"]
        )
        return (
            not changes_display,
            direct["status_code"] != "E",
            -direct["lower"],
            direct["upper"] - direct["lower"],
            direct["upper"],
            row["presentation_id"],
        )

    expected_ids = set()
    for members in by_class.values():
        if members[0]["distance_class"]["is_exact"]:
            expected_ids.update(row["presentation_id"] for row in members)
        else:
            expected_ids.add(min(members, key=representative_key)["presentation_id"])

    assert len(presentation_ids) == len(set(presentation_ids)) == 839
    assert set(presentation_ids) == expected_ids
    assert "CL-5cd6065f" not in presentation_ids
    assert sum(
        line.startswith(r"\texttt{") and "H$_c$" in line
        for line in text.splitlines()
    ) == 6
    table_iv_onward = text[text.index(r"\label{tab:w5-certified-summary}") :]
    assert r"\shortstack" not in table_iv_onward
    assert "Continued on next page" not in text
    assert "Every retained row has either an exact distance" in text
    assert "Explicit connected BB presentations of components" in text
    assert "all four exact PBB component classes with parameters $\\code{12,2,3}$" in text


def test_compact_catalogue_status_and_class_effect_columns_match_sidecar():
    text = SUPPLEMENT_TABLES.read_text(encoding="utf-8")
    catalogue_lines = [
        line for line in text.splitlines() if re.match(r"^\\texttt\{", line)
    ]
    by_id = {row["presentation_id"]: row for row in load_catalogue()}

    for line in catalogue_lines:
        columns = line.removesuffix(r"\\").split(" & ")
        match = re.match(
            r"^\\texttt\{((?:CS|CL|PS|PL)-[0-9a-f]{8})/[CP][0-9]{3}\}$",
            columns[0],
        )
        assert match is not None
        row = by_id[match.group(1)]
        direct = row["distance_direct"]
        merged = row["distance_class"]
        expected_status = (
            "Exact"
            if direct["status_code"] == "E"
            else ("Interval" if direct["upper_is_supported"] else "Estimate")
        )
        strengthened = not direct["is_exact"] and (
            merged["lower"] > direct["lower"]
            or merged["upper"] < direct["upper"]
        )
        support_upgrade = (
            merged["upper_is_supported"] and not direct["upper_is_supported"]
        )
        if not strengthened and not support_upgrade:
            expected_effect = "--"
        elif merged["is_exact"]:
            expected_effect = "Exact"
        elif not strengthened:
            expected_effect = "Supported"
        else:
            expected_effect = "Tighter"
        assert columns[4] == expected_status
        assert columns[5] == expected_effect

    assert r"C$\to$E" not in text
    assert r"C$\to$C" not in text

    exact_representative = next(
        line
        for line in catalogue_lines
        if line.startswith(r"\texttt{CL-afd67a34/C434}")
    )
    assert r"$10$ & Exact & -- & $0.93$" in exact_representative

    expected_nonexact_representatives = {
        "CS-2a6c3475": (
            "C006",
            r"5;\widehat{10}",
            "Estimate",
            "Tighter",
            r"0.83;\widehat{3.33}",
        ),
        "CS-5cdbb079": (
            "C045",
            r"5;\widehat{9}",
            "Estimate",
            "Tighter",
            r"0.62;\widehat{2.00}",
        ),
        "CL-ac793cd4": (
            "C426",
            r"5;\widehat{22}",
            "Estimate",
            "Tighter",
            r"0.23;\widehat{4.48}",
        ),
        "CL-9b2cfe69": (
            "C458",
            r"5;\widehat{18}",
            "Estimate",
            "Tighter",
            r"0.23;\widehat{3.00}",
        ),
        "CL-6e6caeb4": (
            "C466",
            "[5,12]",
            "Estimate",
            "Tighter",
            "[0.23,1.33]",
        ),
    }
    for presentation_id, values in expected_nonexact_representatives.items():
        class_label, distance, status, effect, fom = values
        line = next(
            row
            for row in catalogue_lines
            if row.startswith(rf"\texttt{{{presentation_id}/")
        )
        assert line.startswith(rf"\texttt{{{presentation_id}/{class_label}}}")
        assert rf"${distance}$ & {status} & {effect} & ${fom}$" in line

    support_upgrade = next(
        row
        for row in catalogue_lines
        if row.startswith(r"\texttt{CS-0a81f585/C041}")
    )
    assert r"$[5,9]$ & Estimate & Supported & $[0.62,2.00]$" in support_upgrade

    assert "S (exact low-weight symplectic result)" not in text
    assert "W (symplectic witness)" not in text
