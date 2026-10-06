#!/usr/bin/env python3
"""Build the weight-five campaign comparison data and exact-only figure.

The script deliberately separates exact distances, certified lower bounds,
decoder estimates, reported distances, and explicit upper bounds.  A retained
logical operator or MILP incumbent supports ``d <= d_upper``; a BP--OSD scalar
whose operator was not retained is recorded only as an estimated endpoint.  A
completed low-weight exhaustive search can additionally certify a lower endpoint.

The CSV retains every aggregated point, including rigorous intervals, decoder
estimates, reported endpoints, disconnected parent presentations, and their
evidence status.
Current weight-five rows are read only from the normalized publication
catalogue; raw discovery logs are not reinterpreted here.
The publication figure is deliberately stricter: it shows only exact-distance
records.  Every current weight-five parent is first reduced to one connected
component, so a replicated parent cannot inflate the displayed length.  Prior
campaign BB/PBB rows with stored polynomial supports receive the same
translation-subgroup reduction.  Every QEC Challenge anchor is reduced using
the invariant row-space audit of its submitted check matrices.  Other
literature anchors are retained at check weights four, five, six, and eight
with their connectivity status stated explicitly.

Run from anywhere in the repository::

    UV_CACHE_DIR=/tmp/qcode-uv-cache MPLCONFIGDIR=/tmp/qcode-mpl-cache \
      uv run python paper/2610.06623/figures/plot_weight5_pareto_comparison.py

Outputs are deterministic and written beside this script.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.connectivity import bicycle_translation_component_count

RESULTS = ROOT / "results"
OUTDIR = Path(__file__).resolve().parent
CSV_PATH = OUTDIR / "weight5_pareto_comparison_data.csv"
PDF_PATH = OUTDIR / "fig_weight5_pareto_comparison.pdf"
SVG_PATH = OUTDIR / "fig_weight5_pareto_comparison.svg"
PNG_PATH = OUTDIR / "fig_weight5_pareto_comparison.png"
CHALLENGE_SNAPSHOT = OUTDIR / "qldpc_challenge_bb_snapshot.json"
CHALLENGE_WEIGHT5_SNAPSHOT = OUTDIR / "qldpc_challenge_weight5_snapshot.json"
CHALLENGE_CONNECTIVITY_AUDIT = (
    RESULTS / "weight5_challenge_connectivity_audit.jsonl"
)
COMPONENT_CERTIFICATES = RESULTS / "weight5_component_certifications.jsonl"
PBB_COMPONENT_CERTIFICATES = RESULTS / "weight5_pbb_component_certifications.jsonl"
LIN_PRYADKO_AUDIT = RESULTS / "weight5_lin_pryadko_audit.json"
PUBLICATION_CATALOGUE = RESULTS / "weight5_publication_catalogue.jsonl"
PUBLICATION_MANIFEST = RESULTS / "weight5_publication_manifest.json"
COMPONENT_CLASSES = RESULTS / "weight5_component_classes.jsonl"
RATE_D_MAX = 30.0
FOM_MIN = 0.035
FOM_MAX = 33.0
RIGOROUS_UPPER_METHODS = {"H", "Hc", "I", "M", "S", "W"}

COLORS = {4: "#CC79A7", 5: "#0072B2", 6: "#D55E00", 8: "#009E73"}
CURRENT_CSS_MARKER = "*"
CURRENT_PBB_MARKER = "D"
REFERENCE_MARKER = "o"
COHORT_ORDER = {
    "current_css_w5": 0,
    "current_pbb_w5": 1,
    "current_component_w5": 2,
    "current_pbb_component_w5": 3,
    "current_component_class_w5": 4,
    "prior_css_w6": 5,
    "prior_css_w8": 6,
    "prior_pbb_w8": 7,
    "voss_w5": 8,
    "lin_pryadko_w5": 9,
    "khesin_lu_w5": 10,
    "khesin_lu_w6": 11,
    "bravyi_w6": 12,
    "wang_mueller_w6": 13,
    "liang_w6": 14,
    "qian_li_w6": 15,
    "qian_li_w8": 16,
    "symons_w8": 17,
    "qec_challenge_w4": 18,
    "qec_challenge_w5": 19,
    "qec_challenge_w6": 20,
    "qec_challenge_w8": 21,
}

CHALLENGE_COMMIT = "3b50503b058a0c09500c1017b54c3d547e1101cc"

def read_jsonl(relative_path: str) -> list[dict[str, Any]]:
    path = RESULTS / relative_path
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_challenge_connectivity_audit() -> dict[str, dict[str, Any]]:
    """Load and validate the pinned Challenge row-space audit artifact."""
    if not CHALLENGE_CONNECTIVITY_AUDIT.exists():
        raise FileNotFoundError(
            "missing required Challenge connectivity artifact: "
            f"{CHALLENGE_CONNECTIVITY_AUDIT}"
        )
    rows = [
        json.loads(line)
        for line in CHALLENGE_CONNECTIVITY_AUDIT.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    manifests = [row for row in rows if row.get("record_type") == "artifact_manifest"]
    audit_rows = [
        row
        for row in rows
        if row.get("record_type") == "qldpc_challenge_connectivity"
    ]
    if len(manifests) != 1:
        raise ValueError("Challenge connectivity artifact must have one manifest")
    manifest = manifests[0]
    if manifest.get("upstream", {}).get("commit") != CHALLENGE_COMMIT:
        raise ValueError("Challenge connectivity artifact uses the wrong commit")
    expected_snapshots = {
        "actual_weight_five": CHALLENGE_WEIGHT5_SNAPSHOT,
        "bb": CHALLENGE_SNAPSHOT,
    }
    for name, path in expected_snapshots.items():
        source = manifest.get("source_snapshots", {}).get(name, {})
        if (
            source.get("path") != path.relative_to(ROOT).as_posix()
            or source.get("sha256") != sha256_file(path)
        ):
            raise ValueError(
                f"Challenge connectivity artifact is stale relative to {path}"
            )
    if len(audit_rows) != int(manifest.get("counts", {}).get("records", -1)):
        raise ValueError("Challenge connectivity artifact record count is invalid")
    by_slug = {str(row["slug"]): row for row in audit_rows}
    if len(by_slug) != 102 or len(by_slug) != len(audit_rows):
        raise ValueError("expected 102 unique rows in Challenge connectivity artifact")
    for slug, row in by_slug.items():
        reduction = row.get("reduction")
        if not isinstance(reduction, dict) or reduction.get(
            "distance_transfer_validated"
        ) is not True:
            raise ValueError(f"Challenge row {slug} lacks a validated reduction")
    return by_slug


def audited_challenge_fields(
    snapshot_row: dict[str, Any],
    audit: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Return component-reduced parameters and provenance for one row."""
    slug = str(snapshot_row["slug"])
    if slug not in audit:
        raise ValueError(f"Challenge connectivity audit lacks {slug}")
    row = audit[slug]
    source_parameters = row["parameters"]
    expected = (
        int(snapshot_row["n"]),
        int(snapshot_row["k"]),
        int(snapshot_row["d"]),
        int(snapshot_row["max_check_weight"]),
        str(snapshot_row["distance_status"]),
    )
    observed = (
        int(source_parameters["n"]),
        int(source_parameters["k"]),
        int(source_parameters["d"]),
        int(source_parameters["max_check_weight"]),
        str(source_parameters["distance_status"]),
    )
    if observed != expected:
        raise ValueError(f"Challenge audit/source mismatch for {slug}")

    reduction = row["reduction"]
    method = str(reduction["method"])
    n, k = int(reduction["n"]), int(reduction["k"])
    connectivity = row["connectivity"]
    if method == "connected":
        connectivity_status = "row-space connected"
        decomposition = "connected by invariant row-space audit"
        construction_role = "audited Challenge presentation"
    elif method == "homogeneous_isomorphic":
        count = int(connectivity["n_components"])
        connectivity_status = "derived connected component"
        decomposition = f"{count}x[[{n},{k}]]"
        construction_role = "component of audited Challenge parent"
    elif method == "unique_positive_k":
        trivial = [
            component
            for component in connectivity["components"]
            if int(component["k"]) == 0
        ]
        connectivity_status = "derived connected logical component"
        decomposition = (
            f"[[{n},{k}]] plus {len(trivial)} k=0 stabilizer-state factors"
        )
        construction_role = "logical component of audited Challenge parent"
    else:
        raise ValueError(f"unsupported Challenge reduction method for {slug}: {method}")
    return {
        "n": n,
        "k": k,
        "connectivity_status": connectivity_status,
        "decomposition": decomposition,
        "presentation_agrees": bool(
            connectivity["presentation_partition_agrees"]
        ),
        "construction_role": construction_role,
        "parent_presentation_id": slug,
        "component_digest": str(
            connectivity.get("component_canonical_digest_sha256") or ""
        ),
    }


def normalized_terms(record: dict[str, Any], field: str) -> tuple[tuple[int, int], ...]:
    """Canonicalize a polynomial support modulo the recorded lattice."""
    ell, m = int(record["ell"]), int(record["m"])
    return tuple(
        sorted({(int(a) % ell, int(b) % m) for a, b in record.get(field, [])})
    )


def load_explicit_component_matches() -> list[dict[str, Any]]:
    """Load the certified explicit connected BB realizations."""
    if not COMPONENT_CERTIFICATES.exists():
        raise FileNotFoundError(
            f"missing required component certificate artifact: {COMPONENT_CERTIFICATES}"
        )
    rows = [
        json.loads(line)
        for line in COMPONENT_CERTIFICATES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    parent_count = 0
    explicit_rows: list[dict[str, Any]] = []
    for row in rows:
        kind = row.get("record_type")
        if kind == "component_distance_certificate":
            parent_count += 1
        elif kind == "explicit_bb_component_match":
            explicit_rows.append(row)
    if parent_count != 7 or len(explicit_rows) != 3:
        raise ValueError(
            "expected seven parent certificates and three explicit BB component matches, "
            f"got {parent_count} and {len(explicit_rows)}"
        )
    return explicit_rows


def load_explicit_pbb_component_matches() -> list[dict[str, Any]]:
    """Load the four certified explicit connected PBB realizations."""
    if not PBB_COMPONENT_CERTIFICATES.exists():
        raise FileNotFoundError(
            "missing required PBB component certificate artifact: "
            f"{PBB_COMPONENT_CERTIFICATES}"
        )
    rows = [
        json.loads(line)
        for line in PBB_COMPONENT_CERTIFICATES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    explicit_rows = [
        row for row in rows if row.get("record_type") == "explicit_pbb_component_match"
    ]
    if len(explicit_rows) != 4:
        raise ValueError(
            f"expected four explicit PBB component matches, got {len(explicit_rows)}"
        )
    return explicit_rows


def positive_int(value: Any, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return result


def base_record(
    *,
    cohort: str,
    generation: str,
    structure: str,
    weight: int,
    n: int,
    k: int,
    d: int,
    status: str,
    evidence: str,
    source: str,
    label: str = "",
    d_lower: int | None = None,
    connectivity_status: str = "not_audited",
    decomposition: str = "not audited",
    presentation_agrees: bool | None = None,
    presentation_class_id: str = "not audited",
    campaign_origin: str = "not applicable",
    construction_role: str = "reference",
    parent_presentation_id: str = "",
    component_digest: str = "",
    source_campaign: str = "",
    catalogue_status_code: str = "",
    class_status_code: str = "",
    class_d_lower: int | None = None,
    class_d_upper: int | None = None,
    class_d_estimate: int | None = None,
    upper_is_supported: bool | None = None,
    class_upper_is_supported: bool | None = None,
) -> dict[str, Any]:
    if status not in {"exact", "interval", "reported", "upper_bound"}:
        raise ValueError(f"unknown distance status: {status}")
    n, k, d = positive_int(n, "n"), positive_int(k, "k"), positive_int(d, "d")
    if status == "exact":
        d_lower = d
    elif status == "interval":
        d_lower = positive_int(d_lower, "d_lower")
        if d_lower > d:
            raise ValueError(f"empty distance interval: [{d_lower}, {d}]")
    elif d_lower is not None:
        raise ValueError(f"d_lower is not valid for distance status {status}")
    if upper_is_supported is None and status in {"exact", "interval", "upper_bound"}:
        upper_is_supported = True
    if status == "exact" and upper_is_supported is not True:
        raise ValueError("an exact distance requires supported upper evidence")
    if upper_is_supported is False and status == "reported":
        raise ValueError("a reported literature value is not a decoder estimate")
    if class_upper_is_supported is None and class_status_code == "E":
        class_upper_is_supported = True
    if class_upper_is_supported is True and (
        class_d_upper is None or class_d_estimate is not None
    ):
        raise ValueError("supported class endpoint is assigned inconsistently")
    if class_upper_is_supported is False and (
        class_d_upper is not None or class_d_estimate is None
    ):
        raise ValueError("estimated class endpoint is assigned inconsistently")
    return {
        "cohort": cohort,
        "generation": generation,
        "structure": structure,
        "max_check_weight": int(weight),
        "n": n,
        "k": k,
        "d": d,
        "d_lower": d_lower,
        "d_upper": d if upper_is_supported else None,
        "d_estimate": d if upper_is_supported is False else None,
        "upper_is_supported": upper_is_supported,
        "distance_status": status,
        "evidence": evidence,
        "source": source,
        "label": label,
        "connectivity_status": connectivity_status,
        "decomposition": decomposition,
        "presentation_agrees": presentation_agrees,
        "presentation_class_id": presentation_class_id,
        "campaign_origin": campaign_origin,
        "construction_role": construction_role,
        "parent_presentation_id": parent_presentation_id,
        "component_digest": component_digest,
        "source_campaign": source_campaign,
        "catalogue_status_code": catalogue_status_code,
        "class_status_code": class_status_code,
        "class_d_lower": class_d_lower,
        "class_d_upper": class_d_upper,
        "class_d_estimate": class_d_estimate,
        "class_upper_is_supported": class_upper_is_supported,
    }


def archive_evidence_text(distance: dict[str, Any]) -> str:
    """Summarize normalized direct-distance evidence without reinterpreting it."""
    evidence = distance["evidence"]
    exact = sorted({str(method) for method in evidence["exact_methods"]})
    lower = sorted(
        {(int(item["bound"]), str(item["method"])) for item in evidence["lower_bounds"]}
    )
    upper = sorted(
        {(int(item["bound"]), str(item["method"])) for item in evidence["upper_bounds"]}
    )
    unknown_upper_methods = {
        method for _, method in upper if method not in RIGOROUS_UPPER_METHODS | {"B"}
    }
    if unknown_upper_methods:
        raise ValueError(f"unknown upper-evidence methods: {unknown_upper_methods}")
    supported_upper = [
        item for item in upper if item[1] in RIGOROUS_UPPER_METHODS
    ]
    estimated_upper = [item for item in upper if item[1] == "B"]
    parts = []
    if exact:
        parts.append("exact methods " + "/".join(exact))
    if lower:
        parts.append(
            "certified lower "
            + ", ".join(f"{bound} ({method})" for bound, method in lower)
        )
    if supported_upper:
        parts.append(
            "supported upper "
            + ", ".join(
                f"{bound} ({method})" for bound, method in supported_upper
            )
        )
    if estimated_upper:
        parts.append(
            "estimated endpoint "
            + ", ".join(
                f"{bound} ({method})" for bound, method in estimated_upper
            )
        )
    return "normalized direct evidence: " + "; ".join(parts)


def load_current_publication_catalogue() -> list[dict[str, Any]]:
    """Load current weight-five presentations from the normalized archive."""
    manifest = json.loads(PUBLICATION_MANIFEST.read_text(encoding="utf-8"))
    catalogue_artifact = manifest["output_artifacts"]["catalogue"]
    rows = [
        json.loads(line)
        for line in PUBLICATION_CATALOGUE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = []
    seen_ids: set[str] = set()
    status_map = {"E": "exact", "C": "interval", "U": "upper_bound"}
    for row in rows:
        if (
            row.get("schema_version") != 2
            or row.get("record_type") != "weight5_presentation"
        ):
            raise ValueError("unsupported weight-five publication catalogue record")
        identifier = str(row["presentation_id"])
        if identifier in seen_ids:
            raise ValueError(f"duplicate publication presentation: {identifier}")
        seen_ids.add(identifier)

        family = str(row["family"])
        if family not in {"CSS", "PBB"}:
            raise ValueError(f"unexpected weight-five family: {family}")
        parameters = row["parameters"]
        distance = row["distance_direct"]
        class_distance = row["distance_class"]
        status_code = str(distance["status_code"])
        status = status_map[status_code]
        upper = positive_int(distance["upper"], "publication distance upper")
        lower = distance["lower"] if status == "interval" else None
        upper_is_supported = distance.get("upper_is_supported")
        class_upper_is_supported = class_distance.get("upper_is_supported")
        if not isinstance(upper_is_supported, bool) or not isinstance(
            class_upper_is_supported, bool
        ):
            raise ValueError(f"missing endpoint support status for {identifier}")
        if status == "exact" and not upper_is_supported:
            raise ValueError(f"exact distance lacks supported evidence: {identifier}")

        connectivity = row["connectivity"]
        connectivity_status = str(connectivity["state"])
        component = connectivity["component_code"]
        if connectivity_status == "connected":
            decomposition = "connected"
        elif connectivity_status == "disconnected" and component is not None:
            decomposition = (
                f"{positive_int(component['multiplicity'], 'component multiplicity')}x"
                f"[[{positive_int(component['n'], 'component n')},"
                f"{positive_int(component['k'], 'component k')}]]"
            )
        else:
            raise ValueError(f"invalid normalized connectivity for {identifier}")

        origin = row["provenance"]["origin"]
        campaign_origin = "campaign catalogue"
        if family == "PBB":
            campaign_origin = "fixed safety net"
            if not origin["fixed_safety_net"]:
                campaign_origin = "search generated"
        certificate = row.get("component_certification")
        component_digest = ""
        if certificate is not None:
            component_digest = str(
                certificate["component_certificate"]["canonical_digest_sha256"]
            )

        records.append(
            base_record(
                cohort=f"current_{family.lower()}_w5",
                generation="current",
                structure=family,
                weight=5,
                n=parameters["n"],
                k=parameters["k"],
                d=upper,
                d_lower=lower,
                status=status,
                evidence=archive_evidence_text(distance),
                source="results/weight5_publication_catalogue.jsonl",
                connectivity_status=connectivity_status,
                decomposition=decomposition,
                presentation_agrees=connectivity["presentation_partition_agrees"],
                presentation_class_id=str(row["equivalence"]["class_id"]),
                campaign_origin=campaign_origin,
                construction_role="parent presentation",
                parent_presentation_id=identifier,
                component_digest=component_digest,
                source_campaign=str(row["campaign"]),
                catalogue_status_code=status_code,
                class_status_code=str(class_distance["status_code"]),
                class_d_lower=class_distance["lower"],
                class_d_upper=(
                    class_distance["upper"]
                    if class_upper_is_supported
                    else None
                ),
                class_d_estimate=(
                    None
                    if class_upper_is_supported
                    else class_distance["upper"]
                ),
                upper_is_supported=upper_is_supported,
                class_upper_is_supported=class_upper_is_supported,
            )
        )
    expected = int(manifest["counts"]["retained_presentations"])
    if len(records) != expected:
        raise ValueError(
            f"publication manifest declares {expected} current records, "
            f"found {len(records)}"
        )
    if (
        int(catalogue_artifact["records"]) != expected
        or str(catalogue_artifact["sha256"]) != sha256_file(PUBLICATION_CATALOGUE)
    ):
        raise ValueError("publication manifest does not describe the catalogue file")
    return records


def load_component_classes() -> list[dict[str, Any]]:
    """Load one primitive record per component-canonical weight-five class."""
    if not COMPONENT_CLASSES.exists():
        raise FileNotFoundError(
            f"missing required component-class artifact: {COMPONENT_CLASSES}"
        )
    rows = [
        json.loads(line)
        for line in COMPONENT_CLASSES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifests = [row for row in rows if row.get("record_type") == "artifact_manifest"]
    classes = [
        row for row in rows if row.get("record_type") == "weight5_component_class"
    ]
    if len(manifests) != 1:
        raise ValueError("component-class artifact must contain one manifest")
    manifest = manifests[0]
    if manifest.get("schema_version") != 2:
        raise ValueError("unsupported component-class artifact schema")
    source = manifest["source_file"]
    source_records = int(source["records"])
    source_digest = str(source["sha256"])
    actual_source_records = sum(
        1
        for line in PUBLICATION_CATALOGUE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    actual_source_digest = sha256_file(PUBLICATION_CATALOGUE)
    if (source_records, source_digest) != (
        actual_source_records,
        actual_source_digest,
    ):
        raise ValueError(
            "component-class artifact is stale relative to the publication "
            "catalogue"
        )
    expected = int(manifest["counts"]["component_classes"])
    if len(classes) != expected:
        raise ValueError(
            f"component-class artifact declares {expected} classes, found {len(classes)}"
        )

    status_map = {"E": "exact", "C": "interval", "U": "upper_bound"}
    records = []
    for row in classes:
        if row.get("schema_version") != 2:
            raise ValueError("unsupported component-class record schema")
        parameters = row["parameters"]
        distance = row["distance"]
        status_code = str(distance["status_code"])
        status = status_map[status_code]
        n = positive_int(parameters["n"], "component-class n")
        k = positive_int(parameters["k"], "component-class k")
        upper = positive_int(distance["upper"], "component-class distance upper")
        lower = distance["lower"] if status == "interval" else None
        upper_is_supported = distance.get("upper_is_supported")
        if not isinstance(upper_is_supported, bool):
            raise ValueError("component class lacks endpoint support status")
        expected_estimate = None if upper_is_supported else upper
        if distance.get("estimated_upper") != expected_estimate:
            raise ValueError("component-class estimate disagrees with support status")
        records.append(
            base_record(
                cohort="current_component_class_w5",
                generation="current",
                structure=str(row["family"]),
                weight=5,
                n=n,
                k=k,
                d=upper,
                d_lower=lower,
                status=status,
                evidence=(
                    (
                        "supported distance bound transferred from "
                        "component-canonical parent classes"
                        if upper_is_supported
                        else "estimated endpoint transferred from "
                        "component-canonical parent classes"
                    )
                ),
                source="results/weight5_component_classes.jsonl",
                connectivity_status="component canonicalized",
                decomposition="connected component",
                presentation_agrees=True,
                presentation_class_id=str(row["component_class_id"]),
                campaign_origin="campaign component quotient",
                construction_role="component-canonical class",
                parent_presentation_id="; ".join(
                    row["parent_presentation_ids"]
                ),
                component_digest=str(row["component_digest_sha256"]),
                source_campaign="component quotient",
                catalogue_status_code=status_code,
                class_status_code=status_code,
                class_d_lower=distance["lower"],
                class_d_upper=upper if upper_is_supported else None,
                class_d_estimate=None if upper_is_supported else upper,
                upper_is_supported=upper_is_supported,
                class_upper_is_supported=upper_is_supported,
            )
        )
    return records


def reduced_prior_parameters(
    row: dict[str, Any],
    *,
    family: str,
    a_field: str = "A_terms",
    b_field: str = "B_terms",
    c_field: str | None = None,
    d_field: str | None = None,
) -> dict[str, Any]:
    """Apply the translation-subgroup component audit to a prior campaign row."""
    A = normalized_terms(row, a_field)
    B = normalized_terms(row, b_field)
    C = normalized_terms(row, c_field) if c_field is not None else ()
    D = normalized_terms(row, d_field) if d_field is not None else ()
    multiplicity = bicycle_translation_component_count(
        int(row["ell"]), int(row["m"]), A, B, C, D
    )
    n, k = int(row["n"]), int(row["k"])
    if n % multiplicity or k % multiplicity:
        raise ValueError(
            f"{family} comparator [[{n},{k}]] is incompatible with "
            f"translation multiplicity {multiplicity}"
        )
    if multiplicity == 1:
        return {
            "n": n,
            "k": k,
            "connectivity_status": "translation-connected",
            "decomposition": "connected by translation-subgroup audit",
            "construction_role": "audited prior campaign presentation",
        }
    component_n, component_k = n // multiplicity, k // multiplicity
    return {
        "n": component_n,
        "k": component_k,
        "connectivity_status": "derived connected component",
        "decomposition": f"{multiplicity}x[[{component_n},{component_k}]]",
        "construction_role": "component of audited prior campaign parent",
    }


def load_prior_css() -> list[dict[str, Any]]:
    """Load prior CSS points from Campaigns 1--4 using only proof flags.

    The Campaigns 1--3 catalog has no machine-readable optimality field: its
    ``CONFIRMED``/``TIGHTER`` labels compare ILP values with BP--OSD and are not
    solver certificates.  Those values are therefore upper bounds.  Catalog
    points already present in the newer Campaign-4 file are not repeated.
    """
    source = "campaign4_milp_verified.jsonl"
    records = []
    campaign4_displays: set[tuple[int, int, int, int]] = set()
    for row in read_jsonl(source):
        weight = len(normalized_terms(row, "A_terms")) + len(
            normalized_terms(row, "B_terms")
        )
        if weight not in {6, 8}:
            continue
        reduced = reduced_prior_parameters(row, family="CSS")
        campaign4_displays.add((weight, int(row["n"]), int(row["k"]), int(row["d"])))
        exact = row.get("d_is_exact") is True
        records.append(
            base_record(
                cohort=f"prior_css_w{weight}",
                generation="prior",
                structure="CSS",
                weight=weight,
                n=reduced["n"],
                k=reduced["k"],
                d=row["d"],
                status="exact" if exact else "upper_bound",
                evidence=(
                    "MILP certificate"
                    if exact
                    else "logical-operator witness (incomplete MILP or low-weight search)"
                ),
                source=f"results/{source}",
                connectivity_status=reduced["connectivity_status"],
                decomposition=reduced["decomposition"],
                presentation_agrees=True,
                construction_role=reduced["construction_role"],
            )
        )

    catalog_path = RESULTS / "ilp_catalog.json"
    with catalog_path.open(encoding="utf-8") as handle:
        catalog = json.load(handle)
    catalog_points: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for section in ("n=144", "n=288", "n=360"):
        for row in catalog[section]:
            weight = len({tuple(term) for term in row["A"]}) + len(
                {tuple(term) for term in row["B"]}
            )
            point = (
                weight,
                2 * int(row["ell"]) * int(row["m"]),
                int(row["k"]),
                int(row["ilp_d"]),
            )
            if weight in {6, 8} and point not in campaign4_displays:
                catalog_points.setdefault(point, row)
    for (weight, n, k, d), row in sorted(catalog_points.items()):
        # The Challenge entry 144-24-6 is the same (12,6) construction as
        # the catalog row retained above.  Its d_exact certificate upgrades
        # that earlier incumbent rather than creating a duplicate point.
        challenge_exact_upgrade = (weight, n, k, d) == (6, 144, 24, 6)
        reduced = reduced_prior_parameters(
            {**row, "n": n, "k": k},
            family="CSS",
            a_field="A",
            b_field="B",
        )
        records.append(
            base_record(
                cohort=f"prior_css_w{weight}",
                generation="prior",
                structure="CSS",
                weight=weight,
                n=reduced["n"],
                k=reduced["k"],
                d=d,
                status="exact" if challenge_exact_upgrade else "upper_bound",
                evidence=(
                    "QEC Challenge exact-distance certificate"
                    if challenge_exact_upgrade
                    else "ILP incumbent without a machine-readable optimality flag"
                ),
                source=(
                    "results/ilp_catalog.json; QEC Challenge "
                    f"{CHALLENGE_COMMIT[:8]} (codes/144-24-6.json)"
                    if challenge_exact_upgrade
                    else "results/ilp_catalog.json"
                ),
                connectivity_status=reduced["connectivity_status"],
                decomposition=reduced["decomposition"],
                presentation_agrees=True,
                construction_role=reduced["construction_role"],
            )
        )
    return records


def pbb_max_check_weight(row: dict[str, Any]) -> int:
    """Return the actual largest PBB generator support after cancellations."""
    a = set(normalized_terms(row, "A_terms"))
    b = set(normalized_terms(row, "B_terms"))
    c = set(normalized_terms(row, "C_terms"))
    d = set(normalized_terms(row, "D_terms"))
    return max(len(a | c) + len(b | d), len(a) + len(b))


def load_prior_pbb() -> list[dict[str, Any]]:
    """Load BLISS-distinct weight-eight PBB comparison points."""
    source = "campaign7_publication_merged.jsonl"
    records = []
    for row in read_jsonl(source):
        weight = pbb_max_check_weight(row)
        if weight != 8:
            continue
        exact = row.get("d_is_exact") is True
        if not exact and not (
            row.get("trust_level") == "TRUSTED" and row.get("d_is_upper_bound") is True
        ):
            raise ValueError("weight-eight prior PBB row lacks exact or trusted-UB status")
        reduced = reduced_prior_parameters(
            row,
            family="PBB",
            c_field="C_terms",
            d_field="D_terms",
        )
        records.append(
            base_record(
                cohort="prior_pbb_w8",
                generation="prior",
                structure="PBB",
                weight=8,
                n=reduced["n"],
                k=reduced["k"],
                d=row["d"],
                status="exact" if exact else "upper_bound",
                evidence="MILP certificate" if exact else "trusted logical-operator witness",
                source=f"results/{source}",
                connectivity_status=reduced["connectivity_status"],
                decomposition=reduced["decomposition"],
                presentation_agrees=True,
                construction_role=reduced["construction_role"],
            )
        )
    return records


def load_exact_lin_pryadko_rows() -> list[dict[str, Any]]:
    """Load archived cyclic constructions independently MILP-certified here."""
    if not LIN_PRYADKO_AUDIT.exists():
        raise FileNotFoundError(
            f"missing required Lin--Pryadko audit artifact: {LIN_PRYADKO_AUDIT}"
        )
    audit = json.loads(LIN_PRYADKO_AUDIT.read_text(encoding="utf-8"))
    if audit.get("archive", {}).get("commit") != (
        "403d194c3f98f0cadc236aecbc4a8b6139ccf23c"
    ):
        raise ValueError("unexpected Lin--Pryadko archive revision")
    nonabelian_rows = audit.get("nonabelian_frontier_rows") or []
    rows = [
        *(audit.get("rows") or []),
        *[
            row
            for row in nonabelian_rows
            if row.get("exact_milp", {}).get("exact") is True
        ],
    ]
    if len(rows) != 7:
        raise ValueError(f"expected seven exact Lin--Pryadko audit rows, got {len(rows)}")
    incomplete_labels = {
        row.get("label")
        for row in nonabelian_rows
        if row.get("milp_attempt", {}).get("all_objectives_optimal") is False
    }
    if incomplete_labels != {"LP-168-4-14", "LP-180-4-15", "LP-192-4-16"}:
        raise ValueError(
            f"unexpected incomplete Lin--Pryadko audit rows: {incomplete_labels}"
        )

    records = []
    for row in rows:
        reconstructed = row["reconstructed_code"]
        milp = row["exact_milp"]
        if (
            reconstructed.get("connected") is not True
            or reconstructed.get("stabilizer_weight") != 5
            or milp.get("exact") is not True
            or milp.get("all_objectives_optimal") is not True
            or milp.get("logical_objectives_optimal")
            != milp.get("logical_objectives_total")
        ):
            raise ValueError(f"incomplete Lin--Pryadko audit row: {row.get('label')}")
        n = positive_int(reconstructed["n"], "Lin--Pryadko n")
        k = positive_int(reconstructed["k"], "Lin--Pryadko k")
        d = positive_int(milp["d"], "Lin--Pryadko exact distance")
        records.append(
            base_record(
                cohort="lin_pryadko_w5",
                generation="literature",
                structure=(
                    "nonabelian 2BGA"
                    if row.get("family_classification", {}).get(
                        "ambient_group_nonabelian"
                    )
                    else "cyclic 2BGA"
                ),
                weight=5,
                n=n,
                k=k,
                d=d,
                status="exact",
                evidence="independent all-logical MILP certificate",
                source="results/weight5_lin_pryadko_audit.json",
                label=rf"$[[{n},{k},{d}]]$",
                connectivity_status="connected",
                decomposition="connected",
                presentation_agrees=True,
                campaign_origin="Lin--Pryadko 2BGA archive",
                construction_role="prior construction; exact certificate here",
            )
        )
    return records


def load_literature() -> list[dict[str, Any]]:
    """Published comparison anchors; ``<=`` is retained as an upper bound."""
    voss = [
        (30, 4, 5, "exact", "published distance; independently exhaustive-verified"),
        (72, 4, 8, "reported", "published distance; no local optimality certificate"),
        (96, 4, 8, "reported", "published distance; no local optimality certificate"),
    ]
    # Selected parameter rows from the public archive accompanying
    # arXiv:2306.16400.  The archive's GAP code calls QDistRnd with 100,000
    # trials, so these are reported randomized values rather than exact
    # certificates under the evidence convention used by this figure.
    lin_pryadko_reported = [
        (120, 4, 10),
        (140, 6, 10),
        (168, 4, 14),
        (180, 4, 13),
        (180, 4, 14),
        (180, 4, 15),
        (192, 4, 16),
    ]
    bravyi = [
        (72, 12, 6, "exact"),
        (90, 8, 10, "exact"),
        (108, 8, 10, "exact"),
        (144, 12, 12, "exact"),
        (288, 12, 18, "exact"),
        (360, 12, 24, "upper_bound"),
    ]
    records = [
        base_record(
            cohort="voss_w5",
            generation="literature",
            structure="literature",
            weight=5,
            n=n,
            k=k,
            d=d,
            status=status,
            evidence=evidence,
            source="Voss et al., arXiv:2406.19151",
            label=rf"$[[{n},{k},{d}]]$",
        )
        for n, k, d, status, evidence in voss
    ]
    records.extend(
        base_record(
            cohort="lin_pryadko_w5",
            generation="literature",
            structure="2BGA",
            weight=5,
            n=n,
            k=k,
            d=d,
            status="reported",
            evidence="public-archive QDistRnd value; no exact certificate",
            source=(
                "Lin and Pryadko, arXiv:2306.16400; 2BGA-codes archive "
                "403d194c3f98"
            ),
            label=rf"$[[{n},{k},{d}]]$",
        )
        for n, k, d in lin_pryadko_reported
    )
    records.extend(load_exact_lin_pryadko_rows())
    # Khesin--Lu also report [[30,4,5]], but it is BLISS-equivalent to the
    # Voss code above and is therefore not entered a second time.
    records.append(
        base_record(
            cohort="khesin_lu_w5",
            generation="literature",
            structure="literature",
            weight=5,
            n=36,
            k=4,
            d=6,
            status="exact",
            evidence="published exact distance; independently exhaustive-verified",
            source="Khesin and Lu, arXiv:2603.05496",
            label=r"$[[36,4,6]]$",
        )
    )
    records.append(
        base_record(
            cohort="khesin_lu_w6",
            generation="literature",
            structure="literature",
            weight=6,
            n=60,
            k=4,
            d=10,
            status="exact",
            evidence="published exact distance",
            source="Khesin and Lu, arXiv:2603.05496",
            label=r"$[[60,4,10]]$",
        )
    )
    records.extend(
        base_record(
            cohort="bravyi_w6",
            generation="literature",
            structure="literature",
            weight=6,
            n=n,
            k=k,
            d=d,
            status=status,
            evidence=(
                "published exact distance"
                if status == "exact"
                else "published upper bound"
            ),
            source="Bravyi et al., arXiv:2308.07915",
            label=(rf"$[[{n},{k},{d}]]$" if status == "exact" else rf"$[[{n},{k},\leq {d}]]$"),
        )
        for n, k, d, status in bravyi
    )
    records.append(
        base_record(
            cohort="wang_mueller_w6",
            generation="literature",
            structure="literature",
            weight=6,
            n=150,
            k=16,
            d=8,
            status="reported",
            evidence="published distance; no local optimality certificate",
            source="Wang and Mueller, arXiv:2408.10001",
            label=r"$[[150,16,8]]$",
        )
    )
    for n, k, d in (
        (96, 4, 12),
        (120, 8, 12),
        (140, 6, 14),
        (180, 8, 16),
        (254, 14, 16),
        (294, 10, 20),
        (340, 16, 18),
    ):
        records.append(
            base_record(
                cohort="liang_w6",
                generation="literature",
                structure="literature",
                weight=6,
                n=n,
                k=k,
                d=d,
                status="exact",
                evidence="published exact integer-programming distance",
                source=(
                    "Liang, Liu, Song, and Chen, arXiv:2503.03827; "
                    "PRX Quantum 6, 020357 (2025)"
                ),
                label=rf"$[[{n},{k},{d}]]$",
            )
        )
    # Frontier-relevant exact codes in Qian--Li's Tables I and II.
    for weight, n, k, d in (
        (6, 336, 12, 20),
        (8, 224, 22, 16),
        (8, 288, 24, 18),
        (8, 378, 32, 19),
    ):
        records.append(
            base_record(
                cohort=f"qian_li_w{weight}",
                generation="literature",
                structure="coset-orbit balanced product",
                weight=weight,
                n=n,
                k=k,
                d=d,
                status="exact",
                evidence=(
                    "published exact MILP distance; published overall weight "
                    "also constrains total qubit degree"
                ),
                source="Qian and Li, arXiv:2608.08996v1",
                label=rf"$[[{n},{k},{d}]]$",
            )
        )
    records.append(
        base_record(
            cohort="symons_w8",
            generation="literature",
            structure="literature",
            weight=8,
            n=144,
            k=14,
            d=14,
            status="exact",
            evidence="published exact mixed-integer-programming distance",
            source="Symons, Rajput, and Browne, arXiv:2511.13560",
            label=r"$[[144,14,14]]$",
        )
    )
    return records


def load_qec_challenge_weight5(
    audit: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Load every actual weight-five entry at the pinned Challenge commit.

    The Challenge's public weight classes are upper-bound buckets: its
    ``weight-6`` class contains checks of maximum row weight five as well as
    six.  These five records are selected by their recomputed maximum row
    weight, independently of their heterogeneous family tags.
    """
    with CHALLENGE_WEIGHT5_SNAPSHOT.open(encoding="utf-8") as handle:
        snapshot = json.load(handle)
    if snapshot.get("commit") != CHALLENGE_COMMIT:
        raise ValueError(
            "QEC Challenge weight-five snapshot is not at the pinned commit"
        )
    raw = snapshot.get("records")
    if not isinstance(raw, list) or len(raw) != 5:
        raise ValueError("expected five actual-weight-five QEC Challenge entries")
    if len({row.get("slug") for row in raw}) != len(raw):
        raise ValueError("duplicate slug in QEC Challenge weight-five snapshot")
    if any(row.get("max_check_weight") != 5 for row in raw):
        raise ValueError("non-weight-five row in QEC Challenge weight-five snapshot")
    for row in raw:
        code_digest = row.get("code_sha256")
        certificate_digest = row.get("certificate_sha256")
        if not isinstance(code_digest, str) or len(code_digest) != 64:
            raise ValueError("invalid source-code hash in Challenge weight-five snapshot")
        if row.get("distance_status") == "exact":
            if not isinstance(certificate_digest, str) or len(certificate_digest) != 64:
                raise ValueError(
                    "exact Challenge weight-five row lacks a certificate hash"
                )
        elif certificate_digest is not None:
            raise ValueError(
                "upper-bound Challenge weight-five row unexpectedly cites a certificate"
            )
    status_counts = Counter(str(row.get("distance_status")) for row in raw)
    if status_counts != Counter({"exact": 3, "upper_bound": 2}):
        raise ValueError(
            f"unexpected Challenge weight-five proof-status counts: {status_counts}"
        )

    records = []
    for row in raw:
        status = str(row["distance_status"])
        reduced = audited_challenge_fields(row, audit)
        records.append(
            base_record(
                cohort="qec_challenge_w5",
                generation="challenge",
                structure=str(row["family"]),
                weight=5,
                n=reduced["n"],
                k=reduced["k"],
                d=row["d"],
                status=status,
                evidence=(
                    "QEC Challenge exact-distance certificate"
                    if status == "exact"
                    else "QEC Challenge verifier-confirmed logical witness"
                ),
                source=(
                    "Unitary Foundation QEC Challenge "
                    f"{CHALLENGE_COMMIT[:8]} (codes/{row['slug']}.json); "
                    "results/weight5_challenge_connectivity_audit.jsonl"
                ),
                connectivity_status=reduced["connectivity_status"],
                decomposition=reduced["decomposition"],
                presentation_agrees=reduced["presentation_agrees"],
                construction_role=reduced["construction_role"],
                parent_presentation_id=reduced["parent_presentation_id"],
                component_digest=reduced["component_digest"],
                source_campaign="QEC Challenge",
            )
        )
    return records


def load_qec_challenge_bb(
    existing_records: list[dict[str, Any]],
    audit: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Load the pinned QEC Challenge BB entries at weights four, six, and eight.

    The upstream family label records provenance; the snapshot recomputes the
    actual maximum row weight from the submitted check matrices.  Challenge
    entries that duplicate an existing parameter/proof point are suppressed
    here because this is a parameter comparison, not a construction census.
    The fixed snapshot contains all 97 BB-tagged entries.
    """
    with CHALLENGE_SNAPSHOT.open(encoding="utf-8") as handle:
        snapshot = json.load(handle)
    if snapshot.get("commit") != CHALLENGE_COMMIT:
        raise ValueError("QEC Challenge snapshot is not at the pinned commit")
    raw = snapshot.get("records")
    if not isinstance(raw, list) or len(raw) != 97:
        raise ValueError("expected 97 BB-tagged QEC Challenge snapshot entries")
    status_counts: dict[str, int] = defaultdict(int)
    for row in raw:
        status_counts[str(row.get("distance_status"))] += 1
    if dict(status_counts) != {"exact": 55, "upper_bound": 42}:
        raise ValueError(f"unexpected Challenge proof-status counts: {status_counts}")

    proof_strength = {"reported": 0, "upper_bound": 1, "interval": 2, "exact": 3}
    existing_strength: dict[tuple[int, int, int, int], int] = defaultdict(lambda: -1)
    for row in existing_records:
        key = (row["max_check_weight"], row["n"], row["k"], row["d"])
        existing_strength[key] = max(
            existing_strength[key], proof_strength[row["distance_status"]]
        )

    expected_suppressed = {
        "96-4-12",
        "72-12-6",
        "90-8-10",
        "108-8-10",
        "120-8-12",
        "140-6-14",
        "144-12-12",
        "144-16-8",
        "144-24-6",
        "162-36-4",
        "288-12-18",
        "288-32-8",
        "288-48-6",
        "288-46-8",
        "288-50-8",
        "340-16-18",
        "360-16-14",
        "360-24-10",
        "360-32-6",
        "360-12-24",
        "12-4-2",
    }
    suppressed: set[str] = set()
    records = []
    for row in raw:
        weight = positive_int(row["max_check_weight"], "Challenge max_check_weight")
        if weight not in {4, 6, 8}:
            continue
        status = str(row["distance_status"])
        if status not in {"exact", "upper_bound"}:
            raise ValueError(f"unexpected Challenge distance status: {status}")
        reduced = audited_challenge_fields(row, audit)
        key = (weight, reduced["n"], reduced["k"], int(row["d"]))
        if existing_strength[key] >= proof_strength[status]:
            suppressed.add(str(row["slug"]))
            continue
        records.append(
            base_record(
                cohort=f"qec_challenge_w{weight}",
                generation="challenge",
                structure="BB",
                weight=weight,
                n=reduced["n"],
                k=reduced["k"],
                d=row["d"],
                status=status,
                evidence=(
                    "QEC Challenge exact-distance certificate"
                    if status == "exact"
                    else "QEC Challenge verifier-confirmed logical witness"
                ),
                source=(
                    "Unitary Foundation QEC Challenge "
                    f"{CHALLENGE_COMMIT[:8]} (codes/{row['slug']}.json); "
                    "results/weight5_challenge_connectivity_audit.jsonl"
                ),
                connectivity_status=reduced["connectivity_status"],
                decomposition=reduced["decomposition"],
                presentation_agrees=reduced["presentation_agrees"],
                construction_role=reduced["construction_role"],
                parent_presentation_id=reduced["parent_presentation_id"],
                component_digest=reduced["component_digest"],
                source_campaign="QEC Challenge",
            )
        )
    if suppressed != expected_suppressed:
        raise ValueError(
            "unexpected QEC Challenge overlap set: "
            f"missing={sorted(expected_suppressed - suppressed)}, "
            f"extra={sorted(suppressed - expected_suppressed)}"
        )
    if len(records) != 70:
        raise ValueError(f"expected 70 new Challenge comparison rows, got {len(records)}")
    return records


def load_explicit_components(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert certified component-to-BB matches into primitive plot records."""
    records = []
    for row in rows:
        component = row["explicit_bb_component"]
        match = row["match"]
        parent = row["parent_source_identity"]
        if component.get("connected") is not True:
            raise ValueError(f"explicit component is not connected: {row['match_id']}")
        if match.get("bliss_isomorphic") is not True or match.get(
            "distance_transfer_valid"
        ) is not True:
            raise ValueError(f"component match is not certified: {row['match_id']}")
        if positive_int(component["stabilizer_weight"], "component weight") != 5:
            raise ValueError(f"component is not weight five: {row['match_id']}")
        digest = str(match["canonical_digest_sha256"])
        n = positive_int(component["n"], "component n")
        k = positive_int(component["k"], "component k")
        d = positive_int(component["d_exact"], "component d")
        records.append(
            base_record(
                cohort="current_component_w5",
                generation="current",
                structure="CSS BB",
                weight=5,
                n=n,
                k=k,
                d=d,
                status="exact",
                evidence=(
                    "exact parent distance transferred through a certified "
                    "colored-BLISS component match"
                ),
                source="results/weight5_component_certifications.jsonl",
                label=rf"$[[{n},{k},{d}]]$",
                connectivity_status="derived connected component",
                decomposition="connected explicit BB realization",
                presentation_agrees=True,
                presentation_class_id=f"TC-{digest[:10]}",
                campaign_origin="derived from campaign parent",
                construction_role="explicit connected component",
                parent_presentation_id=str(parent["presentation_id"]),
                component_digest=digest,
                source_campaign="derived component",
                catalogue_status_code="E",
                class_status_code="E",
                class_d_lower=d,
                class_d_upper=d,
            )
        )
    return records


def load_explicit_pbb_components(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert certified PBB component matches into primitive plot records."""
    records = []
    for row in rows:
        component = row["explicit_pbb_component"]
        match = row["match"]
        parent = row["parent_source_identity"]
        if component.get("connected") is not True:
            raise ValueError(f"explicit PBB component is not connected: {row['match_id']}")
        if match.get("bliss_isomorphic") is not True or match.get(
            "distance_transfer_valid"
        ) is not True:
            raise ValueError(f"PBB component match is not certified: {row['match_id']}")
        if positive_int(component["stabilizer_weight"], "PBB component weight") != 5:
            raise ValueError(f"PBB component is not weight five: {row['match_id']}")
        digest = str(match["canonical_digest_sha256"])
        n = positive_int(component["n"], "PBB component n")
        k = positive_int(component["k"], "PBB component k")
        d = positive_int(component["d_exact"], "PBB component d")
        records.append(
            base_record(
                cohort="current_pbb_component_w5",
                generation="current",
                structure="PBB",
                weight=5,
                n=n,
                k=k,
                d=d,
                status="exact",
                evidence=(
                    "exact parent distance transferred through a certified "
                    "colored-BLISS PBB component match"
                ),
                source="results/weight5_pbb_component_certifications.jsonl",
                label=rf"$[[{n},{k},{d}]]$",
                connectivity_status="derived connected component",
                decomposition="connected explicit PBB realization",
                presentation_agrees=True,
                presentation_class_id=f"TP-{digest[:10]}",
                campaign_origin="derived from campaign parent",
                construction_role="explicit connected PBB component",
                parent_presentation_id=str(parent["presentation_id"]),
                component_digest=digest,
                source_campaign="derived component",
                catalogue_status_code="E",
                class_status_code="E",
                class_d_lower=d,
                class_d_upper=d,
            )
        )
    return records


def load_records() -> list[dict[str, Any]]:
    challenge_audit = load_challenge_connectivity_audit()
    records = load_current_publication_catalogue()
    records.extend(load_component_classes())
    records.extend(load_explicit_components(load_explicit_component_matches()))
    records.extend(
        load_explicit_pbb_components(load_explicit_pbb_component_matches())
    )
    records.extend(load_prior_css())
    records.extend(load_prior_pbb())
    records.extend(load_literature())
    records.extend(load_qec_challenge_weight5(challenge_audit))
    records.extend(load_qec_challenge_bb(records, challenge_audit))
    return records


def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Whether left dominates right in rate, distance, and block length."""
    lhs = (left["rate"], float(left["d"]), -float(left["n"]))
    rhs = (right["rate"], float(right["d"]), -float(right["n"]))
    weak = all(a >= b - 1e-12 for a, b in zip(lhs, rhs))
    strict = any(a > b + 1e-12 for a, b in zip(lhs, rhs))
    return weak and strict


def aggregate_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse identical displayed points and attach frontier flags."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record["cohort"],
            record["n"],
            record["k"],
            record["d"],
            record["d_lower"],
            record["distance_status"],
            record["connectivity_status"],
            record["construction_role"],
            record["campaign_origin"],
            record["source_campaign"],
            record["catalogue_status_code"],
            record["upper_is_supported"],
            record["class_status_code"],
            record["class_d_lower"],
            record["class_d_upper"],
            record["class_d_estimate"],
            record["class_upper_is_supported"],
        )
        groups[key].append(record)

    aggregated = []
    for _, members in groups.items():
        first = members[0]
        row = dict(first)
        row["multiplicity"] = len(members)
        row["evidence"] = "; ".join(sorted({member["evidence"] for member in members}))
        row["source"] = "; ".join(sorted({member["source"] for member in members}))
        connectivity_states = {member["connectivity_status"] for member in members}
        row["connectivity_status"] = (
            next(iter(connectivity_states))
            if len(connectivity_states) == 1
            else "mixed"
        )
        row["decomposition"] = "; ".join(
            sorted({member["decomposition"] for member in members})
        )
        agreement = {
            member["presentation_agrees"]
            for member in members
            if member["presentation_agrees"] is not None
        }
        row["presentation_agrees"] = (
            "" if not agreement else (len(agreement) == 1 and True in agreement)
        )
        class_ids = sorted(
            {
                member["presentation_class_id"]
                for member in members
                if member["presentation_class_id"] != "not audited"
            }
        )
        row["presentation_class_id"] = "; ".join(class_ids) or "not audited"
        row["presentation_class_count"] = len(class_ids)
        row["campaign_origin"] = "; ".join(
            sorted({member["campaign_origin"] for member in members})
        )
        row["construction_role"] = "; ".join(
            sorted({member["construction_role"] for member in members})
        )
        row["parent_presentation_id"] = "; ".join(
            sorted(
                {
                    member["parent_presentation_id"]
                    for member in members
                    if member["parent_presentation_id"]
                }
            )
        )
        row["component_digest"] = "; ".join(
            sorted(
                {
                    member["component_digest"]
                    for member in members
                    if member["component_digest"]
                }
            )
        )
        row["rate"] = row["k"] / row["n"]
        row["fom"] = row["k"] * row["d"] ** 2 / row["n"]
        row["fom_upper"] = row["fom"] if row["d_upper"] is not None else None
        row["fom_estimate"] = (
            row["fom"] if row["d_estimate"] is not None else None
        )
        row["rate_panel_clipped"] = (
            row["distance_status"] in {"interval", "upper_bound"}
            and row["d"] > RATE_D_MAX
        )
        row["fom_panel_clipped"] = (
            row["distance_status"] in {"interval", "upper_bound"}
            and row["fom"] > FOM_MAX
        )
        row["plot_clipped"] = row["rate_panel_clipped"] or row["fom_panel_clipped"]
        if not row["label"]:
            if row["d_estimate"] is not None and row["d_lower"] is not None:
                distance = rf"d\geq {row['d_lower']};\ \widehat d={row['d_estimate']}"
            elif row["d_estimate"] is not None:
                distance = rf"\widehat d={row['d_estimate']}"
            elif row["distance_status"] == "upper_bound":
                distance = rf"\leq {row['d']}"
            elif row["distance_status"] == "interval":
                distance = rf"{row['d_lower']}\leq d\leq {row['d']}"
            else:
                distance = str(row["d"])
            row["label"] = rf"$[[{row['n']},{row['k']},{distance}]]$"
        row["exact_pareto"] = False
        row["upper_bound_envelope"] = False
        row["rate_distance_front"] = False
        row["length_fom_front"] = False
        row["figure_exact_eligible"] = False
        row["figure_exclusion_reason"] = ""
        aggregated.append(row)

    exact = [row for row in aggregated if row["distance_status"] == "exact"]
    upper = [
        row
        for row in aggregated
        if row["distance_status"] in {"interval", "upper_bound"}
        and row["d_upper"] is not None
    ]
    for row in exact:
        row["exact_pareto"] = not any(
            other is not row and dominates(other, row) for other in exact
        )
    # Reported literature values and B-only decoder estimates lack a local
    # certificate and therefore do not define the upper-bound envelope.
    upper_comparators = exact + upper
    for row in upper:
        row["upper_bound_envelope"] = not any(
            other is not row and dominates(other, row) for other in upper_comparators
        )

    return sorted(
        aggregated,
        key=lambda row: (
            COHORT_ORDER[row["cohort"]],
            row["n"],
            row["k"],
            row["d"],
            row["distance_status"],
        ),
    )


def mark_figure_selection(rows: list[dict[str, Any]]) -> None:
    """Mark exact, primitive-aware per-weight records for the main figure."""

    generation_priority = {
        "literature": 0,
        "challenge": 1,
        "prior": 2,
        "current": 3,
    }

    for row in rows:
        if row["distance_status"] != "exact":
            row["figure_exclusion_reason"] = "distance not exact"
        elif row["d"] <= 2:
            row["figure_exclusion_reason"] = "distance at most two"
        elif (
            row["generation"] == "current"
            and row["cohort"] != "current_component_class_w5"
        ):
            row["figure_exclusion_reason"] = (
                "superseded by complete component-canonical catalogue"
            )
        elif (
            row["generation"] == "current"
            and row["connectivity_status"] != "component canonicalized"
        ):
            row["figure_exclusion_reason"] = "component connectivity unresolved"
        else:
            row["figure_exact_eligible"] = True

    def mark_exact_front(
        weight: int,
        *,
        flag: str,
        dominates_panel: Any,
        coordinate: Any,
        preference: Any,
    ) -> None:
        candidates = [
            row
            for row in rows
            if row["max_check_weight"] == weight
            and row["figure_exact_eligible"]
        ]
        frontier = [
            row
            for row in candidates
            if not any(
                other is not row and dominates_panel(other, row)
                for other in candidates
            )
        ]
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in frontier:
            grouped[coordinate(row)].append(row)
        for coincident in grouped.values():
            chosen = min(
                coincident,
                key=lambda row: (
                    *preference(row),
                    generation_priority[row["generation"]],
                    COHORT_ORDER[row["cohort"]],
                ),
            )
            chosen[flag] = True

    for weight in (4, 5, 6, 8):
        mark_exact_front(
            weight,
            flag="rate_distance_front",
            dominates_panel=lambda left, right: (
                left["rate"] >= right["rate"] - 1e-12
                and left["d"] >= right["d"]
                and (
                    left["rate"] > right["rate"] + 1e-12
                    or left["d"] > right["d"]
                )
            ),
            coordinate=lambda row: (round(row["rate"], 12), row["d"]),
            preference=lambda row: (row["n"],),
        )
        mark_exact_front(
            weight,
            flag="length_fom_front",
            dominates_panel=lambda left, right: (
                left["n"] <= right["n"]
                and left["fom"] >= right["fom"] - 1e-12
                and (left["n"] < right["n"] or left["fom"] > right["fom"] + 1e-12)
            ),
            coordinate=lambda row: (row["n"], round(row["fom"], 12)),
            preference=lambda row: (-row["d"],),
        )


def validate(records: list[dict[str, Any]], aggregated: list[dict[str, Any]]) -> None:
    publication_manifest = json.loads(
        PUBLICATION_MANIFEST.read_text(encoding="utf-8")
    )
    publication_counts = publication_manifest["counts"]
    component_artifact_rows = [
        json.loads(line)
        for line in COMPONENT_CLASSES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    component_manifest = next(
        row
        for row in component_artifact_rows
        if row.get("record_type") == "artifact_manifest"
    )
    status_map = {"E": "exact", "C": "interval", "U": "upper_bound"}

    expected_raw = {
        ("current_component_w5", "exact"): 3,
        ("current_pbb_component_w5", "exact"): 4,
        ("prior_css_w6", "exact"): 410,
        ("prior_css_w6", "upper_bound"): 252,
        ("prior_css_w8", "exact"): 194,
        ("prior_css_w8", "upper_bound"): 281,
        ("prior_pbb_w8", "exact"): 96,
        ("prior_pbb_w8", "upper_bound"): 54,
        ("voss_w5", "exact"): 1,
        ("voss_w5", "reported"): 2,
        ("lin_pryadko_w5", "exact"): 7,
        ("lin_pryadko_w5", "reported"): 7,
        ("khesin_lu_w5", "exact"): 1,
        ("khesin_lu_w6", "exact"): 1,
        ("bravyi_w6", "exact"): 5,
        ("bravyi_w6", "upper_bound"): 1,
        ("wang_mueller_w6", "reported"): 1,
        ("liang_w6", "exact"): 7,
        ("qian_li_w6", "exact"): 1,
        ("qian_li_w8", "exact"): 3,
        ("symons_w8", "exact"): 1,
        ("qec_challenge_w4", "exact"): 3,
        ("qec_challenge_w4", "upper_bound"): 2,
        ("qec_challenge_w5", "exact"): 3,
        ("qec_challenge_w5", "upper_bound"): 2,
        ("qec_challenge_w6", "exact"): 34,
        ("qec_challenge_w6", "upper_bound"): 24,
        ("qec_challenge_w8", "exact"): 2,
        ("qec_challenge_w8", "upper_bound"): 5,
    }
    for campaign, counts in publication_counts[
        "presentations_by_campaign_and_direct_status"
    ].items():
        cohort = "current_css_w5" if campaign.startswith("css-") else "current_pbb_w5"
        for status_code, count in counts.items():
            if count:
                key = (cohort, status_map[status_code])
                expected_raw[key] = expected_raw.get(key, 0) + int(count)
    for status_code, count in component_manifest["counts"][
        "component_classes_by_status"
    ].items():
        if count:
            expected_raw[("current_component_class_w5", status_map[status_code])] = int(
                count
            )
    observed: dict[tuple[str, str], int] = defaultdict(int)
    for record in records:
        observed[(record["cohort"], record["distance_status"])] += 1
    if dict(observed) != expected_raw:
        raise ValueError(f"unexpected cohort/status counts: {dict(observed)}")
    if len(records) != sum(expected_raw.values()):
        raise ValueError(f"unexpected total record count: {len(records)}")
    for record in records:
        supported = record["upper_is_supported"]
        expected_upper = record["d"] if supported is True else None
        expected_estimate = record["d"] if supported is False else None
        if (
            record["d_upper"] != expected_upper
            or record["d_estimate"] != expected_estimate
        ):
            raise ValueError(
                "distance endpoint was assigned to the wrong support field: "
                f"{record['cohort']} [[{record['n']},{record['k']}]]"
            )
    if any(
        row["upper_bound_envelope"] and row["upper_is_supported"] is not True
        for row in aggregated
    ):
        raise ValueError("an estimated endpoint entered the upper-bound envelope")
    challenge_w4 = {
        (row["n"], row["k"], row["d"], row["distance_status"])
        for row in records
        if row["cohort"] == "qec_challenge_w4"
    }
    if challenge_w4 != {
        (84, 2, 9, "exact"),
        (16, 2, 4, "exact"),
        (16, 2, 4, "upper_bound"),
        (64, 2, 8, "upper_bound"),
    }:
        raise ValueError(f"unexpected weight-four Challenge set: {challenge_w4}")
    challenge_w5 = {
        (row["n"], row["k"], row["d"], row["distance_status"])
        for row in records
        if row["cohort"] == "qec_challenge_w5"
    }
    if challenge_w5 != {
        (18, 4, 3, "upper_bound"),
        (40, 6, 4, "exact"),
        (40, 10, 4, "exact"),
        (45, 9, 3, "exact"),
        (676, 71, 4, "upper_bound"),
    }:
        raise ValueError(f"unexpected weight-five Challenge set: {challenge_w5}")
    current_parents = [
        row
        for row in records
        if row["generation"] == "current"
        and row["construction_role"] == "parent presentation"
    ]
    expected_parent_count = int(publication_counts["retained_presentations"])
    if len(current_parents) != expected_parent_count or any(
        row["d"] <= 2 for row in current_parents
    ):
        raise ValueError(
            f"expected {expected_parent_count} current parents retained by "
            "distance upper endpoint >=3"
        )
    if {row["source"] for row in current_parents} != {
        "results/weight5_publication_catalogue.jsonl"
    }:
        raise ValueError(
            "current weight-five rows must come only from the publication archive"
        )
    campaign_counts: dict[str, int] = defaultdict(int)
    for row in current_parents:
        campaign_counts[row["source_campaign"]] += 1
    expected_campaign_counts = {
        campaign: sum(int(count) for count in status_counts.values())
        for campaign, status_counts in publication_counts[
            "presentations_by_campaign_and_direct_status"
        ].items()
    }
    if dict(campaign_counts) != expected_campaign_counts:
        raise ValueError(f"unexpected normalized campaign counts: {dict(campaign_counts)}")
    retained_current_classes = {
        row["presentation_class_id"]
        for row in current_parents
    }
    expected_parent_classes = int(publication_counts["retained_classes"])
    if len(retained_current_classes) != expected_parent_classes:
        raise ValueError(
            f"expected {expected_parent_classes} retained current "
            "stored-presentation classes, got "
            f"{len(retained_current_classes)}"
        )
    class_status = {
        class_id: next(
            row["class_status_code"]
            for row in current_parents
            if row["presentation_class_id"] == class_id
        )
        for class_id in retained_current_classes
    }
    class_status_counts: dict[str, int] = defaultdict(int)
    for status_code in class_status.values():
        class_status_counts[status_code] += 1
    expected_class_status_counts = {
        status: int(count)
        for status, count in publication_counts["classes_by_status"].items()
        if count
    }
    if dict(class_status_counts) != expected_class_status_counts:
        raise ValueError(f"unexpected normalized class statuses: {class_status_counts}")
    component_class_rows = [
        row for row in records if row["cohort"] == "current_component_class_w5"
    ]
    expected_component_classes = int(
        component_manifest["counts"]["component_classes"]
    )
    if len(component_class_rows) != expected_component_classes:
        raise ValueError(
            f"expected {expected_component_classes} component-canonical classes, "
            f"got {len(component_class_rows)}"
        )
    component_parameters = {
        (row["n"], row["k"], row["d"])
        for row in records
        if row["construction_role"] == "explicit connected component"
    }
    if component_parameters != {(90, 4, 9), (96, 4, 10), (140, 6, 10)}:
        raise ValueError(f"unexpected explicit component set: {component_parameters}")
    pbb_component_parameters = {
        (row["n"], row["k"], row["d"])
        for row in records
        if row["construction_role"] == "explicit connected PBB component"
    }
    if pbb_component_parameters != {(144, 4, 8), (180, 2, 11), (216, 4, 10)}:
        raise ValueError(
            f"unexpected explicit PBB component set: {pbb_component_parameters}"
        )
    selected = [
        row
        for row in aggregated
        if row["rate_distance_front"] or row["length_fom_front"]
    ]
    if any(
        row["generation"] == "current"
        and row["construction_role"] == "parent presentation"
        and row["figure_exact_eligible"]
        for row in aggregated
    ):
        raise ValueError("a current parent presentation is figure-eligible")
    if any(row["distance_status"] != "exact" for row in selected):
        raise ValueError("a non-exact row entered the exact-only figure")
    if any(
        row["generation"] == "current"
        and row["cohort"] != "current_component_class_w5"
        for row in selected
    ):
        raise ValueError("a noncanonical current presentation entered the figure")
    rate_front_counts = {
        weight: sum(
            row["rate_distance_front"]
            for row in aggregated
            if row["max_check_weight"] == weight
        )
        for weight in (4, 5, 6, 8)
    }
    fom_front_counts = {
        weight: sum(
            row["length_fom_front"]
            for row in aggregated
            if row["max_check_weight"] == weight
        )
        for weight in (4, 5, 6, 8)
    }
    if any(count == 0 for count in rate_front_counts.values()) or any(
        count == 0 for count in fom_front_counts.values()
    ):
        raise ValueError(
            f"empty exact record curve: rate={rate_front_counts}, FOM={fom_front_counts}"
        )
    expected_w4_fronts = {
        "rate_distance_front": {(84, 2, 9), (16, 2, 4)},
        "length_fom_front": {(16, 2, 4)},
    }
    for flag, expected_w4_front in expected_w4_fronts.items():
        observed_w4_front = {
            (row["n"], row["k"], row["d"])
            for row in aggregated
            if row["max_check_weight"] == 4 and row[flag]
        }
        if observed_w4_front != expected_w4_front:
            raise ValueError(
                f"unexpected weight-four {flag}: {observed_w4_front}"
            )
    expected_w5_fronts = {
        "rate_distance_front": {
            (180, 4, 14),
            (132, 4, 12),
            (140, 6, 10),
            (78, 4, 9),
            (60, 4, 8),
            (48, 4, 7),
            (36, 4, 6),
            (30, 4, 5),
            (40, 10, 4),
        },
        "length_fom_front": {
            (12, 2, 3),
            (30, 4, 5),
            (36, 4, 6),
            (48, 4, 7),
            (60, 4, 8),
            (132, 4, 12),
        },
    }
    for flag, expected in expected_w5_fronts.items():
        observed = {
            (row["n"], row["k"], row["d"])
            for row in aggregated
            if row["max_check_weight"] == 5 and row[flag]
        }
        if observed != expected:
            raise ValueError(f"unexpected weight-five {flag}: {observed}")

    expected_literature_attribution = {
        (30, 4, 5): "voss_w5",
        (36, 4, 6): "khesin_lu_w5",
        (48, 4, 7): "lin_pryadko_w5",
    }
    for parameters, cohort in expected_literature_attribution.items():
        matches = [
            row
            for row in selected
            if (row["n"], row["k"], row["d"]) == parameters
            and row["max_check_weight"] == 5
        ]
        if not matches or any(row["cohort"] != cohort for row in matches):
            raise ValueError(
                f"weight-five frontier attribution for {parameters} is not {cohort}"
            )

    expected_qian_li_attribution = {
        (6, 336, 12, 20): "qian_li_w6",
        (8, 224, 22, 16): "qian_li_w8",
        (8, 288, 24, 18): "qian_li_w8",
        (8, 378, 32, 19): "qian_li_w8",
    }
    for (weight, n, k, d), cohort in expected_qian_li_attribution.items():
        matches = [
            row
            for row in selected
            if row["max_check_weight"] == weight
            and (row["n"], row["k"], row["d"]) == (n, k, d)
        ]
        if len(matches) != 1 or matches[0]["cohort"] != cohort:
            raise ValueError(
                f"Qian--Li frontier attribution for {(n, k, d)} is not {cohort}"
            )

    expected_cross_weight_fronts = {
        (6, "rate_distance_front"): {
            (336, 12, 20),
            (340, 16, 18),
            (254, 14, 16),
            (144, 12, 12),
            (90, 8, 10),
            (72, 8, 8),
            (72, 12, 6),
            (18, 4, 4),
        },
        (6, "length_fom_front"): {
            (18, 4, 4),
            (30, 4, 6),
            (48, 4, 8),
            (60, 4, 10),
            (72, 8, 8),
            (84, 6, 10),
            (90, 8, 10),
            (120, 8, 12),
            (144, 12, 12),
            (254, 14, 16),
            (336, 12, 20),
            (340, 16, 18),
        },
        (8, "rate_distance_front"): {
            (378, 32, 19),
            (224, 22, 16),
            (288, 50, 8),
            (72, 22, 4),
        },
        (8, "length_fom_front"): {
            (18, 2, 3),
            (32, 6, 4),
            (36, 4, 6),
            (60, 4, 8),
            (72, 12, 6),
            (108, 8, 10),
            (144, 14, 14),
            (224, 22, 16),
            (288, 24, 18),
            (378, 32, 19),
        },
    }
    for (weight, flag), expected in expected_cross_weight_fronts.items():
        observed = {
            (row["n"], row["k"], row["d"])
            for row in aggregated
            if row["max_check_weight"] == weight and row[flag]
        }
        if observed != expected:
            raise ValueError(f"unexpected weight-{weight} {flag}: {observed}")


def write_csv(rows: list[dict[str, Any]]) -> None:
    fields = [
        "cohort",
        "generation",
        "structure",
        "max_check_weight",
        "n",
        "k",
        "d",
        "d_lower",
        "d_upper",
        "d_estimate",
        "upper_is_supported",
        "distance_status",
        "catalogue_status_code",
        "class_status_code",
        "class_d_lower",
        "class_d_upper",
        "class_d_estimate",
        "class_upper_is_supported",
        "rate",
        "fom",
        "fom_upper",
        "fom_estimate",
        "multiplicity",
        "rate_panel_clipped",
        "fom_panel_clipped",
        "plot_clipped",
        "exact_pareto",
        "upper_bound_envelope",
        "rate_distance_front",
        "length_fom_front",
        "figure_exact_eligible",
        "figure_exclusion_reason",
        "connectivity_status",
        "decomposition",
        "presentation_agrees",
        "presentation_class_id",
        "presentation_class_count",
        "campaign_origin",
        "source_campaign",
        "construction_role",
        "parent_presentation_id",
        "component_digest",
        "evidence",
        "source",
        "label",
    ]
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            output = {field: row[field] for field in fields}
            output["rate"] = f"{row['rate']:.12g}"
            output["fom"] = f"{row['fom']:.12g}"
            output["fom_upper"] = (
                "" if row["fom_upper"] is None else f"{row['fom_upper']:.12g}"
            )
            output["fom_estimate"] = (
                ""
                if row["fom_estimate"] is None
                else f"{row['fom_estimate']:.12g}"
            )
            writer.writerow(output)


def plot_exact_record_curve(
    ax: Any,
    rows: list[dict[str, Any]],
    *,
    flag: str,
    xfield: str,
    yfield: str,
) -> None:
    """Draw a sparse exact frontier, with lines used only as visual guides."""
    for weight in (4, 5, 6, 8):
        frontier = sorted(
            (row for row in rows if row[flag] and row["max_check_weight"] == weight),
            key=lambda row: row[xfield],
        )
        ax.plot(
            [row[xfield] for row in frontier],
            [row[yfield] for row in frontier],
            color=COLORS[weight],
            linewidth=1.05,
            alpha=0.72,
            zorder=2,
        )
        for selector, marker, size in (
            (
                lambda row: row["generation"] == "current"
                and row["structure"] == "CSS",
                CURRENT_CSS_MARKER,
                70.0,
            ),
            (
                lambda row: row["generation"] == "current"
                and row["structure"] == "PBB",
                CURRENT_PBB_MARKER,
                46.0,
            ),
            (
                lambda row: row["generation"] != "current",
                REFERENCE_MARKER,
                31.0,
            ),
        ):
            subset = [row for row in frontier if selector(row)]
            if not subset:
                continue
            is_current = subset[0]["generation"] == "current"
            ax.scatter(
                [row[xfield] for row in subset],
                [row[yfield] for row in subset],
                s=size,
                marker=marker,
                facecolors=COLORS[weight],
                edgecolors="#222222" if is_current else "white",
                linewidths=0.75 if is_current else 0.55,
                zorder=4,
            )


def find_row(rows: list[dict[str, Any]], **criteria: Any) -> dict[str, Any]:
    matches = [
        row for row in rows if all(row[field] == value for field, value in criteria.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"annotation target is not unique: {criteria}")
    return matches[0]


def annotate_row(
    ax: Any,
    row: dict[str, Any],
    *,
    xfield: str,
    yfield: str,
    text: str,
    offset: tuple[int, int],
) -> None:
    ax.annotate(
        text,
        (row[xfield], row[yfield]),
        xytext=offset,
        textcoords="offset points",
        fontsize=6.7,
        color="#222222",
        arrowprops={"arrowstyle": "-", "color": "#555555", "lw": 0.45},
        bbox={
            "boxstyle": "round,pad=0.10",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.86,
        },
        zorder=8,
    )


def make_figure(rows: list[dict[str, Any]]) -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.7,
            "svg.hashsalt": "weight5-pareto-comparison-v2",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, (ax_rate, ax_fom) = plt.subplots(1, 2, figsize=(7.2, 3.72))

    plot_exact_record_curve(
        ax_rate,
        rows,
        flag="rate_distance_front",
        xfield="rate",
        yfield="d",
    )
    ax_rate.set_xlabel(r"Encoding rate $k/n$")
    ax_rate.set_ylabel(r"Distance $d$")
    ax_rate.set_title("(a) Rate--distance Pareto points")
    ax_rate.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax_rate.set_xlim(0.0, 0.33)
    ax_rate.set_ylim(2.0, 26.0)

    plot_exact_record_curve(
        ax_fom,
        rows,
        flag="length_fom_front",
        xfield="n",
        yfield="fom",
    )
    ax_fom.set_xlabel(r"Block length $n$")
    ax_fom.set_ylabel(r"$k d^2/n$")
    ax_fom.set_title(r"(b) Length--FOM Pareto points")
    ax_fom.set_xlim(8, 445)
    ax_fom.set_ylim(0, 33)

    for ax in (ax_rate, ax_fom):
        ax.grid(True, which="major", color="#D0D0D0", linewidth=0.45, alpha=0.7)
        ax.grid(True, which="minor", color="#E8E8E8", linewidth=0.3, alpha=0.5)
        ax.set_axisbelow(True)

    annotate_row(
        ax_rate,
        find_row(
            rows,
            cohort="qec_challenge_w4",
            n=84,
            k=2,
            d=9,
            distance_status="exact",
        ),
        xfield="rate",
        yfield="d",
        text=r"QEC $[[84,2,9]]$",
        offset=(3, 12),
    )
    annotate_row(
        ax_rate,
        find_row(
            rows,
            cohort="qec_challenge_w4",
            n=16,
            k=2,
            d=4,
            distance_status="exact",
        ),
        xfield="rate",
        yfield="d",
        text=r"QEC component $[[16,2,4]]$",
        offset=(8, -17),
    )
    annotate_row(
        ax_rate,
        find_row(
            rows,
            cohort="qec_challenge_w5",
            n=40,
            k=10,
            d=4,
            distance_status="exact",
        ),
        xfield="rate",
        yfield="d",
        text=r"QEC $[[40,10,4]]$",
        offset=(-22, 10),
    )

    annotate_row(
        ax_rate,
        find_row(
            rows,
            cohort="current_component_class_w5",
            n=180,
            k=4,
            d=14,
            distance_status="exact",
        ),
        xfield="rate",
        yfield="d",
        text=r"this work $[[180,4,14]]$",
        offset=(10, -9),
    )
    annotate_row(
        ax_rate,
        find_row(
            rows,
            cohort="current_component_class_w5",
            n=140,
            k=6,
            d=10,
            distance_status="exact",
        ),
        xfield="rate",
        yfield="d",
        text=r"component $[[140,6,10]]$",
        offset=(8, -15),
    )
    annotate_row(
        ax_fom,
        find_row(
            rows,
            cohort="current_component_class_w5",
            n=12,
            k=2,
            d=3,
            distance_status="exact",
        ),
        xfield="n",
        yfield="fom",
        text=r"$[[12,2,3]]$",
        offset=(18, -2),
    )
    annotate_row(
        ax_fom,
        find_row(
            rows,
            cohort="lin_pryadko_w5",
            n=132,
            k=4,
            d=12,
            distance_status="exact",
        ),
        xfield="n",
        yfield="fom",
        text=r"LP $[[132,4,12]]$",
        offset=(9, -18),
    )
    annotate_row(
        ax_fom,
        find_row(
            rows,
            cohort="lin_pryadko_w5",
            n=60,
            k=4,
            d=8,
            distance_status="exact",
        ),
        xfield="n",
        yfield="fom",
        text=r"LP $[[60,4,8]]$",
        offset=(-13, 19),
    )
    handles = [
        Line2D([], [], color=COLORS[4], marker="o", label=r"weight $4$"),
        Line2D([], [], color=COLORS[5], marker="o", label=r"weight $5$"),
        Line2D([], [], color=COLORS[6], marker="o", label=r"weight $6$"),
        Line2D([], [], color=COLORS[8], marker="o", label=r"weight $8$"),
        Line2D(
            [],
            [],
            marker=CURRENT_CSS_MARKER,
            linestyle="none",
            color="#444444",
            markersize=8,
            label="this work: CSS component",
        ),
        Line2D(
            [],
            [],
            marker=CURRENT_PBB_MARKER,
            linestyle="none",
            color="#444444",
            markersize=5.5,
            label="this work: PBB component",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color="#444444",
            label="prior work/literature/Challenge",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=4,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.1,
    )
    fig.text(
        0.5,
        0.165,
        "Exact distances only; uncertain endpoints and unreduced campaign "
        "parents are excluded.\nThe $[[12,2,3]]$ PBB component is the "
        "first available point; shorter lengths are unsampled.",
        ha="center",
        va="center",
        fontsize=6.8,
        color="#444444",
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.92, bottom=0.30, wspace=0.25)

    metadata = {
        "Title": "Weight-five quantum-code Pareto comparison",
        "Author": "qcode-discovery",
        "Subject": (
            "Per-weight exact record curves using component-canonical "
            "weight-five campaign codes."
        ),
        "Creator": "plot_weight5_pareto_comparison.py",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(PDF_PATH, metadata=metadata)
    svg_metadata = {
        "Title": metadata["Title"],
        "Creator": metadata["Creator"],
        "Description": metadata["Subject"],
        "Date": None,
    }
    fig.savefig(SVG_PATH, metadata=svg_metadata)
    fig.savefig(
        PNG_PATH,
        dpi=220,
        metadata={
            "Title": metadata["Title"],
            "Author": metadata["Author"],
            "Description": metadata["Subject"],
            "Software": metadata["Creator"],
        },
    )
    plt.close(fig)


def print_summary(records: list[dict[str, Any]], aggregated: list[dict[str, Any]]) -> None:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in records:
        counts[(row["cohort"], row["distance_status"])] += 1
    for cohort in COHORT_ORDER:
        exact = counts.get((cohort, "exact"), 0)
        interval = counts.get((cohort, "interval"), 0)
        reported = counts.get((cohort, "reported"), 0)
        upper = counts.get((cohort, "upper_bound"), 0)
        print(
            f"{cohort:18s} exact={exact:3d}  interval={interval:3d}  "
            f"reported={reported:3d}  upper_bound={upper:3d}"
        )
    print(f"retained source and component records: {len(records)}")
    print(f"aggregated comparison rows: {len(aggregated)}")
    print(f"exact Pareto points: {sum(row['exact_pareto'] for row in aggregated)}")
    print(
        "upper-bound envelope: "
        f"{sum(row['upper_bound_envelope'] for row in aggregated)}"
    )
    print(
        "decoder-estimated endpoints: "
        f"{sum(row['d_estimate'] is not None for row in aggregated)}"
    )
    for weight in (4, 5, 6, 8):
        rate_count = sum(
            row["rate_distance_front"]
            for row in aggregated
            if row["max_check_weight"] == weight
        )
        fom_count = sum(
            row["length_fom_front"]
            for row in aggregated
            if row["max_check_weight"] == weight
        )
        print(
            f"weight-{weight} exact figure fronts: "
            f"rate--distance={rate_count}, length--FOM={fom_count}"
        )
    print(
        "exact rows eligible before panel-specific frontier reduction: "
        f"{sum(row['figure_exact_eligible'] for row in aggregated)}"
    )
    safety = sum(
        row["multiplicity"]
        for row in aggregated
        if row["cohort"] == "current_pbb_w5"
        and row["campaign_origin"] == "fixed safety net"
    )
    searched = sum(
        row["multiplicity"]
        for row in aggregated
        if row["cohort"] == "current_pbb_w5"
        and row["campaign_origin"] == "search generated"
    )
    print(f"retained PBB origin: fixed safety net={safety}, search generated={searched}")
    print(
        "high nonexact endpoints clipped from both panels: "
        f"{sum(row['plot_clipped'] for row in aggregated)}"
    )
    print(f"wrote {CSV_PATH.relative_to(ROOT)}")
    print(f"wrote {PDF_PATH.relative_to(ROOT)}")
    print(f"wrote {SVG_PATH.relative_to(ROOT)}")
    print(f"wrote {PNG_PATH.relative_to(ROOT)}")


def main() -> None:
    records = load_records()
    aggregated = aggregate_records(records)
    mark_figure_selection(aggregated)
    validate(records, aggregated)
    write_csv(aggregated)
    make_figure(aggregated)
    print_summary(records, aggregated)


if __name__ == "__main__":
    main()
