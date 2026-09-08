#!/usr/bin/env python3
"""Build the normalized publication archive for the weight-five campaigns.

The campaign JSONL files are append-only observation logs.  This script does
not modify them.  It applies the same normalization and evidence rules used by
``generate_weight5_supplement.py``, adds exact distances transferred from the
independently certified connected components, filters presentations whose
best reported upper endpoint is at most two, merges the deterministic CSS
weight-four exclusion and explicit-witness audits, and emits:

* one self-contained JSON object per retained presentation; and
* a manifest that pins every input by SHA-256 and records audit totals.

The direct distance fields describe evidence attached to that presentation.
The class distance fields additionally transfer compatible evidence between
colored-BLISS stored-generator Tanner-isomorphic presentations.

Usage:
    python scripts/build_weight5_publication_catalogue.py
    python scripts/build_weight5_publication_catalogue.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import generate_weight5_supplement as supplement  # noqa: E402


SCHEMA_VERSION = 2
CATALOGUE_SCHEMA = "weight5_publication_catalogue_v2"
MANIFEST_SCHEMA = "weight5_publication_manifest_v2"
DEFAULT_COMPONENT_CERTIFICATES = (
    ROOT / "results" / "weight5_component_certifications.jsonl"
)
DEFAULT_CSS_LOW_WEIGHT_AUDIT = (
    ROOT / "results" / "weight5_css_low_weight_audit.jsonl"
)
DEFAULT_CSS_WEIGHT5_WITNESSES = (
    ROOT / "results" / "weight5_css_weight5_witnesses.jsonl"
)
DEFAULT_CSS_UPPER_BOUND_WITNESSES = (
    ROOT / "results" / "weight5_css_upper_bound_witnesses.jsonl"
)
DEFAULT_CATALOGUE_OUTPUT = ROOT / "results" / "weight5_publication_catalogue.jsonl"
DEFAULT_MANIFEST_OUTPUT = ROOT / "results" / "weight5_publication_manifest.json"

# These are scientific invariants of the submitted data snapshot, not loose
# expectations.  A source edit must be reviewed before these values change.
EXPECTED_PRESENTATION_STATUS = {"E": 398, "C": 744}
EXPECTED_CLASS_STATUS = {"E": 227, "C": 403}
EXPECTED_NONEXACT_PRESENTATION_UPPER_SUPPORT = {
    "estimated": 680,
    "supported": 64,
}
EXPECTED_NONEXACT_CLASS_UPPER_SUPPORT = {
    "estimated": 350,
    "supported": 53,
}
EXPECTED_RETAINED_PRESENTATIONS = 1_142
EXPECTED_RETAINED_CLASSES = 630
EXPECTED_SOURCE_PRESENTATIONS = 1_297
EXPECTED_COMPONENT_CERTIFICATES = 7
EXPECTED_EXPLICIT_BB_MATCHES = 3
EXPECTED_MISSING_CSS_ATTRIBUTION = {
    "CL-929c2a70",
    "CL-d9f65a7e",
    "CL-a25b2baf",
    "CL-4b943aa2",
    "CL-437b6e5f",
    "CL-b9275593",
}

EVIDENCE_METHODS = {
    "B": "BP-OSD or another decoder estimate (upper endpoint; operator not retained)",
    "H": "exhaustive low-weight hash search (exact)",
    "Hc": "exact component enumeration transferred to an isomorphic direct sum",
    "I": "MILP incumbent (upper bound)",
    "L4": "exhaustive exclusion through weight four (lower bound five)",
    "L6": "exhaustive exclusion through weight six (lower bound seven)",
    "M": "MILP optimum (exact)",
    "S": "independent exact solver result",
    "W": "independent witness (upper bound)",
}


@dataclass(frozen=True)
class PublicationBuild:
    """Fully normalized in-memory publication snapshot."""

    all_specs: tuple[supplement.Spec, ...]
    retained_specs: tuple[supplement.Spec, ...]
    classes: tuple[supplement.PresentationClass, ...]
    component_manifest: dict[str, Any]
    component_certificates: dict[str, dict[str, Any]]
    explicit_component_matches: dict[str, tuple[dict[str, Any], ...]]
    css_low_weight_manifest: dict[str, Any]
    css_low_weight_records: dict[str, dict[str, Any]]
    css_weight5_witness_manifest: dict[str, Any]
    css_weight5_witness_records: dict[str, dict[str, Any]]
    css_upper_bound_witness_manifest: dict[str, Any]
    css_upper_bound_witness_records: dict[str, dict[str, Any]]
    raw_rows_by_campaign: dict[str, int]
    verification_rows_by_campaign: dict[str, int]
    verification_recorded_at_by_id: dict[str, tuple[str, ...]]


def _json_line(row: dict[str, Any]) -> str:
    return json.dumps(row, separators=(",", ":"), sort_keys=True)


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _status(lower: int, upper: int, exact: bool) -> str:
    if exact:
        return "E"
    return "C" if lower > 0 else "U"


def _status_name(code: str, upper_is_supported: bool) -> str:
    if code == "E":
        return "exact"
    if code == "C":
        return (
            "certified_interval"
            if upper_is_supported
            else "certified_lower_with_estimated_upper"
        )
    return "supported_upper_only" if upper_is_supported else "estimated_upper_only"


def _bound_evidence(items: Iterable[tuple[int, str]]) -> list[dict[str, Any]]:
    return [
        {"bound": bound, "method": method}
        for bound, method in sorted(set(items), key=lambda item: (item[0], item[1]))
    ]


def _distance_payload(
    *,
    n: int,
    k: int,
    lower: int,
    upper: int,
    exact: bool,
    lower_evidence: Iterable[tuple[int, str]] = (),
    upper_evidence: Iterable[tuple[int, str]] = (),
    exact_methods: Iterable[str] = (),
) -> dict[str, Any]:
    """Serialize bounds without representing an absent lower proof as zero."""
    code = _status(lower, upper, exact)
    materialized_lower: int | None = lower if lower > 0 else None
    lower_items = tuple(lower_evidence)
    upper_items = tuple(upper_evidence)
    exact_method_items = set(exact_methods)
    upper_is_supported = supplement.upper_endpoint_is_rigorous(
        upper_items, upper
    )
    if exact and not upper_is_supported:
        raise ValueError("exact distance lacks rigorous upper-bound evidence")
    if exact and not exact_method_items:
        closing_lower_methods = {
            method for bound, method in lower_items if bound == lower
        }
        closing_upper_methods = {
            method
            for bound, method in upper_items
            if bound == upper and method in supplement.RIGOROUS_UPPER_METHODS
        }
        exact_method_items.update(
            f"{lower_method}+{upper_method}"
            for lower_method in closing_lower_methods
            for upper_method in closing_upper_methods
        )
        if not exact_method_items:
            raise ValueError("exact distance lacks an explicit evidence derivation")
    return {
        "status": _status_name(code, upper_is_supported),
        "status_code": code,
        "lower": materialized_lower,
        "upper": upper,
        "estimated_upper": None if upper_is_supported else upper,
        "is_exact": exact,
        "lower_is_certified": lower > 0,
        "upper_is_supported": upper_is_supported,
        "fom_lower": (
            k * lower * lower / n if materialized_lower is not None else None
        ),
        "fom_upper": k * upper * upper / n if upper_is_supported else None,
        "fom_estimate": (
            None if upper_is_supported else k * upper * upper / n
        ),
        "evidence": {
            "exact_methods": sorted(exact_method_items),
            "lower_bounds": _bound_evidence(lower_items),
            "upper_bounds": _bound_evidence(upper_items),
        },
    }


def _class_distance_differs(
    direct: dict[str, Any], class_distance: dict[str, Any]
) -> bool:
    """Report a numerical tightening or an upgrade to rigorous upper evidence."""
    fields = ("lower", "upper", "is_exact", "upper_is_supported")
    return any(direct[field] != class_distance[field] for field in fields)


def _load_component_certificates(
    path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, tuple[dict[str, Any], ...]],
]:
    if not path.exists():
        raise FileNotFoundError(
            f"missing component-certificate artifact: {_relative(path)}"
        )
    rows = supplement.load_jsonl(path)
    manifests = [row for row in rows if row.get("record_type") == "artifact_manifest"]
    certificates = [
        row for row in rows if row.get("record_type") == "component_distance_certificate"
    ]
    matches = [
        row for row in rows if row.get("record_type") == "explicit_bb_component_match"
    ]
    unknown = [
        row.get("record_type")
        for row in rows
        if row.get("record_type")
        not in {
            "artifact_manifest",
            "component_distance_certificate",
            "explicit_bb_component_match",
        }
    ]
    if unknown:
        raise ValueError(f"unknown component-certificate record types: {unknown}")
    if len(manifests) != 1:
        raise ValueError(
            f"{_relative(path)} must contain exactly one artifact_manifest row"
        )
    if len(certificates) != EXPECTED_COMPONENT_CERTIFICATES:
        raise ValueError(
            f"{_relative(path)} contains {len(certificates)} component certificates; "
            f"expected {EXPECTED_COMPONENT_CERTIFICATES}"
        )
    if len(matches) != EXPECTED_EXPLICIT_BB_MATCHES:
        raise ValueError(
            f"{_relative(path)} contains {len(matches)} explicit BB matches; "
            f"expected {EXPECTED_EXPLICIT_BB_MATCHES}"
        )
    recorded_counts = manifests[0].get("record_counts")
    expected_record_counts = {
        "component_distance_certificate": len(certificates),
        "explicit_bb_component_match": len(matches),
    }
    if recorded_counts != expected_record_counts:
        raise ValueError(
            "component artifact manifest record counts disagree with its contents"
        )

    source_files = manifests[0].get("source_files")
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("component artifact manifest lacks source-file hashes")
    for relative_path, expected_digest in source_files.items():
        source_path = (ROOT / str(relative_path)).resolve()
        try:
            source_path.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise ValueError(
                f"component artifact references a path outside the repository: "
                f"{relative_path!r}"
            ) from exc
        if not source_path.is_file():
            raise ValueError(
                f"component artifact source does not exist: {relative_path!r}"
            )
        actual_digest = _sha256_path(source_path)
        if actual_digest != expected_digest:
            raise ValueError(
                f"component artifact source hash mismatch for {relative_path}: "
                f"{actual_digest} != {expected_digest}"
            )

    by_id: dict[str, dict[str, Any]] = {}
    for row in certificates:
        identity = row.get("source_identity")
        if not isinstance(identity, dict):
            raise ValueError("component certificate lacks source_identity")
        spec_id = str(identity.get("presentation_id") or "")
        if not spec_id:
            raise ValueError("component certificate lacks a presentation_id")
        if spec_id in by_id:
            raise ValueError(f"duplicate component certificate for {spec_id}")
        by_id[spec_id] = row
    matches_by_parent: dict[str, list[dict[str, Any]]] = {}
    seen_match_ids: set[str] = set()
    for row in matches:
        match_id = str(row.get("match_id") or "")
        identity = row.get("parent_source_identity")
        if not match_id or match_id in seen_match_ids:
            raise ValueError(f"missing or duplicate explicit BB match ID: {match_id!r}")
        if not isinstance(identity, dict) or not identity.get("presentation_id"):
            raise ValueError(f"{match_id}: missing parent presentation identity")
        seen_match_ids.add(match_id)
        parent_id = str(identity["presentation_id"])
        matches_by_parent.setdefault(parent_id, []).append(row)
    normalized_matches = {
        parent_id: tuple(sorted(parent_matches, key=lambda item: item["match_id"]))
        for parent_id, parent_matches in matches_by_parent.items()
    }
    return manifests[0], by_id, normalized_matches


def _validate_and_merge_component_certificate(
    spec: supplement.Spec, row: dict[str, Any]
) -> None:
    identity = row["source_identity"]
    if identity.get("campaign") != spec.campaign.slug:
        raise ValueError(
            f"{spec.spec_id}: component certificate campaign "
            f"{identity.get('campaign')!r} disagrees with {spec.campaign.slug!r}"
        )
    if identity.get("raw_path") != _relative(spec.campaign.raw_path):
        raise ValueError(f"{spec.spec_id}: component certificate raw path disagrees")
    parent = row.get("parent_spec")
    decomposition = row.get("decomposition")
    component = row.get("component_certificate")
    transfer = row.get("distance_transfer")
    if not all(isinstance(value, dict) for value in (parent, decomposition, component, transfer)):
        raise ValueError(f"{spec.spec_id}: malformed component certificate")

    assert isinstance(parent, dict)
    assert isinstance(decomposition, dict)
    assert isinstance(component, dict)
    assert isinstance(transfer, dict)
    expected_parent = {
        "ell": spec.ell,
        "m": spec.m,
        "A_terms": [list(term) for term in spec.A],
        "B_terms": [list(term) for term in spec.B],
        "n": spec.n,
        "k": spec.k,
    }
    for key, expected in expected_parent.items():
        actual = parent.get(key)
        if key in {"A_terms", "B_terms"}:
            actual = [list(term) for term in supplement.normalize_terms(
                actual, f"{spec.spec_id}: component {key}"
            )]
        if actual != expected:
            raise ValueError(
                f"{spec.spec_id}: component certificate {key}={actual!r}, "
                f"expected {expected!r}"
            )

    info = supplement.connectivity_info(spec)
    if info.is_connected or not info.homogeneous or info.components_isomorphic is not True:
        raise ValueError(
            f"{spec.spec_id}: exact distance cannot be transferred from the "
            "claimed component decomposition"
        )
    checks = {
        "n_components": info.n_components,
        "components_bliss_isomorphic": True,
        "k_additivity_verified": True,
        "rowspace_partition_equals_presentation_partition": True,
    }
    for key, expected in checks.items():
        if decomposition.get(key) != expected:
            raise ValueError(
                f"{spec.spec_id}: decomposition {key}={decomposition.get(key)!r}, "
                f"expected {expected!r}"
            )
    sizes = decomposition.get("component_sizes")
    if sizes != [info.base_n] * info.n_components:
        raise ValueError(f"{spec.spec_id}: component sizes disagree with audit")
    if decomposition.get("component_dimensions") != [info.base_k] * info.n_components:
        raise ValueError(f"{spec.spec_id}: component dimensions disagree with audit")
    if transfer.get("valid") is not True:
        raise ValueError(f"{spec.spec_id}: component distance transfer is not valid")

    distances = {
        supplement.positive_int(parent.get("d_exact"), f"{spec.spec_id}: parent d"),
        supplement.positive_int(
            component.get("d_exact"), f"{spec.spec_id}: component d"
        ),
        supplement.positive_int(
            transfer.get("parent_d_exact"), f"{spec.spec_id}: transferred d"
        ),
    }
    if len(distances) != 1:
        raise ValueError(
            f"{spec.spec_id}: component and parent exact distances disagree: "
            f"{sorted(distances)}"
        )
    component_n = supplement.positive_int(
        component.get("n"), f"{spec.spec_id}: component n"
    )
    component_k = supplement.positive_int(
        component.get("k"), f"{spec.spec_id}: component k"
    )
    if (component_n, component_k) != (info.base_n, info.base_k):
        raise ValueError(
            f"{spec.spec_id}: certified component [[{component_n},{component_k}]] "
            f"disagrees with audited [[{info.base_n},{info.base_k}]]"
        )
    d_exact = next(iter(distances))
    d_x = supplement.positive_int(
        component.get("d_x_exact"), f"{spec.spec_id}: component X distance"
    )
    d_z = supplement.positive_int(
        component.get("d_z_exact"), f"{spec.spec_id}: component Z distance"
    )
    if min(d_x, d_z) != d_exact:
        raise ValueError(
            f"{spec.spec_id}: component d={d_exact} disagrees with "
            f"min(d_X,d_Z)={min(d_x, d_z)}"
        )
    if (
        decomposition.get("component_presentation_canonical_digest_sha256")
        != component.get("canonical_digest_sha256")
        or decomposition.get("component_rowspace_digest_sha256")
        != component.get("rowspace_digest_sha256")
    ):
        raise ValueError(f"{spec.spec_id}: component digests disagree internally")
    certificate_id = str(row.get("certificate_id") or "")
    if not certificate_id:
        raise ValueError(f"{spec.spec_id}: component certificate lacks an ID")
    if certificate_id in spec.component_certificate_ids:
        raise ValueError(f"{spec.spec_id}: duplicate component certificate ID")
    spec.component_certificate_ids.add(certificate_id)
    spec.add_exact(d_exact, "Hc")


def _validate_explicit_component_match(
    spec: supplement.Spec, row: dict[str, Any]
) -> None:
    match_id = str(row.get("match_id") or "")
    parent = row.get("parent_spec")
    component = row.get("explicit_bb_component")
    match = row.get("match")
    if not all(isinstance(value, dict) for value in (parent, component, match)):
        raise ValueError(f"{match_id}: malformed explicit BB component match")
    assert isinstance(parent, dict)
    assert isinstance(component, dict)
    assert isinstance(match, dict)
    identity = row.get("parent_source_identity")
    if not isinstance(identity, dict):
        raise ValueError(f"{match_id}: missing parent source identity")
    if identity.get("campaign") != spec.campaign.slug:
        raise ValueError(f"{match_id}: parent campaign disagrees with catalogue")
    if identity.get("raw_path") != _relative(spec.campaign.raw_path):
        raise ValueError(f"{match_id}: parent raw path disagrees with catalogue")
    info = supplement.connectivity_info(spec)
    if info.is_connected:
        raise ValueError(f"{match_id}: parent presentation is connected")
    expected_parent = {
        "ell": spec.ell,
        "m": spec.m,
        "n": spec.n,
        "k": spec.k,
        "n_components": info.n_components,
    }
    for key, expected in expected_parent.items():
        if parent.get(key) != expected:
            raise ValueError(f"{match_id}: parent {key} disagrees with catalogue")
    for key, expected in {
        "A_terms": spec.A,
        "B_terms": spec.B,
    }.items():
        actual = supplement.normalize_terms(
            parent.get(key), f"{match_id}: parent {key}"
        )
        if actual != expected:
            raise ValueError(f"{match_id}: parent {key} disagrees with catalogue")
    direct_lower, direct_upper, direct_exact = spec.final_bounds()
    if not direct_exact or direct_lower != direct_upper:
        raise ValueError(f"{match_id}: parent does not have an exact distance")
    if parent.get("d_exact") != direct_upper or component.get("d_exact") != direct_upper:
        raise ValueError(f"{match_id}: parent/component exact distance disagrees")
    if (component.get("n"), component.get("k")) != (info.base_n, info.base_k):
        raise ValueError(f"{match_id}: explicit BB component has wrong dimensions")
    if component.get("connected") is not True or component.get("stabilizer_weight") != 5:
        raise ValueError(f"{match_id}: explicit component is not connected weight five")
    if (
        match.get("bliss_isomorphic") is not True
        or match.get("distance_transfer_valid") is not True
        or match.get("equivalence_relation") != supplement.CLASS_EQUIVALENCE_RELATION
    ):
        raise ValueError(f"{match_id}: explicit component match is not certified")

    ell = supplement.positive_int(component.get("ell"), f"{match_id}: ell")
    m = supplement.positive_int(component.get("m"), f"{match_id}: m")
    A = supplement.normalize_terms(component.get("A_terms"), f"{match_id}: A_terms")
    B = supplement.normalize_terms(component.get("B_terms"), f"{match_id}: B_terms")
    if len(A) + len(B) != 5:
        raise ValueError(f"{match_id}: explicit BB presentation is not weight five")
    code = supplement.build_bb_code(ell, m, list(A), list(B))
    if (int(code.num_qudits), int(code.dimension)) != (
        component.get("n"),
        component.get("k"),
    ):
        raise ValueError(f"{match_id}: rebuilt explicit BB dimensions disagree")
    if not supplement.decompose(code).is_connected:
        raise ValueError(f"{match_id}: rebuilt explicit BB component is disconnected")
    digest = supplement.canonical_signature_digest(supplement.canonical_hash(code))
    if digest != match.get("canonical_digest_sha256"):
        raise ValueError(f"{match_id}: rebuilt explicit BB digest disagrees")


def _normalized_timestamp(value: object, context: str) -> str:
    """Return an explicit UTC ISO-8601 timestamp from source metadata."""
    if isinstance(value, bool):
        raise ValueError(f"{context}: invalid timestamp {value!r}")
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{context}: invalid timestamp {value!r}") from exc
        if parsed.tzinfo is None:
            raise ValueError(f"{context}: timestamp lacks a timezone: {value!r}")
        return parsed.astimezone(timezone.utc).isoformat()
    raise ValueError(f"{context}: invalid timestamp {value!r}")


def _verification_recorded_times(
    campaign: supplement.Campaign,
    specs_by_key: dict[tuple, supplement.Spec],
) -> dict[str, tuple[str, ...]]:
    """Materialize verifier timestamps without upgrading their semantics."""
    completed: dict[str, list[str]] = {}
    for line_number, row in enumerate(
        supplement.load_jsonl(campaign.verification_path), 1
    ):
        key = supplement.key_for(campaign, row)
        if key not in specs_by_key:
            raise ValueError(
                f"{campaign.slug}: verification timestamp row has no specification"
            )
        raw_timestamp = row.get("verified_at")
        if raw_timestamp is None:
            raw_timestamp = row.get("_saved_at")
        if raw_timestamp is None:
            raise ValueError(
                f"{_relative(campaign.verification_path)}:{line_number} lacks a "
                "verifier timestamp"
            )
        spec_id = specs_by_key[key].spec_id
        completed.setdefault(spec_id, []).append(
            _normalized_timestamp(
                raw_timestamp,
                f"{_relative(campaign.verification_path)}:{line_number}",
            )
        )
    return {spec_id: tuple(sorted(values)) for spec_id, values in completed.items()}


def build_publication_snapshot(
    component_path: Path,
    css_low_weight_path: Path = DEFAULT_CSS_LOW_WEIGHT_AUDIT,
    css_weight5_witness_path: Path = DEFAULT_CSS_WEIGHT5_WITNESSES,
    css_upper_bound_witness_path: Path = (
        DEFAULT_CSS_UPPER_BOUND_WITNESSES
    ),
) -> PublicationBuild:
    """Load raw observations and materialize the reviewed publication state."""
    (
        component_manifest,
        component_certificates,
        explicit_component_matches,
    ) = _load_component_certificates(component_path)
    specs_by_id: dict[str, supplement.Spec] = {}
    all_specs: list[supplement.Spec] = []
    raw_rows_by_campaign: dict[str, int] = {}
    verification_rows_by_campaign: dict[str, int] = {}
    verification_recorded_at_by_id: dict[str, tuple[str, ...]] = {}

    for campaign in supplement.CAMPAIGNS:
        specs_by_key, raw_rows = supplement.load_specs(campaign)
        event_rows = supplement.merge_events(campaign, specs_by_key)
        attribution_rows = supplement.merge_historical_attribution(
            campaign, specs_by_key
        )
        if event_rows and attribution_rows:
            raise ValueError(f"{campaign.slug}: mixed native and recovered provenance")
        verification_rows = supplement.merge_verification(campaign, specs_by_key)
        recorded_times = _verification_recorded_times(campaign, specs_by_key)
        overlap = set(verification_recorded_at_by_id) & set(recorded_times)
        if overlap:
            raise ValueError(
                f"verification timestamps cross campaign boundaries: {sorted(overlap)}"
            )
        verification_recorded_at_by_id.update(recorded_times)
        raw_rows_by_campaign[campaign.slug] = raw_rows
        verification_rows_by_campaign[campaign.slug] = verification_rows
        for spec in specs_by_key.values():
            spec.final_bounds()
            if spec.spec_id in specs_by_id:
                raise ValueError(f"truncated specification-ID collision: {spec.spec_id}")
            specs_by_id[spec.spec_id] = spec
            all_specs.append(spec)

    if len(all_specs) != EXPECTED_SOURCE_PRESENTATIONS:
        raise ValueError(
            f"source catalogue has {len(all_specs)} specifications; "
            f"expected {EXPECTED_SOURCE_PRESENTATIONS}"
        )
    unknown_certificates = set(component_certificates) - set(specs_by_id)
    if unknown_certificates:
        raise ValueError(
            "component certificates have no source presentation: "
            f"{sorted(unknown_certificates)}"
        )
    for spec_id, row in component_certificates.items():
        _validate_and_merge_component_certificate(specs_by_id[spec_id], row)
    unknown_matches = set(explicit_component_matches) - set(specs_by_id)
    if unknown_matches:
        raise ValueError(
            f"explicit BB matches have no source presentation: {sorted(unknown_matches)}"
        )
    for spec_id, rows in explicit_component_matches.items():
        for row in rows:
            _validate_explicit_component_match(specs_by_id[spec_id], row)

    css_low_weight_manifest, css_low_weight_records = (
        supplement.load_and_merge_css_low_weight_audit(
            specs_by_id, css_low_weight_path
        )
    )
    (
        css_weight5_witness_manifest,
        css_weight5_witness_records,
    ) = supplement.load_and_merge_css_weight5_witnesses(
        specs_by_id, css_weight5_witness_path
    )
    (
        css_upper_bound_witness_manifest,
        css_upper_bound_witness_records,
    ) = supplement.load_and_merge_css_upper_bound_witnesses(
        specs_by_id, css_upper_bound_witness_path
    )

    all_specs.sort(key=supplement.sort_key)
    retained_specs = [
        spec for spec in all_specs if supplement.retain_for_catalogue(spec)
    ]
    classes = supplement.build_presentation_classes(retained_specs)
    build = PublicationBuild(
        all_specs=tuple(all_specs),
        retained_specs=tuple(retained_specs),
        classes=tuple(classes),
        component_manifest=component_manifest,
        component_certificates=component_certificates,
        explicit_component_matches=explicit_component_matches,
        css_low_weight_manifest=css_low_weight_manifest,
        css_low_weight_records=css_low_weight_records,
        css_weight5_witness_manifest=css_weight5_witness_manifest,
        css_weight5_witness_records=css_weight5_witness_records,
        css_upper_bound_witness_manifest=(
            css_upper_bound_witness_manifest
        ),
        css_upper_bound_witness_records=(
            css_upper_bound_witness_records
        ),
        raw_rows_by_campaign=raw_rows_by_campaign,
        verification_rows_by_campaign=verification_rows_by_campaign,
        verification_recorded_at_by_id=verification_recorded_at_by_id,
    )
    _validate_snapshot_totals(build)
    return build


def _validate_snapshot_totals(build: PublicationBuild) -> None:
    direct = Counter(supplement.status_for(spec) for spec in build.retained_specs)
    class_status = Counter(
        supplement.presentation_class_status(group) for group in build.classes
    )
    direct_support = Counter(
        (
            "supported"
            if supplement.upper_endpoint_is_rigorous(
                spec.upper_evidence, spec.final_bounds()[1]
            )
            else "estimated"
        )
        for spec in build.retained_specs
        if not spec.final_bounds()[2]
    )
    class_support = Counter(
        "supported" if group.upper_is_supported else "estimated"
        for group in build.classes
        if not group.exact
    )
    if len(build.retained_specs) != EXPECTED_RETAINED_PRESENTATIONS:
        raise ValueError(
            f"retained {len(build.retained_specs)} presentations; "
            f"expected {EXPECTED_RETAINED_PRESENTATIONS}"
        )
    if len(build.classes) != EXPECTED_RETAINED_CLASSES:
        raise ValueError(
            f"found {len(build.classes)} presentation classes; "
            f"expected {EXPECTED_RETAINED_CLASSES}"
        )
    if dict(direct) != EXPECTED_PRESENTATION_STATUS:
        raise ValueError(
            f"unexpected direct-distance totals: {dict(direct)}; "
            f"expected {EXPECTED_PRESENTATION_STATUS}"
        )
    if dict(class_status) != EXPECTED_CLASS_STATUS:
        raise ValueError(
            f"unexpected class-distance totals: {dict(class_status)}; "
            f"expected {EXPECTED_CLASS_STATUS}"
        )
    if dict(direct_support) != EXPECTED_NONEXACT_PRESENTATION_UPPER_SUPPORT:
        raise ValueError(
            f"unexpected direct upper-support totals: {dict(direct_support)}; "
            f"expected {EXPECTED_NONEXACT_PRESENTATION_UPPER_SUPPORT}"
        )
    if dict(class_support) != EXPECTED_NONEXACT_CLASS_UPPER_SUPPORT:
        raise ValueError(
            f"unexpected class upper-support totals: {dict(class_support)}; "
            f"expected {EXPECTED_NONEXACT_CLASS_UPPER_SUPPORT}"
        )
    if any(spec.final_bounds()[1] <= 2 for spec in build.retained_specs):
        raise ValueError("publication catalogue retained a distance endpoint <= 2")

    missing_css = {
        spec.spec_id
        for spec in build.retained_specs
        if spec.campaign.family == "CSS" and not spec.model_aliases
    }
    if missing_css != EXPECTED_MISSING_CSS_ATTRIBUTION:
        raise ValueError(
            f"unexpected retained CSS attribution gaps: {sorted(missing_css)}"
        )


def _source_paths(
    component_path: Path,
    css_low_weight_path: Path,
    css_weight5_witness_path: Path = DEFAULT_CSS_WEIGHT5_WITNESSES,
    css_upper_bound_witness_path: Path = (
        DEFAULT_CSS_UPPER_BOUND_WITNESSES
    ),
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for campaign in supplement.CAMPAIGNS:
        paths[f"{campaign.slug}:raw_log"] = campaign.raw_path
        if campaign.event_path is not None:
            paths[f"{campaign.slug}:discovery_events"] = campaign.event_path
        if campaign.attribution_path is not None:
            paths[f"{campaign.slug}:attribution_sidecar"] = campaign.attribution_path
        paths[f"{campaign.slug}:distance_verification"] = campaign.verification_path
    paths["all:component_certifications"] = component_path
    paths["all:css_low_weight_audit"] = css_low_weight_path
    paths["all:css_weight5_witnesses"] = css_weight5_witness_path
    paths["all:css_upper_bound_witnesses"] = (
        css_upper_bound_witness_path
    )
    return paths


def _software_paths() -> dict[str, Path]:
    return {
        "publication_catalogue_generator": Path(__file__).resolve(),
        "source_normalizer": ROOT / "scripts" / "generate_weight5_supplement.py",
        "component_certificate_generator": (
            ROOT / "scripts" / "certify_weight5_components.py"
        ),
        "css_low_weight_audit_generator": (
            ROOT / "scripts" / "audit_weight5_css_low_weight.py"
        ),
        "css_weight5_witness_generator": (
            ROOT / "scripts" / "audit_weight5_css_weight5_witnesses.py"
        ),
        "css_upper_bound_witness_generator": (
            ROOT / "scripts" / "audit_weight5_css_upper_bound_witnesses.py"
        ),
        "dependency_manifest": ROOT / "pyproject.toml",
        "dependency_lock": ROOT / "uv.lock",
        "bb_constructor": ROOT / "evaluation" / "bb_code.py",
        "pbb_constructor": ROOT / "evaluation" / "pbb_code.py",
        "connectivity_audit": ROOT / "evaluation" / "connectivity.py",
        "tanner_canonicalizer": ROOT / "evaluation" / "tanner_equivalence.py",
        "fixed_pbb_safety_net": ROOT / "evolve" / "seed_solution_weight5_pbb.py",
    }


def _artifact_descriptor(path: Path, *, jsonl: bool = False) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "path": _relative(path),
        "sha256": _sha256_path(path),
        "bytes": path.stat().st_size,
    }
    if jsonl:
        descriptor["records"] = sum(
            bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines()
        )
    return descriptor


def _campaign_sources(
    campaign: supplement.Campaign,
    source_descriptors: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    keys = [
        ("raw_log", f"{campaign.slug}:raw_log"),
        ("distance_verification", f"{campaign.slug}:distance_verification"),
    ]
    if campaign.event_path is not None:
        keys.append(("discovery_events", f"{campaign.slug}:discovery_events"))
    if campaign.attribution_path is not None:
        keys.append(("attribution_sidecar", f"{campaign.slug}:attribution_sidecar"))
    return {role: source_descriptors[key] for role, key in keys}


def _normalized_aliases(values: Iterable[str | None]) -> list[str]:
    return sorted("seed/null" if value is None else str(value) for value in values)


def _record_for_spec(
    spec: supplement.Spec,
    group: supplement.PresentationClass,
    class_label: str,
    source_descriptors: dict[str, dict[str, Any]],
    component_certificates: dict[str, dict[str, Any]],
    explicit_component_matches: dict[str, tuple[dict[str, Any], ...]],
    css_low_weight_records: dict[str, dict[str, Any]],
    css_weight5_witness_records: dict[str, dict[str, Any]],
    css_upper_bound_witness_records: dict[str, dict[str, Any]],
    verification_recorded_at: tuple[str, ...],
) -> dict[str, Any]:
    direct_lower, direct_upper, direct_exact = spec.final_bounds()
    direct = _distance_payload(
        n=spec.n,
        k=spec.k,
        lower=direct_lower,
        upper=direct_upper,
        exact=direct_exact,
        lower_evidence=spec.lower_evidence,
        upper_evidence=spec.upper_evidence,
        exact_methods=spec.proof_sources,
    )
    exact_donors = [
        member.spec_id for member in group.members if member.final_bounds()[2]
    ]
    decisive_lower_evidence = sorted(
        {
            (member.spec_id, bound, method)
            for member in group.members
            for bound, method in member.lower_evidence
            if bound == group.lower
        }
    )
    decisive_upper_evidence = sorted(
        {
            (member.spec_id, bound, method)
            for member in group.members
            for bound, method in member.upper_evidence
            if bound == group.upper
        }
    )
    class_distance = _distance_payload(
        n=spec.n,
        k=spec.k,
        lower=group.lower,
        upper=group.upper,
        exact=group.exact,
        lower_evidence=(
            (bound, method)
            for _, bound, method in decisive_lower_evidence
        ),
        upper_evidence=(
            (bound, method)
            for _, bound, method in decisive_upper_evidence
        ),
        exact_methods=(
            method
            for member in group.members
            for method in member.proof_sources
        ),
    )
    class_distance.update(
        {
            "inherited_or_tightened": _class_distance_differs(
                direct, class_distance
            ),
            "exact_evidence_member_ids": exact_donors,
            "evidence": {
                "exact_methods": class_distance["evidence"]["exact_methods"],
                "decisive_lower_bounds": [
                    {
                        "presentation_id": presentation_id,
                        "bound": bound,
                        "method": method,
                    }
                    for presentation_id, bound, method in decisive_lower_evidence
                ],
                "decisive_upper_bounds": [
                    {
                        "presentation_id": presentation_id,
                        "bound": bound,
                        "method": method,
                    }
                    for presentation_id, bound, method in decisive_upper_evidence
                ],
            },
        }
    )

    info = supplement.connectivity_info(spec)
    component_code = None
    if not info.is_connected:
        component_code = {
            "n": info.base_n,
            "k": info.base_k,
            "multiplicity": info.n_components,
            "distance_equals_parent": info.components_isomorphic is True,
        }

    fixed_safety_net = spec.fixed_safety_net
    aliases = _normalized_aliases(spec.model_aliases)
    possible_aliases = _normalized_aliases(
        set(spec.possible_model_aliases) - set(spec.model_aliases)
    )
    missing_reason = None
    if spec.campaign.family == "CSS" and not spec.model_aliases:
        missing_reason = "excluded_by_credibility_event_gate"
    if spec.attribution_status is not None:
        attribution_status = spec.attribution_status
    elif spec.event_count:
        attribution_status = "native_discovery_events"
    else:
        attribution_status = "unavailable"

    record_sources = _campaign_sources(spec.campaign, source_descriptors)
    class_component_donors = [
        member_id
        for member_id in exact_donors
        if member_id in component_certificates
    ]
    class_explicit_matches = [
        member.spec_id
        for member in group.members
        if member.spec_id in explicit_component_matches
    ]
    if (
        spec.spec_id in component_certificates
        or class_component_donors
        or class_explicit_matches
    ):
        record_sources["component_certifications"] = source_descriptors[
            "all:component_certifications"
        ]
    if fixed_safety_net:
        record_sources["fixed_safety_net_definition"] = source_descriptors[
            "software:fixed_pbb_safety_net"
        ]

    css_low_weight_record = css_low_weight_records.get(spec.spec_id)
    if css_low_weight_record is not None:
        record_sources["css_low_weight_audit"] = source_descriptors[
            "all:css_low_weight_audit"
        ]
        css_low_weight_audit = {
            "algorithm": css_low_weight_record["algorithm"],
            "audited_class_id": css_low_weight_record["class_identity"]["class_id"],
            "representative_id": css_low_weight_record["representative"][
                "presentation_id"
            ],
            "transferred_by_tanner_isomorphism": (
                spec.spec_id
                != css_low_weight_record["representative"]["presentation_id"]
            ),
            "result": css_low_weight_record["audit_evidence"]["result"],
            "exact_distance": css_low_weight_record["audit_evidence"][
                "exact_distance"
            ],
            "certified_lower_bound": css_low_weight_record["audit_evidence"][
                "certified_lower_bound"
            ],
        }
    else:
        css_low_weight_audit = None

    css_weight5_witness_record = css_weight5_witness_records.get(spec.spec_id)
    if css_weight5_witness_record is not None:
        record_sources["css_weight5_witnesses"] = source_descriptors[
            "all:css_weight5_witnesses"
        ]
        css_weight5_witness_certification = {
            "algorithm": css_weight5_witness_record["algorithm"],
            "historical_upper_bound": css_weight5_witness_record[
                "historical_upper_bound"
            ],
            "certified_exact_distance": 5,
            "sector_weights": {
                sector: css_weight5_witness_record[
                    "replacement_explicit_witnesses"
                ][sector]["weight"]
                for sector in ("X", "Z")
            },
        }
    else:
        css_weight5_witness_certification = None

    css_upper_bound_witness_record = (
        css_upper_bound_witness_records.get(spec.spec_id)
    )
    if css_upper_bound_witness_record is not None:
        record_sources["css_upper_bound_witnesses"] = source_descriptors[
            "all:css_upper_bound_witnesses"
        ]
        css_upper_bound_witness_correction = {
            "algorithm": css_upper_bound_witness_record["algorithm"],
            "historical_upper_bound": css_upper_bound_witness_record[
                "historical_upper_bound"
            ],
            "corrected_upper_bound": css_upper_bound_witness_record[
                "corrected_upper_bound"
            ],
            "sector_weights": {
                sector: css_upper_bound_witness_record[
                    "replacement_explicit_witnesses"
                ][sector]["weight"]
                for sector in ("X", "Z")
            },
        }
    else:
        css_upper_bound_witness_correction = None

    component_certificate = component_certificates.get(spec.spec_id)
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "weight5_presentation",
        "presentation_id": spec.spec_id,
        "family": spec.campaign.family,
        "campaign": spec.campaign.slug,
        "parameters": {
            "n": spec.n,
            "k": spec.k,
            "ell": spec.ell,
            "m": spec.m,
        },
        "generators": {
            "A_terms": [list(term) for term in spec.A],
            "B_terms": [list(term) for term in spec.B],
            "C_terms": [list(term) for term in spec.C],
            "D_terms": [list(term) for term in spec.D],
        },
        "distance_direct": direct,
        "distance_class": class_distance,
        "equivalence": {
            "relation": supplement.CLASS_EQUIVALENCE_RELATION,
            "canonicalizer": {
                "name": "python-igraph/BLISS",
                "version": supplement.distribution_version("igraph"),
            },
            "class_id": group.class_id,
            "class_label": class_label,
            "canonical_digest_sha256": group.digest,
            "class_size": len(group.members),
            "member_ids": [member.spec_id for member in group.members],
        },
        "connectivity": {
            "state": "connected" if info.is_connected else "disconnected",
            "is_connected": info.is_connected,
            "n_components": info.n_components,
            "homogeneous": info.homogeneous,
            "components_bliss_isomorphic": info.components_isomorphic,
            "presentation_partition_agrees": info.presentation_agrees,
            "component_code": component_code,
        },
        "provenance": {
            "observation_count": spec.observations,
            "verification_row_count": spec.verification_rows,
            "verification_recorded_at": list(verification_recorded_at),
            "verification_recorded_at_latest": (
                verification_recorded_at[-1] if verification_recorded_at else None
            ),
            "event_count": spec.event_count,
            "attribution_status": attribution_status,
            "missing_attribution_reason": missing_reason,
            "model_aliases": aliases,
            "possible_additional_model_aliases": possible_aliases,
            "program_ids": sorted(spec.program_ids),
            "program_source_hashes": sorted(spec.source_hashes),
            "origin": {
                "fixed_safety_net": fixed_safety_net,
                "model_attributed_observations_present": any(
                    alias != "seed/null" for alias in aliases
                ),
            },
        },
        "component_certification": component_certificate,
        "css_low_weight_audit": css_low_weight_audit,
        "css_weight5_witness_certification": (
            css_weight5_witness_certification
        ),
        "css_upper_bound_witness_correction": (
            css_upper_bound_witness_correction
        ),
        "explicit_bb_component_matches": list(
            explicit_component_matches.get(spec.spec_id, ())
        ),
        "source_artifacts": record_sources,
    }


def render_catalogue(
    build: PublicationBuild, source_descriptors: dict[str, dict[str, Any]]
) -> str:
    class_of = {
        spec.spec_id: group for group in build.classes for spec in group.members
    }
    class_labels = supplement.presentation_class_labels(list(build.classes))
    records = [
        _record_for_spec(
            spec,
            class_of[spec.spec_id],
            class_labels[class_of[spec.spec_id].class_id],
            source_descriptors,
            build.component_certificates,
            build.explicit_component_matches,
            build.css_low_weight_records,
            build.css_weight5_witness_records,
            build.css_upper_bound_witness_records,
            build.verification_recorded_at_by_id.get(spec.spec_id, ()),
        )
        for spec in build.retained_specs
    ]
    records.sort(
        key=lambda row: (
            row["family"],
            row["parameters"]["n"],
            -row["parameters"]["k"],
            row["presentation_id"],
        )
    )
    return "".join(_json_line(row) + "\n" for row in records)


def _counter_dict(values: Iterable[str]) -> dict[str, int]:
    counts = Counter(values)
    return {key: counts[key] for key in sorted(counts)}


def _nested_status_counts(
    specs: Iterable[supplement.Spec],
) -> dict[str, dict[str, int]]:
    nested: dict[str, Counter[str]] = {}
    for spec in specs:
        nested.setdefault(spec.campaign.slug, Counter())[supplement.status_for(spec)] += 1
    return {
        campaign: {code: counts[code] for code in ("E", "C", "U")}
        for campaign, counts in sorted(nested.items())
    }


def render_manifest(
    build: PublicationBuild,
    catalogue_payload: str,
    catalogue_path: Path,
    source_descriptors: dict[str, dict[str, Any]],
) -> str:
    direct_status = _counter_dict(
        supplement.status_for(spec) for spec in build.retained_specs
    )
    class_status = _counter_dict(
        supplement.presentation_class_status(group) for group in build.classes
    )
    nonexact_direct_support = _counter_dict(
        (
            "supported"
            if supplement.upper_endpoint_is_rigorous(
                spec.upper_evidence, spec.final_bounds()[1]
            )
            else "estimated"
        )
        for spec in build.retained_specs
        if not spec.final_bounds()[2]
    )
    nonexact_class_support = _counter_dict(
        "supported" if group.upper_is_supported else "estimated"
        for group in build.classes
        if not group.exact
    )
    excluded = [
        spec for spec in build.all_specs if not supplement.retain_for_catalogue(spec)
    ]
    disconnected_specs = sum(
        not supplement.connectivity_info(spec).is_connected
        for spec in build.retained_specs
    )
    disconnected_classes = sum(
        group.connectivity == "disconnected" for group in build.classes
    )
    component_exact_retained = sorted(
        set(build.component_certificates)
        & {spec.spec_id for spec in build.retained_specs}
    )
    component_exact_excluded = sorted(
        set(build.component_certificates)
        - {spec.spec_id for spec in build.retained_specs}
    )
    retained_ids = {spec.spec_id for spec in build.retained_specs}
    low_weight_retained = sorted(set(build.css_low_weight_records) & retained_ids)
    low_weight_excluded = sorted(set(build.css_low_weight_records) - retained_ids)
    weight5_witness_retained = sorted(
        set(build.css_weight5_witness_records) & retained_ids
    )
    weight5_witness_excluded = sorted(
        set(build.css_weight5_witness_records) - retained_ids
    )
    upper_bound_witness_retained = sorted(
        set(build.css_upper_bound_witness_records) & retained_ids
    )
    upper_bound_witness_excluded = sorted(
        set(build.css_upper_bound_witness_records) - retained_ids
    )
    catalogue_bytes = catalogue_payload.encode("utf-8")
    recorded_times_by_campaign: dict[str, list[str]] = {}
    campaign_of = {spec.spec_id: spec.campaign.slug for spec in build.all_specs}
    for spec_id, timestamps in build.verification_recorded_at_by_id.items():
        recorded_times_by_campaign.setdefault(campaign_of[spec_id], []).extend(
            timestamps
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": MANIFEST_SCHEMA,
        "catalogue_schema": CATALOGUE_SCHEMA,
        "producer": {
            "path": _relative(Path(__file__)),
            "sha256": _sha256_path(Path(__file__)),
        },
        "normalization": {
            "presentation_key": [
                "campaign",
                "ell",
                "m",
                "sorted A_terms",
                "sorted B_terms",
                "sorted C_terms",
                "sorted D_terms",
            ],
            "equivalence_relation": supplement.CLASS_EQUIVALENCE_RELATION,
            "canonicalizer": {
                "name": "python-igraph/BLISS",
                "version": supplement.distribution_version("igraph"),
            },
            "distance_filter": {
                "rule": "retain iff best reported distance endpoint >= 3",
                "warning": (
                    "For U records this rule does not certify d>=3; their true "
                    "distance may be one or two."
                ),
            },
            "raw_inputs_modified": False,
        },
        "distance_semantics": {
            "E": "exact distance",
            "C": (
                "certified positive lower bound plus a reported endpoint; "
                "upper_is_supported distinguishes rigorous upper evidence "
                "from a decoder estimate"
            ),
            "U": (
                "reported endpoint only; no positive lower bound is certified; "
                "upper_is_supported distinguishes rigorous upper evidence "
                "from a decoder estimate"
            ),
            "upper_is_supported": (
                "true exactly when the decisive upper endpoint has H, Hc, I, "
                "M, S, or W evidence; false for a B-only endpoint"
            ),
            "estimated_upper_and_fom": (
                "For a B-only decisive endpoint, upper retains the reported "
                "numeric estimate for compatibility, estimated_upper and "
                "fom_estimate carry its explicit estimate semantics, and "
                "fom_upper is null so it cannot be consumed as a certified "
                "upper-bound FOM. For rigorous endpoints estimated_upper and "
                "fom_estimate are null."
            ),
            "exact_methods": (
                "direct exact certificates are named by method; exact distances "
                "closed by matching endpoints use a composite lower+upper label "
                "such as L4+W"
            ),
            "status_labels": {
                "certified_interval": (
                    "certified lower and rigorous upper endpoints"
                ),
                "certified_lower_with_estimated_upper": (
                    "certified lower bound plus a B-only estimated endpoint"
                ),
                "supported_upper_only": (
                    "rigorous upper endpoint without a certified positive lower"
                ),
                "estimated_upper_only": (
                    "B-only estimated endpoint without a certified positive lower"
                ),
            },
            "evidence_methods": EVIDENCE_METHODS,
            "class_transfer": (
                "Certified lower bounds are combined within each colored-BLISS "
                "stored-generator Tanner-isomorphism class. The smallest "
                "reported endpoint is retained separately, with "
                "upper_is_supported distinguishing rigorous upper evidence "
                "from a decoder estimate."
            ),
        },
        "counts": {
            "source_observations": sum(build.raw_rows_by_campaign.values()),
            "source_presentations": len(build.all_specs),
            "source_verification_rows": sum(
                build.verification_rows_by_campaign.values()
            ),
            "retained_presentations": len(build.retained_specs),
            "retained_presentations_with_verification": sum(
                spec.verification_rows > 0 for spec in build.retained_specs
            ),
            "excluded_presentations": len(excluded),
            "retained_classes": len(build.classes),
            "presentations_by_family": _counter_dict(
                spec.campaign.family for spec in build.retained_specs
            ),
            "classes_by_family": _counter_dict(
                group.family for group in build.classes
            ),
            "presentations_by_direct_status": direct_status,
            "classes_by_status": class_status,
            "nonexact_presentations_by_upper_support": nonexact_direct_support,
            "nonexact_classes_by_upper_support": nonexact_class_support,
            "presentations_by_campaign_and_direct_status": _nested_status_counts(
                build.retained_specs
            ),
            "source_verification_rows_by_campaign": dict(
                sorted(build.verification_rows_by_campaign.items())
            ),
            "retained_verified_presentations_by_campaign": _counter_dict(
                spec.campaign.slug
                for spec in build.retained_specs
                if spec.verification_rows
            ),
            "excluded_by_campaign": _counter_dict(
                spec.campaign.slug for spec in excluded
            ),
            "excluded_by_direct_status": _counter_dict(
                supplement.status_for(spec) for spec in excluded
            ),
            "excluded_by_upper_endpoint": _counter_dict(
                str(spec.final_bounds()[1]) for spec in excluded
            ),
            "disconnected_presentations": disconnected_specs,
            "disconnected_classes": disconnected_classes,
            "component_certificates": len(build.component_certificates),
            "component_certificates_retained": len(component_exact_retained),
            "component_certificates_excluded_d_le_2": len(component_exact_excluded),
            "explicit_bb_component_matches": sum(
                len(rows) for rows in build.explicit_component_matches.values()
            ),
            "css_low_weight_audit_targets": len(build.css_low_weight_records),
            "css_low_weight_audit_targets_retained": len(low_weight_retained),
            "css_low_weight_audit_targets_excluded_d_le_2": len(
                low_weight_excluded
            ),
            "css_weight5_witness_certifications": len(
                build.css_weight5_witness_records
            ),
            "css_weight5_witness_certifications_retained": len(
                weight5_witness_retained
            ),
            "css_weight5_witness_certifications_excluded_d_le_2": len(
                weight5_witness_excluded
            ),
            "css_upper_bound_witness_corrections": len(
                build.css_upper_bound_witness_records
            ),
            "css_upper_bound_witness_corrections_retained": len(
                upper_bound_witness_retained
            ),
            "css_upper_bound_witness_corrections_excluded_d_le_2": len(
                upper_bound_witness_excluded
            ),
            "retained_css_missing_attribution": len(
                EXPECTED_MISSING_CSS_ATTRIBUTION
            ),
        },
        "verification_recorded_time_ranges": {
            campaign: {
                "records": len(timestamps),
                "earliest": min(timestamps),
                "latest": max(timestamps),
                "source_field": (
                    "_saved_at" if campaign.startswith("css-") else "verified_at"
                ),
                "semantics": (
                    "legacy_first_pass_not_continuation_completion"
                    if campaign == "pbb-large"
                    else "verifier_output_timestamp"
                ),
            }
            for campaign, timestamps in sorted(recorded_times_by_campaign.items())
        },
        "component_certification": {
            "artifact_manifest": build.component_manifest,
            "retained_presentation_ids": component_exact_retained,
            "excluded_presentation_ids": component_exact_excluded,
            "explicit_bb_match_parent_ids": sorted(build.explicit_component_matches),
        },
        "css_low_weight_audit": {
            "artifact_manifest": build.css_low_weight_manifest,
            "retained_presentation_ids": low_weight_retained,
            "excluded_presentation_ids": low_weight_excluded,
        },
        "css_weight5_witness_certification": {
            "artifact_manifest": build.css_weight5_witness_manifest,
            "retained_presentation_ids": weight5_witness_retained,
            "excluded_presentation_ids": weight5_witness_excluded,
        },
        "css_upper_bound_witness_correction": {
            "artifact_manifest": build.css_upper_bound_witness_manifest,
            "retained_presentation_ids": upper_bound_witness_retained,
            "excluded_presentation_ids": upper_bound_witness_excluded,
        },
        "source_artifacts": {
            key: source_descriptors[key]
            for key in sorted(source_descriptors)
            if not key.startswith("software:")
        },
        "software_artifacts": {
            key.removeprefix("software:"): source_descriptors[key]
            for key in sorted(source_descriptors)
            if key.startswith("software:")
        },
        "output_artifacts": {
            "catalogue": {
                "path": _relative(catalogue_path),
                "sha256": _sha256_bytes(catalogue_bytes),
                "bytes": len(catalogue_bytes),
                "records": len(build.retained_specs),
            }
        },
    }
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def _build_source_descriptors(
    component_path: Path,
    css_low_weight_path: Path,
    css_weight5_witness_path: Path = DEFAULT_CSS_WEIGHT5_WITNESSES,
    css_upper_bound_witness_path: Path = (
        DEFAULT_CSS_UPPER_BOUND_WITNESSES
    ),
) -> dict[str, dict[str, Any]]:
    descriptors = {
        key: _artifact_descriptor(path, jsonl=True)
        for key, path in _source_paths(
            component_path,
            css_low_weight_path,
            css_weight5_witness_path,
            css_upper_bound_witness_path,
        ).items()
    }
    descriptors.update(
        {
            f"software:{key}": _artifact_descriptor(path)
            for key, path in _software_paths().items()
        }
    )
    return descriptors


def _write_or_check(path: Path, payload: str, check: bool) -> bool:
    if check:
        if not path.exists():
            print(f"missing generated file: {path}", file=sys.stderr)
            return False
        if path.read_text(encoding="utf-8") != payload:
            print(f"generated file is stale: {path}", file=sys.stderr)
            return False
        print(f"up to date: {_relative(path)}")
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    print(f"wrote {_relative(path)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--component-certificates",
        type=Path,
        default=DEFAULT_COMPONENT_CERTIFICATES,
    )
    parser.add_argument(
        "--css-low-weight-audit",
        type=Path,
        default=DEFAULT_CSS_LOW_WEIGHT_AUDIT,
    )
    parser.add_argument(
        "--css-weight5-witnesses",
        type=Path,
        default=DEFAULT_CSS_WEIGHT5_WITNESSES,
    )
    parser.add_argument(
        "--css-upper-bound-witnesses",
        type=Path,
        default=DEFAULT_CSS_UPPER_BOUND_WITNESSES,
    )
    parser.add_argument(
        "--catalogue-output", type=Path, default=DEFAULT_CATALOGUE_OUTPUT
    )
    parser.add_argument(
        "--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT
    )
    parser.add_argument(
        "--check", action="store_true", help="fail if generated files are stale"
    )
    args = parser.parse_args()

    component_path = args.component_certificates.resolve()
    css_low_weight_path = args.css_low_weight_audit.resolve()
    css_weight5_witness_path = args.css_weight5_witnesses.resolve()
    css_upper_bound_witness_path = (
        args.css_upper_bound_witnesses.resolve()
    )
    catalogue_path = args.catalogue_output.resolve()
    manifest_path = args.manifest_output.resolve()
    initial_source_descriptors = _build_source_descriptors(
        component_path,
        css_low_weight_path,
        css_weight5_witness_path,
        css_upper_bound_witness_path,
    )
    build = build_publication_snapshot(
        component_path,
        css_low_weight_path,
        css_weight5_witness_path,
        css_upper_bound_witness_path,
    )
    source_descriptors = _build_source_descriptors(
        component_path,
        css_low_weight_path,
        css_weight5_witness_path,
        css_upper_bound_witness_path,
    )
    if source_descriptors != initial_source_descriptors:
        raise RuntimeError(
            "a publication input changed during catalogue generation; rerun from "
            "a stable workspace"
        )
    catalogue_payload = render_catalogue(build, source_descriptors)
    manifest_payload = render_manifest(
        build, catalogue_payload, catalogue_path, source_descriptors
    )

    catalogue_ok = _write_or_check(catalogue_path, catalogue_payload, args.check)
    manifest_ok = _write_or_check(manifest_path, manifest_payload, args.check)
    if not catalogue_ok or not manifest_ok:
        return 1
    print(
        "publication snapshot: "
        f"{len(build.retained_specs)} presentations / {len(build.classes)} classes; "
        f"direct E/C/U={EXPECTED_PRESENTATION_STATUS['E']}/"
        f"{EXPECTED_PRESENTATION_STATUS['C']}/"
        f"{EXPECTED_PRESENTATION_STATUS.get('U', 0)}; "
        f"class E/C/U={EXPECTED_CLASS_STATUS['E']}/"
        f"{EXPECTED_CLASS_STATUS['C']}/"
        f"{EXPECTED_CLASS_STATUS.get('U', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
