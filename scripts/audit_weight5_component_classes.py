#!/usr/bin/env python3
"""Canonicalize the connected components of the weight-five catalogue.

The publication catalogue groups stored *parent* presentations.  A parent can
be a repeated direct sum, so parent-level classes can overcount the underlying
connected codes.  This audit replaces each parent by one component of its
generator-basis-invariant stabilizer partition and applies the same colored
BLISS canonicalization to that component.

The output is deterministic JSON Lines: one manifest followed by one record
per component-presentation class.  Certified bounds are combined across all
parent classes reducing to the same component, while decoder-only endpoints
remain explicitly marked as estimates.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.bb_code import build_bb_code  # noqa: E402
from evaluation.connectivity import (  # noqa: E402
    component_canonical_digest,
    decompose,
    stabilizer_components,
)
from evaluation.pbb_code import build_pbb_code  # noqa: E402


SCHEMA_VERSION = 2
CATALOGUE = ROOT / "results" / "weight5_publication_catalogue.jsonl"
DEFAULT_OUTPUT = ROOT / "results" / "weight5_component_classes.jsonl"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_line(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def _build_code(row: dict[str, Any]):
    parameters = row["parameters"]
    generators = row["generators"]
    args = (
        int(parameters["ell"]),
        int(parameters["m"]),
        [tuple(term) for term in generators["A_terms"]],
        [tuple(term) for term in generators["B_terms"]],
    )
    if row["family"] == "CSS":
        return build_bb_code(*args)
    if row["family"] == "PBB":
        return build_pbb_code(
            *args,
            [tuple(term) for term in generators["C_terms"]] or None,
            [tuple(term) for term in generators["D_terms"]] or None,
        )
    raise ValueError(f"unknown family: {row['family']!r}")


def _load_catalogue() -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in CATALOGUE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 1_142:
        raise ValueError(f"expected 1,142 catalogue rows, found {len(rows)}")
    if any(
        row.get("schema_version") != 2
        or row.get("record_type") != "weight5_presentation"
        for row in rows
    ):
        raise ValueError("component audit requires publication catalogue schema v2")
    return rows


def _parent_class_reductions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["equivalence"]["class_id"])].append(row)
    if len(grouped) != 630:
        raise ValueError(f"expected 630 parent classes, found {len(grouped)}")

    reductions = []
    for parent_class_id, members in sorted(grouped.items()):
        representative = min(members, key=lambda row: row["presentation_id"])
        family = str(representative["family"])
        if {row["family"] for row in members} != {family}:
            raise ValueError(f"{parent_class_id}: mixed code families")

        parameters = representative["parameters"]
        connectivity = representative["connectivity"]
        distance = representative["distance_class"]
        distance_tuple = (
            distance["lower"],
            distance["upper"],
            bool(distance["is_exact"]),
            distance.get("upper_is_supported"),
        )
        if not isinstance(distance_tuple[3], bool):
            raise ValueError(
                f"{parent_class_id}: missing upper-endpoint support status"
            )
        if distance_tuple[2] and not distance_tuple[3]:
            raise ValueError(
                f"{parent_class_id}: exact distance lacks supported upper evidence"
            )
        for member in members:
            member_distance = member["distance_class"]
            if (
                member_distance["lower"],
                member_distance["upper"],
                bool(member_distance["is_exact"]),
                member_distance.get("upper_is_supported"),
            ) != distance_tuple:
                raise ValueError(f"{parent_class_id}: inconsistent class evidence")
            if member["connectivity"]["state"] != connectivity["state"]:
                raise ValueError(f"{parent_class_id}: inconsistent connectivity")

        if connectivity["is_connected"]:
            multiplicity = 1
            component_n = int(parameters["n"])
            component_k = int(parameters["k"])
            digest = str(representative["equivalence"]["canonical_digest_sha256"])
            qubits = list(range(component_n))
        else:
            code = _build_code(representative)
            decomposition = decompose(code)
            recorded_component = connectivity["component_code"]
            if (
                decomposition.is_connected
                or not decomposition.presentation_agrees
                or connectivity["components_bliss_isomorphic"] is not True
                or recorded_component is None
            ):
                raise ValueError(f"{parent_class_id}: uncertified component reduction")
            multiplicity = int(connectivity["n_components"])
            component_n = int(recorded_component["n"])
            component_k = int(recorded_component["k"])
            if (
                decomposition.n_components,
                decomposition.base_n,
                decomposition.base_k,
            ) != (multiplicity, component_n, component_k):
                raise ValueError(f"{parent_class_id}: decomposition mismatch")
            digest = component_canonical_digest(code)
            qubits = stabilizer_components(code)[0]

        reductions.append(
            {
                "family": family,
                "parent_class_id": parent_class_id,
                "parent_class_label": representative["equivalence"]["class_label"],
                "parent_n": int(parameters["n"]),
                "parent_k": int(parameters["k"]),
                "parent_presentation_ids": sorted(
                    str(row["presentation_id"]) for row in members
                ),
                "representative_parent_id": representative["presentation_id"],
                "representative_component_qubits": qubits,
                "multiplicity": multiplicity,
                "component_n": component_n,
                "component_k": component_k,
                "component_digest": digest,
                "distance_lower": distance["lower"],
                "distance_upper": int(distance["upper"]),
                "distance_exact": bool(distance["is_exact"]),
                "distance_upper_is_supported": bool(
                    distance["upper_is_supported"]
                ),
            }
        )
    return reductions


def _component_classes(reductions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in reductions:
        grouped[(row["family"], row["component_digest"])].append(row)

    preliminary = []
    for (family, digest), parents in grouped.items():
        nk = {(row["component_n"], row["component_k"]) for row in parents}
        if len(nk) != 1:
            raise ValueError(f"component {digest}: inconsistent dimensions {nk}")
        component_n, component_k = next(iter(nk))
        exact_values = {
            row["distance_upper"] for row in parents if row["distance_exact"]
        }
        if len(exact_values) > 1:
            raise ValueError(
                f"component {digest}: conflicting exact distances {exact_values}"
            )
        lowers = [
            int(row["distance_lower"])
            for row in parents
            if row["distance_lower"] is not None
        ]
        lower = max(lowers) if lowers else None
        upper = min(int(row["distance_upper"]) for row in parents)
        upper_is_supported = any(
            row["distance_upper"] == upper
            and row["distance_upper_is_supported"]
            for row in parents
        )
        if exact_values:
            exact = next(iter(exact_values))
            if (lower is not None and lower > exact) or upper < exact:
                raise ValueError(f"component {digest}: incompatible distance evidence")
            lower = upper = exact
            upper_is_supported = True
        if lower is not None and lower > upper:
            raise ValueError(f"component {digest}: empty distance interval")
        closes_exactly = bool(exact_values) or (
            lower is not None and lower == upper and upper_is_supported
        )
        status = "E" if closes_exactly else ("C" if lower is not None else "U")
        if status == "E":
            status_name = "exact"
        elif status == "C":
            status_name = (
                "certified_interval"
                if upper_is_supported
                else "certified_lower_with_estimated_upper"
            )
        else:
            status_name = (
                "supported_upper_only"
                if upper_is_supported
                else "estimated_upper_only"
            )
        endpoint_fom = component_k * upper * upper / component_n
        preliminary.append(
            {
                "schema_version": SCHEMA_VERSION,
                "record_type": "weight5_component_class",
                "family": family,
                "component_digest_sha256": digest,
                "parameters": {"n": component_n, "k": component_k},
                "distance": {
                    "status": status_name,
                    "status_code": status,
                    "lower": lower,
                    "upper": upper,
                    "estimated_upper": (
                        None if upper_is_supported else upper
                    ),
                    "is_exact": status == "E",
                    "upper_is_supported": upper_is_supported,
                    "fom_lower": (
                        None if lower is None else component_k * lower * lower / component_n
                    ),
                    "fom_upper": endpoint_fom if upper_is_supported else None,
                    "fom_estimate": (
                        None if upper_is_supported else endpoint_fom
                    ),
                },
                "parent_class_ids": sorted(row["parent_class_id"] for row in parents),
                "parent_class_labels": sorted(
                    row["parent_class_label"] for row in parents
                ),
                "parent_class_count": len(parents),
                "parent_presentation_ids": sorted(
                    presentation_id
                    for row in parents
                    for presentation_id in row["parent_presentation_ids"]
                ),
                "parent_presentation_count": sum(
                    len(row["parent_presentation_ids"]) for row in parents
                ),
                "parent_multiplicities": sorted(
                    {int(row["multiplicity"]) for row in parents}
                ),
                "representative_parent": {
                    "presentation_id": min(
                        str(row["representative_parent_id"]) for row in parents
                    ),
                    "component_qubits": next(
                        row["representative_component_qubits"]
                        for row in sorted(
                            parents, key=lambda item: item["representative_parent_id"]
                        )
                    ),
                },
            }
        )

    preliminary.sort(
        key=lambda row: (
            row["family"],
            row["parameters"]["n"],
            -row["parameters"]["k"],
            row["component_digest_sha256"],
        )
    )
    counters: Counter[str] = Counter()
    seen_ids: set[str] = set()
    for row in preliminary:
        family = str(row["family"])
        counters[family] += 1
        prefix = "CC" if family == "CSS" else "PC"
        row["component_class_label"] = f"{prefix}{counters[family]:03d}"
        class_id = f"{prefix}-{row['component_digest_sha256'][:10]}"
        if class_id in seen_ids:
            raise ValueError(f"truncated component-class ID collision: {class_id}")
        seen_ids.add(class_id)
        row["component_class_id"] = class_id
    return preliminary


def build_rows() -> list[dict[str, Any]]:
    catalogue = _load_catalogue()
    reductions = _parent_class_reductions(catalogue)
    classes = _component_classes(reductions)
    family_counts = Counter(row["family"] for row in classes)
    status_counts = Counter(row["distance"]["status_code"] for row in classes)
    collapsed_parent_classes = sum(row["parent_class_count"] - 1 for row in classes)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "artifact_manifest",
        "artifact": "weight5_component_classes",
        "generator": "scripts/audit_weight5_component_classes.py",
        "equivalence_relation": "colored_stored_generator_tanner_isomorphism_v1_after_component_reduction",
        "source_file": {
            "path": "results/weight5_publication_catalogue.jsonl",
            "sha256": _sha256(CATALOGUE),
            "records": len(catalogue),
        },
        "counts": {
            "parent_presentation_classes": len(reductions),
            "component_classes": len(classes),
            "collapsed_parent_classes": collapsed_parent_classes,
            "component_classes_by_family": dict(sorted(family_counts.items())),
            "component_classes_by_status": dict(sorted(status_counts.items())),
        },
        "semantics": (
            "Each retained parent presentation is replaced by one member of its "
            "homogeneous, pairwise-isomorphic connected-component orbit before "
            "colored-BLISS canonicalization; this remains a stored-generator "
            "equivalence and not complete stabilizer-code equivalence. The "
            "distance upper_is_supported flag is true only when the decisive "
            "endpoint has non-decoder evidence; a B-only value is retained in "
            "upper and estimated_upper as an estimate, with fom_upper left null."
        ),
    }
    return [manifest, *classes]


def render(rows: list[dict[str, Any]]) -> str:
    return "".join(_json_line(row) for row in rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    payload = render(build_rows())
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != payload:
            print(f"stale component-class artifact: {args.output}", file=sys.stderr)
            return 1
        print(f"component-class artifact is current: {args.output}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    manifest = json.loads(payload.splitlines()[0])
    print(
        "component-class audit: "
        f"{manifest['counts']['parent_presentation_classes']} parent classes -> "
        f"{manifest['counts']['component_classes']} component classes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
