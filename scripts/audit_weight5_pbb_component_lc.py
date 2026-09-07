#!/usr/bin/env python3
"""Complete LC-to-CSS audit for the frontier ``[[12,2,3]]`` components.

The component-class catalogue contains four distinct PBB classes with these
parameters.  This script reconstructs an induced stabilizer row space for one
representative of each class and exhaustively decides whether any tensor
product of single-qubit Clifford gates maps that row space to CSS form.

Usage::

    python scripts/audit_weight5_pbb_component_lc.py
    python scripts/audit_weight5_pbb_component_lc.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.clifford_equivalence import (  # noqa: E402
    _independent_gf2_rows,
    exact_local_clifford_css_search,
)
from evaluation.connectivity import (  # noqa: E402
    _stabilizer_basis,
    stabilizer_components,
)
from evaluation.pbb_code import build_pbb_code  # noqa: E402


CATALOGUE_PATH = ROOT / "results" / "weight5_publication_catalogue.jsonl"
COMPONENT_CLASSES_PATH = ROOT / "results" / "weight5_component_classes.jsonl"
OUTPUT_PATH = ROOT / "results" / "weight5_pbb_component_lc_audit.jsonl"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix_digest(matrix: np.ndarray) -> str:
    normalized = np.ascontiguousarray(matrix, dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(json.dumps(list(normalized.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def _records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _component_stabilizer(parent: dict[str, Any], qubits: list[int]) -> np.ndarray:
    parameters = parent["parameters"]
    generators = parent["generators"]
    code = build_pbb_code(
        int(parameters["ell"]),
        int(parameters["m"]),
        [tuple(term) for term in generators["A_terms"]],
        [tuple(term) for term in generators["B_terms"]],
        [tuple(term) for term in generators["C_terms"]] or None,
        [tuple(term) for term in generators["D_terms"]] or None,
    )
    parent_basis = _stabilizer_basis(code)
    parent_n = int(code.num_qudits)
    indices = np.asarray(qubits, dtype=int)
    if len(set(qubits)) != len(qubits) or np.any(indices < 0) or np.any(indices >= parent_n):
        raise ValueError("component qubit list is not a valid subset of the parent")
    recomputed_components = stabilizer_components(code)
    if sorted(qubits) not in recomputed_components:
        raise ValueError("stored qubit list is not a finest row-space component")

    mask = np.zeros(parent_n, dtype=bool)
    mask[indices] = True
    support = (parent_basis[:, :parent_n] != 0) | (parent_basis[:, parent_n:] != 0)
    crosses_boundary = support.any(axis=1) & (support & mask).any(axis=1) & (support & ~mask).any(axis=1)
    if np.any(crosses_boundary):
        raise ValueError("purported component is crossed by a stabilizer-basis row")
    keep = support.any(axis=1) & (support & mask).any(axis=1)
    restricted = np.hstack(
        [parent_basis[keep][:, indices], parent_basis[keep][:, parent_n + indices]]
    )
    return _independent_gf2_rows(restricted)


def _commutes(stabilizer: np.ndarray) -> bool:
    n = stabilizer.shape[1] // 2
    x_part = stabilizer[:, :n]
    z_part = stabilizer[:, n:]
    products = (x_part @ z_part.T + z_part @ x_part.T) % 2
    return not np.any(products)


def build_records() -> list[dict[str, Any]]:
    catalogue = {
        row["presentation_id"]: row
        for row in _records(CATALOGUE_PATH)
        if row.get("record_type") != "artifact_manifest"
    }
    component_classes = [
        row
        for row in _records(COMPONENT_CLASSES_PATH)
        if row.get("record_type") == "weight5_component_class"
        and row.get("family") == "PBB"
        and row.get("parameters") == {"k": 2, "n": 12}
        and row.get("distance", {}).get("is_exact")
        and row.get("distance", {}).get("lower") == 3
    ]
    component_classes.sort(key=lambda row: row["component_class_label"])
    if len(component_classes) != 4:
        raise ValueError(
            "expected exactly four exact PBB [[12,2,3]] component classes; "
            f"found {len(component_classes)}"
        )

    output: list[dict[str, Any]] = []
    for component_class in component_classes:
        representative = component_class["representative_parent"]
        presentation_id = representative["presentation_id"]
        parent = catalogue[presentation_id]
        stabilizer = _component_stabilizer(
            parent, [int(qubit) for qubit in representative["component_qubits"]]
        )
        n = stabilizer.shape[1] // 2
        rank = stabilizer.shape[0]
        if n != 12 or n - rank != 2:
            raise ValueError(
                f"{component_class['component_class_label']} reconstructed as "
                f"[[{n},{n-rank}]], expected [[12,2]]"
            )
        if not _commutes(stabilizer):
            raise ValueError(
                f"{component_class['component_class_label']} stabilizers do not commute"
            )

        result = exact_local_clifford_css_search(stabilizer)
        output.append(
            {
                "record_type": "weight5_pbb_component_lc_audit",
                "schema_version": 1,
                "component_class_id": component_class["component_class_id"],
                "component_class_label": component_class["component_class_label"],
                "parameters": {"n": n, "k": n - rank, "d": 3},
                "representative_parent": representative,
                "component_rowspace_digest_sha256": _matrix_digest(stabilizer),
                "commutation_verified": True,
                "search": result,
                "conclusion": (
                    "lc_css_equivalent"
                    if result["is_lc_css"]
                    else "not_lc_css_equivalent"
                ),
            }
        )

    manifest = {
        "record_type": "artifact_manifest",
        "schema_version": 1,
        "artifact": "weight5_pbb_component_lc_audit",
        "generator": "scripts/audit_weight5_pbb_component_lc.py",
        "scope": "all four exact PBB [[12,2,3]] component classes",
        "method": (
            "exhaustive branch-and-bound over all per-qubit Clifford actions "
            "modulo phase, using the generator-basis-invariant CSS rank criterion"
        ),
        "counts": {
            "component_classes": len(output),
            "lc_css_equivalent": sum(row["search"]["is_lc_css"] for row in output),
            "certified_not_lc_css_equivalent": sum(
                not row["search"]["is_lc_css"] for row in output
            ),
        },
        "source_files": {
            str(CATALOGUE_PATH.relative_to(ROOT)): _sha256(CATALOGUE_PATH),
            str(COMPONENT_CLASSES_PATH.relative_to(ROOT)): _sha256(
                COMPONENT_CLASSES_PATH
            ),
        },
    }
    return [manifest, *output]


def render(records: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    generated = render(build_records())
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != generated:
            raise SystemExit(f"stale or missing artifact: {args.output}")
        try:
            display_path = args.output.relative_to(ROOT)
        except ValueError:
            display_path = args.output
        print(f"verified {display_path}")
        return

    args.output.write_text(generated, encoding="utf-8")
    try:
        display_path = args.output.relative_to(ROOT)
    except ValueError:
        display_path = args.output
    print(f"wrote {display_path}")


if __name__ == "__main__":
    main()
