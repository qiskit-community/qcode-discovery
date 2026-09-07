#!/usr/bin/env python3
"""Reconstruct explicit logical witnesses for every direct ``L4+B`` CSS row.

The historical BP--OSD API used by the weight-five campaign returned only a
scalar upper bound.  Consequently the raw campaign JSONL files record
``d = 5`` but not the corresponding operator.  After the deterministic
weight-four audit proved ``d >= 5``, those scalar observations made 45 direct
catalogue rows exact.  This script supplies the missing independently
checkable upper-bound evidence without rerunning BP--OSD.

For each such presentation and each CSS sector, translation invariance lets a
weight-five support be shifted until one of its qubits is the origin of its
left or right bicycle block.  The other four qubits are split into two pairs.
Pair syndromes are indexed and collided against the anchored syndrome.  Every
reported support is then checked directly to have zero syndrome and nonzero
remainder modulo the same-type stabilizer row space.

The checked-in artifact is deterministic: it contains no timestamps or
runtime measurements.

Usage::

    python scripts/audit_weight5_css_weight5_witnesses.py
    python scripts/audit_weight5_css_weight5_witnesses.py --check
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parent.parent
CATALOGUE = ROOT / "results" / "weight5_publication_catalogue.jsonl"
LOW_WEIGHT_AUDIT = ROOT / "results" / "weight5_css_low_weight_audit.jsonl"
DEFAULT_OUTPUT = ROOT / "results" / "weight5_css_weight5_witnesses.jsonl"

SCHEMA_VERSION = 1
ARTIFACT_SCHEMA = "weight5_css_weight5_witnesses_v1"
ALGORITHM = "translation_anchored_pair_pair_collision_v1"

sys.path.insert(0, str(ROOT))
from scripts.audit_weight5_css_low_weight import (  # noqa: E402
    _columns_from_rows,
    _matrix_digest,
    _qubit_labels,
    _rowspace_remainder_columns,
    _support_indices,
    _verified_witness,
    _xor_columns,
    build_bb_check_rows,
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _json_line(value: object) -> str:
    return _canonical_json(value).decode("ascii")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _has_bound(
    evidence: Iterable[dict[str, Any]], method: str, bound: int
) -> bool:
    return any(
        item.get("method") == method and int(item.get("bound", -1)) == bound
        for item in evidence
    )


def _is_direct_l4_b_exact_five(row: dict[str, Any]) -> bool:
    direct = row.get("distance_direct", {})
    evidence = direct.get("evidence", {})
    return bool(
        row.get("family") == "CSS"
        and direct.get("is_exact") is True
        and direct.get("lower") == 5
        and direct.get("upper") == 5
        and _has_bound(evidence.get("lower_bounds", []), "L4", 5)
        and _has_bound(evidence.get("upper_bounds", []), "B", 5)
    )


def _candidate_key(row: dict[str, Any]) -> tuple[object, ...]:
    params = row["parameters"]
    generators = row["generators"]
    return (
        int(params["ell"]),
        int(params["m"]),
        tuple(sorted(tuple(term) for term in generators["A_terms"])),
        tuple(sorted(tuple(term) for term in generators["B_terms"])),
    )


def _raw_key(row: dict[str, Any]) -> tuple[object, ...] | None:
    try:
        return (
            int(row["ell"]),
            int(row["m"]),
            tuple(sorted(tuple(term) for term in row["A_terms"])),
            tuple(sorted(tuple(term) for term in row["B_terms"])),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _index_raw_observations(
    catalogue_rows: Sequence[dict[str, Any]],
) -> tuple[dict[tuple[object, ...], list[dict[str, Any]]], list[Path]]:
    paths = sorted(
        {
            ROOT / row["source_artifacts"]["raw_log"]["path"]
            for row in catalogue_rows
        }
    )
    index: dict[tuple[object, ...], list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        for observation in _load_jsonl(path):
            key = _raw_key(observation)
            if key is not None:
                index[key].append(observation)
    return index, paths


def find_weight_five_witness(
    check_rows: Sequence[int],
    stabilizer_rows: Sequence[int],
    num_qubits: int,
    group_order: int,
) -> dict[str, Any]:
    """Find and directly validate a weight-five logical in one CSS sector.

    The search is only a witness finder, not the lower-bound proof.  The
    independent ``L4`` artifact supplies exhaustive exclusion through weight
    four.  Retaining every pair in each syndrome bucket avoids heuristic
    choices when enforcing disjoint pair supports.
    """
    check_columns = _columns_from_rows(check_rows, num_qubits)
    remainder_columns, stabilizer_rank = _rowspace_remainder_columns(
        stabilizer_rows, num_qubits
    )

    total_pairs_indexed = 0
    total_pairs_probed = 0
    for anchor in (0, group_order):
        if not 0 <= anchor < num_qubits:
            continue
        vertices = [index for index in range(num_qubits) if index != anchor]
        pair_by_syndrome: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for offset, left in enumerate(vertices):
            left_bit = 1 << left
            left_syndrome = check_columns[left]
            left_remainder = remainder_columns[left]
            for right in vertices[offset + 1 :]:
                support = left_bit | (1 << right)
                pair_by_syndrome[left_syndrome ^ check_columns[right]].append(
                    (left_remainder ^ remainder_columns[right], support)
                )
                total_pairs_indexed += 1

        anchor_bit = 1 << anchor
        anchor_syndrome = check_columns[anchor]
        anchor_remainder = remainder_columns[anchor]
        for offset, left in enumerate(vertices):
            left_bit = 1 << left
            left_syndrome = check_columns[left]
            left_remainder = remainder_columns[left]
            for right in vertices[offset + 1 :]:
                total_pairs_probed += 1
                probe_support = left_bit | (1 << right)
                probe_remainder = left_remainder ^ remainder_columns[right]
                target = anchor_syndrome ^ left_syndrome ^ check_columns[right]
                for indexed_remainder, indexed_support in pair_by_syndrome.get(
                    target, ()
                ):
                    if indexed_support & probe_support:
                        continue
                    support = anchor_bit | indexed_support | probe_support
                    if support.bit_count() != 5:
                        continue
                    if anchor_remainder ^ indexed_remainder ^ probe_remainder == 0:
                        continue
                    _verified_witness(
                        support, check_columns, remainder_columns
                    )
                    return {
                        "qubit_indices_zero_based": _support_indices(support),
                        "anchor_qubit": anchor,
                        "pairs_indexed": total_pairs_indexed,
                        "pairs_probed": total_pairs_probed,
                        "stabilizer_rank": stabilizer_rank,
                        "weight": 5,
                        "zero_check_syndrome": (
                            _xor_columns(check_columns, support) == 0
                        ),
                        "outside_stabilizer_rowspace": (
                            _xor_columns(remainder_columns, support) != 0
                        ),
                    }
    raise ValueError("no translation-anchored weight-five logical was found")


def _low_weight_by_class() -> dict[str, dict[str, Any]]:
    rows = _load_jsonl(LOW_WEIGHT_AUDIT)
    return {
        row["class_identity"]["class_id"]: row
        for row in rows
        if row.get("record_type") == "css_low_weight_class_audit"
    }


def _audit_row(
    row: dict[str, Any],
    low_weight: dict[str, dict[str, Any]],
    raw_index: dict[tuple[object, ...], list[dict[str, Any]]],
) -> dict[str, Any]:
    params = row["parameters"]
    generators = row["generators"]
    ell, m = int(params["ell"]), int(params["m"])
    num_qubits = int(params["n"])
    group_order = ell * m
    if num_qubits != 2 * group_order:
        raise ValueError(f"{row['presentation_id']}: n != 2 ell m")

    class_id = row["equivalence"]["class_id"]
    lower_audit = low_weight[class_id]
    if lower_audit["audit_evidence"] != {
        "result": "excluded_through_weight_4",
        "exact_distance": None,
        "certified_lower_bound": 5,
    }:
        raise ValueError(f"{row['presentation_id']}: missing L4 lower bound")

    rows_x, rows_z = build_bb_check_rows(
        ell, m, generators["A_terms"], generators["B_terms"]
    )
    witness_x = find_weight_five_witness(
        rows_z, rows_x, num_qubits, group_order
    )
    witness_z = find_weight_five_witness(
        rows_x, rows_z, num_qubits, group_order
    )
    witness_x["qubits"] = _qubit_labels(
        witness_x["qubit_indices_zero_based"], ell, m
    )
    witness_z["qubits"] = _qubit_labels(
        witness_z["qubit_indices_zero_based"], ell, m
    )

    observations = [
        item
        for item in raw_index.get(_candidate_key(row), [])
        if int(item.get("d", -1)) == 5
    ]
    if not observations:
        raise ValueError(f"{row['presentation_id']}: no historical d=5 record")
    witness_like_keys = {
        key
        for observation in observations
        for key in observation
        if "witness" in key.lower() or key in {"L", "logical_operator"}
    }
    if witness_like_keys:
        raise ValueError(
            f"{row['presentation_id']}: unexpected persisted witness keys "
            f"{sorted(witness_like_keys)}"
        )

    rank_x = witness_x["stabilizer_rank"]
    rank_z = witness_z["stabilizer_rank"]
    rebuilt_k = num_qubits - rank_x - rank_z
    if rebuilt_k != int(params["k"]):
        raise ValueError(
            f"{row['presentation_id']}: rebuilt k={rebuilt_k} != {params['k']}"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "css_weight5_witness_audit",
        "algorithm": ALGORITHM,
        "presentation_id": row["presentation_id"],
        "class_id": class_id,
        "parameters": {
            "ell": ell,
            "m": m,
            "n": num_qubits,
            "k_catalogue": int(params["k"]),
            "k_rebuilt": rebuilt_k,
            "d_exact": 5,
        },
        "generators": {
            "A_terms": generators["A_terms"],
            "B_terms": generators["B_terms"],
        },
        "matrix_digests": {
            "H_X_sha256": _matrix_digest(rows_x, num_qubits),
            "H_Z_sha256": _matrix_digest(rows_z, num_qubits),
        },
        "lower_bound": {
            "method": "L4",
            "certified": 5,
            "audit_class_id": class_id,
            "audit_representative_id": lower_audit["representative"][
                "presentation_id"
            ],
        },
        "historical_upper_bound": {
            "method": "B",
            "value": 5,
            "matching_scalar_observations": len(observations),
            "operator_was_persisted": False,
        },
        "replacement_explicit_witnesses": {
            "X": witness_x,
            "Z": witness_z,
        },
        "conclusion": (
            "Both CSS sectors have directly validated weight-five logical "
            "operators; combined with the independent exhaustive L4 exclusion, "
            "d_X=d_Z=d=5."
        ),
    }


def build_records() -> list[dict[str, Any]]:
    catalogue_rows = _load_jsonl(CATALOGUE)
    targets = sorted(
        (row for row in catalogue_rows if _is_direct_l4_b_exact_five(row)),
        key=lambda row: row["presentation_id"],
    )
    if len(targets) != 45:
        raise ValueError(f"expected 45 direct L4+B rows, found {len(targets)}")
    low_weight = _low_weight_by_class()
    raw_index, raw_paths = _index_raw_observations(targets)
    audits = [_audit_row(row, low_weight, raw_index) for row in targets]

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "artifact_manifest",
        "artifact_schema": ARTIFACT_SCHEMA,
        "generator": "scripts/audit_weight5_css_weight5_witnesses.py",
        "algorithm": ALGORITHM,
        "semantics": (
            "Explicit independently reconstructed X- and Z-sector logical "
            "witnesses for all direct CSS catalogue rows made exact by L4+B."
        ),
        "historical_bp_osd_persistence": (
            "The campaign records contain scalar d=5 observations but no "
            "operator vectors; this artifact supplies replacement witnesses."
        ),
        "counts": {
            "presentations": len(audits),
            "presentation_classes": len({row["class_id"] for row in audits}),
            "explicit_witnesses": 2 * len(audits),
            "all_x_witnesses_valid": all(
                row["replacement_explicit_witnesses"]["X"][
                    "zero_check_syndrome"
                ]
                and row["replacement_explicit_witnesses"]["X"][
                    "outside_stabilizer_rowspace"
                ]
                for row in audits
            ),
            "all_z_witnesses_valid": all(
                row["replacement_explicit_witnesses"]["Z"][
                    "zero_check_syndrome"
                ]
                and row["replacement_explicit_witnesses"]["Z"][
                    "outside_stabilizer_rowspace"
                ]
                for row in audits
            ),
        },
        "source_artifacts": {
            "publication_catalogue": {
                "path": _relative(CATALOGUE),
                "sha256": _sha256(CATALOGUE),
            },
            "low_weight_audit": {
                "path": _relative(LOW_WEIGHT_AUDIT),
                "sha256": _sha256(LOW_WEIGHT_AUDIT),
            },
            "raw_campaign_logs": [
                {"path": _relative(path), "sha256": _sha256(path)}
                for path in raw_paths
            ],
        },
    }
    return [manifest, *audits]


def _serialized(records: Sequence[dict[str, Any]]) -> bytes:
    return ("\n".join(_json_line(row) for row in records) + "\n").encode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail unless the existing artifact is byte-for-byte current",
    )
    args = parser.parse_args()
    payload = _serialized(build_records())
    if args.check:
        if not args.output.exists() or args.output.read_bytes() != payload:
            print(f"stale or missing artifact: {args.output}", file=sys.stderr)
            return 1
        print(f"verified {args.output.relative_to(ROOT)}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(f"wrote {args.output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
