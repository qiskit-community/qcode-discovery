#!/usr/bin/env python3
"""Audit connectivity of the pinned QEC Challenge comparison records.

The two paper snapshots intentionally contain only compact metadata.  This
script consumes a checkout of the upstream QEC Challenge repository at the
commit named by those snapshots, rebuilds every selected CSS check matrix,
and writes a deterministic connectivity artifact.  The artifact is sufficient
for the comparison-figure generator; the upstream checkout is needed only to
regenerate or freshness-check the audit.

Example::

    python scripts/audit_qldpc_challenge_connectivity.py \
        --source-dir /path/to/qldpc-challenge
    python scripts/audit_qldpc_challenge_connectivity.py \
        --source-dir /path/to/qldpc-challenge --check
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version as distribution_version
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.connectivity import (  # noqa: E402
    component_canonical_digest,
    components_isomorphic,
    decompose,
    stabilizer_components,
)


SCHEMA_VERSION = 1
EXPECTED_COMMIT = "3b50503b058a0c09500c1017b54c3d547e1101cc"
SNAPSHOTS = {
    "actual_weight_five": ROOT
    / "paper"
    / "2610.06623"
    / "figures"
    / "qldpc_challenge_weight5_snapshot.json",
    "bb": ROOT
    / "paper"
    / "2610.06623"
    / "figures"
    / "qldpc_challenge_bb_snapshot.json",
}
DEFAULT_OUTPUT = ROOT / "results" / "weight5_challenge_connectivity_audit.jsonl"


class _ChallengeCSSCode:
    """Minimal CSS-code interface consumed by ``evaluation.connectivity``."""

    def __init__(
        self,
        matrix_x: np.ndarray,
        matrix_z: np.ndarray,
        *,
        dimension: int,
    ) -> None:
        self.matrix_x = np.asarray(matrix_x, dtype=np.uint8)
        self.matrix_z = np.asarray(matrix_z, dtype=np.uint8)
        self.num_qudits = int(self.matrix_x.shape[1])
        self.dimension = int(dimension)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_line(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def _gf2_rank(matrix: np.ndarray) -> int:
    work = np.asarray(matrix, dtype=np.uint8).copy() % 2
    row = 0
    for column in range(work.shape[1]):
        candidates = np.flatnonzero(work[row:, column])
        if not candidates.size:
            continue
        pivot = row + int(candidates[0])
        if pivot != row:
            work[[row, pivot]] = work[[pivot, row]]
        mask = work[:, column].astype(bool)
        mask[row] = False
        work[mask] ^= work[row]
        row += 1
        if row == work.shape[0]:
            break
    return row


def _support_matrix(rows: object, n: int, label: str) -> np.ndarray:
    if not isinstance(rows, list):
        raise ValueError(f"{label} checks must be a list")
    matrix = np.zeros((len(rows), n), dtype=np.uint8)
    for row_index, support in enumerate(rows):
        if not isinstance(support, list) or not support:
            raise ValueError(f"{label} check {row_index} has invalid support")
        normalized: list[int] = []
        for qubit in support:
            if isinstance(qubit, bool) or not isinstance(qubit, int):
                raise ValueError(f"{label} check {row_index} has a non-integer index")
            if not 0 <= qubit < n:
                raise ValueError(
                    f"{label} check {row_index} index {qubit} is outside [0,{n})"
                )
            normalized.append(qubit)
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{label} check {row_index} repeats a qubit")
        matrix[row_index, normalized] = 1
    return matrix


def _validate_logical_witness(
    *,
    side: str,
    raw_distance: dict[str, Any],
    matrix_x: np.ndarray,
    matrix_z: np.ndarray,
    component_of: dict[int, int],
) -> dict[str, Any]:
    detail = raw_distance.get(side)
    if not isinstance(detail, dict):
        raise ValueError(f"distance.{side} is missing")
    value = int(detail["value"])
    support = detail.get("witness")
    if not isinstance(support, list) or not support:
        raise ValueError(f"distance.{side}.witness is missing")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in support):
        raise ValueError(f"distance.{side}.witness has a non-integer index")
    n = matrix_x.shape[1]
    if any(not 0 <= index < n for index in support):
        raise ValueError(f"distance.{side}.witness has an out-of-range index")
    if len(support) != len(set(support)) or len(support) != value:
        raise ValueError(f"distance.{side}.witness has the wrong Hamming weight")

    vector = np.zeros(n, dtype=np.uint8)
    vector[support] = 1
    if side == "X":
        commuting_checks, same_type_checks = matrix_z, matrix_x
    elif side == "Z":
        commuting_checks, same_type_checks = matrix_x, matrix_z
    else:  # pragma: no cover - internal call site is fixed
        raise ValueError(f"unknown logical side: {side}")
    if np.any((commuting_checks @ vector) % 2):
        raise ValueError(f"distance.{side}.witness anticommutes with a stabilizer")
    if _gf2_rank(np.vstack([same_type_checks, vector])) != (
        _gf2_rank(same_type_checks) + 1
    ):
        raise ValueError(f"distance.{side}.witness is a stabilizer")

    component_indices = sorted({component_of[index] for index in support})
    if len(component_indices) != 1:
        raise ValueError(
            f"distance.{side}.witness spans components {component_indices}"
        )
    return {
        "value": value,
        "confidence": str(detail.get("confidence", "")),
        "component_index": component_indices[0],
        "validated": True,
    }


def _git_head(source_dir: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _validate_checkout(source_dir: Path) -> None:
    if _git_head(source_dir) != EXPECTED_COMMIT:
        raise ValueError(
            f"QEC Challenge checkout must be at pinned commit {EXPECTED_COMMIT}"
        )
    modified = subprocess.run(
        [
            "git",
            "-C",
            str(source_dir),
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            "codes",
            "certs",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if modified:
        raise ValueError(
            "QEC Challenge checkout has tracked modifications under codes/ or certs/"
        )


def _load_snapshot_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    source_metadata: dict[str, Any] = {}
    for name, path in SNAPSHOTS.items():
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        if snapshot.get("commit") != EXPECTED_COMMIT:
            raise ValueError(f"{path}: unexpected QEC Challenge commit")
        raw_records = snapshot.get("records")
        if not isinstance(raw_records, list):
            raise ValueError(f"{path}: records must be a list")
        source_metadata[name] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(path),
            "records": len(raw_records),
        }
        for raw in raw_records:
            slug = str(raw["slug"])
            if slug in records:
                raise ValueError(f"Challenge slug occurs in both snapshots: {slug}")
            records[slug] = {**raw, "snapshot": name}
    if len(records) != 102:
        raise ValueError(f"expected 102 unique Challenge rows, found {len(records)}")
    return [records[slug] for slug in sorted(records)], source_metadata


def _audit_record(source_dir: Path, snapshot: dict[str, Any]) -> dict[str, Any]:
    slug = str(snapshot["slug"])
    code_path = source_dir / "codes" / f"{slug}.json"
    if not code_path.is_file():
        raise FileNotFoundError(f"missing pinned Challenge code: {code_path}")
    raw = json.loads(code_path.read_text(encoding="utf-8"))
    if raw.get("code_type") != "CSS":
        raise ValueError(f"{slug}: expected a CSS check matrix")

    n, k = int(raw["n"]), int(raw["k"])
    d = int(raw["distance"]["d"])
    if (n, k, d) != (int(snapshot["n"]), int(snapshot["k"]), int(snapshot["d"])):
        raise ValueError(f"{slug}: snapshot parameters do not match the source code")
    matrix_x = _support_matrix(raw["checks"]["X"], n, "X")
    matrix_z = _support_matrix(raw["checks"]["Z"], n, "Z")
    if np.any((matrix_x @ matrix_z.T) % 2):
        raise ValueError(f"{slug}: CSS checks do not commute")
    rank_x, rank_z = _gf2_rank(matrix_x), _gf2_rank(matrix_z)
    computed_k = n - rank_x - rank_z
    if computed_k != k:
        raise ValueError(f"{slug}: stored k={k}, recomputed k={computed_k}")
    max_check_weight = max(
        int(matrix_x.sum(axis=1).max(initial=0)),
        int(matrix_z.sum(axis=1).max(initial=0)),
    )
    if max_check_weight != int(snapshot["max_check_weight"]):
        raise ValueError(f"{slug}: snapshot check weight does not match source")

    code_sha256 = _sha256(code_path)
    expected_code_sha256 = snapshot.get("code_sha256")
    if expected_code_sha256 is not None and code_sha256 != expected_code_sha256:
        raise ValueError(f"{slug}: source-code hash does not match the snapshot")

    status = str(snapshot["distance_status"])
    certificate_path = source_dir / "certs" / f"{slug}.json"
    certificate_sha256 = None
    if status == "exact":
        if not certificate_path.is_file():
            raise FileNotFoundError(f"{slug}: exact row has no certificate")
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        if certificate.get("d_exact") is not True or int(certificate["d"]) != d:
            raise ValueError(f"{slug}: invalid exact-distance certificate")
        certificate_sha256 = _sha256(certificate_path)
        expected_certificate_sha256 = snapshot.get("certificate_sha256")
        if (
            expected_certificate_sha256 is not None
            and certificate_sha256 != expected_certificate_sha256
        ):
            raise ValueError(f"{slug}: certificate hash does not match the snapshot")
    elif status != "upper_bound":
        raise ValueError(f"{slug}: unsupported distance status {status!r}")
    elif certificate_path.exists():
        raise ValueError(f"{slug}: upper-bound row unexpectedly has a certificate")

    code = _ChallengeCSSCode(matrix_x, matrix_z, dimension=k)
    components = stabilizer_components(code)
    decomposition = decompose(code)
    if decomposition.k_additivity_ok is False:
        raise ValueError(f"{slug}: component dimensions do not sum to k")
    if decomposition.n_components != len(components):
        raise ValueError(f"{slug}: inconsistent component count")
    component_k = decomposition.component_k if decomposition.component_k else [k]
    component_rows = [
        {"index": index, "n": len(qubits), "k": int(component_k[index])}
        for index, qubits in enumerate(components)
    ]
    component_of = {
        qubit: component_index
        for component_index, qubits in enumerate(components)
        for qubit in qubits
    }
    witnesses = {
        side: _validate_logical_witness(
            side=side,
            raw_distance=raw["distance"],
            matrix_x=matrix_x,
            matrix_z=matrix_z,
            component_of=component_of,
        )
        for side in ("X", "Z")
    }
    if min(item["value"] for item in witnesses.values()) != d:
        raise ValueError(f"{slug}: witness distances do not match d={d}")

    positive_k = [item["index"] for item in component_rows if item["k"] > 0]
    isomorphic: bool | None = None
    component_digest: str | None = None
    if decomposition.is_connected:
        method = "connected"
        selected_index = 0
    elif decomposition.homogeneous and len(set(component_k)) == 1:
        isomorphic = components_isomorphic(code)
        if isomorphic is not True:
            method = "unresolved"
            selected_index = None
        else:
            method = "homogeneous_isomorphic"
            selected_index = 0
            component_digest = component_canonical_digest(code)
    elif len(positive_k) == 1:
        method = "unique_positive_k"
        selected_index = positive_k[0]
        if any(
            witness["component_index"] != selected_index
            for witness in witnesses.values()
        ):
            raise ValueError(f"{slug}: witness is outside the unique logical component")
    else:
        method = "unresolved"
        selected_index = None

    reduced = None
    if selected_index is not None:
        selected = component_rows[selected_index]
        reduced = {
            "n": selected["n"],
            "k": selected["k"],
            "d": d,
            "distance_status": status,
            "component_index": selected_index,
            "method": method,
            "distance_transfer_validated": True,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "qldpc_challenge_connectivity",
        "slug": slug,
        "snapshot": str(snapshot["snapshot"]),
        "source": {
            "code_path": f"codes/{slug}.json",
            "code_sha256": code_sha256,
            "certificate_path": (
                f"certs/{slug}.json" if certificate_sha256 is not None else None
            ),
            "certificate_sha256": certificate_sha256,
        },
        "parameters": {
            "n": n,
            "k": k,
            "d": d,
            "max_check_weight": max_check_weight,
            "distance_status": status,
            "family": str(snapshot.get("family") or raw.get("family") or "BB"),
        },
        "matrix_validation": {
            "x_rows": int(matrix_x.shape[0]),
            "z_rows": int(matrix_z.shape[0]),
            "rank_x": rank_x,
            "rank_z": rank_z,
            "commutes": True,
            "computed_k": computed_k,
        },
        "connectivity": {
            "n_components": decomposition.n_components,
            "components": component_rows,
            "is_connected": decomposition.is_connected,
            "homogeneous": decomposition.homogeneous,
            "k_additivity_verified": (
                True if decomposition.is_connected else decomposition.k_additivity_ok
            ),
            "presentation_partition_agrees": decomposition.presentation_agrees,
            "components_bliss_isomorphic": isomorphic,
            "component_canonical_digest_sha256": component_digest,
            "positive_k_component_indices": positive_k,
        },
        "witnesses": witnesses,
        "reduction": reduced,
    }


def build_rows(source_dir: Path) -> list[dict[str, Any]]:
    source_dir = source_dir.resolve()
    _validate_checkout(source_dir)
    snapshot_rows, source_metadata = _load_snapshot_records()
    audit_rows = [_audit_record(source_dir, row) for row in snapshot_rows]
    unresolved = [row["slug"] for row in audit_rows if row["reduction"] is None]
    if unresolved:
        raise ValueError(f"unresolved Challenge reductions: {unresolved}")

    method_counts = Counter(row["reduction"]["method"] for row in audit_rows)
    disconnected_by_weight = Counter(
        str(row["parameters"]["max_check_weight"])
        for row in audit_rows
        if not row["connectivity"]["is_connected"]
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "artifact_manifest",
        "artifact": "weight5_challenge_connectivity_audit",
        "generator": "scripts/audit_qldpc_challenge_connectivity.py",
        "upstream": {
            "repository": "https://github.com/unitaryfoundation/qldpc-challenge",
            "commit": EXPECTED_COMMIT,
        },
        "source_snapshots": source_metadata,
        "software": {
            "python-igraph": distribution_version("igraph"),
        },
        "counts": {
            "records": len(audit_rows),
            "connected": sum(
                row["connectivity"]["is_connected"] for row in audit_rows
            ),
            "disconnected": sum(
                not row["connectivity"]["is_connected"] for row in audit_rows
            ),
            "reduction_methods": dict(sorted(method_counts.items())),
            "disconnected_by_weight": dict(sorted(disconnected_by_weight.items())),
        },
        "semantics": (
            "Every record in the two pinned Challenge snapshots is rebuilt from "
            "its submitted CSS check matrices and decomposed using the invariant "
            "stabilizer-row-space partition. Homogeneous reductions require "
            "pairwise colored-BLISS isomorphism. A heterogeneous parent is reduced "
            "only when exactly one component has positive logical dimension and "
            "both submitted logical witnesses lie in that component."
        ),
    }
    return [manifest, *audit_rows]


def render(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(_json_line(row) for row in rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="QEC Challenge checkout at the pinned commit",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    payload = render(build_rows(args.source_dir))
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != payload:
            print(f"stale Challenge connectivity artifact: {args.output}", file=sys.stderr)
            return 1
        print(f"Challenge connectivity artifact is current: {args.output}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    manifest = json.loads(payload.splitlines()[0])
    print(
        f"Challenge connectivity audit: {manifest['counts']['disconnected']}/"
        f"{manifest['counts']['records']} disconnected; wrote {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
