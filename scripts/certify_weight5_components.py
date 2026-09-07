#!/usr/bin/env python3
"""Certify exact distances through disconnected weight-five CSS components.

Seven high-dimension CSS-large presentations were not selected for the
original parent-level MILP audit because their discovery-time distance bounds
were loose.  Each presentation is, however, a direct sum of small isomorphic
components.  This script:

* rebuilds each parent from its stored polynomial tuple;
* checks that the presentation-Tanner and stabilizer-row-space partitions
  agree, that component dimensions add to the parent dimension, and that all
  component presentations are BLISS-isomorphic;
* compares the restricted presentation row space with the row space induced
  from the full parent stabilizer;
* computes both exact CSS distances of one component by exhaustive
  ``qldpc`` enumeration and transfers their minimum to the isomorphic direct
  sum; and
* records explicit connected BB representatives for selected exact parents,
  including every derived component needed by the comparison frontier,
  together with a colored-BLISS presentation match.

The raw append-only campaign logs are read but never modified.  Output is
canonical JSON Lines with no timestamp, so ``--check`` is deterministic.

Usage:
    uv run python scripts/certify_weight5_components.py
    uv run python scripts/certify_weight5_components.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.bb_code import build_bb_code  # noqa: E402
from evaluation.connectivity import (  # noqa: E402
    components_isomorphic,
    decompose,
    stabilizer_components,
    tanner_components,
)
from evaluation.pbb_code import _gf2_rref  # noqa: E402
from evaluation.tanner_equivalence import (  # noqa: E402
    _extract_check_matrices,
    canonical_digest,
    canonical_hash,
)
from qldpc.codes import CSSCode  # noqa: E402
from qldpc.objects import Pauli  # noqa: E402


SCHEMA_VERSION = 1
LARGE_RAW_RELATIVE_PATH = Path(
    "results/evolution/weight5_css_large_v1/all_codes_weight5_css.jsonl"
)
LARGE_VERIFICATION_RELATIVE_PATH = Path(
    "results/weight5_css_large_v1_milp_verified.jsonl"
)
SMALL_RAW_RELATIVE_PATH = Path(
    "results/evolution/weight5_css_small_v1/all_codes_weight5_css.jsonl"
)
SMALL_VERIFICATION_RELATIVE_PATH = Path(
    "results/weight5_css_small_v1_milp_verified.jsonl"
)
DEFAULT_OUTPUT = ROOT / "results/weight5_component_certifications.jsonl"


Terms = tuple[tuple[int, int], ...]
SpecKey = tuple[int, int, Terms, Terms]


@dataclass(frozen=True)
class Target:
    presentation_id: str
    ell: int
    m: int
    A: Terms
    B: Terms
    expected_n: int
    expected_k: int
    expected_multiplicity: int
    expected_component_n: int
    expected_component_k: int
    expected_distance: int
    campaign: str = "css-large"

    @property
    def key(self) -> SpecKey:
        return self.ell, self.m, _normalise_terms(self.A), _normalise_terms(self.B)


@dataclass(frozen=True)
class ExplicitRepresentative:
    parent: Target
    ell: int
    m: int
    A: Terms
    B: Terms


# Expected values below are assertions, not input evidence.  Every distance is
# recomputed from one induced component when this script runs.
HIGH_K_TARGETS = (
    Target(
        "CL-5cd6065f", 18, 12,
        ((0, 0), (9, 0)), ((0, 0), (6, 4), (12, 2)),
        432, 72, 18, 24, 4, 2,
    ),
    Target(
        "CL-4b943aa2", 15, 14,
        ((0, 0), (5, 12), (10, 8)), ((0, 0), (10, 0)),
        420, 60, 10, 42, 6, 3,
    ),
    Target(
        "CL-929c2a70", 16, 12,
        ((0, 0), (4, 4), (12, 8)), ((0, 0), (4, 0)),
        384, 64, 16, 24, 4, 3,
    ),
    Target(
        "CL-b9275593", 18, 12,
        ((0, 0), (0, 4)), ((0, 0), (6, 2), (12, 4)),
        432, 48, 12, 36, 4, 3,
    ),
    Target(
        "CL-437b6e5f", 15, 14,
        ((0, 0), (5, 4), (10, 6)), ((0, 0), (0, 6)),
        420, 40, 10, 42, 4, 5,
    ),
    Target(
        "CL-d9f65a7e", 16, 12,
        ((0, 0), (4, 2), (10, 7)), ((0, 0), (2, 3)),
        384, 32, 8, 48, 4, 3,
    ),
    Target(
        "CL-a25b2baf", 16, 12,
        ((0, 0), (4, 2), (4, 10)), ((0, 0), (2, 9)),
        384, 32, 8, 48, 4, 6,
    ),
)


# These connected BB presentations were derived by restricting the parent
# lattices to one connected orbit.  BLISS equality below proves equality of
# their stored-generator Tanner presentations up to qubit/check permutations.
EXPLICIT_REPRESENTATIVES = (
    ExplicitRepresentative(
        Target(
            "CL-885dfa58", 16, 12,
            ((0, 0), (0, 4), (2, 8)), ((0, 0), (3, 6)),
            384, 16, 4, 96, 4, 10,
        ),
        16, 3,
        ((0, 0), (0, 1), (2, 2)), ((0, 0), (3, 0)),
    ),
    ExplicitRepresentative(
        Target(
            "CL-f8ef7ae2", 15, 14,
            ((0, 0), (0, 13), (12, 11)), ((0, 0), (3, 7)),
            420, 18, 3, 140, 6, 10,
        ),
        5, 14,
        ((0, 0), (0, 13), (4, 11)), ((0, 0), (1, 7)),
    ),
    ExplicitRepresentative(
        Target(
            "CS-d473da9b", 10, 9,
            ((0, 0), (4, 8), (6, 7)), ((0, 0), (4, 6)),
            180, 8, 2, 90, 4, 9,
            campaign="css-small",
        ),
        5, 9,
        ((0, 0), (2, 8), (3, 7)), ((0, 0), (2, 6)),
    ),
)


def _source_paths(target: Target) -> tuple[Path, Path]:
    if target.campaign == "css-small":
        return SMALL_RAW_RELATIVE_PATH, SMALL_VERIFICATION_RELATIVE_PATH
    if target.campaign == "css-large":
        return LARGE_RAW_RELATIVE_PATH, LARGE_VERIFICATION_RELATIVE_PATH
    raise ValueError(f"unknown CSS campaign: {target.campaign!r}")


class _MatrixCSSCode:
    """Minimal raw-presentation wrapper accepted by BLISS helpers."""

    def __init__(self, matrix_x: np.ndarray, matrix_z: np.ndarray):
        self.matrix_x = np.asarray(matrix_x, dtype=np.uint8)
        self.matrix_z = np.asarray(matrix_z, dtype=np.uint8)


def _normalise_terms(terms: Iterable[Iterable[int]]) -> Terms:
    return tuple(sorted((int(x), int(y)) for x, y in terms))


def _record_key(row: dict) -> SpecKey:
    return (
        int(row["ell"]),
        int(row["m"]),
        _normalise_terms(row["A_terms"]),
        _normalise_terms(row["B_terms"]),
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _presentation_id(key: SpecKey, campaign: str = "css-large") -> str:
    ell, m, A, B = key
    payload = [campaign, ell, m, A, B, (), ()]
    digest = _sha256_bytes(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    )[:8]
    prefix = "CS" if campaign == "css-small" else "CL"
    return f"{prefix}-{digest}"


def _load_jsonl_index(path: Path) -> dict[SpecKey, list[dict]]:
    index: dict[SpecKey, list[dict]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            entry = {
                "line_number": line_number,
                "canonical_record_sha256": _sha256_bytes(_canonical_json(row)),
                "record": row,
            }
            index.setdefault(_record_key(row), []).append(entry)
    return index


def _rref_basis(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.uint8) % 2
    rref, pivots = _gf2_rref(matrix)
    return np.asarray(rref[: len(pivots)], dtype=np.uint8)


def _matrix_digest(matrix: np.ndarray) -> str:
    matrix = np.ascontiguousarray(np.asarray(matrix, dtype=np.uint8) % 2)
    shape = json.dumps(list(matrix.shape), separators=(",", ":")).encode("ascii")
    return _sha256_bytes(shape + b"\0" + np.packbits(matrix, axis=None).tobytes())


def _symplectic_css(matrix_x: np.ndarray, matrix_z: np.ndarray) -> np.ndarray:
    matrix_x = np.asarray(matrix_x, dtype=np.uint8) % 2
    matrix_z = np.asarray(matrix_z, dtype=np.uint8) % 2
    n = matrix_x.shape[1]
    return np.vstack(
        [
            np.hstack([matrix_x, np.zeros((matrix_x.shape[0], n), dtype=np.uint8)]),
            np.hstack([np.zeros((matrix_z.shape[0], n), dtype=np.uint8), matrix_z]),
        ]
    )


def _restrict_presentation(
    code, qubits: list[int]
) -> tuple[_MatrixCSSCode, np.ndarray]:
    """Restrict supplied checks and independently recover the induced row space."""
    matrix_x, matrix_z = _extract_check_matrices(code)
    n = int(code.num_qudits)
    indices = np.asarray(qubits, dtype=int)
    selected = np.zeros(n, dtype=bool)
    selected[indices] = True

    def restricted_rows(matrix: np.ndarray) -> np.ndarray:
        support = matrix != 0
        keep = support.any(axis=1) & ~(support & ~selected).any(axis=1)
        return np.asarray(matrix[keep][:, indices], dtype=np.uint8)

    component = _MatrixCSSCode(restricted_rows(matrix_x), restricted_rows(matrix_z))

    parent_basis = _rref_basis(_symplectic_css(matrix_x, matrix_z))
    support = (parent_basis[:, :n] != 0) | (parent_basis[:, n:] != 0)
    keep = support.any(axis=1) & ~(support & ~selected).any(axis=1)
    induced = np.hstack(
        [parent_basis[keep][:, indices], parent_basis[keep][:, n + indices]]
    )
    return component, _rref_basis(induced)


def _component_checks(code) -> dict:
    rowspace_partition = stabilizer_components(code)
    presentation_partition = tanner_components(code)
    if rowspace_partition != presentation_partition:
        raise AssertionError("presentation and stabilizer-row-space partitions differ")

    decomposition = decompose(code)
    if not decomposition.presentation_agrees:
        raise AssertionError("decompose() did not confirm partition agreement")
    if decomposition.k_additivity_ok is not True:
        raise AssertionError("component dimensions do not add to parent k")
    if components_isomorphic(code) is not True:
        raise AssertionError("components are not BLISS-isomorphic")

    component_codes: list[_MatrixCSSCode] = []
    component_digests: list[str] = []
    component_rowspace_digests: list[str] = []
    for qubits in rowspace_partition:
        component, induced_rowspace = _restrict_presentation(code, qubits)
        presented_rowspace = _rref_basis(
            _symplectic_css(component.matrix_x, component.matrix_z)
        )
        if not np.array_equal(presented_rowspace, induced_rowspace):
            raise AssertionError("restricted presentation does not span induced row space")
        component_codes.append(component)
        component_digests.append(canonical_digest(component))
        component_rowspace_digests.append(_matrix_digest(induced_rowspace))

    if len(set(component_digests)) != 1:
        raise AssertionError("component canonical digests differ")
    if len(set(component_rowspace_digests)) != 1:
        # Equal BLISS presentations imply equivalent row spaces, but their
        # literal RREF matrices can differ because component-local qubit order
        # differs.  Therefore this is recorded rather than required.
        literal_rowspaces_identical = False
    else:
        literal_rowspaces_identical = True

    return {
        "partition": rowspace_partition,
        "decomposition": decomposition,
        "component_codes": component_codes,
        "component_digests": component_digests,
        "component_rowspace_digests": component_rowspace_digests,
        "literal_rowspaces_identical": literal_rowspaces_identical,
    }


def _source_identity(
    target: Target,
    raw_indexes: dict[Path, dict[SpecKey, list[dict]]],
) -> dict:
    raw_relative_path, _ = _source_paths(target)
    raw_index = raw_indexes[raw_relative_path]
    observations = raw_index.get(target.key, [])
    if not observations:
        raise AssertionError(f"{target.presentation_id} is absent from the raw archive")
    if _presentation_id(target.key, target.campaign) != target.presentation_id:
        raise AssertionError(f"presentation ID mismatch for {target.presentation_id}")
    for observation in observations:
        row = observation["record"]
        if (int(row["n"]), int(row["k"])) != (target.expected_n, target.expected_k):
            raise AssertionError(f"stored parameter mismatch for {target.presentation_id}")
    return {
        "campaign": target.campaign,
        "presentation_id": target.presentation_id,
        "raw_path": raw_relative_path.as_posix(),
        "observation_line_numbers": [row["line_number"] for row in observations],
        "observation_canonical_sha256": [
            row["canonical_record_sha256"] for row in observations
        ],
    }


def _verify_target_shape(target: Target, checks: dict, code) -> None:
    decomposition = checks["decomposition"]
    actual = (
        int(code.num_qudits),
        int(code.dimension),
        decomposition.n_components,
        decomposition.base_n,
        decomposition.base_k,
    )
    expected = (
        target.expected_n,
        target.expected_k,
        target.expected_multiplicity,
        target.expected_component_n,
        target.expected_component_k,
    )
    if actual != expected:
        raise AssertionError(
            f"unexpected decomposition for {target.presentation_id}: "
            f"expected {expected}, got {actual}"
        )


def _distance_certificate_row(
    target: Target,
    raw_indexes: dict[Path, dict[SpecKey, list[dict]]],
) -> dict:
    code = build_bb_code(target.ell, target.m, list(target.A), list(target.B))
    checks = _component_checks(code)
    _verify_target_shape(target, checks, code)

    first = checks["component_codes"][0]
    exact_code = CSSCode(first.matrix_x, first.matrix_z)
    if (int(exact_code.num_qudits), int(exact_code.dimension)) != (
        target.expected_component_n,
        target.expected_component_k,
    ):
        raise AssertionError(f"induced component parameter mismatch for {target.presentation_id}")

    distance_x = int(exact_code.get_distance_exact(Pauli.X))
    distance_z = int(exact_code.get_distance_exact(Pauli.Z))
    distance = min(distance_x, distance_z)
    if distance != target.expected_distance:
        raise AssertionError(
            f"distance mismatch for {target.presentation_id}: "
            f"expected {target.expected_distance}, got {distance}"
        )

    decomposition = checks["decomposition"]
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "component_distance_certificate",
        "certificate_id": f"W5CC-{target.presentation_id.removeprefix('CL-')}",
        "source_identity": _source_identity(target, raw_indexes),
        "parent_spec": {
            "family": "CSS",
            "ell": target.ell,
            "m": target.m,
            "A_terms": [list(term) for term in target.A],
            "B_terms": [list(term) for term in target.B],
            "n": target.expected_n,
            "k": target.expected_k,
            "d_exact": distance,
        },
        "decomposition": {
            "n_components": decomposition.n_components,
            "component_sizes": decomposition.component_sizes,
            "component_dimensions": decomposition.component_k,
            "rowspace_partition_equals_presentation_partition": True,
            "restricted_presentation_spans_induced_rowspace": True,
            "k_additivity_verified": True,
            "components_bliss_isomorphic": True,
            "component_presentation_canonical_digest_sha256": checks[
                "component_digests"
            ][0],
            "component_rowspace_digest_sha256": checks[
                "component_rowspace_digests"
            ][0],
            "literal_component_rowspaces_identical_before_relabeling": checks[
                "literal_rowspaces_identical"
            ],
        },
        "component_certificate": {
            "component_index": 0,
            "qubit_indices": checks["partition"][0],
            "n": target.expected_component_n,
            "k": target.expected_component_k,
            "d_exact": distance,
            "d_x_exact": distance_x,
            "d_z_exact": distance_z,
            "distance_method": "qldpc.CSSCode.get_distance_exact exhaustive enumeration",
            "canonical_digest_sha256": checks["component_digests"][0],
            "rowspace_digest_sha256": checks["component_rowspace_digests"][0],
        },
        "distance_transfer": {
            "parent_d_exact": distance,
            "valid": True,
            "reason": (
                "The parent is a direct sum of pairwise BLISS-isomorphic "
                "components, so its distance equals the exactly enumerated "
                "component distance."
            ),
        },
    }


def _exact_source_identity(
    target: Target,
    verification_indexes: dict[Path, dict[SpecKey, list[dict]]],
) -> dict:
    _, verification_relative_path = _source_paths(target)
    verification_index = verification_indexes[verification_relative_path]
    rows = verification_index.get(target.key, [])
    exact_rows = [
        row for row in rows
        if row["record"].get("d_is_exact") is True
        and int(row["record"]["d"]) == target.expected_distance
    ]
    if len(exact_rows) != 1:
        raise AssertionError(
            f"expected one exact parent certificate for {target.presentation_id}, "
            f"found {len(exact_rows)}"
        )
    row = exact_rows[0]
    return {
        "path": verification_relative_path.as_posix(),
        "line_number": row["line_number"],
        "canonical_record_sha256": row["canonical_record_sha256"],
        "method": row["record"].get("stage"),
    }


def _explicit_representative_row(
    item: ExplicitRepresentative,
    raw_indexes: dict[Path, dict[SpecKey, list[dict]]],
    verification_indexes: dict[Path, dict[SpecKey, list[dict]]],
) -> dict:
    parent = build_bb_code(
        item.parent.ell, item.parent.m, list(item.parent.A), list(item.parent.B)
    )
    checks = _component_checks(parent)
    _verify_target_shape(item.parent, checks, parent)
    induced = checks["component_codes"][0]

    representative = build_bb_code(item.ell, item.m, list(item.A), list(item.B))
    rep_decomposition = decompose(representative)
    if not rep_decomposition.is_connected:
        raise AssertionError("explicit component representative is disconnected")
    if (int(representative.num_qudits), int(representative.dimension)) != (
        item.parent.expected_component_n,
        item.parent.expected_component_k,
    ):
        raise AssertionError("explicit component representative has wrong parameters")
    if canonical_hash(induced) != canonical_hash(representative):
        raise AssertionError("explicit BB representative does not BLISS-match component")

    digest = canonical_digest(representative)
    if digest != checks["component_digests"][0]:
        raise AssertionError("canonical digest mismatch after positive BLISS comparison")

    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "explicit_bb_component_match",
        "match_id": (
            f"W5BB-{item.parent.expected_component_n}-"
            f"{item.parent.expected_component_k}-{item.parent.expected_distance}"
        ),
        "parent_source_identity": _source_identity(item.parent, raw_indexes),
        "parent_exact_distance_source": _exact_source_identity(
            item.parent, verification_indexes
        ),
        "parent_spec": {
            "ell": item.parent.ell,
            "m": item.parent.m,
            "A_terms": [list(term) for term in item.parent.A],
            "B_terms": [list(term) for term in item.parent.B],
            "n": item.parent.expected_n,
            "k": item.parent.expected_k,
            "d_exact": item.parent.expected_distance,
            "n_components": item.parent.expected_multiplicity,
        },
        "explicit_bb_component": {
            "ell": item.ell,
            "m": item.m,
            "A_terms": [list(term) for term in item.A],
            "B_terms": [list(term) for term in item.B],
            "n": item.parent.expected_component_n,
            "k": item.parent.expected_component_k,
            "d_exact": item.parent.expected_distance,
            "connected": True,
            "stabilizer_weight": len(item.A) + len(item.B),
        },
        "match": {
            "equivalence_relation": "colored_stored_generator_tanner_isomorphism_v1",
            "bliss_isomorphic": True,
            "canonical_digest_sha256": digest,
            "distance_transfer_valid": True,
            "distance_transfer_reason": (
                "The exact-distance parent is a direct sum of isomorphic "
                "components, and this connected BB presentation BLISS-matches "
                "one induced component."
            ),
        },
    }


def build_artifact_rows() -> list[dict]:
    raw_relative_paths = {LARGE_RAW_RELATIVE_PATH, SMALL_RAW_RELATIVE_PATH}
    verification_relative_paths = {
        LARGE_VERIFICATION_RELATIVE_PATH,
        SMALL_VERIFICATION_RELATIVE_PATH,
    }
    raw_indexes = {
        path: _load_jsonl_index(ROOT / path) for path in raw_relative_paths
    }
    verification_indexes = {
        path: _load_jsonl_index(ROOT / path) for path in verification_relative_paths
    }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "artifact_manifest",
        "artifact": "weight5_component_certifications",
        "generator": "scripts/certify_weight5_components.py",
        "semantics": (
            "Exact distances of induced components and their pairwise-isomorphic "
            "direct-sum parents; explicit BB component rows inherit exact parent "
            "distance only after a positive colored-BLISS match."
        ),
        "source_files": {
            path.as_posix(): _sha256_file(ROOT / path)
            for path in sorted(raw_relative_paths | verification_relative_paths)
        },
        "software": {
            "qldpc": distribution_version("qldpc"),
            "python-igraph": distribution_version("igraph"),
        },
        "record_counts": {
            "component_distance_certificate": len(HIGH_K_TARGETS),
            "explicit_bb_component_match": len(EXPLICIT_REPRESENTATIVES),
        },
    }
    certificates = [
        _distance_certificate_row(target, raw_indexes)
        for target in sorted(HIGH_K_TARGETS, key=lambda item: item.presentation_id)
    ]
    representatives = [
        _explicit_representative_row(item, raw_indexes, verification_indexes)
        for item in sorted(
            EXPLICIT_REPRESENTATIVES,
            key=lambda item: item.parent.expected_component_n,
        )
    ]
    return [manifest, *certificates, *representatives]


def render_artifact(rows: list[dict]) -> str:
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if output is stale")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    rendered = render_artifact(build_artifact_rows())
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != rendered:
            print(f"stale component-certification artifact: {args.output}", file=sys.stderr)
            return 1
        print(f"component-certification artifact is current: {args.output}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"wrote {len(HIGH_K_TARGETS)} exact component certificates and "
          f"{len(EXPLICIT_REPRESENTATIVES)} explicit BB matches to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
