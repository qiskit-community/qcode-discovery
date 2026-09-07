#!/usr/bin/env python3
"""Certify explicit connected PBB realizations of disconnected components.

Four exact PBB-large presentations split into pairwise-isomorphic components.
This script rebuilds each exact parent and a smaller PBB presentation, then
checks the complete distance-transfer chain:

* the parent dimensions and homogeneous direct-sum decomposition;
* agreement of the stored-generator and stabilizer-row-space partitions;
* pairwise colored-BLISS isomorphism of the parent components;
* exact all-logical MILP evidence for the parent distance;
* connectedness and dimensions of the smaller PBB presentation; and
* colored-BLISS equality between that presentation and an induced component.

The raw append-only campaign logs and verifier output are read but never
modified.  Output is canonical JSON Lines without timestamps, so ``--check``
is deterministic.

Usage:
    uv run python scripts/certify_weight5_pbb_components.py
    uv run python scripts/certify_weight5_pbb_components.py --check
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

from evaluation.connectivity import (  # noqa: E402
    components_isomorphic,
    decompose,
    stabilizer_components,
    tanner_components,
)
from evaluation.noncss_gate import passes_noncss_gate  # noqa: E402
from evaluation.pbb_code import _gf2_rref, build_pbb_code  # noqa: E402
from evaluation.tanner_equivalence import (  # noqa: E402
    _extract_symplectic_matrix,
    canonical_digest_noncss,
    canonical_hash_noncss,
)


SCHEMA_VERSION = 1
RAW_RELATIVE_PATH = Path(
    "results/evolution/weight5_pbb_large_v1/all_codes_weight5_pbb.jsonl"
)
VERIFICATION_RELATIVE_PATH = Path(
    "results/weight5_pbb_large_v1_deep_milp_verify.jsonl"
)
DEFAULT_OUTPUT = ROOT / "results/weight5_pbb_component_certifications.jsonl"


Terms = tuple[tuple[int, int], ...]
SpecKey = tuple[int, int, Terms, Terms, Terms, Terms]


@dataclass(frozen=True)
class PbbTarget:
    presentation_id: str
    ell: int
    m: int
    A: Terms
    B: Terms
    C: Terms
    D: Terms
    expected_n: int
    expected_k: int
    expected_multiplicity: int
    expected_component_n: int
    expected_component_k: int
    expected_distance: int

    @property
    def key(self) -> SpecKey:
        return (
            self.ell,
            self.m,
            _normalise_terms(self.A),
            _normalise_terms(self.B),
            _normalise_terms(self.C),
            _normalise_terms(self.D),
        )


@dataclass(frozen=True)
class ExplicitPbbRepresentative:
    parent: PbbTarget
    ell: int
    m: int
    A: Terms
    B: Terms
    C: Terms
    D: Terms


# Expected parameters are assertions.  Exact distances come from the pinned
# all-logical MILP rows, not from the discovery-time ``d`` field.
EXPLICIT_PBB_REPRESENTATIVES = (
    ExplicitPbbRepresentative(
        PbbTarget(
            "PL-8ee754a5",
            15,
            12,
            ((0, 0), (2, 4)),
            ((0, 0), (2, 6), (4, 0)),
            ((0, 0), (2, 4)),
            ((0, 0), (4, 0)),
            360,
            4,
            2,
            180,
            2,
            11,
        ),
        15,
        6,
        ((0, 0), (2, 2)),
        ((0, 0), (2, 3), (4, 0)),
        ((0, 0), (2, 2)),
        ((0, 0), (4, 0)),
    ),
    ExplicitPbbRepresentative(
        PbbTarget(
            "PL-ae4cc6ad",
            18,
            12,
            ((0, 0), (4, 2)),
            ((0, 0), (1, 6), (2, 0)),
            (),
            ((0, 0), (2, 0)),
            432,
            8,
            2,
            216,
            4,
            10,
        ),
        18,
        6,
        ((0, 0), (4, 1)),
        ((0, 0), (1, 3), (2, 0)),
        (),
        ((0, 0), (2, 0)),
    ),
    ExplicitPbbRepresentative(
        PbbTarget(
            "PL-36c5ce3b",
            18,
            12,
            ((0, 0), (2, 2)),
            ((0, 0), (1, 6), (2, 0)),
            ((0, 0), (2, 2)),
            ((0, 0), (2, 0)),
            432,
            8,
            2,
            216,
            4,
            10,
        ),
        18,
        6,
        ((0, 0), (2, 1)),
        ((0, 0), (1, 3), (2, 0)),
        ((0, 0), (2, 1)),
        ((0, 0), (2, 0)),
    ),
    ExplicitPbbRepresentative(
        PbbTarget(
            "PL-aefa6773",
            18,
            12,
            ((0, 0), (3, 3)),
            ((0, 0), (1, 6), (2, 0)),
            ((0, 0), (3, 3)),
            ((1, 6),),
            432,
            12,
            3,
            144,
            4,
            8,
        ),
        18,
        4,
        ((0, 0), (3, 1)),
        ((0, 0), (1, 2), (2, 0)),
        ((0, 0), (3, 1)),
        ((1, 2),),
    ),
)


class _MatrixNonCSSCode:
    """Minimal raw-presentation wrapper accepted by non-CSS BLISS helpers."""

    def __init__(self, matrix: np.ndarray):
        self.matrix = np.asarray(matrix, dtype=np.uint8)


def _normalise_terms(terms: Iterable[Iterable[int]]) -> Terms:
    return tuple(sorted((int(x), int(y)) for x, y in terms))


def _record_key(row: dict) -> SpecKey:
    return (
        int(row["ell"]),
        int(row["m"]),
        _normalise_terms(row["A_terms"]),
        _normalise_terms(row["B_terms"]),
        _normalise_terms(row.get("C_terms") or ()),
        _normalise_terms(row.get("D_terms") or ()),
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


def _presentation_id(key: SpecKey) -> str:
    ell, m, A, B, C, D = key
    payload = ["pbb-large", ell, m, A, B, C, D]
    digest = _sha256_bytes(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    )[:8]
    return f"PL-{digest}"


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


def _restrict_presentation(
    code, qubits: list[int]
) -> tuple[_MatrixNonCSSCode, np.ndarray]:
    """Restrict supplied PBB rows and independently recover induced row space."""
    matrix = _extract_symplectic_matrix(code)
    n = int(code.num_qudits)
    indices = np.asarray(qubits, dtype=int)
    selected = np.zeros(n, dtype=bool)
    selected[indices] = True

    support = (matrix[:, :n] != 0) | (matrix[:, n:] != 0)
    keep = support.any(axis=1) & ~(support & ~selected).any(axis=1)
    restricted = np.hstack(
        [matrix[keep][:, indices], matrix[keep][:, n + indices]]
    )

    parent_basis = _rref_basis(matrix)
    basis_support = (parent_basis[:, :n] != 0) | (parent_basis[:, n:] != 0)
    basis_keep = basis_support.any(axis=1) & ~(
        basis_support & ~selected
    ).any(axis=1)
    induced = np.hstack(
        [
            parent_basis[basis_keep][:, indices],
            parent_basis[basis_keep][:, n + indices],
        ]
    )
    return _MatrixNonCSSCode(restricted), _rref_basis(induced)


def _component_checks(target: PbbTarget, code) -> dict:
    rowspace_partition = stabilizer_components(code)
    presentation_partition = tanner_components(code)
    if rowspace_partition != presentation_partition:
        raise AssertionError(
            f"{target.presentation_id}: presentation and row-space partitions differ"
        )

    decomposition = decompose(code)
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
    if not decomposition.presentation_agrees:
        raise AssertionError(f"{target.presentation_id}: partition agreement failed")
    if decomposition.k_additivity_ok is not True:
        raise AssertionError(f"{target.presentation_id}: dimension additivity failed")
    if components_isomorphic(code) is not True:
        raise AssertionError(f"{target.presentation_id}: components are not isomorphic")

    components: list[_MatrixNonCSSCode] = []
    component_digests: list[str] = []
    rowspace_digests: list[str] = []
    for qubits in rowspace_partition:
        component, induced_rowspace = _restrict_presentation(code, qubits)
        if not np.array_equal(_rref_basis(component.matrix), induced_rowspace):
            raise AssertionError(
                f"{target.presentation_id}: restricted rows do not span component"
            )
        components.append(component)
        component_digests.append(canonical_digest_noncss(component))
        rowspace_digests.append(_matrix_digest(induced_rowspace))
    if len(set(component_digests)) != 1:
        raise AssertionError(f"{target.presentation_id}: component digests differ")

    return {
        "partition": rowspace_partition,
        "decomposition": decomposition,
        "components": components,
        "component_digests": component_digests,
        "rowspace_digests": rowspace_digests,
        "literal_rowspaces_identical": len(set(rowspace_digests)) == 1,
    }


def _source_identity(
    target: PbbTarget, raw_index: dict[SpecKey, list[dict]]
) -> dict:
    observations = raw_index.get(target.key, [])
    if not observations:
        raise AssertionError(f"{target.presentation_id} is absent from the raw archive")
    if _presentation_id(target.key) != target.presentation_id:
        raise AssertionError(f"presentation ID mismatch for {target.presentation_id}")
    for observation in observations:
        row = observation["record"]
        if (int(row["n"]), int(row["k"])) != (
            target.expected_n,
            target.expected_k,
        ):
            raise AssertionError(f"stored parameter mismatch for {target.presentation_id}")
    return {
        "campaign": "pbb-large",
        "presentation_id": target.presentation_id,
        "raw_path": RAW_RELATIVE_PATH.as_posix(),
        "observation_line_numbers": [row["line_number"] for row in observations],
        "observation_canonical_sha256": [
            row["canonical_record_sha256"] for row in observations
        ],
    }


def _exact_source_identity(
    target: PbbTarget, verification_index: dict[SpecKey, list[dict]]
) -> dict:
    rows = verification_index.get(target.key, [])
    exact_rows = []
    for row in rows:
        record = row["record"]
        milp_distance = record.get("milp_d")
        total = record.get("total_logicals")
        optimal = record.get("logicals_optimal")
        if (
            record.get("milp_exact") is True
            and isinstance(milp_distance, int)
            and not isinstance(milp_distance, bool)
            and milp_distance == target.expected_distance
            and isinstance(total, int)
            and not isinstance(total, bool)
            and isinstance(optimal, int)
            and not isinstance(optimal, bool)
            and total > 0
            and optimal == total
        ):
            exact_rows.append(row)
    if len(exact_rows) != 1:
        raise AssertionError(
            f"expected one exact parent certificate for {target.presentation_id}, "
            f"found {len(exact_rows)}"
        )
    row = exact_rows[0]
    record = row["record"]
    return {
        "path": VERIFICATION_RELATIVE_PATH.as_posix(),
        "line_number": row["line_number"],
        "canonical_record_sha256": row["canonical_record_sha256"],
        "method": "all-logical MILP optimum",
        "milp_d": int(record["milp_d"]),
        "logicals_optimal": int(record["logicals_optimal"]),
        "total_logicals": int(record["total_logicals"]),
    }


def _validate_weight_five(item: ExplicitPbbRepresentative) -> None:
    support_a = set(item.A)
    support_b = set(item.B)
    support_c = set(item.C)
    support_d = set(item.D)
    if not support_c <= support_a or not support_d <= support_b:
        raise AssertionError("explicit PBB violates perturbation support containment")
    mixed_weight = len(support_a | support_c) + len(support_b | support_d)
    second_weight = len(support_a) + len(support_b)
    if (mixed_weight, second_weight) != (5, 5):
        raise AssertionError(
            f"explicit PBB does not have row weights five: "
            f"{mixed_weight}, {second_weight}"
        )


def _local_clifford_css_screen(item: ExplicitPbbRepresentative, code) -> dict:
    """Return a compact, deterministic summary of the implemented LC screens.

    This deliberately records the tested scope rather than calling the result
    a complete local-Clifford non-equivalence certificate.  In particular,
    arbitrary nonuniform assignments from the ``{SH,HSH}`` macro-class and
    assignments mixing macro-classes remain outside the implemented checks.
    """
    result = passes_noncss_gate(
        item.ell,
        item.m,
        list(item.A),
        list(item.B),
        list(item.C),
        list(item.D),
        code,
    )
    lc_details = result["lc_css_details"]
    uniform = lc_details["uniform_bruteforce"]
    nonuniform = lc_details["nonuniform_exact"]
    hadamard = result["css_details"]
    if hadamard is None:
        raise AssertionError("Hadamard screen unexpectedly short-circuited")
    if len(uniform["all_results"]) != 36:
        raise AssertionError("uniform LC screen did not test all 36 assignments")
    if (
        result["passes"] is not True
        or result["is_lc_css"] is not False
        or result["is_css"] is not False
        or uniform["group_css_assignments"]
        or nonuniform["consistent"] is not False
    ):
        raise AssertionError(
            "explicit PBB component failed the implemented non-CSS gate"
        )
    return {
        "passes_implemented_checks": True,
        "css_equivalent_under_tested_classes": False,
        "uniform_blockwise": {
            "assignments_tested": 36,
            "group_css_assignments": [],
        },
        "nonuniform_IS_or_HHS": {
            "affine_system_consistent": False,
            "constraint_rank": int(nonuniform["constraint_rank"]),
            "num_variables": int(nonuniform["num_vars"]),
        },
        "nonuniform_hadamard_on_stored_generators": {
            "is_css": False,
            "reason": str(hadamard["reason"]),
            "num_y_incidents": int(hadamard["num_y_qubits"]),
        },
        "scope": {
            "covered": (
                "all 36 uniform per-block single-qubit Clifford assignments; "
                "all nonuniform assignments within the {I,S} and {H,HS} "
                "macro-classes; arbitrary per-qubit Hadamards on the stored "
                "generators"
            ),
            "not_covered": (
                "nonuniform {SH,HSH} assignments and assignments mixing "
                "Clifford macro-classes"
            ),
            "complete_local_clifford_orbit_test": False,
        },
    }


def _explicit_representative_row(
    item: ExplicitPbbRepresentative,
    raw_index: dict[SpecKey, list[dict]],
    verification_index: dict[SpecKey, list[dict]],
) -> dict:
    target = item.parent
    parent = build_pbb_code(
        target.ell,
        target.m,
        list(target.A),
        list(target.B),
        list(target.C),
        list(target.D),
    )
    checks = _component_checks(target, parent)
    induced = checks["components"][0]

    _validate_weight_five(item)
    representative = build_pbb_code(
        item.ell,
        item.m,
        list(item.A),
        list(item.B),
        list(item.C),
        list(item.D),
    )
    rep_decomposition = decompose(representative)
    if not rep_decomposition.is_connected:
        raise AssertionError("explicit PBB component representative is disconnected")
    if (int(representative.num_qudits), int(representative.dimension)) != (
        target.expected_component_n,
        target.expected_component_k,
    ):
        raise AssertionError("explicit PBB component representative has wrong parameters")
    if canonical_hash_noncss(induced) != canonical_hash_noncss(representative):
        raise AssertionError("explicit PBB representative does not BLISS-match component")

    digest = canonical_digest_noncss(representative)
    if digest != checks["component_digests"][0]:
        raise AssertionError("canonical digest mismatch after positive BLISS comparison")

    lc_screen = _local_clifford_css_screen(item, representative)

    decomposition = checks["decomposition"]
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "explicit_pbb_component_match",
        "match_id": (
            f"W5PBB-{target.expected_component_n}-{target.expected_component_k}-"
            f"{target.expected_distance}-{target.presentation_id.removeprefix('PL-')}"
        ),
        "parent_source_identity": _source_identity(target, raw_index),
        "parent_exact_distance_source": _exact_source_identity(
            target, verification_index
        ),
        "parent_spec": {
            "family": "PBB",
            "ell": target.ell,
            "m": target.m,
            "A_terms": [list(term) for term in target.A],
            "B_terms": [list(term) for term in target.B],
            "C_terms": [list(term) for term in target.C],
            "D_terms": [list(term) for term in target.D],
            "n": target.expected_n,
            "k": target.expected_k,
            "d_exact": target.expected_distance,
            "n_components": target.expected_multiplicity,
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
            "component_rowspace_digest_sha256": checks["rowspace_digests"][0],
            "literal_component_rowspaces_identical_before_relabeling": checks[
                "literal_rowspaces_identical"
            ],
        },
        "explicit_pbb_component": {
            "ell": item.ell,
            "m": item.m,
            "A_terms": [list(term) for term in item.A],
            "B_terms": [list(term) for term in item.B],
            "C_terms": [list(term) for term in item.C],
            "D_terms": [list(term) for term in item.D],
            "n": target.expected_component_n,
            "k": target.expected_component_k,
            "d_exact": target.expected_distance,
            "connected": True,
            "stabilizer_weight": 5,
        },
        "local_clifford_css_screen": lc_screen,
        "match": {
            "equivalence_relation": (
                "colored_stored_generator_tanner_isomorphism_v1"
            ),
            "bliss_isomorphic": True,
            "canonical_digest_sha256": digest,
            "distance_transfer_valid": True,
            "distance_transfer_reason": (
                "The MILP-certified parent is a direct sum of pairwise "
                "BLISS-isomorphic components, and this connected PBB "
                "presentation BLISS-matches one induced component."
            ),
        },
    }


def build_artifact_rows() -> list[dict]:
    raw_path = ROOT / RAW_RELATIVE_PATH
    verification_path = ROOT / VERIFICATION_RELATIVE_PATH
    raw_index = _load_jsonl_index(raw_path)
    verification_index = _load_jsonl_index(verification_path)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "artifact_manifest",
        "artifact": "weight5_pbb_component_certifications",
        "generator": "scripts/certify_weight5_pbb_components.py",
        "semantics": (
            "Explicit connected PBB presentations inherit exact distance only "
            "after an exact parent MILP certificate, a homogeneous isomorphic "
            "direct-sum audit, and a positive colored-BLISS component match. "
            "Each presentation is also rerun through the implemented, explicitly "
            "scoped local-Clifford-to-CSS screens."
        ),
        "source_files": {
            RAW_RELATIVE_PATH.as_posix(): _sha256_file(raw_path),
            VERIFICATION_RELATIVE_PATH.as_posix(): _sha256_file(verification_path),
        },
        "software": {
            "qldpc": distribution_version("qldpc"),
            "python-igraph": distribution_version("igraph"),
        },
        "record_counts": {
            "explicit_pbb_component_match": len(EXPLICIT_PBB_REPRESENTATIVES),
        },
    }
    representatives = [
        _explicit_representative_row(item, raw_index, verification_index)
        for item in sorted(
            EXPLICIT_PBB_REPRESENTATIVES,
            key=lambda item: (
                item.parent.expected_component_n,
                item.parent.presentation_id,
            ),
        )
    ]
    return [manifest, *representatives]


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
            print(f"stale PBB component-certification artifact: {args.output}", file=sys.stderr)
            return 1
        print(f"PBB component-certification artifact is current: {args.output}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(
        f"wrote {len(EXPLICIT_PBB_REPRESENTATIVES)} explicit PBB matches "
        f"to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
