#!/usr/bin/env python3
"""Replace loose CSS campaign upper endpoints with explicit witnesses.

Nine CSS presentations carried a decisive upper endpoint supported only by
a stored BP--OSD-family scalar.  A tenth presentation had a tighter
nonoptimal MILP incumbent.  Independent randomized GF(2) kernel/permutation
searches find a strictly lighter logical operator for every presentation.
Each candidate is then re-verified with the bitmask-int GF(2) routines used by
:mod:`scripts.audit_weight5_css_low_weight`, so search and acceptance use
separate implementations.  These witnesses tighten upper bounds only; the
independent exhaustive weight-four audit supplies the lower bound ``d >= 5``.

Search-history note: an early version of the pairwise-combination search
combined two rows that were each individually confirmed to be outside the
stabilizer row space, without re-checking that their XOR combination was
*also* outside it.  Two such rows can be representatives of the same
logical coset (differing only by a stabilizer element), and their XOR then
cancels back into the row space -- producing a spuriously light but invalid
candidate.  The search below re-verifies every combined candidate before
accepting it; :func:`_verified_witness` then independently re-confirms the
final choice via the unrelated bitmask-int implementation.

The checked-in artifact still contains a fixed, explicitly recorded
pseudo-random seed per sector (there is no combinatorially exhaustive
alternative at this weight), so ``--check`` reproducibility depends on
``numpy.random.default_rng``'s bit-stream stability across numpy versions,
unlike the purely combinatorial determinism of the other audit scripts in
this suite.

This artifact's own correction is merged back into the publication
catalogue (``scripts/build_weight5_publication_catalogue.py``), so it must
not depend on that generated catalogue itself -- doing so would make the
catalogue depend on an artifact that depends on the catalogue's own prior
(uncorrected) content, and the dependency would go stale the moment the
catalogue is regenerated with the correction applied. Instead, like
:mod:`scripts.audit_weight5_css_low_weight`, this script rebuilds specs
directly from the raw campaign sources via
``generate_weight5_supplement.build_catalogue(...)`` (with its own merge
disabled), so its true data dependencies are the raw campaign logs, their
verification and attribution sidecars, the component certificates, and the
weight-four low-weight audit -- never the generated catalogue.

Usage::

    python scripts/audit_weight5_css_upper_bound_witnesses.py
    python scripts/audit_weight5_css_upper_bound_witnesses.py --check
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
LOW_WEIGHT_AUDIT = ROOT / "results" / "weight5_css_low_weight_audit.jsonl"
DEFAULT_OUTPUT = (
    ROOT / "results" / "weight5_css_upper_bound_witnesses.jsonl"
)

SCHEMA_VERSION = 1
ARTIFACT_SCHEMA = "weight5_css_upper_bound_witnesses_v1"
ALGORITHM = "randomized_gf2_kernel_permutation_search_v1"

DEFAULT_SEARCH_TRIALS = 400
SEARCH_SEED_X = 1000
SEARCH_SEED_Z = 2000
KEEP_LOGICAL = 8
KEEP_GAUGE = 8


@dataclass(frozen=True)
class Target:
    prior_method: str
    prior_value: int
    corrected_value: int
    search_trials: int = DEFAULT_SEARCH_TRIALS


sys.path.insert(0, str(ROOT))
import scripts.generate_weight5_supplement as supplement  # noqa: E402
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


TARGETS = {
    presentation_id: Target(
        prior_method,
        prior_value,
        corrected_value,
        search_trials=(
            8_000
            if presentation_id == "CL-f5754da6"
            else DEFAULT_SEARCH_TRIALS
        ),
    )
    for presentation_id, (
        prior_method,
        prior_value,
        corrected_value,
    ) in supplement.EXPECTED_CSS_UPPER_BOUND_WITNESSES.items()
}


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


def _spec_key(spec: supplement.Spec) -> tuple[object, ...]:
    return (
        int(spec.ell),
        int(spec.m),
        tuple(sorted(tuple(term) for term in spec.A)),
        tuple(sorted(tuple(term) for term in spec.B)),
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


def _index_observations(
    paths: Sequence[Path],
) -> dict[tuple[object, ...], list[dict[str, Any]]]:
    index: dict[tuple[object, ...], list[dict[str, Any]]] = {}
    for path in paths:
        for observation in _load_jsonl(path):
            key = _raw_key(observation)
            if key is not None:
                index.setdefault(key, []).append(observation)
    return index


def _low_weight_by_class() -> dict[str, dict[str, Any]]:
    rows = _load_jsonl(LOW_WEIGHT_AUDIT)
    return {
        row["class_identity"]["class_id"]: row
        for row in rows
        if row.get("record_type") == "css_low_weight_class_audit"
    }


# --- GF(2) linear algebra over dense bit matrices (the search engine) ------


def _bitmask_rows_to_matrix(rows: Sequence[int], n: int) -> np.ndarray:
    matrix = np.zeros((len(rows), n), dtype=np.uint8)
    for row_index, row in enumerate(rows):
        pending = int(row)
        while pending:
            least_bit = pending & -pending
            matrix[row_index, least_bit.bit_length() - 1] = 1
            pending ^= least_bit
    return matrix


def _rref_gf2(matrix: np.ndarray) -> tuple[np.ndarray, list[int]]:
    reduced = (matrix.copy() % 2).astype(np.uint8)
    num_rows, num_cols = reduced.shape
    pivot_row = 0
    pivot_columns: list[int] = []
    for column in range(num_cols):
        if pivot_row >= num_rows:
            break
        nonzero = np.nonzero(reduced[pivot_row:, column])[0]
        if nonzero.size == 0:
            continue
        found = pivot_row + int(nonzero[0])
        if found != pivot_row:
            reduced[[pivot_row, found]] = reduced[[found, pivot_row]]
        mask = reduced[:, column].astype(bool)
        mask[pivot_row] = False
        reduced[mask] ^= reduced[pivot_row]
        pivot_columns.append(column)
        pivot_row += 1
    return reduced[:pivot_row].copy(), pivot_columns


def _kernel_basis_gf2(matrix: np.ndarray, n: int) -> np.ndarray:
    reduced, pivot_columns = _rref_gf2(matrix)
    pivot_set = set(pivot_columns)
    free_columns = [column for column in range(n) if column not in pivot_set]
    basis = np.zeros((len(free_columns), n), dtype=np.uint8)
    for basis_index, free_column in enumerate(free_columns):
        basis[basis_index, free_column] = 1
        for pivot_index, pivot_column in enumerate(pivot_columns):
            basis[basis_index, pivot_column] = reduced[pivot_index, free_column]
    return basis


def _reduce_many_against_rref(
    vectors: np.ndarray, rref: np.ndarray, pivot_columns: list[int]
) -> np.ndarray:
    reduced = vectors.copy()
    for pivot_index, pivot_column in enumerate(pivot_columns):
        mask = reduced[:, pivot_column].astype(bool)
        if mask.any():
            reduced[mask] ^= rref[pivot_index]
    return reduced


def _not_in_rowspace(
    vector: np.ndarray, rref: np.ndarray, pivot_columns: list[int]
) -> bool:
    reduced = vector.copy()
    for pivot_index, pivot_column in enumerate(pivot_columns):
        if reduced[pivot_column]:
            reduced = reduced ^ rref[pivot_index]
    return bool(reduced.any())


def _search_lightest_witness(
    kernel_basis: np.ndarray,
    self_rref: np.ndarray,
    self_pivots: list[int],
    n: int,
    trials: int,
    seed: int,
) -> tuple[int | None, np.ndarray | None]:
    """Randomized search for a lightweight vector in ``kernel_basis``'s span
    that lies outside ``row(self_rref)`` (i.e. a genuine logical operator).

    Each trial re-derives a fresh RREF of the kernel basis under a random
    column permutation (favoring different, usually sparser, generators),
    keeps the lightest rows, and tries every accepted row alone plus every
    pairwise XOR combination among the kept logical rows and kept
    stabilizer-rowspace ("gauge") rows.  Every pairwise candidate is
    re-verified with :func:`_not_in_rowspace` before being accepted --
    without this, two logical-flagged rows from the same coset can cancel
    back into the row space and produce an invalid but spuriously light
    candidate (see module docstring).
    """
    rng = np.random.default_rng(seed)
    best_weight: int | None = None
    best_vector: np.ndarray | None = None
    for _ in range(trials):
        permutation = rng.permutation(n)
        permuted_basis = kernel_basis[:, permutation]
        permuted_rref, _ = _rref_gf2(permuted_basis)
        restored = np.zeros_like(permuted_rref)
        restored[:, permutation] = permuted_rref

        reduced = _reduce_many_against_rref(restored, self_rref, self_pivots)
        is_logical = reduced.any(axis=1)
        weights = restored.sum(axis=1, dtype=np.int64)

        logical_indices = np.nonzero(is_logical)[0]
        gauge_indices = np.nonzero(~is_logical)[0]
        if logical_indices.size == 0:
            continue
        logical_order = logical_indices[
            np.argsort(weights[logical_indices])[:KEEP_LOGICAL]
        ]
        gauge_order = (
            gauge_indices[np.argsort(weights[gauge_indices])[:KEEP_GAUGE]]
            if gauge_indices.size
            else np.array([], dtype=int)
        )
        logical_rows = restored[logical_order]
        logical_weights = weights[logical_order]
        gauge_rows = restored[gauge_order]

        for i in range(len(logical_order)):
            weight = int(logical_weights[i])
            if weight == 0:
                continue
            if best_weight is None or weight < best_weight:
                best_weight, best_vector = weight, logical_rows[i].copy()

        for i in range(len(logical_order)):
            for j in range(i + 1, len(logical_order)):
                candidate = logical_rows[i] ^ logical_rows[j]
                weight = int(candidate.sum(dtype=np.int64))
                if weight == 0 or (best_weight is not None and weight >= best_weight):
                    continue
                if not _not_in_rowspace(candidate, self_rref, self_pivots):
                    continue
                best_weight, best_vector = weight, candidate.copy()

        for i in range(len(logical_order)):
            for g in range(len(gauge_order)):
                candidate = logical_rows[i] ^ gauge_rows[g]
                weight = int(candidate.sum(dtype=np.int64))
                if weight == 0 or (best_weight is not None and weight >= best_weight):
                    continue
                if not _not_in_rowspace(candidate, self_rref, self_pivots):
                    continue
                best_weight, best_vector = weight, candidate.copy()
    return best_weight, best_vector


def _find_witness(
    rows_opposite: Sequence[int],
    rows_self: Sequence[int],
    n: int,
    ell: int,
    m: int,
    sector: str,
    trials: int,
    seed: int,
) -> dict[str, Any]:
    """Find, then independently re-verify, one lightweight sector witness.

    ``rows_opposite``/``rows_self`` follow the same convention as elsewhere
    in this audit suite: a sector-``X`` witness must commute with ``H_Z``
    (``rows_opposite``) and lie outside ``row(H_X)`` (``rows_self``).

    The search itself runs on a dense numpy bit matrix; the acceptance
    check reruns entirely on the bitmask-int implementation from
    :mod:`scripts.audit_weight5_css_low_weight` (``_columns_from_rows``,
    ``_rowspace_remainder_columns``, ``_verified_witness``), so the two
    steps are algorithmically independent of each other.
    """
    opposite_matrix = _bitmask_rows_to_matrix(rows_opposite, n)
    self_matrix = _bitmask_rows_to_matrix(rows_self, n)
    kernel_basis = _kernel_basis_gf2(opposite_matrix, n)
    self_rref, self_pivots = _rref_gf2(self_matrix)

    weight, vector = _search_lightest_witness(
        kernel_basis, self_rref, self_pivots, n, trials, seed
    )
    if vector is None:
        raise ValueError(f"no witness found in {trials} trials for sector {sector}")

    support_indices = sorted(int(i) for i in np.nonzero(vector)[0])
    support_mask = 0
    for index in support_indices:
        support_mask |= 1 << index

    check_columns = _columns_from_rows(rows_opposite, n)
    remainder_columns, stabilizer_rank = _rowspace_remainder_columns(rows_self, n)
    _verified_witness(support_mask, check_columns, remainder_columns)

    return {
        "operator_type": sector,
        "qubit_indices_zero_based": support_indices,
        "qubits": _qubit_labels(support_indices, ell, m),
        "weight": int(weight),
        "stabilizer_rank": stabilizer_rank,
        "zero_check_syndrome": _xor_columns(check_columns, support_mask) == 0,
        "outside_stabilizer_rowspace": (
            _xor_columns(remainder_columns, support_mask) != 0
        ),
        "weight_matches_support_len": len(support_indices) == int(weight),
        "search_trials": trials,
        "search_seed": seed,
    }


def _audit_row(
    spec: supplement.Spec,
    target: Target,
    class_id: str,
    low_weight: dict[str, dict[str, Any]],
    source_index: dict[tuple[object, ...], list[dict[str, Any]]],
) -> dict[str, Any]:
    ell, m = int(spec.ell), int(spec.m)
    num_qubits = int(spec.n)

    lower_audit = low_weight[class_id]
    if lower_audit["audit_evidence"] != {
        "result": "excluded_through_weight_4",
        "exact_distance": None,
        "certified_lower_bound": 5,
    }:
        raise ValueError(f"{spec.spec_id}: missing L4 lower bound")

    if (target.prior_value, target.prior_method) not in spec.upper_evidence:
        raise ValueError(
            f"{spec.spec_id}: missing prior {target.prior_method}="
            f"{target.prior_value} evidence"
        )
    _, current_upper, _ = spec.final_bounds()
    if target.prior_value != int(current_upper):
        raise ValueError(
            f"{spec.spec_id}: expected current upper {target.prior_value}, "
            f"found {current_upper}"
        )

    observations = [
        item
        for item in source_index.get(_spec_key(spec), [])
        if int(item.get("d", -1)) == target.prior_value
    ]
    if not observations:
        raise ValueError(
            f"{spec.spec_id}: no matching historical record for "
            f"d={target.prior_value}"
        )
    observed_stages = sorted({item.get("stage") for item in observations})
    expected_stages = (
        {"refined_estimate", "osd_cs_verified"}
        if target.prior_method == "B"
        else {"milp_incumbent"}
    )
    if any(stage not in expected_stages for stage in observed_stages):
        raise ValueError(
            f"{spec.spec_id}: unexpected evaluation stage(s) "
            f"{observed_stages}"
        )
    if target.prior_method == "I" and any(
        item.get("d_is_exact") is not False
        or (item.get("milp_details") or {}).get("exact") is not False
        for item in observations
    ):
        raise ValueError(f"{spec.spec_id}: prior MILP evidence is not an incumbent")
    witness_like_keys = {
        key
        for observation in observations
        for key in observation
        if "witness" in key.lower() or key in {"L", "logical_operator"}
    }
    if witness_like_keys:
        raise ValueError(
            f"{spec.spec_id}: unexpected persisted witness keys "
            f"{sorted(witness_like_keys)}"
        )

    generators = {
        "A_terms": [list(term) for term in spec.A],
        "B_terms": [list(term) for term in spec.B],
    }
    rows_x, rows_z = build_bb_check_rows(
        ell, m, generators["A_terms"], generators["B_terms"]
    )
    witness_x = _find_witness(
        rows_z,
        rows_x,
        num_qubits,
        ell,
        m,
        "X",
        target.search_trials,
        SEARCH_SEED_X,
    )
    witness_z = _find_witness(
        rows_x,
        rows_z,
        num_qubits,
        ell,
        m,
        "Z",
        target.search_trials,
        SEARCH_SEED_Z,
    )

    rebuilt_k = num_qubits - witness_x["stabilizer_rank"] - witness_z["stabilizer_rank"]
    if rebuilt_k != int(spec.k):
        raise ValueError(
            f"{spec.spec_id}: rebuilt k={rebuilt_k} != {spec.k}"
        )

    corrected_value = min(witness_x["weight"], witness_z["weight"])
    if corrected_value != target.corrected_value:
        raise ValueError(
            f"{spec.spec_id}: expected corrected upper bound "
            f"{target.corrected_value}, found {corrected_value}"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "css_upper_bound_witness_correction",
        "algorithm": ALGORITHM,
        "presentation_id": spec.spec_id,
        "class_id": class_id,
        "parameters": {
            "ell": ell,
            "m": m,
            "n": num_qubits,
            "k_catalogue": int(spec.k),
            "k_rebuilt": rebuilt_k,
        },
        "generators": generators,
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
            "method": target.prior_method,
            "stages_observed": observed_stages,
            "value": target.prior_value,
            "matching_scalar_observations": len(observations),
            "operator_was_persisted": False,
        },
        "corrected_upper_bound": {
            "method": "W",
            "value": corrected_value,
        },
        "replacement_explicit_witnesses": {
            "X": witness_x,
            "Z": witness_z,
        },
        "conclusion": (
            f"The lighter independently validated CSS logical witness has "
            f"weight {corrected_value}, strictly below the prior "
            f"{target.prior_method} endpoint {target.prior_value}. Together "
            f"with exhaustive exclusion through weight four, this tightens "
            f"the previously reported endpoint range [5, "
            f"{target.prior_value}] to the witnessed interval [5, "
            f"{corrected_value}]."
        ),
    }


def build_records() -> list[dict[str, Any]]:
    campaign_data = supplement.build_catalogue(
        include_css_low_weight_audit=False,
        include_css_weight5_witnesses=False,
        include_css_upper_bound_witnesses=False,
    )
    all_specs = [spec for _, specs, _, _ in campaign_data for spec in specs]
    specs_by_id: dict[str, supplement.Spec] = {}
    for spec in all_specs:
        if spec.spec_id in specs_by_id:
            raise ValueError(f"duplicate spec_id in retained catalogue: {spec.spec_id}")
        specs_by_id[spec.spec_id] = spec
    supplement.load_and_merge_css_low_weight_audit(specs_by_id)
    missing = [pid for pid in TARGETS if pid not in specs_by_id]
    if missing:
        raise ValueError(f"target presentation(s) not found in catalogue: {missing}")
    targets = sorted(
        ((specs_by_id[pid], target) for pid, target in TARGETS.items()),
        key=lambda item: item[0].spec_id,
    )
    if len(targets) != len(TARGETS):
        raise ValueError(
            f"expected {len(TARGETS)} target rows, found {len(targets)}"
        )

    class_of = {
        spec.spec_id: (
            f"TC-{supplement.connectivity_info(spec).presentation_digest[:10]}"
        )
        for spec, _ in targets
    }

    low_weight = _low_weight_by_class()
    observation_paths = sorted(
        {
            spec.campaign.raw_path
            if target.prior_method == "B"
            else spec.campaign.verification_path
            for spec, target in targets
        }
    )
    source_index = _index_observations(observation_paths)
    audits = [
        _audit_row(
            spec,
            target,
            class_of[spec.spec_id],
            low_weight,
            source_index,
        )
        for spec, target in targets
    ]

    catalogue_inputs = sorted(
        {
            path
            for campaign in supplement.CAMPAIGNS
            for path in (
                campaign.raw_path,
                campaign.event_path,
                campaign.verification_path,
                campaign.attribution_path,
            )
            if path is not None
        }
        | {supplement.COMPONENT_CERTIFICATION_PATH}
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "artifact_manifest",
        "artifact_schema": ARTIFACT_SCHEMA,
        "generator": (
            "scripts/audit_weight5_css_upper_bound_witnesses.py"
        ),
        "algorithm": ALGORITHM,
        "semantics": (
            "Explicit independently reconstructed X- and Z-sector logical "
            "witnesses that strictly tighten ten CSS proposal-level upper "
            "endpoints. Nine supersede decoder scalars and one tightens a "
            "nonoptimal MILP incumbent; none establishes exact distance."
        ),
        "decoder_scalar_provenance": (
            "The refined_estimate and osd_cs_verified evaluation stages "
            "record only a decoder-heuristic scalar distance, with no "
            "logical operator. This artifact supplies independently verified "
            "explicit witnesses for the nine targeted decoder endpoints."
        ),
        "counts": {
            "presentations": len(audits),
            "presentation_classes": len({row["class_id"] for row in audits}),
            "explicit_witnesses": 2 * len(audits),
            "prior_methods": {
                method: sum(
                    row["historical_upper_bound"]["method"] == method
                    for row in audits
                )
                for method in ("B", "I")
            },
            "all_x_witnesses_valid": all(
                row["replacement_explicit_witnesses"]["X"]["zero_check_syndrome"]
                and row["replacement_explicit_witnesses"]["X"][
                    "outside_stabilizer_rowspace"
                ]
                for row in audits
            ),
            "all_z_witnesses_valid": all(
                row["replacement_explicit_witnesses"]["Z"]["zero_check_syndrome"]
                and row["replacement_explicit_witnesses"]["Z"][
                    "outside_stabilizer_rowspace"
                ]
                for row in audits
            ),
        },
        "source_artifacts": {
            "low_weight_audit": {
                "path": _relative(LOW_WEIGHT_AUDIT),
                "sha256": _sha256(LOW_WEIGHT_AUDIT),
            },
            "catalogue_inputs": [
                {"path": _relative(path), "sha256": _sha256(path)}
                for path in catalogue_inputs
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
