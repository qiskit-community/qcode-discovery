#!/usr/bin/env python3
"""Deterministically audit every direct-U CSS row through weight four.

The pre-audit normalized campaign snapshot contains 858 retained CSS
presentations with an upper endpoint but no certified positive lower bound
(status ``U``).  This script groups those presentations by colored
stored-generator Tanner isomorphism, rebuilds one representative of each
class directly over ``Z_ell x Z_m``, and exhaustively determines whether
either CSS logical sector contains an operator of weight at most four.

For a CSS sector with check matrix ``H`` and same-type stabilizer matrix
``S``, a support ``v`` is a nontrivial logical exactly when

    H v = 0  and  v is not in row(S).

Every support of size at most four is the symmetric difference of two
supports of size at most two.  We therefore hash the pair

    (H v, v mod row(S))

for the empty support, all singletons, and all unordered pairs.  Equal check
syndromes with unequal row-space remainders give a logical.  The search is
performed in three ordered phases (weights <=2, then 3, then 4), so a found
witness is accompanied by exhaustive exclusion of every smaller weight.  If
no witness is found, the sector has certified distance at least five.

The implementation uses Python integers as bit vectors.  It is deterministic,
has no solver or randomized dependency, and reconstructs the BB check rows
without invoking a decoder.  The output has no timestamp or runtime field, so
``--check`` is byte-for-byte reproducible; elapsed time is printed separately.

Usage:
    python scripts/audit_weight5_css_low_weight.py
    python scripts/audit_weight5_css_low_weight.py --check
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "results" / "weight5_css_low_weight_audit.jsonl"

SCHEMA_VERSION = 1
ARTIFACT_SCHEMA = "weight5_css_low_weight_audit_v1"
ALGORITHM = "css_weight_le_4_syndrome_quotient_mitm_v1"
EQUIVALENCE_RELATION = "colored_stored_generator_tanner_isomorphism_v1"

sys.path.insert(0, str(ROOT))
from scripts import generate_weight5_supplement as supplement  # noqa: E402


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _json_line(value: object) -> str:
    return _canonical_json(value).decode("ascii")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def _group_index(x_exp: int, y_exp: int, m: int) -> int:
    return x_exp * m + y_exp


def _shift_index(
    index: int, dx: int, dy: int, ell: int, m: int
) -> int:
    x_exp, y_exp = divmod(index, m)
    return _group_index((x_exp + dx) % ell, (y_exp + dy) % m, m)


def _validated_terms(
    terms: Iterable[Iterable[object]], ell: int, m: int, label: str
) -> tuple[tuple[int, int], ...]:
    normalized: list[tuple[int, int]] = []
    for term in terms:
        try:
            x_raw, y_raw = term
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} terms must be exponent pairs") from exc
        if isinstance(x_raw, bool) or isinstance(y_raw, bool):
            raise ValueError(f"{label} exponents must be integers")
        try:
            x_exp, y_exp = int(x_raw), int(y_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} exponents must be integers") from exc
        if x_exp != x_raw or y_exp != y_raw:
            raise ValueError(f"{label} exponents must be integers")
        if not (0 <= x_exp < ell and 0 <= y_exp < m):
            raise ValueError(
                f"{label} exponent {(x_exp, y_exp)} outside Z_{ell} x Z_{m}"
            )
        normalized.append((x_exp, y_exp))
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} contains duplicate monomials")
    return tuple(sorted(normalized))


def build_bb_check_rows(
    ell: int,
    m: int,
    A_terms: Iterable[Iterable[object]],
    B_terms: Iterable[Iterable[object]],
) -> tuple[list[int], list[int]]:
    """Return integer-bit rows of ``H_X=[A|B]`` and ``H_Z=[B^T|A^T]``.

    A monomial ``x^a y^b`` sends group index ``g`` to ``g+(a,b)`` in
    the regular representation.  Transposition reverses that shift.
    """
    if isinstance(ell, bool) or isinstance(m, bool):
        raise ValueError("lattice dimensions must be positive integers")
    ell, m = int(ell), int(m)
    if ell <= 0 or m <= 0:
        raise ValueError("lattice dimensions must be positive integers")
    A = _validated_terms(A_terms, ell, m, "A")
    B = _validated_terms(B_terms, ell, m, "B")
    if len(A) + len(B) != 5:
        raise ValueError(
            f"weight-five BB presentation expected, got {len(A) + len(B)} terms"
        )

    group_order = ell * m
    rows_x: list[int] = []
    rows_z: list[int] = []
    for group_element in range(group_order):
        row_x = 0
        for dx, dy in A:
            row_x ^= 1 << _shift_index(group_element, dx, dy, ell, m)
        for dx, dy in B:
            row_x ^= 1 << (
                group_order + _shift_index(group_element, dx, dy, ell, m)
            )
        rows_x.append(row_x)

        row_z = 0
        for dx, dy in B:
            row_z ^= 1 << _shift_index(group_element, -dx, -dy, ell, m)
        for dx, dy in A:
            row_z ^= 1 << (
                group_order + _shift_index(group_element, -dx, -dy, ell, m)
            )
        rows_z.append(row_z)
    return rows_x, rows_z


def _gf2_rref_int(rows: Iterable[int], num_columns: int) -> tuple[list[int], list[int]]:
    """Canonical GF(2) RREF for rows represented by nonnegative integers."""
    if num_columns < 0:
        raise ValueError("num_columns must be nonnegative")
    column_mask = (1 << num_columns) - 1
    reduced = [int(row) & column_mask for row in rows if int(row) & column_mask]
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(num_columns):
        bit = 1 << column
        found = next(
            (index for index in range(pivot_row, len(reduced)) if reduced[index] & bit),
            None,
        )
        if found is None:
            continue
        reduced[pivot_row], reduced[found] = reduced[found], reduced[pivot_row]
        pivot = reduced[pivot_row]
        for index, row in enumerate(reduced):
            if index != pivot_row and row & bit:
                reduced[index] = row ^ pivot
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == len(reduced):
            break
    basis = reduced[:pivot_row]
    if len(basis) != len(pivot_columns):
        raise AssertionError("RREF rank bookkeeping failed")
    return basis, pivot_columns


def _columns_from_rows(rows: Sequence[int], num_columns: int) -> list[int]:
    columns = [0] * num_columns
    for row_index, row in enumerate(rows):
        pending = int(row)
        while pending:
            least_bit = pending & -pending
            column = least_bit.bit_length() - 1
            if column >= num_columns:
                raise ValueError("row has a bit outside the declared matrix width")
            columns[column] |= 1 << row_index
            pending ^= least_bit
    return columns


def _rowspace_remainder_columns(
    stabilizer_rows: Sequence[int], num_columns: int
) -> tuple[list[int], int]:
    """Return ``e_j mod row(S)`` for every coordinate and ``rank(S)``."""
    basis, pivots = _gf2_rref_int(stabilizer_rows, num_columns)
    pivot_to_row = dict(zip(pivots, basis, strict=True))
    pivot_mask = sum(1 << pivot for pivot in pivots)
    remainders: list[int] = []
    for column in range(num_columns):
        unit = 1 << column
        remainder = (
            pivot_to_row[column] ^ unit if column in pivot_to_row else unit
        )
        if remainder & pivot_mask:
            raise AssertionError("RREF remainder retained a pivot coordinate")
        remainders.append(remainder)
    return remainders, len(pivots)


def _xor_columns(columns: Sequence[int], support_mask: int) -> int:
    value = 0
    pending = support_mask
    while pending:
        least_bit = pending & -pending
        value ^= columns[least_bit.bit_length() - 1]
        pending ^= least_bit
    return value


def _support_indices(support_mask: int) -> list[int]:
    indices: list[int] = []
    pending = support_mask
    while pending:
        least_bit = pending & -pending
        indices.append(least_bit.bit_length() - 1)
        pending ^= least_bit
    return indices


def _verified_witness(
    support_mask: int,
    check_columns: Sequence[int],
    remainder_columns: Sequence[int],
) -> None:
    if support_mask <= 0:
        raise AssertionError("a logical witness must have nonempty support")
    if _xor_columns(check_columns, support_mask) != 0:
        raise AssertionError("reported logical witness has nonzero check syndrome")
    if _xor_columns(remainder_columns, support_mask) == 0:
        raise AssertionError("reported logical witness lies in the stabilizer row space")


def audit_css_sector(
    check_rows: Sequence[int],
    stabilizer_rows: Sequence[int],
    num_qubits: int,
) -> dict[str, Any]:
    """Determine a CSS sector's exact distance if <=4, else prove it >=5."""
    check_columns = _columns_from_rows(check_rows, num_qubits)
    remainder_columns, stabilizer_rank = _rowspace_remainder_columns(
        stabilizer_rows, num_qubits
    )

    # Phase 1: empty/singleton halves.  A collision proves weight one or two.
    base_by_syndrome: dict[int, tuple[int, int]] = {0: (0, 0)}
    best_small: int | None = None
    for qubit in range(num_qubits):
        syndrome = check_columns[qubit]
        remainder = remainder_columns[qubit]
        support = 1 << qubit
        prior = base_by_syndrome.get(syndrome)
        if prior is not None and prior[0] != remainder:
            witness = support ^ prior[1]
            if best_small is None or witness.bit_count() < best_small.bit_count():
                best_small = witness
        else:
            base_by_syndrome.setdefault(syndrome, (remainder, support))
    if best_small is not None:
        _verified_witness(best_small, check_columns, remainder_columns)
        return {
            "result": "exact",
            "exact_distance": best_small.bit_count(),
            "certified_lower_bound": best_small.bit_count(),
            "witness_indices": _support_indices(best_small),
            "stabilizer_rank": stabilizer_rank,
            "singletons_examined": num_qubits,
            "pairs_examined": 0,
        }

    # Phases 2 and 3 share the pair enumeration.  Pair/singleton collisions
    # are weight three.  Pair/pair collisions are retained but cannot be
    # returned until the full scan has excluded every weight-three support.
    pair_by_syndrome: dict[int, tuple[int, int]] = {}
    witness_weight_four: int | None = None
    pairs_examined = 0
    for left in range(num_qubits):
        syndrome_left = check_columns[left]
        remainder_left = remainder_columns[left]
        left_bit = 1 << left
        for right in range(left + 1, num_qubits):
            pairs_examined += 1
            syndrome = syndrome_left ^ check_columns[right]
            remainder = remainder_left ^ remainder_columns[right]
            support = left_bit | (1 << right)

            base = base_by_syndrome.get(syndrome)
            if base is not None and base[0] != remainder:
                witness = support ^ base[1]
                _verified_witness(witness, check_columns, remainder_columns)
                if witness.bit_count() != 3:
                    raise AssertionError(
                        "pair/base collision had unexpected weight after <=2 exclusion"
                    )
                return {
                    "result": "exact",
                    "exact_distance": 3,
                    "certified_lower_bound": 3,
                    "witness_indices": _support_indices(witness),
                    "stabilizer_rank": stabilizer_rank,
                    "singletons_examined": num_qubits,
                    "pairs_examined": pairs_examined,
                }

            prior_pair = pair_by_syndrome.get(syndrome)
            if prior_pair is not None and prior_pair[0] != remainder:
                witness = support ^ prior_pair[1]
                if witness_weight_four is None:
                    witness_weight_four = witness
            else:
                pair_by_syndrome.setdefault(syndrome, (remainder, support))

    if witness_weight_four is not None:
        _verified_witness(
            witness_weight_four, check_columns, remainder_columns
        )
        if witness_weight_four.bit_count() != 4:
            raise AssertionError(
                "pair/pair collision had unexpected weight after <=2 exclusion"
            )
        return {
            "result": "exact",
            "exact_distance": 4,
            "certified_lower_bound": 4,
            "witness_indices": _support_indices(witness_weight_four),
            "stabilizer_rank": stabilizer_rank,
            "singletons_examined": num_qubits,
            "pairs_examined": pairs_examined,
        }

    return {
        "result": "lower_bound",
        "exact_distance": None,
        "certified_lower_bound": 5,
        "witness_indices": None,
        "stabilizer_rank": stabilizer_rank,
        "singletons_examined": num_qubits,
        "pairs_examined": pairs_examined,
    }


def audit_css_rows(
    rows_x: Sequence[int], rows_z: Sequence[int], num_qubits: int
) -> dict[str, Any]:
    """Audit both CSS sectors and return their combined distance evidence."""
    columns_x = _columns_from_rows(rows_x, num_qubits)
    columns_z = _columns_from_rows(rows_z, num_qubits)
    for row in rows_x:
        if _xor_columns(columns_z, row):
            raise ValueError("H_X H_Z^T is nonzero: supplied checks do not commute")
    for row in rows_z:
        if _xor_columns(columns_x, row):
            raise ValueError("H_Z H_X^T is nonzero: supplied checks do not commute")

    # An X-type operator is checked by H_Z and is trivial modulo row(H_X),
    # while a Z-type operator reverses those roles.
    sector_x = audit_css_sector(rows_z, rows_x, num_qubits)
    sector_z = audit_css_sector(rows_x, rows_z, num_qubits)
    sector_x["check_rank"] = sector_z["stabilizer_rank"]
    sector_z["check_rank"] = sector_x["stabilizer_rank"]
    exact_sector_values = [
        sector["exact_distance"]
        for sector in (sector_x, sector_z)
        if sector["exact_distance"] is not None
    ]
    if exact_sector_values:
        distance = min(exact_sector_values)
        return {
            "result": "exact_low_weight",
            "exact_distance": distance,
            "certified_lower_bound": distance,
            "sectors": {"X": sector_x, "Z": sector_z},
        }
    return {
        "result": "excluded_through_weight_4",
        "exact_distance": None,
        "certified_lower_bound": 5,
        "sectors": {"X": sector_x, "Z": sector_z},
    }


def _matrix_digest(rows: Sequence[int], num_qubits: int) -> str:
    width = (num_qubits + 7) // 8
    digest = hashlib.sha256()
    digest.update(_canonical_json({"rows": len(rows), "columns": num_qubits}))
    digest.update(b"\0")
    for row in rows:
        digest.update(int(row).to_bytes(width, "little"))
    return digest.hexdigest()


def _qubit_labels(indices: Sequence[int], ell: int, m: int) -> list[dict[str, Any]]:
    group_order = ell * m
    labels: list[dict[str, Any]] = []
    for index in indices:
        block_index, group_index = divmod(index, group_order)
        x_exp, y_exp = divmod(group_index, m)
        labels.append(
            {
                "index": index,
                "block": "left" if block_index == 0 else "right",
                "x": x_exp,
                "y": y_exp,
            }
        )
    return labels


def _decorate_sector(
    sector: dict[str, Any], name: str, ell: int, m: int
) -> dict[str, Any]:
    decorated = dict(sector)
    decorated["operator_type"] = name
    decorated["check_matrix"] = "H_Z" if name == "X" else "H_X"
    decorated["stabilizer_rowspace"] = "row(H_X)" if name == "X" else "row(H_Z)"
    indices = decorated.pop("witness_indices")
    decorated["witness"] = (
        None
        if indices is None
        else {
            "qubit_indices_zero_based": indices,
            "qubits": _qubit_labels(indices, ell, m),
        }
    )
    return decorated


def _baseline_catalogue_rows() -> list[dict[str, Any]]:
    """Rebuild the retained snapshot before this audit is merged.

    This deliberately calls the supplement normalizer with its low-weight
    merge disabled.  The audit is therefore reproducible from primary logs
    and independent certificates and never depends on its own downstream
    publication catalogue.
    """
    campaign_data = supplement.build_catalogue(
        include_css_low_weight_audit=False,
        include_css_weight5_witnesses=False,
        include_css_upper_bound_witnesses=False,
    )
    specs = [spec for _, retained, _, _ in campaign_data for spec in retained]
    classes = supplement.build_presentation_classes(specs)
    labels = supplement.presentation_class_labels(classes)
    rows: list[dict[str, Any]] = []
    for group in classes:
        member_ids = [spec.spec_id for spec in group.members]
        class_status = supplement.presentation_class_status(group)
        for spec in group.members:
            direct_lower, direct_upper, direct_exact = spec.final_bounds()
            direct_upper_is_supported = supplement.upper_endpoint_is_rigorous(
                spec.upper_evidence, direct_upper
            )
            rows.append(
                {
                    "presentation_id": spec.spec_id,
                    "family": spec.campaign.family,
                    "parameters": {
                        "ell": spec.ell,
                        "m": spec.m,
                        "n": spec.n,
                        "k": spec.k,
                    },
                    "generators": {
                        "A_terms": [list(term) for term in spec.A],
                        "B_terms": [list(term) for term in spec.B],
                    },
                    "distance_direct": {
                        "status_code": supplement.status_for(spec),
                        "lower": direct_lower or None,
                        "upper": direct_upper,
                        "is_exact": direct_exact,
                        "upper_is_supported": direct_upper_is_supported,
                    },
                    "distance_class": {
                        "status_code": class_status,
                        "lower": group.lower or None,
                        "upper": group.upper,
                        "is_exact": group.exact,
                        "upper_is_supported": group.upper_is_supported,
                        "inherited_or_tightened": (
                            (direct_lower, direct_upper, direct_exact)
                            != (group.lower, group.upper, group.exact)
                        ),
                    },
                    "equivalence": {
                        "relation": EQUIVALENCE_RELATION,
                        "class_id": group.class_id,
                        "class_label": labels[group.class_id],
                        "canonical_digest_sha256": group.digest,
                        "class_size": len(group.members),
                        "member_ids": member_ids,
                    },
                }
            )
    return rows


def _source_artifacts() -> dict[str, dict[str, Any]]:
    paths: dict[str, Path] = {
        "component_certifications": supplement.COMPONENT_CERTIFICATION_PATH,
        "source_normalizer": Path(supplement.__file__).resolve(),
    }
    for campaign in supplement.CAMPAIGNS:
        paths[f"{campaign.slug}:raw_log"] = campaign.raw_path
        paths[f"{campaign.slug}:distance_verification"] = campaign.verification_path
        if campaign.event_path is not None:
            paths[f"{campaign.slug}:discovery_events"] = campaign.event_path
        if campaign.attribution_path is not None:
            paths[f"{campaign.slug}:attribution_sidecar"] = campaign.attribution_path
    return {
        key: {"path": _relative(path), "sha256": _sha256_path(path)}
        for key, path in sorted(paths.items())
    }


def _class_groups(
    catalogue_rows: Sequence[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]]:
    all_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    targets_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in catalogue_rows:
        class_id = row["equivalence"]["class_id"]
        all_by_class[class_id].append(row)
        if row["family"] == "CSS" and row["distance_direct"]["status_code"] == "U":
            targets_by_class[class_id].append(row)

    groups = []
    for class_id in sorted(targets_by_class):
        members = sorted(
            all_by_class[class_id], key=lambda row: row["presentation_id"]
        )
        targets = sorted(
            targets_by_class[class_id], key=lambda row: row["presentation_id"]
        )
        declared = set(members[0]["equivalence"]["member_ids"])
        observed = {row["presentation_id"] for row in members}
        if declared != observed:
            raise ValueError(f"{class_id}: catalogue class membership is incomplete")
        if any(row["family"] != "CSS" for row in members):
            raise ValueError(f"{class_id}: a CSS class contains a non-CSS member")
        if any(
            row["equivalence"]["relation"] != EQUIVALENCE_RELATION
            for row in members
        ):
            raise ValueError(f"{class_id}: unexpected equivalence relation")
        # ``inherited_or_tightened`` is intentionally presentation-local;
        # every genuinely class-level distance field must agree.
        shared_distance_payloads = {
            _canonical_json(
                {
                    key: value
                    for key, value in row["distance_class"].items()
                    if key != "inherited_or_tightened"
                }
            )
            for row in members
        }
        if len(shared_distance_payloads) != 1:
            raise ValueError(f"{class_id}: members disagree on class evidence")
        groups.append((class_id, members, targets))
    return groups


def _post_audit_distance(
    source_distance: dict[str, Any], audit: dict[str, Any]
) -> dict[str, Any]:
    source_lower = source_distance["lower"] or 0
    source_upper = int(source_distance["upper"])
    source_exact = bool(source_distance["is_exact"])
    source_upper_is_supported = bool(source_distance["upper_is_supported"])
    audit_exact = audit["exact_distance"]
    if audit_exact is not None:
        if source_lower > audit_exact or source_upper < audit_exact:
            raise ValueError(
                f"low-weight exact distance {audit_exact} conflicts with source "
                f"interval [{source_lower or None},{source_upper}]"
            )
        lower = upper = audit_exact
        exact = True
        upper_is_supported = True
        evidence_method = "H"
    else:
        audit_lower = int(audit["certified_lower_bound"])
        if source_upper < audit_lower:
            raise ValueError(
                f"weight-four exclusion conflicts with source upper bound {source_upper}"
            )
        lower = max(source_lower, audit_lower)
        upper = source_upper
        exact = source_exact or (lower == upper and source_upper_is_supported)
        upper_is_supported = source_upper_is_supported
        if source_exact:
            lower = upper = source_upper
        evidence_method = "L4"
    return {
        "status_code": "E" if exact else "C",
        "status": (
            "exact"
            if exact
            else (
                "certified_interval"
                if upper_is_supported
                else "certified_lower_with_estimated_upper"
            )
        ),
        "lower": lower,
        "upper": upper,
        "is_exact": exact,
        "upper_is_supported": upper_is_supported,
        "new_evidence_method": evidence_method,
    }


def _audit_class(
    class_id: str,
    members: Sequence[dict[str, Any]],
    targets: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    # Pick a direct-U member so every computation is visibly rooted in the
    # population under review; class equivalence transfers it to all members.
    representative = targets[0]
    parameters = representative["parameters"]
    generators = representative["generators"]
    ell, m = int(parameters["ell"]), int(parameters["m"])
    num_qubits = int(parameters["n"])
    expected_k = int(parameters["k"])
    rows_x, rows_z = build_bb_check_rows(
        ell, m, generators["A_terms"], generators["B_terms"]
    )
    if num_qubits != 2 * ell * m:
        raise ValueError(f"{class_id}: n != 2 ell m")
    audit = audit_css_rows(rows_x, rows_z, num_qubits)
    rank_x = audit["sectors"]["X"]["stabilizer_rank"]
    rank_z = audit["sectors"]["Z"]["stabilizer_rank"]
    computed_k = num_qubits - rank_x - rank_z
    if computed_k != expected_k:
        raise ValueError(
            f"{class_id}: rebuilt dimension {computed_k} != catalogue k={expected_k}"
        )

    source_distance = representative["distance_class"]
    post_audit = _post_audit_distance(source_distance, audit)
    sectors = {
        name: _decorate_sector(sector, name, ell, m)
        for name, sector in audit["sectors"].items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "css_low_weight_class_audit",
        "algorithm": ALGORITHM,
        "class_identity": {
            "class_id": class_id,
            "class_label": representative["equivalence"]["class_label"],
            "equivalence_relation": representative["equivalence"]["relation"],
            "canonical_digest_sha256": representative["equivalence"][
                "canonical_digest_sha256"
            ],
            "class_size": len(members),
            "all_member_ids": [row["presentation_id"] for row in members],
            "target_direct_u_member_ids": [
                row["presentation_id"] for row in targets
            ],
        },
        "representative": {
            "presentation_id": representative["presentation_id"],
            "ell": ell,
            "m": m,
            "n": num_qubits,
            "k_catalogue": expected_k,
            "k_rebuilt": computed_k,
            "A_terms": generators["A_terms"],
            "B_terms": generators["B_terms"],
            "H_X_sha256": _matrix_digest(rows_x, num_qubits),
            "H_Z_sha256": _matrix_digest(rows_z, num_qubits),
            "checks_commute": True,
        },
        "search": {
            "maximum_weight": 4,
            "half_support_maximum_weight": 2,
            "half_supports_per_sector": (
                1 + num_qubits + num_qubits * (num_qubits - 1) // 2
            ),
            "X": sectors["X"],
            "Z": sectors["Z"],
        },
        "audit_evidence": {
            "result": audit["result"],
            "exact_distance": audit["exact_distance"],
            "certified_lower_bound": audit["certified_lower_bound"],
        },
        "source_class_distance": {
            "status_code": source_distance["status_code"],
            "lower": source_distance["lower"],
            "upper": source_distance["upper"],
            "is_exact": source_distance["is_exact"],
            "upper_is_supported": source_distance["upper_is_supported"],
        },
        "post_audit_class_distance": post_audit,
        "transfer": {
            "valid": True,
            "reason": (
                "a colored stored-generator Tanner isomorphism supplies qubit "
                "and same-type check permutations, which preserve CSS logical "
                "weight and stabilizer membership"
            ),
            "target_presentations_covered": len(targets),
        },
    }


def _manifest_record(
    catalogue_rows: Sequence[dict[str, Any]],
    audit_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    direct_u_targets = [
        row
        for row in catalogue_rows
        if row["family"] == "CSS" and row["distance_direct"]["status_code"] == "U"
    ]
    outcome_specs: Counter[str] = Counter()
    post_status_specs: Counter[str] = Counter()
    source_status_classes: Counter[str] = Counter()
    post_status_classes: Counter[str] = Counter()
    exact_distances_classes: Counter[str] = Counter()
    exact_distances_specs: Counter[str] = Counter()
    for row in audit_rows:
        covered = row["transfer"]["target_presentations_covered"]
        outcome = row["audit_evidence"]["result"]
        outcome_specs[outcome] += covered
        post_status = row["post_audit_class_distance"]["status_code"]
        post_status_specs[post_status] += covered
        source_status_classes[row["source_class_distance"]["status_code"]] += 1
        post_status_classes[post_status] += 1
        exact_distance = row["audit_evidence"]["exact_distance"]
        if exact_distance is not None:
            exact_distances_classes[str(exact_distance)] += 1
            exact_distances_specs[str(exact_distance)] += covered

    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "artifact_manifest",
        "artifact_schema": ARTIFACT_SCHEMA,
        "algorithm": {
            "name": ALGORITHM,
            "maximum_weight": 4,
            "deterministic": True,
            "randomness": None,
            "solver": None,
            "proof": (
                "Every support of weight at most four partitions into two "
                "supports of weight at most two. Equal H-syndrome halves "
                "produce a kernel vector; unequal canonical remainders modulo "
                "the same-type stabilizer row space prove it nontrivial."
            ),
            "phase_order": [
                "empty/singleton collisions (weights 1-2)",
                "pair versus empty/singleton collisions (weight 3)",
                "pair versus pair collisions (weight 4)",
            ],
        },
        "scope": {
            "selection": "CSS presentations with direct status U",
            "target_presentations": len(direct_u_targets),
            "target_classes": len(audit_rows),
            "target_classes_by_source_class_status": dict(
                sorted(source_status_classes.items())
            ),
            "class_optimization": EQUIVALENCE_RELATION,
            "representatives_audited": len(audit_rows),
        },
        "counts": {
            "audit_outcomes_by_class": dict(
                sorted(Counter(
                    row["audit_evidence"]["result"] for row in audit_rows
                ).items())
            ),
            "audit_outcomes_by_target_presentation": dict(sorted(outcome_specs.items())),
            "low_weight_exact_classes_by_distance": dict(
                sorted(exact_distances_classes.items(), key=lambda item: int(item[0]))
            ),
            "low_weight_exact_presentations_by_distance": dict(
                sorted(exact_distances_specs.items(), key=lambda item: int(item[0]))
            ),
            "post_audit_status_by_class": dict(sorted(post_status_classes.items())),
            "post_audit_status_by_target_presentation": dict(
                sorted(post_status_specs.items())
            ),
            "presentations_proved_d_le_2": sum(
                count
                for distance, count in exact_distances_specs.items()
                if int(distance) <= 2
            ),
        },
        "source_artifacts": _source_artifacts(),
        "producer": {
            "path": _relative(Path(__file__)),
            "sha256": _sha256_path(Path(__file__)),
        },
        "evidence_semantics": {
            "H": (
                "exact distance at most four: exhaustive exclusion below the "
                "reported weight plus an explicitly verified logical witness"
            ),
            "L4": "exhaustive exclusion of all X- and Z-type logicals through weight four",
            "class_transfer": (
                "the audit result is transferred only within the catalogue's "
                "colored stored-generator Tanner-isomorphism class"
            ),
        },
    }


def build_artifact_rows(
) -> list[dict[str, Any]]:
    catalogue_rows = _baseline_catalogue_rows()
    if len(catalogue_rows) != 1_144:
        raise ValueError(
            f"expected 1,144 publication rows, found {len(catalogue_rows)}"
        )
    groups = _class_groups(catalogue_rows)
    if len(groups) != 492:
        raise ValueError(f"expected 492 CSS classes containing direct-U rows, found {len(groups)}")
    target_count = sum(len(targets) for _, _, targets in groups)
    if target_count != 858:
        raise ValueError(f"expected 858 direct-U CSS targets, found {target_count}")

    audit_rows = [
        _audit_class(class_id, members, targets)
        for class_id, members, targets in groups
    ]
    manifest = _manifest_record(
        catalogue_rows,
        audit_rows,
    )
    return [manifest, *audit_rows]


def render_artifact(rows: Sequence[dict[str, Any]]) -> str:
    return "".join(_json_line(row) + "\n" for row in rows)


def _write_or_check(path: Path, payload: str, check: bool) -> bool:
    if check:
        if not path.exists():
            print(f"missing generated file: {_relative(path)}", file=sys.stderr)
            return False
        if path.read_text(encoding="utf-8") != payload:
            print(f"generated file is stale: {_relative(path)}", file=sys.stderr)
            return False
        print(f"up to date: {_relative(path)}")
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    print(f"wrote {_relative(path)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    rows = build_artifact_rows()
    payload = render_artifact(rows)
    if not _write_or_check(args.output, payload, args.check):
        return 1
    elapsed = time.perf_counter() - started
    manifest = rows[0]
    print(
        "audited "
        f"{manifest['scope']['target_presentations']} direct-U CSS presentations "
        f"via {manifest['scope']['representatives_audited']} class representatives "
        f"in {elapsed:.3f}s"
    )
    print(_json_line(manifest["counts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
