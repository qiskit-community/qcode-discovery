#!/usr/bin/env python3
"""Verify Campaign 4 exact-distance transfers from one exact BB representative."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.bb_code import build_bb_code
from evaluation.tanner_equivalence import _extract_check_matrices, canonical_digest

SOURCE = ROOT / "results" / "campaign4_reverified.jsonl"
FROZEN_SOURCE_BLOB_SHA1 = "3fcddce65c1711041cc82eddebee0242a201b61e"
REFERENCE = {
    "ell": 12, "m": 12,
    "A_terms": [[0, 0], [2, 3], [1, 0]],
    "B_terms": [[0, 0], [1, 2], [2, 1]],
}

# Each entry is:
# (A_terms, B_terms, L_matrix, L_translation, R_matrix, R_translation)
TARGETS = [
    (
        [[0, 0], [1, 2], [2, 1]], [[0, 0], [10, 1], [11, 2]],
        [[1, 11], [11, 0]], [0, 0], [[1, 11], [11, 0]], [10, 0],
    ),
    (
        [[0, 0], [3, 1], [3, 2]], [[0, 0], [1, 2], [2, 1]],
        [[0, 11], [1, 11]], [0, 0], [[0, 11], [1, 11]], [11, 0],
    ),
    (
        [[0, 0], [3, 2], [0, 1]], [[0, 0], [1, 2], [2, 1]],
        [[0, 1], [1, 0]], [0, 0], [[0, 1], [1, 0]], [0, 0],
    ),
    (
        [[0, 0], [1, 3], [2, 3]], [[0, 0], [1, 2], [2, 1]],
        [[1, 11], [0, 11]], [0, 0], [[1, 11], [0, 11]], [0, 11],
    ),
    (
        [[0, 0], [1, 3], [2, 3]], [[0, 0], [3, 1], [9, 2]],
        [[11, 8], [0, 11]], [0, 0], [[11, 8], [0, 11]], [7, 11],
    ),
]

def ident(ell, m, a_terms, b_terms):
    return {
        "ell": ell, "m": m,
        "A_terms": a_terms, "B_terms": b_terms,
    }

def key(value):
    return (
        int(value["ell"]), int(value["m"]),
        tuple(map(tuple, value["A_terms"])),
        tuple(map(tuple, value["B_terms"])),
    )

def gf2_rank(matrix):
    a = np.asarray(matrix, dtype=np.uint8).copy() & 1
    rank = 0
    for col in range(a.shape[1]):
        pivots = np.flatnonzero(a[rank:, col])
        if not len(pivots):
            continue
        pivot = rank + int(pivots[0])
        a[[rank, pivot]] = a[[pivot, rank]]
        for row in range(a.shape[0]):
            if row != rank and a[row, col]:
                a[row] ^= a[rank]
        rank += 1
        if rank == a.shape[0]:
            break
    return rank

def same_rowspace(left, right):
    left = np.asarray(left, dtype=np.uint8) & 1
    right = np.asarray(right, dtype=np.uint8) & 1
    rl, rr = gf2_rank(left), gf2_rank(right)
    return rl == rr == gf2_rank(np.vstack((left, right)))

def affine_map(matrix, translation):
    return {
        "target_sector": None,
        "matrix": matrix,
        "translation": translation,
    }

def map_spec(l_matrix, l_translation, r_matrix, r_translation):
    left = affine_map(l_matrix, l_translation)
    right = affine_map(r_matrix, r_translation)
    left["target_sector"] = "L"
    right["target_sector"] = "R"
    return {"L": left, "R": right}

def qubit_permutation(spec, ell, m):
    cells = ell * m
    perm = np.empty(2 * cells, dtype=int)
    for sector, source_base in (("L", 0), ("R", cells)):
        rule = spec[sector]
        target_base = 0 if rule["target_sector"] == "L" else cells
        matrix, shift = rule["matrix"], rule["translation"]
        for a in range(ell):
            for b in range(m):
                aa = (matrix[0][0] * a + matrix[0][1] * b + shift[0]) % ell
                bb = (matrix[1][0] * a + matrix[1][1] * b + shift[1]) % m
                perm[source_base + a * m + b] = target_base + aa * m + bb
    assert sorted(perm.tolist()) == list(range(2 * cells))
    return perm

def main():
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line]
    assert len(rows) == 39

    by_id = {key(row): row for row in rows}
    assert len(by_id) == 39

    reference = by_id[key(REFERENCE)]
    assert (reference["n"], reference["k"], reference["d"]) == (288, 12, 12)
    assert reference["d_is_exact"] is True

    ref_code = build_bb_code(
        reference["ell"], reference["m"],
        reference["A_terms"], reference["B_terms"],
    )
    hx_ref, hz_ref = _extract_check_matrices(ref_code)
    ref_digest = canonical_digest(ref_code)

    target_records = []
    target_keys = set()
    for a_terms, b_terms, lm, lt, rm, rt in TARGETS:
        target_id = ident(12, 12, a_terms, b_terms)
        spec = map_spec(lm, lt, rm, rt)
        target_records.append((target_id, spec))
        target_keys.add(key(target_id))

    classes = defaultdict(set)
    codes = {}
    for row in rows:
        row_key = key(row)
        code = build_bb_code(row["ell"], row["m"], row["A_terms"], row["B_terms"])
        codes[row_key] = code
        classes[canonical_digest(code)].add(row_key)

    # Full 39-row audit: this exact-distance class contains exactly six records.
    assert classes[ref_digest] == target_keys | {key(REFERENCE)}

    for target_id, spec in target_records:
        row = by_id[key(target_id)]
        assert (row["n"], row["k"], row["d"]) == (288, 12, 12)
        assert row["d_is_exact"] is True
        assert row["stage"] == "milp_incumbent"
        assert row["milp_details"]["exact"] is False

        assert row["distance_transfer"] == {
            "valid": True,
            "method": "colored_bliss_permutation_equivalence",
            "reference_d_exact": 12,
            "source_path": "results/campaign4_reverified.jsonl",
            "source_git_blob_sha1": FROZEN_SOURCE_BLOB_SHA1,
            "reference": REFERENCE,
            "qubit_map_from_reference": spec,
        }

        code = codes[key(target_id)]
        assert canonical_digest(code) == ref_digest
        hx_target, hz_target = _extract_check_matrices(code)
        perm = qubit_permutation(spec, 12, 12)
        assert same_rowspace(hx_ref, hx_target[:, perm])
        assert same_rowspace(hz_ref, hz_target[:, perm])

    other_multi = [
        members for digest, members in classes.items()
        if digest != ref_digest and len(members) > 1
    ]
    assert len(other_multi) == 4

    print(json.dumps({
        "status": "PASS",
        "rows_checked": len(rows),
        "equivalence_classes": len(classes),
        "reference_class_size": len(classes[ref_digest]),
        "transfers_verified": len(target_records),
        "other_multi_member_classes_without_transfer": len(other_multi),
        "frozen_source_blob_sha1": FROZEN_SOURCE_BLOB_SHA1,
    }, indent=2))

if __name__ == "__main__":
    main()
