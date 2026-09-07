"""Reproduce the Lin--Pryadko weight-five prior-art audit.

The Lin--Pryadko 2BGA archive stores group elements by their one-based
positions in ``GAP``'s ``Elements(SmallGroup(order, id))`` list.  This script
freezes the relevant archive rows and their independently decoded supports,
rebuilds the CSS checks, and verifies:

* the five short prior codes have the advertised dimensions and connected
  Tanner graphs;
* every recorded exact-distance claim closed all ``2k`` SciPy/HiGHS MILPs;
* the campaign's connected ``[[96,4,10]]`` component maps to the archived
  ``48 2 4 10 [2] [3,39]`` row by an explicit CRT relabeling and block swap;
* the archived ``[[132,4,12]]`` representative over
  ``SmallGroup(66,1)=C11 x S3`` is an ambient-nonabelian quasi-abelian
  lifted-product code over ``F2[C11]`` with zero rank defects;
* the attempted ``[[168,4,14]]``, ``[[180,4,15]]``, and
  ``[[192,4,16]]`` checks retain every incumbent and timeout without
  promoting their reported distances to exact results.

The checked-in JSON is deterministic and needs no external checkout.  For an
independent solver rerun (roughly two minutes on the audit machine), use::

    python scripts/audit_lin_pryadko_weight5.py --run-milp

Optional ``--archive-zip`` and ``--nonabelian-archive-zip`` arguments check
the pinned upstream zips and relevant member files byte-for-byte.  ``--write``
regenerates the checked-in JSON.

Important provenance caveat: the upstream ``checkdata`` routine calls
``DistRandCSS`` from QDistRnd.  Its archived distances are probabilistic upper
bounds, not proofs.  Exactness below comes only from the independent MILPs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = ROOT / "results" / "weight5_lin_pryadko_audit.json"

ARCHIVE = {
    "repository": "https://github.com/QEC-pages/2BGA-codes",
    "commit": "403d194c3f98f0cadc236aecbc4a8b6139ccf23c",
    "abelian_zip_sha256": (
        "5500708bf369baae14fb008e7a6d01147b77efb2ff9d798dac3247a57a6f63dd"
    ),
    "nonabelian_zip_sha256": (
        "5962039bbfa3729a80f5001ff44dbefd57dac689d605fbaae7f247ff34fb5258"
    ),
    "readme_sha256": (
        "8a0b260216f7e5c28af162aaabb5b3be6344f7becc4669a5660a32e72d4e637e"
    ),
    "gap_source_sha256": (
        "5a2912f0db3a9a3ad27523b643975f1470951fd31ecf234e6d30f1e69b88296b"
    ),
    "distance_method": {
        "routine": "QDistRnd DistRandCSS",
        "archive_call": (
            "DistRandCSS(TransposedMat(gxz[2]),gxz[1],100000,1: "
            "field:=F,maxav:=10)"
        ),
        "interpretation": (
            "probabilistic search; an archived d is a reported upper bound "
            "unless an independent exact computation certifies it"
        ),
    },
}

GAP_DECODER = {
    "method": (
        "For each cyclic SmallGroup, choose the first full-order element z "
        "in GAP Elements(G), then solve Elements(G)[i] = z^e.  The order-48 "
        "row is subsequently acted on by the unit e -> 19e mod 48 so it "
        "uses the same generator as the campaign CRT presentation."
    ),
    "audit_environment": {
        "gap_version": "4.17dev",
        "gap_commit": "de5dc7576e0c83747e37f4278d57c4e2e4bca248",
        "smallgrp_version": "1.7.0",
        "smallgrp_commit": "0b54328cdef811622c0e7f5498d4868300719c07",
    },
    "portability_caveat": (
        "The archive does not pin GAP/SmallGrp and stores positions rather "
        "than group words.  The decoded powers below are therefore frozen "
        "audit data; re-decoding should record the GAP and SmallGrp versions."
    ),
    "nonabelian_cross_version_check": {
        "smallgrp_versions": ["1.5.1", "1.7.0"],
        "rows": [
            "LP-132-4-12",
            "LP-168-4-14",
            "LP-180-4-15",
            "LP-192-4-16",
        ],
        "result": "all selected GAP element indices decode identically",
    },
}

# ``file_sha256`` covers the entire archive member, not just the selected row.
# Exponent zero (the identity) is implicit in each archive index list.
ARCHIVE_ROWS = (
    {
        "label": "LP-48-4-7",
        "member": "diswtabelian_wt5wtL2_order24_k4.txt",
        "file_sha256": "d29ea1c7370d58ebad03f4872769f95f389504e26c8011794823cb3221ef9184",
        "archive_row": "24 2 4 7 [2] [3,20]",
        "group_order": 24,
        "small_group_id": 2,
        "reported_k": 4,
        "reported_d": 7,
        "a_indices": (2,),
        "b_indices": (3, 20),
        "index_exponents": {2: 9, 3: 16, 20: 11},
        "generator_automorphism": 1,
    },
    {
        "label": "LP-60-4-8",
        "member": "diswtabelian_wt5wtL2_order30_k4.txt",
        "file_sha256": "cd1137867a1b3f4c170e665f5de0ccbaa06084f4f4c9b02fc6747224ff6a6259",
        "archive_row": "30 4 4 8 [6] [3,13]",
        "group_order": 30,
        "small_group_id": 4,
        "reported_k": 4,
        "reported_d": 8,
        "a_indices": (6,),
        "b_indices": (3, 13),
        "index_exponents": {6: 21, 3: 10, 13: 26},
        "generator_automorphism": 1,
    },
    {
        "label": "LP-66-4-8",
        "member": "diswtabelian_wt5wtL2_order33_k4.txt",
        "file_sha256": "6400dd67f99045403986120e63fdb9de7fcb3f930f136be6164f734e0c7d2aa7",
        "archive_row": "33 1 4 8 [3] [2,16]",
        "group_order": 33,
        "small_group_id": 1,
        "reported_k": 4,
        "reported_d": 8,
        "a_indices": (3,),
        "b_indices": (2, 16),
        "index_exponents": {3: 12, 2: 22, 16: 26},
        "generator_automorphism": 1,
    },
    {
        "label": "LP-78-4-9",
        "member": "diswtabelian_wt5wtL2_order39_k4.txt",
        "file_sha256": "6aa2d537c2dd42b7750a196448a2b1c76846323322c0e4cc3663b98c9aed67da",
        "archive_row": "39 2 4 9 [3] [2,19]",
        "group_order": 39,
        "small_group_id": 2,
        "reported_k": 4,
        "reported_d": 9,
        "a_indices": (3,),
        "b_indices": (2, 19),
        "index_exponents": {3: 27, 2: 13, 19: 5},
        "generator_automorphism": 1,
    },
    {
        "label": "LP-84-4-9",
        "member": "diswtabelian_wt5wtL2_order42_k4.txt",
        "file_sha256": "f2b175d91cfb677261baf9c58d36d4094918b793a9bc6680424144ba42d118cc",
        "archive_row": "42 6 4 9 [6] [11,25]",
        "group_order": 42,
        "small_group_id": 6,
        "reported_k": 4,
        "reported_d": 9,
        "a_indices": (6,),
        "b_indices": (11, 25),
        "index_exponents": {6: 15, 11: 1, 25: 38},
        "generator_automorphism": 1,
    },
    {
        "label": "LP-96-4-10",
        "member": "diswtabelian_wt5wtL2_order48_k4.txt",
        "file_sha256": "9a4220fdb2fcc207f89713e365b61a7dfeccf803261f8b9eec2ffc9c9f41f1b4",
        "archive_row": "48 2 4 10 [2] [3,39]",
        "group_order": 48,
        "small_group_id": 2,
        "reported_k": 4,
        "reported_d": 10,
        "a_indices": (2,),
        "b_indices": (3, 39),
        "index_exponents": {2: 33, 3: 16, 39: 38},
        "generator_automorphism": 19,
    },
)

NONABELIAN_FRONTIER_ROWS = (
    {
        "label": "LP-132-4-12",
        "member": "diswtnonabelian_wt5wtL2_order66_k4.txt",
        "file_sha256": "b822bc574436da45d69feda8217ac1208a3fe69f62cfbc536d6287d62910d8ac",
        "archive_row": "66 1 4 12 [5] [14,39]",
        "group_order": 66,
        "small_group_id": 1,
        "reported_k": 4,
        "reported_d": 12,
        "a_indices": (5,),
        "b_indices": (14, 39),
        # Coordinates are c^a r^b s^d in C11 x S3.
        "index_coordinates": {
            5: (1, 0, 1),
            14: (2, 1, 0),
            39: (5, 2, 0),
        },
    },
    {
        "label": "LP-168-4-14",
        "member": "diswtnonabelian_wt5wtL2_order84_k4.txt",
        "file_sha256": "f5ccdd35c779d190b134067fe8fe4a7e11c9d96a1f4e1b1a1fcdb583252572b9",
        "archive_row": "84 3 4 14 [6] [11,60]",
        "group_order": 84,
        "small_group_id": 3,
        "reported_k": 4,
        "reported_d": 14,
        "a_indices": (6,),
        "b_indices": (11, 60),
        # Coordinates are c^a r^b s^d in C7 x (C3 : C4).
        "index_coordinates": {
            6: (1, 0, 1),
            11: (1, 1, 0),
            60: (3, 2, 2),
        },
    },
    {
        "label": "LP-180-4-15",
        "member": "diswtnonabelian_wt5wtL2_order90_k4.txt",
        "file_sha256": "7fe4d79401013b7bbb2229e3a33ae8e17f10c384a91641fa5bc2e075ac5fb9d1",
        "archive_row": "90 6 4 15 [10] [6,38]",
        "group_order": 90,
        "small_group_id": 6,
        "reported_k": 4,
        "reported_d": 15,
        "a_indices": (10,),
        "b_indices": (6, 38),
        # Coordinates are c^a r^b s^d in C15 x S3.
        "index_coordinates": {
            10: (1, 0, 0),
            6: (10, 0, 1),
            38: (11, 1, 0),
        },
    },
    {
        "label": "LP-192-4-16",
        "member": "diswtnonabelian_wt5wtL2_order96_k4.txt",
        "file_sha256": "b85064ccd7beec88d4388fe3b59acbda29eccc22058e3ad4741576bded886658",
        "archive_row": "96 4 4 16 [3] [8,46]",
        "group_order": 96,
        "small_group_id": 4,
        "reported_k": 4,
        "reported_d": 16,
        "a_indices": (3,),
        "b_indices": (8, 46),
        # Coordinates are c^a r^b s^d in C16 x S3.
        "index_coordinates": {
            3: (1, 0, 0),
            8: (1, 0, 1),
            46: (12, 1, 0),
        },
    },
)

# Filled from a complete ``--run-milp`` pass.  The individual objective
# values are retained, rather than just their minimum, so the artifact shows
# that every member of both logical bases reached solver-proven optimality.
RECORDED_MILP: dict[str, dict[str, Any]] = {
    "LP-48-4-7": {
        "d": 7,
        "d_x": 7,
        "d_z": 7,
        "exact": True,
        "all_objectives_optimal": True,
        "logical_objectives_optimal": 8,
        "logical_objectives_total": 8,
        "x_objectives": [{"weight": 7, "optimal": True}] * 4,
        "z_objectives": [{"weight": 7, "optimal": True}] * 4,
    },
    "LP-60-4-8": {
        "d": 8,
        "d_x": 8,
        "d_z": 8,
        "exact": True,
        "all_objectives_optimal": True,
        "logical_objectives_optimal": 8,
        "logical_objectives_total": 8,
        "x_objectives": [{"weight": 8, "optimal": True}] * 4,
        "z_objectives": [{"weight": 8, "optimal": True}] * 4,
    },
    "LP-66-4-8": {
        "d": 8,
        "d_x": 8,
        "d_z": 8,
        "exact": True,
        "all_objectives_optimal": True,
        "logical_objectives_optimal": 8,
        "logical_objectives_total": 8,
        "x_objectives": [{"weight": 8, "optimal": True}] * 4,
        "z_objectives": [{"weight": 8, "optimal": True}] * 4,
    },
    "LP-78-4-9": {
        "d": 9,
        "d_x": 9,
        "d_z": 9,
        "exact": True,
        "all_objectives_optimal": True,
        "logical_objectives_optimal": 8,
        "logical_objectives_total": 8,
        "x_objectives": [{"weight": 9, "optimal": True}] * 4,
        "z_objectives": [{"weight": 9, "optimal": True}] * 4,
    },
    "LP-84-4-9": {
        "d": 9,
        "d_x": 9,
        "d_z": 9,
        "exact": True,
        "all_objectives_optimal": True,
        "logical_objectives_optimal": 8,
        "logical_objectives_total": 8,
        "x_objectives": [
            {"weight": 9, "optimal": True},
            {"weight": 9, "optimal": True},
            {"weight": 10, "optimal": True},
            {"weight": 10, "optimal": True},
        ],
        "z_objectives": [
            {"weight": 10, "optimal": True},
            {"weight": 10, "optimal": True},
            {"weight": 9, "optimal": True},
            {"weight": 9, "optimal": True},
        ],
    },
    "LP-96-4-10": {
        "d": 10,
        "d_x": 10,
        "d_z": 10,
        "exact": True,
        "all_objectives_optimal": True,
        "logical_objectives_optimal": 8,
        "logical_objectives_total": 8,
        "x_objectives": [{"weight": 10, "optimal": True}] * 4,
        "z_objectives": [{"weight": 10, "optimal": True}] * 4,
    },
    "LP-132-4-12": {
        "d": 12,
        "d_x": 12,
        "d_z": 12,
        "exact": True,
        "all_objectives_optimal": True,
        "logical_objectives_optimal": 8,
        "logical_objectives_total": 8,
        "parallel_wall_seconds_observed": 91.74,
        "x_objectives": [
            {"weight": 12, "optimal": True, "seconds_observed": 37.74},
            {"weight": 12, "optimal": True, "seconds_observed": 90.53},
            {"weight": 12, "optimal": True, "seconds_observed": 51.34},
            {"weight": 12, "optimal": True, "seconds_observed": 55.36},
        ],
        "z_objectives": [
            {"weight": 12, "optimal": True, "seconds_observed": 91.73},
            {"weight": 12, "optimal": True, "seconds_observed": 47.42},
            {"weight": 12, "optimal": True, "seconds_observed": 59.41},
            {"weight": 12, "optimal": True, "seconds_observed": 56.93},
        ],
    },
    "LP-168-4-14": {
        "d": None,
        "d_upper_bound": 14,
        "d_x_upper_bound": 14,
        "d_z_upper_bound": 14,
        "exact": False,
        "all_objectives_optimal": False,
        "logical_objectives_optimal": 1,
        "logical_objectives_total": 8,
        "timed_out_with_incumbent": 7,
        "timeout_per_logical_seconds": 300,
        "x_objectives": [
            {
                "weight": 14,
                "optimal": False,
                "status": "time_limit_with_incumbent",
                "seconds_observed": 300.03,
            },
            {
                "weight": 14,
                "optimal": False,
                "status": "time_limit_with_incumbent",
                "seconds_observed": 300.04,
            },
            {
                "weight": 14,
                "optimal": False,
                "status": "time_limit_with_incumbent",
                "seconds_observed": 300.08,
            },
            {
                "weight": 14,
                "optimal": False,
                "status": "time_limit_with_incumbent",
                "seconds_observed": 300.05,
            },
        ],
        "z_objectives": [
            {
                "weight": 14,
                "optimal": True,
                "status": "optimal",
                "seconds_observed": 59.03,
            },
            {
                "weight": 14,
                "optimal": False,
                "status": "time_limit_with_incumbent",
                "seconds_observed": 300.04,
            },
            {
                "weight": 14,
                "optimal": False,
                "status": "time_limit_with_incumbent",
                "seconds_observed": 300.04,
            },
            {
                "weight": 14,
                "optimal": False,
                "status": "time_limit_with_incumbent",
                "seconds_observed": 300.06,
            },
        ],
        "conclusion": (
            "Every logical-sector MILP found a weight-14 representative, but "
            "seven hit the time limit.  This independently verifies only "
            "d<=14, not d=14."
        ),
    },
    "LP-180-4-15": {
        "d": None,
        "d_upper_bound": 15,
        "d_x_upper_bound": 15,
        "d_z_upper_bound": 15,
        "exact": False,
        "all_objectives_optimal": False,
        "logical_objectives_optimal": 0,
        "logical_objectives_total": 8,
        "timed_out_with_incumbent": 8,
        "timeout_per_logical_seconds": 300,
        "x_objectives": [
            {
                "weight": 15,
                "optimal": False,
                "status": "time_limit_with_incumbent",
                "seconds_observed": 300.038,
            },
            {
                "weight": 15,
                "optimal": False,
                "status": "time_limit_with_incumbent",
                "seconds_observed": 300.078,
            },
            {
                "weight": 15,
                "optimal": False,
                "status": "time_limit_with_incumbent",
                "seconds_observed": 300.033,
            },
            {
                "weight": 15,
                "optimal": False,
                "status": "time_limit_with_incumbent",
                "seconds_observed": 300.076,
            },
        ],
        "z_objectives": [
            {
                "weight": 15,
                "optimal": False,
                "status": "time_limit_with_incumbent",
                "seconds_observed": 300.028,
            },
            {
                "weight": 15,
                "optimal": False,
                "status": "time_limit_with_incumbent",
                "seconds_observed": 300.024,
            },
            {
                "weight": 15,
                "optimal": False,
                "status": "time_limit_with_incumbent",
                "seconds_observed": 300.05,
            },
            {
                "weight": 15,
                "optimal": False,
                "status": "time_limit_with_incumbent",
                "seconds_observed": 300.081,
            },
        ],
        "conclusion": (
            "Every logical-sector MILP found a weight-15 representative, but "
            "all eight hit the time limit.  This independently verifies only "
            "d<=15, not d=15."
        ),
    },
    "LP-192-4-16": {
        "d": None,
        "d_upper_bound": 16,
        "d_x_upper_bound": 16,
        "d_z_upper_bound": 16,
        "exact": False,
        "all_objectives_optimal": False,
        "logical_objectives_optimal": 1,
        "logical_objectives_total": 8,
        "timed_out_with_incumbent": 7,
        "timeout_per_logical_seconds": 300,
        "x_objectives": [
            {
                "weight": 16,
                "optimal": False,
                "status": "time_limit_with_incumbent",
                "seconds_observed": 300.033,
            },
            {
                "weight": 16,
                "optimal": False,
                "status": "time_limit_with_incumbent",
                "seconds_observed": 300.049,
            },
            {
                "weight": 16,
                "optimal": False,
                "status": "time_limit_with_incumbent",
                "seconds_observed": 300.072,
            },
            {
                "weight": 16,
                "optimal": False,
                "status": "time_limit_with_incumbent",
                "seconds_observed": 300.055,
            },
        ],
        "z_objectives": [
            {
                "weight": 16,
                "optimal": True,
                "status": "optimal",
                "seconds_observed": 133.118,
            },
            {
                "weight": 16,
                "optimal": False,
                "status": "time_limit_with_incumbent",
                "seconds_observed": 300.052,
            },
            {
                "weight": 16,
                "optimal": False,
                "status": "time_limit_with_incumbent",
                "seconds_observed": 300.036,
            },
            {
                "weight": 16,
                "optimal": False,
                "status": "time_limit_with_incumbent",
                "seconds_observed": 300.055,
            },
        ],
        "conclusion": (
            "Every logical-sector MILP found a weight-16 representative, but "
            "seven hit the time limit.  This independently verifies only "
            "d<=16, not d=16."
        ),
    },
}

BRANCH_COMPONENT = {
    "match_id": "W5BB-96-4-10",
    "parent_presentation_id": "CL-885dfa58",
    "parent_parameters": {"n": 384, "k": 16, "d": 10, "components": 4},
    "group": "C16 x C3",
    "ell": 16,
    "m": 3,
    "a_terms": ((0, 0), (0, 1), (2, 2)),
    "b_terms": ((0, 0), (3, 0)),
    "parameters": {"n": 96, "k": 4, "d": 10},
}


def _canonical_row(text: str) -> str:
    """Normalize archive whitespace while preserving list punctuation."""
    return (
        re.sub(r"\s+", " ", text.strip())
        .replace("[ ", "[")
        .replace(" ]", "]")
        .replace(", ", ",")
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _matrix_sha256(matrix: np.ndarray) -> str:
    matrix = np.asarray(matrix, dtype=np.uint8) % 2
    header = f"{matrix.shape[0]}x{matrix.shape[1]}:".encode("ascii")
    return _sha256(header + np.packbits(matrix, axis=None).tobytes())


def cyclic_group_algebra_matrix(order: int, support: tuple[int, ...]) -> np.ndarray:
    """Return the left-regular matrix of ``sum(z^e for e in support)``."""
    matrix = np.zeros((order, order), dtype=np.uint8)
    columns = np.arange(order)
    for exponent in support:
        matrix[(columns + exponent) % order, columns] ^= 1
    return matrix


def cyclic_css_checks(
    order: int, a_support: tuple[int, ...], b_support: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``H_X=[A|B]`` and ``H_Z=[B^T|A^T]`` for a cyclic 2BGA code."""
    matrix_a = cyclic_group_algebra_matrix(order, a_support)
    matrix_b = cyclic_group_algebra_matrix(order, b_support)
    matrix_x = np.hstack((matrix_a, matrix_b))
    matrix_z = np.hstack((matrix_b.T, matrix_a.T))
    if np.any((matrix_x @ matrix_z.T) % 2):
        raise AssertionError("constructed CSS checks do not commute")
    return matrix_x, matrix_z


# Coordinate model for SmallGroup(66,1) = C11 x S3.  Elements are normal
# words c^a r^b s^d with s*r*s = r^-1.  This avoids making the reproducible
# audit depend on GAP at runtime while retaining the exact left/right actions.
Group66Element = tuple[int, int, int]
GROUP66_IDENTITY: Group66Element = (0, 0, 0)
GROUP66_ELEMENTS: tuple[Group66Element, ...] = tuple(
    (a_exp, r_exp, s_exp)
    for a_exp in range(11)
    for r_exp in range(3)
    for s_exp in range(2)
)


def group66_multiply(left: Group66Element, right: Group66Element) -> Group66Element:
    """Multiply normal words in C11 x <r,s | r^3=s^2=1, srs=r^-1>."""
    left_c, left_r, left_s = left
    right_c, right_r, right_s = right
    acted_right_r = right_r if left_s == 0 else -right_r
    return (
        (left_c + right_c) % 11,
        (left_r + acted_right_r) % 3,
        (left_s + right_s) % 2,
    )


def _group66_inverse(element: Group66Element) -> Group66Element:
    c_exp, r_exp, s_exp = element
    inverse_r = -r_exp if s_exp == 0 else r_exp
    return ((-c_exp) % 11, inverse_r % 3, s_exp)


def _group66_power(element: Group66Element, exponent: int) -> Group66Element:
    result = GROUP66_IDENTITY
    for _ in range(exponent):
        result = group66_multiply(result, element)
    return result


def _group66_order(element: Group66Element) -> int:
    result = GROUP66_IDENTITY
    for exponent in range(1, 67):
        result = group66_multiply(result, element)
        if result == GROUP66_IDENTITY:
            return exponent
    raise AssertionError(f"element has no order dividing 66: {element}")


def _group66_generated_subgroup(
    generators: tuple[Group66Element, ...],
) -> set[Group66Element]:
    subgroup = {GROUP66_IDENTITY}
    frontier = [GROUP66_IDENTITY]
    moves = (*generators, *(_group66_inverse(item) for item in generators))
    while frontier:
        current = frontier.pop()
        for move in moves:
            product = group66_multiply(current, move)
            if product not in subgroup:
                subgroup.add(product)
                frontier.append(product)
    return subgroup


def _regular_group66_matrix(
    support: tuple[Group66Element, ...], *, side: str
) -> np.ndarray:
    positions = {element: index for index, element in enumerate(GROUP66_ELEMENTS)}
    matrix = np.zeros((66, 66), dtype=np.uint8)
    for column, element in enumerate(GROUP66_ELEMENTS):
        for shift in support:
            if side == "left":
                target = group66_multiply(shift, element)
            elif side == "right":
                target = group66_multiply(element, shift)
            else:
                raise ValueError(f"unknown regular-action side: {side}")
            matrix[positions[target], column] ^= 1
    return matrix


def _nonabelian_132_blocks() -> tuple[np.ndarray, np.ndarray]:
    row = NONABELIAN_FRONTIER_ROWS[0]
    coordinates = row["index_coordinates"]
    support_a = (GROUP66_IDENTITY, *(coordinates[index] for index in row["a_indices"]))
    support_b = (GROUP66_IDENTITY, *(coordinates[index] for index in row["b_indices"]))
    matrix_a = _regular_group66_matrix(support_a, side="left")
    matrix_b = _regular_group66_matrix(support_b, side="right")
    return matrix_a, matrix_b


def nonabelian_132_css_checks() -> tuple[np.ndarray, np.ndarray]:
    matrix_a, matrix_b = _nonabelian_132_blocks()
    matrix_x = np.hstack((matrix_a, matrix_b))
    matrix_z = np.hstack((matrix_b.T, matrix_a.T))
    if np.any((matrix_x @ matrix_z.T) % 2):
        raise AssertionError("constructed nonabelian CSS checks do not commute")
    return matrix_x, matrix_z


# Coordinate model for C_t x (C3 : C_q), with q even and the generator s
# acting on r by inversion.  This covers the order-84 and order-90 frontier
# rows while keeping their reconstruction independent of GAP at runtime.
MetacyclicElement = tuple[int, int, int]


def _metacyclic_elements(
    central_order: int, s_order: int
) -> tuple[MetacyclicElement, ...]:
    return tuple(
        (c_exp, r_exp, s_exp)
        for c_exp in range(central_order)
        for r_exp in range(3)
        for s_exp in range(s_order)
    )


def _metacyclic_multiply(
    left: MetacyclicElement,
    right: MetacyclicElement,
    *,
    central_order: int,
    s_order: int,
) -> MetacyclicElement:
    left_c, left_r, left_s = left
    right_c, right_r, right_s = right
    acted_right_r = right_r if left_s % 2 == 0 else -right_r
    return (
        (left_c + right_c) % central_order,
        (left_r + acted_right_r) % 3,
        (left_s + right_s) % s_order,
    )


def _metacyclic_inverse(
    element: MetacyclicElement, *, central_order: int, s_order: int
) -> MetacyclicElement:
    c_exp, r_exp, s_exp = element
    inverse_r = -r_exp if s_exp % 2 == 0 else r_exp
    return ((-c_exp) % central_order, inverse_r % 3, (-s_exp) % s_order)


def _metacyclic_power(
    element: MetacyclicElement,
    exponent: int,
    *,
    central_order: int,
    s_order: int,
) -> MetacyclicElement:
    result = (0, 0, 0)
    for _ in range(exponent):
        result = _metacyclic_multiply(
            result,
            element,
            central_order=central_order,
            s_order=s_order,
        )
    return result


def _metacyclic_order(
    element: MetacyclicElement, *, central_order: int, s_order: int
) -> int:
    group_order = central_order * 3 * s_order
    for exponent in range(1, group_order + 1):
        if _metacyclic_power(
            element,
            exponent,
            central_order=central_order,
            s_order=s_order,
        ) == (0, 0, 0):
            return exponent
    raise AssertionError(f"element has no order dividing {group_order}: {element}")


def _metacyclic_generated_subgroup(
    generators: tuple[MetacyclicElement, ...],
    *,
    central_order: int,
    s_order: int,
) -> set[MetacyclicElement]:
    identity = (0, 0, 0)
    subgroup = {identity}
    frontier = [identity]
    moves = (
        *generators,
        *(
            _metacyclic_inverse(
                item, central_order=central_order, s_order=s_order
            )
            for item in generators
        ),
    )
    while frontier:
        current = frontier.pop()
        for move in moves:
            product = _metacyclic_multiply(
                current,
                move,
                central_order=central_order,
                s_order=s_order,
            )
            if product not in subgroup:
                subgroup.add(product)
                frontier.append(product)
    return subgroup


def _regular_metacyclic_matrix(
    central_order: int,
    s_order: int,
    support: tuple[MetacyclicElement, ...],
    *,
    side: str,
) -> np.ndarray:
    elements = _metacyclic_elements(central_order, s_order)
    positions = {element: index for index, element in enumerate(elements)}
    matrix = np.zeros((len(elements), len(elements)), dtype=np.uint8)
    for column, element in enumerate(elements):
        for shift in support:
            if side == "left":
                target = _metacyclic_multiply(
                    shift,
                    element,
                    central_order=central_order,
                    s_order=s_order,
                )
            elif side == "right":
                target = _metacyclic_multiply(
                    element,
                    shift,
                    central_order=central_order,
                    s_order=s_order,
                )
            else:
                raise ValueError(f"unknown regular-action side: {side}")
            matrix[positions[target], column] ^= 1
    return matrix


def _nonabelian_168_blocks() -> tuple[np.ndarray, np.ndarray]:
    row = next(
        item for item in NONABELIAN_FRONTIER_ROWS if item["label"] == "LP-168-4-14"
    )
    coordinates = row["index_coordinates"]
    support_a = ((0, 0, 0), *(coordinates[index] for index in row["a_indices"]))
    support_b = ((0, 0, 0), *(coordinates[index] for index in row["b_indices"]))
    matrix_a = _regular_metacyclic_matrix(7, 4, support_a, side="left")
    matrix_b = _regular_metacyclic_matrix(7, 4, support_b, side="right")
    return matrix_a, matrix_b


def nonabelian_168_css_checks() -> tuple[np.ndarray, np.ndarray]:
    matrix_a, matrix_b = _nonabelian_168_blocks()
    matrix_x = np.hstack((matrix_a, matrix_b))
    matrix_z = np.hstack((matrix_b.T, matrix_a.T))
    if np.any((matrix_x @ matrix_z.T) % 2):
        raise AssertionError("constructed nonabelian CSS checks do not commute")
    return matrix_x, matrix_z


def _nonabelian_180_blocks() -> tuple[np.ndarray, np.ndarray]:
    row = next(
        item for item in NONABELIAN_FRONTIER_ROWS if item["label"] == "LP-180-4-15"
    )
    coordinates = row["index_coordinates"]
    support_a = ((0, 0, 0), *(coordinates[index] for index in row["a_indices"]))
    support_b = ((0, 0, 0), *(coordinates[index] for index in row["b_indices"]))
    matrix_a = _regular_metacyclic_matrix(15, 2, support_a, side="left")
    matrix_b = _regular_metacyclic_matrix(15, 2, support_b, side="right")
    return matrix_a, matrix_b


def nonabelian_180_css_checks() -> tuple[np.ndarray, np.ndarray]:
    matrix_a, matrix_b = _nonabelian_180_blocks()
    matrix_x = np.hstack((matrix_a, matrix_b))
    matrix_z = np.hstack((matrix_b.T, matrix_a.T))
    if np.any((matrix_x @ matrix_z.T) % 2):
        raise AssertionError("constructed nonabelian CSS checks do not commute")
    return matrix_x, matrix_z


def _nonabelian_192_blocks() -> tuple[np.ndarray, np.ndarray]:
    row = next(
        item for item in NONABELIAN_FRONTIER_ROWS if item["label"] == "LP-192-4-16"
    )
    coordinates = row["index_coordinates"]
    support_a = ((0, 0, 0), *(coordinates[index] for index in row["a_indices"]))
    support_b = ((0, 0, 0), *(coordinates[index] for index in row["b_indices"]))
    matrix_a = _regular_metacyclic_matrix(16, 2, support_a, side="left")
    matrix_b = _regular_metacyclic_matrix(16, 2, support_b, side="right")
    return matrix_a, matrix_b


def nonabelian_192_css_checks() -> tuple[np.ndarray, np.ndarray]:
    matrix_a, matrix_b = _nonabelian_192_blocks()
    matrix_x = np.hstack((matrix_a, matrix_b))
    matrix_z = np.hstack((matrix_b.T, matrix_a.T))
    if np.any((matrix_x @ matrix_z.T) % 2):
        raise AssertionError("constructed nonabelian CSS checks do not commute")
    return matrix_x, matrix_z


def _gf2_rank(matrix: np.ndarray) -> int:
    work = np.asarray(matrix, dtype=np.uint8).copy() % 2
    rank = 0
    for column in range(work.shape[1]):
        pivots = np.flatnonzero(work[rank:, column])
        if not pivots.size:
            continue
        pivot = rank + int(pivots[0])
        work[[rank, pivot]] = work[[pivot, rank]]
        for row in np.flatnonzero(work[:, column]):
            if row != rank:
                work[row] ^= work[rank]
        rank += 1
        if rank == work.shape[0]:
            break
    return rank


def _tanner_component_count(matrix_x: np.ndarray, matrix_z: np.ndarray) -> int:
    """Count qubit components of the stored CSS Tanner presentation."""
    n = matrix_x.shape[1]
    parent = list(range(n))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for row in np.vstack((matrix_x, matrix_z)):
        support = np.flatnonzero(row)
        if support.size:
            first = int(support[0])
            for qubit in support[1:]:
                union(first, int(qubit))
    return len({find(qubit) for qubit in range(n)})


def decoded_supports(row: dict[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    order = row["group_order"]
    multiplier = row["generator_automorphism"]
    exponents = row["index_exponents"]

    def decode(indices: tuple[int, ...]) -> tuple[int, ...]:
        return (0, *(multiplier * exponents[index] % order for index in indices))

    return decode(row["a_indices"]), decode(row["b_indices"])


def _structural_record(
    row: dict[str, Any], milp_evidence: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    a_support, b_support = decoded_supports(row)
    matrix_x, matrix_z = cyclic_css_checks(row["group_order"], a_support, b_support)
    n = matrix_x.shape[1]
    k = n - _gf2_rank(matrix_x) - _gf2_rank(matrix_z)
    component_count = _tanner_component_count(matrix_x, matrix_z)
    if k != row["reported_k"]:
        raise AssertionError(f"{row['label']}: decoded k={k}, expected {row['reported_k']}")
    if component_count != 1:
        raise AssertionError(f"{row['label']}: decoded code has {component_count} components")

    return {
        "label": row["label"],
        "archive": {
            "member": row["member"],
            "member_sha256": row["file_sha256"],
            "row": row["archive_row"],
            "group_order": row["group_order"],
            "small_group_id": row["small_group_id"],
            "a_element_indices": list(row["a_indices"]),
            "b_element_indices": list(row["b_indices"]),
            "reported_k": row["reported_k"],
            "reported_d_upper_bound": row["reported_d"],
        },
        "cyclic_decode": {
            "structure": f"C{row['group_order']}",
            "index_to_exponent_before_automorphism": {
                str(index): exponent
                for index, exponent in sorted(row["index_exponents"].items())
            },
            "generator_automorphism_multiplier": row["generator_automorphism"],
            "a_support_exponents": list(a_support),
            "b_support_exponents": list(b_support),
        },
        "reconstructed_code": {
            "n": n,
            "k": k,
            "fom_kd2_over_n": row["reported_k"] * row["reported_d"] ** 2 / n,
            "stabilizer_weight": int(matrix_x[0].sum()),
            "connected": component_count == 1,
            "tanner_components": component_count,
            "translation_gcd": math.gcd(row["group_order"], *a_support, *b_support),
            "matrix_x_sha256": _matrix_sha256(matrix_x),
            "matrix_z_sha256": _matrix_sha256(matrix_z),
        },
        "exact_milp": milp_evidence.get(row["label"]),
    }


def _nonabelian_structural_record(
    row: dict[str, Any], milp_evidence: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Dispatch reconstruction of a nonabelian archive row."""
    if row["label"] == "LP-168-4-14":
        return _nonabelian_168_structural_record(row, milp_evidence)
    if row["label"] == "LP-180-4-15":
        return _nonabelian_180_structural_record(row, milp_evidence)
    if row["label"] == "LP-192-4-16":
        return _nonabelian_192_structural_record(row, milp_evidence)
    if row["label"] != "LP-132-4-12":
        raise ValueError(f"unsupported nonabelian row: {row['label']}")

    coordinates = row["index_coordinates"]
    g5 = coordinates[5]
    g14 = coordinates[14]
    g39 = coordinates[39]
    support_a = (GROUP66_IDENTITY, g5)
    support_b = (GROUP66_IDENTITY, g14, g39)

    subgroup_a = _group66_generated_subgroup((g5,))
    subgroup_b = _group66_generated_subgroup((g14, g39))
    intersection = subgroup_a & subgroup_b
    central_c11 = {(exponent, 0, 0) for exponent in range(11)}
    double_coset = {
        group66_multiply(left, right)
        for left in subgroup_a
        for right in subgroup_b
    }

    def is_abelian(group: set[Group66Element]) -> bool:
        return all(
            group66_multiply(left, right) == group66_multiply(right, left)
            for left in group
            for right in group
        )

    def is_normal(
        subgroup: set[Group66Element], ambient: set[Group66Element]
    ) -> bool:
        return all(
            group66_multiply(
                group66_multiply(element, item), _group66_inverse(element)
            )
            in subgroup
            for element in ambient
            for item in subgroup
        )

    if _group66_order(g5) != 22:
        raise AssertionError("archive element g5 should have order 22")
    if _group66_order(g14) != 33 or _group66_order(g39) != 33:
        raise AssertionError("archive elements g14 and g39 should have order 33")
    if _group66_power(g14, 8) != g39:
        raise AssertionError("expected g39 = g14^8")
    if _group66_power(g5, 2) != _group66_power(g14, 12):
        raise AssertionError("expected overlap relation g5^2 = g14^12")
    if len(subgroup_a) != 22 or len(subgroup_b) != 33:
        raise AssertionError("unexpected support-subgroup orders")
    if intersection != central_c11:
        raise AssertionError("support-subgroup intersection is not central C11")
    if not is_abelian(intersection):
        raise AssertionError("support-subgroup intersection is not abelian")
    if not all(
        group66_multiply(item, element) == group66_multiply(element, item)
        for item in intersection
        for element in GROUP66_ELEMENTS
    ):
        raise AssertionError("support-subgroup intersection is not central")
    if not is_normal(intersection, subgroup_a) or not is_normal(
        intersection, subgroup_b
    ):
        raise AssertionError("support-subgroup intersection is not normal")
    if double_coset != set(GROUP66_ELEMENTS):
        raise AssertionError("support subgroups do not cover the ambient group")

    matrix_a, matrix_b = _nonabelian_132_blocks()
    matrix_x, matrix_z = nonabelian_132_css_checks()
    matrix_ab = (matrix_a @ matrix_b) % 2
    rank_a = _gf2_rank(matrix_a)
    rank_b = _gf2_rank(matrix_b)
    rank_ab = _gf2_rank(matrix_ab)
    rank_x = _gf2_rank(matrix_x)
    rank_z = _gf2_rank(matrix_z)
    n = matrix_x.shape[1]
    k = n - rank_x - rank_z
    subsystem_k = row["group_order"] - rank_a - rank_b + rank_ab
    delta_x = row["group_order"] - subsystem_k - rank_x
    delta_z = row["group_order"] - subsystem_k - rank_z
    component_count = _tanner_component_count(matrix_x, matrix_z)

    if (rank_a, rank_b, rank_ab, rank_x, rank_z) != (63, 62, 61, 64, 64):
        raise AssertionError("unexpected reconstructed rank tuple")
    if k != row["reported_k"] or component_count != 1:
        raise AssertionError("unexpected dimension or connectivity")
    if subsystem_k != 2 or (delta_x, delta_z) != (0, 0):
        raise AssertionError("unexpected block-erasure dimension or rank defects")
    if not np.all(matrix_x.sum(axis=1) == 5) or not np.all(
        matrix_z.sum(axis=1) == 5
    ):
        raise AssertionError("reconstructed checks are not uniformly weight five")

    return {
        "label": row["label"],
        "archive": {
            "member": row["member"],
            "member_sha256": row["file_sha256"],
            "row": row["archive_row"],
            "group_order": row["group_order"],
            "small_group_id": row["small_group_id"],
            "a_element_indices": list(row["a_indices"]),
            "b_element_indices": list(row["b_indices"]),
            "reported_k": row["reported_k"],
            "reported_d_upper_bound": row["reported_d"],
        },
        "group_decode": {
            "structure": "C11 x S3",
            "presentation": (
                "<c,r,s | c^11=r^3=s^2=1, [c,r]=[c,s]=1, "
                "s*r*s=r^-1>"
            ),
            "coordinate_convention": "(a,b,d) denotes c^a r^b s^d",
            "coordinate_enumeration": (
                "lexicographic a=0..10, b=0..2, d=0..1; matrix hashes use "
                "this order, not GAP Elements(G) order"
            ),
            "index_to_coordinate": {
                str(index): list(coordinate)
                for index, coordinate in sorted(coordinates.items())
            },
            "index_to_word": {"5": "c*s", "14": "c^2*r", "39": "c^5*r^2"},
            "element_orders": {"5": 22, "14": 33, "39": 33},
            "relation": "g39 = g14^8",
            "support_generator_overlap_relation": "g5^2 = g14^12 = c^2",
            "a_support_coordinates": [list(item) for item in support_a],
            "b_support_coordinates": [list(item) for item in support_b],
            "regular_representation": {
                "A": "I + L(g5)",
                "B": "I + R(g14) + R(g39)",
                "H_X": "[A | B]",
                "H_Z": "[B^T | A^T]",
            },
        },
        "support_subgroups": {
            "G_a": {"generator": "g5", "structure": "C22", "order": 22},
            "G_b": {"generator": "g14", "structure": "C33", "order": 33},
            "intersection": {
                "symbol": "N",
                "generator": "c",
                "structure": "C11",
                "order": 11,
                "abelian": True,
                "central_in_ambient_group": True,
                "normal_in_G_a": True,
                "normal_in_G_b": True,
            },
            "double_coset_order": len(double_coset),
            "double_coset_covers_ambient_group": True,
            "quotient_by_intersection": {
                "ambient_group": "S3",
                "G_a_over_N": "C2=<s>",
                "G_b_over_N": "C3=<r>",
                "a_support": "1+s",
                "b_support": "1+r+r^2",
            },
            "extension_splitting": {
                "G_a_over_N_splits": True,
                "G_b_over_N_splits": True,
            },
        },
        "reconstructed_code": {
            "n": n,
            "k": k,
            "fom_kd2_over_n": row["reported_k"] * row["reported_d"] ** 2 / n,
            "stabilizer_weight": 5,
            "connected": component_count == 1,
            "tanner_components": component_count,
            "ranks": {
                "A": rank_a,
                "B": rank_b,
                "AB": rank_ab,
                "H_X": rank_x,
                "H_Z": rank_z,
            },
            "block_erasure_subsystem_k": subsystem_k,
            "rank_defects": {"delta_X": delta_x, "delta_Z": delta_z},
            "matrix_a_sha256": _matrix_sha256(matrix_a),
            "matrix_b_sha256": _matrix_sha256(matrix_b),
            "matrix_ab_sha256": _matrix_sha256(matrix_ab),
            "matrix_x_sha256": _matrix_sha256(matrix_x),
            "matrix_z_sha256": _matrix_sha256(matrix_z),
        },
        "family_classification": {
            "ambient_group_nonabelian": True,
            "lin_pryadko_quasi_abelian_lifted_product": True,
            "equivalent_hypergraph_product_base_ring": "F2[C11]",
            "semi_abelian_zero_rank_defects": True,
            "essentially_nonabelian_in_rank_defect_sense": False,
            "equivalence_to_abelian_2BGA_established": False,
            "conclusion": (
                "This representative is nonabelian only at the ambient-group "
                "level: its abelian normal support intersection N=C11 puts it "
                "in the established quasi-abelian lifted-product class.  The "
                "classification does not by itself prove equivalence to an "
                "abelian 2BGA presentation."
            ),
        },
        "exact_milp": milp_evidence.get(row["label"]),
    }


def _nonabelian_168_structural_record(
    row: dict[str, Any], milp_evidence: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Reconstruct the order-84 row without overclaiming its classification."""
    central_order = 7
    s_order = 4
    identity = (0, 0, 0)
    elements = _metacyclic_elements(central_order, s_order)
    coordinates = row["index_coordinates"]
    g6 = coordinates[6]
    g11 = coordinates[11]
    g60 = coordinates[60]
    support_a = (identity, g6)
    support_b = (identity, g11, g60)

    subgroup_a = _metacyclic_generated_subgroup(
        (g6,), central_order=central_order, s_order=s_order
    )
    subgroup_b = _metacyclic_generated_subgroup(
        (g11, g60), central_order=central_order, s_order=s_order
    )
    intersection = subgroup_a & subgroup_b
    expected_intersection = {
        (c_exp, 0, s_exp) for c_exp in range(7) for s_exp in (0, 2)
    }
    double_coset = {
        _metacyclic_multiply(
            left,
            right,
            central_order=central_order,
            s_order=s_order,
        )
        for left in subgroup_a
        for right in subgroup_b
    }

    def multiply(
        left: MetacyclicElement, right: MetacyclicElement
    ) -> MetacyclicElement:
        return _metacyclic_multiply(
            left,
            right,
            central_order=central_order,
            s_order=s_order,
        )

    def inverse(element: MetacyclicElement) -> MetacyclicElement:
        return _metacyclic_inverse(
            element, central_order=central_order, s_order=s_order
        )

    def is_normal(
        subgroup: set[MetacyclicElement], ambient: set[MetacyclicElement]
    ) -> bool:
        return all(
            multiply(multiply(element, item), inverse(element)) in subgroup
            for element in ambient
            for item in subgroup
        )

    if (
        _metacyclic_order(g6, central_order=7, s_order=4),
        _metacyclic_order(g11, central_order=7, s_order=4),
        _metacyclic_order(g60, central_order=7, s_order=4),
    ) != (28, 21, 42):
        raise AssertionError("unexpected order-84 archive element orders")
    if _metacyclic_power(g60, 26, central_order=7, s_order=4) != g11:
        raise AssertionError("expected g11 = g60^26")
    if _metacyclic_power(g6, 2, central_order=7, s_order=4) != (
        _metacyclic_power(g60, 3, central_order=7, s_order=4)
    ):
        raise AssertionError("expected overlap relation g6^2 = g60^3")
    if len(subgroup_a) != 28 or len(subgroup_b) != 42:
        raise AssertionError("unexpected order-84 support-subgroup orders")
    if intersection != expected_intersection or len(intersection) != 14:
        raise AssertionError("unexpected order-84 support intersection")
    intersection_abelian = all(
        multiply(left, right) == multiply(right, left)
        for left in intersection
        for right in intersection
    )
    intersection_central = all(
        multiply(item, element) == multiply(element, item)
        for item in intersection
        for element in elements
    )
    normal_in_a = is_normal(intersection, subgroup_a)
    normal_in_b = is_normal(intersection, subgroup_b)
    if not all((intersection_abelian, intersection_central, normal_in_a, normal_in_b)):
        raise AssertionError("unexpected order-84 intersection properties")
    if double_coset != set(elements):
        raise AssertionError("order-84 support subgroups do not cover the ambient group")

    matrix_a, matrix_b = _nonabelian_168_blocks()
    matrix_x, matrix_z = nonabelian_168_css_checks()
    matrix_ab = (matrix_a @ matrix_b) % 2
    rank_a = _gf2_rank(matrix_a)
    rank_b = _gf2_rank(matrix_b)
    rank_ab = _gf2_rank(matrix_ab)
    rank_x = _gf2_rank(matrix_x)
    rank_z = _gf2_rank(matrix_z)
    n = matrix_x.shape[1]
    k = n - rank_x - rank_z
    subsystem_k = row["group_order"] - rank_a - rank_b + rank_ab
    delta_x = row["group_order"] - subsystem_k - rank_x
    delta_z = row["group_order"] - subsystem_k - rank_z
    component_count = _tanner_component_count(matrix_x, matrix_z)
    if (rank_a, rank_b, rank_ab, rank_x, rank_z) != (81, 74, 71, 82, 82):
        raise AssertionError("unexpected order-84 reconstructed rank tuple")
    if k != row["reported_k"] or component_count != 1:
        raise AssertionError("unexpected order-84 dimension or connectivity")
    if subsystem_k != 0 or (delta_x, delta_z) != (2, 2):
        raise AssertionError("unexpected order-84 subsystem dimension or rank defects")
    if not np.all(matrix_x.sum(axis=1) == 5) or not np.all(
        matrix_z.sum(axis=1) == 5
    ):
        raise AssertionError("order-84 reconstructed checks are not weight five")

    return {
        "label": row["label"],
        "archive": {
            "member": row["member"],
            "member_sha256": row["file_sha256"],
            "row": row["archive_row"],
            "group_order": row["group_order"],
            "small_group_id": row["small_group_id"],
            "a_element_indices": list(row["a_indices"]),
            "b_element_indices": list(row["b_indices"]),
            "reported_k": row["reported_k"],
            "reported_d_upper_bound": row["reported_d"],
        },
        "group_decode": {
            "structure": "C7 x (C3 : C4)",
            "presentation": (
                "<c,r,s | c^7=r^3=s^4=1, [c,r]=[c,s]=1, "
                "s*r*s^-1=r^-1>"
            ),
            "coordinate_convention": "(a,b,d) denotes c^a r^b s^d",
            "coordinate_enumeration": (
                "lexicographic a=0..6, b=0..2, d=0..3; matrix hashes use "
                "this order, not GAP Elements(G) order"
            ),
            "index_to_coordinate": {
                str(index): list(coordinate)
                for index, coordinate in sorted(coordinates.items())
            },
            "index_to_word": {"6": "c*s", "11": "c*r", "60": "c^3*r^2*s^2"},
            "element_orders": {"6": 28, "11": 21, "60": 42},
            "relation": "g11 = g60^26",
            "support_generator_overlap_relation": "g6^2 = g60^3 = c^2*s^2",
            "a_support_coordinates": [list(item) for item in support_a],
            "b_support_coordinates": [list(item) for item in support_b],
            "regular_representation": {
                "A": "I + L(g6)",
                "B": "I + R(g11) + R(g60)",
                "H_X": "[A | B]",
                "H_Z": "[B^T | A^T]",
            },
        },
        "support_subgroups": {
            "G_a": {"generator": "g6", "structure": "C28", "order": 28},
            "G_b": {"generator": "g60", "structure": "C42", "order": 42},
            "intersection": {
                "symbol": "N",
                "generator": "g6^2 = g60^3",
                "structure": "C14",
                "order": 14,
                "abelian": intersection_abelian,
                "central_in_ambient_group": intersection_central,
                "normal_in_G_a": normal_in_a,
                "normal_in_G_b": normal_in_b,
            },
            "double_coset_order": len(double_coset),
            "double_coset_covers_ambient_group": True,
            "quotient_by_intersection": {
                "ambient_group": "S3",
                "G_a_over_N": "C2",
                "G_b_over_N": "C3",
                "a_support": "1+s",
                "b_support": "1+r+r^2",
            },
            "extension_splitting": {
                "G_a_over_N_splits": False,
                "obstruction": (
                    "C28 has no C2 complement to N=C14: its unique order-two "
                    "subgroup lies inside N"
                ),
            },
        },
        "reconstructed_code": {
            "n": n,
            "k": k,
            "reported_fom_upper_bound": row["reported_k"]
            * row["reported_d"] ** 2
            / n,
            "stabilizer_weight": 5,
            "connected": component_count == 1,
            "tanner_components": component_count,
            "ranks": {
                "A": rank_a,
                "B": rank_b,
                "AB": rank_ab,
                "H_X": rank_x,
                "H_Z": rank_z,
            },
            "block_erasure_subsystem_k": subsystem_k,
            "rank_defects": {"delta_X": delta_x, "delta_Z": delta_z},
            "matrix_a_sha256": _matrix_sha256(matrix_a),
            "matrix_b_sha256": _matrix_sha256(matrix_b),
            "matrix_ab_sha256": _matrix_sha256(matrix_ab),
            "matrix_x_sha256": _matrix_sha256(matrix_x),
            "matrix_z_sha256": _matrix_sha256(matrix_z),
        },
        "family_classification": {
            "ambient_group_nonabelian": True,
            "nonzero_rank_defects_verified": True,
            "essentially_nonabelian_in_rank_defect_sense": True,
            "lin_pryadko_statement_as_written_counterexample": True,
            "source_caveat": (
                "Statements 8--9 of arXiv:2306.16400 assert that an abelian "
                "intersection N normal in both support groups yields the "
                "quasi-abelian/zero-defect reduction.  Their proof assumes "
                "without justification that the extensions by N split.  Here "
                "G_a=C28 is a nonsplit extension of N=C14 by C2, and the direct "
                "matrix ranks give delta_X=delta_Z=2.  Thus the row is a "
                "counterexample to the zero-defect statement as written; a "
                "split-extension hypothesis repairs the application."
            ),
        },
        "milp_attempt": milp_evidence.get(row["label"]),
    }


def _nonabelian_180_structural_record(
    row: dict[str, Any], milp_evidence: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    central_order = 15
    s_order = 2
    identity = (0, 0, 0)
    elements = _metacyclic_elements(central_order, s_order)
    coordinates = row["index_coordinates"]
    g10 = coordinates[10]
    g6 = coordinates[6]
    g38 = coordinates[38]
    support_a = (identity, g10)
    support_b = (identity, g6, g38)
    subgroup_a = _metacyclic_generated_subgroup(
        (g10,), central_order=central_order, s_order=s_order
    )
    subgroup_b = _metacyclic_generated_subgroup(
        (g6, g38), central_order=central_order, s_order=s_order
    )
    intersection = subgroup_a & subgroup_b
    expected_intersection = {(c_exp, 0, 0) for c_exp in range(15)}

    def multiply(
        left: MetacyclicElement, right: MetacyclicElement
    ) -> MetacyclicElement:
        return _metacyclic_multiply(
            left,
            right,
            central_order=central_order,
            s_order=s_order,
        )

    double_coset = {
        multiply(left, right) for left in subgroup_a for right in subgroup_b
    }
    if (
        _metacyclic_order(g10, central_order=15, s_order=2),
        _metacyclic_order(g6, central_order=15, s_order=2),
        _metacyclic_order(g38, central_order=15, s_order=2),
    ) != (15, 6, 15):
        raise AssertionError("unexpected order-90 archive element orders")
    if len(subgroup_a) != 15 or len(subgroup_b) != 90:
        raise AssertionError("unexpected order-90 support-subgroup orders")
    if intersection != expected_intersection:
        raise AssertionError("unexpected order-90 support intersection")
    if not all(
        multiply(item, element) == multiply(element, item)
        for item in intersection
        for element in elements
    ):
        raise AssertionError("order-90 support intersection is not central")
    if double_coset != set(elements):
        raise AssertionError("order-90 support subgroups do not cover the ambient group")

    matrix_a, matrix_b = _nonabelian_180_blocks()
    matrix_x, matrix_z = nonabelian_180_css_checks()
    matrix_ab = (matrix_a @ matrix_b) % 2
    rank_a = _gf2_rank(matrix_a)
    rank_b = _gf2_rank(matrix_b)
    rank_ab = _gf2_rank(matrix_ab)
    rank_x = _gf2_rank(matrix_x)
    rank_z = _gf2_rank(matrix_z)
    n = matrix_x.shape[1]
    k = n - rank_x - rank_z
    subsystem_k = row["group_order"] - rank_a - rank_b + rank_ab
    delta_x = row["group_order"] - subsystem_k - rank_x
    delta_z = row["group_order"] - subsystem_k - rank_z
    component_count = _tanner_component_count(matrix_x, matrix_z)
    if (rank_a, rank_b, rank_ab, rank_x, rank_z) != (84, 78, 74, 88, 88):
        raise AssertionError("unexpected order-90 reconstructed rank tuple")
    if k != row["reported_k"] or component_count != 1:
        raise AssertionError("unexpected order-90 dimension or connectivity")
    if subsystem_k != 2 or (delta_x, delta_z) != (0, 0):
        raise AssertionError("unexpected order-90 subsystem dimension or rank defects")
    if not np.all(matrix_x.sum(axis=1) == 5) or not np.all(
        matrix_z.sum(axis=1) == 5
    ):
        raise AssertionError("order-90 reconstructed checks are not weight five")

    return {
        "label": row["label"],
        "archive": {
            "member": row["member"],
            "member_sha256": row["file_sha256"],
            "row": row["archive_row"],
            "group_order": row["group_order"],
            "small_group_id": row["small_group_id"],
            "a_element_indices": list(row["a_indices"]),
            "b_element_indices": list(row["b_indices"]),
            "reported_k": row["reported_k"],
            "reported_d_upper_bound": row["reported_d"],
        },
        "group_decode": {
            "structure": "C15 x S3",
            "presentation": (
                "<c,r,s | c^15=r^3=s^2=1, [c,r]=[c,s]=1, "
                "s*r*s=r^-1>"
            ),
            "coordinate_convention": "(a,b,d) denotes c^a r^b s^d",
            "coordinate_enumeration": (
                "lexicographic a=0..14, b=0..2, d=0..1; matrix hashes use "
                "this order, not GAP Elements(G) order"
            ),
            "index_to_coordinate": {
                str(index): list(coordinate)
                for index, coordinate in sorted(coordinates.items())
            },
            "index_to_word": {"6": "c^10*s", "10": "c", "38": "c^11*r"},
            "element_orders": {"6": 6, "10": 15, "38": 15},
            "a_support_coordinates": [list(item) for item in support_a],
            "b_support_coordinates": [list(item) for item in support_b],
            "regular_representation": {
                "A": "I + L(g10)",
                "B": "I + R(g6) + R(g38)",
                "H_X": "[A | B]",
                "H_Z": "[B^T | A^T]",
            },
        },
        "support_subgroups": {
            "G_a": {"generator": "g10", "structure": "C15", "order": 15},
            "G_b": {
                "generators": ["g6", "g38"],
                "structure": "C15 x S3",
                "order": 90,
            },
            "intersection": {
                "symbol": "N=G_a",
                "generator": "c=g10",
                "structure": "C15",
                "order": 15,
                "abelian": True,
                "central_in_ambient_group": True,
                "normal_in_G_a": True,
                "normal_in_G_b": True,
            },
            "double_coset_order": len(double_coset),
            "double_coset_covers_ambient_group": True,
            "quotient_by_intersection": {
                "ambient_group": "S3",
                "G_a_over_N": "trivial",
                "G_b_over_N": "S3",
                "a_support": "1+c lies in F2[N]",
                "b_support": "1+s+r",
            },
            "extension_splitting": {
                "G_a_over_N_splits": True,
                "G_b_over_N_splits": True,
            },
        },
        "reconstructed_code": {
            "n": n,
            "k": k,
            "reported_fom_upper_bound": row["reported_k"]
            * row["reported_d"] ** 2
            / n,
            "stabilizer_weight": 5,
            "connected": component_count == 1,
            "tanner_components": component_count,
            "ranks": {
                "A": rank_a,
                "B": rank_b,
                "AB": rank_ab,
                "H_X": rank_x,
                "H_Z": rank_z,
            },
            "block_erasure_subsystem_k": subsystem_k,
            "rank_defects": {"delta_X": delta_x, "delta_Z": delta_z},
            "matrix_a_sha256": _matrix_sha256(matrix_a),
            "matrix_b_sha256": _matrix_sha256(matrix_b),
            "matrix_ab_sha256": _matrix_sha256(matrix_ab),
            "matrix_x_sha256": _matrix_sha256(matrix_x),
            "matrix_z_sha256": _matrix_sha256(matrix_z),
        },
        "family_classification": {
            "ambient_group_nonabelian": True,
            "lin_pryadko_quasi_abelian_lifted_product": True,
            "equivalent_hypergraph_product_base_ring": "F2[C15]",
            "semi_abelian_zero_rank_defects": True,
            "essentially_nonabelian_in_rank_defect_sense": False,
            "equivalence_to_abelian_2BGA_established": False,
        },
        "milp_attempt": milp_evidence.get(row["label"]),
    }


def _nonabelian_192_structural_record(
    row: dict[str, Any], milp_evidence: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    central_order = 16
    s_order = 2
    identity = (0, 0, 0)
    elements = _metacyclic_elements(central_order, s_order)
    coordinates = row["index_coordinates"]
    g3 = coordinates[3]
    g8 = coordinates[8]
    g46 = coordinates[46]
    support_a = (identity, g3)
    support_b = (identity, g8, g46)
    subgroup_a = _metacyclic_generated_subgroup(
        (g3,), central_order=central_order, s_order=s_order
    )
    subgroup_b = _metacyclic_generated_subgroup(
        (g8, g46), central_order=central_order, s_order=s_order
    )
    intersection = subgroup_a & subgroup_b
    expected_intersection = {(c_exp, 0, 0) for c_exp in range(0, 16, 2)}

    def multiply(
        left: MetacyclicElement, right: MetacyclicElement
    ) -> MetacyclicElement:
        return _metacyclic_multiply(
            left,
            right,
            central_order=central_order,
            s_order=s_order,
        )

    def inverse(element: MetacyclicElement) -> MetacyclicElement:
        return _metacyclic_inverse(
            element, central_order=central_order, s_order=s_order
        )

    def is_normal(
        subgroup: set[MetacyclicElement], ambient: set[MetacyclicElement]
    ) -> bool:
        return all(
            multiply(multiply(element, item), inverse(element)) in subgroup
            for element in ambient
            for item in subgroup
        )

    double_coset = {
        multiply(left, right) for left in subgroup_a for right in subgroup_b
    }
    if (
        _metacyclic_order(g3, central_order=16, s_order=2),
        _metacyclic_order(g8, central_order=16, s_order=2),
        _metacyclic_order(g46, central_order=16, s_order=2),
    ) != (16, 16, 12):
        raise AssertionError("unexpected order-96 archive element orders")
    if _metacyclic_power(g3, 2, central_order=16, s_order=2) != (
        _metacyclic_power(g8, 2, central_order=16, s_order=2)
    ):
        raise AssertionError("expected overlap relation g3^2 = g8^2")
    if len(subgroup_a) != 16 or len(subgroup_b) != 48:
        raise AssertionError("unexpected order-96 support-subgroup orders")
    if intersection != expected_intersection:
        raise AssertionError("unexpected order-96 support intersection")
    intersection_abelian = all(
        multiply(left, right) == multiply(right, left)
        for left in intersection
        for right in intersection
    )
    intersection_central = all(
        multiply(item, element) == multiply(element, item)
        for item in intersection
        for element in elements
    )
    normal_in_a = is_normal(intersection, subgroup_a)
    normal_in_b = is_normal(intersection, subgroup_b)
    if not all((intersection_abelian, intersection_central, normal_in_a, normal_in_b)):
        raise AssertionError("unexpected order-96 intersection properties")
    if double_coset != set(elements):
        raise AssertionError("order-96 support subgroups do not cover the ambient group")

    matrix_a, matrix_b = _nonabelian_192_blocks()
    matrix_x, matrix_z = nonabelian_192_css_checks()
    matrix_ab = (matrix_a @ matrix_b) % 2
    rank_a = _gf2_rank(matrix_a)
    rank_b = _gf2_rank(matrix_b)
    rank_ab = _gf2_rank(matrix_ab)
    rank_x = _gf2_rank(matrix_x)
    rank_z = _gf2_rank(matrix_z)
    n = matrix_x.shape[1]
    k = n - rank_x - rank_z
    subsystem_k = row["group_order"] - rank_a - rank_b + rank_ab
    delta_x = row["group_order"] - subsystem_k - rank_x
    delta_z = row["group_order"] - subsystem_k - rank_z
    component_count = _tanner_component_count(matrix_x, matrix_z)
    if (rank_a, rank_b, rank_ab, rank_x, rank_z) != (90, 92, 86, 94, 94):
        raise AssertionError("unexpected order-96 reconstructed rank tuple")
    if k != row["reported_k"] or component_count != 1:
        raise AssertionError("unexpected order-96 dimension or connectivity")
    if subsystem_k != 0 or (delta_x, delta_z) != (2, 2):
        raise AssertionError("unexpected order-96 subsystem dimension or rank defects")
    if not np.all(matrix_x.sum(axis=1) == 5) or not np.all(
        matrix_z.sum(axis=1) == 5
    ):
        raise AssertionError("order-96 reconstructed checks are not weight five")

    return {
        "label": row["label"],
        "archive": {
            "member": row["member"],
            "member_sha256": row["file_sha256"],
            "row": row["archive_row"],
            "group_order": row["group_order"],
            "small_group_id": row["small_group_id"],
            "a_element_indices": list(row["a_indices"]),
            "b_element_indices": list(row["b_indices"]),
            "reported_k": row["reported_k"],
            "reported_d_upper_bound": row["reported_d"],
        },
        "group_decode": {
            "structure": "C16 x S3",
            "presentation": (
                "<c,r,s | c^16=r^3=s^2=1, [c,r]=[c,s]=1, "
                "s*r*s=r^-1>"
            ),
            "coordinate_convention": "(a,b,d) denotes c^a r^b s^d",
            "coordinate_enumeration": (
                "lexicographic a=0..15, b=0..2, d=0..1; matrix hashes use "
                "this order, not GAP Elements(G) order"
            ),
            "index_to_coordinate": {
                str(index): list(coordinate)
                for index, coordinate in sorted(coordinates.items())
            },
            "index_to_word": {"3": "c", "8": "c*s", "46": "c^12*r"},
            "element_orders": {"3": 16, "8": 16, "46": 12},
            "support_generator_overlap_relation": "g3^2 = g8^2 = c^2",
            "a_support_coordinates": [list(item) for item in support_a],
            "b_support_coordinates": [list(item) for item in support_b],
            "regular_representation": {
                "A": "I + L(g3)",
                "B": "I + R(g8) + R(g46)",
                "H_X": "[A | B]",
                "H_Z": "[B^T | A^T]",
            },
        },
        "support_subgroups": {
            "G_a": {"generator": "g3", "structure": "C16", "order": 16},
            "G_b": {
                "generators": ["g8", "g46"],
                "structure": "C3 : C16",
                "order": 48,
            },
            "intersection": {
                "symbol": "N",
                "generator": "c^2=g3^2=g8^2",
                "structure": "C8",
                "order": 8,
                "abelian": intersection_abelian,
                "central_in_ambient_group": intersection_central,
                "normal_in_G_a": normal_in_a,
                "normal_in_G_b": normal_in_b,
            },
            "double_coset_order": len(double_coset),
            "double_coset_covers_ambient_group": True,
            "quotient_by_intersection": {
                "ambient_group": "C2 x S3",
                "G_a_over_N": "C2",
                "G_b_over_N": "S3",
                "a_support": "1+c_bar",
                "b_support": "1+c_bar*s+r",
            },
            "extension_splitting": {
                "G_a_over_N_splits": False,
                "obstruction": (
                    "C16 has no C2 complement to N=C8: its unique order-two "
                    "subgroup lies inside N"
                ),
            },
        },
        "reconstructed_code": {
            "n": n,
            "k": k,
            "reported_fom_upper_bound": row["reported_k"]
            * row["reported_d"] ** 2
            / n,
            "stabilizer_weight": 5,
            "connected": component_count == 1,
            "tanner_components": component_count,
            "ranks": {
                "A": rank_a,
                "B": rank_b,
                "AB": rank_ab,
                "H_X": rank_x,
                "H_Z": rank_z,
            },
            "block_erasure_subsystem_k": subsystem_k,
            "rank_defects": {"delta_X": delta_x, "delta_Z": delta_z},
            "matrix_a_sha256": _matrix_sha256(matrix_a),
            "matrix_b_sha256": _matrix_sha256(matrix_b),
            "matrix_ab_sha256": _matrix_sha256(matrix_ab),
            "matrix_x_sha256": _matrix_sha256(matrix_x),
            "matrix_z_sha256": _matrix_sha256(matrix_z),
        },
        "family_classification": {
            "ambient_group_nonabelian": True,
            "nonzero_rank_defects_verified": True,
            "essentially_nonabelian_in_rank_defect_sense": True,
            "lin_pryadko_statement_as_written_counterexample": True,
            "source_caveat": (
                "Statements 8--9 of arXiv:2306.16400 assert that an abelian "
                "intersection N normal in both support groups yields the "
                "quasi-abelian/zero-defect reduction.  Their proof assumes "
                "without justification that the extensions by N split.  Here "
                "G_a=C16 is a nonsplit extension of N=C8 by C2, and the direct "
                "matrix ranks give delta_X=delta_Z=2.  Thus the row is a "
                "counterexample to the zero-defect statement as written; a "
                "split-extension hypothesis repairs the application."
            ),
        },
        "milp_attempt": milp_evidence.get(row["label"]),
    }


def branch_to_cyclic_exponent(x_exponent: int, y_exponent: int) -> int:
    """Map C16 x C3 to C48 in the generator used by the archived row.

    Set ``x_prime=x^3``.  Since ``3^-1 = 11 (mod 16)``, an original term
    ``x^i y^j`` is ``x_prime^(11i) y^j``.  The CRT embedding
    ``x_prime -> z^3`` and ``y -> z^16`` then gives the formula below.
    """
    return (3 * ((11 * x_exponent) % 16) + 16 * y_exponent) % 48


def _abelian_group_algebra_matrix(
    ell: int, m: int, support: tuple[tuple[int, int], ...]
) -> np.ndarray:
    elements = [(x_exp, y_exp) for x_exp in range(ell) for y_exp in range(m)]
    positions = {element: index for index, element in enumerate(elements)}
    matrix = np.zeros((ell * m, ell * m), dtype=np.uint8)
    for column, (x_exp, y_exp) in enumerate(elements):
        for shift_x, shift_y in support:
            target = ((x_exp + shift_x) % ell, (y_exp + shift_y) % m)
            matrix[positions[target], column] ^= 1
    return matrix


def _equivalence_record() -> dict[str, Any]:
    branch_a = BRANCH_COMPONENT["a_terms"]
    branch_b = BRANCH_COMPONENT["b_terms"]
    mapped_a = tuple(branch_to_cyclic_exponent(*term) for term in branch_a)
    mapped_b = tuple(branch_to_cyclic_exponent(*term) for term in branch_b)

    archive_row = next(row for row in ARCHIVE_ROWS if row["label"] == "LP-96-4-10")
    archive_a, archive_b = decoded_supports(archive_row)

    # Verify the relabeling at matrix level, not only as a support-set claim.
    elements = [(x_exp, y_exp) for x_exp in range(16) for y_exp in range(3)]
    permutation = np.zeros((48, 48), dtype=np.uint8)
    images = []
    for source_index, element in enumerate(elements):
        target_index = branch_to_cyclic_exponent(*element)
        permutation[target_index, source_index] = 1
        images.append(target_index)
    if sorted(images) != list(range(48)):
        raise AssertionError("CRT map is not bijective")

    branch_a_matrix = _abelian_group_algebra_matrix(16, 3, branch_a)
    branch_b_matrix = _abelian_group_algebra_matrix(16, 3, branch_b)
    mapped_a_matrix = permutation @ branch_a_matrix @ permutation.T
    mapped_b_matrix = permutation @ branch_b_matrix @ permutation.T
    if not np.array_equal(mapped_a_matrix, cyclic_group_algebra_matrix(48, mapped_a)):
        raise AssertionError("CRT relabeling failed for branch A")
    if not np.array_equal(mapped_b_matrix, cyclic_group_algebra_matrix(48, mapped_b)):
        raise AssertionError("CRT relabeling failed for branch B")
    if set(mapped_a) != set(archive_b) or set(mapped_b) != set(archive_a):
        raise AssertionError("archive supports do not match after block exchange")

    return {
        "result": "exact_presentation_equivalence",
        "branch_component": {
            **BRANCH_COMPONENT,
            "a_terms": [list(term) for term in branch_a],
            "b_terms": [list(term) for term in branch_b],
        },
        "map": {
            "x_prime_definition": "x_prime = x^3 in C16",
            "inverse_of_3_mod_16": 11,
            "crt_embedding": {"x_prime": "z^3", "y": "z^16"},
            "formula": "(i,j) -> 3*(11*i mod 16)+16*j mod 48",
            "bijection_verified": True,
            "mapped_branch_a": list(mapped_a),
            "mapped_branch_b": list(mapped_b),
        },
        "archive_cyclic_supports": {
            "a": list(archive_a),
            "b": list(archive_b),
        },
        "required_code_relabeling": [
            "CRT permutation of the 48 group coordinates in each qubit block",
            "exchange the two 48-qubit blocks",
        ],
        "matrix_conjugation_verified": True,
        "conclusion": (
            "The campaign [[96,4,10]] component and the archived "
            "SmallGroup(48,2) row define the same CSS code up to qubit/check "
            "permutation and exchange of the two bicycle blocks."
        ),
    }


def build_audit(
    milp_evidence: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if milp_evidence is None:
        milp_evidence = RECORDED_MILP
    rows = [_structural_record(row, milp_evidence) for row in ARCHIVE_ROWS]
    nonabelian_rows = [
        _nonabelian_structural_record(row, milp_evidence)
        for row in NONABELIAN_FRONTIER_ROWS
    ]
    return {
        "schema_version": 1,
        "record_type": "lin_pryadko_weight5_prior_art_audit",
        "archive": ARCHIVE,
        "gap_index_decoder": GAP_DECODER,
        "milp_method": {
            "implementation": "evaluation.distance_milp.ilp_min_weight",
            "formulation": "Landahl-Anderson-Rice binary MILP",
            "solver": "scipy.optimize.milp (HiGHS)",
            "timeout_per_logical_seconds": 300,
            "early_stop": None,
            "exactness_rule": (
                "all k X-type and all k Z-type logical objectives return a "
                "solver-proven optimum; d=min(d_X,d_Z)"
            ),
        },
        "rows": rows,
        "nonabelian_frontier_rows": nonabelian_rows,
        "campaign_96_equivalence": _equivalence_record(),
        "summary": {
            "prior_rows_exactly_certified": [
                row["label"] for row in rows if row["label"] != "LP-96-4-10"
            ],
            "archive_96_is_rediscovery": True,
            "all_six_connected": all(
                row["reconstructed_code"]["connected"] for row in rows
            ),
            "nonabelian_rows_exactly_certified": [
                row["label"]
                for row in nonabelian_rows
                if row.get("exact_milp") and row["exact_milp"]["exact"]
            ],
            "nonabelian_rows_quasi_abelian": [
                row["label"]
                for row in nonabelian_rows
                if row["family_classification"].get(
                    "lin_pryadko_quasi_abelian_lifted_product", False
                )
            ],
            "nonabelian_rows_with_incomplete_milp_attempts": [
                row["label"]
                for row in nonabelian_rows
                if row.get("milp_attempt") and not row["milp_attempt"]["exact"]
            ],
            "nonabelian_rows_with_nonzero_rank_defects": [
                row["label"]
                for row in nonabelian_rows
                if row["family_classification"].get(
                    "nonzero_rank_defects_verified", False
                )
            ],
        },
    }


def render_audit(audit: dict[str, Any]) -> str:
    return json.dumps(audit, indent=2, sort_keys=True) + "\n"


def _verify_archive_rows(
    path: Path, expected_zip_sha256: str, rows: tuple[dict[str, Any], ...]
) -> None:
    raw_zip = path.read_bytes()
    if _sha256(raw_zip) != expected_zip_sha256:
        raise AssertionError(f"unexpected archive hash for {path}")
    with zipfile.ZipFile(path) as archive:
        for row in rows:
            member_data = archive.read(row["member"])
            if _sha256(member_data) != row["file_sha256"]:
                raise AssertionError(f"unexpected member hash: {row['member']}")
            member_rows = {
                _canonical_row(line) for line in member_data.decode().splitlines()
            }
            if _canonical_row(row["archive_row"]) not in member_rows:
                raise AssertionError(f"archive row not found in {row['member']}")


def verify_archive_zip(path: Path) -> None:
    _verify_archive_rows(path, ARCHIVE["abelian_zip_sha256"], ARCHIVE_ROWS)


def verify_nonabelian_archive_zip(path: Path) -> None:
    _verify_archive_rows(
        path, ARCHIVE["nonabelian_zip_sha256"], NONABELIAN_FRONTIER_ROWS
    )


def _qldpc_code(row: dict[str, Any]):
    from qldpc import codes

    if row["label"] == "LP-132-4-12":
        matrix_x, matrix_z = nonabelian_132_css_checks()
    elif row["label"] == "LP-168-4-14":
        matrix_x, matrix_z = nonabelian_168_css_checks()
    elif row["label"] == "LP-180-4-15":
        matrix_x, matrix_z = nonabelian_180_css_checks()
    elif row["label"] == "LP-192-4-16":
        matrix_x, matrix_z = nonabelian_192_css_checks()
    else:
        a_support, b_support = decoded_supports(row)
        matrix_x, matrix_z = cyclic_css_checks(
            row["group_order"], a_support, b_support
        )
    return codes.CSSCode(matrix_x, matrix_z)


def run_exact_milp(timeout: int = 300) -> dict[str, dict[str, Any]]:
    """Run every logical objective and return deterministic proof metadata."""
    from evaluation.distance_milp import get_code_matrices, ilp_min_weight

    evidence: dict[str, dict[str, Any]] = {}
    exact_rows = (
        *ARCHIVE_ROWS,
        *(
            row
            for row in NONABELIAN_FRONTIER_ROWS
            if RECORDED_MILP[row["label"]]["exact"]
        ),
    )
    for row in exact_rows:
        code = _qldpc_code(row)
        matrix_x, matrix_z, logical_x, logical_z = get_code_matrices(code)
        z_objectives = []
        x_objectives = []
        print(f"MILP {row['label']} ({2 * code.dimension} objectives)", file=sys.stderr)
        for logical in logical_x:
            weight, optimal = ilp_min_weight(matrix_x, logical, timeout=timeout)
            z_objectives.append({"weight": weight, "optimal": bool(optimal)})
        for logical in logical_z:
            weight, optimal = ilp_min_weight(matrix_z, logical, timeout=timeout)
            x_objectives.append({"weight": weight, "optimal": bool(optimal)})

        all_objectives = x_objectives + z_objectives
        if any(item["weight"] is None for item in all_objectives):
            raise AssertionError(f"{row['label']}: MILP returned no incumbent")
        d_x = min(item["weight"] for item in x_objectives)
        d_z = min(item["weight"] for item in z_objectives)
        all_optimal = all(item["optimal"] for item in all_objectives)
        if not all_optimal:
            raise AssertionError(f"{row['label']}: at least one objective did not close")
        if min(d_x, d_z) != row["reported_d"]:
            raise AssertionError(
                f"{row['label']}: exact distance {min(d_x, d_z)} != "
                f"archive value {row['reported_d']}"
            )
        evidence[row["label"]] = {
            "d": min(d_x, d_z),
            "d_x": d_x,
            "d_z": d_z,
            "exact": True,
            "all_objectives_optimal": True,
            "logical_objectives_optimal": len(all_objectives),
            "logical_objectives_total": 2 * code.dimension,
            "x_objectives": x_objectives,
            "z_objectives": z_objectives,
        }
    return evidence


def _validate_milp_evidence(
    observed: dict[str, dict[str, Any]], expected: dict[str, dict[str, Any]]
) -> None:
    expected_exact = {
        label: record for label, record in expected.items() if record["exact"]
    }
    if set(observed) != set(expected_exact):
        raise AssertionError("MILP evidence labels differ from the checked-in audit")
    for label in observed:
        stable_fields = (
            "d",
            "d_x",
            "d_z",
            "exact",
            "all_objectives_optimal",
            "logical_objectives_optimal",
            "logical_objectives_total",
        )
        for field in stable_fields:
            if observed[label][field] != expected_exact[label][field]:
                raise AssertionError(
                    f"{label}: rerun {field}={observed[label][field]!r}, "
                    f"expected {expected_exact[label][field]!r}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run-milp",
        action="store_true",
        help="rerun all 56 exact logical-objective MILPs",
    )
    parser.add_argument(
        "--timeout-per-logical",
        type=int,
        default=300,
        help="HiGHS time limit for each logical objective (default: 300s)",
    )
    parser.add_argument(
        "--archive-zip",
        type=Path,
        help="optional pinned upstream abelian.zip to hash and inspect",
    )
    parser.add_argument(
        "--nonabelian-archive-zip",
        type=Path,
        help="optional pinned upstream nonabelian.zip to hash and inspect",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help=f"write the deterministic artifact to {DEFAULT_OUTPUT}",
    )
    args = parser.parse_args()

    if args.archive_zip:
        verify_archive_zip(args.archive_zip)
        print(f"archive snapshot verified: {args.archive_zip}")
    if args.nonabelian_archive_zip:
        verify_nonabelian_archive_zip(args.nonabelian_archive_zip)
        print(f"archive snapshot verified: {args.nonabelian_archive_zip}")

    observed_evidence = None
    if args.run_milp:
        observed_evidence = run_exact_milp(args.timeout_per_logical)
        _validate_milp_evidence(observed_evidence, RECORDED_MILP)
        print("all recorded exact MILP results reproduced")

    # Keep a validation-only rerun from making the deterministic artifact
    # depend on logical-basis choices in a newer qldpc release.  ``--write``
    # deliberately records the newly observed per-logical objective values.
    evidence_for_render = RECORDED_MILP
    if args.write and observed_evidence:
        evidence_for_render = {**RECORDED_MILP, **observed_evidence}
    rendered = render_audit(build_audit(evidence_for_render))

    if args.write:
        DEFAULT_OUTPUT.write_text(rendered, encoding="utf-8")
        print(f"wrote {DEFAULT_OUTPUT}")
    else:
        if not DEFAULT_OUTPUT.exists():
            raise FileNotFoundError(f"checked-in artifact missing: {DEFAULT_OUTPUT}")
        if DEFAULT_OUTPUT.read_text(encoding="utf-8") != rendered:
            raise AssertionError(f"checked-in artifact is stale: {DEFAULT_OUTPUT}")
        print(f"checked-in artifact verified: {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
