#!/usr/bin/env python3
"""Independent binary check of the fixed weight-five PBB baseline family.

The family is defined over

    F_2[x,y] / (x^ell - 1, y^m - 1)

by ``(A, B, C, D) = (1+x+x^2, 1+y, x, 1+y)``.  This script deliberately
does not use qLDPC or the repository's PBB constructor.  It builds every
translated stabilizer as a Python integer, performs Gaussian elimination over
F_2, and checks the explicit logical operator used in the accompanying proof
note.

What is certified for every checked instance is

* ``k = 2*gcd(m, 2)``; and
* nontrivial logicals of weights ``2*ell/3`` and ``m``, hence
  ``d <= min(2*ell/3, m)``.

The script does not claim or test a general matching distance lower bound.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import itertools
import json
import math
from typing import Iterable, Sequence


Term = tuple[int, int]


@dataclass(frozen=True)
class VerificationResult:
    ell: int
    m: int
    n: int
    top_rank: int
    bottom_rank: int
    intersection_dimension: int
    stabilizer_rank: int
    k: int
    expected_k: int
    stabilizers_commute: bool
    horizontal_logical_commutes: bool
    horizontal_logical_is_nontrivial: bool
    horizontal_logical_weight: int
    expected_horizontal_weight: int
    vertical_logical_commutes: bool
    vertical_logical_is_nontrivial: bool
    vertical_logical_weight: int
    expected_vertical_weight: int
    distance_upper_bound: int


def _index(x_exp: int, y_exp: int, ell: int, m: int) -> int:
    return (x_exp % ell) * m + (y_exp % m)


def _translated_polynomial(
    terms: Iterable[Term], shift: Term, ell: int, m: int
) -> int:
    """Return the translated group-ring element as an ``ell*m``-bit integer."""

    value = 0
    shift_x, shift_y = shift
    for x_exp, y_exp in terms:
        value ^= 1 << _index(x_exp + shift_x, y_exp + shift_y, ell, m)
    return value


def _star(terms: Iterable[Term], ell: int, m: int) -> tuple[Term, ...]:
    return tuple(((-x_exp) % ell, (-y_exp) % m) for x_exp, y_exp in terms)


def _symplectic_row(
    x_left: int,
    x_right: int,
    z_left: int,
    z_right: int,
    block_size: int,
) -> int:
    """Pack ``(X_left, X_right | Z_left, Z_right)`` into one integer."""

    n = 2 * block_size
    x_part = x_left | (x_right << block_size)
    z_part = z_left | (z_right << block_size)
    return x_part | (z_part << n)


def build_stabilizers(ell: int, m: int) -> tuple[list[int], list[int]]:
    """Build the two translated stabilizer blocks without external packages."""

    if ell < 3 or ell % 3:
        raise ValueError("ell must be a positive multiple of three with ell >= 3")
    if m < 2:
        raise ValueError("m must be at least two for the weight-five family")

    block_size = ell * m
    A = ((0, 0), (1, 0), (2, 0))
    B = ((0, 0), (0, 1))
    C = ((1, 0),)
    D = B
    A_star = _star(A, ell, m)
    B_star = _star(B, ell, m)

    top: list[int] = []
    bottom: list[int] = []
    for x_exp in range(ell):
        for y_exp in range(m):
            shift = (x_exp, y_exp)
            top.append(
                _symplectic_row(
                    _translated_polynomial(A, shift, ell, m),
                    _translated_polynomial(B, shift, ell, m),
                    _translated_polynomial(C, shift, ell, m),
                    _translated_polynomial(D, shift, ell, m),
                    block_size,
                )
            )
            bottom.append(
                _symplectic_row(
                    0,
                    0,
                    _translated_polynomial(B_star, shift, ell, m),
                    _translated_polynomial(A_star, shift, ell, m),
                    block_size,
                )
            )
    return top, bottom


def _gf2_basis(rows: Iterable[int]) -> dict[int, int]:
    """Return a row-echelon XOR basis keyed by its highest set bit."""

    basis: dict[int, int] = {}
    for candidate in rows:
        row = candidate
        while row:
            pivot = row.bit_length() - 1
            prior = basis.get(pivot)
            if prior is None:
                basis[pivot] = row
                break
            row ^= prior
    return basis


def _gf2_rank(rows: Iterable[int]) -> int:
    return len(_gf2_basis(rows))


def _in_span(row: int, basis: dict[int, int]) -> bool:
    while row:
        pivot = row.bit_length() - 1
        prior = basis.get(pivot)
        if prior is None:
            return False
        row ^= prior
    return True


def _symplectic_inner(left: int, right: int, n: int) -> int:
    mask = (1 << n) - 1
    left_x, left_z = left & mask, left >> n
    right_x, right_z = right & mask, right >> n
    return ((left_x & right_z).bit_count() + (left_z & right_x).bit_count()) & 1


def _rows_commute(rows: Sequence[int], n: int) -> bool:
    return all(
        _symplectic_inner(row, other, n) == 0
        for index, row in enumerate(rows)
        for other in rows[index + 1 :]
    )


def horizontal_logical(ell: int, m: int) -> int:
    """Return ``(0,Q | 0,Q)``, where ``wt(Q)=2*ell/3`` and ``A Q=0``."""

    if ell < 3 or ell % 3:
        raise ValueError("ell must be a positive multiple of three with ell >= 3")
    if m < 2:
        raise ValueError("m must be at least two for the weight-five family")

    block_size = ell * m
    q_terms = tuple(
        term
        for j in range(ell // 3)
        for term in ((3 * j, 0), (3 * j + 1, 0))
    )
    q = _translated_polynomial(q_terms, (0, 0), ell, m)
    return _symplectic_row(0, q, 0, q, block_size)


def vertical_logical(ell: int, m: int) -> int:
    """Return the pure-Z logical ``(0,0 | 0,h)``, ``h=sum_j y^j``."""

    if ell < 3 or ell % 3:
        raise ValueError("ell must be a positive multiple of three with ell >= 3")
    if m < 2:
        raise ValueError("m must be at least two for the weight-five family")

    block_size = ell * m
    h_terms = tuple((0, j) for j in range(m))
    h = _translated_polynomial(h_terms, (0, 0), ell, m)
    return _symplectic_row(0, 0, 0, h, block_size)


def pauli_weight(row: int, n: int) -> int:
    mask = (1 << n) - 1
    return ((row & mask) | (row >> n)).bit_count()


def exact_distance_through(ell: int, m: int, max_weight: int) -> int | None:
    """Exhaust all Paulis through ``max_weight`` and return the first logical.

    This is intended only for small counterexamples.  Its cost is exponential
    in ``max_weight``; the structural checks in :func:`verify_instance` are the
    scalable part of this module.
    """

    if max_weight < 1:
        return None
    top, bottom = build_stabilizers(ell, m)
    stabilizers = top + bottom
    stabilizer_basis = _gf2_basis(stabilizers)
    n = 2 * ell * m
    for weight in range(1, max_weight + 1):
        for support in itertools.combinations(range(n), weight):
            for labels in itertools.product((1, 2, 3), repeat=weight):
                x_part = 0
                z_part = 0
                for position, label in zip(support, labels):
                    if label & 1:
                        x_part |= 1 << position
                    if label & 2:
                        z_part |= 1 << position
                candidate = x_part | (z_part << n)
                if all(
                    _symplectic_inner(candidate, row, n) == 0
                    for row in stabilizers
                ) and not _in_span(candidate, stabilizer_basis):
                    return weight
    return None


def verify_instance(ell: int, m: int) -> VerificationResult:
    top, bottom = build_stabilizers(ell, m)
    stabilizers = top + bottom
    block_size = ell * m
    n = 2 * block_size

    top_rank = _gf2_rank(top)
    bottom_rank = _gf2_rank(bottom)
    stabilizer_basis = _gf2_basis(stabilizers)
    stabilizer_rank = len(stabilizer_basis)
    intersection_dimension = top_rank + bottom_rank - stabilizer_rank

    horizontal = horizontal_logical(ell, m)
    vertical = vertical_logical(ell, m)
    expected_k = 2 * math.gcd(m, 2)
    expected_horizontal_weight = 2 * ell // 3
    expected_vertical_weight = m
    result = VerificationResult(
        ell=ell,
        m=m,
        n=n,
        top_rank=top_rank,
        bottom_rank=bottom_rank,
        intersection_dimension=intersection_dimension,
        stabilizer_rank=stabilizer_rank,
        k=n - stabilizer_rank,
        expected_k=expected_k,
        stabilizers_commute=_rows_commute(stabilizers, n),
        horizontal_logical_commutes=all(
            _symplectic_inner(horizontal, row, n) == 0 for row in stabilizers
        ),
        horizontal_logical_is_nontrivial=not _in_span(
            horizontal, stabilizer_basis
        ),
        horizontal_logical_weight=pauli_weight(horizontal, n),
        expected_horizontal_weight=expected_horizontal_weight,
        vertical_logical_commutes=all(
            _symplectic_inner(vertical, row, n) == 0 for row in stabilizers
        ),
        vertical_logical_is_nontrivial=not _in_span(vertical, stabilizer_basis),
        vertical_logical_weight=pauli_weight(vertical, n),
        expected_vertical_weight=expected_vertical_weight,
        distance_upper_bound=min(expected_horizontal_weight, expected_vertical_weight),
    )
    return result


def assert_verified(result: VerificationResult) -> None:
    expected_intersection = 2 if result.m % 2 == 0 else 0
    block_size = result.ell * result.m
    expected = {
        "top rank": (result.top_rank, block_size),
        "bottom rank": (result.bottom_rank, block_size - 2),
        "intersection dimension": (
            result.intersection_dimension,
            expected_intersection,
        ),
        "dimension": (result.k, result.expected_k),
        "horizontal logical weight": (
            result.horizontal_logical_weight,
            result.expected_horizontal_weight,
        ),
        "vertical logical weight": (
            result.vertical_logical_weight,
            result.expected_vertical_weight,
        ),
    }
    failures = [
        f"{name}: got {actual}, expected {wanted}"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    for name, passed in (
        ("stabilizer commutativity", result.stabilizers_commute),
        ("horizontal logical commutativity", result.horizontal_logical_commutes),
        (
            "horizontal logical nontriviality",
            result.horizontal_logical_is_nontrivial,
        ),
        ("vertical logical commutativity", result.vertical_logical_commutes),
        ("vertical logical nontriviality", result.vertical_logical_is_nontrivial),
    ):
        if not passed:
            failures.append(f"{name}: failed")
    if failures:
        raise AssertionError(
            f"baseline verification failed for (ell,m)=({result.ell},{result.m}): "
            + "; ".join(failures)
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ell",
        nargs="+",
        type=int,
        default=[3, 6, 9, 12, 15, 18],
        help="multiples of three to check (default: 3 6 9 12 15 18)",
    )
    parser.add_argument(
        "--m",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5, 8, 9, 12, 15],
        help="second lattice sizes to check (default: 2 3 4 5 8 9 12 15)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON lines")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for ell in args.ell:
        for m in args.m:
            result = verify_instance(ell, m)
            assert_verified(result)
            if args.json:
                print(json.dumps(asdict(result), sort_keys=True))
            else:
                print(
                    f"(ell,m)=({ell},{m}): [[{result.n},{result.k},d]], "
                    f"certified d <= {result.distance_upper_bound} "
                    f"(horizontal {result.horizontal_logical_weight}, "
                    f"vertical {result.vertical_logical_weight})"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
