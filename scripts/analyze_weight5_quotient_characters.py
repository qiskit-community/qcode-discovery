#!/usr/bin/env python3
"""Classify weight-five BB support pairs by small quotient characters.

For a homomorphism

    phi: Z_ell x Z_m -> Z_q,

the script evaluates the two BB polynomials at the character
``x^i y^j -> omega^phi(i,j)`` over GF(4), GF(8), and GF(16) for quotient
orders 3, 5, 7, and 15.  A simultaneous zero gives an explicit common-kernel
certificate for CSS codes and identifies the underlying ``(A,B)`` backbone
for PBB codes.  For coprime CSS lattices the script also computes and factors
the exact common univariate cyclic gcd, retaining factor multiplicity.  The
output additionally records lattice availability, the distinction between
the block-erasure quantity k_S and the common kernel, and a descriptive
distance-evidence control for the noncyclic support strata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOGUE = ROOT / "results" / "weight5_publication_catalogue.jsonl"
DEFAULT_OUTPUT = ROOT / "results" / "weight5_quotient_character_analysis.json"

QUOTIENT_ORDERS = (3, 5, 7, 15)

# Each entry gives an irreducible field modulus and an element of the stated
# multiplicative order.  For orders 5 and 15 we use the same GF(16), generated
# by a root t of t^4+t+1; t has order 15 and t^3 has order 5.
FIELD_SPEC = {
    3: (0b111, 0b10),   # t^2 + t + 1; t has order 3
    5: (0b10011, 0b1000),  # t^4 + t + 1; t^3 has order 5
    7: (0b1011, 0b10),  # t^3 + t + 1; t has order 7
    15: (0b10011, 0b10),  # t^4 + t + 1; t has order 15
}

# Irreducible factors whose roots have the small odd orders audited here.
# Polynomials are packed with coefficient of z^i in bit i.
CYCLOTOMIC_FACTORS = (
    (3, 0b111),       # z^2 + z + 1
    (5, 0b11111),     # z^4 + z^3 + z^2 + z + 1
    (7, 0b1011),      # z^3 + z + 1
    (7, 0b1101),      # z^3 + z^2 + 1
    (15, 0b10011),    # z^4 + z + 1
    (15, 0b11001),    # z^4 + z^3 + 1
)


def gf_multiply(left: int, right: int, modulus: int) -> int:
    degree = modulus.bit_length() - 1
    value = 0
    while right:
        if right & 1:
            value ^= left
        right >>= 1
        left <<= 1
        if left & (1 << degree):
            left ^= modulus
    return value


@cache
def primitive_powers(order: int) -> tuple[int, ...]:
    modulus, generator = FIELD_SPEC[order]
    powers = [1]
    for _ in range(1, order):
        powers.append(gf_multiply(powers[-1], generator, modulus))
    if len(set(powers)) != order:
        raise ValueError(f"chosen field element is not primitive of order {order}")
    if gf_multiply(powers[-1], generator, modulus) != 1:
        raise ValueError(f"chosen field element does not have order {order}")
    return tuple(powers)


def evaluate_support(
    terms: Iterable[Iterable[int]], u: int, v: int, order: int
) -> int:
    powers = primitive_powers(order)
    value = 0
    for i, j in terms:
        value ^= powers[(u * int(i) + v * int(j)) % order]
    return value


def quotient_certificates(row: dict[str, Any], order: int) -> list[tuple[int, int]]:
    """Return all surjective homomorphisms giving a common character zero."""
    ell = int(row["parameters"]["ell"])
    m = int(row["parameters"]["m"])
    generators = row["generators"]
    certificates = []
    for u in range(order):
        for v in range(order):
            if math.gcd(order, u, v) != 1:
                continue
            if ell * u % order or m * v % order:
                continue
            if evaluate_support(generators["A_terms"], u, v, order):
                continue
            if evaluate_support(generators["B_terms"], u, v, order):
                continue
            certificates.append((u, v))
    return certificates


def polynomial_remainder(dividend: int, divisor: int) -> int:
    while dividend and dividend.bit_length() >= divisor.bit_length():
        dividend ^= divisor << (dividend.bit_length() - divisor.bit_length())
    return dividend


def polynomial_gcd(left: int, right: int) -> int:
    while right:
        left, right = right, polynomial_remainder(left, right)
    return left


def polynomial_divmod(dividend: int, divisor: int) -> tuple[int, int]:
    """Divide packed GF(2) polynomials, returning quotient and remainder."""
    if not divisor:
        raise ZeroDivisionError("polynomial division by zero")
    quotient = 0
    while dividend and dividend.bit_length() >= divisor.bit_length():
        shift = dividend.bit_length() - divisor.bit_length()
        quotient ^= 1 << shift
        dividend ^= divisor << shift
    return quotient, dividend


def polynomial_derivative(polynomial: int) -> int:
    """Return the formal derivative of a packed GF(2) polynomial."""
    derivative = 0
    for exponent in range(1, polynomial.bit_length(), 2):
        if polynomial >> exponent & 1:
            derivative |= 1 << (exponent - 1)
    return derivative


def cyclic_polynomial(terms: Iterable[Iterable[int]], ell: int, m: int) -> int:
    """Map a coprime bivariate lattice to F_2[z]/(z^(ell*m)-1)."""
    if math.gcd(ell, m) != 1:
        raise ValueError("cyclic_polynomial requires a coprime lattice")
    size = ell * m
    x_exponent = m * pow(m, -1, ell) % size
    y_exponent = ell * pow(ell, -1, m) % size
    polynomial = 0
    for i, j in terms:
        exponent = (int(i) * x_exponent + int(j) * y_exponent) % size
        polynomial ^= 1 << exponent
    return polynomial


def polynomial_string(polynomial: int) -> str:
    if not polynomial:
        return "0"
    terms = []
    for exponent in range(polynomial.bit_length() - 1, -1, -1):
        if not polynomial >> exponent & 1:
            continue
        terms.append("1" if exponent == 0 else "z" if exponent == 1 else f"z^{exponent}")
    return "+".join(terms)


def cyclic_common_factor(row: dict[str, Any]) -> str | None:
    ell = int(row["parameters"]["ell"])
    m = int(row["parameters"]["m"])
    if math.gcd(ell, m) != 1:
        return None
    a = cyclic_polynomial(row["generators"]["A_terms"], ell, m)
    b = cyclic_polynomial(row["generators"]["B_terms"], ell, m)
    modulus = (1 << (ell * m)) | 1
    return polynomial_string(polynomial_gcd(polynomial_gcd(a, b), modulus))


def cyclic_common_factor_details(row: dict[str, Any]) -> dict[str, Any] | None:
    """Return the exact cyclic gcd together with audited factor multiplicities."""
    ell = int(row["parameters"]["ell"])
    m = int(row["parameters"]["m"])
    if math.gcd(ell, m) != 1:
        return None
    modulus = (1 << (ell * m)) | 1
    a = cyclic_polynomial(row["generators"]["A_terms"], ell, m)
    b = cyclic_polynomial(row["generators"]["B_terms"], ell, m)
    common = polynomial_gcd(polynomial_gcd(a, b), modulus)
    remaining = common
    factors = []
    for order, factor in CYCLOTOMIC_FACTORS:
        multiplicity = 0
        while remaining != 1:
            quotient, remainder = polynomial_divmod(remaining, factor)
            if remainder:
                break
            remaining = quotient
            multiplicity += 1
        if multiplicity:
            factors.append(
                {
                    "order": order,
                    "polynomial": polynomial_string(factor),
                    "degree": factor.bit_length() - 1,
                    "multiplicity": multiplicity,
                }
            )
    derivative = polynomial_derivative(common)
    squarefree = common == 1 or polynomial_gcd(common, derivative) == 1
    return {
        "polynomial": polynomial_string(common),
        "degree_with_multiplicity": common.bit_length() - 1,
        "factors": factors,
        "unclassified_factor": (
            None if remaining == 1 else polynomial_string(remaining)
        ),
        "maximum_factor_multiplicity": max(
            (factor["multiplicity"] for factor in factors), default=0
        ),
        "squarefree": squarefree,
    }


def add_group_elements(
    left: tuple[int, int], right: tuple[int, int], ell: int, m: int
) -> tuple[int, int]:
    """Add two elements of ``Z_ell x Z_m`` in coordinate form."""
    return ((left[0] + right[0]) % ell, (left[1] + right[1]) % m)


def element_order(element: tuple[int, int], ell: int, m: int) -> int:
    """Return the additive order of an element of ``Z_ell x Z_m``."""
    i, j = element
    return math.lcm(ell // math.gcd(ell, i), m // math.gcd(m, j))


def generated_subgroup(
    generators: Iterable[tuple[int, int]], ell: int, m: int
) -> frozenset[tuple[int, int]]:
    """Enumerate the subgroup generated by coordinate pairs."""
    normalized = [(i % ell, j % m) for i, j in generators]
    subgroup = {(0, 0)}
    pending = [(0, 0)]
    while pending:
        element = pending.pop()
        for generator in normalized:
            candidate = add_group_elements(element, generator, ell, m)
            if candidate not in subgroup:
                subgroup.add(candidate)
                pending.append(candidate)
    return frozenset(subgroup)


def normalized_support_subgroup(
    terms: Iterable[Iterable[int]], ell: int, m: int
) -> tuple[frozenset[tuple[int, int]], list[tuple[int, int]]]:
    """Return the subgroup generated by support differences.

    Translating a group-algebra polynomial by a monomial does not change its
    code up to a coordinate permutation.  Subtracting the first support point
    therefore gives the support group of a canonical identity-containing
    translate, even when the stored polynomial itself lacks an identity term.
    """
    support = [tuple(map(int, term)) for term in terms]
    if not support:
        raise ValueError("a CSS support must contain at least one term")
    base_i, base_j = support[0]
    differences = [
        ((i - base_i) % ell, (j - base_j) % m) for i, j in support[1:]
    ]
    return generated_subgroup(differences, ell, m), differences


def cyclic_generator(
    subgroup: frozenset[tuple[int, int]],
    ell: int,
    m: int,
    preferred: Iterable[tuple[int, int]] = (),
) -> tuple[int, int] | None:
    """Return a deterministic generator if ``subgroup`` is cyclic.

    A stored support difference is preferred when it already generates the
    subgroup, which keeps the reported directions aligned with the displayed
    polynomials.
    """
    target_order = len(subgroup)
    for element in preferred:
        if element_order(element, ell, m) == target_order:
            return element
    return next(
        (
            element
            for element in sorted(subgroup)
            if element_order(element, ell, m) == target_order
        ),
        None,
    )


def first_shared_power(
    generator_a: tuple[int, int] | None,
    generator_b: tuple[int, int] | None,
    ell: int,
    m: int,
) -> dict[str, Any] | None:
    """Give the first nonidentity equality between two cyclic directions."""
    if generator_a is None or generator_b is None:
        return None
    order_a = element_order(generator_a, ell, m)
    order_b = element_order(generator_b, ell, m)
    powers_b: dict[tuple[int, int], int] = {}
    element = (0, 0)
    for exponent_b in range(1, order_b):
        element = add_group_elements(element, generator_b, ell, m)
        powers_b.setdefault(element, exponent_b)
    element = (0, 0)
    for exponent_a in range(1, order_a):
        element = add_group_elements(element, generator_a, ell, m)
        if element in powers_b:
            return {
                "A_exponent": exponent_a,
                "B_exponent": powers_b[element],
                "element": list(element),
                "element_order": element_order(element, ell, m),
            }
    return None


def cyclic_extension_splits(group_order: int, subgroup_order: int) -> bool:
    """Whether ``1 -> C_subgroup -> C_group -> C_quotient -> 1`` splits.

    A cyclic group has a complement to its (unique) subgroup of order ``c``
    exactly when ``c`` and the quotient order are coprime.  This distinction
    matters here: the normal-intersection reduction in Lin--Pryadko's printed
    proof explicitly decomposes both support groups as semidirect products,
    which is not automatic for a nonsplit cyclic extension.
    """
    if group_order % subgroup_order:
        raise ValueError("subgroup order must divide cyclic group order")
    return math.gcd(subgroup_order, group_order // subgroup_order) == 1


def _gf2_rank_bitrows(rows: Iterable[int]) -> int:
    """Rank a binary matrix represented by Python-integer rows."""
    basis: dict[int, int] = {}
    for source in rows:
        row = source
        while row:
            pivot = row.bit_length() - 1
            if pivot in basis:
                row ^= basis[pivot]
                continue
            basis[pivot] = row
            break
    return len(basis)


def _abelian_regular_bitrows(
    terms: Iterable[Iterable[int]], ell: int, m: int
) -> list[int]:
    """Build a regular group-algebra matrix as packed binary rows."""
    size = ell * m
    rows = [0] * size
    normalized_terms = [(int(i) % ell, int(j) % m) for i, j in terms]
    for source_i in range(ell):
        for source_j in range(m):
            column = source_i * m + source_j
            for shift_i, shift_j in normalized_terms:
                target_i = (source_i + shift_i) % ell
                target_j = (source_j + shift_j) % m
                rows[target_i * m + target_j] ^= 1 << column
    return rows


def _product_support(
    left: Iterable[Iterable[int]],
    right: Iterable[Iterable[int]],
    ell: int,
    m: int,
) -> list[tuple[int, int]]:
    """Multiply two abelian group-algebra supports over GF(2)."""
    parity: set[tuple[int, int]] = set()
    for left_i, left_j in left:
        for right_i, right_j in right:
            term = (
                (int(left_i) + int(right_i)) % ell,
                (int(left_j) + int(right_j)) % m,
            )
            if term in parity:
                parity.remove(term)
            else:
                parity.add(term)
    return sorted(parity)


def two_block_rank_defects(row: dict[str, Any]) -> dict[str, Any]:
    """Compute the block-erasure dimension and rank defects for a CSS BB row."""
    ell = int(row["parameters"]["ell"])
    m = int(row["parameters"]["m"])
    size = ell * m
    a_terms = row["generators"]["A_terms"]
    b_terms = row["generators"]["B_terms"]
    a_rows = _abelian_regular_bitrows(a_terms, ell, m)
    b_rows = _abelian_regular_bitrows(b_terms, ell, m)
    ab_rows = _abelian_regular_bitrows(
        _product_support(a_terms, b_terms, ell, m), ell, m
    )
    inverse_a_rows = _abelian_regular_bitrows(
        [((-int(i)) % ell, (-int(j)) % m) for i, j in a_terms], ell, m
    )
    inverse_b_rows = _abelian_regular_bitrows(
        [((-int(i)) % ell, (-int(j)) % m) for i, j in b_terms], ell, m
    )
    rank_a = _gf2_rank_bitrows(a_rows)
    rank_b = _gf2_rank_bitrows(b_rows)
    rank_ab = _gf2_rank_bitrows(ab_rows)
    rank_x = _gf2_rank_bitrows(
        left | (right << size) for left, right in zip(a_rows, b_rows)
    )
    rank_z = _gf2_rank_bitrows(
        left | (right << size)
        for left, right in zip(inverse_b_rows, inverse_a_rows)
    )
    subsystem_k = size - rank_a - rank_b + rank_ab
    common_kernel_k = size - _gf2_rank_bitrows([*a_rows, *b_rows])
    delta_x = size - subsystem_k - rank_x
    delta_z = size - subsystem_k - rank_z
    common_kernel_excess = common_kernel_k - subsystem_k
    if common_kernel_excess < 0:
        raise AssertionError("k_S cannot exceed the common right-kernel dimension")
    return {
        "ranks": {
            "A": rank_a,
            "B": rank_b,
            "AB": rank_ab,
            "H_X": rank_x,
            "H_Z": rank_z,
        },
        "block_erasure_subsystem_k": subsystem_k,
        "block_erasure_is_common_kernel_lower_bound": True,
        "kernel_sum_equality": common_kernel_excess == 0,
        "common_kernel_excess_over_k_S": common_kernel_excess,
        "rank_defects": {"delta_X": delta_x, "delta_Z": delta_z},
        "logical_dimension": 2 * subsystem_k + delta_x + delta_z,
        "logical_dimension_from_common_kernel": 2 * common_kernel_k,
        "common_right_kernel_dimension": common_kernel_k,
    }


def support_subgroup_metrics(row: dict[str, Any]) -> dict[str, Any]:
    """Describe the two normalized CSS support directions."""
    ell = int(row["parameters"]["ell"])
    m = int(row["parameters"]["m"])
    generators = row["generators"]
    subgroup_a, differences_a = normalized_support_subgroup(
        generators["A_terms"], ell, m
    )
    subgroup_b, differences_b = normalized_support_subgroup(
        generators["B_terms"], ell, m
    )
    generator_a = cyclic_generator(subgroup_a, ell, m, differences_a)
    generator_b = cyclic_generator(subgroup_b, ell, m, differences_b)
    generated_together = generated_subgroup(
        [*differences_a, *differences_b], ell, m
    )
    intersection_order = len(subgroup_a & subgroup_b)
    invariant_factor_small = math.gcd(ell, m)
    extension_splitting = None
    if generator_a is not None and generator_b is not None:
        extension_splitting = {
            "A": cyclic_extension_splits(len(subgroup_a), intersection_order),
            "B": cyclic_extension_splits(len(subgroup_b), intersection_order),
        }
        extension_splitting["both"] = (
            extension_splitting["A"] and extension_splitting["B"]
        )
    return {
        "ambient_invariant_factors": [
            invariant_factor_small,
            math.lcm(ell, m),
        ],
        "ambient_order": ell * m,
        "A": {
            "order": len(subgroup_a),
            "is_cyclic": generator_a is not None,
            "cyclic_generator": list(generator_a) if generator_a else None,
            "normalized_support_differences": [list(term) for term in differences_a],
        },
        "B": {
            "order": len(subgroup_b),
            "is_cyclic": generator_b is not None,
            "cyclic_generator": list(generator_b) if generator_b else None,
            "normalized_support_differences": [list(term) for term in differences_b],
        },
        "intersection_order": intersection_order,
        "cyclic_support_extensions_split": extension_splitting,
        "generated_together_order": len(generated_together),
        "generates_ambient_group": len(generated_together) == ell * m,
        "first_shared_power": first_shared_power(
            generator_a, generator_b, ell, m
        ),
    }


def _histogram(values: Iterable[int]) -> dict[str, int]:
    return {
        str(value): count
        for value, count in sorted(Counter(values).items())
    }


def _distance_control_summary(
    class_members: Iterable[list[dict[str, Any]]],
) -> dict[str, Any]:
    """Summarize class-level distance evidence without treating censoring as data."""
    representatives = []
    for members in class_members:
        distances = {
            (
                bool(row["distance_class"]["is_exact"]),
                int(row["distance_class"]["lower"]),
                int(row["distance_class"]["upper"]),
            )
            for row in members
        }
        if len(distances) != 1:
            raise ValueError(
                "class-level distance is inconsistent within Tanner class "
                f"{members[0]['equivalence']['class_id']}"
            )
        representatives.append(members[0])

    exact = [row for row in representatives if row["distance_class"]["is_exact"]]
    return {
        "classes": len(representatives),
        "exact_classes": len(exact),
        "certified_interval_classes": len(representatives) - len(exact),
        "exact_distance_histogram": _histogram(
            int(row["distance_class"]["upper"]) for row in exact
        ),
        "certified_lower_bound_histogram": _histogram(
            int(row["distance_class"]["lower"]) for row in representatives
        ),
        "classes_with_certified_lower_bound_at_least_5": sum(
            int(row["distance_class"]["lower"]) >= 5 for row in representatives
        ),
        "exact_classes_with_distance_at_least_10": sum(
            int(row["distance_class"]["upper"]) >= 10 for row in exact
        ),
        "logical_dimension_histogram": _histogram(
            int(row["parameters"]["k"]) for row in representatives
        ),
        "lattice_class_histogram": dict(
            sorted(
                Counter(
                    f"{row['parameters']['ell']}x{row['parameters']['m']}"
                    for row in representatives
                ).items()
            )
        ),
    }


def cyclic_quotient_available(ell: int, m: int, order: int) -> bool:
    """Whether Z_ell x Z_m has a surjective homomorphism onto C_order."""
    return any(
        math.gcd(order, u, v) == 1
        and ell * u % order == 0
        and m * v % order == 0
        for u in range(order)
        for v in range(order)
    )


def summarize_lattice_strata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count connected CSS Tanner classes by lattice and quotient stratum."""
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["family"] == "CSS" and row["connectivity"]["is_connected"]:
            by_class[row["equivalence"]["class_id"]].append(row)

    by_lattice: dict[tuple[int, int], list[list[dict[str, Any]]]] = defaultdict(list)
    for members in by_class.values():
        lattices = {
            (int(row["parameters"]["ell"]), int(row["parameters"]["m"]))
            for row in members
        }
        if len(lattices) != 1:
            raise ValueError(
                "connected Tanner class spans multiple lattices: "
                f"{members[0]['equivalence']['class_id']}"
            )
        by_lattice[next(iter(lattices))].append(members)

    output = {}
    for (ell, m), classes in sorted(by_lattice.items()):
        order_counts = {}
        for order in QUOTIENT_ORDERS:
            order_counts[str(order)] = sum(
                any(quotient_certificates(row, order) for row in members)
                for members in classes
            )
        cyclic_ambient = math.gcd(ell, m) == 1
        two_direction = 0
        if not cyclic_ambient:
            two_direction = sum(
                support_subgroup_metrics(members[0])["A"]["is_cyclic"]
                and support_subgroup_metrics(members[0])["B"]["is_cyclic"]
                for members in classes
            )
        output[f"{ell}x{m}"] = {
            "ell": ell,
            "m": m,
            "ambient_order": ell * m,
            "ambient_exponent": math.lcm(ell, m),
            "cyclic_ambient": cyclic_ambient,
            "ambient_order_divisible_by": [
                order for order in QUOTIENT_ORDERS if ell * m % order == 0
            ],
            "cyclic_quotient_orders_available": [
                order
                for order in QUOTIENT_ORDERS
                if cyclic_quotient_available(ell, m, order)
            ],
            "connected_css_classes": len(classes),
            "exact_connected_css_classes": sum(
                bool(members[0]["distance_class"]["is_exact"])
                for members in classes
            ),
            "quotient_stratum_class_counts": order_counts,
            "two_direction_cyclic_support_classes": two_direction,
        }
    return output


def summarize_support_strata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify connected CSS Tanner classes by ambient/support cyclicity."""
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["family"] == "CSS" and row["connectivity"]["is_connected"]:
            by_class[row["equivalence"]["class_id"]].append(row)

    classes_with_cyclic_ambient_presentation = 0
    noncyclic_two_direction_classes = 0
    noncyclic_rank_two_support_classes = 0
    exact_noncyclic_two_direction_classes = 0
    noncyclic_two_direction_split_classes = 0
    noncyclic_two_direction_nonsplit_classes = 0
    noncyclic_two_direction_zero_defect_classes = 0
    noncyclic_two_direction_nonzero_defect_classes = 0
    noncyclic_two_direction_nonsplit_zero_defect_classes = 0
    noncyclic_two_direction_nonsplit_nonzero_defect_classes = 0
    two_direction_members: list[list[dict[str, Any]]] = []
    rank_two_members: list[list[dict[str, Any]]] = []
    common_kernel_comparisons: Counter[tuple[int, int]] = Counter()
    for members in by_class.values():
        has_cyclic_ambient = any(
            math.gcd(
                int(row["parameters"]["ell"]),
                int(row["parameters"]["m"]),
            )
            == 1
            for row in members
        )
        support_flags = {
            metrics["A"]["is_cyclic"] and metrics["B"]["is_cyclic"]
            for row in members
            for metrics in [support_subgroup_metrics(row)]
        }
        if len(support_flags) != 1:
            raise ValueError(
                "two-direction cyclic-support status is inconsistent within "
                f"Tanner class {members[0]['equivalence']['class_id']}"
            )
        splitting_flags = {
            metrics["cyclic_support_extensions_split"]["both"]
            for row in members
            for metrics in [support_subgroup_metrics(row)]
            if metrics["cyclic_support_extensions_split"] is not None
        }
        if len(splitting_flags) > 1:
            raise ValueError(
                "cyclic-support extension splitting is inconsistent within "
                f"Tanner class {members[0]['equivalence']['class_id']}"
            )
        if has_cyclic_ambient:
            classes_with_cyclic_ambient_presentation += 1
            continue
        if support_flags == {True}:
            two_direction_members.append(members)
            noncyclic_two_direction_classes += 1
            rank_structures = [two_block_rank_defects(row) for row in members]
            kernel_pairs = {
                (
                    item["block_erasure_subsystem_k"],
                    item["common_right_kernel_dimension"],
                )
                for item in rank_structures
            }
            if len(kernel_pairs) != 1:
                raise ValueError(
                    "common-kernel comparison is inconsistent within Tanner class "
                    f"{members[0]['equivalence']['class_id']}"
                )
            common_kernel_comparisons[next(iter(kernel_pairs))] += 1
            defect_flags = {
                any(item["rank_defects"].values()) for item in rank_structures
            }
            if len(defect_flags) != 1:
                raise ValueError(
                    "rank-defect status is inconsistent within Tanner class "
                    f"{members[0]['equivalence']['class_id']}"
                )
            has_rank_defect = defect_flags == {True}
            if has_rank_defect:
                noncyclic_two_direction_nonzero_defect_classes += 1
            else:
                noncyclic_two_direction_zero_defect_classes += 1
            if splitting_flags == {True}:
                noncyclic_two_direction_split_classes += 1
            else:
                noncyclic_two_direction_nonsplit_classes += 1
                if has_rank_defect:
                    noncyclic_two_direction_nonsplit_nonzero_defect_classes += 1
                else:
                    noncyclic_two_direction_nonsplit_zero_defect_classes += 1
            if members[0]["distance_class"]["is_exact"]:
                exact_noncyclic_two_direction_classes += 1
        else:
            rank_two_members.append(members)
            noncyclic_rank_two_support_classes += 1

    target_examples = {}
    for n, k, d in ((120, 4, 10), (432, 4, 10)):
        candidates = [
            row
            for row in rows
            if row["family"] == "CSS"
            and row["connectivity"]["is_connected"]
            and row["parameters"]["n"] == n
            and row["parameters"]["k"] == k
            and row["distance_class"]["is_exact"]
            and row["distance_class"]["upper"] == d
            and math.gcd(row["parameters"]["ell"], row["parameters"]["m"]) > 1
        ]
        if not candidates:
            raise ValueError(f"missing exact connected noncyclic [[{n},{k},{d}]]")
        row = min(candidates, key=lambda item: item["presentation_id"])
        target_examples[f"{n}_{k}_{d}"] = {
            "presentation_id": row["presentation_id"],
            "class_id": row["equivalence"]["class_id"],
            "parameters": row["parameters"],
            "generators": {
                "A_terms": row["generators"]["A_terms"],
                "B_terms": row["generators"]["B_terms"],
            },
            "support_subgroups": support_subgroup_metrics(row),
            "two_block_rank_structure": two_block_rank_defects(row),
        }

    return {
        "connected_css_classes": len(by_class),
        "classes_with_cyclic_ambient_presentation": (
            classes_with_cyclic_ambient_presentation
        ),
        "classes_with_only_noncyclic_ambient_presentations": (
            noncyclic_two_direction_classes + noncyclic_rank_two_support_classes
        ),
        "noncyclic_only_two_direction_cyclic_support_classes": (
            noncyclic_two_direction_classes
        ),
        "noncyclic_only_split_cyclic_support_classes": (
            noncyclic_two_direction_split_classes
        ),
        "noncyclic_only_nonsplit_cyclic_support_classes": (
            noncyclic_two_direction_nonsplit_classes
        ),
        "noncyclic_only_two_direction_zero_rank_defect_classes": (
            noncyclic_two_direction_zero_defect_classes
        ),
        "noncyclic_only_two_direction_nonzero_rank_defect_classes": (
            noncyclic_two_direction_nonzero_defect_classes
        ),
        "noncyclic_only_nonsplit_zero_rank_defect_classes": (
            noncyclic_two_direction_nonsplit_zero_defect_classes
        ),
        "noncyclic_only_nonsplit_nonzero_rank_defect_classes": (
            noncyclic_two_direction_nonsplit_nonzero_defect_classes
        ),
        "noncyclic_only_rank_two_support_classes": (
            noncyclic_rank_two_support_classes
        ),
        "exact_noncyclic_only_two_direction_cyclic_support_classes": (
            exact_noncyclic_two_direction_classes
        ),
        "common_kernel_vs_k_S_for_two_direction_classes": {
            "k_S_is_a_lower_bound": True,
            "classes_with_equality": sum(
                count
                for (subsystem_k, common_kernel_k), count
                in common_kernel_comparisons.items()
                if subsystem_k == common_kernel_k
            ),
            "classes_with_strict_inequality": sum(
                count
                for (subsystem_k, common_kernel_k), count
                in common_kernel_comparisons.items()
                if subsystem_k < common_kernel_k
            ),
            "class_histogram": {
                f"k_S={subsystem_k},common_kernel={common_kernel_k}": count
                for (subsystem_k, common_kernel_k), count
                in sorted(common_kernel_comparisons.items())
            },
        },
        "distance_control": {
            "two_direction_cyclic_support": _distance_control_summary(
                two_direction_members
            ),
            "other_noncyclic_support": _distance_control_summary(rank_two_members),
            "interpretation": (
                "Descriptive class counts only: exact certification was targeted and "
                "the interval endpoints are censored, so these counts do not estimate "
                "a distance distribution or establish predictive value."
            ),
        },
        "exact_examples": target_examples,
    }


def summarize(
    rows: list[dict[str, Any]], predicate: Any, family: str = "CSS"
) -> dict[str, Any]:
    selected = [row for row in rows if row["family"] == family and predicate(row)]
    matches = {
        order: {
            row["presentation_id"]: quotient_certificates(row, order)
            for row in selected
        }
        for order in QUOTIENT_ORDERS
    }
    for order in matches:
        matches[order] = {key: value for key, value in matches[order].items() if value}
    class_ids = {row["equivalence"]["class_id"] for row in selected}
    order_classes = {
        order: {
            row["equivalence"]["class_id"]
            for row in selected
            if row["presentation_id"] in matches[order]
        }
        for order in QUOTIENT_ORDERS
    }
    cyclic = [
        row
        for row in selected
        if math.gcd(int(row["parameters"]["ell"]), int(row["parameters"]["m"])) == 1
    ]
    return {
        "presentations": len(selected),
        "classes": len(class_ids),
        "cyclic_presentations": len(cyclic),
        "cyclic_classes": len({row["equivalence"]["class_id"] for row in cyclic}),
        "order_3_presentations": len(matches[3]),
        "order_3_classes": len(order_classes[3]),
        "order_5_presentations": len(matches[5]),
        "order_5_classes": len(order_classes[5]),
        "order_7_presentations": len(matches[7]),
        "order_7_classes": len(order_classes[7]),
        "order_15_presentations": len(matches[15]),
        "order_15_classes": len(order_classes[15]),
        "order_3_or_7_presentations": len(set(matches[3]) | set(matches[7])),
        "order_3_or_7_classes": len(order_classes[3] | order_classes[7]),
        "order_3_and_7_classes": len(order_classes[3] & order_classes[7]),
        "order_3_or_7_or_15_presentations": len(
            set(matches[3]) | set(matches[7]) | set(matches[15])
        ),
        "order_3_or_7_or_15_classes": len(
            order_classes[3] | order_classes[7] | order_classes[15]
        ),
        "order_3_or_5_or_7_or_15_presentations": len(
            set(matches[3]) | set(matches[5]) | set(matches[7]) | set(matches[15])
        ),
        "order_3_or_5_or_7_or_15_classes": len(
            order_classes[3]
            | order_classes[5]
            | order_classes[7]
            | order_classes[15]
        ),
    }


def summarize_cyclic_gcd_multiplicities(
    rows: list[dict[str, Any]], predicate: Any
) -> dict[str, Any]:
    """Audit exact gcd degrees and small cyclotomic-factor multiplicities."""
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["family"] == "CSS" and predicate(row):
            by_class[row["equivalence"]["class_id"]].append(row)

    details = []
    for class_id, members in sorted(by_class.items()):
        cyclic_members = [
            row
            for row in members
            if math.gcd(
                int(row["parameters"]["ell"]), int(row["parameters"]["m"])
            )
            == 1
        ]
        if not cyclic_members:
            continue
        row = min(cyclic_members, key=lambda item: item["presentation_id"])
        factor = cyclic_common_factor_details(row)
        if factor is None:  # pragma: no cover - guarded by cyclic_members
            raise AssertionError("cyclic row did not produce a cyclic gcd")
        details.append((class_id, row, factor))

    profile_counts: Counter[tuple[Any, ...]] = Counter()
    factor_occurrences: Counter[tuple[int, int]] = Counter()
    maximum_multiplicities = Counter()
    dimension_mismatches = []
    for class_id, row, factor in details:
        signature = (
            factor["polynomial"],
            factor["degree_with_multiplicity"],
            tuple(
                (
                    item["order"],
                    item["polynomial"],
                    item["degree"],
                    item["multiplicity"],
                )
                for item in factor["factors"]
            ),
            factor["unclassified_factor"],
            factor["squarefree"],
        )
        profile_counts[signature] += 1
        maximum_multiplicities[factor["maximum_factor_multiplicity"]] += 1
        for item in factor["factors"]:
            factor_occurrences[(item["order"], item["multiplicity"])] += 1
        if 2 * factor["degree_with_multiplicity"] != int(row["parameters"]["k"]):
            dimension_mismatches.append(class_id)

    profiles = []
    for signature, count in sorted(profile_counts.items(), key=lambda item: item[0][0]):
        polynomial, degree, factors, unclassified, squarefree = signature
        profiles.append(
            {
                "common_factor": polynomial,
                "degree_with_multiplicity": degree,
                "factors": [
                    {
                        "order": order,
                        "polynomial": factor,
                        "degree": factor_degree,
                        "multiplicity": multiplicity,
                    }
                    for order, factor, factor_degree, multiplicity in factors
                ],
                "unclassified_factor": unclassified,
                "squarefree": squarefree,
                "classes": count,
            }
        )

    return {
        "cyclic_classes": len(details),
        "factorization_profiles": profiles,
        "factor_occurrences_by_order_and_multiplicity": {
            f"order_{order}_multiplicity_{multiplicity}": count
            for (order, multiplicity), count in sorted(factor_occurrences.items())
        },
        "classes_by_maximum_factor_multiplicity": {
            str(multiplicity): count
            for multiplicity, count in sorted(maximum_multiplicities.items())
        },
        "classes_with_repeated_irreducible_factor": sum(
            factor["maximum_factor_multiplicity"] > 1
            for _, _, factor in details
        ),
        "classes_with_unclassified_factor": sum(
            factor["unclassified_factor"] is not None
            for _, _, factor in details
        ),
        "dimension_formula_mismatch_class_ids": dimension_mismatches,
        "multiplicity_semantics": (
            "degree_with_multiplicity is the degree of gcd(a,b,z^N-1); "
            "each listed irreducible-factor multiplicity is retained."
        ),
    }


def build_analysis(
    rows: list[dict[str, Any]], catalogue_path: Path = DEFAULT_CATALOGUE
) -> dict[str, Any]:
    connected_cyclic_gcd = summarize_cyclic_gcd_multiplicities(
        rows, lambda row: row["connectivity"]["is_connected"]
    )
    return {
        "schema_version": 1,
        "record_type": "weight5_quotient_character_analysis",
        "source_catalogue": {
            "path": str(catalogue_path.relative_to(ROOT))
            if catalogue_path.is_relative_to(ROOT)
            else str(catalogue_path),
            "sha256": hashlib.sha256(catalogue_path.read_bytes()).hexdigest(),
        },
        "all_css": summarize(rows, lambda row: True),
        "connected_css": summarize(
            rows, lambda row: row["connectivity"]["is_connected"]
        ),
        "connected_pbb_backbone": summarize(
            rows, lambda row: row["connectivity"]["is_connected"], family="PBB"
        ),
        "exact_connected_css": summarize(
            rows,
            lambda row: row["connectivity"]["is_connected"]
            and row["distance_class"]["is_exact"],
        ),
        # Retain the original factor-count field for downstream readers.
        "connected_cyclic_common_factors_by_class": {
            profile["common_factor"]: profile["classes"]
            for profile in connected_cyclic_gcd["factorization_profiles"]
        },
        "connected_cyclic_gcd_multiplicities": connected_cyclic_gcd,
        "connected_css_lattice_strata": summarize_lattice_strata(rows),
        "connected_css_support_strata": summarize_support_strata(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "catalogue",
        nargs="?",
        type=Path,
        default=DEFAULT_CATALOGUE,
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--write",
        action="store_true",
        help=f"write the deterministic analysis to {DEFAULT_OUTPUT}",
    )
    action.add_argument(
        "--check",
        action="store_true",
        help=f"verify that {DEFAULT_OUTPUT} is current",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.catalogue.read_text().splitlines() if line]
    rendered = json.dumps(
        build_analysis(rows, args.catalogue.resolve()), indent=2, sort_keys=True
    ) + "\n"
    if args.write:
        args.output.write_text(rendered, encoding="utf-8")
        print(f"wrote {args.output}")
    elif args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"stale generated artifact: {args.output}")
        print(f"verified {args.output}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
