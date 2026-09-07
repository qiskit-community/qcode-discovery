"""Bivariate bicycle code construction and parameter computation.

Thin wrapper around ``qldpc.codes.BBCode`` that converts exponent-based
representations to sympy polynomials and provides a unified interface
for the evaluation pipeline.

A bivariate bicycle (BB) code is defined by two trinomials over the ring
``F_2[x, y] / (x^ell - 1, y^m - 1)``.  Each trinomial is represented in
this module as a list of three ``(x_exp, y_exp)`` tuples -- for example,
``[(3, 0), (0, 1), (0, 2)]`` encodes the polynomial ``x^3 + y + y^2``.

The module exposes four public helpers used throughout the project:

* :func:`terms_to_poly` -- convert exponent tuples to a sympy expression.
* :func:`validate_terms` -- check that a term list is a well-formed trinomial
  for a given ``(ell, m)`` lattice.
* :func:`build_bb_code` -- construct a :class:`qldpc.codes.BBCode` object.
* :func:`get_code_params_fast` -- return ``(n, k)`` without computing distance.

Typical usage::

    code = build_bb_code(12, 6,
                         [(3, 0), (0, 1), (0, 2)],   # A = x^3 + y + y^2
                         [(0, 3), (1, 0), (2, 0)])    # B = y^3 + x + x^2
    n, k = get_code_params_fast(code)                 # (144, 12)
"""

from __future__ import annotations

import operator

import sympy
from sympy.abc import x, y
from qldpc import codes


def terms_to_poly(terms: list[tuple[int, int]]) -> sympy.Expr:
    """Convert exponent pairs to a sympy polynomial.

    Args:
        terms: List of (x_exp, y_exp) tuples, e.g. [(3,0), (1,0), (0,1)].

    Returns:
        Sympy expression like x**3 + x + y.
    """
    monomials = []
    for x_exp, y_exp in terms:
        monomials.append(x ** x_exp * y ** y_exp)
    return sum(monomials)


def validate_terms(
    ell: int, m: int, terms: list[tuple[int, int]], name: str = "polynomial",
    min_terms: int = 2, max_terms: int = 6,
) -> None:
    """Validate that terms define a proper polynomial for the given lattice.

    Raises ValueError if:
    - Term count outside [min_terms, max_terms]
    - An exponent is not an integer (e.g. a float from a buggy generator) --
      ``x_exp < ell``/``x_exp % ell`` below would otherwise accept it
      silently, and a non-integer exponent cannot be a valid cyclic-group
      element
    - Duplicate monomials (after reducing mod ell, m)
    - Exponents out of range
    """
    if not (min_terms <= len(terms) <= max_terms):
        raise ValueError(
            f"{name} must have {min_terms}-{max_terms} terms, got {len(terms)}"
        )

    reduced = set()
    for x_exp, y_exp in terms:
        try:
            x_exp = operator.index(x_exp)
            y_exp = operator.index(y_exp)
        except TypeError as exc:
            raise ValueError(f"{name} exponents must be integers") from exc
        if not (0 <= x_exp < ell):
            raise ValueError(
                f"{name}: x-exponent {x_exp} out of range [0, {ell})"
            )
        if not (0 <= y_exp < m):
            raise ValueError(
                f"{name}: y-exponent {y_exp} out of range [0, {m})"
            )
        monomial = (x_exp % ell, y_exp % m)
        if monomial in reduced:
            raise ValueError(
                f"{name}: duplicate monomial x^{x_exp}*y^{y_exp}"
            )
        reduced.add(monomial)


def build_bb_code(
    ell: int,
    m: int,
    A_terms: list[tuple[int, int]],
    B_terms: list[tuple[int, int]],
) -> codes.BBCode:
    """Construct a BBCode from lattice dimensions and exponent lists.

    Args:
        ell: Cyclic group order for x.
        m: Cyclic group order for y.
        A_terms: 3 monomials [(a1,b1), (a2,b2), (a3,b3)] for polynomial A.
        B_terms: 3 monomials [(b1,c1), (b2,c2), (b3,c3)] for polynomial B.

    Returns:
        A qldpc BBCode instance.
    """
    poly_a = terms_to_poly(A_terms)
    poly_b = terms_to_poly(B_terms)
    return codes.BBCode({x: ell, y: m}, poly_a, poly_b)


def get_code_params_fast(code: codes.BBCode) -> tuple[int, int]:
    """Get (n, k) quickly -- no distance computation.

    Returns:
        (n, k) where n = num_qubits, k = dimension.
    """
    return code.num_qudits, code.dimension
