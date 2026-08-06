"""Weight-5 backbone and support-containment checks (Phase A2).

The evaluator, not only the seed generator, must enforce the campaign's weight
definition (`plans/direction2_weight5_campaigns.md`, Phase A2):

CSS candidates: ``len(A)+len(B)==5`` with a canonical ``2+3`` split (after
normalization -- i.e. ``{len(A), len(B)} == {2, 3}``, so the mirrored ``3+2``
assignment is not treated as a distinct case).

PBB candidates additionally require ``supp(C) subseteq supp(A)`` and
``supp(D) subseteq supp(B)`` as reduced exponent-pair sets -- checked both
symbolically (on the term lists) and on the constructed circulant matrices,
since a bare SymPy expression can silently diverge from the matrix qLDPC
actually builds (see the `_poly_to_matrix` gotcha in `evaluation/pbb_code.py`).
Given a weight-5 `A,B` backbone, this containment is exactly the condition
under which the PBB row weight `|supp(A) union supp(C)| + |supp(B) union
supp(D)|` stays at 5.
"""

from __future__ import annotations

import numpy as np


def _reduced(terms) -> set[tuple[int, int]]:
    return set(tuple(t) for t in (terms or []))


def css_row_weight(A_terms, B_terms) -> int:
    """CSS stabilizer-generator Pauli row weight: ``|supp(A)| + |supp(B)|``."""
    return len(_reduced(A_terms)) + len(_reduced(B_terms))


def css_weight_ok(
    A_terms,
    B_terms,
    target_weight: int = 5,
    canonical_split: tuple[int, int] | None = (2, 3),
) -> bool:
    """Check the CSS backbone hits ``target_weight`` with the canonical split.

    ``canonical_split=None`` checks only the total weight, ignoring how it is
    divided between A and B. The default ``(2, 3)`` accepts either
    assignment (``|A|=2,|B|=3`` or ``|A|=3,|B|=2``) since swapping A and B
    only exchanges qubit sectors.
    """
    a = _reduced(A_terms)
    b = _reduced(B_terms)
    if len(a) + len(b) != target_weight:
        return False
    if canonical_split is None:
        return True
    return sorted((len(a), len(b))) == sorted(canonical_split)


def pbb_support_contained(base_terms, perturb_terms) -> bool:
    """Check ``supp(perturb) subseteq supp(base)`` as reduced exponent-pair sets."""
    return _reduced(perturb_terms).issubset(_reduced(base_terms))


def pbb_row_weight(A_terms, B_terms, C_terms=None, D_terms=None) -> int:
    """PBB Block-1 row weight: ``|supp(A)∪supp(C)| + |supp(B)∪supp(D)|``."""
    a = _reduced(A_terms)
    b = _reduced(B_terms)
    c = _reduced(C_terms)
    d = _reduced(D_terms)
    return len(a | c) + len(b | d)


def pbb_weight_ok(
    A_terms,
    B_terms,
    C_terms=None,
    D_terms=None,
    target_weight: int = 5,
    canonical_split: tuple[int, int] | None = (2, 3),
) -> bool:
    """Check the full PBB weight-5 condition: CSS backbone + support containment."""
    if not css_weight_ok(A_terms, B_terms, target_weight=target_weight, canonical_split=canonical_split):
        return False
    if not pbb_support_contained(A_terms, C_terms):
        return False
    if not pbb_support_contained(B_terms, D_terms):
        return False
    return True


def matrix_support_contained(mat_base: np.ndarray, mat_perturb: np.ndarray) -> bool:
    """Matrix-level cross-check that ``supp(mat_perturb) subseteq supp(mat_base))``.

    Required because a bare SymPy expression (as opposed to an explicit
    ``sympy.Poly``) can make qLDPC's ``eval()`` silently produce a circulant
    matrix that diverges from the symbolic term list -- this checks the
    actual constructed matrices, independent of `pbb_support_contained`.
    """
    base_nz = (np.asarray(mat_base) % 2) != 0
    perturb_nz = (np.asarray(mat_perturb) % 2) != 0
    return bool(np.all(~perturb_nz | base_nz))
