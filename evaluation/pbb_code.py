"""Perturbed bivariate bicycle (PBB) code construction.

Extends bivariate bicycle (BB) codes to the non-CSS regime by adding
circulant perturbation polynomials (C, D) to the z-part of the first
stabilizer block, matching the stabilizer matrix in the paper.

A CSS BB code has stabilizer matrix (symplectic form)::

    Block 1 (X-stabs):  x-part = [A | B],     z-part = [0 | 0]
    Block 2 (Z-stabs):  x-part = [0 | 0],     z-part = [B^T | A^T]

A PBB code replaces block 1 with mixed stabilizers::

    H = ( A  B  |  C  D )
        ( 0  0  | B^T A^T)

i.e. block-1 z-part is ``[C | D]`` (C left, D right), matching the
paper convention.  When C = D = 0 this is exactly the CSS BB code.

Commutativity constraints:

- Cross-block (block-1 with block-2): automatic for abelian group-ring
  circulants.
- Within-block-1: ``(A @ C.T + B @ D.T) % 2`` must be symmetric over
  F_2; i.e. ``A C^T + B D^T`` is symmetric.

Usage::

    code = build_pbb_code(6, 3,
                          [(3, 0), (0, 1), (0, 2)],   # A
                          [(0, 1), (1, 0), (2, 0)],    # B
                          [(1, 1), (2, 2)],             # C (perturbation, left)
                          [(0, 0), (3, 1)])             # D (perturbation, right)
"""

from __future__ import annotations

import numpy as np
import sympy
from sympy.abc import x, y
from qldpc import codes
# qldpc's `QuditCode` is the generic Galois-stabilizer class; all codes
# constructed here live over GF(2), so they are qubit stabilizer codes.
from qldpc.codes import QuditCode


def _terms_to_sympy_poly(terms: list[tuple[int, int]]) -> sympy.Poly:
    """Convert exponent pairs to a sympy Poly with explicit generators.

    CRITICAL: Must use sympy.Poly(expr, x, y), NOT bare expression.
    qldpc's QCCode.eval() treats bare expressions as single monomials,
    silently producing wrong circulant matrices for multi-term polynomials.
    """
    expr = sum(x ** xe * y ** ye for xe, ye in terms)
    return sympy.Poly(expr, x, y)


def _poly_to_matrix(bb_code: codes.BBCode, terms: list[tuple[int, int]]) -> np.ndarray:
    """Convert polynomial terms to a binary circulant matrix using qldpc's eval.

    Uses the BBCode's eval() to ensure correct group element ordering.
    """
    poly = _terms_to_sympy_poly(terms)
    mat = np.array(bb_code.eval(poly).lift().T, dtype=int) % 2
    return mat


def check_commutativity(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray
) -> bool:
    """Check within-block-1 commutativity: ``A C^T + B D^T`` symmetric over F_2."""
    M = (A @ C.T + B @ D.T) % 2
    return np.array_equal(M, M.T)


def validate_pbb_terms(
    ell: int, m: int, terms: list[tuple[int, int]], name: str = "polynomial"
) -> None:
    """Validate polynomial terms for PBB construction.

    Unlike BB codes which require exactly 3 terms (trinomials), PBB
    perturbation polynomials can have any number of terms >= 1.
    """
    if len(terms) == 0:
        raise ValueError(f"{name} must have at least 1 term")

    reduced = set()
    for x_exp, y_exp in terms:
        if not (0 <= x_exp < ell):
            raise ValueError(f"{name}: x-exponent {x_exp} out of range [0, {ell})")
        if not (0 <= y_exp < m):
            raise ValueError(f"{name}: y-exponent {y_exp} out of range [0, {m})")
        monomial = (x_exp % ell, y_exp % m)
        if monomial in reduced:
            raise ValueError(f"{name}: duplicate monomial x^{x_exp}*y^{y_exp}")
        reduced.add(monomial)


def build_pbb_code(
    ell: int,
    m: int,
    A_terms: list[tuple[int, int]],
    B_terms: list[tuple[int, int]],
    C_terms: list[tuple[int, int]] | None = None,
    D_terms: list[tuple[int, int]] | None = None,
) -> codes.BBCode | QuditCode:
    """Construct a perturbed bivariate bicycle (PBB) code.

    If C_terms and D_terms are both None or empty, returns a CSS BBCode.
    Otherwise, returns a non-CSS qubit stabilizer code (qldpc's ``QuditCode``)
    with block-1 z-part = [C | D] matching the paper's stabilizer matrix.

    Args:
        ell: Cyclic group order for x.
        m: Cyclic group order for y.
        A_terms: Polynomial exponent pairs for A (base code).
        B_terms: Polynomial exponent pairs for B (base code).
        C_terms: Perturbation polynomial C (z-part of block 1, left).
        D_terms: Perturbation polynomial D (z-part of block 1, right).

    Returns:
        qldpc ``BBCode`` if CSS, qldpc ``QuditCode`` if non-CSS
        (both representing qubit stabilizer codes over GF(2)).

    Raises:
        ValueError: If terms are invalid or commutativity is violated.
    """
    is_css = (not C_terms) and (not D_terms)

    from evaluation.bb_code import terms_to_poly
    poly_a = terms_to_poly(A_terms)
    poly_b = terms_to_poly(B_terms)
    bb = codes.BBCode({x: ell, y: m}, poly_a, poly_b)

    if is_css:
        return bb

    dim = ell * m

    # C and D are independently optional: an empty/None component becomes
    # an (ell*m, ell*m) zero matrix rather than being rejected outright.
    # Only a nonempty component is validated and converted via qLDPC's eval.
    # (is_css above already handles the C=D=empty case; this branch always
    # has at least one nonempty component.)
    if C_terms:
        validate_pbb_terms(ell, m, C_terms, "C polynomial")
        mat_C = _poly_to_matrix(bb, C_terms)
    else:
        mat_C = np.zeros((dim, dim), dtype=int)

    if D_terms:
        validate_pbb_terms(ell, m, D_terms, "D polynomial")
        mat_D = _poly_to_matrix(bb, D_terms)
    else:
        mat_D = np.zeros((dim, dim), dtype=int)

    mat_A = _poly_to_matrix(bb, A_terms)
    mat_B = _poly_to_matrix(bb, B_terms)

    if not check_commutativity(mat_A, mat_B, mat_C, mat_D):
        raise ValueError(
            "Commutativity violated: (A @ C^T + B @ D^T) % 2 is not symmetric"
        )

    zero = np.zeros((dim, dim), dtype=int)

    block1_x = np.hstack([mat_A, mat_B])        # (dim, n)
    block1_z = np.hstack([mat_C, mat_D])        # z-part = [C | D], matches paper
    block2_x = np.hstack([zero, zero])          # (dim, n)
    block2_z = np.hstack([mat_B.T % 2, mat_A.T % 2])  # (dim, n)

    top = np.hstack([block1_x, block1_z])       # (dim, 2n)
    bottom = np.hstack([block2_x, block2_z])    # (dim, 2n)
    symplectic = np.vstack([top, bottom]) % 2   # (2*dim, 2n)

    return QuditCode(symplectic)


def get_pbb_params_fast(code: codes.BBCode | QuditCode) -> tuple[int, int]:
    """Get (n, k) quickly -- no distance computation."""
    return code.num_qudits, code.dimension


def _gf2_rref(mat: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Row-reduce a binary matrix over GF(2). Returns (rref, pivots).

    Elimination is vectorized over rows (boolean-mask XOR) rather than a
    Python-level loop -- see ``_gf2_rank`` in clifford_equivalence.py for
    the same restructuring and why it matters at large matrix sizes.
    """
    M = mat.copy() % 2
    rows, cols = M.shape
    pivots = []
    r = 0
    for c in range(cols):
        if r == rows:
            break
        nz = np.flatnonzero(M[r:, c])
        if nz.size == 0:
            continue
        found = r + int(nz[0])
        if found != r:
            M[[r, found]] = M[[found, r]]
        pivots.append(c)
        mask = M[:, c].astype(bool)
        mask[r] = False
        if mask.any():
            M[mask] ^= M[r]
        r += 1
    return M, pivots


def _gf2_nullspace(mat: np.ndarray) -> np.ndarray:
    """Compute the null space of a binary matrix over GF(2).

    Returns a matrix whose rows form a basis for the null space.
    """
    M, pivots = _gf2_rref(mat)
    rows, cols = M.shape
    len(pivots)
    free_cols = [c for c in range(cols) if c not in pivots]

    if not free_cols:
        return np.zeros((0, cols), dtype=int)

    null_vecs = []
    for fc in free_cols:
        vec = np.zeros(cols, dtype=int)
        vec[fc] = 1
        for i, pc in enumerate(pivots):
            vec[pc] = M[i, fc]
        null_vecs.append(vec)
    return np.array(null_vecs, dtype=int) % 2


def _compute_logicals_gf2(code: QuditCode) -> np.ndarray:
    """Compute logical operators for a (possibly non-CSS) stabilizer code.

    The logicals are elements of the symplectic complement S^perp that
    are not in the row space of S. We find S^perp via the null space
    of the symplectic form applied to S, then extract a basis for the
    quotient S^perp / rowspace(S).

    Returns (2k, 2n) binary matrix of logical operators, or empty if k=0.
    """
    stab = np.array(code.matrix, dtype=int) % 2
    num_stabs, two_n = stab.shape
    n = two_n // 2
    k = code.dimension

    if k == 0:
        return np.zeros((0, two_n), dtype=int)

    omega_stab = np.hstack([stab[:, n:], stab[:, :n]])
    complement = _gf2_nullspace(omega_stab)

    if complement.shape[0] == 0:
        return np.zeros((0, two_n), dtype=int)

    stab_rref, stab_pivots = _gf2_rref(stab)
    stab_rank = len(stab_pivots)

    working = stab_rref[:stab_rank].copy()
    current_rank = stab_rank
    logicals = []

    for row in complement:
        aug = np.vstack([working, row.reshape(1, -1)]) % 2
        aug_rref, aug_pivots = _gf2_rref(aug)
        if len(aug_pivots) > current_rank:
            logicals.append(row)
            working = aug_rref[:len(aug_pivots)]
            current_rank = len(aug_pivots)
            if len(logicals) >= 2 * k:
                break

    if not logicals:
        return np.zeros((0, two_n), dtype=int)

    return np.array(logicals, dtype=int) % 2


def get_symplectic_logicals(code: QuditCode) -> np.ndarray:
    """Get logical operators as a (2k, 2n) symplectic matrix.

    For non-CSS codes, logicals mix X and Z. Each row is
    [x_1..x_n | z_1..z_n]. The symplectic weight of a row is
    the number of qubits i where x_i or z_i (or both) is nonzero.

    Uses our own GF(2) computation instead of qldpc's get_logical_ops(),
    which has bugs for some non-CSS codes (singular matrix errors).
    """
    return _compute_logicals_gf2(code)


def symplectic_weight(vec: np.ndarray) -> int:
    """Compute symplectic weight of a (2n,) vector [x|z].

    Weight = number of qubits i where x_i or z_i is nonzero.

    Thin wrapper around ``qldpc.math.symplectic_weight`` (added in
    0.2.x); we keep the wrapper here to give callers a stable Python
    ``int`` return type (qldpc returns a 0-d numpy array for single-
    vector input) and to keep the public import path inside
    ``evaluation.pbb_code``.
    """
    from qldpc.math import symplectic_weight as _qldpc_sw
    return int(_qldpc_sw(vec))


def symplectic_weight_bound_pbb(code: QuditCode) -> int:
    """Upper bound on distance from symplectic logical operator weights.

    Returns the minimum symplectic weight across all logical operators.
    """
    logicals = get_symplectic_logicals(code)
    if logicals.size == 0:
        return code.num_qudits
    return min(symplectic_weight(logicals[i]) for i in range(logicals.shape[0]))
