"""Seed solution for the weight-5 CSS campaign (Phase B1).

This seed restricts the BB-code search to CSS candidates whose stabilizer-
generator Pauli row weight is exactly 5, split canonically as ``2+3``
between the two defining polynomials A and B (``len(A)+len(B)==5`` with
``{len(A), len(B)} == {2, 3}``; see ``evaluation/weight_enforcement.py``,
``css_weight_ok``).  The evaluator (``openevolve_evaluator_weight5_css.py``)
re-checks this condition on every candidate before construction -- this
seed's job is only to *generate* weight-5 candidates efficiently, not to be
the sole enforcement point.

Three generation strategies (Phase B1):

1. **Mixed-monomial bounded sweep** (principal, evolvable family) --
   ``A = [(0,0), (a1,a2)]``, ``B = [(0,0), (b1,0), (b2,b3)]`` with every
   exponent bounded by ``min(ell-1, 6)`` (x-axis) / ``min(m-1, 6)`` (y-axis).
   This is the family the LLM is expected to refine.
2. **Univariate/HGP calibration baseline** -- ``A`` univariate in one axis
   (2 terms), ``B`` univariate in the other axis (3 terms), and the
   symmetric assignment.  A small, deliberately modest labeled set: a
   comparison baseline against the mixed-monomial family, not the
   evolution target.
3. **Coordinate perturbations** -- +/-1 (mod ell or mod m, staying in
   range) steps on Strategy 1's grid, seeded from the 32 corner points of
   the bounded box (interior points already have both neighbors covered
   by Strategy 1's own exhaustive sweep; only a boundary value's +/-1 step
   can land *outside* the box, e.g. past the ``min(*, 6)`` cap or wrapping
   to ``ell-1``/``m-1`` from 0).  Explores the neighborhood just past the
   bounded sweep without recreating its combinatorics.

``_safety_net_candidates(ell, m)`` -- defined OUTSIDE the EVOLVE-BLOCK, so
an LLM mutation cannot remove it.  Returns mixed-monomial weight-5 (2+3)
candidates verified (see module-level test invocations during Phase B
development) to give k >= 4 AND be translation-connected (see
``evaluation.connectivity``) at every Stage-1 lattice in both the "small"
and "large" lattice profiles:  (6,6), (6,9), (9,8), (12,9), (12,12),
(15,12) -- the univariate candidates used here previously were all
translation-disconnected direct sums at every one of those lattices, which
the connectivity gate now added to the evaluator (Reviewer-M1) would strip
from every candidate list, silently emptying the safety net.  This
guarantees Stage 1 of the cascade always has something to score even if
every evolved strategy above regresses to
k=0.

``generate_candidates(ell, m)`` -- the evolved function, returning
``list[tuple[list[tuple[int,int]], list[tuple[int,int]]]]`` i.e. a list of
``(A_terms, B_terms)`` pairs.  Raw counts are intentionally unbounded here
(3,125 at (6,6), 4,500 at (6,9), 7,776 at (9,8) for Strategy 1 alone) --
capping to a manageable per-lattice budget is the evaluator's job (Phase
A5, ``evaluation.candidate_selection``), not the seed's.
"""

from __future__ import annotations

import itertools

# ---------------------------------------------------------------------------
# Safety net -- OUTSIDE EVOLVE-BLOCK, non-removable by evolution.
# ---------------------------------------------------------------------------
# Mixed-monomial weight-5 (2+3) candidates.  Verified via
# evaluation.connectivity.bicycle_translation_is_connected to be
# translation-CONNECTED, and to give k == 4, at every Stage-1 lattice in
# both lattice profiles (small: (6,6),(6,9),(9,8); large:
# (12,9),(12,12),(15,12)) -- comfortably above the default
# QCODE_MIN_K_THRESHOLD (4).  The previous univariate safety net
# ([(0,0),(0,2)]/[(0,0),(0,4)] against [(0,0),(2,0),(4,0)]) gave the same
# k but was translation-DISCONNECTED (a replicated direct sum) at every one
# of those six lattices, so the connectivity gate wired into the evaluator
# (Reviewer-M1) rejected both entries unconditionally and left Stage 1 with
# nothing to score under the default "reject" policy. Small fixed
# exponents (<=2) keep both candidates in range for every lattice actually
# used by this campaign (all have ell>=6, m>=6); the range guard below
# drops a candidate defensively if ever called with a smaller lattice.
_SAFETY_NET_RAW = [
    ([(0, 0), (0, 1)], [(0, 0), (1, 0), (2, 0)]),
    ([(0, 0), (0, 1)], [(0, 0), (1, 1), (2, 2)]),
]


def _safety_net_candidates(ell, m):
    """Return hardcoded weight-5 fallback candidates, filtered to fit
    (ell, m).  OUTSIDE EVOLVE-BLOCK -- the LLM cannot remove or alter this.

    Without this, if every evolved strategy collapses to k=0 (e.g. an LLM
    mutation that breaks Strategy 1's exponent bounds), Stage 1's cascade
    threshold is never met, Stage 2 never runs, and the LLM never receives
    distance feedback to recover from.
    """
    out = []
    for A, B in _SAFETY_NET_RAW:
        if all(0 <= x < ell and 0 <= y < m for x, y in A) and all(
            0 <= x < ell and 0 <= y < m for x, y in B
        ):
            out.append((list(A), list(B)))
    return out


# EVOLVE-BLOCK-START
def _generate_candidates_evolved(
    ell: int, m: int
) -> list[tuple[list[tuple[int, int]], list[tuple[int, int]]]]:
    """Generate weight-5 CSS candidate (A_terms, B_terms) pairs.

    Every candidate returned here is intended to satisfy the campaign's
    weight-5 "2+3" backbone (``len(A)+len(B)==5``, ``{len(A),len(B)}=={2,3}``)
    by construction.  The evaluator independently re-checks this via
    ``evaluation.weight_enforcement.css_weight_ok`` and silently rejects
    anything that doesn't -- so a mutation that drifts away from weight 5
    will fail evaluation rather than corrupt results, but will also stop
    contributing to the score.  Stay within the 2+3 backbone.

    This is the EVOLVABLE half of candidate generation only -- it does
    NOT include the safety net (that's prepended by the non-evolvable
    ``generate_candidates`` wrapper below, outside this block).

    Args:
        ell: Cyclic group order for x.
        m: Cyclic group order for y.

    Returns:
        List of (A_terms, B_terms) pairs to evaluate.
    """
    candidates = []
    seen = set()

    def _add(A_terms, B_terms):
        """Add a candidate if both polynomials have distinct monomials
        and the pair hasn't been emitted yet."""
        if len(set(A_terms)) != len(A_terms):
            return
        if len(set(B_terms)) != len(B_terms):
            return
        key = (tuple(sorted(A_terms)), tuple(sorted(B_terms)))
        if key in seen:
            return
        seen.add(key)
        candidates.append((list(A_terms), list(B_terms)))

    # -----------------------------------------------------------------
    # Strategy 1 (principal, evolvable): mixed-monomial bounded sweep
    # -----------------------------------------------------------------
    # A = 1 + x^a1*y^a2                       (2 terms)
    # B = 1 + x^b1 + x^b2*y^b3                 (3 terms)
    #
    # a1, b1, b2 in [1, min(ell-1, 6)]; a2, b3 in [1, min(m-1, 6)].
    # Every term in A and every term in B is automatically distinct from
    # the others because a1,a2,b1,b2,b3 >= 1 (so no term collides with the
    # constant (0,0)) and b3 >= 1 (so (b1,0) != (b2,b3)) -- no dedup check
    # needed inside the loop itself, though _add still guards defensively.
    Nx = min(ell - 1, 6)
    Ny = min(m - 1, 6)

    if Nx >= 1 and Ny >= 1:
        for a1 in range(1, Nx + 1):
            for a2 in range(1, Ny + 1):
                A = [(0, 0), (a1, a2)]
                for b1 in range(1, Nx + 1):
                    for b2 in range(1, Nx + 1):
                        for b3 in range(1, Ny + 1):
                            B = [(0, 0), (b1, 0), (b2, b3)]
                            _add(A, B)

    # -----------------------------------------------------------------
    # Strategy 2: labeled univariate/HGP calibration baseline
    # -----------------------------------------------------------------
    # NOT the target of evolution -- this is a small, deliberately modest
    # comparison set so the evaluator's feedback can show the LLM how the
    # mixed-monomial family (Strategy 1) performs relative to the
    # univariate/HGP family (highest known k, but d<=4 per the campaign
    # docs).  A = 1 + x^a (2 terms, univariate-x), B = 1 + y^b1 + y^b2
    # (3 terms, univariate-y) -- and the symmetric assignment.
    Cx = min(ell - 1, 4)
    Cy = min(m - 1, 4)

    if Cx >= 1 and Cy >= 2:
        for a in range(1, Cx + 1):
            A = [(0, 0), (a, 0)]
            for b1 in range(1, Cy + 1):
                for b2 in range(b1 + 1, Cy + 1):
                    B = [(0, 0), (0, b1), (0, b2)]
                    _add(A, B)

    if Cy >= 1 and Cx >= 2:
        for a in range(1, Cy + 1):
            A = [(0, 0), (0, a)]
            for b1 in range(1, Cx + 1):
                for b2 in range(b1 + 1, Cx + 1):
                    B = [(0, 0), (b1, 0), (b2, 0)]
                    _add(A, B)

    # -----------------------------------------------------------------
    # Strategy 3: coordinate perturbations of mixed-monomial survivors
    # -----------------------------------------------------------------
    # Strategy 1 bounds every exponent at min(ell-1,6)/min(m-1,6); a +/-1
    # step (mod ell or mod m, staying in range) from an INTERIOR grid
    # point just lands on another interior point Strategy 1 already
    # generated. Only a BOUNDARY value's step can reach outside the box
    # (past the cap, or wrapping 0 -> ell-1/m-1), so perturbations are
    # seeded from the box's 2^5 = 32 corners (each parameter in
    # {1, Nx}/{1, Ny}) rather than every one of Strategy 1's raw
    # candidates -- this keeps Strategy 3 O(1) per lattice (<=320 raw
    # candidates) instead of 10x-ing Strategy 1's combinatorics.
    if Nx >= 1 and Ny >= 1:
        corner_x = sorted({1, Nx})
        corner_y = sorted({1, Ny})
        # Position order: (a1, a2, b1, b2, b3); positions 0,2,3 are
        # x-exponents (mod ell), positions 1,4 are y-exponents (mod m).
        x_positions = {0, 2, 3}
        for corner in itertools.product(corner_x, corner_y, corner_x, corner_x, corner_y):
            for pos in range(5):
                modulus = ell if pos in x_positions else m
                for delta in (-1, 1):
                    pert = list(corner)
                    pert[pos] = (pert[pos] + delta) % modulus
                    pa1, pa2, pb1, pb2, pb3 = pert
                    A = [(0, 0), (pa1, pa2)]
                    B = [(0, 0), (pb1, 0), (pb2, pb3)]
                    _add(A, B)

    return candidates
# EVOLVE-BLOCK-END


def generate_candidates(
    ell: int, m: int
) -> list[tuple[list[tuple[int, int]], list[tuple[int, int]]]]:
    """Entry point actually loaded by the evaluator (see
    ``_load_generate_candidates`` in ``openevolve_evaluator_weight5_css.py``,
    which looks up this exact name).

    OUTSIDE EVOLVE-BLOCK -- unlike ``_generate_candidates_evolved`` above,
    an LLM mutation of this program cannot remove or bypass the call to
    ``_safety_net_candidates`` here, only rewrite the evolvable helper's
    internals. Always prepends the safety net, then extends with whatever
    the evolvable strategies produce, deduplicating against the combined
    set.
    """
    candidates = list(_safety_net_candidates(ell, m))
    seen = {
        (tuple(sorted(a)), tuple(sorted(b))) for a, b in candidates
    }

    for A_terms, B_terms in _generate_candidates_evolved(ell, m):
        if len(set(A_terms)) != len(A_terms):
            continue
        if len(set(B_terms)) != len(B_terms):
            continue
        key = (tuple(sorted(A_terms)), tuple(sorted(B_terms)))
        if key in seen:
            continue
        seen.add(key)
        candidates.append((list(A_terms), list(B_terms)))

    return candidates
