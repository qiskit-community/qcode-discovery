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
3. **Cubic-character structural families** -- primitive mixed directions
   are related so that A and B share a nontrivial order-three Fourier mode.
   Exact progression/complementary-diagonal families are supplemented by
   transverse kernel shifts that preserve k>0 without remaining HGP-like.

``_safety_net_candidates(ell, m)`` -- defined OUTSIDE the EVOLVE-BLOCK, so
an LLM mutation cannot remove it.  Returns a univariate weight-5 (2+3)
candidate verified (see module-level test invocations during Phase B
development) to give k > 0 -- in fact k >= 4 -- at every Stage-1 lattice in
both the "small" and "large" lattice profiles:  (6,6), (6,9), (9,8),
(12,9), (12,12), (15,12).  This guarantees Stage 1 of the cascade always
has something to score even if every evolved strategy above regresses to
k=0.

``generate_candidates(ell, m)`` -- the evolved function, returning
``list[tuple[list[tuple[int,int]], list[tuple[int,int]]]]`` i.e. a list of
``(A_terms, B_terms)`` pairs.  Raw counts are intentionally unbounded here
(3,125 at (6,6), 4,500 at (6,9), 7,776 at (9,8) for Strategy 1 alone) --
capping to a manageable per-lattice budget is the evaluator's job (Phase
A5, ``evaluation.candidate_selection``), not the seed's.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Safety net -- OUTSIDE EVOLVE-BLOCK, non-removable by evolution.
# ---------------------------------------------------------------------------
# Univariate weight-5 (2+3) candidates.  Verified to give k > 0 at every
# Stage-1 lattice in both lattice profiles (small: (6,6),(6,9),(9,8); large:
# (12,9),(12,12),(15,12)) -- in fact k in {4, 8, 16, 32} at those six
# lattices, comfortably above the default QCODE_MIN_K_THRESHOLD (4).  Small
# fixed exponents (<=4) keep both candidates in range for every lattice
# actually used by this campaign (all have ell>=6, m>=6); the range guard
# below drops a candidate defensively if ever called with a smaller lattice.
_SAFETY_NET_RAW = [
    ([(0, 0), (0, 2)], [(0, 0), (2, 0), (4, 0)]),
    ([(0, 0), (0, 4)], [(0, 0), (2, 0), (4, 0)]),
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
    """Return deduplicated mixed BB pairs with exact weight-five 2+3 support.

    The principal families have a guaranteed common Fourier character;
    the wrapper separately prepends the fixed HGP safety candidates.
    """
    candidates = []
    seen = set()

    def _add(A_terms, B_terms):
        """Add one distinct, in-range weight-5 candidate."""
        if len(A_terms) + len(B_terms) != 5:
            return
        if {len(A_terms), len(B_terms)} != {2, 3}:
            return
        if len(set(A_terms)) != len(A_terms):
            return
        if len(set(B_terms)) != len(B_terms):
            return
        if not all(0 <= x < ell and 0 <= y < m
                   for x, y in A_terms + B_terms):
            return
        key = (tuple(sorted(A_terms)), tuple(sorted(B_terms)))
        if key in seen:
            return
        seen.add(key)
        candidates.append((list(A_terms), list(B_terms)))

    # Shared bounded representatives for the structured mixed family.
    # Do not emit the unconstrained five-parameter sweep: most of it has
    # k=0, and it dilutes common-root candidates under hash selection.
    Nx = min(ell - 1, 6)
    Ny = min(m - 1, 6)

    # The fixed HGP calibration set is emitted after the mixed families.

    # -----------------------------------------------------------------
    # Principal strategy: cyclotomic roots in G/<u>
    # -----------------------------------------------------------------
    # For A=1+X^u, any character trivial on <u> zeros A.  Choose v whose
    # order in G/<u> supports a sparse binary cyclotomic trinomial.  The
    # resulting B shares that character zero, so k>0 is structural rather
    # than accidental.  Adding multiples of u to B's exponents preserves
    # the zero while producing complementary and bent mixed diagonals.
    vectors = [
        (x, y)
        for x in range(Nx + 1)
        for y in range(Ny + 1)
        if (x, y) != (0, 0)
    ]

    def _cyclic_subgroup(u):
        subgroup = set()
        point = (0, 0)
        while point not in subgroup:
            subgroup.add(point)
            point = (
                (point[0] + u[0]) % ell,
                (point[1] + u[1]) % m,
            )
        return subgroup

    def _quotient_order(v, subgroup):
        point = (0, 0)
        for order in range(1, ell * m + 1):
            point = (
                (point[0] + v[0]) % ell,
                (point[1] + v[1]) % m,
            )
            if point in subgroup:
                return order
        return 0

    # (required order, p, q) represents the trinomial 1+z^p+z^q.
    # The cubic relation parametrizes the complementary phase pattern in
    # the latest d>=4 artifacts.  Orders 7, 9, and 15 add larger Fourier
    # orbits, which can increase k while retaining sparse weight-5 checks.
    root_templates = (
        (3, 1, 2),
        (7, 1, 3),
        (7, 2, 3),
        (9, 3, 6),
        (15, 1, 4),
        (15, 3, 4),
    )

    # Each pair independently shears the two nonconstant B terms by u.
    # These include exact progressions, opposite diagonal lifts, and more
    # strongly bent variants without resorting to isolated exponent nudges.
    lifts = (
        (0, 0),
        (0, 1), (0, -1),
        (1, 0), (-1, 0),
        (1, -1), (-1, 1),
        (1, 1), (-1, -1),
        (2, -1), (-2, 1),
        (1, -2), (-1, 2),
    )

    for u in vectors:
        subgroup = _cyclic_subgroup(u)
        binomial = [(0, 0), u]

        for v in vectors:
            # Rationally collinear directions only disguise a univariate
            # cyclic code as a mixed polynomial.  Keep transverse winding.
            if u[0] * v[1] == u[1] * v[0]:
                continue

            quotient_order = _quotient_order(v, subgroup)
            if quotient_order < 3:
                continue

            for required_order, p, q in root_templates:
                if quotient_order % required_order:
                    continue

                for first_lift, second_lift in lifts:
                    first = (
                        (p * v[0] + first_lift * u[0]) % ell,
                        (p * v[1] + first_lift * u[1]) % m,
                    )
                    second = (
                        (q * v[0] + second_lift * u[0]) % ell,
                        (q * v[1] + second_lift * u[1]) % m,
                    )

                    # Avoid B being formed by merely adding a term to A.
                    if first == u or second == u:
                        continue

                    trinomial = [(0, 0), first, second]
                    if not any(
                        x != 0 and y != 0
                        for x, y in binomial + trinomial
                    ):
                        continue

                    # Exercise both legal 2+3 placements.  Every call is
                    # still exactly weight five.
                    _add(binomial, trinomial)
                    _add(trinomial, binomial)

    # -----------------------------------------------------------------
    # Balanced order-three chords
    # -----------------------------------------------------------------
    # This parametrizes the d>=4 artifact relation
    #
    #   1 + X^(w+v) + X^(w-v),    1 + X^(3w),
    #
    # where v is nonzero 3-torsion modulo <w>.  A character trivial on w
    # and nontrivial on v zeros both polynomials.  Independent multiples
    # of u=3w shear the two chord endpoints while preserving that root.
    x_thirds = (0, ell // 3, 2 * ell // 3) if ell % 3 == 0 else (0,)
    y_thirds = (0, m // 3, 2 * m // 3) if m % 3 == 0 else (0,)
    torsion = [
        (x, y)
        for x in x_thirds
        for y in y_thirds
        if (x, y) != (0, 0)
    ]
    chord_lifts = (
        (0, 0),
        (1, 0), (0, 1),
        (1, -1), (-1, 1),
        (1, 1), (-1, -1),
    )

    for w in vectors:
        u = ((3 * w[0]) % ell, (3 * w[1]) % m)
        if u == (0, 0):
            continue

        w_subgroup = _cyclic_subgroup(w)
        for v in torsion:
            # Require a genuinely transverse order-three phase.  This
            # excludes augmented copies and one-dimensional chord supports.
            if v in w_subgroup:
                continue
            if w[0] * v[1] == w[1] * v[0]:
                continue

            for first_lift, second_lift in chord_lifts:
                first = (
                    (w[0] + v[0] + first_lift * u[0]) % ell,
                    (w[1] + v[1] + first_lift * u[1]) % m,
                )
                second = (
                    (w[0] - v[0] + second_lift * u[0]) % ell,
                    (w[1] - v[1] + second_lift * u[1]) % m,
                )
                trinomial = [(0, 0), first, second]
                binomial = [(0, 0), u]

                if first == u or second == u:
                    continue
                if not any(
                    x != 0 and y != 0
                    for x, y in trinomial + binomial
                ):
                    continue

                _add(trinomial, binomial)
                _add(binomial, trinomial)

    # Fixed univariate/HGP calibration yardstick; do not grow this set.
    Cx = min(ell - 1, 4)
    Cy = min(m - 1, 4)

    if Cx >= 1 and Cy >= 2:
        for a in range(1, Cx + 1):
            A = [(0, 0), (a, 0)]
            for b1 in range(1, Cy + 1):
                for b2 in range(b1 + 1, Cy + 1):
                    _add(A, [(0, 0), (0, b1), (0, b2)])

    if Cy >= 1 and Cx >= 2:
        for a in range(1, Cy + 1):
            A = [(0, 0), (0, a)]
            for b1 in range(1, Cx + 1):
                for b2 in range(b1 + 1, Cx + 1):
                    _add(A, [(0, 0), (b1, 0), (b2, 0)])

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
