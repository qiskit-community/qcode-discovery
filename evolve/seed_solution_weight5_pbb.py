"""Seed solution for the weight-5 non-CSS PBB campaign (Phase C1).

This seed restricts the search to perturbed bivariate bicycle (PBB) codes
built on the canonical weight-5 "2+3" CSS backbone -- the *same* backbone
ansatz as the sibling weight-5 CSS campaign's
``seed_solution_weight5_css.py`` (re-derived independently here, not
imported, per Phase C1 scoping)::

    A = [(0, 0), (a1, a2)]                     # 2 terms
    B = [(0, 0), (b1, 0), (b2, b3)]             # 3 terms
    a1, b1, b2 in [1, min(ell-1, 6)]; a2, b3 in [1, min(m-1, 6)]

Only the canonical ``2+3`` (A, B) assignment is used -- the mirrored
``3+2`` assignment ``(B, A, D, C)`` is deliberately NOT generated, since it
is just a qubit-sector permutation of the same code (Phase C1 scoping).

PBB perturbation and weight convention (paper convention, matching
``evaluation/pbb_code.py``)::

    Block 1 (mixed):   x-part = [A | B],   z-part = [C | D]   (C left, D right)
    Block 2 (pure Z):  x-part = [0 | 0],   z-part = [B^T | A^T]
    Commutativity: (A @ C^T + B @ D^T) % 2 must be symmetric.

Given the weight-5 (A, B) backbone, the PBB row weight
``|supp(A) union supp(C)| + |supp(B) union supp(D)|`` stays at 5 exactly
when ``supp(C) subseteq supp(A)`` and ``supp(D) subseteq supp(B)`` -- so
every candidate below is built by construction as a *subset* of the
backbone's own terms, which keeps weight-5 and containment automatic
without needing a runtime check inside the generator.

Any nontrivial (C, D) pair is a valid non-CSS PBB candidate here,
INCLUDING pairs where exactly one of C, D is empty (e.g. C != [], D == []
is still a genuinely mixed, non-CSS stabilizer set once C is nonempty --
only C == D == [] reduces all the way back to plain CSS). Such
exactly-one-empty pairs are intentionally retained, not filtered out.

Three generation strategies (Phase C1):

1. **All subset patterns** (principal, evolvable family) -- for a modest,
   bounded sweep of backbones, enumerate EVERY (C, D) with C subseteq A's
   2 terms and D subseteq B's 3 terms: 4 subsets of A x 8 subsets of B = 32
   combinations, minus the excluded (C=[], D=[]) CSS case = 31 per
   backbone. Full powerset, uniform over subset size.
2. **Sparse-biased perturbations** -- a WIDER, strided sweep of backbones
   (covering the full [1, min(ell-1,6)]/[1, min(m-1,6)] range that
   Strategy 1 only partially covers), but with only a SMALL, bounded
   number of *sparse* (C, D) patterns per backbone (single-element C or D,
   and a couple of two-element combinations) instead of the full
   powerset -- distinct in kind from Strategy 1, not just a subsample of
   it.
3. **Coordinate perturbations of retained non-CSS candidates** -- for a
   bounded seed set of Strategy-1 output (the backbone box's corners),
   perturb exactly ONE backbone exponent by +/-1 (mod ell or mod m, as
   appropriate) and remap C/D by STRUCTURAL ROLE so containment on the new
   backbone is preserved automatically (see ``_perturb_one`` docstring for
   the exact remap rule and its one drop case).

``_safety_net_candidates(ell, m)`` -- defined OUTSIDE the EVOLVE-BLOCK, and
invoked from ``generate_candidates``'s own non-evolvable wrapper (also
outside the block; only the internal ``_generate_candidates_evolved``
helper is mutable), so an LLM mutation can neither remove nor bypass it.

Each entry keeps ONE of (C, D) equal to the FULL corresponding backbone
term and the other a size-1 subset chosen so its shift, doubled, lands on
the backbone's own third term. This guarantees commutativity
unconditionally, for ANY (ell, m): ``A @ A^T`` is always symmetric for any
matrix A (transpose of a shift matrix is its group inverse, and the
group-algebra product ``f(x,y) * f(x^-1,y^-1)`` is invariant under
``x -> x^-1, y -> y^-1`` by commutativity of the product), so with
``C = A`` and ``D`` a single term of B chosen as ``b1`` where B's third
term is ``2*b1`` (mod ell), ``B @ D^T`` expands to ``x^-b1 + 1 + x^b1``,
itself symmetric under ``x -> x^-1`` -- so ``A @ C^T + B @ D^T`` is a sum
of two independently-symmetric matrices, hence symmetric, regardless of
(ell, m). This can never raise PBB's "Commutativity violated" build
error, for ell >= 3 and m >= 2 (the minimum range needed for the
backbone's own exponents 0, 1, 2 to be valid and distinct).

This replaces an earlier construction that instead set ``C = A`` AND
``D = B`` together (the full backbone on both sides). That pair is also
unconditionally commuting by the same argument, but was found (Commit-3
review) to be a hardcoded, lattice-independent LC-CSS-equivalence case in
``evaluation/clifford_equivalence.py`` -- i.e. ``passes_noncss_gate``
never actually passed for it, at any (ell, m). The two entries below were
each verified empirically via ``evaluation.noncss_gate.passes_noncss_gate``
(not just argued algebraically) at every Stage-1 lattice in both lattice
profiles: (6,6), (6,9), (9,8) for "small"; (12,9), (12,12), (15,12) for
"large" -- see this task's verification transcript for the exact check.

``generate_candidates(ell, m)`` -- the evolved function, returning
``list[tuple[A_terms, B_terms, C_terms, D_terms]]``. Raw counts are
intentionally unbounded -- capping to a manageable per-lattice budget is
the evaluator's job (Phase A5, ``evaluation.candidate_selection``), not
the seed's.
"""

from __future__ import annotations

import itertools

# ---------------------------------------------------------------------------
# Safety net -- OUTSIDE EVOLVE-BLOCK, non-removable by evolution.
# ---------------------------------------------------------------------------
# Each entry is a full (A, B, C, D) 4-tuple on the canonical weight-5 "2+3"
# backbone with A=[(0,0),(0,1)], B=[(0,0),(1,0),(2,0)] (or its A/B-swapped
# mirror), verified via evaluation.noncss_gate.passes_noncss_gate (not just
# argued algebraically -- see module docstring) at every Stage-1 lattice:
# (6,6), (6,9), (9,8), (12,9), (12,12), (15,12).
_SAFETY_NET_RAW = [
    ([(0, 0), (0, 1)], [(0, 0), (1, 0), (2, 0)], [(0, 0), (0, 1)], [(1, 0)]),
    ([(0, 0), (1, 0), (2, 0)], [(0, 0), (0, 1)], [(1, 0)], [(0, 0), (0, 1)]),
]


def _safety_net_candidates(ell, m):
    """Return hardcoded, gate-verified weight-5 non-CSS fallback
    candidates for (ell, m).

    OUTSIDE EVOLVE-BLOCK, and called only from ``generate_candidates``'s
    own non-evolvable wrapper -- the LLM can neither remove nor bypass
    this. Without it, if every evolved strategy collapses to k=0 or to
    candidates that all fail the "Commutativity violated" build check
    (commutativity is highly (ell, m)-specific -- see module docstring),
    Stage 1's cascade threshold is never met, Stage 2 never runs, and the
    LLM never receives distance feedback to recover from.

    Requires ell >= 3 and m >= 2 for the backbone's exponents to be valid
    and distinct; entries that don't fit a given (ell, m) are skipped
    (an empty safety net is a degraded fallback, not a crash).
    """
    out = []
    for A, B, C, D in _SAFETY_NET_RAW:
        if not all(0 <= x < ell and 0 <= y < m for x, y in A):
            continue
        if not all(0 <= x < ell and 0 <= y < m for x, y in B):
            continue
        if {len(set(A)), len(set(B))} != {2, 3}:
            continue
        out.append((list(A), list(B), list(C), list(D)))
    return out


def _all_subsets(terms):
    """All subsets of ``terms`` (a short list of distinct tuples), as lists,
    from the empty set up to the full set -- used for Strategy 1's full
    powerset enumeration."""
    out = []
    for r in range(len(terms) + 1):
        for combo in itertools.combinations(terms, r):
            out.append(list(combo))
    return out


def _perturb_one(A, B, C, D, pos, delta, ell, m):
    """Perturb ONE backbone exponent by +/-1 (mod ell or mod m) and remap
    C/D to preserve containment on the new backbone.

    ``pos`` indexes the 5 backbone exponents in order (a1, a2, b1, b2, b3):
    0,1 belong to A's second term (a1 is mod ell, a2 is mod m); 2,3,4
    belong to B's second/third terms (b1, b2 are mod ell, b3 is mod m).
    Only ONE of A, B actually changes for a given ``pos`` -- the other
    polynomial, and therefore the perturbation subset built on it (D when
    A changes, C when B changes), is passed through unmodified.

    Remap rule (structural-role remap, not element-search): each backbone
    polynomial has one "moving" non-constant term whose exponents shift
    with ``pos``, plus zero or more terms that don't move for this
    ``pos``. C (subset of A's 2 terms) or D (subset of B's 3 terms) is
    remapped by replacing, if present, the OLD moving term with the NEW
    moving term; the constant (0,0) and any other unmoved term already in
    C/D are left as-is. Because every element of the original C/D is
    either (0,0) or the moving term (the only two possible members for A;
    for B, the moved or an explicitly-unmoved term), this always yields a
    valid subset of the NEW backbone -- no element is ever dropped for
    "not mapping cleanly".

    The one case this function DOES drop (returns None) is backbone
    collapse: if the perturbed exponent lands on a value that makes the
    moving term coincide with another term already in A/B (e.g. wrapping
    b1 to 0 collides with B's own constant term), the backbone no longer
    has the required 2 (or 3) distinct terms, and the candidate is
    discarded rather than silently built on a degenerate backbone.
    """
    a1, a2 = A[1]
    b1 = B[1][0]
    b2, b3 = B[2]

    if pos == 0:
        new_a1 = (a1 + delta) % ell
        old_term, new_term = (a1, a2), (new_a1, a2)
        new_A = [(0, 0), new_term]
        new_B = B
        new_C = [new_term if t == old_term else t for t in C]
        new_D = D
    elif pos == 1:
        new_a2 = (a2 + delta) % m
        old_term, new_term = (a1, a2), (a1, new_a2)
        new_A = [(0, 0), new_term]
        new_B = B
        new_C = [new_term if t == old_term else t for t in C]
        new_D = D
    elif pos == 2:
        new_b1 = (b1 + delta) % ell
        old_term, new_term = (b1, 0), (new_b1, 0)
        new_B = [(0, 0), new_term, (b2, b3)]
        new_A = A
        new_D = [new_term if t == old_term else t for t in D]
        new_C = C
    elif pos == 3:
        new_b2 = (b2 + delta) % ell
        old_term, new_term = (b2, b3), (new_b2, b3)
        new_B = [(0, 0), (b1, 0), new_term]
        new_A = A
        new_D = [new_term if t == old_term else t for t in D]
        new_C = C
    elif pos == 4:
        new_b3 = (b3 + delta) % m
        old_term, new_term = (b2, b3), (b2, new_b3)
        new_B = [(0, 0), (b1, 0), new_term]
        new_A = A
        new_D = [new_term if t == old_term else t for t in D]
        new_C = C
    else:
        raise ValueError(f"pos must be 0..4, got {pos}")

    if len(set(new_A)) != 2 or len(set(new_B)) != 3:
        return None  # backbone collapsed -- drop
    return new_A, new_B, new_C, new_D


# EVOLVE-BLOCK-START
def _generate_candidates_evolved(
    ell: int, m: int,
) -> list[tuple[list[tuple[int, int]], list[tuple[int, int]],
                list[tuple[int, int]], list[tuple[int, int]]]]:
    """Generate weight-5 non-CSS PBB candidate (A, B, C, D) 4-tuples.

    Every candidate is built directly as C subseteq A, D subseteq B on the
    canonical weight-5 "2+3" backbone (A = [(0,0),(a1,a2)],
    B = [(0,0),(b1,0),(b2,b3)]) -- so containment (and therefore row
    weight 5) holds automatically by construction. The evaluator
    independently re-checks this (both symbolically and against the
    constructed circulant matrices) and re-checks commutativity, k, and
    the non-CSS gate -- this seed's job is only to generate promising
    candidates efficiently.

    This is the EVOLVABLE half of candidate generation only -- it does
    NOT include the safety net (that's prepended by the non-evolvable
    ``generate_candidates`` wrapper below, outside this block).

    Args:
        ell: Cyclic group order for x.
        m: Cyclic group order for y.

    Returns:
        List of (A_terms, B_terms, C_terms, D_terms) 4-tuples to evaluate.
    """
    candidates = []
    seen = set()

    def _add(A, B, C, D):
        """Add a 4-tuple candidate if not a duplicate and not the excluded
        (C=[], D=[]) plain-CSS case."""
        if not C and not D:
            return  # excluded: reduces to plain CSS, not a PBB candidate
        key = (
            tuple(sorted(A)), tuple(sorted(B)),
            tuple(sorted(C)), tuple(sorted(D)),
        )
        if key in seen:
            return
        seen.add(key)
        candidates.append((list(A), list(B), list(C), list(D)))

    Nx = min(ell - 1, 6)
    Ny = min(m - 1, 6)

    # -----------------------------------------------------------------
    # Strategy 1 (principal, evolvable): all subset patterns
    # -----------------------------------------------------------------
    # Modest, bounded backbone sweep (capped at 3 per exponent, well
    # inside the [1, Nx]/[1, Ny] box) x full (C, D) powerset per backbone
    # (4 subsets of A x 8 subsets of B = 32, minus the excluded (C=[],
    # D=[]) case = 31 per backbone). Capping the backbone range (rather
    # than the full min(ell-1,6)/min(m-1,6) box) keeps the raw count in
    # the same order of magnitude as the sibling CSS seed's own Strategy 1
    # (~a few thousand to ~10k) despite the 31x fan-out per backbone --
    # Strategy 2 below covers the wider box instead, with sparse (C, D)
    # sampling per backbone rather than the full powerset.
    Nx1 = min(Nx, 3)
    Ny1 = min(Ny, 3)

    if Nx1 >= 1 and Ny1 >= 1:
        for a1 in range(1, Nx1 + 1):
            for a2 in range(1, Ny1 + 1):
                A = [(0, 0), (a1, a2)]
                for b1 in range(1, Nx1 + 1):
                    for b2 in range(1, Nx1 + 1):
                        for b3 in range(1, Ny1 + 1):
                            B = [(0, 0), (b1, 0), (b2, b3)]
                            if len(set(B)) != 3:
                                continue
                            c_subsets = _all_subsets(A)
                            d_subsets = _all_subsets(B)
                            for C in c_subsets:
                                for D in d_subsets:
                                    _add(A, B, C, D)

    # -----------------------------------------------------------------
    # Strategy 2: sparse-biased perturbations over a wider backbone set
    # -----------------------------------------------------------------
    # Strided sweep across the FULL [1, Nx]/[1, Ny] box (stride 2, so it
    # reaches exponents up to Nx/Ny that Strategy 1's cap of 3 does not),
    # but with only a SMALL, fixed number of sparse (C, D) patterns per
    # backbone: each single-element subset of A paired with D=[], each
    # single-element subset of B paired with C=[], and each single
    # element of A paired with each single element of B (both nonempty,
    # 2 terms total) -- a bounded sample biased toward |C|,|D| small,
    # deliberately NOT the full powerset Strategy 1 already covers.
    if Nx >= 1 and Ny >= 1:
        stride_x = sorted(set(range(1, Nx + 1, 2)) | {Nx})
        stride_y = sorted(set(range(1, Ny + 1, 2)) | {Ny})
        for a1 in stride_x:
            for a2 in stride_y:
                A = [(0, 0), (a1, a2)]
                for b1 in stride_x:
                    for b2 in stride_x:
                        for b3 in stride_y:
                            B = [(0, 0), (b1, 0), (b2, b3)]
                            if len(set(B)) != 3:
                                continue
                            a_singletons = [[t] for t in A]
                            b_singletons = [[t] for t in B]
                            for C in a_singletons:
                                _add(A, B, C, [])
                            for D in b_singletons:
                                _add(A, B, [], D)
                            for C in a_singletons:
                                for D in b_singletons:
                                    _add(A, B, C, D)

    # -----------------------------------------------------------------
    # Strategy 3: coordinate perturbations of retained non-CSS candidates
    # -----------------------------------------------------------------
    # Seeded from the Strategy-1 backbone box's corners (2^5 = 32
    # backbones, each paired with its own 31 non-(C=[],D=[]) subset
    # patterns from Strategy 1's own powerset -- 992 base candidates),
    # perturb exactly one backbone exponent by +/-1 (mod ell or mod m) via
    # _perturb_one, which remaps C/D by structural role so containment on
    # the new backbone holds automatically (see _perturb_one docstring).
    if Nx1 >= 1 and Ny1 >= 1:
        corner_x = sorted({1, Nx1})
        corner_y = sorted({1, Ny1})
        for a1, a2, b1, b2, b3 in itertools.product(
            corner_x, corner_y, corner_x, corner_x, corner_y
        ):
            A = [(0, 0), (a1, a2)]
            B = [(0, 0), (b1, 0), (b2, b3)]
            if len(set(B)) != 3:
                continue
            for C in _all_subsets(A):
                for D in _all_subsets(B):
                    if not C and not D:
                        continue
                    for pos in range(5):
                        modulus = ell if pos in (0, 2, 3) else m
                        for delta in (-1, 1):
                            result = _perturb_one(A, B, C, D, pos, delta, ell, m)
                            if result is None:
                                continue
                            _add(*result)

    return candidates
# EVOLVE-BLOCK-END


def generate_candidates(
    ell: int, m: int,
) -> list[tuple[list[tuple[int, int]], list[tuple[int, int]],
                list[tuple[int, int]], list[tuple[int, int]]]]:
    """Entry point actually loaded by the evaluator (see
    ``_load_generate_candidates`` in ``openevolve_evaluator_weight5_pbb.py``,
    which looks up this exact name).

    OUTSIDE EVOLVE-BLOCK -- unlike ``_generate_candidates_evolved`` above,
    an LLM mutation of this program cannot remove or bypass the call to
    ``_safety_net_candidates`` here, only rewrite the evolvable helper's
    internals. Always prepends the safety net, then extends with whatever
    the evolvable strategies produce, deduplicating against the combined
    set and re-applying the plain-CSS exclusion (C == D == []) so the
    evolved half can't reintroduce it either.
    """
    candidates = list(_safety_net_candidates(ell, m))
    seen = {
        (tuple(sorted(a)), tuple(sorted(b)), tuple(sorted(c)), tuple(sorted(d)))
        for a, b, c, d in candidates
    }

    for A, B, C, D in _generate_candidates_evolved(ell, m):
        if not C and not D:
            continue  # excluded: reduces to plain CSS, not a PBB candidate
        key = (
            tuple(sorted(A)), tuple(sorted(B)),
            tuple(sorted(C)), tuple(sorted(D)),
        )
        if key in seen:
            continue
        seen.add(key)
        candidates.append((list(A), list(B), list(C), list(D)))

    return candidates


if __name__ == "__main__":
    # Quick self-test: containment holds by construction for every
    # candidate, cross-checked against evaluation.weight_enforcement's
    # own containment helper (Phase C1 verification requirement).
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from evaluation.weight_enforcement import pbb_support_contained, pbb_weight_ok

    for ell, m in [(6, 6), (6, 9), (9, 8)]:
        cands = generate_candidates(ell, m)
        assert cands, f"no candidates at ({ell},{m})"
        one_empty = 0
        for A, B, C, D in cands:
            assert pbb_support_contained(A, C), (A, C)
            assert pbb_support_contained(B, D), (B, D)
            assert pbb_weight_ok(A, B, C, D), (A, B, C, D)
            if (not C) != (not D) and (C or D):
                one_empty += 1
        print(
            f"({ell},{m}): {len(cands)} candidates, "
            f"{one_empty} exactly-one-empty, containment OK"
        )
    print("self-test passed")
