"""Combined non-CSS classification gate for PBB candidates (Phase A4).

A weight-5 PBB candidate is only retained if it passes both repository
equivalence checks, run with their actual APIs
(`plans/direction2_weight5_campaigns.md`, Phase A4):

    verify_not_lc_css(ell, m, A_terms, B_terms, C_terms, D_terms)["is_lc_css"] is False
    is_equivalently_css(constructed_code)["is_css"] is False

Passing both means "not equivalent to CSS under implemented checks" -- it is
not a proof against every local-Clifford pattern (see the LC equivalence
Theorem 3 gotcha in CLAUDE.md: the {SH,HSH} non-uniform case is not
exhaustively checked).

`verify_not_lc_css` is the cheaper, purely algebraic check (brute-force over
36 uniform Clifford assignments plus a GF(2) affine solve) and is run first;
`is_equivalently_css` needs the constructed code object and is only run if
the first check has not already determined the candidate is LC-CSS
equivalent (short-circuit, since the conjunction is already False either way
once one check reports True).
"""

from __future__ import annotations

from evaluation.clifford_equivalence import is_equivalently_css, verify_not_lc_css


def _canonical_key(ell, m, A_terms, B_terms, C_terms, D_terms) -> tuple:
    def _norm(terms):
        return tuple(sorted(tuple(t) for t in (terms or [])))

    return (ell, m, _norm(A_terms), _norm(B_terms), _norm(C_terms), _norm(D_terms))


def passes_noncss_gate(
    ell: int,
    m: int,
    A_terms: list[tuple[int, int]],
    B_terms: list[tuple[int, int]],
    C_terms: list[tuple[int, int]] | None,
    D_terms: list[tuple[int, int]] | None,
    constructed_code,
    *,
    cache: dict | None = None,
) -> dict:
    """Check both non-CSS gates; short-circuits and caches by canonical key.

    Returns a dict with ``passes`` (bool), ``is_lc_css`` (bool),
    ``is_css`` (bool or None -- None if short-circuited before this check
    ran), and ``lc_css_details``/``css_details`` (the raw dicts from the
    underlying checks, ``css_details`` is None if short-circuited).

    ``cache`` (if supplied) is a plain dict keyed by the canonical
    ``(ell, m, A, B, C, D)`` tuple, shared across calls by the caller (e.g.
    a module-level or evaluator-instance dict) so repeated candidates across
    an evolutionary run are not re-verified.
    """
    key = _canonical_key(ell, m, A_terms, B_terms, C_terms, D_terms)
    if cache is not None and key in cache:
        return cache[key]

    lc_result = verify_not_lc_css(ell, m, A_terms, B_terms, C_terms, D_terms)
    is_lc_css = lc_result["is_lc_css"]

    if is_lc_css:
        result = {
            "passes": False,
            "is_lc_css": True,
            "is_css": None,
            "lc_css_details": lc_result,
            "css_details": None,
        }
    else:
        css_result = is_equivalently_css(constructed_code)
        is_css = css_result["is_css"]
        result = {
            "passes": not is_css,
            "is_lc_css": False,
            "is_css": is_css,
            "lc_css_details": lc_result,
            "css_details": css_result,
        }

    if cache is not None:
        cache[key] = result
    return result
