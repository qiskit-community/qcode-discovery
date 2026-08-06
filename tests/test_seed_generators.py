"""Tests wiring the weight-5 seed generators' correctness properties into
pytest (Phase B1/C1).

evolve/seed_solution_weight5_pbb.py already has a containment/weight-ok
self-test, but it only runs via `python -m evolve.seed_solution_weight5_pbb`
(guarded by `if __name__ == "__main__":`), so it is invisible to
`pytest`. This file promotes that self-test (extended with the sibling
CSS generator and a few plan-specific bullets: modulus-N lattices never
generating exponent N, and identical-input determinism) into the
automated suite.
"""

from __future__ import annotations

from evaluation.noncss_gate import passes_noncss_gate
from evaluation.pbb_code import build_pbb_code
from evaluation.weight_enforcement import (
    css_weight_ok,
    pbb_row_weight,
    pbb_support_contained,
    pbb_weight_ok,
)
from evolve import seed_solution_weight5_css as css_mod
from evolve import seed_solution_weight5_pbb as pbb_mod
from evolve.seed_solution_weight5_css import generate_candidates as css_generate
from evolve.seed_solution_weight5_pbb import generate_candidates as pbb_generate

LATTICES = [(6, 6), (6, 9), (9, 8)]

# All six Stage-1 lattices (small + large profiles) -- the safety net is
# specifically claimed (module docstrings) to work at every one of these,
# not just the three in LATTICES above.
ALL_STAGE1_LATTICES = [(6, 6), (6, 9), (9, 8), (12, 9), (12, 12), (15, 12)]


class TestCssSeedGenerator:
    def test_every_candidate_satisfies_weight5_2_3_backbone(self):
        for ell, m in LATTICES:
            cands = css_generate(ell, m)
            assert cands, f"no candidates at ({ell},{m})"
            for A, B in cands:
                assert css_weight_ok(A, B), (ell, m, A, B)

    def test_modulus_lattice_never_generates_exponent_equal_to_modulus(self):
        # A generator using `% ell`/`% m` wraparound must never emit an
        # exponent equal to ell or m itself (Python's `%` guarantees this,
        # but the plan calls this out explicitly as a test bullet since a
        # future non-modular edit could regress it silently).
        ell, m = 6, 6
        for A, B in css_generate(ell, m):
            for x, y in A + B:
                assert x != ell and x < ell
                assert y != m and y < m

    def test_deterministic_identical_output_across_repeated_calls(self):
        ell, m = 9, 8
        c1 = css_generate(ell, m)
        c2 = css_generate(ell, m)
        assert c1 == c2

    def test_safety_net_survives_and_is_weight5(self):
        from evolve.seed_solution_weight5_css import _safety_net_candidates

        for ell, m in LATTICES:
            net = _safety_net_candidates(ell, m)
            assert net, f"safety net empty at ({ell},{m})"
            for A, B in net:
                assert css_weight_ok(A, B)

    def test_generate_candidates_keeps_safety_net_even_if_evolved_half_is_gutted(
        self, monkeypatch
    ):
        # Regression for Commit-3 Finding 2's EVOLVE-BLOCK restructuring:
        # the safety-net call site used to live INSIDE the evolvable
        # function, so an LLM mutation that rewrote/deleted it could take
        # the safety net down with it. Now the safety net is prepended by
        # the non-evolvable `generate_candidates` wrapper, which calls the
        # evolvable half only under a different name
        # (`_generate_candidates_evolved`) and merely extends its own
        # unconditional safety-net seed with whatever that returns. Simulate
        # the worst-case mutation -- the evolvable half returning nothing at
        # all -- and confirm the safety net still comes through.
        monkeypatch.setattr(css_mod, "_generate_candidates_evolved", lambda ell, m: [])
        for ell, m in LATTICES:
            net = css_mod._safety_net_candidates(ell, m)
            assert net, f"safety net empty at ({ell},{m})"
            cands = css_mod.generate_candidates(ell, m)
            for A, B in net:
                assert (list(A), list(B)) in cands


class TestPbbSeedGenerator:
    def test_every_candidate_satisfies_containment_and_weight5(self):
        for ell, m in LATTICES:
            cands = pbb_generate(ell, m)
            assert cands, f"no candidates at ({ell},{m})"
            for A, B, C, D in cands:
                assert pbb_support_contained(A, C), (ell, m, A, C)
                assert pbb_support_contained(B, D), (ell, m, B, D)
                assert pbb_weight_ok(A, B, C, D), (ell, m, A, B, C, D)
                assert pbb_row_weight(A, B, C, D) == 5

    def test_no_candidate_is_the_excluded_css_calibration_case(self):
        # (C=[], D=[]) must follow the sibling CSS evaluator's fast path,
        # not be emitted as a PBB candidate.
        for ell, m in LATTICES:
            for A, B, C, D in pbb_generate(ell, m):
                assert C or D, (ell, m, A, B, C, D)

    def test_exactly_one_empty_pairs_are_constructed(self):
        for ell, m in LATTICES:
            cands = pbb_generate(ell, m)
            one_empty = [
                (A, B, C, D) for A, B, C, D in cands
                if bool(C) != bool(D)
            ]
            assert one_empty, f"no exactly-one-empty candidates at ({ell},{m})"

    def test_both_nonempty_pairs_are_also_constructed(self):
        for ell, m in LATTICES:
            cands = pbb_generate(ell, m)
            both = [(A, B, C, D) for A, B, C, D in cands if C and D]
            assert both, f"no both-nonempty candidates at ({ell},{m})"

    def test_non_safety_net_candidates_either_build_or_fail_commutativity_cleanly(self):
        # Containment guarantees row weight 5, but NOT commutativity -- an
        # arbitrary C subseteq A, D subseteq B subset can still violate
        # "(A @ C^T + B @ D^T) % 2 symmetric" (only the safety net's
        # C=A, D=B identity guarantees it unconditionally; see that
        # generator's module docstring). The real evaluator
        # (_build_and_check in openevolve_evaluator_weight5_pbb.py) already
        # relies on this by catching the ValueError as a rejected-not-
        # crashed candidate -- this test pins that exact contract: every
        # candidate either builds, or fails with precisely this
        # commutativity ValueError, never some other exception type.
        ell, m = 6, 6
        cands = pbb_generate(ell, m)
        built, rejected = 0, 0
        for A, B, C, D in cands[:200]:
            try:
                code = build_pbb_code(ell, m, A, B, C, D)
                assert code is not None
                built += 1
            except ValueError as e:
                assert "Commutativity violated" in str(e)
                rejected += 1
        assert built > 0
        assert built + rejected == len(cands[:200])

    def test_safety_net_candidates_always_build_successfully(self):
        # Unlike the general population above, the C=A, D=B safety net is
        # guaranteed to commute unconditionally -- it must never raise.
        from evolve.seed_solution_weight5_pbb import _safety_net_candidates

        for ell, m in LATTICES:
            for A, B, C, D in _safety_net_candidates(ell, m):
                code = build_pbb_code(ell, m, A, B, C, D)
                assert code is not None

    def test_modulus_lattice_never_generates_exponent_equal_to_modulus(self):
        ell, m = 6, 6
        for A, B, C, D in pbb_generate(ell, m):
            for x, y in A + B + C + D:
                assert x != ell and x < ell
                assert y != m and y < m

    def test_deterministic_identical_output_across_repeated_calls(self):
        ell, m = 9, 8
        c1 = pbb_generate(ell, m)
        c2 = pbb_generate(ell, m)
        assert c1 == c2

    def test_safety_net_survives_and_is_weight5_and_nonempty(self):
        from evolve.seed_solution_weight5_pbb import _safety_net_candidates

        for ell, m in LATTICES:
            net = _safety_net_candidates(ell, m)
            assert net, f"safety net empty at ({ell},{m})"
            for A, B, C, D in net:
                assert pbb_weight_ok(A, B, C, D)
                assert C and D

    def test_both_raw_safety_net_entries_survive_at_every_lattice(self):
        # Regression for round-2 finding #2 (part A): _safety_net_candidates'
        # guard used to read `if len(set(A)) != 2 or len(set(B)) != 3:
        # continue`, which always discarded _SAFETY_NET_RAW's second,
        # A/B-mirrored entry (len(set(A))==3, len(set(B))==2) -- so only
        # ONE of the two raw entries ever survived, at every lattice. The
        # existing test above only asserts the net is *nonempty*, which a
        # single surviving entry also satisfies, so it never caught this.
        # The fixed guard is `if {len(set(A)), len(set(B))} != {2, 3}:
        # continue`, which accepts both orderings.
        from evolve.seed_solution_weight5_pbb import _safety_net_candidates

        for ell, m in ALL_STAGE1_LATTICES:
            net = _safety_net_candidates(ell, m)
            assert len(net) == 2, (ell, m, net)
            shapes = {(len(set(A)), len(set(B))) for A, B, C, D in net}
            assert shapes == {(2, 3), (3, 2)}, (ell, m, shapes)

    def test_safety_net_candidates_actually_pass_the_noncss_gate(self):
        # Regression for Commit-3 Finding 2: the *old* C=A, D=B safety net
        # built successfully and satisfied pbb_weight_ok, but was never
        # actually non-CSS -- passes_noncss_gate rejected it at every
        # lattice as a hardcoded LC-CSS-equivalence special case (see
        # module docstring). "Builds and is weight-5" is necessary but not
        # sufficient; this test closes that gap by running the real gate,
        # at every Stage-1 lattice in both lattice profiles, not just the
        # three small ones.
        from evolve.seed_solution_weight5_pbb import _safety_net_candidates

        for ell, m in ALL_STAGE1_LATTICES:
            net = _safety_net_candidates(ell, m)
            assert net, f"safety net empty at ({ell},{m})"
            for A, B, C, D in net:
                code = build_pbb_code(ell, m, A, B, C, D)
                result = passes_noncss_gate(ell, m, A, B, C, D, code)
                assert result["passes"], (ell, m, A, B, C, D, result)
                assert result["is_lc_css"] is False, (ell, m, A, B, C, D, result)
                assert result["is_css"] is False, (ell, m, A, B, C, D, result)

    def test_generate_candidates_keeps_safety_net_even_if_evolved_half_is_gutted(
        self, monkeypatch
    ):
        # PBB-side counterpart of the CSS regression above: the safety net
        # must survive an evolvable half that contributes nothing, because
        # it is seeded unconditionally by the non-evolvable
        # `generate_candidates` wrapper rather than by the (LLM-mutable)
        # `_generate_candidates_evolved` helper.
        monkeypatch.setattr(pbb_mod, "_generate_candidates_evolved", lambda ell, m: [])
        for ell, m in LATTICES:
            net = pbb_mod._safety_net_candidates(ell, m)
            assert net, f"safety net empty at ({ell},{m})"
            cands = pbb_mod.generate_candidates(ell, m)
            for A, B, C, D in net:
                assert (list(A), list(B), list(C), list(D)) in cands
