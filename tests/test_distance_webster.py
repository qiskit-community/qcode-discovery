"""Tier 0 bring-up tests for the Webster ``codedistance`` adapter.

These validate the wiring in ``evaluation/distance_webster.py`` before any
campaign runs (plan section 1, "Tier 0"):

  - CSS distance matches known values on the gross [[72,12,6]] code (both
    components), with a verified witness.
  - The A=B d=2 trap is detected by the exact method.
  - Non-CSS symplectic layout ([X|Z]) is correct: BZ on a known [[36,4,6]]
    PBB code agrees with its recorded distance and the witness verifies.
  - ``result_status`` classification is correct per method.
  - Witness verification rejects a trivial (stabilizer) operator.
  - The subprocess-timeout path returns the same answer in-process does.

The whole module is skipped if the optional ``codedistance`` package
(the ``webster`` dependency group) is not installed.
"""

import numpy as np
import pytest

pytest.importorskip("codedistance")

# Requires the optional `codedistance` package (uv sync --group webster).
pytestmark = pytest.mark.webster

from evaluation.bb_code import build_bb_code
from evaluation.pbb_code import build_pbb_code
from evaluation.distance_webster import (
    webster_distance_css,
    webster_distance_noncss,
    verify_witness_css,
    verify_witness_noncss,
)

# [[72,12,6]] gross code (Bravyi et al. arXiv:2308.07915).
GROSS_A = [(3, 0), (0, 1), (0, 2)]
GROSS_B = [(0, 3), (1, 0), (2, 0)]

# A known [[36,4,6]] non-CSS PBB code from campaign 7
# (results/campaign7_publication_merged.jsonl, smallest n).
PBB_ELL, PBB_M = 6, 3
PBB_A = [(0, 1), (3, 2), (4, 1)]
PBB_B = [(3, 1), (4, 0), (4, 1)]
PBB_C = [(0, 2), (2, 0), (4, 1)]
PBB_D = [(1, 2), (2, 0), (3, 0), (3, 1), (4, 2), (5, 2)]


@pytest.fixture(scope="module")
def gross_code():
    return build_bb_code(6, 6, GROSS_A, GROSS_B)


@pytest.fixture(scope="module")
def pbb_code():
    return build_pbb_code(PBB_ELL, PBB_M, PBB_A, PBB_B, PBB_C, PBB_D)


class TestCSSExact:
    @pytest.mark.parametrize("component", ["Z", "X"])
    def test_gross_bz_distance_is_6(self, gross_code, component):
        res = webster_distance_css(
            gross_code, method="BZDistMW", component=component, seed=42
        )
        assert res.d == 6
        assert res.n == 72
        assert res.k == 12
        assert res.result_status == "proven_exact"

    @pytest.mark.parametrize("component", ["Z", "X"])
    def test_gross_witness_verifies(self, gross_code, component):
        res = webster_distance_css(
            gross_code, method="BZDistMW", component=component, seed=42
        )
        v = verify_witness_css(gross_code, res.L, component, res.d)
        assert v["verified"], v
        assert v["weight"] == 6
        assert v["commutes"]
        assert v["anticommutes_with_logical"]

    def test_ab_trap_is_distance_2(self):
        """A=B codes have d=2 (paper Theorem 1); the exact method must see it."""
        code = build_bb_code(6, 6, GROSS_A, GROSS_A)
        res = webster_distance_css(code, method="BZDistMW", component="Z", seed=42)
        assert res.d == 2
        assert res.result_status == "proven_exact"


class TestNonCSSLayout:
    def test_pbb_bz_matches_recorded_distance(self, pbb_code):
        res = webster_distance_noncss(pbb_code, method="BZDistMW", seed=42)
        assert res.d == 6  # recorded distance for this [[36,4,6]] code
        assert res.n == 36
        assert res.result_status == "proven_exact"

    def test_pbb_witness_layout_and_verification(self, pbb_code):
        res = webster_distance_noncss(pbb_code, method="BZDistMW", seed=42)
        # Witness must be a 2n-vector in [LX | LZ] layout.
        assert len(res.L) == 2 * pbb_code.num_qudits
        v = verify_witness_noncss(pbb_code, res.L, res.d)
        assert v["verified"], v
        assert v["commutes"]
        assert v["anticommutes_with_logical"]
        assert v["weight"] == 6


class TestStatusClassification:
    def test_mipdist_no_budget_is_incumbent(self, gross_code):
        """Without an explicit maxTime we cannot recover solver optimality, so
        MIPDist is conservatively an incumbent upper bound (consistent across the
        in-process and subprocess paths)."""
        res = webster_distance_css(gross_code, method="MIPDist", component="Z", seed=42)
        assert res.d == 6
        assert res.result_status == "incumbent"

    def test_mipdist_under_budget_is_still_incumbent(self, gross_code):
        """MIPDist does not expose OR-Tools status, so wall-clock under budget is
        diagnostic only and must not be promoted to an optimality proof."""
        res = webster_distance_css(
            gross_code, method="MIPDist", component="Z", seed=42,
            params={"maxTime": 600},
        )
        assert res.d == 6
        assert res.result_status == "incumbent"
        assert res.optimality_source == "solver_status_not_exposed"

    def test_heuristic_methods_classified(self, gross_code):
        for method in ["QDistRndMW", "QDistEvol", "decoderDist"]:
            res = webster_distance_css(
                gross_code, method=method, component="Z", seed=42,
                params={"iterCount": 500},
            )
            assert res.d == 6, method
            assert res.result_status == "heuristic", method


class TestWitnessRejection:
    def test_stabilizer_is_not_a_valid_logical(self, gross_code):
        """A stabilizer row commutes but is trivial -> verification must fail."""
        hz = np.asarray(gross_code.matrix_z, dtype=np.int64) % 2
        stab_row = hz[0]
        v = verify_witness_css(gross_code, stab_row, "Z", int(stab_row.sum()))
        # Commutes with X-checks but anticommutes with no X-logical.
        assert v["commutes"]
        assert not v["anticommutes_with_logical"]
        assert not v["verified"]

    def test_wrong_length_noncss_witness_rejected(self, pbb_code):
        bad = np.ones(pbb_code.num_qudits, dtype=np.int64)  # n, not 2n
        v = verify_witness_noncss(pbb_code, bad, 6)
        assert not v["len_ok"]
        assert not v["verified"]

    def test_wrong_length_css_witness_rejected(self, gross_code):
        bad = np.ones(gross_code.num_qudits - 1, dtype=np.int64)
        v = verify_witness_css(gross_code, bad, "Z", 6)
        assert not v["len_ok"]
        assert not v["verified"]


class TestSubprocessTimeout:
    def test_subprocess_path_matches_inprocess(self, gross_code):
        """The spawned-process path (used by the runner) returns the same d."""
        res = webster_distance_css(
            gross_code, method="BZDistMW", component="Z", seed=42, timeout_s=120
        )
        assert res.d == 6
        assert res.result_status == "proven_exact"
        v = verify_witness_css(gross_code, res.L, "Z", res.d)
        assert v["verified"], v

    def test_noncss_result_reports_code_dimension(self, pbb_code):
        res = webster_distance_noncss(
            pbb_code, method="BZDistMW", seed=42, timeout_s=120
        )
        assert res.k == pbb_code.dimension
