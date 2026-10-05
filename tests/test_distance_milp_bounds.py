"""Regression tests: MILP early stops and timeouts must not certify distances."""

import numpy as np
from qldpc.codes import QuditCode

import evaluation.distance_milp as distance_milp
from evaluation.bb_code import build_bb_code
from evaluation.distance_milp import (
    compute_distance_milp,
    compute_distance_milp_symplectic,
)


def _five_qubit_code_plus_free_qubit():
    """[[5,1,3]] with an unprotected sixth qubit: [[6,2,1]]."""
    rows = []
    for shift in range(4):
        x = np.zeros(6, dtype=int)
        z = np.zeros(6, dtype=int)
        for offset, pauli in enumerate("XZZX"):
            q = (shift + offset) % 5
            if pauli == "X":
                x[q] = 1
            else:
                z[q] = 1
        rows.append(np.concatenate([x, z]))
    return QuditCode(np.array(rows))


class TestSymplecticEarlyStop:
    def test_early_stop_never_reports_exact_for_partial_sweep(self):
        code = _five_qubit_code_plus_free_qubit()
        d, details = compute_distance_milp_symplectic(
            code, timeout_per_logical=30, total_timeout=60, early_stop=4
        )
        # Every logical has weight <= 4, so the sweep stops after one of the
        # four objectives; the true distance (1) may be among those skipped.
        assert details["k"] == 2
        assert details["num_logicals_checked"] == 1 < details["total_logicals"]
        assert details["exact"] is False
        assert d >= 1

    def test_full_sweep_finds_true_distance(self):
        code = _five_qubit_code_plus_free_qubit()
        d, details = compute_distance_milp_symplectic(
            code, timeout_per_logical=30, total_timeout=60, early_stop=None
        )
        assert d == 1
        assert details["exact"] is True
        assert details["num_logicals_checked"] == details["total_logicals"] == 4


class TestTimeoutGivesNoLowerBound:
    def test_symplectic_zero_budget(self):
        code = _five_qubit_code_plus_free_qubit()
        d, details = compute_distance_milp_symplectic(
            code, timeout_per_logical=30, total_timeout=1e-9, early_stop=4
        )
        assert details["all_timeout"] is True
        assert details["d_is_lower_bound"] is False
        assert details["exact"] is False
        assert d == code.num_qudits  # vacuous upper bound, not early_stop+1

    def test_css_zero_budget(self):
        code = build_bb_code(6, 3, [(0, 0), (0, 1), (0, 2)], [(1, 0), (0, 0), (0, 1)])
        d, details = compute_distance_milp(
            code, timeout_per_logical=30, total_timeout=1e-9, early_stop=4
        )
        assert details["all_timeout"] is True
        assert details["d_is_lower_bound"] is False
        assert details["exact"] is False
        assert d == code.num_qudits


class TestTimeoutAfterAttemptedObjective:
    """The zero-budget tests above never call the solver at all (the
    ``remaining <= 0`` check short-circuits before the first ``ilp_min_weight``
    call). This exercises the other way ``all_timeout`` can arise: the solver
    is actually invoked -- possibly for every logical -- but each call times
    out with no incumbent (``(None, False)``), so ``any_z_found``/
    ``any_x_found``/``any_found`` all stay False despite a non-trivial number
    of attempts."""

    def test_css_attempted_objective_gives_no_lower_bound(self, monkeypatch):
        code = build_bb_code(6, 3, [(0, 0), (0, 1), (0, 2)], [(1, 0), (0, 0), (0, 1)])
        calls = []

        def fake_ilp(check_matrix, logical_op, timeout=30):
            calls.append(None)
            return None, False  # solver ran, timed out with no incumbent

        monkeypatch.setattr(distance_milp, "ilp_min_weight", fake_ilp)
        d, details = compute_distance_milp(
            code, timeout_per_logical=30, total_timeout=60, early_stop=4
        )
        assert len(calls) >= 1
        assert details["all_timeout"] is True
        assert details["d_is_lower_bound"] is False
        assert details["exact"] is False
        assert d == code.num_qudits

    def test_symplectic_attempted_objective_gives_no_lower_bound(self, monkeypatch):
        code = _five_qubit_code_plus_free_qubit()
        calls = []

        def fake_ilp_symp(stabilizer_matrix, logical_op, timeout=30):
            calls.append(None)
            return None, False

        monkeypatch.setattr(distance_milp, "ilp_min_weight_symplectic", fake_ilp_symp)
        d, details = compute_distance_milp_symplectic(
            code, timeout_per_logical=30, total_timeout=60, early_stop=4
        )
        assert len(calls) >= 1
        assert details["all_timeout"] is True
        assert details["d_is_lower_bound"] is False
        assert details["exact"] is False
        assert d == code.num_qudits


class TestCssXLoopEarlyStop:
    def test_x_loop_early_exit_is_not_exact(self, monkeypatch):
        # Every Z objective stays above early_stop; the first X objective
        # drops to it, so the X loop exits before the remaining X logicals.
        code = build_bb_code(6, 3, [(0, 0), (0, 1), (0, 2)], [(1, 0), (0, 0), (0, 1)])
        k = code.dimension
        assert k >= 2
        calls = []

        def fake_ilp(check_matrix, logical_op, timeout=30):
            calls.append(None)
            return (6, True) if len(calls) <= k else (3, True)

        monkeypatch.setattr(distance_milp, "ilp_min_weight", fake_ilp)
        d, details = compute_distance_milp(
            code, timeout_per_logical=30, total_timeout=60, early_stop=4
        )
        assert d == 3
        assert details["num_logicals_checked"] == k + 1 < details["total_logicals"]
        assert details["exact"] is False
