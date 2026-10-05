"""MILP-based exact distance computation for quantum codes.

Uses the integer-programming formulation introduced by Landahl, Anderson, and
Rice (arXiv:1108.5738, 2011) and reused by Bravyi et al. (arXiv:2308.07915,
which cites Landahl-Anderson-Rice as the source of the method) with
optimizations for use during evolutionary search:

  - Early exit when d drops to ``early_stop`` threshold (the result is then
    an upper bound only: ``exact`` requires every logical objective solved)
  - Cross-type early exit: skip d_X if d_Z already ≤ early_stop
  - Per-code total timeout (not just per-logical)
  - d_Z computed before d_X (cheaper to determine d is low)

Supports both CSS codes (``ilp_min_weight``, ``compute_distance_milp``)
and non-CSS codes (``ilp_min_weight_symplectic``,
``compute_distance_milp_symplectic``).

Core solver: HiGHS via ``scipy.optimize.milp``.  Thread-safe (no SIGALRM).
"""

from __future__ import annotations

import logging
import time

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

from qldpc.objects import Pauli

logger = logging.getLogger(__name__)


def get_code_matrices(code):
    """Extract check matrices and logical operators from a qldpc BBCode.

    Returns (hx, hz, lx, lz) -- all binary numpy arrays.
    """
    hx = np.array(code.matrix_x, dtype=int) % 2
    hz = np.array(code.matrix_z, dtype=int) % 2
    lx = np.array(code.get_logical_ops(Pauli.X), dtype=int) % 2
    lz = np.array(code.get_logical_ops(Pauli.Z), dtype=int) % 2
    return hx, hz, lx, lz


def symplectic_weight_bound(code):
    """Upper bound on code distance from symplectic logical operator weights.

    Returns the minimum Hamming weight across all logical operators obtained
    via Gaussian elimination (the initial symplectic basis).  This is a valid
    upper bound on d because any logical operator witnesses d ≤ weight.

    Cost: milliseconds (no solver, just GF(2) linear algebra already done
    by qldpc internally).

    Returns:
        (d_upper, d_x_upper, d_z_upper) -- ints.
    """
    lx = np.array(code.get_logical_ops(Pauli.X), dtype=int) % 2
    lz = np.array(code.get_logical_ops(Pauli.Z), dtype=int) % 2

    d_x_upper = int(np.min(np.sum(lx, axis=1))) if lx.size > 0 else code.num_qudits
    d_z_upper = int(np.min(np.sum(lz, axis=1))) if lz.size > 0 else code.num_qudits
    d_upper = min(d_x_upper, d_z_upper)

    return d_upper, d_x_upper, d_z_upper


def ilp_min_weight(check_matrix, logical_op, timeout=30):
    """Find minimum-weight operator orthogonal to checks, anticommuting with logical_op.

    Formulation: binary variables x_j for each of n qubits, with mod-2
    constraints encoded as integer equalities using slack variables.

    Args:
        check_matrix: (m, n) binary matrix of stabilizer checks.
        logical_op: (n,) binary vector of a logical operator.
        timeout: solver time limit in seconds.

    Returns:
        (weight, optimal) tuple: weight is the minimum weight found (int),
        optimal is True if proven optimal.  Returns (None, False) when no
        feasible solution was found at all (infeasibility or no incumbent).
    """
    m, n = check_matrix.shape
    num_vars = n + m + 1  # [x_0..x_{n-1}, s_0..s_{m-1}, t]

    # Objective: minimize Hamming weight
    c = np.zeros(num_vars)
    c[:n] = 1.0

    # Constraint matrix: stabilizer orthogonality + logical anticommutation
    rows = []
    for r in range(m):
        row = np.zeros(num_vars)
        row[:n] = check_matrix[r]
        row[n + r] = -2
        rows.append(row)

    row = np.zeros(num_vars)
    row[:n] = logical_op
    row[n + m] = -2
    rows.append(row)

    A = np.array(rows)

    b_lb = np.zeros(m + 1)
    b_ub = np.zeros(m + 1)
    b_lb[m] = 1
    b_ub[m] = 1

    constraints = LinearConstraint(A, b_lb, b_ub)

    # Bounds
    lb = np.zeros(num_vars)
    ub = np.ones(num_vars)
    for r in range(m):
        ub[n + r] = np.ceil(np.sum(check_matrix[r]) / 2)
    ub[n + m] = np.ceil(np.sum(logical_op) / 2)

    bounds = Bounds(lb, ub)
    integrality = np.ones(num_vars)

    opts = {"presolve": True}
    if 0 < timeout < 1e9:
        opts["time_limit"] = timeout

    result = milp(
        c=c,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        options=opts,
    )

    if result.x is not None:
        w = int(round(result.fun))
        return w, result.success  # success=True means proven optimal
    return None, False


def compute_distance_milp(
    code,
    *,
    timeout_per_logical: int = 30,
    total_timeout: int = 120,
    early_stop: int | None = 4,
    verbose: bool = False,
) -> tuple[int, dict]:
    """Compute exact code distance via MILP with early-exit optimizations.

    Optimizations over the basic ILP approach:
    1. Stops as soon as d drops to ``early_stop`` (most bad codes have d=2-4).
    2. Computes d_Z first; skips d_X if d_Z ≤ early_stop.
    3. Respects a total time budget across all logicals.
    4. Adapts per-logical timeout to ensure broad coverage: when k is large,
       uses shorter per-logical timeouts to check more logicals (coverage
       matters more than per-logical optimality for finding min-weight).

    Args:
        code: qldpc BBCode instance.
        timeout_per_logical: Max seconds per individual ILP solve.
        total_timeout: Max total seconds for the entire distance computation.
        early_stop: Stop immediately when d ≤ this value; the result is
            then an upper bound with ``exact=False``. Pass ``None`` to
            disable early stopping and iterate over every logical (required
            when the goal is to certify an exact distance).
        verbose: Log progress.

    Returns:
        (d, details) where d is the exact distance (or best upper bound on
        timeout) and details contains d_x, d_z, num_logicals_checked, time_s,
        and exact (bool indicating whether the result is provably exact).
    """
    # Convert 0 = unlimited to effectively infinite budget
    if timeout_per_logical <= 0:
        timeout_per_logical = float("inf")
    if total_timeout <= 0:
        total_timeout = float("inf")

    n = code.num_qudits
    k = code.dimension
    if k == 0:
        return n, {"d_x": n, "d_z": n, "k": 0, "exact": True,
                    "num_logicals_checked": 0, "total_logicals": 0,
                    "time_s": 0.0}

    hx, hz, lx, lz = get_code_matrices(code)
    t_start = time.monotonic()
    logicals_checked = 0
    logicals_optimal = 0   # Proven optimal by solver
    logicals_incumbent = 0  # Feasible solution found but not proven optimal
    all_solved = True  # Track whether all logicals were solved (no timeouts)

    # Per-logical timeout is passed through unmodified.  The caller
    # (evaluator) sets the budget; total_timeout is enforced via the
    # _remaining() check before each logical.  This avoids the old
    # adaptive formula that starved per-logical time when k was large
    # (e.g. k=24 with total=300s → only 6s/logical, far too low).

    def _remaining():
        return max(0, total_timeout - (time.monotonic() - t_start))

    # --- Z-distance: min-weight Z-op commuting with X-checks ---
    d_z = n
    any_z_found = False  # Any feasible solution (optimal or incumbent)
    for i in range(k):
        remaining = _remaining()
        if remaining <= 0:
            all_solved = False
            break
        timeout = min(timeout_per_logical, remaining)
        w, optimal = ilp_min_weight(hx, lx[i], timeout=timeout)
        logicals_checked += 1
        if w is not None:
            d_z = min(d_z, w)
            any_z_found = True
            if optimal:
                logicals_optimal += 1
            else:
                logicals_incumbent += 1
                all_solved = False
            if verbose:
                tag = "" if optimal else " (incumbent)"
                logger.info("Z[%d]: d=%d%s (%.1fs)", i, w, tag,
                            time.monotonic() - t_start)
        else:
            all_solved = False
            if verbose:
                logger.info("Z[%d]: no solution (%.1fs)", i,
                            time.monotonic() - t_start)
        if early_stop is not None and d_z <= early_stop:
            break

    # --- Cross-type early exit ---
    # d = min(d_X, d_Z). If d_Z is already very low, no point computing d_X.
    if early_stop is not None and d_z <= early_stop:
        d = d_z
        elapsed = time.monotonic() - t_start
        # d_z ≤ early_stop was found as a feasible solution (optimal or
        # incumbent). Either way, d ≤ d_z is a valid upper bound.
        # exact=False because d_x was not computed -- we cannot prove d_x >= d_z.
        return d, {
            "d_x": n,  # Not computed
            "d_z": d_z,
            "k": k,
            "exact": False,
            "d_x_computed": False,
            "num_logicals_checked": logicals_checked,
            "logicals_optimal": logicals_optimal,
            "logicals_incumbent": logicals_incumbent,
            "total_logicals": 2 * k,
            "time_s": elapsed,
            "timeout_per_logical": timeout_per_logical,
        }

    # --- X-distance: min-weight X-op commuting with Z-checks ---
    d_x = n
    any_x_found = False
    for i in range(k):
        remaining = _remaining()
        if remaining <= 0:
            all_solved = False
            break
        timeout = min(timeout_per_logical, remaining)
        w, optimal = ilp_min_weight(hz, lz[i], timeout=timeout)
        logicals_checked += 1
        if w is not None:
            d_x = min(d_x, w)
            any_x_found = True
            if optimal:
                logicals_optimal += 1
            else:
                logicals_incumbent += 1
                all_solved = False
            if verbose:
                tag = "" if optimal else " (incumbent)"
                logger.info("X[%d]: d=%d%s (%.1fs)", i, w, tag,
                            time.monotonic() - t_start)
        else:
            all_solved = False
            if verbose:
                logger.info("X[%d]: no solution (%.1fs)", i,
                            time.monotonic() - t_start)
        # Early exit: d_X already below d_Z, no need to check more
        if early_stop is not None and d_x <= early_stop:
            break

    elapsed = time.monotonic() - t_start

    # If no feasible solution was found on either side, distance is unknown.
    # A timeout proves nothing about small logicals, so return the vacuous
    # upper bound d=n and no lower bound (see compute_distance_milp_symplectic).
    if not any_z_found and not any_x_found:
        return n, {
            "d_x": 0,
            "d_z": 0,
            "k": k,
            "exact": False,
            "d_x_computed": True,
            "num_logicals_checked": logicals_checked,
            "logicals_optimal": logicals_optimal,
            "logicals_incumbent": logicals_incumbent,
            "total_logicals": 2 * k,
            "time_s": elapsed,
            "timeout_per_logical": timeout_per_logical,
            "all_timeout": True,
            "d_is_lower_bound": False,
        }

    # Use the best feasible values found. Unsolved sides stay at n
    # (trivially valid upper bound).
    d = min(d_x, d_z)
    # An early exit inside the X loop skips the remaining X logicals, whose
    # minimum could be smaller; only a complete sweep certifies d.
    if logicals_checked < 2 * k:
        all_solved = False

    return d, {
        "d_x": d_x if any_x_found else 0,
        "d_z": d_z if any_z_found else 0,
        "k": k,
        "exact": all_solved,
        "d_x_computed": True,
        "num_logicals_checked": logicals_checked,
        "logicals_optimal": logicals_optimal,
        "logicals_incumbent": logicals_incumbent,
        "total_logicals": 2 * k,
        "time_s": elapsed,
        "timeout_per_logical": timeout_per_logical,
    }


# ---------------------------------------------------------------------------
# Non-CSS (symplectic) MILP formulation
# ---------------------------------------------------------------------------


def ilp_min_weight_symplectic(stabilizer_matrix, logical_op, timeout=30):
    """Find minimum symplectic-weight Pauli in the coset logical_op + stabilizers.

    Uses the symplectic ILP of Landahl, Anderson, and Rice (arXiv:1108.5738,
    2011) with the standard linear encoding of the per-qubit binary OR
    (w_j = x_j OR z_j) via w_j >= x_j and w_j >= z_j; the upper-bound
    constraint w_j <= x_j + z_j is omitted because the minimization objective
    drives w_j down to max(x_j, z_j) on its own. (Note: this is the convex-hull
    description of binary OR, not McCormick relaxation -- McCormick envelopes
    apply to bilinear products of continuous variables.)

    For non-CSS codes, each Pauli operator is (x_1..x_n, z_1..z_n) and its
    symplectic weight is the number of qubits i where x_i OR z_i is nonzero.

    Variables:
      - x_j, z_j: binary, the Pauli operator on qubit j  (2n vars)
      - w_j: binary, 1 if qubit j has nontrivial support   (n vars)
      - s_r: integer slack for mod-2 commutation constraints (num_stabs vars)
      - t:   integer slack for the anticommutation constraint (1 var)

    Objective: minimize sum(w_j)

    Constraints:
      - w_j >= x_j  and  w_j >= z_j  (symplectic weight linearization)
      - For each stabilizer s: sum_j(s_xj * z_j + s_zj * x_j) - 2*s_r = 0
        (commutation, mod-2 encoded via integer slack)
      - For the target logical L: sum_j(L_xj * z_j + L_zj * x_j) - 2*t = 1
        (anticommutation)

    Args:
        stabilizer_matrix: (num_stabs, 2n) binary symplectic matrix.
        logical_op: (2n,) binary vector of a logical operator.
        timeout: solver time limit in seconds.

    Returns:
        (weight, optimal) tuple.  Returns (None, False) if no feasible
        solution found.
    """
    num_stabs, two_n = stabilizer_matrix.shape
    n = two_n // 2

    # Variable layout: [x_0..x_{n-1}, z_0..z_{n-1}, w_0..w_{n-1},
    #                   s_0..s_{num_stabs-1}, t]
    num_vars = 2 * n + n + num_stabs + 1
    idx_x = slice(0, n)
    idx_z = slice(n, 2 * n)
    idx_w = slice(2 * n, 3 * n)
    # Stabilizer slack vars (s_0..s_{num_stabs-1}) live at indices
    # [3n, 3n + num_stabs); they are indexed directly below.
    idx_t = 3 * n + num_stabs

    # Objective: minimize sum(w_j)
    c = np.zeros(num_vars)
    c[idx_w] = 1.0

    # --- Constraints ---
    rows = []
    row_lb = []
    row_ub = []

    # 1. w_j >= x_j  =>  w_j - x_j >= 0
    for j in range(n):
        row = np.zeros(num_vars)
        row[2 * n + j] = 1   # w_j
        row[j] = -1           # -x_j
        rows.append(row)
        row_lb.append(0)
        row_ub.append(np.inf)

    # 2. w_j >= z_j  =>  w_j - z_j >= 0
    for j in range(n):
        row = np.zeros(num_vars)
        row[2 * n + j] = 1   # w_j
        row[n + j] = -1       # -z_j
        rows.append(row)
        row_lb.append(0)
        row_ub.append(np.inf)

    # 3. Commutation with each stabilizer:
    #    sum_j(s_xj * z_j + s_zj * x_j) - 2*s_r = 0
    for r in range(num_stabs):
        row = np.zeros(num_vars)
        s_x = stabilizer_matrix[r, :n]   # X-part of stabilizer
        s_z = stabilizer_matrix[r, n:]   # Z-part of stabilizer
        # symplectic inner product: s_x . z + s_z . x
        row[idx_z] = s_x      # s_xj * z_j
        row[idx_x] = s_z      # s_zj * x_j
        row[3 * n + r] = -2   # -2 * s_r (slack)
        rows.append(row)
        row_lb.append(0)
        row_ub.append(0)

    # 4. Anticommutation with target logical:
    #    sum_j(L_xj * z_j + L_zj * x_j) - 2*t = 1
    row = np.zeros(num_vars)
    L_x = logical_op[:n]
    L_z = logical_op[n:]
    row[idx_z] = L_x
    row[idx_x] = L_z
    row[idx_t] = -2
    rows.append(row)
    row_lb.append(1)
    row_ub.append(1)

    A_mat = np.array(rows)
    constraints = LinearConstraint(A_mat, row_lb, row_ub)

    # Bounds
    lb = np.zeros(num_vars)
    ub = np.ones(num_vars)
    # Slack vars for stabilizer commutation: s_r can be up to ceil(weight/2)
    for r in range(num_stabs):
        ub[3 * n + r] = np.ceil(np.sum(stabilizer_matrix[r]) / 2)
    # Slack for anticommutation
    ub[idx_t] = np.ceil(np.sum(logical_op) / 2)

    bounds = Bounds(lb, ub)
    integrality = np.ones(num_vars)

    opts = {"presolve": True}
    if 0 < timeout < 1e9:
        opts["time_limit"] = timeout

    result = milp(
        c=c,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        options=opts,
    )

    if result.x is not None:
        w = int(round(result.fun))
        return w, result.success
    return None, False


def compute_distance_milp_symplectic(
    code,
    *,
    timeout_per_logical: int = 30,
    total_timeout: int = 120,
    early_stop: int | None = 4,
    verbose: bool = False,
) -> tuple[int, dict]:
    """Compute code distance via symplectic MILP for non-CSS codes.

    Unlike the CSS version which separates d_X and d_Z, this formulation
    works with the full symplectic representation and minimizes symplectic
    weight directly.

    Args:
        code: A qubit stabilizer code, non-CSS (qldpc's ``QuditCode``).
        timeout_per_logical: Max seconds per individual ILP solve.
        total_timeout: Max total seconds for the entire computation.
        early_stop: Stop immediately when d <= this value; the result is
            then an upper bound with ``exact=False``. Pass ``None`` to
            disable early stopping and iterate over every logical (required
            when the goal is to certify an exact distance).
        verbose: Log progress.

    Returns:
        (d, details) where d is the distance (or best upper bound on
        timeout) and details dict.
    """
    if timeout_per_logical <= 0:
        timeout_per_logical = float("inf")
    if total_timeout <= 0:
        total_timeout = float("inf")

    n = code.num_qudits
    k = code.dimension
    if k == 0:
        return n, {"k": 0, "exact": True, "num_logicals_checked": 0,
                    "total_logicals": 0, "time_s": 0.0}

    # Get stabilizer matrix and logical operators
    # Use our own GF(2) computation for logicals -- qldpc's get_logical_ops()
    # has bugs for some non-CSS codes (singular matrix errors).
    from evaluation.pbb_code import get_symplectic_logicals
    stab_matrix = np.array(code.matrix, dtype=int) % 2
    logicals = get_symplectic_logicals(code)
    num_logicals = logicals.shape[0]  # 2k logicals

    t_start = time.monotonic()
    logicals_checked = 0
    logicals_optimal = 0
    logicals_incumbent = 0
    all_solved = True
    d_best = n
    any_found = False

    def _remaining():
        return max(0, total_timeout - (time.monotonic() - t_start))

    for i in range(num_logicals):
        remaining = _remaining()
        if remaining <= 0:
            all_solved = False
            break
        timeout = min(timeout_per_logical, remaining)
        w, optimal = ilp_min_weight_symplectic(stab_matrix, logicals[i],
                                                timeout=timeout)
        logicals_checked += 1
        if w is not None:
            d_best = min(d_best, w)
            any_found = True
            if optimal:
                logicals_optimal += 1
            else:
                logicals_incumbent += 1
                all_solved = False
            if verbose:
                tag = "" if optimal else " (incumbent)"
                logger.info("L[%d]: d=%d%s (%.1fs)", i, w, tag,
                            time.monotonic() - t_start)
        else:
            all_solved = False
            if verbose:
                logger.info("L[%d]: no solution (%.1fs)", i,
                            time.monotonic() - t_start)
        if early_stop is not None and d_best <= early_stop:
            break

    elapsed = time.monotonic() - t_start

    # Every objective must be solved to optimality for d to be exact; an
    # early stop leaves unchecked logicals whose minimum could be smaller.
    if logicals_checked < num_logicals:
        all_solved = False

    if not any_found:
        # No logical was solved.  A timeout says nothing about whether small
        # logicals exist (the solver may simply not have reached them), so no
        # lower bound follows.  Report d = n (vacuous upper bound) so the
        # standard ``milp_worked = d_milp < n`` check at call sites correctly
        # rejects the result.
        return n, {
            "k": k,
            "exact": False,
            "num_logicals_checked": logicals_checked,
            "logicals_optimal": logicals_optimal,
            "logicals_incumbent": logicals_incumbent,
            "total_logicals": num_logicals,
            "time_s": elapsed,
            "timeout_per_logical": timeout_per_logical,
            "all_timeout": True,
            "d_is_lower_bound": False,
        }

    return d_best, {
        "k": k,
        "exact": all_solved,
        "num_logicals_checked": logicals_checked,
        "logicals_optimal": logicals_optimal,
        "logicals_incumbent": logicals_incumbent,
        "total_logicals": num_logicals,
        "time_s": elapsed,
        "timeout_per_logical": timeout_per_logical,
    }
