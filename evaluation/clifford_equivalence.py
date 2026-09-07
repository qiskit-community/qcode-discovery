"""Decide whether a stabilizer generating set becomes CSS under
single-qubit Hadamards.

Given a stabilizer code with generators g_1, ..., g_M acting on n qubits,
this module decides whether there exists a subset J subset {1,...,n}
such that conjugating each generator by H_J = tensor_{j in J} H_j produces
a generating set in which every g_r is either pure-X (no Z factors) or
pure-Z (no X factors).  The decision runs in time linear in the support
size of the symplectic matrix and uses only elementary stabilizer-formalism
facts; it does not rely on the mirror-code theorems of Khesin & Lu
(arXiv:2603.05496) even though those theorems were the original
inspiration for the parity-2-coloring formulation.

Derivation
----------
Write each Pauli factor as a pair (x, z) in F_2^2:

    I = (0,0)   X = (1,0)   Z = (0,1)   Y = (1,1)  (mod global phase).

Conjugation by H on qubit j swaps coordinates: (x_j, z_j) -> (z_j, x_j).
H sends Y to -Y, so the symplectic representation of Y is invariant.

Let s_j in {0,1} indicate j in J, and let c_r in {0,1} encode the desired
post-conjugation type of g_r (c_r = 1 = pure-X, c_r = 0 = pure-Z).

(a) **Y obstruction.**  If any g_r has Y at qubit j (both x_{rj} = 1 and
    z_{rj} = 1), conjugation by any H_J leaves both coordinates set, so
    g_r remains mixed.  No J can fix this -- it is independent of c_r.

(b) **No Y, parity 2-coloring.**  Suppose no generator has Y support.  At
    each qubit j on which g_r acts non-trivially the local Pauli is X or
    Z; let t_{rj} = 1 for X and 0 for Z.  Because there is no Y on
    (r, j), the Z-bit equals 1 - t_{rj}.  After conjugation,

        post-X-bit at (r, j) = t_{rj} XOR s_j.

    Demanding post-X-bit = c_r at every j in supp(g_r) gives

        s_j = c_r XOR t_{rj}      (*)

    For two generators r_1, r_2 with j in supp(r_1) cap supp(r_2) the
    right-hand sides must agree:

        c_{r_1} XOR c_{r_2} = t_{r_1, j} XOR t_{r_2, j}.

    Equal Pauli types at j force equal colors; opposite types force
    different colors.  These parity constraints are a 2-coloring problem
    on a graph with same-color and must-differ edges, decidable in linear
    time by union-find with parity.  When a consistent c exists, (*)
    recovers a valid J.  The supplied generators are H-CSS-reducible iff
    the constraint graph is consistent.

Scope and caveats
-----------------
* "Equivalently CSS" here means **CSS at the level of the supplied
  generators** under single-qubit Hadamards only.  No S, S^dagger, or
  other Cliffords are considered; broader notions of local Clifford
  equivalence are handled by the helpers below.
* The decision is sound -- a True result exhibits an explicit J -- but is
  not complete at the *group* level.  A code can be H-CSS-equivalent at
  the group level while the supplied generators have Y support and
  trigger the Y-obstruction here, because Y support in a generator can
  in principle cancel against Y support in another generator under row
  reduction.  For BB and PBB stabilizers the supplied generators are the
  canonical block-circulant ones; in practice the generator-level and
  group-level questions agree on those codes, but this module does not
  prove that.
* For broader local-Clifford-to-CSS reductions, see ``verify_lc_bruteforce``
  (brute force over the 36 uniform per-block Clifford patterns of
  {I, S, H, HS, SH, HSH}^2), ``verify_uniform_reduction_exact`` (exact
  GF(2) solve covering all non-uniform {I, S} and {H, HS} patterns), and
  ``verify_not_lc_css`` (combined check).  Non-uniform {SH, HSH} patterns
  and patterns mixing macro-classes across qubits are not exhaustively
  covered.

Usage::

    from evaluation.clifford_equivalence import is_equivalently_css
    result = is_equivalently_css(code)
    print(result["is_css"], result["reason"])
"""

from __future__ import annotations

import numpy as np
from qldpc.codes import QuditCode


def is_equivalently_css(code: QuditCode) -> dict:
    """Decide whether the supplied generators become pure-X / pure-Z under H_J.

    Solves the parity-2-coloring problem described in the module docstring:
    finds a subset J of qubits and a 0/1 target type for each generator
    such that conjugation by H_J turns every supplied generator into a
    pure-X or pure-Z operator, or proves no such J exists.

    A True result is sound -- the returned coloring c, together with
    s_j = c_r XOR t_{rj} for any r acting on qubit j, gives an explicit
    J.  A False result rules out any single-qubit-Hadamard reduction
    *of the supplied generators*; the stabilizer group may still be
    H-CSS-equivalent under a different (row-reduced) generating set --
    see "Scope and caveats" in the module docstring.

    Args:
        code: A qubit stabilizer code (qldpc's ``QuditCode``; over GF(2)).

    Returns:
        Dict with keys:
        - is_css (bool): True iff a consistent 2-coloring exists.
        - reason (str): Human-readable explanation.
        - num_y_qubits (int): Total number of (r, j) pairs with Y support.
        - coloring (list[int] | None): For each generator, 0 = target
          pure-Z, 1 = target pure-X; ``None`` when ``is_css`` is False.
    """
    stab = np.array(code.matrix, dtype=int) % 2
    num_stabs, two_n = stab.shape
    n = two_n // 2

    S_x = stab[:, :n]  # X-parts
    S_z = stab[:, n:]   # Z-parts

    # Step 1: Check for Y support (both X and Z on same qubit)
    # Y_support[r, j] = S_x[r,j] AND S_z[r,j]
    Y_support = S_x & S_z
    y_qubits_per_stab = np.sum(Y_support, axis=1)
    total_y = int(np.sum(Y_support > 0))

    if np.any(y_qubits_per_stab > 0):
        # Stabilizers with Y support cannot be made pure-X or pure-Z
        # by single-qubit Hadamards alone (H swaps X↔Z but Y→-Y,
        # which doesn't help).
        num_y_stabs = int(np.sum(y_qubits_per_stab > 0))
        return {
            "is_css": False,
            "reason": f"{num_y_stabs}/{num_stabs} stabilizers have Y support",
            "num_y_qubits": total_y,
            "coloring": None,
        }

    # Step 2: Build constraint graph for 2-coloring
    # For each qubit j, determine which stabilizers act on it and with which Pauli.
    # Since no Y support exists, each stabilizer acts with X only or Z only on each qubit.
    #
    # Constraint: if stabilizers r1 and r2 both act on qubit j,
    # - Same type (both X or both Z): they must have SAME color
    # - Different type (one X, one Z): they must have DIFFERENT color
    #
    # We use Union-Find with parity to solve this efficiently.

    parent = list(range(num_stabs))
    rank = [0] * num_stabs
    parity = [0] * num_stabs  # parity[i] = XOR distance from i to root

    def find(x):
        if parent[x] != x:
            root = find(parent[x])
            parity[x] ^= parity[parent[x]]
            parent[x] = root
        return parent[x]

    def union(x, y, must_differ):
        """Union x and y with constraint: must_differ=True means different colors."""
        rx, ry = find(x), find(y)
        px, py = parity[x], parity[y]

        if rx == ry:
            # Check consistency
            expected_parity = px ^ py
            if must_differ:
                return expected_parity == 1  # OK if they differ
            else:
                return expected_parity == 0  # OK if they agree
        else:
            # Merge
            target_parity = px ^ py ^ (1 if must_differ else 0)
            if rank[rx] < rank[ry]:
                parent[rx] = ry
                parity[rx] = target_parity
            elif rank[rx] > rank[ry]:
                parent[ry] = rx
                parity[ry] = target_parity
            else:
                parent[ry] = rx
                parity[ry] = target_parity
                rank[rx] += 1
            return True

    # Process qubit-by-qubit constraints
    for j in range(n):
        # Find all stabilizers acting on qubit j
        x_stabs = np.where(S_x[:, j] > 0)[0]
        z_stabs = np.where(S_z[:, j] > 0)[0]

        # Same-type pairs must have SAME color
        for i in range(len(x_stabs)):
            for k in range(i + 1, len(x_stabs)):
                if not union(x_stabs[i], x_stabs[k], must_differ=False):
                    return {
                        "is_css": False,
                        "reason": f"Constraint conflict at qubit {j} (X-X)",
                        "num_y_qubits": 0,
                        "coloring": None,
                    }
        for i in range(len(z_stabs)):
            for k in range(i + 1, len(z_stabs)):
                if not union(z_stabs[i], z_stabs[k], must_differ=False):
                    return {
                        "is_css": False,
                        "reason": f"Constraint conflict at qubit {j} (Z-Z)",
                        "num_y_qubits": 0,
                        "coloring": None,
                    }

        # Cross-type pairs must have DIFFERENT color
        for xi in x_stabs:
            for zi in z_stabs:
                if not union(xi, zi, must_differ=True):
                    return {
                        "is_css": False,
                        "reason": f"Constraint conflict at qubit {j} (X-Z cross)",
                        "num_y_qubits": 0,
                        "coloring": None,
                    }

    # Extract coloring
    coloring = [0] * num_stabs
    for i in range(num_stabs):
        find(i)
        coloring[i] = parity[i]

    return {
        "is_css": True,
        "reason": "2-colorable -- equivalently CSS via Hadamard conjugation",
        "num_y_qubits": 0,
        "coloring": coloring,
    }


# --- Local Clifford equivalence (paper Appendix E) ---
#
# All functions below use the paper convention: block-1 stabilizer Z-part
# is [C | D] (C left, D right), matching H = (A B C D ; 0 0 B^T A^T).
# They accept (A_terms, B_terms, C_terms, D_terms) as stored in the JSONL
# catalog and as returned by ``build_pbb_code``.


def _gf2_rank(mat: np.ndarray) -> int:
    """Compute rank of a binary matrix over GF(2) via row reduction.

    Elimination is vectorized over rows (boolean-mask XOR) rather than a
    Python-level loop -- this is the dominant cost of the non-CSS gate
    (called ~100x per candidate via ``verify_lc_bruteforce``), so the
    per-row Python loop dominated wall-clock at large lattice sizes.
    """
    M = mat.copy() % 2
    rows, cols = M.shape
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
        mask = M[:, c].astype(bool)
        mask[r] = False
        if mask.any():
            M[mask] ^= M[r]
        r += 1
    return r


def _is_css_group(stab: np.ndarray) -> bool:
    """Check if stabilizer GROUP is CSS (allows generator recombination).

    Unlike _is_css_matrix (which checks each generator individually), this
    checks if the group decomposes into pure-X and pure-Z subgroups:
    rank([X|Z]) = rank(X) + rank(Z).
    """
    n = stab.shape[1] // 2
    X = stab[:, :n]
    Z = stab[:, n:]
    return _gf2_rank(stab) == _gf2_rank(X) + _gf2_rank(Z)


_SINGLE_QUBIT_CLIFFORD_COLUMNS = (
    ("I", 0, 1),
    ("S", 0, 2),
    ("H", 1, 0),
    ("HS", 2, 0),
    ("SH", 1, 2),
    ("HSH", 2, 1),
)


def _independent_gf2_rows(matrix: np.ndarray) -> np.ndarray:
    """Return a deterministic independent basis for a binary row space."""
    reduced = np.asarray(matrix, dtype=np.uint8).copy() % 2
    row_count, column_count = reduced.shape
    pivot_row = 0
    for column in range(column_count):
        if pivot_row == row_count:
            break
        candidates = np.flatnonzero(reduced[pivot_row:, column])
        if candidates.size == 0:
            continue
        selected = pivot_row + int(candidates[0])
        if selected != pivot_row:
            reduced[[pivot_row, selected]] = reduced[[selected, pivot_row]]
        mask = reduced[:, column].astype(bool)
        mask[pivot_row] = False
        reduced[mask] ^= reduced[pivot_row]
        pivot_row += 1
    return reduced[:pivot_row]


def _binary_column_as_int(column: np.ndarray) -> int:
    """Pack a short binary column into a Python integer."""
    packed = 0
    for bit, value in enumerate(column):
        if int(value) & 1:
            packed |= 1 << bit
    return packed


def _insert_xor_basis(basis: tuple[int, ...], vector: int) -> tuple[int, ...]:
    """Insert ``vector`` into a canonical reduced XOR basis."""
    reduced = vector
    for row in basis:
        reduced = min(reduced, reduced ^ row)
    if reduced == 0:
        return basis
    updated = [min(row, row ^ reduced) for row in basis]
    updated.append(reduced)
    updated.sort(reverse=True)
    return tuple(updated)


def exact_local_clifford_css_search(stab: np.ndarray) -> dict:
    """Decide full single-qubit-LC equivalence to CSS for a small code.

    The input is any binary symplectic generator matrix ``[X | Z]``.  For
    an independent rank-``r`` row basis, a transformed stabilizer group is
    CSS exactly when

    ``rank(X_transformed) + rank(Z_transformed) == r``.

    On each qubit a Clifford modulo phase maps the ordered ``(X,Z)`` column
    pair to one of the six ordered pairs of distinct vectors drawn from
    ``X``, ``Z``, and ``X+Z``.  The search below exhausts those choices.  It
    prunes a branch as soon as the two monotonically increasing column-space
    ranks sum to more than ``r`` and memoizes every exhausted rank-state.
    This is intended for the small component codes highlighted in the paper,
    not for hundreds of qubits.

    A negative result is exhaustive over all ``6**n`` local Clifford
    assignments, including mixed macro-classes and arbitrary nonuniform
    ``{SH,HSH}`` choices.  A positive result includes an explicit gate on
    every qubit and is rechecked directly against the group-level CSS rank
    criterion before return.
    """
    raw = np.asarray(stab)
    if raw.ndim != 2 or raw.shape[1] % 2:
        raise ValueError("stab must be a two-dimensional [X | Z] matrix")
    if np.any((raw != 0) & (raw != 1)):
        raise ValueError("stab must be binary")
    matrix = raw.astype(np.uint8)

    basis_matrix = _independent_gf2_rows(matrix)
    rank = int(basis_matrix.shape[0])
    num_qubits = matrix.shape[1] // 2
    if rank == 0:
        return {
            "is_lc_css": True,
            "complete": True,
            "num_qubits": num_qubits,
            "stabilizer_rank": 0,
            "assignments_covered": f"6^{num_qubits}",
            "gate_assignment": ["I"] * num_qubits,
            "search_nodes": 1,
            "memoized_unsat_states": 0,
            "memo_hits": 0,
            "rank_prunes": 0,
        }

    x_columns = [
        _binary_column_as_int(basis_matrix[:, qubit])
        for qubit in range(num_qubits)
    ]
    z_columns = [
        _binary_column_as_int(basis_matrix[:, num_qubits + qubit])
        for qubit in range(num_qubits)
    ]

    choices_by_qubit: list[list[tuple[int, int, str]]] = []
    for x_column, z_column in zip(x_columns, z_columns):
        axes = (x_column, z_column, x_column ^ z_column)
        # Degenerate local projections can make different gates act
        # identically.  Keeping one deterministic representative preserves
        # exhaustive coverage while reducing the search tree.
        unique: dict[tuple[int, int], str] = {}
        for gate, x_axis, z_axis in _SINGLE_QUBIT_CLIFFORD_COLUMNS:
            unique.setdefault((axes[x_axis], axes[z_axis]), gate)
        choices_by_qubit.append(
            [(new_x, new_z, gate) for (new_x, new_z), gate in unique.items()]
        )

    order = sorted(
        range(num_qubits),
        key=lambda qubit: (
            len(choices_by_qubit[qubit]),
            sum(
                int(new_x != 0) + int(new_z != 0)
                for new_x, new_z, _gate in choices_by_qubit[qubit]
            ),
            -qubit,
        ),
        reverse=True,
    )
    memoized_unsat: set[
        tuple[int, tuple[int, ...], tuple[int, ...]]
    ] = set()
    stats = {"nodes": 0, "memo_hits": 0, "rank_prunes": 0}

    def search(
        depth: int,
        x_basis: tuple[int, ...],
        z_basis: tuple[int, ...],
        partial_assignment: list[tuple[int, str]],
    ) -> list[tuple[int, str]] | None:
        stats["nodes"] += 1
        if len(x_basis) + len(z_basis) > rank:
            stats["rank_prunes"] += 1
            return None
        if depth == num_qubits:
            if len(x_basis) + len(z_basis) == rank:
                return partial_assignment
            return None

        state = (depth, x_basis, z_basis)
        if state in memoized_unsat:
            stats["memo_hits"] += 1
            return None

        qubit = order[depth]
        next_states = []
        for new_x, new_z, gate in choices_by_qubit[qubit]:
            next_x_basis = _insert_xor_basis(x_basis, new_x)
            next_z_basis = _insert_xor_basis(z_basis, new_z)
            next_states.append(
                (
                    len(next_x_basis) + len(next_z_basis),
                    gate,
                    next_x_basis,
                    next_z_basis,
                )
            )
        # Trying the smallest partial rank first finds positive witnesses
        # quickly without changing the exhaustive negative result.
        for _rank_sum, gate, next_x_basis, next_z_basis in sorted(next_states):
            witness = search(
                depth + 1,
                next_x_basis,
                next_z_basis,
                [*partial_assignment, (qubit, gate)],
            )
            if witness is not None:
                return witness

        memoized_unsat.add(state)
        return None

    witness = search(0, (), (), [])
    gate_assignment = None
    if witness is not None:
        gate_assignment = ["I"] * num_qubits
        for qubit, gate in witness:
            gate_assignment[qubit] = gate

        transformed = basis_matrix.copy()
        transformed_x = transformed[:, :num_qubits]
        transformed_z = transformed[:, num_qubits:]
        for qubit, gate in enumerate(gate_assignment):
            new_x, new_z = _apply_clifford_to_block(
                transformed_x[:, qubit : qubit + 1],
                transformed_z[:, qubit : qubit + 1],
                gate,
            )
            transformed_x[:, qubit] = new_x[:, 0]
            transformed_z[:, qubit] = new_z[:, 0]
        if not _is_css_group(transformed):
            raise AssertionError("internal LC witness failed direct CSS check")

    return {
        "is_lc_css": witness is not None,
        "complete": True,
        "num_qubits": num_qubits,
        "stabilizer_rank": rank,
        "assignments_covered": f"6^{num_qubits}",
        "gate_assignment": gate_assignment,
        "search_nodes": stats["nodes"],
        "memoized_unsat_states": len(memoized_unsat),
        "memo_hits": stats["memo_hits"],
        "rank_prunes": stats["rank_prunes"],
    }


def is_lc_equivalent_css(
    A_terms: list[tuple[int, int]],
    B_terms: list[tuple[int, int]],
    C_terms: list[tuple[int, int]] | None,
    D_terms: list[tuple[int, int]] | None,
) -> dict:
    """Check if a PBB code is LC-equivalent to CSS (sufficient conditions).

    For PBB codes with circulant polynomial matrices over
    R = F_2[x,y]/(x^ell-1, y^m-1), uniform per-block S-gate patterns
    give four sufficient algebraic conditions.

    Paper convention (Z-part of block-1 = [C | D]):
      Block-1 qubit Z-source = C_terms
      Block-2 qubit Z-source = D_terms

    The four conditions (S-gate patterns that yield CSS):
      0. Already CSS:       C = 0 and D = 0
      1. S on block-2 only: C = 0 and D = B
      2. S on block-1 only: D = 0 and C = A
      3. S on both blocks:  C = A and D = B

    Returns dict with ``is_lc_css``, ``condition``, ``gate_assignment``.
    """
    A = frozenset(tuple(t) for t in A_terms)
    B = frozenset(tuple(t) for t in B_terms)
    C = frozenset(tuple(t) for t in C_terms) if C_terms else frozenset()
    D = frozenset(tuple(t) for t in D_terms) if D_terms else frozenset()
    c_zero = len(C) == 0
    d_zero = len(D) == 0

    if d_zero and c_zero:
        return {"is_lc_css": True, "condition": "C=D=0 (already CSS)",
                "gate_assignment": "identity"}

    if c_zero and D == B:
        return {"is_lc_css": True, "condition": "C=0, D=B",
                "gate_assignment": "S on block-2 only"}

    if d_zero and C == A:
        return {"is_lc_css": True, "condition": "D=0, C=A",
                "gate_assignment": "S on block-1 only"}

    if C == A and D == B:
        return {"is_lc_css": True, "condition": "C=A, D=B",
                "gate_assignment": "S on both blocks"}

    return {"is_lc_css": False, "condition": None, "gate_assignment": None}


_S_GATE_LABELS = {
    (0, 0): ("identity (group-level)", "none (row ops only)"),
    (1, 0): ("S on block-1 (group-level)", "S on block-1"),
    (0, 1): ("S on block-2 (group-level)", "S on block-2"),
    (1, 1): ("S on both blocks (group-level)", "S on both blocks"),
}


def is_lc_equivalent_css_group(
    ell: int,
    m: int,
    A_terms: list[tuple[int, int]],
    B_terms: list[tuple[int, int]],
    C_terms: list[tuple[int, int]] | None,
    D_terms: list[tuple[int, int]] | None,
) -> dict:
    """Generalized LC equivalence check (group CSS, Theorem thm:lc_equiv).

    Checks whether the stabilizer GROUP (not just generators) becomes CSS
    after uniform per-block S-gate application.  This is the rank condition:

        rowspan([C + s1*A | D + s2*B]) ⊆ rowspan([B^T | A^T])

    for some (s1, s2) in {0,1}^2.  Strictly more general than
    ``is_lc_equivalent_css()`` which only covers the p=0 special cases.

    Returns dict with ``is_lc_css``, ``s1``, ``s2``, ``condition``,
    ``gate_assignment``.
    """
    from evaluation.bb_code import terms_to_poly
    from evaluation.pbb_code import _poly_to_matrix
    from sympy.abc import x, y
    from qldpc import codes

    if (not C_terms) and (not D_terms):
        return {
            "is_lc_css": True, "s1": 0, "s2": 0,
            "condition": "C=D=0 (already CSS)",
            "gate_assignment": "identity",
        }

    dim = ell * m
    poly_a = terms_to_poly(A_terms)
    poly_b = terms_to_poly(B_terms)
    bb = codes.BBCode({x: ell, y: m}, poly_a, poly_b)

    mat_A = _poly_to_matrix(bb, A_terms)
    mat_B = _poly_to_matrix(bb, B_terms)
    mat_C = (_poly_to_matrix(bb, C_terms)
             if C_terms else np.zeros((dim, dim), dtype=int))
    mat_D = (_poly_to_matrix(bb, D_terms)
             if D_terms else np.zeros((dim, dim), dtype=int))

    block2_z = np.hstack([mat_B.T % 2, mat_A.T % 2])
    rank_b2 = _gf2_rank(block2_z)

    for s1 in (0, 1):
        for s2 in (0, 1):
            modified_z = np.hstack([
                (mat_C + s1 * mat_A) % 2,
                (mat_D + s2 * mat_B) % 2,
            ])
            combined = np.vstack([modified_z, block2_z]) % 2
            if _gf2_rank(combined) == rank_b2:
                label, gate = _S_GATE_LABELS[(s1, s2)]
                return {
                    "is_lc_css": True,
                    "s1": s1,
                    "s2": s2,
                    "condition": label,
                    "gate_assignment": gate,
                }

    return {
        "is_lc_css": False, "s1": None, "s2": None,
        "condition": None, "gate_assignment": None,
    }


def _apply_clifford_to_block(
    X_block: np.ndarray, Z_block: np.ndarray, gate: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a uniform Clifford gate to a qubit block.

    Valid gates: "I", "S", "H", "HS", "SH", "HSH" -- all six
    single-qubit Cliffords modulo phases.

    Args:
        X_block, Z_block: (num_stabs, block_dim) binary matrices.
        gate: Clifford gate name.

    Returns:
        (new_X_block, new_Z_block).
    """
    if gate == "I":
        return X_block.copy(), Z_block.copy()
    if gate == "S":
        return X_block.copy(), (X_block + Z_block) % 2
    if gate == "H":
        return Z_block.copy(), X_block.copy()
    if gate == "HS":
        return (X_block + Z_block) % 2, X_block.copy()
    if gate == "SH":
        return Z_block.copy(), (X_block + Z_block) % 2
    if gate == "HSH":
        return (X_block + Z_block) % 2, Z_block.copy()
    raise ValueError(f"Gate must be I, S, H, HS, SH, or HSH; got {gate!r}")


def _is_css_matrix(stab: np.ndarray) -> tuple[bool, int, int, int]:
    """Check whether every row of a symplectic matrix is pure-X or pure-Z.

    Returns ``(is_css, pure_x_count, pure_z_count, mixed_count)``.
    """
    n = stab.shape[1] // 2
    X = stab[:, :n]
    Z = stab[:, n:]
    x_any = np.any(X, axis=1)
    z_any = np.any(Z, axis=1)
    pure_x = int(np.sum(x_any & ~z_any))
    pure_z = int(np.sum(~x_any & z_any))
    mixed = int(np.sum(x_any & z_any))
    return mixed == 0, pure_x, pure_z, mixed


_MACRO_PURE_Z = ("I", "S")
_MACRO_PURE_X = ("H", "HS")
_MACRO_Y = ("SH", "HSH")


def _macro_class(g: str) -> str:
    if g in _MACRO_PURE_Z: return "Z"
    if g in _MACRO_PURE_X: return "X"
    return "Y"


def verify_lc_bruteforce(
    ell: int,
    m: int,
    A_terms: list[tuple[int, int]],
    B_terms: list[tuple[int, int]],
    C_terms: list[tuple[int, int]] | None,
    D_terms: list[tuple[int, int]] | None,
) -> dict:
    """Verify LC equivalence by testing all 36 uniform Clifford assignments.

    Constructs the PBB stabilizer matrix independently and applies each
    of the 36 uniform per-block assignments (6 gates x 6 gates), covering
    all six single-qubit Cliffords modulo phases.

    Checks both generator-level CSS (_is_css_matrix) and group-level CSS
    (_is_css_group).  The group check is the authoritative criterion
    matching ``is_lc_equivalent_css_group()``.

    Returns dict with ``css_assignments`` (generator), ``group_css_assignments``,
    ``mixed_macro_group_css`` (mixed-macro hits, expected empty),
    ``all_results``, ``algebraic_result``, ``group_result``, and ``consistent`` flag.
    """
    from evaluation.bb_code import terms_to_poly
    from evaluation.pbb_code import _poly_to_matrix
    from sympy.abc import x, y
    from qldpc import codes

    dim = ell * m
    n = 2 * dim

    poly_a = terms_to_poly(A_terms)
    poly_b = terms_to_poly(B_terms)
    bb = codes.BBCode({x: ell, y: m}, poly_a, poly_b)

    mat_A = _poly_to_matrix(bb, A_terms)
    mat_B = _poly_to_matrix(bb, B_terms)
    mat_C = (_poly_to_matrix(bb, C_terms)
             if C_terms else np.zeros((dim, dim), dtype=int))
    mat_D = (_poly_to_matrix(bb, D_terms)
             if D_terms else np.zeros((dim, dim), dtype=int))

    zero = np.zeros((dim, dim), dtype=int)

    block1_x = np.hstack([mat_A, mat_B])
    block1_z = np.hstack([mat_C, mat_D])
    block2_x = np.hstack([zero, zero])
    block2_z = np.hstack([mat_B.T % 2, mat_A.T % 2])

    stab = np.vstack([
        np.hstack([block1_x, block1_z]),
        np.hstack([block2_x, block2_z]),
    ]) % 2

    css_assignments: list[str] = []
    group_css_assignments: list[str] = []
    mixed_macro_group_css: list[str] = []
    all_results: dict[str, dict] = {}

    all_gates = ("I", "S", "H", "HS", "SH", "HSH")
    for g1 in all_gates:
        for g2 in all_gates:
            transformed = stab.copy()
            X = transformed[:, :n]
            Z = transformed[:, n:]
            new_x1, new_z1 = _apply_clifford_to_block(
                X[:, :dim], Z[:, :dim], g1)
            new_x2, new_z2 = _apply_clifford_to_block(
                X[:, dim:], Z[:, dim:], g2)
            X[:, :dim] = new_x1
            Z[:, :dim] = new_z1
            X[:, dim:] = new_x2
            Z[:, dim:] = new_z2

            is_css, px, pz, mx = _is_css_matrix(transformed)
            is_group_css = _is_css_group(transformed)
            key = f"{g1},{g2}"
            is_mixed = _macro_class(g1) != _macro_class(g2)
            all_results[key] = {
                "is_css": is_css, "is_group_css": is_group_css,
                "pure_x": px, "pure_z": pz, "mixed": mx,
                "is_mixed_macro": is_mixed,
            }
            if is_css:
                css_assignments.append(key)
            if is_group_css:
                group_css_assignments.append(key)
                if is_mixed:
                    mixed_macro_group_css.append(key)

    algebraic = is_lc_equivalent_css(A_terms, B_terms, C_terms, D_terms)
    group = is_lc_equivalent_css_group(ell, m, A_terms, B_terms, C_terms, D_terms)

    return {
        "css_assignments": css_assignments,
        "group_css_assignments": group_css_assignments,
        "mixed_macro_group_css": mixed_macro_group_css,
        "all_results": all_results,
        "algebraic_result": algebraic,
        "group_result": group,
        "consistent": bool(group_css_assignments) == group["is_lc_css"],
    }


def _solve_column_cancel(
    target: np.ndarray, source: np.ndarray,
) -> tuple[bool, np.ndarray | None]:
    """Solve ``target[:,j] = s_j * source[:,j]`` over F_2 for each column j.

    This checks whether a per-qubit S-pattern can cancel the target using
    the source.  For circulant matrices, only uniform patterns (all-0 or
    all-1) are ever feasible.

    Returns ``(feasible, pattern)`` where ``pattern[j]`` is 0 or 1.
    """
    dim = target.shape[1]
    pattern = np.zeros(dim, dtype=int)

    for j in range(dim):
        tgt = target[:, j]
        src = source[:, j]

        if not np.any(tgt):
            pattern[j] = 0
        elif np.array_equal(tgt, src):
            pattern[j] = 1
        else:
            return False, None

    return True, pattern


def check_lc_pattern_feasibility(
    ell: int,
    m: int,
    A_terms: list[tuple[int, int]],
    B_terms: list[tuple[int, int]],
    C_terms: list[tuple[int, int]] | None,
    D_terms: list[tuple[int, int]] | None,
) -> dict:
    """Check per-qubit S-pattern feasibility for LC equivalence.

    Directly solves the column-wise feasibility problem: for each qubit j,
    can we choose s_j in {0,1} such that the Z-part of block-1 stabilizers
    vanishes?  For circulant matrices, the only feasible patterns are
    uniform (all-0 or all-1), confirming the theorem's algebraic reduction.

    Two macro-classes are checked (paper convention, Z-part = [C | D]):
      {I,S}: block-1 stabs become pure-X when Z-part = C + A*diag(s1) = 0
             and D + B*diag(s2) = 0
      {H,HS}: block-1 stabs become pure-Z when A + C*diag(h1) = 0
              and B + D*diag(h2) = 0

    Returns dict with feasibility and uniformity data per macro-class.
    """
    from evaluation.bb_code import terms_to_poly
    from evaluation.pbb_code import _poly_to_matrix
    from sympy.abc import x, y
    from qldpc import codes

    dim = ell * m
    poly_a = terms_to_poly(A_terms)
    poly_b = terms_to_poly(B_terms)
    bb = codes.BBCode({x: ell, y: m}, poly_a, poly_b)

    mat_A = _poly_to_matrix(bb, A_terms)
    mat_B = _poly_to_matrix(bb, B_terms)
    mat_C = (_poly_to_matrix(bb, C_terms)
             if C_terms else np.zeros((dim, dim), dtype=int))
    mat_D = (_poly_to_matrix(bb, D_terms)
             if D_terms else np.zeros((dim, dim), dtype=int))

    results = {}

    # {I,S}: need A*diag(s1) = C on block-1 qubits, B*diag(s2) = D on block-2 qubits
    feas_b1, pat_b1 = _solve_column_cancel(mat_C, mat_A)
    feas_b2, pat_b2 = _solve_column_cancel(mat_D, mat_B)
    uniform_b1 = pat_b1 is not None and len(set(pat_b1.tolist())) <= 1
    uniform_b2 = pat_b2 is not None and len(set(pat_b2.tolist())) <= 1
    results["IS_class"] = {
        "feasible": feas_b1 and feas_b2,
        "block1_feasible": feas_b1,
        "block2_feasible": feas_b2,
        "block1_uniform": uniform_b1,
        "block2_uniform": uniform_b2,
    }

    # {H,HS}: need C*diag(h1) = A on block-1 qubits, D*diag(h2) = B on block-2 qubits
    feas_b1h, pat_b1h = _solve_column_cancel(mat_A, mat_C)
    feas_b2h, pat_b2h = _solve_column_cancel(mat_B, mat_D)
    uniform_b1h = pat_b1h is not None and len(set(pat_b1h.tolist())) <= 1
    uniform_b2h = pat_b2h is not None and len(set(pat_b2h.tolist())) <= 1
    results["HHS_class"] = {
        "feasible": feas_b1h and feas_b2h,
        "block1_feasible": feas_b1h,
        "block2_feasible": feas_b2h,
        "block1_uniform": uniform_b1h,
        "block2_uniform": uniform_b2h,
    }

    any_feasible = (results["IS_class"]["feasible"]
                    or results["HHS_class"]["feasible"])
    all_uniform = True
    for cls_data in results.values():
        if cls_data["feasible"]:
            if not (cls_data["block1_uniform"] and cls_data["block2_uniform"]):
                all_uniform = False

    results["any_feasible"] = any_feasible
    results["all_patterns_uniform"] = all_uniform or not any_feasible

    return results


def verify_uniform_reduction_exact(
    ell: int,
    m: int,
    A_terms: list[tuple[int, int]],
    B_terms: list[tuple[int, int]],
    C_terms: list[tuple[int, int]] | None,
    D_terms: list[tuple[int, int]] | None,
) -> dict:
    """Exactly verify that non-uniform S-patterns cannot create new LC-CSS
    equivalences beyond what uniform patterns achieve.

    Constructs the full GF(2) linear system from the correct per-qubit
    S-gate action (right multiplication ``A * diag(s1)``).

    For each null-vector w of L = rowspan([B^T | A^T]) and each stabilizer
    row g, the constraint w . row_g([C + A*diag(s1) | D + B*diag(s2)]) = 0
    is affine-linear in (s1, s2) in F_2^{2N}.  We check consistency and
    whether uniform solutions cover all feasible cases.

    The {H, HS} macro-class yields the identical constraint system (the
    X-part [C + A*diag(h1) | D + B*diag(h2)] must lie in the same L),
    so one solve covers both macro-classes.

    Returns dict with:
    - ``reduction_holds``: True if consistent → uniform solution exists
    - ``consistent``: whether any (possibly non-uniform) solution exists
    - ``uniform_solutions``: list of (s1, s2) uniform pairs that work
    - ``constraint_rank``: rank of the constraint system
    - ``num_vars``: 2*N (total variables)
    """
    from evaluation.bb_code import terms_to_poly
    from evaluation.pbb_code import _poly_to_matrix, _gf2_nullspace
    from sympy.abc import x, y
    from qldpc import codes

    dim = ell * m

    poly_a = terms_to_poly(A_terms)
    poly_b = terms_to_poly(B_terms)
    bb = codes.BBCode({x: ell, y: m}, poly_a, poly_b)

    mat_A = _poly_to_matrix(bb, A_terms)
    mat_B = _poly_to_matrix(bb, B_terms)
    mat_C = (_poly_to_matrix(bb, C_terms)
             if C_terms else np.zeros((dim, dim), dtype=int))
    mat_D = (_poly_to_matrix(bb, D_terms)
             if D_terms else np.zeros((dim, dim), dtype=int))

    L = np.hstack([mat_B.T % 2, mat_A.T % 2])
    L_perp = _gf2_nullspace(L)
    num_w = L_perp.shape[0]

    if num_w == 0:
        return {
            "reduction_holds": True, "consistent": True,
            "uniform_solutions": [(0, 0), (1, 0), (0, 1), (1, 1)],
            "constraint_rank": 0, "num_vars": 2 * dim,
        }

    rows_list = []
    rhs_list = []

    for wi in range(num_w):
        w = L_perp[wi]
        w1, w2 = w[:dim], w[dim:]
        for g in range(dim):
            a_row = mat_A[g] % 2
            b_row = mat_B[g] % 2
            c_row = mat_C[g] % 2
            d_row = mat_D[g] % 2

            const = (np.dot(w1, c_row) + np.dot(w2, d_row)) % 2

            coeff = np.zeros(2 * dim, dtype=int)
            coeff[:dim] = (w1 * a_row) % 2
            coeff[dim:] = (w2 * b_row) % 2

            rows_list.append(coeff)
            rhs_list.append(const)

    constraint_mat = np.array(rows_list, dtype=int) % 2
    rhs_vec = np.array(rhs_list, dtype=int) % 2

    uniform_solutions = []
    for s1_val in (0, 1):
        for s2_val in (0, 1):
            test = np.zeros(2 * dim, dtype=int)
            if s1_val:
                test[:dim] = 1
            if s2_val:
                test[dim:] = 1
            if np.all((constraint_mat @ test + rhs_vec) % 2 == 0):
                uniform_solutions.append((s1_val, s2_val))

    aug = np.hstack([constraint_mat, rhs_vec.reshape(-1, 1)]) % 2
    rank_aug = _gf2_rank(aug)
    rank_mat = _gf2_rank(constraint_mat)
    consistent = (rank_aug == rank_mat)

    reduction_holds = (not consistent) or len(uniform_solutions) > 0

    return {
        "reduction_holds": reduction_holds,
        "consistent": consistent,
        "uniform_solutions": uniform_solutions,
        "constraint_rank": rank_mat,
        "num_vars": 2 * dim,
    }


def verify_not_lc_css(
    ell: int,
    m: int,
    A_terms: list[tuple[int, int]],
    B_terms: list[tuple[int, int]],
    C_terms: list[tuple[int, int]] | None,
    D_terms: list[tuple[int, int]] | None,
) -> dict:
    """Comprehensive verification that a PBB code is not LC-equivalent to CSS.

    Combines two checks:

    1. **36 uniform assignments**: brute-force test of all 6x6 per-block
       Clifford assignments, checking group-level CSS.

    2. **{I,S}/{H,HS} non-uniform exact**: solves the full GF(2) affine
       system to check if ANY per-qubit pattern in the {I,S} macro-class
       achieves group-CSS.  The {H,HS} class yields the identical system.

    Note: non-uniform {SH,HSH} patterns are tested only at the uniform
    level (check 1).  Mixed-class assignments (different macro-classes on
    different qubits) are also not covered.

    Returns dict with:
    - ``is_lc_css``: True if any check found LC-CSS equivalence
    - ``uniform_bruteforce``: results from 36-assignment brute-force
    - ``nonuniform_exact``: results from GF(2) system solve
    """
    bf = verify_lc_bruteforce(ell, m, A_terms, B_terms, C_terms, D_terms)
    uniform_css = len(bf["group_css_assignments"]) > 0

    exact = verify_uniform_reduction_exact(
        ell, m, A_terms, B_terms, C_terms, D_terms)
    nonuniform_css = exact["consistent"]

    is_lc_css = uniform_css or nonuniform_css

    return {
        "is_lc_css": is_lc_css,
        "uniform_bruteforce": bf,
        "nonuniform_exact": exact,
    }
