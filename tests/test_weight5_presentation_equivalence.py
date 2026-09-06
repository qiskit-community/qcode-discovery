"""Regression tests for the weight-five presentation-equivalence audit."""

import numpy as np
import pytest

from evaluation.bb_code import build_bb_code
from evaluation.pbb_code import build_pbb_code
from evaluation.tanner_equivalence import (
    canonical_digest,
    canonical_digest_noncss,
    canonical_hash,
    canonical_hash_noncss,
    build_colored_tanner_graph,
    extract_full_vertex_isomorphism,
    extract_qubit_permutation,
)


def test_reviewer_pbb_sector_exchange_pair_is_equivalent():
    left = build_pbb_code(
        6, 6,
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (0, 1)],
        [(1, 0)],
    )
    right = build_pbb_code(
        6, 6,
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (0, 1)],
        [(1, 0)],
        [(0, 0), (0, 1)],
    )
    assert canonical_hash_noncss(left) == canonical_hash_noncss(right)
    assert canonical_digest_noncss(left) == canonical_digest_noncss(right)


def test_lattice_multiplier_is_captured_by_css_bliss_class():
    A = [(0, 0), (1, 1), (5, 5)]
    B = [(0, 0), (0, 1)]
    transformed_A = [((5 * a) % 6, (3 * b) % 10) for a, b in A]
    transformed_B = [((5 * a) % 6, (3 * b) % 10) for a, b in B]
    left = build_bb_code(6, 10, A, B)
    right = build_bb_code(6, 10, transformed_A, transformed_B)
    assert canonical_hash(left) == canonical_hash(right)
    assert canonical_digest(left) == canonical_digest(right)


def test_digest_retains_css_check_colors():
    class _CSSCode:
        def __init__(self, matrix_x, matrix_z):
            self.matrix_x = matrix_x
            self.matrix_z = matrix_z

    x_only = _CSSCode(np.array([[1]]), np.zeros((0, 1), dtype=int))
    z_only = _CSSCode(np.zeros((0, 1), dtype=int), np.array([[1]]))
    assert canonical_digest(x_only) != canonical_digest(z_only)
    assert extract_full_vertex_isomorphism(x_only, z_only) is None
    assert extract_qubit_permutation(x_only, z_only) is None


def test_extracted_mapping_preserves_colored_tanner_graph():
    class _CSSCode:
        def __init__(self, matrix_x, matrix_z):
            self.matrix_x = matrix_x
            self.matrix_z = matrix_z

    left = _CSSCode(
        np.array([[1, 0, 1], [0, 1, 1]], dtype=int),
        np.array([[1, 1, 0]], dtype=int),
    )
    qubit_map = [2, 0, 1]
    x_row_map = [1, 0]
    right_x = np.zeros((2, 3), dtype=int)
    right_z = np.zeros((1, 3), dtype=int)
    for row, mapped_row in enumerate(x_row_map):
        for qubit, mapped_qubit in enumerate(qubit_map):
            right_x[mapped_row, mapped_qubit] = left.matrix_x[row, qubit]
    for qubit, mapped_qubit in enumerate(qubit_map):
        right_z[0, mapped_qubit] = left.matrix_z[0, qubit]
    right = _CSSCode(right_x, right_z)

    mapping = extract_full_vertex_isomorphism(left, right)
    assert mapping is not None
    graph_left, colors_left = build_colored_tanner_graph(left)
    graph_right, colors_right = build_colored_tanner_graph(right)
    assert [colors_right[mapping[index]] for index in range(len(mapping))] == colors_left
    mapped_edges = {
        tuple(sorted((mapping[source], mapping[target])))
        for source, target in graph_left.get_edgelist()
    }
    right_edges = {tuple(sorted(edge)) for edge in graph_right.get_edgelist()}
    assert mapped_edges == right_edges
    assert extract_qubit_permutation(left, right) == mapping[:3]


def test_tanner_builders_reject_malformed_matrix_shapes():
    class _CSSCode:
        matrix_x = np.zeros((1, 2), dtype=int)
        matrix_z = np.zeros((1, 3), dtype=int)

    class _NonCSSCode:
        matrix = np.zeros((1, 3), dtype=int)

    with pytest.raises(ValueError, match="same number of qubit columns"):
        canonical_hash(_CSSCode())
    with pytest.raises(ValueError, match="even number of columns"):
        canonical_hash_noncss(_NonCSSCode())
