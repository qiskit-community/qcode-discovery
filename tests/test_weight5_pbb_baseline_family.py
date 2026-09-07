"""Focused checks for the fixed weight-five PBB baseline family."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_weight5_pbb_baseline_family.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "verify_weight5_pbb_baseline_family", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def verifier():
    return _load_script()


@pytest.mark.parametrize(
    ("ell", "m"),
    [
        (3, 2),
        (3, 3),
        (6, 5),
        (6, 6),
        (9, 8),
        (9, 9),
        (12, 9),
        (15, 14),
        (18, 12),
    ],
)
def test_dimension_and_explicit_logical(verifier, ell: int, m: int):
    result = verifier.verify_instance(ell, m)
    verifier.assert_verified(result)


def test_rejects_degenerate_or_out_of_scope_parameters(verifier):
    with pytest.raises(ValueError, match="multiple of three"):
        verifier.verify_instance(5, 4)
    with pytest.raises(ValueError, match="at least two"):
        verifier.verify_instance(6, 1)


def test_cli_smoke(verifier, capsys):
    assert verifier.main(["--ell", "3", "6", "--m", "2", "3"]) == 0
    output = capsys.readouterr().out
    assert "[[12,4,d]], certified d <= 2 (horizontal 2, vertical 2)" in output
    assert "[[18,2,d]], certified d <= 2 (horizontal 2, vertical 3)" in output
    assert "[[24,4,d]], certified d <= 2 (horizontal 4, vertical 2)" in output
    assert "[[36,2,d]], certified d <= 3 (horizontal 4, vertical 3)" in output


def test_small_m_is_a_counterexample_to_the_unqualified_distance_formula(verifier):
    result = verifier.verify_instance(6, 2)
    verifier.assert_verified(result)
    assert result.vertical_logical_weight == 2
    assert result.vertical_logical_is_nontrivial
    assert result.vertical_logical_weight < 2 * result.ell // 3


@pytest.mark.parametrize(
    ("ell", "m", "distance"),
    [(6, 2, 2), (6, 3, 3), (18, 2, 2)],
)
def test_exact_small_m_counterexamples(verifier, ell: int, m: int, distance: int):
    assert verifier.exact_distance_through(ell, m, distance) == distance
