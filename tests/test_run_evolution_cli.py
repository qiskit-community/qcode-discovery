"""Tests for evolve/run_evolution.py's --evaluator CLI flag (Phase A1).

Covers the plan's "Backward compatibility" test bullets: existing
configs/behavior are unaffected when no weight-5-specific flags are
supplied, and both fresh and resumed runs honor an explicit --evaluator
override which takes precedence over --noncss/--milp.

These tests never reach OpenEvolve's actual evolution loop (no LLM calls,
no network) -- _run_fresh/_run_resume are monkeypatched to raise a
sentinel exception carrying the exact kwargs main() resolved and would
have passed them, which the test then asserts against.
"""

from __future__ import annotations

import os
import sys

import pytest

import evolve.run_evolution as run_evolution


@pytest.fixture(autouse=True)
def _restore_qcode_run_name_env():
    # main() sets os.environ["QCODE_RUN_NAME"] as a side effect (to route
    # evaluator-subprocess JSONL logging) -- restore it after each test so
    # this file doesn't leak state into other tests in the same session.
    original = os.environ.get("QCODE_RUN_NAME")
    yield
    if original is None:
        os.environ.pop("QCODE_RUN_NAME", None)
    else:
        os.environ["QCODE_RUN_NAME"] = original


class _Captured(Exception):
    def __init__(self, kind, **kwargs):
        self.kind = kind
        self.kwargs = kwargs
        super().__init__(kind)


def _fake_run_fresh(config, output_dir, iterations, seed=None, evaluator=None):
    raise _Captured("fresh", output_dir=output_dir, seed=seed, evaluator=evaluator)


def _fake_run_resume(config, output_dir, iterations, checkpoint_path, seed=None, evaluator=None):
    raise _Captured(
        "resume", output_dir=output_dir, checkpoint_path=checkpoint_path,
        seed=seed, evaluator=evaluator,
    )


@pytest.fixture
def patched_runners(monkeypatch):
    monkeypatch.setattr(run_evolution, "_run_fresh", _fake_run_fresh)
    monkeypatch.setattr(run_evolution, "_run_resume", _fake_run_resume)


def _run_main_with_argv(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["run_evolution.py"] + argv)
    with pytest.raises(_Captured) as exc_info:
        run_evolution.main()
    return exc_info.value


class TestBackwardCompatibleDefaults:
    def test_no_flags_uses_default_css_evaluator_and_seed(self, monkeypatch, patched_runners, tmp_path):
        captured = _run_main_with_argv(monkeypatch, [
            "--output", str(tmp_path / "out"), "--iterations", "1",
        ])
        assert captured.kind == "fresh"
        assert captured.kwargs["evaluator"] == run_evolution.EVALUATOR
        assert captured.kwargs["seed"] == run_evolution.SEED_SOLUTION

    def test_noncss_flag_alone_selects_noncss_evaluator_and_seed(self, monkeypatch, patched_runners, tmp_path):
        captured = _run_main_with_argv(monkeypatch, [
            "--noncss", "--output", str(tmp_path / "out"), "--iterations", "1",
        ])
        assert captured.kwargs["evaluator"] == run_evolution.EVALUATOR_NONCSS
        assert captured.kwargs["seed"] == run_evolution.SEED_SOLUTION_NONCSS


class TestExplicitEvaluatorOverride:
    OVERRIDE_EVALUATOR = "evolve/openevolve_evaluator_weight5_css.py"

    def test_explicit_evaluator_overrides_default(self, monkeypatch, patched_runners, tmp_path):
        captured = _run_main_with_argv(monkeypatch, [
            "--evaluator", self.OVERRIDE_EVALUATOR,
            "--output", str(tmp_path / "out"), "--iterations", "1",
        ])
        assert captured.kwargs["evaluator"] == self.OVERRIDE_EVALUATOR

    def test_explicit_evaluator_overrides_noncss(self, monkeypatch, patched_runners, tmp_path):
        captured = _run_main_with_argv(monkeypatch, [
            "--noncss", "--evaluator", self.OVERRIDE_EVALUATOR,
            "--output", str(tmp_path / "out"), "--iterations", "1",
        ])
        assert captured.kwargs["evaluator"] == self.OVERRIDE_EVALUATOR

    def test_explicit_evaluator_honored_on_resume(self, monkeypatch, patched_runners, tmp_path):
        checkpoint_dir = tmp_path / "checkpoint_100"
        checkpoint_dir.mkdir()
        captured = _run_main_with_argv(monkeypatch, [
            "--evaluator", self.OVERRIDE_EVALUATOR,
            "--resume", str(checkpoint_dir),
            "--output", str(tmp_path / "out"), "--iterations", "1",
        ])
        assert captured.kind == "resume"
        assert captured.kwargs["evaluator"] == self.OVERRIDE_EVALUATOR
        assert captured.kwargs["checkpoint_path"] == str(checkpoint_dir)

    def test_nonexistent_evaluator_path_exits_cleanly(self, monkeypatch, patched_runners, tmp_path):
        monkeypatch.setattr(sys, "argv", [
            "run_evolution.py",
            "--evaluator", "/nonexistent/evaluator_does_not_exist.py",
            "--output", str(tmp_path / "out"), "--iterations", "1",
        ])
        with pytest.raises(SystemExit):
            run_evolution.main()
