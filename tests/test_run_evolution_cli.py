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

import json
import os
import sys
from pathlib import Path

import pytest

import evolve.run_evolution as run_evolution


_QCODE_ENV_VARS = ("QCODE_RUN_NAME", "QCODE_RUN_CONFIG_HASH", "QCODE_RUN_SEED_HASH")


@pytest.fixture(autouse=True)
def _restore_qcode_run_name_env():
    # main() sets these os.environ vars as a side effect (to route
    # evaluator-subprocess JSONL logging and discovery-event config/seed
    # hashes) -- restore them after each test so this file doesn't leak
    # state into other tests in the same session.
    originals = {var: os.environ.get(var) for var in _QCODE_ENV_VARS}
    yield
    for var, original in originals.items():
        if original is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = original


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


class TestRunManifestWiring:
    """main() writes run_manifest.json into the caller-supplied --output
    directory (never into the fixed results/evolution/<run_id>/ tree used
    by discovery_events.py/provenance_reducer.py) -- see run_manifest.py's
    manifest_path parameter and its docstring for why.
    """

    def test_manifest_written_into_output_dir_not_repo_results_tree(
        self, monkeypatch, patched_runners, tmp_path,
    ):
        out_dir = tmp_path / "out"
        _run_main_with_argv(monkeypatch, [
            "--output", str(out_dir), "--iterations", "1",
        ])

        manifest_path = out_dir / "run_manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["campaign_name"] == Path(run_evolution.EVALUATOR).stem
        assert manifest["seed_path"] == run_evolution.SEED_SOLUTION
        assert isinstance(manifest["model_aliases"], list)
        assert "effective_gates" in manifest

        # Never polluted the real repo's results/evolution/<run_name>/ tree.
        run_name = out_dir.name
        real_results_manifest = (
            Path(run_evolution.PROJECT_ROOT) / "results" / "evolution" / run_name / "run_manifest.json"
        )
        assert not real_results_manifest.exists()


class TestConfigSeedHashEnvExport:
    """main() must export QCODE_RUN_CONFIG_HASH/QCODE_RUN_SEED_HASH so
    discovery_events.append_discovery_event()'s env-var fallback (see its
    docstring) picks up real hashes instead of silently recording null on
    every event -- see evolve/discovery_events.py:169-170.
    """

    def test_config_and_seed_hash_env_vars_exported(self, monkeypatch, patched_runners, tmp_path):
        from evolve.discovery_events import file_content_hash

        monkeypatch.delenv("QCODE_RUN_CONFIG_HASH", raising=False)
        monkeypatch.delenv("QCODE_RUN_SEED_HASH", raising=False)

        _run_main_with_argv(monkeypatch, [
            "--output", str(tmp_path / "out"), "--iterations", "1",
        ])

        expected_config_hash = file_content_hash(run_evolution.DEFAULT_CONFIG)
        expected_seed_hash = file_content_hash(run_evolution.SEED_SOLUTION)
        assert expected_config_hash is not None
        assert expected_seed_hash is not None
        assert os.environ.get("QCODE_RUN_CONFIG_HASH") == expected_config_hash
        assert os.environ.get("QCODE_RUN_SEED_HASH") == expected_seed_hash

    def test_noncss_config_and_seed_hash_env_vars_exported(self, monkeypatch, patched_runners, tmp_path):
        from evolve.discovery_events import file_content_hash

        monkeypatch.delenv("QCODE_RUN_CONFIG_HASH", raising=False)
        monkeypatch.delenv("QCODE_RUN_SEED_HASH", raising=False)

        _run_main_with_argv(monkeypatch, [
            "--noncss", "--output", str(tmp_path / "out"), "--iterations", "1",
        ])

        assert os.environ.get("QCODE_RUN_CONFIG_HASH") == file_content_hash(run_evolution.DEFAULT_CONFIG_NONCSS)
        assert os.environ.get("QCODE_RUN_SEED_HASH") == file_content_hash(run_evolution.SEED_SOLUTION_NONCSS)


class TestModelAttributionWiring:
    """model_attribution.jsonl must be installed at the FIXED
    results/evolution/<run_id>/ convention regardless of --output, because
    provenance_reducer.py only ever looks there (unlike run_manifest.json,
    which is deliberately also allowed to live in a custom --output dir --
    see run_manifest.py's manifest_path docstring for that one exception).
    A --output pointing outside that tree must not divert
    model_attribution.jsonl away from where the reducer will look for it.
    """

    def test_attribution_installed_at_fixed_results_path_not_custom_output(
        self, monkeypatch, patched_runners, tmp_path,
    ):
        fixed_base = tmp_path / "fixed_results" / "evolution"
        monkeypatch.setattr(run_evolution, "EVOLUTION_BASE", str(fixed_base))

        captured_paths = []

        import evolve.model_attribution as model_attribution
        monkeypatch.setattr(model_attribution, "install", lambda path: captured_paths.append(path))

        custom_out = tmp_path / "somewhere" / "else"
        _run_main_with_argv(monkeypatch, [
            "--output", str(custom_out), "--iterations", "1",
        ])

        assert len(captured_paths) == 1
        run_name = custom_out.name
        expected = str(fixed_base / run_name / "model_attribution.jsonl")
        assert captured_paths[0] == expected
        # Must NOT have been diverted to the custom --output directory.
        assert not captured_paths[0].startswith(str(custom_out))
