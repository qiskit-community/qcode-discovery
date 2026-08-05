"""Tests for per-model attribution (evolve/model_attribution.py).

The attribution-log writer is pure-Python (no OpenEvolve needed); the tagging
tests require the optional `openevolve` package (the `evolve` group).
"""

import json
import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_attribution_log_writer(tmp_path):
    from evolve.model_attribution import _append_attribution

    class FakeProgram:
        id = "abc123"
        iteration_found = 7
        parent_id = "parent42"
        metadata = {"generating_model": "vendor/model-C", "island": 1}
        metrics = {"combined_score": 0.5, "cells_solved": 2.0}

    path = tmp_path / "attr.jsonl"
    _append_attribution(path, FakeProgram())
    _append_attribution(path, FakeProgram())
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec == {
        "program_id": "abc123",
        "iteration": 7,
        "parent_id": "parent42",
        "generating_model": "vendor/model-C",
        "combined_score": 0.5,
    }


class TestTagging:
    @pytest.fixture(autouse=True)
    def _need_openevolve(self):
        pytest.importorskip("openevolve")

    def test_generated_program_is_tagged_and_consumed(self):
        from openevolve.database import Program
        from evolve.model_attribution import install_tagging, note_model, _LAST_MODEL
        assert install_tagging() is True

        note_model("vendor/model-A")
        try:
            child = Program(id="c1", code="x = 1")
            assert child.metadata.get("generating_model") == "vendor/model-A"
            assert _LAST_MODEL[0] is None  # consumed
            # A subsequent Program (e.g. a reconstructed parent) is NOT tagged.
            other = Program(id="c2", code="y = 2")
            assert "generating_model" not in other.metadata
        finally:
            _LAST_MODEL[0] = None

    def test_existing_model_not_overwritten(self):
        from openevolve.database import Program
        from evolve.model_attribution import install_tagging, note_model, _LAST_MODEL
        install_tagging()
        note_model("vendor/model-B")
        try:
            p = Program(id="p", code="z", metadata={"generating_model": "preset"})
            assert p.metadata["generating_model"] == "preset"
        finally:
            _LAST_MODEL[0] = None

    def test_global_survives_asyncio_run_boundary(self):
        # Regression: in the real worker, _sample_model runs inside
        # asyncio.run(generate_with_context(...)) and the child Program is built
        # in the sync frame afterwards. A ContextVar set in the coroutine would
        # be lost here; the module global must survive.
        import asyncio
        from openevolve.database import Program
        from evolve.model_attribution import install_tagging, note_model, _LAST_MODEL
        install_tagging()
        _LAST_MODEL[0] = None

        async def _generate():
            note_model("vendor/model-Z")  # as the patched _sample_model does

        asyncio.run(_generate())
        child = Program(id="boundary", code="x")  # sync frame, after asyncio.run
        try:
            assert child.metadata.get("generating_model") == "vendor/model-Z"
        finally:
            _LAST_MODEL[0] = None

    def test_no_model_set_leaves_untagged(self):
        from openevolve.database import Program
        from evolve.model_attribution import install_tagging, _LAST_MODEL
        install_tagging()
        _LAST_MODEL[0] = None
        p = Program(id="seed", code="s")  # e.g. the initial seed program
        assert "generating_model" not in p.metadata

    def test_attribution_log_retargets_on_reinstall(self, tmp_path):
        # Re-invoking with a new path must write to the NEW path, not the first.
        from openevolve.database import ProgramDatabase
        from evolve.model_attribution import install_attribution_log, _ATTRIB_LOG_PATH
        p1, p2 = tmp_path / "run1.jsonl", tmp_path / "run2.jsonl"
        install_attribution_log(p1)
        assert _ATTRIB_LOG_PATH[0] == p1
        install_attribution_log(p2)
        assert _ATTRIB_LOG_PATH[0] == p2  # retargeted, not frozen in the closure

    def test_install_tagging_idempotent(self):
        from openevolve.database import Program
        from openevolve.llm.ensemble import LLMEnsemble
        from evolve.model_attribution import install_tagging
        install_tagging()
        install_tagging()
        assert getattr(Program.__init__, "_attrib_wrapped", False) is True
        assert getattr(LLMEnsemble._sample_model, "_attrib_wrapped", False) is True

    def test_openevolve_evaluator_import_installs_tagging(self):
        # Importing the CSS evaluator (as a worker would) must install the
        # tagging patch -- this is what makes attribution work in spawn workers.
        from openevolve.database import Program
        # Reset the wrapped marker is not possible cleanly; instead just import and
        # assert the patch is present afterward.
        spec = importlib.util.spec_from_file_location(
            "openevolve_eval_attrib_check",
            PROJECT_ROOT / "evolve" / "openevolve_evaluator.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert getattr(Program.__init__, "_attrib_wrapped", False) is True
