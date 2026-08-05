"""Per-model attribution for OpenEvolve runs.

Records which ensemble model produced each evolved program — the WP9 per-model
mutation-attribution piece, useful for the search-vs-recall ablation and for
understanding which model contributes what.

Why this is non-trivial: OpenEvolve's ``LLMEnsemble`` picks a model in
``_sample_model`` but ``generate_with_context`` returns only text, so the child
``Program`` (built right after) never learns its model. Generation runs in
**spawn** worker processes, so a main-process monkeypatch never reaches them.
And in the process-parallel path the worker does
``llm_response = asyncio.run(generate_with_context(...))`` and then builds the
child ``Program`` in the *synchronous* frame afterwards — so a ``ContextVar``
set inside that coroutine is invisible to the program-construction frame (the
coroutine runs in a copied context). We therefore use a **module global**, which
persists across the ``asyncio.run`` boundary and is safe because each worker runs
exactly one iteration at a time.

- :func:`install_tagging` wraps ``LLMEnsemble._sample_model`` to record the
  sampled model, and ``Program.__init__`` to copy it into
  ``metadata["generating_model"]`` — *consumed* on first use, so only the
  freshly generated child is tagged, never the parent programs a worker
  reconstructs from a database snapshot before generating. Installed from each
  campaign evaluator's import (which the worker loads, via
  ``Evaluator._load_evaluation_function``, before the first LLM call).
- :func:`install_attribution_log` wraps ``ProgramDatabase.add`` (main process,
  where accepted programs arrive with their worker-set metadata) to append one
  JSONL record per accepted program.

Limitation: attribution assumes the *generation* ensemble is what samples a
model before the child is built. LLM-based evaluation (a second ensemble
sampling between generation and child construction) would clobber the global;
all current campaigns use deterministic evaluators, so this does not arise.

Everything is idempotent and defensive: a failure here must never break a run.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

# Process-global (NOT a contextvar — see module docstring re: asyncio.run).
# A one-element list is a simple mutable cell. Safe under OpenEvolve's
# one-iteration-per-worker execution; documented limitation otherwise.
_LAST_MODEL: list = [None]
# Mutable so re-invoking install_attribution_log with a new path retargets the
# already-wrapped ProgramDatabase.add (e.g. multiple runs in one process).
_ATTRIB_LOG_PATH: list = [None]
_LOG_LOCK = threading.Lock()


def note_model(name) -> None:
    """Record the model about to generate the next program."""
    if name:
        _LAST_MODEL[0] = str(name)


def install_tagging() -> bool:
    """Tag each generated Program with the model that produced it.

    Idempotent; returns False if OpenEvolve is unavailable.
    """
    try:
        from openevolve.llm.ensemble import LLMEnsemble
        from openevolve.database import Program
    except Exception:
        return False

    if not getattr(LLMEnsemble._sample_model, "_attrib_wrapped", False):
        _orig_sample = LLMEnsemble._sample_model

        def _sample_model(self):
            model = _orig_sample(self)
            try:
                note_model(getattr(model, "model", None) or getattr(model, "name", None))
            except Exception:
                pass
            return model

        _sample_model._attrib_wrapped = True
        LLMEnsemble._sample_model = _sample_model

    if not getattr(Program.__init__, "_attrib_wrapped", False):
        _orig_init = Program.__init__

        def _init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            try:
                name = _LAST_MODEL[0]
                meta = getattr(self, "metadata", None)
                if name and isinstance(meta, dict) and not meta.get("generating_model"):
                    meta["generating_model"] = name
                    # Consume: only the just-generated child is tagged, not the
                    # parent programs reconstructed from the db snapshot.
                    _LAST_MODEL[0] = None
            except Exception:
                pass

        _init._attrib_wrapped = True
        Program.__init__ = _init

    return True


def _append_attribution(path, program) -> None:
    """Append one attribution record for ``program`` to the JSONL at ``path``."""
    meta = getattr(program, "metadata", None) or {}
    metrics = getattr(program, "metrics", None) or {}
    record = {
        "program_id": getattr(program, "id", None),
        "iteration": getattr(program, "iteration_found", None),
        "parent_id": getattr(program, "parent_id", None),
        "generating_model": meta.get("generating_model"),
        "combined_score": metrics.get("combined_score"),
    }
    with _LOG_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def install_attribution_log(path) -> bool:
    """Append a JSONL attribution record for every accepted program.

    Wraps ``ProgramDatabase.add`` (main process). Idempotent — re-invoking with a
    new ``path`` retargets the existing wrapper rather than double-wrapping.
    Returns False if OpenEvolve is unavailable.
    """
    try:
        from openevolve.database import ProgramDatabase
    except Exception:
        return False

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _ATTRIB_LOG_PATH[0] = path  # current target (read by the wrapper)

    if getattr(ProgramDatabase.add, "_attrib_wrapped", False):
        return True

    _orig_add = ProgramDatabase.add

    def _add(self, program, *args, **kwargs):
        result = _orig_add(self, program, *args, **kwargs)
        try:
            target = _ATTRIB_LOG_PATH[0]
            if target is not None:
                _append_attribution(target, program)
        except Exception:
            pass
        return result

    _add._attrib_wrapped = True
    ProgramDatabase.add = _add
    return True


def install(attribution_log_path=None) -> None:
    """Install tagging always; install the JSONL log if a path is given."""
    install_tagging()
    if attribution_log_path:
        install_attribution_log(attribution_log_path)
