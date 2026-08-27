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

Phase E1 — cross-process visibility of ``program_id`` during evaluation
-------------------------------------------------------------------------
Traced the actual call chain in the installed ``openevolve`` package
(``openevolve/process_parallel.py`` + ``openevolve/evaluator.py``) to answer,
conclusively, where ``Evaluator.evaluate_program`` actually executes:

``ProcessParallelController._submit_iteration`` calls
``self.executor.submit(_run_iteration_worker, iteration, db_snapshot,
parent.id, inspiration_ids)`` — ``_run_iteration_worker`` is the function that
runs *inside* the ``ProcessPoolExecutor`` worker process (that's the whole
point of submitting it to the executor). Inside that same function, after LLM
generation, the worker does::

    child_metrics = asyncio.run(_worker_evaluator.evaluate_program(child_code, child_id))
    ...
    child_program = Program(id=child_id, ..., parent_id=parent.id,
                             iteration_found=iteration, metrics=child_metrics, ...)

``_worker_evaluator`` is itself constructed inside that worker process, lazily,
by ``_lazy_init_worker_components()`` (called at the top of
``_run_iteration_worker``). So ``Evaluator.evaluate_program`` — and therefore
our repo's ``evaluate_stage1``/``evaluate_stage2``/etc., reached via
``Evaluator._direct_evaluate``'s or ``_cascade_evaluate``'s
``loop.run_in_executor(None, fn, program_path)`` — runs on a **thread of that
same worker OS process**, never on the controller process. ``run_in_executor``
with ``executor=None`` uses the event loop's default thread pool, which lives
in the *same* process as the event loop created by ``asyncio.run`` here — it is
not a second process.

Consequence: a plain module global set by a wrapper around
``Evaluator.evaluate_program`` — exactly the same mechanism already used for
``_LAST_MODEL`` above — is visible to ``_run_evaluation()`` when it later reads
it, because both run in the same OS process, and (per the ``_LAST_MODEL``
docstring above) each worker process executes exactly one iteration
(therefore exactly one ``evaluate_program`` call) at a time. No cross-process
channel (env var re-injection, files, etc.) is needed for ``program_id``;
:func:`peek_program_id` reads ``_CURRENT_PROGRAM_ID`` set by the
``evaluate_program`` wrapper installed in :func:`install_tagging`, set for the
duration of that call and restored afterward.

This also confirms, independently of the plan's own claim, that
``parent_id``/``iteration_found`` are genuinely unavailable inside
``evaluate_program``: the ``Program(...)`` object above — the only place those
values exist — is constructed *after* ``evaluate_program`` returns. See
``evolve/provenance_reducer.py`` (Phase E4) for how those fields get
backfilled post-hoc via a join on ``program_id``.

:func:`peek_model` / :func:`peek_program_id` are **non-consuming** reads (a
plain ``return _X[0]``), unlike ``note_model``'s value in ``_LAST_MODEL``,
which ``Program.__init__``'s wrapper *consumes* (sets back to ``None``) on
first use. That consumption happens fine *after* evaluation for our purposes:
OpenEvolve evaluates the child (our ``_run_evaluation()`` calls
``peek_model()``) strictly before building the child ``Program`` object (which
is what consumes ``_LAST_MODEL``), so the model set during generation is still
present, unconsumed, whenever a per-code result is stamped.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

# Process-global (NOT a contextvar — see module docstring re: asyncio.run).
# A one-element list is a simple mutable cell. Safe under OpenEvolve's
# one-iteration-per-worker execution; documented limitation otherwise.
_LAST_MODEL: list = [None]
# Program id of whichever evaluate_program() call is currently in flight in
# this OS process (set by the wrapper installed below). Same one-element-list
# convention as _LAST_MODEL, for the same reason (survives the asyncio.run
# boundary; safe because a worker process evaluates one program at a time).
_CURRENT_PROGRAM_ID: list = [None]
# Mutable so re-invoking install_attribution_log with a new path retargets the
# already-wrapped ProgramDatabase.add (e.g. multiple runs in one process).
_ATTRIB_LOG_PATH: list = [None]
_LOG_LOCK = threading.Lock()


def note_model(name) -> None:
    """Record the model about to generate the next program."""
    if name:
        _LAST_MODEL[0] = str(name)


def peek_model() -> str | None:
    """Return the currently-noted generating model, WITHOUT consuming it.

    Unlike ``note_model``'s value, which ``Program.__init__``'s wrapper
    consumes (resets to ``None``) the first time a child is constructed,
    this is a plain read with no side effects. Safe to call any number of
    times during evaluation (before the child ``Program`` is built).
    """
    return _LAST_MODEL[0]


def peek_program_id() -> str | None:
    """Return the ``program_id`` of the ``evaluate_program()`` call currently
    in flight in this process, or ``None`` if none is (e.g. outside OpenEvolve,
    or the wrapper isn't installed). Non-consuming; see :func:`peek_model`."""
    return _CURRENT_PROGRAM_ID[0]


def install_tagging() -> bool:
    """Tag each generated Program with the model that produced it, and make
    the in-flight ``program_id`` readable during evaluation.

    Idempotent; returns False if OpenEvolve is unavailable.
    """
    try:
        from openevolve.llm.ensemble import LLMEnsemble
        from openevolve.database import Program
        from openevolve.evaluator import Evaluator
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

    if not getattr(Evaluator.evaluate_program, "_attrib_wrapped", False):
        _orig_evaluate_program = Evaluator.evaluate_program

        async def _evaluate_program(self, program_code, program_id: str = "", *args, **kwargs):
            # Set for the duration of this call; restore afterward (save/
            # restore rather than blind-clear, in case of any re-entrancy —
            # though per the module docstring a worker process only ever has
            # one evaluate_program() call in flight at a time).
            previous = _CURRENT_PROGRAM_ID[0]
            _CURRENT_PROGRAM_ID[0] = program_id or None
            try:
                return await _orig_evaluate_program(self, program_code, program_id, *args, **kwargs)
            finally:
                _CURRENT_PROGRAM_ID[0] = previous

        _evaluate_program._attrib_wrapped = True
        Evaluator.evaluate_program = _evaluate_program

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
        # Safety net, not attribution: OpenEvolve's own cascade-evaluation Stage
        # 1/2/3 timeout paths (openevolve/evaluator.py) return a hardcoded
        # metrics dict ({"stage1_passed", "error", "timeout"}) that never
        # includes any project-specific MAP-Elites feature_dimensions (e.g.
        # weight-5 PBB's "lattices_with_high_k"/"num_high_k", CSS's
        # "term_count"/"pattern_type"). database.py's _calculate_feature_coords
        # has no default for a missing custom dimension -- it raises ValueError
        # and kills the whole run on the next add(), including the initial seed
        # program. Bolted onto this wrapper because it is the one place all
        # four weight-5 campaigns already monkeypatch ProgramDatabase.add.
        try:
            for dim in getattr(self.config, "feature_dimensions", None) or ():
                if dim not in ("complexity", "diversity", "score") and dim not in program.metrics:
                    program.metrics[dim] = 0.0
        except Exception:
            pass
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
