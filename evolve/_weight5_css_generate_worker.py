"""Standalone candidate-generation worker for the weight-5 CSS evaluator.

Extracted to a top-level, dependency-light module (mirrors
``_noncss_distance_worker.py``) so a fresh ``multiprocessing`` child spawned
to bound an evolved, untrusted ``generate_candidates(ell, m)`` call doesn't
have to import the full evaluation stack (qldpc/numpy/scipy/sympy) just to
find the worker function -- that import cost would otherwise be paid on
every per-lattice call instead of once per long-lived pool worker.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def generate_candidates_worker(program_path: str, ell: int, m: int, result_queue) -> None:
    """Load ``generate_candidates`` from ``program_path`` and call it.

    Puts ``("ok", candidates)`` or ``("error", message)`` on
    ``result_queue``. Runs in a fresh subprocess: an evolved ``generate_fn``
    that loops forever or blows up combinatorially can only be stopped by
    killing the OS process it runs in (a thread-based timeout can't forcibly
    stop a blocking call already in progress).
    """
    try:
        if not Path(program_path).exists():
            result_queue.put(("error", f"Evolved program not found: {program_path}"))
            return
        spec = importlib.util.spec_from_file_location("evolved_program", program_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "generate_candidates"):
            result_queue.put(("error", "Evolved program missing generate_candidates function"))
            return
        candidates = module.generate_candidates(ell, m)
        result_queue.put(("ok", candidates))
    except Exception as e:
        result_queue.put(("error", f"{type(e).__name__}: {e}"))
