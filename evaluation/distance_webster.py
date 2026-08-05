"""Adapter around the Webster et al. ``codedistance`` package (arXiv:2603.22532).

This wraps the two public entrypoints of ``codedistance`` so the rest of our
codebase can cross-check distances without touching numpy/symplectic plumbing:

    codedistance.CSScodeDistance(Hx, Hz, method, params, component, seed)
    codedistance.codeDistance(H, L, tB, method, params, seed)

Both return a dict ``{'n', 'k', 'd', 'L', 'T', 'R', 'progress'}``.  ``L`` is a
binary witness of weight ``d``:

  - CSS, ``component='Z'``: an ``n``-vector, a Z-type logical (commutes with X
    checks).  ``component='X'`` gives an X-type logical.
  - non-CSS (``tB=2``): a ``2n``-vector ``[LX | LZ]`` in the same block layout
    as the symplectic input ``H = [HX | HZ]``.

The package itself is an optional dependency (the ``webster`` group in
``pyproject.toml``); import it lazily so the rest of the package works without
it.  See ``investigations/webster_comparison_plan.md`` for the full plan.

Notes on exactness / ``result_status``
--------------------------------------
The public result dict has no ``exact`` field.  We classify locally:

  - ``BZDistMW`` is Brouwer-Zimmermann enumeration: exact iff it runs to
    completion (i.e. the call returns without being killed by the outer
    timeout).  -> ``proven_exact``.
  - ``MIPDist`` (OR-tools/SCIP) does not surface solver optimality in the
    public dict, so we never claim ``proven_exact`` from it -> ``incumbent``.
    A *verified witness* of weight below our exact distance is still
    authoritative regardless of optimality, which is the comparison that
    matters.
  - ``QDistEvol`` / ``decoderDist`` are heuristics -> ``heuristic`` (upper
    bound).
  - Killed by the outer timeout -> ``timeout``; raised an exception or no
    result -> ``failed``.

Never infer ``proven_exact`` from the method name alone; see the plan's
risk #10.
"""

from __future__ import annotations

import multiprocessing as mp
import time
import re
from dataclasses import dataclass

import numpy as np

# MIPDist writes its OR-tools wall-clock into ``progress`` as ``T:<int>ms``.
# This is useful for diagnosing whether it likely hit the time limit, but it is
# not an optimality certificate. The public codedistance API discards the
# OR-Tools solver status, so MIPDist remains an incumbent unless we patch or wrap
# the package to expose that status explicitly.
_MIP_WALLCLOCK_RE = re.compile(r"T:(\d+)ms")

# Method classification (plan section 2 / adapter sketch).
PROVEN_EXACT_IF_COMPLETED = {"BZDistMW"}
SOLVER_METHODS_NEED_STATUS = {"MIPDist", "pySATDist", "GurobiDist"}
HEURISTIC_METHODS = {
    "QDistEvol",
    "QDistRndMW",
    "decoderDist",
    "connectedClusterMW",
    "UndetectableErrorMW",
    "GraphLikeErrorMW",
}

# Outer process-timeout defaults per method (seconds). The outer deadline is
# authoritative: Webster's own ``maxTime`` is not honored by every pure-Python
# method (plan section 6).
DEFAULT_TIMEOUTS = {
    "BZDistMW": 600,
    "MIPDist": 660,
    "QDistEvol": 360,
    "decoderDist": 360,
}


@dataclass
class WebsterResult:
    """Normalized result from a single ``codedistance`` call."""

    d: int
    L: np.ndarray  # binary witness; len n (CSS) or 2n (non-CSS)
    T: int
    R: int
    runtime_s: float
    method: str
    params: dict
    n: int
    k: int
    result_status: str  # proven_exact | incumbent | heuristic | timeout | failed
    optimality_source: str
    progress: str = ""
    error: str | None = None

    @property
    def method_claims_exact(self) -> bool:
        return self.result_status == "proven_exact"


def _classify_status(
    method: str,
    timed_out: bool,
    errored: bool,
    progress: str = "",
    maxtime_s: float | None = None,
) -> tuple[str, str]:
    """Return ``(result_status, optimality_source)`` for a completed call.

    For solver methods (``MIPDist``) the public dict hides optimality. A wall
    clock shorter than ``maxTime`` is not a solver status, so these results stay
    ``incumbent`` unless the package is changed to surface optimality directly.
    """
    if errored:
        return "failed", "error"
    if timed_out:
        return "timeout", "timeout"
    if method in PROVEN_EXACT_IF_COMPLETED:
        return "proven_exact", "method_completed"
    if method in SOLVER_METHODS_NEED_STATUS:
        m = _MIP_WALLCLOCK_RE.search(progress or "")
        wall_ms = int(m.group(1)) if m else None
        if maxtime_s is not None and wall_ms is not None:
            # 2% margin guards against rounding right at the time limit.
            if wall_ms >= maxtime_s * 1000 * 0.98:
                return "incumbent", "solver_hit_time_limit"
            return "incumbent", "solver_status_not_exposed"
        # Optimality not recoverable from the public dict -> treat as upper bound.
        return "incumbent", "not_exposed"
    return "heuristic", "trials"


def _pack_dict(res: dict) -> dict:
    """Convert a ``codedistance`` result dict to plain picklable Python."""
    L = np.asarray(res["L"], dtype=np.int64).ravel() % 2
    return {
        "n": int(res["n"]),
        "k": int(res["k"]),
        "d": int(res["d"]),
        "L": L.tolist(),
        "T": int(res.get("T", 1)),
        "R": int(res.get("R", 1)),
        "progress": str(res.get("progress", "")),
    }


# --- subprocess workers (module-level so multiprocessing 'spawn' can pickle) ---

def _worker_css(q, Hx, Hz, method, component, seed, params):
    try:
        import codedistance as cd

        res = cd.CSScodeDistance(
            np.asarray(Hx, dtype=np.int8),
            np.asarray(Hz, dtype=np.int8),
            method=method,
            params=dict(params),
            component=component,
            seed=seed,
        )
        q.put(("ok", _pack_dict(res)))
    except Exception as exc:  # pragma: no cover - reported to parent
        q.put(("err", repr(exc)))


def _worker_noncss(q, H, method, seed, params):
    try:
        import codedistance as cd

        res = cd.codeDistance(
            np.asarray(H, dtype=np.int8),
            L=None,
            tB=2,
            method=method,
            params=dict(params),
            seed=seed,
        )
        q.put(("ok", _pack_dict(res)))
    except Exception as exc:  # pragma: no cover - reported to parent
        q.put(("err", repr(exc)))


def _run_timed(target, args, timeout_s):
    """Run ``target(q, *args)`` in a spawned process with an outer deadline.

    Returns ``(packed_dict_or_None, timed_out, errored, error_msg, runtime_s)``.
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=target, args=(q, *args))
    t0 = time.time()
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return None, True, False, None, time.time() - t0
    runtime = time.time() - t0
    try:
        tag, payload = q.get_nowait()
    except Exception:
        return None, False, True, "no result returned from subprocess", runtime
    if tag == "err":
        return None, False, True, payload, runtime
    return payload, False, False, None, runtime


# --------------------------- public entrypoints -------------------------------

def webster_distance_css(
    code,
    method: str = "BZDistMW",
    component: str = "Z",
    seed: int = 42,
    params: dict | None = None,
    timeout_s: float | None = None,
) -> WebsterResult:
    """Compute the CSS distance (one component) via ``codedistance``.

    Args:
        code: a qldpc ``CSSCode`` / ``BBCode`` exposing ``matrix_x``/``matrix_z``.
        method: Webster method name.
        component: ``'Z'`` (default) or ``'X'``.
        seed: RNG seed for reproducibility.
        params: extra Webster params (e.g. ``{'iterCount': 10000}``).
        timeout_s: outer process deadline; ``None`` runs in-process (no timeout).

    Returns:
        A :class:`WebsterResult`. ``result_status`` reflects exactness.
    """
    params = dict(params or {})
    # Capture the caller's budget *before* the call: codedistance mutates the
    # params dict in place (injecting its 8h default maxTime), and that mutation
    # is visible in-process but not across the subprocess boundary. Reading it
    # here keeps optimality classification consistent across both paths.
    maxtime_s = params.get("maxTime")
    Hx = np.asarray(code.matrix_x, dtype=np.int8) % 2
    Hz = np.asarray(code.matrix_z, dtype=np.int8) % 2

    if timeout_s is None:
        import codedistance as cd

        t0 = time.time()
        try:
            res = cd.CSScodeDistance(
                Hx, Hz, method=method, params=params, component=component, seed=seed
            )
            packed = _pack_dict(res)
            runtime, timed_out, errored, err = time.time() - t0, False, False, None
        except Exception as exc:
            packed, runtime, timed_out, errored, err = (
                None,
                time.time() - t0,
                False,
                True,
                repr(exc),
            )
    else:
        packed, timed_out, errored, err, runtime = _run_timed(
            _worker_css, (Hx, Hz, method, component, seed, params), timeout_s
        )

    return _build_result(
        packed, method, params, component, code, runtime, timed_out, errored, err,
        maxtime_s=maxtime_s,
    )


def webster_distance_noncss(
    code,
    method: str = "BZDistMW",
    seed: int = 42,
    params: dict | None = None,
    timeout_s: float | None = None,
) -> WebsterResult:
    """Compute the non-CSS distance via ``codedistance`` (symplectic ``tB=2``).

    Args:
        code: a qldpc ``QuditCode`` exposing a ``(rows, 2n)`` symplectic
            ``matrix`` in ``[HX | HZ]`` layout (as built by
            ``evaluation.pbb_code.build_pbb_code``).
    """
    params = dict(params or {})
    maxtime_s = params.get("maxTime")  # capture before codedistance mutates params
    H = np.asarray(code.matrix, dtype=np.int8) % 2

    if timeout_s is None:
        import codedistance as cd

        t0 = time.time()
        try:
            res = cd.codeDistance(H, L=None, tB=2, method=method, params=params, seed=seed)
            packed = _pack_dict(res)
            runtime, timed_out, errored, err = time.time() - t0, False, False, None
        except Exception as exc:
            packed, runtime, timed_out, errored, err = (
                None,
                time.time() - t0,
                False,
                True,
                repr(exc),
            )
    else:
        packed, timed_out, errored, err, runtime = _run_timed(
            _worker_noncss, (H, method, seed, params), timeout_s
        )

    return _build_result(
        packed, method, params, component=None, code=code, runtime=runtime,
        timed_out=timed_out, errored=errored, err=err, maxtime_s=maxtime_s,
    )


def _build_result(packed, method, params, component, code, runtime, timed_out, errored, err,
                  maxtime_s=None):
    progress = packed["progress"] if packed else ""
    status, opt_source = _classify_status(
        method, timed_out, errored or packed is None,
        progress=progress, maxtime_s=maxtime_s,
    )
    if packed is None:
        return WebsterResult(
            d=-1,
            L=np.zeros(0, dtype=np.int64),
            T=0,
            R=0,
            runtime_s=runtime,
            method=method,
            params=params,
            n=int(getattr(code, "num_qudits", 0)),
            k=int(getattr(code, "dimension", 0)),
            result_status=status,
            optimality_source=opt_source,
            progress="",
            error=err,
        )
    k_reported = int(packed["k"])
    k_from_code = int(getattr(code, "dimension", 0))
    if component is None and k_reported == 0 and k_from_code:
        k_reported = k_from_code

    return WebsterResult(
        d=packed["d"],
        L=np.asarray(packed["L"], dtype=np.int64),
        T=packed["T"],
        R=packed["R"],
        runtime_s=runtime,
        method=method,
        params=params,
        n=packed["n"],
        k=k_reported,
        result_status=status,
        optimality_source=opt_source,
        progress=packed["progress"],
    )


# --------------------------- witness verification -----------------------------

def verify_witness_css(code, L, component: str, claimed_d: int) -> dict:
    """Verify a CSS witness ``L`` against ``code``'s checks and logicals.

    A genuine logical of weight ``d`` must (a) commute with the opposite-type
    checks, (b) anticommute with at least one opposite-type logical (which
    also certifies non-triviality), and (c) have Hamming weight ``= claimed_d``.
    """
    from qldpc.objects import Pauli

    L = np.asarray(L, dtype=np.int64).ravel() % 2
    n = int(code.num_qudits)
    weight = int(L.sum())
    len_ok = len(L) == n

    if not len_ok:
        return {
            "len_ok": False,
            "weight": weight,
            "weight_matches_d": weight == claimed_d,
            "commutes": False,
            "anticommutes_with_logical": False,
            "verified": False,
        }

    hx = np.asarray(code.matrix_x, dtype=np.int64) % 2
    hz = np.asarray(code.matrix_z, dtype=np.int64) % 2

    if component == "Z":
        # Z-type logical: commutes with X-checks, anticommutes with an X-logical.
        commutes = int((hx @ L % 2).sum()) == 0
        lx = np.asarray(code.get_logical_ops(Pauli.X), dtype=np.int64) % 2
        anticommutes = bool(np.any((lx @ L) % 2 == 1)) if lx.size else False
    else:
        commutes = int((hz @ L % 2).sum()) == 0
        lz = np.asarray(code.get_logical_ops(Pauli.Z), dtype=np.int64) % 2
        anticommutes = bool(np.any((lz @ L) % 2 == 1)) if lz.size else False

    weight_ok = weight == claimed_d
    return {
        "len_ok": len_ok,
        "weight": weight,
        # Diagnostic only: does the witness weight match the package's *reported*
        # d? For timed-out incumbents these differ (reported = objective bound,
        # witness weight = the actual logical found). Not part of validity.
        "weight_matches_d": weight_ok,
        "commutes": bool(commutes),
        "anticommutes_with_logical": anticommutes,
        # A valid witness is a genuine nontrivial logical: commutes with checks
        # AND anticommutes with a logical. Its weight (``weight`` above) is then a
        # sound upper bound on d regardless of the reported value.
        "verified": bool(len_ok and commutes and anticommutes),
    }


def verify_witness_noncss(code, L, claimed_d: int) -> dict:
    """Verify a non-CSS symplectic witness ``L = [LX | LZ]`` (length ``2n``).

    Uses our own ``get_symplectic_logicals`` (plan section 3.3) rather than
    qldpc's non-CSS ``get_logical_ops`` path, which is buggy on some PBB codes.
    The reported distance is the *symplectic weight*: number of qubits with any
    X or Z support.
    """
    from evaluation.pbb_code import get_symplectic_logicals

    L = np.asarray(L, dtype=np.int64).ravel() % 2
    n = int(code.num_qudits)
    len_ok = len(L) == 2 * n

    if not len_ok:
        return {
            "len_ok": False,
            "weight": -1,
            "weight_matches_d": False,
            "commutes": False,
            "anticommutes_with_logical": False,
            "verified": False,
        }

    LX, LZ = L[:n], L[n:]
    S = np.asarray(code.matrix, dtype=np.int64) % 2
    SX, SZ = S[:, :n], S[:, n:]
    commutes = int(((SX @ LZ + SZ @ LX) % 2).sum()) == 0

    G = np.asarray(get_symplectic_logicals(code), dtype=np.int64) % 2  # (2k, 2n) [X|Z]
    GX, GZ = G[:, :n], G[:, n:]
    sp = (GX @ LZ + GZ @ LX) % 2
    anticommutes = bool(np.any(sp == 1)) if G.size else False

    symplectic_weight = int(np.sum(LX | LZ))
    weight_ok = symplectic_weight == claimed_d
    return {
        "len_ok": len_ok,
        "weight": symplectic_weight,
        # Diagnostic only (see verify_witness_css): reported-d match is not part
        # of validity -- timed-out incumbents return a valid witness whose weight
        # differs from the reported objective.
        "weight_matches_d": weight_ok,
        "commutes": bool(commutes),
        "anticommutes_with_logical": anticommutes,
        "verified": bool(commutes and anticommutes),
    }
