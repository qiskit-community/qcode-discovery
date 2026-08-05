#!/usr/bin/env python
"""Tier 1 runner for the Webster ``codedistance`` cross-check.

Implements Comparison A (exact claims + witness cross-check) and Comparison C
(non-CSS symplectic linearization sanity) from
``investigations/webster_comparison_plan.md``: run Webster's independent methods
on our exact/incumbent codes and check that no verified witness comes in *below*
a distance we claim is exact.

Catalogs (per the user's direction):
  * CSS  : ``results/ilp_catalog.json``  -- grouped by ``n=...`` + bravyi_baselines;
           each record carries ``ilp_d`` / ``ilp_d_x`` / ``ilp_d_z`` and a
           table ``label``. The label is not a solver certificate; only the
           Bravyi baseline anchors are treated as CSS exact claims here.
  * PBB  : ``results/campaign7_publication_merged.jsonl`` -- BLISS-deduplicated;
           exact iff (milp_exact or d_is_exact or trust_level=='EXACT') and not
           d_is_upper_bound.

Design decisions, all driven by Tier-0 timing (see plan section 13):
  * **Trust the witness, not the reported d.** The authoritative Webster value
    is the recomputed weight of the returned witness (Hamming for CSS, symplectic
    for non-CSS), via ``evaluation.distance_webster.verify_witness_*``.
  * **MIPDist is the workhorse; BZDistMW only on small codes.** MIP gives useful
    verified witnesses on the gross [[144,12,12]] and larger catalog rows, while
    BZ times out quickly beyond the small anchors. BZ blows up on the 2n-length
    symplectic form of non-CSS codes. So we run BZ only when the effective length
    (n for CSS, 2n for PBB) is <= a small cap; MIP runs on everything with an
    explicit ``maxTime`` but remains an incumbent unless solver status is exposed.
  * **Resumable.** Each (code_id, method, component) result is appended to the
    output JSONL; re-running skips work already present.

Outputs (under ``investigations/``):
  * ``webster_tier1_codes.json``  -- latest resolved selection.
  * ``webster_tier1_codes_<id>.json`` / ``webster_selection_history.jsonl``
                                    -- stable snapshots of each selection.
  * ``webster_results.jsonl``     -- one record per (code, method, component).
  * ``webster_summary.jsonl``     -- one verdict record per code.

Usage::

    uv run --group dev --group webster python investigations/compare_webster.py --select-only
    uv run --group dev --group webster python investigations/compare_webster.py --limit 24
    uv run --group dev --group webster python investigations/compare_webster.py --pbb-only --all-n
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluation.bb_code import build_bb_code  # noqa: E402
from evaluation.pbb_code import build_pbb_code  # noqa: E402

INVEST = REPO / "investigations"
CSS_CATALOG = REPO / "results" / "ilp_catalog.json"
PBB_CATALOG = REPO / "results" / "campaign7_publication_merged.jsonl"
SELECTION_OUT = INVEST / "webster_tier1_codes.json"
SELECTION_HISTORY_OUT = INVEST / "webster_selection_history.jsonl"
RESULTS_OUT = INVEST / "webster_results.jsonl"
SUMMARY_OUT = INVEST / "webster_summary.jsonl"

_LABEL_RE = re.compile(r"\[\[(\d+),(\d+),(<=)?(\d+)\]\]")
_MIP_WALLCLOCK_RE = re.compile(r"T:(\d+)ms")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _terms(seq):
    """Normalize a polynomial term list to a tuple-of-tuples (hashable)."""
    return tuple((int(a), int(b)) for a, b in seq)


def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def record_fingerprint(rec: dict) -> str:
    """Stable fingerprint for the source code data used by a Webster run."""
    keys = [
        "code_type", "ell", "m", "n", "k",
        "A_terms", "B_terms", "C_terms", "D_terms",
        "ours_d", "ours_d_x", "ours_d_z", "ours_d_milp",
        "ours_exact", "ours_exact_source", "source", "source_group",
        "source_index", "source_label", "bliss_hash",
    ]
    payload = {k: rec.get(k) for k in keys if k in rec}
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()[:16]


def method_budget(method: str, mip_maxtime: int, bz_timeout: int) -> tuple[dict, int]:
    """Return Webster params and outer timeout for one method call."""
    if method == "MIPDist":
        return {"maxTime": mip_maxtime}, mip_maxtime + 60
    return {}, bz_timeout


def effective_result_status(webster: dict) -> tuple[str, str]:
    """Conservative status normalization for old and new result records."""
    raw_status = webster.get("result_status")
    raw_source = webster.get("optimality_source")
    if webster.get("method") != "MIPDist":
        return raw_status, raw_source
    if raw_status in {"failed", "timeout"}:
        return raw_status, raw_source

    params = webster.get("params") or {}
    maxtime_s = params.get("maxTime")
    m = _MIP_WALLCLOCK_RE.search(webster.get("progress") or "")
    wall_ms = int(m.group(1)) if m else None
    if maxtime_s is not None and wall_ms is not None:
        if wall_ms >= maxtime_s * 1000 * 0.98:
            return "incumbent", "solver_hit_time_limit"
        return "incumbent", "solver_status_not_exposed"
    return "incumbent", "not_exposed"


# --------------------------------------------------------------------------- #
# Catalog loading -> normalized code records.
# --------------------------------------------------------------------------- #
def load_css_pool() -> list[dict]:
    """Normalize ``ilp_catalog.json`` into CSS code records.

    Returns dicts with: code_type='css', code_id, ell, m, n, k, A_terms,
    B_terms, ours_d, ours_d_x, ours_d_z, ours_exact, ours_status, ours_bposd,
    source, source_group.
    """
    data = json.loads(CSS_CATALOG.read_text())
    out: list[dict] = []
    for group, rows in data.items():
        for idx, r in enumerate(rows):
            label = r.get("label", "")
            m_lbl = _LABEL_RE.search(label)
            if not m_lbl:
                continue
            n = int(m_lbl.group(1))
            k = int(m_lbl.group(2))
            ours_d = r.get("ilp_d")
            if ours_d is None:
                continue
            ell, mm = int(r["ell"]), int(r["m"])
            A = _terms(r["A"])
            B = _terms(r["B"])
            # ``ilp_catalog.json`` does not carry per-row solver proof metadata:
            # ``status`` is BP-OSD-vs-ILP agreement, and the label is a paper
            # display bound. Keep the known Bravyi anchors as exact claims; all
            # other CSS rows are row-level corroboration targets.
            css_exact = group == "bravyi_baselines"
            out.append(
                {
                    "code_type": "css",
                    "code_id": f"css_{group}_{ell}x{mm}_{idx}",
                    "ell": ell,
                    "m": mm,
                    "n": n,
                    "k": k,
                    "A_terms": [list(t) for t in A],
                    "B_terms": [list(t) for t in B],
                    "ours_d": int(ours_d),
                    "ours_d_x": (int(r["ilp_d_x"]) if r.get("ilp_d_x") is not None else None),
                    "ours_d_z": (int(r["ilp_d_z"]) if r.get("ilp_d_z") is not None else None),
                    "ours_exact": css_exact,
                    "ours_exact_source": (
                        "bravyi_baseline" if css_exact else
                        "not_certified_in_ilp_catalog"
                    ),
                    "ours_status": r.get("status"),
                    "ours_bposd": r.get("bp_osd_d"),
                    "source": CSS_CATALOG.name,
                    "source_group": group,
                    "source_index": idx,
                    "source_label": label,
                    "_dedupe_key": ("css", ell, mm, A, B),
                }
            )
    return out


def load_pbb_pool() -> list[dict]:
    """Normalize the PBB JSONL into non-CSS code records (deduped by bliss_hash).

    Returns dicts with: code_type='pbb', code_id, ell, m, n, k, A/B/C/D_terms,
    ours_d, ours_d_milp, ours_d_bposd, ours_exact, ours_milp_exact, ours_trust,
    bliss_hash, source.
    """
    rows = [json.loads(l) for l in PBB_CATALOG.read_text().splitlines() if l.strip()]
    out: list[dict] = []
    for idx, r in enumerate(rows):
        exact = (
            r.get("milp_exact") is True
            or r.get("d_is_exact") is True
            or r.get("trust_level") == "EXACT"
        )
        exact = exact and not r.get("d_is_upper_bound")
        bliss = r.get("bliss_hash")
        dedupe = ("pbb", bliss) if bliss else (
            "pbb", r["ell"], r["m"],
            _terms(r["A_terms"]), _terms(r["B_terms"]),
            _terms(r.get("C_terms") or []), _terms(r.get("D_terms") or []),
        )
        out.append(
            {
                "code_type": "pbb",
                "code_id": r.get("code_id") or f"pbb_{r['ell']}x{r['m']}_{idx}",
                "ell": int(r["ell"]),
                "m": int(r["m"]),
                "n": int(r["n"]),
                "k": int(r["k"]),
                "A_terms": r["A_terms"],
                "B_terms": r["B_terms"],
                "C_terms": r.get("C_terms"),
                "D_terms": r.get("D_terms"),
                "ours_d": int(r["d"]),
                "ours_d_milp": r.get("d_milp"),
                "ours_d_bposd": r.get("d_bposd"),
                "ours_exact": bool(exact),
                "ours_milp_exact": r.get("milp_exact"),
                "ours_trust": r.get("trust_level"),
                "bliss_hash": bliss,
                "source": PBB_CATALOG.name,
                "source_index": idx,
                "_dedupe_key": dedupe,
            }
        )
    return out


def dedupe(records: list[dict]) -> list[dict]:
    """Keep one record per ``_dedupe_key`` (first wins; deterministic order)."""
    seen: dict = {}
    for r in records:
        key = r["_dedupe_key"]
        if key not in seen:
            seen[key] = r
    return list(seen.values())


# --------------------------------------------------------------------------- #
# Selection.
# --------------------------------------------------------------------------- #
def select(css: list[dict], pbb: list[dict], max_n: int,
           min_d: int = 0, exact_only: bool = False,
           code_ids: set[str] | None = None) -> tuple[list[dict], list[dict]]:
    """Filter to ``n <= max_n`` (and optionally ``ours_d >= min_d`` / exact) and
    order exact-first, smallest-n-first.

    If ``code_ids`` is given, select exactly those codes by id and ignore the
    other filters (used for targeted reruns, e.g. the looser incumbents).
    """

    def order_key(r):
        return (r["n"], 0 if r["ours_exact"] else 1, r["ours_d"], r["code_id"])

    def keep(r):
        if code_ids is not None:
            return r["code_id"] in code_ids
        return (r["n"] <= max_n and r["ours_d"] >= min_d
                and (r["ours_exact"] or not exact_only))

    css_sel = sorted([r for r in css if keep(r)], key=order_key)
    pbb_sel = sorted([r for r in pbb if keep(r)], key=order_key)
    return css_sel, pbb_sel


def interleave(css: list[dict], pbb: list[dict]) -> list[dict]:
    """Alternate CSS/PBB so a --limit yields a balanced mix."""
    out, i, j = [], 0, 0
    while i < len(css) or j < len(pbb):
        if i < len(css):
            out.append(css[i]); i += 1
        if j < len(pbb):
            out.append(pbb[j]); j += 1
    return out


# --------------------------------------------------------------------------- #
# Running one (code, method, component).
# --------------------------------------------------------------------------- #
def _bz_eligible(rec: dict, bz_max_eff: int) -> bool:
    eff = rec["n"] if rec["code_type"] == "css" else 2 * rec["n"]
    return eff <= bz_max_eff


def _agreement(ours_d: int, ours_exact: bool, d_witness, witness_verified: bool,
               result_status: str) -> dict:
    """Compare a verified Webster witness weight against our distance."""
    if d_witness is None or not witness_verified:
        return {"matches": None, "lower_than_ours": None, "higher_than_ours": None,
                "comparison_strength": "unverified", "critical": False}
    matches = d_witness == ours_d
    lower = d_witness < ours_d
    higher = d_witness > ours_d
    if result_status == "proven_exact":
        strength = "proof"
    elif result_status in ("incumbent", "heuristic"):
        strength = "corroboration"
    else:
        strength = "unverified"
    # CRITICAL: a *verified* witness lighter than a distance we call exact means
    # our exact value is wrong (or the code mapping is). Highest-impact signal.
    critical = bool(lower and ours_exact)
    return {"matches": matches, "lower_than_ours": lower, "higher_than_ours": higher,
            "comparison_strength": strength, "critical": critical}


def build_code(rec: dict):
    if rec["code_type"] == "css":
        return build_bb_code(rec["ell"], rec["m"], rec["A_terms"], rec["B_terms"])
    return build_pbb_code(rec["ell"], rec["m"], rec["A_terms"], rec["B_terms"],
                          rec.get("C_terms"), rec.get("D_terms"))


def run_one(rec: dict, method: str, component: str | None,
            code, mip_maxtime: int, bz_timeout: int) -> dict:
    """Run one method (one component for CSS) and return a result record."""
    from evaluation.distance_webster import (
        webster_distance_css, webster_distance_noncss,
        verify_witness_css, verify_witness_noncss,
    )

    params, timeout_s = method_budget(method, mip_maxtime, bz_timeout)

    if rec["code_type"] == "css":
        res = webster_distance_css(code, method=method, component=component,
                                   seed=42, params=params, timeout_s=timeout_s)
        if res.L is not None and len(res.L):
            v = verify_witness_css(code, res.L, component, res.d)
        else:
            v = {}
        ours_d = rec["ours_d_z"] if component == "Z" else rec["ours_d_x"]
        ours_d = ours_d if ours_d is not None else rec["ours_d"]
    else:
        res = webster_distance_noncss(code, method=method, seed=42,
                                      params=params, timeout_s=timeout_s)
        v = verify_witness_noncss(code, res.L, res.d) if (res.L is not None and len(res.L)) else {}
        ours_d = rec["ours_d"]

    d_witness = v.get("weight")
    verified = bool(v.get("verified"))
    agree = _agreement(ours_d, rec["ours_exact"], d_witness, verified, res.result_status)

    # Store the witness compactly as its support (indices where L==1); far
    # smaller than base64 for sparse logicals and directly inspectable.
    support = [int(i) for i, b in enumerate(res.L) if int(b) % 2] if res.L is not None else []

    return {
        "code_id": rec["code_id"],
        "code_type": rec["code_type"],
        "ell": rec["ell"], "m": rec["m"], "n": rec["n"], "k": rec["k"],
        "ours": {
            "d": rec["ours_d"],
            "d_component": ours_d,
            "exact": rec["ours_exact"],
            "exact_source": rec.get("ours_exact_source"),
            "status": rec.get("ours_status") or rec.get("ours_trust"),
            "bposd": rec.get("ours_bposd") or rec.get("ours_d_bposd"),
        },
        "webster": {
            "method": method,
            "component": component,
            "d_reported": res.d,
            "d_witness": d_witness,        # authoritative
            "witness_verified": verified,
            "witness_commutes": v.get("commutes"),
            "witness_anticommutes": v.get("anticommutes_with_logical"),
            "weight_matches_reported_d": v.get("weight_matches_d"),
            "L_support": support,
            "L_len": (len(res.L) if res.L is not None else 0),
            "T": res.T, "R": res.R,
            "runtime_s": round(res.runtime_s, 2),
            "result_status": res.result_status,
            "optimality_source": res.optimality_source,
            "params": res.params,
            "progress": (res.progress or "")[:300],
            "error": res.error,
        },
        "source": {
            "file": rec.get("source"),
            "group": rec.get("source_group"),
            "index": rec.get("source_index"),
            "label": rec.get("source_label"),
            "bliss_hash": rec.get("bliss_hash"),
            "record_hash": record_fingerprint(rec),
        },
        "run": {
            "params_key": _canonical_json(params),
            "timeout_s": timeout_s,
        },
        "agreement": agree,
        "timestamp": _now(),
    }


def build_failed_record(rec: dict, method: str, component: str | None,
                        error: str, mip_maxtime: int, bz_timeout: int) -> dict:
    params, timeout_s = method_budget(method, mip_maxtime, bz_timeout)
    if rec["code_type"] == "css":
        ours_d = rec["ours_d_z"] if component == "Z" else rec["ours_d_x"]
        ours_d = ours_d if ours_d is not None else rec["ours_d"]
    else:
        ours_d = rec["ours_d"]
    return {
        "code_id": rec["code_id"],
        "code_type": rec["code_type"],
        "ell": rec["ell"], "m": rec["m"], "n": rec["n"], "k": rec["k"],
        "ours": {
            "d": rec["ours_d"],
            "d_component": ours_d,
            "exact": rec["ours_exact"],
            "exact_source": rec.get("ours_exact_source"),
            "status": rec.get("ours_status") or rec.get("ours_trust"),
            "bposd": rec.get("ours_bposd") or rec.get("ours_d_bposd"),
        },
        "webster": {
            "method": method,
            "component": component,
            "d_reported": -1,
            "d_witness": None,
            "witness_verified": False,
            "witness_commutes": False,
            "witness_anticommutes": False,
            "weight_matches_reported_d": False,
            "L_support": [],
            "L_len": 0,
            "T": 0, "R": 0,
            "runtime_s": 0,
            "result_status": "failed",
            "optimality_source": "build_failed",
            "params": params,
            "progress": "",
            "error": error,
        },
        "source": {
            "file": rec.get("source"),
            "group": rec.get("source_group"),
            "index": rec.get("source_index"),
            "label": rec.get("source_label"),
            "bliss_hash": rec.get("bliss_hash"),
            "record_hash": record_fingerprint(rec),
        },
        "run": {
            "params_key": _canonical_json(params),
            "timeout_s": timeout_s,
        },
        "agreement": _agreement(ours_d, rec["ours_exact"], None, False, "failed"),
        "timestamp": _now(),
    }


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #
def done_key_from_result(r: dict) -> tuple:
    params_key = (r.get("run") or {}).get("params_key")
    if params_key is None:
        params_key = _canonical_json(r["webster"].get("params") or {})
    record_hash = (r.get("source") or {}).get("record_hash")
    return (
        r["code_id"],
        r["webster"]["method"],
        r["webster"]["component"],
        params_key,
        record_hash,
    )


def job_key(rec: dict, method: str, component: str | None,
            mip_maxtime: int, bz_timeout: int) -> tuple:
    params, _timeout_s = method_budget(method, mip_maxtime, bz_timeout)
    return (
        rec["code_id"],
        method,
        component,
        _canonical_json(params),
        record_fingerprint(rec),
    )


def load_done(path: Path) -> set[tuple]:
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            done.add(done_key_from_result(r))
    return done


def is_done(done: set[tuple], key: tuple) -> bool:
    """Return true for exact key matches or legacy rows missing record hashes."""
    return key in done or (*key[:4], None) in done


def jobs_for(rec: dict, methods: list[str], bz_max_eff: int):
    """Yield (method, component) jobs for a code."""
    comps = ["Z", "X"] if rec["code_type"] == "css" else [None]
    for method in methods:
        if method == "BZDistMW" and not _bz_eligible(rec, bz_max_eff):
            continue
        for comp in comps:
            yield method, comp


def normalize_result_record(r: dict) -> dict:
    """Apply the current conservative interpretation to one result record.

    This keeps older JSONL rows usable after review fixes: MIPDist is never
    promoted to a proof from wall-clock progress, and CSS exactness is not
    inferred from the display label in ``ilp_catalog.json``.
    """
    w = r.get("webster") or {}
    status, opt_source = effective_result_status(w)
    w["result_status"] = status
    w["optimality_source"] = opt_source

    ours = r.get("ours") or {}
    if r.get("code_type") == "css":
        css_exact = str(r.get("code_id", "")).startswith("css_bravyi_baselines_")
        ours["exact"] = css_exact
        ours["exact_source"] = (
            "bravyi_baseline" if css_exact else "not_certified_in_ilp_catalog"
        )

    ours_d = ours.get("d_component", ours.get("d"))
    d_witness = w.get("d_witness")
    verified = bool(w.get("witness_verified"))
    if ours_d is not None:
        r["agreement"] = _agreement(
            int(ours_d), bool(ours.get("exact")), d_witness, verified, status
        )
    r["webster"] = w
    r["ours"] = ours
    return r


def normalize_results_file(path: Path) -> int:
    """Rewrite a results JSONL file with current status/exactness rules."""
    if not path.exists():
        return 0
    before = path.read_text().splitlines()
    after = []
    changed = 0
    for line in before:
        if not line.strip():
            continue
        original = json.loads(line)
        normalized = normalize_result_record(json.loads(line))
        if normalized != original:
            changed += 1
        after.append(json.dumps(normalized))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(after) + ("\n" if after else ""))
    tmp.replace(path)
    return changed


def write_summary(results_path: Path, summary_path: Path) -> dict:
    """Aggregate per-(code,method,component) results into per-code verdicts."""
    by_code: dict[str, list[dict]] = {}
    for line in results_path.read_text().splitlines():
        if line.strip():
            r = normalize_result_record(json.loads(line))
            by_code.setdefault(r["code_id"], []).append(r)

    tally = {"codes": 0, "critical": 0, "disagree_nonexact": 0, "all_agree": 0,
             "proofs": 0}
    with summary_path.open("w") as fh:
        for code_id, recs in sorted(by_code.items()):
            r0 = recs[0]
            ours_d = r0["ours"]["d"]
            ours_exact = r0["ours"]["exact"]
            # Code-level Webster distance = min verified witness weight across all
            # methods/components (per method, then overall).
            verified = [x for x in recs if x["webster"]["witness_verified"]]
            wmin = min((x["webster"]["d_witness"] for x in verified), default=None)
            critical = any(x["agreement"]["critical"] for x in recs)
            recs_with_status = [
                (x, *effective_result_status(x["webster"]))
                for x in recs
            ]
            has_proof = any(status == "proven_exact" and x["webster"]["witness_verified"]
                            for x, status, _opt_source in recs_with_status)
            # Did the verified min match our claimed distance?
            agree = (wmin == ours_d) if wmin is not None else None
            verdict = (
                "CRITICAL_LOWER" if critical else
                "no_witness" if wmin is None else
                "agree" if agree else
                ("webster_lower" if wmin < ours_d else "webster_higher")
            )
            rec = {
                "code_id": code_id, "code_type": r0["code_type"],
                "n": r0["n"], "k": r0["k"],
                "ours_d": ours_d, "ours_exact": ours_exact,
                "webster_min_verified_d": wmin,
                "webster_d_per_method": {
                    f'{x["webster"]["method"]}/{x["webster"]["component"]}/'
                    f'{_canonical_json(x["webster"].get("params") or {})}/'
                    f'{x.get("timestamp", "no_ts")}':
                    {"d": x["webster"]["d_witness"],
                     "status": status,
                     "optimality_source": opt_source,
                     "verified": x["webster"]["witness_verified"]}
                    for x, status, opt_source in recs_with_status
                },
                "webster_runs": [
                    {
                        "method": x["webster"]["method"],
                        "component": x["webster"]["component"],
                        "params": x["webster"].get("params") or {},
                        "d": x["webster"]["d_witness"],
                        "status": status,
                        "optimality_source": opt_source,
                        "verified": x["webster"]["witness_verified"],
                        "timestamp": x.get("timestamp"),
                    }
                    for x, status, opt_source in recs_with_status
                ],
                "agree_with_ours": agree,
                "has_independent_proof": has_proof,
                "verdict": verdict,
            }
            fh.write(json.dumps(rec) + "\n")
            tally["codes"] += 1
            if critical:
                tally["critical"] += 1
            if has_proof:
                tally["proofs"] += 1
            if agree is True:
                tally["all_agree"] += 1
            elif agree is False and not ours_exact:
                tally["disagree_nonexact"] += 1
    return tally


def write_selection(args, max_n: int, css_sel: list[dict], pbb_sel: list[dict]) -> Path:
    requested_code_ids = sorted(set(args.code_ids)) if args.code_ids else None
    args_payload = {
        "css_only": args.css_only,
        "pbb_only": args.pbb_only,
        "all_n": args.all_n,
        "max_n": args.max_n,
        "min_d": args.min_d,
        "exact_only": args.exact_only,
        "selection_mode": "code_ids" if requested_code_ids else "filters",
        "code_ids": requested_code_ids,
    }
    selection_id = hashlib.sha256(_canonical_json({
        "args": args_payload,
        "css_ids": [r["code_id"] for r in css_sel],
        "pbb_ids": [r["code_id"] for r in pbb_sel],
    }).encode()).hexdigest()[:10]
    payload = {
        "generated_at": _now(),
        "selection_id": selection_id,
        "max_n": max_n,
        "args": args_payload,
        "css_count": len(css_sel),
        "pbb_count": len(pbb_sel),
        "css": css_sel,
        "pbb": pbb_sel,
    }
    snapshot = INVEST / f"webster_tier1_codes_{selection_id}.json"
    snapshot.write_text(json.dumps(payload, indent=1) + "\n")
    latest = dict(payload)
    latest["snapshot_file"] = snapshot.name
    SELECTION_OUT.write_text(json.dumps(latest, indent=1) + "\n")
    with SELECTION_HISTORY_OUT.open("a") as fh:
        fh.write(json.dumps({
            "generated_at": payload["generated_at"],
            "selection_id": selection_id,
            "snapshot_file": snapshot.name,
            "args": args_payload,
            "css_count": len(css_sel),
            "pbb_count": len(pbb_sel),
        }) + "\n")
    return snapshot


def main() -> int:
    ap = argparse.ArgumentParser(description="Tier 1 Webster cross-check runner.")
    ap.add_argument("--normalize-results", action="store_true",
                    help="Rewrite webster_results.jsonl with current conservative "
                         "status/exactness rules before continuing.")
    ap.add_argument("--summary-only", action="store_true",
                    help="Regenerate webster_summary.jsonl from existing results and stop.")
    ap.add_argument("--select-only", action="store_true", help="Resolve + write selection, then stop.")
    ap.add_argument("--limit", type=int, default=24, help="Max codes to run (0 = all selected).")
    ap.add_argument("--methods", nargs="+", default=["MIPDist", "BZDistMW"],
                    choices=["MIPDist", "BZDistMW"])
    ap.add_argument("--css-only", action="store_true")
    ap.add_argument("--pbb-only", action="store_true")
    ap.add_argument("--all-n", action="store_true", help="Include n>144 (288, 360).")
    ap.add_argument("--max-n", type=int, default=144)
    ap.add_argument("--min-d", type=int, default=0, help="Only codes with ours_d >= this.")
    ap.add_argument("--exact-only", action="store_true", help="Only codes we mark exact.")
    ap.add_argument("--code-ids", nargs="+", default=None,
                    help="Run exactly these code_ids (ignores n/d/exact filters and --limit).")
    ap.add_argument("--bz-max-eff", type=int, default=80,
                    help="Run BZDistMW only when effective length (n CSS / 2n PBB) <= this.")
    ap.add_argument("--mip-maxtime", type=int, default=300, help="MIPDist maxTime (s).")
    ap.add_argument("--bz-timeout", type=int, default=120, help="BZDistMW outer timeout (s).")
    ap.add_argument("--workers", type=int, default=1,
                    help="Concurrent jobs. Each solve is single-threaded, so set "
                         "this to ~#cores. Jobs are independent; scaling is near-linear.")
    args = ap.parse_args()

    if args.normalize_results:
        changed = normalize_results_file(RESULTS_OUT)
        print(f"Normalized {changed} records in {RESULTS_OUT.relative_to(REPO)}")

    if args.summary_only:
        tally = write_summary(RESULTS_OUT, SUMMARY_OUT)
        print(f"Summary -> {SUMMARY_OUT.relative_to(REPO)}")
        print(f"  codes={tally['codes']} agree={tally['all_agree']} "
              f"independent_proofs={tally['proofs']} "
              f"CRITICAL={tally['critical']} disagree(non-exact)={tally['disagree_nonexact']}")
        return 0

    max_n = 10_000 if args.all_n else args.max_n

    css = dedupe(load_css_pool()) if not args.pbb_only else []
    pbb = dedupe(load_pbb_pool()) if not args.css_only else []
    code_ids = set(args.code_ids) if args.code_ids else None
    css_sel, pbb_sel = select(css, pbb, max_n, min_d=args.min_d,
                              exact_only=args.exact_only, code_ids=code_ids)
    if code_ids:
        selected_ids = {r["code_id"] for r in css_sel + pbb_sel}
        missing = sorted(code_ids - selected_ids)
        if missing:
            print("ERROR: --code-ids did not match any selected code: "
                  + ", ".join(missing), file=sys.stderr)
            print("       Check the ids and any --css-only/--pbb-only pool filter.",
                  file=sys.stderr)
            return 2

    selection_snapshot = write_selection(args, max_n, css_sel, pbb_sel)
    scope = "--code-ids targeted set" if code_ids else f"n<={max_n}"
    print(f"Selection: {len(css_sel)} CSS + {len(pbb_sel)} PBB codes ({scope}) "
          f"-> {SELECTION_OUT.relative_to(REPO)} "
          f"(snapshot {selection_snapshot.relative_to(REPO)})")
    print(f"  CSS exact: {sum(r['ours_exact'] for r in css_sel)} | "
          f"PBB exact: {sum(r['ours_exact'] for r in pbb_sel)}")
    if args.select_only:
        return 0

    worklist = interleave(css_sel, pbb_sel)
    if args.limit and not code_ids:  # --code-ids runs exactly the named set
        worklist = worklist[: args.limit]

    # Flatten to independent (code, method, component) jobs, skipping any already
    # in the output (resumable). Each job is CPU-bound and single-threaded, so we
    # dispatch up to --workers of them concurrently.
    done = load_done(RESULTS_OUT)
    jobs = [
        (rec, method, comp)
        for rec in worklist
        for method, comp in jobs_for(rec, args.methods, args.bz_max_eff)
        if not is_done(done, job_key(rec, method, comp, args.mip_maxtime, args.bz_timeout))
    ]
    print(f"Running {len(jobs)} jobs over {len(worklist)} codes with methods {args.methods} "
          f"(BZ only when eff<= {args.bz_max_eff}); workers={args.workers}; "
          f"resumable -> {RESULTS_OUT.relative_to(REPO)}")

    def do_job(job):
        rec, method, comp = job
        try:
            code = build_code(rec)
        except Exception as exc:  # noqa: BLE001
            return ("build_failed", rec, method, comp, repr(exc))
        return ("ok", run_one(rec, method, comp, code, args.mip_maxtime, args.bz_timeout))

    def emit(fh, n_done, outcome):
        # Single writer (main thread) -> no lock needed.
        if outcome[0] == "build_failed":
            _, rec, method, comp, exc = outcome
            r = build_failed_record(rec, method, comp, exc, args.mip_maxtime, args.bz_timeout)
            fh.write(json.dumps(r) + "\n"); fh.flush()
            print(f"[{n_done}/{len(jobs)}] {rec['code_id']} {method}/{comp}: BUILD FAILED {exc}")
            return
        r = outcome[1]
        fh.write(json.dumps(r) + "\n"); fh.flush()
        w, a = r["webster"], r["agreement"]
        flag = " *** CRITICAL ***" if a.get("critical") else ""
        print(f"[{n_done}/{len(jobs)}] {r['code_id']} {w['method']}/{w['component']}: "
              f"ours_d={r['ours']['d_component']} webster_d={w['d_witness']} "
              f"({w['result_status']}, verified={w['witness_verified']}, "
              f"{w['runtime_s']}s){flag}")

    t_start = time.time()
    with RESULTS_OUT.open("a") as fh:
        if args.workers <= 1:
            for i, job in enumerate(jobs, 1):
                emit(fh, i, do_job(job))
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            # Threads block on the adapter's per-call subprocess, so the GIL is
            # not contended: each in-flight job is its own OS process on its own
            # core, with the hard-kill timeout preserved.
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(do_job, job): job for job in jobs}
                for i, fut in enumerate(as_completed(futs), 1):
                    emit(fh, i, fut.result())

    tally = write_summary(RESULTS_OUT, SUMMARY_OUT)
    dt = time.time() - t_start
    print(f"\nDone in {dt:.0f}s. Summary -> {SUMMARY_OUT.relative_to(REPO)}")
    print(f"  codes={tally['codes']} agree={tally['all_agree']} "
          f"independent_proofs={tally['proofs']} "
          f"CRITICAL={tally['critical']} disagree(non-exact)={tally['disagree_nonexact']}")
    if tally["critical"]:
        print("  !!! CRITICAL disagreements found: a verified witness is lighter than "
              "a distance we call exact. Inspect webster_summary.jsonl (verdict=CRITICAL_LOWER).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
