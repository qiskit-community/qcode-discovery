#!/usr/bin/env python
"""Tier 2 (Comparison B): does QDistEvol recover the MILP value where BP-OSD
overestimated?

The paper hypothesizes (paper.tex:700) that Webster's QDistEvol "may reduce the
overestimation gap" left by BP-OSD. This script measures it directly on the codes
where BP-OSD most overestimated, by running two Webster heuristics --

  * decoderDist : Webster's BP-OSD (via the `ldpc` package) -- the same algorithm
                  class as our pipeline's estimator;
  * QDistEvol   : Webster's headline evolutionary search --

and comparing each method's *verified witness weight* (recomputed + checked via
evaluation.distance_webster.verify_witness_*) to:
  - our recorded MILP value (exact only when the source exact flag is set;
    otherwise an incumbent upper bound), and
  - our BP-OSD value (the overestimate we are trying to beat).

Both methods are heuristic upper bounds, so the question per code is simply: how
close does each get to the MILP value? "Gap closed" = witness weight reaches
(<=) the MILP value.

Golden set:
  * CSS  : the 9 `status=="BP-OSD OVERESTIMATED"` rows of results/ilp_catalog.json
           (ilp_d was independently corroborated by Webster MIPDist already, but
           remains an incumbent unless separately certified).
  * PBB  : the largest d_bposd - d_milp gaps in campaign7_publication_merged.jsonl
           (deduped by bliss_hash), where decoderDist often returns no verified
           logical and the contrast with QDistEvol is sharpest.

Output: investigations/tier2_results.jsonl (one row per code) + a printed table.
Reuses the Tier-0/1 adapter; touches none of the committed runner.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluation.bb_code import build_bb_code  # noqa: E402
from evaluation.pbb_code import build_pbb_code  # noqa: E402
from evaluation.distance_webster import (  # noqa: E402
    webster_distance_css, webster_distance_noncss,
    verify_witness_css, verify_witness_noncss,
)

CSS_CATALOG = REPO / "results" / "ilp_catalog.json"
PBB_CATALOG = REPO / "results" / "campaign7_publication_merged.jsonl"
OUT = REPO / "investigations" / "tier2_results.jsonl"

METHODS = ["decoderDist", "QDistEvol"]
HEUR_MAXTIME = 120        # Webster internal time cap per call (s)
ITERCOUNT = 20000
OUTER_TIMEOUT = 180       # hard kill (s)


def css_overestimate() -> list[dict]:
    cat = json.loads(CSS_CATALOG.read_text())
    out = []
    for grp, rows in cat.items():
        for i, r in enumerate(rows):
            if r.get("status") != "BP-OSD OVERESTIMATED":
                continue
            out.append({
                "code_id": f"css_{grp}_{r['ell']}x{r['m']}_{i}",
                "type": "css", "ell": int(r["ell"]), "m": int(r["m"]),
                "n": int(r["label"].split(",")[0].lstrip("[")),
                "k": int(r["label"].split(",")[1]),
                "A": r["A"], "B": r["B"],
                "source": {
                    "file": "ilp_catalog.json",
                    "group": grp,
                    "index": i,
                    "label": r.get("label"),
                    "status": r.get("status"),
                },
                "milp_d": int(r["ilp_d"]), "milp_exact": False,  # catalog incumbents
                "bposd_d": int(r["bp_osd_d"]),
            })
    return out


def pbb_overestimate(topk: int = 8) -> list[dict]:
    rows = [json.loads(l) for l in PBB_CATALOG.read_text().splitlines() if l.strip()]
    cand = [r for r in rows if r.get("d_milp") and r.get("d_bposd")
            and r["d_bposd"] > r["d_milp"]]
    seen = {}
    for r in sorted(cand, key=lambda r: -(r["d_bposd"] - r["d_milp"])):
        h = r.get("bliss_hash") or (r["ell"], r["m"], str(r["A_terms"]))
        if h not in seen:
            seen[h] = r
    picked = sorted(seen.values(), key=lambda r: -(r["d_bposd"] - r["d_milp"]))[:topk]
    return [{
        "code_id": r.get("code_id") or f"pbb_{r['ell']}x{r['m']}_{r.get('bliss_hash', 'nohash')}",
        "type": "pbb", "ell": int(r["ell"]), "m": int(r["m"]),
        "n": int(r["n"]), "k": int(r["k"]),
        "A": r["A_terms"], "B": r["B_terms"],
        "C": r.get("C_terms"), "D": r.get("D_terms"),
        "source": {
            "file": "campaign7_publication_merged.jsonl",
            "source_index": r.get("source_index"),
            "bliss_hash": r.get("bliss_hash"),
            "d_is_upper_bound": bool(r.get("d_is_upper_bound")),
        },
        "milp_d": int(r["d_milp"]),
        "milp_exact": bool(r.get("milp_exact") or r.get("d_is_exact")),
        "bposd_d": int(r["d_bposd"]),
    } for r in picked]


def _params(method):
    return {"iterCount": ITERCOUNT, "maxTime": HEUR_MAXTIME}


def run_method(rec: dict, method: str) -> dict:
    """Run one heuristic on one code; return min verified witness weight."""
    t0 = time.time()
    if rec["type"] == "css":
        code = build_bb_code(rec["ell"], rec["m"], rec["A"], rec["B"])
        weights, statuses, optimality_sources = [], [], []
        for comp in ("Z", "X"):
            res = webster_distance_css(code, method=method, component=comp, seed=42,
                                       params=_params(method), timeout_s=OUTER_TIMEOUT)
            statuses.append(res.result_status)
            optimality_sources.append(res.optimality_source)
            if res.L is not None and len(res.L):
                v = verify_witness_css(code, res.L, comp, res.d)
                if v["verified"]:
                    weights.append(v["weight"])
        wmin = min(weights) if weights else None
    else:
        code = build_pbb_code(rec["ell"], rec["m"], rec["A"], rec["B"],
                              rec.get("C"), rec.get("D"))
        res = webster_distance_noncss(code, method=method, seed=42,
                                      params=_params(method), timeout_s=OUTER_TIMEOUT)
        statuses = [res.result_status]
        optimality_sources = [res.optimality_source]
        wmin = None
        if res.L is not None and len(res.L):
            v = verify_witness_noncss(code, res.L, res.d)
            if v["verified"]:
                wmin = v["weight"]
                weights = [wmin]
            else:
                weights = []
        else:
            weights = []
    return {
        "d": wmin,
        "statuses": statuses,
        "optimality_sources": optimality_sources,
        "verified_witnesses": len(weights),
        "runtime_s": round(time.time() - t0, 1),
    }


def main():
    codes = css_overestimate() + pbb_overestimate(topk=8)
    print(f"Tier 2: {len(codes)} codes "
          f"({sum(c['type']=='css' for c in codes)} CSS + "
          f"{sum(c['type']=='pbb' for c in codes)} PBB), methods={METHODS}\n")
    jobs = [(c, m) for c in codes for m in METHODS]
    results = {}  # code_id -> {method: {...}}
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(run_method, c, m): (c, m) for c, m in jobs}
        for fut in as_completed(futs):
            c, m = futs[fut]
            results.setdefault(c["code_id"], {})[m] = fut.result()

    rows = []
    print(f"{'code':22s} {'n':>4} {'k':>3} {'MILP':>4} {'BPOSD':>5} "
          f"{'decoderDist':>11} {'QDistEvol':>9}  gap-closed?")
    for c in codes:
        r = results[c["code_id"]]
        dd = r["decoderDist"]["d"]
        qe = r["QDistEvol"]["d"]
        qe_closed = qe is not None and qe <= c["milp_d"]
        dd_closed = dd is not None and dd <= c["milp_d"]
        rows.append({
            "code_id": c["code_id"], "type": c["type"], "n": c["n"], "k": c["k"],
            "source": c["source"],
            "milp_d": c["milp_d"], "milp_exact": c["milp_exact"], "bposd_d": c["bposd_d"],
            "decoderDist_d": dd, "qdistevol_d": qe,
            "qdistevol_closed_gap": qe_closed, "decoderdist_closed_gap": dd_closed,
            "decoderDist_statuses": r["decoderDist"]["statuses"],
            "qdistevol_statuses": r["QDistEvol"]["statuses"],
            "decoderDist_optimality_sources": r["decoderDist"]["optimality_sources"],
            "qdistevol_optimality_sources": r["QDistEvol"]["optimality_sources"],
            "decoderDist_verified_witnesses": r["decoderDist"]["verified_witnesses"],
            "qdistevol_verified_witnesses": r["QDistEvol"]["verified_witnesses"],
            "decoderDist_runtime_s": r["decoderDist"]["runtime_s"],
            "qdistevol_runtime_s": r["QDistEvol"]["runtime_s"],
        })
        mark = "QE✓" if qe_closed else "QE✗"
        mark += " DD✓" if dd_closed else " DD✗"
        print(f"{c['code_id']:22s} {c['n']:>4} {c['k']:>3} {c['milp_d']:>4} "
              f"{c['bposd_d']:>5} {str(dd):>11} {str(qe):>9}  {mark}")

    OUT.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    nqe = sum(r["qdistevol_closed_gap"] for r in rows)
    ndd = sum(r["decoderdist_closed_gap"] for r in rows)
    print(f"\nGap closed (witness weight <= MILP value):  "
          f"QDistEvol {nqe}/{len(rows)}   decoderDist {ndd}/{len(rows)}")
    print(f"-> {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
