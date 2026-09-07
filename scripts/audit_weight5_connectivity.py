"""Connectivity/decomposition audit for the weight-5 CSS and PBB campaigns.

Rebuilds every certified weight-5 code from its stored polynomial terms
(never from guessed/derived parameters) and runs it through
``evaluation.connectivity`` to compute its generator-basis-invariant
stabilizer-row-space partition and compare it with the stored-presentation
Tanner partition. A disconnected ``[[n,k,d]]`` code is a direct sum of ``m``
structurally independent components; when the components are pairwise
isomorphic (checked rigorously via BLISS canonical hashing), the code is
exactly ``m x [[n/m, k/m, d]]`` -- not a genuinely new, larger, connected
construction, even though FOM = k*d^2/n is unchanged by the collapse.

Inputs (weight-5 campaign artifacts only -- see CLAUDE.md scope note):
  - results/weight5_css_small_v1_milp_verified.jsonl
  - results/weight5_css_large_v1_milp_verified.jsonl
  - results/weight5_pbb_small_v1_deep_milp_verify.jsonl
  - results/weight5_pbb_large_v1_deep_milp_verify.jsonl
  - the weight-5 subset of results/pareto_front_v2.json's exact_front
    (identified by exact (ell,m,A_terms,B_terms) match against the CSS
    files above -- cross-checked to be all 13 entries)

Output: results/weight5_connectivity_audit.jsonl, one structural row per
certification-target input record, with the original identity fields plus
n_components/component_sizes/is_connected/homogeneous/component_k/
k_additivity_ok/base_n/base_k/components_isomorphic/presentation_agrees and
source_file.  Distance fields are deliberately omitted because some legacy
verification rows retain stale pre-continuation values; distance evidence is
merged separately by the paper/table generators.

Does NOT modify results/pareto_front_v2.json or evaluation/pareto_v2.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.bb_code import build_bb_code
from evaluation.pbb_code import build_pbb_code
from evaluation.connectivity import decompose, components_isomorphic

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

CSS_FILES = [
    "weight5_css_small_v1_milp_verified.jsonl",
    "weight5_css_large_v1_milp_verified.jsonl",
]
PBB_FILES = [
    "weight5_pbb_small_v1_deep_milp_verify.jsonl",
    "weight5_pbb_large_v1_deep_milp_verify.jsonl",
]

OUTPUT_PATH = RESULTS_DIR / "weight5_connectivity_audit.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _terms(raw) -> list[tuple[int, int]]:
    return [tuple(t) for t in raw]


def _poly_key(rec: dict) -> tuple:
    return (
        rec["ell"],
        rec["m"],
        tuple(map(tuple, rec["A_terms"])),
        tuple(map(tuple, rec["B_terms"])),
    )


def _audit_record(rec: dict, *, is_css: bool, source_file: str) -> dict:
    ell, m = rec["ell"], rec["m"]
    A = _terms(rec["A_terms"])
    B = _terms(rec["B_terms"])
    if is_css:
        code = build_bb_code(ell, m, A, B)
    else:
        C = _terms(rec["C_terms"]) if rec.get("C_terms") else None
        D = _terms(rec["D_terms"]) if rec.get("D_terms") else None
        code = build_pbb_code(ell, m, A, B, C, D)

    if code.num_qudits != rec["n"] or code.dimension != rec["k"]:
        raise ValueError(
            f"Rebuilt code [[{code.num_qudits},{code.dimension}]] does not match "
            f"stored record [[{rec['n']},{rec['k']}]] for {source_file}: {rec}"
        )

    result = decompose(code)
    iso = components_isomorphic(code) if not result.is_connected else None

    return {
        "source_file": source_file,
        "code_family": "CSS" if is_css else "PBB",
        "ell": ell,
        "m": m,
        "A_terms": rec["A_terms"],
        "B_terms": rec["B_terms"],
        "C_terms": rec.get("C_terms"),
        "D_terms": rec.get("D_terms"),
        "n": rec["n"],
        "k": rec["k"],
        "catalogue_relevant": int(rec["d"]) > 2,
        "n_components": result.n_components,
        "component_sizes": result.component_sizes,
        "is_connected": result.is_connected,
        "homogeneous": result.homogeneous,
        "component_k": result.component_k,
        "k_additivity_ok": result.k_additivity_ok,
        "base_n": result.base_n,
        "base_k": result.base_k,
        "components_isomorphic": iso,
        "presentation_agrees": result.presentation_agrees,
    }


def _load_exact_front_weight5_subset(css_keys: set[tuple]) -> list[dict]:
    pareto_path = RESULTS_DIR / "pareto_front_v2.json"
    archive = json.loads(pareto_path.read_text())
    matched = [e for e in archive["exact_front"] if _poly_key(e) in css_keys]
    return matched


def main() -> None:
    css_records = []
    for fname in CSS_FILES:
        for rec in _load_jsonl(RESULTS_DIR / fname):
            css_records.append((rec, fname))

    pbb_records = []
    for fname in PBB_FILES:
        for rec in _load_jsonl(RESULTS_DIR / fname):
            pbb_records.append((rec, fname))

    css_keys = {_poly_key(rec) for rec, _ in css_records}

    audit_rows = []
    for rec, fname in css_records:
        audit_rows.append(_audit_record(rec, is_css=True, source_file=fname))
    for rec, fname in pbb_records:
        audit_rows.append(_audit_record(rec, is_css=False, source_file=fname))

    with open(OUTPUT_PATH, "w") as f:
        for row in audit_rows:
            f.write(json.dumps(row) + "\n")

    css_rows = [r for r in audit_rows if r["code_family"] == "CSS"]
    pbb_rows = [r for r in audit_rows if r["code_family"] == "PBB"]
    css_disc = [r for r in css_rows if not r["is_connected"]]
    pbb_disc = [r for r in pbb_rows if not r["is_connected"]]

    print(f"All certification inputs: CSS {len(css_disc)}/{len(css_rows)} "
          f"disconnected ({100 * len(css_disc) / len(css_rows):.1f}%); "
          f"PBB {len(pbb_disc)}/{len(pbb_rows)} disconnected "
          f"({100 * len(pbb_disc) / len(pbb_rows):.1f}%)")
    retained_css = [r for r in css_rows if r["catalogue_relevant"]]
    retained_pbb = [r for r in pbb_rows if r["catalogue_relevant"]]
    retained_css_disc = [r for r in retained_css if not r["is_connected"]]
    retained_pbb_disc = [r for r in retained_pbb if not r["is_connected"]]
    print(
        f"Retained d>2 targets: CSS {len(retained_css_disc)}/{len(retained_css)} "
        f"disconnected ({100 * len(retained_css_disc) / len(retained_css):.1f}%); "
        f"PBB {len(retained_pbb_disc)}/{len(retained_pbb)} disconnected "
        f"({100 * len(retained_pbb_disc) / len(retained_pbb):.1f}%)"
    )

    disconnected_rows = css_disc + pbb_disc
    non_iso = [
        r for r in disconnected_rows if r["components_isomorphic"] is False
    ]
    unresolved_iso = [
        r for r in disconnected_rows if r["components_isomorphic"] is None
    ]
    if non_iso:
        print(f"WARNING: {len(non_iso)} disconnected records have "
              f"NON-isomorphic components (= m x [base] framing breaks down):")
        for r in non_iso:
            print(f"  [[{r['n']},{r['k']}]] ell={r['ell']} m={r['m']} "
                  f"sizes={r['component_sizes']}")
    if unresolved_iso:
        print(
            f"WARNING: {len(unresolved_iso)} disconnected records have "
            "unresolved component isomorphism:"
        )
        for r in unresolved_iso:
            print(f"  [[{r['n']},{r['k']}]] ell={r['ell']} m={r['m']} "
                  f"sizes={r['component_sizes']}")
    if not non_iso and not unresolved_iso:
        print(f"All {len(disconnected_rows)} disconnected records have "
              "pairwise-isomorphic components (= m x [base] framing holds exactly).")

    bad_k = [r for r in audit_rows
             if r["n_components"] > 1 and r["k_additivity_ok"] is False]
    if bad_k:
        print(f"WARNING: {len(bad_k)} disconnected records fail k-additivity "
              f"(component dimensions don't sum to reported k):")
        for r in bad_k:
            print(f"  [[{r['n']},{r['k']}]] component_k={r['component_k']}")

    presentation_mismatches = [r for r in audit_rows if not r["presentation_agrees"]]
    if presentation_mismatches:
        print(
            f"WARNING: {len(presentation_mismatches)} records differ between "
            "stored-Tanner and invariant row-space partitions"
        )
    else:
        print("Stored-Tanner and invariant row-space partitions agree for all inputs.")

    print()
    print("Weight-5 subset of results/pareto_front_v2.json exact_front:")
    ef_rows = _load_exact_front_weight5_subset(css_keys)
    print(f"  matched {len(ef_rows)}/13 exact_front entries as weight-5 CSS records")
    ef_disc = 0
    for e in ef_rows:
        A = _terms(e["A_terms"])
        B = _terms(e["B_terms"])
        code = build_bb_code(e["ell"], e["m"], A, B)
        result = decompose(code, d=e["d"])
        status = "connected" if result.is_connected else (
            f"= {result.n_components} x [[{result.base_n},{result.base_k},{result.base_d}]]"
        )
        if not result.is_connected:
            ef_disc += 1
        print(f"  [[{e['n']},{e['k']},{e['d']}]] ell={e['ell']} m={e['m']}: {status}")
    print(f"  {ef_disc}/{len(ef_rows)} disconnected")

    print()
    print(f"Wrote {len(audit_rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
