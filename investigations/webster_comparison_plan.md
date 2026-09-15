# Webster `codedistance` vs. our BP-OSD / MILP — comparison plan

**Status:** executed once; current version incorporates the 2026-06-09 review fixes.
**Output destination:** `investigations/` (exploratory, not publication-grade).
**Reference:** Webster, Jacob, Higgott, "Distance algorithms for classical codes, quantum codes and circuits," arXiv:2603.22532 (`pip install codedistance`).
**Codes under test:** CSS rows from [results/ilp_catalog.json](../results/ilp_catalog.json) and non-CSS PBB rows from [results/campaign7_publication_merged.jsonl](../results/campaign7_publication_merged.jsonl). `ilp_catalog.json` is the CSS row-level MILP catalog used by the paper tables; `campaign7_publication_merged.jsonl` carries BLISS hashes and exact/upper-bound flags for PBB.

---

## 1. Why this is worth doing

The paper already cites Webster et al. four times — at [paper.tex:128](../paper/2606.02418/paper.tex#L128), [paper.tex:317](../paper/2606.02418/paper.tex#L317), [paper.tex:322](../paper/2606.02418/paper.tex#L322), and [paper.tex:699](../paper/2606.02418/paper.tex#L699) — as concurrent work that confirms BP-OSD overestimation and offers `QDistEvol` as a possible remedy. Running their package on our catalog gives three useful cross-checks:

1. **Exact MILP is exact.** Records with a source-specific exact flag (`d_is_exact=True`, `milp_exact=True`, or `trust_level=="EXACT"` with no upper-bound flag) claim a *proven* minimum weight via our MILP pipeline. Webster's `BZDistMW` (Brouwer-Zimmermann enumeration) is an independent exact method when it completes. Webster's `MIPDist` is an independent solver formulation, but its public result dictionary does not expose OR-tools optimality status; treat it as a witness-producing upper-bound cross-check, not a proof. If Webster returns a verified witness lower than any of our exact distances, one side has a formulation bug. This is the highest-value test in the entire plan.
2. **BP-OSD ensemble at 150k trials is well-calibrated.** [results/ensemble_verification_150k.json](../results/ensemble_verification_150k.json) is our strongest stochastic upper bound. `QDistEvol` is Webster's headline heuristic. The paper hypothesizes ([paper.tex:700](../paper/2606.02418/paper.tex#L700)) that QDistEvol "may reduce the overestimation gap" — but we have not measured this on our codes. This is exactly the experiment the paper points at.
3. **The non-CSS symplectic MILP linearization is correct.** Our non-CSS exact distance (`evaluation/distance_milp.py`) uses a binary-OR linearization of per-qubit support. Webster's non-CSS pipeline (`codeDistance(H, L, tB=2)`) takes a symplectic two-block matrix and uses a different internal formulation. Agreement on small non-CSS codes is the cleanest cross-check we can run on the linearization.

The user explicitly chose **investigations/ exploratory** as the destination — so this is "look first, decide later," not a publication-grade verification sweep. That argues for tight scope and rich per-code logging over breadth.

---

## 2. The package, in one screen

`pip install codedistance` exposes:

```python
codedistance.CSScodeDistance(Hx, Hz, method, params, component, seed) -> dict
codedistance.codeDistance(H, L, tB, method, params, seed) -> dict
```

**Inputs.**
- CSS: separate binary `Hx`, `Hz` (numpy `int8`, may be over-complete). `component='Z'` (default) or `'X'`.
- Non-CSS: `H` is the symplectic stabilizer matrix in two-block form `[HX | HZ]`, with `tB=2`. `L=None` is allowed — they will compute a logical basis internally.

**Output.** `{n, k, d, L, T, R, progress}`:
- `d` — distance estimate. Exact for enumeration methods only when they complete. Solver methods are exact only if optimality status is actually available; `codedistance==0.0.8` does not expose that status for `MIPDist`, so those runs are incumbents/upper bounds.
- `L` — a binary witness: a logical (or codeword for classical) of weight `d`. For non-CSS in two-block form, `L` is in the same `[LX | LZ]` layout.
- `T` — total trials run; `R` — trials that hit `d`.

The public Webster result dictionary has no `exact` or solver-status field. Our adapter must add a local `result_status` (`proven_exact`, `incumbent`, `heuristic`, `timeout`, `failed`) based on method semantics and outer timeout state. Do not infer `proven_exact` from solver wall-clock alone.

**Methods we will use** (per the user's selection):

| Webster method | Type                  | Backend                         | Maps to our analog               | Notes                                                                                                    |
|----------------|-----------------------|---------------------------------|----------------------------------|----------------------------------------------------------------------------------------------------------|
| `BZDistMW`     | Exact if completed     | Pure-Python Brouwer-Zimmermann  | None                             | Independent exact algorithm class. Exponential in `n-k`; small codes only.                               |
| `MIPDist`      | Solver-based           | OR-tools / SCIP                 | Our HiGHS MILP (`distance_milp`) | Different formulation/backend. With the public API, record as incumbent; lower verified witnesses matter. |
| `QDistEvol`    | Heur.                  | Pure-Python evolutionary        | Our BP-OSD upper bounds          | Headline algorithm of arXiv:2603.22532. Compare at matched wall-clock.                                   |
| `decoderDist`  | Heur.                  | `ldpc` BP-OSD                   | Our `qldpc` BP-OSD               | Same algorithm class, different impl. Reproducibility check.                                             |

**Methods we are deliberately NOT running.** `dist_m4ri_*` (needs C build), `magma*` (license), `DistRndGAP` (GAP+Guava), `GurobiDist` (license), `qubitserf*` (C build), `CLISATDist` (SAT binary), `pySATDist` (python-sat — out of scope per user's chosen four).

**Default method,** for what it's worth, is `dist_m4ri_RW`. We are not using it.

---

## 3. Mapping our codes to Webster's input format

This is mechanically straightforward — we already build everything Webster needs.

### 3.1 CSS (BB + Campaign 4 mixed-monomial)

```python
from evaluation.bb_code import build_bb_code  # returns qldpc BBCode / CSSCode
import numpy as np
code = build_bb_code(ell, m, A_terms, B_terms)
Hx = np.array(code.matrix_x, dtype=np.int8) % 2
Hz = np.array(code.matrix_z, dtype=np.int8) % 2
result = CSScodeDistance(Hx, Hz, method='BZDistMW', component='Z', seed=42)
```

Run for `component='X'` and `component='Z'` separately; `d = min(d_X, d_Z)`.

### 3.2 Non-CSS (Campaign 5 PBB)

`evaluation/pbb_code.py` builds the symplectic stabilizer matrix at line 163 (`np.vstack([top, bottom]) % 2`, shape `(2*dim, 2n)`). The internal convention is `[X-part | Z-part]` per row — this matches Webster's `[HX | HZ]` exactly. **Verify in Tier 0** before scaling.

```python
from evaluation.pbb_code import build_pbb_code
code = build_pbb_code(ell, m, A_terms, B_terms, C_terms, D_terms)
H = np.array(code.matrix, dtype=np.int8) % 2  # already (rows, 2n) symplectic
params = {'MIP_solver': 'SCIP', 'maxTime': 600}
result = codeDistance(H, L=None, tB=2, method='MIPDist', params=params, seed=42)
```

> **Risk:** if `code.matrix` from qldpc happens to use `[Z | X]` layout (some packages do), Webster will silently compute a *different* code's distance and answers will look correct numerically but be wrong. Tier 0 includes a layout-validation test.

### 3.3 Witness verification

This is the part that catches the most bugs. After Webster returns `result['L']`:

- **CSS, component='Z'**: `L` is a binary `n`-vector representing a Z-type logical. Verify:
  - `(Hx @ L) % 2 == 0` (commutes with X-checks)
  - `np.sum(L) == result['d']`
  - `L` is not in `rowspan(Hz)` (non-trivial)
  - For at least one row `lx_i` of `code.get_logical_ops(Pauli.X)`: `(lx_i @ L) % 2 == 1`
- **CSS, component='X'**: mirror the above with `(Hz @ L) % 2 == 0`, `L not in rowspan(Hx)`, and anticommutation against at least one row of `code.get_logical_ops(Pauli.Z)`.
- **Non-CSS**: `L = [LX | LZ]`, length `2n`. Verify:
  - Symplectic commutation: `(SX @ LZ + SZ @ LX) % 2 == 0` for all stabilizer rows
  - Symplectic weight: `sum(LX[j] | LZ[j] for j in range(n)) == result['d']`
  - Non-trivial: `L` not in the symplectic span of stabilizers
  - Anticommutation with at least one row from `evaluation.pbb_code.get_symplectic_logicals(code)`, not qldpc's non-CSS `get_logical_ops()` path

A witness that fails verification means *either* a block-order mismatch *or* a Webster bug — both are findings worth recording.

---

## 4. Three independent comparisons (this is the actual experiment)

### 4.1 Comparison A — Exact claims + witness cross-check (Webster vs our MILP)

**Question.** Do Webster's independent methods ever return a verified witness
below a distance we treat as exact, and do their verified witnesses agree with
the CSS/PBB catalog values in the executed scope?

**Codes.** CSS rows with `ilp_d` and `n ≤ 144` from `ilp_catalog.json` as row-level corroboration targets; only the `bravyi_baselines` anchors are treated as CSS exact claims because the catalog labels/status fields are not solver certificates. Plus all deduplicated non-CSS classes with an exact flag (`milp_exact=True`, `d_is_exact=True`, or `trust_level=="EXACT"`) and no `d_is_upper_bound=True`, from `campaign7_publication_merged.jsonl`. Plus the known A=B d=2 trap (one canonical instance in Tier 0 tests).

**Methods.** `BZDistMW` and `MIPDist` for each code, both `component='X'` and `component='Z'` for CSS.

**Pass criterion.** Our exact `d` and every completed `BZDistMW` result match exactly. `MIPDist` equality is corroborating evidence rather than proof unless the package is patched to expose OR-tools status. Each Webster `L` passes witness verification.

**Failure modes worth flagging in the writeup.**
- BZ returns a verified witness with `d < d_ours_exact`: catastrophic — our exact result is wrong or the code mapping is wrong.
- BZ completes with `d > d_ours_exact`: Webster's BZ run or our witness/mapping interpretation is wrong, because our exact witness should be a valid upper bound.
- MIP returns a verified witness with `d < d_ours_exact`: same severity as BZ, even if MIP optimality is not exposed.
- MIP returns `d > d_ours_exact` without an optimality certificate: not a disagreement; record as a weaker upper-bound/incumbent result.
- Any timeout: not a failure. Record the outer timeout, Webster `progress`, and whether a verified witness was returned.

### 4.2 Comparison B — Heuristic ⇄ Heuristic (QDistEvol vs our 150k BP-OSD)

**Question.** Does QDistEvol find lower-weight logicals than our BP-OSD ensemble at matched compute?

**Codes.** The codes where the paper claims BP-OSD overestimated:
- CSS rows from [results/ensemble_verification_150k.json](../results/ensemble_verification_150k.json) where `claimed_d > verified_d` or where any decoder batch exceeds `verified_d`
- Non-CSS rows from `campaign7_publication_merged.jsonl` with both `d_milp` and `d_bposd`, largest `d_bposd - d_milp` gaps first
- The `[[144, 32, ?]]` cases mentioned in CLAUDE.md ("range 6 to 18 across batches")
- The `[[48, 5, 10]]` reference Webster themselves cite ([paper.tex:129](../paper/2606.02418/paper.tex#L129)), only if we can reconstruct or import the same code unambiguously

Estimate ~15-20 codes after deduplication. Pull CSS cases from `ensemble_verification_150k.json` using `verified_d`, `claimed_d`, `decoder_summary`, `total_trials`, and `total_time_s`. Pull PBB cases from the JSONL by filtering `d_milp_initial - d_milp` and `d_bposd - d_milp` deltas. For non-exact PBB rows, label `d_milp` as an incumbent upper bound, not ground truth.

**Method.** `QDistEvol` and `decoderDist`. Match wall-clock to a single 150k-trial BP-OSD run for that code, ballpark a few minutes per code. Set `seed=42` for reproducibility; set `maxTime` to match.

**Pass criterion.** This is not a pass/fail — it's a measurement. We log `(d_ours_bposd_150k, d_QDistEvol, d_decoderDist, d_truth_or_incumbent)` and chart the gap-closing per code.

**What "good" looks like.** If QDistEvol consistently matches exact `d_milp` on codes where our BP-OSD overestimates, that's strong support for [paper.tex:700](../paper/2606.02418/paper.tex#L700)'s suggestion to integrate it. If it only improves incumbents or does not improve them, that's a finding too.

### 4.3 Comparison C — Symplectic linearization sanity (non-CSS small codes)

**Question.** Is our binary-OR linearization in `evaluation/distance_milp.py::ilp_min_weight_symplectic` actually computing the symplectic minimum weight correctly?

**Codes.** All deduplicated non-CSS codes with an exact flag and no upper-bound flag, AND `n ≤ 100`.

**Method.** `BZDistMW` via `codeDistance(H, None, tB=2)`. Compare numeric `d`; verify witness has correct symplectic weight.

This is a subset of Comparison A but worth listing separately because the failure mode is different — a linearization bug doesn't show up on CSS.

---

## 5. Codes to run, concretely

Programmatic selection criteria (no hand-curation, but with explicit deduplication):

**Input normalization.**
- CSS source: `ilp_catalog.json` contains the row-level CSS MILP catalog used for the paper tables. Rebuild each row from `(ell, m, A, B)` and preserve `source_group`, `source_index`, and `source_label`; do not infer exactness from a `<=` label.
- Non-CSS source: `campaign7_publication_merged.jsonl` is already BLISS-deduplicated in intent, but still use `bliss_hash` when present and fall back to a normalized tuple key when missing.
- Preserve `source_file`, `source_index`, `dedupe_key`, `distance_is_exact`, and `distance_is_upper_bound` in the resolved code lists.

**Tier 1 set.** From the normalized CSS and non-CSS pools, select:
- CSS: rows from `ilp_catalog.json` with `n ≤ 144` and `ilp_d` present. These are row-level corroboration targets; do not derive exactness from the display label.
- non-CSS: records with `n ≤ max_n`, exact flag true (`milp_exact == True` OR `d_is_exact == True` OR `trust_level == "EXACT"`), and `d_is_upper_bound != True`.

For a dry run, cap at 20 deduplicated codes and prefer (a) smaller `n`, (b)
non-CSS small codes needed for Comparison C, (c) higher `k`, (d) one canonical
A=B d=2 instance. The executed Tier 1 sweep used `--limit 0` for the full
selected CSS and PBB scopes.

**Tier 2 set (BP-OSD overestimate codes).**
- CSS: from `ensemble_verification_150k.json`, select rows with `claimed_d > verified_d`, or rows where any decoder batch summary has `d_max > verified_d`. Use `verified_d` as the 150k BP-OSD upper bound.
- Non-CSS: from `campaign7_publication_merged.jsonl`, select rows with `d_milp` and `d_bposd` both present and `d_bposd > d_milp + 2` (or `d_bposd / d_milp ≥ 1.3`, our existing trust ratio threshold). If `d_is_upper_bound=True`, report gaps against the MILP incumbent, not truth.

Cap Tier 2 at 20 deduplicated codes, balancing CSS and non-CSS cases instead of
allowing one source file to dominate.

The latest resolved code list goes in `investigations/webster_tier1_codes.json`;
stable selection snapshots go in `webster_tier1_codes_<selection_id>.json` and
`webster_selection_history.jsonl`.

---

## 6. Method parameters, runtime budgets

Hand-tuned, not load-bearing:

| Method        | Params                                                                             | outer timeout per call |
|---------------|------------------------------------------------------------------------------------|------------------------|
| `BZDistMW`    | `{'LOCheck': 0}`                                                                    | 600 s                  |
| `MIPDist`     | `{'maxTime': 600}`                                                                  | 660 s                  |
| `QDistEvol`   | `{'genCount': 100, 'offspring': 10, 'iterCount': 10000, 'maxTime': 300}`            | 360 s                  |
| `decoderDist` | `{'iterCount': 10000, 'decoder': 'bposd', 'maxTime': 300, 'LOCheck': 0}`            | 360 s                  |

Each call seeded with `seed=42` for reproducibility. Treat the outer process deadline as authoritative: Webster's `maxTime` is not guaranteed to be honored by every pure-Python method. Use a separate process, not a thread, so runaway BZ/evolution calls can be terminated cleanly.

Total Tier 1 runtime estimate: 20 codes × 4 methods × ~few minutes ≈ several hours wall-clock if serialized; faster with `multiprocessing` (each code-method independent). Tier 2 same order.

---

## 7. Output schema

One JSONL record per `(code, method)` pair, written to `investigations/webster_results.jsonl`:

```json
{
  "code_id": "ell12_m6_idx3",
  "ell": 12, "m": 6,
  "n": 144, "k": 12,
  "code_type": "css",
  "A_terms": [...], "B_terms": [...],
  "C_terms": null, "D_terms": null,
  "ours": {
    "d": 12,
    "d_milp": 12, "d_bposd": 12,
    "milp_exact": true,
    "trust_level": "EXACT"
  },
  "webster": {
    "method": "BZDistMW",
    "component": "Z",
    "d": 12,
    "T": 1, "R": 1,
    "L_witness": "<base64 of binary L>",
    "L_weight_check": 12,
    "L_commutes": true,
    "L_anticommutes_with_logical": true,
    "L_verified": true,
    "runtime_s": 42.7,
    "params": {"seed": 42},
    "method_claims_exact": true,
    "result_status": "proven_exact",
    "optimality_source": "method_completed",
    "progress": "..."
  },
  "agreement": {
    "matches_ours": true,
    "lower_than_ours": false,
    "higher_than_ours": false,
    "comparison_strength": "proof"
  },
  "timestamp": "2026-06-08T..."
}
```

A second JSONL `investigations/webster_summary.jsonl` aggregates per code: `{code_id, ours_d, webster_d_per_method, all_proven_exact_agree, witness_verified_count, ...}`.

A short markdown writeup `investigations/webster_findings.md` is produced at the end with: methodology summary, per-tier disagreement table, the 1-2 most interesting cases, and a recommendation on whether this should graduate to a publication-grade scripts/ run.

---

## 8. Files to add

```
investigations/
├── webster_comparison_plan.md          (this document)
├── compare_webster.py                  (CLI: --tier 1|2, --methods, --codes-file)
├── webster_tier1_codes.json            (resolved code list, generated)
├── webster_tier2_codes.json            (resolved code list, generated)
├── webster_results.jsonl               (per-(code,method) results)
├── webster_summary.jsonl               (per-code aggregation)
└── webster_findings.md                 (writeup at end)

evaluation/
└── distance_webster.py                 (thin adapter — see below)

tests/
└── test_distance_webster.py            (Tier 0 sanity: gross [[72,12,6]], Bravyi [[144,12,12]], A=B d=2)

pyproject.toml                          (add a separate 'webster' dependency group; do not add these deps to dev)
```

### `evaluation/distance_webster.py` adapter (sketch)

```python
"""Thin adapter around the codedistance package for our BBCode/QuditCode objects.

Returns dataclasses that match our existing distance result shape, so callers
can compare to evaluator.py outputs without re-implementing dict plumbing.
"""
from dataclasses import dataclass
import numpy as np
import codedistance as cd

@dataclass
class WebsterResult:
    d: int
    L: np.ndarray
    T: int
    R: int
    runtime_s: float
    method: str
    params: dict
    method_claims_exact: bool
    result_status: str  # proven_exact | incumbent | heuristic | timeout | failed
    optimality_source: str

def webster_distance_css(code, method='BZDistMW', component='Z', seed=42, **params):
    Hx = np.asarray(code.matrix_x, dtype=np.int8) % 2
    Hz = np.asarray(code.matrix_z, dtype=np.int8) % 2
    # ... call CSScodeDistance, time it, wrap in WebsterResult
    ...

def webster_distance_noncss(code, method='BZDistMW', seed=42, **params):
    H = np.asarray(code.matrix, dtype=np.int8) % 2
    # ... call codeDistance(H, None, tB=2, ...), time it, wrap
    ...

def verify_witness_css(code, L, component, claimed_d): ...
def verify_witness_noncss(code, L, claimed_d): ...

PROVEN_EXACT_IF_COMPLETED = {'BZDistMW'}
SOLVER_METHODS_NEED_STATUS = {'MIPDist', 'pySATDist', 'GurobiDist'}
HEURISTIC_METHODS = {'QDistEvol', 'QDistRndMW', 'decoderDist',
                     'connectedClusterMW', 'UndetectableErrorMW'}
```

---

## 9. Risks, gotchas, and how we'll catch them

1. **Symplectic block-order mismatch.** Webster expects `[HX | HZ]`. qldpc's internal convention should match (per `pbb_code.py` line 163), but a silent off-by-one in column ordering would produce numerically plausible but semantically wrong distances. **Mitigation:** Tier 0 test runs Webster on the gross code [[72,12,6]] and confirms `d=6`; runs on a non-CSS code with known exact `d` from our MILP and confirms agreement; explicitly checks witness commutation.
2. **`code.matrix_x` row span vs. row count.** Webster says "Hx, Hz do not need to be full rank" — so passing the unreduced matrix is fine. But if our `code.matrix_x` happens to drop rows (canonicalization), the dimension reported by Webster may differ from ours. **Mitigation:** assert `result['n'] == code.num_qudits` and `result['k'] == code.dimension` after every call.
3. **BZDistMW exponential blow-up.** Brouwer–Zimmermann is exponential in `n-k`. For `[[144, 12, 12]]` (`n-k = 132`) it may not finish. **Mitigation:** Tier 1 cap at `n ≤ 144` and enforce an outer process timeout of 600s; record partial results without treating timeout as failure.
4. **`MIPDist` SCIP backend installs.** Requires `ortools`. Add `codedistance`/`ortools` to a separate optional `webster` dependency group, not the normal `dev` group. **Mitigation:** Tier 0 validates the isolated install before any campaign runs.
5. **`decoderDist` uses `ldpc` package's BP-OSD, not `qldpc`'s.** Channel model and decoder schedule may differ subtly from our pipeline. This isn't a bug, it's the *point* — but we should flag in the writeup that "BP-OSD impl A vs B" is what's being compared, not "BP-OSD vs new method."
6. **`QDistEvol` non-determinism.** Even with `seed=42`, multiprocessing or thread interactions may break reproducibility. **Mitigation:** record `T`, `R`, and `progress` so re-running can be diagnosed; rerun any QDistEvol result that beats exact truth or materially tightens an incumbent at least once before reporting.
7. **MILP "incumbent" records are upper bounds, not truth.** `campaign7_publication_merged.jsonl` has many records with `milp_exact=False` and `d_is_upper_bound=True`. **Mitigation:** Comparison A filters to exact records only (`d_is_upper_bound != True` plus an exact flag). Incumbent records appear in Comparison B as part of the "BP-OSD overestimate" picture, with our `d_milp` treated as an upper bound, not ground truth.
8. **Witness `L` representation ambiguity.** Webster's `L` for non-CSS in two-block form should be `[LX | LZ]` (length `2n`) but the README isn't 100% explicit. **Mitigation:** Tier 0 test verifies `len(L) == 2*n` for non-CSS; falls back to `len(L) == n` interpretation if not (some methods may return only the X-part).
9. **`pip install codedistance` may have transitive deps that conflict with our `qldpc` pin.** `ldpc` (used by `decoderDist`) in particular has been known to have version conflicts. **Mitigation:** install in a separate `--group webster` and document the failure path; if conflicts, isolate Webster runs via `uv run --group webster`.
10. **Webster `MIPDist` optimality is hidden.** The public result only gives `d`, `L`, and `progress`; OR-tools status is discarded. Wall-clock under `maxTime` is diagnostic, not a proof. **Mitigation:** mark `MIPDist` results as `incumbent` unless the package is patched/wrapped to capture solver status.

---

## 10. Success criteria for the investigation

Not pass/fail in the test sense — this is exploration. We're done when we can answer all of:

- [ ] **Comparison A:** for every Tier 1 exact code, does completed `BZDistMW` agree with our exact `d`, and does `MIPDist` return no lower verified witness?
- [ ] **Comparison B:** for every Tier 2 code, by how much does QDistEvol close the BP-OSD overestimation gap, in absolute units and as a fraction of `(d_bposd - d_truth_or_incumbent)`?
- [ ] **Comparison C:** does our non-CSS symplectic linearization agree with BZDistMW on every small non-CSS code?
- [ ] **Witness verification:** does every Webster `L` returned commute with our stabilizers, anticommute with at least one of our logicals, and have the claimed weight?
- [ ] One short writeup at `investigations/webster_findings.md` documenting the answers, plus a recommendation: "graduate to scripts/ for full sweep" or "stop here, no surprises found."

---

## 11. Stop conditions / when to escalate

- **Stop and ask** if any Tier 1 code shows `d_ours != d_BZDistMW` after `BZDistMW` completes and its witness verifies. This is the highest-impact case because paper claims rest on it.
- **Stop and ask** if any Webster method returns a verified witness with `d < d_ours` for a code whose local distance is exact.
- **Stop and ask** if any Webster witness fails the commutation/anticommutation check despite `result_status == "proven_exact"`.
- **Continue but flag** if QDistEvol or decoderDist returns a `d` lower than our non-exact `d_milp` incumbent; that is useful tightening, not a contradiction.
- **Continue routinely** if heuristics overestimate vs MILP — that's normal and is the measurement we want.

---

## 12. Open questions worth resolving before Tier 1 runs

1. Does the user want results posted as a follow-up table in the paper, or kept private for now? (The paper already cites Webster — adding empirical comparison would be a natural addendum to Section "Distance certification" but is out of scope as currently approved.)
2. Should Comparison B match wall-clock to *one* 150k-trial BP-OSD batch, or to the full 150k-trial ensemble (3 decoders × 10 batches × 5000 trials)? Default in this plan is *one batch* — fairer apples-to-apples — but the ensemble is what we report in the paper.
3. If `pip install codedistance` introduces a `ldpc` version conflict with `qldpc`, do we (a) isolate via separate uv group, (b) skip `decoderDist`, or (c) downgrade something else? Default in this plan is (a).

---

## 13. Tier 0 results (executed 2026-06-08/09)

Tier 0 bring-up is **done and green**. Adapter: `evaluation/distance_webster.py`;
tests: `tests/test_distance_webster.py` (15 passed, marker `webster`, group
`webster`). `codedistance==0.0.8` installs cleanly alongside our `qldpc`/`ldpc`
pins — **no conflict**, so open question 3 resolves to "no isolation needed"
(still kept in its own group for hygiene; `uv.lock` tracks it).

Empirical findings that change the Tier 1/2/C approach:

* **Layout confirmed.** Our `QuditCode.matrix` is `[HX|HZ]`, matching Webster's
  `tB=2` input exactly. Witnesses verify (commute + anticommute + correct weight)
  on the gross [[72,12,6]] CSS code (both components, all four methods agree
  `d=6`), the A=B trap (`d=2`), and a [[36,4,6]] non-CSS code.
* **Trust the witness, not the reported `d`.** The adapter recomputes the
  witness's own weight (Hamming/symplectic) as authoritative; the package's `d`
  can differ from it on timed-out solver runs (saw reported `d=25`, witness
  symplectic weight 22 on [[360,12,20]]).
* **MIPDist optimality is not recoverable from the public API.** The package
  discards the OR-tools status and returns only `d`, `L`, and wall-clock progress.
  The adapter therefore records every MIPDist result as an incumbent. Verified
  witnesses still matter: a lighter witness than our exact value would be a
  contradiction.
* **BZDistMW blows up on non-CSS.** Fine on CSS at n=72 (~0.5s) and small non-CSS
  n=36 (~6s), but did not finish on n=360 non-CSS in minutes. **Comparison A/C
  for non-CSS leans on MIPDist witnesses + QDistEvol, not BZ proofs.** Every call
  runs in a hard-killable subprocess (`timeout_s`).
* **decoderDist fails on non-CSS** (random/unachievable syndromes — our known
  issue): returned d=126 on [[360,12,20]] (true 20). Keep it only as a CSS BP-OSD
  reproducibility check; drop it from non-CSS comparisons.
* **QDistEvol is the strong heuristic**: hit exact d=20 on [[360,12,20]] where
  decoderDist gave 126. This is the right tool for Comparison B.

**Tier 1 status: DONE for CSS n≤144 + exact non-CSS n≤180** via
`investigations/compare_webster.py` (selection, parallel `--workers`, resumable).
**189 codes checked, 0 contradictions, 16 codes independently proven exact by BZ**; see
`webster_findings.md` for the full breakdown and `webster_results.jsonl` /
`webster_summary.jsonl` for raw data. Remaining: the 13 looser n=108/144/180 codes (longer
budget), the n=288/360 "BP-OSD OVERESTIMATED" set (`--all-n`), and Tier 2
(QDistEvol).
