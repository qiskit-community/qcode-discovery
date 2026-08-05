# Webster `codedistance` cross-check -- findings

Status: **Tier 1 complete for CSS n<=144, exact non-CSS n<=180, and the CSS
n=288/360 "BP-OSD OVERESTIMATED" subset.** The post-review interpretation checks
198 distinct codes with **0 contradictions**. Tier 2 (Comparison B, QDistEvol vs
BP-OSD) is also complete -- all three planned comparisons are done. See
`webster_comparison_plan.md` for the design. Raw data: `webster_results.jsonl` (271 normalized method-runs),
`webster_summary.jsonl` (per-code verdicts), `webster_selection_history.jsonl`,
and the frozen selection snapshots `webster_tier1_codes_8fffd262df.json` (n=288)
and `webster_tier1_codes_a4a4c72c24.json` (n=360). The n=288/360 delta rows are
also kept separately in `webster_results_n288.jsonl` and
`webster_results_n360.jsonl`.

Reproduce:

```bash
uv sync --group dev --group webster
uv run --group webster python -m pytest tests/test_distance_webster.py -q
uv run --group webster python investigations/compare_webster.py --normalize-results --summary-only
uv run --group webster python investigations/compare_webster.py --css-only --limit 0 --workers 12
uv run --group webster python investigations/compare_webster.py --pbb-only --min-d 7 --max-n 180 \
    --exact-only --limit 0 --mip-maxtime 120 --workers 12
# CSS n=288/360 BP-OSD-OVERESTIMATED subset. Code-ids are derived from
# ilp_catalog.json rows with status=="BP-OSD OVERESTIMATED"; MIP-only
# (BZ auto-skipped at these sizes). These reproduce the committed delta logs.
uv run --group webster python investigations/compare_webster.py --css-only \
    --code-ids css_n=288_12x12_0 css_n=288_12x12_22 \
    css_n=288_12x12_5 css_n=288_24x6_21 \
    --mip-maxtime 3600 --workers 16
uv run --group webster python investigations/compare_webster.py --css-only \
    --code-ids css_n=360_15x12_19 css_n=360_15x12_20 \
    css_n=360_15x12_44 css_n=360_15x12_45 css_n=360_15x12_46 \
    --mip-maxtime 3600 --workers 10
```

## Methodology (short)

For each code we treat the **returned witness** as the primary object: recompute
its weight (Hamming for CSS, symplectic for non-CSS), verify that it commutes
with all checks and anticommutes with a logical, and compare that verified
weight to our catalog value. The package's reported `d` is kept for diagnostics
only because timed-out MIPDist runs can report an objective above their returned
witness weight.

* **BZDistMW** (Brouwer-Zimmermann): independent exact enumeration; `proven_exact`
  when it completes. It was run only for small effective lengths.
* **MIPDist** (OR-tools/SCIP): independent solver formulation, but the public
  `codedistance` result does not expose OR-tools optimality status. Every MIPDist
  row is therefore an `incumbent` upper-bound witness, even when its wall-clock
  progress is below `maxTime`.

Highest-impact signal: **a verified witness lighter than a distance we call
exact** (`verdict=CRITICAL_LOWER`). None observed.

## Results (post-review)

**198 distinct codes (43 CSS + 155 PBB), 271 method-runs, 0 CRITICAL, 0
unverified witnesses.** Methods: 254 MIPDist incumbent runs + 17 BZDistMW
`proven_exact` runs. At code level, **16 codes have an independent BZ proof**.

Per-code verdict (from `webster_summary.jsonl`):

| verdict | codes | meaning |
|---------|-------|---------|
| **agree** | **196** | Webster's minimum verified witness weight equals our catalog value |
| `webster_higher` | 2 | MIPDist returned a heavier valid logical than our value; a looser upper bound, not a contradiction |
| **CRITICAL** | **0** | no verified witness below a claimed-exact distance |

The two remaining `webster_higher` rows are `9_6_0126` and `9_6_0133` (both
n=108, our d=10): MIPDist returned weight 12 even at a 600 s budget. Weight 12 ≥
10 is consistent — not evidence against our catalog, just an un-tightened MIP
incumbent on a hard code.

### CSS (Comparison A) -- 43 codes, all agree

| group | n | d | codes | Webster evidence |
|-------|---|---|-------|------------------|
| Bravyi [[72,12,6]] | 72 | 6 | 1 | MIP+BZ, X&Z; BZ gives a proof |
| Bravyi [[144,12,12]] | 144 | 12 | 1 | MIP, X&Z; verified incumbent witnesses |
| catalog rows | 144 | 2 | 21 | MIP, X&Z; verified incumbent witnesses |
| catalog rows | 144 | 4 | 5 | MIP, X&Z; verified incumbent witnesses |
| catalog rows | 144 | 6 | 6 | MIP, X&Z; verified incumbent witnesses |

Important correction: `ilp_catalog.json` is a row-level CSS MILP catalog, but its
display labels and `status` field are not solver certificates. The 32 non-Bravyi
n=144 CSS rows are therefore corroborated by independent MIPDist witnesses, not
upgraded to independently proven exact by Webster's public API.

### CSS n=288/360 -- "BP-OSD OVERESTIMATED" subset (the headline cross-check)

The 9 CSS rows flagged `status == "BP-OSD OVERESTIMATED"` in `ilp_catalog.json`
are where our pipeline most aggressively corrected BP-OSD. Webster's MIPDist
(OR-tools/SCIP, independent of our SciPy/HiGHS MILP implementation) returned a
verified logical whose weight **matches our `ilp_d`, in both X and Z, on all 9**,
with no lighter witness and no heavier per-code minimum in this run set:

| code | BP-OSD d | our `ilp_d` / **Webster witness** | BP-OSD / witness |
|------|----------|-----------------------------------|------------------|
| [[360,40]] | 24 | **2** | **12.0x** |
| [[360,40]] | 22 | **2** | 11.0x |
| [[360,40]] | 20 | **2** | 10.0x |
| [[288,32]] | 22 | **4** | 5.5x |
| [[360,32]] | 20 | **4** | 5.0x |
| [[360,32]] | 16 | **4** | 4.0x |
| [[288,32]] | 20 | **6** | 3.3x |
| [[288,32]] | 12 | **4** | 3.0x |
| [[288,24]] | 18 | **12** | 1.5x |

Why this matters: a *verified* witness of weight `w` is a hard certificate that
`d <= w`. So Webster independently **proves** the upper bounds that refute the
BP-OSD estimates, including the BP-OSD-24 row ([[360,40]]: witness 2 certifies
`d <= 2`, so BP-OSD/witness = 12x), regardless of solver optimality. This
independently substantiates the BP-OSD-overestimate correction at the level needed
for the paper: an outside solver found witnesses at the corrected upper bounds,
including the weight-2 A=B/degenerate family where BP-OSD reports 20-24. All runs
are `incumbent` (SCIP optimality is not exposed by the public API) and these rows
are catalog incumbents too, so "agree" means two independent solvers/formulations
reach the *same* upper bound. Equality with our `ilp_d` is corroboration, not an
independent exact-distance proof. No `webster_lower` was found. Low-d codes solved
in ~20-30 s even at n=288/360; the one d=12 code took ~700 s and stayed incumbent,
so higher-weight searches can be materially more expensive.

### PBB / non-CSS (Comparison C) -- 155 codes, 153 agree, 2 looser, 0 critical

Verdicts after the 600 s rerun of the original 13 looser incumbents (`--code-ids
... --mip-maxtime 600 --workers 12`):

| n | d=6 | d=8 | d=10 | d=12 |
|---|-----|-----|------|------|
| 36 | 15 agree | - | - | - |
| 72 | - | 20 agree | - | - |
| 108 | - | 31 agree | 40 agree / 2 looser | 5 agree |
| 144 | - | 15 agree | 3 agree | 13 agree |
| 180 | - | 2 agree | 4 agree | 5 agree |

The original 13 looser rows were re-run at a 600 s MIP budget: **11 closed to
exact agreement** (MIP found the weight-10/12 logical the 120 s run missed), and
**2 remain** (`9_6_0126`, `9_6_0133`, n=108 d=10 → weight 12 even at 600 s). The
resume key includes the MIP params, so the 600 s rows coexist with the 120 s rows
and `write_summary` takes the per-code minimum verified witness weight. The two
holdouts are hard MIP instances, not contradictions; closing them would need a
longer budget or a different exact method (e.g. BZ is infeasible at 2n=216).

## Tier 2 (Comparison B) -- does QDistEvol close the BP-OSD gap?

The paper's open question (`paper.tex` line 700): can Webster's QDistEvol reduce the
overestimation gap left by BP-OSD? Measured directly on the 17 codes where BP-OSD
most overestimated -- the 9 CSS "OVERESTIMATED" rows plus the 8 largest
`d_bposd - d_milp` PBB gaps -- by running QDistEvol and decoderDist (Webster's
BP-OSD via the `ldpc` package) and comparing each *verified* witness weight to our
recorded MILP value (exact where flagged, otherwise an incumbent upper bound).
Script: `investigations/tier2_qdistevol.py`; data: `tier2_results.jsonl`.

| set | decoderDist (BP-OSD) reaches recorded MILP | QDistEvol reaches recorded MILP |
|-----|--------------------------------------------|---------------------------------|
| CSS (9) | 9/9 | 9/9 |
| PBB / non-CSS (8) | 0/8 (no verified logical) | 8/8 |
| **total** | **9/17** | **17/17** |

QDistEvol closed the gap on **every** code -- matching the recorded MILP value on
16 and *improving* one MILP incumbent (`pbb_30x6_c447dcb2d101746f`, BLISS
`c447dcb2d101746f`: recorded incumbent 24 -> a verified weight-20 logical, so
that catalog value is not tight). It is cheap (~46-133 s/code), so it is viable
as an in-loop distance estimator.

Two nuances sharpen the paper's BP-OSD story:

- On **CSS**, Webster's decoderDist *also* matches the recorded `ilp_d` values
  (9/9). So the BP-OSD overestimates recorded in `ilp_catalog.json` (bp_osd_d
  12-24 vs recorded `ilp_d` 2-12) reflect *our pipeline's* BP-OSD
  configuration/sampling, not an intrinsic BP-OSD limitation -- a well-run BP-OSD
  finds witnesses at those weights too.
- On **non-CSS**, decoderDist timed out with no verified logical on all 8;
  QDistEvol is the only Tier-2 heuristic that returned verified witnesses,
  matching or improving the recorded MILP value on every one (BP-OSD reported
  30-44; recorded MILP/QDistEvol 8-24). This is the regime where a heuristic
  alternative to BP-OSD genuinely matters.

Follow-up worth doing: rerun MILP on `pbb_30x6_c447dcb2d101746f` / BLISS
`c447dcb2d101746f` to confirm d<=20 (QDistEvol found a lighter logical than the
recorded incumbent).

## Coverage: done vs. what's left

**Done:** all CSS n<=144 (34 codes), the 9 CSS n=288/360 "BP-OSD OVERESTIMATED"
rows, and exact non-CSS n<=180 with d>6 plus the n=36 PBB baselines. The non-CSS
symplectic linearization held across all 155 PBB codes checked: every returned
witness was a valid logical and none was lighter than our recorded distance.

**Not yet covered:**

* The 2 remaining `webster_higher` PBB rows (`9_6_0126`, `9_6_0133`). Would need a
  budget beyond 600 s or a different exact method; not a contradiction. Rerun via
  `--code-ids 9_6_0126 9_6_0133 --pbb-only --mip-maxtime 1800 --workers 12`.
* The rest of CSS n=288 / n=360 (the non-"OVERESTIMATED", higher-d rows).
* PBB n=360 exact rows and non-exact PBB rows.
* MILP rerun on `pbb_30x6_c447dcb2d101746f` / BLISS `c447dcb2d101746f`
  (QDistEvol found weight 20 below the recorded 24).

## Recommendation

All three comparisons corroborated our pipeline with zero contradictions across
198 codes: an independent OR-tools/SCIP MIP matched our `ilp_d` on the
overestimated codes (up to 12x), the non-CSS symplectic linearization held, and
Tier 2 showed QDistEvol closes the BP-OSD gap against the recorded MILP value on
17/17 (uniquely so on non-CSS, where decoderDist returned no verified logical).
The cross-check is in a self-contained, citable state. Optional follow-ups: the
higher-d n=288/360 rows, the 2 hard PBB holdouts, the
`pbb_30x6_c447dcb2d101746f` MILP recheck, and folding the verification into the
paper.
