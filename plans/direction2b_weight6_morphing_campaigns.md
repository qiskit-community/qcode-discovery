# Direction 2b: Morphing-Driven Campaigns (mid-cycle `d≈18`/`d≈24` targets)

**Status: PLAN ONLY.** No code, config, or seed files are created or modified by this
document. Every code/YAML block below is a design sketch for a future development pass;
blocks copied verbatim from an existing file are marked "(verbatim)", everything else is
a paraphrase/recommendation. This mirrors the "1st plan, 2nd dev" split used for prior
directions.

**Implementation branch/base**: this document's checklist items fork files
(`seed_solution_weight5_css.py`, `openevolve_evaluator_weight5_css.py`,
`config_weight5_css_small/large.yaml`, and the shared
`distance_schema.py`/`discovery_events.py`/`pareto_v2.py`/`candidate_selection.py`/`gates.py`/
`model_attribution.py`/`--evaluator` infra) that now live on `public/main` (merged there from
`weight5-campaigns`). Branch from `public/main` at its current tip — **not** from
`weight5-campaigns`, which may carry additional unmerged commits (e.g. LLM-ensemble config
tweaks) that `public/main` lacks; the two branches are not guaranteed to be commit-identical even
though `weight5-campaigns`'s infra work has been merged. Re-check `git log public/main` and
`git merge-base weight5-campaigns public/main` immediately before starting, in case more work has
landed on either branch since this document was written.

**Split out from `plans/direction2_weight5_campaigns.md`.** This content was originally a
"## Direction 2b" section alongside Direction 2a's four weight-5 FOM-max campaigns; it now
lives in its own file because the two tracks have diverged enough in objective (constrained
`d`-floor search here vs. open FOM-maximization there) to warrant separate documents. **This
document depends on `plans/direction2_weight5_campaigns.md` for shared infrastructure**:
Phase A's env-var-overridable gates (`QCODE_MIN_K_THRESHOLD`, `QCODE_MIN_RELEVANT_D`,
`QCODE_SAVE_TRUST_RATIO`, etc.), Phase B item 10's `seed_solution_weight5_css.py` (forked
below for the `W5` morph campaigns), Phase B3's MAP-Elites feature-dimension mechanism
(reconfigured below), and Phase D's Webster distance-backend work. Cross-references to that
document are written out by file name below rather than by bare section name, since
"above"/"below" does not resolve across the two files.

## Motivation and scope

This is a second, independent campaign track alongside Direction 2a's four campaigns, not a
replacement for them. It targets a different objective: rather than open-ended
FOM-maximization, it searches for the smallest `n` achieving a target `(k,d)` pair, where the
target `d` values (`≈18`, `≈24`) come from the Shaw-Terhal code-morphing technique rather than
from Direction 2a's weight-5 scaling-law orientation points.

Shaw & Terhal ("Lowering Connectivity Requirements For Bivariate Bicycle Codes Using
Morphing Circuits," arXiv:2407.16336, PRL 134, 090602 (2025)) describe a circuit that morphs
a larger "mid-cycle" BB code into a smaller "end-cycle" BB code with **identical `k`**
(logical qubits are preserved by construction — the contraction is implemented via
ancilla-reset stabilizers, so no logical information is carried by the qubits being removed)
and no vulnerable window (every circuit step is a valid stabilizer state). The mid-cycle
code's distance `d` upper-bounds how far the end-cycle code's distance can degrade
(`d_tilde ≥ d/3` proven for BB-style depth-3 contraction, `~d/2` empirically in the paper's
worked examples). Concretely, the paper's own Table I example morphs a mid-cycle
`[[288,12,18]]` (two-gross) into an end-cycle code matching gross's own `[[144,12,12]]`. This
track searches for the *next* rung up: mid-cycle codes with `d≈18` and `d≈24` that could serve
as the higher-distance side of a similar morphing pair.

**Disambiguation**: Shaw-Terhal's own "degree-5" language refers to *physical qubit
connectivity degree* in the syndrome-extraction circuit graph, not stabilizer check weight —
their mid-cycle codes are standard **weight-6** BB codes (gross and two-gross both are), and
their derived end-cycle codes are typically weight-9. No weight-5 *stabilizer* code appears
anywhere in that paper. This document therefore searches both weight-5 and weight-6
**stabilizer-weight** BB codes as two separate campaign threads for these mid-cycle targets,
independent of the morphing paper's own connectivity-degree terminology.

**Scope:**
- Both weight-5 and weight-6 are in scope, as two separate campaign threads (not one campaign
  covering both).
- `d≈18` is searched even though two-gross already answers a version of it — a smaller-`n`
  alternative at the same `(k,d)` is still a valuable find. `d≈24` is the higher-priority open
  target: no currently cited candidate has *certified* `d≥24` (this repo's own
  `evolve/openevolve_evaluator.py` hardcodes a `[[360,12,24]]` "bravyi" reference candidate
  with no certification metadata attached — its `d=24` is a bare literal, not run through this
  repo's own distance cascade or tagged with a `result_status`). The verify-first checks below
  settle this before the campaign relies on it as an anchor.
- These four campaigns are added alongside Direction 2a's existing four (`W5-CSS-Small/Large`,
  `W5-PBB-Small/Large`), not a replacement for them.
- **Non-BB weight-5/6 families (e.g. non-abelian 2BGA, subsystem/gauge-weight-5
  constructions) are out of scope here** — deferred to a separate future plan document. This
  document only covers standard two-variable BB codes.
- **Non-CSS/PBB variants are also out of scope for this track specifically** (unlike Direction
  2a, which includes PBB campaigns). Shaw-Terhal's morphing construction is defined and
  analyzed only for CSS BB codes; extending it to non-CSS/PBB codes is unstudied territory
  this plan does not attempt to speculate into.

## Naming and targets

| Name | Weight | `n` anchor | `d_floor` | `k_floor` | Grounding |
|------|--------|-----------|-----------|-----------|-----------|
| `W6-CSS-Morph18` | 6 | beat 288 (two-gross) | 18 | 12 | Two-gross itself: `[[288,12,18]]`, but `d=18` is BP-OSD-estimated only, not MILP-proven — see Risk |
| `W6-CSS-Morph24` | 6 | ~360 (existing, uncertified `[[360,12,24]]` "bravyi" reference already hardcoded in this repo, `evolve/openevolve_evaluator.py:1040-1053`) or ~432 (generalized-family conjecture, unverified) — whichever the Phase G verify-first checks confirm | 24 | 12 | Two literature anchors, both unverified against this repo's own distance cascade until Phase G runs: (a) the existing `n=360` reference candidate, a bare literal with no certification metadata; (b) Tour de gross (arXiv:2506.03094) Conclusion's generalized gross-family formula, `r=2,b=1` instance — neither had ever been numerically checked within this codebase prior to direct construction (see Phase G items 1-2 — each has now had only its `k=12` half confirmed by direct construction; the `d=24` half of both is unrun distance certification, per Success Criteria M4) |
| `W5-CSS-D18` | 5 | ~150-260 (2BGA scaling-law extrapolation) | 18 | 12 | `weight-5-codes-survey.md` §5.2 fit, extrapolated well past its fitted range (`n≤100/200`) — low confidence |
| `W5-CSS-D24` | 5 | ~243-435 (same fit, extrapolated further) | 24 | 12 | Same source, extrapolated even further — lower confidence still |

Same "**floor, not ceiling**" principle as Direction 2a applies to `d`: a code that overshoots
a campaign's `d_floor` at the same or smaller `n,k` is strictly better, not off-target. Same
principle applies to `k_floor`.

**Stage-1/Stage-2 `(ℓ,m)` lattice lists for the `morph18`/`morph24`/`d18`/`d24`
`QCODE_LATTICE_PROFILE` values.** These use the same `n=2·ℓ·m` convention and
"nice-factorization" style as the existing evaluator's own `STAGE1_LATTICES`/
`STAGE2_LATTICES` constants, and include this document's own three literature anchors
verbatim. Every `(ℓ,m)` pair below has been confirmed to construct as a valid `qldpc.BBCode`
(direct construction check against `evaluation/bb_code.py`, `n=2ℓm` confirmed for all 26
distinct lattices across this section) — no coprimality or other constraint blocks any of
them.

- **`morph18`/`morph24` Stage 1** (fast pilot, `W6`): `(12,12)` [two-gross, `n=288`], `(16,9)`
  [`n=288`, alternate shape at the same `n`], `(30,6)` [bravyi, `n=360`], `(18,12)` [`r=2,b=1`,
  `n=432`]. Each of these four lattices carries a permanent, non-removable, verified safety
  candidate — see checklist item 10.
- **`morph18`/`morph24` Stage 2** (fuller sweep, `W6`): Stage 1 plus `(20,9)` [`n=360`],
  `(18,10)` [`n=360`], `(24,9)` [`n=432`], `(36,6)` [`n=432`], `(21,10)` [`n=420`], `(30,7)`
  [`n=420`], `(16,14)` [`n=448`], plus, for `morph18` specifically, explicit sub-288 lattices
  — every lattice listed so far has `n≥288`, so `W6-CSS-Morph18`'s own M2 success criterion
  (`n<288`, below) needs at least one lattice actually below that threshold to ever be
  satisfiable: `(12,6)`/`(6,12)` [`n=144`, reused verbatim from the base evaluator's own
  existing `STAGE2_LATTICES` default — not a new invented lattice], `(12,10)` [`n=240`],
  `(20,6)` [`n=240`], `(16,8)` [`n=256`], `(14,10)` [`n=280`]. `(12,6)`/`(6,12)` is the gross
  code's own lattice (`[[144,12,12]]`) — a `d≥18` result there would mean beating gross's own
  distance at gross's own `n`, a materially harder ask than beating two-gross's `n=288` at a
  *different*, larger lattice; this is flagged since M2 as worded only requires `n<288` and
  does not distinguish which sub-288 lattice a candidate comes from.
- **`d18`/`d24` Stage 1/Stage 2** (`W5`): no literature anchor exists at this weight (see Risk
  item 3 below), so there is no fixed "must include" lattice the way `W6` has three. Instead,
  checklist item 14's Stage-0 audit samples from a **fixed candidate superset, decided here,
  independent of the audit's own results**, and the audit narrows this fixed set down to the
  campaign's actual Stage-1/Stage-2 profile. Fixed superset (nice-factorization `(ℓ,m)` pairs
  spanning the extrapolated ranges above):
  - `D18` (`~150-260`): `(10,8)` [`n=160`], `(12,8)` [`n=192`], `(10,10)` [`n=200`], `(12,9)`
    [`n=216`], `(15,8)` [`n=240`], `(16,8)` [`n=256`].
  - `D24` (`~243-435`): `(15,9)` [`n=270`], `(16,9)` [`n=288`], `(15,10)` [`n=300`], `(18,9)`
    [`n=324`], `(15,12)`/`(18,10)` [`n=360`], `(18,12)` [`n=432`].

  Checklist item 14 defines the audit's exact reproducibility spec (fixed seed, one
  deterministic call per lattice, per-lattice pass criterion, minimum number of passing
  lattices) and item 15 defines how its output becomes a persisted, hashed manifest that the
  evaluator actually loads — see both below. If fewer than the required minimum number of
  lattices clear the per-lattice bar, the campaign does not launch at all.

**Every lattice in every profile above is capped at ≤500 candidates entering the
quick-distance/audit stage per run** — the same bound Direction 2a already established for
its own lattices (`plans/direction2_weight5_campaigns.md`: `n<500` scope statement,
`--max-audit-candidates 500`, and the "at most 500 `k>0` candidates per lattice without
replacement" selection rule) — applied here identically rather than restated as a new rule.

**`k_floor=12` is a single, fixed value used consistently everywhere in this document — the
search score (Step 2), the authoritative archive scan (Step 2), and the Success Criteria
table (M1/M2) all read this same number.** `k_floor=12` matches every literature anchor this
document actually targets (bravyi `n=360` has `k=12`; the `r=2,b=1` conjecture has `k=12`;
two-gross has `k=12`) — a lower floor would accept codes with less `k` than the very
architecture these campaigns are meant to be comparable to.

**`k`'s role here is different from Direction 2a's `k`-maximization and needs its own
justification.** Morphing preserves `k` exactly between mid-cycle and end-cycle codes by
construction — there is no structural reason `k` must be `12` specifically; `k_floor=12` is
used here purely because it keeps discovered codes directly comparable to (and potentially
pairable with) the gross-family architecture this track is anchored on. This is a judgment
call made in this plan, not an explicit external requirement — flagged in Risk item 4 below.

**`k_floor` and `d_floor` are *not* implemented as `evaluation/evaluator.py`'s
`MIN_K_THRESHOLD` gate (nor `evolve/openevolve_evaluator.py`'s `MIN_RELEVANT_D` gate).** Direct
reading of `evaluation/evaluator.py:160-163` shows that gate is a hard reject: any candidate
with `k < gates.min_k_threshold()` short-circuits to `result["stage"]="k_low"` and `return None`
before any distance stage ever runs, with a large fixed negative score
(`SCORE_K_LOW_PENALTY + k`, i.e. roughly the same deeply-negative value for every
sub-threshold `k`). Setting that gate to `12` would outright discard every `k=4..11`
candidate pre-distance, destroying exactly the exploratory gradient a search over `n≈288-450`
lattices needs. `MIN_K_THRESHOLD` stays at its existing repo-wide permissive default (`4`,
`evaluation/gates.py`'s `_INT_DEFAULTS["QCODE_MIN_K_THRESHOLD"]`, read via `gates.min_k_threshold()`
— unset the env var override, or set it no higher than that default) for these campaigns;
`k_floor=12`/`d_floor` are implemented purely as parameters of
the new constrained search score described in Step 2 below, which is mandatory infrastructure
for this track — not an evaluator-level cutoff. A discovered code with `k=14` or `16` at an
excellent `(n,d)` is still kept and reported under this design, since the score's feasible
tier is `k≥k_floor`, not `k==k_floor`. Flagged in Risk Assessment below as something a future
implementation pass should revisit once a concrete morphing-circuit pairing is actually
attempted.

## Technical Approach

### Phase G: verify-before-search, then reuse existing search infra

**Step 1 — three near-zero-cost verifications, before writing any new seed or config.**

None of the three checks below resolves `d` in "one evaluation" or with near-certain
confidence once *exactness* is the bar. `results/campaign7_report.md` documents that at
`n=360` (the exact scale of item 1's bravyi candidate below, and comparable to item 2's
`n=432` `r=2,b=1` candidate) this repo's own MILP/HiGHS backend proved only 1 of the 30
deep-MILP-verified `(30,6)`/`n=360` codes EXACT (report §4.3's per-lattice table), leaving the
rest as unproven `milp_incumbent` upper bounds — and a previously-reported top `n=360` code,
`[[360,10,24]]`, was later corrected downward to `d=16` under deeper exact verification (report
§4.3's Key Findings). Two caveats on reusing this figure here: **(i)** the report is entirely about non-CSS PBB
codes verified via the *symplectic* MILP formulation (`evaluation/distance_milp.py`'s non-CSS
path, per this repo's own module-level documentation); this campaign's CSS candidates are
verified via the separate *Hamming-weight* CSS MILP formulation in the same module, so "1 of 30"
is directional precedent for HiGHS's exactness-proving behavior at a comparable
variable-count/lattice scale, not a direct measurement of CSS MILP difficulty at `n=360`.
**(ii)** a different lattice in the same report, `(15,6)`/`n=180`, has its own 36-code
deep-MILP subset (6 of 36 EXACT) — that `36` is `n=180`'s code count, not `n=360`'s (which is
30); the two must not be conflated. Each of the three checks below therefore actually decomposes into
three separately-budgeted stages, not one evaluator call: **(a)** cheap construction and
exact-`k` (GF(2) rank — near-instant); **(b)** a bounded witness search establishing `d_upper`
(BP-OSD and/or a time-boxed MILP incumbent — cheap, but only an upper bound); and **(c)** a
separately-budgeted lower-bound/exact-certification pass (MILP proven-optimal or Webster) that,
per the report above, may not terminate cheaply at this `n`-range and should be given its own
time box rather than assumed to finish within the same near-zero-cost check. A check that
completes stages (a)-(b) but times out on (c) should be reported as "`d_upper=X`, not yet
exact" — a real, useful result — not treated as a failed or incomplete verification.

1. **The existing `[[360,12,24]]` "bravyi" reference candidate, already hardcoded in this
   repo.** `evolve/openevolve_evaluator.py`'s `_REFERENCE_CODES` list (lines 1040-1053) already
   contains `{"ell": 30, "m": 6, "A": [(9,0),(0,1),(0,2)], "B": [(0,3),(25,0),(26,0)], "d": 24,
   "k": 12, "name": "[[360,12,24]] bravyi"}` — a candidate at exactly this campaign's target
   `(k,d)=(12,24)` and at `n=360`, smaller than the generalized-family conjecture's own
   `r=2,b=1` anchor (`n=432`, item 2 below). This entry's `d=24` is a bare literal with no
   `stage`/`result_status` certification metadata — it was added to `_REFERENCE_CODES` for
   perturbation-gradient analysis, not vetted as a verified distance claim. Directly construct
   and evaluate this candidate through `evaluation/bb_code.py`/`evaluator.py` (and,
   given the D4 certification contract's exactness bar in
   `plans/direction2_weight5_campaigns.md`, MILP/Webster-verify rather than stopping at a
   BP-OSD estimate) before treating either `n=360` or `n=432` as this campaign's real anchor.
   **Confirmed by direct construction**: `n=360, k=12` (exact `k`, stage (a)) — the `k=12` half
   of the claim holds; stage (b)/(c) distance certification is separate follow-on work, not yet
   run.
   - If stages (b)-(c) confirm `d=24` (certified, not just BP-OSD-estimated): this becomes the
     primary literature anchor for `W6-CSS-Morph24` — smaller than the conjectural `r=2,b=1`
     instance — and the search then targets beating `n=360` at `d≥24,k≥12`, with item 2 below
     demoted to "is the general conjecture also true," not "what's the anchor."
   - If distance fails to confirm (`d<24`, or only BP-OSD-supportable): the `_REFERENCE_CODES`
     entry itself is misleading and should be flagged/corrected as a standalone existing-file
     fix, independent of this campaign; item 2's `r=2,b=1` check (at `n=432`) becomes the
     primary anchor candidate instead.
2. **The `r=2,b=1` generalized-gross-family candidate.** Tour de gross's Conclusion speculates
   a family parameterized by integer `r≥1` and bit `b∈{0,1}`, reusing gross's own monomials
   (`A=1+y+x³y⁻¹`, `B=1+x+x⁻¹y⁻³`) at `ℓ=6(r+b)`, `m=6r` (so `n=72r(r+b)`), conjectured `k=12`,
   conjectured `d=6(2r+b-1)`. This reproduces gross (`r=1,b=1`) and two-gross (`r=2,b=0`)
   exactly. The next untested member, `r=2,b=1`, gives `ℓ=18, m=12` (`n=432`), conjectured
   `d=6(4+1-1)=24` — exactly the `W6-CSS-Morph24` target, and the paper's own authors never
   instantiated it. As a CSS candidate tuple (per the domain-context `list[tuple[int,int]]`
   convention, normalized to non-negative exponents mod `(ℓ,m)`: `x³y⁻¹` → `(3,11)`,
   `x⁻¹y⁻³` → `(17,9)`): `A_terms=[(0,0),(0,1),(3,11)]`, `B_terms=[(0,0),(1,0),(17,9)]`,
   `ell=18, m=12`. **Confirmed by direct construction** via `evaluation/bb_code.py`'s
   `build_bb_code`/`get_code_params_fast`: `n=432, k=12` —
   the `k=12` half of the claim holds; per the three-stage budgeting note above, exact
   distance certification (stage (c)) is separately budgeted and has not been run yet, and
   `campaign7_report.md`'s own findings suggest it may not complete cheaply at this `n`.
   - If stages (b)-(c) confirm `d=24`: a second literature-anchored floor for
     `W6-CSS-Morph24`, alongside whatever item 1's `n=360` bravyi check establishes — search
     then targets beating `min(360, 432)` (i.e. `360` if item 1 confirmed it, `432` otherwise)
     at `d≥24,k≥12`.
   - If `d<24`: the conjecture is wrong for this instance; still useful — gives an empirical
     ceiling and the campaign proceeds as a genuinely open search instead of a "beat the
     anchor" search — **unless Phase G item 1 already confirmed `n=360, k=12, d=24`**, in which
     case `n=360` remains the anchor regardless of this item's own outcome, and the campaign is
     still a "beat the anchor" search at `n<360`, not an open one; this item's `d<24` result on
     its own only rules out `432` as a *second* anchor, it does not by itself remove item 1's.
   - If `k≠12`: this cannot happen — `k=12` is already confirmed exact for this instance (see
     above) — so the family's `k`-preservation assumption holds for at least this `(r,b)`.
3. **Two-gross's own `d=18` claim.** The source paper only supports this via BP-OSD decoding
   (336 weight-18 X logicals found by randomized-prior decoding; explicitly *not*
   MILP/exhaustively proven — the paper itself hedges this as a set it "strongly believes" is
   complete). Per this repo's own known BP-OSD variance/overestimation issues (see CLAUDE.md's
   "Known issues" section), MILP- or Webster-verifying two-gross directly is worth doing before
   treating `n=288` as a trustworthy baseline for `W6-CSS-Morph18`'s "beat this" framing.

**Step 2 — the objective is genuinely constrained, not FOM-equivalent, so a new search-time
score is mandatory; the archive readout also needs a dedicated design.**

The actual objective is *constrained*: minimize `n` subject to `k≥k_floor` and
`d_lower≥d_floor`, where `k` and `d` are floors a candidate must clear, not exact targets — a
candidate is free to overshoot either. Under plain FOM-maximization (`k·d²/n`), a code with
`k=20,d=30,n=600` (FOM=30) outscores a code with `k=12,d=18,n=200` (FOM=19.4), even though the
second is exactly what this campaign wants (floors cleared, smaller `n`) and the first is not
preferred by this objective at all. Search pressure driven by plain FOM can therefore spend its
budget rewarding larger-`(k,d)` codes at the expense of small feasible ones — and a post-hoc
exact scan over the archive (below) cannot repair this after the fact: it can only report the
best of what the search actually explored, and if FOM-driven selection never favored (or
actively out-competed) small feasible codes, none may exist in the archive to find.

**Design: a mandatory constrained search score**, used as `combined_score` for these four
campaigns in place of plain FOM, with **smooth joint progress below both floors, a dominant
"certified" tier gated on `d_lower` (never `d_upper`), and `-n` only inside that tier**.

The score's dominant tier gates on `d_lower`, not `d_upper`, and handles `d_upper is None`
explicitly. Per `evaluation/distance_schema.py`'s own docstring, `d_lower` is "0 if
uncertified" while `d_upper` is a verified-witness upper bound that can be `None`. Gating the
dominant, `-n`-rewarding tier on an upper bound would let a code whose distance decoder simply
failed to find a low-weight logical (a witness-search failure, not a code property) dominate
the search over a genuinely better, smaller, lower-`d_upper` code — the same D4
certification-contract distinction this document applies everywhere else (Step 2's own
archive readout, below). Schema v2 defines `d_upper=None` as a real, valid state ("no verified
witness exists yet"); a bare `d_upper >= d_floor` comparison would raise `TypeError` the first
time the cascade returns a candidate before any witness search has run. The score below gates
the dominant tier on `d_lower` (always numeric, defaulting to `0`, per schema v2 — safe to
compare unguarded), adds a `None`-safe intermediate "witnessed, not certified" tier keyed on
`d_upper`, and only ever reads `d_upper` through a `None`-guarded local:

```python
# score inputs per persisted candidate, using evaluation/distance_schema.py's schema-v2
# vocabulary (NOT the legacy flat "d" field the base evaluators still use -- see the Step 3
# fork decision below for why this campaign's evaluator must read schema-v2 fields):
#   k        -- exact, GF(2) rank (cheap, stage (a))
#   d_lower  -- rigorous lower bound; 0 if uncertified (distance_schema.py's own convention --
#               only ever raised by completed exhaustive enumeration or MILP-proven-optimal,
#               same D4 vocabulary as the archive readout below). Always numeric, never None.
#   d_upper  -- BP-OSD / MILP-incumbent witness (stage (b) -- a witness, NOT authoritative),
#               or **None if no verified witness exists yet** (distance_schema.py's own
#               convention). Must be None-guarded before any comparison.
#   n        -- exact, from the lattice
# campaign parameters: k_floor=12, d_floor=18 or 24 (this document's Naming and targets table)

FEASIBLE_BASE = 1_000  # must exceed 2*(max(n)+2) across the lattice sweep, so that even the
                        # WITNESSED tier (FEASIBLE_BASE/2 - n) stays above the smooth
                        # sub-floor tier's bound (<2) for every n in the search space; with
                        # max(n)~450 (this document's own n-range) 1_000 clears 2*452=904
                        # comfortably -- re-check this bound if the lattice lists above change

d_upper_safe = d_upper if d_upper is not None else 0  # None means "no witness yet," not
                                                        # "d=0 proven" -- treat as zero
                                                        # evidence, never compare None
                                                        # directly (a bare
                                                        # `d_upper >= d_floor` throws
                                                        # TypeError before any witness exists)

if k >= k_floor and d_lower >= d_floor:
    # CERTIFIED tier: d_lower is a rigorous proof the floor is cleared -- matches this
    # document's "target-achieving archive" tier in the three-tier readout below.
    score = FEASIBLE_BASE - n
elif k >= k_floor and d_upper_safe >= d_floor:
    # WITNESSED tier: a real, useful signal ("plausible, not refuted" -- see the Launch
    # Commands vocabulary below) worth more search pressure than sub-floor progress, but
    # strictly below the certified tier since a witness is not yet a proof.
    score = (FEASIBLE_BASE / 2) - n
else:
    k_progress = min(k, k_floor) / k_floor                     # smooth, in [0, 1)
    d_progress = min(max(d_upper_safe, 0), d_floor) / d_floor   # smooth, in [0, 1)
    score = k_progress + d_progress    # joint progress, in [0, 2) -- always below both tiers
```

This score is not lexicographic — `k_progress` and `d_progress` are summed, so an improvement
in either floor's approach moves the score, and neither floor unconditionally dominates the
other. It gives smooth gradient below the floor (`d=6→12→16→22` all score differently via
`d_progress`) instead of flat pass/fail terms. It names its distance inputs explicitly as
`d_lower`/`d_upper` (schema v2), never a bare unspecified `d`, and never compares a possibly-
`None` value unguarded. This score governs *search pressure only* — the WITNESSED tier and the
smooth sub-floor tier both use the cheap, exploratory `d_upper` witness and are explicitly not
a claim of achieved distance; authoritative success is decided solely by the three-tier
archive readout below, which reads `d_lower`.

**How this plugs into `combined_score`.** `evolve/openevolve_evaluator_weight5_css.py` (this
campaign's own fork base, per Step 3 below) computes its `combined_score` via
`per_lattice_best[key] = max(per_lattice_best.get(key, 0.0), credible_fom)` inside a loop over
every lattice a program's candidates were evaluated against, then `combined_score =
sum(per_lattice_best.values())`, the same pattern already live in the base
`evolve/openevolve_evaluator.py`. **This campaign's fork must override the reduction step
itself, not just the per-candidate `score` fed into it**: summing `FEASIBLE_BASE - n` once per
lattice that clears the floor rewards *breadth of coverage*, not the *smallest single `n`* — a
CERTIFIED candidate at `n=288` on one lattice scores `1000-288=712`, while a CERTIFIED
candidate that merely clears the floor on two separate `n=448` lattices scores
`(1000-448)+(1000-448)=552+552=1104`, beating the genuinely smaller, more desirable code. This
is the *opposite* of the "smallest `n` for `(k,d)`" objective this whole campaign exists to
pursue, and it is not the same situation as Direction 2a's plain-FOM campaigns, where
`combined_score` legitimately rewards a program that produces good codes on many different
lattices (there, more coverage of good codes really is better; here, `n` is a single scalar
being minimized, and "found it twice, on two big lattices" is not better than "found it once,
on a small one").

**Fix: `combined_score = max(per_lattice_best.values())`** (maximum across lattices, not
`sum`), applied only to this campaign's own score (never touching Direction 2a's own plain-FOM
`combined_score`, which correctly keeps `sum` — the pathology above is specific to a
floor-gated, `FEASIBLE_BASE - n` style score, not to `combined_score` in general). This is a
genuine, intentional divergence from the base evaluator's reduction pattern for this
campaign's score only — checklist item 4 below states so explicitly. Lattice-coverage (how
many lattices a program clears the floor on) is a real, separate signal worth keeping, but it
belongs in a distinct MAP-Elites diversity metric, not folded additively into the primary
minimize-`n` score — this document does not design that separate metric; it is flagged as
follow-on work, not blocking these four campaigns' launch.

**Reading "smallest `n` for `(k,d)`" off the MAP-Elites archive/grid itself does not work,
even if the feature dimensions are reconfigured to `(k,d)`.** Direct inspection of the
installed `openevolve` package (`openevolve/database.py`'s `_calculate_feature_coords`,
`_scale_feature_value`, and `_update_feature_stats`) shows feature-dimension bins are computed
via **dynamic min/max scaling** — each dimension's observed min/max is updated on every
`add()` call, and a program's bin index is `int(scaled_value * num_bins)` against *whatever the
current observed range is at that moment*. Two consequences follow:
- **Bins are ranges, not exact-value buckets**, and their boundaries drift as the run
  progresses and new extreme `k`/`d` values are observed — a candidate stored early under one
  bin index is not guaranteed to still land in that same bin index if recomputed later against
  a rescaled range. Two genuinely different `(k,d)` pairs can share a cell (if the observed
  range is wide relative to `num_bins`), and the "elite" comparison within that cell
  (`combined_score`) then compares candidates across different `(k,d)` targets.
- Setting `num_bins` very high per dimension narrows this but never eliminates it (bin drift as
  the observed range changes over the run is structural, not a resolution-of-grid issue), and
  there is no config knob that pins bin boundaries to specific integer `k`/`d` values.

**Design: treat the MAP-Elites grid as what it actually is (a diversity-maintenance mechanism
for the search itself) and get the headline result from an exact, non-binned, three-tier
readout instead — never from `combined_score` or the nearest MAP-Elites cell.** The readout is
not exact-only — it does not require `distance_schema.is_exact(d_lower, d_upper, bound_sources)`
(defined below) — that would be stricter than the actual success bar, since a rigorous lower
bound `d_lower≥18` already proves the floor is cleared even if the best upper-bound witness for
that same candidate is higher and has not (yet) been certified exact by that function (e.g.
`d_lower≥18, d_upper=20`, floor cleared, exactness not yet certified).

- **Data source.** `all_codes.jsonl` is a dead filename: it is written only by the historical
  Campaign-4 run (`evolve/run_evolution.py`'s legacy default), and no evaluator this document
  actually launches writes that exact path — each writes a campaign-suffixed sibling instead
  (confirmed by direct source reading: the noncss evaluator writes `all_codes_noncss.jsonl`;
  the newer weight5-campaign evaluators write `all_codes_weight5_css.jsonl`/
  `all_codes_weight5_pbb.jsonl`). `discovery_events.jsonl` and `pareto_front_v2.json` are wired
  into code but are populated only by a completed run's own output — the readout must not assume
  either file pre-exists, and must tolerate a fresh campaign's first run creating both from
  scratch. This campaign's evaluator (forked from
  `evolve/openevolve_evaluator_weight5_css.py`, per Step 3 below) already calls
  `evolve/discovery_events.py`'s `_emit_discovery_events` (writing a per-run
  `discovery_events.jsonl`) and `evaluation/pareto_v2.py`'s `update_pareto_front_v2` (writing
  `results/pareto_front_v2.json`) as a matter of inherited plumbing — the readout script scans
  those two, not `all_codes.jsonl`. **`discovery_events.jsonl` is the primary source for this
  readout specifically because it is append-only**: unlike either Pareto front (v1 or v2),
  which retains only currently-surviving Pareto-optimal candidates, `discovery_events.jsonl`
  records every event as it happens, which is required for "preserve every model
  rediscovery" — a code found repeatedly by independent search branches should not silently
  collapse to a single Pareto-front entry when counting rediscoveries. Cross-check against
  `results/pareto_front.json` (legacy v1, still populated by every evaluator variant) as a
  fallback, not a primary source.
  - `discovery_events.py` currently promotes only the legacy `d`/`d_is_exact` fields to each
    event's top level, by deliberate design (its own docstring states this is "deliberately
    NOT dependent on any schema-v2 fields such as `d_lower`/`d_upper`... that a separate,
    concurrently in-progress piece of work may add"). Everything else on the result dict passes
    through to each event's catch-all `extra` dict. The Step 3 fork-scoped fix below
    (`r.update(build_v2_record(r))`, applied before `_emit_discovery_events` is called) means
    the result dict already carries `d_lower`/`d_upper`/`bound_sources` by the time that call
    happens — so `extra` carries them automatically, with **zero changes needed to
    `discovery_events.py` itself**.
- **Exactness criterion (tier 2).** Bare `d_lower==d_upper` is insufficient, per
  `evaluation/distance_schema.py`'s own stated invariant (its module docstring calls this "the
  single most important invariant in this module"): two independent heuristic estimates (e.g.
  two separate BP-OSD batches, or a BP-OSD estimate that happens to coincide with an unrelated
  MILP incumbent bound) can numerically coincide without either constituting a proof. Use the
  module's own `is_exact(d_lower, d_upper, bound_sources)` function instead — it additionally
  requires at least one entry of `bound_sources` to be in the module's `EXACT_SOURCE_ALLOWLIST`
  (`exact_milp`, `exhaustive_hash`, `self_dual_proof`, `webster_proven_exact`,
  `symplectic_low_d_proof`). This function is real and already implemented and tested — call
  it directly rather than re-deriving exactness from equality by hand. Every place in this
  document that determines whether a distance is *exact* uses `distance_schema.is_exact(...)`,
  not bare equality — including the Launch Commands and Success Criteria sections below.

Scan the two sources above into three explicit, non-exclusive categories, reusing the same
certification vocabulary as `plans/direction2_weight5_campaigns.md`'s D4 contract (`d_lower`
only ever rises via completed exhaustive enumeration or MILP-proven-optimal):
1. **Target-achieving archive**: `k≥k_floor ∧ d_lower≥d_floor`, regardless of exactness. Take
   the minimum `n` across this set — this is the sole authoritative source for the Success
   Criteria table's M1/M2 rows below. Non-empty ⇒ the campaign's numeric target is met.
2. **Exact-distance subset**: `distance_schema.is_exact(d_lower, d_upper, bound_sources)`
   (fully proven distance, independent of whether it clears `d_floor`). Reported for
   transparency and for tracking distance-certification cost (per the three-stage budgeting
   note above); not itself required for M1/M2.
3. **Exploratory candidates**: `k≥k_floor ∧ d_upper≥d_floor` but no proof yet that
   `d_lower≥d_floor`. Ranked by `d_upper` only, reported explicitly as "witnessed, not
   certified" — never treated as meeting M1/M2 on its own; see the Launch Commands section
   below for the corresponding vocabulary.

**Certification/provenance contract: how a later MILP/Webster-certified `d_lower` for an
already-discovered code flows back into the discovery record.** Two candidate contracts were
considered: **(a)** extend `discovery_events.jsonl`'s append-only schema with a field
referencing an earlier event (e.g. `supersedes_event_id`), so a later certification becomes its
own new event linked to the original discovery; or **(b)** re-submit the certified result
through `evaluation/pareto_v2.py`'s existing `update_pareto_front_v2()`, keyed by `_code_key`
(`evaluation/results.py`), and let its existing merge logic fold the stronger bound into the
already-present entry. Direct source reading of both real files settles this in favor of
**(b)**:

- `update_pareto_front_v2()`/its private `_merge_records()` helper (`evaluation/pareto_v2.py`)
  already do exactly this — every call reloads the full persisted `all_records` set, looks up
  each incoming v2 record's `_code_key`, and where an entry already exists, folds the old and
  new `(d_lower, d_upper)` together via `distance_schema.merge_bounds()` (re-deriving
  `d_is_exact`/`certified_fom` from the merged result) rather than blindly overwriting or
  appending — a later `exact_milp`/`webster_proven_exact` bound legitimately strengthens an
  earlier BP-OSD-only entry for the *same* code, in place. This machinery is real, already
  implemented, and already correct — confirmed by direct source reading, not proposed new code.
- Contract (a) has no supporting machinery today: `discovery_events.jsonl` is deliberately
  append-only with no cross-event reference field of any kind (the closest existing field,
  `parent_id`, refers to the OpenEvolve *program* lineage graph, not to another discovery
  event), and `evolve/provenance_reducer.py`'s own quality-comparison logic
  (`_is_better_quality`/`_quality_fields_from_event`) reasons only about the legacy scalar
  `d`/`d_is_exact` pair, never `d_lower`/`d_upper` ranges — building (a) would mean inventing a
  new linking field and extending that reducer's comparison logic, real new work in a shared
  production file, for a result strictly weaker than what (b) already provides for free.

**Design: certification updates flow back through `results/pareto_front_v2.json` only, by
re-calling `update_pareto_front_v2()` with a fresh v2 record for the same code, keyed by
`_code_key`.** `discovery_events.jsonl` keeps its existing, narrower job — an append-only
"who/when found what" provenance log — and is not where certification updates get linked;
checklist item 23 (MILP/Webster re-verification) feeds its result back through this path, not
into the event log.

**Concurrency safety for `pareto_front_v2.json`.** `update_pareto_front_v2()`'s full
load→merge→write transaction already runs under a single-writer lock: an exclusive
`fcntl.flock` on a sibling `.<name>.lock` file, acquired before `load_pareto_v2` and released in
a `finally` block only after the atomic temp-file-plus-`os.replace` write completes
(`evaluation/pareto_v2.py`, docstring at lines 318-329, lock acquired at line 336, released at
line 418) — confirmed by direct reading of the current function body, not a gap this plan needs
to close. This closes the race that would otherwise be reachable given every real campaign
config overrides `parallel_evaluations` upward from its `1` default (`config.yaml`→6,
`config_ansatz.yaml`→6, `config_ansatz_server.yaml`→24, `config_noncss.yaml`→4,
`config_weight5_pbb_small.yaml`→6, `config_weight5_pbb_large.yaml`→16,
`config_weight5_css_small.yaml`→6, `config_weight5_css_large.yaml`→24) — so multiple
`ProcessPoolExecutor` worker processes do call `update_pareto_front_v2()` concurrently in every
real run, not just hypothetically. Because the lock lives inside the function itself, every
caller is automatically protected — item 6's two new forks and the existing
`evolve/openevolve_evaluator_weight5_pbb.py` need no call-site wrapping for this function.
Checklist item 7 records this verification rather than proposing a fix.

**Remaining gap in the same category, not closed by the above, out of this plan's scope**: the
legacy v1 `update_pareto_front()` (`evaluation/results.py:75`) has no equivalent lock — direct
reading of `evaluation/results.py` finds no `fcntl`/`flock` usage anywhere in that file. Every
evaluator variant, including this campaign's forks, still calls the v1 function alongside the v2
one (Step 2 above), so the same class of race remains reachable there; fixing it means touching
shared production code outside this plan's scope.

**Caveat, not blocking**: `_code_key` (`evaluation/results.py`) is a syntactic tuple key (`ell,
m,` sorted `A_terms`, sorted `B_terms`[, sorted `C_terms`/`D_terms`]), not a
graph-isomorphism-invariant hash — `pareto_v2.py` does not import or use
`evaluation/tanner_equivalence.py`'s `canonical_hash` at all. Two permutation-equivalent codes
represented with differently-ordered or differently-labeled monomial terms would land as two
separate `pareto_front_v2` entries rather than merging onto one, so a later certification pass
must re-use the *same* term representation the discovery run originally produced (not merely an
equivalent one) for the merge in this section to actually fire. Fixing `_code_key` itself to use
`canonical_hash` would close this gap but touches shared production code outside this plan's
scope — flagged as follow-on work, not blocking. Separately, this readout logic must not assume
`results/pareto_front_v2.json` or `discovery_events.jsonl` already exist — a fresh campaign's own
first run may be what materializes either file for the first time, and the readout must tolerate
that (not just the steady-state case where both already have prior entries).

Doing the readout as an exact log scan (rather than through `combined_score`) also sidesteps
any interaction between a lexicographic score and openevolve's `_is_better` stochastic/pareto
selection logic — the readout never touches `combined_score` at all, so that interaction is not
a concern for correctness of the reported result (it can still affect what the search
*explores*, which is exactly what the constrained score above is for). **The MAP-Elites
feature-dimension choice itself is not load-bearing for correctness** — it can stay whatever
Direction 2a's Phase B3 settles on (diversity is a useful search heuristic regardless of
whether it's bucketed by `(k,d)`).

**Step 3 — seeds and evaluators: fork the schema-v2 weight-5 evaluator lineage, not the base
evaluator; reuse existing routing/weight helpers, don't reinvent them.**

A separate, concurrent development pass (building Direction 2a's own weight-5 CSS/PBB
campaigns) has landed real infrastructure this track reuses directly: `--evaluator` routing in
`evolve/run_evolution.py`, the `QCODE_LATTICE_PROFILE` mechanism, and a generalized
weight-enforcement helper. The design below builds on that infrastructure rather than
duplicating it.

- **Evaluator: fork `evolve/openevolve_evaluator_weight5_css.py`, not the base
  `evolve/openevolve_evaluator.py`.** `openevolve_evaluator_weight5_css.py` already imports
  `evaluation/distance_schema.py` (this campaign's own Step 2 score needs real
  `d_lower`/`d_upper` fields, which the base evaluator's legacy flat `d`/`d_is_exact` dict does
  not carry), already calls `evolve/discovery_events.py`'s `_emit_discovery_events` and
  `evaluation/pareto_v2.py`'s `update_pareto_front_v2` (this campaign's own readout fix above
  depends on both), and already implements the `QCODE_LATTICE_PROFILE` mechanism described
  next. Forking it means this campaign's `W6-CSS-Morph18`/`Morph24` evaluator (new file,
  `evolve/openevolve_evaluator_weight6_morph.py`) inherits all three for free and only needs to
  add this document's own `k_floor`/`d_floor`-gated score (Step 2) in place of that file's
  existing plain-FOM `credible_fom`. `openevolve_evaluator_weight5_css.py` itself does not
  implement any `k_floor`/`d_floor`-gated score today (its `combined_score` uses the exact same
  plain-FOM `per_lattice_best`/`max`/`sum` pattern as the base evaluator, with zero references
  to `d_lower`/`d_upper`/floor concepts anywhere in the file) — the constrained score is new
  work on both the 2a and 2b sides; only the schema-v2/discovery-events/pareto-v2/lattice-
  profile plumbing around it is reused.
  - **`W5-CSS-D18`/`W5-CSS-D24`: fork into `evolve/openevolve_evaluator_weight5_d18d24.py`,
    do not reuse `evolve/openevolve_evaluator_weight5_css.py` in place.** Same-file reuse (a
    `QCODE_K_FLOOR` env-var check gating which score function runs) was considered and
    rejected: a real pipeline-ordering bug in the shared file (below) means this campaign's
    score cannot correctly read `d_lower`/`d_upper` from that file's existing structure without
    a fix, and applying that fix in place would silently change evaluation-pipeline ordering
    for Direction 2a's own W5 CSS campaigns, which already use that file unmodified. The fork
    (`evolve/openevolve_evaluator_weight5_d18d24.py`, mirroring the `W6` fork above) adds
    `d18`/`d24` lattice-profile values to its (inherited) `QCODE_LATTICE_PROFILE` mechanism —
    sourced from checklist item 15's persisted, hashed Stage-0 manifest, not a hardcoded Python
    list — substitutes this campaign's `k_floor`/`d_floor`-gated score (Step 2) for its
    existing plain-FOM per-candidate value at the same aggregation call site, and applies the
    ordering fix below.
- **Weight enforcement: call the existing `evaluation/weight_enforcement.py`'s
  `css_weight_ok()`, do not write a new `validate_pair_weight()`.**
  `evaluation/weight_enforcement.py` already contains a generalized `css_weight_ok(A_terms,
  B_terms, target_weight=5, canonical_split=(2,3))` (regular, not keyword-only, parameters).
  For `W6`, call it as `css_weight_ok(A_terms, B_terms, target_weight=6, canonical_split=(3,3))`
  — an explicit call-site override of the existing defaults, not a change to the function
  itself, and not a new function. For `W5-D18`/`D24`, call it with **no override at all**:
  `css_weight_ok(A_terms, B_terms)` already matches `target_weight=5, canonical_split=(2,3)` by
  default. Wire this call into the forked/reused evaluator immediately after `validate_terms()`
  and before construction, mirroring `validate_terms()`'s reject convention. Add regression
  tests confirming a hand-built 4-term-`A`/3-term-`B` candidate is rejected under the
  `target_weight=6` call and a `3+3` candidate is rejected under the default `target_weight=5`
  call (confirming mutated generators cannot silently escape the requested weight) — this
  test-writing work is still new (the helper itself is not), so it stays in the checklist.
- **Lattice-profile routing: extend the existing `QCODE_LATTICE_PROFILE` mechanism, do not
  build a new one.** `evolve/openevolve_evaluator_weight5_css.py` already reads
  `os.environ.get("QCODE_LATTICE_PROFILE", "small")` fresh on every call (deliberately never
  cached — relevant for concurrent/parallel evaluation, and a detail this campaign's fork must
  preserve rather than accidentally memoizing) and already branches on the returned string to
  select a lattice list. Both forks add two new profile values each — `morph18`/`morph24` for
  `W6`, `d18`/`d24` for `W5` — to that same existing mechanism, each swapping in a lattice list
  re-centered on `n≈288-450` per the concrete Stage-1/Stage-2 lists in the Naming and targets
  section above (the `d18`/`d24` values load their lattice list from checklist item 15's
  manifest rather than a hardcoded list), rather than falling through to that mechanism's own
  `"small"` default (whose lattice range does not reach the `~432` `r=2,b=1` anchor).
- **`QCODE_MIN_K_THRESHOLD` must NOT be set to this campaign's `k_floor` value;
  `QCODE_MIN_RELEVANT_D` is moot for this campaign's evaluator lineage either way, for a more
  direct reason than "leave it permissive."** `QCODE_MIN_K_THRESHOLD` is a real, hard
  evaluator-level cutoff, read via `evaluation/gates.py`'s `min_k_threshold()`:
  `evaluation/evaluator.py:160-163` short-circuits (`return None` before any distance stage
  runs) every candidate with `k < gates.min_k_threshold()`. Setting it to `12` would discard
  every `k=4..11` candidate pre-distance, destroying the exploratory gradient this search needs
  (same point as the Naming and targets section above) — it stays at its repo-wide default (`4`,
  `evaluation/gates.py`'s `_INT_DEFAULTS`).
  `QCODE_MIN_RELEVANT_D` has no effect on this campaign at all: this campaign's evaluator
  lineage (`evolve/openevolve_evaluator_weight5_css.py`, forked by both
  `evolve/openevolve_evaluator_weight6_morph.py` and
  `evolve/openevolve_evaluator_weight5_d18d24.py` below) never reads `min_relevant_d` anywhere in
  the file (confirmed by direct grep) — it scores every candidate through a different
  `credible_fom`/`TRUST_FULL` trust-decay scheme gated by `QCODE_SAVE_TRUST_RATIO` (already set
  explicitly in the Launch Commands blocks below), not a `d`-threshold gate. The
  flat-`0.01`-below-threshold mapping lives only in the
  *base* evaluator's `evaluate_stage2_milp` function (`evolve/openevolve_evaluator.py:1003-1007`,
  reading `gates.min_relevant_d()` at line 865 — default `6`, `evaluation/gates.py`'s
  `_INT_DEFAULTS`) — a function this campaign's forks never call. `QCODE_MIN_RELEVANT_D` is left
  unset for these campaigns regardless, both because it is inert here and in case a future
  maintenance pass reuses `evaluate_stage2_milp`-style logic in this lineage. The actual floors
  are carried into the new constrained score (Step 2 above) via two new env vars this track
  adds, `QCODE_K_FLOOR` and `QCODE_D_FLOOR` (read by the new score directly, not by either gate
  above).
- Explicit routing at launch time: every Launch Commands block below passes `--evaluator <path>`
  explicitly (`evolve/run_evolution.py`'s `--evaluator` flag — evaluated last in that file's
  dispatch logic and unconditionally overrides the hardcoded `EVALUATOR`/`EVALUATOR_NONCSS`
  constants with an arbitrary user-supplied path). Omitting it would silently fall through to
  the base CSS evaluator, which has none of this section's schema-v2/discovery-events/lattice-
  profile plumbing — see the Launch Commands section below for the exact invocations,
  including the `W5` blocks' corrected evaluator path.
- All four should land after Direction 2a's Phase D (Webster) is available, or at minimum
  should MILP-verify any promising find before reporting it — BP-OSD alone is not trustworthy
  at `d≈18-24` given this repo's own documented BP-OSD-overestimation issue, and the whole point
  of this track is to find codes trustworthy enough to actually build a morphing pair around.

**Pipeline-ordering fix, required for the Step 2 score to actually read `d_lower`/`d_upper`.**
`evaluate_stage2`'s scoring loop (`per_lattice_best`, initialized at line 582, of
`evolve/openevolve_evaluator_weight5_css.py`) reads straight off `metrics["all_results"]` — the
raw legacy result dicts `evaluation/evaluator.py`'s `_make_result_template()` constructs (`ell,
m, A_terms, B_terms, n, k, d, d_is_exact, distance_trusted, fom, encoding_rate, score, stage` —
no `d_lower`/`d_upper` anywhere in that shape) — via `r.get("d", 0)`. `build_v2_record(r)` is
called for the first and only time over a hundred lines later, at line 717, inside the
persistence block, strictly after the combined score (`combined =
sum(per_lattice_best.values())`, line 605) has already been finalized and after
`save_code`/`update_pareto_front` (v1, lines 709-710) have already run against the same raw
dicts. `v2_records = [build_v2_record(r) for r in credible_to_save]` builds *separate* v2
objects rather than merging their fields back onto `r` — so `save_code`, `update_pareto_front`
(v1), and `_emit_discovery_events` (line 722, called with `credible_to_save`, the raw list, not
`v2_records`) never see `d_lower`/`d_upper` at all. Only `update_pareto_front_v2` (line 718)
ever receives a v2 record.

**Fix, scoped entirely to this campaign's own fork(s) — no change to the shared base file
`evolve/openevolve_evaluator_weight5_css.py` itself:** immediately after
`_run_evaluation`/`evaluate_batch` returns each raw result `r` (i.e., at the top of whatever
loop currently produces `metrics["all_results"]`, before that list is handed to the
`per_lattice_best` scoring loop), materialize and merge: `r.update(build_v2_record(r))`. This
must happen once, per candidate, before any of the following read `r`: the Step 2 score's
`per_lattice_best` loop (needs `d_lower`/`d_upper`, not legacy `d`), `credible_to_save`/
`best_credible` construction, and `_emit_discovery_events`. (`_log_code_jsonl` already runs
earlier, at line 447, inside `_run_evaluation`'s per-lattice loop — strictly before this merge
point regardless of exact line, and its current hardcoded field list does not consume
`d_lower`/`d_upper`/`bound_sources` anyway, so it is intentionally excluded here rather than
impossible to satisfy.) Because `r` is
merged in place rather than shadowed by a separate object, every downstream consumer that
already reads `r` (including ones this plan does not otherwise touch) gets `d_lower`/
`d_upper`/`bound_sources` for free, with no further call-site changes needed. This same fix
also resolves the readout's data-source design above: once `r` carries schema-v2 fields before
`_emit_discovery_events` runs, `discovery_events.py`'s own pass-through-to-`extra` behavior
(anything not in its `_TOP_LEVEL_RESULT_FIELDS` allowlist lands in each event's `extra` dict)
means `d_lower`/`d_upper`/`bound_sources` reach `discovery_events.jsonl` automatically, with
**zero changes to `discovery_events.py` itself**.

A thin fork (`evolve/openevolve_evaluator_weight5_d18d24.py`, mirroring the `W6` fork) confines
this fix to this campaign only, rather than changing evaluation-pipeline ordering for every
campaign that already uses the shared base file unmodified (Direction 2a's own W5 CSS runs).

## Implementation Checklist

### Phase G: Morphing-Driven Campaigns

Numbered 1-24, under the "Phase G" label used throughout the Technical Approach section above.

1. [ ] Verify-first: directly construct and evaluate the existing hardcoded `[[360,12,24]]`
   "bravyi" reference candidate (`evolve/openevolve_evaluator.py:1040-1053`'s `_REFERENCE_CODES`,
   `ℓ=30,m=6,n=360`, `A=[(9,0),(0,1),(0,2)]`, `B=[(0,3),(25,0),(26,0)]`) via
   `evaluation/bb_code.py`/`evaluator.py`, then MILP/Webster-verify (not BP-OSD-only) — this
   candidate's `d=24` is currently a bare uncertified literal and, if confirmed, is a smaller
   anchor (`n=360`) than the `r=2,b=1` conjecture below (`n=432`); run this check first.
   **Exact `k=12` already confirmed** by direct construction; distance certification (stages
   (b)-(c)) is the remaining work.
2. [ ] Verify-first: directly construct and evaluate the `r=2,b=1` generalized-gross-family
   candidate (`ℓ=18,m=12,n=432`, `A_terms=[(0,0),(0,1),(3,11)]`,
   `B_terms=[(0,0),(1,0),(17,9)]` — normalized to non-negative exponents mod `(ℓ,m)`) via
   `evaluation/bb_code.py`/`evaluator.py` — this is one candidate, not a search, but per the
   three-stage budgeting note above, treat exact certification (stage (c)) as separately
   time-boxed from the cheap `k`/`d_upper` stages (a)-(b) — per `campaign7_report.md` (subject
   to Step 1's caveat on that report being a non-CSS/symplectic-formulation precedent, not a
   direct CSS measurement), exact MILP certification is not reliably cheap at this `n`. **Exact
   `k=12` already confirmed** by direct construction; distance certification is the remaining
   work.
3. [ ] MILP- or Webster-verify two-gross's own `[[288,12,18]]` claim (currently BP-OSD-only)
   before finalizing `W6-CSS-Morph18`'s "beat n=288" framing.
4. [ ] Implement the mandatory constrained score from Step 2 above — the CERTIFIED tier
   (`FEASIBLE_BASE - n`) gated on `k≥k_floor ∧ d_lower≥d_floor`, the WITNESSED tier
   (`FEASIBLE_BASE/2 - n`) gated on `k≥k_floor ∧ d_upper_safe≥d_floor` with `d_upper`
   explicitly `None`-guarded, else smooth `k_progress+d_progress` — reading `k_floor`/`d_floor`
   from two new env vars `QCODE_K_FLOOR`/`QCODE_D_FLOOR`, substituted for `credible_fom` at the
   existing `per_lattice_best`/`max` call site in item 6's evaluator fork, **but with the
   across-lattice reduction changed from `sum(per_lattice_best.values())` to
   `max(per_lattice_best.values())` for this score specifically** — summing rewards
   multi-lattice breadth over the single smallest `n`, contradicting this campaign's own
   minimize-`n` objective (worked counterexample in Step 2). This is a real code change to the
   reduction step, not a drop-in substitution — gated to these four campaigns only, leaving
   Direction 2a's plain-FOM `combined_score` (which correctly keeps `sum`) untouched for its
   own campaigns. Lattice-coverage tracking, if wanted, is a separate MAP-Elites metric — not
   designed here. **Add tests covering**: (i) the CERTIFIED/WITNESSED tier boundary — a
   candidate at exactly `k=k_floor, d_lower=d_floor` scores CERTIFIED, one at
   `k=k_floor, d_lower=d_floor-1` (with a `d_upper_safe≥d_floor`) falls through to WITNESSED, not
   CERTIFIED, and one at `k=k_floor-1` (any `d`) falls all the way through to the smooth
   `k_progress+d_progress` tier regardless of `d`; (ii) the WITNESSED tier's `d_upper` `None`
   guard specifically — a candidate with `d_upper=None` (no upper-bound witness yet),
   `d_lower<d_floor`, and `k≥k_floor` must fall through to the smooth tier, not be treated as
   vacuously satisfying `d_upper_safe≥d_floor` (the `d_lower<d_floor` condition is required so the
   candidate doesn't instead hit the CERTIFIED branch, which is checked first and depends only on
   `d_lower`); and (iii) the `max` vs `sum` reduction change itself — a constructed
   two-lattice case where one lattice scores high and the other low reduces to the high lattice's
   own score alone under `max(per_lattice_best.values())`, confirmed different from (and lower
   than) what `sum(per_lattice_best.values())` would have produced on the identical inputs, and a
   same-file regression test confirming Direction 2a's own `combined_score` call site is
   untouched (still uses `sum`, not `max`).
5. [ ] Implement the three-tier archive-scan readout from Step 2 above (target-achieving /
   exact-distance subset via `distance_schema.is_exact()` / exploratory) as a standalone script
   that **joins two sources by canonical code key (`evaluation/results.py`'s `_code_key`)
   instead of reading either alone**: `discovery_events.jsonl` (every matching event, in
   arrival order — the provenance source, giving `generating_model`/`program_id`/`stage`/
   timestamp for every independent rediscovery of that code, plus that event's own
   `d_lower`/`d_upper` out of its `extra` dict, contract (a) in Step 2's discovery) and
   `results/pareto_front_v2.json`'s `all_records` (the certification source — any later
   MILP/Webster bound item 23 writes, which only ever reaches this file, never the immutable
   event log). **`pareto_front_v2.json`'s `all_records` entry for a code key is the sole
   authority for the `d_lower`/`d_upper` the readout uses in tier/success determination — never
   a bound recomputed from raw event fields.** `update_pareto_front_v2()` (`evaluation/
   pareto_v2.py`) already merges every incoming result for a code key through
   `distance_schema.merge_bounds()` under its own single-writer lock, and deliberately
   quarantines (into a separate `quarantined` list, never `all_records`) any merge that would
   produce `upper < lower` — reading the readout's bound straight from `all_records` inherits
   that merge discipline for free. Recomputing `max(d_lower)`/`min(d_upper)` directly across raw
   event fields instead would bypass `merge_bounds()`'s source-kind gating (a BP-OSD-only event's
   `d_lower=0` can never be raised by a heuristic source, only by `"exhaustive"`/`"solver_proof"`)
   and could numerically reconstruct exactly the conflicting bound combination
   `update_pareto_front_v2` already declined to merge — the one failure mode this design must
   avoid. Discovery events are the readout's **provenance-only** source: for a code key with a
   `pareto_front_v2.json` record, the readout takes `d_lower`/`d_upper` from that record alone
   and takes `generating_model`/`program_id`/`stage`/timestamp attribution from the **union of
   every discovery event matching that code key** (independent rediscovery of the same code by
   multiple models/programs/runs must all survive in the readout's attribution, not collapse to
   a single one). **The join is an explicit inner join, scoped to the requested run's own event
   log**: a code key is included in a given run's readout iff it has **both** a
   `pareto_front_v2.json` `all_records` entry **and** at least one matching event in *that run's
   own* `discovery_events.jsonl` (`results/evolution/<run_id>/discovery_events.jsonl` —
   `evolve/discovery_events.py`'s own path layout keeps each run's events in a separate,
   run-scoped file, while `pareto_front_v2.json` is a single file shared across every run). A
   `pareto_front_v2.json` record for a code key with **no** matching event in the requested run's
   own log — e.g. a record produced by a different run, or by a later item-23 certification pass
   that never calls `_emit_discovery_events` — is **excluded from that run's readout entirely**,
   never reported with empty or borrowed-from-elsewhere provenance. A later item-23 certification
   pass may still update a `pareto_front_v2.json` record's non-distance fields (e.g. re-stamping
   `candidate_origin`) after the join without itself adding a new discovery event — the inner
   join only requires the requested run's own log to contain a matching event, not that every
   field on the current record trace back to that run. Symmetrically, a code key present in
   `discovery_events.jsonl` with no matching
   `pareto_front_v2.json` record (should not happen in steady state, since both are written from
   the same `credible_to_save` batch — Step 3) is logged and excluded from tier/success
   determination entirely, not used to fall back to an event-derived bound. **Excludes any event
   or record whose `candidate_origin` is `"fixed_reference"` (checklist items 10 and 12 — the
   `W6` `_LAST_ORIGIN_BY_KEY` mechanism and the `W5-D18`/`D24` `_FIXED_REFERENCE_KEYS` mechanism
   respectively) from every tier/success determination** — a permanent safety-net or
   manifest-witness candidate re-evaluated every run is not a discovery regardless of what bound
   it ends up carrying; see the Success Criteria table's M1/M2 gate text below, which states this
   exclusion explicitly, and M3's own reporting (below) inherits the same exclusion since it
   reads from the same joined readout. This works with zero changes to `discovery_events.py`
   itself once item 6's forks apply the `r.update(build_v2_record(r))` ordering fix (Step 3),
   which makes every result dict carry `d_lower`/`d_upper` before `_emit_discovery_events` runs,
   and once items 10 and 12's forks stamp `candidate_origin` before scoring. **Add tests covering**: (i) two independent runs/programs rediscovering the same
   code key, both preserved (not deduped away) in the joined readout's attribution, while the
   bound used for tier/success comes only from the `pareto_front_v2.json` record; (ii) a
   rediscovery followed by a later, separate item-23 certification run that raises `d_lower`
   only in `pareto_front_v2.json`, confirming the readout's tier/success check reads that raised
   bound, not a stale event-only one; and (iii) a constructed case where a BP-OSD-only event's
   raw `d_lower=0`/`d_upper=<target>` and the certified `pareto_front_v2.json` record's
   `d_lower=<target>`/`d_upper=None` would, if maxed/minned together across sources naively,
   produce a tighter-looking but unmerged bound than the record alone — confirming the readout
   reports the record's own bound and does not reconstruct that unmerged combination; and (iv) a
   `pareto_front_v2.json` record whose code key has a matching event in a *different* run's
   `discovery_events.jsonl` but no matching event in the requested run's own log is excluded
   entirely from the requested run's readout — confirming the join is a per-run inner join, not
   a union across every run's event log.
6. [ ] Fork `evolve/openevolve_evaluator_weight5_css.py` into
   `evolve/openevolve_evaluator_weight6_morph.py` for the `W6` campaigns, and into
   `evolve/openevolve_evaluator_weight5_d18d24.py` for the `W5-D18`/`D24` campaigns (fork for
   both, do not reuse the base file in place for either — see Step 3's pipeline-ordering fix
   for why same-file reuse is unsafe). Both forks inherit the base file's existing
   `distance_schema`/`discovery_events`/`pareto_v2` wiring and `QCODE_LATTICE_PROFILE`
   mechanism for free (per Step 3 above, do not fork or modify the base
   `evolve/openevolve_evaluator.py` itself) and must each apply the
   `r.update(build_v2_record(r))` ordering fix (Step 3) before their own copy of the scoring
   loop runs.
7. [x] `evaluation/pareto_v2.py`'s `update_pareto_front_v2()` already
   wraps its full read→merge→write transaction in a single-writer `fcntl.flock` on a sibling
   lock file (lock acquired at line 336, released in a `finally` block at line 418, confirmed by
   direct source reading — see the Concurrency safety discussion above). Item 6's two forks and
   the existing `evolve/openevolve_evaluator_weight5_pbb.py` need no additional locking around
   this call site; the lock lives inside the function, not at each caller. Flagged, not fixed
   here (shared production code, out of this plan's scope): the legacy v1 `update_pareto_front`
   call site (`evaluation/results.py`) has no equivalent lock — a candidate for the same fix in a
   future pass that touches that shared file directly.
8. [ ] Add `morph18`/`morph24` (`W6`) and `d18`/`d24` (`W5`) values to the existing
   `QCODE_LATTICE_PROFILE` mechanism (already implemented in
   `openevolve_evaluator_weight5_css.py` — extend it directly and/or in item 6's fork, do not
   build a new profile-selection mechanism), using the concrete Stage-1/Stage-2 `(ℓ,m)` lattice
   lists in the Naming and targets section above (the `d18`/`d24` values load their lattice
   list from item 15's persisted manifest, not a hardcoded list), each capped at ≤500
   candidates entering the quick-distance/audit stage per lattice (the same
   `n<500`/`--max-audit-candidates 500` bound Direction 2a already established).
9. [ ] Wire `evaluation/weight_enforcement.py`'s existing `css_weight_ok(A_terms, B_terms,
   target_weight=6, canonical_split=(3,3))` (`W6`, explicit override) and `css_weight_ok(A_terms,
   B_terms)` (`W5`, already matches the function's own `target_weight=5, canonical_split=(2,3)`
   defaults) into the item 6 evaluator(s) immediately after `validate_terms()` and before
   construction — no new gate function, only new call sites; add regression tests confirming a
   mutated 4-term-`A`/3-term-`B` candidate is rejected under the `W6` call and a `3+3` candidate
   is rejected under the `W5` call.
10. [ ] `evolve/seed_solution_weight6_morph.py` — fork of `seed_solution.py`/
    `seed_solution_ansatz.py` (x/y-swap family), lattice sweep re-centered on `n≈288-450`,
    paired with item 6's evaluator fork.

    **Fixed-reference table**: a permanent, non-removable safety candidate for each of the four
    `W6` Stage-1 lattices, mirroring `seed_solution_ansatz.py`'s own `_safety_net_codes(ell, m)`
    pattern (a fixed list living outside the `EVOLVE-BLOCK` markers, unconditionally injected
    whenever `(ell, m)` matches, so the LLM cannot remove it during evolution). All four verified
    by direct construction via `evaluation/bb_code.py` (`build_bb_code`/`get_code_params_fast`,
    read-only execution, no repository files modified):

    | `(ℓ,m)` | `A_terms` | `B_terms` | `n` | `k` | Provenance |
    |---|---|---|---|---|---|
    | `(12,12)` | `[(0,0),(0,1),(3,11)]` | `[(0,0),(1,0),(11,9)]` | 288 | 12 | two-gross itself |
    | `(16,9)` | `[(9,5),(1,0),(1,4)]` | `[(0,5),(4,3),(4,1)]` | 288 | 16 | found via randomized search |
    | `(30,6)` | `[(9,0),(0,1),(0,2)]` | `[(0,3),(25,0),(26,0)]` | 360 | 12 | bravyi; already present in `seed_solution_ansatz.py`'s `REFERENCE_CODES` — re-add explicitly in this fork rather than relying on inheritance |
    | `(18,12)` | `[(0,0),(0,1),(3,11)]` | `[(0,0),(1,0),(17,9)]` | 432 | 12 | `r=2,b=1` gross-family instance — same candidate as checklist item 2's verify-first check |

    All four entries satisfy the inherited evaluator's Stage-1 requirement (`evaluate_stage1`
    requires a valid code with `k>0` at every Stage-1 lattice or the whole batch's score is
    zeroed to a near-zero per-lattice-coverage penalty) with room to spare — every entry reaches
    `k≥12`, not merely `k>0`. `(16,9)`'s "found via randomized search" provenance is not
    otherwise reproducible from anything currently in the repository, unlike `(30,6)` (a
    literature citation) and `(18,12)` (Phase G item 2's own verify-first script) — persist the
    actual search artifact that produced it (e.g. `evolve/data/w6_safety_net_16_9_search.py`
    with a fixed seed) alongside this fork, so it carries the same reproducibility guarantee the
    other three entries already have.

    **`candidate_origin` contract**: every result dict is stamped `"fixed_reference"` or
    `"evolved"`. It cannot live inside the candidate tuple itself — it must be carried as a
    side-channel map keyed by canonical candidate key, because every consumer this fork inherits
    hard-rejects anything but a bare `(A_terms, B_terms)` 2-tuple: `_filter_weight5` does
    `if not candidate_selection.has_arity(cand, 2): rejected += 1; continue`, and
    `evaluate_batch` does a strict `for A_terms, B_terms in candidates:` unpack — appending
    `candidate_origin` as a 3rd tuple element would make every candidate for every lattice
    length-3 and both call sites would discard the *entire* batch, including the safety-net
    entries themselves, before evaluation ever runs. `_make_result_template()`'s output has no
    fixed arity, so stamping the field onto the **result dict** instead adds a new key with zero
    shape-compatibility issue anywhere downstream.

    **Mechanism**:
    - The outer `generate_candidates` entry point — the function that builds
      `origin_by_key`/`_LAST_ORIGIN_BY_KEY` and unconditionally prepends the fixed four-entry
      table — must itself live after `# EVOLVE-BLOCK-END`, not inside the `EVOLVE-BLOCK`.
      Mirror `seed_solution_weight5_css.py`'s split structure (a non-evolvable outer
      `generate_candidates`, placed after the block, that unconditionally prepends the fixed
      table and performs the `global _LAST_ORIGIN_BY_KEY` reassignment below, delegating only
      the LLM-mutable x/y-swap strategy logic — mirrored from `seed_solution_ansatz.py` — to an
      inner, evolvable helper analogous to `_generate_candidates_evolved`) rather than
      `seed_solution.py`/`seed_solution_ansatz.py`'s own shape, which puts the entire
      `generate_candidates` function inside the block. If the outer entry point were mutable, an
      LLM mutation could silently drop or break the `origin_by_key` construction with nothing
      downstream able to detect it — `candidate_origin` never feeds `combined_score`, so a
      broken label produces no score signal, and the evaluator-side retarget-not-delete
      mechanism (below) still forces fixed-reference candidates into evaluation regardless of
      what a mutated `generate_candidates` returns, so they would just resolve via
      `origin_by_key.get(key, "evolved")`'s fallback and get silently mislabeled `"evolved"` for
      the rest of that lineage's run.
    - `generate_candidates(ell, m)` continues to return plain `(A_terms, B_terms)` 2-tuples,
      unchanged in shape, for both the fixed table and the `EVOLVE-BLOCK` strategies. In the
      same call, before returning, it also builds `origin_by_key: dict[str, str]` mapping
      `canonical_candidate_key((A_terms, B_terms))` → `"fixed_reference"` (for the four-entry
      table) or `"evolved"` (for everything else), and exposes it via a module-level attribute
      `_LAST_ORIGIN_BY_KEY` (initialized to `{}` at module scope) that `generate_candidates`
      overwrites on every call using an explicit `global _LAST_ORIGIN_BY_KEY` declaration —
      without it, the assignment creates a local variable shadowing the module-level name
      instead of rebinding it, and the module attribute the evaluator reads via `__globals__`
      would silently stay `{}` forever. The evaluator retrieves this via the bound
      `generate_candidates` function's own `__globals__` attribute: `_load_generate_candidates`
      execs the seed module and returns only the bound function, discarding the module object,
      but a function's `__globals__` is a live reference to its defining module's `__dict__`, so
      the module-level dict is still reachable from the bare function handle.
    - **The read-and-stamp must happen per-lattice, inside `_run_evaluation`'s own per-lattice
      loop — never once, after all lattices have been merged.** `generate_candidates` overwrites
      `_LAST_ORIGIN_BY_KEY` on every call (a rebind, not a merge), so a single read taken after
      the loop finishes would reflect only the most-recently-processed lattice and silently
      mislabel every earlier lattice's fixed-reference candidates as `"evolved"`. The fix:
      immediately after each lattice's own `candidates = generate_fn(ell, m)` call, capture that
      lattice's own snapshot (`origin_by_key = generate_fn.__globals__.get("_LAST_ORIGIN_BY_KEY",
      {})`), and use it to stamp `r["candidate_origin"] = origin_by_key.get(
      canonical_candidate_key((r["A_terms"], r["B_terms"])), "evolved")` onto that same lattice's
      own results, before they're merged into `all_results` — never using a dict captured on a
      different iteration. This is a change to the fork's own copy of `_run_evaluation`'s
      per-lattice loop body only; `_load_generate_candidates`'s and `_run_evaluation`'s
      signatures are unchanged. Step 3's `r.update(build_v2_record(r))` fix runs later,
      post-merge — `candidate_origin` must already be present on every result dict before that
      pass runs, since by then the per-lattice snapshots are no longer distinguishable from each
      other.
    - `_emit_discovery_events` is shared base-file code (`evolve/openevolve_evaluator.py`, which
      Step 3 forbids modifying) and cannot be edited to add a skip check. Instead, at the fork's
      own call site, build a filtered list — `credible_for_events = [r for r in credible_to_save
      if r.get("candidate_origin") != "fixed_reference"]` — and pass that instead of
      `credible_to_save`. `save_code`/`update_pareto_front`/`update_pareto_front_v2` keep
      receiving the unfiltered `credible_to_save` (fixed-reference candidates are harmless there
      — checklist item 5's readout-side exclusion already keeps them out of success/readout
      determination); only the discovery-event log, which `evolve/model_attribution.py`'s
      per-program attribution reads, excludes them at the source. `evolve/model_attribution.py`'s
      attribution is per-*program* (the whole evaluated batch), not per-candidate, so without
      this filter a hardcoded safety candidate returned alongside genuinely LLM-mutated
      candidates in the same batch is indistinguishable from them and gets stamped with that
      batch's `generating_model`/`program_id` like any real discovery — a permanent reference
      candidate re-evaluated every run must never be written to `discovery_events.jsonl` as one
      (see checklist item 5's matching exclusion and the Success Criteria table's M1/M2 gate
      text).
    - **Retarget, don't delete, the inherited weight-5 safety-net-reservation-and-reinsertion
      block.** The base `_run_evaluation` does two independent things: (a) it calls
      `_css_safety_net_candidates(ell, m)` — the weight-5 seed's own hardcoded table — and
      appends its output directly onto `candidates`, after `_filter_weight5` (the gate this fork
      retargets to weight-6 per item 9) has already run and before `evaluate_batch`; and
      (b) independently of which table it reads from, it reserves pre-build-cap budget for
      whatever that table holds and unconditionally re-appends after selection any entry the
      hash-stratified pre-build cap (item 8, ≤500) would otherwise have dropped — the only
      mechanism that guarantees a fixed-reference candidate survives that cap regardless of raw
      candidate volume. Only (a) is harmful here: it reinserts weight-5-shaped candidates that
      bypass item 9's weight-6 enforcement and never pass through `generate_fn`, so they never
      appear in `origin_by_key` and get defaulted to `"evolved"` — reintroducing, via a second,
      unaccounted-for fixed list, exactly the mislabeling this item exists to prevent (at least
      two such leftover weight-5 entries build into genuinely valid, high-`k` codes at 3 of this
      campaign's 4 `W6` Stage-1 lattices, so this pollutes `credible_to_save`,
      `pareto_front_v2.json`, and — since they read `"evolved"` — `discovery_events.jsonl` on
      every run, not just in a corner case). (b) is **not** made redundant by `origin_by_key` —
      `origin_by_key` guarantees correct labeling, not survival to evaluation; this item's own
      four-entry table is bucketed into an ordinary structural stratum alongside evolved
      candidates with no dedicated bucket of its own, so without (b)'s reservation it is exactly
      as exposed to being hash-sampled out of a crowded stratum whenever raw candidate volume
      exceeds the cap, and losing a fixed-reference entry at even one lattice trips
      `evaluate_stage1`'s zero-score coverage penalty for a reason unrelated to the LLM's own
      candidates. **The fix**: delete only the weight-5-specific data source — replace
      `_css_safety_net_candidates(ell, m)` with an equivalent lookup against this item's own
      four-entry table (exposed from this seed file as a same-shaped `_safety_net_candidates(ell,
      m)` helper, imported under a distinct alias such as `_w6_safety_net_candidates`) — keep the
      surrounding reservation/reinsertion *structure* unchanged. Because `origin_by_key` is keyed
      by `canonical_candidate_key`, a force-reinserted entry resolves to the same key as one that
      survived selection unaided, so no change to the stamp mechanism is needed for entries
      reinserted this way.
11. [ ] `evolve/config_weight6_morph18.yaml`, `config_weight6_morph24.yaml` — reuse Direction
    2a's Phase A gates at their **repo-wide permissive defaults** (`QCODE_MIN_K_THRESHOLD=4`,
    `QCODE_MIN_RELEVANT_D=6` — do NOT set these to `12`/`18`/`24`, per the Step 3 design above),
    plus the new `QCODE_K_FLOOR=12`, `QCODE_D_FLOOR=18`/`24`, and
    `QCODE_LATTICE_PROFILE=morph18`/`morph24`. Base the `models:` block on
    `evolve/config_weight5_css_small.yaml`/`_large.yaml`'s own current models — `aws/claude-opus-5`
    (weight `0.33`), `azure/gpt-5.6-sol` (weight `0.34`), `gcp/gemini-3.6-flash` (weight `0.33`),
    all `temperature: 1.0`, `reasoning_effort: "high"` — not `config_noncss.yaml`'s older alias
    set (`aws/claude-opus-4-6`, `azure/gpt-5.3-codex`, `gcp/gemini-3.1-pro-preview`), since this
    fork's lineage is the weight5_css line, not the non-CSS PBB line.
12. [ ] `evolve/seed_solution_weight5_d18d24.py` — fork of Direction 2a's
    `seed_solution_weight5_css.py`, lattice sweep re-centered on the extrapolated `n`-ranges in
    the targets table above. **Inherits the safe EVOLVE-BLOCK split for free**: unlike item 10's
    fork basis (`seed_solution.py`/`seed_solution_ansatz.py`, which puts the entire
    `generate_candidates` inside the `EVOLVE-BLOCK`), `seed_solution_weight5_css.py` already has
    the real, loaded `generate_candidates` sitting *after* `# EVOLVE-BLOCK-END`,
    unconditionally prepending `_safety_net_candidates(ell, m)` before extending with the fully
    evolvable `_generate_candidates_evolved`. This fork needs no separate EVOLVE-BLOCK-placement
    fix — only item 10's fork basis had that gap.

    **Origin-tagging design — simpler than item 10's, by construction**: rather than mirroring
    item 10's `_LAST_ORIGIN_BY_KEY`/`__globals__` per-lattice-capture mechanism (needed there
    because it depends on state a *mutable* `generate_candidates` call produces, which is also
    why item 10 needs the EVOLVE-BLOCK-placement fix and a timing-sensitive per-lattice
    snapshot), this fork defines a static, evaluator-owned `_FIXED_REFERENCE_KEYS:
    frozenset[tuple[int, int, str]]`, built once at **import time** of
    `seed_solution_weight5_d18d24.py` — not
    inside `generate_candidates`, not touching `_LAST_ORIGIN_BY_KEY` or any `global` rebind —
    from two fixed, non-evolutionary sources:
    - this fork's own (inherited, unchanged) `_safety_net_candidates(ell, m)` table, evaluated
      once for every `(ell, m)` in this campaign's fixed lattice superset (item 14's 6-lattice
      `D18`/7-lattice `D24` list — small and static, so a one-time import-time loop over it costs
      nothing); and
    - item 15's persisted manifest (`stage0_profile_d18.json`/`stage0_profile_d24.json`,
      selected by `QCODE_LATTICE_PROFILE`)'s `reference_candidate_by_lattice` field (added by
      item 15 below) — one witnessing candidate per lattice that cleared item 14's own
      per-lattice `k≥12` pass criterion, sourced via the same module-level `_MANIFEST` variable
      described in the bootstrap-case paragraph below (`_MANIFEST["reference_candidate_by_lattice"]`),
      never a second, independent file read — so `_FIXED_REFERENCE_KEYS`'s import-time
      construction is also covered by that paragraph's absent-file default, and `_MANIFEST` must
      be assigned earlier in the module's top-level code than `_FIXED_REFERENCE_KEYS` is built.

    `evaluation/candidate_selection.py`'s `canonical_candidate_key` takes only a term-tuple
    (e.g. `(A_terms, B_terms)`) — it has no `(ell, m)` component, so the same polynomial pair at
    two different lattices collapses to the same key. **`_FIXED_REFERENCE_KEYS` must therefore
    be keyed on `(ell, m, canonical_candidate_key((A_terms, B_terms)))`, never on the bare
    `canonical_candidate_key` result alone** — otherwise a lattice-`X` witness and a
    coincidentally-identical-polynomial evolved candidate at lattice `Y != X` would collide on
    the same key and the evolved candidate would be falsely tagged `fixed_reference`. Every
    entry from both sources above is converted to this triple and unioned into the single
    `_FIXED_REFERENCE_KEYS` frozenset (still built once, at import time, from data already on
    disk). **Origin tagging is a pure membership check the evaluator can run on any result
    dict, at any point after it's built, with no per-lattice capture-and-stamp ordering
    requirement at all**: `r["candidate_origin"] = "fixed_reference" if (r["ell"], r["m"],
    candidate_selection.canonical_candidate_key((r["A_terms"], r["B_terms"]))) in
    seed_solution_weight5_d18d24._FIXED_REFERENCE_KEYS else "evolved"`. There is no mutable,
    per-call side channel here for an EVOLVE-BLOCK mutation to break, because
    `_FIXED_REFERENCE_KEYS` never depends on any particular `generate_candidates` call's return
    value in the first place.

    **`generate_candidates(ell, m)`'s own outer, non-evolvable body** (mirroring
    `seed_solution_weight5_css.py`'s placement identically after `# EVOLVE-BLOCK-END`)
    additionally prepends, alongside the inherited `_safety_net_candidates(ell,
    m)` call, this lattice's own manifest witness if one exists —
    `_MANIFEST["reference_candidate_by_lattice"].get(f"{ell},{m}")` — before extending with
    `_generate_candidates_evolved(ell, m)` and deduplicating, unchanged in structure from the
    base file otherwise. **This function is what the real evolutionary run calls (`_run_evaluation`'s
    per-lattice loop); item 14's Stage-0 audit below must never call it.**

    **Audit-only generation path (prevents re-audit self-contamination)**: item 14's Stage-0
    audit exists to independently validate whether a lattice still clears the gate — but
    `generate_candidates` prepends the *existing* manifest's own witness, so an audit that called
    it would rediscover its own prior output inside the very pool it is auditing, and could
    rubber-stamp an obsolete manifest as still passing instead of re-deriving the answer from
    scratch. This fork therefore also exposes `generate_candidates_audit_only(ell, m)`: the same
    non-evolvable outer body and the same inherited `_safety_net_candidates(ell, m)` prepend (a
    fixed, non-manifest-derived table — including it is not self-contamination), extended with
    `_generate_candidates_evolved(ell, m)` and deduplicated exactly as above, but with the
    `_MANIFEST` witness prepend removed entirely — no read of `_MANIFEST` anywhere in this
    function. Item 14's Reproducibility spec below calls `generate_candidates_audit_only`,
    never plain `generate_candidates`, for exactly this reason.

    **Bootstrap case — manifest file absent entirely**: if the profile's manifest JSON file
    (`stage0_profile_d18.json`/`stage0_profile_d24.json`) is absent from disk entirely — the exact
    state item 14's own *first* Stage-0 audit run is guaranteed to encounter, since item 15's
    manifest is that audit's own output and cannot exist before the audit that produces it has
    run — `_MANIFEST` must default to `{"reference_candidate_by_lattice": {}}` at import time
    rather than raising. This extends the missing-per-lattice-key handling item 15 specifies for
    an existing manifest (a missing key means "no witness for this lattice," never a schema
    error) to the file's total absence. This default is irrelevant to item 14's own audit itself —
    `generate_candidates_audit_only` never reads `_MANIFEST` at all (see above), so the audit path
    has no bootstrap dependency to satisfy. The default exists for the other three real
    `_MANIFEST` consumers, each of which must still import cleanly before any manifest has ever
    been written to disk: plain `generate_candidates` (the real evolutionary path
    `_run_evaluation` calls), the import-time construction of `_FIXED_REFERENCE_KEYS` (above), and
    the evaluator-side retarget fix's manifest-witness union lookup (below).

    **Resolves the four-lattice coverage gap directly**: item 15's manifest only records a
    `reference_candidate_by_lattice` entry for a lattice that already cleared item 14's own
    per-lattice pass criterion (at least 1 candidate with exact `k≥12` in the audited Stage-0
    sample) — so the witness prepended above is guaranteed non-degenerate (`k≥12`) at every
    lattice that actually enters this campaign's real Stage-1 sweep, by construction, rather than
    by hand-picking a replacement candidate for the four lattices (`(10,8), (10,10), (16,8),
    (16,9)`) where the *inherited* `_safety_net_candidates` table (tuned for weight-5-CSS's own,
    smaller, lattice range) evaluates to `k=0`. Unlike item 10's own `(16,9)` entry, this witness
    has a built-in, fully reproducible provenance — it is exactly the candidate item 14's own
    `QCODE_STAGE0_SEED=42`-seeded audit already produced and item 15 already persists, not a
    separately hand-picked or randomly-searched addition.

    **Evaluator-side retarget-not-delete fix (item 6's `openevolve_evaluator_weight5_d18d24.py`
    fork)**: this fork inherits, unchanged, the same pre-build-cap reservation-and-forced-
    reinsertion block item 10 retargets for the `W6` fork, which imports `_safety_net_candidates
    as _css_safety_net_candidates` and unconditionally re-inserts, after the hash-stratified
    pre-build-cap selection (item 8, `≤500`) runs, any entry that selection would otherwise have
    dropped. Mirror item 10's fix exactly: retain the reservation/reinsertion
    *structure* unchanged, but replace the data source with this fork's own combined
    fixed-reference set for that lattice — `_safety_net_candidates(ell, m)` (inherited) unioned
    with `_MANIFEST["reference_candidate_by_lattice"].get(f"{ell},{m}")` (if present) — imported
    under a distinct alias (e.g. `_d18d24_safety_net_candidates`). Without this, the manifest
    witness is exactly as exposed to being hash-sampled out of a crowded structural stratum as
    the `W6` table was before item 10's own fix, defeating the coverage guarantee above whenever
    a lattice's raw candidate volume exceeds the cap.

    **`credible_for_events` filter, mirrored from item 10**: at this fork's own
    `_emit_discovery_events(credible_to_save)` call site, build and pass
    `credible_for_events = [r for r in credible_to_save if r.get("candidate_origin") !=
    "fixed_reference"]` instead of the unfiltered list — for the same reason item 10 states (a
    permanent, unconditionally-re-evaluated reference candidate is not an LLM discovery and must
    never appear in `discovery_events.jsonl`). `save_code`/`update_pareto_front`/
    `update_pareto_front_v2` keep receiving the unfiltered `credible_to_save`, unchanged.

    **Add tests, for both this fork and item 10's `W6` fork**: (i) a hardcoded/manifest-sourced
    fixed-reference candidate that would fail the `≤500` pre-build cap on hash-rank alone (a
    constructed crowded-stratum scenario) still appears in that lattice's post-selection
    `candidates` list; (ii) that same candidate's result dict is stamped
    `candidate_origin=="fixed_reference"` and is present in `pareto_front_v2.json`'s
    `all_records` but absent from `discovery_events.jsonl`; (iii) an ordinarily-evolved candidate
    that happens to coincide with a fixed-reference candidate's exact `(A_terms, B_terms)` **at
    the same lattice** is still tagged `"fixed_reference"` (membership is on
    `(ell, m, content)`, not on which code path produced it — a deliberate, documented
    consequence of a pure key-membership design, not a bug); (iv) for this fork specifically —
    since its `_FIXED_REFERENCE_KEYS` is a single frozenset unioned across every lattice in the
    fixed superset, unlike item 10's per-call-rebuilt `origin_by_key` — the identical
    `(A_terms, B_terms)` pair used as one lattice's manifest witness must **not** be tagged
    `"fixed_reference"` when it appears as an ordinarily-evolved candidate at a *different*
    lattice, confirming the key is the `(ell, m, canonical_candidate_key(...))` triple, not the
    bare candidate-content key alone; (v) for this fork specifically, a lattice with no
    `reference_candidate_by_lattice` manifest entry (i.e. one outside the cleared set) still
    runs `generate_candidates` without error and tags nothing from the manifest as
    fixed-reference, only the inherited `_safety_net_candidates` entries; and (vi) with the
    manifest JSON file entirely absent from disk (the bootstrap case above), the module still
    imports successfully and `_FIXED_REFERENCE_KEYS` contains only the safety-net-derived keys,
    with no exception raised at import time or at any `generate_candidates` call.
13. [ ] `evolve/config_weight5_d18.yaml`, `evolve/config_weight5_d24.yaml` — same
    permissive-default design as item 11, plus `QCODE_LATTICE_PROFILE=d18`/`d24`, and the same
    `evolve/config_weight5_css_small.yaml`/`_large.yaml`-sourced `models:` block item 11
    specifies (`aws/claude-opus-5`/`azure/gpt-5.6-sol`/`gcp/gemini-3.6-flash`, weights
    `0.33`/`0.34`/`0.33`, `temperature: 1.0`, `reasoning_effort: "high"`) — this fork's basis is
    the same weight5_css lineage as item 12's seed fork, so the alias set matches, not
    `config_noncss.yaml`'s older set.
14. [ ] **Pre-pilot gate**: before launching **any** `W5-CSS-D18`/`D24` LLM call — including the
    pilot itself, not only a full/paid run, since a pilot still incurs real LLM API cost — run
    a deterministic Stage-0 exact-`k` audit against the fixed candidate superset defined in the
    Naming and targets section above. `weight-5-codes-survey.md`'s own `d=g+f·n^b` fit is
    measured at only two data points (`k=2`, `k=4`) with no `k≥12` data point anywhere in the
    source — it provides no support for `k≥12` feasibility at this weight, and this audit
    surfaces that gap before money is spent rather than after.

    **Reproducibility spec**: for each `(ℓ,m)` in the fixed superset (`D18`: 6 lattices; `D24`:
    7 lattices — `(15,12)` and `(18,10)` are two distinct lattices that happen to share `n=360`,
    both counted separately, per this document's own established slash-notation convention (see
    the Stage-2 sub-288 `(12,6)`/`(6,12)` pair above, likewise counted as 2)), seed Python's
    `random` module with the fixed constant `QCODE_STAGE0_SEED=42`
    immediately before calling item 12's seed file's `generate_candidates_audit_only(ell, m)`
    exactly once — **never plain `generate_candidates`**, which prepends the existing manifest's
    own witness and would let a re-audit rediscover and rubber-stamp its own prior output (item
    12's audit-only path above) — and take its full returned batch — one deterministic call per
    lattice, the
    same call pattern the real evaluator uses for every other stage in this document (one call
    per lattice per evaluation), not a separate ad hoc sampling procedure. If a single call's
    batch exceeds 500 candidates, select deterministically via
    `evaluation/candidate_selection.py`'s `stratified_select(...)` (`run_seed=QCODE_STAGE0_SEED`,
    this lattice's `ell`/`m`, `max_total=500`, a single stratum since this is one seed-file
    call) rather than positional truncation — the same canonical hash-stratified selector item
    8's per-lattice bound already uses elsewhere, so a differently-ordered but otherwise
    identical candidate population always yields the identical audited sample. **Do not assume a
    single seed-file call stays comfortably under the cap**: the analogous weight-5-CSS seed
    (`seed_solution_weight5_css.py`, this item's own fork basis for item 12) emits a constant
    **7,956** candidates from `generate_candidates(ell, m)` at every lattice at or past the
    saturation condition `ell-1≥6, m-1≥6` — which covers essentially this item's entire `D18`/
    `D24` fixed superset — regardless of the seed file's own docstring, which cites lower figures
    for "Strategy 1 alone," a strict subset of what `generate_candidates` actually returns.
    `stratified_select`'s deterministic-sampling path above should therefore be treated as the
    expected case at this item's own superset lattices, not a rarely-hit fallback.
    Compute exact `k`
    (GF(2) rank — stage (a) only, no distance work) for every candidate in the batch.

    **Per-lattice pass criterion**: a lattice clears the gate iff (a) its deterministic sample
    contains at least 1 candidate with exact `k≥12`, **and** (b) a bounded distance-plausibility
    audit does not show every such `k≥12` candidate stuck at a degenerate distance. Exact `k`
    alone does not rule out a lattice whose every qualifying candidate has, say, `d=2`
    (`A=B` always gives `d=2`, Theorem 1 in `paper/`, undetected by BP-OSD per CLAUDE.md's own
    documented gotcha) or `d≤4` more generally — a lattice like that clears the `k`-only bar
    with no real prospect of ever reaching `d_floor=18/24`. Run `evaluation/distance.py`'s
    `estimate_distance` (a single cheap BP-OSD quick pass, ~100 trials, stage (b) cost only —
    not the full refined/exact cascade) against every `k≥12` candidate in the lattice's sample;
    the lattice fails criterion (b), and therefore fails the gate regardless of (a), if every one
    of those quick passes returns `d≤4`. **Record the witnessing candidate**: for every lattice
    that clears both (a) and (b), record the specific candidate that witnesses criterion (a) —
    the highest exact-`k` candidate in the sample (ties broken by lexicographically smallest
    `canonical_candidate_key((A_terms, B_terms))`, for a fully deterministic pick) — alongside its
    exact `k`. Item 15 below persists this as `reference_candidate_by_lattice`, and item 12's
    fork prepends it at that lattice unconditionally, closing the coverage gap described there.

    **Go/no-go criterion**: **go** iff at least 2 lattices in the fixed superset clear the
    per-lattice bar; **no-go** otherwise. Requiring at least 2 lattices (rather than a single
    global candidate count) rules out the case where one lattice alone supplies every
    qualifying candidate while every other lattice in the superset fails outright — a
    single-lattice pass would not actually demonstrate a workable per-lattice profile.
    `W5-CSS-D18`/`D24` should be re-scoped (smaller `k_floor`, a different fixed superset, or
    deferred entirely) before any further `W5` LLM spend, pilot or full, if no-go.

    The clearing lattices become that campaign's Stage-1 list; the full cleared subset becomes
    Stage 2 (Naming and targets above). This check does not replace Risk item 3 below — it
    makes that risk empirically checkable before money is spent, rather than leaving it as an
    unquantified caveat.
15. [ ] **New**: persist item 14's audit results as a hashed manifest, and make the `d18`/`d24`
    evaluator fork load it instead of hardcoding a Python list. Implement item 14's audit itself
    as a standalone, named script, `evolve/stage0_audit_d18d24.py` (not an unnamed "the audit
    script" — every other artifact this checklist references gets a concrete filename, and this
    one needs one too so the manifest's own digest of it, below, names something that exists).
    Write JSON to `evolve/data/stage0_profile_d18.json` / `stage0_profile_d24.json`, containing:
    a `manifest_schema_version` (starts at `1`, bumped on any field-shape change so an old
    manifest cannot be silently misread by a newer loader), the campaign name,
    `seed_source_file` **and its SHA-256 hex digest** (over the file's raw bytes at audit time —
    a filename alone does not prove which version of `generate_candidates` actually ran; the
    manifest is only as fresh as the seed content it hashes), a SHA-256 digest of
    `evolve/stage0_audit_d18d24.py` itself (or its own version string, if versioned separately),
    a version identifier (a digest or explicit version string) each for **both** implementations
    the audit's two stages depended on: the evaluator/rank implementation stage (a)'s exact-`k`
    computation used (`evaluation/evaluator.py`'s GF(2)-rank path) **and** the distance
    implementation stage (b)'s plausibility audit used (`evaluation/distance.py`'s
    `estimate_distance`) — these can drift independently of each other and of the seed file, so
    one combined digest is not enough to tell which one changed, the git commit hash the audit
    ran at **and a `git_dirty` boolean** (recorded for provenance/audit-trail purposes only — an
    audit run against uncommitted local changes is not reproducible from the commit alone and
    must say so on the record — but, per check (vii) below, this field never gates manifest
    validity), a **`critical_path_hashes`** dict — a SHA-256 hex digest, keyed by path, of
    `evaluation/candidate_selection.py` (the one critical implementation dependency —
    `canonical_candidate_key`/`stratified_select` — not already covered by the seed-file,
    evaluator/distance, or audit-script digests elsewhere in this manifest), the **audit parameters actually used**
    (`QCODE_STAGE0_SEED`, the `k≥12` threshold, stage (b)'s BP-OSD trial count and its `d≤4`
    degenerate-distance threshold — every numeric knob item 14's pass criterion depends on,
    explicit in the manifest rather than implicit in whatever the audit script's defaults
    happened to be that day), the fixed lattice superset, a per-lattice breakdown (candidate
    count returned, count with `k≥12`, the distance-plausibility audit outcome from item 14's
    criterion (b), whether that lattice cleared), **`reference_candidate_by_lattice`**: for every
    lattice in `cleared_lattices`, a `"{ell},{m}"`-keyed entry recording item 14's witnessing
    candidate (`A_terms`, `B_terms`, its exact `k`) — the deterministic highest-`k`,
    lexicographically-smallest-key pick item 14 above now specifies — omitted entirely for any
    lattice not in `cleared_lattices` (item 12's fork must treat a missing key as "no manifest
    witness for this lattice," never as a schema error), the derived `cleared_lattices` list, the
    overall `go`/`no-go` boolean, and a SHA-256 hex digest (`manifest_hash`) computed over the
    canonical (sorted-keys, no-whitespace) JSON encoding of every other field.

    `evolve/openevolve_evaluator_weight5_d18d24.py`'s `QCODE_LATTICE_PROFILE` branch for
    `d18`/`d24` loads this file at import time and, before trusting `cleared_lattices` or
    `reference_candidate_by_lattice`, runs all of the following, **in this order** — every check
    after (i) assumes the manifest has the current schema shape, so (i) must gate all the rest:
    (i) checks `manifest_schema_version` against this loader's own
    `CURRENT_MANIFEST_SCHEMA_VERSION` constant and requires an exact match — a version mismatch
    means the remaining fields may not even have the shape the checks below assume, so no other
    check should run against a manifest that fails this one; (ii) recomputes `manifest_hash` over
    the loaded fields and requires an exact match; (iii) recomputes the seed file's SHA-256 from
    the file on disk right now and requires it match the manifest's recorded digest — a manifest
    is stale, not just internally consistent, if the seed file it was computed against has since
    changed, even though `manifest_hash` alone would not detect that; (iv) runs a semantic
    validation pass: `campaign`/`QCODE_LATTICE_PROFILE` in the manifest matches the profile
    actually being loaded (a `d18` manifest must never satisfy a `d24` load), `go` is `true`,
    `cleared_lattices` has at least 2 unique entries, and `cleared_lattices ⊆ fixed_superset` (no
    lattice outside the declared fixed superset can appear as cleared); **(v) recomputes both
    implementation-version identifiers live** — re-derives the same digest/version string for
    `evaluation/evaluator.py`'s current GF(2)-rank path and `evaluation/distance.py`'s current
    `estimate_distance` and requires each match the manifest's recorded pair — a manifest whose
    seed file and commit hash both still match can nonetheless be silently invalid if the shared
    `evaluator.py`/`distance.py` it depended on has since changed underneath it, which (iii) alone
    cannot detect since it only re-hashes the seed file; **(vi) validates recorded audit
    parameters against this loader's own current expected constants** (`QCODE_STAGE0_SEED`,
    `k≥12`, the BP-OSD trial count, the `d≤4` threshold) and requires an exact match — a manifest
    computed under different parameters (e.g. fewer BP-OSD trials, or a different `k` floor) must
    never be silently reused once those parameters change; **(vii) recomputes, live, the SHA-256
    digest of every path listed in the manifest's own `critical_path_hashes` field** (currently
    just `evaluation/candidate_selection.py`) **and requires each match the manifest's recorded
    value.** This is a deliberately narrow, path-scoped check, not a whole-working-tree
    cleanliness gate: it does **not** require `git_dirty` to be `false` and does **not** require
    the current commit hash to equal the manifest's recorded commit. (iii), (v), (vi), and (ix)
    already re-derive, live, the seed-file/evaluator/distance-implementation/audit-parameter/
    audit-script state that matters; this check adds the one remaining critical-path dependency
    those four don't cover. Together, manifest validity is gated entirely by content hashes of
    named implementation/configuration files — never by whether the working tree as a whole is
    clean. An unrelated commit, or an untracked result artifact from this document's own `W6`
    pilots (which the Launch Commands sequencing below runs first, before any `W5-CSS-D18`/`D24`
    launch, and which write only under `results/` — never under any path this manifest hashes),
    must not force a redundant re-audit of an otherwise still-valid manifest. The manifest's own
    recorded `git_dirty` boolean (above) is retained for provenance/audit-trail purposes only and
    never participates in this loader's pass/fail decision;
    **(viii) recomputes `go` and `cleared_lattices` from the manifest's own stored
    per-lattice breakdown** by re-applying item 14's exact per-lattice pass criterion ((a) at
    least one `k≥12` entry, (b) not every such entry stuck at `d≤4`) and the go/no-go rule (at
    least 2 clearing lattices) to those stored per-lattice fields, and requires the result match
    the manifest's own top-level `go`/`cleared_lattices` fields exactly — this catches a
    hand-edited or buggy-writer manifest whose top-level derived fields disagree with their own
    underlying per-lattice data, which no hash check alone can catch since a hand-edited manifest
    can still recompute a self-consistent `manifest_hash` over its own (already-wrong) fields;
    and **(ix) recomputes the SHA-256 digest (or version string) of
    `evolve/stage0_audit_d18d24.py` itself, live, from disk, and requires it match the manifest's
    recorded value** — none of (iii)/(v)/(vi)/(viii) re-derives the audit script's own logic (they
    re-derive the seed file, the evaluator/distance implementations, the audit parameters, and the
    manifest's internal per-lattice consistency respectively, but not the script that actually
    produced the per-lattice breakdown in the first place), so without this check a change to the
    audit script itself (e.g. its per-lattice candidate-count computation or superset construction)
    made between manifest-write and load time — and, per (vii) above, no longer caught by a
    commit-hash-equality requirement — would go undetected.
    Any of (i)-(ix) failing raises immediately rather than silently running against a
    schema-mismatched, hand-edited, stale, differently-audited, critical-path-stale, or
    internally-inconsistent manifest. This makes "what the audit found," "what the evaluator
    runs," and "what the seed file, evaluator, and distance implementation currently contain" the
    same verified-consistent artifact by construction, rather than
    independently-maintained copies that can drift apart un-noticed.
16. [ ] Extract Shaw-Terhal's exact sufficient condition from the arXiv:2407.16336 full text
    (research-only — the condition is stated for general two-block group-algebra (2BGA) codes,
    not narrowly for BB codes; the extraction target is the condition's precise hypotheses, not
    just its scope) and define a structured **morphability result schema** (e.g. a
    dict/dataclass with one boolean-or-detail field per item 17-21 below, plus supporting
    evidence) that all of items 17-21 populate — this schema, not prose, is what downstream
    reporting (item 24) reads.
17. [ ] Implement the morphability predicate function codifying item 16's extracted condition;
    reproduce gross and two-gross as regression tests (both must pass; both are the paper's own
    worked examples).
18. [ ] Implement the end-cycle partner constructor (the paper's own ancilla-reset contraction
    recipe) — a concrete mid-cycle-code → end-cycle-code map, not merely assumed to exist
    because a mid-cycle candidate has high `d` (Morphability Requirements item 2 below).
    Immediately after constructing each end-cycle code, recheck its exact `k` via this repo's
    own GF(2)-rank machinery (`evaluation/evaluator.py`'s existing rank path — the same one used
    for every other campaign's exact-`k`, not a new computation) and record whether it equals
    the mid-cycle code's `k` (Morphability Requirements item 3 below) — this was the paper's own
    `k`-preservation claim, re-verified per candidate rather than assumed, and it must run
    *before* items 19-21 spend effort on a specific constructed pair whose `k`-preservation was
    never actually checked. Populate item 16's result schema with the outcome.
19. [ ] Implement a per-timestep stabilizer-validity check against the circuit item 18
    constructs (Morphability Requirements item 4 below — the paper's "no vulnerable window"
    claim, re-checked per candidate rather than assumed to transfer from the paper's own
    examples).
20. [ ] Implement a circuit degree/scheduling/connectivity check against item 18's constructed
    circuit (Morphability Requirements item 5 below — the paper's actual physical motivation,
    "degree-5" connectivity).
21. [ ] Implement end-cycle distance re-derivation (for the proven `d_tilde≥d/3` bound) and
    re-measurement (for the empirical `~d/2` ratio) against item 18's specific constructed pair
    (Morphability Requirements item 6 below) — populate item 16's result schema with both, not
    just whichever is cheaper.
22. [ ] Pilot `W6-CSS-Morph18` first (cheapest to validate — literature anchor already exists at
    `n=288`), then `W6-CSS-Morph24`, then either `W5` `D18`/`D24` exploration — item 14's Stage-0
    audit must return a go signal (item 15's manifest) before *any* `W5` LLM call, including
    this pilot, not only a subsequent full/paid run. See the Launch Commands section's
    per-campaign gating list for which Phase G items gate `Morph18`/`Morph24` specifically.
23. [ ] MILP/Webster re-verify (Direction 2a's Phase D path) any discovery before reporting —
    BP-OSD alone is not trustworthy at `d≈18-24` per this repo's documented overestimation
    issue. Feed the certified result back by re-calling `update_pareto_front_v2()` (through
    item 7's single-writer lock) with a fresh v2 record for the same code, re-using the same
    `A_terms`/`B_terms` **term content** the discovery run originally produced — `_code_key`
    sorts each term list internally, so mere re-ordering of the same terms is harmless and
    still merges onto the existing entry; what must actually match is the underlying term set
    itself, not a differently-generated (even if isomorphic) representation of the same code —
    see the `_code_key` non-isomorphism-invariance caveat above, which this item's own re-call
    must respect, not the ordering framing this item previously used.
24. [ ] Re-run items 16-21's morphability predicate suite against any weight-5 `D18`/`D24`
    discovery specifically to determine whether the general-2BGA sufficient condition is
    actually satisfied by a `2+3` BB instance — only once a weight-5 candidate has passed this
    check should its campaign name change from `D18`/`D24` back to `Morph18`/`Morph24`.

## Risk Assessment

Numbered 1-4.

1. **Two-gross's `d=18` is BP-OSD-only, not MILP-proven.** `W6-CSS-Morph18`'s "beat n=288"
   framing could be chasing an inflated baseline, since this repo's own known issues note
   BP-OSD can overestimate distance by up to 12x for high-rate codes. Mitigated by Phase G item
   3 above (verify two-gross directly before relying on it).
2. **The generalized gross-family `d` formula is conjectural and has never been checked at any
   `r>2` instance.** If the `r=2,b=1` check (Phase G item 2) refutes `d=24` for that candidate,
   `W6-CSS-Morph24` loses that literature anchor — but per Phase G item 1's check, it may still
   have the existing `n=360` "bravyi" reference candidate as an anchor if that one confirms;
   only if *both* checks fail to confirm `d=24` does the campaign become a fully open search
   with no anchor at all, a materially different, harder problem than either single-anchor
   framing suggests.
3. **Weight-5's structural distance ceiling (`d≤O(n^(2/3))`, see
   `plans/direction2_weight5_campaigns.md`'s Risk item 2) applies with extra force here.**
   `d≈18`/`d≈24` targets are well past the only concrete weight-5 anchor in the literature
   (`[[30,4,5]]`, and even that is a trivariate, not a standard 2-variable BB, construction).
   `W5-CSS-D18`/`D24` may be searching a regime that is theoretically unreachable at competitive
   `k`, and the fragile scaling-law extrapolation (already stretched past its fitted range in
   Direction 2a) is stretched considerably further here, giving essentially no real prior
   either way.
4. **`k_floor=12` is a judgment call this plan made, not an explicit user requirement — and it
   is a floor, not an equality constraint.** If a real morphing-circuit implementation later
   needs exact-`k` matching (rather than merely `k≥12`) between the mid-cycle and end-cycle
   codes it pairs, a discovered code at `k=14` or `16` with excellent `(n,d)` might not actually
   be usable as a direct swap-in for the gross/two-gross-style architecture this track is
   oriented toward. Flagged for revisiting once a concrete morphing-circuit pairing is
   attempted, not resolved here.

## Morphability Requirements (what makes a discovered code a genuine morphing candidate)

None of this document's numeric success criteria (M1-M4, below) by themselves establish that a
discovered mid-cycle code can actually be morphed into a usable end-cycle pair. Meeting the
`(n,k,d)` bar via the D4 certification contract is necessary but not sufficient. Before
reporting any discovery from this track as a genuine morphing candidate — as opposed to merely
"a high-distance BB code that happens to meet the numeric target" — the following must also be
checked against Shaw & Terhal (arXiv:2407.16336):

1. **Shaw-Terhal's contraction/sufficient condition is stated for *general two-block
   group-algebra (2BGA) codes*, not narrowly for weight-6 BB codes.** Shaw & Terhal's own
   arXiv:2407.16336 abstract states "a sufficient condition for its applicability to two-block
   group algebra codes" (not "to weight-6 BB codes"); this repo's own
   `weight-5-codes-survey.md` §4.4 independently states the same thing ("The paper also gives a
   sufficient condition for applying morphing circuits to general 2BGA codes"). Standard BB
   codes — of *any* stabilizer weight, including weight-5 — are themselves 2-generator instances
   of 2BGA codes, so the sufficient condition's stated scope already covers weight-5 BB codes as
   a special case; the open question is whether a *specific* weight-5 candidate's structure
   satisfies that condition's concrete hypotheses, not whether the condition could ever apply to
   weight-5 in principle. The paper's exact condition still needs to be extracted from the full
   text (see checklist item 16) and evaluated against whatever candidate is discovered; a
   candidate that meets `(n,k,d)` numerically but has not been checked against this condition
   has not been shown to be morphable at all, regardless of its weight.
   - **Additional corroborating source**: Chao (NVIDIA), "Block algebra for morphing circuits,"
     arXiv:2606.12724 (submitted 2026-06-10), gives four CNOT-based CSS morphing-circuit
     constructions (ST, I, II, III, IV) in block-algebra notation; its Construction I is
     described as recovering Shaw-Terhal's own result specifically for 2BGA codes, reinforcing
     that the sufficient condition is a 2BGA-level, not BB-weight-specific, statement. No
     explicit weight-5 discussion was found in Chao's abstract alone — full-text extraction
     during implementation (checklist item 16) is still needed to determine whether Chao's
     constructions add anything weight-5-specific beyond re-deriving Shaw-Terhal.
2. **An explicit end-cycle partner construction.** "Morphable" is a relational property between
   a mid-cycle code and a specific end-cycle code, not a property of the mid-cycle code alone. A
   concrete end-cycle candidate (following the paper's own contraction recipe) must actually be
   constructed, not merely assumed to exist because the mid-cycle code has high `d`.
3. **Exact `k` preservation, checked, not assumed.** The paper's `k`-preservation claim is a
   property of the *specific* contraction map; it should be re-verified (via this repo's own
   GF(2)-rank machinery, not taken on faith) for whatever end-cycle code is actually constructed
   in item 2 — especially for morph campaigns that discover a candidate with `k≠12` under this
   plan's `k_floor` policy (see the `k`-discussion in Naming and targets above).
4. **Valid stabilizers throughout the circuit, not just at the two endpoints.** The paper's
   headline "no vulnerable window" claim is itself a nontrivial per-timestep property of the
   specific contraction circuit — it should be checked for the actual mid-cycle/end-cycle pair
   constructed here, not assumed to transfer automatically from the paper's own worked examples
   to a newly-discovered code.
5. **Circuit degree/depth/scheduling checks.** Even if `d`-degradation and stabilizer-validity
   hold algebraically, the paper's practical motivation is *physical qubit connectivity* (hence
   "degree-5" in its title) — a discovered code's contraction circuit should be checked against
   whatever connectivity/scheduling constraints motivated the search in the first place, not
   just its algebraic code parameters.
6. **End-cycle/circuit-level distance bounds, re-derived per candidate, not assumed from the
   paper's proven/empirical ratios.** `d_tilde≥d/3` is proven for BB-style depth-3 contraction
   generally, and `~d/2` is only an empirical observation from the paper's own worked examples —
   neither should be treated as a guarantee for a newly-discovered mid-cycle code without
   re-deriving (for the proven bound) or re-measuring (for the empirical one) against the
   *specific* contraction actually constructed in item 2.

**The weight-5 branch (`W5-CSS-D18`/`W5-CSS-D24`) is still more speculative than the `W6`
branch against this checklist — but the reason is empirical, not foundational.** Shaw-Terhal's
sufficient condition is stated at the general-2BGA level, which already covers weight-5 BB
codes in principle. The real asymmetry is: gross and two-gross give this document *worked,
checkable weight-6 examples* of the condition actually being satisfied, while no weight-5 BB
code has ever been checked against the condition by anyone — this plan included. That is a gap
in demonstrated instances, not a gap in applicable theory. Per checklist item 24 (above), the
campaigns are named `D18`/`D24` rather than `Morph18`/`Morph24` specifically because that check
has not yet been run for any weight-5 candidate — a `W5` discovery meeting the numeric `(n,k,d)`
bar should be reported as exactly that (a numeric match) until item 24's predicate check is
actually run against it, at which point — if it passes — the campaign's own name should change
back to `Morph18`/`Morph24` to reflect the upgrade from "numeric match" to "checked morphing
candidate."

**Single contract, stated once here and nowhere contradicted elsewhere in this document**:
Implementation Checklist items 16-21 (the morphability predicate/end-cycle-constructor
infrastructure) do **not** gate the launch of any numeric search pilot or full run —
`W6-CSS-Morph18`/`Morph24` and `W5-CSS-D18`/`D24` may all launch with only the verify-first
(Phase G items 1-3) and Stage-0 (items 14-15) gates satisfied, per the Launch Commands
section's per-campaign gating list below, which lists items 16-21 nowhere. Items 16-21 **do**
gate two things downstream of launch: calling any single discovery "a morphing candidate" (as
opposed to "a high-distance BB code meeting the morphing-motivated numeric target"), and any
publication of a morphing claim. Numeric search is useful on its own and can proceed the moment
its own gates clear; the morphability predicate is a reporting/publication gate layered on top
of already-launched search, not a prerequisite to starting it. The predicate itself (extracting
Shaw-Terhal's exact condition, implementing an end-cycle constructor, and reproducing
gross/two-gross as regression tests) is split across items 16-21 above by morphability sub-task
(item 18 covers both the end-cycle constructor and the immediate `k`-preservation recheck,
since the latter must run against the same object the former constructs; every other item is
one item per sub-task), each populating the item-16 result schema, rather than one bundled item
covering research, predicate design, construction, and tests all at once. Checklist item 24
covers re-running that same predicate suite against any actual weight-5 discovery once one
exists; the same re-run requirement applies symmetrically to any `W6` discovery before it is
reported as a checked morphing candidate rather than a numeric match, even though gross and
two-gross already satisfy the predicate in the literature — a *newly discovered* `W6` code
still needs its own predicate check, not an inherited pass from the anchors it was searched
alongside.

## Success Criteria

Each row states the actual pass/fail gate: the concrete condition that determines whether that
tier is met, checkable directly against a run's output.

| Tier | Criterion | Gate (measured against actual run output) |
|------|-----------|------------|
| M1 (ambitious) | `W6-CSS-Morph24` finds a weight-6 CSS BB code with `k≥k_floor=12` and `d_lower≥d_floor=24` (target-achieving tier, per the three-tier archive readout in Step 2 — floor, not exact cap) per `plans/direction2_weight5_campaigns.md`'s D4 certification contract, at `n<n_anchor` (**strictly less than** — this tier's objective is to beat the anchor, not tie it), where **`n_anchor` is conditional**: `n_anchor=360` if Phase G item 1 (bravyi) confirms `k=12,d=24`; else `n_anchor=432` if Phase G item 2 (`r=2,b=1`) confirms; else **no anchor exists** — see the Gate column's explicit no-anchor branch, not a fully-open, unconstrained-`n` search. **This is a numeric match, not by itself a "morphing candidate" claim — see the Morphability Requirements section above** | **Met iff** the checklist item 5 joined archive readout reports at least one code, **excluding any `candidate_origin=="fixed_reference"` entry** (checklist item 10 — the bravyi/`r=2,b=1` anchors are themselves permanent safety-net candidates in this campaign's own seed fork and must never satisfy this gate merely by being re-evaluated every run), attributed to the `W6-CSS-Morph24` run with merged `k≥12 ∧ d_lower≥24` at `n<n_anchor` (target-achieving tier) **when an `n_anchor` exists** (i.e. Phase G item 1 or item 2 confirmed). **No-anchor branch (both Phase G items 1 and 2 fail to confirm `k=12,d=24`)**: `n_anchor` is undefined, so the `n<n_anchor` clause cannot be evaluated at all — M1 is **not applicable** in this branch, not silently reinterpreted as an unconstrained search. Report any merged `k≥12 ∧ d_lower≥24` code found under this run as an M3-style frontier point (achieved `(n,k,d)`, stated plainly) instead, and record M1 itself as N/A with the reason ("no confirmed anchor"). **Not found within budget** (anchor-exists branch only) — a finite unsuccessful run never proves unattainability, it only shows nothing was found under the budget actually spent — if an `n_anchor` exists (Phase G item 1 or item 2 confirmed `k=12,d=24`) *and* the `W6-CSS-Morph24` run itself surfaces no merged `k≥12 ∧ d_lower≥24` entry with `n<n_anchor`; in that case M1 should be reported as not met within this run's budget, not as proven impossible, and not re-estimated as a probability. (If instead both Phase G items 1 and 2 return a `d_upper` witness below `24` — anchor refuted, not merely unconfirmed, and note that "refuted" is one way to fail to confirm, not a separate case — that is the **no-anchor branch above**, not this one: M1 is N/A there regardless of what the run surfaces. The Launch Commands note's/Risk item 2's/Phase G item 2's "fully open search" language describes that branch's search-framing/objective for the campaign itself, a separate concept from M1's own N/A gate disposition here.) |
| M2 (valuable) | `W6-CSS-Morph18` finds a weight-6 CSS BB code with `k≥k_floor=12` and `d_lower≥d_floor=18` (target-achieving tier, floor, not exact cap) per the same D4 certification contract, at `n<288` (beats two-gross's own `n`). **Same caveat as M1: numeric match only, see Morphability Requirements above** | **Met iff** the checklist item 5 joined archive readout reports at least one code, **excluding any `candidate_origin=="fixed_reference"` entry** (checklist item 10), attributed to the `W6-CSS-Morph18` run with merged `k≥12 ∧ d_lower≥18` at `n<288`. **Not met** if the pilot (checklist item 22, 50 iterations) and any follow-on run complete without such an entry — two-gross itself required a broad search, so a negative result here is informative on its own (feeds Risk item 1 — two-gross's own `d=18` being BP-OSD-only, the risk this tier's own framing is built on) and does not by itself block M3/M4 |
| M3 (informative) | Either `W5` morph campaign maps out an empirical frontier of achievable `(n,k,d)` under this plan's stated search space, lattice range, and compute budget (see Direction 2a's Compute Budget section, reused here) at the `d≈18`/`24` targets, independent of whether it beats weight-6 — a statement about what was found under a declared, bounded search, not a proof that the extrapolated 2BGA scaling law is confirmed or refuted in general | **Reported, not pass/fail, per completed `W5-CSS-D18`/`D24` run (pilot or full)**, read from the same checklist item 5 joined readout used for M1/M2 and therefore inheriting its `candidate_origin=="fixed_reference"` exclusion (checklist item 12) — the frontier claim is the *full* multiset of `(n,k,d)` tuples the run actually logs, **excluding item 12's manifest-witness and inherited-safety-net candidates**, reported as-is however small, not characterized as more complete than it is; a run producing only one or two entries should be described as exactly that ("found N points"), not as having "mapped an empirical frontier." **Not produced at all if** checklist item 14's Stage-0 audit returns a no-go (see item 14's per-lattice/minimum-passing-lattices criteria) and the `W5` runs are consequently never launched |
| M4 (baseline) | All three verify-first checks (Phase G items 1-3: the bravyi `n=360` reference, the `r=2,b=1` conjecture, and two-gross's own `d=18`) complete their cheap stages (a)-(b) — exact-`k` and a `d_upper` witness — regardless of outcome, producing a checkable exact-`k`-plus-upper-bound result for three open questions the source literature itself left unverified. A `d_upper` witness alone only bounds distance from above; it does not establish the true value | **Met iff** all three of Phase G items 1-3 return a stage (a)-(b) result (exact `k` plus a `d_upper` witness), independent of whether that result confirms or refutes the literature claim being checked. Items 1 and 2 have **already returned their exact-`k` half** (`k=12` confirmed for both by direct construction); the `d_upper` witness half remains for all three. **Stage (c) exact certification for all three is tracked separately** and is not required for M4 — size its own time box from `campaign7_report.md`'s observed MILP/HiGHS behavior at comparable `n` (`n≥180-360`; non-CSS/symplectic-formulation precedent, per Step 1's caveat) rather than treating it as part of this baseline gate |

M4's cheap stages alone justify Phase G's first step regardless of whether any search campaign
ships: stages (a)-(b) cost a handful of direct evaluator calls per check and resolve `k` and a
`d_upper` witness for a conjecture the generalized gross-family's own originating paper never
checked. Reaching full exact certification (stage (c)) for all three, however, is a
separately-budgeted effort that should not be assumed to complete on the same near-zero-cost
timeline — size its time box using `campaign7_report.md`'s observed MILP/HiGHS behavior at
comparable `n` (non-CSS/symplectic-formulation precedent — see Step 1's caveat), not the
cheap-stage cost.

## Launch Commands

**Note**: none of these can run until `plans/direction2_weight5_campaigns.md`'s Phase A and
this document's own Phase G checklist items land. The three verify-first checks (Phase G items
1-3: bravyi `n=360`, `r=2,b=1` at `n=432`, and two-gross's own `d=18`) are not
`run_evolution.py` invocations at all — each is a one-off script calling
`evaluation/bb_code.py`/`evaluator.py` directly against a single candidate.

**This document inherits, and does not restate its own copy of, Direction 2a's "Before any
pilot" go/no-go gate** (`plans/direction2_weight5_campaigns.md`'s Compute Budget and Decision
Gates section): primary-source fit constants checked, **all three LiteLLM model aliases and
reasoning settings smoke-tested**, the relevant Stage-0 report reviewed, **the provenance event
log and `evolve/provenance_reducer.py` reducer tests pass**, v2 distance/archive schema tests
pass, **the full `uv run python -m pytest tests/ -v` suite passes with no regression in
existing campaigns**, and **the configured Codex pre-review hook has been invoked on the staged
implementation diff**. Every launch block below is additionally gated on this checklist, not
only on the per-campaign items enumerated below — stated explicitly here so it is a checked
requirement of *this* document's own launch sequence, not an assumption silently carried over
from Direction 2a without being re-affirmed.

**The `morph18`/`morph24` Stage-2 lattice profile (17 lattices — Stage 1's 4 plus Stage 2's 7
plus `morph18`'s own 6 sub-288 additions, Naming and targets above) is 3-6x larger than any
single Stage-2 lattice count Direction 2a's own Compute Budget section was calibrated against**
(Direction 2a's small profile: 3 Stage-1 + 3 Stage-2; large profile: 3 Stage-1 + 3 Stage-2 + 5
boundary-extension, 11 total at most; the base evaluator's own default `STAGE2_LATTICES` is
smaller still). The per-lattice CPU-hour ceilings in that section (4/8 CPU-hours cheap census,
4 CPU-hours LC-CSS, 4/24 CPU-hours certification) are *per lattice*, but a 17-lattice full sweep
still multiplies total wall-clock/CPU spend well past anything Direction 2a's own campaigns
incurred at once. Do not adopt Direction 2a's budget unchanged for the `morph18`/`morph24`
Stage-2 profile: benchmark the actual 17-lattice sweep's aggregate cost (cheap census +
LC-CSS + certification, at both the 50-iteration pilot and 300-iteration full-run token/CPU
scales) before committing to a full run, and record the result alongside the Stage-0 manifest
(item 15) or an equivalent artifact.

**Per-campaign gating** (each launch block below depends on a specific anchor's stages (a)-(b),
not a blanket "all three Phase G items before anything"):
- `W6-CSS-Morph18` depends on Phase G item 3 (two-gross's own `d=18` claim) reaching stages
  (a)-(b).
- `W6-CSS-Morph24` depends on Phase G items 1 and 2 (bravyi `n=360`, `r=2,b=1` `n=432`) both
  reaching stages (a)-(b) — either can serve as the anchor (see Step 1 above), and checklist
  item 22 additionally requires the `Morph18` pilot to run first as the cheaper validation.
- The `W5` `D18`/`D24` pilots depend on checklist item 14's Stage-0 audit returning a go signal
  (item 15's manifest) — **not** on Phase G items 1-3 at all, since those are CSS-BB-trinomial
  anchors specific to the weight-6 morphing target and have no bearing on the `W5` lattice-sweep
  search space.
- **None of the four launch blocks above depend on checklist items 16-21** (the morphability
  predicate/end-cycle-constructor infrastructure) — those gate reporting any resulting discovery
  as a checked morphing candidate and gate publication of morphing claims (see the
  Morphability Requirements section above and checklist item 24), not the launch of numeric
  search itself. This is stated explicitly here, not left to be inferred from their absence
  above, since checklist items 16-21 (above) sit physically between the Phase G/Stage-0 items
  each launch block above actually depends on and this Launch Commands section — a reader
  working through the checklist in order could otherwise assume they gate everything after
  them. The contract is: not blocking, stated once (Morphability Requirements section above)
  and repeated here for anyone reading the Launch Commands section on its own.

A `d_upper` witness at the target weight never "confirms" the floor is met — only a `d_lower`
bound at or above the target does; a `d_upper` witness only ever proves `d≤d_upper` (an upper
bound), and finding nothing smaller than the target during stages (a)-(b) means the target is
**"plausible, not refuted," not "confirmed."** Per Risk item 2 above, if *both* item 1 (bravyi)
and item 2 (`r=2,b=1`) come back with a `d_upper` witness below `24` (i.e. the target is
actually refuted, not merely unconfirmed), `W6-CSS-Morph24`'s framing changes from "beat this
anchor" to "fully open search." The pilots below may still start once stages (a)-(b) complete
for the relevant anchor per the per-campaign gating above — a `d_upper` witness is a real,
useful result for search pressure on its own (see the constrained score, Step 2) — but any
report of these campaigns' outcomes as meeting M1/M2 must cite `d_lower`, per the three-tier
archive readout in Step 2 above, not a `d_upper` witness; full exact certification (stage (c),
`distance_schema.is_exact(d_lower, d_upper, bound_sources)`) is separately time-boxed and should
not block starting the pilots below if it is still running, but is required before claiming the
*exact* distance rather than just the floor.

**Unlike Direction 2a's launch commands (`plans/direction2_weight5_campaigns.md`), the four
below deliberately omit `QCODE_FOM_THRESHOLD_REFINE`/`_EXACT`.** This is intentional, not an
omission: these campaigns' target FOM (`k_floor=12, d≥18` → `FOM≈13.5` at `n=288`;
`k_floor=12, d≥24` → `FOM≈16` at `n=432`) already clears the default `6.0`/`8.0` refine/exact
gates with room to spare, so the defaults don't need overriding here. Re-check this arithmetic
against whatever `k,d` the pilots actually produce before assuming it still holds at smaller `n`
or lower `k`.

**The four blocks below leave `QCODE_MIN_K_THRESHOLD`/`QCODE_MIN_RELEVANT_D` unset rather than
setting them to this track's floors.** Setting `QCODE_MIN_K_THRESHOLD` directly to `10` would
hard-reject every `k<10` candidate pre-distance, destroying the search gradient these campaigns
need (per the Step 3 design above); it is left unset (repo-wide default `4`).
`QCODE_MIN_RELEVANT_D` is left unset too, though it is inert for this campaign's evaluator
lineage either way — see the Step 3 design above for why (it never reads that gate; the
flat-`0.01` mapping is specific to the base evaluator's `evaluate_stage2_milp`, a function this
lineage never calls). The actual floors are
carried by the two new env vars `QCODE_K_FLOOR=12`,
`QCODE_D_FLOOR=18`/`24` that the new constrained `combined_score` (checklist item 4) reads
instead. The `W6` blocks also add `QCODE_LATTICE_PROFILE=morph18`/`morph24` (checklist item 8)
so the evaluator's lattice sweep actually reaches `n≈288-450` rather than falling through to the
CSS evaluator's existing default lattices (largest `n=360`); the `W5` blocks add the matching
`QCODE_LATTICE_PROFILE=d18`/`d24` for the same reason.

**Every block below passes `--evaluator <path>` explicitly.** `evolve/run_evolution.py`'s
`--evaluator` flag overrides the hardcoded `EVALUATOR`/`EVALUATOR_NONCSS` constants
unconditionally, so without it these commands would silently run against the base
`openevolve_evaluator.py` — which has none of the `QCODE_K_FLOOR`/`QCODE_D_FLOOR`/
`QCODE_LATTICE_PROFILE` plumbing this campaign depends on, and would ignore the env vars above
entirely rather than erroring. The `W6` blocks pass `--evaluator
evolve/openevolve_evaluator_weight6_morph.py` (checklist item 6's fork); the `W5` blocks pass
`--evaluator evolve/openevolve_evaluator_weight5_d18d24.py` (checklist item 6's other fork,
per the finalized fork decision in Step 3 above — not the base `weight5_css.py` evaluator,
which lacks the `k_floor`/`d_floor`-gated score, the `max`-not-`sum` reduction, and the
pipeline-ordering fix this campaign's score depends on).

`W6-CSS-Morph18` pilot (50 iterations; validate before `W6-CSS-Morph24`/either `W5` morph
campaign, per checklist item 22's sequencing rule):
```bash
QCODE_K_FLOOR=12 QCODE_D_FLOOR=18 QCODE_LATTICE_PROFILE=morph18 \
QCODE_SAVE_FOM_THRESHOLD_CSS=2.0 QCODE_SAVE_TRUST_RATIO=2.0 \
  uv run python evolve/run_evolution.py \
  --evaluator evolve/openevolve_evaluator_weight6_morph.py \
  --config evolve/config_weight6_morph18.yaml \
  --seed evolve/seed_solution_weight6_morph.py \
  --iterations 50 \
  --run-name weight6_css_morph18_pilot_v1
```

`W6-CSS-Morph24` (after the `r=2,b=1` verify-first check and the `Morph18` pilot):
```bash
QCODE_K_FLOOR=12 QCODE_D_FLOOR=24 QCODE_LATTICE_PROFILE=morph24 \
QCODE_SAVE_FOM_THRESHOLD_CSS=2.0 QCODE_SAVE_TRUST_RATIO=2.0 \
  uv run python evolve/run_evolution.py \
  --evaluator evolve/openevolve_evaluator_weight6_morph.py \
  --config evolve/config_weight6_morph24.yaml \
  --seed evolve/seed_solution_weight6_morph.py \
  --iterations 300 \
  --run-name weight6_css_morph24_v1 \
  --wandb
```

`W5-CSS-D18`/`W5-CSS-D24` pilots (only after the `W6` morph pilots above show the infra works
end-to-end — weight-5's grounding here is far weaker, per Risk #3):
```bash
QCODE_K_FLOOR=12 QCODE_D_FLOOR=18 QCODE_LATTICE_PROFILE=d18 \
QCODE_SAVE_FOM_THRESHOLD_CSS=2.0 QCODE_SAVE_TRUST_RATIO=2.0 \
  uv run python evolve/run_evolution.py \
  --evaluator evolve/openevolve_evaluator_weight5_d18d24.py \
  --config evolve/config_weight5_d18.yaml \
  --seed evolve/seed_solution_weight5_d18d24.py \
  --iterations 50 \
  --run-name weight5_css_d18_pilot_v1
```
```bash
QCODE_K_FLOOR=12 QCODE_D_FLOOR=24 QCODE_LATTICE_PROFILE=d24 \
QCODE_SAVE_FOM_THRESHOLD_CSS=2.0 QCODE_SAVE_TRUST_RATIO=2.0 \
  uv run python evolve/run_evolution.py \
  --evaluator evolve/openevolve_evaluator_weight5_d18d24.py \
  --config evolve/config_weight5_d24.yaml \
  --seed evolve/seed_solution_weight5_d18d24.py \
  --iterations 50 \
  --run-name weight5_css_d24_pilot_v1
```

## Sources and confidence

**Confidence.** Every technical claim in this design (Naming and targets, Technical Approach
Steps 1-3, Implementation Checklist, Risk Assessment, Morphability Requirements, Success
Criteria, Launch Commands) has been checked against this repository's actual source — file
paths, line numbers, gate defaults, evaluator code paths, and cited report figures are confirmed
by direct reading or read-only execution of the real files, not asserted from memory. It has
not been through a `codex exec` pass — the one described in
`plans/direction2_weight5_campaigns.md`'s Sources section covers Direction 2a only.

- Shaw, M. H. & Terhal, B. M., "Lowering Connectivity Requirements For Bivariate Bicycle Codes
  Using Morphing Circuits," arXiv:2407.16336 / PRL 134, 090602 (2025): primary source for the
  morphing mechanics (ancilla-reset contraction preserving `k`, no vulnerable window,
  `d_tilde≥d/3` proof, `~d/2` empirical, weight-9 end-cycle codes) and the
  `[[288,12,18]]→[[144,12,12]]` worked example (its Table I). Read via full-text agent research,
  not independently re-verified against the arXiv LaTeX source directly — lower confidence than
  the Tour de gross citations below, which were full-text grepped directly.
- Beverland et al., "Tour de gross: A modular quantum computer based on bivariate bicycle
  codes," arXiv:2506.03094: source for gross/two-gross's own parameters (confirmed unchanged
  from Bravyi et al.'s original gross-code paper), the explicit hedge that two-gross's `d=18` is
  BP-OSD/randomized-prior-decoding-only (336 weight-18 X logicals, "strongly believe" complete —
  not MILP/exhaustively proven), and the Conclusion's generalized gross-family conjecture
  (`ℓ=6(r+b), m=6r, k=12, d=6(2r+b-1)`) that motivates the `r=2,b=1` verify-first check. Verified
  via full-text grep of the paper's own LaTeX e-print source by a research agent, not just its
  abstract. The `r=2,b=1` candidate's specific `n=432` and
  conjectured `d=24` were computed directly from the stated formula; its `k=12` half and `n=432`
  have since been independently confirmed by direct construction (Phase G item 2).
- `weight-5-codes-survey.md` §5.2's fit is reused here from Direction 2a, but stretched
  considerably further outside its fitted range (`n≤100/200`) than Direction 2a's own `d≈10/20`
  orientation points — treat `W5-CSS-D18`/`D24`'s `n`-anchors as even more speculative than
  Direction 2a's already-hedged numbers, not equally hedged.
- Ott, T., Hetényi, B. & Beverland, M., independent MILP/exhaustive re-confirmation of gross's
  `d=12` (arXiv:2502.16408) — cited for context on why gross's own distance is considered solid
  while two-gross's is not; not independently re-read in full — relayed via research-agent
  synthesis only.
- CLAUDE.md's own "Known issues and developer gotchas" section (BP-OSD variance, overestimation
  up to 12x) — used directly to motivate Risk item 1 and the general "MILP/Webster-verify before
  trusting" framing throughout this document.

**Open verification caveats:**
- The Stage-0 profile manifest (checklist item 15) is a new design, checked against the real
  source of `evolve/provenance_reducer.py`'s locking pattern but not yet exercised end-to-end
  against a genuine concurrent multi-writer run.
- The pareto_v2 single-writer lock (checklist item 7) is *not* a new design:
  `evaluation/pareto_v2.py`'s `update_pareto_front_v2()` already implements it in production
  code (confirmed by direct source reading). This plan's role here was verifying that fact, not
  designing the lock — but the lock itself has not been exercised against a genuine concurrent
  run, only single-writer usage.
- Checklist item 16's full-text extraction of Shaw-Terhal's exact sufficient condition has not
  been performed yet — only its scope (stated for general 2BGA codes, not narrowly for BB codes)
  has been confirmed from the survey and the paper's own abstract.
- Distance certification (stages (b)-(c)) for Phase G items 1-3 has not been run yet — only the
  exact-`k` half of items 1 and 2 has been confirmed so far (see Success Criteria M4).

This document's checklist (above) is the authoritative list of what remains to be built; the
Naming and targets, Technical Approach, Risk Assessment, and Morphability Requirements sections
above are the current, final design — not a draft awaiting further reconciliation.
