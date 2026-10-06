# Direction 2: Weight-5 Quantum LDPC Discovery Campaigns

## Status

Implementation plan. Infrastructure development may begin immediately. Stage-0 audits,
model-proxy smoke tests, and primary-source verification are mandatory before any paid LLM
pilot. Full campaigns remain gated on pilot results and distance-certification capacity.

The work starts from `public/main` on branch `weight5-campaigns-plan`. Existing unrelated
worktree changes must be preserved.

## Objective

Build four OpenEvolve campaigns for stabilizer-generator weight-5 codes with `n<500`:

| Campaign | Family | Initial n range | Distance orientation |
|---|---|---:|---:|
| `W5-CSS-Small` | CSS bivariate bicycle / 2BGA | 72-180 | `d≥10` |
| `W5-CSS-Large` | CSS bivariate bicycle / 2BGA | 216-494 | `d≥20` |
| `W5-PBB-Small` | non-CSS perturbed BB | 72-180 | `d≥10` |
| `W5-PBB-Large` | non-CSS perturbed BB | 216-494 | `d≥20` |

The distance values orient lattice selection; they are floors, not caps. The optimization
objective remains:

```text
FOM = k d² / n
```

Codes with larger distance, larger FOM, or a better Pareto trade-off outside a campaign's
nominal range must be retained.

The campaigns use these LiteLLM aliases:

- `aws/claude-opus-5`
- `azure/gpt-5.6-sol`
- `gcp/gemini-3.6-flash`

Every saved discovery must retain the exact model alias and evolved program that produced
it, including independent rediscoveries of the same code.

## Scientific Scope and Assumptions

### Weight convention

This plan targets stabilizer-generator Pauli row weight 5. It does not classify a code as
weight-5 merely because it has gauge checks of weight 5 or qubit degree 5.

For CSS BB codes:

```text
|supp(A)| + |supp(B)| = 5
```

The canonical split is `|A|=2`, `|B|=3`. Swapping `A` and `B` only exchanges the two
qubit sectors, so the `3+2` mirror is not generated separately.

For PBB codes with stabilizer layout

```text
H = ( A  B | C   D )
    ( 0  0 | Bᵀ Aᵀ )
```

the Block-1 row weight is

```text
|supp(A) ∪ supp(C)| + |supp(B) ∪ supp(D)|.
```

Given a weight-5 `A,B` backbone, row weight is exactly 5 if and only if

```text
supp(C) ⊆ supp(A)
supp(D) ⊆ supp(B).
```

Empty `C` or empty `D` is allowed independently. Only `C=D=∅` is the CSS case.

### Literature anchors

`investigations/weight-5-codes-survey.md` supplies the initial literature context. Its reported 2BGA fit
parameters are hypotheses until checked directly against Lin and Pryadko,
*Quantum two-block group algebra codes*, arXiv:2306.16400, §5.2:

- `k=2`: `f=1.295±0.027`, `g=-2.528±0.272`, `b=0.55`
- `k=4`: `f=1.244`, `g=-1.916`, `b=0.50`

Using `d=g+f n^b`, these imply approximate orientation points:

| k | `d≈10` | `d≈20` |
|---:|---:|---:|
| 2 | `n≈62` | `n≈180` |
| 4 | `n≈92` | `n≈310` |

The primary-source check must confirm the fit form and constants before these values are
used to approve a pilot. The campaign does not depend on the fits being correct: Stage 0
measures the actual bounded families before LLM spending.

The first implementation covers BB/2BGA CSS codes and PBB codes. If Stage 0 shows that
both bounded families cannot clear the exploration gates, the next design step is a
separate campaign for more general group-algebra or balanced-product constructions rather
than increasing LLM spend on an unproductive ansatz.

## Campaign Lattices

For all BB/PBB lattices, `n=2ℓm`.

### Small profile

| Stage | `(ℓ,m)` | n |
|---|---|---:|
| Stage 1 | `(6,6)`, `(6,9)`, `(9,8)` | 72, 108, 144 |
| Stage 2 | `(6,10)`, `(9,9)`, `(10,9)` | 120, 162, 180 |

### Large profile

| Stage | `(ℓ,m)` | n |
|---|---|---:|
| Stage 1 | `(12,9)`, `(12,12)`, `(15,12)` | 216, 288, 360 |
| Stage 2 | `(16,12)`, `(15,14)`, `(18,12)` | 384, 420, 432 |
| Boundary extension | `(20,10)`, `(21,11)`, `(17,14)`, `(22,11)`, `(19,13)` | 400, 462, 476, 484, 494 |

CSS and PBB use the same profiles so their results are comparable at matched block sizes.
Construct every listed lattice with qLDPC during preflight before treating the tables as
final.

## Phase A: Shared Infrastructure

### A1. Explicit evaluator selection

Add `--evaluator PATH` to `evolve/run_evolution.py`. It must default to the current CSS or
non-CSS evaluator exactly as today, validate that the file exists, and pass the selected
path through fresh and resumed runs.

Create:

| File | Purpose |
|---|---|
| `evolve/openevolve_evaluator_weight5_css.py` | CSS weight/backbone gates and lattice profiles |
| `evolve/openevolve_evaluator_weight5_pbb.py` | PBB weight, commutativity, and non-CSS gates |
| `evolve/seed_solution_weight5_css.py` | CSS candidate generator |
| `evolve/seed_solution_weight5_pbb.py` | PBB candidate generator |
| `evolve/prompt_context_weight5_css.md` | CSS campaign prompt |
| `evolve/prompt_context_weight5_pbb.md` | PBB campaign prompt |
| `evolve/config_weight5_css_small.yaml` | CSS small config |
| `evolve/config_weight5_css_large.yaml` | CSS large config |
| `evolve/config_weight5_pbb_small.yaml` | PBB small config |
| `evolve/config_weight5_pbb_large.yaml` | PBB large config |

Shared behavior should be factored into helpers rather than copied from the existing
evaluators. Existing campaign defaults must remain unchanged.

### A2. Weight enforcement

The evaluator, not only the seed, enforces the campaign definition.

CSS candidates are rejected unless:

- both term lists are valid for the lattice;
- `len(A_terms)+len(B_terms)==5`;
- the canonical split is `2+3` after normalization;
- the constructed code has `k>0`.

PBB candidates are rejected unless:

- the `A,B` backbone satisfies the CSS checks above;
- `C⊆A` and `D⊆B` as reduced exponent-pair sets;
- the constructed matrices independently confirm the same support containment;
- `A Cᵀ + B Dᵀ` is symmetric over GF(2);
- the code passes both non-CSS checks in A4.

Matrix-level validation is mandatory because symbolic polynomials can diverge from
constructed circulants if qLDPC receives a bare SymPy expression instead of an explicit
`sympy.Poly`.

### A3. Gate configuration

Make these values environment-overridable while retaining current defaults for all
existing campaigns:

| Environment variable | Existing default | Weight-5 value |
|---|---:|---:|
| `QCODE_MIN_K_THRESHOLD` | 4 | 2 |
| `QCODE_MIN_K_THRESHOLD_NONCSS` | 1 (`k>0`) | 2 |
| `QCODE_MIN_RELEVANT_D` | 6 | 4 |
| `QCODE_MIN_RELEVANT_D_NONCSS` | 5 | 5 |
| `QCODE_FOM_THRESHOLD_REFINE` | 6.0 | 3.0 |
| `QCODE_FOM_THRESHOLD_EXACT` | 8.0 | 5.0 |
| `QCODE_SAVE_FOM_THRESHOLD_CSS` | 6.0 | 2.0 |
| `QCODE_SAVE_FOM_THRESHOLD_NONCSS` | 4.0 | 2.0 |
| `QCODE_SAVE_TRUST_RATIO` | 1.3 | 2.0 |
| `QCODE_SAVE_TRUST_RATIO_NONCSS` | 2.5 | 2.5 |

Thread `QCODE_FOM_THRESHOLD_EXACT` through both CSS evaluator call sites, including the
normal path that currently passes `float("inf")`.

An exact result bypasses heuristic trust and FOM persistence gates. The overrides apply
only to heuristic selection and persistence.

### A4. PBB builder and non-CSS classification

Update `evaluation/pbb_code.py` so `C` and `D` are independently optional:

- `C=D=∅` returns the existing CSS `BBCode`;
- otherwise an empty component becomes an `ℓm×ℓm` zero matrix;
- a nonempty component is validated normally;
- commutativity is checked on the resulting matrices.

This is part of the search definition. Valid weight-5, non-LC-CSS candidates exist with
exactly one empty perturbation component, so the builder must not discard them.

Every retained PBB candidate must pass both repository checks with their actual APIs:

```python
verify_not_lc_css(
    ell, m, A_terms, B_terms, C_terms, D_terms
)["is_lc_css"] is False

is_equivalently_css(constructed_code)["is_css"] is False
```

Run cheap algebraic LC-CSS pattern checks first, cache the full result by canonical
`(ell,m,A,B,C,D)`, and defer the full pair of checks until after weight, commutativity,
and `k` filters. Passing these checks means “not equivalent to CSS under implemented
checks”; it is not a proof against every local-Clifford pattern.

Correct `evolve/config_noncss.yaml` and the new PBB prompt to use `[C|D]` and
`A Cᵀ+B Dᵀ`. They must say that any nontrivial `(C,D)` pair is only a non-CSS candidate,
including pairs with exactly one empty component.

### A5. Candidate selection before distance

Candidate reduction has two distinct stages.

#### Pre-build candidate cap

Replace the current positional truncation (`candidates[:5000]` in the CSS evaluator and
`candidates[:3000]` in the PBB evaluator). Positional truncation makes results depend on
generator loop order and can remove entire exponent regions or perturbation strategies.

For normal evolutionary evaluations, canonicalize and deduplicate the raw candidate list,
partition it by generator strategy, guarantee every nonempty strategy a minimum quota,
and select the remaining pre-build slots by sorting
`SHA256(run_seed || lattice || strategy || canonical_candidate_key)`. Retain the existing
5,000 CSS and 3,000 PBB limits as backward-compatible defaults, exposed as:

- `QCODE_MAX_BUILD_CANDIDATES_CSS`
- `QCODE_MAX_BUILD_CANDIDATES_NONCSS`

Record raw, deduplicated, per-strategy, selected, and rejected counts in evaluator metrics.
The Stage-0 census uses its own driver and bypasses this runtime cap because its cheap
population statistics are explicitly exhaustive.

#### Post-build distance selection

Replace pure-`k` distance selection in both evaluator families with deterministic
stratified selection. At minimum use these bands:

- `k∈{2,3}`
- `k∈{4,5}`
- `k≥6`

Every nonempty band receives at least one distance slot. Remaining slots are distributed
proportionally, with deterministic SHA-256 ordering inside each band. This applies to the
CSS hard `k≥8`/pure-`k` path and the PBB descending-`k`/unique-`k` path.

Do not select by FOM before a verified distance witness exists.

### A6. Model configuration

All configs use:

```yaml
llm:
  api_base: "http://localhost:4000/v1"
  temperature: 1.0
  models:
    - name: "aws/claude-opus-5"
      weight: 0.33
      temperature: 1.0
      reasoning_effort: "high"
    - name: "azure/gpt-5.6-sol"
      weight: 0.33
      temperature: 1.0
      reasoning_effort: "high"
    - name: "gcp/gemini-3.6-flash"
      weight: 0.34
      temperature: 1.0
      reasoning_effort: "high"
  max_tokens: 16384
  timeout: 300
  retries: 4
  retry_delay: 10
```

Before a pilot, source `.env` without printing it and query `/v1/models`. Make one minimal
request to each exact alias using the proposed temperature and reasoning settings. Record
only alias, success/failure, response model, latency, and supported-parameter errors; never
log the key or authorization header.

## Phase B: CSS Campaigns

### B1. Search strategies

The principal strategy uses mixed monomials:

```python
A = [(0, 0), (a1, a2)]
B = [(0, 0), (b1, 0), (b2, b3)]
```

with axis-valid positive ranges:

```text
a1,b1,b2 ∈ [1, min(ell-1, 6)]
a2,b3    ∈ [1, min(m-1, 6)]
```

The raw bounded count is `N_x³N_y²`, where `N_x=min(ell-1,6)` and
`N_y=min(m-1,6)`. It is 3,125 at `(6,6)`, 4,500 at `(6,9)`, and 7,776 at
`(9,8)`.

Use three generator strategies:

1. mixed-monomial bounded sweep;
2. a labeled univariate/HGP calibration baseline, not evolved as the principal family;
3. coordinate perturbations of mixed-monomial survivors.

A safety-net candidate outside the EVOLVE block must preserve `k>0` at every Stage-1
lattice if an LLM mutation breaks the evolvable strategies.

### B2. CSS Stage 0

Stage 0 has two distinct outputs.

#### Exhaustive cheap census

For every bounded candidate at every Stage-1 lattice:

1. canonicalize the `2+3` representation;
2. validate exponent ranges and distinct terms;
3. construct the code;
4. compute exact `n,k`;
5. record full and per-`k`-band counts.

No BP-OSD, hash-distance, MILP, or Webster call runs over the full pool.

#### Deterministic sampled distance audit

Select at most 500 `k>0` candidates per lattice without replacement:

1. partition by the A5 `k` bands;
2. guarantee a minimum quota for each nonempty band;
3. distribute remaining quota proportionally;
4. sort within a band by `SHA256(stage0_seed || canonical_candidate_key)`;
5. record the seed, stratum sizes, sample sizes, and inclusion probabilities.

Run the quick-distance cascade on the sample. Report per-stratum rates with 95% Wilson
intervals and compute the overall rate as a population-size-weighted stratified estimate;
do not apply an unweighted binomial interval to unequal quotas. Once a verified witness
supplies `d_upper`, use
`exploratory_fom=k·d_upper²/n` plus `k`-band quotas to prioritize the smaller exact-
certification queue.

Stage 0 is approved only if its report clearly distinguishes exhaustive `n,k` population
statistics from sampled distance estimates.

### B3. Prompt and MAP-Elites

The prompt states:

- exact `2+3` weight-5 enforcement;
- mixed monomials are the principal family;
- the univariate family is calibration only;
- maximize `k d²/n`;
- `d≈10` and `d≈20` are orientation floors, not stopping points.

Use built-in MAP-Elites dimensions for the first pilot. Custom dimensions are allowed only
after the evaluator returns corresponding metric keys and tests prove that missing metrics
cannot crash database insertion.

## Phase C: PBB Campaigns

### C1. Search strategies

Use only the canonical `2+3` backbone. The simultaneous transformation
`(A,B,C,D)→(B,A,D,C)` is a qubit-sector permutation, so a mirrored `3+2` generator is
redundant.

Use three perturbation strategies over the canonical backbone:

1. all subset patterns `C⊆A`, `D⊆B`, including exactly-one-empty pairs;
2. a distribution biased toward sparse perturbations, kept distinct from the uniform
   subset strategy;
3. coordinate perturbations of retained non-CSS candidates while preserving containment.

The seed and evaluator both enforce row weight, commutativity, and the combined non-CSS
gate.

### C2. PBB Stage 0

#### Exhaustive cheap census

For every canonical backbone, enumerate all 32 subset pairs:

- `(C,D)=(∅,∅)` is counted as a CSS calibration case and excluded from the PBB survivor
  pool;
- all other 31 pairs are PBB candidates;
- exactly-one-empty pairs are constructed with the zero-matrix builder path.

Run canonicalization, symbolic and matrix weight checks, construction, commutativity,
cheap LC-CSS pattern rejection, and exact `k` over the full bounded pool. Record exact
population counts after every cheap filter.

#### Deterministic sampled gate/distance audit

If more than 500 survivors remain at a lattice, select at most 500 using the same seeded,
`k`-stratified SHA-256 protocol as CSS Stage 0. Run the combined LC-CSS gate on that sample.
Report each stratum's rejection rate with a 95% Wilson interval. Compute the overall rate
as a population-size-weighted stratified estimate with a stratified variance/interval;
never use an unweighted pooled rate when quotas differ, and never describe an estimate as
an exact full-population rate.

Run bounded distance work only on candidates that pass both checks:

- `n≤216`: exhaustive hashing through weight 6;
- `n>216`: exhaustive hashing through weight 4;
- MILP/Webster only within the certification budget.

Report separately:

1. exhaustive cheap-census counts and `k` distribution;
2. sampled LC-CSS yield estimates;
3. sampled `(n,k,d_lower,d_upper)` distribution;
4. sampling seed, inclusion metadata, timings, and memory usage.

The PBB pilot cannot start until this report shows whether the constrained family has a
credible path beyond the known low-distance examples.

## Phase D: Distance Certification and Result Semantics

### D1. Distance methods

Use existing exhaustive hashing and MILP as the initial certification sources. Integrate
Webster's `codedistance` adapter as follows:

- CSS: call Webster at the persistence boundary, not inside every BP-OSD trial loop;
- PBB: benchmark `QDistEvol`, then replace the Stage-3 BP-OSD fallback only after worker
  timeouts and concurrency have been recalibrated;
- publication verification: update `scripts/verify_publication.py` first;
- every in-loop Webster call passes both an outer `timeout_s` and an inner
  `params={"maxTime": ...}`;
- every returned witness passes the appropriate repository verifier before affecting a
  bound.

Webster status mapping:

| Status | Effect |
|---|---|
| `proven_exact` | May establish exact distance after witness verification |
| `incumbent` | Verified upper bound only |
| `heuristic` | Verified upper bound only |
| `timeout` / `failed` | Log the attempt; do not change stored bounds |

For CSS, track X- and Z-component bounds separately.

### D2. Versioned distance schema

Every result uses `distance_schema_version: 2` and contains:

```json
{
  "distance_schema_version": 2,
  "d_lower": 0,
  "d_upper": null,
  "d_is_exact": false,
  "bound_sources": [],
  "component_bounds": null,
  "certified_fom": null,
  "exploratory_fom": null
}
```

Rules:

- `d_lower=0` means no positive lower bound has been certified.
- `d_upper=null` means no verified logical witness exists.
- A new lower bound merges with `max(old,new)`.
- A new upper bound merges with `min(old,new)`, ignoring `null`.
- If a merge produces `d_upper<d_lower`, quarantine the record as a certification conflict;
  never clamp or silently choose one method.
- Exact means `d_upper` is not null, `d_lower==d_upper`, and the proof/witness sources
  support equality. Equality of default values never implies exactness.
- `certified_fom=k·d_lower²/n` is present only when `d_lower>0`.
- `exploratory_fom=k·d_upper²/n` is present only when `d_upper` is not null.
- A bare ambiguous `fom` must not decide publication or certified Pareto membership.

Only these events can raise `d_lower`:

1. completed exhaustive enumeration finding no logical through weight `w`, which raises
   the lower bound to `w+1`;
2. solver-proven infeasibility or optimality with the solver's exact/proof flag present.

BP-OSD, QDistEvol, decoderDist, randomized sampling, timeouts, and failed searches never
raise `d_lower`.

### D3. CSS component aggregation

For CSS store:

```json
{
  "component_bounds": {
    "X": {"lower": 0, "upper": null, "sources": []},
    "Z": {"lower": 0, "upper": null, "sources": []}
  }
}
```

Aggregate quantum bounds as:

```text
d_lower = min(X.lower, Z.lower)
d_upper = min(all non-null component upper bounds), or null if neither exists
```

Quantum distance is exact whenever the aggregate lower and verified upper coincide. This
includes both components being exact, or one component being exact at `d` while the other
has a certified lower bound at least `d`.

### D4. Versioned three-tier archive

Create `results/pareto_front_v2.json`:

```json
{
  "schema_version": 2,
  "generated_at": "...",
  "exact_front": [],
  "lower_bound_front": [],
  "exploratory_witnesses": []
}
```

Before dominance calculation, merge all evidence and provenance for a code key.

- `exact_front`: `d_is_exact=true`; dominance uses exact `d`.
- `lower_bound_front`: `d_lower>0` and not exact; dominance uses `d_lower`.
- `exploratory_witnesses`: `d_lower=0`, `d_upper!=null`; ranking uses `d_upper` and is
  explicitly non-certified.

Each tier is computed independently. Exploratory entries cannot evict certified entries,
and lower-bound entries cannot evict exact entries.

Add a migration reader for legacy lists. A legacy record with only bare `d` is migrated as
exploratory (`d_lower=0`, `d_upper=d`) unless an existing exact/proof flag is recognized.
Never infer exactness from a historical numeric distance alone. Preserve the legacy file;
write only the new v2 artifact. Update all consumers and tests for the new object return
type.

## Phase E: Provenance and Concurrent Persistence

### E1. Runtime attribution

Extend `evolve/model_attribution.py` with non-consuming `peek_model()` and
`peek_program_id()` accessors. Wrap `Evaluator.evaluate_program(program_code, program_id)`
to set the current program ID for the duration of evaluation. Stamp each per-code result
once in `_run_evaluation()` with:

- `run_id`
- `program_id`
- `generating_model`
- `model_alias`
- `source_hash` of the complete evolved program source
- canonical `code_key`

The model alias comes from the generation ensemble selection, not from an evaluator model.

### E2. Authoritative append-only discovery events

Worker processes must not perform read-modify-write updates on shared JSON arrays. Each
run writes authoritative events to:

```text
results/evolution/<run_id>/discovery_events.jsonl
```

Each event includes:

```text
event_id = SHA256(run_id || program_id || code_key)
run_id, program_id, model_alias, generating_model
source_hash, code_key, ell, m, A_terms, B_terms, C_terms, D_terms
distance schema fields, stage, timestamp
config_hash, seed_hash, git_commit
parent_id=null, iteration=null initially
```

Use an inter-process `fcntl.flock` around a single append operation. The event ID makes
evaluation retries idempotent while retaining genuine rediscoveries by distinct programs.

### E3. Run manifest

At launch, write `run_manifest.json` atomically with:

- run ID and campaign name;
- git commit and dirty-state marker;
- config and seed paths plus content hashes;
- evaluator path and content hash;
- model aliases and weights;
- effective gate environment variables;
- Stage-0 seed or evolution RNG seed;
- approved budget deviations.

Do not record API keys or authorization headers.

### E4. Post-run enrichment and reduction

OpenEvolve evaluates a child before constructing its `Program`, so `parent_id` and
`iteration` are unavailable at discovery-event write time. After the run:

1. join `discovery_events.jsonl` to `model_attribution.jsonl` by `program_id`;
2. fill `parent_id` and `iteration` where available;
3. leave them null with an explicit `join_status` if the program was not accepted or the
   attribution record is missing;
4. retain the model/program/source provenance regardless of join success.

Run a single reducer after enrichment. It merges discovery events by code key while
appending a distinct `discovered_by` entry keyed by `(run_id,program_id)`. The reducer,
not evaluator workers, updates global discovered-code and Pareto artifacts.

Protect reduction with an inter-process lock. Write a temporary file in the destination
directory, flush and `fsync`, then publish with `os.replace`. This prevents record loss
within one parallel campaign and across accidentally overlapping campaigns.

The authoritative event logs are never deleted and can regenerate every derived artifact.

## Compute Budget and Decision Gates

### Stage-0 limits

- Full cheap enumeration contains no distance or full LC-CSS calls, but it is still
  compute-bounded. Before each lattice, benchmark a deterministic 100-candidate slice,
  record construction/rank time and peak memory, and extrapolate the full cheap-census
  cost.
- The default cheap-census ceiling is 4 CPU-hours per small-profile lattice and 8
  CPU-hours per large-profile lattice. If the projection or measured run exceeds the
  ceiling, stop and either optimize, narrow the declared exponent space, or approve a
  manifest-recorded budget increase. Never silently sample the cheap population and still
  label its statistics exhaustive.
- Cache work at the backbone level: canonical keys, the BB object, `A/B` circulants,
  per-term circulants, the fixed Block-2 matrix, and reusable rank inputs. Derive subset
  `C/D` matrices from cached term matrices instead of rebuilding the same qLDPC objects 31
  times. Cache keys include the lattice and normalized terms.
- The census writes checkpoints containing the enumeration cursor, cache version, counts,
  timings, and a hash of the census configuration. Resume only when the hash matches.
  Checkpoint/resume output must be identical to an uninterrupted run.
- At most 500 candidates per lattice enter the combined LC-CSS or quick-distance audit
  without an explicit, manifest-recorded approval.
- Sampling is deterministic and stratified; an unspecified random slice is not allowed.
- `n≤216` exhaustive hashing stops at weight 6.
- `n>216` exhaustive hashing stops at weight 4.
- Only one `n≥216` high-memory hash or large-MILP job may run per worker.
- Smaller certification concurrency is at most `floor(available_RAM/12GB)` and must be at
  least one only when the host actually has sufficient memory.

### CPU limits

Track cheap enumeration, LC-CSS, and distance-certification costs separately:

| Scope | Cheap census | LC-CSS budget | Certification budget |
|---|---:|---:|---:|
| Small Stage 0 | 4 CPU-hours per lattice | 4 CPU-hours per lattice | 4 CPU-hours per lattice |
| Large Stage 0 | 8 CPU-hours per lattice | 4 CPU-hours per lattice | 4 CPU-hours per lattice |
| 50-iteration pilot | n/a | 4 CPU-hours | 4 CPU-hours per lattice |
| 300-iteration full run | n/a | 24 CPU-hours | 24 CPU-hours per lattice |

Use a 10-minute per-invocation MILP/Webster ceiling initially. Recalibrate it from Stage-0
measurements before large-n pilots. Keep no more than 100 unresolved rigorous intervals per
lattice without an explicit prioritization decision.

### Token limits

At `max_tokens=16384`, the configuration ceilings are approximately:

- 50 iterations: 0.82M output tokens, excluding separately accounted reasoning tokens;
- 300 iterations: 4.9M output tokens, excluding separately accounted reasoning tokens.

Check proxy rate and budget limits before every full campaign.

### Go/no-go gates

#### Before any pilot

- primary-source fit constants checked;
- all model aliases and reasoning settings smoke-tested;
- Stage-0 report reviewed;
- provenance event log and reducer tests pass;
- v2 distance/archive schema tests pass;
- no regression in existing campaigns.

#### Stage-0 exploration gate

This is a feasibility screen, not a distance proof: `d_upper≥MIN_RELEVANT_D` means the
smallest verified witness found in the sampled audit has weight at least
`MIN_RELEVANT_D`, not that the true distance is bounded below by it. A single high-weight
witness on an otherwise-unsearched candidate is not sufficient. The gate requires, at a
proposed pilot lattice, that the smallest verified witness among all sampled candidates
that were actually searched down to `MIN_RELEVANT_D-1` (i.e., the search had the
opportunity to find a smaller logical and did not) has weight `≥MIN_RELEVANT_D`, or that a
sampled candidate already carries a certified `d_lower≥MIN_RELEVANT_D`. A lattice that
cannot clear this heuristic gate is removed from the pilot profile.

#### Pilot-to-full-run gate

At least one pilot discovery must have:

- PBB: certified `d_lower≥5`;
- CSS: certified `d_lower≥7`;
- `certified_fom` at or above the campaign refinement threshold;
- complete `(run_id,program_id,model_alias,source_hash)` provenance.

#### Publication gate

Any external distance or Pareto claim requires:

- `d_is_exact=true` under schema v2;
- a verified logical witness;
- inclusion in the exact front;
- post-hoc verification through the publication script;
- complete provenance and run manifest.

## Tests

### Backward compatibility

- existing CSS and PBB configs retain current behavior when no weight-5 variables or
  evaluator override are supplied;
- fresh and resumed runs honor `--evaluator`;
- all existing tests pass.

### Weight and construction

- CSS rejects every non-5 backbone mutation;
- PBB rejects support outside `C⊆A`, `D⊆B` symbolically and at matrix level;
- PBB constructs and validates both exactly-one-empty cases;
- `(∅,∅)` still follows the CSS fast path;
- every retained PBB seed has row weight exactly 5 and commutes;
- both LC-CSS checks receive the correct argument types.

### Selection and Stage 0

- modulus-6 lattices never generate exponent 6;
- expected bounded counts are reproduced;
- raw candidate lists larger than 5,000 CSS or 3,000 PBB entries are sampled by canonical
  hash across every generator strategy, never by prefix;
- changing generator loop order does not change the selected raw candidate set;
- identical Stage-0 seed and input produce identical samples;
- every nonempty `k` band receives its quota;
- sample metadata and Wilson intervals are emitted;
- no pre-distance code path ranks by FOM.
- cached and uncached cheap-census paths produce identical filters and `k` values;
- checkpoint/resume produces byte-equivalent census counts and candidate keys;
- exceeding a cheap-census budget stops with a resumable checkpoint and never converts an
  exhaustive report into an unlabeled sample.

### Distance

- heuristic failure never raises `d_lower`;
- exhaustive no-logical-through-`w` raises it to `w+1`;
- a verified witness tightens only `d_upper`;
- merge conflicts are quarantined;
- CSS component aggregation handles null upper bounds;
- exact results bypass heuristic gates;
- legacy bare-`d` records migrate as exploratory;
- exploratory entries cannot evict certified fronts.

### Provenance and concurrency

- each discovery event has model, program, source, config, seed, and git identity;
- two programs from one model rediscovering a code produce two `discovered_by` entries;
- parallel writers produce valid JSONL with no lost events;
- concurrent reducers are serialized;
- interruption before `os.replace` leaves the previous derived artifact intact;
- parent and iteration enrichment is correct and missing joins remain explicit.

## Rollout

1. Implement Phase A, schema-v2 distance persistence, and authoritative provenance events.
2. Run unit tests and the full regression suite.
3. Verify the primary literature values and LiteLLM model aliases.
4. Run CSS and PBB Stage-0 audits for the small profile.
5. Pilot `W5-CSS-Small` for 50 iterations.
6. Pilot `W5-PBB-Small` for 50 iterations if its Stage-0 gate passes.
7. Integrate and benchmark the full Webster paths.
8. Run large-profile Stage 0 only for families whose small pilot passes.
9. Pilot large campaigns before approving any 300-iteration run.
10. Reduce events, enrich provenance, and reverify exact-front candidates after each run.

## Launch Interface

These commands become valid after their referenced files and environment hooks exist.

### Stage 0

```bash
uv run python scripts/run_weight5_stage0.py \
  --family css \
  --profile small \
  --seed 42 \
  --max-audit-candidates 500 \
  --cheap-cpu-hours-per-lattice 4 \
  --checkpoint-dir results/stage0/weight5_css_small
```

Use `--family pbb` for PBB. Large-profile runs set
`--cheap-cpu-hours-per-lattice 8`. The command writes the census configuration hash,
benchmark projection, checkpoints, exact cheap-population counts, sample inclusion
metadata, and budget decisions into its output directory.

### CSS small pilot

```bash
QCODE_MIN_K_THRESHOLD=2 \
QCODE_MIN_RELEVANT_D=4 \
QCODE_LATTICE_PROFILE=small \
QCODE_FOM_THRESHOLD_REFINE=3.0 \
QCODE_FOM_THRESHOLD_EXACT=5.0 \
QCODE_SAVE_FOM_THRESHOLD_CSS=2.0 \
QCODE_SAVE_TRUST_RATIO=2.0 \
uv run python evolve/run_evolution.py \
  --evaluator evolve/openevolve_evaluator_weight5_css.py \
  --config evolve/config_weight5_css_small.yaml \
  --seed evolve/seed_solution_weight5_css.py \
  --iterations 50 \
  --run-name weight5_css_small_pilot_v1
```

### CSS large run

```bash
QCODE_MIN_K_THRESHOLD=2 \
QCODE_MIN_RELEVANT_D=4 \
QCODE_LATTICE_PROFILE=large \
QCODE_FOM_THRESHOLD_REFINE=3.0 \
QCODE_FOM_THRESHOLD_EXACT=5.0 \
QCODE_SAVE_FOM_THRESHOLD_CSS=2.0 \
QCODE_SAVE_TRUST_RATIO=2.0 \
uv run python evolve/run_evolution.py \
  --evaluator evolve/openevolve_evaluator_weight5_css.py \
  --config evolve/config_weight5_css_large.yaml \
  --seed evolve/seed_solution_weight5_css.py \
  --iterations 300 \
  --run-name weight5_css_large_v1 \
  --wandb
```

### PBB small pilot

```bash
QCODE_MIN_K_THRESHOLD_NONCSS=2 \
QCODE_MIN_RELEVANT_D_NONCSS=5 \
QCODE_LATTICE_PROFILE=small \
QCODE_FOM_THRESHOLD_REFINE=3.0 \
QCODE_FOM_THRESHOLD_EXACT=5.0 \
QCODE_SAVE_FOM_THRESHOLD_NONCSS=2.0 \
QCODE_SAVE_TRUST_RATIO_NONCSS=2.5 \
uv run python evolve/run_evolution.py \
  --noncss \
  --evaluator evolve/openevolve_evaluator_weight5_pbb.py \
  --config evolve/config_weight5_pbb_small.yaml \
  --seed evolve/seed_solution_weight5_pbb.py \
  --iterations 50 \
  --run-name weight5_pbb_small_pilot_v1
```

### PBB large run

```bash
QCODE_MIN_K_THRESHOLD_NONCSS=2 \
QCODE_MIN_RELEVANT_D_NONCSS=5 \
QCODE_LATTICE_PROFILE=large \
QCODE_FOM_THRESHOLD_REFINE=3.0 \
QCODE_FOM_THRESHOLD_EXACT=5.0 \
QCODE_SAVE_FOM_THRESHOLD_NONCSS=2.0 \
QCODE_SAVE_TRUST_RATIO_NONCSS=2.5 \
uv run python evolve/run_evolution.py \
  --noncss \
  --evaluator evolve/openevolve_evaluator_weight5_pbb.py \
  --config evolve/config_weight5_pbb_large.yaml \
  --seed evolve/seed_solution_weight5_pbb.py \
  --iterations 300 \
  --run-name weight5_pbb_large_v1 \
  --wandb
```

## Implementation Checklist

### Preflight

- [ ] Confirm branch base remains `public/main` and preserve unrelated worktree changes.
- [ ] Verify every proposed `(ℓ,m)` pair constructs with the installed qLDPC version.
- [ ] Check Lin-Pryadko §5.2 directly and record the verified fit values.
- [ ] Smoke-test all three LiteLLM aliases using `.env` without exposing credentials.

### Core evaluator work

- [ ] Add and test `--evaluator` for fresh and resumed runs.
- [ ] Add all gate environment variables with backward-compatible defaults.
- [ ] Implement CSS and PBB weight-5 evaluators and shared helpers.
- [ ] Replace positional raw-candidate caps with canonical, strategy-stratified hash
      sampling and expose the CSS/PBB cap variables.
- [ ] Implement canonical `2+3` normalization and post-build `k`-stratified distance
      selection.
- [ ] Fix the current non-CSS prompt's `[C|D]` convention and classification language.
- [ ] Fix `build_pbb_code()` for independently empty `C` or `D`.
- [ ] Implement combined, cached LC-CSS screening.

### Distance and persistence

- [ ] Implement distance schema v2 and monotonic bound merging.
- [ ] Implement CSS component-bound aggregation.
- [ ] Implement exact-result gate bypass.
- [ ] Implement the versioned three-tier archive and conservative legacy migration.
- [ ] Add bounded Webster calls and witness verification.
- [ ] Benchmark `QDistEvol` before replacing the PBB BP-OSD fallback.

### Provenance

- [ ] Add non-consuming model/program accessors.
- [ ] Stamp model, program, source hash, and run identity on every result.
- [ ] Write locked append-only discovery events from workers.
- [ ] Write an atomic run manifest.
- [ ] Implement parent/iteration enrichment.
- [ ] Implement a locked, atomic single-process reducer and `discovered_by` merge.

### Stage 0 and campaigns

- [ ] Implement `scripts/run_weight5_stage0.py`, the shared deterministic audit sampler,
      benchmark projection, cache, checkpoint/resume, budget stop, and coverage report.
- [ ] Verify cached enumeration exactly matches an uncached reference on small fixtures.
- [ ] Run and review CSS-small Stage 0.
- [ ] Run and review PBB-small Stage 0, including all 31 nontrivial subset pairs.
- [ ] Create CSS and PBB seeds, prompts, and four configs.
- [ ] Run the CSS-small pilot and apply its go/no-go gate.
- [ ] Run the PBB-small pilot only if its Stage-0 gate passes.
- [ ] Gate all large work on the corresponding small pilot.

### Final validation

- [ ] Run `uv run python -m pytest tests/ -v`.
- [ ] Run focused concurrency, migration, provenance, and distance-bound tests.
- [ ] Run `scripts/verify_publication.py --help` and validate the final Webster CLI before
      documenting a publication command.
- [ ] Invoke the configured Codex pre-review hook on the staged implementation diff and
      address blocking findings before commit.

## Success Criteria

The implementation succeeds if it provides:

1. four launchable, backward-compatible campaign configurations;
2. mechanical enforcement of genuine stabilizer row weight 5;
3. reproducible Stage-0 evidence with explicit population/sample boundaries;
4. lossless model/program provenance under parallel evaluation;
5. versioned, proof-aware distance intervals and Pareto archives;
6. at least one pilot that passes its certified go/no-go gate, or a well-supported bounded
   negative result showing that the tested ansatz should be replaced.

Scientific discoveries count as publication-ready only when they meet the publication
gate. A negative result is scoped to the enumerated/sampled BB or PBB families and stated
compute budget; it is not a no-go theorem for all weight-5 codes.

## Sources

- `investigations/weight-5-codes-survey.md`
- Lin and Pryadko, *Quantum two-block group algebra codes*, arXiv:2306.16400
- `evaluation/evaluator.py`
- `evaluation/pbb_code.py`
- `evaluation/clifford_equivalence.py`
- `evaluation/distance_webster.py`
- `evaluation/results.py`
- `evolve/openevolve_evaluator.py`
- `evolve/openevolve_evaluator_noncss.py`
- `evolve/_noncss_distance_worker.py`
- `evolve/model_attribution.py`
- `evolve/run_evolution.py`
- `results/campaign7_report.md`
- `investigations/webster_findings.md`
- `investigations/webster_comparison_plan.md`
