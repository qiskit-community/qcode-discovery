# Weight-5 CSS Discovery — Domain Knowledge for LLM

## What You're Evolving

You are evolving a Python function `generate_candidates(ell, m)` that returns candidate bivariate bicycle (BB) quantum error-correcting code polynomial pairs. Each candidate is a pair of polynomials (A, B) represented as lists of `(x_exponent, y_exponent)` tuples.

**Hard constraint — read this first**: every candidate you return MUST have exactly 5 total terms, split canonically as 2+3 between the two polynomials: `len(A) + len(B) == 5` and `{len(A), len(B)} == {2, 3}` (A has 2 terms and B has 3, or vice versa). This is the code's *stabilizer-generator Pauli row weight* — each X-check and Z-check acts on exactly 5 qubits. The evaluator enforces this independently of your code (`evaluation.weight_enforcement.css_weight_ok`, checked before construction) — a candidate that drifts to weight 4 or 6 is silently dropped, not corrected. It costs you evaluation budget for zero credit. Stay exactly at weight 5.

## Mathematical Background

A BB code is defined over the ring F_2[x,y]/(x^ell - 1, y^m - 1) by two polynomials A(x,y) and B(x,y). Each monomial x^a * y^b corresponds to a shift operator P_m^b (tensor) P_ell^a acting on the ell x m torus.

Code parameters: [[n, k, d]] where n = 2*ell*m, k = logical qubits (exact, via GF(2) rank), d = code distance (estimated via BP-OSD, verified via MILP for high-rate codes).

**Figure of merit**: FOM = k * d^2 / n. Higher is better, with no ceiling — keep pushing past any milestone.

## The Scientific Question

**What is the best achievable FOM among weight-5 CSS BB codes?**

Weight-3 trinomial codes (Campaigns 1-3) and weight-4-6 mixed-monomial codes with a *variable* term budget (Campaign 4) have both been explored. This campaign fixes the row weight at exactly 5 — the natural next rung above weight-3 hardware-realistic codes — and asks how far FOM can be pushed under that fixed physical-weight budget, restricted to the 2+3 split.

Two structural families matter here:

1. **Mixed-monomial family (principal, evolvable target)**: A has a genuine mixed term x^a·y^b (both exponents > 0), coupling the two cyclic dimensions diagonally. This is the family you are here to refine — it is essentially unexplored at weight 5 and is expected to be the source of any real improvement.
2. **Univariate/HGP family (calibration baseline only)**: A and B are both purely axis-aligned (hypergraph-product-like). This family is known to give very high k but typically d <= 4 (structural ceiling from prior campaigns) — it is included as a small, fixed comparison set so the evaluator's feedback can show you, lattice by lattice, how the mixed-monomial family's k and d trade off against this baseline. **Do not spend your evolution budget growing the univariate/HGP set** — treat it as a fixed yardstick, not a target for expansion.

## Known Best Codes (weight-3 baseline, for orientation — not weight-5 codes)

These are *not* weight-5 and cannot be reproduced directly under the 2+3 constraint, but they show what FOM values are achievable once distance gets large — use them to calibrate ambition, not as candidates to imitate structurally:

| Code | ell | m | FOM | Family |
|------|-----|---|-----|--------|
| [[72,12,6]] | 6 | 6 | 6.0 | x/y-swap (weight-3) |
| [[144,12,12]] | 12 | 6 | 12.0 | x/y-swap (weight-3) |
| [[288,12,18]] | 12 | 12 | 13.5 | x/y-swap (weight-3) |
| [[360,12,<=24]] | 30 | 6 | 19.2 | x/y-swap (weight-3) |
| [[360,40,<=20]] | 15 | 12 | 44.4 | constant-monomial (weight-3) |

**Orientation floors, not stopping points**: a weight-5 code reaching **d ~ 10** is a first checkpoint showing the mixed-monomial ansatz can beat the weight-3 baseline's distance at comparable n; reaching **d ~ 20** is a second checkpoint on par with the best weight-3 codes above. Neither is a ceiling — if you find candidates trending past d=20, keep extrapolating the same structural relationship to larger lattices rather than stopping.

## The Ansatz (Phase B1 seed strategies)

The seed (`seed_solution_weight5_css.py`) starts from this bounded template — refine it, don't just resweep it:

```
A = 1 + x^a1*y^a2                  (2 terms, mixed monomial)
B = 1 + x^b1 + x^b2*y^b3            (3 terms: constant + pure-x + mixed)
```
with `a1, b1, b2 in [1, min(ell-1, 6)]` and `a2, b3 in [1, min(m-1, 6)]`.

Also present: a small labeled univariate/HGP calibration set (`A = 1 + x^a`, `B = 1 + y^b1 + y^b2`, and the x/y-symmetric version), and coordinate perturbations (+/-1 mod ell or mod m) seeded from the bounded box's corners.

Directions worth exploring beyond the seed's fixed template, while staying at weight 5 / 2+3:
- Different placements of the mixed term in A (e.g. `A = x^a1 + x^a1*y^a2`, no constant) vs anchoring on the identity.
- Different term compositions in B: two mixed terms + one pure term, or one pure-x + one pure-y + one mixed.
- Exponent relationships between A and B's mixed term(s) — complementary diagonals, coprime shift ratios, shared or reflected shift vectors.
- Whether asymmetric growth (b1, b2 far apart) or coupled growth (b1, b2 related by a fixed offset) produces better distance as lattices scale up.

## Known Dead Ends (avoid these)

- **Self-dual-like structure**: if A and B end up with identical monomial sets, d = 2 exactly (proven for weight-3; the pathology recurs whenever A and B generate the same ideal). At weight 5 with a 2+3 split, A and B can never literally be equal (different term counts), but near-duplicate structure (e.g. B built by adding one term to a copy of A) tends to inherit the same low-distance behavior — avoid it.
- **Growing the univariate/HGP baseline**: it is a calibration reference, not the evolution target. Spending mutation budget deepening this family wastes iterations that should go to the mixed-monomial family.
- **Drifting off weight 5**: any mutation that changes term counts away from the 2+3 split gets silently rejected by the evaluator (`evaluation.weight_enforcement.css_weight_ok`) before construction — zero credit, no partial score, no error message pinpointing which candidate. Double-check term counts after any structural change to the generator.
- **Minor perturbations with no theory behind them**: isolated +/-1 exponent nudges without a structural rationale rarely move the needle once the bounded box has been swept (Strategy 3 already covers the boundary neighborhood exhaustively).

## Constraints on Your Function

- Return type: `list[tuple[list[tuple[int,int]], list[tuple[int,int]]]]`
- Every candidate: `len(A) + len(B) == 5` and `{len(A), len(B)} == {2, 3}` — checked independently by the evaluator; non-conforming candidates are dropped silently, not corrected.
- Exponent ranges: `0 <= x_exp < ell`, `0 <= y_exp < m`
- All terms within each polynomial must be distinct (post mod-reduction)
- Do not remove or weaken the safety-net candidates defined outside the EVOLVE-BLOCK — they guarantee Stage 1 always has something to score even if your evolved strategies regress.
- Raw candidate counts do not need to be pre-capped — the evaluator caps per-lattice via deterministic hash-stratified selection (`evaluation.candidate_selection`). Focus on candidate *quality* (structural diversity, weight-5 conformance), not on manually limiting count.
- Use standard Python + `itertools` only (no numpy/scipy required, though allowed).

## Evaluator Feedback

### Metrics

| Metric | Meaning | What to optimize |
|--------|---------|-----------------|
| `combined_score` | Primary fitness: sum of best credible FOM per lattice | Maximize |
| `best_fom` | Highest FOM across all evaluated lattices | No ceiling — keep pushing past d~10, d~20 checkpoints |
| `num_valid` | Candidates with k > 0 (after weight-5 filtering) | Higher = mixed-monomial structure is finding kernel |
| `num_high_k` | Candidates with k >= 8 | Breadth of promising candidates reaching the distance stage |
| `lattices_with_high_k` | Number of lattices with at least one k >= 8 code | Consistency of the ansatz across lattice sizes |
| `total_candidates` | Total generated (pre-filter) | Informational — no need to self-limit |

### Artifacts

- **`best_code`**: Top credible code (trusted BP-OSD distance) found so far, with A and B terms and FOM.
- **`selection_metrics`** (when the pre-build cap engaged): per-lattice raw/deduped/selected/rejected counts by structural stratum — tells you whether your strategies are being evenly sampled or crowded out by one dominant stratum.
- **`errors`**: per-lattice bookkeeping, including how many raw candidates were rejected by weight-5 enforcement at each lattice — a nonzero, growing rejection count across generations is a signal your mutations are drifting off the 2+3 constraint.
- **`top_codes`**: top 5 credible codes by FOM.
- **`structural_analysis`**: for codes with d >= 4, a term-by-term breakdown (pure-x, pure-y, constant, mixed counts and shift vectors) for A and B — use this to identify which mixed-term placements are working.
- **`summary`**: aggregate stats across all evaluated lattices, including weight-5 rejection totals and per-lattice credible-FOM breakdown.

### Evaluation Stages

1. **Stage 1** (~2-5s): k-only screening. Lattices are chosen by `QCODE_LATTICE_PROFILE` (env var, default `"small"`): small profile uses (6,6), (6,9), (9,8); large profile uses (12,9), (12,12), (15,12). Must produce k > 0 at ALL Stage-1 lattices to advance — the safety net (outside your control) guarantees this baseline; your job is to also get novel weight-5 mixed-monomial candidates to k > 0 at all three.
2. **Stage 2** (~30-60s): Full BP-OSD distance estimation. Small profile: (6,10), (9,9), (10,9). Large profile: (16,12), (15,14), (18,12). Distance is filtered by a d/sqrt(n) trust ratio — only codes with `d <= 1.3*sqrt(n)` are fully trusted; codes above `2.0*sqrt(n)` fall back to a k/n-only credible FOM; between the two, credit interpolates linearly.

### Tips

1. **If most candidates have k = 0**: the constant term `(0,0)` in A or B often anchors the kernel — check both are keeping it. Very large or exactly-shared exponents between A's mixed term and B's pure-x term can also degenerate k to 0; try decoupling them.
2. **If k > 0 but d stays low (d <= 4)**: the algebraic structure is too regular — this is exactly the fate of the univariate/HGP calibration baseline. Try breaking symmetry in the mixed term's exponent ratio, or move the mixed term from A into B (2+3 permits either assignment).
3. **If a code reaches d >= 4**: study the `structural_analysis` artifact — which term(s) are mixed, and what's their shift vector relative to the other polynomial's terms? Encode the winning relationship as a parametrized template, not a single hardcoded pair, so it can be swept across lattice sizes.
4. **Push past the checkpoints**: d ~ 10 and d ~ 20 are orientation markers from the weight-3 baseline, not goals. If a template is trending upward as lattices grow, keep extrapolating it to larger (ell, m) rather than stopping once a checkpoint is cleared.
5. **Don't let the calibration baseline crowd out the principal family**: if `selection_metrics` shows the univariate/HGP stratum dominating the per-lattice budget, shrink it — it exists for comparison, not for growth.
