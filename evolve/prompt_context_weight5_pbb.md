# Weight-5 Non-CSS PBB Code Discovery — Domain Knowledge for LLM

## What You're Evolving

You are evolving a Python function `generate_candidates(ell, m)` that returns candidate **weight-5, non-CSS perturbed bivariate bicycle (PBB)** quantum error-correcting codes. Each candidate is a 4-tuple `(A_terms, B_terms, C_terms, D_terms)` of polynomial exponent pairs, restricted to the canonical weight-5 **"2+3" CSS backbone**.

## Hard constraints

1. **Backbone weight-5, canonical 2+3 split only.** `A` has exactly 2 distinct terms, `B` has exactly 3 distinct terms (never the mirrored 3+2 assignment — that's just a qubit-sector permutation of the same code, out of scope for this campaign):

   ```
   A = [(0, 0), (a1, a2)]
   B = [(0, 0), (b1, 0), (b2, b3)]
   a1, b1, b2 in [1, min(ell-1, 6)];  a2, b3 in [1, min(m-1, 6)]
   ```

2. **Containment.** `supp(C) ⊆ supp(A)` and `supp(D) ⊆ supp(B)` as exponent-pair sets. This is not optional — it is exactly the condition under which the PBB row weight stays at 5 (see below), and the evaluator independently re-checks it both symbolically AND against the constructed circulant matrices. Violating containment gets a candidate silently dropped, no partial credit.

3. **Row weight exactly 5.** `|supp(A) ∪ supp(C)| + |supp(B) ∪ supp(D)| == 5`. Given the weight-5 backbone above, this is automatic once containment holds — build C, D as *subsets* of A's and B's own terms and you get it for free.

4. **Any nontrivial (C, D) pair is valid, INCLUDING exactly-one-empty pairs.** `C != []` with `D == []` (or vice versa) is still genuinely non-CSS once the nonempty one is nonempty — do not discard these. Only `C == [] and D == []` is excluded (that reduces to plain CSS, a different campaign).

## Mathematical Background

A PBB code extends the bivariate bicycle (BB) code by adding perturbation polynomials to the stabilizer matrix. **Convention (paper convention, matching `evaluation/pbb_code.py` — note this is `[C|D]` and `A@C^T + B@D^T`, NOT `[D|C]`/`A@D^T+B@C^T`):**

```
Block 1 (mixed):   x-part = [A | B],   z-part = [C | D]   (C left, D right)
Block 2 (pure Z):  x-part = [0 | 0],   z-part = [B^T | A^T]
```

When C = D = [] (empty), this is exactly a CSS BB code — that is the excluded calibration case, not a target. Non-empty C and/or D make the code **genuinely non-CSS** (subject to the non-CSS gate below) — it cannot always be transformed to CSS by single-qubit Clifford operations.

Each polynomial is defined over `F_2[x,y]/(x^ℓ-1, y^m-1)`:
- A: exactly 2 terms (backbone). B: exactly 3 terms (backbone).
- C: subset of A's terms. D: subset of B's terms.
- Exponents: `0 ≤ x-exp < ℓ`, `0 ≤ y-exp < m`. Each term is `(x_exponent, y_exponent)`.

Code parameters:
- n = 2·ℓ·m physical qubits
- k = logical qubits (depends on A, B, C, D together — exact via GF(2) rank)
- d = code distance (adaptive: exact d≤6 at n≤216, exact d≤4 at n>216, MILP for the rest)

**Commutativity constraint**: `(A @ C^T + B @ D^T) % 2` must be symmetric over GF(2). The evaluator checks this at build time (catching the resulting `ValueError` without crashing) — invalid candidates are silently dropped, but generating many invalid candidates wastes your budget. Commutativity is highly `(ell, m)`-specific: a `(C, D)` pair that commutes at one lattice size can violate commutativity at another. One trick that ALWAYS commutes, at any `(ell, m)`: setting `C` to ALL of `A`'s terms and `D` to ALL of `B`'s terms (since `A @ A^T` and `B @ B^T` are each automatically symmetric, their sum is symmetric) — useful as a fallback, not a target to converge on repeatedly.

## Scoring

FOM = k·d²/n — higher is better, among candidates that pass the **non-CSS gate** (see below).

**Codes with d ≤ 4 score ZERO** in the trusted/verified sense. The evaluator has a 3-tier adaptive distance pipeline:

1. **Hash-based exact check** (O(n³) symplectic weight): n ≤ 216 checks d ≤ 6 exactly; n > 216 checks d ≤ 4 exactly (the O(n³) triple loop is too slow for weight-5/6 at large n).
2. **MILP symplectic** for codes beyond hash range: adaptive timeouts (15-60s per logical). Partial results are valid upper bounds even if not every logical solves in time.
3. **BP-OSD fallback**: only used if MILP yields nothing useful. Known to overestimate distance for non-CSS codes by up to several times — treat BP-OSD-only estimates as provisional.

## Non-CSS gate

A candidate that builds successfully and has k>0 is not automatically credited — it must also pass a **non-CSS classification gate** confirming it is not just LC-equivalent (or exactly equivalent) to a CSS code:

1. `verify_not_lc_css` — cheap, purely algebraic check over the term lists (no code construction needed beyond the build you already did).
2. `is_equivalently_css` — more expensive check on the constructed code object, only run if (1) didn't already rule the candidate out.

Both are run internally by the evaluator AFTER weight/containment/commutativity/k filters — you never need to call these yourself, but be aware that a candidate with high k and passable d that is secretly CSS-equivalent gets zero credit. Diversifying HOW C and D perturb the backbone (not just how big they are) is the way to avoid this.

## Strategies to Explore

### Backbone diversity
Only Base2/Base7e-style structural insight from the plain non-CSS campaign carries over loosely: most (A,B) backbones give d≤4 structurally, and you won't know until you try. Sweep `(a1, a2, b1, b2, b3)` broadly across `[1, min(ell-1,6)]` / `[1, min(m-1,6)]`, not just corners.

### (C, D) pattern diversity within containment
For a fixed backbone, C ranges over subsets of A's 2 terms (4 options including empty) and D over subsets of B's 3 terms (8 options including empty) — 32 combinations minus the excluded (∅,∅) = 31 genuinely distinct (C,D) patterns per backbone. Try all of them for a promising backbone before moving on; don't fixate on the full-C/full-D safety-net trick.

### Coordinate perturbations
Once you have a promising backbone + (C,D) pattern, try nudging one backbone exponent by ±1 (mod ell or mod m as appropriate) and remap C/D to preserve containment on the new backbone by tracking which *term* moved (not by hand-searching a fresh subset) — reuse the pattern that pointed you there.

## Constraints on Your Function

- Must return `list[tuple[list[tuple[int,int]], list[tuple[int,int]], list[tuple[int,int]], list[tuple[int,int]]]]`
- A_terms: exactly 2 distinct tuples, always including `(0, 0)`. B_terms: exactly 3 distinct tuples, always including `(0, 0)`.
- C_terms: subset of A_terms. D_terms: subset of B_terms. Not both empty.
- x-exponents in `[1, min(ell-1,6)]`, y-exponents in `[1, min(m-1,6)]` for the backbone's non-constant terms.
- Do not manually cap candidate counts — the evaluator caps per-lattice via hash-stratified selection (`evaluation.candidate_selection`). Focus on structural diversity within the weight-5 constraint instead.
- **NEVER drift off the 2+3 backbone split or off containment** — silently zero-credited, not corrected.
- **NEVER return `(C=[], D=[])`** — that's plain CSS, a different campaign, and is explicitly excluded.

## Evaluator Feedback

### Metrics
| Metric | Meaning |
|--------|---------|
| `combined_score` | Sum of best trust-adjusted FOM per lattice |
| `best_fom` | Highest FOM found |
| `num_valid` | Candidates with k > 0 that passed the non-CSS gate |
| `num_high_k` | Codes with k ≥ 8 |

### Evaluation Stages

1. **Stage 1** (quick, ~5-30s): k-only screening at the Stage-1 lattices for the active `QCODE_LATTICE_PROFILE`. Must produce valid, gate-passing codes (k > 0) at ALL Stage-1 lattices to clear the cascade threshold.
2. **Stage 2** (full, ~120-1200s): adaptive distance across Stage-1 + Stage-2 lattices for the active profile.

### Tips
1. **If most candidates have k=0**: check the constant term `(0,0)` is retained in both A and B, and that no perturbed backbone term collapses onto another term already present.
2. **If num_valid is very low**: many (backbone, C, D) combinations either fail commutativity or fail the non-CSS gate (secretly CSS-equivalent). Diversify backbone AND pattern together, not just pattern.
3. **If k is high but d≤4**: the backbone itself is likely bad for distance regardless of (C,D) — try a different backbone before optimizing perturbations further.
4. **Target**: any trusted FOM > 0 with a genuinely non-CSS, gate-passing code beats the excluded CSS-calibration baseline at the same lattice — that's the entire point of this campaign.
