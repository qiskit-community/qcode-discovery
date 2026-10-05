# Tests and Analysis Scripts

This directory contains both the automated pytest suite and standalone analysis scripts that support the paper's claims.

## Automated Tests

```bash
uv run python -m pytest tests/ -v              # all tests (~2s)
uv run python -m pytest tests/ -v -m "not slow" # skip distance estimation
```

| File | Tests | Description |
|------|-------|-------------|
| `test_known_codes.py` | 40 | Validates polynomial construction, input validation, code building, all 14 known code parameters, FOM computation, cascade stages, batch evaluation, JSON persistence, Pareto front, seed solution, and run tracking. |
| `test_pbb_code.py` | — | PBB code construction, commutativity check, symplectic logicals. |
| `test_bposd_noncss.py` | — | Non-CSS BP-OSD: achievable-syndrome sampling, channel decomposition, low-weight logical detection. |
| `test_clifford_equivalence.py` | — | LC equivalence: Hadamard 2-coloring, S-gate algebraic check, group-level rank check, GF(2) affine system solver. |
| `test_mirror_code.py` | — | Mirror code construction (Khesin & Lu). |
| `test_parallel_k_screening.py` | — | Parallel MILP k-screening. |
| `test_ansatz.py` | — | Campaign 4 mixed-monomial ansatz evaluation. |
| `test_k_formula_verification.py` | — | k = 8ℓ/3 formula verification for univariate codes. |
| `test_cm_verification.py` | — | Constant-monomial code verification. |

## Multi-Decoder Verification Protocol

These standalone scripts implement the multi-decoder verification protocol used for publication claims.

### Publication protocol (150k trials)

3 decoder configurations × 10 batches × 5,000 trials = 150,000 total per code. Verified distance = global minimum across all 30 batches.

**Decoder configurations** (defined in `soak_test.py:65`):

| Config | BP method | OSD method | OSD order |
|--------|-----------|------------|-----------|
| 1 | product_sum | OSD_0 | 0 |
| 2 | product_sum | OSD_CS | 10 |
| 3 | minimum_sum | OSD_CS | 10 |

| Script | Runtime | Description |
|--------|---------|-------------|
| `soak_test.py` | 30-120 min/code | Publication-quality 150k-trial protocol. 9 priority codes at n=144, 288, 360. Reports CONFIRMED/REVISED with comparison vs Bravyi et al. |
| `verify_bravyi_codes.py` | ~2h | Re-verifies 3 Bravyi et al. baselines ([[144,12,12]], [[288,12,18]], [[360,12,≤24]]) under the identical 150k protocol. Addresses asymmetric baseline comparison concern. |
| `verify_ensemble_codes.py` | ~24h (60k), ~60h (150k) | Two-round verification of all 145 ensemble-discovered codes. Use `--round 60k` or `--round 150k`. |
| `extended_verification.py` | ~8h | 1.5M-trial verification (10× protocol: 3 × 10 × 50,000) on 3 headline codes. Tests bound stability vs 150k. |

```bash
uv run python tests/soak_test.py                     # all codes, full protocol
uv run python tests/soak_test.py --codes "360,40"     # specific code
uv run python tests/verify_ensemble_codes.py --round 60k   # first round
uv run python tests/verify_ensemble_codes.py --round 150k  # second round
uv run python tests/extended_verification.py           # 1.5M trials on headlines
```

## MILP Distance Verification

| Script | Runtime | Description | Output |
|--------|---------|-------------|--------|
| `ilp_catalog.py` | hours | ILP exact distance for all 97 CSS catalog codes. Parses polynomials directly from `paper/2606.02418/paper.tex`. Contains its own local ILP solver (independent of `evaluation/distance_milp.py`) for cross-validation. | `results/ilp_catalog.json` |
| `milp_optimality_audit.py` | hours | Per-logical MILP optimality check on key d≥12 codes ([[144,12,12]], [[288,16,12]], [[288,24,12]], [[360,16,14]]). Reports whether solver proved optimality for each of 2k logical operators. | `results/milp_optimality_audit.json` |
| `verify_campaign4_milp.py` | ~24h | MILP-verify all Campaign 4 (mixed-monomial) codes, ordered by FOM. Incremental, resumable. | `results/campaign4_milp_verified.jsonl` |
| `reverify_top_incumbents.py` | hours | Re-verify top 39 Campaign 4 incumbents with 10× MILP timeouts. | `results/campaign4_reverified.jsonl` |
| `verify_novel_incumbents.py` | hours | Full MILP verification (300s/logical, 7200s total timeout) for BLISS-confirmed novel codes. | — |
| `verify_ga_distances.py` | ~1h | Verify GA-found high-k codes have d≤2 via weight-2 logical search. | — |

```bash
uv run python tests/ilp_catalog.py                    # all catalog codes
uv run python tests/milp_optimality_audit.py           # d=12/14 audit
uv run python tests/milp_optimality_audit.py --timeout 300  # extended timeout
```

## BP-OSD Attack Scripts

Aggressive BP-OSD attacks on MILP incumbent codes to find tighter upper bounds.

| Script | Description | Output |
|--------|-------------|--------|
| `bp_osd_intensive_search.py` | 300k-trial intensive BP-OSD search (3 decoder configs) on top codes. | `results/bp_osd_attack.json` |
| `bp_osd_intensive_search_batch2.py` | Second batch of 300k searches. | `results/bp_osd_attack_batch2.json` |

## Code Deduplication (BLISS)

Uses BLISS graph isomorphism (via `python-igraph`) to compute canonical forms of Tanner graphs and identify equivalent codes.

| Script | Description |
|--------|-------------|
| `paper_equivalence_check.py` | BLISS Tanner graph deduplication for all CSS catalog codes (n=144, 288, 360) plus 7 known reference codes. Reports equivalence classes, cross-table matches, and final tally of distinct codes. |
| `check_all_equivalences.py` | BLISS deduplication for Campaign 5 (non-CSS PBB) codes. Groups by (n,k). |
| `check_code_equivalence.py` | Three-level equivalence check for specific codes: (1) polynomial-level algebraic automorphisms, (2) BLISS Tanner graph isomorphism, (3) explicit permutation verification (H_X and H_Z exactly preserved). |

## A=B Distance Trap

Theorem 1 in the paper (Sec III.D) proves that every BB code with A = B
and k > 0 has distance exactly 2: identical X- and Z-stabilizer
polynomials force identical column pairs in H_Z, producing a weight-2
codeword `v = e_i + e_{i+ℓm}` that is a non-trivial logical.  See the
paper for the algebraic argument.

Standalone post-hoc investigations live in [`investigations/`](../investigations/);
the d=2 outlier investigation script is at
`investigations/investigate_d2_outlier.py`.

## Threshold Simulations

Code capacity threshold simulations using BP-OSD (OSD-CS order 7, product_sum, 20 BP iterations), 100,000 shots per error rate.

| Script | Noise model | Code type | Output |
|--------|-------------|-----------|--------|
| `threshold_simulation.py` | Bit-flip (code capacity) | CSS | `results/threshold_simulation.json` |
| `threshold_simulation_noncss.py` | Bit-flip (code capacity) | Non-CSS PBB | `results/threshold_simulation_noncss.json` |
| `threshold_simulation_independent_xz.py` | Depolarizing input + iid X/Z decoder prior (mismatched; lower bound on optimal depolarizing threshold) | Non-CSS PBB | `results/threshold_simulation_depolarizing.json` |
| `threshold_simulation_missing.py` | Bit-flip | Mixed (Table V gaps) | `results/threshold_simulation_missing.json` |

## Ablation and Analysis

| Script | Runtime | Description | Output |
|--------|---------|-------------|--------|
| `ablation_study.py` | ~30 min | 3-arm comparison: seed baseline vs random trinomials vs LLM evolution. Two phases: `--phase 1` (deterministic, ~5 min) and `--phase 2` (stochastic with `--runs N`, ~30 min). | `results/ablation_study.json` |
| `ablation_k_only.py` | — | 8-arm k-only ablation across 8 lattices. | `results/ablation_k_only.json` |
| `ablation_extended.py` | — | Extended ablation with distance estimation. | `results/ablation_extended.json` |
| `ablation_ga_generators.py` | — | GA-on-generators ablation. | `results/ablation_ga_generators.json` |
| `milp_ga_codes.py` | — | MILP on 10 best GA codes (confirms FOM ≤ 4.0). | `results/milp_ga_codes.json` |
| `enumerate_constant_monomial.py` | ~1 min | Enumerates k values for the constant-monomial BB code family across all (ℓ,m) with ℓ·m ≤ 250. | `results/constant_monomial_enumeration.json` |
| `per_batch_distributions.py` | — | Per-batch decoder distribution analysis. | `results/per_batch_distributions.json` |

```bash
uv run python tests/ablation_study.py --phase 1       # deterministic arms
uv run python tests/ablation_study.py --phase 2 --runs 5  # stochastic arms
uv run python tests/enumerate_constant_monomial.py     # constant-monomial enumeration
```

## PBB Code Surveys

| Script | Description |
|--------|-------------|
| `pbb_survey.py` | Exhaustive small-lattice PBB code survey with MILP distance. |
| `pbb_survey_milp_66.py` | PBB survey at (6,6) lattice. |
| `benchmark_noncss.py` | Non-CSS comparative benchmark. |
| `lattice_survey.py` | Lattice-level code survey. |
| `generate_pbb_catalog_rows.py` | Generate LaTeX PBB catalog tables for `paper/2606.02418/pbb_catalog_tables.tex`. |

## Validation Priority Guide

When manually validating the paper's computational claims, prioritize in this order:

1. **MILP ILP formulation** (`evaluation/distance_milp.py:66` and `:333`): These produce the "exact distance" claims. Verify the mod-2 constraint encoding via integer slack variables. Check on a known code (e.g. Gross [[144,12,12]]) that MILP returns d=12.
2. **FOM computation** (`evaluation/evaluator.py:94`): Trivial formula (k·d²/n) but used everywhere. Spot-check on a few catalog codes.
3. **Non-CSS code construction** (`evaluation/pbb_code.py`): Verify the paper-convention stabilizer matrix `H = (A B C D ; 0 0 B^T A^T)` (block-1 z-part `[C | D]`, commutativity `A C^T + B D^T` symmetric). The `_poly_to_matrix` function must use `sympy.Poly` — documented bug class.
4. **LC equivalence** (`evaluation/clifford_equivalence.py:278` and `:610`): Underpins the classification of 285/295 PBB codes as genuinely non-CSS (9 Hadamard-CSS via `is_equivalently_css`, 1 uniform-S-CSS via `is_lc_equivalent_css_group`, 285 pass all tested Clifford reductions). Known issues in `paper/2606.02418/theorem3_issues.md`.
5. **Low-weight logical enumeration** (`evaluation/distance_bposd_noncss.py:40` and `:118`): Catches BP-OSD blind spots. Verify on known d=2 codes (A=B cases) that they find weight-2 logicals.
6. **Soak test protocol** (`tests/soak_test.py`): Verify multi-decoder protocol produces stable bounds. Extended 1.5M verification should match 150k results for well-behaved codes.
7. **Tanner graph deduplication** (`tests/paper_equivalence_check.py`): Verify known-equivalent codes are correctly identified.
8. **ILP catalog cross-validation** (`tests/ilp_catalog.py`): Contains its own ILP implementation independent of `evaluation/distance_milp.py` — compare results for consistency.
