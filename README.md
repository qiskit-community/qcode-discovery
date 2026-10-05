# qcode-discovery

Discovering bivariate bicycle (BB) and perturbed bivariate bicycle (PBB) quantum LDPC codes via LLM-guided evolutionary search.

Five evolution campaigns employing six LLMs from three families discover **465 distinct codes** (97 CSS, 368 non-CSS) with CSS encoding dimensions up to k = 54 (prior best: k = 16). MILP distance computation reveals a sharp rate-distance tradeoff and that BP-OSD overestimates distance by up to 12x for high-rate codes. Total cost: ~US$400 over ~140 hours.

**Paper**: "Evolutionary Discovery of Bivariate Bicycle Codes with LLM-Guided Search" — see [`paper/2606.02418/paper.tex`](paper/2606.02418/paper.tex).

## Headline Results

### Best CSS BB Codes (MILP-verified distances)

| Code | Lattice | A polynomial | B polynomial | d (MILP) | FOM | Notes |
|------|---------|-------------|-------------|----------|-----|-------|
| **[[288,24,12]]** | (12,12) | x⁶ + y + y² | y³ + x² + x⁴ | **12** (exact) | 12.0 | Direct sum of 2 gross codes |
| **[[288,50,8]]** | (18,8) | 1+y⁵+x+xy⁵ | 1+y+x⁵+x⁵y | **8** (exact) | 11.1 | Cross-factored, weight-8 |
| **[[360,20,≤14]]** | (30,6) | 1+x+x³y⁴+x⁶y² | 1+y²+x³y²+x⁶y⁴ | ≤14 | ≤10.9 | Mixed-monomial |
| **[[360,16,≤14]]** | (15,12) | y² + y⁴ + x³ | y⁶ + x² + x⁴ | ≤14 | ≤8.7 | x/y-swap |
| **[[288,16,12]]** | (12,12) | x³ + y + y² | y³ + x + x² | **12** (exact) | 8.0 | All shifts ≤ 3 hops |
| **[[144,8,12]]** | (12,6) | 1+xy²+xy³ | 1+x²y³+x³y² | **12** (exact) | 8.0 | Mixed-monomial |

### Best Non-CSS PBB Codes (MILP-verified)

| Code | Lattice | d (MILP) | FOM | Notes |
|------|---------|----------|-----|-------|
| **[[360,12,≤24]]** | (30,6) | ≤24 | ≤19.2 | Highest trusted PBB FOM, LER < p at all tested rates |
| **[[360,12,≤20]]** | (30,6) | ≤20 | ≤13.3 | Additional simulated high-distance PBB code |
| **[[144,12,12]]** | (12,6) | **12** (exact) | 12.0 | Non-CSS gross code (matches CSS FOM) |
| **[[108,8,10]]** | (9,6) | **10** (exact) | 7.4 | |
| **[[72,12,6]]** | (6,6) | **6** (exact) | 6.0 | Highest-FOM n=72 PBB representative |

FOM = kd²/n. Comparison baseline: Bravyi et al. 2024 (arXiv:2308.07915). "Exact" = MILP proven optimal (MIP gap = 0).

### Key Findings

- **A = B distance trap**: All BB codes with A = B have d = 2 exactly (Theorem 1). BP-OSD fails to detect this even at 1.5M trials.
- **[[288,24,12]] decomposition**: Tanner graph analysis reveals this is a direct sum of two [[144,12,12]] gross codes — no error-correction advantage over two independent copies.
- **BP-OSD overestimation**: Up to 12x for high-rate codes. The [[360,40,2]] yielded d_BP ≤ 24 at 150k trials vs d_MILP = 2.
- **Rate-distance tradeoff**: Indecomposable d = 12 CSS codes limited to k ≤ 16; k > 24 implies d ≤ 8. Non-CSS PBB codes occupy the same envelope.
- **Univariate/HGP codes**: The dominant high-k family (k = 40 at n = 360) has d ≤ 4 universally. k = 8ℓ/3 proven via CRT decomposition.

### Evolution Campaigns

| Campaign | Type | Models | Iters | Pop. | Codes | Time | Cost |
|----------|------|--------|-------|------|-------|------|------|
| 1 | CSS | Gemini 3 Flash | 100 | 100 | 9 | 5h | $15 |
| 2 | CSS | Ensemble (3 models) | 251 | 100 | 0* | 9.5h | $25 |
| 3 | CSS | Ensemble (3 models) | 500 | 1,000 | 145 | 21h | $50 |
| 4 | CSS | Ensemble (3 models) | 300 | 750 | 45 | 92h | $47 |
| 5 | PBB | Ensemble (3 models) | 500 | 200 | 368 | 11h | $100 |

*All Campaign 2 codes subsumed by Campaign 3. "Ensemble" = Claude Opus 4.6 + GPT-5.2/5.3 + Gemini 3 Pro/3.1 Pro with equal selection weight. Campaigns 1-3 ran on Apple M4 Max; Campaigns 4-5 on a 64-core server.

## How It Works

### What Is a BB Code?

A bivariate bicycle code is defined by two polynomials over `F_2[x,y]/(x^ℓ - 1, y^m - 1)`. Code parameters: **n = 2·ℓ·m** physical qubits, **k** logical qubits (exact via GF(2) rank), **d** code distance (expensive — see Distance Estimation). CSS commutativity is automatic, so the search is purely combinatorial.

A **perturbed bivariate bicycle (PBB) code** augments the construction with two additional polynomials (C, D) creating mixed X/Z stabilizers. When C = D = 0, the code reduces to a CSS BB code.

### Architecture

```
                    ┌─────────────────────────┐
                    │    Evolutionary Loop     │
                    │  (OpenEvolve + LLM       │
                    │   ensemble: 3 models)    │
                    │                          │
                    │  Evolves the strategy    │
                    │  in generate_candidates  │
                    └───────────┬──────────────┘
                                │ proposes polynomial tuples
                                ▼
┌──────────────────────────────────────────────────────────────┐
│                    Evaluation Cascade                         │
│                                                              │
│  Stage 1: k-only screening (~2s)  — GF(2) rank on 2 lattices│
│  Stage 2: BP-OSD distance (~30-60s) — on 8 lattices          │
│  Stage 3: MILP exact distance (Campaigns 4-5, in-loop)       │
│                                                              │
│  Post-hoc: MILP verification on all discovered codes         │
│            150k multi-decoder BP-OSD protocol                │
│            3-tier adaptive pipeline (non-CSS)                │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Results & Tracking  │
                    │  - JSON persistence  │
                    │  - JSONL run logs    │
                    │  - W&B dashboards    │
                    └──────────────────────┘
```

### Known Benchmark Codes (Bravyi et al. 2024)

| Code | ℓ | m | A polynomial | B polynomial | FOM |
|------|---|---|-------------|-------------|-----|
| [[72,12,6]] | 6 | 6 | x³ + y + y² | y³ + x + x² | 6.0 |
| [[90,8,10]] | 15 | 3 | x⁹ + y + y² | 1 + x² + x⁷ | 8.9 |
| [[144,12,12]] | 12 | 6 | x³ + y + y² | y³ + x + x² | **12.0** |
| [[288,12,18]] | 12 | 12 | x³ + y² + y⁷ | y³ + x + x² | 13.5 |
| [[360,12,≤24]] | 30 | 6 | x⁹ + y + y² | y³ + x²⁵ + x²⁶ | 19.2 |

Source: Bravyi et al. 2024 (arXiv:2308.07915)

## Setup

Requires Python ≥ 3.10. Uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Core dependencies only
uv sync

# With dev tools (pytest) — recommended
uv sync --group dev

# Everything (dev + evolution + tracking)
uv sync --group dev --group evolve --group tracking
```

> **Important:** `uv sync` removes packages from groups not listed in the command. Always include all groups you need in a single call.

### Dependencies

| Group | Packages | Purpose |
|-------|----------|---------|
| core | qldpc, ldpc, numpy, scipy, galois, sympy | BB/PBB code construction, decoding, algebra |
| dev | pytest, python-igraph | Test suite, graph isomorphism |
| evolve | openevolve, litellm[proxy] | Evolutionary search + LLM proxy |
| tracking | wandb | Experiment tracking dashboards |

## Usage

### Run the evaluation pipeline

```bash
# Evaluate all 18 target lattices (quick mode — k only, ~30s)
uv run python main.py --quick

# Evaluate specific lattices with distance estimation
uv run python main.py --lattices 12,6 6,6

# Verbose output
uv run python main.py --lattices 6,6 --quick -v
```

### Run tests

```bash
# All tests (~1s)
uv run python -m pytest tests/ -v

# Skip slow tests (distance estimation)
uv run python -m pytest tests/ -v -m "not slow"

# Single test
uv run python -m pytest tests/test_known_codes.py::TestEvaluator::test_gross_code_quick -v
```

### Use as a library

```python
from evaluation.evaluator import evaluate_candidate

# Evaluate the gross code
result = evaluate_candidate(
    12, 6,
    [(3, 0), (0, 1), (0, 2)],  # A = x^3 + y + y^2
    [(0, 3), (1, 0), (2, 0)],  # B = y^3 + x + x^2
)
print(f"[[{result['n']},{result['k']},{result['d']}]] FOM={result['fom']:.1f}")
# [[144,12,12]] FOM=12.0
```

```python
from evaluation.pbb_code import build_pbb_code
# Build a non-CSS PBB code (4 polynomials)
code = build_pbb_code(12, 6, A_terms, B_terms, C_terms, D_terms)
```

```python
from evaluation.clifford_equivalence import is_lc_equivalent_css, verify_lc_bruteforce

# Algebraic check: can this PBB code be made CSS by per-qubit Cliffords?
result = is_lc_equivalent_css(A_terms, B_terms, C_terms, D_terms)
print(result)  # {'equivalent': False, 'conditions': {...}}

# Brute-force verification (tries all 16 uniform Clifford assignments)
result = verify_lc_bruteforce(12, 6, A_terms, B_terms, C_terms, D_terms)
print(result)  # {'equivalent': False, 'algebraic_match': True, ...}
```

## Project Structure

```
qcode-discovery/
├── main.py                                # CLI entry point for manual evaluation
├── pyproject.toml                         # uv project config + dependency groups
├── evaluation/                            # Core evaluation pipeline
│   ├── bb_code.py                         #   CSS BB code construction (qldpc.BBCode)
│   ├── pbb_code.py                        #   PBB non-CSS code construction
│   ├── distance.py                        #   BP-OSD distance estimation (CSS)
│   ├── distance_milp.py                   #   MILP exact distance (CSS + symplectic)
│   ├── distance_bposd_noncss.py           #   BP-OSD for non-CSS (achievable-syndrome sampling)
│   ├── evaluator.py                       #   Multi-stage evaluation cascade
│   ├── clifford_equivalence.py            #   LC equivalence to CSS (6 public functions, see below)
│   ├── mirror_code.py                     #   Mirror code construction (Khesin & Lu)
│   ├── results.py                         #   JSON persistence & Pareto front
│   ├── tracking.py                        #   Run logging and metrics (JSONL)
│   └── threshold.py                       #   Threshold simulation stub (not implemented)
├── evolve/                                # LLM-guided evolutionary search
│   ├── seed_solution.py                   #   CSS seed (Campaigns 1-3)
│   ├── seed_solution_ansatz.py            #   Mixed-monomial seed (Campaign 4)
│   ├── seed_solution_noncss.py            #   PBB 4-tuple seed (Campaign 5)
│   ├── openevolve_evaluator.py            #   CSS evaluator adapter
│   ├── openevolve_evaluator_noncss.py     #   Non-CSS evaluator (3-tier pipeline)
│   ├── _noncss_distance_worker.py        #   Multiprocessing worker for non-CSS distance
│   ├── run_evolution.py                   #   Entry point (--noncss for Campaign 5)
│   ├── config.yaml                        #   Config: Campaigns 1-3
│   ├── config_ansatz.yaml                 #   Config: Campaign 4
│   ├── config_ansatz_server.yaml          #   Config: Campaign 4 (64-core server)
│   ├── config_noncss.yaml                 #   Config: Campaign 5
│   ├── prompt_context.md                  #   Domain knowledge: Campaigns 1-3
│   ├── prompt_context_ansatz.md           #   Domain knowledge: Campaign 4
│   └── prompt_context_noncss.md           #   Domain knowledge: Campaign 5
├── scripts/                               # Verification & publication scripts
│   ├── verify_campaign7.py                #   Campaign 5 initial BLISS dedup + verification
│   ├── verify_campaign7d.py               #   Campaign 5 variant verification
│   ├── verify_deep_milp.py                #   Deep MILP (up to 14400s/logical, 60 workers)
│   ├── verify_deep_milp_remote_n360.py    #   Remote MILP for n=360 codes
│   ├── verify_deep_milp_remote_standalone.py #  Standalone remote MILP worker
│   ├── verify_from_scratch.py             #   Full re-verification from raw data
│   ├── verify_publication.py              #   Merge all verification passes → publication
│   ├── verify_top_codes.py                #   Top codes verification
│   ├── verify_top_codes_deep.py           #   Extended MILP on top codes
│   ├── extract_witnesses.py               #   Extract witness (low-weight) operators
│   └── test_fast_weight_check.py          #   Fast weight-check utility
├── tests/                                 # Test suite + analysis scripts
│   ├── test_known_codes.py                #   pytest regression suite (40 tests)
│   ├── test_bposd_noncss.py              #   Non-CSS BP-OSD tests
│   ├── test_pbb_code.py                   #   PBB code construction tests
│   ├── test_clifford_equivalence.py       #   Clifford equivalence tests
│   ├── test_mirror_code.py                #   Mirror code tests
│   ├── test_ansatz.py                     #   Campaign 4 mixed-monomial tests
│   ├── test_cm_verification.py            #   Constant-monomial verification tests
│   ├── test_k_formula_verification.py     #   k = 8ℓ/3 formula verification
│   ├── test_parallel_k_screening.py       #   Parallel k-screening tests
│   ├── ilp_catalog.py                     #   MILP catalog of all 97 CSS codes
│   ├── soak_test.py                       #   150k multi-decoder protocol
│   ├── threshold_simulation.py            #   CSS code capacity simulation
│   ├── threshold_simulation_noncss.py     #   Non-CSS code capacity simulation (X-only noise)
│   ├── threshold_simulation_independent_xz.py # PBB depolarizing channel, iid X/Z decoder prior (n ≤ 144)
│   ├── threshold_simulation_360_12_20_depol.py # PBB [[360,12,≤20]] depolarizing channel
│   ├── threshold_simulation_360_12_24_depol.py # PBB [[360,12,≤24]] depolarizing channel
│   ├── threshold_simulation_360_12_24.py  #   PBB [[360,12,≤24]] X-only channel
│   ├── threshold_simulation_missing.py    #   Gap-filling simulations (extra CSS + [[360,12,≤20]])
│   ├── verify_decomposition_288_24_12.py  #   [[288,24,12]] direct-sum proof
│   ├── verify_bravyi_codes.py            #   Bravyi baseline MILP validation
│   ├── verify_ensemble_codes.py          #   Two-round ensemble verification (60k/150k)
│   ├── verify_campaign4_milp.py          #   Campaign 4 MILP on all codes
│   ├── verify_ga_distances.py            #   Exhaustive GA distance verification
│   ├── verify_novel_incumbents.py        #   Novel incumbent verification
│   ├── reverify_top_incumbents.py        #   Extended MILP on C4 top 39 codes
│   ├── check_all_equivalences.py         #   BLISS Tanner-graph deduplication
│   ├── check_code_equivalence.py         #   Pairwise code equivalence checking
│   ├── paper_equivalence_check.py        #   Paper-specific equivalence checks
│   ├── ablation_study.py                 #   Original 3-arm ablation
│   ├── ablation_k_only.py               #   8-arm k-only ablation
│   ├── ablation_extended.py             #   Extended ablation with distance
│   ├── ablation_ga_generators.py        #   GA-on-generators ablation
│   ├── milp_ga_codes.py                 #   MILP on best GA codes
│   ├── milp_optimality_audit.py         #   d=12/14 per-logical optimality audit
│   ├── bp_osd_intensive_search.py       #   300k-trial BP-OSD search on top codes
│   ├── bp_osd_intensive_search_batch2.py #  Batch 2 of the same search
│   ├── per_batch_distributions.py       #   Per-batch decoder distribution analysis
│   ├── extended_verification.py         #   1.5M trial extended verification
│   ├── enumerate_constant_monomial.py   #   Factored-product enumeration
│   ├── generate_pbb_catalog_rows.py     #   Generate LaTeX PBB catalog tables
│   ├── benchmark_noncss.py              #   Non-CSS comparative benchmark
│   ├── lattice_survey.py                #   Lattice-level code survey
│   ├── pbb_survey.py                    #   PBB code survey at small lattices
│   └── pbb_survey_milp_66.py            #   PBB MILP survey at (6,6)
├── results/                               # Verified results and evolution artifacts
│   ├── ilp_catalog.json                   #   97 CSS codes with MILP distances
│   ├── campaign7_publication_merged.jsonl  #   368 non-CSS PBB codes (publication-quality)
│   ├── threshold_simulation*.json         #   Code capacity simulation data
│   ├── ablation_*.json                    #   Ablation study results
│   ├── evolution_gemini3flash_seed42/     #   Campaign 1 checkpoints
│   ├── evolution_ensemble1_pop100/        #   Campaign 2 checkpoints
│   ├── evolution_ensemble2_pop1000/       #   Campaign 3 checkpoints
│   ├── evolution_ansatz_campaign4/        #   Campaign 4 checkpoints
│   └── evolution/                         #   Campaign 5 runs
│       └── campaign7/                    #     Published Campaign 5 checkpoints
├── paper/                                 # Paper sources (one subfolder per paper)
│   └── 2606.02418/                        #   "Evolutionary Discovery of BB Codes" paper
│       ├── paper.tex                      #     Main paper
│       ├── supplemental.tex               #     Supplemental material
│       ├── css_catalog_tables.tex         #     Auto-generated CSS catalog
│       ├── pbb_catalog_tables.tex         #     Auto-generated PBB catalog
│       └── figures/                       #     Figure generation scripts + PDFs
└── plans/                                 # Research direction plans
```

See [results/README.md](results/README.md) for details on each data file. See [tests/README.md](tests/README.md) for documentation on all analysis scripts.

## Distance Estimation

### Methods

| Method | Code | Speed | Accuracy | Paper Section |
|--------|------|-------|----------|---------------|
| BP-OSD (OSD_0) | `evaluation/distance.py` | Fast | Loose (up to 12x overestimate) | Sec V.A |
| BP-OSD (OSD-CS order 10) | `evaluation/distance.py` | Medium | Tighter | Sec V.A |
| MILP exact (CSS) | `evaluation/distance_milp.py` | Minutes-hours | Exact or upper bound | Sec V.B |
| MILP symplectic (non-CSS) | `evaluation/distance_milp.py` | ~4x slower than CSS | Exact or upper bound | Sec V.B |
| BP-OSD non-CSS | `evaluation/distance_bposd_noncss.py` | Fast | Requires achievable-syndrome sampling | Sec V.C |
| Hash-exact enumeration | `evaluation/distance_bposd_noncss.py` | Fast for d ≤ 6 | Exact | Sec V.C |
| Multi-decoder soak test | `tests/soak_test.py` | 30-120 min/code | Publication-quality | Sec V.A |

### Key insight: BP-OSD is unreliable for high-rate codes

MILP ground truth reveals that BP-OSD overestimates distance by 3-12x for codes with k/n > 0.1. The trust filter in the evolution cascade (d/√n ≤ 1.3 trusted, ≥ 2.0 discarded) mitigates this but is imperfect. For non-CSS codes, standard random syndrome sampling fails entirely — achievable-syndrome sampling (computing the reachable syndrome subspace via GF(2) null-space projection) is required.

## Known Limitations

- **BP-OSD overestimation:** Up to 12x for high-rate codes (k/n > 0.1). A single BP-OSD run is unreliable — estimates can range from 6 to 18 across independent batches on the same code. All publication distance claims use 150,000+ trials (3 decoders × 10 batches × 5,000 trials) via `tests/soak_test.py`.
- **MILP incumbents:** Some large codes have `milp_incumbent` status (feasible solution found, but not proven optimal). This gives a valid upper bound on d, not exact d. The `milp_details["exact"]` flag is the authoritative indicator of proven optimality. See `tests/milp_optimality_audit.py` for per-logical audit.
- **LC equivalence coverage:** `evaluation/clifford_equivalence.py` exhaustively tests all 36 uniform per-block Clifford assignments (6×6 over {I, S, H, HS, SH, HSH}). Non-uniform {I,S}, {H,HS}, and Hadamard patterns are handled by polynomial-time checks. Non-uniform {SH,HSH} patterns on block 1 and cross-family non-uniform Clifford assignments remain unchecked.
- **qldpc `get_logical_ops()` bug:** The library's logical operator computation fails on some non-CSS codes (singular matrix errors). The codebase uses its own GF(2)-based `get_symplectic_logicals()` at `evaluation/pbb_code.py` instead.
- **`_poly_to_matrix` bug class:** `evaluation/pbb_code.py` must use `sympy.Poly(expr, x, y)`, not a bare sympy expression — qldpc silently produces wrong circulant matrices for multi-term polynomials otherwise.
- **PBB C/D convention:** `evaluation/pbb_code.py` and the catalog in `results/campaign7_publication_merged.jsonl` follow paper Eq. 2: the block-1 Z-part is `[C \| D]`. An earlier `ptb_code.py` (now renamed) used `[D \| C]`; legacy JSON/JSONL data was migrated by `scripts/migrate_cd_convention.py`. PBB polynomial tuples written by hand should match paper Table V (e.g. for the [[144,12,12]] PBB at (12,6): C = y + x³y, D = y³ + x³y³). Passing legacy-convention values to `build_pbb_code` raises `ValueError: Commutativity violated`.

## LC Equivalence to CSS (`evaluation/clifford_equivalence.py`)

This module determines whether a non-CSS PBB code can be transformed into a CSS code via per-qubit single-qubit Clifford operations. It underpins the paper's claim that 357 of 368 discovered PBB codes are CSS-inequivalent within the tested LC families (Sec. III.D, App. F): 10 are Hadamard-equivalent to CSS via non-uniform per-qubit `H` (caught by `is_equivalently_css`), 1 is LC-equivalent via uniform `S` on both blocks (caught by `is_lc_equivalent_css_group`), and the remaining 357 fail every tested Clifford reduction.

### Functions (in escalating generality)

| Function | Line | Scope | Description |
|----------|------|-------|-------------|
| `is_equivalently_css(code)` | 41 | Hadamard only | 2-coloring of constraint graph via Union-Find with parity. For mirror codes (no Y support). |
| `is_lc_equivalent_css(A, B, C, D)` | 220 | Generator-level, uniform S | Tests 4 algebraic conditions: C=D=0 / C=0,D=B / D=0,C=A / C=A,D=B. Frozenset comparison. |
| `is_lc_equivalent_css_group(ell, m, A, B, C, D)` | 278 | Group-level, uniform S | Rank condition: `rowspan([C+s1*A \| D+s2*B]) ⊆ rowspan([B^T \| A^T])` for (s1,s2) in {0,1}². Strictly more general than generator-level. |
| `verify_lc_bruteforce(ell, m, A, B, C, D)` | 406 | Uniform, 6 gates | Tests all 36 assignments of {I,S,H,HS,SH,HSH} × {I,S,H,HS,SH,HSH} on 2 blocks. Checks both generator-level and group-level CSS. |
| `check_lc_pattern_feasibility(ell, m, A, B, C, D)` | 538 | Per-qubit {I,S} and {H,HS} | Column-wise feasibility: can per-qubit s_j cancel the Z-part? For circulant matrices, only uniform patterns are feasible. |
| `verify_uniform_reduction_exact(ell, m, A, B, C, D)` | 620 | Per-qubit {I,S}, non-uniform | Full GF(2) affine system solver. Determines if **any** (possibly non-uniform) S-pattern makes the group CSS within the covered Clifford family. |

The typical verification pipeline runs `is_lc_equivalent_css_group` (fast algebraic check) on all codes, then `verify_lc_bruteforce` and `verify_uniform_reduction_exact` for independent confirmation.

### Reproducing the 11 / 357 CSS-equivalence accounting

The paper's §III.D / Appendix F accounting on the 368-code PBB catalog (10 Hadamard-CSS + 1 uniform-S-CSS + 357 CSS-inequivalent) is reproduced end-to-end in ~2 min by:

```bash
PYTHONPATH=. uv run python scripts/run_lc_analysis.py
```

This loads `results/campaign7_dedup.jsonl` (368 records), runs `is_equivalently_css` (Hadamard 2-coloring) + `is_lc_equivalent_css_group` (uniform per-block S/H) + `verify_uniform_reduction_exact` ({I,S}/{H,HS} GF(2)) on each, writes `results/campaign7_lc_analysis.jsonl`, and prints the 10 / 1 / 11 / 357 totals. The lone uniform-S code's polynomials (the `[[36,4,6]]` at `(6,3)` with $C = y + x^3 y$, $D = xy + x^4 y$) are visible in the per-record output.

## Ablation Study

Eight arms compared on the same 8 lattices using deterministic Σ_k = Σ max{k} per lattice:

| Arm | Budget | Σ_k | Best FOM |
|-----|--------|-----|----------|
| Random (10³/lattice) | 8,000 | 172 | — |
| Random (10⁴/lattice) | 80,000 | 299 ± 26 | — |
| Seed | 21,962 | 276 | — |
| GA (exponent tuples) | 80,000 | 703 ± 142 | 4.0 (d≤2 at standard lattices) |
| GA-G (generators) | 14,000 | 1,037 ± 22 | d≤2 everywhere |
| Campaign 1 (Flash) | 213,216 | 704 | **12.0** (indecomposable: 8.0) |

GA-G exceeds LLM on Σ_k but exclusively through degenerate d ≤ 2 codes. The LLM's advantage is semantic understanding: discovering algebraic families with d ≥ 6 and FOM up to 12.0.

Scripts: [`tests/ablation_k_only.py`](tests/ablation_k_only.py), [`tests/ablation_ga_generators.py`](tests/ablation_ga_generators.py), [`tests/milp_ga_codes.py`](tests/milp_ga_codes.py).

## Structural Families

Four structural families emerged, each with a distinct rate-distance profile:

1. **Univariate (HGP of cyclic codes)**: A = f(y), B = g(x). Highest k (up to 40 at n=360) but d ≤ 4 universally. k = 8ℓ/3 proven via CRT.
2. **x/y-swap**: A = x^a + y^b + y^c, B = y^d + x^e + x^f. Only trinomial family with d ≥ 6. Best: [[288,16,12]].
3. **Mixed-monomial** (Campaign 4): 4-6 term polynomials with x^a y^b terms. Access new (k,d) combinations. Best: [[288,50,8]] (FOM = 11.1).
4. **Non-CSS PBB** (Campaign 5): 4-tuples (A, B, C, D). Match CSS FOM = 12.0 with non-CSS stabilizers. Best trusted upper bound: [[360,12,≤24]] (FOM ≤ 19.2); [[360,12,≤20]] is an additional simulated high-distance code.

## Evolutionary Search with OpenEvolve

The evolutionary loop uses [OpenEvolve](https://github.com/codelion/openevolve) to evolve the `generate_candidates` function via LLM-guided mutations.

```bash
# CSS evolution (Campaigns 1-3)
uv run python evolve/run_evolution.py \
    --models anthropic/claude-opus-4-6 openai/gpt-5.2 gemini/gemini-3-pro \
    --iterations 500 --run-name ensemble_v2

# Campaign 4 (mixed-monomial) — paper run used config_ansatz_server.yaml (pop=750)
uv run python evolve/run_evolution.py \
    --config evolve/config_ansatz_server.yaml \
    --seed evolve/seed_solution_ansatz.py \
    --iterations 300

# Campaign 5 (non-CSS PBB)
uv run python evolve/run_evolution.py --noncss --iterations 200
```

A LiteLLM-compatible proxy must be running. See [`evolve/config.yaml`](evolve/config.yaml) for full configuration options.

---

## Paper-to-Repository Mapping

This section maps every section, figure, table, and claim in the paper and supplemental material to the corresponding code, scripts, and data files in this repository.

### Campaign Naming Convention

The repository uses internal numbering that differs from the paper:

| Paper | Repo Name | Run Directory | Config |
|-------|-----------|---------------|--------|
| Campaign 1 | Campaign 1 (Flash) | `results/evolution_gemini3flash_seed42/` | `evolve/config.yaml` |
| Campaign 2 | Ensemble #1 (pop=100) | `results/evolution_ensemble1_pop100/` | `evolve/config.yaml` |
| Campaign 3 | Ensemble #2 (pop=1000) | `results/evolution_ensemble2_pop1000/` | `evolve/config.yaml` |
| Campaign 4 | Ansatz | `results/evolution_ansatz_campaign4/` | `evolve/config_ansatz_server.yaml` |
| Campaign 5 | Campaign 7 | `results/evolution/campaign7/` | `evolve/config_noncss.yaml` |

### Paper Sections → Code

| Paper Section | Key Files |
|--------------|-----------|
| **Sec III.A** BB codes | `evaluation/bb_code.py` |
| **Sec III.B** PBB codes | `evaluation/pbb_code.py`, `evaluation/clifford_equivalence.py` |
| **Sec III.C** Figure of merit | `evaluation/evaluator.py` (FOM = kd²/n computed in `evaluate_candidate()`) |
| **Sec III.D** A=B distance trap (Theorem 1) | Algebraic proof in paper Sec III.D |
| **Sec IV.A** Evolutionary framework | `evolve/run_evolution.py`, `evolve/config*.yaml`, `evolve/prompt_context*.md` |
| **Sec IV.B** Evaluation cascade | `evolve/openevolve_evaluator.py`, `evolve/openevolve_evaluator_noncss.py`, `evaluation/evaluator.py` |
| **Sec IV.C** Campaign configs | `evolve/seed_solution.py` (C1-3), `evolve/seed_solution_ansatz.py` (C4), `evolve/seed_solution_noncss.py` (C5) |
| **Sec V.A** BP-OSD limits | `evaluation/distance.py`, `tests/soak_test.py` |
| **Sec V.B** MILP exact distance | `evaluation/distance_milp.py`, `tests/ilp_catalog.py`, `tests/milp_optimality_audit.py` |
| **Sec V.C** Non-CSS pipeline | `evaluation/distance_bposd_noncss.py` (achievable-syndrome sampling), `evaluation/distance_milp.py` (symplectic MILP) |
| **Sec V.D** Trust boundaries | `evaluation/evaluator.py` (`DISTANCE_TRUST_RATIO`, `DISTANCE_UNTRUST_RATIO`) |
| **Sec VI.A** Structural families | `tests/check_all_equivalences.py` (BLISS dedup), `tests/verify_decomposition_288_24_12.py` (direct-sum proof), `tests/test_k_formula_verification.py` (k = 8ℓ/3) |
| **Sec VI.B** Rate-distance tradeoff | `paper/2606.02418/figures/plot_pareto_frontier.py` |
| **Sec VI.C** Comparison with prior art | `tests/verify_bravyi_codes.py`, `results/ilp_catalog.json` |
| **Sec VI.D** Code capacity simulations | `tests/threshold_simulation.py`, `tests/threshold_simulation_noncss.py`, `tests/threshold_simulation_independent_xz.py` (depolarizing channel + iid X/Z decoder, n ≤ 144), `tests/threshold_simulation_360_12_20_depol.py`, `tests/threshold_simulation_360_12_24_depol.py`, `tests/threshold_simulation_360_12_24.py`, `tests/threshold_simulation_missing.py` |
| **Sec VI.E** BP-OSD overestimation | `tests/bp_osd_intensive_search.py`, `tests/bp_osd_intensive_search_batch2.py`, `tests/per_batch_distributions.py`, `tests/extended_verification.py` |
| **Sec VI.F** Ablation | `tests/ablation_k_only.py`, `tests/ablation_ga_generators.py`, `tests/ablation_extended.py`, `tests/milp_ga_codes.py`, `tests/verify_ga_distances.py` |
| **App C** Representative LLM mutations | `results/evolution_gemini3flash_seed42/` (Campaign 1 checkpoint diffs) |
| **App F** LC-CSS equivalence accounting | `evaluation/clifford_equivalence.py` (`is_lc_equivalent_css`, `verify_lc_bruteforce`, `check_lc_pattern_feasibility`) |
| **SM Sec I** CSS catalog | `paper/2606.02418/css_catalog_tables.tex` ← `results/ilp_catalog.json` |
| **SM Sec II** PBB catalog | `paper/2606.02418/pbb_catalog_tables.tex` ← `results/campaign7_publication_merged.jsonl`, `tests/generate_pbb_catalog_rows.py` |
| **SM Sec III** Per-decoder comparison | `results/soak_test_publication.json`, `results/ensemble_verification_150k.json` |
| **SM Sec IV** BP-OSD per-batch analysis | `tests/per_batch_distributions.py`, `results/per_batch_distributions.json` |
| **SM Sec V** Distance tightening | `results/ensemble_verification_60k.json` → `results/ensemble_verification_150k.json` |
| **SM Sec VI** Univariate families | `results/ilp_catalog.json` (UV-pattern filter), `tests/test_k_formula_verification.py` |
| **SM Sec VII** Full threshold data | `results/threshold_simulation*.json` (all 6 simulation files) |
| **SM Sec VIII** Utility stratification | `results/ilp_catalog.json`, `results/campaign7_publication_merged.jsonl` |
| **SM Sec IX** Novelty assessment | `tests/check_all_equivalences.py`, `tests/paper_equivalence_check.py` |
| **SM Sec X** Campaign details | Evolution run directories (see Campaign Naming Convention above) |
| **SM Sec XI** Ablation methodology | `tests/ablation_k_only.py`, `tests/ablation_ga_generators.py`, `tests/verify_ga_distances.py`, `tests/milp_ga_codes.py` |
| **SM Sec XII** MILP formulations | `evaluation/distance_milp.py` (CSS Hamming weight; symplectic uses the per-qubit-OR encoding $w_j \geq x_j$, $w_j \geq z_j$, $w_j \leq x_j + z_j$ -- the convex hull of binary OR for $\{0,1\}$ variables, not the McCormick envelope) |

### Figures → Generation Scripts

All figure scripts are in `paper/2606.02418/figures/`. To regenerate:

```bash
cd paper/2606.02418/figures
uv run python <script>.py
```

| Figure | Script | Data Sources |
|--------|--------|-------------|
| **Fig 1**: Evolution pipeline | `fig_pipeline.py` | (schematic, no data) |
| **Fig 2**: LER curves | `fig_ler_curves.py` | `results/threshold_simulation.json`, `results/threshold_simulation_noncss.json`, `results/threshold_simulation_milp.json`, `results/threshold_simulation_depolarizing.json`, `results/threshold_simulation_360_12_20_depol.json`, `results/threshold_simulation_360_12_24_depol.json` |
| **Fig 3**: Pareto frontier (k vs d, FOM vs n) | `plot_pareto_frontier.py` | `results/ilp_catalog.json`, `results/campaign7_publication_merged.jsonl` |
| **Fig 4**: Distance tightening | `fig_tightening.py` | `results/soak_test_publication.json`, `results/ensemble_verification_150k.json` |
| **SM Fig**: Fitness trajectories (C1-C3) | `fig_fitness_trajectory.py` | `results/evolution_gemini3flash_seed42/`, `results/evolution_ensemble1_pop100/`, `results/evolution_ensemble2_pop1000/` |
| **SM Fig**: Per-batch distributions | `tests/per_batch_distributions.py` | `results/per_batch_distributions.json` |

### Tables → Data Files

#### Main Paper

| Table | Content | Data Source |
|-------|---------|-------------|
| **Table I**: Campaign summary | 5 campaigns | Campaign run directories |
| **Table II**: Best PBB codes per lattice | 7 non-CSS codes | `results/campaign7_publication_merged.jsonl` |
| **Table III**: Best CSS codes (MILP) | 12 CSS codes with d_BP vs d_MILP | `results/ilp_catalog.json` + `results/soak_test_publication.json` |
| **Table IV**: Comparison with prior art | k-ratio at n=144,288,360 | `results/ilp_catalog.json` |
| **Table V**: Practically relevant codes | d ≥ 8, FOM ≥ 6 | `results/ilp_catalog.json` + `results/campaign7_publication_merged.jsonl` |
| **Table VI**: CSS threshold data | 8 CSS codes at p=1,3,5,8% | `results/threshold_simulation.json`, `results/threshold_simulation_milp.json` |
| **Table VII**: Non-CSS threshold data | 5 PBB + CSS baseline | `results/threshold_simulation_noncss.json`, `results/threshold_simulation_depolarizing.json`, `results/threshold_simulation_360_12_20_depol.json` and `results/threshold_simulation_360_12_24_depol.json` (depol rows for the (30,6) lattice), `results/threshold_simulation_missing.json` ([[360,12,≤20]] X-only row), `results/threshold_simulation_360_12_24.json` ([[360,12,≤24]] X-only row) |
| **Table VIII**: Ablation Σ_k summary | 8 arms | `results/ablation_k_only.json` |
| **Table IX**: Per-lattice max k | All arms × 8 lattices | `results/ablation_k_only.json`, `results/ablation_study.json` |

#### Supplemental Material

| Table | Content | Data Source |
|-------|---------|-------------|
| **Tables cat144/cat288/cat360**: CSS catalog | 225 representations → 97 distinct | `paper/2606.02418/css_catalog_tables.tex` ← `results/ilp_catalog.json` |
| **Tables pbb_cat***: PBB catalog | 368 non-CSS codes | `paper/2606.02418/pbb_catalog_tables.tex` ← `results/campaign7_publication_merged.jsonl` |
| **Tab: Decoder rates**: Per-decoder success by k/n | Decoder comparison | `results/soak_test_publication.json`, `results/ensemble_verification_150k.json` |
| **Tab: Per-batch stats**: 5 codes × 3 decoders | Per-batch distributions | `results/per_batch_distributions.json` |
| **Tab: Extended verification**: 1.5M trials | 4 codes at 10x depth | `results/extended_verification_1500k.json`, `results/gross_verification_1500k.json` |
| **Tab: Tightening examples**: 60k→150k | Distance tightening | `results/soak_test_publication.json` vs `results/ensemble_verification_60k.json` |
| **Tab: Univariate families**: HGP codes | All UV codes with BP vs MILP | `results/ilp_catalog.json` (UV-pattern filter) |
| **Full threshold tables**: CSS + non-CSS | 12 error rates × all codes | `results/threshold_simulation*.json` |

### Key Claims → Verification Scripts

| Paper Claim | Script | Result File |
|-------------|--------|-------------|
| **Theorem 1**: A=B ⟹ d=2 | Paper Sec III.D | Algebraic proof |
| **[[288,24,12]] = 2 × gross code** | `tests/verify_decomposition_288_24_12.py` | Tanner graph connectivity: 2 components |
| **k = 8ℓ/3 (univariate)** | `tests/test_k_formula_verification.py` | Verified on 1,680 parameter combinations |
| **LC-CSS accounting for PBB codes** | `evaluation/clifford_equivalence.py`, `tests/test_clifford_equivalence.py` | Algebraic + brute-force + feasibility checks on all 368 codes |
| **BP-OSD 12x overestimate** | `tests/bp_osd_intensive_search.py`, `tests/bp_osd_intensive_search_batch2.py` | `results/bp_osd_attack.json` |
| **97 distinct CSS codes** | `tests/check_all_equivalences.py` | BLISS Tanner-graph deduplication |
| **368 distinct PBB codes** | `scripts/verify_campaign7.py`, `scripts/verify_publication.py` | `results/campaign7_dedup.jsonl` (BLISS-deduplicated), `results/campaign7_publication_merged.jsonl` (publication-quality MILP+BP-OSD verification) |
| **MILP validation on Bravyi baselines** | `tests/verify_bravyi_codes.py` | `results/bravyi_verified.json` |
| **MILP optimality audit (d=12, d=14)** | `tests/milp_optimality_audit.py` | `results/milp_optimality_audit.json` |
| **GA-G Σ_k=1037 but d≤2** | `tests/ablation_ga_generators.py` | `results/ablation_ga_generators.json` |
| **GA MILP verification (FOM ≤ 4.0)** | `tests/milp_ga_codes.py` | `results/milp_ga_codes.json` |
| **Deep MILP: 33 corrections (22%)** | `scripts/verify_deep_milp.py` | `results/campaign7_deep_milp.jsonl` |
| **150k multi-decoder protocol** | `tests/soak_test.py` | `results/soak_test_publication.json` |
| **1.5M extended verification** | `tests/extended_verification.py` | `results/extended_verification_1500k.json` |
| **Code capacity: LER < p** | `tests/threshold_simulation.py`, `tests/threshold_simulation_noncss.py` | `results/threshold_simulation*.json` |

### Campaign 4 (Mixed-Monomial) Verification

| Script | Purpose | Output |
|--------|---------|--------|
| `tests/verify_campaign4_milp.py` | MILP on all C4 codes | `results/campaign4_milp_verified.jsonl` |
| `tests/reverify_top_incumbents.py` | Extended MILP on top 39 | `results/campaign4_reverified.jsonl` |
| `tests/bp_osd_intensive_search.py` | 300k BP-OSD on top 6 | `results/bp_osd_attack.json` |
| `tests/enumerate_constant_monomial.py` | Factored-product enumeration | `results/constant_monomial_enumeration.json` |

### Campaign 5 (PBB Non-CSS) Verification

| Script | Purpose | Output |
|--------|---------|--------|
| `scripts/verify_campaign7.py` | Initial BLISS dedup + verification | `results/campaign7_dedup.jsonl` |
| `scripts/verify_deep_milp.py` | Deep MILP (up to 14400s/logical, 60 workers) | `results/campaign7_deep_milp.jsonl` |
| `scripts/verify_publication.py` | Publication MILP+BP-OSD verification pass | `results/campaign7_publication.jsonl` (merged into `results/campaign7_publication_merged.jsonl` with deep MILP evidence) |
| `scripts/merge_deep_milp_results.py` | Merge deep MILP evidence into publication catalog | `results/campaign7_publication_merged.jsonl` |
| `tests/generate_pbb_catalog_rows.py` | Generate LaTeX catalog | `paper/2606.02418/pbb_catalog_tables.tex` |

### Result Files Reference

| File | Content | Paper Reference |
|------|---------|----------------|
| `results/ilp_catalog.json` | 97 CSS codes with MILP distances | Tables III, IV, V, SM catalogs |
| `results/campaign7_publication_merged.jsonl` | 368 PBB codes (publication-quality) | Table II, SM PBB catalog |
| `results/campaign7_dedup.jsonl` | BLISS-deduplicated PBB codes | Sec VI.A |
| `results/campaign7_deep_milp.jsonl` | Deep MILP on 149 PBB rows | Sec V.C |
| `results/soak_test_publication.json` | 9 C1 codes at 150k | Table III, SM decoders |
| `results/ensemble_verification_150k.json` | 145 ensemble codes at 150k | Fig 4, SM tightening |
| `results/ensemble_verification_60k.json` | 145 codes at 60k | SM tightening |
| `results/bravyi_verified.json` | 3 Bravyi baselines at 150k | Sec V.B validation |
| `results/extended_verification_1500k.json` | 3 codes at 1.5M | SM extended verification |
| `results/gross_verification_1500k.json` | Gross code at 1.5M | SM extended verification |
| `results/per_batch_distributions.json` | Per-batch decoder data | SM per-batch analysis |
| `results/milp_optimality_audit.json` | d=12/14 optimality audit | Sec V.B |
| `results/threshold_simulation.json` | CSS X-only simulation | Table VI, SM full threshold |
| `results/threshold_simulation_milp.json` | C4 codes simulation | Table VI (C4 rows) |
| `results/threshold_simulation_noncss.json` | PBB X-only simulation | Table VII, SM non-CSS |
| `results/threshold_simulation_depolarizing.json` | PBB depolarizing | Table VII, SM depol |
| `results/threshold_simulation_missing.json` | Gap-filling simulations | Table VI |
| `results/ablation_k_only.json` | 8-arm ablation | Table VIII, IX |
| `results/ablation_study.json` | Original 3-arm ablation | App B |
| `results/ablation_extended.json` | Equal-budget random + GA baselines (5 seeds × 8 lattices) | SM ablation |
| `results/ablation_ga_generators.json` | GA-G ablation | SM ablation |
| `results/milp_ga_codes.json` | MILP on 10 best GA codes (FOM ≤ 4.0) | Sec VI.F, SM ablation |
| `results/bp_osd_attack.json` | 300k BP-OSD attacks | Sec VI.E |
| `results/campaign4_milp_verified.jsonl` | C4 MILP results | Table III (C4 rows) |
| `results/campaign4_reverified.jsonl` | C4 extended MILP | SM campaigns |
| `results/bp_osd_attack_batch2.json` | Batch 2 BP-OSD attacks | Sec VI.E |
| `results/constant_monomial_enumeration.json` | Factored-product enumeration | Sec VI.A (mixed-monomial) |
| `results/threshold_simulation_360_12_20_depol.json` | [[360,12,≤20]] depolarizing sim (regen via `tests/threshold_simulation_360_12_20_depol.py`) | Table VII |
| `results/threshold_simulation_360_12_24_depol.json` | [[360,12,≤24]] depolarizing sim (regen via `tests/threshold_simulation_360_12_24_depol.py`) | Table VII |
| `results/threshold_simulation_360_12_24.json` | [[360,12,≤24]] X-only sim (regen via `tests/threshold_simulation_360_12_24.py`) | Table VII |
| `results/campaign7_report.md` | C5 summary report | — |
| `results/benchmark_noncss.json` | Non-CSS benchmark | SM utility |

---

## Tracking & Observability

Every `main.py` run is logged to `results/runs/<run_id>/`. W&B integration is built into the evolutionary search via `--wandb`.

## Citation

If you use this software or its results, please cite the associated paper:

> J. Cruz-Benito, A. W. Cross, D. Kremer, and I. Faro,
> "Evolutionary Discovery of Bivariate Bicycle Codes with LLM-Guided Search,"
> arXiv:2606.02418 [quant-ph], 2026. https://arxiv.org/abs/2606.02418

BibTeX:

```bibtex
@misc{cruzbenito2026qcodediscovery,
  title         = {Evolutionary Discovery of Bivariate Bicycle Codes with LLM-Guided Search},
  author        = {Cruz-Benito, Juan and Cross, Andrew W. and Kremer, David and Faro, Ismael},
  year          = {2026},
  eprint        = {2606.02418},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  doi           = {10.48550/arXiv.2606.02418},
  url           = {https://arxiv.org/abs/2606.02418}
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is also provided.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
