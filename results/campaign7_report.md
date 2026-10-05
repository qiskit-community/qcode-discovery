# Campaign 7: Non-CSS PBB Code Discovery via LLM-Guided Evolution

> Note: this report is retained as a campaign log. Headline parameters below have
> been reconciled with `results/campaign7_publication_merged.jsonl`; exactness
> means all relevant MILP logicals were proven optimal, while TRUSTED/PARTIAL
> entries remain upper bounds.

## 1. Executive Summary

Campaign 7 applied LLM-guided evolutionary search to discover non-CSS quantum LDPC codes using the Perturbed Bivariate Bicycle (PBB) construction. Over 500 iterations with a 3-model ensemble (Claude Opus 4.6, GPT-5.3-Codex, Gemini 3.1 Pro), the campaign explored 7 lattice dimensions from n=36 to n=360. From **18,588 raw code evaluations**, a two-stage deduplication pipeline (tuple-level then BLISS Tanner-graph canonicalization) yielded **368 structurally unique codes** — each with a distinct Tanner graph verified by BLISS canonical hashing.


**Key results:**

- **[[72,12,6]]** — FOM=6.00, exact low-weight verified. The highest-FOM n=72 PBB representative in the publication catalog.
- **[[108,8,10]]** — FOM=7.41, MILP-exact verified (deep verification, all 16 logicals proven optimal). Note: codes initially reported as [[108,8,12]] by BP-OSD were corrected to d=10 by deep MILP.
- **[[144,12,12]]** — FOM=12.00, matching the Gross code [[144,12,12]] figure of merit. Deep MILP verified all 24 logicals for 7 of 14 codes in this family; the remaining 7 are trusted upper bounds.
- **[[360,12,≤24]]** — FOM≤19.20, MILP upper bound. Largest lattice explored (n=360) and highest trusted PBB FOM in the publication catalog.
- **[[360,12,≤20]]** — FOM≤13.33, MILP upper bound. Additional high-distance n=360 code used in simulations.
- **[[360,60,6]]** — MILP-exact. 60 logical qubits from 360 physical (rate 1/6), the highest-rate non-CSS code found.

Of the 368 codes, 251 have EXACT (proven optimal) distance and 110 have TRUSTED (valid upper bound) distance; 7 additional codes are PARTIAL upper bounds with $d/\sqrt{n} \geq 1.5$. Deep MILP verification on 149 PBB catalog entries found 33 downward distance corrections, confirming that BP-OSD can overestimate distance for non-CSS codes — all TRUSTED and PARTIAL distances should be interpreted as upper bounds.

These results demonstrate that non-CSS PBB codes can match the gross-code FOM exactly and produce larger-FOM upper bounds at n=360, while LLM-guided evolution is effective at navigating the non-CSS search space.

### Comparison with concurrent work

Khesin and Lu (arXiv:2603.05496, March 2026) independently introduced "mirror codes," a construction that generalizes abelian two-block group algebra codes beyond CSS. Their best weight-6 non-CSS codes include [[60,4,10]], [[36,6,6]], [[48,8,6]], and [[85,8,9]]. Our PBB construction is related but distinct: mirror codes are parameterized by a group G and two subsets, while PBB codes independently specify all four stabilizer blocks (A, B, C, D), reducing to CSS BB codes when C=D=0. Our [[108,8,10]] (FOM=7.41) compares favorably with their best codes at similar block lengths. The two approaches are complementary — mirror codes emphasize fault-tolerant circuit constructions, while our work focuses on code parameter optimization via evolutionary search.

## 2. Background and Motivation

### 2.1 PBB Code Construction

A Perturbed Bivariate Bicycle (PBB) code is defined by four polynomials (A, B, C, D) over F₂[x,y]/(xˡ−1, yᵐ−1). The stabilizer matrix has the structure:

```
Block 1: x-part = [A | B],  z-part = [C | D]
Block 2: x-part = [0 | 0],  z-part = [B^T | A^T]
```

When C = D = 0, this reduces to a standard CSS bivariate bicycle (BB) code. Non-zero C, D create genuinely non-CSS codes by coupling X and Z stabilizers in block 1. The commutativity constraint requires (A·C^T + B·D^T) mod 2 to be symmetric.

**Parameters:** n = 2ℓm physical qubits, k logical qubits (computed via GF(2) rank), d code distance (computed via MILP and BP-OSD). The figure of merit is FOM = k·d²/n; higher FOM indicates more efficient encoding.

### 2.2 Why non-CSS?

CSS codes impose a structural constraint: X and Z stabilizers are decoupled, which simplifies syndrome extraction but limits the code space. Non-CSS codes can access structurally different stabilizer patterns at the same block length. In the publication catalog, the best n=72 PBB representative is [[72,12,6]] (FOM=6.0), while higher-distance small PBB examples such as [[72,4,8]] trade rate for distance.

### 2.3 Infrastructure Development

The evaluation infrastructure was developed iteratively prior to the production campaign:

1. **Baseline**: CSS bivariate bicycle codes with BP-OSD distance estimation. Established [[72,12,6]] FOM=6.0 as the CSS reference.
2. **Hash-exact distance**: Added weight-enumeration for exact d ≤ 6 at n ≤ 216, eliminating BP-OSD uncertainty for low-distance codes.
3. **MILP symplectic formulation**: Per-logical integer linear programs for d ≥ 7, providing provable upper bounds.
4. **Adaptive pipeline**: Combined all three tiers (§3.2) with lattice-dependent timeouts, enabling scaling to n=360.

The production campaign reported here uses the final infrastructure: 500 iterations across 7 lattices, yielding 368 unique verified codes.

## 3. Methods

### 3.1 Evolutionary Framework

The search uses OpenEvolve, an LLM-guided program synthesis framework. The evolved artifact is a Python function `generate_candidates(ell, m)` that returns lists of (A, B, C, D) polynomial tuples for a given lattice dimension. The LLM mutates this function based on evaluation feedback.

**Configuration:**

| Parameter | Value |
|-----------|-------|
| LLM ensemble | Claude Opus 4.6 (34%), GPT-5.3-Codex (33%), Gemini 3.1 Pro (33%) |
| Population | MAP-Elites with 4 islands, migration every 15 iterations |
| Iterations | 500 (352 productive evaluations) |
| Parallel evaluations | 4 programs simultaneously |
| Evaluation timeout | 1500s per program |
| Distance workers | 16 (ProcessPoolExecutor) |

**Evaluation cascade:**

1. **Stage 1 (k-only, ~2s):** Construct PBB codes on 2 fast lattices [(6,6), (6,3)], compute k via GF(2) rank. Programs with no valid k>0 codes are rejected.
2. **Stage 2 (distance, ~60–400s):** For programs passing stage 1, compute distance on 7 lattices using the adaptive pipeline (§3.2). Score = Σ(weighted FOM across lattices).

### 3.2 Adaptive Distance Pipeline

Distance computation is the computational bottleneck. We developed a 3-tier adaptive pipeline:

**Tier 1 — Hash-based exact enumeration (d ≤ 6):**

For codes with n ≤ 216, we enumerate all symplectic Pauli operators up to weight 6 using extended syndrome columns. This is an O(n³) algorithm building dictionaries of XOR patterns:
- Weight 1–2: trivial check
- Weight 3–4: pair-XOR dictionary lookup
- Weight 5–6: triple-XOR dictionary for combined pairs

If a weight-w logical is found, d = w exactly. If no logical ≤ weight 6 is found, d ≥ 7 and we proceed to tier 2.

For n > 216, only weight ≤ 4 is checked (the weight-5/6 triple-XOR dictionary requires ~89 GB at n=360).

**Tier 2 — MILP symplectic formulation:**

For each of 2k logical operators, we solve an integer linear program minimizing symplectic weight subject to commutation with all stabilizers and anticommutation with the target logical. The solver is HiGHS via `scipy.optimize.milp`.

Adaptive timeouts by lattice size:

| n range | Per-logical timeout | Total timeout |
|---------|-------------------|---------------|
| n ≤ 108 | 15s | 90s |
| n ≤ 216 | 30s | 180s |
| n > 216 | 60s | 360s |

Partial MILP results are valid upper bounds: if p of 2k logicals are solved, d ≤ min(solved weights).

**Tier 3 — BP-OSD fallback:**

When MILP times out without solving all logicals, we run belief propagation with ordered statistics decoding (BP-OSD) using multi-channel symplectic decomposition. 5 decoder runs (seeds 42, 137, 271, 314, 997) with 1000, 1000, 1000, 500, 500 trials each (3500 total) provide a statistical upper bound via d = min(all runs).

The final distance is d = min(d_hash, d_milp, d_bposd).

### 3.3 Trust Levels

Each verified code is assigned a trust level:

- **EXACT**: Distance proven optimal. Either hash-exact (all weights enumerated) or MILP-exact (all 2k logicals solved to optimality by HiGHS).
- **TRUSTED**: Distance is a valid upper bound from at least 2 independent methods (MILP partial + BP-OSD), with consistency checks (d_milp ≈ d_bposd, d/√n ≥ 0.5, d_bposd spread < 3×d).

### 3.4 Deduplication

The evolution produces many evaluations of the same or equivalent codes. We apply a two-stage deduplication pipeline to identify the structurally distinct codes.

**Stage 1 — Tuple dedup with quality filter.** Identical polynomial tuples (A, B, C, D, ℓ, m) are merged, keeping the best FOM per tuple. A selection threshold of **d ≥ 5 and k > 0** is applied to exclude trivial codes (k = 0 encodes no information; d ≤ 4 offers insufficient error correction for practical use). Of the 18,588 raw evaluations, 15,152 satisfy d ≥ 5; after merging identical tuples this yields **720 tuple-unique codes**.

**Stage 2 — BLISS canonical hashing.** Different polynomial tuples can produce isomorphic Tanner graphs (and thus identical codes). We canonicalize each code's Tanner graph using the BLISS algorithm (Junttila & Kaski, 2007) via igraph's `canonical_permutation`. The graph has 3 vertex colors: qubits (n nodes, color 0), X-checks (n_checks nodes, color 1), and Z-checks (n_checks nodes, color 2). A SHA-256 hash of the canonical edge list serves as the code's fingerprint. Codes with identical BLISS hashes are graph-isomorphic and are merged, keeping the best-verified representative: EXACT over TRUSTED, then lowest d (most conservative).

This collapses **720 → 368 BLISS-unique codes** (1.96× dedup ratio). The effect is strongly lattice-dependent:

| Lattice | Raw evals | d ≥ 5 | Tuple-unique | BLISS-unique | Dedup ratio |
|---------|-----------|-------|--------------|--------------|-------------|
| (3,6) n=36 | 361 | 3 | 2 | 2 | 1.0× |
| (6,3) n=36 | 3,380 | 495 | 46 | 45 | 1.0× |
| (6,6) n=72 | 3,390 | 3,255 | 43 | 34 | 1.3× |
| (9,6) n=108 | 3,110 | 3,086 | 209 | 107 | 2.0× |
| (12,6) n=144 | 2,555 | 2,530 | 139 | 66 | 2.1× |
| (15,6) n=180 | 3,019 | 3,017 | 158 | 64 | 2.5× |
| (30,6) n=360 | 2,773 | 2,766 | 123 | 50 | 2.5× |
| **Total** | **18,588** | **15,152** | **720** | **368** | **2.0×** |

The dedup ratio increases with lattice size: at n=36, most distinct tuples produce distinct codes, but at n=144–360, roughly 1 in 2–3 tuple-unique codes is a relabeling of another. This may reflect larger automorphism groups of codes on bigger lattices — more permutations of (x,y) exponents yield equivalent codes.

### 3.5 Deep MILP Verification

For publication-quality results, 149 PBB catalog entries across (6,6), (9,6), (12,6), (15,6), and (30,6) lattices underwent deep MILP verification: all 2k logicals per code are submitted as independent ILPs to a process pool with up to 60 workers and up to 14400s per-logical timeout in the deepest reruns. This enables solving all logicals in parallel rather than sequentially with a shared timeout. Verification ran on a single 64-core server.

### 3.6 Target Lattices

| Lattice (ℓ, m) | n = 2ℓm | Distance method | Role |
|----------------|---------|-----------------|------|
| (3, 6) | 36 | exact_w6 | Fast validation |
| (6, 3) | 36 | exact_w6 | Transposed structure |
| (6, 6) | 72 | exact_w6 + MILP | Primary target |
| (9, 6) | 108 | exact_w6 + MILP | Medium |
| (12, 6) | 144 | exact_w6 + MILP | Large |
| (15, 6) | 180 | exact_w6 + MILP | Large |
| (30, 6) | 360 | exact_w4 + MILP + BPOSD | Largest |

### 3.7 Seed Solution

The initial `generate_candidates` function contains 7 strategies:

1. **Known bases**: Enumerate C,D perturbations for 5 known (A,B) base pairs.
2. **x/y-swap structure**: A = x^a + y^b + y^c, B = y^d + x^e + x^f pattern.
3. **Systematic trinomial pairs**: Enumerate all trinomial pairs at small (ℓ,m).
4. **Cross-lattice scaling**: Lift known bases from (6,6) to larger lattices.
5. **Spread exponents**: At large ℓ, use exponents spanning the full range [0, ℓ).

## 4. Results

### 4.1 Campaign Statistics

| Metric | Value |
|--------|-------|
| Total iterations | 500 |
| Successful evaluations | 352 |
| Total code evaluations | 18,588 |
| Evaluations with k > 0, d ≥ 5 | 15,152 |
| Tuple-unique codes | 720 |
| **BLISS-unique codes** | **368** |
| Distinct (n, k, d) parameter sets | 78 |
| Distinct CSS base pairs (A, B) | 173 (202 counting per-lattice) |
| EXACT (distance proven optimal) | 251 |
| TRUSTED/PARTIAL upper bounds | 117 (110 TRUSTED, 7 PARTIAL with $d/\sqrt{n} \geq 1.5$) |
| Deep MILP verified | 149 catalog entries (63 exact outcomes, 33 distance corrections) |
| Codes with FOM ≥ 8.0 | 53 |
| Codes with FOM ≥ 12.0 | 25 |
| Best combined score | 111.34 |
| Wall clock time | ~11 hours (evolution) + separate deep MILP batches on a 64-core server |

The discovery funnel shows that evolutionary search explored the space broadly (18,588 evaluations) but most candidates are re-evaluations or isomorphic relabelings of a smaller set of 368 structurally distinct codes. See §3.4 for the per-lattice breakdown.

### 4.2 Best Codes per Lattice

| Lattice | Best code | FOM | d/√n | Trust | Method | Codes found |
|---------|-----------|-----|------|-------|--------|-------------|
| (3,6) n=36 | [[36,4,6]] | 4.00 | 1.000 | EXACT | exact_w6 | 2 |
| (6,3) n=36 | [[36,4,6]] | 4.00 | 1.000 | EXACT | exact_w6 | 45 |
| (6,6) n=72 | [[72,12,6]] | 6.00 | 0.707 | EXACT | exact_w6 | 34 |
| (9,6) n=108 | [[108,8,10]] | 7.41 | 0.962 | EXACT | deep_milp | 107 |
| (12,6) n=144 | [[144,12,12]] | 12.00 | 1.000 | EXACT | deep_milp_exact | 66 |
| (15,6) n=180 | [[180,6,20]] | 13.33 | 1.491 | TRUSTED | milp+bposd | 64 |
| (30,6) n=360 | [[360,12,24]] | 19.20 | 1.265 | TRUSTED | milp+bposd | 50 |

### 4.3 Deep MILP Verification Results

All 149 deep-MILP records across (6,6), (9,6), (12,6), (15,6), and (30,6) underwent verification: all 2k logicals per code submitted as independent ILPs to a process pool with up to 60 workers and up to 14400s per-logical timeout. Verification was performed on a server with 64 cores.

**Summary by lattice:**

| Lattice | Codes | Logicals | EXACT | Partial | Corrections | Runtime |
|---------|-------|----------|-------|---------|-------------|---------|
| (6,6) n=72 | 5 | 40 | 5 | 0 | 0 | parallel batch |
| (9,6) n=108 | 42 | 476 | 33 | 9 | 13 | parallel batch |
| (12,6) n=144 | 36 | 692 | 18 | 18 | 2 | parallel batch |
| (15,6) n=180 | 36 | 416 | 6 | 30 | 8 | parallel batch |
| (30,6) n=360 | 30 | 536 | 1 | 29 | 10 | parallel batch |
| **Total** | **149** | **2,160** | **63** | **86** | **33** | **separate parallel runs** |

**Distance corrections (33 total):**

Deep MILP revealed that BP-OSD systematically overestimated distance, particularly at larger block lengths. All corrections are downward — the MILP found lower-weight logical operators than BP-OSD detected.

| Lattice | Correction | Count |
|---------|-----------|-------|
| (9,6) | k=4: d=12→10 | 1 |
| (9,6) | k=4: d=14→12 | 1 |
| (9,6) | k=6: d=12→10 | 1 |
| (9,6) | k=8: d=12→10 | 10 |
| (12,6) | k=4: d=14→12 | 1 |
| (12,6) | k=8: d=10→8 | 1 |
| (15,6) | k=4: d=14→12 | 2 |
| (15,6) | k=4: d=16→14 | 2 |
| (15,6) | k=4: d=18→12 | 1 |
| (15,6) | k=4: d=18→16 | 1 |
| (15,6) | k=8: d=12→10 | 2 |
| (30,6) | k=4: d=26→20 | 2 |
| (30,6) | k=4: d=26→24 | 1 |
| (30,6) | k=8: d=18→16 | 2 |
| (30,6) | k=8: d=20→16 | 1 |
| (30,6) | k=10: d=18→16 | 2 |
| (30,6) | k=10: d=20→16 | 1 |
| (30,6) | k=10: d=24→16 | 1 |

**Key findings:**

- **BP-OSD overestimates distance systematically.** The overestimation worsens with block length: at n=108, 13 of 42 deep-MILP records were corrected; at n=360, 10 of 30 codes were corrected. The largest correction was [[360,10,24]]→d=16 (FOM dropped from 16.0 to 7.1).
- **HiGHS struggles at n ≥ 180.** At (15,6), 6 of 36 codes achieved EXACT status. At (30,6), only 1 code was proven EXACT. The remaining codes have valid MILP incumbent upper bounds but not proven optimal.
- **MILP incumbents are tighter than BP-OSD.** Even without proving optimality, the MILP solver found incumbents (feasible solutions) with lower weight than BP-OSD in 33 cases.
- **[[144,12,12]] confirmed EXACT for seven representatives.** 7 of 14 codes in this family had all 24 logicals proven optimal at d=12, confirming this is a genuine Gross-code-matching non-CSS parameter set; the other 7 remain trusted upper bounds.

### 4.4 Top FOM Codes

After deep MILP corrections, the highest-FOM codes in the catalog are:

| Code | FOM | Trust | d/√n | Count |
|------|-----|-------|------|-------|
| [[360,12,24]] | 19.20 | TRUSTED | 1.265 | 1 |
| [[360,10,22]] | 13.44 | TRUSTED | 1.160 | 1 |
| [[360,12,20]] | 13.33 | TRUSTED | 1.054 | 1 |
| [[180,6,20]] | 13.33 | TRUSTED | 1.491 | 1 |
| [[144,12,12]] | 12.00 | EXACT/TRUSTED | 1.000 | 14 |
| [[180,8,16]] | 11.38 | TRUSTED | 1.193 | 1 |
| [[360,10,20]] | 11.11 | TRUSTED | 1.054 | 5 |

Twenty-five codes match or exceed the Gross code [[144,12,12]] FOM of 12.0 when PARTIAL upper bounds are included; among EXACT/TRUSTED rows, 18 codes meet or exceed FOM 12.0 and 46 have FOM ≥ 8.0.

Note: The original top code [[360,10,24]] (FOM=16.0) was corrected to d=16 (FOM=7.1) by deep MILP. The [[360,12,24]], [[360,12,20]], and [[360,10,20]] rows are upper bounds; all TRUSTED distances may still tighten downward under deeper exact verification.

### 4.5 High-Rate Codes

The evolution discovered several high-rate non-CSS codes with fully verified distance:

| Code | Rate k/n | FOM | Trust | Method |
|------|---------|-----|-------|--------|
| [[360,60,6]] | 1/6 = 0.167 | 6.00 | EXACT | milp_exact |
| [[360,50,6]] | 1/7.2 = 0.139 | 5.00 | EXACT | milp_exact |
| [[144,24,6]] | 1/6 = 0.167 | 6.00 | EXACT | exact_w6 |
| [[180,20,6]] | 1/9 = 0.111 | 4.00 | EXACT | exact_w6 |
| [[360,24,8]] | 1/15 = 0.067 | 4.27 | EXACT | milp_exact |

The [[360,60,6]] code encodes 60 logical qubits in 360 physical qubits with verified distance 6 — a rate of 1/6, which is unusually high for a quantum LDPC code.

### 4.6 Complete Code Catalog

368 BLISS-unique codes across 7 lattices, collapsed into 78 parameter sets (n, k, d).
The **Codes** column shows how many structurally distinct codes (non-isomorphic Tanner graphs)
share the same parameters — each has different polynomial tuples (A, B, C, D).
**Bases** counts distinct CSS base pairs (A, B); the remaining variation comes from C, D perturbations.
Full polynomial specifications for all 368 codes are in §5.

The parameter-set summary below is generated from `results/campaign7_publication_merged.jsonl` after the deep-MILP merge. Distances without EXACT status remain upper bounds even when shown without a `<=` marker in the code label.

#### n = 36 — 47 codes, 2 parameter sets

| Code | FOM | d/√n | Codes | EXACT | TRUSTED | PARTIAL | Bases |
|------|-----|------|-------|-------|---------|---------|-------|
| [[36,4,6]] | 4.00 | 1.000 | 45 | 45 | 0 | 0 | 3 |
| [[36,2,6]] | 2.00 | 1.000 | 2 | 2 | 0 | 0 | 1 |

#### n = 72 — 34 codes, 4 parameter sets

| Code | FOM | d/√n | Codes | EXACT | TRUSTED | PARTIAL | Bases |
|------|-----|------|-------|-------|---------|---------|-------|
| [[72,12,6]] | 6.00 | 0.707 | 9 | 9 | 0 | 0 | 3 |
| [[72,10,6]] | 5.00 | 0.707 | 2 | 2 | 0 | 0 | 1 |
| [[72,4,8]] | 3.56 | 0.943 | 20 | 20 | 0 | 0 | 2 |
| [[72,4,6]] | 2.00 | 0.707 | 3 | 3 | 0 | 0 | 3 |

#### n = 108 — 107 codes, 14 parameter sets

| Code | FOM | d/√n | Codes | EXACT | TRUSTED | PARTIAL | Bases |
|------|-----|------|-------|-------|---------|---------|-------|
| [[108,8,10]] | 7.41 | 0.962 | 18 | 18 | 0 | 0 | 7 |
| [[108,6,10]] | 5.56 | 0.962 | 8 | 8 | 0 | 0 | 5 |
| [[108,4,12]] | 5.33 | 1.155 | 4 | 3 | 1 | 0 | 4 |
| [[108,8,8]] | 4.74 | 0.770 | 1 | 1 | 0 | 0 | 1 |
| [[108,12,6]] | 4.00 | 0.577 | 1 | 1 | 0 | 0 | 1 |
| [[108,4,10]] | 3.70 | 0.962 | 11 | 10 | 1 | 0 | 10 |
| [[108,6,8]] | 3.56 | 0.770 | 25 | 20 | 5 | 0 | 13 |
| [[108,2,12]] | 2.67 | 1.155 | 2 | 2 | 0 | 0 | 2 |
| [[108,4,8]] | 2.37 | 0.770 | 9 | 9 | 0 | 0 | 9 |
| [[108,6,6]] | 2.00 | 0.577 | 2 | 2 | 0 | 0 | 2 |
| [[108,2,10]] | 1.85 | 0.962 | 8 | 6 | 2 | 0 | 5 |
| [[108,4,6]] | 1.33 | 0.577 | 15 | 15 | 0 | 0 | 14 |
| [[108,2,8]] | 1.19 | 0.770 | 1 | 1 | 0 | 0 | 1 |
| [[108,2,6]] | 0.67 | 0.577 | 2 | 2 | 0 | 0 | 2 |

#### n = 144 — 66 codes, 12 parameter sets

| Code | FOM | d/√n | Codes | EXACT | TRUSTED | PARTIAL | Bases |
|------|-----|------|-------|-------|---------|---------|-------|
| [[144,12,12]] | 12.00 | 1.000 | 14 | 7 | 7 | 0 | 9 |
| [[144,10,12]] | 10.00 | 1.000 | 9 | 4 | 5 | 0 | 6 |
| [[144,24,6]] | 6.00 | 0.500 | 1 | 1 | 0 | 0 | 1 |
| [[144,8,10]] | 5.56 | 0.833 | 1 | 1 | 0 | 0 | 1 |
| [[144,12,8]] | 5.33 | 0.667 | 2 | 2 | 0 | 0 | 2 |
| [[144,10,8]] | 4.44 | 0.667 | 10 | 10 | 0 | 0 | 4 |
| [[144,16,6]] | 4.00 | 0.500 | 3 | 3 | 0 | 0 | 3 |
| [[144,4,12]] | 4.00 | 1.000 | 7 | 2 | 5 | 0 | 7 |
| [[144,8,8]] | 3.56 | 0.667 | 4 | 3 | 1 | 0 | 3 |
| [[144,12,6]] | 3.00 | 0.500 | 5 | 5 | 0 | 0 | 3 |
| [[144,4,10]] | 2.78 | 0.833 | 2 | 2 | 0 | 0 | 2 |
| [[144,8,6]] | 2.00 | 0.500 | 8 | 8 | 0 | 0 | 3 |

#### n = 180 — 64 codes, 21 parameter sets

| Code | FOM | d/√n | Codes | EXACT | TRUSTED | PARTIAL | Bases |
|------|-----|------|-------|-------|---------|---------|-------|
| [[180,6,21]] | 14.70 | 1.565 | 1 | 0 | 0 | 1 | 1 |
| [[180,6,20]] | 13.33 | 1.491 | 1 | 0 | 1 | 0 | 1 |
| [[180,8,16]] | 11.38 | 1.193 | 1 | 0 | 1 | 0 | 1 |
| [[180,6,18]] | 10.80 | 1.342 | 3 | 0 | 3 | 0 | 3 |
| [[180,8,14]] | 8.71 | 1.043 | 1 | 0 | 1 | 0 | 1 |
| [[180,6,16]] | 8.53 | 1.193 | 6 | 0 | 6 | 0 | 5 |
| [[180,6,14]] | 6.53 | 1.043 | 9 | 0 | 9 | 0 | 7 |
| [[180,8,12]] | 6.40 | 0.894 | 8 | 3 | 5 | 0 | 5 |
| [[180,4,16]] | 5.69 | 1.193 | 2 | 0 | 2 | 0 | 2 |
| [[180,6,12]] | 4.80 | 0.894 | 1 | 0 | 1 | 0 | 1 |
| [[180,8,10]] | 4.44 | 0.745 | 8 | 1 | 7 | 0 | 5 |
| [[180,4,14]] | 4.36 | 1.043 | 5 | 0 | 5 | 0 | 5 |
| [[180,12,8]] | 4.27 | 0.596 | 2 | 2 | 0 | 0 | 2 |
| [[180,20,6]] | 4.00 | 0.447 | 1 | 1 | 0 | 0 | 1 |
| [[180,4,12]] | 3.20 | 0.894 | 3 | 0 | 3 | 0 | 3 |
| [[180,12,6]] | 2.40 | 0.447 | 1 | 1 | 0 | 0 | 1 |
| [[180,4,10]] | 2.22 | 0.745 | 5 | 2 | 3 | 0 | 3 |
| [[180,2,14]] | 2.18 | 1.043 | 1 | 0 | 1 | 0 | 1 |
| [[180,2,12]] | 1.60 | 0.894 | 3 | 2 | 1 | 0 | 3 |
| [[180,2,10]] | 1.11 | 0.745 | 1 | 1 | 0 | 0 | 1 |
| [[180,4,6]] | 0.80 | 0.447 | 1 | 1 | 0 | 0 | 1 |

#### n = 360 — 50 codes, 25 parameter sets

| Code | FOM | d/√n | Codes | EXACT | TRUSTED | PARTIAL | Bases |
|------|-----|------|-------|-------|---------|---------|-------|
| [[360,10,40]] | 44.44 | 2.108 | 1 | 0 | 0 | 1 | 1 |
| [[360,10,32]] | 28.44 | 1.687 | 2 | 0 | 0 | 2 | 2 |
| [[360,10,30]] | 25.00 | 1.581 | 3 | 0 | 0 | 3 | 3 |
| [[360,12,24]] | 19.20 | 1.265 | 1 | 0 | 1 | 0 | 1 |
| [[360,10,22]] | 13.44 | 1.160 | 1 | 0 | 1 | 0 | 1 |
| [[360,12,20]] | 13.33 | 1.054 | 1 | 0 | 1 | 0 | 1 |
| [[360,10,20]] | 11.11 | 1.054 | 5 | 0 | 5 | 0 | 4 |
| [[360,10,18]] | 9.00 | 0.949 | 2 | 0 | 2 | 0 | 2 |
| [[360,12,16]] | 8.53 | 0.843 | 1 | 0 | 1 | 0 | 1 |
| [[360,8,18]] | 7.20 | 0.949 | 1 | 0 | 1 | 0 | 1 |
| [[360,10,16]] | 7.11 | 0.843 | 6 | 0 | 6 | 0 | 5 |
| [[360,16,12]] | 6.40 | 0.632 | 1 | 0 | 1 | 0 | 1 |
| [[360,4,24]] | 6.40 | 1.265 | 1 | 0 | 1 | 0 | 1 |
| [[360,60,6]] | 6.00 | 0.316 | 1 | 1 | 0 | 0 | 1 |
| [[360,8,16]] | 5.69 | 0.843 | 7 | 0 | 7 | 0 | 4 |
| [[360,50,6]] | 5.00 | 0.316 | 2 | 2 | 0 | 0 | 2 |
| [[360,12,12]] | 4.80 | 0.632 | 1 | 0 | 1 | 0 | 1 |
| [[360,4,20]] | 4.44 | 1.054 | 3 | 0 | 3 | 0 | 3 |
| [[360,24,8]] | 4.27 | 0.422 | 1 | 1 | 0 | 0 | 1 |
| [[360,40,6]] | 4.00 | 0.316 | 1 | 1 | 0 | 0 | 1 |
| [[360,20,8]] | 3.56 | 0.422 | 1 | 1 | 0 | 0 | 1 |
| [[360,4,16]] | 2.84 | 0.843 | 1 | 0 | 1 | 0 | 1 |
| [[360,8,10]] | 2.22 | 0.527 | 2 | 2 | 0 | 0 | 2 |
| [[360,12,6]] | 1.20 | 0.316 | 1 | 0 | 1 | 0 | 1 |
| [[360,4,10]] | 1.11 | 0.527 | 3 | 2 | 1 | 0 | 3 |

### 4.7 Structural Patterns and Code Families

Analysis of the 368 codes reveals several structural regularities.

#### 4.7.1 Base Pair Structure

All 368 codes use **trinomial** bases (|A| = |B| = 3). There are 173 globally distinct (A, B) bases, of which **15 span multiple lattices** and account for 168 codes (46% of the catalog). The remaining 158 bases are lattice-specific.

**B almost always contains the identity monomial (0,0).** 218 of 368 codes (59%) have B containing the constant term, typically as B = 1 + (two further terms). A almost never contains the identity (only 1 code). This asymmetry reflects the PBB construction: B contributes the transpose block B^T, and including the identity keeps the stabilizer structure well-conditioned.

**Exponent types.** The dominant pattern is **all-mixed** monomials (every term has both x and y exponents nonzero): 222 codes (60%) have all-mixed A. The classic x/y-swap pattern (A has 2+ pure-y terms, B has 2+ pure-x terms) accounts for only 53 codes (14%). The evolution moved beyond the swap structure found in known CSS BB codes.

#### 4.7.2 Perturbation (C, D) Patterns

The optimal perturbation size is **(|C| = 2, |D| = 2)**, accounting for 207 codes (56%) with the highest average FOM (6.40). Minimal perturbations (|C| = |D| = 1) underperform (62 codes, avg FOM 3.31), and large perturbations (|C| + |D| ≥ 6) cap distance at d = 8. The sweet spot is adding exactly 2 terms to each of C and D.

A striking finding: **C's x-exponents are drawn from A's x-exponents** in the top-10 FOM codes (100% of them). Over the full catalog this holds in 55% of codes. The perturbation appears to work best as a "y-rotation" of the base's x-skeleton — reusing A's x-structure while shifting y-exponents.

#### 4.7.3 Cross-Lattice Families

**Family α: A = y + x³y² + x⁴y, B = x³y + x⁴ + x⁴y** (59 codes, 5 lattices)

The most prolific family. At (6,3) it produces many [[36,4,6]] codes; at (6,6), the reconciled publication catalog contains [[72,4,8]] representatives rather than the earlier draft's higher-distance claim. The family trades rate for distance and does not dominate the n=72 PBB frontier.

**Family β: A = xy² + x⁴y³ + x⁴y⁴, B = 1 + xy⁵ + x⁵y⁴** (19 codes, 5 lattices)

The quality leader. This base spans from (6,6) to (30,6), producing [[72,12,6]] (CSS-equivalent at (6,6)), [[108,8,10]] (deep MILP exact), and [[360,10,20]] (FOM=11.11 at (30,6)). The (30,6) result remains useful, though the reconciled top trusted PBB row is [[360,12,24]].

**Family γ: A = y + y² + x³, B = y³ + x + x² (and variant B = y³ + x² + x⁴)** (6 codes, 3 lattices)

The only top-performing pure x/y-swap family. Achieves [[144,12,12]] FOM=12.00 at (12,6) — the highest confirmed FOM at that lattice. Distance saturates at d=12 across lattices while k grows from 8 at (9,6) to 12 at (12,6), suggesting a distance ceiling for the swap structure.

**Family δ: A = y² + y⁴ + x³, B = y³ + x² + x⁴** (5 codes, 3 lattices)

All codes use minimal perturbations (|C| = |D| = 1). Achieves [[144,12,12]] at (12,6). This is the only prolific family with exclusively single-monomial C, D.

#### 4.7.4 High-Rate Family

The high-rate codes ([[360,60,6]], [[360,50,6]], [[144,24,6]]) share a consistent exponent template: A = (a,2) + (b,3) + (b,4), B = (0,0) + (a,5) + (c,4). The y-exponents {2,3,4} in A and {0,5,4} in B are fixed; only x-exponents vary. This pattern produces high k with low d=6, suggesting a code family with growing rate but distance bounded by the perturbation.

#### 4.7.5 Parameter Landscape

The k distribution is skewed: k=4 accounts for 140 codes (38%), while k ≥ 12 accounts for 51 codes (14%). This reflects the search's scoring function, which rewards FOM = k·d²/n — at moderate k, the d² term dominates.

30 of 78 parameter triples are singletons (exactly one code). These isolated codes may represent unexplored frontiers where additional search could find sibling codes.

## 5. Polynomial Specifications

Complete polynomial tuples for all 368 BLISS-unique non-CSS PBB codes are
available in machine-readable form at
`results/campaign7_publication_merged.jsonl`.  Each record carries the
lattice $(\ell, m)$, polynomials $A, B, C, D$ as exponent lists, the BLISS
canonical hash, MILP and BP-OSD distance data, trust level (EXACT /
TRUSTED / PARTIAL), figure of merit, and verification metadata.

The catalog is also rendered as LaTeX tables (one per lattice) in
`paper/2606.02418/pbb_catalog_tables.tex`, generated by
`tests/generate_pbb_catalog_rows.py`.


## 6. Computational Resources

| Resource | Specification |
|----------|-------------|
| Server | 64 cores, 251 GB RAM, Linux (RHEL 9) |
| Python | 3.9.25 |
| Key packages | qldpc 0.2.6, scipy 1.17.0, numpy 2.3.5, igraph 1.0.0, HiGHS 1.13.1 |
| Evolution runtime | ~11 hours (500 iterations, 352 productive) |
| LLM budget | ~$100 across 3 models |
| BLISS dedup | ~5 min (720 codes, 8 workers) |
| Publication verification | ~4 hours (368 codes, 8 workers) |
| Deep MILP verification | Separate parallel batches on 64-core server (149 catalog entries, 2,160 logicals, up to 60 workers, up to 14400s/logical) |

### 6.1 Memory Profile

| Lattice | Weight-6 hash RAM | MILP RAM/logical | Peak concurrent |
|---------|-------------------|-----------------|-----------------|
| n=72 | ~700 MB | ~300 MB | 16 workers |
| n=108 | ~2 GB | ~500 MB | 16 workers |
| n=144 | ~5.5 GB | ~800 MB | 16 workers |
| n=180 | ~11 GB | ~1.2 GB | 16 workers |
| n=360 | ~89 GB (w6, skipped) | ~2 GB | 16 workers |

## 7. Limitations and Future Work

### 7.1 Distance Verification Gap

After verification, 117 of 368 codes (32%) remain TRUSTED or PARTIAL — their distances are valid upper bounds but not proven optimal. These are concentrated at larger lattices: 50 at (15,6) and 40 at (30,6), where HiGHS struggles to prove optimality within the per-logical time budget. The deep MILP campaign produced 63 exact outcomes and found 33 distance corrections, but the solver's limitations at n ≥ 180 (720 binary variables per ILP for n=360) leave a gap. Based on the 22% correction rate observed in deep MILP (33 of 149 catalog entries), some remaining TRUSTED distances — particularly at (15,6) and (30,6) — may be overestimates. Readers should treat all TRUSTED and PARTIAL distances as upper bounds.

**Potential improvements:**
1. Alternative ILP solvers may outperform HiGHS on these instances
2. Symmetry-breaking constraints or warm-starting from incumbent solutions
3. Cross-validation with GAP/QDistRnd exact distance computation for selected codes

### 7.2 Search Space Coverage

The current search explores PBB codes with trinomial A,B bases (3 terms each) and 1–2 term C,D perturbations. Higher-term polynomials, non-abelian groups, or alternative constructions (e.g., mirror codes from Khesin & Lu) remain unexplored.

### 7.3 Circuit-Level Analysis

This campaign focused on code parameters (n, k, d, FOM). Fault-tolerant syndrome extraction circuits, logical error rate simulations, and threshold estimates are left for future work. The mirror codes paper (arXiv:2603.05496) provides initial circuit constructions for weight-6 non-CSS codes that could be applied to our codes.

## 8. Files and Reproducibility

### 8.1 Code Files (tracked in repository)

| File | Description |
|------|-------------|
| `evaluation/pbb_code.py` | PBB code construction and GF(2) utilities |
| `evaluation/distance_bposd_noncss.py` | Hash-based exact distance + BP-OSD for non-CSS |
| `evaluation/distance_milp.py` | MILP symplectic distance formulation |
| `evolve/seed_solution_noncss.py` | Seed solution with 7 strategies |
| `evolve/openevolve_evaluator_noncss.py` | 2-stage evaluation cascade |
| `evolve/_noncss_distance_worker.py` | Adaptive distance pipeline worker |
| `evolve/config_noncss.yaml` | Evolution configuration |
| `evolve/prompt_context_noncss.md` | Domain knowledge for LLM mutations |
| `scripts/verify_publication.py` | Publication-quality verification with BLISS dedup |
| `scripts/verify_from_scratch.py` | Full pipeline: load raw → tuple dedup → BLISS dedup → verify missing |
| `scripts/verify_deep_milp.py` | Deep MILP per-logical parallel verification |
| `scripts/verify_deep_milp_remote_standalone.py` | Self-contained deep MILP for (15,6) n=180 |
| `scripts/verify_deep_milp_remote_n360.py` | Self-contained deep MILP for (30,6) n=360 |

### 8.2 Result Files

| File | Description |
|------|-------------|
| `results/campaign7_publication_merged.jsonl` | 368 verified codes (BLISS-deduplicated, all corrections applied) |
| `results/campaign7_deep_milp.jsonl` | 149 deep MILP verification records (merged from all servers) |
| `results/campaign7_dedup.jsonl` | 368 BLISS-deduplicated codes with group sizes (intermediate) |
| `results/evolution/campaign7/all_codes_noncss.jsonl` | 18,588 raw evolution evaluations (main run) |
| `results/evolution/campaign7d/all_codes_noncss.jsonl` | 24,014 raw evolution evaluations (campaign7d run) |
| `results/evolution/campaign7e/all_codes_noncss.jsonl` | 23,905 raw evolution evaluations (campaign7e run) |
| `results/evolution/campaign7/best/` | Best evolved program and metadata |
| `results/evolution/campaign7d/best/` | Best evolved program from campaign7d |
| `results/evolution/campaign7e/best/` | Best evolved program from campaign7e |
| `results/campaign7_report.md` | This report |

**Note on evolution runs:** The 368-code catalog was produced from the main campaign7 run (18,588 evaluations). Campaign7d and 7e were supplementary runs that explored the same lattice set; their raw evaluations are included for completeness but independent verification showed their distance estimates were unreliable at (6,6) (BP-OSD overestimation), and no additional codes beyond those in the main run survived publication-quality verification.

### 8.3 Verification Protocol

Each code undergoes:
1. **Construction**: Build PBB code, verify commutativity algebraically
2. **k computation**: GF(2) rank of full stabilizer matrix
3. **Hash-exact distance**: Enumerate symplectic weights ≤ 6 (or ≤ 4 at n > 216)
4. **MILP distance**: Per-logical ILP with publication budgets (300s/logical for n ≤ 108, 600s for n ≤ 180, 900s for n ≤ 360)
5. **BP-OSD**: 1000 trials × 3 seeds as fallback upper bound
6. **Deep MILP**: All 2k logicals per code, up to 14400s/logical timeout, up to 60 workers (for 149 deep-MILP records)
7. **Trust assignment**: EXACT if all logicals solved to optimality, TRUSTED if valid upper bound from multiple methods

## References

- Bravyi, S. et al. "High-threshold and low-overhead fault-tolerant quantum memory." Nature 627, 778–782 (2024); arXiv:2308.07915. [Source of the [[144,12,12]] "Gross code"]
- Khesin, A.B. & Lu, J.Z. "Mirror codes: High-threshold quantum LDPC codes beyond the CSS regime." arXiv:2603.05496 (March 2026).
- Panteleev, P. & Kalachev, G. "Asymptotically good quantum and locally testable classical LDPC codes." STOC 2022.
- Roffe, J. et al. "Bias-tailored quantum LDPC codes." arXiv:2202.01702 (2022). [BP-OSD decoder]
