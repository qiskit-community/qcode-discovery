# Weight-5 Quantum LDPC Codes

## A Survey of Bivariate Bicycle, Group-Algebra, and Balanced-Product Constructions

*Verified edition — August 2026*

---

## Contents

1. [Scope, method, and confidence conventions](#1-scope-method-and-confidence-conventions)
2. [Executive summary](#2-executive-summary)
3. [What "weight-5" means: three distinct notions](#3-what-weight-5-means-three-distinct-notions)
4. [Bivariate bicycle codes and the route to weight 5](#4-bivariate-bicycle-codes-and-the-route-to-weight-5)
5. [Group algebra codes: 2BGA and generalized bicycle](#5-group-algebra-codes-2bga-and-generalized-bicycle)
6. [Balanced product codes](#6-balanced-product-codes)
7. [Adjacent low-weight families](#7-adjacent-low-weight-families)
8. [Decoding](#8-decoding)
9. [Bounds and weight-reduction theory](#9-bounds-and-weight-reduction-theory)
10. [Hardware and experimental status](#10-hardware-and-experimental-status)
11. [Comparative parameter table](#11-comparative-parameter-table)
12. [State of the art and open problems](#12-state-of-the-art-and-open-problems)
13. [Bibliography with verification status](#13-bibliography-with-verification-status)
14. [Limitations of this survey](#14-limitations-of-this-survey)

---

## 1. Scope, method, and confidence conventions

This survey covers quantum low-density parity-check (qLDPC) codes with an emphasis on check weight 5, across three mathematically related families: bivariate bicycle (BB) codes, two-block group algebra (2BGA) and generalized bicycle (GB) codes, and balanced-product codes. Coverage runs roughly 2020 through July 2026, with adjacent low-weight families, decoders, bounds, and hardware results included where they bear on the weight-5 question.

Every citation below was subjected to an adversarial verification pass: the arXiv identifier, title, author list, and publication venue were checked against primary records, and quantitative claims were checked against abstracts or full text where budget allowed. Because a survey of this kind is only as good as its weakest citation, confidence is marked inline rather than left implicit:

| Mark | Meaning |
|:---:|:---|
| **✓** | Identifier, title, authors, and venue confirmed against primary records; quoted claims confirmed against the abstract or full text. |
| **~** | Paper confirmed to exist with correct metadata, but the specific number or parameter attributed to it was taken from the abstract, a listing page, or secondary coverage and was **not** independently reconfirmed against the full text. |
| **⚠** | Claim could not be confirmed at all and should be re-checked against the source before being relied on. |

Where an earlier draft of this survey was wrong, the correction is stated explicitly rather than silently applied, so that readers working from the earlier version can reconcile the two.

---

## 2. Executive summary

**The weight-5 landscape is genuinely thin, and that is the headline finding rather than a gap in coverage.** Despite a very active 2024–2026 literature on low-weight qLDPC codes, there is essentially one well-characterized family of genuinely weight-5 stabilizer codes: the multivariate (trivariate) bicycle codes of Voss, Xian, Haug, and Bharti. Systematic searches over 2BGA, coprime BB, balanced-product, and tile-code spaces have concentrated at weight 4, 6, 8, and 9, largely bypassing weight 5. Anyone expecting a rich weight-5 catalogue comparable to the weight-6 BB literature will not find one.

**Three things support the weight-5 target theoretically.** Hastings proved that *any* qLDPC code can be converted into one whose stabilizers all have weight at most 5, at the cost of only a constant-factor increase in physical qubits and a constant-factor reduction in distance — weight 5 is a universal target, not an arbitrary one. Second, for abelian 2BGA codes of weight *w*, geometric locality in *D* = *w* − 2 dimensions gives a Bravyi–Terhal-type ceiling *d* = O(*n*^(2/3)) at weight 5. Third, the Bravyi–Poulin–Terhal bound *kd*² ≤ O(*n*) explains why any high-rate weight-5 code must be non-planar: the long-range check is structurally unavoidable.

**Practical routes to "weight-5 hardware" split three ways, and conflating them overstates maturity.** Genuinely weight-5 stabilizers (trivariate bicycle codes), degree-5 qubit connectivity with higher-weight stabilizers (Shaw–Terhal morphing circuits, which reach degree 5 with weight-*9* stabilizers), and weight-4 gauge checks inferring higher-weight stabilizers (subsystem BB codes) are three different engineering propositions. The literature frequently blurs them.

**The most important corrections to the earlier draft** were metadata errors rather than conceptual ones: a wrong paper title, a fabricated author on the localized-statistics-decoding paper, a misattributed radial-codes citation, a wrong software repository, a wrong rate-distance bound in the IBM search results, and a threshold figure that needed a version caveat. The load-bearing conceptual claims — Hastings weight-5, the BB/GB/2BGA relationships, the distance bounds, and the scarcity of weight-5 balanced-product results — all survived verification intact.

---

## 3. What "weight-5" means: three distinct notions

This distinction is the single most important framing in the survey, and it was explicitly validated during verification. Three quantities get called "weight 5," and they impose different hardware demands:

**(a) Stabilizer support** — the number of qubits in a stabilizer generator. This is the standard meaning of check weight and the one that governs measurement-circuit depth and the propagation of measurement errors. Trivariate bicycle codes achieve genuine weight-5 support.

**(b) Qubit degree / connectivity** — the number of couplers each physical qubit needs. This is often the binding constraint on superconducting hardware, and it is *not* the same as stabilizer support. Shaw and Terhal's morphing circuits reach **degree-5 connectivity while using weight-9 stabilizers** [✓ arXiv:2407.16336], which is a direct demonstration that the two notions come apart.

**(c) Gauge check weight** — in subsystem codes, the weight of the gauge operators actually measured, with higher-weight stabilizer syndromes inferred by multiplying gauge outcomes. Liang and Chen's subsystem BB codes measure **weight-4 gauge operators** while realizing BB logical structure [✓ arXiv:2605.04151].

A code advertised as "low weight" in sense (b) or (c) may be considerably harder to build than one that is low weight in sense (a), or considerably easier — the comparison is not well-posed until the sense is fixed. Throughout this survey, unqualified "weight" means (a), and (b) and (c) are named explicitly.

---

## 4. Bivariate bicycle codes and the route to weight 5

### 4.1 The weight-6 baseline

**Bravyi, Cross, Gambetta, Maslov, Rall, and Yoder**, *High-threshold and low-overhead fault-tolerant quantum memory*, Nature **627**, 778 (2024), arXiv:2308.07915 [✓].

BB codes are CSS codes over the ring *R* = 𝔽₂[*x*,*y*]/(*x*^ℓ − 1, *y*^m − 1), with *H_X* = [*A* | *B*] and *H_Z* = [*B*ᵀ | *A*ᵀ]. Taking *A* and *B* as weight-3 polynomials yields weight-6 stabilizers and degree-6 qubit connectivity.

The flagship **[[144,12,12]] "gross code"** uses 288 physical qubits (144 data + 144 check) and a depth-7 syndrome-extraction schedule.

> **Correction (threshold).** The published Nature abstract reports a threshold of **0.7%** for the standard circuit-based noise model, "on par with the surface code." An earlier draft of this survey and some pre-revision arXiv versions cite 0.8%; the arXiv record notes explicitly that numerical results were revised after a simulation-software bug was fixed. **Use 0.7%**, and treat 0.8% as a superseded pre-revision value [✓].

The paper's own overhead comparison — roughly 3,000 physical qubits for a surface-code encoding of 12 logical qubits at comparable protection, versus 288 — gives the widely quoted better-than-10:1 saving [✓].

### 4.2 Multivariate / trivariate bicycle codes: the one real weight-5 family

**Voss, Sim Jian Xian, Haug, and Bharti**, *Multivariate Bicycle Codes*, arXiv:2406.19151, Phys. Rev. A **111**, L060401 (2025) [✓].

> *Note on title history:* v1 of this preprint was titled "Trivariate Bicycle Codes"; the published version is "Multivariate Bicycle Codes." Both names appear in citing literature.

The construction extends the group algebra to 𝔽₂[Z_{ℓ₁} × ⋯ × Z_{ℓ_r}], with the *r* = 3 (trivariate) case the practical focus. The abstract's central claim is confirmed verbatim: weight-5 codes are offered as "more amenable to near-term experimental setups," and the flagship result is that "we can encode 4 logical qubits with distance 5 into 60 physical qubits using weight-5 check measurements of circuit depth 7, while a surface code with these parameters requires 200 physical qubits" [✓].

Two points of convention deserve care, because the earlier draft handled them loosely:

- The code is **[[30,4,5]]** in the standard convention: *n* = 30 **data** qubits, rate 4/30 ≈ 0.133. The **60** figure in the abstract is 30 data + 30 syndrome/ancilla qubits, i.e. total qubits during syndrome extraction.
- The correct headline comparison is therefore **3.3× fewer total physical qubits** (60 vs 200), not a 3.3× rate improvement. On data qubits alone the rate advantage over a *kd*²-scaling surface code is larger; the earlier draft's "improves the rate by ~3.3×" conflated the two.

The depth-7 circuit is conjectured distance-preserving (*d*_circ ≤ 5, believed but not proven tight), against conventional depth-10 [✓]. Bi-planarity — Tanner-graph thickness 2 — is proven for all TB codes of weight ≤ 6 under the appropriate *A*/*B* split [✓]. Fault-tolerant logical circuits are constructed from automorphism-group transversal operations composed with SWAPs [~].

Best TB codes by weight, as reported in the paper [~ — reported from the paper, not independently re-derived]:

| Weight | Code | Rate |
|:---:|:---|:---|
| 4 | [[144,2,12]] | 1/72 |
| 5 | **[[30,4,5]]** | 2/15 |
| 6 | [[30,6,4]] | 1/5 |
| 7 | [[30,4,5]] | 2/15 |

Pseudo-thresholds for the [[30,4,5]] under circuit-level depolarizing noise are reported in the range ~1.2 × 10⁻³ to 4.0 × 10⁻³ across observables, with noise suppression comparable to the weight-6 [[72,12,6]] and [[72,8,6]] BB codes at less than half the qubit count [~].

> **⚠ To verify before relying on it:** the description of the [[30,4,5]] stabilizers as "four local checks plus one long-range check" appears in the earlier draft but could not be confirmed against the paper's actual polynomial support. Check the TB generator polynomials directly.

### 4.3 Coprime BB codes — and the absence of weight 5

**Wang and Mueller**, *Coprime Bivariate Bicycle Codes and Their Layouts on Cold Atoms*, arXiv:2408.10001, Quantum **10**, 2009 (2026) [✓].

Setting *z* = *xy* with ℓ and *m* coprime collapses the bivariate polynomials to univariate ones, allowing the rate to be fixed in advance. Reported codes include [[30,4,6]], [[42,6,6]], [[70,6,8]], [[108,12,6]], [[126,12,10]], and [[154,6,16]] [~ — individual entries not each re-verified]. The paper's tables are explicitly labeled weight-6 and weight-8.

**No weight-5 coprime BB codes are reported.** This was checked specifically and is consistent with the source [✓]. The paper's main contribution is a neutral-atom (AOD) layout reducing atom-move time and count.

**Postema and Kokkelmans**, arXiv:2502.17052 [✓ exists, ! title corrected].

> **Correction.** The actual title is *Existence and Characterisation of **Bivariate** Bicycle Codes* — not "…Coprime Bivariate Bicycle Codes" as the earlier draft had it. The paper predicts code dimension and existence conditions from ring structure. **⚠** The "asymptotic badness" result attributed to it in the earlier draft was not evident from the abstract and should be re-checked or dropped.

### 4.4 Morphing circuits: degree 5 with weight-9 stabilizers

**Shaw and Terhal**, *Lowering Connectivity Requirements for Bivariate Bicycle Codes Using Morphing Circuits*, arXiv:2407.16336, Phys. Rev. Lett. **134**, 090602 (2025) [✓].

A new family of BB codes whose parity-check circuits need only **degree-5 connectivity**, including a [[144,12,12]] gross-code variant with **weight-9 stabilizers** measured in only **6 CNOT rounds** — one fewer than Bravyi et al. The paper also gives a sufficient condition for applying morphing circuits to general 2BGA codes [✓].

This is the cleanest illustration of notion (b) versus notion (a) in §3: connectivity drops to 5 while stabilizer weight *rises* to 9. For hardware whose binding constraint is coupler count per qubit rather than measurement-circuit weight, this is frequently the better trade than accepting the smaller distances of true weight-5 codes.

**Chao (NVIDIA)**, *Block algebra for morphing circuits*, arXiv:2606.12724, submitted 10 June 2026 [✓].

> **Addition.** The earlier draft cited this paper without naming its author. It is single-authored by **Rui Chao**. Four CNOT-based CSS morphing-circuit constructions in block-algebra notation; Construction I recovers Shaw–Terhal for 2BGA codes.

### 4.5 Subsystem BB codes: weight-4 gauge checks

**Liang and Chen**, *Topological subsystem bivariate bicycle codes with four-qubit check operators*, arXiv:2605.04151 (2026) [✓].

A translation-invariant CSS subsystem construction realizing BB logical structure using **weight-4 gauge** measurements, with stabilizer syndromes inferred by multiplying gauge outcomes. Reported codes: [[27,6,3]], **[[75,10,5]]**, and [[108,12,6]] — the last carrying six times the logical qubits of a subsystem surface code at the same block length and distance [✓]. The [[75,10,5]] reportedly reduces under a finite-depth Clifford circuit to a [[50,10,5]] BB stabilizer code [~].

This is the weight-4 endpoint of the "how low can measured check weight go" question, and it belongs in the weight-5 discussion — but as notion (c), not notion (a).

### 4.6 Layouts, architectures, and code discovery

**Berthusen, Devulapalli, Schoute, Childs, Gullans, Gorshkov, and Gottesman**, *Toward a 2D Local Implementation of Quantum LDPC Codes*, arXiv:2404.17676, PRX Quantum **6**, 010306 (2025) [✓]. A bilayer architecture measuring some generators less frequently; BB codes reach surface-code-comparable logical error rates with fewer physical qubits under 2D-local gates.

**Choe, Steffan, Vigneau, Parrado-Rodríguez, Ku, Leib, Fernandes Pereira, and Šimkovic (IQM)**, *Barbell Codes: qLDPC Codes for Superconducting Quantum Hardware*, arXiv:2606.06062 (2026) [✓ exists]. Derived from tile codes; a six-qubit star lattice with near-local couplers only between X and Z stabilizers, plus superdense syndrome extraction. Fewer than 30 data qubits per logical qubit; a distance-14 code reaching per-round logical error ≈ 1.4 × 10⁻⁷ at *p* = 10⁻⁴; a distance-11 code with 400 data qubits reaching ≈ 8.8 × 10⁻⁷ at *p* = 10⁻³, nearly three orders of magnitude below 16 patches of a distance-5 surface code at equal qubit budget [~ — from the paper's own reporting and press coverage, not independently re-derived].

**Steffan, Choe, Breuckmann, Fernandes Pereira, and Eberhardt**, *Tile Codes: High-Efficiency Quantum Codes on a Lattice with Boundary*, arXiv:2504.09171, Phys. Rev. Lett. (2025) [✓, ! author order corrected — the arXiv listing leads with **Steffan**, not Choe as the earlier draft had it]. The basis for the Barbell construction.

**Cruz-Benito, Cross, Kremer, and Faro (IBM)**, *Evolutionary Discovery of Bivariate Bicycle Codes with LLM-Guided Search*, arXiv:2606.02418 (2026) [✓ exists]. From the abstract: approximately 1,650 evolutionary iterations screened about 2 × 10⁵ candidate codes; at block length *n* ≤ 360 the workflow identified **465 distinct candidate codes — 97 CSS bivariate-bicycle codes and 368 non-CSS perturbed variants** — including an indecomposable [[288,16,12]] and higher-weight codes with up to *k* = 50 at *d* = 8 [✓].

> **Correction (rate-distance bound).** The earlier draft stated "*k* > 24 implies *d* ≤ 4." The paper's associated repository states **"*k* > 24 implies *d* ≤ 8"** alongside "indecomposable *d* = 12 CSS codes limited to *k* ≤ 16." Use *d* ≤ 8 unless a narrower subfamily claim is located in the paper body.

The paper also reports that BP-OSD **overestimates code distance by up to 12×** for high-rate codes [✓] — a methodological warning that applies to any quoted *d* in this literature (see §12).

**Yoder et al.**, *Tour de gross: A modular quantum computer based on bivariate bicycle codes*, arXiv:2506.03094 (June 2025) [✓, ! full title added]. States verbatim that "an order of magnitude larger logical circuits can be implemented with a given number of physical qubits… than on surface code architectures."

**Symons, Rajput, and Browne**, *Sequences of Bivariate Bicycle Codes from Covering Graphs*, arXiv:2511.13560 [✓].

### 4.7 Logical gates on BB codes

**Eberhardt and Steffan**, *Logical Operators and Fold-Transversal Gates of Bivariate Bicycle Codes*, arXiv:2407.03973, IEEE Trans. Inf. Theory **71**, 1140 (2025) [✓ — venue corroborated across multiple independent citing bibliographies; no journal-ref line was visible on the arXiv landing page]. Explicit toric-like logical bases, fold-transversal Clifford gates, and planar/open-boundary pruning.

**Liang and Chen**, *Self-dual bivariate bicycle codes with transversal Clifford gates*, arXiv:2510.05211 [✓]. Weight-8 self-dual construction.

**Xu, Zhou, Bluvstein, Cain, Kalinowski, Preskill, Lukin, and Maskara**, *Batched high-rate logical operations for quantum LDPC codes*, arXiv:2510.06159 [✓, ! title corrected — the earlier draft rendered this as "batched logical operations"].

**Tiew and Breuckmann**, *Low-Overhead Entangling Gates from Generalised Dehn Twists*, arXiv:2411.03302, IEEE Trans. Inf. Theory **71**, 5452 (2025) [✓]. See §6.

---

## 5. Group algebra codes: 2BGA and generalized bicycle

### 5.1 Foundations

**Lin and Pryadko**, *Quantum two-block group algebra codes*, arXiv:2306.16400, Phys. Rev. A **109**, 022407 (2024) [✓].

2BGA codes LP(*a*,*b*) are defined by a pair of group-algebra elements *a*, *b* ∈ 𝔽_q[*G*], with length *n* = 2|*G*|. They are the smallest (1 × 1) lifted-product codes and reduce to GB codes when *G* is cyclic. The paper establishes permutation-equivalence criteria and parameter bounds, and enumerates all connected 2BGA codes with generator weights *W* ≤ 8 for abelian groups up to *n* ≤ 100 and non-abelian groups up to *n* ≤ 200 [✓]. Code data: `github.com/QEC-pages/2BGA-codes` [✓].

### 5.2 The weight-5 situation in 2BGA

Weight 5 corresponds to *W_a* = 2, *W_b* = 3. The central structural finding stands: **Lin and Pryadko report no individual weight-5 headline [[*n*,*k*,*d*]] code.** Their tabulated example codes are weight-8. Weight-5 data appears only in aggregate distance-versus-*n* scaling fits of the general form *d* = *g* + *f*·*n*^*b* (the fitted exponent *b* is not fixed at 1/2 — see the *k*-dependent values below).

> **⚠ Hedged claims.** The following numbers appeared in the earlier draft and **could not be independently reconfirmed** against the paper's tables within the verification budget. They are reproduced here marked as unverified; anyone relying on them should re-open the PDF.
>
> - Weight-5 fits: *k* = 2 → *f* = 1.295 ± 0.027, *g* = −2.528 ± 0.272, exponent *b* = 0.55; *k* = 4 → *f* = 1.244, *g* = −1.916, *b* = 0.50.
> - Monotonic slope progression *f*: *W* = 4 → 1.00, *W* = 5 → 1.30, *W* = 6 → 1.57, *W* = 7 → 1.78, *W* = 8 → 1.80.
> - Tabulated weight-8 codes [[72,8,9]], [[80,8,10]], [[96,8,12]], and the non-abelian [[24,5,3]] from *A*₄.
> - The claim that odd total weights (including *W* = 5) produce many trivial *k* = 0 codes while even-even weights guarantee *k* ≥ 2.

The authors' own caution — that the sample is too small to resolve asymptotic scaling — applies with full force at weight 5.

### 5.3 Related group-algebra work

**Wang, Lin, and Pryadko**, *Abelian and non-abelian quantum two-block codes*, arXiv:2305.06890 [✓]. Dimension formulas by group type; criterion for essentially-non-abelian codes.

**Kovalev and Pryadko**, *Quantum Kronecker sum-product low-density parity-check codes with finite rate*, arXiv:1212.6703, Phys. Rev. A **88**, 012311 (2013) [✓, ! citation supplied — the earlier draft referenced this as the GB origin without an identifier]. The hyperbicycle construction, of which GB and hypergraph-product codes are limiting cases.

**Wang and Pryadko**, *Distance bounds for generalized bicycle codes*, arXiv:2203.17216 [✓]. Maps a weight-*w* GB code to a code local in **D ≤ w − 1** dimensions (*D* ≤ *w* − 2 if ℓ is prime), giving *d* ≤ O(*n*^(1−1/D)); confirms a *d* ≥ O(*n*^(1/2)) existence bound.

**Lin, Liu, Lim, and Pryadko**, *Single-shot and two-shot decoding with generalized bicycle codes*, arXiv:2502.19406 (2025) [✓]. GB codes have naturally redundant minimum-weight stabilizer generators enabling single- and two-shot decoding; the [[126,12,10]] GB63 example has ℓ = 63, syndrome distance *d_S* = 3, circuit distance *d_C* = 10 [✓].

**(2,2)-Generalized Bicycle Codes: Classification and Comparison with weight-4 Surface Codes**, arXiv:2507.21237 [✓]. Three optimal families [[2*n*²,2,*n*]], [[4*r*²,2,2*r*]], [[(2*t*+1)²+1,2,2*t*+1]]; weight-4 CSS codes from Cayley graphs.

**Dastbasteh, Etxezarreta et al.**, *Generalized Bicycle Codes with Low Connectivity: Minimum Distance and Hook Errors*, arXiv:2508.09082 [✓, ! full title includes "and Hook Errors"]. Highly degenerate families [[*d*²+1,2,*d*]] for odd *d* ≥ 3 and [[*d*²,2,*d*]] for even *d* ≥ 4, each check qubit touching exactly four data qubits, with a fault-tolerant logical CNOT via relabeling.

**Pacenti and Vasić**, *Quantum Margulis codes*, arXiv:2409.09830 [✓]. 2BGA codes via the left-right Cayley complex, adapting the classical Margulis construction.

### 5.4 How the families relate

The following relationships are standard and accepted in the literature, and were confirmed during verification [✓]:

- **BB codes = 2BGA codes over the abelian group Z_ℓ × Z_m.**
- **GB codes = 2BGA codes over cyclic Z_ℓ.**
- **2BGA codes are the smallest (1 × 1) lifted-product codes**, and can be written directly as balanced-product codes.

This chain is why the three families in this survey are not independent research programmes but nested special cases of one algebraic construction.

---

## 6. Balanced product codes

### 6.1 Foundations

**Breuckmann and Eberhardt**, *Balanced Product Quantum Codes*, arXiv:2012.09271, IEEE Trans. Inf. Theory **67**, 6653 (2021) [✓].

The first explicit non-random LDPC family with *K* ∈ Θ(*N*^(4/5)) and *D* ∈ Ω(*N*^(3/5)), obtained by distance-balancing a subsystem family with Θ(*N*^(2/3))/Ω(*N*^(1/3)) parameters [✓]. The construction quotients a hypergraph product by a shared group symmetry, and admits non-abelian twisting.

Reduction relationships [✓, with one refinement]:

- **Trivial group action → hypergraph product.** Exact.
- **Free action → lifted product.** Correct as informal shorthand; for cyclic groups the balanced product overlaps fiber-bundle and lifted-product constructions.

### 6.2 Role in asymptotically good codes

> **Nuance added.** The earlier draft asserted flatly that all three asymptotically good qLDPC constructions "are variants of the balanced product over the left-right Cayley complex." This overstates the case and should be softened to: the three constructions **can be unified within the balanced-product / left-right Cayley complex framework**. The statement is most literal for Dinur–Hsieh–Lin–Vidick and Leverrier–Zémor; Panteleev–Kalachev is presented by its authors as a lifted product, a related special case.

Correct citations [✓]:

- **Panteleev and Kalachev**, *Asymptotically good quantum and locally testable classical LDPC codes*, arXiv:2111.03654, STOC 2022.
- **Dinur, Hsieh, Lin, and Vidick**, *Good quantum LDPC codes with linear time decoders*, arXiv:2206.07750, STOC 2023.
- **Leverrier and Zémor**, *Quantum Tanner codes*, arXiv:2202.13641.

### 6.3 Low-weight balanced product — and the weight-5 void

**Verification confirmed the earlier draft's central negative claim: weight-5 balanced-product results are essentially absent from the literature.** Low-weight balanced-product work clusters at weight 6, 8, and 9. No dedicated weight-5 balanced-product or weight-5 lifted-product family surfaced. Any claim of a broad non-trivariate weight-5 family is unsubstantiated.

**Tiew and Breuckmann**, arXiv:2411.03302, IEEE Trans. Inf. Theory **71**, 5452 (2025) [✓]. Balanced-product cyclic codes with logical gates via generalized Dehn twists. The [[90,8,10]] code is confirmed as the *q* = 5 member of the family [[18*q*,8,≤2*q*]] (*C*_{3q} over *C*_q), which saturates the distance bound; numerical search was conducted at check weights 6 and 8 [✓].

**Kang, Lin, Yao, Gökduman, Meinking, and Brown**, *QUITS* simulator, arXiv:2504.02673, accepted in Quantum (2025) [✓, ! authors supplied].

> **Correction (threshold).** The balanced-product-cyclic threshold is **≈ 0.47%**, stated verbatim in the paper — not "~0.4%" as the earlier draft had it. For context the same paper reports HGP ≈ 0.23% and QLP ≈ 0.24% [✓].

**Hong et al. (NVIDIA)**, *Quantum LDPC codes with design rate 1/5 and good performance below 1000 physical qubits*, arXiv:2607.27644, submitted 30 July 2026 [✓ exists — the unusually high sequence number is genuine]. Check weight 9, non-abelian Z_ℓ ⋊ Z_m metacyclic ("ZSZ") groups, decoded with GPU-accelerated Relay-BP at ~1–2 ms latency under 0.1% idling-free circuit noise [~].

**Demystifying the Balanced Product Code: A Review**, arXiv:2505.13679 (2025) [✓, addition]. A useful cross-check on the balanced-product/lifted-product relationships asserted in §5.4.

---

## 7. Adjacent low-weight families

These were absent from the earlier draft and materially affect the weight-5 hardware discussion.

**Pecorari, Jandura, Brennen, and Pupillo**, *La-cross codes*, arXiv:2404.13010, Nature Communications **16**, 1111 (2025) [✓, addition]. Hypergraph product of cyclic codes with seed polynomial 1 + *x* + *x*^k, giving **variable check weight** — weight-4 surface-like at boundaries rising to weight-6 in the bulk. A leading low-weight, long-range-connected neutral-atom family, and directly relevant to the weight-4/5/6 hardware-target debate.

**Scruby, Hillmann, and Roffe**, *Radial codes*, arXiv:2406.14445, PRX Quantum (2026) [✓, addition].

> **Correction (attribution).** The earlier draft attributed radial codes to "Scruby, Pesah, et al." This is wrong. Radial codes are **Scruby, Hillmann, and Roffe**. Pesah co-authored the *different* **Quantum Rainbow Codes**, arXiv:2408.13130 [✓].

Radial codes are lifted products of quasi-cyclic codes with parameters ⟦2*r*²*s*, 2(*r*−1)², ≤2*s*⟧ and are single-shot decodable. Check weight grows with *r* rather than being fixed at 6, which sharpens the connectivity discussion: they are the natural higher-connectivity, single-shot alternative to BB codes.

**Mian, Gwilliam, and Krastanov**, *Multivariate Multicycle Codes for Complete Single-Shot Decoding*, arXiv:2601.18879 [✓, addition]. Directly addresses single-shot and measurement-error robustness for low-weight multivariate-bicycle-type codes — the one Part-3 gap the earlier draft flagged without filling.

---

## 8. Decoding

**BP-OSD** — Panteleev–Kalachev; Roffe's implementation. The generic qLDPC baseline, slow because of Gaussian elimination, and prone to **overestimating distance by up to 12×** on high-rate codes [✓ arXiv:2606.02418].

**Ambiguity Clustering** — **Wolanski and Barber (Riverlane)**, arXiv:2406.14527 [✓]. Confirmed verbatim: "with 0.3% circuit-level depolarising noise, AC is up to 27x faster than BP-OSD with matched accuracy," and it "decodes the 144-qubit Gross code in 135µs per round of syndrome extraction on an M2 CPU" — fast enough for neutral-atom and trapped-ion cycle times.

**Relay-BP** — **Müller, Alexander, Beverland, Bühler, Johnson, Maurer, and Vandeth**, *Improved belief propagation is sufficient for real-time decoding of quantum memory*, arXiv:2506.01779 [✓, ! title supplied]. Message passing with disordered and negative memory strengths. Confirmed verbatim: it achieves high accuracy "significantly outperforming BP+OSD+CS-10 for bivariate-bicycle codes and comparable to min-weight-matching for surface codes," roughly one to two orders of magnitude better than BP+OSD+CS-10 for gross and two-gross codes at *p* = 3 × 10⁻³ [~ for the latter figure]. FPGA follow-up: arXiv:2510.21600 [✓], reporting a **24 ns BP iteration time** on the [[144,12,12]] gross code and sub-microsecond per-cycle decoding at *p* < 3 × 10⁻³ — note this is a **distinct author set** led by Maurer and Bühler (IBM Quantum).

**Localized Statistics Decoding (BP-LSD)** — **Hillmann, Berent, Quintavalle, Eisert, Wille, and Roffe**, arXiv:2406.18655, Nature Communications **16**, 8214 (2025) [✓, ! authors corrected].

> **Correction (authors).** The earlier draft listed this as "Hillmann, Berent, Townsend-Teague, Eisert, Roffe, Strikis." **There is no Townsend-Teague and no Strikis on this paper.** The correct list is Hillmann, Berent, **Quintavalle**, Eisert, **Wille**, Roffe. This was the most serious factual error in the earlier draft.

**Matching decoder for BB codes** — **Sahay, Williamson, and Brown**, *A matching decoder for bivariate bicycle codes*, arXiv:2602.22770 (2026) [✓, ! title corrected to singular "decoder"]. Introduces the "cylinder trick," exploiting an equivalence between BB codes and copies of the toric code.

**Machine-learning decoders** — **Blue, Avlani, He, Ziyin, and Chuang**, *Machine Learning Decoding of Circuit-Level Noise for Bivariate Bicycle Codes*, arXiv:2504.13043, Quantum **10**, 2149 (2026) [✓].

---

## 9. Bounds and weight-reduction theory

### 9.1 Locality bounds

**Bravyi, Poulin, and Terhal**, *Tradeoffs for reliable quantum information storage in 2D systems*, arXiv:0909.5200, Phys. Rev. Lett. **104**, 050503 (2010) [✓]. A 2D-local [[*n*,*k*,*d*]] stabilizer code satisfies **kd² ≤ O(n)**; in *D* dimensions, *kd*^(2/(D−1)) ≤ O(*n*). The Bravyi–Terhal bound gives *d* ≤ O(*n*^((D−1)/D)).

This is the structural reason high-rate BB and TB codes require non-planar (bi-planar, thickness-2) connectivity: the surface code already saturates *kd*² = O(*n*), so beating it demands long-range checks.

**Dai and Li**, *Locality vs Quantum Codes*, arXiv:2409.15203 [✓ content verified verbatim; STOC 2025 acceptance plausible but not confirmed on the arXiv record]. Above the BPT bound, any 2D embedding requires Ω(max(*k*,*d*)) interactions of length Ω(ℓ\*), where ℓ\* = max(*d*/√*n*, (*kd*²/*n*)^(1/4)) — a quantitative statement of how much non-locality a high-rate weight-5 code needs.

**A variant of the Bravyi–Terhal bound**, arXiv:2502.04995 [✓]. Abelian 2BGA codes of weight *w* are geometrically local in **D = w − 2** dimensions, with *d* ≤ *m*√(γ_D)(√*D* + 4ρ)·*n*^((D−1)/D), where γ_D is the *D*-dimensional Hermite constant.

> **Nuance.** This is **consistent** with the *D* ≤ *w* − 1 bound of arXiv:2203.17216, not in conflict with it: the latter treats cyclic GB codes (*D* ≤ *w* − 1, tightening to *w* − 2 for prime ℓ), while arXiv:2502.04995 treats general abelian 2BGA codes at *D* = *w* − 2. The earlier draft's worry that it had conflated two bounds was unfounded — but they should be presented as two related results.

**At weight 5**, *D* = *w* − 2 = 3, giving *d* ≤ O(*n*^(2/3)). The arithmetic and applicability were verified [✓].

### 9.2 Weight reduction

**Hastings**, *On Quantum Weight Reduction*, arXiv:2102.10030 [✓] — and *Weight reduction for quantum codes*, arXiv:1611.03790, Quantum Inf. Comput. **17**, 1307 (2017) [✓].

The load-bearing claim is **exactly right**, confirmed verbatim: any LDPC code "may be turned into a code where **all stabilizers have weight at most 5** at the cost of at most a constant factor increase in number of physical qubits and constant factor reduction in distance." Weight 5 — not 6, not some other constant — is the universal target, and this theorem is the principal theoretical justification for treating weight 5 as a natural design point.

**Tan and Stambler**, *Effective Distance of Higher Dimensional HGPs and Weight-Reduced Quantum LDPC Codes*, arXiv:2409.02193, Quantum **9**, 1897 (2025) [✓]. Fault tolerance of the coning procedure.

**Sabo et al.**, *Weight-Reduced Stabilizer Codes with Lower Overhead*, arXiv:2402.05228, PRX Quantum **5**, 040302 (2024) [✓, ! arXiv identifier and title supplied]. Reduces product-code checks to ≤ 6.

**Hsieh, Li, and Lin**, *Simplified Quantum Weight Reduction with Optimal Bounds*, arXiv:2510.09601 [✓, ! authors supplied]. Check weight 5, qubit weight 6, giving [[O(*nw*²log *w*), *k*, Ω(*dw*)]]; symmetric X/Z coning; surpasses the √*n* distance barrier on random dense CSS codes.

**Yuan, Baspin, and Williamson**, *Quantum Weight Reduction with Layer Codes*, arXiv:2603.04883 (2026) [✓ exists, ! authors supplied]. Check weight 6, total qubit degree 6.

---

## 10. Hardware and experimental status

**Superconducting.** **Wang, Lu, Zhang et al.**, *Demonstration of low-overhead quantum error correction codes*, arXiv:2505.09684, Nature Physics (2026), DOI 10.1038/s41567-025-03157-4 [✓]. A distance-4 BB code (4 logical qubits) and a distance-3 qLDPC code (6 logical qubits) on the **32-qubit "Kunlun"** long-range-coupled transmon processor, with weight-6 stabilizers measured simultaneously. Logical error rates per logical qubit per cycle: **(8.91 ± 0.17)%** for the distance-4 BB code and **(7.77 ± 0.12)%** for the distance-3 qLDPC code [✓].

> **⚠** The abstract says "distance-4 bivariate bicycle code." The explicit **[[18,4,4]]** designation used in the earlier draft should be confirmed in the paper body rather than assumed from the abstract.

**Neutral atoms.** Coprime-BB cold-atom layouts [✓ arXiv:2408.10001]; **Xu et al.**, *Constant-Overhead Fault-Tolerant Quantum Computation with Reconfigurable Atom Arrays*, arXiv:2308.08648, Nature Physics **20**, 1084 (2024) [✓]; **Poole, Graham, Perlin, Otten, and Saffman**, *Architecture for fast implementation of qLDPC codes with optimized Rydberg gates*, arXiv:2404.18809, Phys. Rev. A **111**, 022433 (2025) [✓, ! identifier and article number supplied].

**Software.** The reference implementation is **`Infleqtion/qLDPC`** (PyPI package `qldpc`), maintained by **Michael A. Perlin** [✓, ! corrected].

> **Correction.** The earlier draft named this "Infleqtion/qLDPCOrg." That repository name is wrong. It should also not be confused with the unrelated `exp_ldpc` repository.

---

## 11. Comparative parameter table

Entries verified where checkable; note the weight-notion column, which the earlier draft's table omitted and which is essential to reading the comparison honestly.

| Code | Family | Weight | Notion (§3) | Rate *k*/*n* | Notes |
|:---|:---|:---:|:---:|:---:|:---|
| Rotated surface, *d*=5, *k*=4 | topological | 4 | (a) | 4/100 = 0.040 (data qubits; *n*=*kd*²=100) | 2D-local baseline; ~200 physical qubits including syndrome/ancilla overhead |
| **TB [[30,4,5]]** | multivariate bicycle | **5** | **(a)** | 4/30 ≈ 0.133 | Depth-7 circuit, conjectured distance-preserving (not proven tight); bi-planar; 60 total qubits vs 200 for surface |
| SBB [[75,10,5]] | subsystem BB | 4 **gauge** | **(c)** | 10/75 ≈ 0.133 | Infers higher-weight stabilizers; *not* a weight-5 code |
| BB [[72,12,6]] | bivariate bicycle | 6 | (a) | 12/72 ≈ 0.167 | Depth-7 circuit; degree-6 |
| BB [[144,12,12]] gross | bivariate bicycle | 6 | (a) | 12/144 ≈ 0.083 | 0.7% threshold; better than 10:1 vs surface |
| Morphed gross [[144,12,12]] | BB + morphing | 9 | **(b)** degree-5 | 0.083 | Degree-5 wiring, 6 CNOT rounds |
| GB [[126,12,10]] | generalized bicycle | 6 | (a) | 12/126 ≈ 0.095 | Single-/two-shot decodable; *d_S* = 3 |
| BPC [[90,8,10]] | balanced-product cyclic | 6 | (a) | 8/90 ≈ 0.089 | *q* = 5 member of [[18*q*,8,≤2*q*]]; 0.47% threshold |
| Radial ⟦2*r*²*s*, 2(*r*−1)², ≤2*s*⟧ | lifted product | grows with *r* | (a) | varies | Single-shot decodable |
| La-cross | hypergraph product | 4–6 variable | (a) | varies | Weight-4 at boundary, weight-6 in bulk |

**Reading caveat.** Thresholds and logical error rates in this table and throughout the survey come from different noise models — code-capacity, phenomenological, circuit-level, and cold-atom global-laser — so cross-family comparisons are indicative, not apples-to-apples.

---

## 12. State of the art and open problems

### The state of the art, stated plainly

For a **genuinely weight-5 code** (notion (a)), the state of the art is the **Voss et al. TB [[30,4,5]]**, and it is close to being the only entry. It is well-characterized: a depth-7 circuit conjectured (not proven) to be distance-preserving, proven bi-planarity, pseudo-thresholds in the low 10⁻³ range, and a 3.3× total-qubit saving over the equivalent surface code. Nothing else verified in this survey occupies the same slot.

For **degree-5 hardware** (notion (b)), the state of the art is **Shaw–Terhal morphing circuits**, which retain full gross-code parameters at degree-5 connectivity and 6 CNOT rounds. For most superconducting constraints this dominates true weight-5 codes, because it does not pay the distance penalty.

For **minimal measured check weight** (notion (c)), the state of the art is **Liang–Chen subsystem BB codes** at weight-4 gauge checks.

In **2BGA/GB**, weight 5 is a documented near-void: Lin and Pryadko's enumeration yields scaling fits but no headline weight-5 code, and the best-characterized low-weight GB results sit at weight 4 (the (2,2) and low-connectivity families) or weight 6. In **balanced product**, weight 5 is a complete void; work clusters at weight 6, 8, and 9.

### Open problems

1. **Find a weight-5 code family with gross-code-scale efficiency.** If a weight-5 family were found with *kd*²/*n* comparable to the gross code (≈ 12) at *n* ≲ 300, with a distance-preserving depth-≤7 circuit, it would displace both the morphing-circuit and subsystem routes. No such family exists today.
2. **Systematic weight-5 search.** No dedicated computational search over weight-5 code spaces (quasi-cyclic, HGP variant, twisted-toric, or otherwise) surfaced for 2025–2026. Given that BB-space searches have been productive, this looks like a genuine unexplored direction, and the odd-weight *k* = 0 problem in 2BGA suggests non-abelian groups and coprime constructions as the places to look.
3. **Finite-size weight-5 balanced products.** Balanced-product asymptotic distance claims are proven only under expander-existence assumptions, and finite-size weight-5 balanced-product codes are entirely unexplored. This is the clearest structural gap in the field.
4. **Is weight 5 the right hardware target at all?** Verification found no paper explicitly arguing the case either way. Given that morphing circuits achieve degree 5 with weight-9 stabilizers, and La-cross codes deliberately accept variable weight 4–6, the premise that weight-5 stabilizers are the natural engineering objective deserves a direct critical treatment that does not yet exist.
5. **Distance verification discipline.** Since BP-OSD overestimates distance by up to 12× on high-rate codes, quoted distances across this literature should be treated as upper-bound estimates unless verified by a solver-proven-optimal MILP result or a completed exhaustive enumeration. A MILP *incumbent* (feasible but not proven optimal) or a QDistRnd/QDistEvol heuristic run is itself only another upper-bound estimate, not a verification method.

### Practical recommendations

- **Near-term weight-5 experiment:** start from the TB [[30,4,5]] (arXiv:2406.19151), benchmarked against weight-6 [[72,12,6]] / [[72,8,6]] at your target physical error rate (~10⁻³) to confirm the qubit savings hold.
- **If connectivity is the binding constraint:** use morphing circuits (arXiv:2407.16336) rather than a weight-5 code, and keep the gross-code parameters.
- **If local measured weight is the constraint:** evaluate subsystem BB [[75,10,5]] (arXiv:2605.04151) and Barbell codes (arXiv:2606.06062).
- **Decoding:** Relay-BP or Ambiguity Clustering, not vanilla BP-OSD. Relay-BP leads on the accuracy/latency frontier and is FPGA-portable at 24 ns per iteration; fall back to the BB matching decoder (arXiv:2602.22770) if ASIC latency targets cannot be met.
- **Algebraic exploration:** use the Lin–Pryadko enumeration data (`github.com/QEC-pages/2BGA-codes`) with the `Infleqtion/qLDPC` package, prioritizing non-abelian groups given the high *k* = 0 rate at odd weights.

---

## 13. Bibliography with verification status

All entries verified to exist with the metadata shown unless marked otherwise. **Bold** marks corrections relative to the earlier draft of this survey.

### Bivariate bicycle

- Bravyi, Cross, Gambetta, Maslov, Rall, Yoder. *High-threshold and low-overhead fault-tolerant quantum memory.* arXiv:2308.07915. Nature **627**, 778 (2024). **Threshold 0.7% (published); 0.8% is a superseded pre-revision value.**
- Voss, Xian, Haug, Bharti. *Multivariate Bicycle Codes.* arXiv:2406.19151. Phys. Rev. A **111**, L060401 (2025). *(v1 titled "Trivariate Bicycle Codes.")*
- Wang, Mueller. *Coprime Bivariate Bicycle Codes and Their Layouts on Cold Atoms.* arXiv:2408.10001. Quantum **10**, 2009 (2026).
- Postema, Kokkelmans. ***Existence and Characterisation of Bivariate Bicycle Codes.*** arXiv:2502.17052. **Title corrected; "asymptotic badness" claim unverified.**
- Shaw, Terhal. *Lowering Connectivity Requirements for Bivariate Bicycle Codes Using Morphing Circuits.* arXiv:2407.16336. Phys. Rev. Lett. **134**, 090602 (2025).
- **Chao, R.** *Block algebra for morphing circuits.* arXiv:2606.12724 (2026). **Author supplied.**
- Liang, Chen. *Topological subsystem bivariate bicycle codes with four-qubit check operators.* arXiv:2605.04151 (2026).
- Liang, Chen. *Self-dual bivariate bicycle codes with transversal Clifford gates.* arXiv:2510.05211.
- Berthusen, Devulapalli, Schoute, Childs, Gullans, Gorshkov, Gottesman. *Toward a 2D Local Implementation of Quantum LDPC Codes.* arXiv:2404.17676. PRX Quantum **6**, 010306 (2025).
- Choe, Steffan, Vigneau, Parrado-Rodríguez, Ku, Leib, Fernandes Pereira, Šimkovic. *Barbell Codes: qLDPC Codes for Superconducting Quantum Hardware.* arXiv:2606.06062 (2026).
- **Steffan**, Choe, Breuckmann, Fernandes Pereira, Eberhardt. *Tile Codes: High-Efficiency Quantum Codes on a Lattice with Boundary.* arXiv:2504.09171. Phys. Rev. Lett. (2025). **Author order corrected.**
- Cruz-Benito, Cross, Kremer, Faro. *Evolutionary Discovery of Bivariate Bicycle Codes with LLM-Guided Search.* arXiv:2606.02418 (2026). ***k* > 24 ⟹ *d* ≤ 8 (not ≤ 4).**
- Yoder et al. ***Tour de gross: A modular quantum computer based on bivariate bicycle codes.*** arXiv:2506.03094 (2025). **Full title supplied.**
- Symons, Rajput, Browne. *Sequences of Bivariate Bicycle Codes from Covering Graphs.* arXiv:2511.13560.
- Eberhardt, Steffan. *Logical Operators and Fold-Transversal Gates of Bivariate Bicycle Codes.* arXiv:2407.03973. IEEE Trans. Inf. Theory **71**, 1140 (2025).
- Xu, Zhou, Bluvstein, Cain, Kalinowski, Preskill, Lukin, Maskara. ***Batched high-rate logical operations for quantum LDPC codes.*** arXiv:2510.06159. **Title corrected.**

### Group algebra

- Lin, Pryadko. *Quantum two-block group algebra codes.* arXiv:2306.16400. Phys. Rev. A **109**, 022407 (2024). **Numeric fit constants unverified — see §5.2.**
- Wang, Lin, Pryadko. *Abelian and non-abelian quantum two-block codes.* arXiv:2305.06890.
- **Kovalev, Pryadko.** *Quantum Kronecker sum-product low-density parity-check codes with finite rate.* **arXiv:1212.6703.** Phys. Rev. A **88**, 012311 (2013). **Citation supplied.**
- Wang, Pryadko. *Distance bounds for generalized bicycle codes.* arXiv:2203.17216.
- Lin, Liu, Lim, Pryadko. *Single-shot and two-shot decoding with generalized bicycle codes.* arXiv:2502.19406 (2025).
- *(2,2)-Generalized Bicycle Codes: Classification and Comparison with weight-4 Surface Codes.* arXiv:2507.21237.
- Dastbasteh, Etxezarreta et al. *Generalized Bicycle Codes with Low Connectivity: Minimum Distance **and Hook Errors**.* arXiv:2508.09082.
- Pacenti, Vasić. *Quantum Margulis codes.* arXiv:2409.09830.

### Balanced product and good qLDPC

- Breuckmann, Eberhardt. *Balanced Product Quantum Codes.* arXiv:2012.09271. IEEE Trans. Inf. Theory **67**, 6653 (2021).
- Breuckmann, Eberhardt. *Quantum Low-Density Parity-Check Codes.* PRX Quantum **2**, 040101 (2021). *(Standard review — added.)*
- Tiew, Breuckmann. *Low-Overhead Entangling Gates from Generalised Dehn Twists.* arXiv:2411.03302. IEEE Trans. Inf. Theory **71**, 5452 (2025).
- **Kang, Lin, Yao, Gökduman, Meinking, Brown.** *QUITS: A modular Monte Carlo simulator for qLDPC codes.* arXiv:2504.02673. Quantum (2025). **BPC threshold 0.47%.**
- **Hong et al.** *Quantum LDPC codes with design rate 1/5 and good performance below 1000 physical qubits.* arXiv:2607.27644 (2026). **Authors supplied.**
- *Demystifying the Balanced Product Code: A Review.* arXiv:2505.13679 (2025). *(Added.)*
- Panteleev, Kalachev. *Asymptotically good quantum and locally testable classical LDPC codes.* arXiv:2111.03654. STOC 2022. *(Added.)*
- Dinur, Hsieh, Lin, Vidick. *Good quantum LDPC codes with linear time decoders.* arXiv:2206.07750. STOC 2023. *(Added.)*
- Leverrier, Zémor. *Quantum Tanner codes.* arXiv:2202.13641. *(Added.)*

### Adjacent low-weight families

- **Pecorari, Jandura, Brennen, Pupillo.** *La-cross codes.* arXiv:2404.13010. Nature Communications **16**, 1111 (2025). **Added.**
- **Scruby, Hillmann, Roffe.** *Radial codes.* arXiv:2406.14445. PRX Quantum (2026). **Added; attribution corrected from "Scruby, Pesah."**
- Pesah et al. *Quantum Rainbow Codes.* arXiv:2408.13130. *(Distinct from radial codes.)*
- **Mian, Gwilliam, Krastanov.** *Multivariate Multicycle Codes for Complete Single-Shot Decoding.* arXiv:2601.18879. **Added.**

### Decoding

- Wolanski, Barber. *Ambiguity Clustering.* arXiv:2406.14527.
- Müller, Alexander, Beverland, Bühler, Johnson, Maurer, Vandeth. ***Improved belief propagation is sufficient for real-time decoding of quantum memory.*** arXiv:2506.01779. **Title supplied.**
- Maurer, Bühler et al. *FPGA Relay-BP.* arXiv:2510.21600. *(Distinct author set.)*
- Hillmann, Berent, **Quintavalle**, Eisert, **Wille**, Roffe. *Localized statistics decoding.* arXiv:2406.18655. Nature Communications **16**, 8214 (2025). **Author list corrected.**
- Sahay, Williamson, Brown. ***A matching decoder for bivariate bicycle codes.*** arXiv:2602.22770 (2026). **Title corrected.**
- Blue, Avlani, He, Ziyin, Chuang. *Machine Learning Decoding of Circuit-Level Noise for Bivariate Bicycle Codes.* arXiv:2504.13043. Quantum **10**, 2149 (2026).

### Bounds and weight reduction

- Bravyi, Poulin, Terhal. *Tradeoffs for reliable quantum information storage in 2D systems.* arXiv:0909.5200. Phys. Rev. Lett. **104**, 050503 (2010).
- Dai, Li. *Locality vs Quantum Codes.* arXiv:2409.15203. *(STOC 2025 acceptance unconfirmed on arXiv record.)*
- *A variant of the Bravyi–Terhal bound.* arXiv:2502.04995. *(D = w − 2 for abelian 2BGA; consistent with arXiv:2203.17216.)*
- Hastings. *On Quantum Weight Reduction.* arXiv:2102.10030. **Weight ≤ 5 confirmed verbatim.**
- Hastings. *Weight reduction for quantum codes.* arXiv:1611.03790. Quantum Inf. Comput. **17**, 1307 (2017).
- Tan, Stambler. *Effective Distance of Higher Dimensional HGPs and Weight-Reduced Quantum LDPC Codes.* arXiv:2409.02193. Quantum **9**, 1897 (2025).
- Sabo et al. ***Weight-Reduced Stabilizer Codes with Lower Overhead.*** **arXiv:2402.05228.** PRX Quantum **5**, 040302 (2024). **Identifier and title supplied.**
- **Hsieh, Li, Lin.** *Simplified Quantum Weight Reduction with Optimal Bounds.* arXiv:2510.09601. **Authors supplied.**
- **Yuan, Baspin, Williamson.** *Quantum Weight Reduction with Layer Codes.* arXiv:2603.04883 (2026). **Authors supplied.**

### Hardware

- Wang, Lu, Zhang et al. *Demonstration of low-overhead quantum error correction codes.* arXiv:2505.09684. Nature Physics (2026), DOI 10.1038/s41567-025-03157-4.
- Xu et al. *Constant-Overhead Fault-Tolerant Quantum Computation with Reconfigurable Atom Arrays.* arXiv:2308.08648. Nature Physics **20**, 1084 (2024).
- **Poole, Graham, Perlin, Otten, Saffman.** *Architecture for fast implementation of qLDPC codes with optimized Rydberg gates.* **arXiv:2404.18809.** Phys. Rev. A **111**, 022433 (2025). **Identifier supplied.**

### Software and reference

- **`Infleqtion/qLDPC`** (PyPI: `qldpc`), maintained by **Michael A. Perlin**. **Corrected from "qLDPCOrg."**
- `github.com/QEC-pages/2BGA-codes` — Lin–Pryadko 2BGA enumeration data.
- Error Correction Zoo. arXiv:2606.11484 (2026). *(Added as a coverage benchmark.)*

---

## 14. Limitations of this survey

**Verification depth is uneven.** Some quantitative claims were confirmed only from abstracts, arXiv listing pages, citing bibliographies, or press coverage rather than full-text PDFs, because the verification budget was exhausted before every primary source could be opened. Every such instance carries a **~** mark inline. The items most in need of a direct full-text read are the Lin–Pryadko numeric fit constants (§5.2), the Voss et al. weight-by-weight table and pseudo-thresholds (§4.2), the Barbell logical error rates (§4.6), and the explicit [[18,4,4]] designation in the Kunlun experiment (§10).

**One numeric conflict is presented rather than resolved.** The gross-code threshold appears as 0.7% (published Nature) and 0.8% (pre-revision arXiv). Both are given, along with the documented reason for the discrepancy, rather than one being silently chosen.

**Journal references for a handful of items** (arXiv:2306.16400, 2305.06890, 2502.19406, 2511.13560) were not visible on their arXiv landing pages and were corroborated through independent citing bibliographies. Where a venue is asserted, it is consistent across multiple sources, but a landing-page confirmation would be stronger.

**Distance claims should be treated as upper bounds** unless independently verified by a solver-proven-optimal MILP result or a completed exhaustive enumeration, given the documented tendency of BP-OSD to overestimate distance by up to 12× on high-rate codes. A MILP *incumbent* or a QDistRnd/QDistEvol run is itself an upper-bound heuristic, not a verification method.

**Several 2026 identifiers correspond to very recent preprints** (2601.x through 2607.x) that are not yet peer-reviewed. All were confirmed to exist — an initial suspicion that some were fabricated proved unfounded — but their results are unrefereed, and the Barbell and design-rate-1/5 performance figures are simulation results rather than experimental ones.

**Cross-family threshold comparisons are indicative only,** since the underlying noise models differ.