# Large Language Model-Guided Discovery of Weight-Five Bivariate Bicycle Codes

[arXiv:2610.06623](https://arxiv.org/abs/2610.06623)

Sources in this folder: [`weight5_paper.tex`](weight5_paper.tex) (main paper), [`weight5_supplement.tex`](weight5_supplement.tex) (supplemental material), [`weight5_supplemental_tables.tex`](weight5_supplemental_tables.tex) (auto-generated catalog tables), [`weight5_references.bib`](weight5_references.bib) / [`weight5_supplementNotes.bib`](weight5_supplementNotes.bib) (bibliographies), [`figures/`](figures/) (figure generation script + data + PDF), [`2610.06623v1.pdf`](2610.06623v1.pdf) (compiled v1).

See the top-level [`README.md`](../../README.md) for project setup, generic usage, and the shared BB/PBB code model.

## Overview

This paper extends the evolutionary BB/PBB discovery workflow from [paper/2606.02418](../2606.02418/) to **weight-five** stabilizers: CSS codes with a binomial–trinomial (2+3) support split, and support-contained PBB codes whose mixed rows also have Pauli weight five. Four 300-iteration OpenEvolve campaigns (CSS-small, CSS-large, PBB-small, PBB-large) produce a catalogue of **1,142 retained proposals** (912 CSS, 230 PBB) forming 630 presentation classes, reduced to 608 connected-component classes after a basis-independent connectivity audit (409 of 1,142 proposals, 35.8%, are disconnected direct sums).

Headline certified results:

- Connected CSS: **[[96,4,10]]**, **[[140,6,10]]**, **[[180,4,14]]** (FOMs 4.17, 4.29, 4.356) — the latter two lie on the exact rate–distance frontier compiled in the paper.
- Connected PBB: **[[216,4,10]]** (FOM 1.85), attained by two distinct component classes — the strongest certified connected PBB point.
- A post-search comparison certifies seven imported Lin–Pryadko 2BGA-archive constructions, including **[[132,4,12]]** (FOM 4.364, over the nonabelian group $C_{11}\times S_3$) — the highest exact weight-five FOM in the compiled comparison.
- Of 936 distinct CSS proposals with a recorded positive distance, 816 (87.18%) are certified at d≥5, versus 72.50–74.26% for three stratified-uniform random-sampling controls at the same per-lattice/dimension-band allocation.
- An algebraic classification organizes all 329 connected CSS presentation classes into order-3, order-7, and order-15 cyclotomic-kernel strata (no order-5 stratum is possible for this family — proven in the paper).

## Headline Weight-Five Codes

### Selected connected CSS presentations

| Code | (ℓ,m) | A(x,y) | B(x,y) | FOM | Evidence |
|------|-------|--------|--------|-----|----------|
| [[90,4,9]] | (5,9) | 1+x²y⁸+x³y⁷ | 1+x²y⁶ | 3.60 | M+component |
| [[96,4,10]] | (16,3) | 1+y+x²y² | 1+x³ | 4.17 | M+component |
| [[120,4,10]] | (6,10) | 1+xy+x⁵y⁵ | 1+y | 3.33 | M |
| [[140,6,10]] | (5,14) | 1+y¹³+x⁴y¹¹ | 1+xy⁷ | 4.29 | M+component |
| [[180,4,13]] | (10,9) | 1+x²y³ | 1+x⁵y+x⁷y⁸ | 3.76 | M |
| **[[180,4,14]]** | (10,9) | 1+x²y²+x⁵y | 1+x³y³ | **4.356** | M |
| [[432,4,10]] | (18,12) | 1+x⁴y³ | 1+xy¹¹+x²y¹⁰ | 0.93 | M+class |

"M" = exact MILP certification over all logical sectors; "M+component"/"M+class" transfer that certificate from an isomorphic component or presentation-class member. See paper Table III (`tab:css`) for the full list including nonexact interval rows.

### Selected PBB presentations

| Code | (ℓ,m) | (A,B) | (C,D) | FOM | Structure | Evidence |
|------|-------|-------|-------|-----|-----------|----------|
| [[144,4,6]] | (9,8) | (1+x⁶y, 1+x²y⁴+x⁴) | (0, x²y⁴) | 1.00 | conn. | Exh.+M |
| [[144,4,8]] | (18,4) | (1+x³y, 1+xy²+x²) | (1+x³y, xy²) | 1.78 | conn. | M+component |
| [[180,2,11]] | (15,6) | (1+x²y², 1+x²y³+x⁴) | (1+x²y², 1+x⁴) | 1.34 | conn. | M+component |
| **[[216,4,10]]** | (18,6) | (1+x²y, 1+xy³+x²) | (1+x²y, 1+x²) | **1.85** | conn. | M+component |
| [[360,4,11]] | (15,12) | (1+x²y⁴, 1+x²y⁶+x⁴) | (1+x²y⁴, 1+x⁴) | 1.34 | 2×[[180,2,11]] | M |

See paper Table V (`tab:pbb`) for the full list, including the two distinct [[216,4,10]] component classes and the fixed order-three geometric baseline family.

## Evolution Campaigns

Four 300-iteration OpenEvolve campaigns, each with a small- and large-lattice profile. An ensemble of Claude Opus 5 and GPT-5.6 Sol generated program variants (model labels retained as provenance). The CSS fitness function favored high-FOM weight-five 2+3 candidates; the PBB fitness function prioritized non-CSS candidates with d≥5.

| Family | Lattice profile | Stage-2 (distance-evaluated) lattices | Config | Seed |
|--------|-----------------|----------------------------------------|--------|------|
| CSS | small | (6,10), (9,9), (10,9) | `evolve/config_weight5_css_small.yaml` | `evolve/seed_solution_weight5_css.py` |
| CSS | large | (16,12), (15,14), (18,12) | `evolve/config_weight5_css_large.yaml` | `evolve/seed_solution_weight5_css.py` |
| PBB | small | (6,10), (9,9), (10,9) | `evolve/config_weight5_pbb_small.yaml` | `evolve/seed_solution_weight5_pbb.py` |
| PBB | large | (16,12), (15,14), (18,12) | `evolve/config_weight5_pbb_large.yaml` | `evolve/seed_solution_weight5_pbb.py` |

Each profile also has stage-1 (cheap k-only) lattices: small = (6,6),(6,9),(9,8); large = (12,9),(12,12),(15,12).

```bash
# Shared deterministic census/budget sampler (run once per family/profile before a full campaign)
uv run python scripts/run_weight5_stage0.py \
  --family css \
  --profile small \
  --seed 42 \
  --max-audit-candidates 500 \
  --cheap-cpu-hours-per-lattice 4 \
  --checkpoint-dir results/stage0/weight5_css_small

# CSS small pilot
QCODE_MIN_K_THRESHOLD=2 QCODE_MIN_RELEVANT_D=4 QCODE_LATTICE_PROFILE=small \
QCODE_FOM_THRESHOLD_REFINE=3.0 QCODE_FOM_THRESHOLD_EXACT=5.0 \
QCODE_SAVE_FOM_THRESHOLD_CSS=2.0 QCODE_SAVE_TRUST_RATIO=2.0 \
uv run python evolve/run_evolution.py \
  --evaluator evolve/openevolve_evaluator_weight5_css.py \
  --config evolve/config_weight5_css_small.yaml \
  --seed evolve/seed_solution_weight5_css.py \
  --iterations 50 --run-name weight5_css_small_pilot_v1

# CSS large run (paper run)
QCODE_MIN_K_THRESHOLD=2 QCODE_MIN_RELEVANT_D=4 QCODE_LATTICE_PROFILE=large \
QCODE_FOM_THRESHOLD_REFINE=3.0 QCODE_FOM_THRESHOLD_EXACT=5.0 \
QCODE_SAVE_FOM_THRESHOLD_CSS=2.0 QCODE_SAVE_TRUST_RATIO=2.0 \
uv run python evolve/run_evolution.py \
  --evaluator evolve/openevolve_evaluator_weight5_css.py \
  --config evolve/config_weight5_css_large.yaml \
  --seed evolve/seed_solution_weight5_css.py \
  --iterations 300 --run-name weight5_css_large_v1 --wandb

# PBB small pilot
QCODE_MIN_K_THRESHOLD_NONCSS=2 QCODE_MIN_RELEVANT_D_NONCSS=5 QCODE_LATTICE_PROFILE=small \
QCODE_FOM_THRESHOLD_REFINE=3.0 QCODE_FOM_THRESHOLD_EXACT=5.0 \
QCODE_SAVE_FOM_THRESHOLD_NONCSS=2.0 QCODE_SAVE_TRUST_RATIO_NONCSS=2.5 \
uv run python evolve/run_evolution.py \
  --noncss \
  --evaluator evolve/openevolve_evaluator_weight5_pbb.py \
  --config evolve/config_weight5_pbb_small.yaml \
  --seed evolve/seed_solution_weight5_pbb.py \
  --iterations 50 --run-name weight5_pbb_small_pilot_v1

# PBB large run (paper run)
QCODE_MIN_K_THRESHOLD_NONCSS=2 QCODE_MIN_RELEVANT_D_NONCSS=5 QCODE_LATTICE_PROFILE=large \
QCODE_FOM_THRESHOLD_REFINE=3.0 QCODE_FOM_THRESHOLD_EXACT=5.0 \
QCODE_SAVE_FOM_THRESHOLD_NONCSS=2.0 QCODE_SAVE_TRUST_RATIO_NONCSS=2.5 \
uv run python evolve/run_evolution.py \
  --noncss \
  --evaluator evolve/openevolve_evaluator_weight5_pbb.py \
  --config evolve/config_weight5_pbb_large.yaml \
  --seed evolve/seed_solution_weight5_pbb.py \
  --iterations 300 --run-name weight5_pbb_large_v1 --wandb
```

Use `--family pbb` for the Stage-0 sampler's PBB variant. Large-profile Stage-0 runs set `--cheap-cpu-hours-per-lattice 8`. See [`../../plans/direction2_weight5_campaigns.md`](../../plans/direction2_weight5_campaigns.md) for the full campaign plan this was implemented from.

## Weight-Five Code, Data, and Figure Map

### Evolve (campaign configuration)

| File | Purpose |
|------|---------|
| `evolve/config_weight5_css_{small,large}.yaml` | CSS OpenEvolve configs (small/large lattice profile) |
| `evolve/config_weight5_pbb_{small,large}.yaml` | PBB OpenEvolve configs (small/large lattice profile) |
| `evolve/openevolve_evaluator_weight5_css.py` | CSS evaluator: 2+3 weight-5 constraint, k-band stratification, BP-OSD→MILP cascade |
| `evolve/openevolve_evaluator_weight5_pbb.py` | PBB evaluator: support-containment ansatz, d≥5 fitness weighting |
| `evolve/seed_solution_weight5_{css,pbb}.py` | Seed `generate_candidates` for each family |
| `evolve/prompt_context_weight5_{css,pbb}.md` | Domain knowledge fed to the LLM ensemble |
| `evolve/_weight5_css_generate_worker.py` | Multiprocessing worker for CSS candidate generation |

### Scripts (verification, audits, publication build)

| Script | Purpose |
|--------|---------|
| `scripts/verify_weight5_{css_small,css_large,pbb_small}_v1_milp.py`, `verify_weight5_pbb_small_milp_deep.py` | Per-campaign MILP verification passes |
| `scripts/merge_weight5_{css_small,css_large,pbb_small,pbb_large}_v1_milp.py` | Merge MILP evidence into per-campaign records |
| `scripts/prepare_weight5_pbb_large_winners.py`, `recover_weight5_pbb_attribution.py` | PBB-large winner selection and LLM-attribution recovery |
| `scripts/audit_weight5_css_low_weight.py` | Catalogue-wide exhaustive weight-≤4 exclusion search |
| `scripts/audit_weight5_css_upper_bound_witnesses.py` | Explicit witnesses tightening CSS upper-bound endpoints |
| `scripts/audit_weight5_css_weight5_witnesses.py` | Explicit weight-five logical witnesses (both sectors) for exact CSS presentations |
| `scripts/audit_weight5_component_classes.py` | Component-level BLISS canonicalization (630 parent → 608 component classes) |
| `scripts/certify_weight5_components.py`, `certify_weight5_pbb_components.py` | Transfer exact distances from disconnected parents to connected components |
| `scripts/audit_weight5_connectivity.py` | Basis-independent connectivity partition (row-reduced symplectic check matrix) |
| `scripts/audit_weight5_pbb_component_lc.py` | Exhaustive LC-to-CSS screen for the four [[12,2,3]] PBB component classes |
| `scripts/audit_lin_pryadko_weight5.py` | Reconstruct + MILP-certify Lin–Pryadko 2BGA archive constructions |
| `scripts/audit_qldpc_challenge_connectivity.py` | Connectivity + evidence audit of the Unitary Foundation QEC Challenge comparison snapshot |
| `scripts/analyze_weight5_quotient_characters.py` | Order-3/7/15 cyclotomic-kernel stratification |
| `scripts/audit_weight5_referee_v4.py` | Targeted checks requested in referee report v4 |
| `scripts/run_weight5_uniform_baseline.py` | Stratified-uniform transfer-control sampler (3 replicates) |
| `scripts/build_weight5_publication_catalogue.py` | Builds `results/weight5_publication_catalogue.jsonl` + manifest from all of the above |
| `scripts/generate_weight5_supplement.py` | Generates `weight5_supplemental_tables.tex` from the publication catalogue |
| `scripts/run_weight5_stage0.py` | Shared deterministic census/budget sampler (see campaign commands above) |

### Results (catalogue and audit data)

| File | Content |
|------|---------|
| `results/weight5_publication_catalogue.jsonl` | 1,142 retained presentations in 630 colored-BLISS classes — the canonical catalogue |
| `results/weight5_publication_manifest.json` | Hash manifest + accounting (398 exact / 744 nonexact presentations) |
| `results/weight5_component_certifications.jsonl` | 7 exact connected-component distance certificates + 3 explicit BB matches |
| `results/weight5_css_low_weight_audit.jsonl` | Weight-≤4 audit of 492 CSS class representatives |
| `results/weight5_css_upper_bound_witnesses.jsonl` | Witnesses tightening 10 CSS upper endpoints |
| `results/weight5_component_classes.jsonl` | 608 component-level classes (from 630 parent classes) |
| `results/weight5_challenge_connectivity_audit.jsonl` | Connectivity audit of all 102 QEC Challenge comparison entries |
| `results/weight5_pbb_component_certifications.jsonl` | 4 explicit connected PBB component matches |
| `results/weight5_pbb_component_lc_audit.jsonl` | Complete LC-to-CSS audit of the 4 PBB [[12,2,3]] component classes |
| `results/weight5_lin_pryadko_audit.json` | 10 reconstructed weight-5 2BGA representatives (7 exact MILP, 3 upper-bound) |
| `results/weight5_quotient_character_analysis.json` | C3/C5/C7/C15 quotient coverage + cyclic-gcd multiplicity analysis |
| `results/weight5_css_weight5_witnesses.jsonl` | 90 validated weight-5 logical witnesses (45 exact CSS presentations) |
| `results/weight5_referee_v4_checks.json` | Referee-v4 focused verification |
| `results/weight5_uniform_baseline/` | Stratified-uniform transfer control: 3 replicates, per-lattice summaries, aggregate comparison |

Full one-line descriptions for every file above are in [`../../results/README.md`](../../results/README.md).

### Figures

| Figure | Script | Data |
|--------|--------|------|
| Weight-5 Pareto comparison (Fig. in `weight5_paper.tex`) | `figures/plot_weight5_pareto_comparison.py` | `figures/weight5_pareto_comparison_data.csv`, `figures/qldpc_challenge_weight5_snapshot.json`, `figures/qldpc_challenge_bb_snapshot.json`, `figures/codetables_short_snapshot.json` → `figures/fig_weight5_pareto_comparison.pdf` |

```bash
cd paper/2610.06623/figures
uv run python plot_weight5_pareto_comparison.py
```

### Tests

The weight-5 evaluators, audits, catalogue builder, and transfer control each have dedicated coverage in `tests/`: `test_weight5_css_evaluator.py`, `test_weight5_pbb_evaluator.py`, `test_weight5_css_low_weight_audit.py`, `test_weight5_css_weight5_witnesses.py`, `test_weight5_component_class_support.py`, `test_weight5_component_certifications.py`, `test_weight5_pbb_component_certifications.py`, `test_weight5_pbb_component_lc_audit.py`, `test_weight5_presentation_class_sidecar.py`, `test_weight5_presentation_equivalence.py`, `test_weight5_pareto_endpoint_semantics.py`, `test_weight5_publication_catalogue.py`, `test_weight5_quotient_characters.py`, `test_weight5_referee_v4_checks.py`, `test_weight5_challenge_connectivity_audit.py`, `test_weight5_pbb_baseline_family.py`, `test_weight5_uniform_baseline.py`, `test_lin_pryadko_weight5_audit.py`, `test_run_weight5_stage0.py`.

```bash
uv run python -m pytest tests/ -k weight5 -v
```

## Reproduce

```bash
# Rebuild the publication catalogue from all campaign/audit inputs
uv run python scripts/build_weight5_publication_catalogue.py

# Regenerate the supplemental LaTeX tables from the catalogue
uv run python scripts/generate_weight5_supplement.py
```

Note: paths above (`evolve/...`, `scripts/...`, `results/...`, `tests/...`, `plans/...`) are relative to the repository root, not to this `paper/2610.06623/` folder.
