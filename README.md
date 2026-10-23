# qcode-discovery

Discovering bivariate bicycle (BB) and perturbed bivariate bicycle (PBB) quantum LDPC codes via LLM-guided evolutionary search.

An evolutionary loop (OpenEvolve + an LLM ensemble) evolves Python functions that generate candidate polynomial pairs/tuples over `F_2[x,y]/(x^ℓ-1, y^m-1)`. Each candidate is scored through a multi-stage evaluation cascade (GF(2) rank → BP-OSD distance estimate → MILP exact distance) to find high-FOM (FOM = kd²/n) BB and PBB codes.

## Papers

This repository backs two papers, each with its own headline results, campaign details, and repo-to-paper mapping:

- **"Evolutionary Discovery of Bivariate Bicycle Codes with LLM-Guided Search"** — weight-6+ BB/PBB codes, 5 campaigns, 465 discovered codes. [arXiv:2606.02418](https://arxiv.org/abs/2606.02418). Sources: [`paper/2606.02418/`](paper/2606.02418/). **Docs: [`paper/2606.02418/README.md`](paper/2606.02418/README.md)**.
- **"Large Language Model-Guided Discovery of Weight-Five Bivariate Bicycle Codes"** — weight-5 BB/PBB codes, 4 campaigns, 1,142 retained proposals. [arXiv:2610.06623](https://arxiv.org/abs/2610.06623). Sources: [`paper/2610.06623/`](paper/2610.06623/). **Docs: [`paper/2610.06623/README.md`](paper/2610.06623/README.md)**.

## How It Works

### What Is a BB Code?

A bivariate bicycle code is defined by two polynomials over `F_2[x,y]/(x^ℓ - 1, y^m - 1)`. Code parameters: **n = 2·ℓ·m** physical qubits, **k** logical qubits (exact via GF(2) rank), **d** code distance (expensive — distance estimation and exact MILP computation are described in each paper's doc). CSS commutativity is automatic, so the search is purely combinatorial.

A **perturbed bivariate bicycle (PBB) code** augments the construction with two additional polynomials (C, D) creating mixed X/Z stabilizers. When C = D = 0, the code reduces to a CSS BB code.

### Architecture

```
                    ┌─────────────────────────┐
                    │    Evolutionary Loop     │
                    │  (OpenEvolve + LLM       │
                    │   ensemble)              │
                    │                          │
                    │  Evolves the strategy    │
                    │  in generate_candidates  │
                    └───────────┬──────────────┘
                                │ proposes polynomial tuples
                                ▼
┌──────────────────────────────────────────────────────────────┐
│                    Evaluation Cascade                         │
│                                                              │
│  Stage 1: k-only screening (~2s)  — GF(2) rank              │
│  Stage 2: BP-OSD distance estimate                           │
│  Stage 3: MILP exact distance (in-loop, selected campaigns) │
│                                                              │
│  Post-hoc: MILP verification on all discovered codes         │
│            multi-decoder BP-OSD protocol                      │
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
│   ├── clifford_equivalence.py            #   LC equivalence to CSS
│   ├── mirror_code.py                     #   Mirror code construction (Khesin & Lu)
│   ├── results.py                         #   JSON persistence & Pareto front
│   ├── tracking.py                        #   Run logging and metrics (JSONL)
│   └── threshold.py                       #   Threshold simulation stub (not implemented)
├── evolve/                                # LLM-guided evolutionary search (seeds, configs,
│                                           #   evaluators, and prompt context per campaign family —
│                                           #   see each paper's doc for the full list)
├── scripts/                               # Verification & publication scripts (per-paper docs
│                                           #   map these to specific claims/figures/tables)
├── tests/                                 # Test suite + analysis scripts — see tests/README.md
├── results/                                # Verified results and evolution artifacts — see results/README.md
├── paper/                                 # Paper sources (one subfolder per paper)
│   ├── 2606.02418/                        #   "Evolutionary Discovery of BB Codes" — see its README.md
│   └── 2610.06623/                        #   "Weight-Five BB Codes" — see its README.md
└── plans/                                 # Research direction plans
```

See [results/README.md](results/README.md) for details on each data file. See [tests/README.md](tests/README.md) for documentation on all analysis scripts. See each paper's `README.md` for the full file-by-file mapping between that paper's sections/figures/tables/claims and the code and data that produced them.

## Evolutionary Search with OpenEvolve

The evolutionary loop uses [OpenEvolve](https://github.com/codelion/openevolve) to evolve the `generate_candidates` function via LLM-guided mutations. Each paper's campaigns use their own seeds, configs, and evaluators — see that paper's `README.md` for exact launch commands. A representative invocation:

```bash
# CSS evolution
uv run python evolve/run_evolution.py \
    --models anthropic/claude-opus-4-6 openai/gpt-5.2 gemini/gemini-3-pro \
    --iterations 500 --run-name ensemble_v2
```

A LiteLLM-compatible proxy must be running. See [`evolve/config.yaml`](evolve/config.yaml) for full configuration options.

## Tracking & Observability

Every `main.py` run is logged to `results/runs/<run_id>/`. W&B integration is built into the evolutionary search via `--wandb`.

## Citation

If you use this software or its results, please cite the associated papers:

> J. Cruz-Benito, A. W. Cross, D. Kremer, and I. Faro,
> "Evolutionary Discovery of Bivariate Bicycle Codes with LLM-Guided Search,"
> arXiv:2606.02418 [quant-ph], 2026. https://arxiv.org/abs/2606.02418

> J. Cruz-Benito,
> "Large Language Model-Guided Discovery of Weight-Five Bivariate Bicycle Codes,"
> arXiv:2610.06623 [quant-ph], 2026. https://arxiv.org/abs/2610.06623

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

@misc{cruzbenito2026weight5bb,
  title         = {Large Language Model-Guided Discovery of Weight-Five Bivariate Bicycle Codes},
  author        = {Cruz-Benito, Juan},
  year          = {2026},
  eprint        = {2610.06623},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  doi           = {10.48550/arXiv.2610.06623},
  url           = {https://arxiv.org/abs/2610.06623}
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is also provided.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
