# Weight-five uniform-control pilot

This directory is a reproducibility smoke test, not the matched-budget result
for the manuscript.  It ran seed 42 on all six CSS Stage-2 lattices with two
selected candidates per k band (24 distance evaluations total), 20 initial
BP--OSD trials, three 10-trial refinements, and no exact-search or MILP stage.
The production protocol uses the historical per-lattice quotas (6,444
distance evaluations per seed), 1,000 initial trials, three 500-trial
refinements, the 200-trial OSD-CS pass, and the extended MILP verification of
finalists.

The pilot completed selection, distance evaluation, the exhaustive
weight-<=4 audit, Tanner canonicalization, overlap accounting, and summary
generation on every lattice.  It recorded 267 without-replacement draws to
fill 24 pilot slots; 19 decoder estimates were at least five, 17 survived the
independent low-weight audit with certified d>=5, and those contained 13
connected stored-generator Tanner classes.  These tiny-sample rates must not
be used as scientific estimates.

Measured on the local environment, selection took 13.9 CPU-wall seconds,
two-worker distance evaluation 22.0 seconds, and two-worker low-weight
certification 6.6 seconds.  Scaling only the selected count and decoder trial
count already puts a three-seed production run in the multi-day range on two
workers, before the historical 300-second exact attempts and up-to-four-hour
MILP finalist jobs.  The production implementation is therefore resumable
and supports process-level parallelism and per-invocation work caps.

Reproduce this pilot from an empty output directory with:

```bash
MPLCONFIGDIR=/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/run_weight5_uniform_baseline.py \
  --pilot --stage all --profile all --seeds 42 --workers 2 \
  --output-dir results/weight5_uniform_baseline_pilot
```

See `protocol.json` for source hashes, historical gates, exact quotas, search
space sizes, software versions, and the explicitly documented replay
differences.  `summary.json` aggregates the per-lattice summaries.
