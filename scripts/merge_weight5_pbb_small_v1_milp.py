#!/usr/bin/env python3
"""Merge MILP-verified weight5_pbb_small_v1 winners into pareto_front_v2.json.

Reads results/weight5_pbb_small_v1_deep_milp_verify.jsonl (produced by
scripts/verify_weight5_pbb_small_milp.py + verify_weight5_pbb_small_milp_deep.py),
translates the campaign's on-disk PBB vocabulary (d_method/milp_exact) into the
d_is_exact/d_is_upper_bound flags evaluation.distance_schema.build_v2_record
recognizes -- the live evaluator sets those flags in memory
(evolve/_noncss_distance_worker.py), but evolve/openevolve_evaluator_weight5_pbb.py's
_log_code_jsonl() drops them before persisting to disk, so build_v2_record cannot
recognize these records unmodified. Builds a schema-v2 record for each via
build_v2_record(), and merges them into the shared three-tier archive via
update_pareto_front_v2().

These 30 codes never reached the in-loop v2 merge during the campaign itself: their
FOM (~1.0) never crossed openevolve_evaluator_weight5_pbb.py's
save_fom_threshold_noncss() gate (default 4.0), which skips the entire save block
(legacy + v2) for every candidate below it.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.distance_schema import build_v2_record
from evaluation.pareto_v2 import update_pareto_front_v2, load_pareto_v2

SRC = "results/weight5_pbb_small_v1_deep_milp_verify.jsonl"

records = []
with open(SRC) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f"Loaded {len(records)} verified results from {SRC}")

v2_records = []
skipped = []
for r in records:
    d_method = r.get("d_method") or ""
    hash_d = r.get("hash_d")
    milp_d = r.get("milp_d")
    certified = d_method.startswith("exact_w") and hash_d is not None and hash_d == milp_d
    if not certified:
        skipped.append(r)
        continue

    synth = dict(r)
    synth["d"] = hash_d
    synth["d_is_exact"] = True
    synth["d_is_upper_bound"] = False
    milp_exact = bool(r.get("milp_exact"))
    synth["milp_exact"] = milp_exact

    extra_sources = ["exhaustive_hash"] if milp_exact else None
    v2_records.append(build_v2_record(synth, bound_sources=extra_sources))

print(f"Built {len(v2_records)} v2 records; skipped {len(skipped)} uncertified/contradicting")
for r in skipped:
    print(
        f"  SKIPPED: ({r.get('ell')},{r.get('m')}) d_method={r.get('d_method')!r} "
        f"hash_d={r.get('hash_d')} milp_d={r.get('milp_d')}"
    )

before = load_pareto_v2()
print(
    f"Before merge: exact_front={len(before['exact_front'])} "
    f"lower_bound_front={len(before['lower_bound_front'])} "
    f"exploratory_witnesses={len(before['exploratory_witnesses'])} "
    f"all_records={len(before['all_records'])} "
    f"quarantined={len(before.get('quarantined', []))}"
)

for r in v2_records:
    print(
        f"  [[{r['n']},{r['k']},d_lower={r['d_lower']},d_upper={r['d_upper']}]] "
        f"exact={r['d_is_exact']} sources={r['bound_sources']}"
    )

after = update_pareto_front_v2(v2_records)
print(
    f"After merge: exact_front={len(after['exact_front'])} "
    f"lower_bound_front={len(after['lower_bound_front'])} "
    f"exploratory_witnesses={len(after['exploratory_witnesses'])} "
    f"all_records={len(after['all_records'])} "
    f"quarantined={len(after.get('quarantined', []))}"
)
