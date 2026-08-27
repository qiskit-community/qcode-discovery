#!/usr/bin/env python3
"""Merge MILP-verified weight5_css_small_v1 exact_timeout results into pareto_front_v2.json.

Reads results/weight5_css_small_v1_milp_verified.jsonl (produced by
scripts/verify_weight5_css_small_v1_milp.py), builds a schema-v2 record for
each via build_v2_record(), and merges them into the shared three-tier
archive via update_pareto_front_v2().
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.distance_schema import build_v2_record
from evaluation.pareto_v2 import update_pareto_front_v2, load_pareto_v2

SRC = "results/weight5_css_small_v1_milp_verified.jsonl"

results = []
with open(SRC) as f:
    for line in f:
        line = line.strip()
        if line:
            results.append(json.loads(line))

print(f"Loaded {len(results)} verified results from {SRC}")

before = load_pareto_v2()
print(
    f"Before merge: exact_front={len(before['exact_front'])} "
    f"lower_bound_front={len(before['lower_bound_front'])} "
    f"exploratory_witnesses={len(before['exploratory_witnesses'])} "
    f"all_records={len(before['all_records'])} "
    f"quarantined={len(before['quarantined'])}"
)

v2_records = [build_v2_record(r) for r in results if r.get("d", 0) > 0]
print(f"Built {len(v2_records)} v2 records (d>0)")

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
    f"quarantined={len(after['quarantined'])}"
)
