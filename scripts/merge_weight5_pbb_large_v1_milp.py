#!/usr/bin/env python3
"""Merge deep-MILP-verified weight5_pbb_large_v1 winners into pareto_front_v2.json.

Reads results/weight5_pbb_large_v1_deep_milp_verify.jsonl (produced by running
scripts/verify_weight5_pbb_small_milp.py -- misnamed but generic, taking
--input/--output as CLI args -- against results/weight5_pbb_large_v1_winners.jsonl
with --timeout 1800). Unlike the small campaign, none of these 51 codes have a
d_method starting with exact_w: the campaign's own worker
(evolve/_noncss_distance_worker.py) only hash-searches up to weight 4 for
n>216, so every one of them fell through to d_method="milp+bposd" with no
prior exact certification at all. Their foms (0.28-2.78) are also all well
under openevolve_evaluator_weight5_pbb.py's save_fom_threshold_noncss() gate
(default 4.0), so unlike some weight5_pbb_small_v1 codes, none of these 51
have ANY existing record -- exact or heuristic -- in the shared v2 archive yet
(confirmed: zero key overlap against the current all_records).

IMPORTANT: for this campaign the verify script's "hash_d" field name is a
misnomer -- it is just c["d"]/d_bposd carried through unchanged, NOT an
exhaustive hash-search result (that's only true for weight5_pbb_small_v1,
whose own worker actually ran the hash search; see
merge_weight5_pbb_small_v1_milp.py). So "agrees_with_hash" here really means
"does the MILP-proven distance match the original BP-OSD estimate" -- it is
not a second independent-method cross-check the way it is for the small
campaign, and is not treated as extra provenance here.

Certification per code: MILP proved every one of its 2k logical operators
optimal (logicals_optimal == total_logicals) iff milp_exact is True, in which
case milp_d IS the exact distance. Otherwise only an upper-bound witness is
available: the deep MILP run found actual codeword-logical pairs at weight
milp_d for every code (never None, and never looser than the original
d_bposd across all 51 records), so min(milp_d, d_bposd) is always a sound --
and for 5 of the 51, strictly tighter -- upper bound than the original
discovery-time estimate. Both cases route through build_v2_record after
synthesizing the flat d_is_exact/d_is_upper_bound vocabulary it recognizes;
non-exact records get no extra bound_sources tag, matching how
weight5_css_large_v1's non-exact MILP incumbents were merged
(merge_weight5_css_large_v1_milp.py).
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.distance_schema import build_v2_record
from evaluation.pareto_v2 import update_pareto_front_v2, load_pareto_v2

SRC = "results/weight5_pbb_large_v1_deep_milp_verify.jsonl"

records = []
with open(SRC) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f"Loaded {len(records)} verified results from {SRC}")

v2_records = []
n_exact = 0
for r in records:
    is_exact = r["logicals_optimal"] == r["total_logicals"]
    milp_d = r.get("milp_d")
    d_bposd = r.get("d_bposd", r.get("d"))
    d_ub = min(milp_d, d_bposd) if milp_d is not None else d_bposd

    synth = dict(r)
    synth["d"] = d_ub
    synth["d_is_exact"] = is_exact
    synth["d_is_upper_bound"] = not is_exact
    synth["milp_exact"] = bool(r.get("milp_exact"))

    v2_records.append(build_v2_record(synth))
    if is_exact:
        n_exact += 1

print(
    f"Built {len(v2_records)} v2 records: {n_exact} exact (MILP-proven), "
    f"{len(v2_records) - n_exact} upper-bound-only"
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
