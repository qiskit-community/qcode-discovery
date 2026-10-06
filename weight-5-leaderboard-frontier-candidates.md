# Weight-5 Leaderboard Frontier Candidates

Report date: 2026-09-09 (analysis performed 2026-09-08).
Branch: `weight5-campaigns`.

## 1. Context

On 2026-09-07 four weight-5 codes discovered by this repo's evolutionary
campaigns were submitted to the Unitary Foundation `qldpc-challenge`
leaderboard:

| Code | PR | Status |
|---|---|---|
| [[90,4,9]] | [#918](https://github.com/unitaryfoundation/qldpc-challenge/pull/918) | merged |
| [[180,4,14]] | [#916](https://github.com/unitaryfoundation/qldpc-challenge/pull/916) | merged |
| [[140,6,10]] | [#915](https://github.com/unitaryfoundation/qldpc-challenge/pull/915) | merged |
| [[96,4,10]] | [#914](https://github.com/unitaryfoundation/qldpc-challenge/pull/914) | merged |

This report answers: **are there other codes in this branch's results that
could be additional, non-dominated leaderboard entries, and are they free of
overlap with literature already surveyed in `weight-5-codes-survey.md` /
`paper/2610.06623/weight5_paper.tex`?**

Method: (a) direct inspection of `results/weight5_publication_catalogue.jsonl`,
`results/weight5_css_upper_bound_witnesses.jsonl`, and
`paper/2610.06623/weight5_paper.tex`; (b) two background research agents — one
cataloguing candidates against the paper's own literature-exclusion findings,
one surveying the live `qldpc-challenge` board's rules and current frontier;
(c) independent re-verification of both agents' key claims by re-running the
board's own Pareto-domination rule against a freshly pulled local clone of
`/root/qldpc-challenge` (531 entries as of 2026-09-08).

## 2. Literature caveats on already-submitted codes (informational, no action needed)

From `paper/2610.06623/weight5_paper.tex`, Table `tab:headline-provenance`, and
`results/weight5_lin_pryadko_audit.json`:

- **[[90,4,9]]** (PR #918) is FOM-dominated by Lin-Pryadko's exact archived
  **[[78,4,9]]** (FOM 4.15 vs. our 3.60). This is not a domination *on the
  qldpc-challenge board* (no such code is currently seeded there) — it's a
  fact about the wider literature, already implicitly acknowledged by the
  paper's own comparison section.
- **[[96,4,10]]** (PR #914) is a confirmed rediscovery: `summary.archive_96_is_rediscovery: true`
  in `weight5_lin_pryadko_audit.json`, matching archive row `LP-96-4-10`
  (construction `diswtabelian_wt5wtL2_order48_k4.txt`, C₄₈ ambient group).
  This was already disclosed in the paper's provenance table.

Neither requires further action; both were already known/disclosed at
submission time.

## 3. Candidates checked and rejected

### [[216,4,10]] (non-CSS/PBB, exact, FOM 1.85)

This is the paper's own headline PBB result (fully exact-certified,
M+component evidence tier — two non-isomorphic component classes,
`W5PBB-216-4-10-36c5ce3b` and `W5PBB-216-4-10-ae4cc6ad`, ℓ=18, m=6, each
BLISS-matched to a component of an exact-MILP-certified [[432,8,10]] parent).

Despite the strong evidentiary tier, direct verification against the live
board's own Pareto rule shows it is **dominated**:

```
[[216,4,10]] non-CSS exact -> dominators:
  codes/154-6-11.json  (n=154, k=6, d=11, w=5)
  codes/182-6-12.json  (n=182, k=6, d=12, w=5)
  codes/180-4-14.json  (n=180, k=4, d=14, w=5)   <- our own PR #916
  codes/96-4-10.json   (n=96,  k=4, d=10, w=5)   <- our own PR #914
  codes/140-6-10.json  (n=140, k=6, d=10, w=5)   <- our own PR #915
```

Three of the five dominators are our own already-submitted codes. **Not a
frontier candidate**, despite being exact.

### [[120,4,10]] (new literature-exclusion finding)

`weight5_paper.tex` (~lines 624–656): a quasi-abelian lifted-product
reduction identifies this code with an HGP code over 𝔽₂[C₅] (Lin-Pryadko
construction). **Excluded** as a literature rediscovery. (Note: this
supersedes an earlier internal note treating [[120,4,10]] as a clean
discovery — the paper's more recent algebraic-classification analysis
caught the reduction that the original campaign missed.)

### Self-dominated / redundant candidates

These are technically non-dominated by *external* literature, but are
Pareto-dominated by one of the candidates in Section 4 (same n,k, weight-5,
strictly worse d) or by our own submitted codes — not worth separate PRs:

- **[[180,4,13]]** (`CS-6a4be7ad`, exact, FOM 3.756) — dominated by our own
  submitted [[180,4,14]] at identical (n,k).
- **[[420,4,≤21]]** (`CL-170eab7b`, FOM≤4.20) and three **[[420,4,≤20]]**
  classes (`CL-3ba7a33d`, `CL-68eefaad`, `CL-fb0e1722`, FOM≤3.81) — all
  dominated by [[420,4,≤22]] below (same n, k; lower d).
- **[[432,8,≤14]]** (`CL-a4b8c21d`, FOM≤3.63) — dominated by [[432,8,≤15]]
  below (same n, k; lower d).

## 4. Non-dominated, non-literature-matched candidates

All are CSS, connected (single BLISS component, `class_size=1`), weight-5
(2-term + 3-term support split), n ≤ 432 (within the board's n ≤ 700 cap),
and **not present in the Lin-Pryadko 2BGA archive** (n=420/432 exceeds its
enumeration range: abelian ambient groups through n=100, nonabelian through
n=200) and not flagged anywhere in the paper's rediscovery table
(`explicit_bb_component_matches: []` in the catalogue for every one of them).

Domination was checked directly against all 531 current `codes/*.json`
entries using the board's own rule (`site/build.py:pareto()`): a candidate
(n,k,d,w) is dominated by any board entry (n',k',d',w') with n'≤n, k'≥k,
d'≥d, w'≤w, and at least one strict inequality.

| Code | ℓ,m | A | B | FOM≤ | Evidence tier | Presentation id(s) / class | Board dominators |
|---|---|---|---|---|---|---|---|
| **[[432,4,≤23]]** | 18,12 | 1+x⁴y³ | 1+x⁵y¹¹+x¹⁰y | 4.90 | **W** (explicit witness, tightened 24→23 on 2026-09-08) | `CL-f5754da6` / `TC-9efa3eab1a` | NONE |
| [[420,4,≤22]] (class A) | 15,14 | 1+x³y | 1+x¹⁰y⁶+x¹¹y² | 4.61 | I (MILP incumbent; informally stress-tested 8000 trials, held) | `CL-9fbce668` / `TC-5f95e88575` | NONE |
| [[420,4,≤22]] (class B, non-isomorphic to A) | 15,14 | 1+x⁴y³+x⁸y¹¹ | 1+x³y³ | 4.61 | I (same stress-test tier) | `CL-c271b632` / `TC-482c979e17` | NONE |
| [[432,8,≤15]] | 18,12 | 1+x⁴y⁶ | 1+xy⁵+x⁵y¹⁰ | 4.17 | I (same stress-test tier) | `CL-ab07a1d1` / `TC-b6a8f8ca19` | NONE |
| [[420,8,≤13]] (2 non-isomorphic classes) | 15,14 | — | — | 3.22 | **W** (witness-corrected 2026-09-08, same batch as top entry) | `CL-4a95cc39`, `CL-d192c3d1` | NONE |
| [[420,6,≤16]] | 15,14 | 1+x⁴y⁷ | 1+x¹⁴y²+x¹⁴y⁶ | 3.66 | I — **not yet stress-tested at all**, newly surfaced | `CL-dcca76b0` | NONE |

Every candidate's certified **lower** bound is d=5 (`css_low_weight_audit`,
algorithm `css_weight_le_4_syndrome_quotient_mitm_v1`, result
`excluded_through_weight_4`). The upper bounds above are the only distance
evidence beyond that floor.

Direct domination-check output (re-run 2026-09-08 against a freshly pulled
`/root/qldpc-challenge`, 531 entries):

```
[[432,4,<=23]] CL-f5754da6 (W)              -> dominators: NONE
[[420,4,<=22]] CL-9fbce668 (I)               -> dominators: NONE
[[420,4,<=22]] CL-c271b632 (I)               -> dominators: NONE
[[432,8,<=15]] CL-ab07a1d1 (I)               -> dominators: NONE
[[420,6,<=16]] CL-dcca76b0 (I)                -> dominators: NONE
[[420,8,<=13]] CL-4a95cc39/d192c3d1 (W)       -> dominators: NONE
```

Existing weight-5 board entries besides our four submissions (for context —
none of them come close to n=420/432): [[18,4,3]], [[40,10,4]], [[40,6,4]],
[[45,9,3]], [[60,4,8]], [[154,6,11]], [[182,6,12]], [[676,71,4]].

## 5. The critical evidentiary caveat

Unlike the 4 already-submitted codes (which the paper certifies as **exact**,
evidence tier M or M+component), **none of the six candidates above have an
exact certified distance.** Each is an open interval `[5, upper]` where the
upper bound comes from one of two methods:

- **W** — an explicit logical operator was reconstructed via randomized
  GF(2) kernel/permutation search and independently verified against the
  syndrome and row-space constraints. This proves `d ≤ upper`; it does
  **not** prove `d = upper`.
- **I** — a MILP run found an incumbent solution of that weight within its
  time budget (1800s/logical, then a deeper 14400s/logical pass) but did
  **not** prove optimality (0 of 8–16 logicals proved optimal in every case
  checked). This is weaker than W: it is only the best solution MILP
  happened to find, not a verified-minimal witness.

This distinction matters because of what happened during this same
investigation: on 2026-09-08, an audit script
(`scripts/audit_weight5_css_upper_bound_witnesses.py`,
`results/weight5_css_upper_bound_witnesses.jsonl`) ran the W-style witness
search against 10 other presentation classes that had only "B"-tier
(raw BP-OSD decoder heuristic, no witness) or one "I"-tier upper bound.
**9 of 10 "B"-tier bounds collapsed by 30–60%** (e.g. claimed upper 22 → 14,
23 → 12, 14 → 9), while the single "I"-tier bound in that batch only
tightened slightly (24 → 23 — this is exactly the [[432,4,≤23]] candidate
above). A separate, earlier informal stress test (400 then 8000 trials) on
5 other "I"-tier candidates found 4 held exactly and 1 tightened slightly —
consistent with "I" being materially more trustworthy than "B", but still
not proof of optimality.

**Practical implication:** the board's own `verify/certify.py` explicitly
supports `confidence: "upper_bound"` submissions (not just exact), so
submitting these as-is with honest confidence labeling is legitimate
practice on this board. However, given this project's own repeated
experience that unverified upper bounds can collapse hard, the following
hardening pass is recommended before submission:

1. **[[432,4,≤23]]** and the two **[[420,8,≤13]]** classes — already
   witness-hardened (W-tier) as of 2026-09-08. Ready to prepare as PRs.
2. **[[420,4,≤22]]** (both classes) and **[[432,8,≤15]]** — informally
   stress-tested (8000 trials) but not yet run through the formal witness
   audit script. Recommend running it before submitting, though prior
   informal results are reassuring.
3. **[[420,6,≤16]]** — newly surfaced by this investigation, has had *no*
   independent stress-testing at all. Recommend hardening first; treat its
   upper bound of 16 as the least trustworthy of the six.

## 6. Recommendation

Prepare submission PRs in the order of readiness above, starting with
[[432,4,≤23]] and the two [[420,8,≤13]] classes. For the [[420,4,≤22]] pair
(same n,k, non-isomorphic constructions), consider submitting only the
better-hardened one and mentioning the other as an alternate,
isomorphism-inequivalent construction in the research note, since both
occupy the same point in the board's Pareto space.

No branches, commits, or PRs should be created against
`/root/qldpc-challenge` without explicit sign-off, per this project's
standing risk-tolerance practice for actions visible to others.
