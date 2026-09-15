# LC equivalence Theorem 3: known issue with the necessity direction

Theorem 3 in the paper claims an "if and only if" characterization of
when a non-CSS PBB code is LC-equivalent to a CSS code, in terms of a
*uniform* per-block Clifford S-pattern (the same S-gate choice applied
to every qubit in a block).

The sufficiency direction ("uniform pattern found => LC-CSS") holds.
The necessity direction ("LC-CSS => a uniform pattern exists", i.e.
"only if") is false in general: counterexamples exist at (ell=2, m=4)
with non-uniform S-patterns that achieve CSS equivalence but no
uniform pattern does.

The paper should state this as a **sufficient condition**, not an
"if and only if".

## Where this is checked computationally

- `evaluation/clifford_equivalence.py:680` (`verify_uniform_reduction_exact`)
  builds the exact GF(2) linear system for whether non-uniform S-patterns
  can produce an LC-CSS equivalence beyond what uniform patterns find.
- `evaluation/clifford_equivalence.py:339` (`is_lc_equivalent_css_group`)
  and `evaluation/clifford_equivalence.py:790` (`verify_not_lc_css`) are
  the group-level and combined (36 uniform assignments + exact non-uniform
  system) checks used against the full 368-code catalog.
- `tests/test_clifford_equivalence.py` (`TestUniformReduction`) exercises
  `verify_uniform_reduction_exact` against synthetic and real codes and
  cross-checks it against `is_lc_equivalent_css_group`
  (`test_agrees_with_group_check`), including running the exact reduction
  over the full published catalog
  (`test_all_368_uniform_reduction_holds`).

This class of proposition should be backed by computational
verification (as above) rather than a hand proof.
