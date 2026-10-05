"""Versioned, proof-aware distance-bound schema ("schema v2").

This module is purely additive: it does not modify ``evaluation/evaluator.py``
or ``evaluation/results.py``.  It defines a stricter representation of code
distance than the legacy flat ``result["d"]`` field, one that distinguishes
*proof-backed* bounds from *heuristic* estimates.

Core idea
---------
A v1 result dict (as produced by ``evaluate_candidate``/``evaluate_candidate_
milp``) carries a single ``d`` field that conflates three very different
kinds of evidence:

1. A proven fact (self-dual A==B ⟹ d=2; MILP-proved optimal distance).
2. A proven one-sided bound (MILP infeasibility at weight w ⟹ d ≥ w+1).
3. A heuristic upper bound (BP-OSD/OSD-CS output, or an MILP incumbent that
   was not proven optimal) -- these can and do overestimate/underestimate
   in practice and must never be treated as a lower bound.

Schema v2 splits this into ``d_lower`` (0 if uncertified) and ``d_upper``
(``None`` if no verified witness exists), plus ``bound_sources`` recording
*which* mechanisms produced the current bounds, and ``d_is_exact`` which is
only ever true when the two coincide *and* provenance backs that equality.

THE SINGLE MOST IMPORTANT INVARIANT IN THIS MODULE
---------------------------------------------------
``d_lower == d_upper`` is NEVER, by itself, sufficient to call a result
exact.  Two independent heuristic estimates can easily coincide by chance
(e.g. two BP-OSD batches both returning d=8) without either constituting a
proof.  ``is_exact()`` additionally requires a recognized proof-source label
in ``bound_sources``.  Every code path in this module that raises
``d_lower`` is gated on ``new_source_kind`` for exactly this reason -- see
``merge_bounds``.

Judgment call: the "d<=2, BP-OSD trusted" case
-----------------------------------------------
``evaluation/evaluator.py`` treats a BP-OSD estimate of d<=2 as
``distance_trusted=True`` "because low distances are reliably detected" --
but it does *not* set ``d_is_exact=True`` for that case (only the self-dual
shortcut and the real exact/MILP paths do).  The task instructions offered
two options:

  (a) Treat ``d<=2 and distance_trusted`` as a *de facto* proof
      (``d_lower=d_upper=d``, added to the exactness allow-list), since so
      few low-weight logicals exist that BP-OSD is unlikely to miss one.
  (b) Stay strictly conservative: ``d_lower=0, d_upper=d,
      bound_sources=["bp_osd_heuristic"]`` -- same as any other heuristic
      estimate.

This module implements **option (b)**.  Rationale: the evaluator's own code
never sets ``d_is_exact=True`` for this branch, so treating it as exact here
would be *this module* inventing a stronger claim than the upstream cascade
itself makes. Nothing in the evaluator's docstrings or comments claims d<=2
BP-OSD detection is a proof (only that it's empirically reliable) --
"reliable" and "proven" are different categories, and this module's entire
purpose is being conservative about ``d_lower``.  If a future exhaustive/
Webster verification confirms these cases, they will naturally be upgraded
via ``merge_bounds`` with ``new_source_kind="exhaustive"`` or
``"solver_proof"``.

Other ambiguity resolutions (documented for the record)
---------------------------------------------------------
* ``stage == "symplectic_low_d"`` sets ``result["d_is_exact"] = d_symp <= 2``
  (see ``evaluation/evaluator.py``): a Gaussian-elimination proof that BB
  codes with k>0 have d>=2, not a BP-OSD heuristic. ``build_v2_record``
  recognizes this explicitly (``d_lower=d_upper=d``, tagged
  ``"symplectic_low_d_proof"``, allow-listed for exactness) rather than
  letting it fall through to the generic ``d in (1,2) and distance_trusted``
  bucket -- falling through would silently downgrade a proof the evaluator
  itself asserts (via ``d_is_exact=True``) into an unproven heuristic bound,
  which is exactly the kind of misrepresentation this module exists to
  prevent. This is a distinct code path from the "d<=2, BP-OSD trusted"
  judgment call above: that one applies when the evaluator has NOT set
  ``d_is_exact=True`` (a genuine heuristic estimate); this one applies only
  when it has (a genuine proof), so the two never compete for the same
  result.
* ``build_v2_record``'s ``bound_sources`` keyword argument is interpreted as
  *additional* provenance labels supplied by the caller (e.g. a Webster
  verification label) to be appended after the derived labels -- it does
  not replace or gate the derivation rules above.
* ``migrate_legacy_record``'s "unless an existing exact/proof flag is
  recognized" clause (D4 in the plan) is legacy-migration-specific: it is
  not one of the general ``build_v2_record`` derivation rules (which only
  recognize exactness via ``self_dual_d2``, ``stage=="exact"``,
  ``milp_details["exact"]``, or the flat ``d_is_exact``/``d_is_upper_bound``
  pair below). A bare legacy record that already carries ``d_is_exact=True``
  but whose ``stage`` doesn't match any of those (e.g. an old/malformed
  record with the stage field dropped) is recognized as exact by
  ``migrate_legacy_record`` specifically, tagged with the existing
  ``"exact_milp"`` label (there being no more specific historical
  provenance available -- see ``migrate_legacy_record`` docstring).
* ``evolve/_noncss_distance_worker.py`` (the PBB adaptive distance pipeline
  used by ``evolve/openevolve_evaluator_weight5_pbb.py``) does not use the
  CSS cascade's ``stage``/nested ``milp_details`` vocabulary at all -- it
  sets a flat ``d_is_exact``/``d_is_upper_bound``/``milp_exact`` triple
  instead. ``d_is_exact=True and d_is_upper_bound=False`` means the same
  D2 lower-bound-raising event as the two branches above it (solver-proven
  optimality when ``milp_exact=True``, else completed exhaustive
  enumeration up to the returned weight) and is recognized as such, tagged
  ``"exact_milp"`` or the (already allow-listed, previously unused)
  ``"exhaustive_hash"`` label accordingly. This is not a new judgment call
  weakening D2 -- it is the same rule applied to a second field vocabulary.
"""

from __future__ import annotations

SCHEMA_VERSION = 2

# ── Exceptions ────────────────────────────────────────────────────


class DistanceBoundConflict(Exception):
    """Raised when merging distance bounds would produce ``upper < lower``.

    This is a certification conflict -- e.g. a heuristic upper-bound witness
    contradicts an already-certified lower bound -- and must never be
    resolved automatically. ``merge_bounds`` and ``merge_component_bound``
    only ever *raise* this; the CALLER (e.g.
    ``evaluation.pareto_v2.update_pareto_front_v2``) is responsible for
    catching it and quarantining the offending record.
    """


# ── Monotonic bound merging ──────────────────────────────────────

# Only these source kinds may ever raise a lower bound. Any other kind
# (heuristic BP-OSD/OSD-CS/QDistEvol estimates, MILP incumbents that were
# not proven optimal, timeouts, failed searches, ...) must never move
# d_lower, no matter how large or how "trusted" the new estimate is.
_LOWER_RAISING_SOURCE_KINDS = frozenset({"exhaustive", "solver_proof"})


def merge_bounds(
    old_lower: int,
    old_upper: int | None,
    new_lower: int,
    new_upper: int | None,
    *,
    new_source_kind: str,
) -> tuple[int, int | None]:
    """Monotonically merge a (lower, upper) distance-bound pair.

    ``lower = max(old_lower, new_lower)`` -- but ONLY if ``new_source_kind``
    is ``"exhaustive"`` (enumeration found no logical through weight w, so
    lower may rise to w+1) or ``"solver_proof"`` (MILP/Webster proved
    infeasibility/optimality with its exact flag set). For any other
    ``new_source_kind`` (heuristic estimates, unverified witnesses,
    timeouts, ...), ``new_lower`` is ignored entirely and ``lower`` stays
    ``old_lower`` -- never let a heuristic source raise the lower bound,
    even numerically.

    ``upper = min(old_upper, new_upper)`` (ignoring ``None``) regardless of
    source kind -- a valid witness/counting argument is a valid upper bound
    no matter how it was found, and ``min`` can never raise ``upper`` above
    what is already recorded.

    Raises:
        DistanceBoundConflict: if the resulting ``upper`` is not ``None``
            and is strictly less than the resulting ``lower``. Never
            resolved here -- propagates to the caller.
    """
    if new_source_kind in _LOWER_RAISING_SOURCE_KINDS:
        lower = max(old_lower, new_lower)
    else:
        lower = old_lower

    uppers = [u for u in (old_upper, new_upper) if u is not None]
    upper = min(uppers) if uppers else None

    if upper is not None and upper < lower:
        raise DistanceBoundConflict(
            "Merging distance bounds would produce upper < lower: "
            f"old=(lower={old_lower}, upper={old_upper}), "
            f"new=(lower={new_lower}, upper={new_upper}, "
            f"source_kind={new_source_kind!r}) -> "
            f"resulting (lower={lower}, upper={upper})"
        )

    return lower, upper


# ── CSS component (X/Z) bounds ───────────────────────────────────

COMPONENT_X = "X"
COMPONENT_Z = "Z"
COMPONENT_KEYS = (COMPONENT_X, COMPONENT_Z)


def empty_component_bounds() -> dict:
    """Return a fresh, empty CSS X/Z component-bounds structure."""
    return {
        COMPONENT_X: {"lower": 0, "upper": None, "sources": []},
        COMPONENT_Z: {"lower": 0, "upper": None, "sources": []},
    }


def merge_component_bound(
    component_bounds: dict,
    component: str,
    new_lower: int,
    new_upper: int | None,
    *,
    source_kind: str,
    source_label: str,
) -> dict:
    """Return an updated copy of ``component_bounds`` with one component merged.

    Merges ``(new_lower, new_upper)`` into ``component_bounds[component]``
    via :func:`merge_bounds`, appending ``source_label`` to that
    component's ``sources`` list. Does not mutate the input dict.

    Raises:
        DistanceBoundConflict: propagated verbatim from ``merge_bounds`` --
            not caught here.
    """
    updated = {
        key: {
            "lower": val["lower"],
            "upper": val["upper"],
            "sources": list(val["sources"]),
        }
        for key, val in component_bounds.items()
    }

    comp = updated[component]
    lower, upper = merge_bounds(
        comp["lower"], comp["upper"], new_lower, new_upper,
        new_source_kind=source_kind,
    )
    comp["lower"] = lower
    comp["upper"] = upper
    comp["sources"].append(source_label)

    return updated


def aggregate_component_bounds(component_bounds: dict) -> tuple[int, int | None]:
    """CSS-only aggregation of separate X/Z component bounds into (lower, upper).

    ``agg_lower = min(X.lower, Z.lower)`` -- the quantum distance can be no
    larger than the smaller of the two component lower bounds.
    ``agg_upper = min of the non-null uppers among X/Z, or None if both are
    null`` -- must not crash when one or both uppers are null.
    """
    x = component_bounds[COMPONENT_X]
    z = component_bounds[COMPONENT_Z]

    agg_lower = min(x["lower"], z["lower"])

    uppers = [u for u in (x["upper"], z["upper"]) if u is not None]
    agg_upper = min(uppers) if uppers else None

    return agg_lower, agg_upper


# ── Exactness, FOM ────────────────────────────────────────────────

# Closed allow-list of provenance labels that may back an exactness claim.
# Equality of d_lower and d_upper is NEVER sufficient by itself -- see the
# module docstring's "single most important invariant" section.
EXACT_SOURCE_ALLOWLIST = frozenset({
    "exact_milp",
    "exact_enumeration",
    "exhaustive_hash",
    "self_dual_proof",
    "webster_proven_exact",
    "symplectic_low_d_proof",
})


def is_exact(d_lower: int, d_upper: int | None, bound_sources: list[str]) -> bool:
    """True only if the bounds coincide AND a recognized proof source backs it.

    Equality of ``d_lower`` and ``d_upper`` alone is NEVER sufficient: two
    independent heuristic estimates can coincide by chance. At least one
    entry of ``bound_sources`` must be in :data:`EXACT_SOURCE_ALLOWLIST`.
    """
    if d_upper is None:
        return False
    if d_lower != d_upper:
        return False
    return any(s in EXACT_SOURCE_ALLOWLIST for s in bound_sources)


def certified_fom(k: int, n: int, d_lower: int) -> float | None:
    """``k*d_lower**2/n`` if ``d_lower > 0`` (and ``n`` is valid), else ``None``."""
    if d_lower <= 0 or n <= 0:
        return None
    return k * d_lower * d_lower / n


def exploratory_fom(k: int, n: int, d_upper: int | None) -> float | None:
    """``k*d_upper**2/n`` if ``d_upper`` is not ``None`` (and ``n`` is valid), else ``None``."""
    if d_upper is None or n <= 0:
        return None
    return k * d_upper * d_upper / n


# ── Source-kind classification (for merging whole v2 records) ───

# Maps a bound_sources provenance label to the merge_bounds "source_kind"
# it represents. Used by evaluation.pareto_v2 when merging two already-
# built v2 records (which carry a *list* of labels) rather than a single
# fresh measurement. Not part of the plan's literal spec for this module,
# but needed to bridge "list of provenance labels" -> "single source_kind"
# for the record-level merge in pareto_v2 -- documented here as one of the
# ambiguity resolutions.
_SOURCE_KIND_BY_LABEL = {
    "exact_milp": "solver_proof",
    "exact_enumeration": "exhaustive",
    "self_dual_proof": "solver_proof",
    # Generic placeholder for a solver-proven lower bound, used only by
    # synthetic tests exercising the "solver_proof" merge/component-label
    # machinery -- no current producer emits it. (Formerly named
    # "milp_infeasible_early_stop" after the MILP-timeout-implies-a-lower-
    # bound inference that turned out to be invalid; renamed since that
    # inference is not what this placeholder is meant to stand for.)
    "solver_proven_lower_bound": "solver_proof",
    "webster_proven_exact": "solver_proof",
    "exhaustive_hash": "exhaustive",
    "symplectic_low_d_proof": "solver_proof",
}


def source_kind_for_labels(bound_sources: list[str]) -> str:
    """Classify a bound_sources label list into a single merge_bounds source_kind.

    Returns the strongest recognized kind ("solver_proof" or "exhaustive")
    found among the labels, defaulting to the conservative "heuristic" if
    none match (which causes merge_bounds to ignore any lower-bound
    contribution from this record).
    """
    for label in bound_sources or []:
        kind = _SOURCE_KIND_BY_LABEL.get(label)
        if kind is not None:
            return kind
    return "heuristic"


# ── v1 -> v2 record construction ─────────────────────────────────

# Stages where no distance was even attempted (k==0 or below threshold, or
# the candidate never got past validation/construction). These carry no
# bound_sources at all -- there is nothing to attribute a bound to.
_REJECTED_STAGES = frozenset({
    "rejected", "invalid", "construction_error", "k_zero", "k_low",
})


def _milp_sweep_complete(milp_details: dict) -> bool:
    """Whether the record has explicit evidence every logical was checked.

    Early-stopped MILP runs could historically report ``exact=True`` after
    checking only some objectives; such a result is an upper bound only.
    Every current producer of ``milp_details`` (``evaluation/distance_milp.py``)
    always populates both count fields, so requiring them here does not
    downgrade any record this codebase writes today -- it only refuses to
    trust ``exact=True`` from a record that omits the evidence, rather than
    defaulting an unrecognized shape to "trusted".
    """
    checked = milp_details.get("num_logicals_checked")
    total = milp_details.get("total_logicals")
    if checked is None or total is None:
        return False
    # Equality, not >=: checked > total is a malformed record (more
    # objectives claimed checked than the sweep even has), and must fail
    # closed rather than be silently accepted as a complete sweep.
    return checked == total


def build_v2_record(
    result: dict,
    *,
    bound_sources: list[str] | None = None,
    component_bounds: dict | None = None,
) -> dict:
    """Derive a schema-v2 record from a v1-shaped evaluator result dict.

    All keys of ``result`` are passed through unchanged; the following are
    added/overwritten: ``distance_schema_version``, ``d_lower``, ``d_upper``,
    ``d_is_exact`` (recomputed via :func:`is_exact`, ignoring any stale
    ``d_is_exact`` already in ``result``), ``bound_sources``,
    ``component_bounds``, and (only when applicable) ``certified_fom`` /
    ``exploratory_fom``.

    ``bound_sources`` (the keyword argument, not the output key) is an
    optional list of EXTRA provenance labels supplied by the caller (e.g. a
    Webster verification tag) which get appended after the labels this
    function derives from ``result`` -- see module docstring.

    See the module docstring for the full derivation-rule rationale,
    including the (b) conservative judgment call for ``d<=2`` BP-OSD
    "trusted" results.
    """
    d = result.get("d", 0)
    stage = result.get("stage")
    milp_details = result.get("milp_details") or {}

    if stage == "self_dual_d2":
        d_lower, d_upper = 2, 2
        derived_sources = ["self_dual_proof"]
    elif result.get("d_is_exact") is True and stage == "exact" and not milp_details:
        # stage=="exact" is exclusively evaluator.py's brute-force
        # compute_distance_exact path, which never attaches milp_details --
        # the `not milp_details` guard is defensive: a record shaped with
        # both (which no current producer emits) falls through to the MILP
        # branch below instead of bypassing its sweep-completeness check.
        # Tagged with its own label (not "exact_milp") because this proof is
        # exhaustive enumeration, not a MILP solver certificate.
        d_lower = d_upper = d
        derived_sources = ["exact_enumeration"]
    elif milp_details.get("exact") is True and _milp_sweep_complete(milp_details):
        d_lower = d_upper = d
        derived_sources = ["exact_milp"]
    elif result.get("d_is_exact") is True and result.get("d_is_upper_bound") is False:
        # evolve/_noncss_distance_worker.py (the PBB adaptive distance pipeline)
        # uses a flat d_is_exact/d_is_upper_bound/milp_exact vocabulary instead
        # of stage/milp_details -- d_is_exact=True with d_is_upper_bound=False
        # is that pipeline's way of expressing the same D2 lower-bound-raising
        # events as the two branches above: either completed exhaustive
        # enumeration (d_method="exact_w{d}", hash-based) or solver-proven
        # optimality (milp_exact=True). Neither the CSS cascade nor the legacy
        # MILP path ever sets d_is_upper_bound, so this branch cannot fire on
        # their results.
        d_lower = d_upper = d
        derived_sources = ["exact_milp"] if result.get("milp_exact") else ["exhaustive_hash"]
    elif stage == "milp_promising_timeout":
        # The MILP found no feasible solution before timing out.  That proves
        # nothing about small logicals, so no lower bound (and no witness).
        # Legacy evaluator records stored d_lower_bound=early_stop+1 here as
        # if it were certified; it never was.
        d_lower, d_upper = 0, None
        derived_sources = []
    elif stage == "symplectic_low_d" and result.get("d_is_exact") is True:
        # Gaussian-elimination proof (BB codes with k>0 have d>=2), not a
        # BP-OSD heuristic -- see module docstring. Must be checked before
        # the generic "d in (1,2) and distance_trusted" bucket below, or a
        # result the evaluator itself marked d_is_exact=True would be
        # silently downgraded to an unproven heuristic bound.
        d_lower, d_upper = d, d
        derived_sources = ["symplectic_low_d_proof"]
    elif d in (1, 2) and result.get("distance_trusted") is True:
        # Judgment call (b): conservative. See module docstring.
        d_lower, d_upper = 0, d
        derived_sources = ["bp_osd_heuristic"]
    elif stage in _REJECTED_STAGES:
        d_lower, d_upper = 0, None
        derived_sources = []
    else:
        # Ordinary heuristic estimate (BP-OSD quick/refined/OSD-CS, MILP
        # incumbent, PBB equivalents, or any other/unknown stage) -- an
        # upper bound only, never a lower bound.
        d_lower = 0
        d_upper = d if d else None
        derived_sources = ["bp_osd_heuristic"]

    all_sources = list(derived_sources)
    if bound_sources:
        all_sources.extend(bound_sources)

    v2 = dict(result)
    v2["distance_schema_version"] = SCHEMA_VERSION
    v2["d_lower"] = d_lower
    v2["d_upper"] = d_upper
    v2["bound_sources"] = all_sources
    v2["d_is_exact"] = is_exact(d_lower, d_upper, all_sources)
    v2["component_bounds"] = component_bounds

    n = result.get("n", 0)
    k = result.get("k", 0)

    cf = certified_fom(k, n, d_lower)
    if cf is not None:
        v2["certified_fom"] = cf

    ef = exploratory_fom(k, n, d_upper)
    if ef is not None:
        v2["exploratory_fom"] = ef

    return v2


def migrate_legacy_record(old_result: dict) -> dict:
    """Migrate a bare legacy (v1) record into schema v2.

    Reuses :func:`build_v2_record`'s derivation logic unchanged, then:

    1. Tags the resulting ``bound_sources`` with an added
       ``"legacy_migrated"`` marker so provenance of a migrated-vs-fresh
       record is distinguishable later.
    2. Recognizes a pre-existing ``d_is_exact=True`` flag on ``old_result``
       even when its ``stage`` doesn't match one of ``build_v2_record``'s
       specific exactness rules (e.g. an old/malformed record whose
       ``stage`` field was dropped or never matched
       ``"exact"``/``"self_dual_d2"``). This is the plan's explicit D4
       exception: "unless an existing exact/proof flag is recognized" --
       it is legacy-migration-specific, not a general
       ``build_v2_record`` rule (a genuinely fresh v1 result always has a
       stage consistent with how ``d_is_exact`` got set, so this path is a
       legacy-only safety net). The label used is the existing
       ``"exact_milp"`` allow-listed source, since no more specific
       historical provenance is recoverable from a bare ``d_is_exact``
       flag -- documented ambiguity resolution, see module docstring.

    A record with only a bare ``d`` and no ``d_is_exact``/recognizable
    stage migrates as *exploratory* (``d_lower=0, d_upper=d``) -- schema v2
    never infers exactness from a historical numeric distance alone.
    """
    v2 = build_v2_record(old_result)

    if (
        not v2["d_is_exact"]
        and old_result.get("d_is_exact") is True
        and old_result.get("d", 0) > 0
    ):
        d = old_result["d"]
        v2["d_lower"] = d
        v2["d_upper"] = d
        sources = list(v2["bound_sources"])
        if "exact_milp" not in sources:
            sources.append("exact_milp")
        v2["bound_sources"] = sources
        v2["d_is_exact"] = is_exact(d, d, sources)

        n = old_result.get("n", 0)
        k = old_result.get("k", 0)
        cf = certified_fom(k, n, d)
        if cf is not None:
            v2["certified_fom"] = cf
        else:
            v2.pop("certified_fom", None)
        ef = exploratory_fom(k, n, d)
        if ef is not None:
            v2["exploratory_fom"] = ef
        else:
            v2.pop("exploratory_fom", None)

    v2["bound_sources"] = list(v2["bound_sources"]) + ["legacy_migrated"]
    return v2


# ── Webster status mapping (pure lookup, no I/O) ─────────────────

_WEBSTER_STATUS_EFFECTS = {
    "proven_exact": "raises_lower_after_verification",
    "incumbent": "upper_only",
    "heuristic": "upper_only",
    "timeout": "log_only",
    "failed": "log_only",
}


def webster_status_effect(status: str) -> str:
    """Pure lookup of the schema effect of a Webster ``codedistance`` status.

    Makes no actual Webster calls. Raises ``ValueError`` for any
    unrecognized status string.
    """
    try:
        return _WEBSTER_STATUS_EFFECTS[status]
    except KeyError:
        raise ValueError(f"Unrecognized Webster status: {status!r}") from None
