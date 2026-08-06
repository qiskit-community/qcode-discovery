"""Three-tier, proof-aware Pareto archive (schema v2).

This module is purely additive: it adds a new archive file
(``results/pareto_front_v2.json``) alongside the existing legacy
``results/discovered_codes.json`` / ``results/pareto_front.json`` (which are
never written to here -- only read, and only for one-time migration). It
does not modify ``evaluation/results.py``.

The legacy Pareto front (``evaluation.results.update_pareto_front``) ranks
codes by a single ``d`` field that conflates proof and heuristic. This
module instead maintains three independent fronts, computed from disjoint
subsets of the merged record set, so a weaker tier can never structurally
evict a stronger one:

* ``exact_front`` -- ``d_is_exact=True`` records, non-dominated in
  ``(k/n, d_lower, 1/n)``.
* ``lower_bound_front`` -- ``d_lower>0`` and not exact, non-dominated in
  ``(k/n, d_lower, 1/n)``.
* ``exploratory_witnesses`` -- ``d_lower==0`` and ``d_upper is not None``,
  non-dominated in ``(k/n, d_upper, 1/n)``.

A fourth top-level key, ``quarantined``, collects any per-record merge that
would have produced ``d_upper < d_lower`` (a certification conflict) --
:func:`update_pareto_front_v2` catches ``DistanceBoundConflict`` per record
rather than letting one bad record crash the whole update.

A fifth top-level key, ``all_records``, holds the FULL deduplicated
(by code key) record set the three fronts were computed from -- not just
the current front members. This is load-bearing, not redundant: dominance
is recomputed from scratch on every update, and a record dominated in one
round can become non-dominated in a later round without itself changing
(e.g. the record that was dominating it gets upgraded into a stronger tier
and so drops out of the weaker tier's dominance comparison). If only the
three fronts were persisted, a dominated record would be discarded the
moment it first loses, and could never resurface even when it legitimately
belongs in a front again -- so ``update_pareto_front_v2`` always reloads
``all_records`` (falling back to the union of the three fronts only for
archives written before this field existed).

Code identity uses the exact same key convention as
``evaluation.results._code_key`` (imported directly, not reimplemented) so
the v1 and v2 archives can be cross-referenced by an identical key.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
from pathlib import Path

from evaluation import distance_schema
from evaluation.distance_schema import DistanceBoundConflict
from evaluation.evaluator import compute_fom
from evaluation.results import _code_key, load_codes

logger = logging.getLogger(__name__)

PARETO_V2_PATH = Path(__file__).parent.parent / "results" / "pareto_front_v2.json"


def _default_shape() -> dict:
    return {
        "schema_version": distance_schema.SCHEMA_VERSION,
        "generated_at": None,
        "exact_front": [],
        "lower_bound_front": [],
        "exploratory_witnesses": [],
        "all_records": [],
    }


def load_pareto_v2(filepath: Path | str | None = None) -> dict:
    """Load the v2 three-tier archive, or the empty default shape if absent.

    Args:
        filepath: Path to the v2 archive JSON file. Defaults to
            :data:`PARETO_V2_PATH`.

    Returns:
        The archive dict (``schema_version, generated_at, exact_front,
        lower_bound_front, exploratory_witnesses, all_records`` and, once
        written by :func:`update_pareto_front_v2`, ``quarantined``). Never
        raises on a corrupted file -- logs a warning and returns the empty
        default shape instead, matching ``evaluation.results``'s handling.
    """
    path = Path(filepath) if filepath is not None else PARETO_V2_PATH
    if not path.exists():
        return _default_shape()
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Corrupted v2 pareto front file %s: %s", path, e)
        return _default_shape()


def _display_d(upper: int | None) -> int:
    """The legacy flat ``d`` value to display for a given ``d_upper``.

    The legacy field's original meaning is a witnessed/verified distance
    (an evaluator-measured value or a proven upper bound) -- never a
    certified-but-unwitnessed lower bound. Per the schema's own D2 rule,
    ``d_upper=None`` means "no verified logical witness exists", so a
    lower-bound-only record (``d_upper`` still ``None``) must display
    ``d=0``, not its lower bound: showing the lower bound as `d` would
    misrepresent an unwitnessed proof (we know d>=lower, nothing more) as
    if it were an observed/measured distance.
    """
    return upper if upper is not None else 0


def _normalize_legacy_fields(rec: dict) -> dict:
    """Return a copy of ``rec`` with its legacy ``d``/``fom``/``score``
    fields normalized to agree with its own ``d_lower``/``d_upper``/``n``/
    ``k``.

    Applied identically to a never-before-seen record on first insert (see
    :func:`update_pareto_front_v2`'s ``old_rec is None`` branch) and to a
    freshly merged record (see :func:`_merge_records`), so the same
    logical record always displays the same normalized fields regardless
    of which path it went through -- otherwise the two paths would
    disagree and a no-op re-submission of an already-inserted record would
    change its stored fields, breaking idempotency.

    ``d`` uses :func:`_display_d`. ``fom`` is cheap and unambiguous to
    recompute from ``k*d^2/n`` (the same formula ``certified_fom``/
    ``exploratory_fom`` use, and the field ``_dominance_front`` itself
    sorts by) so it's always kept in sync with the normalized ``d``.
    ``score`` has no single formula -- ``evaluator.py`` sets it very
    differently depending on rejection reason (a k-low penalty, an
    encoding-rate proxy, a tiny "promising" marker, ...) that can't be
    reconstructed from a record alone once its `d` has changed, and
    nothing downstream reads it from the persisted archive, so it's
    dropped rather than left stale.
    """
    out = dict(rec)
    out["d"] = _display_d(out.get("d_upper"))
    out["fom"] = compute_fom(out.get("n", 0), out.get("k", 0), out["d"])
    out.pop("score", None)
    return out


def _merge_records(old_rec: dict, new_rec: dict) -> dict:
    """Merge two v2 records for the same code key.

    Non-distance fields are taken from ``new_rec`` (assumed to be the more
    recent measurement). Distance bounds, component bounds, bound_sources,
    d_is_exact, and the derived FOM fields are recomputed from the merge of
    both records via :mod:`evaluation.distance_schema`.

    Raises:
        DistanceBoundConflict: propagated from
            ``distance_schema.merge_bounds`` -- not caught here (the caller,
            :func:`update_pareto_front_v2`, catches it per-record).
    """
    old_lower = old_rec.get("d_lower", 0)
    old_upper = old_rec.get("d_upper")
    new_lower = new_rec.get("d_lower", 0)
    new_upper = new_rec.get("d_upper")
    new_kind = distance_schema.source_kind_for_labels(new_rec.get("bound_sources", []))

    lower, upper = distance_schema.merge_bounds(
        old_lower, old_upper, new_lower, new_upper, new_source_kind=new_kind,
    )

    sources = list(old_rec.get("bound_sources", []))
    for s in new_rec.get("bound_sources", []):
        if s not in sources:
            sources.append(s)

    component_bounds = old_rec.get("component_bounds")
    new_component_bounds = new_rec.get("component_bounds")
    if new_component_bounds:
        if component_bounds is None:
            component_bounds = distance_schema.empty_component_bounds()
        else:
            # Copy (not mutate) before editing in place below -- matches
            # merge_component_bound's own contract of never mutating its input.
            component_bounds = {
                key: {
                    "lower": val["lower"],
                    "upper": val["upper"],
                    "sources": list(val["sources"]),
                }
                for key, val in component_bounds.items()
            }
        for comp in distance_schema.COMPONENT_KEYS:
            comp_new = new_component_bounds.get(comp)
            if not comp_new:
                continue
            comp_kind = distance_schema.source_kind_for_labels(comp_new.get("sources", []))
            comp_entry = component_bounds[comp]
            comp_lower, comp_upper = distance_schema.merge_bounds(
                comp_entry["lower"], comp_entry["upper"],
                comp_new.get("lower", 0), comp_new.get("upper"),
                new_source_kind=comp_kind,
            )
            comp_entry["lower"] = comp_lower
            comp_entry["upper"] = comp_upper
            # Merge the bound once per component (not once per label via
            # merge_component_bound, which unconditionally appends its label
            # with no dedup -- looping it once per label made a repeated /
            # idempotent update accumulate duplicate labels in `sources`
            # indefinitely). Union in only genuinely new labels, mirroring
            # the top-level bound_sources dedup above, so every distinct
            # incoming proof label is still preserved exactly once.
            for comp_label in (comp_new.get("sources") or ["merged"]):
                if comp_label not in comp_entry["sources"]:
                    comp_entry["sources"].append(comp_label)

    merged = dict(new_rec)
    merged["d_lower"] = lower
    merged["d_upper"] = upper
    merged["bound_sources"] = sources
    merged["component_bounds"] = component_bounds
    merged["d_is_exact"] = distance_schema.is_exact(lower, upper, sources)
    # dict(new_rec) above carries over new_rec's raw legacy d/fom/score
    # fields, which can silently diverge from the just-reconciled
    # d_lower/d_upper -- e.g. merging an exact d=6 record with a d=99
    # heuristic estimate must not leave merged["d"] == 99 while
    # d_lower == d_upper == 6 (and fom/score stale against d=99 to match).
    # _normalize_legacy_fields is also applied to a never-before-seen
    # record on first insert (see update_pareto_front_v2's old_rec is None
    # branch), so the same record displays the same fields whether it
    # arrives fresh or via a merge.
    merged = _normalize_legacy_fields(merged)

    n = merged.get("n", 0)
    k = merged.get("k", 0)
    merged.pop("certified_fom", None)
    merged.pop("exploratory_fom", None)
    cf = distance_schema.certified_fom(k, n, lower)
    if cf is not None:
        merged["certified_fom"] = cf
    ef = distance_schema.exploratory_fom(k, n, upper)
    if ef is not None:
        merged["exploratory_fom"] = ef

    return merged


def _dominance_front(records: list[dict], bound_field: str) -> list[dict]:
    """Non-dominated set in ``(k/n, bound_field, 1/n)``, mirroring
    ``evaluation.results.update_pareto_front``'s dominance loop with the
    legacy bare ``d`` replaced by ``bound_field`` (``"d_lower"`` or
    ``"d_upper"`` depending on the tier).
    """
    valid = [
        r for r in records
        if r.get("n", 0) > 0 and r.get("k", 0) > 0 and (r.get(bound_field) or 0) > 0
    ]
    valid.sort(key=lambda r: r.get("fom", 0), reverse=True)

    front: list[dict] = []
    for candidate in valid:
        n = candidate["n"]
        k = candidate["k"]
        d = candidate.get(bound_field) or 0
        rate = k / n if n > 0 else 0

        dominated = False
        for existing in front:
            en, ek = existing["n"], existing["k"]
            ed = existing.get(bound_field) or 0
            e_rate = ek / en if en > 0 else 0
            if e_rate >= rate and ed >= d and en <= n:
                if e_rate > rate or ed > d or en < n:
                    dominated = True
                    break

        if not dominated:
            front = [
                existing for existing in front
                if not (
                    rate >= (existing["k"] / existing["n"]) and
                    d >= (existing.get(bound_field) or 0) and
                    n <= existing["n"] and
                    (rate > (existing["k"] / existing["n"]) or
                     d > (existing.get(bound_field) or 0) or n < existing["n"])
                )
            ]
            front.append(candidate)

    return front


def update_pareto_front_v2(
    v2_records: list[dict],
    *,
    filepath: Path | str | None = None,
    generated_at: str | None = None,
) -> dict:
    """Merge new v2 records into the on-disk three-tier archive and recompute it.

    Each incoming record is merged (by code key) with any existing entry
    already in the archive via :func:`_merge_records` (which uses
    ``distance_schema.merge_bounds`` directly on both the top-level and
    per-component bounds internally). A merge that would raise
    ``DistanceBoundConflict`` is caught per-record: the conflicting
    ``(old, new)`` pair is appended to the ``quarantined`` list instead of
    crashing the whole update.

    The three fronts are then computed independently from disjoint subsets
    of the FULL merged/deduplicated record set (never comparing across
    tiers), so an exploratory record can never evict/dominate an exact or
    lower-bound record for a different code, and a record already present
    in a higher tier still has its stored bounds/provenance kept current by
    the merge step even though it won't additionally appear in a lower
    tier's front. The full merged set is itself persisted as
    ``all_records`` (not just the three front subsets) and reloaded as the
    merge base on the next call -- see the module docstring's ``all_records``
    paragraph for why: a record dominated in this round must remain
    available to resurface in a later round if the record dominating it is
    later promoted to a different tier.

    Writes the result to ``filepath`` (default :data:`PARETO_V2_PATH`)
    atomically (temp file in the same directory + ``os.replace``) and
    returns it.

    The full load-merge-write transaction runs under an exclusive
    ``fcntl.flock`` on a sibling ``.<name>.lock`` file (mirroring
    ``evolve.discovery_events``'s locking pattern), so two concurrent
    callers can't both load the same pre-update archive and then each
    ``os.replace`` their own merge over the other's -- ``os.replace`` is
    atomic per-write, but without a lock around the whole read-modify-write
    cycle a second writer can still silently clobber records the first
    writer just merged in.
    """
    path = Path(filepath) if filepath is not None else PARETO_V2_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / f".{path.name}.lock"

    with open(lock_path, "a+") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            existing = load_pareto_v2(path)
            existing_records = list(existing.get("all_records", []))
            if not existing_records:
                # Backward-compat with archives written before "all_records"
                # existed (or a genuinely empty archive, in which case this
                # is a no-op).
                existing_records = (
                    list(existing.get("exact_front", []))
                    + list(existing.get("lower_bound_front", []))
                    + list(existing.get("exploratory_witnesses", []))
                )
            quarantined = list(existing.get("quarantined", []))

            merged_by_key: dict[tuple, dict] = {}
            for rec in existing_records:
                # Normalize on load too, not just on write: a record
                # persisted by an older version of this module (before
                # _normalize_legacy_fields existed, or under a since-fixed
                # bug) can carry a stale d/fom/score that would otherwise
                # only self-heal if that exact code happens to be
                # resubmitted and merged again -- which may never happen.
                merged_by_key[_code_key(rec)] = _normalize_legacy_fields(rec)

            for new_rec in v2_records:
                key = _code_key(new_rec)
                old_rec = merged_by_key.get(key)
                if old_rec is None:
                    # Normalize the same legacy fields _merge_records would,
                    # so a never-before-seen record's displayed d/fom/score
                    # don't change on a later no-op re-submission of the
                    # identical record (which WOULD go through
                    # _merge_records and normalize them there) -- see
                    # _normalize_legacy_fields's docstring.
                    merged_by_key[key] = _normalize_legacy_fields(new_rec)
                    continue
                try:
                    merged_by_key[key] = _merge_records(old_rec, new_rec)
                except DistanceBoundConflict as e:
                    quarantined.append({
                        "code_key": list(key) if isinstance(key, tuple) else key,
                        "old": old_rec,
                        "new": new_rec,
                        "error": str(e),
                    })

            all_records = list(merged_by_key.values())

            exact_front = _dominance_front(
                [r for r in all_records if r.get("d_is_exact")], "d_lower",
            )
            lower_bound_front = _dominance_front(
                [
                    r for r in all_records
                    if not r.get("d_is_exact") and r.get("d_lower", 0) > 0
                ],
                "d_lower",
            )
            exploratory_witnesses = _dominance_front(
                [
                    r for r in all_records
                    if r.get("d_lower", 0) == 0 and r.get("d_upper") is not None
                ],
                "d_upper",
            )

            output = {
                "schema_version": distance_schema.SCHEMA_VERSION,
                "generated_at": generated_at,
                "exact_front": exact_front,
                "lower_bound_front": lower_bound_front,
                "exploratory_witnesses": exploratory_witnesses,
                "all_records": all_records,
                "quarantined": quarantined,
            }

            tmp_path = path.parent / f".{path.name}.tmp{os.getpid()}"
            with open(tmp_path, "w") as f:
                json.dump(output, f, indent=2, default=str)
            os.replace(tmp_path, path)
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    return output


def migrate_legacy_pareto_to_v2(
    legacy_codes_path: Path | str | None = None,
    legacy_pareto_path: Path | str | None = None,
    v2_filepath: Path | str | None = None,
) -> dict:
    """One-time migration of legacy flat result lists into the v2 archive.

    Reads ``evaluation.results.load_codes(legacy_codes_path)`` (the flat
    discovered-codes list) -- READ ONLY, never writes to the legacy files.
    If ``legacy_pareto_path`` is given and exists, its flat list (same v1
    record shape as ``discovered_codes.json``) is also read read-only and
    included, for completeness (the plan text only explicitly requires
    ``load_codes``; this is a documented, conservative extension since the
    function's purpose is migrating "legacy pareto" data and pareto_front.
    json is itself a flat v1 record list, so skipping it if provided would
    silently drop information the caller explicitly asked to migrate).

    Applies ``distance_schema.migrate_legacy_record`` to each and calls
    :func:`update_pareto_front_v2` with the results, writing only the new
    v2 file.
    """
    legacy_records = list(load_codes(legacy_codes_path))

    if legacy_pareto_path is not None:
        pareto_path = Path(legacy_pareto_path)
        if pareto_path.exists():
            try:
                with open(pareto_path) as f:
                    legacy_records.extend(json.load(f))
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    "Corrupted legacy pareto front file %s: %s", pareto_path, e,
                )

    v2_records = [distance_schema.migrate_legacy_record(r) for r in legacy_records]
    return update_pareto_front_v2(v2_records, filepath=v2_filepath)
