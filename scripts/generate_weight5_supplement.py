#!/usr/bin/env python3
"""Generate the filtered weight-five campaign catalogue as a LaTeX include.

The four ``all_codes`` logs are append-only observation logs, not catalogues:
the same defining tuple can occur many times and its heuristic distance can
change between observations.  This script therefore

* sorts the terms within each polynomial and keys a specification by
  ``(campaign, ell, m, A, B, C, D)``;
* keeps the tightest reported upper endpoint and strongest proved lower bound;
* lets a consistent exact certificate replace both bounds and promotes a
  matching certified lower bound and rigorous upper bound to exact;
* merges native CSS discovery events and reconstructed historical PBB
  generator-attribution sidecars; and
* assigns every retained row to a presentation class defined by colored-BLISS
  stored-generator Tanner isomorphism and writes a machine-readable sidecar; and
* fails loudly on inconsistent dimensions or distance evidence.

The typeset campaign tables contain every proposal in an exact class and one
deterministic representative of each nonexact class; the complete retained
catalogue remains in the machine-readable publication artifact.  The BLISS
relation includes sector exchange, monomial shifts, and
lattice automorphisms that preserve the colored stored generator graph, but
not equivalences requiring stabilizer-basis changes or local Clifford gates.
Specifications are retained when their best reported distance endpoint
is at least three.  Thus exact ``d <= 2``
specifications and unresolved specifications reported at ``d <= 2``
are omitted, while unresolved specifications with upper endpoint at least
three remain visible.

The PBB worker exhaustively excludes logical operators through weight six for
``n <= 216`` and through weight four otherwise before it emits a
``milp+bposd``, ``bposd``, or ``milp_exact`` record.  Those completed searches
give certified lower bounds 7 and 5, respectively.  The completed large-PBB
deep-MILP audit is merged by specification.  Rows for which not every logical
problem reached optimality retain certified lower bounds with rigorous MILP
upper bounds rather than being promoted to exact distance.

The post-hoc CSS audit similarly exhausts both logical sectors through weight
four for every CSS row formerly lacking direct lower-bound evidence.  Its class-representative
results are transferred only across verified colored-Tanner isomorphisms.

Usage:
    python scripts/generate_weight5_supplement.py
    python scripts/generate_weight5_supplement.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "paper" / "2610.06623" / "weight5_supplemental_tables.tex"

sys.path.insert(0, str(ROOT))

from evaluation.bb_code import build_bb_code  # noqa: E402
from evaluation.pbb_code import build_pbb_code  # noqa: E402
from evaluation.connectivity import decompose, components_isomorphic  # noqa: E402
from evaluation.tanner_equivalence import (  # noqa: E402
    canonical_hash,
    canonical_hash_noncss,
    canonical_signature_digest,
)


DEFAULT_CLASS_OUTPUT = ROOT / "results" / "weight5_presentation_classes.jsonl"
COMPONENT_CERTIFICATION_PATH = (
    ROOT / "results" / "weight5_component_certifications.jsonl"
)
PBB_COMPONENT_CERTIFICATION_PATH = (
    ROOT / "results" / "weight5_pbb_component_certifications.jsonl"
)
CSS_LOW_WEIGHT_AUDIT_PATH = (
    ROOT / "results" / "weight5_css_low_weight_audit.jsonl"
)
CSS_WEIGHT5_WITNESSES_PATH = (
    ROOT / "results" / "weight5_css_weight5_witnesses.jsonl"
)
CSS_UPPER_BOUND_WITNESSES_PATH = (
    ROOT / "results" / "weight5_css_upper_bound_witnesses.jsonl"
)
CSS_UPPER_BOUND_WITNESS_SCHEMA_VERSION = 1
CSS_UPPER_BOUND_WITNESS_ARTIFACT_SCHEMA = (
    "weight5_css_upper_bound_witnesses_v1"
)
CSS_UPPER_BOUND_WITNESS_ALGORITHM = (
    "randomized_gf2_kernel_permutation_search_v1"
)
CSS_UPPER_BOUND_WITNESS_GENERATOR = (
    "scripts/audit_weight5_css_upper_bound_witnesses.py"
)
EXPECTED_CSS_UPPER_BOUND_WITNESSES = {
    "CL-1a333605": ("B", 22, 14),
    "CL-23177205": ("B", 22, 14),
    "CL-40a1f698": ("B", 23, 12),
    "CL-4a95cc39": ("B", 14, 13),
    "CL-54d0129e": ("B", 18, 13),
    "CL-85245770": ("B", 17, 14),
    "CL-d192c3d1": ("B", 14, 13),
    "CL-daf9d6b0": ("B", 22, 16),
    "CL-f5754da6": ("I", 24, 23),
    "CS-22b9dba0": ("B", 14, 9),
}
CSS_UPPER_BOUND_WITNESS_SEEDS = {"X": 1000, "Z": 2000}
CSS_UPPER_BOUND_WITNESS_DEFAULT_TRIALS = 400
CSS_UPPER_BOUND_WITNESS_STRESS_TRIALS = {"CL-f5754da6": 8_000}
CSS_WEIGHT5_WITNESS_SCHEMA_VERSION = 1
CSS_WEIGHT5_WITNESS_ARTIFACT_SCHEMA = "weight5_css_weight5_witnesses_v1"
CSS_WEIGHT5_WITNESS_ALGORITHM = "translation_anchored_pair_pair_collision_v1"
CSS_WEIGHT5_WITNESS_GENERATOR = (
    "scripts/audit_weight5_css_weight5_witnesses.py"
)
RIGOROUS_UPPER_METHODS = {"H", "Hc", "I", "M", "S", "W"}
PUBLICATION_CATALOGUE_PATH = ROOT / "results" / "weight5_publication_catalogue.jsonl"
PUBLICATION_MANIFEST_PATH = ROOT / "results" / "weight5_publication_manifest.json"
COMPONENT_CLASSES_PATH = ROOT / "results" / "weight5_component_classes.jsonl"
PBB_COMPONENT_LC_AUDIT_PATH = (
    ROOT / "results" / "weight5_pbb_component_lc_audit.jsonl"
)
CLASS_SCHEMA_VERSION = 2
CLASS_EQUIVALENCE_RELATION = "colored_stored_generator_tanner_isomorphism_v1"

FIXED_PBB_SAFETY_NET = {
    (
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (0, 1)),
        ((1, 0),),
    ),
    (
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (0, 1)),
        ((1, 0),),
        ((0, 0), (0, 1)),
    ),
}


def upper_endpoint_is_rigorous(
    upper_evidence: Iterable[tuple[int, str]], endpoint: int
) -> bool:
    """Whether the decisive endpoint is backed by non-decoder evidence."""
    return any(
        bound == endpoint and method in RIGOROUS_UPPER_METHODS
        for bound, method in upper_evidence
    )


@dataclass(frozen=True)
class Campaign:
    slug: str
    label: str
    family: str
    raw_path: Path
    event_path: Path | None
    verification_path: Path
    expected_raw_rows: int
    expected_specs: int
    attribution_path: Path | None = None


CAMPAIGNS = (
    Campaign(
        "css-small",
        "CSS-small",
        "CSS",
        ROOT / "results/evolution/weight5_css_small_v1/all_codes_weight5_css.jsonl",
        ROOT / "results/evolution/weight5_css_small_v1/discovery_events.jsonl",
        ROOT / "results/weight5_css_small_v1_milp_verified.jsonl",
        5610,
        382,
    ),
    Campaign(
        "css-large",
        "CSS-large",
        "CSS",
        ROOT / "results/evolution/weight5_css_large_v1/all_codes_weight5_css.jsonl",
        ROOT / "results/evolution/weight5_css_large_v1/discovery_events.jsonl",
        ROOT / "results/weight5_css_large_v1_milp_verified.jsonl",
        834,
        554,
    ),
    Campaign(
        "pbb-small",
        "PBB-small",
        "PBB",
        ROOT / "results/evolution/weight5_pbb_small_v1/all_codes_weight5_pbb.jsonl",
        None,
        ROOT / "results/weight5_pbb_small_v1_deep_milp_verify.jsonl",
        4890,
        257,
        ROOT / "results/evolution/weight5_pbb_small_v1/historical_attribution.jsonl",
    ),
    Campaign(
        "pbb-large",
        "PBB-large",
        "PBB",
        ROOT / "results/evolution/weight5_pbb_large_v1/all_codes_weight5_pbb.jsonl",
        None,
        ROOT / "results/weight5_pbb_large_v1_deep_milp_verify.jsonl",
        2589,
        104,
        ROOT / "results/evolution/weight5_pbb_large_v1/historical_attribution.jsonl",
    ),
)


@dataclass
class Spec:
    campaign: Campaign
    ell: int
    m: int
    n: int
    k: int
    A: tuple[tuple[int, int], ...]
    B: tuple[tuple[int, int], ...]
    C: tuple[tuple[int, int], ...]
    D: tuple[tuple[int, int], ...]
    observations: int = 0
    verification_rows: int = 0
    exact_values: set[int] = field(default_factory=set)
    lower_evidence: list[tuple[int, str]] = field(default_factory=list)
    upper_evidence: list[tuple[int, str]] = field(default_factory=list)
    proof_sources: set[str] = field(default_factory=set)
    upper_sources: set[str] = field(default_factory=set)
    model_aliases: set[str | None] = field(default_factory=set)
    possible_model_aliases: set[str | None] = field(default_factory=set)
    attribution_status: str | None = None
    program_ids: set[str] = field(default_factory=set)
    source_hashes: set[str] = field(default_factory=set)
    event_count: int = 0
    fixed_safety_net: bool = False
    component_certificate_ids: set[str] = field(default_factory=set)
    css_low_weight_audit_class_ids: set[str] = field(default_factory=set)
    css_low_weight_audit_transferred: bool = False

    def add_exact(self, value: object, source: str) -> None:
        d = positive_int(value, f"{self.campaign.slug}: exact distance")
        self.exact_values.add(d)
        self.proof_sources.add(source)
        self.lower_evidence.append((d, source))
        self.upper_evidence.append((d, source))

    def add_lower(self, value: object, source: str) -> None:
        d = positive_int(value, f"{self.campaign.slug}: lower bound")
        self.lower_evidence.append((d, source))

    def add_upper(self, value: object, source: str) -> None:
        if value is None:
            return
        d = positive_int(value, f"{self.campaign.slug}: upper bound")
        self.upper_evidence.append((d, source))
        self.upper_sources.add(source)

    def final_bounds(self) -> tuple[int, int, bool]:
        if len(self.exact_values) > 1:
            raise ValueError(
                f"conflicting exact distances for {self.spec_id}: "
                f"{sorted(self.exact_values)}"
            )
        lower = max((d for d, _ in self.lower_evidence), default=0)
        upper = min((d for d, _ in self.upper_evidence), default=None)
        if self.exact_values:
            exact = next(iter(self.exact_values))
            if lower > exact or (upper is not None and upper < exact):
                raise ValueError(
                    f"conflicting bounds for {self.spec_id}: "
                    f"lower={lower}, exact={exact}, upper={upper}"
                )
            return exact, exact, True
        if upper is None:
            raise ValueError(f"no distance witness for {self.spec_id}")
        if lower > upper:
            raise ValueError(
                f"empty distance interval for {self.spec_id}: [{lower}, {upper}]"
            )
        if lower > 0 and lower == upper:
            closing_methods = sorted(
                {
                    method
                    for bound, method in self.upper_evidence
                    if bound == upper
                }
            )
            if not upper_endpoint_is_rigorous(self.upper_evidence, upper):
                raise ValueError(
                    f"unsupported exact-distance closure for {self.spec_id}: "
                    f"lower={lower}, upper methods={closing_methods}"
                )
            return lower, upper, True
        return lower, upper, False

    @property
    def spec_id(self) -> str:
        payload = [
            self.campaign.slug,
            self.ell,
            self.m,
            self.A,
            self.B,
            self.C,
            self.D,
        ]
        digest = hashlib.sha256(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:8]
        prefix = {
            "css-small": "CS",
            "css-large": "CL",
            "pbb-small": "PS",
            "pbb-large": "PL",
        }[self.campaign.slug]
        return f"{prefix}-{digest}"


@dataclass(frozen=True)
class ConnectivityInfo:
    n_components: int
    is_connected: bool
    homogeneous: bool
    components_isomorphic: bool | None
    base_n: int | None
    base_k: int | None
    presentation_agrees: bool
    presentation_signature: tuple
    presentation_digest: str


@dataclass(frozen=True)
class PresentationClass:
    class_id: str
    family: str
    digest: str
    members: tuple[Spec, ...]
    lower: int
    upper: int
    exact: bool
    upper_is_supported: bool
    upper_methods: tuple[str, ...]
    connectivity: str


_CONNECTIVITY_CACHE: dict[str, ConnectivityInfo] = {}


def positive_int(value: object, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be a positive integer, got {value!r}")
    try:
        integer = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be a positive integer, got {value!r}") from exc
    if integer <= 0 or integer != value:
        raise ValueError(f"{context} must be a positive integer, got {value!r}")
    return integer


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected an object in {path}:{line_number}")
            rows.append(row)
    return rows


def normalize_terms(value: object, context: str) -> tuple[tuple[int, int], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list, got {type(value).__name__}")
    terms: list[tuple[int, int]] = []
    for term in value:
        if not isinstance(term, (list, tuple)) or len(term) != 2:
            raise ValueError(f"invalid term {term!r} in {context}")
        a, b = term
        if not isinstance(a, int) or not isinstance(b, int):
            raise ValueError(f"non-integral term {term!r} in {context}")
        terms.append((a, b))
    if len(set(terms)) != len(terms):
        raise ValueError(f"duplicate polynomial term in {context}: {terms!r}")
    return tuple(sorted(terms))


def key_for(campaign: Campaign, row: dict) -> tuple:
    return (
        campaign.slug,
        positive_int(row.get("ell"), f"{campaign.slug}: ell"),
        positive_int(row.get("m"), f"{campaign.slug}: m"),
        normalize_terms(row.get("A_terms"), f"{campaign.slug}: A_terms"),
        normalize_terms(row.get("B_terms"), f"{campaign.slug}: B_terms"),
        normalize_terms(row.get("C_terms"), f"{campaign.slug}: C_terms"),
        normalize_terms(row.get("D_terms"), f"{campaign.slug}: D_terms"),
    )


def validate_shape(spec: Spec) -> None:
    if spec.n != 2 * spec.ell * spec.m:
        raise ValueError(
            f"{spec.spec_id}: n={spec.n}, expected 2*ell*m={2 * spec.ell * spec.m}"
        )
    if spec.k <= 0 or spec.k > spec.n:
        raise ValueError(f"{spec.spec_id}: invalid k={spec.k} for n={spec.n}")
    if len(spec.A) + len(spec.B) != 5:
        raise ValueError(f"{spec.spec_id}: backbone does not have weight five")
    if spec.campaign.family == "CSS":
        if spec.C or spec.D:
            raise ValueError(f"{spec.spec_id}: CSS record has a perturbation")
    else:
        if not (spec.C or spec.D):
            raise ValueError(f"{spec.spec_id}: PBB record has no perturbation")
        if not set(spec.C).issubset(spec.A) or not set(spec.D).issubset(spec.B):
            raise ValueError(f"{spec.spec_id}: PBB support-containment violation")
        if len(set(spec.A) | set(spec.C)) + len(set(spec.B) | set(spec.D)) != 5:
            raise ValueError(f"{spec.spec_id}: PBB row weight is not five")


def make_spec(campaign: Campaign, row: dict) -> Spec:
    spec = Spec(
        campaign=campaign,
        ell=positive_int(row.get("ell"), f"{campaign.slug}: ell"),
        m=positive_int(row.get("m"), f"{campaign.slug}: m"),
        n=positive_int(row.get("n"), f"{campaign.slug}: n"),
        k=positive_int(row.get("k"), f"{campaign.slug}: k"),
        A=normalize_terms(row.get("A_terms"), f"{campaign.slug}: A_terms"),
        B=normalize_terms(row.get("B_terms"), f"{campaign.slug}: B_terms"),
        C=normalize_terms(row.get("C_terms"), f"{campaign.slug}: C_terms"),
        D=normalize_terms(row.get("D_terms"), f"{campaign.slug}: D_terms"),
    )
    if campaign.family == "PBB":
        spec.fixed_safety_net = (spec.A, spec.B, spec.C, spec.D) in FIXED_PBB_SAFETY_NET
    validate_shape(spec)
    return spec


def add_raw_evidence(spec: Spec, row: dict) -> None:
    d = positive_int(row.get("d"), f"{spec.spec_id}: raw d")
    spec.observations += 1

    if spec.campaign.family == "CSS":
        # Every persisted CSS campaign stage is either a decoder estimate or
        # an exact-search timeout retaining its preceding decoder estimate.
        spec.add_upper(d, "B")
        return

    method = str(row.get("d_method") or "")
    if method.startswith("exact_w"):
        suffix = method.removeprefix("exact_w")
        if suffix and int(suffix) != d:
            raise ValueError(f"{spec.spec_id}: {method} disagrees with d={d}")
        spec.add_exact(d, "H")
        return

    # Reaching MILP/BP-OSD means the preceding exhaustive low-weight search
    # completed without finding a logical: through 6 for n<=216, else 4.
    cutoff = 6 if spec.n <= 216 else 4
    spec.add_lower(cutoff + 1, f"L{cutoff}")

    if method == "milp_exact" or row.get("milp_exact") is True:
        exact_d = row.get("d_milp") if row.get("d_milp") is not None else d
        spec.add_exact(exact_d, "M")
        return

    if row.get("d_milp") is not None:
        spec.add_upper(row["d_milp"], "I")
    if row.get("d_bposd") is not None:
        spec.add_upper(row["d_bposd"], "B")
    # The persisted d is the best reported upper endpoint available in-loop.
    spec.add_upper(d, "B" if method == "bposd" else "I")


def load_specs(campaign: Campaign) -> tuple[dict[tuple, Spec], int]:
    rows = load_jsonl(campaign.raw_path)
    if len(rows) != campaign.expected_raw_rows:
        raise ValueError(
            f"{campaign.raw_path.relative_to(ROOT)} has {len(rows)} rows; "
            f"expected {campaign.expected_raw_rows}"
        )
    specs: dict[tuple, Spec] = {}
    for row in rows:
        key = key_for(campaign, row)
        if key not in specs:
            specs[key] = make_spec(campaign, row)
        spec = specs[key]
        n = positive_int(row.get("n"), f"{spec.spec_id}: n")
        k = positive_int(row.get("k"), f"{spec.spec_id}: k")
        if (n, k) != (spec.n, spec.k):
            raise ValueError(
                f"inconsistent dimensions for {spec.spec_id}: "
                f"{(spec.n, spec.k)} versus {(n, k)}"
            )
        add_raw_evidence(spec, row)
    if len(specs) != campaign.expected_specs:
        raise ValueError(
            f"{campaign.raw_path.relative_to(ROOT)} yields {len(specs)} specs; "
            f"expected {campaign.expected_specs}"
        )
    return specs, len(rows)


def merge_events(campaign: Campaign, specs: dict[tuple, Spec]) -> int:
    if campaign.event_path is None:
        return 0
    rows = load_jsonl(campaign.event_path)
    for row in rows:
        key = key_for(campaign, row)
        if key not in specs:
            raise ValueError(
                f"event has no raw-log specification: "
                f"{campaign.event_path.relative_to(ROOT)}"
            )
        spec = specs[key]
        spec.event_count += 1
        spec.model_aliases.add(row.get("model_alias"))
        if row.get("program_id"):
            spec.program_ids.add(str(row["program_id"]))
        if row.get("source_hash"):
            spec.source_hashes.add(str(row["source_hash"]))
    return len(rows)


def merge_historical_attribution(
    campaign: Campaign, specs: dict[tuple, Spec]
) -> int:
    """Merge the immutable PBB provenance sidecar into catalogue rows."""
    if campaign.attribution_path is None:
        return 0
    rows = load_jsonl(campaign.attribution_path)
    seen: set[tuple] = set()
    for row in rows:
        key = key_for(campaign, row)
        if key not in specs:
            raise ValueError(
                "attribution row has no raw-log specification: "
                f"{campaign.attribution_path.relative_to(ROOT)}"
            )
        if key in seen:
            raise ValueError(f"duplicate attribution row for {specs[key].spec_id}")
        seen.add(key)
        spec = specs[key]
        if int(row.get("observation_count", -1)) != spec.observations:
            raise ValueError(
                f"attribution observation count disagrees for {spec.spec_id}"
            )
        status = str(row.get("attribution_status") or "")
        if status not in {"exact", "confirmed_minimum"}:
            raise ValueError(f"invalid attribution status for {spec.spec_id}: {status}")
        possible_aliases = set(row.get("possible_additional_model_aliases") or [])
        if (status == "exact") != (not possible_aliases):
            raise ValueError(
                f"attribution status/possible aliases disagree for {spec.spec_id}"
            )
        spec.attribution_status = status
        spec.model_aliases.update(row.get("model_aliases") or [])
        spec.possible_model_aliases.update(possible_aliases)
        spec.program_ids.update(str(value) for value in row.get("program_ids") or [])
    if seen != set(specs):
        raise ValueError(
            f"{campaign.attribution_path.relative_to(ROOT)} does not cover every spec"
        )
    return len(rows)


def merge_css_verification(
    campaign: Campaign, specs: dict[tuple, Spec], rows: Iterable[dict]
) -> None:
    for row in rows:
        key = key_for(campaign, row)
        if key not in specs:
            raise ValueError(
                f"verification row has no raw-log specification: {campaign.slug}"
            )
        spec = specs[key]
        spec.verification_rows += 1
        stage = str(row.get("stage") or "")
        d = positive_int(row.get("d"), f"{spec.spec_id}: verification d")
        if row.get("d_is_exact") is True:
            source = "M" if stage == "milp_exact" else "S"
            spec.add_exact(d, source)
        else:
            source = "I" if stage.startswith("milp") else "W"
            spec.add_upper(d, source)


def merge_pbb_verification(
    campaign: Campaign, specs: dict[tuple, Spec], rows: Iterable[dict]
) -> None:
    for row in rows:
        key = key_for(campaign, row)
        if key not in specs:
            raise ValueError(
                f"verification row has no raw-log specification: {campaign.slug}"
            )
        spec = specs[key]
        spec.verification_rows += 1
        milp_d = row.get("milp_d")
        hash_d = row.get("hash_d")
        method = str(row.get("d_method") or "")
        logicals_optimal = row.get("logicals_optimal")
        total_logicals = row.get("total_logicals")
        all_optimal = (
            isinstance(logicals_optimal, int)
            and not isinstance(logicals_optimal, bool)
            and isinstance(total_logicals, int)
            and not isinstance(total_logicals, bool)
            and total_logicals > 0
            and logicals_optimal == total_logicals
        )

        if method.startswith("exact_w"):
            # This field is a real exhaustive-hash result for PBB-small.
            spec.add_exact(hash_d, "H")
        # In PBB-large, hash_d is only the discovery-time estimate copied by
        # the generic verifier.  Never interpret it as hash evidence.

        if milp_d is not None:
            if all_optimal:
                spec.add_exact(milp_d, "M")
            else:
                spec.add_upper(milp_d, "I")
        if row.get("d_bposd") is not None:
            spec.add_upper(row["d_bposd"], "B")


def merge_verification(campaign: Campaign, specs: dict[tuple, Spec]) -> int:
    rows = load_jsonl(campaign.verification_path)
    if campaign.family == "CSS":
        merge_css_verification(campaign, specs, rows)
    else:
        merge_pbb_verification(campaign, specs, rows)
    return len(rows)


def merge_component_certifications(
    campaign: Campaign, specs: dict[tuple, Spec]
) -> int:
    """Merge exact parent distances proved by component enumeration.

    These certificates are independent post-hoc evidence.  They must be
    merged before the distance filter so that a newly proved ``d <= 2`` row
    is removed from the publication catalogue.
    """
    if campaign.slug != "css-large":
        return 0
    by_id = {spec.spec_id: spec for spec in specs.values()}
    merged = 0
    for row in load_jsonl(COMPONENT_CERTIFICATION_PATH):
        if row.get("record_type") != "component_distance_certificate":
            continue
        source = row.get("source_identity") or {}
        presentation_id = str(source.get("presentation_id") or "")
        if presentation_id not in by_id:
            raise ValueError(
                "component certificate has no raw-log specification: "
                f"{presentation_id}"
            )
        spec = by_id[presentation_id]
        parent = row.get("parent_spec") or {}
        expected = {
            "ell": spec.ell,
            "m": spec.m,
            "n": spec.n,
            "k": spec.k,
            "A_terms": [list(term) for term in spec.A],
            "B_terms": [list(term) for term in spec.B],
        }
        for field_name, expected_value in expected.items():
            if parent.get(field_name) != expected_value:
                raise ValueError(
                    f"{presentation_id}: component certificate {field_name} "
                    f"disagrees with raw specification"
                )
        transfer = row.get("distance_transfer") or {}
        if transfer.get("valid") is not True:
            raise ValueError(f"{presentation_id}: invalid distance transfer")
        component = row.get("component_certificate") or {}
        exact = positive_int(
            component.get("d_exact"),
            f"{presentation_id}: component exact distance",
        )
        if parent.get("d_exact") != exact or transfer.get("parent_d_exact") != exact:
            raise ValueError(f"{presentation_id}: inconsistent component distance")
        certificate_id = str(row.get("certificate_id") or "")
        if not certificate_id or certificate_id in spec.component_certificate_ids:
            raise ValueError(
                f"{presentation_id}: missing or duplicate component certificate ID"
            )
        spec.component_certificate_ids.add(certificate_id)
        spec.add_exact(exact, "Hc")
        merged += 1
    if merged != 7:
        raise ValueError(f"expected seven component certificates, found {merged}")
    return merged


def load_and_merge_css_low_weight_audit(
    specs_by_id: dict[str, Spec],
    path: Path = CSS_LOW_WEIGHT_AUDIT_PATH,
) -> tuple[dict, dict[str, dict]]:
    """Validate and merge the deterministic weight-four CSS audit.

    The checked-in sidecar was computed on the normalized pre-audit snapshot.
    Its colored-Tanner digest and complete member list are rechecked here
    before any evidence is added.  A class result is then valid for each
    direct-U target because the certified qubit/check permutations preserve
    CSS weight, commutation, and stabilizer-row-space membership.
    """
    rows = load_jsonl(path)
    manifests = [row for row in rows if row.get("record_type") == "artifact_manifest"]
    audits = [
        row
        for row in rows
        if row.get("record_type") == "css_low_weight_class_audit"
    ]
    if len(manifests) != 1 or len(audits) != 492 or len(rows) != 493:
        raise ValueError(
            f"{path.relative_to(ROOT)} must contain one manifest and 492 class audits"
        )
    manifest = manifests[0]
    expected_counts = {
        "audit_outcomes_by_class": {
            "exact_low_weight": 64,
            "excluded_through_weight_4": 428,
        },
        "audit_outcomes_by_target_presentation": {
            "exact_low_weight": 94,
            "excluded_through_weight_4": 764,
        },
        "low_weight_exact_classes_by_distance": {"2": 2, "3": 25, "4": 37},
        "low_weight_exact_presentations_by_distance": {
            "2": 2,
            "3": 42,
            "4": 50,
        },
        "post_audit_status_by_class": {"C": 421, "E": 71},
        "post_audit_status_by_target_presentation": {"C": 735, "E": 123},
        "presentations_proved_d_le_2": 2,
    }
    if (
        manifest.get("artifact_schema") != "weight5_css_low_weight_audit_v1"
        or (manifest.get("algorithm") or {}).get("name")
        != "css_weight_le_4_syndrome_quotient_mitm_v1"
        or (manifest.get("scope") or {}).get("target_presentations") != 858
        or (manifest.get("scope") or {}).get("target_classes") != 492
        or manifest.get("counts") != expected_counts
    ):
        raise ValueError("malformed CSS low-weight audit manifest")

    producer = manifest.get("producer") or {}
    producer_path = ROOT / str(producer.get("path") or "")
    if (
        not producer_path.is_file()
        or source_digest(producer_path) != producer.get("sha256")
    ):
        raise ValueError("CSS low-weight audit producer hash mismatch")
    for descriptor in (manifest.get("source_artifacts") or {}).values():
        source_path = ROOT / str((descriptor or {}).get("path") or "")
        # The sidecar records this generator as the historical normalizer, but
        # publication-only rendering edits must not invalidate the audit.  The
        # normalized member lists and canonical digests are independently
        # rebuilt and checked below, which is the relevant semantic guard.
        if source_path.resolve() == Path(__file__).resolve():
            continue
        if (
            not source_path.is_file()
            or source_digest(source_path) != descriptor.get("sha256")
        ):
            raise ValueError(
                f"CSS low-weight audit source hash mismatch: {source_path}"
            )

    baseline_retained = {
        spec_id: spec
        for spec_id, spec in specs_by_id.items()
        if retain_for_catalogue(spec)
    }
    expected_targets = {
        spec_id
        for spec_id, spec in baseline_retained.items()
        if spec.campaign.family == "CSS" and status_for(spec) == "U"
    }
    if len(expected_targets) != 858:
        raise ValueError(
            f"pre-audit snapshot has {len(expected_targets)} direct-U CSS rows; "
            "expected 858"
        )

    # Rebuild the exact equivalence partition named by the sidecar.
    by_digest: dict[str, set[str]] = defaultdict(set)
    for spec_id, spec in baseline_retained.items():
        if spec.campaign.family == "CSS":
            by_digest[connectivity_info(spec).presentation_digest].add(spec_id)

    target_records: dict[str, dict] = {}
    seen_classes: set[str] = set()
    pending_evidence: list[tuple[Spec, str, int, str, bool]] = []
    for row in audits:
        if row.get("algorithm") != "css_weight_le_4_syndrome_quotient_mitm_v1":
            raise ValueError("CSS low-weight class audit names an unknown algorithm")
        identity = row.get("class_identity") or {}
        class_id = str(identity.get("class_id") or "")
        digest = str(identity.get("canonical_digest_sha256") or "")
        if (
            not class_id
            or class_id in seen_classes
            or class_id != f"TC-{digest[:10]}"
            or identity.get("equivalence_relation") != CLASS_EQUIVALENCE_RELATION
        ):
            raise ValueError(f"invalid or duplicate CSS low-weight class {class_id!r}")
        seen_classes.add(class_id)
        declared_members = set(identity.get("all_member_ids") or [])
        if declared_members != by_digest.get(digest, set()):
            raise ValueError(f"{class_id}: audit member list disagrees with BLISS")
        target_ids = list(identity.get("target_direct_u_member_ids") or [])
        if not target_ids or len(target_ids) != len(set(target_ids)):
            raise ValueError(f"{class_id}: missing or duplicate target IDs")
        if not set(target_ids) <= declared_members:
            raise ValueError(f"{class_id}: target IDs are not class members")

        representative = row.get("representative") or {}
        representative_id = str(representative.get("presentation_id") or "")
        if representative_id not in target_ids:
            raise ValueError(f"{class_id}: representative is not a direct-U target")
        representative_spec = specs_by_id[representative_id]
        expected_representative = {
            "ell": representative_spec.ell,
            "m": representative_spec.m,
            "n": representative_spec.n,
            "k_catalogue": representative_spec.k,
            "A_terms": [list(term) for term in representative_spec.A],
            "B_terms": [list(term) for term in representative_spec.B],
        }
        for key, expected in expected_representative.items():
            if representative.get(key) != expected:
                raise ValueError(
                    f"{class_id}: representative {key} disagrees with source"
                )
        if (
            representative.get("k_rebuilt") != representative_spec.k
            or representative.get("checks_commute") is not True
        ):
            raise ValueError(f"{class_id}: invalid rebuilt CSS checks")

        evidence = row.get("audit_evidence") or {}
        result = evidence.get("result")
        exact_distance = evidence.get("exact_distance")
        certified_lower = evidence.get("certified_lower_bound")
        if result == "exact_low_weight":
            exact = positive_int(exact_distance, f"{class_id}: exact distance")
            if exact > 4 or certified_lower != exact:
                raise ValueError(f"{class_id}: invalid low-weight exact evidence")
            method = "H"
            value = exact
        elif result == "excluded_through_weight_4":
            if exact_distance is not None or certified_lower != 5:
                raise ValueError(f"{class_id}: invalid weight-four exclusion")
            method = "L4"
            value = 5
        else:
            raise ValueError(f"{class_id}: unknown audit result {result!r}")

        sectors = row.get("search") or {}
        sector_exact: list[int] = []
        for sector_name in ("X", "Z"):
            sector = sectors.get(sector_name) or {}
            sector_value = sector.get("exact_distance")
            witness = sector.get("witness")
            if sector_value is None:
                if sector.get("certified_lower_bound") != 5 or witness is not None:
                    raise ValueError(f"{class_id}: malformed {sector_name} exclusion")
                continue
            sector_value = positive_int(
                sector_value, f"{class_id}: {sector_name} exact distance"
            )
            indices = (witness or {}).get("qubit_indices_zero_based")
            if (
                sector_value > 4
                or not isinstance(indices, list)
                or len(indices) != sector_value
                or len(set(indices)) != sector_value
                or any(
                    not isinstance(index, int)
                    or isinstance(index, bool)
                    or not 0 <= index < representative_spec.n
                    for index in indices
                )
            ):
                raise ValueError(f"{class_id}: malformed {sector_name} witness")
            sector_exact.append(sector_value)
        if (min(sector_exact) if sector_exact else None) != exact_distance:
            raise ValueError(f"{class_id}: sector and combined evidence disagree")

        for target_id in target_ids:
            if target_id in target_records:
                raise ValueError(f"duplicate CSS low-weight target {target_id}")
            spec = specs_by_id.get(target_id)
            if spec is None or target_id not in expected_targets:
                raise ValueError(f"unexpected CSS low-weight target {target_id}")
            target_records[target_id] = row
            pending_evidence.append(
                (spec, class_id, value, method, target_id != representative_id)
            )

    if set(target_records) != expected_targets:
        missing = sorted(expected_targets - set(target_records))
        extra = sorted(set(target_records) - expected_targets)
        raise ValueError(
            f"CSS low-weight audit target mismatch: missing={missing}, extra={extra}"
        )

    # Mutate only after every record and the complete coverage set validate.
    for spec, class_id, value, method, transferred in pending_evidence:
        if method == "H":
            spec.add_exact(value, method)
        else:
            spec.add_lower(value, method)
        spec.css_low_weight_audit_class_ids.add(class_id)
        spec.css_low_weight_audit_transferred = transferred
    return manifest, target_records


def _css_witness_source_paths() -> set[Path]:
    """Return every data artifact read while rebuilding witness targets."""
    paths = {
        COMPONENT_CERTIFICATION_PATH,
        CSS_LOW_WEIGHT_AUDIT_PATH,
    }
    for campaign in CAMPAIGNS:
        paths.add(campaign.raw_path)
        paths.add(campaign.verification_path)
        if campaign.event_path is not None:
            paths.add(campaign.event_path)
        if campaign.attribution_path is not None:
            paths.add(campaign.attribution_path)
    return {path.resolve() for path in paths}


def _validate_css_witness_source_manifest(manifest: dict) -> None:
    """Require an exact, hash-pinned list of witness-generator data inputs."""
    source_artifacts = manifest.get("source_artifacts")
    if not isinstance(source_artifacts, dict) or set(source_artifacts) != {
        "catalogue_inputs",
        "low_weight_audit",
    }:
        raise ValueError("malformed CSS witness source manifest")
    low_weight_descriptor = source_artifacts["low_weight_audit"]
    catalogue_descriptors = source_artifacts["catalogue_inputs"]
    if not isinstance(low_weight_descriptor, dict) or not isinstance(
        catalogue_descriptors, list
    ):
        raise ValueError("malformed CSS witness source descriptors")

    expected_sources = _css_witness_source_paths()
    observed_sources: set[Path] = set()
    for descriptor in [low_weight_descriptor, *catalogue_descriptors]:
        if not isinstance(descriptor, dict):
            raise ValueError("malformed CSS witness source descriptor")
        descriptor_path = descriptor.get("path")
        if not isinstance(descriptor_path, str) or not descriptor_path:
            raise ValueError("CSS witness source path is missing")
        source_path = (ROOT / descriptor_path).resolve()
        try:
            expected_relative = source_path.relative_to(ROOT.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"CSS witness source is outside the repository: {descriptor_path}"
            ) from exc
        if descriptor_path != expected_relative or source_path not in expected_sources:
            raise ValueError(f"unexpected CSS witness source: {descriptor_path}")
        if source_path in observed_sources:
            raise ValueError(f"duplicate CSS witness source: {descriptor_path}")
        observed_sources.add(source_path)
        if (
            not source_path.is_file()
            or source_digest(source_path) != descriptor.get("sha256")
        ):
            raise ValueError(f"CSS witness source hash mismatch: {source_path}")
    if observed_sources != expected_sources:
        missing = sorted(
            str(item.relative_to(ROOT))
            for item in expected_sources - observed_sources
        )
        extra = sorted(
            str(item.relative_to(ROOT))
            for item in observed_sources - expected_sources
        )
        raise ValueError(
            f"CSS witness source coverage mismatch: missing={missing}, extra={extra}"
        )
    if (
        (ROOT / low_weight_descriptor["path"]).resolve()
        != CSS_LOW_WEIGHT_AUDIT_PATH.resolve()
    ):
        raise ValueError("CSS witness low-weight source is malformed")


def _css_matrix_rows_as_int(matrix: object, num_columns: int) -> list[int]:
    """Convert a binary check matrix to the audit artifact's integer rows."""
    rows: list[int] = []
    for values in matrix:  # type: ignore[union-attr]
        if len(values) != num_columns:
            raise ValueError("rebuilt CSS check matrix has the wrong width")
        row = 0
        for column, raw_value in enumerate(values):
            value = int(raw_value)
            if value not in (0, 1):
                raise ValueError("rebuilt CSS check matrix is not binary")
            row |= value << column
        rows.append(row)
    return rows


def _css_matrix_digest(rows: list[int], num_columns: int) -> str:
    """Match the canonical check-matrix digest stored by the audit."""
    width = (num_columns + 7) // 8
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"rows": len(rows), "columns": num_columns},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    )
    digest.update(b"\0")
    for row in rows:
        digest.update(row.to_bytes(width, "little"))
    return digest.hexdigest()


def _css_rowspace_basis(
    rows: list[int], num_columns: int
) -> tuple[list[int], list[int]]:
    """Return a GF(2) RREF basis and its pivot columns for integer rows."""
    column_mask = (1 << num_columns) - 1
    reduced = [row & column_mask for row in rows if row & column_mask]
    pivots: list[int] = []
    pivot_row = 0
    for column in range(num_columns):
        bit = 1 << column
        found = next(
            (
                index
                for index in range(pivot_row, len(reduced))
                if reduced[index] & bit
            ),
            None,
        )
        if found is None:
            continue
        reduced[pivot_row], reduced[found] = (
            reduced[found],
            reduced[pivot_row],
        )
        pivot = reduced[pivot_row]
        for index, row in enumerate(reduced):
            if index != pivot_row and row & bit:
                reduced[index] = row ^ pivot
        pivots.append(column)
        pivot_row += 1
        if pivot_row == len(reduced):
            break
    return reduced[:pivot_row], pivots


def _css_support_is_logical(
    support: int,
    check_rows: list[int],
    stabilizer_basis: list[int],
    stabilizer_pivots: list[int],
) -> bool:
    if support <= 0 or any((row & support).bit_count() % 2 for row in check_rows):
        return False
    remainder = support
    for row, pivot in zip(stabilizer_basis, stabilizer_pivots, strict=True):
        if remainder & (1 << pivot):
            remainder ^= row
    return remainder != 0


def _css_qubit_labels(
    indices: list[int], ell: int, m: int
) -> list[dict[str, object]]:
    group_order = ell * m
    labels: list[dict[str, object]] = []
    for index in indices:
        block_index, group_index = divmod(index, group_order)
        x_exp, y_exp = divmod(group_index, m)
        labels.append(
            {
                "index": index,
                "block": "left" if block_index == 0 else "right",
                "x": x_exp,
                "y": y_exp,
            }
        )
    return labels


def _is_css_l4_b5_witness_target(spec: Spec) -> bool:
    """Identify a pre-witness CSS row whose B=5 scalar meets an L4 proof."""
    return (
        spec.campaign.family == "CSS"
        and not spec.exact_values
        and max((bound for bound, _ in spec.lower_evidence), default=0) == 5
        and min((bound for bound, _ in spec.upper_evidence), default=None) == 5
        and (5, "L4") in spec.lower_evidence
        and (5, "B") in spec.upper_evidence
        and not any(
            bound == 5 and method in RIGOROUS_UPPER_METHODS
            for bound, method in spec.upper_evidence
        )
    )


def load_and_merge_css_weight5_witnesses(
    specs_by_id: dict[str, Spec],
    path: Path = CSS_WEIGHT5_WITNESSES_PATH,
) -> tuple[dict, dict[str, dict]]:
    """Validate and merge deterministic witnesses closing the 45 L4+B rows."""
    rows = load_jsonl(path)
    manifests = [row for row in rows if row.get("record_type") == "artifact_manifest"]
    audits = [
        row
        for row in rows
        if row.get("record_type") == "css_weight5_witness_audit"
    ]
    if len(manifests) != 1 or len(audits) != 45 or len(rows) != 46:
        raise ValueError("malformed CSS weight-five witness artifact")
    manifest = manifests[0]
    if (
        manifest.get("schema_version") != CSS_WEIGHT5_WITNESS_SCHEMA_VERSION
        or manifest.get("record_type") != "artifact_manifest"
        or manifest.get("artifact_schema") != CSS_WEIGHT5_WITNESS_ARTIFACT_SCHEMA
        or manifest.get("generator") != CSS_WEIGHT5_WITNESS_GENERATOR
        or manifest.get("algorithm") != CSS_WEIGHT5_WITNESS_ALGORITHM
        or manifest.get("counts")
        != {
            "presentations": 45,
            "presentation_classes": 32,
            "explicit_witnesses": 90,
            "all_x_witnesses_valid": True,
            "all_z_witnesses_valid": True,
        }
    ):
        raise ValueError("malformed CSS weight-five witness manifest")
    _validate_css_witness_source_manifest(manifest)

    expected_targets = {
        spec_id
        for spec_id, spec in specs_by_id.items()
        if _is_css_l4_b5_witness_target(spec)
    }
    if len(expected_targets) != 45:
        raise ValueError(
            "pre-witness snapshot has "
            f"{len(expected_targets)} direct L4+B=5 rows; expected 45"
        )

    target_records: dict[str, dict] = {}
    pending_evidence: list[Spec] = []
    for row in audits:
        if (
            row.get("schema_version") != CSS_WEIGHT5_WITNESS_SCHEMA_VERSION
            or row.get("record_type") != "css_weight5_witness_audit"
            or row.get("algorithm") != CSS_WEIGHT5_WITNESS_ALGORITHM
        ):
            raise ValueError("malformed CSS weight-five witness row")
        presentation_id = str(row.get("presentation_id") or "")
        if (
            presentation_id not in expected_targets
            or presentation_id in target_records
        ):
            raise ValueError(
                f"invalid or duplicate CSS weight-five witness target "
                f"{presentation_id!r}"
            )
        spec = specs_by_id[presentation_id]
        class_id = row.get("class_id")
        if (
            not isinstance(class_id, str)
            or not class_id
            or spec.css_low_weight_audit_class_ids != {class_id}
        ):
            raise ValueError(
                f"{presentation_id}: witness class disagrees with low-weight audit"
            )

        expected_generators = {
            "A_terms": [list(term) for term in spec.A],
            "B_terms": [list(term) for term in spec.B],
        }
        params = row.get("parameters") or {}
        if (
            row.get("generators") != expected_generators
            or params.get("ell") != spec.ell
            or params.get("m") != spec.m
            or params.get("n") != spec.n
            or params.get("k_catalogue") != spec.k
            or params.get("k_rebuilt") != spec.k
            or params.get("d_exact") != 5
        ):
            raise ValueError(
                f"{presentation_id}: rebuilt presentation disagrees with source"
            )

        lower_bound = row.get("lower_bound") or {}
        if (
            lower_bound.get("method") != "L4"
            or lower_bound.get("certified") != 5
            or lower_bound.get("audit_class_id") != class_id
            or not isinstance(lower_bound.get("audit_representative_id"), str)
            or not lower_bound["audit_representative_id"]
        ):
            raise ValueError(
                f"{presentation_id}: malformed lower-bound provenance"
            )
        audit_representative = specs_by_id.get(
            lower_bound["audit_representative_id"]
        )
        if (
            audit_representative is None
            or audit_representative.css_low_weight_audit_class_ids != {class_id}
        ):
            raise ValueError(
                f"{presentation_id}: unknown low-weight audit representative"
            )
        historical = row.get("historical_upper_bound") or {}
        if (
            historical.get("method") != "B"
            or historical.get("value") != 5
            or historical.get("operator_was_persisted") is not False
            or not isinstance(historical.get("matching_scalar_observations"), int)
            or isinstance(historical.get("matching_scalar_observations"), bool)
            or historical["matching_scalar_observations"] <= 0
        ):
            raise ValueError(
                f"{presentation_id}: malformed historical B endpoint"
            )

        code = build_bb_code(spec.ell, spec.m, list(spec.A), list(spec.B))
        if int(code.num_qudits) != spec.n or int(code.dimension) != spec.k:
            raise ValueError(
                f"{presentation_id}: independently rebuilt code disagrees with source"
            )
        rows_x = _css_matrix_rows_as_int(code.matrix_x, spec.n)
        rows_z = _css_matrix_rows_as_int(code.matrix_z, spec.n)
        expected_matrix_digests = {
            "H_X_sha256": _css_matrix_digest(rows_x, spec.n),
            "H_Z_sha256": _css_matrix_digest(rows_z, spec.n),
        }
        if row.get("matrix_digests") != expected_matrix_digests:
            raise ValueError(
                f"{presentation_id}: CSS witness matrix digest mismatch"
            )
        basis_x, pivots_x = _css_rowspace_basis(rows_x, spec.n)
        basis_z, pivots_z = _css_rowspace_basis(rows_z, spec.n)

        witnesses = row.get("replacement_explicit_witnesses") or {}
        if set(witnesses) != {"X", "Z"}:
            raise ValueError(
                f"{presentation_id}: malformed CSS witness sectors"
            )
        for sector_name in ("X", "Z"):
            sector = witnesses.get(sector_name) or {}
            indices = sector.get("qubit_indices_zero_based")
            if (
                not isinstance(indices, list)
                or sector.get("weight") != 5
                or len(indices) != 5
                or len(set(indices)) != 5
                or indices != sorted(indices)
                or any(
                    not isinstance(index, int)
                    or isinstance(index, bool)
                    or not 0 <= index < spec.n
                    for index in indices
                )
            ):
                raise ValueError(
                    f"{presentation_id}: malformed {sector_name} witness support"
                )
            group_order = spec.ell * spec.m
            anchor = sector.get("anchor_qubit")
            if (
                anchor not in {0, group_order}
                or anchor not in indices
                or sector.get("qubits")
                != _css_qubit_labels(indices, spec.ell, spec.m)
                or not isinstance(sector.get("pairs_indexed"), int)
                or isinstance(sector.get("pairs_indexed"), bool)
                or sector["pairs_indexed"] <= 0
                or not isinstance(sector.get("pairs_probed"), int)
                or isinstance(sector.get("pairs_probed"), bool)
                or sector["pairs_probed"] <= 0
                or sector.get("zero_check_syndrome") is not True
                or sector.get("outside_stabilizer_rowspace") is not True
            ):
                raise ValueError(
                    f"{presentation_id}: malformed {sector_name} witness metadata"
                )

            support = sum(1 << index for index in indices)
            if sector_name == "X":
                check_rows = rows_z
                stabilizer_basis = basis_x
                stabilizer_pivots = pivots_x
            else:
                check_rows = rows_x
                stabilizer_basis = basis_z
                stabilizer_pivots = pivots_z
            if sector.get("stabilizer_rank") != len(stabilizer_pivots):
                raise ValueError(
                    f"{presentation_id}: wrong {sector_name} stabilizer rank"
                )
            if not _css_support_is_logical(
                support,
                check_rows,
                stabilizer_basis,
                stabilizer_pivots,
            ):
                raise ValueError(
                    f"{presentation_id}: invalid {sector_name} logical witness"
                )

        target_records[presentation_id] = row
        pending_evidence.append(spec)

    if set(target_records) != expected_targets:
        missing = sorted(expected_targets - set(target_records))
        extra = sorted(set(target_records) - expected_targets)
        raise ValueError(
            "CSS weight-five witness target mismatch: "
            f"missing={missing}, extra={extra}"
        )
    if len({row["class_id"] for row in target_records.values()}) != 32:
        raise ValueError("CSS weight-five witness class coverage mismatch")

    # Mutate only after the complete artifact has independently validated.
    for spec in pending_evidence:
        spec.add_upper(5, "W")
    return manifest, target_records


def load_and_merge_css_upper_bound_witnesses(
    specs_by_id: dict[str, Spec],
    path: Path = CSS_UPPER_BOUND_WITNESSES_PATH,
) -> tuple[dict, dict[str, dict]]:
    """Validate and merge independently reconstructed CSS witnesses.

    The artifact supplies explicit logical operators that tighten nine
    decoder-scalar endpoints and one nonoptimal MILP incumbent.  The merge
    adds only witnessed (``W``) upper bounds and never promotes a distance
    to exact.
    """
    rows = load_jsonl(path)
    manifests = [row for row in rows if row.get("record_type") == "artifact_manifest"]
    audits = [
        row
        for row in rows
        if row.get("record_type") == "css_upper_bound_witness_correction"
    ]
    if len(manifests) != 1 or len(audits) != 10 or len(rows) != 11:
        raise ValueError("malformed CSS upper-bound witness artifact")
    manifest = manifests[0]
    if (
        manifest.get("schema_version")
        != CSS_UPPER_BOUND_WITNESS_SCHEMA_VERSION
        or manifest.get("record_type") != "artifact_manifest"
        or manifest.get("artifact_schema")
        != CSS_UPPER_BOUND_WITNESS_ARTIFACT_SCHEMA
        or manifest.get("generator") != CSS_UPPER_BOUND_WITNESS_GENERATOR
        or manifest.get("algorithm") != CSS_UPPER_BOUND_WITNESS_ALGORITHM
        or manifest.get("counts")
        != {
            "presentations": 10,
            "presentation_classes": 10,
            "explicit_witnesses": 20,
            "prior_methods": {"B": 9, "I": 1},
            "all_x_witnesses_valid": True,
            "all_z_witnesses_valid": True,
        }
    ):
        raise ValueError("malformed CSS upper-bound witness manifest")

    _validate_css_witness_source_manifest(manifest)

    target_records: dict[str, dict] = {}
    pending_evidence: list[tuple[Spec, int]] = []
    for row in audits:
        if (
            row.get("schema_version")
            != CSS_UPPER_BOUND_WITNESS_SCHEMA_VERSION
            or row.get("record_type") != "css_upper_bound_witness_correction"
            or row.get("algorithm") != CSS_UPPER_BOUND_WITNESS_ALGORITHM
        ):
            raise ValueError("malformed CSS upper-bound witness row")
        presentation_id = str(row.get("presentation_id") or "")
        if not presentation_id or presentation_id in target_records:
            raise ValueError(
                f"invalid or duplicate CSS upper-bound witness target "
                f"{presentation_id!r}"
            )
        spec = specs_by_id.get(presentation_id)
        if spec is None:
            raise ValueError(
                f"unknown CSS upper-bound witness target "
                f"{presentation_id!r}"
            )

        expected_generators = {
            "A_terms": [list(term) for term in spec.A],
            "B_terms": [list(term) for term in spec.B],
        }
        params = row.get("parameters") or {}
        if (
            row.get("generators") != expected_generators
            or params.get("ell") != spec.ell
            or params.get("m") != spec.m
            or params.get("n") != spec.n
            or params.get("k_catalogue") != spec.k
            or params.get("k_rebuilt") != spec.k
        ):
            raise ValueError(
                f"{presentation_id}: rebuilt presentation disagrees with source"
            )
        class_id = row.get("class_id")
        if (
            not isinstance(class_id, str)
            or not class_id
            or spec.css_low_weight_audit_class_ids != {class_id}
        ):
            raise ValueError(
                f"{presentation_id}: witness class disagrees with low-weight audit"
            )
        lower_bound = row.get("lower_bound") or {}
        if (
            lower_bound.get("method") != "L4"
            or lower_bound.get("certified") != 5
            or lower_bound.get("audit_class_id") != class_id
            or not isinstance(lower_bound.get("audit_representative_id"), str)
            or not lower_bound["audit_representative_id"]
        ):
            raise ValueError(
                f"{presentation_id}: malformed lower-bound provenance"
            )
        audit_representative = specs_by_id.get(
            lower_bound["audit_representative_id"]
        )
        if (
            audit_representative is None
            or audit_representative.css_low_weight_audit_class_ids != {class_id}
        ):
            raise ValueError(
                f"{presentation_id}: unknown low-weight audit representative"
            )

        code = build_bb_code(spec.ell, spec.m, list(spec.A), list(spec.B))
        if int(code.num_qudits) != spec.n or int(code.dimension) != spec.k:
            raise ValueError(
                f"{presentation_id}: independently rebuilt code disagrees with source"
            )
        rows_x = _css_matrix_rows_as_int(code.matrix_x, spec.n)
        rows_z = _css_matrix_rows_as_int(code.matrix_z, spec.n)
        expected_matrix_digests = {
            "H_X_sha256": _css_matrix_digest(rows_x, spec.n),
            "H_Z_sha256": _css_matrix_digest(rows_z, spec.n),
        }
        if row.get("matrix_digests") != expected_matrix_digests:
            raise ValueError(
                f"{presentation_id}: CSS witness matrix digest mismatch"
            )
        basis_x, pivots_x = _css_rowspace_basis(rows_x, spec.n)
        basis_z, pivots_z = _css_rowspace_basis(rows_z, spec.n)

        historical = row.get("historical_upper_bound") or {}
        corrected = row.get("corrected_upper_bound") or {}
        historical_method = historical.get("method")
        if historical_method not in {"B", "I"} or corrected.get("method") != "W":
            raise ValueError(f"{presentation_id}: unexpected evidence methods")
        historical_value = positive_int(
            historical.get("value"), f"{presentation_id}: historical upper bound"
        )
        new_value = positive_int(
            corrected.get("value"), f"{presentation_id}: corrected upper bound"
        )
        expected_correction = EXPECTED_CSS_UPPER_BOUND_WITNESSES.get(
            presentation_id
        )
        if expected_correction != (
            historical_method,
            historical_value,
            new_value,
        ):
            raise ValueError(
                f"{presentation_id}: unexpected CSS witness correction "
                f"{(historical_method, historical_value, new_value)}"
            )
        if new_value >= historical_value:
            raise ValueError(
                f"{presentation_id}: corrected bound does not improve on the "
                "historical value"
            )
        _, source_upper, source_exact = spec.final_bounds()
        if source_exact or source_upper != historical_value:
            raise ValueError(
                f"{presentation_id}: source endpoint is {source_upper}, not "
                f"the expected {historical_value}"
            )
        if historical.get("operator_was_persisted") is not False:
            raise ValueError(
                f"{presentation_id}: prior evidence persistence is malformed"
            )
        if not any(
            bound == historical_value and method == historical_method
            for bound, method in spec.upper_evidence
        ):
            raise ValueError(
                f"{presentation_id}: expected historical "
                f"{historical_method}={historical_value} "
                "evidence not found on source spec"
            )

        witnesses = row.get("replacement_explicit_witnesses") or {}
        if set(witnesses) != {"X", "Z"}:
            raise ValueError(
                f"{presentation_id}: malformed CSS witness sectors"
            )
        sector_weights = []
        for sector_name in ("X", "Z"):
            sector = witnesses.get(sector_name) or {}
            if (
                sector.get("zero_check_syndrome") is not True
                or sector.get("outside_stabilizer_rowspace") is not True
                or sector.get("weight_matches_support_len") is not True
            ):
                raise ValueError(f"{presentation_id}: unverified {sector_name} witness")
            indices = sector.get("qubit_indices_zero_based")
            weight = sector.get("weight")
            if (
                not isinstance(indices, list)
                or not isinstance(weight, int)
                or isinstance(weight, bool)
                or len(indices) != weight
                or len(set(indices)) != weight
                or any(
                    not isinstance(index, int)
                    or isinstance(index, bool)
                    or not 0 <= index < spec.n
                    for index in indices
                )
            ):
                raise ValueError(
                    f"{presentation_id}: malformed {sector_name} witness support"
                )
            if indices != sorted(indices):
                raise ValueError(
                    f"{presentation_id}: noncanonical {sector_name} witness support"
                )
            expected_trials = CSS_UPPER_BOUND_WITNESS_STRESS_TRIALS.get(
                presentation_id,
                CSS_UPPER_BOUND_WITNESS_DEFAULT_TRIALS,
            )
            if (
                sector.get("operator_type") != sector_name
                or sector.get("qubits")
                != _css_qubit_labels(indices, spec.ell, spec.m)
                or sector.get("search_trials") != expected_trials
                or sector.get("search_seed")
                != CSS_UPPER_BOUND_WITNESS_SEEDS[sector_name]
            ):
                raise ValueError(
                    f"{presentation_id}: malformed {sector_name} witness metadata"
                )

            support = sum(1 << index for index in indices)
            if sector_name == "X":
                check_rows = rows_z
                stabilizer_basis = basis_x
                stabilizer_pivots = pivots_x
            else:
                check_rows = rows_x
                stabilizer_basis = basis_z
                stabilizer_pivots = pivots_z
            if sector.get("stabilizer_rank") != len(stabilizer_pivots):
                raise ValueError(
                    f"{presentation_id}: wrong {sector_name} stabilizer rank"
                )
            if not _css_support_is_logical(
                support,
                check_rows,
                stabilizer_basis,
                stabilizer_pivots,
            ):
                raise ValueError(
                    f"{presentation_id}: invalid {sector_name} logical witness"
                )
            sector_weights.append(weight)
        if min(sector_weights) != new_value:
            raise ValueError(
                f"{presentation_id}: corrected bound disagrees with sector witnesses"
            )

        target_records[presentation_id] = row
        pending_evidence.append((spec, new_value))

    if len(pending_evidence) != 10:
        raise ValueError(
            f"expected 10 CSS upper-bound witness corrections, applied "
            f"{len(pending_evidence)}"
        )
    if set(target_records) != set(EXPECTED_CSS_UPPER_BOUND_WITNESSES):
        raise ValueError("CSS upper-bound witness target set mismatch")
    if len({row["class_id"] for row in target_records.values()}) != 10:
        raise ValueError("CSS upper-bound witness class coverage mismatch")

    # Mutate only after every record validates.
    for spec, new_value in pending_evidence:
        spec.add_upper(new_value, "W")
    return manifest, target_records


def source_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def polynomial(terms: tuple[tuple[int, int], ...]) -> str:
    if not terms:
        return "0"
    monomials = []
    for x_exp, y_exp in terms:
        factors = []
        if x_exp:
            factors.append("x" if x_exp == 1 else f"x^{{{x_exp}}}")
        if y_exp:
            factors.append("y" if y_exp == 1 else f"y^{{{y_exp}}}")
        monomials.append("".join(factors) if factors else "1")
    return "+".join(monomials)


def status_for(spec: Spec) -> str:
    lower, _, exact = spec.final_bounds()
    if exact:
        return "E"
    if lower > 0:
        return "C"
    return "U"


def retain_for_catalogue(spec: Spec) -> bool:
    """Retain specifications whose best reported endpoint is at least three."""
    _, upper, _ = spec.final_bounds()
    return upper >= 3


def distance_text(spec: Spec) -> str:
    lower, upper, exact = spec.final_bounds()
    if exact:
        return str(upper)
    if upper_endpoint_is_rigorous(spec.upper_evidence, upper):
        if lower:
            return rf"[{lower},{upper}]"
        return rf"\leq {upper}"
    if lower:
        return rf"{lower};\widehat{{{upper}}}"
    return rf"\widehat{{{upper}}}"


def spec_upper_is_supported(spec: Spec) -> bool:
    """Whether the proposal's displayed endpoint is a rigorous upper bound."""
    _, upper, _ = spec.final_bounds()
    return upper_endpoint_is_rigorous(spec.upper_evidence, upper)


def endpoint_pair_text(
    lower: float, upper: float, *, upper_is_supported: bool
) -> str:
    """Format a certified lower value with a bound or estimated endpoint."""
    if upper_is_supported:
        return f"[{lower:.2f},{upper:.2f}]"
    return rf"{lower:.2f};\widehat{{{upper:.2f}}}"


def class_changes_display(spec: Spec, group: PresentationClass) -> bool:
    """Whether class merging changes the displayed value or evidence quality."""
    return class_strengthens_row(spec, group) or (
        group.upper_is_supported and not spec_upper_is_supported(spec)
    )


def nonexact_evidence_label(*, upper_is_supported: bool) -> str:
    """Return the compact table label for a nonexact distance record."""
    return "Interval" if upper_is_supported else "Estimate"


def class_effect_label(spec: Spec, group: PresentationClass) -> str:
    """Describe the publication-facing effect of merging class evidence."""
    if not class_changes_display(spec, group):
        return "--"
    if group.exact:
        return "Exact"
    if not class_strengthens_row(spec, group):
        return "Supported"
    return "Tighter"


def fom_text(spec: Spec) -> str:
    lower, upper, exact = spec.final_bounds()
    upper_fom = spec.k * upper * upper / spec.n
    if exact:
        return format_exact_fom(spec.n, spec.k, upper)
    if lower:
        lower_fom = spec.k * lower * lower / spec.n
        return endpoint_pair_text(
            lower_fom,
            upper_fom,
            upper_is_supported=upper_endpoint_is_rigorous(
                spec.upper_evidence, upper
            ),
        )
    if upper_endpoint_is_rigorous(spec.upper_evidence, upper):
        return rf"\leq {upper_fom:.2f}"
    return rf"\widehat{{{upper_fom:.2f}}}"


def format_exact_fom(n: int, k: int, distance: int) -> str:
    """Round exact-code FOMs, retaining precision for a close comparison."""
    value = k * distance * distance / n
    precision = 3 if (n, k, distance) == (180, 4, 14) else 2
    return f"{value:.{precision}f}"


def evidence_text(spec: Spec) -> str:
    _, _, exact = spec.final_bounds()
    order = ("H", "Hc", "M", "S", "L6", "L4", "I", "W", "B")
    if exact and spec.exact_values:
        tags = set(spec.proof_sources)
        # Preserve the useful fact that the one hash-exact PBB-small row had
        # only an incomplete MILP cross-check.
        if "H" in tags and "M" not in tags and "I" in spec.upper_sources:
            tags.add("I")
    else:
        tags = {source for _, source in spec.lower_evidence}
        tags.update(spec.upper_sources)
    # Keep the machine-readable evidence tag as ``H`` while matching the
    # ``Exh.`` label used by the main paper for the same exhaustive search.
    display = {"H": "Exh.", "Hc": r"H$_c$"}
    return "+".join(display.get(tag, tag) for tag in order if tag in tags)


def provenance_text(spec: Spec) -> str:
    if not spec.model_aliases and not spec.possible_model_aliases:
        return "F" if spec.fixed_safety_net else "--"
    aliases = set(spec.model_aliases)
    possible = set(spec.possible_model_aliases) - aliases
    ordered = (
        ("azure/gpt-5.6-sol", "G"),
        ("aws/claude-opus-5", "O"),
        (None, "N"),
    )
    labels = []
    for alias, label in ordered:
        if alias in aliases:
            labels.append(label)
        elif alias in possible:
            labels.append(f"{label}?")
    known = {alias for alias, _ in ordered}
    labels.extend("?" for alias in sorted(aliases - known, key=str))
    labels.extend("??" for alias in sorted(possible - known, key=str))
    if spec.fixed_safety_net:
        labels.append("F")
    return "/".join(labels) if labels else "?"


def connectivity_info(spec: Spec) -> ConnectivityInfo:
    """Compute and cache the invariant stabilizer decomposition of a spec."""
    cached = _CONNECTIVITY_CACHE.get(spec.spec_id)
    if cached is not None:
        return cached
    if spec.campaign.family == "CSS":
        code = build_bb_code(spec.ell, spec.m, list(spec.A), list(spec.B))
        presentation_signature = canonical_hash(code)
    else:
        code = build_pbb_code(
            spec.ell, spec.m, list(spec.A), list(spec.B),
            list(spec.C) or None, list(spec.D) or None,
        )
        presentation_signature = canonical_hash_noncss(code)
    if int(code.num_qudits) != spec.n or int(code.dimension) != spec.k:
        raise ValueError(
            f"{spec.spec_id}: rebuilt [[{code.num_qudits},{code.dimension}]] "
            f"disagrees with stored [[{spec.n},{spec.k}]]"
        )
    presentation_digest = canonical_signature_digest(presentation_signature)
    result = decompose(code)
    if not result.is_connected and result.k_additivity_ok is not True:
        raise ValueError(
            f"{spec.spec_id}: component dimensions are not additive"
        )
    isomorphic = components_isomorphic(code) if not result.is_connected else None
    info = ConnectivityInfo(
        n_components=result.n_components,
        is_connected=result.is_connected,
        homogeneous=result.homogeneous,
        components_isomorphic=isomorphic,
        base_n=result.base_n,
        base_k=result.base_k,
        presentation_agrees=result.presentation_agrees,
        presentation_signature=presentation_signature,
        presentation_digest=presentation_digest,
    )
    _CONNECTIVITY_CACHE[spec.spec_id] = info
    return info


def decomposition_text(spec: Spec) -> str:
    """Invariant decomposition status: 'conn.' or '$m\\times\\code{n0,k0}$'.

    A split stabilizer row space means the reported [[n,k]] is a direct sum of
    m structurally independent components.  When they are pairwise isomorphic
    (the observed case for every disconnected weight-five specification),
    d(whole) = d(component) exactly, so the row's own d/kd^2/n values apply
    unchanged to the smaller base code named here. Rebuilt from the spec's
    own polynomials, not looked up, so this stays correct under re-sorting.
    """
    info = connectivity_info(spec)
    if info.is_connected:
        return "conn."
    if info.homogeneous and info.components_isomorphic:
        return rf"${info.n_components}\times\code{{{info.base_n},{info.base_k}}}$"
    return rf"${info.n_components}\times{{?}}$"


def build_presentation_classes(specs: list[Spec]) -> list[PresentationClass]:
    """Group retained rows into presentation classes by colored Tanner isomorphism.

    This is a conservative quotient of presentations, not a complete code-
    equivalence decision: arbitrary stabilizer-basis changes and local
    Clifford transformations are outside the colored BLISS relation.
    """
    grouped: dict[tuple[str, tuple], list[Spec]] = defaultdict(list)
    for spec in specs:
        signature = connectivity_info(spec).presentation_signature
        grouped[(spec.campaign.family, signature)].append(spec)

    classes = []
    seen_ids: dict[str, tuple[str, str]] = {}
    for (family, signature), members_list in grouped.items():
        members = tuple(sorted(members_list, key=lambda spec: spec.spec_id))
        digest = canonical_signature_digest(signature)
        class_id = f"{'TC' if family == 'CSS' else 'TP'}-{digest[:10]}"
        prior = seen_ids.setdefault(class_id, (family, digest))
        if prior != (family, digest):
            raise ValueError(f"truncated presentation-class ID collision: {class_id}")

        nk = {(spec.n, spec.k) for spec in members}
        if len(nk) != 1:
            raise ValueError(f"{class_id}: isomorphic presentations disagree on (n,k)")
        bounds = [spec.final_bounds() for spec in members]
        exact_values = {upper for lower, upper, exact in bounds if exact}
        if len(exact_values) > 1:
            raise ValueError(
                f"{class_id}: equivalent specifications have conflicting exact "
                f"distances {sorted(exact_values)}"
            )
        lower = max(item[0] for item in bounds)
        upper = min(item[1] for item in bounds)
        if lower > upper:
            raise ValueError(
                f"{class_id}: equivalent specifications have incompatible "
                f"distance bounds [{lower},{upper}]"
            )
        if exact_values and next(iter(exact_values)) != lower:
            exact = next(iter(exact_values))
            if not lower <= exact <= upper:
                raise ValueError(f"{class_id}: exact distance conflicts with bounds")
            lower = upper = exact
        class_upper_evidence = [
            item for spec in members for item in spec.upper_evidence
        ]
        upper_methods = tuple(
            sorted(
                {
                    method
                    for bound, method in class_upper_evidence
                    if bound == upper
                }
            )
        )
        upper_is_supported = upper_endpoint_is_rigorous(
            class_upper_evidence, upper
        )
        class_exact = bool(exact_values)
        if lower == upper and not class_exact:
            if not upper_is_supported:
                raise ValueError(
                    f"{class_id}: B-only endpoint cannot close a class interval"
                )
            class_exact = True

        connectivity_states = {
            "connected" if connectivity_info(spec).is_connected else "disconnected"
            for spec in members
        }
        if len(connectivity_states) != 1:
            raise ValueError(
                f"{class_id}: isomorphic presentations disagree on connectivity"
            )
        classes.append(
            PresentationClass(
                class_id=class_id,
                family=family,
                digest=digest,
                members=members,
                lower=lower,
                upper=upper,
                exact=class_exact,
                upper_is_supported=upper_is_supported,
                upper_methods=upper_methods,
                connectivity=next(iter(connectivity_states)),
            )
        )
    return sorted(
        classes,
        key=lambda group: (
            group.family,
            group.members[0].n,
            group.members[0].k,
            group.class_id,
        ),
    )


def render_presentation_classes(classes: list[PresentationClass]) -> str:
    """Render one deterministic JSONL row per stored-presentation class."""
    def alias_name(alias: str | None) -> str:
        return "seed/null" if alias is None else alias

    lines = []
    class_labels = presentation_class_labels(classes)
    for group in classes:
        representative = group.members[0]
        status_counts = Counter(status_for(spec) for spec in group.members)
        aliases = {
            alias
            for spec in group.members
            for alias in spec.model_aliases
        }
        possible_aliases = {
            alias
            for spec in group.members
            for alias in spec.possible_model_aliases
        } - aliases
        rows = {
            "schema_version": CLASS_SCHEMA_VERSION,
            "equivalence_relation": CLASS_EQUIVALENCE_RELATION,
            "canonicalizer": {
                "name": "python-igraph/BLISS",
                "version": distribution_version("igraph"),
            },
            "class_id": group.class_id,
            "catalogue_label": class_labels[group.class_id],
            "family": group.family,
            "canonical_digest_sha256": group.digest,
            "class_size": len(group.members),
            "member_ids": [spec.spec_id for spec in group.members],
            "campaigns": sorted({spec.campaign.slug for spec in group.members}),
            "model_aliases": sorted((alias_name(alias) for alias in aliases)),
            "possible_model_aliases": sorted(
                (alias_name(alias) for alias in possible_aliases)
            ),
            "n": representative.n,
            "k": representative.k,
            "class_distance_lower": group.lower,
            "class_distance_upper": group.upper,
            "class_distance_exact": group.exact,
            "class_upper_is_supported": group.upper_is_supported,
            "class_upper_methods": list(group.upper_methods),
            "member_direct_status_counts": {
                status: status_counts[status] for status in ("E", "C", "U")
            },
            "connectivity": group.connectivity,
            "representative_id": representative.spec_id,
            "representative": {
                "ell": representative.ell,
                "m": representative.m,
                "A_terms": representative.A,
                "B_terms": representative.B,
                "C_terms": representative.C,
                "D_terms": representative.D,
            },
        }
        lines.append(json.dumps(rows, separators=(",", ":"), sort_keys=True))
    return "\n".join(lines) + "\n"


def presentation_class_labels(
    classes: list[PresentationClass],
) -> dict[str, str]:
    """Assign compact deterministic labels for the supplemental tables."""
    counts: Counter[str] = Counter()
    labels: dict[str, str] = {}
    for group in classes:
        counts[group.family] += 1
        prefix = "C" if group.family == "CSS" else "P"
        labels[group.class_id] = f"{prefix}{counts[group.family]:03d}"
    return labels


def presentation_class_status(group: PresentationClass) -> str:
    if group.exact:
        return "E"
    return "C" if group.lower > 0 else "U"


def class_strengthens_row(spec: Spec, group: PresentationClass) -> bool:
    """Return whether merged class evidence tightens this row's interval."""
    lower, upper, direct_exact = spec.final_bounds()
    if not (lower <= group.lower <= group.upper <= upper):
        raise ValueError(
            f"{spec.spec_id}: class interval [{group.lower},{group.upper}] is "
            f"outside the attached row interval [{lower},{upper}]"
        )
    if direct_exact:
        return False
    return group.lower > lower or group.upper < upper


def catalogue_distance_text(spec: Spec, group: PresentationClass) -> str:
    """Render class distance when merging improves value or evidence quality."""
    if class_changes_display(spec, group):
        if group.exact:
            return str(group.upper)
        if group.upper_is_supported and group.lower:
            return rf"[{group.lower},{group.upper}]"
        if group.upper_is_supported:
            return rf"\leq {group.upper}"
        if group.lower:
            return rf"{group.lower};\widehat{{{group.upper}}}"
        return rf"\widehat{{{group.upper}}}"
    return distance_text(spec)


def catalogue_status_text(spec: Spec) -> str:
    """Render the publication-facing status of the displayed specification."""
    status = status_for(spec)
    if status == "E":
        return "Exact"
    if status == "C":
        return nonexact_evidence_label(
            upper_is_supported=spec_upper_is_supported(spec)
        )
    raise ValueError(
        f"{spec.spec_id}: retained catalogue unexpectedly has only an upper bound"
    )


def catalogue_class_effect_text(spec: Spec, group: PresentationClass) -> str:
    """State explicitly whether class evidence changes the displayed row."""
    return class_effect_label(spec, group)


def typeset_representative(group: PresentationClass) -> Spec:
    """Choose a deterministic representative that exposes class-level evidence."""
    def key(spec: Spec) -> tuple:
        lower, upper, exact = spec.final_bounds()
        return (
            not class_changes_display(spec, group),
            not exact,
            -lower,
            upper - lower,
            upper,
            spec.spec_id,
        )

    return min(group.members, key=key)


def typeset_spec_ids(classes: list[PresentationClass]) -> set[str]:
    """Keep all members of exact classes and one member of each nonexact class."""
    selected: set[str] = set()
    for group in classes:
        if group.exact:
            selected.update(spec.spec_id for spec in group.members)
        else:
            selected.add(typeset_representative(group).spec_id)
    return selected


def catalogue_fom_text(spec: Spec, group: PresentationClass) -> str:
    """Render class FOM when merging improves value or evidence quality."""
    if class_changes_display(spec, group):
        lower_fom = spec.k * group.lower * group.lower / spec.n
        upper_fom = spec.k * group.upper * group.upper / spec.n
        if group.exact:
            return format_exact_fom(spec.n, spec.k, group.upper)
        if group.lower:
            return endpoint_pair_text(
                lower_fom,
                upper_fom,
                upper_is_supported=group.upper_is_supported,
            )
        if group.upper_is_supported:
            return rf"\leq {upper_fom:.2f}"
        return rf"\widehat{{{upper_fom:.2f}}}"
    return fom_text(spec)


def sort_key(spec: Spec) -> tuple:
    lower, upper, exact = spec.final_bounds()
    status_order = 0 if exact else (1 if lower else 2)
    return (
        spec.n,
        -spec.k,
        status_order,
        -lower,
        -upper,
        spec.A,
        spec.B,
        spec.C,
        spec.D,
    )


def render_summary(
    campaign_data: list[tuple[Campaign, list[Spec], int, int]],
    presentation_classes: list[PresentationClass],
) -> list[str]:
    class_of = {
        spec.spec_id: group.class_id
        for group in presentation_classes
        for spec in group.members
    }
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{longtable}{lrrrrrrrr}",
        r"\caption{Catalogue accounting after filtering.  ``Logged obs.'' counts all observations with a positive reported distance; ``Kept obs.'' restricts this count to retained proposals.  ``All'' and ``Kept'' give proposal counts before and after filtering, and ``Classes'' gives the number of retained presentation classes.  Exact and unresolved counts are proposal-level statuses before the final class merge and can include low-weight evidence transferred from a verified representative.  Unresolved rows pair a certified lower bound with either a rigorous upper bound or a decoder estimate.}\label{tab:w5-supp-summary}\\",
        r"\toprule",
        r"Campaign & Logged obs. & Kept obs. & All & Kept & Classes & Removed & Exact & Unresolved\\",
        r"\midrule",
    ]
    totals = Counter()
    for campaign, specs, raw_rows, _ in campaign_data:
        statuses = Counter(status_for(spec) for spec in specs)
        if statuses["U"]:
            raise ValueError(
                f"{campaign.slug}: retained row lacks positive lower-bound evidence"
            )
        class_count = len({class_of[spec.spec_id] for spec in specs})
        retained_observations = sum(spec.observations for spec in specs)
        removed_specs = campaign.expected_specs - len(specs)
        totals.update(
            logged_observations=raw_rows,
            retained_observations=retained_observations,
            all_specs=campaign.expected_specs,
            retained_specs=len(specs),
            classes=class_count,
            removed_specs=removed_specs,
            **statuses,
        )
        lines.append(
            f"{campaign.label} & {raw_rows:,} & {retained_observations:,} & "
            f"{campaign.expected_specs:,} & {len(specs):,} & {class_count:,} & "
            f"{removed_specs:,} & "
            f"{statuses['E']:,} & {statuses['C']:,}\\\\"
        )
    lines.extend(
        [
            r"\midrule",
            f"Total & {totals['logged_observations']:,} & "
            f"{totals['retained_observations']:,} & {totals['all_specs']:,} & "
            f"{totals['retained_specs']:,} & {len(presentation_classes):,} & "
            f"{totals['removed_specs']:,} & "
            f"{totals['E']:,} & {totals['C']:,}\\\\",
            r"\bottomrule",
            r"\end{longtable}",
            r"\endgroup",
        ]
    )
    return lines


def load_explicit_bb_components() -> list[dict]:
    rows = [
        row
        for row in load_jsonl(COMPONENT_CERTIFICATION_PATH)
        if row.get("record_type") == "explicit_bb_component_match"
    ]
    if len(rows) != 3:
        raise ValueError(f"expected three explicit BB component matches, found {len(rows)}")
    for row in rows:
        component = row.get("explicit_bb_component") or {}
        match = row.get("match") or {}
        if component.get("connected") is not True or match.get("bliss_isomorphic") is not True:
            raise ValueError("explicit BB component match is not connected and verified")
        if match.get("distance_transfer_valid") is not True:
            raise ValueError("explicit BB component distance transfer is invalid")
    return sorted(rows, key=lambda row: row["explicit_bb_component"]["n"])


def render_explicit_component_summary() -> list[str]:
    lines = [
        r"\begin{longtable}{llllll}",
        r"\caption{Explicit connected BB presentations of components of certified "
        r"disconnected parents.  Canonical labelling verifies that the corresponding "
        r"presentation graphs are isomorphic, transferring the component's certified distance to "
        r"each presentation.}"
        r"\label{tab:w5-explicit-components}\\",
        r"\toprule",
        r"Code & $(\ell,m)$ & $A$ & $B$ & FOM & Certified parent\\",
        r"\midrule",
    ]
    for row in load_explicit_bb_components():
        component = row["explicit_bb_component"]
        parent = row["parent_spec"]
        fom = component["k"] * component["d_exact"] ** 2 / component["n"]
        lines.append(
            rf"$\code{{{component['n']},{component['k']},{component['d_exact']}}}$ & "
            rf"$({component['ell']},{component['m']})$ & "
            rf"${polynomial(normalize_terms(component['A_terms'], 'explicit A'))}$ & "
            rf"${polynomial(normalize_terms(component['B_terms'], 'explicit B'))}$ & "
            rf"{fom:.2f} & "
            rf"$\code{{{parent['n']},{parent['k']},{parent['d_exact']}}}$\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return lines


def load_explicit_pbb_components() -> list[dict]:
    rows = [
        row
        for row in load_jsonl(PBB_COMPONENT_CERTIFICATION_PATH)
        if row.get("record_type") == "explicit_pbb_component_match"
    ]
    if len(rows) != 4:
        raise ValueError(
            f"expected four explicit PBB component matches, found {len(rows)}"
        )
    for row in rows:
        component = row.get("explicit_pbb_component") or {}
        match = row.get("match") or {}
        if component.get("connected") is not True or match.get("bliss_isomorphic") is not True:
            raise ValueError("explicit PBB component match is not connected and verified")
        if match.get("distance_transfer_valid") is not True:
            raise ValueError("explicit PBB component distance transfer is invalid")
    return sorted(
        rows,
        key=lambda row: (
            row["explicit_pbb_component"]["n"],
            row["match_id"],
        ),
    )


def load_component_class_counts() -> dict[str, object]:
    """Validate and return the component-level class accounting sidecar."""
    rows = load_jsonl(COMPONENT_CLASSES_PATH)
    manifests = [row for row in rows if row.get("record_type") == "artifact_manifest"]
    classes = [
        row for row in rows if row.get("record_type") == "weight5_component_class"
    ]
    if len(manifests) != 1 or len(classes) != 608 or len(rows) != 609:
        raise ValueError(
            "component-class artifact must contain one manifest and 608 classes"
        )

    manifest = manifests[0]
    counts = manifest.get("counts") or {}
    expected_counts = {
        "parent_presentation_classes": 630,
        "component_classes": 608,
        "collapsed_parent_classes": 22,
        "component_classes_by_family": {"CSS": 528, "PBB": 80},
        "component_classes_by_status": {"C": 402, "E": 206},
    }
    if manifest.get("artifact") != "weight5_component_classes" or counts != expected_counts:
        raise ValueError("unexpected component-class manifest accounting")

    source = manifest.get("source_file") or {}
    if (
        source.get("path") != str(PUBLICATION_CATALOGUE_PATH.relative_to(ROOT))
        or source.get("records") != 1_142
        or source.get("sha256") != source_digest(PUBLICATION_CATALOGUE_PATH)
    ):
        raise ValueError("component-class artifact does not match the catalogue")

    actual_family = Counter(str(row.get("family")) for row in classes)
    actual_status = Counter(
        str((row.get("distance") or {}).get("status_code")) for row in classes
    )
    parent_class_ids = {
        str(class_id)
        for row in classes
        for class_id in row.get("parent_class_ids") or []
    }
    if (
        dict(actual_family) != expected_counts["component_classes_by_family"]
        or dict(actual_status) != expected_counts["component_classes_by_status"]
        or len(parent_class_ids) != expected_counts["parent_presentation_classes"]
    ):
        raise ValueError("component-class records disagree with their manifest")
    return counts


def load_pbb_component_lc_audit_counts() -> dict[str, int]:
    """Validate the complete LC-to-CSS audit of the shortest PBB components."""
    rows = load_jsonl(PBB_COMPONENT_LC_AUDIT_PATH)
    manifests = [row for row in rows if row.get("record_type") == "artifact_manifest"]
    records = [
        row
        for row in rows
        if row.get("record_type") == "weight5_pbb_component_lc_audit"
    ]
    expected_counts = {
        "certified_not_lc_css_equivalent": 4,
        "component_classes": 4,
        "lc_css_equivalent": 0,
    }
    if len(manifests) != 1 or len(records) != 4 or len(rows) != 5:
        raise ValueError(
            "PBB component LC audit must contain one manifest and four class records"
        )
    counts = manifests[0].get("counts") or {}
    if (
        manifests[0].get("artifact") != "weight5_pbb_component_lc_audit"
        or counts != expected_counts
    ):
        raise ValueError("unexpected PBB component LC-audit accounting")
    if any(
        row.get("parameters") != {"n": 12, "k": 2, "d": 3}
        or row.get("conclusion") != "not_lc_css_equivalent"
        or row.get("search", {}).get("complete") is not True
        for row in records
    ):
        raise ValueError("PBB component LC audit contains an incomplete class record")
    return counts


def render_explicit_pbb_component_summary() -> list[str]:
    lines = [
        r"\begin{longtable}{llllll}",
        r"\caption{Explicit connected PBB presentations of components of certified "
        r"disconnected parents.  Canonical labelling verifies that the corresponding "
        r"presentation graphs are isomorphic, transferring the component's certified distance to "
        r"each presentation.}"
        r"\label{tab:w5-explicit-pbb-components}\\",
        r"\toprule",
        r"Code & $(\ell,m)$ & $(A,B)$ & $(C,D)$ & FOM & Certified parent\\",
        r"\midrule",
    ]
    for row in load_explicit_pbb_components():
        component = row["explicit_pbb_component"]
        parent = row["parent_spec"]
        fom = component["k"] * component["d_exact"] ** 2 / component["n"]
        polynomials = {
            key: polynomial(normalize_terms(component[f"{key}_terms"], f"explicit {key}"))
            for key in "ABCD"
        }
        lines.append(
            rf"$\code{{{component['n']},{component['k']},{component['d_exact']}}}$ & "
            rf"$({component['ell']},{component['m']})$ & "
            rf"$({polynomials['A']},\ {polynomials['B']})$ & "
            rf"$({polynomials['C']},\ {polynomials['D']})$ & "
            rf"{fom:.2f} & "
            rf"$\code{{{parent['n']},{parent['k']},{parent['d_exact']}}}$\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return lines


def render_exact_parameter_summary(
    presentation_classes: list[PresentationClass],
) -> list[str]:
    """Render one row per exact family/parameter triple after class transfer."""
    aggregates: dict[tuple[str, int, int, int], Counter] = defaultdict(Counter)
    representatives: dict[tuple[str, int, int, int], Spec] = {}

    def representative_key(spec: Spec) -> tuple:
        """Prefer attached exact evidence, then connectivity and stable data."""
        return (
            status_for(spec) != "E",
            not connectivity_info(spec).is_connected,
            spec.ell,
            spec.m,
            spec.A,
            spec.B,
            spec.C,
            spec.D,
            spec.spec_id,
        )

    def generator_pair(
        first: tuple[tuple[int, int], ...],
        second: tuple[tuple[int, int], ...],
    ) -> str:
        """Render one generator pair in a compact, single-line cell."""
        return rf"$({polynomial(first)},\,{polynomial(second)})$"

    for group in presentation_classes:
        if not group.exact:
            continue
        representative = group.members[0]
        key = (group.family, representative.n, representative.k, group.upper)
        aggregates[key].update(
            presentations=len(group.members),
            classes=1,
            connected=int(group.connectivity == "connected"),
            disconnected=int(group.connectivity == "disconnected"),
        )
        candidates = (*group.members,)
        if key in representatives:
            candidates = (representatives[key], *candidates)
        representatives[key] = min(candidates, key=representative_key)

    lines = [
        r"\begingroup",
        r"\renewcommand{\arraystretch}{0.92}",
        r"\setlength{\tabcolsep}{1pt}",
        r"\begin{longtable}{@{}TTTTTTTNNNN@{}}",
        r"\caption{Certified parameter summary after combining evidence within "
        r"presentation classes.  For each exact parameter triple, Props. "
        r"and Cls. respectively count retained proposals and presentation classes; "
        r"Conn./disc. splits the classes into connected and disconnected.  The displayed "
        r"distance and FOM reflect class-level evidence.  Rep. is chosen "
        r"deterministically, preferring an exact proposal-level status and then "
        r"a connected presentation.  The representative columns give its lattice, "
        r"generators (with -- in $(C,D)$ for CSS), and component decomposition.  "
        r"The selected representative need not supply the decisive certificate.  "
        r"An asterisk marks a row containing both connected and disconnected "
        r"classes; in such a row Decomp. still describes only the representative.  "
        r"The seven "
        r"explicit connected-component presentations in "
        r"Tables~\ref{tab:w5-explicit-components} and "
        r"\ref{tab:w5-explicit-pbb-components} "
        r"are excluded from these counts.}"
        r"\label{tab:w5-certified-summary}\\",
        r"\toprule",
        r"Fam. & Code & Rep. & $(\ell,m)$ & $(A,B)$ & $(C,D)$ & "
        r"Decomp. & FOM & Props. & Cls. & Conn./disc.\\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{11}{c}{\tablename\ \thetable\ (continued)}\\",
        r"\toprule",
        r"Fam. & Code & Rep. & $(\ell,m)$ & $(A,B)$ & $(C,D)$ & "
        r"Decomp. & FOM & Props. & Cls. & Conn./disc.\\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for (family, n, k, distance), counts in sorted(aggregates.items()):
        representative = representatives[(family, n, k, distance)]
        mixed_marker = (
            r"\textsuperscript{*}"
            if counts["connected"] and counts["disconnected"]
            else ""
        )
        cd_text = "--"
        if family == "PBB":
            cd_text = generator_pair(representative.C, representative.D)
        lines.append(
            rf"{family} & $\code{{{n},{k},{distance}}}$ & "
            rf"\texttt{{{representative.spec_id}}} & "
            rf"$({representative.ell},{representative.m})$ & "
            rf"{generator_pair(representative.A, representative.B)} & "
            rf"{cd_text} & "
            rf"{decomposition_text(representative)}{mixed_marker} & "
            rf"{format_exact_fom(n, k, distance)} & "
            rf"{counts['presentations']} & {counts['classes']} & "
            rf"{counts['connected']}/{counts['disconnected']}\\"
        )
    lines.extend([r"\end{longtable}", r"\endgroup"])
    return lines


def table_header(
    campaign: Campaign,
    retained_row_count: int,
    displayed_row_count: int,
    class_count: int,
) -> list[str]:
    is_pbb = campaign.family == "PBB"
    polynomial_separator = r"@{\kern.5pt\vrule\kern.5pt}"
    if is_pbb:
        polynomial_columns = (
            "T" + polynomial_separator + "T"
            + polynomial_separator + "T"
            + polynomial_separator + "T"
        )
        columns = "@{}TTTTTTT" + polynomial_columns + "TTTT@{}"
        headings = (
            r"ID/class & $(\ell,m)$ & $\code{n,k}$ & $d$ & Own evidence & "
            r"Class effect & FOM & $A$ & $B$ & $C$ & $D$ & Evid. & "
            r"Obs. & Attr. & Decomp.\\"
        )
    else:
        columns = "@{}TTTTTTTT" + polynomial_separator + "TTTTT@{}"
        headings = (
            r"ID/class & $(\ell,m)$ & $\code{n,k}$ & $d$ & Own evidence & "
            r"Class effect & FOM & $A$ & $B$ & Evid. & Obs. & Attr. & Decomp.\\"
        )
    caption = (
        f"{campaign.label} compact catalogue "
        f"({displayed_row_count:,} displayed of {retained_row_count:,} retained "
        f"proposals; {class_count:,} presentation classes).  The "
        "table includes every proposal in an exact class and one deterministic "
        "representative of every nonexact class.  Own evidence and Evid. describe "
        "the proposal before class merging; Class effect reports when merging makes the "
        r"displayed $d$ and FOM exact, tighter, or rigorously supported.  Column abbreviations, "
        "filtering, and evidence conventions are defined above."
    )
    label = f"tab:w5-supp-{campaign.slug}"
    return [
        r"\begin{longtable}{" + columns + "}",
        rf"\caption{{{caption}}}\label{{{label}}}\\",
        r"\toprule",
        headings,
        r"\midrule",
        r"\endfirsthead",
        rf"\multicolumn{{{15 if is_pbb else 13}}}{{c}}{{\tablename\ \thetable\ (continued)}}\\",
        r"\toprule",
        headings,
        r"\midrule",
        r"\endhead",
        # A forward-looking footer can survive on the terminal page when a
        # longtable ends exactly at a page boundary.  The repeated header on
        # every subsequent page already identifies continued material.
        r"\bottomrule",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]


def table_row(
    spec: Spec, class_label: str, presentation_class: PresentationClass
) -> str:
    common = [
        rf"\texttt{{{spec.spec_id}/{class_label}}}",
        rf"$({spec.ell},{spec.m})$",
        rf"$\code{{{spec.n},{spec.k}}}$",
        rf"${catalogue_distance_text(spec, presentation_class)}$",
        catalogue_status_text(spec),
        catalogue_class_effect_text(spec, presentation_class),
        rf"${catalogue_fom_text(spec, presentation_class)}$",
        rf"${polynomial(spec.A)}$",
        rf"${polynomial(spec.B)}$",
    ]
    if spec.campaign.family == "PBB":
        common.extend([rf"${polynomial(spec.C)}$", rf"${polynomial(spec.D)}$"])
    common.extend(
        [
        evidence_text(spec),
        str(spec.observations),
        ]
    )
    common.append(provenance_text(spec))
    common.append(decomposition_text(spec))
    return " & ".join(common) + r"\\"


def render(
    campaign_data: list[tuple[Campaign, list[Spec], int, int]],
    presentation_classes: list[PresentationClass],
) -> str:
    component_class_counts = load_component_class_counts()
    pbb_component_lc_counts = load_pbb_component_lc_audit_counts()
    class_labels = presentation_class_labels(presentation_classes)
    class_of = {
        spec.spec_id: group.class_id
        for group in presentation_classes
        for spec in group.members
    }
    class_by_id = {group.class_id: group for group in presentation_classes}
    pbb_large = next(
        item for item in campaign_data if item[0].slug == "pbb-large"
    )
    pbb_verify_rows = load_jsonl(pbb_large[0].verification_path)
    verified_timestamps = [
        str(row["verified_at"])
        for row in pbb_verify_rows
        if row.get("verified_at") is not None
    ]
    pbb_recorded_through = max(verified_timestamps, default="not recorded")
    pbb_audit_specs = [spec for spec in pbb_large[1] if spec.verification_rows]
    pbb_audit_statuses = Counter(status_for(spec) for spec in pbb_audit_specs)

    total_logged_observations = sum(
        raw_rows for _, _, raw_rows, _ in campaign_data
    )
    total_retained_observations = sum(
        spec.observations
        for _, specs, _, _ in campaign_data
        for spec in specs
    )
    total_source_specs = sum(
        campaign.expected_specs for campaign, _, _, _ in campaign_data
    )
    total_specs = sum(len(specs) for _, specs, _, _ in campaign_data)
    total_removed_specs = total_source_specs - total_specs
    all_specs = [spec for _, specs, _, _ in campaign_data for spec in specs]
    statuses = Counter(status_for(spec) for spec in all_specs)
    nonexact_direct_evidence = Counter(
        "interval" if spec_upper_is_supported(spec) else "estimate"
        for spec in all_specs
        if status_for(spec) == "C"
    )
    transferred_low_weight_exact = sum(
        status_for(spec) == "E"
        and spec.css_low_weight_audit_transferred
        and "H" in spec.proof_sources
        for spec in all_specs
    )
    transferred_low_weight_completion = sum(
        status_for(spec) == "E"
        and spec.css_low_weight_audit_transferred
        and "H" not in spec.proof_sources
        for spec in all_specs
    )
    if (transferred_low_weight_exact, transferred_low_weight_completion) != (30, 15):
        raise ValueError(
            "unexpected retained E rows with transferred low-weight evidence: "
            f"exact={transferred_low_weight_exact}, "
            f"completed={transferred_low_weight_completion}"
        )
    class_statuses = Counter(
        presentation_class_status(group) for group in presentation_classes
    )
    nonexact_class_evidence = Counter(
        "interval" if group.upper_is_supported else "estimate"
        for group in presentation_classes
        if presentation_class_status(group) == "C"
    )
    if nonexact_direct_evidence != {"interval": 64, "estimate": 680}:
        raise ValueError(
            "unexpected direct nonexact evidence split: "
            f"{nonexact_direct_evidence}"
        )
    if nonexact_class_evidence != {"interval": 53, "estimate": 350}:
        raise ValueError(
            "unexpected class nonexact evidence split: "
            f"{nonexact_class_evidence}"
        )
    specs_by_family = Counter(spec.campaign.family for spec in all_specs)
    class_counts = Counter(
        (group.family, presentation_class_status(group), group.connectivity)
        for group in presentation_classes
    )
    expected_class_counts = {
        ("CSS", "E", "connected"): 38,
        ("CSS", "E", "disconnected"): 108,
        ("CSS", "C", "connected"): 291,
        ("CSS", "C", "disconnected"): 98,
        ("PBB", "E", "connected"): 23,
        ("PBB", "E", "disconnected"): 58,
        ("PBB", "C", "connected"): 10,
        ("PBB", "C", "disconnected"): 4,
    }
    if dict(class_counts) != expected_class_counts:
        raise ValueError(
            f"unexpected stored-presentation class counts: {dict(class_counts)}"
        )
    classes_by_family = Counter(group.family for group in presentation_classes)
    if classes_by_family != {"CSS": 535, "PBB": 95}:
        raise ValueError(f"unexpected presentation classes by family: {classes_by_family}")
    exact_member_classes = {
        group.class_id
        for group in presentation_classes
        if any(status_for(spec) == "E" for spec in group.members)
    }
    if len(exact_member_classes) != 227:
        raise ValueError("expected exact specifications to occupy 227 classes")
    exact_class_members = sum(
        len(group.members) for group in presentation_classes if group.exact
    )
    inherited_exact_statuses = Counter(
        status_for(spec)
        for group in presentation_classes
        if group.exact
        for spec in group.members
        if status_for(spec) != "E"
    )
    if exact_class_members != 436 or inherited_exact_statuses != {"C": 38}:
        raise ValueError(
            "unexpected transfer of exact class evidence: "
            f"members={exact_class_members}, inherited={inherited_exact_statuses}"
        )
    strengthened_status_pairs = Counter(
        (status_for(spec), presentation_class_status(group))
        for group in presentation_classes
        for spec in group.members
        if class_strengthens_row(spec, group)
    )
    if strengthened_status_pairs != {("C", "C"): 99, ("C", "E"): 38}:
        raise ValueError(
            "unexpected class-strengthened catalogue rows: "
            f"{strengthened_status_pairs}"
        )
    disconnected_classes = sum(
        group.connectivity == "disconnected" for group in presentation_classes
    )
    disconnected_exact_classes = sum(
        group.connectivity == "disconnected"
        and presentation_class_status(group) == "E"
        for group in presentation_classes
    )
    connectivity_rows = [(spec, connectivity_info(spec)) for spec in all_specs]
    connectivity_counts = Counter(
        (
            spec.campaign.family,
            status_for(spec),
            "connected" if info.is_connected else "disconnected",
        )
        for spec, info in connectivity_rows
    )
    expected_connectivity = {
        ("CSS", "E", "connected"): 56,
        ("CSS", "E", "disconnected"): 137,
        ("CSS", "C", "connected"): 549,
        ("CSS", "C", "disconnected"): 170,
        ("PBB", "E", "connected"): 107,
        ("PBB", "E", "disconnected"): 98,
        ("PBB", "C", "connected"): 21,
        ("PBB", "C", "disconnected"): 4,
    }
    if dict(connectivity_counts) != expected_connectivity:
        raise ValueError(
            f"unexpected retained-catalogue connectivity counts: "
            f"{dict(connectivity_counts)}"
        )
    if any(not info.presentation_agrees for _, info in connectivity_rows):
        raise ValueError(
            "stored-Tanner and invariant stabilizer partitions disagree"
        )
    if any(
        not info.is_connected and not info.components_isomorphic
        for _, info in connectivity_rows
    ):
        raise ValueError("a retained decomposition has non-isomorphic components")
    disconnected_total = sum(
        count
        for (_, _, state), count in connectivity_counts.items()
        if state == "disconnected"
    )
    disconnected_exact = sum(
        count
        for (_, status, state), count in connectivity_counts.items()
        if status == "E" and state == "disconnected"
    )
    component_certified = [spec for spec in all_specs if spec.component_certificate_ids]
    if len(component_certified) != 6:
        raise ValueError(
            f"expected six retained component-certified rows, found {len(component_certified)}"
        )
    fixed_safety_net = [spec for spec in all_specs if spec.fixed_safety_net]
    if len(fixed_safety_net) != 20:
        raise ValueError(
            f"expected 20 retained fixed-safety-net rows, found {len(fixed_safety_net)}"
        )
    if statuses["U"] or class_statuses["U"]:
        raise ValueError(
            "publication catalogue contains a row without positive lower-bound evidence"
        )
    displayed_ids = typeset_spec_ids(presentation_classes)
    if len(displayed_ids) != 839:
        raise ValueError(
            f"expected 839 compact-table rows, found {len(displayed_ids)}"
        )
    displayed_by_campaign = Counter(
        spec.campaign.slug for spec in all_specs if spec.spec_id in displayed_ids
    )
    expected_displayed_by_campaign = {
        "css-small": 224,
        "css-large": 391,
        "pbb-small": 151,
        "pbb-large": 73,
    }
    if dict(displayed_by_campaign) != expected_displayed_by_campaign:
        raise ValueError(
            "unexpected compact-table campaign counts: "
            f"{dict(displayed_by_campaign)}"
        )

    lines = [
        "% AUTO-GENERATED; do not edit.",
        "% Required in the parent preamble: \\usepackage{longtable}.",
        f"% Retained data rows: {total_specs}; removed data rows: {total_removed_specs}.",
        f"% Retained raw observations: {total_retained_observations}; "
        f"logged raw observations: {total_logged_observations}.",
        f"% PBB-large audit latest embedded timestamp: {pbb_recorded_through}.",
        f"% Retained disconnected proposals: {disconnected_total}/{total_specs}; "
        f"exact disconnected: {disconnected_exact}/{statuses['E']}.",
        f"% Presentation classes: {len(presentation_classes)}/{total_specs}; "
        f"disconnected classes: {disconnected_classes}/{len(presentation_classes)}.",
    ]
    lines.extend(
        [
            "",
            r"\setcounter{table}{0}",
            r"\renewcommand{\thetable}{S\arabic{table}}",
            "",
            r"\section{Weight-five catalogue and supporting audits}",
            r"\label{sec:w5-complete-catalogue}",
            r"\subsection{Catalogue construction}",
            (
                "The source logs contain "
                f"{total_source_specs:,} proposals ({total_logged_observations:,} "
                "observations).  After duplicate observations are merged, the "
                "catalogue retains "
                f"{total_specs:,} proposals ({total_retained_observations:,} "
                r"observations), omitting exact $d\leq2$ rows and rows with a "
                r"reported endpoint $d_{\rm rep}\leq2$.  Canonical labelling "
                f"groups the retained normalized tuples into "
                f"{len(presentation_classes):,} presentation classes, defined here "
                "as stored-generator colored-Tanner isomorphism classes: "
                f"{classes_by_family['CSS']:,} CSS and "
                f"{classes_by_family['PBB']:,} PBB.  This classification captures "
                "sector exchanges, monomial shifts, and lattice automorphisms that "
                "preserve the colored generator graph.  It does not capture "
                "equivalences requiring stabilizer-basis changes or local Clifford "
                f"operations, so {len(presentation_classes):,} remains an upper bound "
                "on the number of inequivalent codes.  ID prefixes CS, CL, PS, "
                "and PL denote CSS-small, CSS-large, PBB-small, and PBB-large; "
                "the C or P label after the slash identifies the presentation "
                "class.  The four campaign tables print all "
                f"{exact_class_members:,} proposals in exact classes and one "
                f"representative of each nonexact class ({len(displayed_ids):,} "
                f"rows total); the complete {total_specs:,}-row catalogue and "
                "supporting evidence are available through the project "
                r"repository~\cite{cruzbenito2026qcode}."
            ),
            "",
            r"\subsection{Distance evidence and class merging}",
            (
                "Duplicate observations are merged by requiring exact values to "
                "agree, maximizing certified lower bounds, and retaining the "
                "smallest reported endpoint together with whether that endpoint is "
                "rigorous or decoder-only.  "
                "For one representative of each of 492 "
                "presentation classes containing all 858 proposals formerly lacking "
                "direct lower-bound evidence, we "
                "exhaustively searched both CSS logical sectors for "
                "nontrivial operators of weight at most four.  The search enumerates "
                "supports of at most two qubits and combines pairs whose syndromes "
                "cancel, rejecting combinations that are stabilizers.  The audit "
                "resolved 64 classes (94 proposals) at low weight: two "
                r"$d=2$ proposals were omitted, and 62 classes (92 retained "
                r"proposals) have $d=3$ or 4.  For the other 428 classes "
                r"(764 proposals), it certified $d\geq5$.  Every "
                "retained row has either an exact distance or a certified "
                "positive lower bound paired with a reported endpoint.  "
                "Before the final class merge, "
                f"{statuses['E']:,} proposals have exact row-level status and "
                f"{statuses['C']:,} have unresolved row-level status.  Of the latter, "
                f"{nonexact_direct_evidence['interval']:,} have rigorous upper "
                "bounds and "
                f"{nonexact_direct_evidence['estimate']:,} retain decoder estimates.  "
                f"{transferred_low_weight_exact:,} proposals with exact row-level "
                "status inherit an exact "
                r"$d\leq4$ result from an audited class representative.  A separate "
                f"set of {transferred_low_weight_completion:,} proposals with exact "
                "row-level status combines a "
                r"transferred $d\geq5$ result with its own weight-five witness.  "
                "These transferred-evidence subsets measure a different property "
                "from the 45 proposals whose decoder estimates of 5 were matched "
                "by reconstructed and independently verified weight-five witnesses."
            ),
            "",
            (
                "Merging distance evidence within each presentation class gives "
                f"{class_statuses['E']:,} exact classes and "
                f"{class_statuses['C']:,} classes with a certified lower bound "
                "and reported endpoint: "
                f"{nonexact_class_evidence['interval']:,} rigorous intervals and "
                f"{nonexact_class_evidence['estimate']:,} lower-bound--plus--estimate "
                "records.  The exact "
                f"classes cover {exact_class_members:,} proposals: "
                f"{statuses['E']:,} already had exact row-level status before this "
                f"merge and {inherited_exact_statuses['C']:,} acquire it by class "
                "isomorphism.  Class merging tightens without establishing an exact value "
                f"for another {strengthened_status_pairs[('C', 'C')]:,} rows.  The "
                "campaign tables separate each proposal's own evidence from the "
                "class effect.  Where a class effect is shown, the displayed "
                r"$d$ and FOM use the class-level evidence, whereas Evid. lists only "
                "evidence attached to that proposal.  The figure of merit is "
                r"$\mathrm{FOM}=kd^2/n$; values are rounded to two decimals unless "
                "three are needed to distinguish a close comparison."
            ),
            "",
            r"\subsection{Connectivity and component classes}",
            (
                "The basis-independent connectivity test finds that "
                f"{disconnected_total:,} of {total_specs:,} retained proposals "
                f"are disconnected, including {disconnected_exact:,} of the "
                f"{statuses['E']:,} proposals with exact row-level status.  By family, "
                f"307 of the {specs_by_family['CSS']:,} CSS proposals and 102 "
                f"of the {specs_by_family['PBB']:,} PBB proposals are disconnected.  "
                "The component partition agrees with the generator-graph partition "
                "for every retained row.  At the class level, "
                f"{disconnected_classes:,} of {len(presentation_classes):,} classes "
                "are disconnected, including "
                f"{disconnected_exact_classes:,} of "
                f"{class_statuses['E']:,} exact classes."
            ),
            "",
            (
                r"The high-$k$ component audit resolves six retained direct sums: "
                r"$10\times\code{42,6,3}$, $16\times\code{24,4,3}$, "
                r"$12\times\code{36,4,3}$, $10\times\code{42,4,5}$, "
                r"$8\times\code{48,4,3}$, and $8\times\code{48,4,6}$.  "
                r"The seventh, $18\times\code{24,4,2}$, is excluded from the "
                r"retained catalogue because $d=2$."
            ),
            "",
            (
                "Selecting one connected component from each parent presentation and "
                "repeating the canonical labelling reduces "
                f"{component_class_counts['parent_presentation_classes']:,} parent "
                "classes to "
                f"{component_class_counts['component_classes']:,} component classes: "
                f"{component_class_counts['component_classes_by_family']['CSS']:,} "
                "CSS and "
                f"{component_class_counts['component_classes_by_family']['PBB']:,} "
                "PBB, of which "
                f"{component_class_counts['component_classes_by_status']['E']:,} "
                "are exact and "
                f"{component_class_counts['component_classes_by_status']['C']:,} "
                "are nonexact.  Because this comparison uses "
                "presentation-graph isomorphism rather than complete "
                "stabilizer-code equivalence, "
                f"{component_class_counts['component_classes']:,} remains an upper "
                "bound on the number of inequivalent component codes."
            ),
            "",
            r"\subsection{Local-Clifford screen}",
            (
                "A separate complete local-Clifford-to-CSS audit covers all four "
                r"exact PBB component classes with parameters $\code{12,2,3}$.  "
                "An exhaustive branch-and-bound search over the "
                r"$6^{12}$ assignments of Pauli-axis permutations to the 12 qubits "
                "shows that none is locally Clifford equivalent to CSS.  The "
                "machine-readable certificates are available through the "
                r"project repository~\cite{cruzbenito2026qcode}."
            ),
            "",
            r"\subsection{Table conventions}",
            (
                "Evidence codes are Exh. (exact exhaustive search), H$_c$ (exact "
                "component enumeration plus isomorphism and dimension additivity), "
                "M (MILP optimality for every logical sector), $L_w$ (exhaustive "
                "exclusion of logical operators through weight $w$), I (MILP incumbent), "
                "W (independently reconstructed and verified logical operator), and B "
                "(BP--OSD/decoder estimate; operator not retained).  For a nonexact "
                r"row, $[L,U]$ denotes a certified lower bound and a rigorous upper "
                r"bound, whereas $L;\widehat{U}$ pairs the certified lower bound "
                "with a decoder estimate; the FOM column uses the same notation.  "
                "Own evidence labels these cases Interval and Estimate, respectively.  "
                "Class effect identifies an exact value, a tighter endpoint, or a "
                "rigorous endpoint supplied by an isomorphic presentation.  Obs. is the "
                "number of raw-log observations, not an equivalence-class "
                "multiplicity.  Decomp. gives the connectivity decomposition "
                "computed from the displayed polynomials: "
                r"\texttt{conn.} for one component, or "
                r"$r\times\code{n_0,k_0}$ for $r$ "
                "pairwise-isomorphic disjoint components with parameters "
                r"$\code{n_0,k_0}$.  Each component has the same true distance "
                "as the direct sum, so the row's exact distance or its certified "
                "lower bound and reported endpoint carry over unchanged."
            ),
            "",
            r"\subsection{Generator attribution and artifacts}",
            (
                "Attr. uses G for GPT-5.6 Sol, O for Claude Opus 5, N for the "
                "seed/null source, F for the fixed PBB baseline evaluated in every "
                "run, and -- when no attribution record exists.  These labels record "
                "associated generator programs, not independent discovery or the "
                "distance-certification method.  "
                "At the proposal level, the retained CSS association counts are "
                "G=739 and O=313, with 181 in both sets, 35 seed/null-only rows, "
                "and six rows without an event.  For PBB-small, the attribution "
                "counts are 90 G, 51 G/O, 2 G/N, and 8 G/O/N; for PBB-large, they "
                "are 8 G, 56 G/O, 11 G/O/N, and 4 O.  F marks 20 retained "
                "fixed-baseline proposals and is orthogonal to the G/O/N "
                "partition: it can be appended to any model-association label and is "
                "not included in those partition counts.  Twenty-five PBB-large "
                "bursts remain ambiguous at the individual-program level, although "
                "their sets of associated models are fixed."
            ),
            "",
            (
                f"For the {len(pbb_audit_specs):,} retained PBB-large targets, the "
                "completed audit certifies "
                f"{pbb_audit_statuses['E']:,} exact distances and leaves "
                f"{pbb_audit_statuses['C']:,} ranges with certified lower bounds "
                "and supported upper endpoints.  Further details, "
                "the complete machine-readable catalogue, and supporting evidence "
                r"are available through the project repository~"
                r"\cite{cruzbenito2026qcode}."
            ),
            "",
        ]
    )
    lines.extend(render_summary(campaign_data, presentation_classes))
    lines.extend(["", *render_explicit_component_summary()])
    lines.extend(["", *render_explicit_pbb_component_summary()])
    lines.extend(["", *render_exact_parameter_summary(presentation_classes)])

    for campaign, specs, _, _ in campaign_data:
        campaign_class_count = len(
            {class_of[spec.spec_id] for spec in specs}
        )
        displayed_specs = [spec for spec in specs if spec.spec_id in displayed_ids]
        lines.extend(
            [
                "",
                r"\begingroup",
                r"\setlength{\tabcolsep}{0.84pt}",
                r"\renewcommand{\arraystretch}{0.96}",
            ]
        )
        lines.extend(
            table_header(
                campaign,
                len(specs),
                len(displayed_specs),
                campaign_class_count,
            )
        )
        for spec in displayed_specs:
            class_id = class_of[spec.spec_id]
            lines.append(
                table_row(spec, class_labels[class_id], class_by_id[class_id])
            )
        lines.extend([r"\end{longtable}", r"\endgroup"])

    lines.append("")
    return "\n".join(lines)


def build_catalogue(
    *,
    include_css_low_weight_audit: bool = True,
    include_css_weight5_witnesses: bool = True,
    include_css_upper_bound_witnesses: bool = True,
) -> list[tuple[Campaign, list[Spec], int, int]]:
    loaded_campaigns: list[
        tuple[Campaign, dict[tuple, Spec], int, int]
    ] = []
    seen_ids: set[str] = set()
    specs_by_id: dict[str, Spec] = {}
    source_spec_count = 0
    for campaign in CAMPAIGNS:
        specs_by_key, raw_rows = load_specs(campaign)
        event_rows = merge_events(campaign, specs_by_key)
        attribution_rows = merge_historical_attribution(campaign, specs_by_key)
        if event_rows and attribution_rows:
            raise ValueError(f"{campaign.slug}: mixed native and recovered provenance")
        verification_rows = merge_verification(campaign, specs_by_key)
        merge_component_certifications(campaign, specs_by_key)
        all_specs = sorted(specs_by_key.values(), key=sort_key)
        source_spec_count += len(all_specs)
        for spec in all_specs:
            spec.final_bounds()
            if spec.spec_id in seen_ids:
                raise ValueError(f"truncated specification-ID collision: {spec.spec_id}")
            seen_ids.add(spec.spec_id)
            specs_by_id[spec.spec_id] = spec
        loaded_campaigns.append(
            (campaign, specs_by_key, raw_rows, verification_rows)
        )

    if source_spec_count != 1297:
        raise ValueError("source catalogue must contain exactly 1,297 specifications")

    if include_css_low_weight_audit:
        load_and_merge_css_low_weight_audit(specs_by_id)
    if include_css_weight5_witnesses:
        load_and_merge_css_weight5_witnesses(specs_by_id)
    if include_css_upper_bound_witnesses:
        load_and_merge_css_upper_bound_witnesses(specs_by_id)

    campaign_data = []
    for campaign, specs_by_key, raw_rows, verification_rows in loaded_campaigns:
        all_specs = sorted(specs_by_key.values(), key=sort_key)
        specs = [spec for spec in all_specs if retain_for_catalogue(spec)]
        campaign_data.append((campaign, specs, raw_rows, verification_rows))
    return campaign_data


def print_audit(
    campaign_data: list[tuple[Campaign, list[Spec], int, int]],
    presentation_classes: list[PresentationClass],
) -> None:
    class_of = {
        spec.spec_id: group.class_id
        for group in presentation_classes
        for spec in group.members
    }
    print(
        "campaign\tlogged_obs\tretained_obs\tall_specs\tretained_specs\t"
        "presentation_classes\tremoved_specs\tverify_rows\texact\tnonexact\tendpoint_only"
    )
    totals = Counter()
    for campaign, specs, raw_rows, verification_rows in campaign_data:
        statuses = Counter(status_for(spec) for spec in specs)
        class_count = len({class_of[spec.spec_id] for spec in specs})
        retained_observations = sum(spec.observations for spec in specs)
        removed_specs = campaign.expected_specs - len(specs)
        print(
            f"{campaign.slug}\t{raw_rows}\t{retained_observations}\t"
            f"{campaign.expected_specs}\t{len(specs)}\t{class_count}\t{removed_specs}\t"
            f"{verification_rows}\t"
            f"{statuses['E']}\t{statuses['C']}\t{statuses['U']}"
        )
        totals.update(
            logged_observations=raw_rows,
            retained_observations=retained_observations,
            all_specs=campaign.expected_specs,
            retained_specs=len(specs),
            removed_specs=removed_specs,
            verification_rows=verification_rows,
            exact=statuses["E"],
            nonexact=statuses["C"],
            endpoint_only=statuses["U"],
        )
    print(
        f"total\t{totals['logged_observations']}\t"
        f"{totals['retained_observations']}\t{totals['all_specs']}\t"
        f"{totals['retained_specs']}\t{len(presentation_classes)}\t"
        f"{totals['removed_specs']}\t"
        f"{totals['verification_rows']}\t{totals['exact']}\t"
        f"{totals['nonexact']}\t{totals['endpoint_only']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output include (default: {DEFAULT_OUTPUT.relative_to(ROOT)})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the existing output differs from freshly rendered input",
    )
    parser.add_argument(
        "--class-output",
        type=Path,
        default=DEFAULT_CLASS_OUTPUT,
        help=(
            "stored-presentation class sidecar "
            f"(default: {DEFAULT_CLASS_OUTPUT.relative_to(ROOT)})"
        ),
    )
    args = parser.parse_args()

    campaign_data = build_catalogue()
    all_specs = [spec for _, specs, _, _ in campaign_data for spec in specs]
    presentation_classes = build_presentation_classes(all_specs)
    output = render(campaign_data, presentation_classes)
    class_output = render_presentation_classes(presentation_classes)
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    class_output_path = (
        args.class_output
        if args.class_output.is_absolute()
        else ROOT / args.class_output
    )

    if args.check:
        if not output_path.exists():
            print(f"missing generated file: {output_path}", file=sys.stderr)
            return 1
        if output_path.read_text(encoding="utf-8") != output:
            print(f"generated file is stale: {output_path}", file=sys.stderr)
            return 1
        if not class_output_path.exists():
            print(f"missing generated file: {class_output_path}", file=sys.stderr)
            return 1
        if class_output_path.read_text(encoding="utf-8") != class_output:
            print(f"generated file is stale: {class_output_path}", file=sys.stderr)
            return 1
        try:
            display_path = output_path.relative_to(ROOT)
        except ValueError:
            display_path = output_path
        print(f"up to date: {display_path}")
        try:
            display_class_path = class_output_path.relative_to(ROOT)
        except ValueError:
            display_class_path = class_output_path
        print(f"up to date: {display_class_path}")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
        class_output_path.parent.mkdir(parents=True, exist_ok=True)
        class_output_path.write_text(class_output, encoding="utf-8")
        try:
            display_path = output_path.relative_to(ROOT)
        except ValueError:
            display_path = output_path
        print(f"wrote {display_path}")
        try:
            display_class_path = class_output_path.relative_to(ROOT)
        except ValueError:
            display_class_path = class_output_path
        print(f"wrote {display_class_path}")

    print_audit(campaign_data, presentation_classes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
