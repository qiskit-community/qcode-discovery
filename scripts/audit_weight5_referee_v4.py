#!/usr/bin/env python3
"""Build the focused evidence artifact requested by the v4 referee report.

This audit is intentionally separate from the manuscript generator.  It checks
four claims that are easy to state imprecisely in prose:

* why the PBB catalogue has no rows on ``(10, 9)`` or ``(16, 12)``;
* agreement of the CSS X/Z low-weight and exact-MILP results;
* what MILP metadata was retained (and, importantly, what was not); and
* ideal conflict-only measurement schedules for two weight-five codes and
  their length-matched weight-six QLDPC Challenge comparators.

The QLDPC Challenge inputs are read from a pinned checkout.  The default path
matches the local audit checkout, but another path may be supplied explicitly::

    python scripts/audit_weight5_referee_v4.py \
        --challenge-root /path/to/qldpc-challenge

The generated JSON embeds the two comparator check matrices and all schedule
layers, so its claims remain independently checkable without that checkout.
No manuscript or supplemental source is read or written.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, deque
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "results" / "weight5_referee_v4_checks.json"
DEFAULT_CHALLENGE_ROOT = Path("/tmp/qldpc-challenge-src")

SCHEMA_VERSION = 1
ARTIFACT_SCHEMA = "weight5_referee_v4_checks_v1"
CHALLENGE_COMMIT = "3b50503b058a0c09500c1017b54c3d547e1101cc"

PBB_EVALUATOR = ROOT / "evolve" / "openevolve_evaluator_weight5_pbb.py"
PBB_EVALUATOR_COMMIT = "aa0f18b600e38c4f5459a1fbfaee0836d9cf99ab"
PBB_EVALUATOR_SHA256 = "b41d27e3784bd4d6aac0155b404761d6595f4610792933810b1900171edd9223"
PBB_RUN_MANIFESTS = {
    "pbb-small": (
        ROOT
        / "results"
        / "evolution"
        / "weight5_pbb_small_v1"
        / "run_manifest.json"
    ),
    "pbb-large": (
        ROOT
        / "results"
        / "evolution"
        / "weight5_pbb_large_v1"
        / "run_manifest.json"
    ),
}
CATALOGUE = ROOT / "results" / "weight5_publication_catalogue.jsonl"
LOW_WEIGHT = ROOT / "results" / "weight5_css_low_weight_audit.jsonl"
WEIGHT_FIVE_WITNESSES = (
    ROOT / "results" / "weight5_css_weight5_witnesses.jsonl"
)
DISTANCE_MILP = ROOT / "evaluation" / "distance_milp.py"

CSS_MILP = {
    "css-small": ROOT / "results" / "weight5_css_small_v1_milp_verified.jsonl",
    "css-large": ROOT / "results" / "weight5_css_large_v1_milp_verified.jsonl",
}
PBB_RAW = {
    "pbb-small": (
        ROOT
        / "results"
        / "evolution"
        / "weight5_pbb_small_v1"
        / "all_codes_weight5_pbb.jsonl"
    ),
    "pbb-large": (
        ROOT
        / "results"
        / "evolution"
        / "weight5_pbb_large_v1"
        / "all_codes_weight5_pbb.jsonl"
    ),
}
PBB_MILP = {
    "pbb-small": ROOT / "results" / "weight5_pbb_small_v1_deep_milp_verify.jsonl",
    "pbb-large": ROOT / "results" / "weight5_pbb_large_v1_deep_milp_verify.jsonl",
}

CHALLENGE_CODES = {
    "96-4-12": {
        "sha256": "1ce3274dc0d81349baa99221bf7cd763cb153db0eb1b45c67589a0c4b4d2bd4b",
        "distance_status": "exact",
    },
    "140-6-14": {
        "sha256": "41b4735be64ab57d41df146c7a4110af3802f2da4c75ab7a2e4b30bda5302c7c",
        "distance_status": "witness_upper_bound",
    },
}

AUTHORS_CODES = {
    "96-4-10": {
        "ell": 16,
        "m": 3,
        "n": 96,
        "k": 4,
        "d": 10,
        "A_terms": [(0, 0), (0, 1), (2, 2)],
        "B_terms": [(0, 0), (3, 0)],
        "source": {
            "path": "results/weight5_component_certifications.jsonl",
            "match_id": "W5BB-96-4-10",
        },
    },
    "140-6-10": {
        "ell": 5,
        "m": 14,
        "n": 140,
        "k": 6,
        "d": 10,
        "A_terms": [(0, 0), (0, 13), (4, 11)],
        "B_terms": [(0, 0), (1, 7)],
        "source": {
            "path": "results/weight5_component_certifications.jsonl",
            "match_id": "W5BB-140-6-10",
        },
    },
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _source(path: Path) -> dict[str, Any]:
    return {
        "path": _relative(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_path(path),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _normalized_terms(value: object) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(tuple(int(x) for x in term) for term in (value or [])))


def _pbb_key(row: dict[str, Any], *, catalogue: bool = False) -> tuple[Any, ...]:
    if catalogue:
        params = row["parameters"]
        generators = row["generators"]
        return (
            int(params["ell"]),
            int(params["m"]),
            _normalized_terms(generators.get("A_terms")),
            _normalized_terms(generators.get("B_terms")),
            _normalized_terms(generators.get("C_terms")),
            _normalized_terms(generators.get("D_terms")),
        )
    return (
        int(row["ell"]),
        int(row["m"]),
        _normalized_terms(row.get("A_terms")),
        _normalized_terms(row.get("B_terms")),
        _normalized_terms(row.get("C_terms")),
        _normalized_terms(row.get("D_terms")),
    )


def _literal_assignment(source: str, name: str) -> Any:
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            value = node.value
            if value is None:
                break
            return ast.literal_eval(value)
    raise ValueError(f"could not find literal assignment {name}")


def _historical_pbb_evaluator() -> str:
    relative_path = PBB_EVALUATOR.relative_to(ROOT).as_posix()
    result = subprocess.run(
        ["git", "show", f"{PBB_EVALUATOR_COMMIT}:{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    if _sha256_bytes(result.stdout) != PBB_EVALUATOR_SHA256:
        raise ValueError("historical PBB evaluator blob hash changed")
    return result.stdout.decode("utf-8")


def audit_pbb_lattice_coverage(catalogue_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    evaluator_source = _historical_pbb_evaluator()
    if "_log_code_jsonl(result, run_name=run_name)" not in evaluator_source:
        raise ValueError("historical distance-result logging call was not found")

    manifests: dict[str, Any] = {}
    for campaign, path in PBB_RUN_MANIFESTS.items():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("git_commit") != PBB_EVALUATOR_COMMIT:
            raise ValueError(f"{campaign}: historical evaluator commit changed")
        if manifest.get("evaluator_hash") != PBB_EVALUATOR_SHA256:
            raise ValueError(f"{campaign}: historical evaluator hash changed")
        manifests[campaign] = _source(path)

    profiles = {
        "pbb-small": {
            "stage1": _literal_assignment(evaluator_source, "_SMALL_STAGE1"),
            "stage2_additions": _literal_assignment(
                evaluator_source, "_SMALL_STAGE2_EXTRA"
            ),
        },
        "pbb-large": {
            "stage1": _literal_assignment(evaluator_source, "_LARGE_STAGE1"),
            "stage2_additions": _literal_assignment(
                evaluator_source, "_LARGE_STAGE2_EXTRA"
            ),
        },
    }

    retained_by_campaign_lattice = Counter(
        (
            row["campaign"],
            int(row["parameters"]["ell"]),
            int(row["parameters"]["m"]),
        )
        for row in catalogue_rows
        if row.get("record_type") == "weight5_presentation"
        and row.get("family") == "PBB"
    )

    output: dict[str, Any] = {}
    all_missing: list[list[int]] = []
    for campaign, stages in profiles.items():
        raw_rows = _load_jsonl(PBB_RAW[campaign])
        if not all(int(row.get("d", 0)) > 0 for row in raw_rows):
            raise ValueError(f"{campaign}: raw observation log contains d <= 0")
        observations = Counter((int(row["ell"]), int(row["m"])) for row in raw_rows)
        distinct: dict[tuple[int, int], set[tuple[Any, ...]]] = {}
        for row in raw_rows:
            lattice = (int(row["ell"]), int(row["m"]))
            distinct.setdefault(lattice, set()).add(_pbb_key(row))

        lattice_records = []
        for role in ("stage1", "stage2_additions"):
            for ell, m in stages[role]:
                count = observations[(ell, m)]
                if count == 0:
                    all_missing.append([ell, m])
                lattice_records.append(
                    {
                        "lattice": [ell, m],
                        "stage_role": (
                            "stage_1" if role == "stage1" else "stage_2_addition"
                        ),
                        "positive_distance_observations": count,
                        "distinct_positive_distance_tuples": len(
                            distinct.get((ell, m), set())
                        ),
                        "retained_catalogue_presentations": retained_by_campaign_lattice[
                            (campaign, ell, m)
                        ],
                    }
                )
        output[campaign] = {
            "raw_log": _source(PBB_RAW[campaign]),
            "lattices": lattice_records,
        }

    if sorted(all_missing) != [[10, 9], [16, 12]]:
        raise ValueError(f"unexpected zero-observation PBB lattices: {all_missing}")
    return {
        "profiles": output,
        "zero_positive_observation_lattices": sorted(all_missing),
        "finding": (
            "Both absent lattices are Stage-2 additions, not Stage-1 lattices. "
            "The retained logs establish that no positive-distance observation "
            "was persisted on either lattice; they do not distinguish candidate "
            "selection, worker failure/timeout, or a structural obstruction."
        ),
        "historical_evaluator": {
            "path": PBB_EVALUATOR.relative_to(ROOT).as_posix(),
            "git_commit": PBB_EVALUATOR_COMMIT,
            "sha256": PBB_EVALUATOR_SHA256,
        },
        "run_manifests": manifests,
        "logging_semantics": (
            "The historical evaluator called its JSONL logger for each returned "
            "distance-worker result. Every surviving raw-log row has d > 0."
        ),
    }


def _sector_outcome(sector: dict[str, Any]) -> tuple[str, int]:
    exact = sector.get("exact_distance")
    if exact is not None:
        return "exact", int(exact)
    return "certified_lower_bound", int(sector["certified_lower_bound"])


def audit_css_sector_agreement() -> dict[str, Any]:
    low_rows = [
        row
        for row in _load_jsonl(LOW_WEIGHT)
        if row.get("record_type") == "css_low_weight_class_audit"
    ]
    outcomes: Counter[str] = Counter()
    disagreements = []
    for row in low_rows:
        x_outcome = _sector_outcome(row["search"]["X"])
        z_outcome = _sector_outcome(row["search"]["Z"])
        if x_outcome != z_outcome:
            disagreements.append(row["class_identity"]["class_id"])
        key = (
            f"exact_{x_outcome[1]}"
            if x_outcome[0] == "exact"
            else f"both_certified_at_least_{x_outcome[1]}"
        )
        outcomes[key] += 1
    if disagreements:
        raise ValueError(f"low-weight X/Z disagreements: {disagreements}")
    expected_outcomes = {
        "both_certified_at_least_5": 428,
        "exact_2": 2,
        "exact_3": 25,
        "exact_4": 37,
    }
    if dict(sorted(outcomes.items())) != expected_outcomes:
        raise ValueError(f"unexpected low-weight outcomes: {outcomes}")

    milp_profiles: dict[str, Any] = {}
    exact_milp_total = 0
    for campaign, path in CSS_MILP.items():
        rows = _load_jsonl(path)
        exact_rows = [
            row
            for row in rows
            if row.get("stage") == "milp_exact"
            and row.get("d_is_exact") is True
            and (row.get("milp_details") or {}).get("exact") is True
            and (row.get("milp_details") or {}).get("d_x_computed") is True
        ]
        disagreements = [
            {
                "ell": row["ell"],
                "m": row["m"],
                "d_x": row["milp_details"]["d_x"],
                "d_z": row["milp_details"]["d_z"],
            }
            for row in exact_rows
            if row["milp_details"]["d_x"] != row["milp_details"]["d_z"]
        ]
        if disagreements:
            raise ValueError(f"{campaign}: exact MILP X/Z disagreements")

        sentinel_rows = [
            row
            for row in rows
            if row.get("milp_details")
            and (
                row["milp_details"].get("d_x_computed") is False
                or row["milp_details"].get("d_x") == row.get("n")
                or row["milp_details"].get("d_z") == row.get("n")
            )
        ]
        milp_profiles[campaign] = {
            "source": _source(path),
            "exact_records": len(exact_rows),
            "exact_records_with_d_x_equal_d_z": len(exact_rows),
            "excluded_uncomputed_sector_sentinel_records": len(sentinel_rows),
        }
        exact_milp_total += len(exact_rows)

    witness_rows = _load_jsonl(WEIGHT_FIVE_WITNESSES)
    witness_manifest = witness_rows[0]
    witness_audits = witness_rows[1:]
    invalid_witnesses = []
    for row in witness_audits:
        for sector in ("X", "Z"):
            witness = row["replacement_explicit_witnesses"][sector]
            if not (
                witness.get("weight") == 5
                and witness.get("zero_check_syndrome") is True
                and witness.get("outside_stabilizer_rowspace") is True
            ):
                invalid_witnesses.append([row["presentation_id"], sector])
    if invalid_witnesses:
        raise ValueError(f"invalid weight-five witnesses: {invalid_witnesses}")

    return {
        "low_weight_audit": {
            "source": _source(LOW_WEIGHT),
            "classes_checked": len(low_rows),
            "x_z_agreements": len(low_rows),
            "outcomes": expected_outcomes,
        },
        "exact_css_milp": {
            "profiles": milp_profiles,
            "exact_records": exact_milp_total,
            "x_z_agreements": exact_milp_total,
            "sentinel_policy": (
                "Only rows with both sectors computed and solver-exact are "
                "compared; n-valued uncomputed-sector sentinels are excluded."
            ),
        },
        "replacement_weight_five_witnesses": {
            "source": _source(WEIGHT_FIVE_WITNESSES),
            "presentations": len(witness_audits),
            "presentation_classes": witness_manifest["counts"][
                "presentation_classes"
            ],
            "explicit_logicals": 2 * len(witness_audits),
            "all_directly_validated": True,
        },
    }


def _find_line(source: str, needle: str) -> int:
    for index, line in enumerate(source.splitlines(), 1):
        if needle in line:
            return index
    raise ValueError(f"could not locate source line containing {needle!r}")


def _catalogue_milp_exact_rows(
    catalogue_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for row in catalogue_rows:
        if row.get("record_type") != "weight5_presentation":
            continue
        direct = row.get("distance_direct") or {}
        exact_methods = (direct.get("evidence") or {}).get("exact_methods") or []
        if direct.get("is_exact") is True and "M" in exact_methods:
            output.append(row)
    return output


def _pbb_exact_source_keys(
    campaign: str,
) -> tuple[set[tuple[Any, ...]], set[tuple[Any, ...]]]:
    raw_exact = {
        _pbb_key(row)
        for row in _load_jsonl(PBB_RAW[campaign])
        if row.get("d_method") == "milp_exact" or row.get("milp_exact") is True
    }
    post_campaign_exact = {
        _pbb_key(row)
        for row in _load_jsonl(PBB_MILP[campaign])
        if isinstance(row.get("logicals_optimal"), int)
        and not isinstance(row.get("logicals_optimal"), bool)
        and row.get("total_logicals", 0) > 0
        and row["logicals_optimal"] == row["total_logicals"]
        and row.get("milp_d") is not None
    }
    return raw_exact, post_campaign_exact


def _recursive_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(_recursive_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_recursive_keys(item))
    return keys


def audit_milp_metadata(catalogue_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    source_text = DISTANCE_MILP.read_text(encoding="utf-8")
    required_snippets = (
        "from scipy.optimize import milp, LinearConstraint, Bounds",
        "return w, result.success",
        "# 1. w_j >= x_j",
        "# 2. w_j >= z_j",
    )
    if not all(snippet in source_text for snippet in required_snippets):
        raise ValueError("MILP implementation no longer matches audited formulation")

    exact_rows = _catalogue_milp_exact_rows(catalogue_rows)
    by_campaign = Counter(row["campaign"] for row in exact_rows)
    by_family = Counter(row["family"] for row in exact_rows)
    expected_campaign = {
        "css-large": 38,
        "css-small": 12,
        "pbb-large": 27,
        "pbb-small": 29,
    }
    if dict(sorted(by_campaign.items())) != expected_campaign:
        raise ValueError(f"unexpected direct-M exact counts: {by_campaign}")

    pbb_reconciliation: dict[str, Any] = {}
    for campaign in ("pbb-small", "pbb-large"):
        raw_exact, post_exact = _pbb_exact_source_keys(campaign)
        catalogue_exact = {
            _pbb_key(row, catalogue=True)
            for row in exact_rows
            if row["campaign"] == campaign
        }
        if raw_exact | post_exact != catalogue_exact:
            raise ValueError(f"{campaign}: exact-M provenance does not reconcile")
        pbb_reconciliation[campaign] = {
            "in_search_exact_unique_tuples": len(raw_exact),
            "post_campaign_all_logical_exact_records": len(post_exact),
            "overlap": len(raw_exact & post_exact),
            "catalogue_direct_M_exact_presentations": len(catalogue_exact),
        }

    exact_source_records: list[dict[str, Any]] = []
    for path in CSS_MILP.values():
        exact_source_records.extend(
            row
            for row in _load_jsonl(path)
            if row.get("stage") == "milp_exact" and row.get("d_is_exact") is True
        )
    for _campaign, path in PBB_MILP.items():
        exact_source_records.extend(
            row
            for row in _load_jsonl(path)
            if isinstance(row.get("logicals_optimal"), int)
            and not isinstance(row.get("logicals_optimal"), bool)
            and row.get("total_logicals", 0) > 0
            and row["logicals_optimal"] == row["total_logicals"]
            and row.get("milp_d") is not None
        )
    retained_keys = _recursive_keys(exact_source_records)
    nonretained_native_fields = {
        "dual_bound",
        "mip_dual_bound",
        "mip_gap",
        "solution",
        "solution_vector",
        "solver_log",
        "solver_status_code",
    }
    unexpectedly_retained = sorted(nonretained_native_fields & retained_keys)
    if unexpectedly_retained:
        raise ValueError(f"retention assessment stale: {unexpectedly_retained}")

    import scipy

    try:
        from scipy.optimize._highspy._core import _Highs

        scipy_bundled_highs = _Highs().version()
    except (ImportError, AttributeError):
        scipy_bundled_highs = None
    try:
        standalone_highspy = importlib.metadata.version("highspy")
    except importlib.metadata.PackageNotFoundError:
        standalone_highspy = None

    return {
        "implementation": {
            "source": _source(DISTANCE_MILP),
            "solver": "scipy.optimize.milp using the HiGHS backend",
            "formulation": (
                "Landahl-Anderson-Rice parity MILP; CSS minimizes Hamming "
                "weight and PBB minimizes symplectic support using binary OR variables"
            ),
            "exactness_rule": "the wrapper records optimal=True exactly when result.success is true",
            "release_environment": {
                "scipy": scipy.__version__,
                "scipy_bundled_highs_core": scipy_bundled_highs,
                "standalone_highspy": standalone_highspy,
                "backend_used": (
                    "the HiGHS core bundled inside SciPy and invoked by "
                    "scipy.optimize.milp"
                ),
                "standalone_highspy_used_by_this_path": False,
            },
            "source_lines": {
                "solver_import": _find_line(source_text, required_snippets[0]),
                "css_success_rule": _find_line(source_text, required_snippets[1]),
                "symplectic_or_x": _find_line(source_text, required_snippets[2]),
                "symplectic_or_z": _find_line(source_text, required_snippets[3]),
            },
        },
        "catalogue_direct_M_exact": {
            "total": len(exact_rows),
            "by_family": dict(sorted(by_family.items())),
            "by_campaign": expected_campaign,
            "pbb_source_reconciliation": pbb_reconciliation,
        },
        "retention": {
            "css_exact_records": (
                "aggregate d_X/d_Z, exact flag, optimal/incumbent counts, and timing"
            ),
            "pbb_post_campaign_exact_records": (
                "per-logical optimum flag, objective weight, and elapsed time"
            ),
            "not_retained": sorted(nonretained_native_fields),
            "native_highs_logs_retained": False,
            "formal_optimality_certificates_retained": False,
            "consequence": (
                "The artifacts record solver-success evidence and summary metadata, "
                "but independent reconstruction of a proof requires rerunning the MILP."
            ),
        },
    }


def _shift_index(index: int, dx: int, dy: int, ell: int, m: int) -> int:
    x, y = divmod(index, m)
    return ((x + dx) % ell) * m + ((y + dy) % m)


def bb_schedule_rows_and_layers(
    ell: int,
    m: int,
    A_terms: Iterable[Sequence[int]],
    B_terms: Iterable[Sequence[int]],
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    """Construct combined H_X/H_Z rows and monomial perfect matchings."""
    A = [tuple(map(int, term)) for term in A_terms]
    B = [tuple(map(int, term)) for term in B_terms]
    group_order = ell * m
    n = 2 * group_order
    rows = [set() for _ in range(n)]
    layers: list[dict[str, Any]] = []

    for polynomial, terms in (("A", A), ("B", B)):
        for dx, dy in terms:
            by_check = [-1] * n
            for group_element in range(group_order):
                if polynomial == "A":
                    q_x = _shift_index(group_element, dx, dy, ell, m)
                    q_z = group_order + _shift_index(
                        group_element, -dx, -dy, ell, m
                    )
                else:
                    q_x = group_order + _shift_index(
                        group_element, dx, dy, ell, m
                    )
                    q_z = _shift_index(group_element, -dx, -dy, ell, m)
                by_check[group_element] = q_x
                by_check[group_order + group_element] = q_z
                rows[group_element].add(q_x)
                rows[group_order + group_element].add(q_z)
            layers.append(
                {
                    "label": f"{polynomial}:x^{dx}y^{dy}",
                    "qubit_by_check": by_check,
                }
            )
    return [sorted(row) for row in rows], layers


def _perfect_matching(rows: Sequence[Sequence[int]], n: int) -> list[int]:
    """Deterministic Hopcroft-Karp matching, returned as qubit_by_check."""
    adjacency = [sorted(set(row)) for row in rows]
    pair_check = [-1] * n
    pair_qubit = [-1] * n
    distance = [0] * n
    infinity = n + 1

    def bfs() -> bool:
        queue: deque[int] = deque()
        shortest = infinity
        for check in range(n):
            if pair_check[check] == -1:
                distance[check] = 0
                queue.append(check)
            else:
                distance[check] = infinity
        while queue:
            check = queue.popleft()
            if distance[check] >= shortest:
                continue
            for qubit in adjacency[check]:
                partner = pair_qubit[qubit]
                if partner == -1:
                    shortest = distance[check] + 1
                elif distance[partner] == infinity:
                    distance[partner] = distance[check] + 1
                    queue.append(partner)
        return shortest != infinity

    def dfs(check: int) -> bool:
        for qubit in adjacency[check]:
            partner = pair_qubit[qubit]
            if partner == -1 or (
                distance[partner] == distance[check] + 1 and dfs(partner)
            ):
                pair_check[check] = qubit
                pair_qubit[qubit] = check
                return True
        distance[check] = infinity
        return False

    while bfs():
        for check in range(n):
            if pair_check[check] == -1:
                dfs(check)
    if any(qubit == -1 for qubit in pair_check):
        raise ValueError("regular bipartite graph did not yield a perfect matching")
    return pair_check


def factor_regular_bipartite(rows: Sequence[Sequence[int]], n: int) -> list[dict[str, Any]]:
    remaining = [set(map(int, row)) for row in rows]
    degrees = {len(row) for row in remaining}
    if len(degrees) != 1:
        raise ValueError(f"check side is not regular: {degrees}")
    degree = next(iter(degrees))
    layers = []
    for layer_index in range(degree):
        matching = _perfect_matching([sorted(row) for row in remaining], n)
        for check, qubit in enumerate(matching):
            remaining[check].remove(qubit)
        layers.append(
            {
                "label": f"matching_{layer_index + 1}",
                "qubit_by_check": matching,
            }
        )
    if any(remaining):
        raise AssertionError("edge factorization left uncoloured edges")
    return layers


def validate_schedule(
    rows: Sequence[Sequence[int]], layers: Sequence[dict[str, Any]], n: int
) -> dict[str, Any]:
    if len(rows) != n:
        raise ValueError(f"expected {n} check rows, got {len(rows)}")
    expected_edges = {
        (check, int(qubit))
        for check, row in enumerate(rows)
        for qubit in row
    }
    if len(expected_edges) != sum(len(row) for row in rows):
        raise ValueError("duplicate edge within a check row")
    check_degrees = Counter(check for check, _ in expected_edges)
    qubit_degrees = Counter(qubit for _, qubit in expected_edges)
    if set(check_degrees) != set(range(n)) or set(qubit_degrees) != set(range(n)):
        raise ValueError("combined graph does not cover every check and qubit")
    if len(set(check_degrees.values())) != 1 or len(set(qubit_degrees.values())) != 1:
        raise ValueError("combined graph is not regular on both bipartitions")
    degree = next(iter(check_degrees.values()))
    if set(qubit_degrees.values()) != {degree}:
        raise ValueError("check and qubit degrees differ")
    if len(layers) != degree:
        raise ValueError("number of layers does not equal graph degree")

    scheduled_edges: set[tuple[int, int]] = set()
    for layer in layers:
        matching = list(map(int, layer["qubit_by_check"]))
        if len(matching) != n or sorted(matching) != list(range(n)):
            raise ValueError(f"{layer['label']}: not a perfect matching")
        edges = {(check, qubit) for check, qubit in enumerate(matching)}
        if not edges <= expected_edges:
            raise ValueError(f"{layer['label']}: schedule contains a non-edge")
        if scheduled_edges & edges:
            raise ValueError(f"{layer['label']}: edge appears in multiple layers")
        scheduled_edges.update(edges)
    if scheduled_edges != expected_edges:
        raise ValueError("schedule does not cover the graph exactly")

    return {
        "checks": n,
        "qubits": n,
        "edges": len(expected_edges),
        "regular_degree": degree,
        "schedule_layers": len(layers),
        "each_layer_is_perfect_matching": True,
        "edge_chromatic_number": degree,
        "graph_digest_sha256": _sha256_bytes(
            _canonical_json(sorted([list(edge) for edge in expected_edges]))
        ),
    }


def _schedule_record(
    *,
    slug: str,
    source_kind: str,
    n: int,
    k: int,
    d: int,
    distance_status: str,
    rows: list[list[int]],
    layers: list[dict[str, Any]],
    source: dict[str, Any],
    generators: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validation = validate_schedule(rows, layers, n)
    record: dict[str, Any] = {
        "slug": slug,
        "source_kind": source_kind,
        "parameters": {"n": n, "k": k, "d": d, "distance_status": distance_status},
        "source": source,
        "combined_check_order": "all X rows followed by all Z rows",
        "source_rows": rows,
        "layers": layers,
        "validation": validation,
        "optimality_reason": (
            f"Maximum degree {validation['regular_degree']} is a lower bound; "
            f"the {validation['schedule_layers']} perfect matchings attain it "
            "(equivalently, by Konig's line-colouring theorem)."
        ),
    }
    if generators is not None:
        record["generators"] = generators
    return record


def audit_schedules(challenge_root: Path) -> dict[str, Any]:
    component_path = ROOT / "results" / "weight5_component_certifications.jsonl"
    component_rows = _load_jsonl(component_path)
    by_match = {
        row["match_id"]: row
        for row in component_rows
        if row.get("record_type") == "explicit_bb_component_match"
    }

    records: list[dict[str, Any]] = []
    for slug, expected in AUTHORS_CODES.items():
        match_id = expected["source"]["match_id"]
        source_row = by_match.get(match_id)
        if source_row is None:
            raise ValueError(f"missing component certificate {match_id}")
        actual = source_row["explicit_bb_component"]
        for key in ("ell", "m", "n", "k", "d_exact", "A_terms", "B_terms"):
            expected_key = "d" if key == "d_exact" else key
            expected_value = expected[expected_key]
            if actual[key] != expected_value and _normalized_terms(actual[key]) != _normalized_terms(expected_value):
                raise ValueError(f"{slug}: component certificate changed at {key}")
        rows, layers = bb_schedule_rows_and_layers(
            expected["ell"],
            expected["m"],
            expected["A_terms"],
            expected["B_terms"],
        )
        records.append(
            _schedule_record(
                slug=slug,
                source_kind="this_work_explicit_connected_BB_component",
                n=expected["n"],
                k=expected["k"],
                d=expected["d"],
                distance_status="exact",
                rows=rows,
                layers=layers,
                source={
                    **expected["source"],
                    "artifact_sha256": _sha256_path(component_path),
                },
                generators={
                    "ell": expected["ell"],
                    "m": expected["m"],
                    "A_terms": [list(term) for term in expected["A_terms"]],
                    "B_terms": [list(term) for term in expected["B_terms"]],
                },
            )
        )

    git_head = subprocess.run(
        ["git", "-C", str(challenge_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if git_head != CHALLENGE_COMMIT:
        raise ValueError(
            f"QLDPC Challenge checkout is {git_head}, expected {CHALLENGE_COMMIT}"
        )

    for slug, expected in CHALLENGE_CODES.items():
        path = challenge_root / "codes" / f"{slug}.json"
        if _sha256_path(path) != expected["sha256"]:
            raise ValueError(f"{slug}: pinned QLDPC Challenge input hash changed")
        row = json.loads(path.read_text(encoding="utf-8"))
        rows = [list(map(int, support)) for support in row["checks"]["X"]]
        rows.extend(list(map(int, support)) for support in row["checks"]["Z"])
        layers = factor_regular_bipartite(rows, int(row["n"]))
        records.append(
            _schedule_record(
                slug=slug,
                source_kind="qldpc_challenge_length_matched_comparator",
                n=int(row["n"]),
                k=int(row["k"]),
                d=int(row["distance"]["d"]),
                distance_status=expected["distance_status"],
                rows=rows,
                layers=layers,
                source={
                    "repository": "https://github.com/unitaryfoundation/qldpc-challenge",
                    "commit": CHALLENGE_COMMIT,
                    "path": f"codes/{slug}.json",
                    "sha256": expected["sha256"],
                },
            )
        )

    records.sort(key=lambda row: (row["parameters"]["n"], row["parameters"]["d"]))
    by_slug = {row["slug"]: row for row in records}
    comparisons = []
    for ours, comparator in (("96-4-10", "96-4-12"), ("140-6-10", "140-6-14")):
        ours_layers = by_slug[ours]["validation"]["schedule_layers"]
        comparator_layers = by_slug[comparator]["validation"]["schedule_layers"]
        comparisons.append(
            {
                "this_work": ours,
                "comparator": comparator,
                "this_work_layers": ours_layers,
                "comparator_layers": comparator_layers,
                "ideal_entangling_layer_reduction": comparator_layers - ours_layers,
            }
        )
    return {
        "model": (
            "Ideal conflict-only schedule: one two-qubit interaction per Tanner "
            "edge, with no check ancilla or data qubit used twice in a layer."
        ),
        "scope_limit": (
            "This is an edge-colouring count, not a hardware-routed syndrome "
            "extraction circuit; it omits preparation, measurement, routing, "
            "gate ordering constraints, hook errors, and noise."
        ),
        "codes": records,
        "length_matched_comparisons": comparisons,
    }


def build_artifact(challenge_root: Path) -> dict[str, Any]:
    catalogue_rows = _load_jsonl(CATALOGUE)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_schema": ARTIFACT_SCHEMA,
        "producer": _source(Path(__file__)),
        "catalogue_source": _source(CATALOGUE),
        "pbb_lattice_coverage": audit_pbb_lattice_coverage(catalogue_rows),
        "css_sector_agreement": audit_css_sector_agreement(),
        "milp_provenance_and_retention": audit_milp_metadata(catalogue_rows),
        "ideal_measurement_schedules": audit_schedules(challenge_root),
    }


def _render(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--challenge-root", type=Path, default=DEFAULT_CHALLENGE_ROOT
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the checked-in artifact differs from a fresh audit",
    )
    args = parser.parse_args()

    rendered = _render(build_artifact(args.challenge_root))
    if args.check:
        if not args.output.exists():
            raise SystemExit(f"missing artifact: {args.output}")
        if args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"artifact is stale: {args.output}")
        print(f"verified {args.output}")
        return 0

    _atomic_write(args.output, rendered)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
