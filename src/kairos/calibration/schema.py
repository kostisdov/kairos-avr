"""Typed contracts for calibration: snapshot manifests, observed profiles, evidence targets and curves,
population specifications, parameter definitions and calibration statuses.

Every contract is a plain dataclass with ``to_dict``/``from_dict`` so artifacts are JSON/YAML/Parquet
friendly and reviewable.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields

SCHEMA_VERSION = "1"

# value provenance (shared vocabulary with the standardization design)
ACCEPTED_ORIGINS = ("observed", "derived")
REJECTED_ORIGINS = ("schema_default", "synthetic_default", "synthetic_companion", "placeholder")
SCAFFOLD_SOURCE_KINDS = ("real_with_synthetic_defaults",)

TARGET_KINDS = ("proportion", "mean", "sd", "quantile", "correlation", "longitudinal_mean", "km_survival",
                "cumulative_incidence", "log_effect")
CURVE_KINDS = ("km_survival", "cumulative_incidence")
ROLES = ("fit", "holdout", "sensitivity", "excluded")
REVIEW_STATUSES = ("proposed", "approved", "rejected", "approved_synthetic_test")
APPROVED = ("approved", "approved_synthetic_test")
UNCERTAINTY_TYPES = ("se", "sd", "ci95", "none")
EFFECT_TYPES = ("cause_specific_log_hr", "subdistribution_log_hr", "log_odds_ratio", "log_risk_ratio")

FIT_STATUSES = ("accepted_for_simulation", "failed_targets", "insufficient_evidence", "nonidentifiable", "incomplete")
VALIDATION_STATUSES = ("passed", "failed", "unavailable")
SUPPORT_LEVELS = ("fit_local", "descriptive_only", "not_estimated")


def _clean(obj):
    if isinstance(obj, float) and obj != obj:
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


class Contract:
    def to_dict(self) -> dict:
        return _clean(asdict(self))

    @classmethod
    def from_dict(cls, d: dict):
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"{cls.__name__}: unknown fields {sorted(unknown)}")
        return cls(**{k: v for k, v in d.items() if k in names})


def fingerprint(obj) -> str:
    return hashlib.sha256(json.dumps(_clean(obj), sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class SnapshotManifest(Contract):
    snapshot_id: str
    source_kind: str = "real"
    purpose: str = "research"
    published: bool = False
    frozen: bool = False
    table_hashes: dict = field(default_factory=dict)
    vocabulary_version: str | None = None
    concept_set_version: str | None = None
    lineage_complete: bool = False
    created_at: str | None = None


@dataclass
class SummaryStat(Contract):
    variable: str
    statistic: str                     # proportion:<level> | mean | sd | q<pp> | correlation:<other> | slope_per_year | annual_mean:<k>
    value: float | None
    se: float | None = None
    n_patients: int = 0
    n_records: int = 0
    kind: str = "baseline"             # baseline | cross_sectional | longitudinal | dependency | annual_descriptive
    window: str = ""
    unit: str = ""
    support: str = "not_estimated"
    reason: str = ""


@dataclass
class ObservedProfile(Contract):
    snapshot_id: str
    population_id: str
    split_role: str
    policy: dict
    persons_in_scope: int
    summaries: list = field(default_factory=list)       # SummaryStat dicts
    exclusions: dict = field(default_factory=dict)
    not_estimated: list = field(default_factory=list)
    snapshot_fingerprint: str = ""
    schema_version: str = SCHEMA_VERSION


@dataclass
class EvidenceTarget(Contract):
    target_id: str
    kind: str
    variable: str
    statistic: str = ""
    estimate: float | None = None
    uncertainty: dict = field(default_factory=lambda: {"type": "none"})
    sample_size: int | None = None
    population: dict = field(default_factory=dict)       # study, arm, route, devices, era, setting, view
    time: dict = field(default_factory=dict)             # origin, horizon_years, unit, censoring, competing_events
    outcome_definition: str | None = None
    effect_type: str | None = None
    unit: str | None = None
    study_group_id: str = ""
    overlapping_groups: list = field(default_factory=list)
    source: dict = field(default_factory=dict)           # citation, location, release_hash, extraction
    review_status: str = "proposed"
    reviewer: str | None = None
    role: str = "excluded"
    compatibility: dict = field(default_factory=lambda: {"status": "unassessed", "reason": ""})
    acceptance: dict = field(default_factory=lambda: {"tolerance": None, "tolerance_type": "absolute", "weight": 1.0,
                                                      "mandatory": True})
    curve_id: str | None = None
    origin: str = "external"                             # external | local_profile | synthetic_recovery

    def standard_error(self) -> float | None:
        """Standard error of the estimate from its declared uncertainty; SD is never taken as SE."""
        u = self.uncertainty or {}
        t = u.get("type", "none")
        if t == "se":
            return float(u["value"])
        if t == "ci95":
            return (float(u["upper"]) - float(u["lower"])) / (2 * 1.959964)
        if t == "sd" and self.sample_size:
            return float(u["value"]) / float(self.sample_size) ** 0.5
        return None


@dataclass
class CurvePoint(Contract):
    target_id: str
    time: float
    estimate: float
    lower: float | None = None
    upper: float | None = None
    interval_type: str = "none"          # pointwise_ci95 | none
    n_at_risk: int | None = None
    digitized: bool = False
    image_ref: str | None = None


@dataclass
class PopulationSpec(Contract):
    population_id: str
    preset: str                          # local_export_like | external_population
    description: str = ""
    index_procedure: str = "aortic valve replacement with a bioprosthesis"
    routes: dict = field(default_factory=dict)          # route -> share, or empty for scenario default
    devices: list = field(default_factory=list)
    implant_years: list = field(default_factory=lambda: [2012, 2020])
    study_end: str = "2026-06-30"
    max_followup_years: float = 8.0
    time_zero: str = "reference_echo"    # KAIROS model time zero; study views declare their own origins
    event_definitions: dict = field(default_factory=lambda: {
        "death": "all_cause_death", "replacement": "non_svd_index_valve_replacement",
        "svd": "kairos_adjudicated_moderate_or_severe_svd_v2"})
    study_views: dict = field(default_factory=lambda: {
        "enrolled_implantation_origin": {"origin": "implantation", "includes_pre_reference_events": True},
        "kairos_reference_eligible": {"origin": "reference_echo", "includes_pre_reference_events": False}})
    sampling_weights: dict = field(default_factory=dict)
    observation_policy: dict = field(default_factory=dict)
    base_scenario: str = "gradual_stenotic"
    base_variant: str | None = None
    scope_statement: str = ""


@dataclass
class ParameterDef(Contract):
    name: str
    path: str                            # dotted path into scenario params
    lower: float
    upper: float
    transform: str = "identity"          # identity | log | logit
    prior_mean: float | None = None
    prior_sd: float | None = None        # on the transformed scale
    status: str = "free"                 # free | fixed
    value: float | None = None           # fixed value or starting value
    evidence_links: list = field(default_factory=list)
    label: str = "assumed"
    notes: str = ""
