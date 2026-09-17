"""Evidence registry, compatibility assessment and curve validation (design section 4.2, G2).

A registry is a YAML file with ``targets`` (:class:`EvidenceTarget`) and ``curves`` (:class:`CurvePoint`,
long form). :func:`validate_evidence` decides, per target and against a :class:`PopulationSpec`, whether the
target may be used and in which role. Decisions are recorded, never silently applied:

* only reviewed targets (``approved``, or ``approved_synthetic_test`` for synthetic recovery tests) can be
  ``fit`` or ``holdout``; automatic extraction may propose but never approve;
* unsupported kinds, a Fine-Gray (subdistribution) effect used as a cause-specific parameter, a 1-KM
  used for a cause with competing events, an endpoint definition that differs from the simulated
  endpoint without a declared reporting operator, a time origin without a matching study view, a horizon
  beyond simulated follow-up, a unit mismatch, suppressed counts, or a missing uncertainty (policy
  ``sensitivity_only``) move the target to ``sensitivity`` or ``excluded`` with the reason;
* curves: ordered times, probabilities in [0, 1], monotone (KM non-increasing, CIF non-decreasing) within a
  small tolerance; a substantive reversal is an error, not a repair;
* study groups: overlapping publications share a group; :func:`study_group_weights` caps the total weight
  per group, so a densely digitized curve or a repeated cohort cannot dominate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from kairos.calibration.schema import (
    APPROVED,
    CURVE_KINDS,
    EFFECT_TYPES,
    TARGET_KINDS,
    CurvePoint,
    EvidenceTarget,
    PopulationSpec,
    fingerprint,
)

# simulated quantities: variable -> (unit, reporting-operator support)
SUPPORTED_VARIABLES = {
    "route": "", "sex": "", "diabetes": "", "dialysis": "", "af": "", "bicuspid": "", "lipid_lowering": "", "design_class": "",
    "age_at_implant": "years", "bsa": "m2", "bmi": "kg/m2", "egfr0": "mL/min/1.73m2", "egfr": "mL/min/1.73m2",
    "reference_mean_gradient": "mmHg", "reference_eoa": "cm2", "reference_dvi": "", "hba1c": "%", "ldl": "mg/dL",
    "phosphate": "mg/dL", "all_cause_death": "", "non_svd_index_valve_replacement": "",
    "kairos_adjudicated_moderate_or_severe_svd_v2": "",
}
CAUSE_VARIABLES = {"all_cause_death": "death", "non_svd_index_valve_replacement": "replacement",
                   "kairos_adjudicated_moderate_or_severe_svd_v2": "svd"}
ORIGIN_TO_VIEW = {"implantation": "enrolled_implantation_origin", "reference_echo": "kairos_reference_eligible"}
MONOTONE_TOL = 1e-3


@dataclass
class EvidenceRegistry:
    targets: list = field(default_factory=list)      # EvidenceTarget dicts
    curves: list = field(default_factory=list)       # CurvePoint dicts
    metadata: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> EvidenceRegistry:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls([EvidenceTarget.from_dict(t).to_dict() for t in raw.get("targets", [])],
                   [CurvePoint.from_dict(c).to_dict() for c in raw.get("curves", [])], raw.get("metadata", {}))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump({"metadata": self.metadata, "targets": self.targets, "curves": self.curves},
                                             sort_keys=False, width=120, allow_unicode=True), encoding="utf-8")

    def fingerprint(self) -> str:
        return fingerprint({"targets": self.targets, "curves": self.curves})

    def merged(self, other: EvidenceRegistry) -> EvidenceRegistry:
        ids = {t["target_id"] for t in self.targets}
        clash = ids & {t["target_id"] for t in other.targets}
        if clash:
            raise ValueError(f"duplicate target ids across registries: {sorted(clash)[:5]}")
        return EvidenceRegistry(self.targets + other.targets, self.curves + other.curves, {**self.metadata, **other.metadata})


@dataclass
class EvidenceValidation:
    targets: list                                    # EvidenceTarget dicts with decided role and compatibility
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    registry_fingerprint: str = ""

    def by_role(self, role: str) -> list:
        return [t for t in self.targets if t["role"] == role]


def validate_curve(points: list[dict], kind: str) -> list[str]:
    """Problems with one curve (empty list when valid). Never repairs."""
    errs = []
    if kind not in CURVE_KINDS:
        return [f"curve kind {kind!r} is not a survival or cumulative incidence curve"]
    df = pd.DataFrame(points)
    if df.empty:
        return ["curve has no points"]
    if not df["time"].is_monotonic_increasing or df["time"].duplicated().any():
        errs.append("curve times must be strictly increasing")
    if (df["time"] < 0).any():
        errs.append("negative curve time")
    if ((df["estimate"] < 0) | (df["estimate"] > 1)).any():
        errs.append("curve estimates must be probabilities in [0, 1]")
    for col in ("lower", "upper"):
        if col in df and df[col].notna().any():
            if ((df[col].dropna() < 0) | (df[col].dropna() > 1)).any():
                errs.append(f"curve {col} limits outside [0, 1]")
    if "lower" in df and "upper" in df:
        both = df.dropna(subset=["lower", "upper"])
        if ((both["lower"] > both["estimate"] + MONOTONE_TOL) | (both["upper"] < both["estimate"] - MONOTONE_TOL)).any():
            errs.append("curve interval does not contain its estimate")
    d = np.diff(df["estimate"].to_numpy(dtype=float))
    if kind == "km_survival" and (d > MONOTONE_TOL).any():
        errs.append(f"survival curve increases by {d.max():.4f} (substantive reversal; not repaired)")
    if kind == "cumulative_incidence" and (d < -MONOTONE_TOL).any():
        errs.append(f"cumulative incidence decreases by {-d.min():.4f} (substantive reversal; not repaired)")
    return errs


def as_target(t: dict) -> EvidenceTarget:
    names = set(EvidenceTarget.__dataclass_fields__)
    return EvidenceTarget(**{k: v for k, v in t.items() if k in names})


def target_standard_error(t: dict) -> float | None:
    """Declared SE, or the binomial SE of a proportion or curve probability with a sample size."""
    se = as_target(t).standard_error()
    curve_like = t["kind"] in ("proportion", "km_survival", "cumulative_incidence")
    if se is None and t.get("sample_size") and curve_like and t.get("estimate") is not None:
        p = min(max(float(t["estimate"]), 1e-6), 1 - 1e-6)
        se = (p * (1 - p) / float(t["sample_size"])) ** 0.5
    return se


def _decide(t: dict, population: PopulationSpec, policy: dict) -> tuple[str, str, str]:
    """(role, compatibility status, reason)."""
    requested = t.get("role", "excluded")
    kind, var = t["kind"], t["variable"]
    if t["review_status"] not in APPROVED:
        return "excluded", "unreviewed", f"review status {t['review_status']!r}: only reviewed targets are used"
    if kind not in TARGET_KINDS:
        return "excluded", "unsupported_kind", f"target kind {kind!r} not supported"
    if var not in SUPPORTED_VARIABLES:
        return "excluded", "no_reporting_operator", f"no simulated reporting operator for variable {var!r}"
    unit = SUPPORTED_VARIABLES[var]
    if t.get("unit") not in (None, "", unit) and unit:
        return "excluded", "unit_mismatch", f"unit {t.get('unit')!r} differs from simulated unit {unit!r}"
    est = t.get("estimate")
    if isinstance(est, str) and est.strip().startswith("<"):
        return "excluded", "suppressed", "suppressed count: not converted to an exact target"
    if kind == "log_effect":
        if t.get("effect_type") not in EFFECT_TYPES:
            return "excluded", "unknown_effect_type", "log effect without an explicit effect type"
        if t["effect_type"] == "subdistribution_log_hr":
            return "sensitivity", "estimand_mismatch", ("subdistribution (Fine-Gray) hazard ratio is not a cause-specific "
                                                        "coefficient; separate scenario or exclude")
    if kind in CURVE_KINDS or kind == "log_effect":
        origin = (t.get("time") or {}).get("origin")
        if origin not in ORIGIN_TO_VIEW:
            return "excluded", "time_origin_mismatch", f"time origin {origin!r} has no simulated study view"
        if var not in CAUSE_VARIABLES:
            return "excluded", "endpoint_mismatch", f"outcome {var!r} is not a simulated event definition"
        if var in CAUSE_VARIABLES and population.event_definitions.get(CAUSE_VARIABLES[var]) != var:
            return "sensitivity", "endpoint_mismatch", "population declares a different event definition for this cause"
        defn = t.get("outcome_definition")
        if defn not in (None, var) and not (t.get("population") or {}).get("reporting_operator"):
            return "sensitivity", "endpoint_mismatch", (f"source endpoint {defn!r} differs from simulated {var!r} and no "
                                                         "reporting operator is declared")
        horizon = float((t.get("time") or {}).get("horizon_years") or 0.0)
        if horizon > float(population.max_followup_years):
            return "sensitivity", "extrapolation", (f"horizon {horizon:g} y beyond simulated follow-up "
                                                    f"{population.max_followup_years:g} y: named extrapolation assumption only")
        if kind == "km_survival" and var != "all_cause_death" and (t.get("time") or {}).get("competing_events", True):
            return "sensitivity", "estimand_mismatch", "1-KM is not a cumulative incidence for a cause with competing events"
    binomial_ok = kind in ("proportion", "km_survival", "cumulative_incidence") and bool(t.get("sample_size"))
    if as_target(t).standard_error() is None and not binomial_ok:
        if policy.get("missing_target_uncertainty_policy", "sensitivity_only") == "sensitivity_only":
            return "sensitivity", "no_uncertainty", "no defensible uncertainty: sensitivity only, no invented standard error"
    if requested in ("fit", "holdout"):
        tol = (t.get("acceptance") or {}).get("tolerance")
        if tol is None and policy.get("target_tolerances_required", True):
            return "sensitivity", "no_tolerance", "fit and holdout targets need a predeclared tolerance"
        return requested, "compatible", ""
    return requested if requested in ("sensitivity", "excluded") else "excluded", "compatible", "role not fit or holdout"


def validate_evidence(registry: EvidenceRegistry, population: PopulationSpec, policy: dict | None = None) -> EvidenceValidation:
    policy = {**{"missing_target_uncertainty_policy": "sensitivity_only", "target_tolerances_required": True}, **(policy or {})}
    out, errors, warnings = [], [], []
    ids = [t["target_id"] for t in registry.targets]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        errors.append(f"duplicate target ids: {sorted(dup)}")
    curves = pd.DataFrame(registry.curves) if registry.curves else pd.DataFrame(columns=["target_id"])
    for t in registry.targets:
        t = dict(t)
        role, status, reason = _decide(t, population, policy)
        if t["kind"] in CURVE_KINDS and t.get("curve_id"):
            pts = curves[curves["target_id"] == t["curve_id"]].sort_values("time").to_dict(orient="records")
            problems = validate_curve(pts, t["kind"])
            if problems:
                errors.extend(f"{t['target_id']}: {p}" for p in problems)
                role, status, reason = "excluded", "invalid_curve", "; ".join(problems)
        t["requested_role"] = t.get("role")
        t["role"], t["compatibility"] = role, {"status": status, "reason": reason}
        out.append(t)
    fit_groups = {t["study_group_id"] for t in out if t["role"] == "fit"}
    hold_groups = {t["study_group_id"] for t in out if t["role"] == "holdout"}
    overlap = fit_groups & hold_groups
    for t in out:
        if t["role"] == "holdout" and t["study_group_id"] in overlap:
            t["role"], t["compatibility"] = "excluded", {"status": "fit_holdout_overlap",
                                                        "reason": "same study group is a fit target: not independent validation"}
            warnings.append(f"{t['target_id']}: holdout shares study group with fit targets; excluded from validation")
        linked = set(t.get("overlapping_groups") or [])
        if t["role"] == "holdout" and linked & fit_groups:
            t["role"], t["compatibility"] = "excluded", {"status": "fit_holdout_overlap",
                                                        "reason": "overlapping publication cohort is used for fitting"}
    return EvidenceValidation(out, errors, warnings, registry.fingerprint())


def study_group_weights(targets: list[dict], cap_per_group: float = 1.0) -> dict:
    """Weight per target: each study group (merged with its declared overlaps) shares at most
    ``cap_per_group`` in total, split in proportion to declared target weights."""
    parent = {}

    def find(g):
        while parent.setdefault(g, g) != g:
            g = parent[g]
        return g

    for t in targets:
        for o in t.get("overlapping_groups") or []:
            parent[find(o)] = find(t["study_group_id"])
    groups: dict = {}
    for t in targets:
        groups.setdefault(find(t["study_group_id"]), []).append(t)
    out = {}
    for members in groups.values():
        raw = np.array([float((m.get("acceptance") or {}).get("weight", 1.0)) for m in members])
        share = raw / raw.sum() * cap_per_group if raw.sum() > 0 else np.zeros(len(members))
        out.update({m["target_id"]: float(w) for m, w in zip(members, share)})
    return out


# --- candidates from the published rates table -------------------------------------------------------------
def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")[:60]


def candidates_from_published_rates(path: str | Path) -> EvidenceRegistry:
    """Proposed (unreviewed) targets from ``data/reference/published_rates.csv``, with a preliminary
    compatibility note. Nothing here is approved: review happens in the registry file."""
    df = pd.read_csv(path)
    targets = []
    for i, r in df.iterrows():
        outcome = str(r["outcome"]).lower()
        study_group = _slug(str(r["study"]).split(" (as cited")[0])
        cited = "as cited" in str(r["study"])
        note, var, kind, estimate = [], None, None, None
        if "enrolled" in outcome:
            continue
        pct = r["estimate_pct"]
        if outcome.startswith("all-cause mortality"):
            var, kind, estimate = "all_cause_death", "km_survival", (1 - pct / 100.0) if pd.notna(pct) else None
            note.append("mortality reported as cumulative percentage; stored as survival")
        elif "svd" in outcome and "freedom" not in outcome and "redo" not in outcome:
            var, kind = "kairos_adjudicated_moderate_or_severe_svd_v2", "cumulative_incidence"
            estimate = pct / 100.0 if pd.notna(pct) else None
            note.append(f"source definition {r['definition']!r} differs from the KAIROS adjudicated endpoint (confirmation, "
                        "reference study): needs a reporting operator before use")
        elif "redo" in outcome or "re-intervention" in outcome or "reoperation" in outcome:
            var, kind = "non_svd_index_valve_replacement", "cumulative_incidence"
            estimate = (1 - pct / 100.0) if ("freedom" in outcome and pd.notna(pct)) else (pct / 100.0 if pd.notna(pct) else None)
            note.append("reintervention of any cause or SVD-related differs from non-SVD replacement; 'freedom from' is 1-KM, "
                        "not a cumulative incidence")
        else:
            note.append("outcome has no simulated reporting operator in generator 2.1")
        if cited:
            note.append("secondary citation: verify against the primary publication before review")
        targets.append(EvidenceTarget(
            target_id=f"pub:{study_group}:{_slug(r['arm'])}:{_slug(outcome)}:{r['time_years']}:{i}",
            kind=kind or "proportion", variable=var or _slug(outcome), statistic=str(r["definition"]),
            estimate=estimate, uncertainty=({"type": "ci95", "lower": r["ci_low"] / 100.0, "upper": r["ci_high"] / 100.0}
                                            if pd.notna(r["ci_low"]) and pd.notna(r["ci_high"]) else {"type": "none"}),
            sample_size=None, population={"study": r["study"], "arm": r["arm"], "device": r["device"]},
            time={"origin": "implantation", "horizon_years": None if pd.isna(r["time_years"]) else float(r["time_years"]),
                  "unit": "years", "censoring": "not reported", "competing_events": True},
            outcome_definition=str(r["definition"]), study_group_id=study_group,
            source={"citation": r["source"], "location": r["source_location"], "extraction": "published_rates.csv",
                    "extraction_confidence": r["confidence"], "retrieved_at": r["retrieved_at"]},
            review_status="proposed", role="excluded",
            compatibility={"status": "unassessed", "reason": "; ".join(note)},
            acceptance={"tolerance": None, "tolerance_type": "absolute", "weight": 1.0, "mandatory": False}).to_dict())
    return EvidenceRegistry(targets, [], {"source": str(path), "status": "proposed candidates; none reviewed",
                                          "note": "approve by editing review_status, role, uncertainty and acceptance.tolerance"})
