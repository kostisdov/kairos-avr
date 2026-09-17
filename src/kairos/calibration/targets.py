"""Reporting operators: the simulated counterpart of every target kind (design sections 5.4 and 6).

Each operator computes a statistic on the study view that matches the target's origin and selection,
never on a more convenient cohort:

* baseline proportions, means, SDs, quantiles and correlations on reference-eligible patients by default,
  or on all enrolled patients when the target's population declares ``view: enrolled_implantation_origin``;
* ``longitudinal_mean`` for laboratory analytes: person-weighted mean slope per year (``slope_per_year``)
  or person-weighted mean value in a year since implantation (``annual_mean:<k>``);
* ``km_survival`` for all-cause death: Kaplan-Meier survival at the horizon (follow-up ends at non-SVD
  replacement in the simulator, which the operator treats as censoring and records);
* ``cumulative_incidence`` for death, non-SVD replacement or adjudicated SVD: Aalen-Johansen with the other
  causes competing, from the declared origin (implantation-origin SVD counts only reference-eligible
  patients' adjudicated endpoints; pre-reference deaths and replacements compete);
* ``log_effect`` with ``effect_type: cause_specific_log_hr``: univariable cause-specific Cox coefficient
  for the declared binary contrast on the matching view.

:class:`TargetSet` expands curve targets into their points; a curve's points share the target's weight so
a densely digitized curve gains no extra influence.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from kairos.calibration.evidence import (
    CAUSE_VARIABLES,
    ORIGIN_TO_VIEW,
    study_group_weights,
    target_standard_error,
)
from kairos.calibration.populations import study_view
from kairos.calibration.schema import CURVE_KINDS

DAYS = 365.25
BASELINE_COLUMNS = {"route", "sex", "diabetes", "dialysis", "af", "bicuspid", "lipid_lowering", "design_class",
                    "age_at_implant", "bsa", "bmi", "egfr0"}
REFERENCE_ECHO = {"reference_mean_gradient": "mean_gradient", "reference_eoa": "eoa", "reference_dvi": "dvi"}
LAB_ANALYTES = {"egfr", "hba1c", "ldl", "phosphate"}


@dataclass
class TargetPoint:
    point_id: str
    target_id: str
    block: str
    kind: str
    variable: str
    statistic: str
    observed: float
    obs_se: float | None
    tolerance: float | None
    weight: float
    mandatory: bool
    horizon: float | None = None
    view: str = "kairos_reference_eligible"
    spec: dict = field(default_factory=dict)


@dataclass
class TargetSet:
    fit: list = field(default_factory=list)
    holdout: list = field(default_factory=list)
    sensitivity: list = field(default_factory=list)
    excluded: list = field(default_factory=list)      # target dicts with reasons
    fingerprint: str = ""


def _view_for(t: dict) -> str:
    pop_view = (t.get("population") or {}).get("view")
    if t["kind"] in (*CURVE_KINDS, "log_effect"):
        return ORIGIN_TO_VIEW[(t.get("time") or {}).get("origin", "reference_echo")]
    return pop_view or "kairos_reference_eligible"


def _points(t: dict, curves: pd.DataFrame, weight: float) -> list[TargetPoint]:
    view = _view_for(t)
    acc = t.get("acceptance") or {}
    if t["kind"] in CURVE_KINDS and t.get("curve_id"):
        pts = curves[curves["target_id"] == t["curve_id"]].sort_values("time")
        if pts.empty:
            return []
        w = weight / len(pts)
        out = []
        for r in pts.itertuples(index=False):
            se = (r.upper - r.lower) / (2 * 1.959964) if (pd.notna(getattr(r, "upper", np.nan)) and pd.notna(getattr(r, "lower", np.nan))) \
                else target_standard_error({**t, "estimate": r.estimate})
            out.append(TargetPoint(f"{t['target_id']}@{r.time:g}", t["target_id"], t["study_group_id"], t["kind"], t["variable"],
                                   t.get("statistic", ""), float(r.estimate), se, acc.get("tolerance"), w, bool(acc.get("mandatory", True)),
                                   float(r.time), view, {"population": t.get("population", {}), "effect_type": t.get("effect_type")}))
        return out
    horizon = (t.get("time") or {}).get("horizon_years")
    return [TargetPoint(t["target_id"], t["target_id"], t["study_group_id"], t["kind"], t["variable"], t.get("statistic", ""),
                        float(t["estimate"]), target_standard_error(t), acc.get("tolerance"), weight,
                        bool(acc.get("mandatory", True)), None if horizon is None else float(horizon), view,
                        {"population": t.get("population", {}), "effect_type": t.get("effect_type")})]


def compile_targets(validation, curves: list | None = None, cap_per_group: float = 1.0) -> TargetSet:
    from kairos.calibration.schema import fingerprint

    curves_df = pd.DataFrame(curves or [], columns=["target_id", "time", "estimate", "lower", "upper"]) if not curves \
        else pd.DataFrame(curves)
    ts = TargetSet(excluded=[t for t in validation.targets if t["role"] == "excluded"])
    for role in ("fit", "holdout", "sensitivity"):
        members = [t for t in validation.targets if t["role"] == role and t.get("estimate") is not None or
                   (t["role"] == role and t.get("curve_id"))]
        weights = study_group_weights(members, cap_per_group)
        pts = [p for t in members for p in _points(t, curves_df, weights[t["target_id"]])]
        setattr(ts, role, pts)
    ts.fingerprint = fingerprint([t for t in validation.targets if t["role"] != "excluded"])
    return ts


# --- operators ---------------------------------------------------------------------------------------------
def _baseline_values(cohort, variable: str, view: str) -> pd.Series:
    if variable in REFERENCE_ECHO:
        ref = cohort.echoes[cohort.echoes["is_reference"].astype(bool)]
        return ref.set_index("patient_id")[REFERENCE_ECHO[variable]].astype(float)
    if variable in ("egfr",):
        labs = cohort.labs[cohort.labs["analyte"] == "egfr"].sort_values("date")
        return labs.groupby("patient_id")["value"].first().astype(float)
    if view == "enrolled_implantation_origin" and variable in ("route", "design_class", "age_at_implant", "sex"):
        return cohort.truth_enrolled.set_index("patient_id")[variable]
    if variable not in BASELINE_COLUMNS:
        raise KeyError(f"no baseline operator for {variable!r}")
    return cohort.patients.set_index("patient_id")[variable]


def _km(times: np.ndarray, events: np.ndarray, horizon: float) -> tuple[float, int]:
    order = np.argsort(times)
    t, e = times[order], events[order]
    surv, n = 1.0, len(t)
    for u in np.unique(t[(e == 1) & (t <= horizon)]):
        at_risk = n - np.searchsorted(t, u, side="left")
        surv *= 1.0 - np.sum((t == u) & (e == 1)) / at_risk
    return float(surv), n


def _aalen_johansen(times: np.ndarray, causes: np.ndarray, cause: int, horizon: float) -> float:
    order = np.argsort(times)
    t, c = times[order], causes[order]
    surv, cif, n = 1.0, 0.0, len(t)
    for u in np.unique(t[(c != 0) & (t <= horizon)]):
        at_risk = n - np.searchsorted(t, u, side="left")
        cif += surv * np.sum((t == u) & (c == cause)) / at_risk
        surv *= 1.0 - np.sum((t == u) & (c != 0)) / at_risk
    return float(cif)


def _competing_frame(view_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Time and first-event cause (1 svd, 2 death, 3 replacement, 0 censored) from a study view."""
    t_end = view_df["t_end"].astype(float).to_numpy()
    cand = np.column_stack([view_df[c].astype(float).to_numpy() for c in ("t_svd", "t_death", "t_replacement")])
    cand = np.where(np.isnan(cand), np.inf, cand)
    first = cand.min(axis=1)
    cause = np.where(np.isfinite(first), cand.argmin(axis=1) + 1, 0)
    time = np.where(np.isfinite(first), first, t_end)
    return np.maximum(time, 0.0), cause


def simulate_point(cohort, point: TargetPoint, cache: dict | None = None) -> tuple[float, float, int]:
    """(simulated value, within-cohort standard error, contributing persons)."""
    cache = {} if cache is None else cache
    kind, var, stat = point.kind, point.variable, point.statistic
    if kind in ("proportion", "mean", "sd", "quantile", "correlation"):
        vals = _baseline_values(cohort, var, point.view)
        if kind == "proportion":
            level = stat.split(":", 1)[1] if ":" in stat else "True"
            ind = vals.astype(str).str.lower().eq(level.lower()).astype(float)
            p = float(ind.mean())
            return p, float(np.sqrt(max(p * (1 - p), 1e-12) / max(len(ind), 1))), int(len(ind))
        x = pd.to_numeric(vals, errors="coerce").dropna()
        if kind == "mean":
            return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))), int(len(x))
        if kind == "sd":
            return float(x.std(ddof=1)), float(x.std(ddof=1) / np.sqrt(2 * (len(x) - 1))), int(len(x))
        if kind == "quantile":
            q = float(stat.lstrip("q")) / 100.0
            return float(x.quantile(q)), float(x.std(ddof=1) * 1.2533 / np.sqrt(len(x))), int(len(x))
        other = pd.to_numeric(_baseline_values(cohort, stat.split(":", 1)[1], point.view), errors="coerce")
        pair = pd.concat([x, other], axis=1, join="inner").dropna()
        r = float(np.corrcoef(pair.iloc[:, 0], pair.iloc[:, 1])[0, 1])
        return r, float((1 - r ** 2) / np.sqrt(max(len(pair) - 3, 1))), int(len(pair))
    if kind == "longitudinal_mean":
        labs = cohort.labs[cohort.labs["analyte"] == var].copy()
        implant = cohort.patients.set_index("patient_id")["implant_date"]
        labs["years"] = [(d - implant[p]).days / DAYS for p, d in zip(labs["patient_id"], labs["date"])]
        if stat == "slope_per_year":
            slopes = [np.polyfit(g["years"], g["value"], 1)[0] for _, g in labs.groupby("patient_id")
                      if len(g) >= 2 and g["years"].max() - g["years"].min() >= 0.5]
            s = np.asarray(slopes, dtype=float)
            return float(s.mean()), float(s.std(ddof=1) / np.sqrt(len(s))), int(len(s))
        k = int(stat.split(":", 1)[1])
        per_person = labs[np.floor(labs["years"]) == k].groupby("patient_id")["value"].mean()
        return float(per_person.mean()), float(per_person.std(ddof=1) / np.sqrt(len(per_person))), int(len(per_person))
    view = cache.get(point.view)
    if view is None:
        view = cache[point.view] = study_view(cohort, point.view)
    if kind == "km_survival":
        time = np.minimum(view["t_end"].astype(float).to_numpy(), np.inf)
        event = view["t_death"].notna().to_numpy().astype(int)
        s, n = _km(np.maximum(time, 0.0), event, float(point.horizon))
        return s, float(np.sqrt(max(s * (1 - s), 1e-12) / n)), n
    if kind == "cumulative_incidence":
        time, cause = _competing_frame(view)
        code = {"svd": 1, "death": 2, "replacement": 3}[CAUSE_VARIABLES[var]]
        f = _aalen_johansen(time, cause, code, float(point.horizon))
        return f, float(np.sqrt(max(f * (1 - f), 1e-12) / len(time))), int(len(time))
    if kind == "log_effect":
        from lifelines import CoxPHFitter

        contrast = (point.spec.get("population") or {}).get("contrast", {"variable": "route", "level": "TAVR"})
        time, cause = _competing_frame(view)
        code = {"svd": 1, "death": 2, "replacement": 3}[CAUSE_VARIABLES[var]]
        df = pd.DataFrame({"t": np.maximum(time, 1e-6), "e": (cause == code).astype(int),
                           "x": view[contrast["variable"]].astype(str).eq(contrast["level"]).astype(float)})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph = CoxPHFitter().fit(df, "t", "e")
        return float(cph.params_["x"]), float(cph.standard_errors_["x"]), int(len(df))
    raise KeyError(f"no reporting operator for kind {kind!r}")
