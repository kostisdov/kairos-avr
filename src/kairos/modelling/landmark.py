"""Landmark dataset builder with the leakage rules of the specification.

One row per patient per prediction time (every echo from the reference study onward), each
carrying only information dated on or before that time:

* time zero is the first adequate echo 30 to 180 days after implantation;
* patients who met the endpoint by the landmark leave the risk set (no row on or after
  the endpoint date);
* the echo that establishes the endpoint is never a predictor of that endpoint: landmarks
  stop strictly before the endpoint date, so that echo can never enter a feature;
* stale echo values (older than 18 months) are flagged, never carried forward silently;
* dp-ucMGP is carried forward for at most twelve months, then marked stale and not used;
* the SVD label follows a named label policy over the adjudication record (endpoint version 2);
  under ``primary`` a patient whose first unresolved candidate precedes adjudication is censored
  there and contributes no landmark on or after it;
* missing medication records mean unknown anticoagulant coverage (features missing), not
  verified absence of anticoagulation;
* synthetic truth columns are refused, and ``FORBIDDEN_PREDICTORS`` lists every column that may
  never become a model feature.

The same feature function serves the training builder and the live predictor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd

from kairos.adjudication.framework import EchoPoint, select_reference, stage_point
from kairos.grades import regurg_ordinal
from kairos.modelling.vitamin_k import dp_ucmgp_features
from kairos.simulation.truth import TRUTH_COLUMNS, truth_columns_in

DAYS = 365.25
CAUSE_CODES = {"censored": 0, "svd": 1, "death": 2, "replacement": 3}
LAB_ANALYTES = ("egfr", "hba1c", "ldl", "phosphate", "lpa", "ntprobnp", "hscrp", "calcium_corrected", "pth", "alp")
OAC = {"VKA", "FXa", "DTI"}
ENDPOINT_VERSION_LANDMARK = "2"
LABEL_POLICIES = ("primary", "sens_uncertain_positive", "sens_uncertain_negative", "legacy_single_study")
EXCLUSION_REASONS = ("no_events_row", "no_echo", "no_reference", "endpoint_at_or_before_reference",
                     "unresolved_candidate_at_or_before_reference", "death_or_replacement_at_or_before_reference",
                     "followup_end_at_or_before_reference", "no_eligible_landmark")
OUTCOME_COLUMNS = frozenset({"time", "event", "time_thromb", "event_thromb"})
AUDIT_COLUMNS = frozenset({"patient_id", "landmark_date", "bootstrap_cluster_id", "row_weight_patient_balanced",
                           "ac_records_available"})
EVENT_TABLE_COLUMNS = frozenset({
    "implant_date", "reference_date", "svd_candidate_date", "svd_confirmation_date", "svd_adjudicated_date",
    "svd_status", "svd_mechanism", "svd_confidence", "svd_adjudicated_phenotype", "svd_uncertain_dates",
    "svd_thrombosis_attributed_dates", "svd_first_unresolved_date", "svd_first_positive_date",
    "adjudication_policy_version", "death_date", "replacement_date", "end_followup_date", "end_reason",
    "first_thrombosis_date"})
# never a predictor: identifiers, outcomes, endpoint and confirmation information, synthetic truth
FORBIDDEN_PREDICTORS = TRUTH_COLUMNS | OUTCOME_COLUMNS | AUDIT_COLUMNS | EVENT_TABLE_COLUMNS


def _years(a: date, b: date) -> float:
    return (a - b).days / DAYS


def mismatch_grade(ieoa: float | None) -> str | None:
    if ieoa is None or not np.isfinite(ieoa):
        return None
    if ieoa > 0.85:
        return "none"
    if ieoa > 0.65:
        return "moderate"
    return "severe"


@dataclass
class ExposureRecord:
    class_: str
    indication: str
    start: date
    stop: date | None
    post_suspicion: bool


EXPOSURE_FEATURES = ("ac_class_current", "ac_indication", "ac_current_status", "ac_cum_vka_years", "ac_cum_oac_years",
                     "ac_days_since_change", "ac_post_suspicion", "on_antiplatelet")


def exposure_features(episodes: Iterable[ExposureRecord], landmark: date, records_available: bool = True) -> dict:
    """Antithrombotic features at ``landmark``. ``records_available=False`` means medication
    coverage is unknown: every feature is missing rather than "never anticoagulated"."""
    if not records_available:
        return {**{k: np.nan for k in EXPOSURE_FEATURES}, "ac_records_available": False}
    eps = [e for e in episodes if e.start <= landmark]
    current = [e for e in eps if (e.stop is None or e.stop > landmark)]
    cur_oac = [e for e in current if e.class_ in OAC]
    cur_ap = [e for e in current if e.class_ in ("SAPT", "DAPT")]
    cum_vka = 0.0
    cum_oac = 0.0
    changes = []
    ever_oac = False
    for e in eps:
        stop = min(e.stop, landmark) if e.stop else landmark
        years = max(0.0, _years(stop, e.start))
        if e.class_ == "VKA":
            cum_vka += years
        if e.class_ in OAC:
            cum_oac += years
            ever_oac = True
        changes.append(e.start)
        if e.stop and e.stop <= landmark:
            changes.append(e.stop)
    if cur_oac:
        status = "current"
    elif ever_oac:
        status = "past"
    else:
        status = "never"
    days_since_change = min((_years(landmark, c) * DAYS for c in changes), default=np.nan)
    return {"ac_class_current": cur_oac[0].class_ if cur_oac else ("DAPT" if any(e.class_ == "DAPT" for e in cur_ap) else ("SAPT" if cur_ap else "none")),
            "ac_indication": cur_oac[0].indication if cur_oac else "none",
            "ac_current_status": status, "ac_cum_vka_years": round(cum_vka, 3), "ac_cum_oac_years": round(cum_oac, 3),
            "ac_days_since_change": days_since_change,
            "ac_post_suspicion": bool(any(e.post_suspicion for e in eps if e.class_ in OAC)),
            "on_antiplatelet": bool(cur_ap), "ac_records_available": True}


def lab_features(labs: Iterable[tuple[date, str, float]], landmark: date) -> dict:
    latest: dict = {}
    egfr_series = []
    for d, analyte, value in labs:
        if d > landmark:
            continue
        if analyte not in latest or d >= latest[analyte][0]:
            latest[analyte] = (d, value)
        if analyte == "egfr":
            egfr_series.append((d, value))
    out = {a: (latest[a][1] if a in latest else np.nan) for a in LAB_ANALYTES}
    slope = np.nan
    if len(egfr_series) >= 2:
        egfr_series.sort()
        t = np.array([_years(d, egfr_series[0][0]) for d, _ in egfr_series])
        v = np.array([x for _, x in egfr_series])
        if t[-1] - t[0] >= 0.5:
            slope = float(np.polyfit(t, v, 1)[0])
    out["egfr_slope"] = slope
    return out


def echo_features(points: list[EchoPoint], ref: EchoPoint, landmark: date, stale_months: int = 18) -> dict:
    usable = [p for p in points if p.date <= landmark]
    cur = usable[-1]
    prev = usable[-2] if len(usable) >= 2 else None
    ar_cur = cur.ar_ordinal if cur.ar_ordinal is not None else np.nan
    ar_ref = ref.ar_ordinal if ref.ar_ordinal is not None else np.nan
    feats = {"ref_gradient": ref.mean_gradient, "ref_eoa": ref.eoa, "ref_dvi": ref.dvi, "ref_ar": ar_ref,
             "ref_lvef": ref.lvef, "ref_svi": ref.svi,
             "current_gradient": cur.mean_gradient,
             "delta_gradient": (cur.mean_gradient - ref.mean_gradient) if (cur.mean_gradient is not None and ref.mean_gradient is not None) else np.nan,
             "last_change_gradient": (cur.mean_gradient - prev.mean_gradient) if (prev and cur.mean_gradient is not None and prev.mean_gradient is not None) else np.nan,
             "last_change_interval_years": _years(cur.date, prev.date) if prev else np.nan,
             "current_eoa": cur.eoa, "delta_eoa": (cur.eoa - ref.eoa) if (cur.eoa is not None and ref.eoa is not None) else np.nan,
             "current_dvi": cur.dvi, "delta_dvi": (cur.dvi - ref.dvi) if (cur.dvi is not None and ref.dvi is not None) else np.nan,
             "current_ar": ar_cur, "delta_ar": (ar_cur - ar_ref) if np.isfinite(ar_cur) and np.isfinite(ar_ref) else np.nan,
             "current_lvef": cur.lvef, "current_svi": cur.svi,
             "time_since_last_echo_years": _years(landmark, cur.date), "n_echoes": len(usable),
             "stale_echo": bool(_years(landmark, cur.date) * 12 > stale_months),
             "current_stage": stage_point(cur, ref).stage if cur.date > ref.date else "0"}
    grads = [(p.date, p.mean_gradient) for p in usable if p.mean_gradient is not None]
    if len(grads) >= 3:
        t = np.array([_years(d, grads[0][0]) for d, _ in grads])
        v = np.array([g for _, g in grads])
        feats["gradient_slope"] = float(np.polyfit(t, v, 1)[0]) if t[-1] > t[0] else np.nan
    else:
        feats["gradient_slope"] = np.nan
    for k, v in list(feats.items()):
        if v is None:
            feats[k] = np.nan
    return feats


def static_features(static: dict, ref: EchoPoint) -> dict:
    bsa = static.get("bsa")
    ieoa = (ref.eoa / bsa) if (ref.eoa is not None and bsa) else np.nan
    keys = ("age_at_implant", "sex", "bsa", "bmi", "route", "design_class", "canonical_model", "generation",
            "size_mm", "implant_year", "diabetes", "egfr0", "dialysis", "af", "bicuspid", "tissue_treatment",
            "diabetes_duration", "lipid_lowering")
    out = {k: static.get(k, np.nan) for k in keys}
    out["ieoa"] = ieoa
    out["mismatch_grade"] = mismatch_grade(ieoa if np.isfinite(ieoa) else None)
    out["dialysis_current"] = static.get("dialysis", np.nan)
    return out


def features_at_landmark(static: dict, points: list[EchoPoint], ref: EchoPoint, landmark: date,
                         labs: Iterable[tuple[date, str, float]] = (), exposures: Iterable[ExposureRecord] = (),
                         implant_date: date | None = None, stale_months: int = 18,
                         dp_ucmgp_carry_months: float = 12.0, exposure_records_available: bool | None = None) -> dict:
    """Every feature the modules can draw on, computed from information dated <= landmark.
    ``exposure_records_available`` defaults to whether any exposure record was supplied."""
    labs = list(labs)
    exposures = list(exposures)
    feats = static_features(static, ref)
    feats.update(echo_features(points, ref, landmark, stale_months))
    feats.update(lab_features(labs, landmark))
    feats.update(dp_ucmgp_features(labs, landmark, dp_ucmgp_carry_months))
    avail = bool(exposures) if exposure_records_available is None else bool(exposure_records_available)
    feats.update(exposure_features(exposures, landmark, avail))
    feats["valve_age_years"] = _years(landmark, implant_date) if implant_date else np.nan
    feats["t_lm"] = _years(landmark, ref.date)
    interval = 1.0
    feats["overdue_flag"] = bool(feats["time_since_last_echo_years"] > interval + 0.25)
    return feats


def _outcome(landmark: date, svd_date, death_date, repl_date, end_date, horizon_years: float,
             thromb_date=None, thromb_end_date=None) -> dict:
    """Cause and time after ``landmark``. ``end_date`` is the end of observation for the primary
    outcome (end of follow-up, or an earlier label-policy censoring date); events after it are
    not observed."""
    candidates = []
    for d, code in ((svd_date, CAUSE_CODES["svd"]), (death_date, CAUSE_CODES["death"]),
                    (repl_date, CAUSE_CODES["replacement"])):
        if d is not None and landmark < d <= end_date:
            candidates.append((d, code))
    if candidates:
        d, code = min(candidates, key=lambda x: (x[0], x[1]))
        t = _years(d, landmark)
    else:
        d, code, t = end_date, 0, _years(end_date, landmark)
    if t > horizon_years:
        t, code = horizon_years, 0
    t = max(t, 1e-3)
    out = {"time": t, "event": code}
    # secondary outcome: first thrombosis episode after the landmark (thrombosis-related dysfunction)
    thromb_end = thromb_end_date or end_date
    if thromb_date is not None and landmark < thromb_date <= thromb_end:   # an episode after follow-up ends is not observed
        tt = _years(thromb_date, landmark)
        out["time_thromb"], out["event_thromb"] = (min(tt, horizon_years), 1 if tt <= horizon_years else 0)
    else:
        te = _years(thromb_end, landmark)
        out["time_thromb"], out["event_thromb"] = (min(max(te, 1e-3), horizon_years), 0)
    return out


def _opt_date(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def resolve_svd_label(e: pd.Series, label_policy: str = "primary") -> tuple[date | None, date | None]:
    """(SVD event date, censoring date) for one events row under a label policy.

    * ``primary``: the adjudicated (confirmed) date; a patient whose first unresolved candidate
      (transient, unconfirmed or another mechanism) precedes it is censored at that candidate
      (default decision 1 of the detailed design).
    * ``sens_uncertain_positive``: the first unresolved candidate counts as the SVD event.
    * ``sens_uncertain_negative``: unresolved candidates are ignored.
    * ``legacy_single_study``: endpoint version 1, the first single study at stage 2/3.
    """
    if label_policy not in LABEL_POLICIES:
        raise ValueError(f"unknown label policy {label_policy!r}; known: {LABEL_POLICIES}")
    if "svd_adjudicated_date" not in e.index:
        raise KeyError("events table has no svd_adjudicated_date: regenerate the cohort with endpoint version 2")
    adj = _opt_date(e.get("svd_adjudicated_date"))
    if label_policy == "legacy_single_study":
        if "svd_first_positive_date" not in e.index:
            raise KeyError("events table has no svd_first_positive_date for the legacy label policy")
        return _opt_date(e.get("svd_first_positive_date")), None
    if label_policy == "sens_uncertain_negative":
        return adj, None
    unresolved = _opt_date(e.get("svd_first_unresolved_date"))
    if unresolved is not None and (adj is None or unresolved < adj):
        return (None, unresolved) if label_policy == "primary" else (unresolved, None)
    return adj, None


@dataclass
class LandmarkBuild:
    rows: pd.DataFrame
    exclusions: dict = field(default_factory=dict)
    label_policy: str = "primary"
    endpoint_version: str = ENDPOINT_VERSION_LANDMARK


def _num(v):
    """Tabular value to float or None (NaN and None are both missing)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def build_landmark(patients: pd.DataFrame, echoes: pd.DataFrame, labs: pd.DataFrame,
                   exposures: pd.DataFrame, events: pd.DataFrame, horizon_years: float = 5.0,
                   stale_months: int = 18, min_days_between: int = 1,
                   time_zero_window=(30, 180), dp_ucmgp_carry_months: float = 12.0,
                   label_policy: str = "primary", event_override: pd.DataFrame | None = None) -> LandmarkBuild:
    """Landmark rows with explicit exclusions. ``event_override`` (indexed by patient_id, column
    ``svd_event_date``) replaces the SVD label for the patients it lists (sensitivity analyses);
    eligibility is rebuilt from it, so no row is kept on or after an allocated event."""
    leaked = truth_columns_in(patients, echoes, labs, exposures, events)
    if leaked:
        raise ValueError(f"synthetic truth columns passed to the landmark builder: {sorted(leaked)}")
    exclusions = dict.fromkeys(EXCLUSION_REASONS, 0)
    rows = []
    ev = events.set_index("patient_id")
    echo_g = {k: g for k, g in echoes.groupby("patient_id")} if len(echoes) else {}
    lab_g = {k: g for k, g in labs.groupby("patient_id")} if len(labs) else {}
    exp_g = {k: g for k, g in exposures.groupby("patient_id")} if len(exposures) else {}
    for _, pt in patients.iterrows():
        pid = pt["patient_id"]
        if pid not in ev.index:
            exclusions["no_events_row"] += 1
            continue
        if pid not in echo_g:
            exclusions["no_echo"] += 1
            continue
        e = ev.loc[pid]
        implant = pt["implant_date"]
        pts = []
        for i, r in enumerate(echo_g[pid].sort_values("date").itertuples(index=False)):
            pts.append(EchoPoint(date=r.date, mean_gradient=_num(r.mean_gradient), eoa=_num(r.eoa), dvi=_num(r.dvi),
                                 ar_ordinal=regurg_ordinal(r.ar_grade), lvef=_num(r.lvef), svi=_num(r.svi), index=i))
        ref = select_reference(pts, implant, time_zero_window)
        if ref is None:
            exclusions["no_reference"] += 1
            continue
        if event_override is not None and pid in event_override.index:
            svd_date, censor_date = _opt_date(event_override.loc[pid, "svd_event_date"]), None
        else:
            svd_date, censor_date = resolve_svd_label(e, label_policy)
        death_date = _opt_date(e["death_date"])
        repl_date = _opt_date(e["replacement_date"])
        end_date = e["end_followup_date"]
        obs_end = min(end_date, censor_date) if censor_date is not None else end_date
        thromb_date = _opt_date(e.get("first_thrombosis_date")) if "first_thrombosis_date" in e.index else None
        # explicit exclusion at or before the reference study, one reason per patient
        if svd_date is not None and svd_date <= ref.date:
            exclusions["endpoint_at_or_before_reference"] += 1
            continue
        if censor_date is not None and censor_date <= ref.date:
            exclusions["unresolved_candidate_at_or_before_reference"] += 1
            continue
        if (death_date is not None and death_date <= ref.date) or (repl_date is not None and repl_date <= ref.date):
            exclusions["death_or_replacement_at_or_before_reference"] += 1
            continue
        if end_date <= ref.date:
            exclusions["followup_end_at_or_before_reference"] += 1
            continue
        static = pt.to_dict()
        lab_rows = [(r.date, r.analyte, r.value) for r in lab_g[pid].itertuples(index=False)] if pid in lab_g else []
        exp_available = pid in exp_g
        exp_rows = [ExposureRecord(r.class_ if hasattr(r, "class_") else r._1, r.indication, r.start_date,
                                   _opt_date(r.stop_date), bool(r.post_suspicion))
                    for r in exp_g[pid].rename(columns={"class": "class_"}).itertuples(index=False)] if exp_available else []
        last_lm = None
        n_rows = 0
        for p in pts:
            if p.date < ref.date:
                continue
            # leakage rules: no landmark on or after the endpoint, a censoring candidate, death,
            # replacement or the end of follow-up
            stops = [d for d in (svd_date, censor_date, death_date, repl_date, end_date) if d is not None]
            if any(p.date >= d for d in stops):
                break
            if last_lm is not None and (p.date - last_lm).days < min_days_between:
                continue
            last_lm = p.date
            feats = features_at_landmark(static, pts, ref, p.date, lab_rows, exp_rows, implant, stale_months,
                                         dp_ucmgp_carry_months, exposure_records_available=exp_available)
            feats.update(_outcome(p.date, svd_date, death_date, repl_date, obs_end, horizon_years, thromb_date, end_date))
            feats["patient_id"] = pid
            feats["landmark_date"] = p.date
            rows.append(feats)
            n_rows += 1
        if n_rows == 0:
            exclusions["no_eligible_landmark"] += 1
    df = pd.DataFrame(rows)
    if len(df):
        df["bootstrap_cluster_id"] = df["patient_id"]
        df["row_weight_patient_balanced"] = 1.0 / df.groupby("patient_id")["patient_id"].transform("size")
    return LandmarkBuild(rows=df, exclusions=exclusions, label_policy=label_policy)


def build_landmark_dataset(patients: pd.DataFrame, echoes: pd.DataFrame, labs: pd.DataFrame,
                           exposures: pd.DataFrame, events: pd.DataFrame, horizon_years: float = 5.0,
                           stale_months: int = 18, min_days_between: int = 1,
                           time_zero_window=(30, 180), dp_ucmgp_carry_months: float = 12.0,
                           label_policy: str = "primary") -> pd.DataFrame:
    """Rows only; see :func:`build_landmark` for the exclusion counts."""
    return build_landmark(patients, echoes, labs, exposures, events, horizon_years, stale_months, min_days_between,
                          time_zero_window, dp_ucmgp_carry_months, label_policy).rows
