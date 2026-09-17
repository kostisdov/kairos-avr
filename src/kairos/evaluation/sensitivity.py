"""Observation-process and onset sensitivity (CR-02, WP-B4).

Four prespecified analyses on common patients and the same patient folds:

1. ``primary``: first adjudicated detection (label policy ``primary``);
2. ``oracle``: the noise-free threshold-crossing date from the evaluation-only truth table as the
   SVD date (synthetic evaluation only); rows at or after it leave the risk set, and a patient
   with an adjudicated endpoint but no crossing has no SVD event;
3. ``interval``: the onset is allocated inside ``(last adequate negative study, adjudicated
   detection]`` at the left end (one day after the negative study), the midpoint, the right end
   (the detection date) and repeated uniform draws; eligibility is rebuilt for every allocation.
   This is an **assumption-based interval-imputation sensitivity**, not an interval-censored
   likelihood and not a formal bound. Separately, ``missing_onset`` gives patients whose trajectory
   crossed the threshold but who were never adjudicated (death, replacement or end of follow-up
   came first) an SVD event at the crossing date: an explicitly labelled assumption informed by
   synthetic truth, the one analysis allowed to place an event where no study observed it;
4. ``visit process``: paired cohorts with identical latent histories and regular versus
   informative attendance, each evaluated pooled, patient-balanced and with inverse visit-intensity
   weights fitted inside the training folds (with positivity diagnostics).

No allocated date exceeds death, replacement or the end of follow-up (asserted), except in the
labelled ``missing_onset`` analysis where the crossing precedes those dates by construction.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from kairos.adjudication.framework import EchoPoint, select_reference, stage_point
from kairos.evaluation.ladder import evaluate_ladder, patient_folds
from kairos.grades import regurg_ordinal
from kairos.modelling.train import landmark_build_from_cohort, support_mode_for
from kairos.simulation.generators import TIME_ZERO_WINDOW_DAYS, Cohort, generate_cohort
from kairos.simulation.scenarios import ScenarioSpec

ALLOCATIONS = ("left", "midpoint", "right", "uniform")
DAYS = 365.25


@dataclass
class SensitivityResult:
    results: pd.DataFrame
    diagnostics: dict = field(default_factory=dict)
    label: str = "synthetic scenario; assumption-based sensitivity, not an interval-censored likelihood"


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def onset_intervals(events: pd.DataFrame, echoes: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    """(last adequate negative study, adjudicated detection] from the observed echoes and the
    adjudication record only. ``defensible`` is false when no negative study precedes detection."""
    implant = patients.set_index("patient_id")["implant_date"]
    ev = events.dropna(subset=["svd_adjudicated_date"]).set_index("patient_id")
    rows = []
    for pid, g in echoes[echoes["patient_id"].isin(ev.index)].groupby("patient_id"):
        pts = [EchoPoint(date=r.date, mean_gradient=_num(r.mean_gradient), eoa=_num(r.eoa), dvi=_num(r.dvi),
                         ar_ordinal=regurg_ordinal(r.ar_grade), index=i)
               for i, r in enumerate(g.sort_values("date").itertuples(index=False))]
        ref = select_reference(pts, implant[pid], TIME_ZERO_WINDOW_DAYS)
        hi = ev.loc[pid, "svd_adjudicated_date"]
        if ref is None:
            continue
        negs = [p.date for p in pts if p.date < hi and p.adequate
                and (p.date == ref.date or stage_point(p, ref).stage in ("0", "1"))]
        lo = max(negs) if negs else None
        rows.append({"patient_id": pid, "interval_lo": lo, "interval_hi": hi, "defensible": lo is not None and lo < hi})
    return pd.DataFrame(rows, columns=["patient_id", "interval_lo", "interval_hi", "defensible"])


def allocate_events(events: pd.DataFrame, intervals: pd.DataFrame, how: str, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Event override (patient_id -> svd_event_date) for defensible intervals; others keep detection."""
    if how not in ALLOCATIONS:
        raise ValueError(f"allocation must be one of {ALLOCATIONS}")
    ends = events.set_index("patient_id")
    out = {}
    for r in intervals.itertuples(index=False):
        if not r.defensible:
            out[r.patient_id] = r.interval_hi
            continue
        width = (r.interval_hi - r.interval_lo).days
        offset = {"left": 1, "midpoint": max(1, width // 2), "right": width}.get(how)
        if how == "uniform":
            offset = int(rng.integers(1, width + 1))
        d = r.interval_lo + pd.Timedelta(days=int(offset)).to_pytimedelta()
        e = ends.loc[r.patient_id]
        limits = [x for x in (e["death_date"], e["replacement_date"], e["end_followup_date"]) if x is not None and pd.notna(x)]
        assert all(d <= x for x in limits), f"allocated onset after death/replacement/follow-up for {r.patient_id}"
        out[r.patient_id] = d
    return pd.DataFrame({"svd_event_date": pd.Series(out, dtype=object)}).rename_axis("patient_id")


def oracle_events(events: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    t = truth.set_index("patient_id")["threshold_crossing_date"]
    return pd.DataFrame({"svd_event_date": events["patient_id"].map(t).astype(object).where(
        events["patient_id"].map(t).notna(), None).to_numpy()}, index=pd.Index(events["patient_id"], name="patient_id"))


def missing_onset_events(events: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    tr = truth.set_index("patient_id")
    ev = events.set_index("patient_id")
    missed = tr.index[tr["missed_crossing"].astype(bool)]
    return pd.DataFrame({"svd_event_date": [tr.loc[p, "threshold_crossing_date"] for p in missed]},
                        index=pd.Index(missed, name="patient_id")).loc[lambda d: d.index.isin(ev.index)]


# --- visit process ----------------------------------------------------------------------------------
def paired_visit_cohorts(spec: ScenarioSpec, n: int, seed: int, namespace: str = "quick",
                         p_missed_informative: float = 0.28) -> tuple[Cohort, Cohort]:
    """Regular and informative attendance with identical latent histories (separate random streams)."""
    regular = copy.deepcopy(spec)
    regular.params["surveillance"]["informative"] = False
    informative = copy.deepcopy(spec)
    informative.params["surveillance"]["informative"] = True
    informative.params["surveillance"]["p_missed_visit"] = p_missed_informative
    return (generate_cohort(regular, n=n, seed=seed, namespace=namespace),
            generate_cohort(informative, n=n, seed=seed, namespace=namespace))


VISIT_COVARIATES = ("delta_gradient", "current_gradient", "valve_age_years", "t_lm", "age_at_implant")


def _gap_table(lm: pd.DataFrame) -> pd.DataFrame:
    d = lm.sort_values(["patient_id", "landmark_date"]).copy()
    nxt = d.groupby("patient_id")["landmark_date"].shift(-1)
    gap = [(b - a).days / DAYS if pd.notna(b) else np.nan for a, b in zip(d["landmark_date"], nxt)]
    d["gap"] = np.where(pd.notna(nxt), gap, d["time"])
    d["visit"] = pd.notna(nxt).astype(float)
    d["gap"] = np.maximum(d["gap"].astype(float), 1.0 / 52)
    return d


def _design(d: pd.DataFrame, med: dict) -> np.ndarray:
    cols = [pd.to_numeric(d[c], errors="coerce").fillna(med[c]).to_numpy(dtype=float) for c in VISIT_COVARIATES]
    route = (d["route"].astype(str) == "TAVR").to_numpy(dtype=float)
    return np.column_stack([np.ones(len(d)), *cols, route])


def visit_intensity_weights(lm_train: pd.DataFrame, lm_test: pd.DataFrame, truncate=(1, 99)) -> tuple[np.ndarray, dict]:
    """Stabilised inverse visit-intensity weights: exponential gap-time (Poisson) model of the time
    to the next attended study given the current landmark covariates, fitted on training rows and
    applied to test rows; truncated at the training percentiles."""
    tr = _gap_table(lm_train)
    med = {c: float(pd.to_numeric(tr[c], errors="coerce").median()) for c in VISIT_COVARIATES}
    Z = _design(tr, med)
    mu_sd = Z[:, 1:].std(axis=0)
    mu_sd[mu_sd == 0] = 1.0
    scale = np.r_[1.0, mu_sd]
    Zs = Z / scale
    y, off = tr["visit"].to_numpy(), np.log(tr["gap"].to_numpy(dtype=float))
    beta = np.zeros(Zs.shape[1])
    beta[0] = np.log(max(y.sum(), 1) / np.exp(off).sum())
    converged = False
    for _ in range(50):
        mu = np.exp(np.clip(Zs @ beta + off, -30, 30))
        step = np.linalg.solve((Zs * mu[:, None]).T @ Zs + 1e-8 * np.eye(Zs.shape[1]), Zs.T @ (y - mu))
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            converged = True
            break
    rate_tr = np.exp(np.clip(Zs @ beta, -30, 30))
    Zt = _design(lm_test, med) / scale
    rate_te = np.exp(np.clip(Zt @ beta, -30, 30))
    stab = float(np.mean(rate_tr))
    w_tr, w_te = stab / rate_tr, stab / rate_te
    lo, hi = np.percentile(w_tr, truncate)
    truncated = float(np.mean((w_te < lo) | (w_te > hi)))
    w_te = np.clip(w_te, lo, hi)
    ess = float(w_te.sum() ** 2 / np.sum(w_te ** 2)) if len(w_te) else 0.0
    return w_te, {"converged": converged, "coefficients": dict(zip(["intercept", *VISIT_COVARIATES, "route_TAVR"],
                                                                     (beta / scale).round(5).tolist())),
                  "weight_min": float(w_te.min()) if len(w_te) else None, "weight_max": float(w_te.max()) if len(w_te) else None,
                  "truncation_bounds": [float(lo), float(hi)], "share_truncated": truncated,
                  "ess_fraction": ess / len(w_te) if len(w_te) else None}


# --- orchestration --------------------------------------------------------------------------------
def _svd_rows(res, analysis: str, allocation: str | None = None, draw: int | None = None) -> pd.DataFrame:
    r = res.results
    r = r[r["state"] == "svd"].copy()
    r["analysis"], r["allocation"], r["draw"] = analysis, allocation, draw
    return r


def run_observation_sensitivity(cohort: Cohort, cfg: dict, steps: list[str], n_splits: int | None = None,
                                seed: int | None = None, uniform_draws: int | None = None, mode: str | None = None,
                                visit_process: bool = True) -> SensitivityResult:
    ev = cfg["evaluation"]
    sens = cfg.get("sensitivity", {})
    mode = mode or support_mode_for(cohort, cfg)
    seed = int(ev["seed"] if seed is None else seed)
    n_splits = int(n_splits or ev["n_splits"])
    draws = int(uniform_draws if uniform_draws is not None else sens.get("uniform_draws", {}).get(mode, 20))
    if len(cohort.truth) == 0:
        raise ValueError("the observation sensitivity needs the evaluation-only truth tables")
    fold_of = dict(zip(cohort.patients["patient_id"],
                       patient_folds(cohort.patients["patient_id"], n_splits, seed)))
    intervals = onset_intervals(cohort.events, cohort.echoes, cohort.patients)
    overrides = {"primary": None, "oracle": oracle_events(cohort.events, cohort.truth),
                 "missing_onset": missing_onset_events(cohort.events, cohort.truth)}
    rng = np.random.default_rng(seed)
    alloc = {a: allocate_events(cohort.events, intervals, a, rng) for a in ("left", "midpoint", "right")}
    uniform = [allocate_events(cohort.events, intervals, "uniform", rng) for _ in range(draws)]

    def build(override):
        return landmark_build_from_cohort(cohort, cfg, "primary", event_override=override).rows

    tables = {"primary": build(None), "oracle": build(overrides["oracle"]), "missing_onset": build(overrides["missing_onset"])}
    tables.update({f"interval_{a}": build(o) for a, o in alloc.items()})
    common = set.intersection(*(set(t["patient_id"]) for t in tables.values()))
    diag = {"n_patients_common": len(common),
            "n_patients_dropped_by_analysis": {k: int(t["patient_id"].nunique() - len(common & set(t["patient_id"])))
                                               for k, t in tables.items()},
            "intervals": {"adjudicated": int(len(intervals)), "defensible": int(intervals["defensible"].sum()) if len(intervals) else 0,
                          "median_width_days": (float((intervals.loc[intervals["defensible"], "interval_hi"]
                                                       - intervals.loc[intervals["defensible"], "interval_lo"]).map(lambda x: x.days).median())
                                                if len(intervals) and intervals["defensible"].any() else None)},
            "uniform_draws": draws, "mode": mode, "n_splits": n_splits}
    truth, events = cohort.truth, cohort.events
    adj = events.set_index("patient_id")["svd_adjudicated_date"]
    cross = truth.set_index("patient_id")["threshold_crossing_date"]
    discordant = [p for p in adj.dropna().index if pd.isna(cross.get(p)) or cross.get(p) > adj[p]]
    delay = truth["detection_delay_days"].dropna()
    diag["truth"] = {"detection_delay_days_median": float(delay.median()) if len(delay) else None,
                     "detection_delay_days_iqr": [float(delay.quantile(0.25)), float(delay.quantile(0.75))] if len(delay) else None,
                     "missed_threshold_crossings": int(truth["missed_crossing"].sum()),
                     "discordant_observed_positives": len(discordant),
                     "adjudicated": int(adj.notna().sum()), "threshold_crossings": int(cross.notna().sum())}
    kw = dict(n_splits=n_splits, n_boot=0, seed=seed, steps=steps, mode=mode, extras=False, fold_of=fold_of,
              states=("svd",))
    frames = []
    for name, lm in tables.items():
        lm = lm[lm["patient_id"].isin(common)].reset_index(drop=True)
        analysis, allocation = (("interval", name.split("_", 1)[1]) if name.startswith("interval_") else (name, None))
        frames.append(_svd_rows(evaluate_ladder(cohort, cfg, lm=lm, **kw), analysis, allocation))
    for i, o in enumerate(uniform):
        lm = build(o)
        lm = lm[lm["patient_id"].isin(common)].reset_index(drop=True)
        frames.append(_svd_rows(evaluate_ladder(cohort, cfg, lm=lm, **kw), "interval", "uniform", i))
    if visit_process:
        from kairos.simulation.scenarios import get_scenario

        spec = get_scenario(cohort.manifest["scenario"], cohort.manifest.get("variant"))
        n = int(cohort.manifest.get("n_requested") or len(cohort.patients))
        reg, inf = paired_visit_cohorts(spec, n, int(cohort.manifest.get("seed", seed)), namespace=cohort.manifest.get("namespace", "quick"))
        diag["visit_process"] = {"five_year_attendance_regular": reg.manifest["five_year_attendance"],
                                 "five_year_attendance_informative": inf.manifest["five_year_attendance"],
                                 "identical_latent_truth": bool(reg.truth[["patient_id", "threshold_crossing_date"]]
                                                                .equals(inf.truth[["patient_id", "threshold_crossing_date"]]))}
        vfold = dict(zip(reg.patients["patient_id"], patient_folds(reg.patients["patient_id"], n_splits, seed)))
        for label, c in (("visits_regular", reg), ("visits_informative", inf)):
            res = evaluate_ladder(c, cfg, **{**kw, "fold_of": vfold, "states": ("svd",)},
                                  extra_weighting=("inverse visit intensity", visit_intensity_weights))
            frames.append(_svd_rows(res, label))
            diag["visit_process"][f"{label}_weights"] = {st: {k: v.get("weighting") for k, v in f.items()}
                                                         for st, f in res.fold_fits.items()}
    out = pd.concat(frames, ignore_index=True)
    out["label"] = "synthetic scenario; sensitivity analysis"
    return SensitivityResult(results=out, diagnostics=diag)


def summarise(res: SensitivityResult) -> pd.DataFrame:
    """Brier, observed and calibration per analysis at each horizon (uniform draws: mean and range)."""
    r = res.results[res.results["weighting"].isin(["pooled landmark rows", "patient balanced", "inverse visit intensity"])]
    keys = ["analysis", "allocation", "weighting", "step", "horizon_years"]
    agg = r.groupby(keys, dropna=False).agg(brier=("brier", "mean"), brier_min=("brier", "min"), brier_max=("brier", "max"),
                                            observed=("observed", "mean"), obs_minus_pred=("obs_minus_pred", "mean"),
                                            n_event_patients=("n_event_patients", "mean"), draws=("brier", "size"))
    return agg.reset_index()
