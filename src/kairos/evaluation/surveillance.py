"""KAIROS against current practice: landmark-level comparison and surveillance-policy simulation.

Why two parts. Landmarks stop strictly before the first echo that meets the endpoint, so at every
landmark the VARC-3 threshold rule on the current echo is negative by construction: the rule is a
detector, the model a predictor, and the moment the rule fires the event has already been dated.
Comparing the two on the same landmark rows therefore cannot measure what the model adds. The
comparison is done in two ways instead:

1. **Landmark level** (the model as a risk score). Out-of-fold 12-month SVD risk from the model is
   scored against the calendar comparator (:mod:`kairos.comparators.calendar`), the valve-age-and-
   type reference model and the rule, with IPCW Brier, AUC, calibration and net benefit across the
   5 to 15 percent threshold band. Reported on all landmarks and on the protocol's primary
   population, which excludes landmarks with prevalent early haemodynamic deterioration.

2. **Surveillance policies over time** (the model as a scheduler). Each held-out patient's noise-free
   valve trajectory (stored by the generator, version 2.2) is replayed under a guideline schedule and
   under the same schedule with KAIROS allowed to bring the next echo forward when the predicted
   12-month SVD risk is at or above the threshold. Guideline echoes are never removed. At every echo
   the VARC-3 rule is applied against the patient's reference study; the first positive echo is the
   detection. Measurement noise is a deterministic function of patient and calendar day, so two
   policies imaging a patient on the same day see the same measurement. Reported: detection delay
   after the latent threshold crossing, lead time of the risk-guided policy over its base schedule
   (paired, per patient), echoes per 1,000 patient-years, and positive echoes before any crossing.
   The design follows the policy simulation of team CardioNTUA (Dyania Health Hackathon 2026).

Everything here is synthetic and illustrative. The simulation assumes full attendance and a single
positive study as detection; both are stated in the report.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from kairos.adjudication.framework import EchoPoint
from kairos.comparators.calendar import CalendarComparator
from kairos.comparators.varc3_hvd import ComparatorContext, evaluate_varc3_comparator
from kairos.evaluation.clinical_comparison import decision_net_benefit
from kairos.evaluation.ladder import patient_folds
from kairos.evaluation.metrics import KMCensoring, evaluate_predictions, ipcw_weights, outcome
from kairos.grades import regurg_ordinal
from kairos.modelling.cif import combine_cause_specific, probabilities_at
from kairos.modelling.landmark import features_at_landmark
from kairos.modelling.train import fit_step, landmark_from_cohort
from kairos.simulation.generators import Cohort, _Trajectory, generator_settings

DAYS = 365.25
NOISE_STREAM = 97          # seed component for replayed measurements; distinct from the generator's streams


# --- configuration --------------------------------------------------------------------------------
@dataclass(frozen=True)
class Policy:
    name: str
    base: str                          # "annual" (ESC/EACTS) or "acc_aha" (2020 ACC/AHA)
    risk_guided: bool = False
    threshold: float = 0.05
    brought_forward_years: float = 0.5
    min_gain_years: float = 0.1        # an echo is brought forward only if it lands this much before the next scheduled one

    @property
    def base_name(self) -> str:
        return f"{self.base}"


def policies_from_config(sc: dict) -> list[Policy]:
    out = []
    for base in sc["base_schedules"]:
        out.append(Policy(base, base))
        for thr in sc["thresholds"]:
            out.append(Policy(f"{base}+kairos@{thr:g}", base, True, float(thr), float(sc["brought_forward_months"]) / 12.0,
                              float(sc["min_gain_months"]) / 12.0))
    return out


def prevalent_early_hvd(lm: pd.DataFrame, pc: dict) -> np.ndarray:
    """Landmarks with prevalent early haemodynamic deterioration, excluded from the primary population
    (protocol: 'prevalent stage 1 or worse'). VARC-3 stage 1 is morphological and not simulated, so a
    haemodynamic proxy is used: a mean-gradient rise of at least ``gradient_rise_mmhg`` over the
    reference study, or intraprosthetic regurgitation up by at least ``ar_increase_grades``."""
    dg = lm["delta_gradient"].to_numpy(float)
    da = lm["delta_ar"].to_numpy(float)
    return (np.nan_to_num(dg, nan=-np.inf) >= float(pc["gradient_rise_mmhg"])) | \
           (np.nan_to_num(da, nan=-np.inf) >= float(pc["ar_increase_grades"]))


# --- model predictions ----------------------------------------------------------------------------
def p_svd_near(pipe, model, rows: pd.DataFrame, horizons, near: float, chunk: int = 1000) -> np.ndarray:
    """12-month SVD probability per row; NaN where the model does not support the row."""
    X = pipe.transform(rows)
    for s in model.strata:
        X[s] = rows[s].astype(str).to_numpy()
    out = np.full(len(rows), np.nan)
    ok, _ = model.row_support(X)
    idx = np.flatnonzero(ok)
    for start in range(0, len(idx), chunk):
        sel = idx[start:start + chunk]
        cifs = combine_cause_specific(model.cumulative_hazards(X.iloc[sel]), model.grid)
        out[sel] = probabilities_at(cifs, model.grid, horizons, near)["svd_near_term"]
    return out


def _step_blocks(cfg: dict, step: str) -> list[str]:
    for s in cfg["ladder"]:
        if s["name"] == step:
            return list(s["blocks"])
    raise KeyError(f"unknown ladder step {step!r}")


# --- the VARC-3 rule on one pair of studies -------------------------------------------------------
def rule_status(ref: dict, cur: dict, location_reported: bool) -> str:
    ctx = ComparatorContext(intraprosthetic_ar_confirmed=True if location_reported else None)
    return evaluate_varc3_comparator(ref, cur, ctx).status


def _study(grad, eoa, dvi, ar) -> dict:
    return {"mean_gradient_mmhg": grad, "eoa_cm2": eoa, "dvi": dvi, "intraprosthetic_ar_grade": ar}


def landmark_rule_status(lm: pd.DataFrame, echoes: pd.DataFrame) -> np.ndarray:
    """Rule status at each landmark: current echo against the reference study."""
    loc = echoes.set_index(["patient_id", "date"])["ar_location"] if "ar_location" in echoes else None
    out = []
    for r in lm.itertuples(index=False):
        reported = False
        if loc is not None:
            v = loc.get((r.patient_id, r.landmark_date))
            reported = isinstance(v, str) and v == "intraprosthetic"
        ref = _study(r.ref_gradient, r.ref_eoa, r.ref_dvi, None if pd.isna(r.ref_ar) else int(r.ref_ar))
        cur = _study(r.current_gradient, r.current_eoa, r.current_dvi, None if pd.isna(r.current_ar) else int(r.current_ar))
        out.append(rule_status(ref, cur, reported))
    return np.array(out, dtype=object)


# --- part 1: landmark-level comparison ------------------------------------------------------------
def landmark_comparison(cohort: Cohort, cfg: dict, sc: dict, n_splits: int = 5, seed: int = 20260916,
                        lm: pd.DataFrame | None = None) -> dict:
    horizons = [float(h) for h in cfg["horizons_years"]]
    near = float(cfg["near_term_horizon_months"]) / 12.0
    lm = landmark_from_cohort(cohort, cfg) if lm is None else lm
    folds = patient_folds(lm["patient_id"], n_splits, seed)
    preds = {name: np.full(len(lm), np.nan) for name in ("model", "reference", "calendar")}
    cal_cards = []
    for k in range(n_splits):
        tr, te = lm[folds != k], lm[folds == k]
        te_idx = np.flatnonzero(folds == k)
        for name, step in (("model", sc["model_step"]), ("reference", "reference")):
            pipe, model, *_ = fit_step(tr, _step_blocks(cfg, step), cfg)
            preds[name][te_idx] = p_svd_near(pipe, model, te, horizons, near)
        cal = CalendarComparator(cut_years=float(sc["calendar"]["cut_years"]), by_route=bool(sc["calendar"]["by_route"]),
                                 horizon_years=near).fit(tr)
        preds["calendar"][te_idx] = cal.predict(te)
        cal_cards.append(cal.card())
    rule = landmark_rule_status(lm, cohort.echoes)
    early = prevalent_early_hvd(lm, sc["primary_population"])
    populations = {"all_landmarks": np.ones(len(lm), bool), "primary_population": ~early}
    thresholds = [round(float(x), 4) for x in np.arange(sc.get("decision_curve_from", sc["decision_band"][0]),
                                                      sc["decision_band"][1] + 1e-9, sc["decision_band_step"])]
    metrics, curves = [], []
    for pop, mask in populations.items():
        sub = lm[mask]
        t, e = sub["time"].to_numpy(float), sub["event"].to_numpy()
        G = KMCensoring().fit(t, e)
        ok = np.all([np.isfinite(preds[n][mask]) for n in preds], axis=0)
        if not ok.any():
            raise ValueError(f"no {pop} landmark has a supported prediction from every score; "
                             "the cohort is too small for the event-support gates")
        for name in preds:
            p = preds[name][mask]
            m = evaluate_predictions(p[ok], t[ok], e[ok], "svd", near, G)
            metrics.append({"population": pop, "score": name, **{k: m[k] for k in (
                "n", "n_events", "observed", "mean_predicted", "brier", "brier_null", "ipa", "auc",
                "cal_slope_logit", "cal_intercept_offset")}})
        y, _ = outcome(t, e, "svd", near)
        w = ipcw_weights(t, e, near, G).w
        r = rule[mask]
        rule_pos = (r == "positive").astype(float)
        for thr in thresholds:
            rec = {"population": pop, "threshold": thr,
                   "assess_all": decision_net_benefit(np.ones(len(sub)), y, w, thr),
                   "assess_none": 0.0, "varc3_rule": decision_net_benefit(rule_pos, y, w, thr)}
            for name in preds:
                p = np.nan_to_num(preds[name][mask], nan=0.0)
                rec[name] = decision_net_benefit(p >= thr, y, w, thr)
                rec[f"{name}_alert_rate"] = float(np.mean(p >= thr))
            curves.append(rec)
        metrics.append({"population": pop, "score": "varc3_rule", "n": int(len(sub)),
                        "rule_positive": int((r == "positive").sum()), "rule_negative": int((r == "negative").sum()),
                        "rule_indeterminate": int((r == "indeterminate").sum())})
    return {"metrics": pd.DataFrame(metrics), "decision_curve": pd.DataFrame(curves),
            "counts": {"landmarks": int(len(lm)), "patients": int(lm["patient_id"].nunique()),
                       "primary_population_landmarks": int((~early).sum()),
                       "excluded_prevalent_early_hvd": int(early.sum())},
            "calendar_cards": cal_cards}


# --- part 2: surveillance-policy simulation ------------------------------------------------------
def base_schedule(base: str, route: str, t0: float, t_end: float) -> np.ndarray:
    """Scheduled echo times (years after implantation) after the reference study."""
    if base == "annual" or (base == "acc_aha" and route != "SAVR"):
        times = t0 + np.arange(1, 100, dtype=float)
    elif base == "acc_aha":
        times = np.array([5.0, 10.0] + [10.0 + k for k in range(1, 90)])
    else:
        raise ValueError(f"unknown base schedule {base!r}")
    return times[(times > t0 + 1e-9) & (times < t_end)]


@dataclass
class _Patient:
    pid: str
    idx: int
    route: str
    implant: date
    static: dict
    labs: list
    traj: _Trajectory
    t0: float
    t_end: float
    crossing: float | None
    episodes: list
    ref_lvef: float
    ref_svi: float


def _load_patients(cohort: Cohort, pids: set) -> dict:
    tr = cohort.truth.set_index("patient_id")
    missing = [c for c in ("traj_t0_years", "traj_t_onset_years") if c not in tr.columns]
    if missing:
        raise KeyError("cohort has no replayable trajectory: regenerate it with generator 2.2 or later")
    ref = cohort.echoes[cohort.echoes["is_reference"]].set_index("patient_id")
    labs = {k: [(r.date, r.analyte, r.value) for r in g.itertuples(index=False)] for k, g in cohort.labs.groupby("patient_id")} \
        if len(cohort.labs) else {}
    out = {}
    for pt in cohort.patients.itertuples(index=False):
        if pt.patient_id not in pids:
            continue
        t = tr.loc[pt.patient_id]
        cross = t["threshold_crossing_date"]
        cross_t = (cross - pt.implant_date).days / DAYS if isinstance(cross, date) else None
        eps = [(( date.fromisoformat(e["start"]) - pt.implant_date).days / DAYS,
                (date.fromisoformat(e["end"]) - pt.implant_date).days / DAYS) for e in json.loads(t["thrombosis_episodes"])]
        traj = _Trajectory(float(t["traj_t_onset_years"]), bool(t["traj_regurgitant"]), float(t["traj_ref_gradient"]),
                           float(t["traj_ref_eoa"]), float(t["traj_ref_dvi"]), str(t["traj_ref_ar"]),
                           float(t["traj_slope_mmhg_per_year"]), float(t["traj_eoa_fall_fraction"]))
        r = ref.loc[pt.patient_id]
        out[pt.patient_id] = _Patient(pt.patient_id, int(pt.patient_id.rsplit("-", 1)[1]), pt.route, pt.implant_date,
                                      pt._asdict(), labs.get(pt.patient_id, []), traj, float(t["traj_t0_years"]),
                                      float(t["traj_t_end_years"]), cross_t, eps, float(r["lvef"]), float(r["svi"]))
    return out


class _Measurer:
    def __init__(self, spec, seed: int, p_location: float):
        p = spec.params
        self.noise = p["measurement_noise"]
        self.thromb_rise = float(p["thrombosis"]["gradient_rise_mmhg"])
        _, reg = generator_settings(p)
        self.vent = reg["ventricle"]
        self.seed = int(seed)
        self.p_location = float(p_location)

    def __call__(self, pt: _Patient, t: float) -> tuple[EchoPoint, dict, bool]:
        day = int(round(t * DAYS))
        rng = np.random.default_rng([self.seed, NOISE_STREAM, pt.idx, day])
        z = rng.normal(size=5)
        u = rng.uniform()
        g, e, _, ar, _ = pt.traj.at(t)
        if any(a <= t <= b for a, b in pt.episodes):
            g += self.thromb_rise
            e *= 0.75
        d = pt.traj.ref_dvi * (e / pt.traj.ref_eoa)
        n, v = self.noise, self.vent
        g_obs = max(2.0, g + float(n["mean_gradient_sd_mmhg"]) * z[0])
        e_obs = max(0.4, e + float(n["eoa_sd_cm2"]) * z[1])
        d_obs = max(0.1, d + float(n["dvi_sd"]) * z[2])
        lvef = float(np.clip(pt.ref_lvef - v["lvef_decline_per_year"] * (t - pt.t0) + v["noise_sd"] * z[3], *v["lvef_obs_clip"]))
        svi = float(np.clip(pt.ref_svi - v["svi_decline_per_year"] * (t - pt.t0) + v["noise_sd"] * z[4], *v["svi_obs_clip"]))
        when = pt.implant + timedelta(days=day)
        ar_o = regurg_ordinal(ar)
        point = EchoPoint(date=when, mean_gradient=round(g_obs, 1), eoa=round(e_obs, 2), dvi=round(d_obs, 2),
                          ar_ordinal=ar_o, lvef=round(lvef, 0), svi=round(svi, 0))
        return point, _study(point.mean_gradient, point.eoa, point.dvi, ar_o), bool(u < self.p_location)


def _reference_point(pt: _Patient) -> tuple[EchoPoint, dict]:
    tr = pt.traj
    point = EchoPoint(date=pt.implant + timedelta(days=int(round(pt.t0 * DAYS))), mean_gradient=round(tr.ref_grad, 1),
                      eoa=round(tr.ref_eoa, 2), dvi=round(tr.ref_dvi, 2), ar_ordinal=regurg_ordinal(tr.ref_ar),
                      lvef=pt.ref_lvef, svi=pt.ref_svi, index=0)
    return point, _study(point.mean_gradient, point.eoa, point.dvi, point.ar_ordinal)


def simulate_policy(patients: list[_Patient], policy: Policy, measure: _Measurer, predictor=None,
                    stale_months: int = 18) -> pd.DataFrame:
    """One row per patient: echoes, detection time and alerts under ``policy``. ``predictor`` maps a
    feature DataFrame to 12-month SVD probabilities (required when the policy is risk guided)."""
    if policy.risk_guided and predictor is None:
        raise ValueError("a risk-guided policy needs a predictor")
    state = {}
    for pt in patients:
        ref_pt, ref_study = _reference_point(pt)
        state[pt.pid] = {"pt": pt, "points": [ref_pt], "ref": ref_pt, "ref_study": ref_study, "t": pt.t0,
                         "sched": base_schedule(policy.base, pt.route, pt.t0, pt.t_end), "next": None,
                         "n_echo": 0, "detect": None, "alerts": 0, "unsupported": 0, "done": False,
                         "positive_indeterminate": 0}

    def next_base(s, t):
        later = s["sched"][s["sched"] > t + 1e-9]
        return float(later[0]) if len(later) else math.inf

    def schedule(batch_ids, ts):
        if policy.risk_guided and batch_ids:
            rows = []
            for pid in batch_ids:
                s = state[pid]
                pt = s["pt"]
                f = features_at_landmark(pt.static, s["points"], s["ref"], s["points"][-1].date, pt.labs, [],
                                         pt.implant, stale_months, exposure_records_available=False)
                f["route"] = pt.route
                rows.append(f)
            probs = predictor(pd.DataFrame(rows))
        else:
            probs = np.full(len(batch_ids), np.nan)
        for pid, t, p in zip(batch_ids, ts, probs):
            s = state[pid]
            nb = next_base(s, t)
            nxt = nb
            if policy.risk_guided:
                if not np.isfinite(p):
                    s["unsupported"] += 1
                elif p >= policy.threshold:
                    s["alerts"] += 1
                    bf = t + policy.brought_forward_years
                    if bf < nb - policy.min_gain_years:
                        nxt = bf
            s["next"] = nxt

    schedule(list(state), [state[k]["t"] for k in state])
    active = [k for k, s in state.items() if not s["done"]]
    while active:
        batch_ids, ts = [], []
        for pid in active:
            s = state[pid]
            t = s["next"]
            if not np.isfinite(t) or t >= s["pt"].t_end:
                s["done"] = True
                continue
            point, study, loc = measure(s["pt"], t)
            s["n_echo"] += 1
            s["t"] = t
            point.index = len(s["points"])
            s["points"].append(point)
            status = rule_status(s["ref_study"], study, loc)
            if status == "positive":
                s["detect"] = t
                s["done"] = True
                continue
            if status == "indeterminate":
                s["positive_indeterminate"] += 1
            batch_ids.append(pid)
            ts.append(t)
        schedule(batch_ids, ts)
        active = [k for k, s in state.items() if not s["done"]]
    rows = []
    for pid, s in state.items():
        pt = s["pt"]
        stop = s["detect"] if s["detect"] is not None else pt.t_end
        rows.append({"patient_id": pid, "route": pt.route, "policy": policy.name, "base": policy.base,
                     "risk_guided": policy.risk_guided, "threshold": policy.threshold if policy.risk_guided else np.nan,
                     "t0": pt.t0, "t_end": pt.t_end, "crossing": pt.crossing, "detect": s["detect"],
                     "n_echoes": s["n_echo"], "surveillance_years": max(stop - pt.t0, 0.0), "alerts": s["alerts"],
                     "unsupported_predictions": s["unsupported"], "indeterminate_echoes": s["positive_indeterminate"]})
    return pd.DataFrame(rows)


def simulate_surveillance(cohort: Cohort, spec, cfg: dict, sc: dict, n_splits: int = 5, seed: int = 20260916,
                          lm: pd.DataFrame | None = None, policies: list[Policy] | None = None,
                          progress=None) -> pd.DataFrame:
    """Out-of-fold policy simulation: the model guiding a patient is never trained on that patient."""
    horizons = [float(h) for h in cfg["horizons_years"]]
    near = float(cfg["near_term_horizon_months"]) / 12.0
    lm = landmark_from_cohort(cohort, cfg) if lm is None else lm
    policies = policies or policies_from_config(sc)
    folds = patient_folds(lm["patient_id"], n_splits, seed)
    fold_of = dict(zip(lm["patient_id"], folds))
    patients = _load_patients(cohort, set(fold_of))
    measure = _Measurer(spec, seed, float(sc["p_ar_location_reported"]))
    stale = int(cfg["landmark"]["stale_echo_months"])
    out = []
    for k in range(n_splits):
        tr = lm[folds != k]
        pipe, model, *_ = fit_step(tr, _step_blocks(cfg, sc["model_step"]), cfg)

        def predictor(df, pipe=pipe, model=model):
            return p_svd_near(pipe, model, df, horizons, near)

        test = [patients[p] for p, f in fold_of.items() if f == k and p in patients]
        for pol in policies:
            res = simulate_policy(test, pol, measure, predictor if pol.risk_guided else None, stale)
            res["fold"] = k
            out.append(res)
            if progress:
                progress(f"fold {k} {pol.name}: {len(res)} patients")
    return pd.concat(out, ignore_index=True)


# --- summaries ------------------------------------------------------------------------------------
def _classify(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    crosses = d["crossing"].notna() & (d["crossing"] > d["t0"]) & (d["crossing"] < d["t_end"])
    d["crosser"] = crosses
    det = d["detect"].notna()
    d["true_detection"] = det & crosses & (d["detect"] >= d["crossing"])
    d["early_positive"] = det & ~d["true_detection"]          # positive echo before any latent crossing
    d["delay_months"] = np.where(d["true_detection"], (d["detect"] - d["crossing"]) * 12.0, np.nan)
    return d


def _paired_stats(gi: pd.DataFrame, bi: pd.DataFrame, idx: np.ndarray) -> tuple[float, float]:
    """(mean lead in months among crossers both policies detect, extra echoes per 1,000 patient-years)
    on the resampled patients ``idx``."""
    gg, bb = gi.iloc[idx], bi.iloc[idx]
    m = gg["crosser"].to_numpy() & gg["true_detection"].to_numpy() & bb["true_detection"].to_numpy()
    ld = (bb["detect"].to_numpy()[m] - gg["detect"].to_numpy()[m]) * 12.0
    ex = 1000.0 * (gg["n_echoes"].sum() / gg["surveillance_years"].sum() - bb["n_echoes"].sum() / bb["surveillance_years"].sum())
    return (float(ld.mean()) if len(ld) else np.nan), float(ex)


def _ci(bs: np.ndarray, col: int) -> list[float] | None:
    if not len(bs):
        return None
    return [round(float(np.nanpercentile(bs[:, col], 2.5)), 2), round(float(np.nanpercentile(bs[:, col], 97.5)), 2)]


def summarise(sim: pd.DataFrame, n_boot: int = 200, seed: int = 20260917) -> dict:
    d = _classify(sim)
    rows = []
    for (pol, route), g in [((p, "all"), g) for p, g in d.groupby("policy", sort=False)] + \
                           [((p, r), g) for (p, r), g in d.groupby(["policy", "route"], sort=False)]:
        py = g["surveillance_years"].sum()
        cr = g[g["crosser"]]
        rows.append({"policy": pol, "route": route, "patients": int(len(g)), "patient_years": round(float(py), 1),
                     "echoes": int(g["n_echoes"].sum()),
                     "echoes_per_1000py": round(1000.0 * g["n_echoes"].sum() / py, 1) if py > 0 else np.nan,
                     "crossers": int(len(cr)), "detected": int(cr["true_detection"].sum()),
                     "detected_pct": round(100.0 * cr["true_detection"].mean(), 1) if len(cr) else np.nan,
                     "delay_median_months": round(float(np.nanmedian(cr["delay_months"])), 1) if cr["true_detection"].any() else np.nan,
                     "delay_p90_months": round(float(np.nanpercentile(cr["delay_months"], 90)), 1) if cr["true_detection"].any() else np.nan,
                     "early_positives": int(g["early_positive"].sum()),
                     "early_positives_per_1000py": round(1000.0 * g["early_positive"].sum() / py, 1) if py > 0 else np.nan,
                     "patients_alerted_pct": round(100.0 * (g["alerts"] > 0).mean(), 1)})
    table = pd.DataFrame(rows)

    paired = []
    guided = d[d["risk_guided"]]
    for pol, g in guided.groupby("policy", sort=False):
        base = d[(~d["risk_guided"]) & (d["base"] == g["base"].iloc[0])].set_index("patient_id")
        gi = g.set_index("patient_id")
        common = gi.index.intersection(base.index)
        gi, bi = gi.loc[common], base.loc[common]
        cr = gi["crosser"]
        both = cr & gi["true_detection"] & bi["true_detection"]
        lead = (bi.loc[both, "detect"] - gi.loc[both, "detect"]) * 12.0
        extra_py = 1000.0 * (gi["n_echoes"].sum() / gi["surveillance_years"].sum() - bi["n_echoes"].sum() / bi["surveillance_years"].sum())

        rng = np.random.default_rng(seed)
        bs = np.array([_paired_stats(gi, bi, rng.integers(0, len(gi), len(gi))) for _ in range(n_boot)]) \
            if n_boot else np.empty((0, 2))
        paired.append({"policy": pol, "versus": g["base"].iloc[0], "threshold": float(g["threshold"].iloc[0]),
                       "crossers": int(cr.sum()), "detected_by_both": int(both.sum()),
                       "only_guided_detected": int((cr & gi["true_detection"] & ~bi["true_detection"]).sum()),
                       "only_base_detected": int((cr & ~gi["true_detection"] & bi["true_detection"]).sum()),
                       "lead_mean_months": round(float(lead.mean()), 2) if len(lead) else np.nan,
                       "lead_mean_ci95": _ci(bs, 0),
                       "lead_median_months": round(float(lead.median()), 2) if len(lead) else np.nan,
                       "earlier_pct": round(100.0 * float((lead > 0.5).mean()), 1) if len(lead) else np.nan,
                       "extra_echoes_per_1000py": round(float(extra_py), 1), "extra_echoes_ci95": _ci(bs, 1)})
    return {"by_policy": table, "paired": pd.DataFrame(paired)}


# --- writer --------------------------------------------------------------------------------------
def _md_table(df: pd.DataFrame, cols: list[str]) -> list[str]:
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |"
            for r in df[cols].itertuples(index=False)]
    return [head, sep, *body]


def write_report(out: Path, scenario: str, lmc: dict, summ: dict, sim: pd.DataFrame, manifest: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    lmc["metrics"].to_csv(out / "landmark_metrics.csv", index=False)
    lmc["decision_curve"].to_csv(out / "decision_curve.csv", index=False)
    summ["by_policy"].to_csv(out / "policies.csv", index=False)
    summ["paired"].to_csv(out / "lead_time.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({**manifest, "landmark_counts": lmc["counts"],
                                                   "calendar_cards": lmc["calendar_cards"]}, indent=2, default=str) + "\n")
    m = lmc["metrics"]
    lines = [f"# KAIROS against current practice: `{scenario}`", "",
             "Synthetic and illustrative. Out-of-fold throughout: no patient is scored or guided by a model trained on them.",
             "", "## 1. The model as a risk score (12-month SVD, landmark level)", ""]
    sc_cols = ["population", "score", "n", "n_events", "observed", "mean_predicted", "brier", "ipa", "auc", "cal_slope_logit"]
    mm = m[m["score"] != "varc3_rule"].copy()
    for c in ("observed", "mean_predicted", "brier", "ipa", "auc", "cal_slope_logit"):
        mm[c] = mm[c].astype(float).round(4)
    lines += _md_table(mm, sc_cols)
    rr = m[m["score"] == "varc3_rule"]
    lines += ["", "VARC-3 rule at the same landmarks (current echo against the reference study): " +
              "; ".join(f"{r.population}: {int(r.rule_positive)} positive, {int(r.rule_negative)} negative, "
                        f"{int(r.rule_indeterminate)} indeterminate" for r in rr.itertuples()) + ".",
              "Landmarks stop before the first echo that meets the endpoint, so the rule is negative there by "
              "construction. It is compared as a surveillance policy in section 2 instead.", "",
              "### Net benefit across the 5 to 15 percent band (primary population)", ""]
    dc = lmc["decision_curve"]
    dcp = dc[dc["population"] == "primary_population"].copy()
    for c in ("model", "calendar", "reference", "varc3_rule", "assess_all"):
        dcp[c] = (dcp[c].astype(float) * 1000).round(2)
    band = [float(x) for x in manifest["settings"]["decision_band"]]
    prim = m[(m["population"] == "primary_population") & (m["score"] == "model")]
    ceiling = float(prim["observed"].iloc[0]) * 1000 if len(prim) else float("nan")
    lines += [f"Net benefit per 1,000 landmarks (true positives net of weighted false positives). The ceiling is the "
              f"12-month SVD incidence, {ceiling:.1f} per 1,000. The protocol band is {band[0]:.0%} to {band[1]:.0%}; "
              "thresholds below it are shown for context.", ""]
    lines += _md_table(dcp, ["threshold", "model", "calendar", "reference", "varc3_rule", "assess_all"])
    lines += ["", "## 2. The model as a scheduler (surveillance-policy simulation)", "",
              "Each held-out patient's noise-free valve trajectory is replayed under a guideline schedule, and under the "
              "same schedule with KAIROS bringing the next echo forward (to 6 months) when the 12-month SVD risk is at or "
              "above the threshold. Guideline echoes are never removed. Detection is the first echo on which the VARC-3 "
              "rule is positive against the reference study.", ""]
    bp = summ["by_policy"]
    lines += _md_table(bp[bp["route"] == "all"], ["policy", "patients", "echoes_per_1000py", "crossers", "detected_pct",
                                                  "delay_median_months", "delay_p90_months", "early_positives_per_1000py",
                                                  "patients_alerted_pct"])
    lines += ["", "### Lead time of KAIROS-guided surveillance over its base schedule (paired, per patient)", ""]
    lines += _md_table(summ["paired"], ["policy", "versus", "crossers", "detected_by_both", "only_guided_detected",
                                        "only_base_detected", "lead_mean_months", "lead_mean_ci95", "lead_median_months",
                                        "earlier_pct", "extra_echoes_per_1000py", "extra_echoes_ci95"])
    lines += ["", "*Lead* is averaged over crossers both policies detect before follow-up ends; *only guided detected* "
              "counts crossers the base schedule missed altogether (death, replacement or the end of follow-up came first)."]
    lines += ["", "### By route", ""]
    lines += _md_table(bp[bp["route"] != "all"], ["policy", "route", "patients", "echoes_per_1000py", "crossers",
                                                  "detected_pct", "delay_median_months", "delay_p90_months"])
    lines += ["", "## Reading this", "",
              "- *Delay* is months from the latent (noise-free) threshold crossing to the first positive echo.",
              "- *Lead* is how many months earlier the KAIROS-guided policy detects the same patient than its base schedule.",
              "- *Early positives* are positive echoes before any latent crossing: measurement noise or thrombosis, not SVD.",
              "- Assumptions: full attendance, a single positive study counts as detection, the regurgitation location is "
              "reported on 90 percent of echoes, echoes brought forward to 6 months. All labelled assumed.",
              "- The simulated patients follow the generator's own model of deterioration, so the result shows what the "
              "model can do if that model of deterioration is right. It is not clinical evidence."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
