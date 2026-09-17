"""Synthetic cohort generator for the six fixed scenarios.

The generator simulates, per patient: static covariates, a reference echo 30 to 180 days
after implantation, latent cause-specific processes (SVD onset with a stenotic or
regurgitant/abrupt phenotype, death, non-SVD index-valve replacement), reversible
thrombosis episodes as intercurrent states, a visit process that can be informative,
measurement noise on echo values, laboratory results with per-marker availability and
dated anticoagulant exposure episodes, including treatment started after a suspicious
echo (reverse causation by construction), and dp-ucMGP for a research-assay substudy.

Random streams (generator version 2). Every patient draws from seven independent streams
keyed by (seed, stream, patient index): static covariates, latent disease and competing
events, visit timing and attendance, echo measurement noise, laboratory draws, treatment
decisions and dp-ucMGP. Noise and laboratory draws exist for every *scheduled* visit, attended
or not. Two cohorts that differ only in the surveillance parameters therefore share every
latent trajectory and every observed value at commonly attended visits.

Endpoint (endpoint version 2). The synthetic label is produced by the same adjudication core
as live eligibility (:func:`kairos.adjudication.framework.adjudicate_series`), applied to the
*observed* echoes and the observed exposure records: first qualifying detection after
confirmation by the next determinable study, or by death / replacement before a second study
was possible. Candidates, confirmation, status and unresolved (transient or unconfirmed) dates
are stored separately in ``events``; the legacy single-study date is kept as
``svd_first_positive_date`` for the before/after comparison only.

Parameter registry (generator 2.1). Distributions that version 2.0 hard-coded (body size, comorbidities,
ventricular function, laboratory processes, device size weighting, calendar) are read from
:mod:`kairos.simulation.registry`; its defaults reproduce version 2.0 exactly in ``scenario`` mode. A
compiled calibrated specification (:mod:`kairos.calibration.compiler`) overrides them and may select
``calibrated`` mode (device era support, correlated reference echo, laboratory random effects). Every
attempted patient, including those who die or are replaced before a reference study, is recorded in the
evaluation-only ``truth_enrolled`` table so implantation-origin study views can be derived.

Truth separation. Latent phenotype, SVD initiation, the noise-free threshold-crossing time,
onset intervals, latent marker levels, true thrombosis episodes and the complete visit schedule
live in ``truth`` and ``truth_visits`` (:mod:`kairos.simulation.truth`). They are saved apart
from the predictor-visible tables and the landmark builder refuses them.

All dates are synthetic. Every number that leaves this module is labelled synthetic by the
manifest that accompanies each cohort.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import namedtuple
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

from kairos.adjudication.framework import (
    DEFAULT_POLICY,
    EchoPoint,
    adjudicate_series,
    select_reference,
    stage_point,
    thrombosis_windows_from_exposures,
)
from kairos.extraction.rules import REFERENCE_DIR, device_table
from kairos.grades import regurg_ordinal
from kairos.io.config import code_revision, get_settings
from kairos.simulation.registry import (
    REGISTRY_VERSION,
    analyte_base,
    analyte_value,
    device_supported_in_year,
    generator_settings,
    provenance_table,
)
from kairos.simulation.scenarios import ScenarioSpec, parameter_table
from kairos.simulation.truth import TRUTH_TABLES, attendance_by_year
from kairos.varc3 import Echo, stage_hvd

DAYS = 365.25
GENERATOR_VERSION = "2.1"
ENDPOINT_VERSION = "2"
NAMESPACES = ("quick", "full")
STREAMS = {"static": 1, "latent": 2, "visits": 3, "measurement": 4, "labs": 5, "treatment": 6, "mgp": 7}
TIME_ZERO_WINDOW_DAYS = (30, 180)
STUDY_END = date(2026, 6, 30)   # version-2.0 default; the registry calendar is authoritative

_Episode = namedtuple("_Episode", "indication start stop")


@dataclass
class Cohort:
    patients: pd.DataFrame
    echoes: pd.DataFrame
    labs: pd.DataFrame
    exposures: pd.DataFrame
    events: pd.DataFrame
    manifest: dict = field(default_factory=dict)
    truth: pd.DataFrame = field(default_factory=pd.DataFrame)
    truth_visits: pd.DataFrame = field(default_factory=pd.DataFrame)
    truth_enrolled: pd.DataFrame = field(default_factory=pd.DataFrame)

    TABLES = ("patients", "echoes", "labs", "exposures", "events")

    def save(self, store, prefix: str) -> list[str]:
        written = []
        for t in self.TABLES:
            store.put_parquet("scenarios", f"{prefix}/{t}.parquet", getattr(self, t))
            written.append(f"{prefix}/{t}.parquet")
        for t in TRUTH_TABLES:  # evaluation-only, kept in their own folder
            store.put_parquet("scenarios", f"{prefix}/truth/{t}.parquet", getattr(self, t))
            written.append(f"{prefix}/truth/{t}.parquet")
        store.put_json("scenarios", f"{prefix}/manifest.json", self.manifest)
        written.append(f"{prefix}/manifest.json")
        return written

    @classmethod
    def load(cls, store, prefix: str) -> Cohort:
        tables = {t: store.get_parquet("scenarios", f"{prefix}/{t}.parquet") for t in cls.TABLES}
        for t in TRUTH_TABLES:
            path = f"{prefix}/truth/{t}.parquet"
            tables[t] = store.get_parquet("scenarios", path) if store.exists("scenarios", path) else pd.DataFrame()
        for df in tables.values():
            for col in df.columns:
                if col.endswith("_date") or col in ("date", "onset_interval_lo", "onset_interval_hi"):
                    df[col] = pd.to_datetime(df[col]).dt.date
                    df[col] = df[col].astype(object).where(df[col].notna(), None)
        manifest = store.get_json("scenarios", f"{prefix}/manifest.json")
        return cls(manifest=manifest, **tables)


@lru_cache(maxsize=1)
def _models_by_class() -> dict:
    dt = device_table()
    out: dict = {}
    for _, r in dt.iterrows():
        sizes = []
        for s in str(r.get("sizes_mm", "")).split(";"):
            s = s.strip()
            if s.isdigit():
                sizes.append(int(s))
        if not sizes:
            continue
        out.setdefault(r["design_class"], []).append(
            {"canonical_model": r["canonical_model"], "route": r["route"], "sizes": sizes,
             "generation": r["generation"] if isinstance(r["generation"], str) else "",
             "tissue_treatment": r["tissue_treatment"] if isinstance(r["tissue_treatment"], str) else "none"})
    return out


def _reference_values_path():
    return get_settings().reference_dir / "prosthetic_valve_reference_values.csv"


@lru_cache(maxsize=1)
def _reference_values() -> pd.DataFrame:
    df = pd.read_csv(_reference_values_path())
    df = df[df["canonical_model"].notna()].copy()
    df["size_mm"] = pd.to_numeric(df["size_mm"], errors="coerce")
    return df


def _ref_draw(rng: np.random.Generator, model: str, size: int, design_class: str, spec: ScenarioSpec,
              reg_echo: dict | None = None, correlated: bool = False):
    rv = _reference_values()
    row = rv[(rv["canonical_model"] == model) & (rv["size_mm"] == size)]
    cls = spec.get(f"reference_echo_by_class.{design_class}") or spec.get("reference_echo_by_class.stented porcine")
    if not row.empty:
        r = row.iloc[0]
        g_m, g_s = r["mean_gradient_mmHg_mean"], r["mean_gradient_sd"]
        e_m, e_s = r["eoa_cm2_mean"], r["eoa_sd"]
        d_m, d_s = r["dvi_mean"], r["dvi_sd"]
        g = (g_m, g_s) if np.isfinite(g_m) and np.isfinite(g_s) else tuple(cls["gradient"])
        e = (e_m, e_s) if np.isfinite(e_m) and np.isfinite(e_s) else tuple(cls["eoa"])
        d = (d_m, d_s) if np.isfinite(d_m) and np.isfinite(d_s) else tuple(cls["dvi"])
        src = "ASE 2024 table row"
    else:
        g, e, d = tuple(cls["gradient"]), tuple(cls["eoa"]), tuple(cls["dvi"])
        src = "class default"
    reg_echo = reg_echo or {"clip_gradient": [4, 30], "clip_eoa": [0.8, 3.0], "clip_dvi": [0.25, 0.8], "correlation": None}
    (glo, ghi), (elo, ehi), (dlo, dhi) = reg_echo["clip_gradient"], reg_echo["clip_eoa"], reg_echo["clip_dvi"]
    if correlated and reg_echo.get("correlation") is not None:
        z = np.linalg.cholesky(np.asarray(reg_echo["correlation"], dtype=float)) @ rng.normal(size=3)
        return (float(np.clip(g[0] + g[1] * z[0], glo, ghi)), float(np.clip(e[0] + e[1] * z[1], elo, ehi)),
                float(np.clip(d[0] + d[1] * z[2], dlo, dhi)), src + " (correlated)")
    grad = float(np.clip(rng.normal(*g), glo, ghi))
    eoa = float(np.clip(rng.normal(*e), elo, ehi))
    dvi = float(np.clip(rng.normal(*d), dlo, dhi))
    return grad, eoa, dvi, src


def _to_date(implant: date, years: float) -> date:
    return implant + timedelta(days=int(round(years * DAYS)))


def _sample_weibull(rng, shape: float, lam: float, eta: float) -> float:
    """Time with hazard h(t) = shape * lam^shape * t^(shape-1) * exp(eta)."""
    u = rng.uniform()
    return ((-math.log(u)) / math.exp(eta)) ** (1.0 / shape) / lam


def _sample_gompertz(rng, h0: float, b: float, eta: float) -> float:
    """Time with hazard h(t) = h0 * exp(eta) * exp(b t)."""
    u = rng.uniform()
    a = h0 * math.exp(eta)
    if b <= 1e-9:
        return -math.log(u) / a
    return math.log(1.0 + b * (-math.log(u)) / a) / b


@dataclass(frozen=True)
class _Trajectory:
    """Noise-free SVD trajectory of one patient (thrombosis effects excluded)."""
    t_onset: float
    regurgitant: bool
    ref_grad: float
    ref_eoa: float
    ref_dvi: float
    ref_ar: str
    slope: float
    eoa_frac: float

    def at(self, tv: float):
        """(gradient, EOA, DVI, regurgitation label, gradient rise) at ``tv`` years after implantation."""
        after = max(0.0, tv - self.t_onset)
        if self.regurgitant:
            return self.ref_grad, self.ref_eoa, self.ref_dvi, ("severe" if after > 0 else self.ref_ar), 0.0
        rise = self.slope * after
        eoa_true = max(0.5, self.ref_eoa * (1.0 - self.eoa_frac * after))
        return self.ref_grad + rise, eoa_true, self.ref_dvi * (eoa_true / self.ref_eoa), self.ref_ar, rise

    def meets_endpoint(self, tv: float, ref_echo: Echo) -> bool:
        g, e, d, a, _ = self.at(tv)
        return stage_hvd(Echo(mean_gradient_mmHg=g, eoa_cm2=e, dvi=d, regurg_grade=regurg_ordinal(a)), ref_echo).stage in ("2", "3")


def threshold_crossing_time(traj: _Trajectory, ref_echo: Echo, t0: float, t_end: float) -> float | None:
    """First time (years after implantation, before ``t_end``) at which the noise-free trajectory
    meets VARC-3 stage 2 or 3 against the reference study: monthly scan, then bisection to about
    three days. Regurgitant (abrupt) failure crosses at onset."""
    if traj.t_onset >= t_end:
        return None
    start = max(traj.t_onset, t0)
    if traj.regurgitant:
        return start
    prev, tt = start, start
    while tt < t_end:
        if traj.meets_endpoint(tt, ref_echo):
            lo, hi = prev, tt
            for _ in range(10 if hi > lo else 0):
                mid = 0.5 * (lo + hi)
                lo, hi = (lo, mid) if traj.meets_endpoint(mid, ref_echo) else (mid, hi)
            return hi
        prev, tt = tt, tt + 1.0 / 12.0
    return t_end - 1e-6 if traj.meets_endpoint(t_end - 1e-6, ref_echo) else None


def patient_streams(seed: int, index: int) -> dict[str, np.random.Generator]:
    return {name: np.random.default_rng([int(seed), sid, int(index)]) for name, sid in STREAMS.items()}


def generate_cohort(spec: ScenarioSpec, n: int | None = None, seed: int = 20260916, namespace: str = "full") -> Cohort:
    n = int(n or spec.get("n_patients"))
    p = spec.params
    models_by_class = _models_by_class()
    noise = p["measurement_noise"]
    svd = p["svd_hazard"]
    bio_mult = float(svd["biomarker_effect_multiplier"])
    death = p["death_hazard"]
    thromb = p["thrombosis"]
    ac = p["anticoagulation"]
    surv = p["surveillance"]
    prog = p["progression"]
    yr_lo, yr_hi = p["implant_year_range"]
    lag_lo, lag_hi = p["reference_echo_lag_days"]
    shape = float(svd["weibull_shape"])
    measured_fraction = p["biomarkers"]["measured_fraction"]
    analytes = list(measured_fraction)
    followup_max = float(p["followup_years_max"])
    vk = p["vitamin_k"]
    # optional, absent from scenarios.yaml: [from, to) years after implantation with no attended visit
    forced_gap = tuple(surv["forced_missed_years"]) if surv.get("forced_missed_years") else None
    mode, reg = generator_settings(p)
    calibrated = mode == "calibrated"
    demo, comb, dev_reg, echo_reg, vent, lab_reg = (reg["demographics"], reg["comorbidity"], reg["device"],
                                                     reg["reference_echo"], reg["ventricle"], reg["labs"])
    study_end = date.fromisoformat(reg["calendar"]["study_end"])
    missing_models = [a for a in analytes if a not in lab_reg["analytes"]]
    if missing_models:
        raise KeyError(f"no synthetic value model for analytes {missing_models}")
    enforce_era = calibrated and bool(dev_reg.get("enforce_era_support"))

    patients, echoes, labs, exposures, events, truth, truth_visits = [], [], [], [], [], [], []
    enrolled = []
    era_rejections = 0
    class_redraws = 0
    exclusions = {"died_or_replaced_before_reference": 0}
    for i in range(n):
        rs = patient_streams(seed, i)
        r_static, r_latent, r_visits, r_meas = rs["static"], rs["latent"], rs["visits"], rs["measurement"]
        r_labs, r_treat, r_mgp = rs["labs"], rs["treatment"], rs["mgp"]
        pid = f"SYN-{spec.name[:12]}-{i:05d}"

        # static covariates --------------------------------------------------------------------
        route = "TAVR" if r_static.uniform() < p["p_tavr"] else "SAVR"
        mix = p["design_class_mix"][route]
        classes, weights = zip(*mix.items())
        design_class = str(r_static.choice(classes, p=np.array(weights) / sum(weights)))
        if enforce_era:   # calibrated mode: calendar first, then only devices still implantable that year
            implant = date(int(r_static.integers(yr_lo, yr_hi + 1)), 1, 1) + timedelta(days=int(r_static.integers(0, 365)))
        candidates = [m for m in models_by_class.get(design_class, []) if m["route"] == route] or \
            [m for cls in models_by_class.values() for m in cls if m["route"] == route]
        if enforce_era:
            supported = [m for m in candidates if device_supported_in_year(m["canonical_model"], implant.year)]
            era_rejections += len(candidates) - len(supported)
            if not supported:
                # the drawn class has no device still implantable that year: redraw among classes that do, and record it
                ok_classes = [c for c in classes if any(device_supported_in_year(m["canonical_model"], implant.year)
                                                        for m in models_by_class.get(c, []) if m["route"] == route)]
                if not ok_classes:
                    raise ValueError(f"no {route} device supported in {implant.year}; revise the design mix or era")
                w_ok = np.array([mix[c] for c in ok_classes], dtype=float)
                design_class = str(r_static.choice(ok_classes, p=w_ok / w_ok.sum()))
                class_redraws += 1
                supported = [m for m in models_by_class.get(design_class, []) if m["route"] == route
                             and device_supported_in_year(m["canonical_model"], implant.year)]
            candidates = supported
        dev = candidates[int(r_static.integers(len(candidates)))]
        sizes = np.array(dev["sizes"])
        w = np.exp(-((sizes - float(dev_reg["size_center_mm"])) ** 2) / float(dev_reg["size_scale"]))
        size = int(r_static.choice(sizes, p=w / w.sum()))
        if not enforce_era:
            implant = date(int(r_static.integers(yr_lo, yr_hi + 1)), 1, 1) + timedelta(days=int(r_static.integers(0, 365)))
        age_m, age_s = p["age_at_implant"][route]
        age = float(np.clip(r_static.normal(age_m, age_s), *demo["age_clip"]))
        female = r_static.uniform() < p["p_female"]
        bsa = float(np.clip(r_static.normal(demo["bsa_mean"]["F"] if female else demo["bsa_mean"]["M"], demo["bsa_sd"]),
                            *demo["bsa_clip"]))
        bmi = float(np.clip(r_static.normal(demo["bmi_mean"], demo["bmi_sd"]), *demo["bmi_clip"]))
        diabetes = r_static.uniform() < comb["p_diabetes"]
        dd = float(r_static.gamma(*comb["diabetes_duration_gamma"]))
        diabetes_duration = dd if diabetes else 0.0
        egfr0 = float(np.clip(r_static.normal(comb["egfr_intercept_at_75"] + comb["egfr_age_slope_per_year"] * (age - 75),
                                              comb["egfr_sd"]), *comb["egfr_clip"]))
        dialysis = bool(egfr0 < comb["dialysis_egfr_threshold"] or r_static.uniform() < comb["p_dialysis_other"])
        af = r_static.uniform() < min(comb["p_af_cap"], ac["p_af"] + comb["af_age_slope_per_year"] * (age - 75))
        bicuspid = r_static.uniform() < (comb["p_bicuspid_under_65"] if age < 65 else comb["p_bicuspid_65_plus"])
        lipid_lowering = r_static.uniform() < comb["p_lipid_lowering"]
        lpa_high = r_static.uniform() < p["biomarkers"]["p_lpa_high"]
        ntprobnp_high = r_static.uniform() < (comb["p_ntprobnp_high_at_75"] + comb["ntprobnp_high_age_slope_per_year"] * (age - 75))
        phosphate0 = float(np.clip(r_static.normal(comb["phosphate_mean"] + (comb["phosphate_dialysis_shift"] if dialysis else 0.0),
                                                   comb["phosphate_sd"]), *comb["phosphate_clip"]))
        regurgitant = r_static.uniform() < p["phenotype"]["p_regurgitant"][route]
        u_oac = r_static.uniform()
        oac_class = ("VKA" if u_oac < ac["p_vka_given_af"] else "FXa") if af else None
        on_vka_from_implant = oac_class == "VKA"
        anticoagulated = oac_class is not None
        ref_grad, ref_eoa, ref_dvi, ref_src = _ref_draw(r_static, dev["canonical_model"], size, design_class, spec,
                                                        echo_reg, correlated=calibrated)
        ar_probs = echo_reg["ar_probs"][route]
        ref_ar = str(r_static.choice(list(ar_probs), p=list(ar_probs.values())))
        lvef0 = float(np.clip(r_static.normal(vent["lvef_mean"], vent["lvef_sd"]), *vent["lvef_clip"]))
        svi0 = float(np.clip(r_static.normal(vent["svi_mean"], vent["svi_sd"]), *vent["svi_clip"]))
        measured = {a: r_static.uniform() < f for a, f in measured_fraction.items()}

        # dp-ucMGP latent level --------------------------------------------------------------------
        mgp_between = float(r_mgp.normal(0.0, float(vk["dp_ucmgp_log_sd_between"])))
        mgp_in_substudy = bool(r_mgp.uniform() < float(vk["dp_ucmgp_substudy_fraction"]))
        mgp_log_base = float(vk["dp_ucmgp_log_mean_pmol_l"]) + mgp_between + float(vk["dp_ucmgp_log_rise_dialysis"]) * dialysis
        mgp_latent_implant = mgp_log_base + float(vk["dp_ucmgp_log_rise_on_vka"]) * on_vka_from_implant

        # latent SVD onset, death, non-SVD replacement, progression --------------------------------
        eta = svd["log_hr_design_class"].get(design_class, 0.0)
        eta += svd["log_hr_age_per_10y"] * (age - 75) / 10.0
        eta += svd["log_hr_small_size_le21"] * (1.0 if size <= 21 else 0.0)
        eta += bio_mult * (svd["log_hr_egfr_per_minus10"] * (75 - egfr0) / 10.0
                           + svd["log_hr_dialysis"] * dialysis + svd["log_hr_diabetes"] * diabetes
                           + svd["log_hr_phosphate_per_mgdl"] * (phosphate0 - 3.5)
                           + (svd["log_hr_lpa_high"] if (lpa_high and not regurgitant) else 0.0)
                           + svd["log_hr_hscrp"] * 0.0 + svd["log_hr_ntprobnp_high"] * ntprobnp_high)
        eta += svd["log_hr_vka_from_implant"] * on_vka_from_implant
        eta += float(svd["log_hr_dp_ucmgp_per_log_unit"]) * (mgp_latent_implant - float(vk["dp_ucmgp_log_mean_pmol_l"]))
        base = float(svd["baseline_annual"][route])
        lam = (10.0 * base) ** (1.0 / shape) / 10.0
        t_onset = _sample_weibull(r_latent, shape, lam, eta)
        eta_d = (death["log_hr_per_year_of_age"] * (age - 79) + death["log_hr_dialysis"] * dialysis
                 + death["log_hr_diabetes"] * diabetes + death["log_hr_af"] * af
                 + death["log_hr_male"] * (not female) + death["log_hr_ntprobnp_high"] * ntprobnp_high
                 + math.log(float(death["multiplier"])))
        t_death = _sample_gompertz(r_latent, float(death["annual_at_age_79"]), float(death["log_hr_per_year_of_age"]), eta_d)
        t_repl = r_latent.exponential(1.0 / float(p["replacement_non_svd"]["annual_hazard"]))
        slope = float(np.exp(r_latent.normal(math.log(prog["gradient_slope_mmhg_per_year"]["median"]),
                                             prog["gradient_slope_mmhg_per_year"]["sigma"])))
        eoa_frac = float(prog["eoa_fall_fraction_per_year_at_median_slope"]) * slope / prog["gradient_slope_mmhg_per_year"]["median"]
        t_admin = min(followup_max, (study_end - implant).days / DAYS)
        t_end = min(t_death, t_repl, t_admin)
        end_reason = "death" if t_end == t_death else ("replacement" if t_end == t_repl else "administrative")

        # reference echo timing (visit stream) ------------------------------------------------------
        t0 = float(r_visits.uniform(lag_lo, lag_hi)) / DAYS
        eligible = t0 < t_end
        enrolled.append({"patient_id": pid, "attempt_index": i, "route": route, "design_class": design_class,
                         "canonical_model": dev["canonical_model"], "size_mm": size, "implant_date": implant,
                         "age_at_implant": round(age, 1), "sex": "F" if female else "M",
                         "planned_reference_date": _to_date(implant, t0), "reference_eligible": bool(eligible),
                         "exclusion_reason": "" if eligible else f"{end_reason}_before_reference",
                         "latent_death_date": _to_date(implant, t_death) if t_death < 60 else None,
                         "latent_replacement_date": _to_date(implant, t_repl) if t_repl < 60 else None,
                         "admin_end_date": _to_date(implant, t_admin), "followup_end_date": _to_date(implant, t_end),
                         "end_reason": end_reason, "initiation_date": _to_date(implant, t_onset) if t_onset < 60 else None,
                         "phenotype_latent": "regurgitant" if regurgitant else "stenotic"})
        if not eligible:  # died or replaced before a reference study: not in the KAIROS cohort
            exclusions["died_or_replaced_before_reference"] += 1
            continue

        exp_start = len(exposures)
        # thrombosis episodes (latent, reversible, intercurrent) ------------------------------------
        rate = float(thromb["annual_hazard"].get(design_class, thromb["annual_hazard"]["default"])) * float(thromb["multiplier"])
        if anticoagulated:
            rate *= math.exp(float(thromb["log_hr_anticoagulated"]))
        episodes = []
        vka_windows = [(0.0, math.inf)] if on_vka_from_implant else []  # years after implant
        t = t0
        while True:
            t += r_latent.exponential(1.0 / rate) if rate > 0 else 1e9
            if t >= t_end:
                break
            dur = float(r_latent.uniform(*thromb["duration_months"])) / 12.0
            treated = r_latent.uniform() < thromb["p_treated"]
            u_cls = r_latent.uniform()
            episodes.append({"start": t, "end": min(t + dur, t_end), "treated": treated})
            if treated:
                cls = "VKA" if u_cls < 0.5 else "FXa"
                if cls == "VKA":
                    vka_windows.append((t + 30 / DAYS, min(t + dur + 90 / DAYS, t_end)))
                exposures.append({"patient_id": pid, "class": cls, "agent": "warfarin" if cls == "VKA" else "apixaban",
                                  "indication": "suspected_valve_thrombosis", "start_date": _to_date(implant, t + 30 / DAYS),
                                  "stop_date": _to_date(implant, min(t + dur + 90 / DAYS, t_end)), "source": "prescription",
                                  "post_suspicion": True})
            t += dur

        # baseline exposures ------------------------------------------------------------------------
        if oac_class:
            exposures.append({"patient_id": pid, "class": oac_class, "agent": "warfarin" if oac_class == "VKA" else "apixaban",
                              "indication": "AF", "start_date": implant, "stop_date": None, "source": "prescription",
                              "post_suspicion": False})
        if route == "TAVR" and not oac_class:
            exposures.append({"patient_id": pid, "class": "DAPT", "agent": "aspirin+clopidogrel", "indication": "postop_prophylaxis",
                              "start_date": implant, "stop_date": _to_date(implant, 0.4), "source": "prescription", "post_suspicion": False})
            exposures.append({"patient_id": pid, "class": "SAPT", "agent": "aspirin", "indication": "other",
                              "start_date": _to_date(implant, 0.4), "stop_date": None, "source": "prescription", "post_suspicion": False})
        elif not oac_class:
            exposures.append({"patient_id": pid, "class": "SAPT", "agent": "aspirin", "indication": "other",
                              "start_date": implant, "stop_date": None, "source": "prescription", "post_suspicion": False})

        # visit schedule and per-visit draws (independent of attendance) ----------------------------
        visit_times = [t0]
        k = 1
        while True:
            tv = t0 + k * float(surv["interval_years"]) + r_visits.normal(0, float(surv["jitter_days_sd"])) / DAYS
            if tv >= t_end:
                break
            visit_times.append(max(tv, visit_times[-1] + 30 / DAYS))
            k += 1
        n_vis = len(visit_times)
        u_attend = r_visits.uniform(size=n_vis)
        z_meas = r_meas.normal(size=(n_vis, 5))
        u_lab = r_labs.uniform(size=(n_vis, len(analytes)))
        z_lab = r_labs.normal(size=(n_vis, len(analytes)))
        u_mgp = r_mgp.uniform(size=n_vis)
        z_mgp = r_mgp.normal(size=n_vis)
        lab_re = {}
        if calibrated:   # per-patient random intercepts and slopes on each analyte's transformed scale
            for analyte in analytes:
                a_spec = lab_reg["analytes"][analyte]
                lab_re[analyte] = (float(a_spec.get("patient_intercept_sd", 0.0)) * float(r_labs.normal()),
                                   float(a_spec.get("patient_slope_sd", 0.0)) * float(r_labs.normal()))
        lab_state = {"egfr0": egfr0, "phosphate0": phosphate0, "diabetes": diabetes, "lipid_lowering": lipid_lowering,
                     "lpa_high": lpa_high, "ntprobnp_high": ntprobnp_high}

        ref_echo = Echo(mean_gradient_mmHg=ref_grad, eoa_cm2=ref_eoa, dvi=ref_dvi, regurg_grade=regurg_ordinal(ref_ar))

        traj = _Trajectory(t_onset, regurgitant, ref_grad, ref_eoa, ref_dvi, ref_ar, slope, eoa_frac)

        suspicious_seen = False
        post_suspicion_started = False
        legacy_first_positive_t = None
        pat_echoes = []
        n_attended = 0
        for j, tv in enumerate(visit_times):
            active_thromb = any(ep["start"] <= tv <= ep["end"] for ep in episodes)
            g_true, eoa_true, _, ar, rise = traj.at(tv)
            if active_thromb:
                g_true += float(thromb["gradient_rise_mmhg"])
                rise += float(thromb["gradient_rise_mmhg"])
                eoa_true *= 0.75
            dvi_true = ref_dvi * (eoa_true / ref_eoa)
            if j == 0:
                attended = True
            else:
                pm = float(surv["p_missed_visit"])
                if bool(surv["informative"]):
                    pm *= float(surv["missing_multiplier_symptomatic"]) if rise >= 8 else float(surv["missing_multiplier_asymptomatic"])
                attended = not (u_attend[j] < min(0.95, pm))
                if forced_gap is not None and forced_gap[0] <= tv < forced_gap[1]:
                    attended = False  # observation-process sensitivity: visits lost in a fixed window
            visit_date = _to_date(implant, tv)
            truth_visits.append({"patient_id": pid, "visit_index": j, "scheduled_date": visit_date,
                                 "years_since_implant": round(tv, 4), "attended": bool(attended), "eligible": True,
                                 "active_thrombosis": bool(active_thromb), "true_mean_gradient": round(g_true, 2),
                                 "true_eoa": round(eoa_true, 3), "true_dvi": round(dvi_true, 3), "true_ar_grade": ar})
            if not attended:
                continue
            n_attended += 1
            zg, ze, zd, zl, zs = z_meas[j]
            grad_obs = max(2.0, g_true + float(noise["mean_gradient_sd_mmhg"]) * zg)
            eoa_obs = max(0.4, eoa_true + float(noise["eoa_sd_cm2"]) * ze)
            dvi_obs = max(0.1, dvi_true + float(noise["dvi_sd"]) * zd)
            lvef = float(np.clip(lvef0 - vent["lvef_decline_per_year"] * (tv - t0) + vent["noise_sd"] * zl, *vent["lvef_obs_clip"]))
            svi = float(np.clip(svi0 - vent["svi_decline_per_year"] * (tv - t0) + vent["noise_sd"] * zs, *vent["svi_obs_clip"]))
            if j == 0:
                grad_obs, eoa_obs, dvi_obs = ref_grad, ref_eoa, ref_dvi
            if mgp_in_substudy and u_mgp[j] < float(vk["dp_ucmgp_p_measured_per_visit"]):
                on_vka_now = any(a <= tv <= b for a, b in vka_windows)
                log_val = (mgp_log_base + float(vk["dp_ucmgp_log_rise_on_vka"]) * on_vka_now
                           + float(vk["dp_ucmgp_measurement_log_sd"]) * z_mgp[j])
                labs.append({"patient_id": pid, "date": visit_date, "analyte": "dp_ucmgp", "value": round(float(np.exp(log_val)), 0)})
            row = {"patient_id": pid, "date": visit_date, "mean_gradient": round(grad_obs, 1),
                   "eoa": round(eoa_obs, 2), "dvi": round(dvi_obs, 2), "ar_grade": ar,
                   "lvef": round(lvef, 0), "svi": round(svi, 0), "is_reference": j == 0}
            echoes.append(row)
            pat_echoes.append(row)
            if j == 0:
                for analyte in analytes:
                    if measured[analyte]:
                        a_spec = lab_reg["analytes"][analyte]
                        b0 = lab_re.get(analyte, (0.0, 0.0))[0]
                        # version 2.0 recorded the natural-scale base value at the reference study, without noise
                        base = analyte_value(a_spec, lab_state, 0.0, 0.0, b0) if b0 else analyte_base(a_spec, lab_state)
                        labs.append({"patient_id": pid, "date": visit_date, "analyte": analyte, "value": round(base, 2)})
                continue
            for a_i, analyte in enumerate(analytes):
                if measured[analyte] and u_lab[j, a_i] < lab_reg["p_measured_per_visit"]:
                    b0, b1 = lab_re.get(analyte, (0.0, 0.0))
                    labs.append({"patient_id": pid, "date": visit_date, "analyte": analyte,
                                 "value": round(analyte_value(lab_reg["analytes"][analyte], lab_state, tv - t0,
                                                              float(z_lab[j, a_i]), b0, b1), 2)})
            # suspicious echo -> possible post-suspicion anticoagulation (reverse causation) ----------
            if (grad_obs - ref_grad) >= 10 and not suspicious_seen:
                suspicious_seen = True
                if not anticoagulated and not post_suspicion_started and r_treat.uniform() < float(ac["p_post_suspicion_start"]):
                    post_suspicion_started = True
                    cls = "VKA" if r_treat.uniform() < 0.5 else "FXa"
                    start_t = tv + float(r_treat.uniform(7, 80)) / DAYS
                    if cls == "VKA":
                        vka_windows.append((start_t, math.inf))
                    exposures.append({"patient_id": pid, "class": cls, "agent": "warfarin" if cls == "VKA" else "apixaban",
                                      "indication": "suspected_valve_thrombosis",
                                      "start_date": _to_date(implant, start_t),
                                      "stop_date": None, "source": "prescription", "post_suspicion": True})
            # legacy (endpoint version 1) label: first single study at stage 2/3 outside a true episode
            if legacy_first_positive_t is None and not active_thromb:
                st = stage_hvd(Echo(mean_gradient_mmHg=grad_obs, eoa_cm2=eoa_obs, dvi=dvi_obs, regurg_grade=regurg_ordinal(ar)), ref_echo)
                if st.stage in ("2", "3"):
                    legacy_first_positive_t = tv

        # adjudication of the observed series, same core as live eligibility -------------------------
        points = [EchoPoint(date=r["date"], mean_gradient=r["mean_gradient"], eoa=r["eoa"], dvi=r["dvi"],
                            ar_ordinal=regurg_ordinal(r["ar_grade"]), lvef=r["lvef"], svi=r["svi"], index=k_)
                  for k_, r in enumerate(pat_echoes)]
        ref_pt = select_reference(points, implant, TIME_ZERO_WINDOW_DAYS)
        pat_exp = exposures[exp_start:]
        episodes_obs = [_Episode(e["indication"], e["start_date"].isoformat(), e["stop_date"].isoformat() if e["stop_date"] else None)
                        for e in pat_exp]
        windows = thrombosis_windows_from_exposures(episodes_obs, points, ref_pt)
        death_date = _to_date(implant, t_death) if end_reason == "death" else None
        repl_date = _to_date(implant, t_repl) if end_reason == "replacement" else None
        rec = adjudicate_series(points, ref_pt, windows, None, death_date, repl_date, DEFAULT_POLICY)

        # noise-free threshold crossing against the reference study ---------------------------------
        crossing_t = threshold_crossing_time(traj, ref_echo, t0, t_end)
        crossing_date = _to_date(implant, crossing_t) if crossing_t is not None else None
        adjudicated = rec.adjudicated_date
        last_negative = None
        if adjudicated is not None and ref_pt is not None:
            negs = [pt.date for pt in points if pt.date < adjudicated
                    and (pt.date == ref_pt.date or stage_point(pt, ref_pt).stage in ("0", "1"))]
            last_negative = max(negs) if negs else None
        first_unresolved = rec.first_unresolved_date

        patients.append({"patient_id": pid, "route": route, "design_class": design_class,
                         "canonical_model": dev["canonical_model"], "generation": dev["generation"],
                         "tissue_treatment": dev["tissue_treatment"], "size_mm": size,
                         "implant_date": implant, "implant_year": implant.year, "reference_date": _to_date(implant, t0),
                         "age_at_implant": round(age, 1), "sex": "F" if female else "M", "bsa": round(bsa, 2),
                         "bmi": round(bmi, 1), "diabetes": bool(diabetes), "diabetes_duration": round(diabetes_duration, 1),
                         "egfr0": round(egfr0, 1), "dialysis": bool(dialysis), "af": bool(af), "bicuspid": bool(bicuspid),
                         "lipid_lowering": bool(lipid_lowering), "dp_ucmgp_substudy": mgp_in_substudy,
                         "reference_source": ref_src})
        first_thromb = min((ep["start"] for ep in episodes), default=None)
        events.append({"patient_id": pid, "implant_date": implant, "reference_date": _to_date(implant, t0),
                       "svd_candidate_date": rec.candidate_date, "svd_confirmation_date": rec.confirmation_date,
                       "svd_adjudicated_date": adjudicated, "svd_status": rec.status, "svd_mechanism": rec.mechanism,
                       "svd_confidence": rec.confidence, "svd_adjudicated_phenotype": rec.phenotype if adjudicated else None,
                       "svd_uncertain_dates": json.dumps([d.isoformat() for d in rec.uncertain_dates]),
                       "svd_thrombosis_attributed_dates": json.dumps([d.isoformat() for d in rec.thrombosis_attributed_dates]),
                       "svd_first_unresolved_date": first_unresolved,
                       "svd_first_positive_date": _to_date(implant, legacy_first_positive_t) if legacy_first_positive_t is not None else None,
                       "adjudication_policy_version": rec.policy_version,
                       "death_date": death_date, "replacement_date": repl_date,
                       "end_followup_date": _to_date(implant, t_end), "end_reason": end_reason,
                       "first_thrombosis_date": _to_date(implant, first_thromb) if first_thromb is not None else None})
        truth.append({"patient_id": pid, "phenotype_latent": "regurgitant" if regurgitant else "stenotic",
                      "lpa_high_latent": bool(lpa_high), "dp_ucmgp_latent_log_implant": round(mgp_latent_implant, 4),
                      "initiation_date": _to_date(implant, t_onset) if t_onset < 60 else None,
                      "initiation_before_end": bool(t_onset < t_end),
                      "threshold_crossing_date": crossing_date,
                      "crossing_before_reference": bool(crossing_t is not None and t_onset <= t0),
                      "detection_delay_days": (adjudicated - crossing_date).days if (adjudicated and crossing_date) else None,
                      "missed_crossing": bool(crossing_date is not None and adjudicated is None),
                      "last_negative_date": last_negative, "onset_interval_lo": last_negative, "onset_interval_hi": adjudicated,
                      "crossing_in_interval": (bool(last_negative < crossing_date <= adjudicated)
                                               if (adjudicated and last_negative and crossing_date) else None),
                      "n_thrombosis_episodes": len(episodes),
                      "thrombosis_episodes": json.dumps([{"start": _to_date(implant, e["start"]).isoformat(),
                                                          "end": _to_date(implant, e["end"]).isoformat(),
                                                          "treated": bool(e["treated"])} for e in episodes]),
                      "n_scheduled_visits": n_vis, "n_attended_visits": n_attended})

    cohort = Cohort(patients=pd.DataFrame(patients), echoes=pd.DataFrame(echoes), labs=pd.DataFrame(labs),
                    exposures=pd.DataFrame(exposures), events=pd.DataFrame(events),
                    truth=pd.DataFrame(truth), truth_visits=pd.DataFrame(truth_visits),
                    truth_enrolled=pd.DataFrame(enrolled))
    if len(cohort.labs) == 0:
        cohort.labs = pd.DataFrame(columns=["patient_id", "date", "analyte", "value"])
    cohort.manifest = build_manifest(spec, n, seed, cohort, namespace=namespace, exclusions=exclusions)
    cohort.manifest.update({"generator_mode": mode, "registry_version": REGISTRY_VERSION,
                            "registry_provenance": provenance_table(reg, (p.get("generator") or {}).get("baseline")),
                            "calendar": reg["calendar"], "era_support": {"enforced": enforce_era,
                                                                         "devices_excluded_by_era": era_rejections,
                                                                         "design_class_redraws": class_redraws,
                                                                         "first_approval_years": "unavailable in device table"},
                            "counts_by_view": {"attempted": n, "enrolled": len(enrolled),
                                               "reference_eligible": int(len(cohort.patients))}})
    return cohort


def _file_sha256(path) -> str | None:
    try:
        return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    except OSError:
        return None


def table_hash(df: pd.DataFrame) -> str:
    """Content hash of a table (CSV text: stable across parquet writers)."""
    return hashlib.sha256(df.to_csv(index=False, lineterminator="\n").encode()).hexdigest()[:16]


def build_manifest(spec: ScenarioSpec, n: int, seed: int, cohort: Cohort, namespace: str = "full",
                   exclusions: dict | None = None) -> dict:
    ev = cohort.events
    has = len(ev) > 0
    unresolved_first = sum(1 for u, a in zip(ev["svd_first_unresolved_date"], ev["svd_adjudicated_date"])
                           if pd.notna(u) and (pd.isna(a) or u < a)) if has else 0
    counts = {"patients": int(len(cohort.patients)), "echoes": int(len(cohort.echoes)),
              "svd_adjudicated": int(ev["svd_adjudicated_date"].notna().sum()) if has else 0,
              "svd_unresolved_before_adjudication": int(unresolved_first),
              "svd_first_positive_legacy": int(ev["svd_first_positive_date"].notna().sum()) if has else 0,
              "svd_status": {k: int(v) for k, v in ev["svd_status"].value_counts().items()} if has else {},
              "deaths": int(ev["death_date"].notna().sum()) if has else 0,
              "replacements": int(ev["replacement_date"].notna().sum()) if has else 0,
              "thrombosis_episodes": int(cohort.truth["n_thrombosis_episodes"].sum()) if len(cohort.truth) else 0}
    att = attendance_by_year(cohort.truth_visits) if len(cohort.truth_visits) else pd.DataFrame()
    five = att[att["year"] == 5] if len(att) else att
    code = code_revision()
    tables = {t: table_hash(getattr(cohort, t)) for t in (*Cohort.TABLES, *TRUTH_TABLES)}
    return {"schema_version": 2, "generator_version": GENERATOR_VERSION, "endpoint_version": ENDPOINT_VERSION,
            "adjudication_policy": asdict(DEFAULT_POLICY),
            "scenario": spec.name, "variant": spec.variant, "key": spec.key, "description": spec.description,
            "evaluates": spec.evaluates, "namespace": namespace, "n_requested": n, "n_retained": int(len(cohort.patients)),
            "exclusions": exclusions or {}, "seed": seed,
            "config_hash": spec.config_hash, "effective_config_hash": spec.effective_config_hash,
            "effective_config": spec.params, "code_revision": code, "git_sha": code["value"],
            "reference_hashes": {"device_table.csv": _file_sha256(REFERENCE_DIR / "device_table.csv"),
                                 "prosthetic_valve_reference_values.csv": _file_sha256(_reference_values_path())},
            "table_hashes": tables, "truth_tables": list(TRUTH_TABLES),
            "random_streams": STREAMS, "generated_at": datetime.now(UTC).isoformat(),
            "label": "synthetic scenario: illustrative, unvalidated", "counts": counts,
            "attendance_by_year": att.round(4).to_dict(orient="records") if len(att) else [],
            "five_year_attendance": (float(five["attendance"].iloc[0]) if len(five) else None),
            "parameters": parameter_table(spec)}


def cohort_prefix(spec: ScenarioSpec, seed: int, n: int | None = None, namespace: str = "full") -> str:
    """Storage prefix: scenario, variant, namespace and a tag over the resolved parameters, seed,
    cohort size and generator/endpoint versions, so a quick run never overwrites a full cohort."""
    if namespace not in NAMESPACES:
        raise ValueError(f"namespace must be one of {NAMESPACES}")
    n = int(n or spec.get("n_patients"))
    raw = f"{spec.effective_config_hash}:{seed}:{n}:{GENERATOR_VERSION}:{ENDPOINT_VERSION}"
    tag = hashlib.sha256(raw.encode()).hexdigest()[:10]
    return f"{spec.name}/{spec.variant or 'default'}/{namespace}/{tag}"
