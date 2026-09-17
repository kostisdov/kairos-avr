"""Patient inputs for the demonstration UI.

Pure functions, no Streamlit, so they are unit-tested: synthetic example patients, conversion
between the editable form and the predict-service request, staging of every echo against the
reference study, biomarker-module coverage and the "hypothetical next echo" builder.

Every example patient is synthetic. Every number the model returns for them is illustrative
and unvalidated. The form exposes only inputs the prototype model actually uses.
"""
from __future__ import annotations

import copy
import math
from collections import OrderedDict
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Any

import pandas as pd
from pydantic import ValidationError

from kairos.adjudication.framework import select_reference, stage_point, to_points
from kairos.extraction.rules import device_info, device_table
from kairos.extraction.schema import (
    EchoObservation,
    ExposureEpisode,
    ExposureTimeline,
    LabObservation,
    Passport,
    PassportSource,
    PatientStatic,
    PredictRequest,
)

DEMO_PASSPORT_ID = "demo-patient"
ROUTES = ["SAVR", "TAVR"]
JURISDICTIONS = ["ESC_EACTS", "ACC_AHA"]
AR_GRADES = ["none", "trace", "mild", "moderate", "severe"]
EXPOSURE_CLASSES = ["VKA", "FXa", "DTI", "SAPT", "DAPT"]
ORAL_ANTICOAGULANTS = {"VKA", "FXa", "DTI"}
INDICATIONS = ["AF", "VTE", "postop_prophylaxis", "suspected_valve_thrombosis", "other", "unknown"]
AC_CHANGES = ["no change", "start VKA", "start factor Xa inhibitor", "stop anticoagulation"]

# Laboratory markers the prototype model was trained on: key -> (label shown in the form, unit).
MODEL_ANALYTES: OrderedDict[str, tuple[str, str]] = OrderedDict([
    ("egfr", ("eGFR (mL/min/1.73 m²)", "mL/min/1.73m2")),
    ("hba1c", ("HbA1c (%)", "%")),
    ("ldl", ("LDL cholesterol (mg/dL)", "mg/dL")),
    ("phosphate", ("Phosphate (mg/dL)", "mg/dL")),
    ("lpa", ("Lipoprotein(a) (nmol/L)", "nmol/L")),
    ("ntprobnp", ("NT-proBNP (pg/mL)", "pg/mL")),
    ("hscrp", ("hs-CRP (mg/L)", "mg/L")),
    ("dp_ucmgp", ("dp-ucMGP (pmol/L)", "pmol/L")),
])
DP_UCMGP_CARRY_MONTHS = 12
ANALYTE_BY_LABEL = {label: key for key, (label, _unit) in MODEL_ANALYTES.items()}

# (model block name in the model card, label, laboratory markers in the form)
MODULES = [
    ("biomarker_renal_metabolic", "Renal and metabolic", ["egfr", "hba1c", "ldl"]),
    ("biomarker_mineral", "Mineral metabolism", ["phosphate"]),
    ("biomarker_lipid", "Lipid-related susceptibility", ["lpa"]),
    ("biomarker_cardiac", "Cardiac response", ["ntprobnp"]),
    ("biomarker_inflammatory", "Inflammatory and molecular", ["hscrp"]),
    ("anticoagulant", "Anticoagulant exposure", []),
    ("vitamin_k", "Vitamin K status (dp-ucMGP substudy)", ["dp_ucmgp"]),
]

ECHO_COLUMNS = ["date", "mean_gradient_mmhg", "eoa_cm2", "dvi", "ar_grade", "lvef_pct", "svi_ml_m2"]
ECHO_NUMERIC = ["mean_gradient_mmhg", "eoa_cm2", "dvi", "lvef_pct", "svi_ml_m2"]
LAB_COLUMNS = ["date", "marker", "value"]
EPISODE_COLUMNS = ["class", "agent", "indication", "start", "stop", "post_suspicion"]
STAGE_NOTES = {"0": "no worsening against the reference study",
               "1": "worsening below the stage 2 threshold",
               "2": "moderate haemodynamic valve deterioration (candidate finding)",
               "3": "severe haemodynamic valve deterioration (candidate finding)"}


class FormError(ValueError):
    """The form cannot be turned into a valid request; ``messages`` lists every problem."""

    def __init__(self, messages: list[str]):
        super().__init__("; ".join(messages))
        self.messages = messages


# --- value helpers -----------------------------------------------------------------------------
def to_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        ts = pd.to_datetime(value)
    except (ValueError, TypeError):
        return None
    return None if pd.isna(ts) else ts.date()


def to_num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _messages(ex: ValidationError, where: str) -> list[str]:
    out = []
    for err in ex.errors():
        loc = ".".join(str(x) for x in err.get("loc", ()) if x != "__root__")
        out.append(f"{where}: {loc + ': ' if loc else ''}{err.get('msg', 'invalid value')}")
    return out


# --- device choices ------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def device_options() -> pd.DataFrame:
    rows = []
    for _, r in device_table().iterrows():
        sizes = sorted({int(s) for s in str(r.get("sizes_mm", "")).split(";") if s.strip().isdigit()})
        if sizes and r["route"] in ROUTES:
            rows.append({"canonical_model": r["canonical_model"], "route": r["route"],
                         "design_class": r["design_class"], "sizes": sizes})
    return pd.DataFrame(rows)


def design_classes(route: str) -> list[str]:
    d = device_options()
    return sorted(d.loc[d["route"] == route, "design_class"].unique().tolist())


def models_for(route: str, design_class: str) -> list[str]:
    d = device_options()
    return sorted(d.loc[(d["route"] == route) & (d["design_class"] == design_class), "canonical_model"].tolist())


def sizes_for(model: str) -> list[int]:
    d = device_options()
    row = d.loc[d["canonical_model"] == model]
    return list(row.iloc[0]["sizes"]) if len(row) else []


def device_summary(model: str | None) -> dict:
    info = device_info(model) if model else {}
    raw = device_table()
    row = raw.loc[raw["canonical_model"] == model] if model else raw.iloc[0:0]
    return {"market_status": info.get("market_status", "unknown"),
            "market_detail": str(row.iloc[0]["market_status"]) if len(row) else "",
            "generation": info.get("generation") or "",
            "tissue_treatment": info.get("tissue_treatment") if isinstance(info.get("tissue_treatment"), str) else ""}


# --- synthetic example patients ------------------------------------------------------------------
def _echo(implant: date, days: int, gradient: float, eoa: float, dvi: float, ar: str = "none",
          lvef: float = 58.0, svi: float = 40.0) -> dict:
    return {"date": implant + timedelta(days=days), "mean_gradient_mmhg": float(gradient), "eoa_cm2": float(eoa),
            "dvi": float(dvi), "ar_grade": ar, "lvef_pct": float(lvef), "svi_ml_m2": float(svi)}


def _lab(implant: date, days: int, analyte: str, value: float) -> dict:
    return {"date": implant + timedelta(days=days), "analyte": analyte, "value": float(value)}


def _episode(cls: str, agent: str, indication: str, start: date, stop: date | None = None,
             post_suspicion: bool = False) -> dict:
    return {"class": cls, "agent": agent, "indication": indication, "start": start, "stop": stop,
            "post_suspicion": post_suspicion}


def _patient(age, sex, bsa, bmi, egfr, diabetes=False, diabetes_years=0.0, dialysis=False, af=False,
             bicuspid=False, lipid_lowering=False) -> dict:
    return {"age_at_implant": float(age), "sex": sex, "bsa_m2": float(bsa), "bmi": float(bmi),
            "egfr_ml_min": float(egfr), "diabetes": diabetes, "diabetes_duration_years": float(diabetes_years),
            "dialysis": dialysis, "atrial_fibrillation": af, "bicuspid_native_valve": bicuspid,
            "lipid_lowering": lipid_lowering}


def _form(route, design_class, model, size, implant, patient, echoes, labs, episodes) -> dict:
    return {"valve": {"route": route, "design_class": design_class, "canonical_model": model, "size_mm": size,
                      "implant_date": implant},
            "patient": patient, "echoes": echoes, "labs": labs, "episodes": episodes,
            "prediction_time": max(e["date"] for e in echoes), "jurisdiction": "ESC_EACTS"}


def preset_trifecta_gradual() -> dict:
    t0 = date(2016, 3, 14)
    echoes = [_echo(t0, 90, 11, 1.70, 0.48), _echo(t0, 455, 12, 1.70, 0.47), _echo(t0, 820, 14, 1.60, 0.45, "trace"),
              _echo(t0, 1185, 17, 1.50, 0.42, "trace", 57, 39), _echo(t0, 1550, 18, 1.45, 0.40, "mild", 56, 38)]
    labs = [_lab(t0, 90, "egfr", 58), _lab(t0, 820, "egfr", 55), _lab(t0, 1550, "egfr", 51),
            _lab(t0, 90, "hba1c", 7.4), _lab(t0, 1185, "hba1c", 7.9), _lab(t0, 90, "ldl", 95), _lab(t0, 90, "lpa", 160)]
    episodes = [_episode("SAPT", "aspirin", "other", t0)]
    return _form("SAVR", "externally mounted pericardial", "Trifecta", 21, t0,
                 _patient(66, "F", 1.80, 27.5, 58, diabetes=True, diabetes_years=8, lipid_lowering=True),
                 echoes, labs, episodes)


def preset_sapien_af_stable() -> dict:
    t0 = date(2019, 5, 20)
    echoes = [_echo(t0, 45, 11, 1.75, 0.52, "trace", 55, 38), _echo(t0, 410, 11, 1.75, 0.51, "trace", 55, 38),
              _echo(t0, 780, 12, 1.70, 0.50, "trace", 54, 37), _echo(t0, 1150, 12, 1.72, 0.50, "trace", 54, 37)]
    labs = [_lab(t0, 45, "egfr", 48), _lab(t0, 780, "egfr", 45), _lab(t0, 45, "ntprobnp", 420),
            _lab(t0, 1150, "ntprobnp", 450), _lab(t0, 780, "hscrp", 2.1)]
    episodes = [_episode("FXa", "apixaban", "AF", t0)]
    return _form("TAVR", "balloon-expandable intra-annular TAVR", "SAPIEN 3 Ultra", 26, t0,
                 _patient(76, "M", 1.95, 26.0, 48, af=True, lipid_lowering=True), echoes, labs, episodes)


def preset_perimount_regurgitation() -> dict:
    t0 = date(2014, 9, 2)
    echoes = [_echo(t0, 120, 10, 1.90, 0.55), _echo(t0, 485, 10, 1.90, 0.55), _echo(t0, 850, 11, 1.85, 0.54),
              _echo(t0, 1215, 11, 1.85, 0.53, "trace"), _echo(t0, 1580, 12, 1.80, 0.52, "severe", 60, 44)]
    labs = [_lab(t0, 120, "egfr", 80), _lab(t0, 1215, "egfr", 78), _lab(t0, 120, "ldl", 110)]
    episodes = [_episode("SAPT", "aspirin", "other", t0)]
    return _form("SAVR", "stented bovine pericardial, internally mounted", "Perimount", 23, t0,
                 _patient(58, "M", 2.05, 29.0, 80), echoes, labs, episodes)


def preset_mosaic_dialysis_vka() -> dict:
    t0 = date(2017, 1, 10)
    echoes = [_echo(t0, 60, 12, 1.50, 0.42, "none", 52, 36), _echo(t0, 425, 13, 1.45, 0.41, "none", 51, 35),
              _echo(t0, 790, 14, 1.40, 0.40, "trace", 50, 35), _echo(t0, 1155, 16, 1.30, 0.38, "trace", 48, 34)]
    labs = [_lab(t0, 60, "egfr", 16), _lab(t0, 790, "egfr", 15), _lab(t0, 60, "phosphate", 5.0),
            _lab(t0, 790, "phosphate", 5.2), _lab(t0, 60, "lpa", 60), _lab(t0, 60, "ntprobnp", 2400),
            _lab(t0, 60, "hba1c", 5.6), _lab(t0, 60, "dp_ucmgp", 1850), _lab(t0, 425, "dp_ucmgp", 2040),
            _lab(t0, 790, "dp_ucmgp", 2210), _lab(t0, 1155, "dp_ucmgp", 2380)]
    episodes = [_episode("VKA", "warfarin", "AF", t0)]
    return _form("SAVR", "stented porcine", "Mosaic", 23, t0,
                 _patient(68, "F", 1.65, 24.0, 16, dialysis=True, af=True, bicuspid=True), echoes, labs, episodes)


def preset_evolut_treated_thrombosis() -> dict:
    t0 = date(2020, 2, 3)
    echoes = [_echo(t0, 40, 8, 2.00, 0.56, "trace"), _echo(t0, 400, 9, 1.95, 0.55, "trace"),
              _echo(t0, 600, 21, 1.40, 0.40, "trace"), _echo(t0, 780, 9, 1.90, 0.55, "trace"),
              _echo(t0, 1150, 10, 1.90, 0.54, "trace")]
    labs = [_lab(t0, 40, "egfr", 62), _lab(t0, 780, "egfr", 60), _lab(t0, 400, "ntprobnp", 700)]
    episodes = [_episode("SAPT", "aspirin", "other", t0),
                _episode("FXa", "apixaban", "suspected_valve_thrombosis", t0 + timedelta(days=620),
                         t0 + timedelta(days=800), post_suspicion=True)]
    return _form("TAVR", "self-expanding supra-annular TAVR", "Evolut PRO", 29, t0,
                 _patient(79, "F", 1.70, 25.0, 62), echoes, labs, episodes)


def preset_blank() -> dict:
    t0 = date(2022, 6, 1)
    return _form("SAVR", "stented bovine pericardial, internally mounted", "Magna Ease", 23, t0,
                 _patient(70, "F", 1.85, 27.0, 70), [_echo(t0, 90, 11, 1.70, 0.48)], [], [])


PRESETS: OrderedDict[str, Any] = OrderedDict([
    ("Surgical Trifecta 21 mm, gradual stenosis, diabetic (66 y)", preset_trifecta_gradual),
    ("Transcatheter SAPIEN 3 Ultra 26 mm, stable, AF on apixaban (76 y)", preset_sapien_af_stable),
    ("Surgical Perimount 23 mm, new severe regurgitation (58 y)", preset_perimount_regurgitation),
    ("Surgical Mosaic 23 mm, dialysis, warfarin, bicuspid (68 y)", preset_mosaic_dialysis_vka),
    ("Transcatheter Evolut PRO 29 mm, treated leaflet thrombosis (79 y)", preset_evolut_treated_thrombosis),
    ("New patient: one reference echo only", preset_blank),
])


# --- frames for the editable tables --------------------------------------------------------------
def echo_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([{c: r.get(c) for c in ECHO_COLUMNS} for r in rows], columns=ECHO_COLUMNS)
    for c in ECHO_NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df


def labs_frame(rows: list[dict]) -> pd.DataFrame:
    recs = [{"date": r.get("date"), "marker": MODEL_ANALYTES[r["analyte"]][0], "value": r.get("value")}
            for r in rows if r.get("analyte") in MODEL_ANALYTES]
    df = pd.DataFrame(recs, columns=LAB_COLUMNS)
    df["value"] = pd.to_numeric(df["value"], errors="coerce").astype(float)
    return df


def episodes_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([{c: r.get(c) for c in EPISODE_COLUMNS} for r in rows], columns=EPISODE_COLUMNS)
    df["post_suspicion"] = df["post_suspicion"].fillna(False).astype(bool)
    return df


def echo_records(df: pd.DataFrame) -> list[dict]:
    return [{c: r.get(c) for c in ECHO_COLUMNS} for r in df.to_dict("records")]


def lab_records(df: pd.DataFrame) -> list[dict]:
    out = []
    for r in df.to_dict("records"):
        out.append({"date": r.get("date"), "analyte": ANALYTE_BY_LABEL.get(r.get("marker"), r.get("marker")),
                    "value": r.get("value")})
    return out


def episode_records(df: pd.DataFrame) -> list[dict]:
    return [{c: r.get(c) for c in EPISODE_COLUMNS} for r in df.to_dict("records")]


# --- form <-> request --------------------------------------------------------------------------------
def request_from_form(form: dict) -> PredictRequest:
    """Validate the whole form and build the predict request; raises FormError listing every problem."""
    errors: list[str] = []
    valve = form.get("valve", {})
    implant = to_date(valve.get("implant_date"))
    if implant is None:
        errors.append("Valve: the implant date is required.")
    passport = None
    model = valve.get("canonical_model") or None
    info = device_info(model) if model else {}
    try:
        passport = Passport(
            passport_id=form.get("passport_id") or DEMO_PASSPORT_ID,
            source=PassportSource(note_ref="demo-form", note_type="operative",
                                  date=(implant or date(2000, 1, 1)).isoformat()),
            route=valve.get("route") or "unknown", canonical_model=model,
            design_class=valve.get("design_class") or info.get("design_class"), generation=info.get("generation"),
            size_mm=int(valve["size_mm"]) if to_num(valve.get("size_mm")) is not None else None,
            implant_date=implant.isoformat() if implant else None, market_status=info.get("market_status", "unknown"))
    except ValidationError as ex:
        errors.extend(_messages(ex, "Valve"))
    pid = passport.passport_id if passport else DEMO_PASSPORT_ID

    echoes = []
    for i, row in enumerate(form.get("echoes", []), start=1):
        d = to_date(row.get("date"))
        values = {k: to_num(row.get(k)) for k in ECHO_NUMERIC}
        ar = row.get("ar_grade") if row.get("ar_grade") in AR_GRADES else None
        if d is None:
            if any(v is not None for v in values.values()):
                errors.append(f"Echo row {i}: the date is missing.")
            continue
        try:
            echoes.append(EchoObservation(passport_id=pid, date=d.isoformat(), ar_grade=ar,
                                          native_vs_prosthetic="prosthetic", source="manual", **values))
        except ValidationError as ex:
            errors.extend(_messages(ex, f"Echo row {i}"))
    if not echoes:
        errors.append("Echo series: add at least one echo; time zero is the first adequate echo 30 to 180 days after implant.")

    labs = []
    for i, row in enumerate(form.get("labs", []), start=1):
        d, value, analyte = to_date(row.get("date")), to_num(row.get("value")), row.get("analyte")
        if d is None and value is None and not analyte:
            continue
        if d is None or value is None or analyte not in MODEL_ANALYTES:
            errors.append(f"Laboratory row {i}: date, marker and value are all required.")
            continue
        labs.append(LabObservation(passport_id=pid, date=d.isoformat(), analyte=analyte, value=value,
                                   unit=MODEL_ANALYTES[analyte][1]))

    episodes = []
    for i, row in enumerate(form.get("episodes", []), start=1):
        start, stop, cls = to_date(row.get("start")), to_date(row.get("stop")), row.get("class")
        if start is None and not cls:
            continue
        if start is None or cls not in EXPOSURE_CLASSES:
            errors.append(f"Anticoagulant row {i}: class and start date are required.")
            continue
        if stop is not None and stop < start:
            errors.append(f"Anticoagulant row {i}: the stop date precedes the start date.")
            continue
        indication = row.get("indication") if row.get("indication") in INDICATIONS else "unknown"
        episodes.append(ExposureEpisode(**{"class": cls, "agent": (row.get("agent") or None), "indication": indication,
                                           "start": start.isoformat(), "stop": stop.isoformat() if stop else None,
                                           "source": "prescription",
                                           "started_within_90d_after_suspicious_echo": bool(row.get("post_suspicion"))}))

    p = form.get("patient", {})
    static = None
    try:
        static = PatientStatic(
            age_at_implant=to_num(p.get("age_at_implant")), sex=p.get("sex") or None, bsa_m2=to_num(p.get("bsa_m2")),
            bmi=to_num(p.get("bmi")), diabetes=bool(p.get("diabetes")),
            diabetes_duration_years=to_num(p.get("diabetes_duration_years")) if p.get("diabetes") else None,
            egfr_ml_min=to_num(p.get("egfr_ml_min")), dialysis=bool(p.get("dialysis")),
            atrial_fibrillation=bool(p.get("atrial_fibrillation")),
            bicuspid_native_valve=bool(p.get("bicuspid_native_valve")), lipid_lowering=bool(p.get("lipid_lowering")))
    except ValidationError as ex:
        errors.extend(_messages(ex, "Patient"))

    prediction_time = to_date(form.get("prediction_time"))
    if prediction_time is None:
        errors.append("Prediction time is required.")
    jurisdiction = form.get("jurisdiction") if form.get("jurisdiction") in JURISDICTIONS else "ESC_EACTS"
    if errors:
        raise FormError(errors)
    return PredictRequest(passport=passport, echo_observations=echoes,
                          exposure=ExposureTimeline(passport_id=pid, episodes=episodes), labs=labs, static=static,
                          prediction_time=prediction_time.isoformat(), jurisdiction=jurisdiction, source_kind="synthetic")


def form_from_request(req: dict) -> dict:
    """Inverse of request_from_form for a predict-request JSON (e.g. the train job's patient)."""
    p = req.get("passport", {})
    route = p.get("route") if p.get("route") in ROUTES else "SAVR"
    model = p.get("canonical_model")
    s = req.get("static") or {}
    patient = _patient(s.get("age_at_implant") or 70, s.get("sex") or "F", s.get("bsa_m2") or 1.85, s.get("bmi") or 27,
                       s.get("egfr_ml_min") or 70, diabetes=bool(s.get("diabetes")),
                       diabetes_years=s.get("diabetes_duration_years") or 0.0, dialysis=bool(s.get("dialysis")),
                       af=bool(s.get("atrial_fibrillation")), bicuspid=bool(s.get("bicuspid_native_valve")),
                       lipid_lowering=bool(s.get("lipid_lowering")))
    echoes = [{"date": to_date(o["date"]), **{k: o.get(k) for k in ECHO_NUMERIC}, "ar_grade": o.get("ar_grade") or "none"}
              for o in req.get("echo_observations", []) if o.get("native_vs_prosthetic", "prosthetic") == "prosthetic"]
    labs = [{"date": to_date(o["date"]), "analyte": o["analyte"], "value": o["value"]}
            for o in req.get("labs", []) if o.get("analyte") in MODEL_ANALYTES]
    episodes = [_episode(e["class"], e.get("agent") or "", e.get("indication") or "unknown", to_date(e["start"]),
                         to_date(e.get("stop")), bool(e.get("started_within_90d_after_suspicious_echo")))
                for e in (req.get("exposure") or {}).get("episodes", []) if e.get("class") in EXPOSURE_CLASSES]
    return {"valve": {"route": route, "design_class": p.get("design_class"), "canonical_model": model,
                      "size_mm": p.get("size_mm"), "implant_date": to_date(p.get("implant_date"))},
            "patient": patient, "echoes": echoes, "labs": labs, "episodes": episodes,
            "prediction_time": to_date(req.get("prediction_time")),
            "jurisdiction": req.get("jurisdiction") if req.get("jurisdiction") in JURISDICTIONS else "ESC_EACTS"}


# --- views derived from the form -----------------------------------------------------------------
def staging_rows(form: dict) -> tuple[list[dict], date | None]:
    """VARC-3 stage of every echo against the reference study; returns (rows, reference date)."""
    implant = to_date(form.get("valve", {}).get("implant_date"))
    obs = []
    for row in form.get("echoes", []):
        d = to_date(row.get("date"))
        if d is None:
            continue
        try:
            obs.append(EchoObservation(passport_id="staging", date=d.isoformat(), native_vs_prosthetic="prosthetic",
                                       ar_grade=row.get("ar_grade") if row.get("ar_grade") in AR_GRADES else None,
                                       **{k: to_num(row.get(k)) for k in ECHO_NUMERIC}))
        except ValidationError:
            continue
    points = to_points(obs)
    ref = select_reference(points, implant)
    rows = []
    for pt in points:
        days = (pt.date - implant).days if implant else None
        if ref is not None and pt is ref:
            stage, note = "reference", "time zero: first adequate echo 30 to 180 days after implant"
        elif ref is None:
            stage, note = "uncertain", "no reference study: needs mean gradient plus EOA or DVI, 30 to 180 days after implant"
        elif pt.date < ref.date:
            stage, note = "before time zero", "not used for staging"
        else:
            result = stage_point(pt, ref)
            stage = result.stage
            note = STAGE_NOTES.get(stage, result.reason[:110])
        rows.append({"date": pt.date, "days after implant": days, "VARC-3 stage": stage, "note": note})
    return rows, (ref.date if ref is not None else None)


def dp_ucmgp_coverage(form: dict) -> str:
    """What the predictor will do with this patient's dp-ucMGP values at the prediction time."""
    pt = to_date(form.get("prediction_time"))
    values = sorted((to_date(r.get("date")), to_num(r.get("value"))) for r in form.get("labs", [])
                    if r.get("analyte") == "dp_ucmgp" and to_date(r.get("date")) is not None
                    and to_num(r.get("value")) is not None and (pt is None or to_date(r.get("date")) <= pt))
    if not values:
        return "no value (the model reports without it, never imputed)"
    d, v = values[-1]
    months = ((pt - d).days / 365.25 * 12.0) if pt else 0.0
    if months > DP_UCMGP_CARRY_MONTHS:
        return f"{v:.0f} pmol/L on {d}, {months:.0f} months old: stale, not used"
    return f"{v:.0f} pmol/L on {d} ({months:.0f} months before the prediction time)"


def module_coverage(form: dict, dropped_modules: list[str] | tuple = (), dp_ucmgp_fitted: bool | None = None) -> list[dict]:
    pt = to_date(form.get("prediction_time"))
    present = {r["analyte"] for r in form.get("labs", [])
               if r.get("analyte") in MODEL_ANALYTES and to_num(r.get("value")) is not None
               and to_date(r.get("date")) is not None and (pt is None or to_date(r.get("date")) <= pt)}
    rows = []
    for block, label, markers in MODULES:
        in_model = block not in set(dropped_modules)
        if block == "vitamin_k":
            in_model = bool(dp_ucmgp_fitted) and "anticoagulant" not in set(dropped_modules)
            rows.append({"module": label, "in the model": "yes" if in_model else "switched off",
                         "this patient": dp_ucmgp_coverage(form)})
            continue
        if block == "anticoagulant":
            n = sum(1 for e in form.get("episodes", []) if to_date(e.get("start")) and (pt is None or to_date(e.get("start")) <= pt))
            patient_data = f"{n} episode(s) started by the prediction time" if n else "no episodes (treated as never exposed)"
        else:
            have = [MODEL_ANALYTES[m][0].split(" (")[0] for m in markers if m in present]
            patient_data = ", ".join(have) if have else "no values (treated as missing, never imputed into a result)"
        rows.append({"module": label, "in the model": "yes" if in_model else "switched off", "this patient": patient_data})
    return rows


def latest_echo(form: dict) -> dict | None:
    dated = [r for r in form.get("echoes", []) if to_date(r.get("date")) is not None]
    return max(dated, key=lambda r: to_date(r["date"])) if dated else None


def at_latest_echo(form: dict) -> dict:
    out = copy.deepcopy(form)
    last = latest_echo(out)
    if last is not None:
        out["prediction_time"] = to_date(last["date"])
    return out


def hypothetical_form(form: dict, months_after: int, gradient: float, eoa: float, dvi: float, ar_grade: str,
                      ac_change: str = "no change") -> dict:
    """The form with one hypothetical next echo, predicted at that echo's date."""
    out = copy.deepcopy(form)
    last = latest_echo(out)
    if last is None:
        raise FormError(["Add at least one echo before adding a hypothetical one."])
    last_date = to_date(last["date"])
    new_date = last_date + timedelta(days=round(months_after * 30.44))
    out["echoes"].append({"date": new_date, "mean_gradient_mmhg": float(gradient), "eoa_cm2": float(eoa), "dvi": float(dvi),
                          "ar_grade": ar_grade, "lvef_pct": to_num(last.get("lvef_pct")), "svi_ml_m2": to_num(last.get("svi_ml_m2"))})
    if ac_change in ("start VKA", "start factor Xa inhibitor"):
        cls, agent = ("VKA", "warfarin") if ac_change == "start VKA" else ("FXa", "apixaban")
        out["episodes"].append(_episode(cls, agent, "other", last_date))
    elif ac_change == "stop anticoagulation":
        for e in out["episodes"]:
            start, stop = to_date(e.get("start")), to_date(e.get("stop"))
            if e.get("class") in ORAL_ANTICOAGULANTS and start and start <= last_date and (stop is None or stop > last_date):
                e["stop"] = last_date
    out["prediction_time"] = new_date
    return out


def apply_passport(form: dict, passport: dict) -> tuple[dict, list[str]]:
    """Copy extracted valve fields into the form where they are usable; returns (form, notes)."""
    out, notes = copy.deepcopy(form), []
    valve = out["valve"]
    model = passport.get("canonical_model")
    options = device_options()
    if model and model in set(options["canonical_model"]):
        row = options.loc[options["canonical_model"] == model].iloc[0]
        valve.update(route=row["route"], design_class=row["design_class"], canonical_model=model)
        size = passport.get("size_mm")
        valve["size_mm"] = size if size in row["sizes"] else row["sizes"][0]
        if size and size not in row["sizes"]:
            notes.append(f"size {size} mm is not a listed size of {model}; the first listed size was used")
    elif model:
        notes.append(f"{model} has no listed sizes in the device table; the valve was not changed")
    elif passport.get("route") in ROUTES:
        notes.append("no valve model was found; only the route is known, so the valve was not changed")
    implant = passport.get("implant_date")
    if implant and len(implant) == 10:
        valve["implant_date"] = to_date(implant)
    elif implant:
        notes.append(f"implant date {implant} is year-only; time zero needs a day-level date, so it was not applied")
    return out, notes


def add_echo_observations(form: dict, observations: list[dict]) -> tuple[dict, int]:
    out, added = copy.deepcopy(form), 0
    for o in observations:
        d = o.get("date")
        if o.get("native_vs_prosthetic") != "prosthetic" or not d or len(d) != 10:
            continue
        out["echoes"].append({"date": to_date(d), **{k: o.get(k) for k in ECHO_NUMERIC},
                              "ar_grade": o.get("ar_grade") or "none"})
        added += 1
    return out, added
