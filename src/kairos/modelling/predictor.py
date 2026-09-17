"""Model bundle and the live predictor behind the predict service.

Rules encoded here (design section 3.4): the four probabilities sum to one within
tolerance; the complement of the SVD cumulative incidence is never presented as anything;
only observations dated on or before the prediction time are used; a passport that already
meets the endpoint (a confirmed adjudicated endpoint, or a replaced index valve) yields
``EndpointMetError`` (HTTP 409) instead of a prediction. Every prediction carries the
illustrative, unvalidated label and states whether a measured dp-ucMGP value was used (the
substudy offset model of :mod:`kairos.modelling.vitamin_k` scales the SVD hazard only).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from kairos import HORIZONS_YEARS, ILLUSTRATIVE_LABEL
from kairos.adjudication.framework import (
    EchoPoint,
    adjudicate,
    select_reference,
    stage_point,
    thrombosis_windows_from_exposures,
    to_points,
)
from kairos.extraction.rules import device_info
from kairos.extraction.schema import (
    Driver,
    Messages,
    ModuleStatus,
    Passport,
    Prediction,
    PredictRequest,
    Reliability,
    ReliabilityReason,
    VitaminKStatus,
    parse_date,
)
from kairos.modelling.base import (  # noqa: F401 - the unsupported errors are re-exported for the service
    ADAPTER_SCHEMA_VERSION,
    LegacyBundleError,
    UnsupportedFitError,
    UnsupportedRouteError,
    library_versions,
)
from kairos.modelling.cause_specific import CauseSpecificCoxModel
from kairos.modelling.cif import INTEGRATION_VERSION, combine_cause_specific, probabilities_at
from kairos.modelling.features import FeaturePipeline
from kairos.modelling.landmark import ExposureRecord, features_at_landmark
from kairos.modelling.modules import guideline_interval_months, load_model_config
from kairos.modelling.vitamin_k import LOG_FEATURE, MARKER, DpUcMgpOffsetModel

FEATURE_LABELS = {
    "design_class": "design class", "canonical_model": "valve model", "route": "route",
    "valve_age_years": "valve age", "t_lm": "time since reference echo", "age_at_implant": "age at implant",
    "ref_gradient": "reference mean gradient", "current_gradient": "current mean gradient",
    "delta_gradient": "gradient change from reference", "last_change_gradient": "most recent gradient change",
    "gradient_slope": "gradient slope", "current_eoa": "current EOA", "delta_eoa": "EOA change",
    "current_dvi": "current DVI", "delta_dvi": "DVI change", "current_ar": "regurgitation grade",
    "delta_ar": "regurgitation change", "size_mm": "label size", "egfr": "eGFR", "egfr0": "eGFR at implant",
    "egfr_slope": "eGFR slope", "dialysis": "dialysis", "diabetes": "diabetes", "hba1c": "HbA1c", "ldl": "LDL",
    "phosphate": "phosphate", "lpa": "lipoprotein(a)", "ntprobnp": "NT-proBNP", "hscrp": "hs-CRP",
    "ac_class_current": "current antithrombotic class", "ac_cum_vka_years": "cumulative VKA exposure",
    "ac_cum_oac_years": "cumulative anticoagulant exposure", "ac_post_suspicion": "anticoagulation started after a suspicious echo",
    "ac_indication": "anticoagulation indication", "ac_current_status": "anticoagulation status",
    "time_since_last_echo_years": "time since last echo", "stale_echo": "stale echo", "overdue_flag": "overdue surveillance",
    "ieoa": "indexed EOA", "mismatch_grade": "patient-prosthesis mismatch", "af": "atrial fibrillation",
    "implant_year": "implant year", "n_echoes": "number of echoes", "current_lvef": "LVEF", "current_svi": "stroke volume index",
    "ref_eoa": "reference EOA", "ref_dvi": "reference DVI", "ref_ar": "reference regurgitation grade", "ref_lvef": "reference LVEF",
    "ref_svi": "reference stroke volume index", "bsa": "body surface area", "bmi": "BMI", "sex": "sex", "generation": "device generation",
    "tissue_treatment": "tissue treatment", "bicuspid": "bicuspid native valve", "lipid_lowering": "lipid-lowering treatment",
    "diabetes_duration": "diabetes duration", "dialysis_current": "dialysis (current)", "last_change_interval_years": "interval of the last change",
    "ac_days_since_change": "days since the last antithrombotic change", "on_antiplatelet": "antiplatelet therapy",
    "calcium_corrected": "corrected calcium", "pth": "PTH", "alp": "alkaline phosphatase",
    LOG_FEATURE: "dp-ucMGP (vitamin K status)",
}


class EndpointMetError(Exception):
    """The passport already meets the endpoint; no prediction is made (HTTP 409)."""


class NoReferenceEchoError(Exception):
    """No qualifying reference echo 30-180 days after implantation (HTTP 400)."""


class PredictionTimeError(Exception):
    """Prediction time before time zero or before the last usable observation (HTTP 400)."""


@dataclass
class ModelBundle:
    """Schema version 2: one fitted model of one family behind the adapter interface
    (``cumulative_hazards``, ``row_support``, ``support_summary``, ``log_risk``), the frozen
    preprocessing and eligibility, and everything needed to reproduce and audit it."""
    pipeline: FeaturePipeline
    model: CauseSpecificCoxModel
    features: list[str]
    ladder_step: str
    scenario_set: str
    model_version: str
    trained_at: str
    module_availability: dict = field(default_factory=dict)
    eligibility: dict = field(default_factory=dict)          # EligibilityManifest.card(), frozen at training
    excluded_features: dict = field(default_factory=dict)    # feature -> reason (eligibility decisions)
    dropped_modules: list = field(default_factory=list)
    training_summary: dict = field(default_factory=dict)
    config_snapshot: dict = field(default_factory=dict)
    model_levels: list = field(default_factory=list)
    vitamin_k: DpUcMgpOffsetModel | None = None
    label: str = ILLUSTRATIVE_LABEL
    integration_version: str = INTEGRATION_VERSION
    support_mode: str = "full"
    schema_version: int = ADAPTER_SCHEMA_VERSION
    family: str = "cox"
    hyperparameters: dict = field(default_factory=dict)
    tuning_record: dict | None = None
    library_versions: dict = field(default_factory=dict)
    transformed_columns: list = field(default_factory=list)
    claimed_routes: list = field(default_factory=list)

    @property
    def cox(self):
        """Deprecated alias of ``model`` (kept for callers written before the family adapters)."""
        return self.model

    @property
    def vitamin_k_model(self) -> DpUcMgpOffsetModel | None:
        vk = getattr(self, "vitamin_k", None)  # bundles saved before dp-ucMGP was consumed lack the field
        return vk if (vk is not None and vk.fitted) else None

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path) -> ModelBundle:
        bundle = joblib.load(Path(path))
        if getattr(bundle, "__dict__", {}).get("schema_version") != ADAPTER_SCHEMA_VERSION or \
                not getattr(bundle, "integration_version", None) or "model" not in bundle.__dict__:
            raise LegacyBundleError(f"model bundle at {path} predates bundle schema {ADAPTER_SCHEMA_VERSION} and the hazard "
                                    f"contract (integration {INTEGRATION_VERSION}); it is rejected, not reinterpreted: retrain it")
        if bundle.integration_version != INTEGRATION_VERSION:
            raise LegacyBundleError(f"bundle integration {bundle.integration_version} differs from {INTEGRATION_VERSION}; retrain it")
        now = library_versions()
        for lib, v in (bundle.library_versions or {}).items():
            if v and now.get(lib) and v.split(".")[:2] != now[lib].split(".")[:2]:
                raise LegacyBundleError(f"bundle trained with {lib} {v}, runtime has {now[lib]}; retrain it")
        return bundle

    def card(self) -> dict:
        return {"ladder_step": self.ladder_step, "scenario_set": self.scenario_set, "model_version": self.model_version,
                "trained_at": self.trained_at, "n_features": len(self.features), "features": self.features,
                "module_availability": self.module_availability, "dropped_modules": self.dropped_modules,
                "eligibility": getattr(self, "eligibility", {}), "excluded_features": getattr(self, "excluded_features", {}),
                "training_summary": self.training_summary, "label": self.label,
                "family": self.family, "schema_version": self.schema_version, "hyperparameters": self.hyperparameters,
                "tuning": ({k: self.tuning_record.get(k) for k in ("status", "selected", "objective", "inner_splits", "seconds")}
                           if self.tuning_record else None),
                "library_versions": self.library_versions, "claimed_routes": self.claimed_routes,
                "event_counts": self.model.event_counts_,
                "integration_version": self.integration_version, "support_mode": self.support_mode,
                "model_support": self.model.support_summary(),
                "dp_ucmgp_substudy": (getattr(self, "vitamin_k", None).card() if getattr(self, "vitamin_k", None) is not None
                                      else {"fitted": False, "reason": "bundle predates dp-ucMGP consumption"})}


def _exposures(req: PredictRequest) -> list[ExposureRecord]:
    if req.exposure is None:
        return []
    out = []
    for e in req.exposure.episodes:
        start = parse_date(e.start)
        if start is None:
            continue
        out.append(ExposureRecord(e.class_, e.indication, start, parse_date(e.stop) if e.stop else None,
                                  e.started_within_90d_after_suspicious_echo))
    return out


def _static(req: PredictRequest, p: Passport) -> dict:
    s = req.static
    d = {"route": p.route if p.route != "unknown" else None, "design_class": p.design_class,
         "canonical_model": p.canonical_model, "generation": p.generation, "size_mm": p.size_mm,
         "implant_year": parse_date(p.implant_date).year if p.implant_date else None,
         "tissue_treatment": device_info(p.canonical_model).get("tissue_treatment") if p.canonical_model else None}
    if s is not None:
        d.update({"age_at_implant": s.age_at_implant, "sex": s.sex, "bsa": s.bsa_m2, "bmi": s.bmi,
                  "diabetes": s.diabetes, "diabetes_duration": s.diabetes_duration_years, "egfr0": s.egfr_ml_min,
                  "dialysis": s.dialysis, "af": s.atrial_fibrillation, "bicuspid": s.bicuspid_native_valve,
                  "lipid_lowering": s.lipid_lowering})
    return {k: (np.nan if v is None else v) for k, v in d.items()}


class Predictor:
    def __init__(self, bundle: ModelBundle, model_cfg: dict | None = None):
        self.bundle = bundle
        self.cfg = model_cfg or load_model_config()
        self.threshold_12m = float(self.cfg["messages"]["earlier_assessment_p_svd_12m_threshold"])
        self.grace_months = float(self.cfg["messages"]["overdue_grace_months"])
        self.stale_months = int(self.cfg["landmark"]["stale_echo_months"])
        self.dp_carry_months = float(self.cfg.get("vitamin_k", {}).get("carry_forward_months", 12))

    # --- pre-checks -----------------------------------------------------------------------
    def _points_and_reference(self, req: PredictRequest, prediction_time: date):
        obs = [o for o in req.echo_observations if parse_date(o.date) is not None and parse_date(o.date) <= prediction_time]
        points = to_points(obs, prosthetic_only=True)
        implant = parse_date(req.passport.implant_date)
        if implant is None:
            raise NoReferenceEchoError("passport has no implant date; time zero cannot be established")
        window = tuple(self.cfg["landmark"]["time_zero_window_days"])
        ref = select_reference(points, implant, window)
        if ref is None:
            raise NoReferenceEchoError(
                f"no adequate reference echo (mean gradient plus EOA or DVI) {window[0]}-{window[1]} days after implantation on or before {prediction_time}")
        if prediction_time < ref.date:
            raise PredictionTimeError("prediction time precedes time zero (the reference echo)")
        return points, ref, implant

    def _check_endpoint(self, req: PredictRequest, points: list[EchoPoint], ref: EchoPoint,
                        prediction_time: date, exposures: list[ExposureRecord]):
        p = req.passport
        for e in p.events:
            d = parse_date(e.date) if e.date else None
            if e.type in ("redo", "valve_in_valve") and d is not None and d <= prediction_time:
                raise EndpointMetError(
                    f"index valve replaced ({e.type} dated {e.date}) on or before the prediction time; the index valve is no longer at risk")
        windows = thrombosis_windows_from_exposures(req.exposure.episodes if req.exposure else [], points, ref)
        decision = adjudicate(points, ref, p.events, windows)
        if decision.met and decision.confidence == "high":
            raise EndpointMetError(
                f"passport already meets the primary endpoint: VARC-3 stage {decision.stage} ({decision.phenotype}) "
                f"on {decision.date}, confirmed by a subsequent study; no prediction is made")
        if not points or all(pt.date < ref.date for pt in points):
            pass
        has_dysfunction = any(e.type == "dysfunction" and e.date and parse_date(e.date) <= prediction_time for e in p.events)
        if has_dysfunction and len([pt for pt in points if pt.date > ref.date]) == 0:
            raise EndpointMetError(
                "the passport carries a prosthetic dysfunction mention on or before the prediction time and no echo "
                "series to adjudicate it; adjudicate the mention before predicting")
        return decision

    # --- main ----------------------------------------------------------------------------
    def predict(self, req: PredictRequest) -> Prediction:
        prediction_time = parse_date(req.prediction_time)
        points, ref, implant = self._points_and_reference(req, prediction_time)
        exposures = _exposures(req)
        decision = self._check_endpoint(req, points, ref, prediction_time, exposures)
        labs = [(parse_date(lab.date), lab.analyte, lab.value) for lab in req.labs if parse_date(lab.date) is not None]
        static = _static(req, req.passport)
        feats = features_at_landmark(static, points, ref, prediction_time, labs, exposures, implant, self.stale_months,
                                     self.dp_carry_months, exposure_records_available=req.exposure is not None)
        row = pd.DataFrame([feats])
        X = self.bundle.pipeline.transform(row)
        for s in self.bundle.model.strata:
            X[s] = row[s].astype("object").astype(str).to_numpy()   # a missing route is an error, never a default
        H = self.bundle.model.cumulative_hazards(X)
        dp_status = self._dp_ucmgp_status(req, row, prediction_time)
        dp_log_hr = None
        if dp_status.measured_value_used:
            vk = self.bundle.vitamin_k_model
            H = vk.adjust(H, row)
            dp_log_hr = float(vk.log_hr(row)[0])
        cifs = combine_cause_specific(H, self.bundle.model.grid)
        near = float(self.cfg["near_term_horizon_months"]) / 12.0
        probs = probabilities_at(cifs, self.bundle.model.grid, list(HORIZONS_YEARS), near)
        p_svd = [float(x) for x in probs["svd"][0]]
        p_death = [float(x) for x in probs["death"][0]]
        p_repl = [float(x) for x in probs["replacement"][0]]
        p_alive = [float(x) for x in probs["alive_intact"][0]]
        closure = max(abs(a + b + c + d - 1.0) for a, b, c, d in zip(p_svd, p_death, p_repl, p_alive))
        if closure > 1e-9:  # algebraic identity of the integration; a violation is a defect, not drift to hide
            raise RuntimeError(f"state probabilities do not sum to one (max deviation {closure:.3g})")
        p_svd_12m = float(probs["svd_near_term"][0])

        drivers = self._drivers(X, row, dp_log_hr)
        reliability = self._reliability(req.passport, row, prediction_time, points, req, decision)
        messages, explanations = self._messages(req, row, points, ref, prediction_time, p_svd_12m, decision)
        return Prediction(
            passport_id=req.passport.passport_id, prediction_time=req.prediction_time,
            p_svd_before_death=p_svd, p_death_before_svd=p_death, p_alive_intact=p_alive,
            p_replaced_non_svd=p_repl, p_svd_12m=p_svd_12m, drivers=drivers, reliability=reliability,
            messages=messages, message_explanations=explanations, dp_ucmgp=dp_status, model_version=self.bundle.model_version,
            scenario_set=self.bundle.scenario_set, model_family=self.bundle.family,
            integration_version=getattr(self.bundle, "integration_version", None),
            bundle_schema_version=getattr(self.bundle, "schema_version", None))

    # --- explanation pieces ----------------------------------------------------------------
    def _dp_ucmgp_status(self, req: PredictRequest, row: pd.DataFrame, prediction_time: date) -> VitaminKStatus:
        obs = [lab for lab in req.labs if lab.analyte == MARKER and parse_date(lab.date) is not None
               and parse_date(lab.date) <= prediction_time and lab.value > 0]
        if not obs:
            return VitaminKStatus(measured_value_used=False, status="not_measured",
                                  note="no dp-ucMGP value on or before the prediction time; the model reports without it (never imputed)")
        latest = max(obs, key=lambda lab: parse_date(lab.date))
        base = {"value_pmol_l": float(latest.value), "measured_on": latest.date, "assay": latest.assay}
        if bool(row["dp_ucmgp_stale"].iloc[0]):
            return VitaminKStatus(measured_value_used=False, status="stale", **base,
                                  note=(f"latest dp-ucMGP is {row['dp_ucmgp_age_months'].iloc[0]:.0f} months old, beyond the "
                                        f"{self.dp_carry_months:.0f}-month carry-forward window; not used"))
        if self.bundle.vitamin_k_model is None:
            vk_any = getattr(self.bundle, "vitamin_k", None)
            reason = vk_any.reason if vk_any is not None else "the model bundle predates dp-ucMGP consumption"
            return VitaminKStatus(measured_value_used=False, status="not_in_model", **base,
                                  note=f"a measured value exists but this model has no dp-ucMGP substudy model: {reason}")
        return VitaminKStatus(measured_value_used=True, status="used", **base,
                              note=(f"measured dp-ucMGP ({latest.value:.0f} pmol/L) used through the substudy offset model: spline main "
                                    f"effect and interaction with cumulative VKA exposure on the SVD hazard; {ILLUSTRATIVE_LABEL}"))

    def _drivers(self, X: pd.DataFrame, row: pd.DataFrame, dp_log_hr: float | None = None, k: int = 5) -> list[Driver]:
        contrib = self.bundle.model.contributions(X, "svd")
        if contrib.empty:
            return []
        grouped: dict = {}
        for col in contrib.columns:
            g = self.bundle.pipeline.group(col)
            grouped[g] = grouped.get(g, 0.0) + float(contrib[col].iloc[0])
        if dp_log_hr is not None:
            grouped[LOG_FEATURE] = dp_log_hr
        top = sorted(grouped.items(), key=lambda kv: -abs(kv[1]))[:k]
        out = []
        for feat, c in top:
            if abs(c) < 1e-6:
                continue
            raw = row[feat].iloc[0] if feat in row else None
            if feat == LOG_FEATURE:
                value = f"{row[MARKER].iloc[0]:.0f} pmol/L"
            elif raw is None or (isinstance(raw, float) and not np.isfinite(raw)):
                value = "missing"
            elif isinstance(raw, (float, np.floating)):
                value = f"{raw:.2f}"
            else:
                value = str(raw)
            out.append(Driver(feature=FEATURE_LABELS.get(feat, feat), value=value, direction="up" if c > 0 else "down"))
        return out

    def _training_scenarios(self) -> list[str]:
        """Training scenarios from bundle metadata only (bundle_evidence, else the scenario_set key frozen at training)."""
        ev = getattr(self.bundle, "bundle_evidence", None) or {}
        sc = ev.get("training_scenarios") or ev.get("scenarios") if isinstance(ev, dict) else None
        if sc:
            return [str(s) for s in (sc if isinstance(sc, (list, tuple)) else [sc])]
        key = str(self.bundle.scenario_set or "").split(":")[0]
        return [s for s in key.split("+") if s]

    def _bundle_reasons(self) -> tuple[list[ReliabilityReason], str | None]:
        reasons: list[ReliabilityReason] = []
        scenarios = self._training_scenarios()
        gradual_only = bool(scenarios) and all(s.startswith("gradual") for s in scenarios)
        scope = None
        if gradual_only:
            scope = "gradual stenotic deterioration; synthetic training only"
            reasons.append(ReliabilityReason(
                code="abrupt_failure_not_reliably_anticipated", severity="warning", scope="bundle",
                message="the model was trained on gradual deterioration only; abrupt valve failure is not reliably anticipated",
                detail={"training_scenarios": scenarios}))
            reasons.append(ReliabilityReason(
                code="phenotype_performance_unestablished", severity="warning", scope="bundle",
                message="performance for regurgitant, mixed or abrupt phenotypes is not established",
                detail={"training_scenarios": scenarios}))
        else:
            scope = f"synthetic training scenarios: {', '.join(scenarios) or 'unknown'}"
        support = self.bundle.training_summary.get("model_support") if isinstance(self.bundle.training_summary, dict) else None
        if isinstance(support, dict) and str(support.get("status", support.get("mode", ""))).lower() in ("insufficient", "unsupported", "partial"):
            reasons.append(ReliabilityReason(code="insufficient_event_support", severity="warning", scope="bundle",
                                             message="event support in training was limited for at least one cause",
                                             detail={"support": {k: v for k, v in support.items() if isinstance(v, (str, int, float, bool))}}))
        return reasons, scope

    def _modules(self) -> dict[str, ModuleStatus]:
        out = {}
        for m, info in (self.bundle.module_availability or {}).items():
            if not isinstance(info, dict):
                continue
            status = str(info.get("status", "full"))
            if status not in ("full", "partial", "unavailable"):
                status = "unavailable" if info.get("available") is False else "partial"
            inc = info.get("included") or []
            exc = info.get("excluded") or {}
            if not isinstance(exc, dict):
                exc = {str(e): str(info.get("reason") or "excluded") for e in exc}
            out[str(m)] = ModuleStatus(status=status, included=[str(i) for i in inc],
                                       excluded={str(k): str(v) for k, v in exc.items()})
        return out

    def _reliability(self, p: Passport, row: pd.DataFrame, prediction_time: date, points: list[EchoPoint],
                     req: PredictRequest | None = None, decision=None) -> Reliability:
        base = self._reliability_core(p, row, prediction_time, points)
        reasons, scope = self._bundle_reasons()
        modules = self._modules()
        if base.stale_echo:
            reasons.append(ReliabilityReason(code="stale_echo", severity="warning", scope="patient",
                                             message=f"the latest echo is older than {self.stale_months} months at the prediction time",
                                             detail={"latest_echo": points[-1].date.isoformat() if points else None}))
        unavailable = sorted(m for m, s in modules.items() if s.status == "unavailable")
        if unavailable:
            reasons.append(ReliabilityReason(code="module_unavailable", severity="info", scope="patient",
                                             message="some input modules were unavailable in training and do not enter this prediction",
                                             detail={"modules": unavailable}))
        grid = getattr(self.bundle.model, "grid", None)
        valve_age = row["valve_age_years"].iloc[0] if "valve_age_years" in row else np.nan
        max_t = float(np.max(grid)) if grid is not None and len(grid) else None
        if max_t is not None and np.isfinite(valve_age) and float(valve_age) > max_t:
            reasons.append(ReliabilityReason(code="outside_training_followup", severity="warning", scope="patient",
                                             message="valve age at prediction lies beyond the follow-up seen in training",
                                             detail={"valve_age_years": round(float(valve_age), 2), "training_max_years": round(max_t, 2)}))
        if decision is not None and getattr(decision, "uncertain_dates", None):
            reasons.append(ReliabilityReason(code="prior_uncertain_finding", severity="info", scope="patient",
                                             message="an earlier stage 2/3 finding resolved on a later study and is classed uncertain",
                                             detail={"dates": [str(d) for d in decision.uncertain_dates]}))
        uses_exposure = any(f.startswith("ac_") for f in self.bundle.features)
        if req is not None and req.exposure is None and uses_exposure:
            reasons.append(ReliabilityReason(code="exposure_coverage_unknown", severity="info", scope="patient",
                                             message="no antithrombotic exposure timeline was supplied; exposure features are unknown"))
        ts = self.bundle.training_summary if isinstance(self.bundle.training_summary, dict) else {}
        evaluation_support = {k: ts[k] for k in ("n_patients", "events", "event_support") if k in ts} or None
        return base.model_copy(update={
            "reasons": reasons, "applicable_scope": scope, "model_family": self.bundle.family,
            "evaluation_support": evaluation_support, "modules": modules,
            "excluded_features": sorted(str(f) for f in (getattr(self.bundle, "excluded_features", {}) or {}))})

    def _reliability_core(self, p: Passport, row: pd.DataFrame, prediction_time: date, points: list[EchoPoint]) -> Reliability:
        if p.canonical_model and p.canonical_model in self.bundle.model_levels:
            device = "model-level"
        elif p.design_class:
            device = "class-level"
        elif p.route in ("SAVR", "TAVR"):
            device = "route-level"
        else:
            device = "none"
        feats = [f for f in self.bundle.features if f in row]
        present = sum(1 for f in feats if not (isinstance(row[f].iloc[0], float) and not np.isfinite(row[f].iloc[0])) and row[f].iloc[0] is not None)
        completeness = present / max(1, len(feats))
        stale = bool(row["stale_echo"].iloc[0]) if "stale_echo" in row else False
        return Reliability(device_evidence=device, data_completeness=round(float(completeness), 3), stale_echo=stale,
                           label=ILLUSTRATIVE_LABEL)

    def _messages(self, req: PredictRequest, row: pd.DataFrame, points: list[EchoPoint], ref: EchoPoint,
                  prediction_time: date, p_svd_12m: float, decision) -> tuple[Messages, dict]:
        expl = {}
        latest = points[-1]
        stage = stage_point(latest, ref).stage if latest.date > ref.date else "0"
        current = stage in ("2", "3")
        if current:
            expl["current_abnormality"] = (f"latest echo ({latest.date}) meets VARC-3 stage {stage} criteria against the reference study; "
                                           "a finding for clinical assessment and a confirmatory study, not a prediction")
        elif decision.uncertain_dates:
            expl["current_abnormality"] = "an earlier stage 2/3 finding resolved on a later study and is classed uncertain"
        else:
            expl["current_abnormality"] = f"latest echo ({latest.date}) is VARC-3 stage {stage} against the reference study"
        earlier = p_svd_12m >= self.threshold_12m
        expl["earlier_assessment"] = (f"predicted 12-month SVD-before-death risk {p_svd_12m:.1%} versus threshold {self.threshold_12m:.0%} "
                                      f"(threshold labelled assumed); {ILLUSTRATIVE_LABEL}")
        route = req.passport.route if req.passport.route in ("SAVR", "TAVR") else "SAVR"
        valve_age = float(row["valve_age_years"].iloc[0]) if "valve_age_years" in row else 0.0
        interval, desc = guideline_interval_months(self.cfg, req.jurisdiction, route, valve_age)
        months_since = float(row["time_since_last_echo_years"].iloc[0]) * 12.0
        overdue = months_since > interval + self.grace_months
        expl["overdue_surveillance"] = (f"{months_since:.0f} months since the last echo; guideline interval {interval:.0f} months "
                                        f"({req.jurisdiction}: {desc}); reminder needs no model and the guideline schedule is unchanged")
        return Messages(current_abnormality=current, earlier_assessment=earlier, overdue_surveillance=overdue), expl


def make_bundle(pipeline: FeaturePipeline, model, features: list[str], ladder_step: str,
                scenario_set: str, model_version: str, availability: dict, dropped: list, summary: dict,
                cfg: dict, model_levels: list, vitamin_k: DpUcMgpOffsetModel | None = None,
                eligibility: dict | None = None, excluded_features: dict | None = None,
                tuning_record: dict | None = None) -> ModelBundle:
    if hasattr(model, "drop_training_data"):
        model.drop_training_data()
    return ModelBundle(pipeline=pipeline, model=model, features=features, ladder_step=ladder_step,
                       family=model.family, hyperparameters=dict(model.hyperparameters), tuning_record=tuning_record,
                       library_versions=library_versions(), transformed_columns=list(pipeline.output_columns_),
                       claimed_routes=list(model.supported_routes),
                       scenario_set=scenario_set, model_version=model_version,
                       trained_at=datetime.now(UTC).isoformat(), module_availability=availability,
                       dropped_modules=dropped, training_summary=summary,
                       config_snapshot={"model": cfg.get("model"), "landmark": cfg.get("landmark"),
                                        "vitamin_k": cfg.get("vitamin_k"), "eligibility": cfg.get("eligibility"),
                                        "support": cfg.get("support"), "families": cfg.get("families")},
                       model_levels=model_levels, vitamin_k=vitamin_k, eligibility=eligibility or {},
                       support_mode=str(model.gates.get("mode", "full")),
                       excluded_features=excluded_features or {})
