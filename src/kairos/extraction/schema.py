"""Pydantic models for every KAIROS payload (design document section 3).

The same models validate service requests and responses and generate the OpenAPI
documents of both HTTP services. Field names and enumerations follow the design verbatim;
a few optional fields (evidence on echo observations, message explanations) are additions
that never change the documented shape.
"""
from __future__ import annotations

import re
import uuid
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kairos import CONDITIONING, HORIZONS_YEARS, ILLUSTRATIVE_LABEL

DATE_RE = re.compile(r"^\d{4}(-\d{2}-\d{2})?$")

NoteType = Literal["operative", "procedure", "progress", "discharge"]
Route = Literal["SAVR", "TAVR", "unknown"]
MarketStatus = Literal["active", "withdrawn", "discontinued", "unknown"]
EvidenceMethod = Literal["rule", "llm"]
EventType = Literal["valve_in_valve", "redo", "dysfunction", "endocarditis", "thrombosis"]
ARGrade = Literal["none", "trace", "mild", "moderate", "severe"]
NativeVsProsthetic = Literal["prosthetic", "native", "uncertain"]
ObservationSource = Literal["rule", "llm", "manual"]
ExposureClass = Literal["VKA", "FXa", "DTI", "SAPT", "DAPT", "none"]
Indication = Literal["AF", "VTE", "postop_prophylaxis", "suspected_valve_thrombosis", "other", "unknown"]
ExposureSource = Literal["prescription", "confirmed_adherence"]
SourceKind = Literal["real", "synthetic"]
Jurisdiction = Literal["ESC_EACTS", "ACC_AHA"]
Analyte = Literal["creatinine", "egfr", "hba1c", "ldl", "non_hdl", "phosphate", "calcium_corrected",
                  "calcium_ionised", "pth", "alp", "lpa", "ntprobnp", "hscrp", "dp_ucmgp", "lvef", "svi"]
DeviceEvidence = Literal["model-level", "class-level", "route-level", "none"]
Mechanism = Literal["svd", "thrombosis", "endocarditis", "non_structural", "uncertain"]
Phenotype = Literal["stenotic", "regurgitant", "mixed", "uncertain"]


def _check_date(v: str | None) -> str | None:
    if v is None or v == "":
        return None
    if not DATE_RE.match(v):
        raise ValueError("date must be YYYY-MM-DD or YYYY")
    if len(v) == 10:
        datetime.strptime(v, "%Y-%m-%d")
    return v


def parse_date(v: str | None) -> date | None:
    """Parse YYYY-MM-DD or YYYY (year-only maps to 1 January of that year)."""
    if not v:
        return None
    if len(v) == 4:
        return date(int(v), 1, 1)
    return datetime.strptime(v, "%Y-%m-%d").date()


def date_precision(v: str | None) -> Literal["day", "year", "none"]:
    if not v:
        return "none"
    return "day" if len(v) == 10 else "year"


def new_passport_id() -> str:
    return str(uuid.uuid4())


class Evidence(BaseModel):
    field: str
    span: tuple[int, int]
    text: str = Field(max_length=160)
    method: EvidenceMethod


class PassportSource(BaseModel):
    note_ref: str = Field(description="opaque note reference; never note text")
    note_type: NoteType
    date: str = Field(description="YYYY-MM-DD or YYYY")

    _v = field_validator("date")(_check_date)


class PassportEvent(BaseModel):
    type: EventType
    date: str | None = None
    span: tuple[int, int] | None = None

    _v = field_validator("date")(_check_date)


class Confidence(BaseModel):
    route: float = Field(0.0, ge=0, le=1)
    canonical_model: float = Field(0.0, ge=0, le=1)
    size_mm: float = Field(0.0, ge=0, le=1)


class Passport(BaseModel):
    passport_id: str = Field(default_factory=new_passport_id)
    source: PassportSource
    route: Route = "unknown"
    canonical_model: str | None = None
    design_class: str | None = None
    generation: str | None = None
    size_mm: int | None = Field(None, ge=15, le=36)
    implant_date: str | None = None
    market_status: MarketStatus = "unknown"
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: Confidence = Field(default_factory=Confidence)
    events: list[PassportEvent] = Field(default_factory=list)

    _v = field_validator("implant_date")(_check_date)


class EchoObservation(BaseModel):
    passport_id: str
    date: str
    mean_gradient_mmhg: float | None = Field(None, ge=0, le=150)
    peak_velocity_ms: float | None = Field(None, ge=0, le=8)
    eoa_cm2: float | None = Field(None, ge=0.1, le=6)
    dvi: float | None = Field(None, ge=0.05, le=1.5)
    ar_grade: ARGrade | None = None
    lvef_pct: float | None = Field(None, ge=5, le=90)
    svi_ml_m2: float | None = Field(None, ge=5, le=120)
    native_vs_prosthetic: NativeVsProsthetic = "uncertain"
    source: ObservationSource = "rule"
    evidence: list[Evidence] = Field(default_factory=list)

    _v = field_validator("date")(_check_date)


class ExposureEpisode(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    class_: ExposureClass = Field(alias="class")
    agent: str | None = None
    indication: Indication = "unknown"
    start: str
    stop: str | None = None
    source: ExposureSource = "prescription"
    started_within_90d_after_suspicious_echo: bool = False

    _v = field_validator("start", "stop")(_check_date)


class INRValue(BaseModel):
    date: str
    value: float = Field(ge=0.5, le=12)

    _v = field_validator("date")(_check_date)


class ExposureTimeline(BaseModel):
    passport_id: str
    episodes: list[ExposureEpisode] = Field(default_factory=list)
    inr: list[INRValue] = Field(default_factory=list)


class LabObservation(BaseModel):
    passport_id: str
    date: str
    analyte: Analyte
    value: float
    unit: str = ""
    assay: str | None = Field(None, max_length=80, description="assay name, recorded for research markers such as dp-ucMGP")

    _v = field_validator("date")(_check_date)


class PatientStatic(BaseModel):
    age_at_implant: float | None = Field(None, ge=18, le=105)
    sex: Literal["F", "M"] | None = None
    bsa_m2: float | None = Field(None, ge=1.0, le=3.0)
    bmi: float | None = Field(None, ge=12, le=70)
    diabetes: bool | None = None
    diabetes_duration_years: float | None = None
    egfr_ml_min: float | None = None
    dialysis: bool | None = None
    smoking: Literal["never", "former", "current"] | None = None
    atrial_fibrillation: bool | None = None
    other_anticoagulation_indication: bool | None = None
    bicuspid_native_valve: bool | None = None
    lipid_lowering: bool | None = None


class Driver(BaseModel):
    feature: str
    value: str
    direction: Literal["up", "down"]
    delta_probability: float | None = None
    method: str | None = None


ReasonCode = Literal["abrupt_failure_not_reliably_anticipated", "phenotype_performance_unestablished",
                     "insufficient_event_support", "outside_training_followup", "unknown_device_or_route",
                     "stale_echo", "module_unavailable", "prior_uncertain_finding", "exposure_coverage_unknown"]


class ReliabilityReason(BaseModel):
    code: ReasonCode
    severity: Literal["info", "warning", "blocking"]
    scope: Literal["bundle", "patient"]
    message: str
    detail: dict = Field(default_factory=dict)


class ModuleStatus(BaseModel):
    status: Literal["full", "partial", "unavailable"]
    included: list[str] = Field(default_factory=list)
    excluded: dict[str, str] = Field(default_factory=dict)


class ErrorDetail(BaseModel):
    code: str
    message: str
    detail: dict = Field(default_factory=dict)


class Reliability(BaseModel):
    device_evidence: DeviceEvidence
    data_completeness: float = Field(ge=0, le=1)
    stale_echo: bool
    label: str = ILLUSTRATIVE_LABEL
    reasons: list[ReliabilityReason] = Field(default_factory=list)
    applicable_scope: str | None = None
    model_family: str | None = None
    evaluation_support: dict | None = None
    modules: dict[str, ModuleStatus] = Field(default_factory=dict)
    excluded_features: list[str] = Field(default_factory=list)

    @field_validator("label")
    @classmethod
    def _label_fixed(cls, v: str) -> str:
        if ILLUSTRATIVE_LABEL not in v:
            raise ValueError("reliability label must carry the illustrative, unvalidated statement")
        return v


DpUcMgpUse = Literal["used", "not_measured", "stale", "not_in_model"]


class VitaminKStatus(BaseModel):
    """Whether a measured dp-ucMGP value entered this prediction (stated on every prediction)."""
    marker: Literal["dp-ucMGP"] = "dp-ucMGP"
    measured_value_used: bool
    status: DpUcMgpUse
    value_pmol_l: float | None = None
    measured_on: str | None = None
    assay: str | None = None
    note: str = ""

    @model_validator(mode="after")
    def _consistent(self) -> VitaminKStatus:
        if self.measured_value_used != (self.status == "used"):
            raise ValueError("measured_value_used must be true exactly when status is 'used'")
        return self


class Messages(BaseModel):
    current_abnormality: bool
    earlier_assessment: bool
    overdue_surveillance: bool


class Prediction(BaseModel):
    passport_id: str
    prediction_time: str
    conditioning: str = CONDITIONING
    horizons_years: list[int] = Field(default_factory=lambda: list(HORIZONS_YEARS))
    p_svd_before_death: list[float]
    p_death_before_svd: list[float]
    p_alive_intact: list[float]
    p_replaced_non_svd: list[float]
    p_svd_12m: float = Field(ge=0, le=1)
    drivers: list[Driver] = Field(default_factory=list)
    reliability: Reliability
    messages: Messages
    message_explanations: dict[str, str] = Field(default_factory=dict)
    dp_ucmgp: VitaminKStatus = Field(default_factory=lambda: VitaminKStatus(
        measured_value_used=False, status="not_measured", note="no dp-ucMGP value supplied"))
    model_version: str
    scenario_set: str
    model_family: str | None = None
    integration_version: str | None = None
    bundle_schema_version: int | None = None

    _v = field_validator("prediction_time")(_check_date)

    @model_validator(mode="after")
    def _check_probabilities(self) -> Prediction:
        if self.horizons_years != list(HORIZONS_YEARS):
            raise ValueError(f"horizons must be {list(HORIZONS_YEARS)}")
        series = [self.p_svd_before_death, self.p_death_before_svd, self.p_alive_intact,
                  self.p_replaced_non_svd]
        for s in series:
            if len(s) != len(HORIZONS_YEARS):
                raise ValueError("each probability list carries one value per horizon")
            if any(p < -1e-9 or p > 1 + 1e-9 for p in s):
                raise ValueError("probabilities must lie in [0, 1]")
        for i in range(len(HORIZONS_YEARS)):
            total = sum(s[i] for s in series)
            if abs(total - 1.0) > 1e-4:
                raise ValueError(f"probabilities at horizon {HORIZONS_YEARS[i]}y sum to {total:.6f}, not 1")
        if self.conditioning != CONDITIONING:
            raise ValueError("conditioning statement is fixed by the specification")
        return self


# --- service payloads -----------------------------------------------------------------
class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, max_length=200_000)
    source_kind: SourceKind
    note_ref: str = Field(min_length=1, max_length=120)
    note_type: NoteType
    date: str
    method: Literal["auto", "rules", "llm"] = "auto"
    passport_id: str | None = None

    _v = field_validator("date")(_check_date)


class Prescreen(BaseModel):
    mentions_prosthesis: bool
    method: Literal["rules", "llm", "rules+llm"]


class ExtractResponse(BaseModel):
    passport: Passport
    echo_observations: list[EchoObservation]
    prescreen: Prescreen
    methods_used: list[EvidenceMethod]
    llm_used: bool
    source_kind: SourceKind
    warnings: list[str] = Field(default_factory=list)
    label: str = ILLUSTRATIVE_LABEL


class PredictRequest(BaseModel):
    passport: Passport
    echo_observations: list[EchoObservation]
    exposure: ExposureTimeline | None = None
    labs: list[LabObservation] = Field(default_factory=list)
    static: PatientStatic | None = None
    prediction_time: str
    jurisdiction: Jurisdiction = "ESC_EACTS"
    source_kind: SourceKind = "synthetic"

    _v = field_validator("prediction_time")(_check_date)


class AdjudicationRequest(BaseModel):
    passport: Passport
    echo_observations: list[EchoObservation]
    exposure: ExposureTimeline | None = None
    source_kind: SourceKind = "synthetic"
    use_llm: bool = False


class CandidateFlagModel(BaseModel):
    date: str
    stage: str
    phenotype: Phenotype
    reason: str


class AdjudicationProposal(BaseModel):
    """A proposal for reviewers, never a label (design section 4)."""
    passport_id: str
    endpoint_met: bool
    endpoint_date: str | None = None
    stage: str | None = None
    mechanism: Mechanism = "uncertain"
    phenotype: Phenotype | None = None
    confidence: Literal["high", "moderate", "low"] = "low"
    candidates: list[CandidateFlagModel] = Field(default_factory=list)
    rationale: str = ""
    reference_source: str = "none"
    llm_proposal: dict | None = None
    status: Literal["proposal"] = "proposal"
    label: str = ILLUSTRATIVE_LABEL


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str
    model_version: str
    allow_real_notes_to_llm: bool | None = None
    llm_configured: bool | None = None
    label: str = ILLUSTRATIVE_LABEL


PAYLOAD_MODELS = {
    "passport": Passport,
    "echo_observation": EchoObservation,
    "exposure_timeline": ExposureTimeline,
    "prediction": Prediction,
    "extract_request": ExtractRequest,
    "extract_response": ExtractResponse,
    "predict_request": PredictRequest,
    "adjudication_proposal": AdjudicationProposal,
}
