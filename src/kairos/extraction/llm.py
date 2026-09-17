"""Hosted-model extraction (Azure OpenAI, Structured Outputs) behind the real-notes flag.

Design section 4: rules first, hosted model second; schema-constrained output (strict JSON
schema); Managed Identity instead of API keys; every call carries the header
``x-kairos-source`` with the value ``real`` or ``synthetic``; the service refuses to call a
model with real text whenever ``ALLOW_REAL_NOTES_TO_LLM`` is off. The flag is read from
:class:`kairos.io.config.Settings` only.

The deployed model is GPT-5.1 (regional Standard, Sweden Central), a reasoning model: it
accepts ``reasoning_effort`` and rejects any temperature other than the default, so
determinism comes from the strict schema and a low reasoning effort rather than from
temperature 0. A deployment of a non-reasoning model is supported by setting its reasoning
effort to an empty string, which sends ``temperature=0`` instead.

``KAIROS_OPENAI_API_VERSION`` selects the endpoint: a dated version uses the classic
``/openai/deployments/<name>`` path through ``AzureOpenAI``; ``v1`` uses the version-free
``/openai/v1/`` path through the standard client.

Prompts and completions are not logged by this module. Model output is merged into the
rules result (rules win on conflict, the model fills gaps and adds evidence quotes that are
located in the note to produce spans).
"""
from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

from kairos.extraction.rules import (
    RulesResult,
    _ev,
    device_info,
    known_models,
    locate_quote,
)
from kairos.extraction.schema import (
    EchoObservation,
    Evidence,
    PassportEvent,
    Prescreen,
)
from kairos.io.config import Settings, azure_credential, get_settings

log = logging.getLogger("kairos.extraction.llm")

COGNITIVE_SCOPE = "https://cognitiveservices.azure.com/.default"


class RealNotesNotPermitted(PermissionError):
    """Raised when real note text would reach a hosted model while the flag is off."""


class LLMNotConfigured(RuntimeError):
    """Raised when the hosted-model path is requested but no deployment is configured."""


# --- strict-schema models for Structured Outputs (all fields required, nullable) -----------
ARGradeLit = Literal["none", "trace", "mild", "moderate", "severe"]


class LLMEcho(BaseModel):
    mean_gradient_mmhg: float | None
    peak_velocity_ms: float | None
    eoa_cm2: float | None
    dvi: float | None
    ar_grade: ARGradeLit | None
    lvef_pct: float | None
    svi_ml_m2: float | None


class LLMEvidence(BaseModel):
    field: Literal["route", "canonical_model", "size_mm", "implant_year", "event",
                   "mean_gradient_mmhg", "eoa_cm2", "dvi", "ar_grade", "lvef_pct"]
    quote: str = Field(description="verbatim quote from the note, at most 120 characters")


class LLMPassportExtraction(BaseModel):
    mentions_aortic_valve_prosthesis: bool
    route: Literal["SAVR", "TAVR", "unknown"]
    canonical_model: str | None = Field(description="one of the allowed canonical model names, or null")
    size_mm: int | None
    implant_year: int | None
    events: list[Literal["valve_in_valve", "redo", "dysfunction", "endocarditis", "thrombosis"]]
    prosthetic_echo: LLMEcho | None = Field(description="values that describe the prosthetic valve")
    native_echo: LLMEcho | None = Field(description="values that describe the native valve before replacement")
    evidence: list[LLMEvidence]


class LLMAdjudication(BaseModel):
    mechanism: Literal["svd", "thrombosis", "endocarditis", "non_structural", "uncertain"]
    confidence: Literal["high", "moderate", "low"]
    rationale: str


class LLMPrescreen(BaseModel):
    mentions_aortic_valve_prosthesis: bool


SYSTEM_PROMPT = (
    "You extract structured facts about a bioprosthetic AORTIC valve from one clinical note. "
    "Return only what the note states; never guess. Use null when a field is not stated. "
    "canonical_model must be exactly one of the allowed names or null. "
    "Sizes are label sizes in millimetres (17 to 34). Mean gradients are in mmHg. "
    "Distinguish values that describe the prosthetic valve from values that describe the native "
    "valve before replacement. Every evidence item quotes the note verbatim (at most 120 characters). "
    "This is a research prototype; output is illustrative and unvalidated."
)

ADJUDICATION_PROMPT = (
    "You assist reviewers who adjudicate bioprosthetic aortic valve dysfunction under VARC-3. "
    "Given extracted echo observations staged against the patient's reference study and any event "
    "mentions, propose the most likely mechanism (structural valve deterioration, thrombosis, "
    "endocarditis, non-structural dysfunction, or uncertain) with a confidence grade and a short "
    "rationale. This is a proposal for reviewers, never a label, and never a treatment recommendation."
)


EXTRACT_MAX_COMPLETION_TOKENS = 6000      # includes reasoning tokens
ADJUDICATE_MAX_COMPLETION_TOKENS = 8000
PRESCREEN_MAX_COMPLETION_TOKENS = 2000
CLIENT_TIMEOUT_SECONDS = 120.0
CLIENT_MAX_RETRIES = 2


def sampling_kwargs(reasoning_effort: str, max_completion_tokens: int) -> dict:
    """Request parameters for one call. Reasoning models (GPT-5 family, o-series) take
    ``reasoning_effort`` and reject a non-default temperature; other chat models get
    ``temperature=0``. ``max_completion_tokens`` covers reasoning plus visible output."""
    kwargs: dict = {"max_completion_tokens": max_completion_tokens}
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    else:
        kwargs["temperature"] = 0
    return kwargs


def build_client(settings: Settings | None = None, api_version: str | None = None):
    s = settings or get_settings()
    if not s.openai_endpoint:
        raise LLMNotConfigured("KAIROS_OPENAI_ENDPOINT is not set")
    version = (api_version or s.openai_api_version).strip()
    if s.openai_api_key:  # local development only; Azure uses the managed identity
        credential = s.openai_api_key
    else:
        from azure.identity import get_bearer_token_provider

        credential = get_bearer_token_provider(azure_credential(s), COGNITIVE_SCOPE)
    if version.lower() == "v1":
        from openai import OpenAI

        return OpenAI(base_url=s.openai_endpoint.rstrip("/") + "/openai/v1/", api_key=credential,
                      timeout=CLIENT_TIMEOUT_SECONDS, max_retries=CLIENT_MAX_RETRIES)
    from openai import AzureOpenAI

    auth = {"api_key": credential} if s.openai_api_key else {"azure_ad_token_provider": credential}
    return AzureOpenAI(azure_endpoint=s.openai_endpoint, api_version=version,
                       timeout=CLIENT_TIMEOUT_SECONDS, max_retries=CLIENT_MAX_RETRIES, **auth)


def describe_error(ex: Exception) -> str:
    """Short, text-free description of a model call failure for logs and warnings."""
    parts = [type(ex).__name__]
    for attr in ("status_code", "code", "param"):
        value = getattr(ex, attr, None)
        if value:
            parts.append(f"{attr}={value}")
    return " ".join(parts)


class LLMExtractor:
    def __init__(self, settings: Settings | None = None, client=None):
        self.settings = settings or get_settings()
        self._client = client

    @property
    def available(self) -> bool:
        return bool(self.settings.openai_endpoint and self.settings.openai_extract_deployment)

    @property
    def adjudication_available(self) -> bool:
        return bool(self.settings.openai_endpoint and self.settings.openai_adjudicate_deployment)

    @property
    def prescreen_available(self) -> bool:
        return bool(self.settings.openai_endpoint and self.settings.openai_prescreen_deployment)

    def guard(self, source_kind: str) -> None:
        """Refuse to send real text to a model while the flag is off (rule 0.3, section 8)."""
        if source_kind == "real" and not self.settings.allow_real_notes_to_llm:
            raise RealNotesNotPermitted(
                "ALLOW_REAL_NOTES_TO_LLM is off: real note text is not sent to a hosted model")

    @property
    def client(self):
        if self._client is None:
            self._client = build_client(self.settings)
        return self._client

    def extract(self, text: str, source_kind: str) -> LLMPassportExtraction:
        self.guard(source_kind)
        if not self.available:
            raise LLMNotConfigured("no extraction deployment configured")
        allowed = ", ".join(known_models())
        completion = self.client.chat.completions.parse(
            model=self.settings.openai_extract_deployment,
            messages=[{"role": "system", "content": SYSTEM_PROMPT + " Allowed canonical model names: " + allowed},
                      {"role": "user", "content": text}],
            response_format=LLMPassportExtraction,
            extra_headers={"x-kairos-source": source_kind},
            **sampling_kwargs(self.settings.openai_extract_reasoning_effort, EXTRACT_MAX_COMPLETION_TOKENS),
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError("model returned no parsable structured output")
        return parsed

    def prescreen(self, text: str, source_kind: str) -> bool | None:
        """Cheap pre-screen; returns None when no pre-screen deployment is configured."""
        if not self.prescreen_available:
            return None
        self.guard(source_kind)
        completion = self.client.chat.completions.parse(
            model=self.settings.openai_prescreen_deployment,
            messages=[{"role": "system", "content": "Does this note mention an aortic valve prosthesis (surgical or transcatheter)? Answer in the schema."},
                      {"role": "user", "content": text[:6000]}],
            response_format=LLMPrescreen, extra_headers={"x-kairos-source": source_kind},
            **sampling_kwargs(self.settings.openai_prescreen_reasoning_effort, PRESCREEN_MAX_COMPLETION_TOKENS))
        parsed = completion.choices[0].message.parsed
        return None if parsed is None else parsed.mentions_aortic_valve_prosthesis

    def adjudicate(self, summary: str, source_kind: str) -> LLMAdjudication:
        self.guard(source_kind)
        if not self.adjudication_available:
            raise LLMNotConfigured("no adjudication deployment configured")
        completion = self.client.chat.completions.parse(
            model=self.settings.openai_adjudicate_deployment,
            messages=[{"role": "system", "content": ADJUDICATION_PROMPT},
                      {"role": "user", "content": summary}],
            response_format=LLMAdjudication,
            extra_headers={"x-kairos-source": source_kind},
            **sampling_kwargs(self.settings.openai_adjudicate_reasoning_effort, ADJUDICATE_MAX_COMPLETION_TOKENS),
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError("model returned no parsable structured output")
        return parsed

    def self_check(self) -> list[dict]:
        """One tiny structured-output call per configured deployment, with a fixed synthetic
        sentence, to confirm the deployment, API version and request parameters work."""
        roles = [("extract", self.settings.openai_extract_deployment, self.settings.openai_extract_reasoning_effort),
                 ("prescreen", self.settings.openai_prescreen_deployment, self.settings.openai_prescreen_reasoning_effort),
                 ("adjudicate", self.settings.openai_adjudicate_deployment, self.settings.openai_adjudicate_reasoning_effort)]
        results = []
        for role, deployment, effort in roles:
            if not (self.settings.openai_endpoint and deployment):
                results.append({"role": role, "deployment": deployment or None, "ok": None, "detail": "not configured"})
                continue
            try:
                completion = self.client.chat.completions.parse(
                    model=deployment,
                    messages=[{"role": "system", "content": "Answer in the schema."},
                              {"role": "user", "content": SELF_CHECK_TEXT}],
                    response_format=LLMPrescreen,
                    extra_headers={"x-kairos-source": "synthetic"},
                    **sampling_kwargs(effort, PRESCREEN_MAX_COMPLETION_TOKENS),
                )
                parsed = completion.choices[0].message.parsed
                usage = getattr(completion, "usage", None)
                details = getattr(usage, "completion_tokens_details", None)
                results.append({"role": role, "deployment": deployment, "ok": bool(parsed and parsed.mentions_aortic_valve_prosthesis),
                                "model": getattr(completion, "model", None),
                                "reasoning_tokens": getattr(details, "reasoning_tokens", None),
                                "detail": "structured output parsed" if parsed else "no parsed output"})
            except Exception as ex:  # noqa: BLE001 - reported, not raised
                results.append({"role": role, "deployment": deployment, "ok": False, "detail": describe_error(ex)})
        return results


SELF_CHECK_TEXT = ("SYNTHETIC CHECK. Surveillance visit after surgical aortic valve replacement with a "
                   "23 mm bioprosthesis; the prosthetic mean gradient is 11 mmHg.")


def _match_model_name(name: str | None) -> str | None:
    if not name:
        return None
    models = known_models()
    if name in models:
        return name
    lower = {m.lower(): m for m in models}
    return lower.get(name.strip().lower())


def _echo_from_llm(e: LLMEcho, passport_id: str, note_date: str, cls: str,
                   evidence: list[Evidence]) -> EchoObservation | None:
    vals = dict(mean_gradient_mmhg=e.mean_gradient_mmhg, peak_velocity_ms=e.peak_velocity_ms,
                eoa_cm2=e.eoa_cm2, dvi=e.dvi, ar_grade=e.ar_grade, lvef_pct=e.lvef_pct,
                svi_ml_m2=e.svi_ml_m2)
    if all(v is None for v in vals.values()):
        return None
    try:
        return EchoObservation(passport_id=passport_id, date=note_date, native_vs_prosthetic=cls,
                               source="llm", evidence=evidence, **vals)
    except Exception:  # noqa: BLE001 - out-of-range model values are dropped, not trusted
        return None


def merge(rules: RulesResult, llm: LLMPassportExtraction, text: str, note_date: str) -> RulesResult:
    """Rules first, model second: fill gaps, add located evidence, record disagreements."""
    p = rules.passport
    warnings = list(rules.warnings)
    llm_evidence: list[Evidence] = []
    for item in llm.evidence:
        span = locate_quote(text, item.quote)
        if span is None:
            continue
        llm_evidence.append(_ev(text, item.field, span[0], span[1], method="llm"))
    p.evidence.extend(llm_evidence)

    llm_model = _match_model_name(llm.canonical_model)
    if llm.canonical_model and llm_model is None:
        warnings.append("model proposed a device name outside the device table; ignored")
    if p.canonical_model is None and llm_model:
        info = device_info(llm_model)
        p.canonical_model = llm_model
        p.design_class = info.get("design_class")
        p.generation = info.get("generation")
        p.market_status = info.get("market_status", "unknown")
        p.confidence.canonical_model = 0.6
        if p.route == "unknown" and info.get("route") in ("SAVR", "TAVR"):
            p.route, p.confidence.route = info["route"], 0.5
    elif p.canonical_model and llm_model:
        if llm_model == p.canonical_model:
            p.confidence.canonical_model = max(p.confidence.canonical_model, 0.95)
        else:
            warnings.append(f"rules and model disagree on the device ({p.canonical_model} vs {llm_model}); rules kept")

    if p.route == "unknown" and llm.route in ("SAVR", "TAVR"):
        p.route, p.confidence.route = llm.route, 0.6
    elif p.route != "unknown" and llm.route in ("SAVR", "TAVR"):
        if llm.route == p.route:
            p.confidence.route = max(p.confidence.route, 0.95)
        else:
            warnings.append(f"rules and model disagree on the route ({p.route} vs {llm.route}); rules kept")

    if p.size_mm is None and llm.size_mm and 17 <= llm.size_mm <= 34:
        p.size_mm, p.confidence.size_mm = llm.size_mm, 0.6
    elif p.size_mm and llm.size_mm:
        if llm.size_mm == p.size_mm:
            p.confidence.size_mm = max(p.confidence.size_mm, 0.95)
        else:
            warnings.append(f"rules and model disagree on the size ({p.size_mm} vs {llm.size_mm}); rules kept")

    if p.implant_date is None and llm.implant_year and 1990 <= llm.implant_year <= 2030:
        p.implant_date = str(llm.implant_year)

    known_events = {e.type for e in p.events}
    for ev in llm.events:
        if ev not in known_events:
            p.events.append(PassportEvent(type=ev, date=note_date, span=None))

    echos = list(rules.echo_observations)
    have_prosthetic = any(o.native_vs_prosthetic == "prosthetic" for o in echos)
    if llm.prosthetic_echo is not None:
        ev = [e for e in llm_evidence if e.field in ("mean_gradient_mmhg", "eoa_cm2", "dvi", "ar_grade", "lvef_pct")]
        obs = _echo_from_llm(llm.prosthetic_echo, p.passport_id, note_date, "prosthetic", ev)
        if obs is not None:
            if not have_prosthetic:
                echos.append(obs)
            else:
                target = next(o for o in echos if o.native_vs_prosthetic == "prosthetic")
                for f in ("mean_gradient_mmhg", "peak_velocity_ms", "eoa_cm2", "dvi", "ar_grade", "lvef_pct", "svi_ml_m2"):
                    if getattr(target, f) is None and getattr(obs, f) is not None:
                        setattr(target, f, getattr(obs, f))
                        target.source = "rule"
    if llm.native_echo is not None and not any(o.native_vs_prosthetic == "native" for o in echos):
        obs = _echo_from_llm(llm.native_echo, p.passport_id, note_date, "native", [])
        if obs is not None:
            echos.append(obs)

    prescreen = Prescreen(mentions_prosthesis=rules.prescreen.mentions_prosthesis or llm.mentions_aortic_valve_prosthesis,
                          method="rules+llm")
    return RulesResult(passport=p, echo_observations=echos, prescreen=prescreen, warnings=warnings)
