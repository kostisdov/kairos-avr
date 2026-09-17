"""kairos-extract: note text in, valve passport JSON out (design section 1, milestone M2).

Rules first, hosted model second. The hosted-model path is gated by the configuration
flag ``ALLOW_REAL_NOTES_TO_LLM``: with the flag off, a request carrying real text that
would reach a model is refused with HTTP 403 unless it explicitly asks for ``rules``.
Note text is never stored or logged; extracted fields and evidence spans are.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from kairos import ILLUSTRATIVE_LABEL, __version__
from kairos.adjudication.framework import (
    adjudicate,
    candidate_flags,
    select_reference,
    thrombosis_windows_from_exposures,
    to_points,
)
from kairos.extraction.llm import (
    LLMExtractor,
    LLMNotConfigured,
    RealNotesNotPermitted,
    describe_error,
    merge,
)
from kairos.extraction.rules import extract_rules
from kairos.extraction.schema import (
    AdjudicationProposal,
    AdjudicationRequest,
    CandidateFlagModel,
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
    parse_date,
)
from kairos.io.config import Settings, get_settings
from kairos.io.db import Repository, get_repository
from kairos.telemetry import configure_telemetry, get_logger

DESCRIPTION = (
    "KAIROS research prototype: valve passport extraction from de-identified note text. "
    "Rules first, hosted model second (flag-gated). Outputs are illustrative and unvalidated; "
    "no clinical claim is made."
)


def create_app(settings: Settings | None = None, repository: Repository | None = None,
               llm: LLMExtractor | None = None, persist: bool = True) -> FastAPI:
    settings = settings or get_settings()
    configure_telemetry("kairos-extract")
    log = get_logger("kairos.extract")
    llm = llm or LLMExtractor(settings)
    app = FastAPI(title="KAIROS extract", version=__version__, description=DESCRIPTION)
    state = {"repo": repository, "repo_failed": False}

    def repo() -> Repository | None:
        if state["repo"] is None and persist and not state["repo_failed"]:
            try:
                state["repo"] = get_repository(settings)
            except Exception as ex:  # noqa: BLE001 - the service works without the database
                state["repo_failed"] = True
                log.warning("database unavailable (%s); passports are not persisted", type(ex).__name__)
        return state["repo"]

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        return HealthResponse(service="kairos-extract", model_version=settings.model_version(),
                              allow_real_notes_to_llm=settings.allow_real_notes_to_llm,
                              llm_configured=llm.available)

    @app.post("/extract", response_model=ExtractResponse,
              responses={403: {"description": "real note text refused while ALLOW_REAL_NOTES_TO_LLM is off"},
                         503: {"description": "hosted-model path requested but not configured"}})
    def extract(req: ExtractRequest, store: bool = Query(True, description="persist extracted fields")) -> ExtractResponse:
        rules = extract_rules(req.text, req.note_ref, req.note_type, req.date, req.passport_id)
        result, methods, llm_used = rules, ["rule"], False
        warnings = list(rules.warnings)
        if req.method in ("auto", "llm"):
            if not llm.available:
                if req.method == "llm":
                    raise HTTPException(status_code=503, detail="hosted-model path is not configured")
                warnings.append("hosted-model path not configured; rules only")
            else:
                try:
                    llm.guard(req.source_kind)
                except RealNotesNotPermitted as ex:
                    raise HTTPException(status_code=403, detail=str(ex)) from ex
                run = rules.prescreen.mentions_prosthesis  # a rules match is never vetoed
                if not run and llm.prescreen_available:
                    try:
                        run = bool(llm.prescreen(req.text, req.source_kind))
                    except Exception as ex:  # noqa: BLE001
                        warnings.append(f"pre-screen call failed ({type(ex).__name__}); treated as mention")
                        run = True
                if not run and req.method == "llm":
                    run = True
                if run:
                    try:
                        out = llm.extract(req.text, req.source_kind)
                        result = merge(rules, out, req.text, req.date)
                        methods.append("llm")
                        llm_used = True
                        warnings = list(result.warnings)
                    except RealNotesNotPermitted as ex:
                        raise HTTPException(status_code=403, detail=str(ex)) from ex
                    except LLMNotConfigured as ex:
                        raise HTTPException(status_code=503, detail=str(ex)) from ex
                    except Exception as ex:  # noqa: BLE001 - rules result still stands
                        log.warning("hosted model call failed: %s", describe_error(ex))
                        warnings.append(f"hosted model call failed ({describe_error(ex)}); rules only")
                else:
                    warnings.append("pre-screen found no aortic valve prosthesis mention; hosted model skipped")
        r = repo()
        if r is not None and store:
            try:
                r.save_passport(result.passport, req.source_kind, methods, settings.model_version())
                r.save_echo_observations(result.echo_observations)
            except Exception as ex:  # noqa: BLE001
                log.warning("persisting the passport failed: %s", type(ex).__name__)
                warnings.append("passport not persisted (database error)")
        log.info("extract source=%s note_type=%s methods=%s llm_used=%s model=%s route=%s n_echo=%d",
                 req.source_kind, req.note_type, "+".join(methods), llm_used, result.passport.canonical_model,
                 result.passport.route, len(result.echo_observations))
        return ExtractResponse(passport=result.passport, echo_observations=result.echo_observations,
                               prescreen=result.prescreen, methods_used=methods, llm_used=llm_used,
                               source_kind=req.source_kind, warnings=warnings, label=ILLUSTRATIVE_LABEL)

    @app.post("/adjudicate/propose", response_model=AdjudicationProposal)
    def propose(req: AdjudicationRequest) -> AdjudicationProposal:
        p = req.passport
        points = to_points(req.echo_observations, prosthetic_only=True)
        ref = select_reference(points, parse_date(p.implant_date))
        episodes = req.exposure.episodes if req.exposure else []
        windows = thrombosis_windows_from_exposures(episodes, points, ref)
        decision = adjudicate(points, ref, p.events, windows)
        flags = candidate_flags(points, ref) if ref is not None else []
        proposal = AdjudicationProposal(
            passport_id=p.passport_id, endpoint_met=decision.met,
            endpoint_date=decision.date.isoformat() if decision.date else None, stage=decision.stage,
            mechanism=decision.mechanism, phenotype=decision.phenotype, confidence=decision.confidence,
            candidates=[CandidateFlagModel(date=f.date.isoformat(), stage=f.stage, phenotype=f.phenotype, reason=f.reason)
                        for f in flags],
            rationale=decision.reason, reference_source=decision.reference_source)
        if req.use_llm:
            if not llm.adjudication_available:
                raise HTTPException(status_code=503, detail="adjudication-assist deployment is not configured")
            try:
                llm.guard(req.source_kind)
                summary = _adjudication_summary(p, points, ref, flags, decision)
                out = llm.adjudicate(summary, req.source_kind)
                proposal.llm_proposal = out.model_dump()
            except RealNotesNotPermitted as ex:
                raise HTTPException(status_code=403, detail=str(ex)) from ex
            except Exception as ex:  # noqa: BLE001
                log.warning("adjudication-assist call failed: %s", describe_error(ex))
                proposal.llm_proposal = {"error": f"call failed ({describe_error(ex)})"}
        return proposal

    @app.get("/passports")
    def list_passports(limit: int = 50) -> list[dict]:
        r = repo()
        if r is None:
            raise HTTPException(status_code=503, detail="database not available")
        return r.list_passports(limit)

    @app.get("/passports/{passport_id}")
    def get_passport(passport_id: str) -> dict:
        r = repo()
        if r is None:
            raise HTTPException(status_code=503, detail="database not available")
        p = r.get_passport(passport_id)
        if p is None:
            raise HTTPException(status_code=404, detail="passport not found")
        return {"passport": p.model_dump(), "echo_observations": [o.model_dump() for o in r.get_echo_observations(passport_id)],
                "exposure": r.get_exposures(passport_id).model_dump(by_alias=True), "label": ILLUSTRATIVE_LABEL}

    return app


def _adjudication_summary(p, points, ref, flags, decision) -> str:
    """Numbers only: staged observations, never note text."""
    lines = [f"Route {p.route}; design class {p.design_class}; model {p.canonical_model}; size {p.size_mm} mm; implant {p.implant_date}."]
    if ref is not None:
        lines.append(f"Reference study {ref.date}: mean gradient {ref.mean_gradient} mmHg, EOA {ref.eoa} cm2, DVI {ref.dvi}, AR ordinal {ref.ar_ordinal}.")
    for pt in points:
        if ref is not None and pt.date <= ref.date:
            continue
        lines.append(f"Echo {pt.date}: mean gradient {pt.mean_gradient} mmHg, EOA {pt.eoa} cm2, DVI {pt.dvi}, AR ordinal {pt.ar_ordinal}, LVEF {pt.lvef}.")
    for f in flags:
        lines.append(f"VARC-3 candidate {f.date}: stage {f.stage}, phenotype {f.phenotype}.")
    if p.events:
        lines.append("Event mentions: " + ", ".join(f"{e.type} ({e.date})" for e in p.events) + ".")
    lines.append(f"Framework proposal: mechanism {decision.mechanism}, confidence {decision.confidence}: {decision.reason}")
    return "\n".join(lines)


app = create_app()
