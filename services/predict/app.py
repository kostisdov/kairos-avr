"""kairos-predict: passport plus echo history (plus optional module inputs) in; the three
probabilities at 1, 3 and 5 years and the 12-month risk out, with drivers, a reliability
statement and the three separated messages (design section 3.4, milestone M4).

Every response carries the label "illustrative, unvalidated: model trained on synthetic
scenarios". A passport that already meets the endpoint returns HTTP 409 with an
explanation instead of a prediction. Every error ``detail`` is an ``ErrorDetail`` object
``{code, message, detail}`` (detailed design WP-D1).

The service loads every model family with an active pointer in its namespace (WP-D3); the
default family comes from ``KAIROS_MODEL_FAMILY``, else ``deployed_model.family`` in
``model.yaml``, else ``cox``. Other families are served only when compatible with the default.
"""
from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from kairos import HORIZONS_YEARS, ILLUSTRATIVE_LABEL, __version__
from kairos.extraction.schema import (
    ErrorDetail,
    HealthResponse,
    Prediction,
    PredictRequest,
    parse_date,
)
from kairos.io.config import Settings, get_settings
from kairos.io.db import Repository, get_repository
from kairos.io.storage import get_store
from kairos.modelling.modules import load_model_config
from kairos.modelling.predictor import (
    EndpointMetError,
    ModelBundle,
    NoReferenceEchoError,
    PredictionTimeError,
    Predictor,
    UnsupportedFitError,
    UnsupportedRouteError,
)
from kairos.modelling.store import active_families, load_active_bundle
from kairos.telemetry import configure_telemetry, get_logger

DESCRIPTION = (
    "KAIROS research prototype: dynamic, competing-risk prediction of structural valve deterioration. "
    "Trained on synthetic scenarios; outputs are illustrative and unvalidated and never a treatment recommendation."
)


class PredictHealth(HealthResponse):
    model_loaded: bool = False
    scenario_set: str | None = None
    ladder_step: str | None = None
    load_error: str | None = None
    default_family: str | None = None
    families: list[str] = Field(default_factory=list)


class TrajectoryItem(BaseModel):
    prediction_time: str
    prediction: Prediction | None = None
    error: str | None = None
    error_detail: ErrorDetail | None = None
    status: int = 200


class TrajectoryResponse(BaseModel):
    passport_id: str
    family: str | None = None
    items: list[TrajectoryItem] = Field(default_factory=list)
    label: str = ILLUSTRATIVE_LABEL


def err(status: int, code: str, message: str, **detail) -> HTTPException:
    return HTTPException(status_code=status, detail=ErrorDetail(code=code, message=message, detail=detail).model_dump())


def load_bundle_from_store(settings: Settings) -> ModelBundle:
    store = get_store(settings)
    path = f"{settings.model_blob_prefix}/bundle.joblib"
    if not store.exists("models", path):
        raise FileNotFoundError(f"models/{path} not found in the artefact store")
    local = Path(tempfile.gettempdir()) / "kairos_predict_bundle.joblib"
    store.get_file("models", path, local)
    return ModelBundle.load(local)


def default_family_name(cfg: dict) -> str:
    env = os.environ.get("KAIROS_MODEL_FAMILY")
    if env:
        return env
    dm = cfg.get("deployed_model") if isinstance(cfg, dict) else None
    if isinstance(dm, dict) and dm.get("family"):
        return str(dm["family"])
    return "cox"


def _horizons(b: ModelBundle) -> list:
    return list(getattr(b, "horizons_years", None) or HORIZONS_YEARS)


def _endpoint_version(b: ModelBundle):
    ds = (b.training_summary or {}).get("dataset", {}) if isinstance(b.training_summary, dict) else {}
    return (ds or {}).get("endpoint_version")


def compatible(b: ModelBundle, ref: ModelBundle) -> bool:
    return (b.integration_version == ref.integration_version and _horizons(b) == _horizons(ref)
            and _endpoint_version(b) == _endpoint_version(ref))


def _classify(ex: Exception) -> tuple[int, str]:
    if isinstance(ex, EndpointMetError):
        return 409, "endpoint_met"
    if isinstance(ex, NoReferenceEchoError):
        return 400, "no_reference_echo"
    if isinstance(ex, PredictionTimeError):
        return 400, "prediction_time_invalid"
    if isinstance(ex, UnsupportedRouteError):
        route = getattr(ex, "route", None)
        missing = route is None or str(route).strip().lower() in ("", "nan", "none", "unknown")
        return 422, ("unknown_device_or_route" if missing else "unsupported_route")
    if isinstance(ex, UnsupportedFitError):
        return 422, "unsupported_fit"
    raise ex


HANDLED = (EndpointMetError, NoReferenceEchoError, PredictionTimeError, UnsupportedRouteError, UnsupportedFitError)


def create_app(settings: Settings | None = None, bundle: ModelBundle | None = None,
               repository: Repository | None = None, persist: bool = True,
               bundles: dict[str, ModelBundle] | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_telemetry("kairos-predict")
    log = get_logger("kairos.predict")
    cfg = load_model_config()
    namespace = os.environ.get("KAIROS_MODEL_NAMESPACE", "full")
    state = {"bundles": {}, "predictors": {}, "default": default_family_name(cfg), "load_error": None,
             "repo": repository, "repo_failed": False}

    def install(bs: dict[str, ModelBundle], preferred: str | None = None) -> None:
        state["bundles"] = dict(bs)
        state["predictors"] = {f: Predictor(b, cfg) for f, b in bs.items()}
        wanted = default_family_name(cfg)
        if wanted in bs:
            state["default"] = wanted
        elif preferred in bs:
            state["default"] = preferred
        elif bs:
            state["default"] = next(iter(bs))

    initial: dict[str, ModelBundle] = dict(bundles or {})
    if bundle is not None:
        initial.setdefault(bundle.family or "cox", bundle)
    if initial:
        install(initial, preferred=(bundle.family if bundle is not None else None))

    def load() -> None:
        try:
            store = get_store(settings)
            loaded: dict[str, ModelBundle] = {}
            errors = []
            for fam in active_families(store, namespace):
                try:
                    b = load_active_bundle(store, namespace, fam)
                    if b is not None:
                        loaded[fam] = b
                except Exception as ex:  # noqa: BLE001
                    errors.append(f"{fam}: {type(ex).__name__}: {ex}")
            if not loaded:
                b = load_bundle_from_store(settings)
                loaded[b.family or "cox"] = b
            install(loaded)
            state["load_error"] = "; ".join(errors)[:300] or None
            log.info("model bundles loaded: %s (default %s)", sorted(loaded), state["default"])
        except Exception as ex:  # noqa: BLE001
            state["load_error"] = f"{type(ex).__name__}: {ex}"[:300]
            log.warning("model bundle not loaded: %s", state["load_error"])

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if not state["bundles"]:
            load()
        yield

    app = FastAPI(title="KAIROS predict", version=__version__, description=DESCRIPTION, lifespan=lifespan)

    def repo() -> Repository | None:
        if state["repo"] is None and persist and not state["repo_failed"]:
            try:
                state["repo"] = get_repository(settings)
            except Exception as ex:  # noqa: BLE001
                state["repo_failed"] = True
                log.warning("database unavailable (%s); predictions are not persisted", type(ex).__name__)
        return state["repo"]

    def default_bundle() -> ModelBundle | None:
        return state["bundles"].get(state["default"])

    def resolve(family: str | None) -> tuple[str, Predictor]:
        if not state["predictors"]:
            raise err(503, "no_model_bundle", f"no model bundle loaded ({state['load_error'] or 'not attempted'}); run the train job")
        fam = family or state["default"]
        ref = default_bundle()
        b = state["bundles"].get(fam)
        if b is None or ref is None or not compatible(b, ref):
            raise err(422, "unsupported_family",
                      f"model family '{fam}' is not loaded or not compatible with the default family '{state['default']}'",
                      family=fam, loaded=sorted(state["bundles"]), default=state["default"])
        return fam, state["predictors"][fam]

    def _predict(req: PredictRequest, family: str | None) -> Prediction:
        _, pred = resolve(family)
        try:
            return pred.predict(req)
        except HANDLED as ex:
            status, code = _classify(ex)
            raise err(status, code, str(ex)) from ex

    @app.get("/healthz", response_model=PredictHealth)
    def healthz() -> PredictHealth:
        b = default_bundle()
        return PredictHealth(service="kairos-predict", model_version=b.model_version if b else settings.model_version(),
                             model_loaded=b is not None, scenario_set=b.scenario_set if b else None,
                             ladder_step=b.ladder_step if b else None, load_error=state["load_error"],
                             default_family=state["default"] if b else None, families=sorted(state["bundles"]))

    @app.post("/predict", response_model=Prediction,
              responses={409: {"description": "passport already meets the endpoint (code endpoint_met)"},
                         400: {"description": "no_reference_echo or prediction_time_invalid"},
                         422: {"description": "unknown_device_or_route, unsupported_route, unsupported_fit or unsupported_family"},
                         503: {"description": "no_model_bundle"}})
    def predict(req: PredictRequest, family: str | None = Query(None)) -> Prediction:
        pred = _predict(req, family)
        r = repo()
        if r is not None:
            try:
                r.save_prediction(pred, req.source_kind)
            except Exception as ex:  # noqa: BLE001
                log.warning("persisting the prediction failed: %s", type(ex).__name__)
        log.info("predict family=%s source=%s p_svd_12m=%.4f drivers=%d", pred.model_family, req.source_kind,
                 pred.p_svd_12m, len(pred.drivers))
        return pred

    @app.post("/predict/trajectory", response_model=TrajectoryResponse)
    def trajectory(req: PredictRequest, family: str | None = Query(None)) -> TrajectoryResponse:
        """One prediction per echo date from time zero to the requested prediction time."""
        fam, pred = resolve(family)
        end = parse_date(req.prediction_time)
        dates = sorted({o.date for o in req.echo_observations
                        if o.native_vs_prosthetic == "prosthetic" and parse_date(o.date) is not None
                        and parse_date(o.date) <= end})
        if req.prediction_time not in dates:
            dates.append(req.prediction_time)
        items = []
        for d in dates:
            sub = req.model_copy(update={"prediction_time": d})
            try:
                items.append(TrajectoryItem(prediction_time=d, prediction=pred.predict(sub)))
            except HANDLED as ex:
                status, code = _classify(ex)
                items.append(TrajectoryItem(prediction_time=d, error=str(ex), status=status,
                                            error_detail=ErrorDetail(code=code, message=str(ex))))
        return TrajectoryResponse(passport_id=req.passport.passport_id, family=fam, items=items)

    @app.get("/models")
    def models() -> dict:
        ref = default_bundle()
        out = []
        for fam, b in sorted(state["bundles"].items()):
            try:
                support = b.model.support_summary()
            except Exception:  # noqa: BLE001
                support = None
            out.append({"family": fam, "model_version": b.model_version, "ladder_step": b.ladder_step,
                        "scenario_set": b.scenario_set, "integration_version": b.integration_version,
                        "schema_version": b.schema_version, "support_mode": b.support_mode, "support": support,
                        "default": fam == state["default"],
                        "compatible_with_default": bool(ref is not None and compatible(b, ref))})
        return {"default_family": state["default"] if out else None, "namespace": namespace, "models": out,
                "label": ILLUSTRATIVE_LABEL}

    @app.get("/model")
    def model_card(family: str | None = Query(None)) -> dict:
        if not state["bundles"]:
            raise err(503, "no_model_bundle", "no model bundle loaded")
        fam = family or state["default"]
        b = state["bundles"].get(fam)
        if b is None:
            raise err(422, "unsupported_family", f"model family '{fam}' is not loaded", family=fam,
                      loaded=sorted(state["bundles"]))
        return b.card()

    @app.get("/guideline/{jurisdiction}")
    def guideline(jurisdiction: str) -> dict:
        g = cfg["guideline_surveillance"]
        if jurisdiction not in g:
            raise err(404, "unknown_jurisdiction", f"unknown jurisdiction; known: {sorted(g)}", known=sorted(g))
        return {"jurisdiction": jurisdiction, **g[jurisdiction], "note": "displayed unchanged; the model never modifies guideline surveillance"}

    @app.post("/reload")
    def reload() -> dict:
        load()
        return {"model_loaded": bool(state["bundles"]), "families": sorted(state["bundles"]),
                "default_family": state["default"], "load_error": state["load_error"]}

    return app


app = create_app()
