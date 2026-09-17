"""Database layer: PostgreSQL Flexible Server in Azure (Entra token auth through the
managed identity), SQLite locally and in tests.

Tables: passports, echo_observations, exposure_episodes, predictions, runs. Extracted fields
and predictions are stored; note text never is (design rule 0.3). Payload columns hold the
validated JSON of the corresponding schema model.
"""
from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    event,
)

from kairos.extraction.schema import (
    EchoObservation,
    ExposureEpisode,
    ExposureTimeline,
    Passport,
    Prediction,
)
from kairos.io.config import Settings, azure_credential, get_settings

PG_TOKEN_SCOPE = "https://ossrdbms-aad.database.windows.net/.default"

metadata = MetaData()

passports = Table(
    "passports", metadata,
    Column("passport_id", String(36), primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("source_kind", String(16), nullable=False),
    Column("note_ref", String(120)),
    Column("note_type", String(16)),
    Column("note_date", String(10)),
    Column("route", String(16)),
    Column("canonical_model", String(64)),
    Column("design_class", String(96)),
    Column("generation", Text),
    Column("size_mm", Integer),
    Column("implant_date", String(10)),
    Column("market_status", String(16)),
    Column("extraction_methods", String(32)),
    Column("model_version", String(64)),
    Column("payload", JSON, nullable=False),
)

echo_observations = Table(
    "echo_observations", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("passport_id", String(36), sa.ForeignKey("passports.passport_id"), nullable=False, index=True),
    Column("date", String(10), nullable=False),
    Column("mean_gradient_mmhg", Float),
    Column("peak_velocity_ms", Float),
    Column("eoa_cm2", Float),
    Column("dvi", Float),
    Column("ar_grade", String(16)),
    Column("lvef_pct", Float),
    Column("svi_ml_m2", Float),
    Column("native_vs_prosthetic", String(16)),
    Column("source", String(16)),
    Column("payload", JSON, nullable=False),
)

exposure_episodes = Table(
    "exposure_episodes", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("passport_id", String(36), sa.ForeignKey("passports.passport_id"), nullable=False, index=True),
    Column("class", String(8), nullable=False),
    Column("agent", String(64)),
    Column("indication", String(32)),
    Column("start", String(10), nullable=False),
    Column("stop", String(10)),
    Column("source", String(24)),
    Column("post_suspicion", Boolean, default=False),
    Column("payload", JSON, nullable=False),
)

predictions = Table(
    "predictions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("passport_id", String(36), sa.ForeignKey("passports.passport_id"), nullable=False, index=True),
    Column("prediction_time", String(10), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("source_kind", String(16)),
    Column("model_version", String(64)),
    Column("scenario_set", String(160)),
    Column("p_svd_12m", Float),
    Column("payload", JSON, nullable=False),
)

runs = Table(
    "runs", metadata,
    Column("run_id", String(64), primary_key=True),
    Column("kind", String(32), nullable=False),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("finished_at", DateTime(timezone=True)),
    Column("status", String(16), nullable=False),
    Column("manifest", JSON),
)


def _now() -> datetime:
    return datetime.now(UTC)


class _TokenCache:
    def __init__(self, credential):
        self.credential = credential
        self.token = None
        self.expires = 0

    def get(self) -> str:
        if self.token is None or self.expires - time.time() < 300:
            t = self.credential.get_token(PG_TOKEN_SCOPE)
            self.token, self.expires = t.token, t.expires_on
        return self.token


def make_engine(settings: Settings | None = None) -> sa.Engine:
    s = settings or get_settings()
    if s.pg_host:
        url = sa.URL.create("postgresql+psycopg", username=s.pg_user or None, host=s.pg_host,
                            port=5432, database=s.pg_db, query={"sslmode": "require"})
        engine = sa.create_engine(url, pool_pre_ping=True, pool_recycle=1800)
        if s.pg_auth == "entra":
            cache = _TokenCache(azure_credential(s))

            @event.listens_for(engine, "do_connect")
            def _inject_token(dialect, conn_rec, cargs, cparams):  # noqa: ANN001
                cparams["password"] = cache.get()
        else:
            @event.listens_for(engine, "do_connect")
            def _inject_password(dialect, conn_rec, cargs, cparams):  # noqa: ANN001
                cparams["password"] = s.pg_password
        return engine
    if s.sqlite_path == ":memory:":
        return sa.create_engine("sqlite://", connect_args={"check_same_thread": False},
                                poolclass=sa.pool.StaticPool)
    path = Path(s.sqlite_path) if s.sqlite_path else s.artifacts_dir / "kairos.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return sa.create_engine(f"sqlite:///{path.as_posix()}")


def init_db(engine: sa.Engine) -> None:
    metadata.create_all(engine)


class Repository:
    def __init__(self, engine: sa.Engine):
        self.engine = engine
        init_db(engine)

    # passports -------------------------------------------------------------------------
    def save_passport(self, p: Passport, source_kind: str, methods: list[str], model_version: str) -> None:
        row = dict(passport_id=p.passport_id, created_at=_now(), source_kind=source_kind,
                   note_ref=p.source.note_ref, note_type=p.source.note_type, note_date=p.source.date,
                   route=p.route, canonical_model=p.canonical_model, design_class=p.design_class,
                   generation=p.generation, size_mm=p.size_mm, implant_date=p.implant_date,
                   market_status=p.market_status, extraction_methods="+".join(methods),
                   model_version=model_version, payload=json.loads(p.model_dump_json()))
        with self.engine.begin() as c:
            existing = c.execute(sa.select(passports.c.passport_id).where(passports.c.passport_id == p.passport_id)).first()
            if existing:
                c.execute(passports.update().where(passports.c.passport_id == p.passport_id).values(**row))
            else:
                c.execute(passports.insert().values(**row))

    def get_passport(self, passport_id: str) -> Passport | None:
        with self.engine.connect() as c:
            r = c.execute(sa.select(passports.c.payload).where(passports.c.passport_id == passport_id)).first()
        return Passport.model_validate(r[0]) if r else None

    def list_passports(self, limit: int = 100) -> list[dict]:
        with self.engine.connect() as c:
            rows = c.execute(sa.select(passports.c.passport_id, passports.c.created_at, passports.c.source_kind,
                                       passports.c.route, passports.c.canonical_model, passports.c.size_mm,
                                       passports.c.implant_date)
                             .order_by(passports.c.created_at.desc()).limit(limit)).mappings().all()
        return [dict(r) for r in rows]

    # echo observations -------------------------------------------------------------------
    def save_echo_observations(self, obs: list[EchoObservation]) -> int:
        if not obs:
            return 0
        rows = [dict(passport_id=o.passport_id, date=o.date, mean_gradient_mmhg=o.mean_gradient_mmhg,
                     peak_velocity_ms=o.peak_velocity_ms, eoa_cm2=o.eoa_cm2, dvi=o.dvi, ar_grade=o.ar_grade,
                     lvef_pct=o.lvef_pct, svi_ml_m2=o.svi_ml_m2, native_vs_prosthetic=o.native_vs_prosthetic,
                     source=o.source, payload=json.loads(o.model_dump_json())) for o in obs]
        with self.engine.begin() as c:
            c.execute(echo_observations.insert(), rows)
        return len(rows)

    def get_echo_observations(self, passport_id: str) -> list[EchoObservation]:
        with self.engine.connect() as c:
            rows = c.execute(sa.select(echo_observations.c.payload)
                             .where(echo_observations.c.passport_id == passport_id)
                             .order_by(echo_observations.c.date)).all()
        return [EchoObservation.model_validate(r[0]) for r in rows]

    # exposures -----------------------------------------------------------------------
    def save_exposures(self, tl: ExposureTimeline) -> int:
        with self.engine.begin() as c:
            c.execute(exposure_episodes.delete().where(exposure_episodes.c.passport_id == tl.passport_id))
            rows = [{"passport_id": tl.passport_id, "class": e.class_, "agent": e.agent,
                     "indication": e.indication, "start": e.start, "stop": e.stop, "source": e.source,
                     "post_suspicion": e.started_within_90d_after_suspicious_echo,
                     "payload": json.loads(e.model_dump_json(by_alias=True))} for e in tl.episodes]
            if rows:
                c.execute(exposure_episodes.insert(), rows)
        return len(tl.episodes)

    def get_exposures(self, passport_id: str) -> ExposureTimeline:
        with self.engine.connect() as c:
            rows = c.execute(sa.select(exposure_episodes.c.payload)
                             .where(exposure_episodes.c.passport_id == passport_id)
                             .order_by(exposure_episodes.c.start)).all()
        return ExposureTimeline(passport_id=passport_id,
                                episodes=[ExposureEpisode.model_validate(r[0]) for r in rows])

    # predictions ---------------------------------------------------------------------
    def save_prediction(self, p: Prediction, source_kind: str) -> None:
        with self.engine.begin() as c:
            c.execute(predictions.insert().values(
                passport_id=p.passport_id, prediction_time=p.prediction_time, created_at=_now(),
                source_kind=source_kind, model_version=p.model_version, scenario_set=p.scenario_set,
                p_svd_12m=p.p_svd_12m, payload=json.loads(p.model_dump_json())))

    def list_predictions(self, passport_id: str) -> list[Prediction]:
        with self.engine.connect() as c:
            rows = c.execute(sa.select(predictions.c.payload)
                             .where(predictions.c.passport_id == passport_id)
                             .order_by(predictions.c.prediction_time, predictions.c.id)).all()
        return [Prediction.model_validate(r[0]) for r in rows]

    # runs ----------------------------------------------------------------------------
    def start_run(self, run_id: str, kind: str, manifest: dict | None = None) -> None:
        with self.engine.begin() as c:
            c.execute(runs.insert().values(run_id=run_id, kind=kind, started_at=_now(), status="running",
                                           manifest=manifest or {}))

    def finish_run(self, run_id: str, status: str = "succeeded", manifest: dict | None = None) -> None:
        with self.engine.begin() as c:
            values = dict(finished_at=_now(), status=status)
            if manifest is not None:
                values["manifest"] = manifest
            c.execute(runs.update().where(runs.c.run_id == run_id).values(**values))

    def list_runs(self, limit: int = 50) -> list[dict]:
        with self.engine.connect() as c:
            rows = c.execute(sa.select(runs).order_by(runs.c.started_at.desc()).limit(limit)).mappings().all()
        return [dict(r) for r in rows]


def get_repository(settings: Settings | None = None) -> Repository:
    return Repository(make_engine(settings))
