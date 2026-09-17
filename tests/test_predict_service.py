"""Contract tests for kairos-predict (milestone M4)."""
import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from kairos import CONDITIONING, ILLUSTRATIVE_LABEL
from kairos.io.config import Settings
from tests.helpers import DETERIORATING, STABLE, cohort_request, passport, request

ROOT = Path(__file__).resolve().parents[1]


def _load_app_module():
    spec = importlib.util.spec_from_file_location("predict_app", ROOT / "services" / "predict" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # pydantic resolves forward references through sys.modules
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def client(small_bundle):
    mod = _load_app_module()
    settings = Settings(KAIROS_SQLITE_PATH=":memory:", KAIROS_OPENAI_ENDPOINT="")
    app = mod.create_app(settings=settings, bundle=small_bundle, persist=True)
    with TestClient(app) as c:
        yield c


def test_health_reports_model(client):
    h = client.get("/healthz").json()
    assert h["model_loaded"] is True and h["ladder_step"] == "core_plus_both" and h["label"] == ILLUSTRATIVE_LABEL


def test_predict_contract(client, small_cohort):
    req, _ = cohort_request(small_cohort, index=1)
    r = client.post("/predict", json=req.model_dump(by_alias=True))
    assert r.status_code == 200, r.text
    body = r.json()
    for key in ("p_svd_before_death", "p_death_before_svd", "p_alive_intact", "p_replaced_non_svd"):
        assert len(body[key]) == 3
    tot = np.sum([body[k] for k in ("p_svd_before_death", "p_death_before_svd", "p_alive_intact", "p_replaced_non_svd")], axis=0)
    assert np.allclose(tot, 1, atol=1e-4)
    assert body["horizons_years"] == [1, 3, 5] and body["conditioning"] == CONDITIONING
    assert body["reliability"]["label"] == ILLUSTRATIVE_LABEL and set(body["messages"]) == {"current_abnormality", "earlier_assessment", "overdue_surveillance"}
    assert body["model_version"] == "0.1.0+test" and body["scenario_set"].startswith("gradual_stenotic")


def test_409_on_endpoint_met(client):
    r = client.post("/predict", json=request(DETERIORATING, "2020-01-01").model_dump(by_alias=True))
    d = r.json()["detail"]
    assert r.status_code == 409 and d["code"] == "endpoint_met" and "endpoint" in d["message"] and "detail" in d


def test_400_without_reference_echo(client):
    r = client.post("/predict", json=request([("2018-01-01", 10, 1.8, 0.5)], "2019-01-01").model_dump(by_alias=True))
    assert r.status_code == 400 and r.json()["detail"]["code"] == "no_reference_echo"


def test_422_when_the_route_is_missing_never_a_default(client):
    req = request(STABLE, "2018-08-01", p=passport(route="unknown"))
    r = client.post("/predict", json=req.model_dump(by_alias=True))
    d = r.json()["detail"]
    assert r.status_code == 422 and d["code"] == "unknown_device_or_route" and "no route is substituted" in d["message"]


def test_legacy_bundle_is_rejected_at_load(tmp_path, small_bundle):
    import joblib

    from kairos.modelling.predictor import LegacyBundleError, ModelBundle

    old = copy.copy(small_bundle)
    old.integration_version = None
    joblib.dump(old, tmp_path / "old.joblib")
    with pytest.raises(LegacyBundleError, match="retrain"):
        ModelBundle.load(tmp_path / "old.joblib")
    v1 = copy.copy(small_bundle)
    v1.__dict__.pop("schema_version")          # a bundle written before the family adapters
    joblib.dump(v1, tmp_path / "v1.joblib")
    with pytest.raises(LegacyBundleError, match="schema"):
        ModelBundle.load(tmp_path / "v1.joblib")
    lib = copy.copy(small_bundle)
    lib.library_versions = {**small_bundle.library_versions, "scikit-learn": "0.1.0"}
    joblib.dump(lib, tmp_path / "lib.joblib")
    with pytest.raises(LegacyBundleError, match="scikit-learn"):
        ModelBundle.load(tmp_path / "lib.joblib")


def test_trajectory_and_guideline(client):
    r = client.post("/predict/trajectory", json=request(STABLE, "2018-08-01").model_dump(by_alias=True))
    assert r.status_code == 200
    items = r.json()["items"]
    assert [i["prediction_time"] for i in items] == ["2016-08-01", "2017-08-01", "2018-08-01"]
    assert all(i["prediction"] for i in items)
    g = client.get("/guideline/ACC_AHA").json()
    assert "unchanged" in g["note"] and client.get("/guideline/nowhere").status_code == 404
    card = client.get("/model").json()
    assert card["label"] == ILLUSTRATIVE_LABEL


def test_service_without_bundle_returns_503(monkeypatch):
    monkeypatch.setenv("KAIROS_MODEL_NAMESPACE", "test-namespace-without-pointers")
    mod = _load_app_module()
    settings = Settings(KAIROS_SQLITE_PATH=":memory:", KAIROS_MODEL_BLOB_PREFIX="does-not-exist", KAIROS_OPENAI_ENDPOINT="")
    with TestClient(mod.create_app(settings=settings, persist=False)) as c:
        assert c.get("/healthz").json()["model_loaded"] is False
        r = c.post("/predict", json=request(STABLE, "2018-08-01").model_dump(by_alias=True))
        assert r.status_code == 503 and r.json()["detail"]["code"] == "no_model_bundle"
        assert c.get("/models").json()["models"] == []


def test_prediction_carries_family_and_versions(client):
    body = client.post("/predict", json=request(STABLE, "2018-08-01").model_dump(by_alias=True)).json()
    assert body["model_family"] == "cox" and body["integration_version"] and body["bundle_schema_version"] >= 2
    codes = {x["code"] for x in body["reliability"]["reasons"]}
    assert {"abrupt_failure_not_reliably_anticipated", "phenotype_performance_unestablished"} <= codes
    assert body["reliability"]["model_family"] == "cox"


def test_family_parameter_and_models_listing(client):
    body = request(STABLE, "2018-08-01").model_dump(by_alias=True)
    assert client.post("/predict?family=cox", json=body).status_code == 200
    r = client.post("/predict?family=gradient_boosting", json=body)
    assert r.status_code == 422 and r.json()["detail"]["code"] == "unsupported_family"
    r = client.post("/predict/trajectory?family=nope", json=body)
    assert r.status_code == 422 and r.json()["detail"]["code"] == "unsupported_family"
    assert client.post("/predict/trajectory?family=cox", json=body).json()["family"] == "cox"
    m = client.get("/models").json()
    assert m["default_family"] == "cox" and [x["family"] for x in m["models"]] == ["cox"]
    assert m["models"][0]["model_version"] == "0.1.0+test" and m["models"][0]["ladder_step"] == "core_plus_both"
    assert client.get("/model?family=cox").status_code == 200
    assert client.get("/model?family=nope").json()["detail"]["code"] == "unsupported_family"
    assert client.get("/healthz").json()["families"] == ["cox"]


def test_second_compatible_family_is_served(small_bundle, monkeypatch):
    monkeypatch.delenv("KAIROS_MODEL_FAMILY", raising=False)
    mod = _load_app_module()
    other = copy.copy(small_bundle)
    other.family = "gradient_boosting"
    incompatible = copy.copy(small_bundle)
    incompatible.integration_version = "other-integration"
    settings = Settings(KAIROS_SQLITE_PATH=":memory:", KAIROS_OPENAI_ENDPOINT="")
    app = mod.create_app(settings=settings, bundles={"cox": small_bundle, "gradient_boosting": other, "weird": incompatible},
                         persist=False)
    body = request(STABLE, "2018-08-01").model_dump(by_alias=True)
    with TestClient(app) as c:
        assert c.post("/predict?family=gradient_boosting", json=body).status_code == 200
        r = c.post("/predict?family=weird", json=body)
        assert r.status_code == 422 and r.json()["detail"]["code"] == "unsupported_family"
        listed = {x["family"]: x["compatible_with_default"] for x in c.get("/models").json()["models"]}
        assert listed == {"cox": True, "gradient_boosting": True, "weird": False}
