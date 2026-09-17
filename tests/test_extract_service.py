"""Contract tests for kairos-extract, including the real-notes flag (milestone M2)."""
import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kairos import ILLUSTRATIVE_LABEL
from kairos.extraction.llm import LLMEcho, LLMEvidence, LLMExtractor, LLMPassportExtraction
from kairos.io.config import Settings

FIX = Path(__file__).parent / "fixtures" / "synthetic_notes"
ROOT = Path(__file__).resolve().parents[1]


def _load_app_module():
    spec = importlib.util.spec_from_file_location("extract_app", ROOT / "services" / "extract" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # pydantic resolves forward references through sys.modules
    spec.loader.exec_module(mod)
    return mod


def _client(**env):
    mod = _load_app_module()
    kwargs = {"KAIROS_SQLITE_PATH": ":memory:", "KAIROS_OPENAI_ENDPOINT": ""}
    kwargs.update(env)
    settings = Settings(**kwargs)
    return TestClient(mod.create_app(settings=settings, persist=True)), mod


def _payload(name="02_savr_op_trifecta_hashsize.txt", **over):
    body = {"text": (FIX / name).read_text(encoding="utf-8"), "source_kind": "synthetic", "note_ref": name,
            "note_type": "operative", "date": "2016", "method": "auto"}
    body.update(over)
    return body


def test_health_and_rules_extraction():
    client, _ = _client()
    h = client.get("/healthz").json()
    assert h["service"] == "kairos-extract" and h["label"] == ILLUSTRATIVE_LABEL and h["llm_configured"] is False
    r = client.post("/extract", json=_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["passport"]["canonical_model"] == "Trifecta" and body["passport"]["size_mm"] == 21
    assert body["methods_used"] == ["rule"] and body["llm_used"] is False
    assert any("not configured" in w for w in body["warnings"])
    # persisted fields, never text
    listed = client.get("/passports").json()
    assert listed and listed[0]["canonical_model"] == "Trifecta"
    got = client.get(f"/passports/{body['passport']['passport_id']}").json()
    assert "text" not in got["passport"] and got["passport"]["route"] == "SAVR"


def test_real_text_is_refused_when_flag_is_off():
    client, _ = _client(KAIROS_OPENAI_ENDPOINT="https://example.openai.azure.com/", ALLOW_REAL_NOTES_TO_LLM="false")
    r = client.post("/extract", json=_payload(source_kind="real"))
    assert r.status_code == 403 and "ALLOW_REAL_NOTES_TO_LLM" in r.json()["detail"]
    r = client.post("/extract", json=_payload(source_kind="real", method="llm"))
    assert r.status_code == 403
    # the rules path never leaves the service and stays available
    r = client.post("/extract", json=_payload(source_kind="real", method="rules"))
    assert r.status_code == 200 and r.json()["llm_used"] is False


def test_llm_path_merges_and_respects_rules(monkeypatch):
    client, mod = _client(KAIROS_OPENAI_ENDPOINT="https://example.openai.azure.com/", ALLOW_REAL_NOTES_TO_LLM="true")
    canned = LLMPassportExtraction(
        mentions_aortic_valve_prosthesis=True, route="SAVR", canonical_model="Magna", size_mm=25, implant_year=2016,
        events=["endocarditis"], prosthetic_echo=LLMEcho(mean_gradient_mmhg=8, peak_velocity_ms=None, eoa_cm2=1.9, dvi=None,
                                                          ar_grade=None, lvef_pct=60, svi_ml_m2=None),
        native_echo=None, evidence=[LLMEvidence(field="canonical_model", quote="Trifecta"), LLMEvidence(field="size_mm", quote="not in note")])
    monkeypatch.setattr(LLMExtractor, "extract", lambda self, text, source_kind: canned)
    r = client.post("/extract", json=_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["llm_used"] is True and body["methods_used"] == ["rule", "llm"]
    assert body["passport"]["canonical_model"] == "Trifecta" and body["passport"]["size_mm"] == 21  # rules win
    assert any("disagree" in w for w in body["warnings"])
    assert any(e["method"] == "llm" for e in body["passport"]["evidence"])
    assert "endocarditis" in {e["type"] for e in body["passport"]["events"]}


def test_llm_requested_but_not_configured_is_503():
    client, _ = _client()
    r = client.post("/extract", json=_payload(method="llm"))
    assert r.status_code == 503


def test_adjudication_proposal_is_a_proposal():
    client, _ = _client()
    pid = "p1"
    payload = {"passport": {"passport_id": pid, "source": {"note_ref": "x", "note_type": "operative", "date": "2015-01-15"},
                            "route": "SAVR", "implant_date": "2015-01-15"},
               "echo_observations": [
                   {"passport_id": pid, "date": "2015-04-01", "mean_gradient_mmhg": 12, "eoa_cm2": 1.8, "dvi": 0.5, "native_vs_prosthetic": "prosthetic"},
                   {"passport_id": pid, "date": "2018-04-01", "mean_gradient_mmhg": 26, "eoa_cm2": 1.2, "dvi": 0.35, "native_vs_prosthetic": "prosthetic"},
                   {"passport_id": pid, "date": "2019-04-01", "mean_gradient_mmhg": 30, "eoa_cm2": 1.1, "dvi": 0.33, "native_vs_prosthetic": "prosthetic"}]}
    r = client.post("/adjudicate/propose", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "proposal" and body["endpoint_met"] is True and body["mechanism"] == "svd"
    assert body["candidates"][0]["stage"] == "2" and body["label"] == ILLUSTRATIVE_LABEL


@pytest.mark.parametrize("note_type", ["operative", "progress"])
def test_note_text_never_appears_in_response(note_type):
    client, _ = _client()
    body = client.post("/extract", json=_payload(note_type=note_type)).json()
    dumped = str(body)
    assert "median sternotomy was" not in dumped
