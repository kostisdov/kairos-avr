"""Request parameters sent to the hosted model (GPT-5.1 reasoning model, or a non-reasoning fallback).

No network: a fake client records every call. These tests pin the non-negotiables of design
section 4 (the ``x-kairos-source`` header on every call, refusal of real text while the flag is
off) and the reasoning-model parameters (``reasoning_effort``, no temperature).
"""
from types import SimpleNamespace

import pytest
from openai import AzureOpenAI, OpenAI

from kairos.extraction.llm import (
    ADJUDICATE_MAX_COMPLETION_TOKENS,
    EXTRACT_MAX_COMPLETION_TOKENS,
    LLMAdjudication,
    LLMExtractor,
    LLMPassportExtraction,
    LLMPrescreen,
    RealNotesNotPermitted,
    build_client,
    describe_error,
    sampling_kwargs,
)
from kairos.io.config import Settings

EXTRACTION = LLMPassportExtraction(
    mentions_aortic_valve_prosthesis=True, route="SAVR", canonical_model="Trifecta", size_mm=21,
    implant_year=2016, events=[], prosthetic_echo=None, native_echo=None, evidence=[])
ADJUDICATION = LLMAdjudication(mechanism="svd", confidence="moderate", rationale="gradient rise with falling EOA")
PRESCREEN = LLMPrescreen(mentions_aortic_valve_prosthesis=True)


class FakeAPIError(Exception):
    status_code = 400
    code = "unsupported_parameter"
    param = "reasoning_effort"


class FakeCompletions:
    def __init__(self, outcome_by_format):
        self.calls = []
        self.outcome_by_format = outcome_by_format

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcome_by_format[kwargs["response_format"]]
        if isinstance(outcome, Exception):
            raise outcome
        usage = SimpleNamespace(completion_tokens_details=SimpleNamespace(reasoning_tokens=12))
        message = SimpleNamespace(parsed=outcome)
        return SimpleNamespace(model="gpt-5.1-2025-11-13", usage=usage, choices=[SimpleNamespace(message=message)])


def fake(outcome_by_format=None):
    outcomes = outcome_by_format or {LLMPassportExtraction: EXTRACTION, LLMAdjudication: ADJUDICATION, LLMPrescreen: PRESCREEN}
    completions = FakeCompletions(outcomes)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def settings(**env):
    values = {"KAIROS_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
              "KAIROS_OPENAI_ADJUDICATE_DEPLOYMENT": "kairos-adjudicate", "KAIROS_SQLITE_PATH": ":memory:"}
    values.update(env)
    return Settings(**values)


def test_sampling_kwargs_for_reasoning_and_non_reasoning_models():
    assert sampling_kwargs("low", 100) == {"max_completion_tokens": 100, "reasoning_effort": "low"}
    assert sampling_kwargs("", 100) == {"max_completion_tokens": 100, "temperature": 0}


def test_extract_sends_reasoning_parameters_and_source_header():
    client, calls = fake()
    out = LLMExtractor(settings(), client=client).extract("SYNTHETIC operative note", "synthetic")
    assert out.canonical_model == "Trifecta"
    call = calls.calls[0]
    assert call["model"] == "kairos-extract" and call["response_format"] is LLMPassportExtraction
    assert call["reasoning_effort"] == "low" and "temperature" not in call
    assert call["max_completion_tokens"] == EXTRACT_MAX_COMPLETION_TOKENS
    assert call["extra_headers"] == {"x-kairos-source": "synthetic"}
    assert "Allowed canonical model names" in call["messages"][0]["content"]


def test_non_reasoning_deployment_gets_temperature_zero():
    client, calls = fake()
    LLMExtractor(settings(KAIROS_OPENAI_EXTRACT_REASONING_EFFORT=""), client=client).extract("SYNTHETIC", "synthetic")
    assert calls.calls[0]["temperature"] == 0 and "reasoning_effort" not in calls.calls[0]


def test_real_text_is_refused_before_any_call_when_flag_is_off():
    client, calls = fake()
    llm = LLMExtractor(settings(ALLOW_REAL_NOTES_TO_LLM="false"), client=client)
    with pytest.raises(RealNotesNotPermitted):
        llm.extract("real note text", "real")
    with pytest.raises(RealNotesNotPermitted):
        llm.adjudicate("numbers only", "real")
    assert calls.calls == []


def test_real_text_carries_the_real_header_when_permitted():
    client, calls = fake()
    LLMExtractor(settings(ALLOW_REAL_NOTES_TO_LLM="true"), client=client).extract("real note text", "real")
    assert calls.calls[0]["extra_headers"] == {"x-kairos-source": "real"}


def test_adjudicate_uses_its_own_deployment_and_effort():
    client, calls = fake()
    out = LLMExtractor(settings(), client=client).adjudicate("Echo 2019: mean gradient 26 mmHg", "synthetic")
    assert out.mechanism == "svd"
    call = calls.calls[0]
    assert call["model"] == "kairos-adjudicate" and call["reasoning_effort"] == "medium"
    assert call["max_completion_tokens"] == ADJUDICATE_MAX_COMPLETION_TOKENS
    assert call["extra_headers"] == {"x-kairos-source": "synthetic"}


def test_self_check_reports_each_role():
    client, calls = fake({LLMPrescreen: PRESCREEN})
    results = {r["role"]: r for r in LLMExtractor(settings(), client=client).self_check()}
    assert results["extract"]["ok"] is True and results["adjudicate"]["ok"] is True
    assert results["prescreen"]["ok"] is None  # no pre-screen deployment configured
    assert results["extract"]["reasoning_tokens"] == 12
    assert all(c["extra_headers"] == {"x-kairos-source": "synthetic"} for c in calls.calls)
    broken, _ = fake({LLMPrescreen: FakeAPIError("bad request")})
    failed = {r["role"]: r for r in LLMExtractor(settings(), client=broken).self_check()}
    assert failed["extract"]["ok"] is False and "param=reasoning_effort" in failed["extract"]["detail"]


def test_build_client_dated_version_and_v1_endpoint():
    s = settings(KAIROS_OPENAI_API_KEY="local-test-key")
    assert s.openai_api_version == "2025-04-01-preview"
    dated = build_client(s)
    assert isinstance(dated, AzureOpenAI)
    v1 = build_client(s, api_version="v1")
    assert type(v1) is OpenAI and str(v1.base_url).endswith("/openai/v1/")


def test_describe_error_has_no_message_text():
    detail = describe_error(FakeAPIError("the note said something private"))
    assert detail == "FakeAPIError status_code=400 code=unsupported_parameter param=reasoning_effort"
