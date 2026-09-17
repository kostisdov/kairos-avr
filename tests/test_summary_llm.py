from types import SimpleNamespace

import pytest

from kairos.demo_patient import PRESETS, request_from_form
from kairos.extraction.llm import RealNotesNotPermitted
from kairos.io.config import Settings
from kairos.summary.llm import InvalidNarrative, PatientSummaryWriter
from kairos.summary.patient_summary import assemble_patient_summary
from kairos.summary.schema import ClinicianFindings, FindingItem


class FakeCompletions:
    def __init__(self, parsed):
        self.parsed = parsed
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(parsed=self.parsed, refusal=None)
        return SimpleNamespace(model="gpt-test", usage=None, choices=[SimpleNamespace(message=message)])


def settings(**values):
    base = {"KAIROS_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
            "KAIROS_OPENAI_SUMMARY_DEPLOYMENT": "summary-deployment",
            "KAIROS_OPENAI_SUMMARY_REASONING_EFFORT": "low"}
    base.update(values)
    return Settings(**base)


def facts(source="synthetic"):
    req = request_from_form(next(iter(PRESETS.values()))())
    pred = {"model_family": "cox", "model_version": "v1", "p_svd_12m": 0.07,
            "messages": {"current_abnormality": False, "earlier_assessment": True, "overdue_surveillance": False},
            "reliability": {"model_family": "cox", "reasons": []}}
    return assemble_patient_summary(req, pred, selected_family="cox", source_provenance=source)


def valid_draft(summary):
    return ClinicianFindings(
        findings=[FindingItem(text="The selected family is Cox.", evidence_ids=["model.family"])],
        interpretation=[FindingItem(text="The supplied comparison category is retained.", evidence_ids=["comparison.category"])],
        recommendations=[],
        limitations=[FindingItem(text="The result is illustrative and unvalidated.", evidence_ids=["limitation.validation"])],
    )


def test_writer_uses_explicit_deployment_schema_guard_and_one_call():
    summary = facts()
    calls = FakeCompletions(valid_draft(summary))
    client = SimpleNamespace(chat=SimpleNamespace(completions=calls))
    draft, metadata = PatientSummaryWriter(settings(), client).write(summary)
    assert draft.findings and metadata.deployment == "summary-deployment"
    assert len(calls.calls) == 1
    call = calls.calls[0]
    assert call["response_format"] is ClinicianFindings and call["extra_headers"] == {"x-kairos-source": "synthetic"}
    assert call["reasoning_effort"] == "low" and call["max_completion_tokens"] == 4000


def test_real_derived_facts_are_refused_before_client_call():
    summary = facts("real")
    calls = FakeCompletions(valid_draft(summary))
    client = SimpleNamespace(chat=SimpleNamespace(completions=calls))
    with pytest.raises(RealNotesNotPermitted):
        PatientSummaryWriter(settings(ALLOW_REAL_NOTES_TO_LLM=False), client).write(summary)
    assert calls.calls == []


def test_fabricated_number_and_unsafe_markup_are_rejected():
    summary = facts()
    bad = valid_draft(summary)
    bad.findings[0].text = "Risk is 99.9%."
    calls = FakeCompletions(bad)
    with pytest.raises(InvalidNarrative):
        PatientSummaryWriter(settings(), SimpleNamespace(chat=SimpleNamespace(completions=calls))).write(summary)
