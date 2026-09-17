from kairos import ILLUSTRATIVE_LABEL
from kairos.demo_patient import PRESETS, request_from_form
from kairos.summary.patient_summary import (
    assemble_patient_summary,
    comparator_for_request,
    comparison_category,
    deterministic_summary,
    export_summary,
    snapshot_hash,
)


def prediction(flag=False, family="cox", version="v1"):
    return {"model_family": family, "model_version": version, "p_svd_12m": 0.07 if flag else 0.02,
            "messages": {"current_abnormality": False, "earlier_assessment": flag, "overdue_surveillance": False},
            "reliability": {"model_family": family, "reasons": []}}


def test_snapshot_hash_changes_with_input_and_provenance():
    req = request_from_form(next(iter(PRESETS.values()))())
    assert snapshot_hash(req, "synthetic") != snapshot_hash(req, "real")
    changed = req.model_copy(deep=True)
    changed.prediction_time = "2025-01-01"
    assert snapshot_hash(req, "synthetic") != snapshot_hash(changed, "synthetic")


def test_all_deterministic_comparison_categories_and_negative_wording():
    assert comparison_category(prediction(True), {"status": "negative"})[0] == "rule_negative_model_flagged"
    assert comparison_category(prediction(False), {"status": "positive"})[0] == "rule_positive_model_not_flagged"
    assert comparison_category(prediction(True), {"status": "positive"})[0] == "both_flagged"
    code, text = comparison_category(prediction(False), {"status": "negative"})
    assert code == "neither_flagged" and "not a declaration" in text
    assert comparison_category(None, {"status": "negative"})[0] == "comparison_incomplete_model_unavailable"


def test_summary_facts_fallback_and_export_retain_provenance():
    req = request_from_form(next(iter(PRESETS.values()))())
    comp = comparator_for_request(req)
    facts = assemble_patient_summary(req, prediction(True), comp, selected_family="cox", source_provenance="synthetic")
    assert facts.returned_family == "cox" and facts.model_version == "v1"
    assert any(f.id == "model.risk_12m" and f.value == 0.07 for f in facts.facts)
    fallback = deterministic_summary(facts)
    assert "Template summary" in fallback and "normal" not in fallback.lower()
    exported = export_summary(facts, fallback)
    assert facts.snapshot_id in exported and ILLUSTRATIVE_LABEL in exported and comp.comparator_id in exported
