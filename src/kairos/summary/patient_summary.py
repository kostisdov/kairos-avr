"""Deterministic Patient Summary assembly, comparison language and fallback export."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from typing import Any, Mapping

from kairos import ILLUSTRATIVE_LABEL
from kairos.adjudication.framework import select_reference, to_points
from kairos.comparators.varc3_hvd import ComparatorContext, ComparatorResult, evaluate_varc3_comparator
from kairos.extraction.schema import PredictRequest, parse_date
from kairos.summary.schema import AllowedAction, ClinicianFindings, PatientSummaryFacts, SummaryFact


def _dict(value: Any) -> dict:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return vars(value)


def snapshot_hash(payload: Any, source_provenance: str = "unknown") -> str:
    raw = _dict(payload)
    canonical = json.dumps({"payload": raw, "source_provenance": source_provenance}, sort_keys=True,
                           separators=(",", ":"), default=str, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def comparator_for_request(request: PredictRequest | Mapping[str, Any], *, source_kind: str | None = None,
                           freshness_days: int = 548) -> ComparatorResult:
    req = request if isinstance(request, PredictRequest) else PredictRequest.model_validate(request)
    prediction_date = parse_date(req.prediction_time)
    points = [p for p in to_points(req.echo_observations) if prediction_date is None or p.date <= prediction_date]
    ref = select_reference(points, parse_date(req.passport.implant_date))
    current = max(points, key=lambda p: (p.date, p.index), default=None)
    fresh = None if current is None or prediction_date is None else (prediction_date - current.date).days <= freshness_days
    source = source_kind or req.source_kind
    # Demo/synthetic observations explicitly define the AR field as prosthetic AR.  Other
    # structured sources require an upstream explicit location contract, which this schema lacks.
    ar_location = True if source == "synthetic" else None
    return evaluate_varc3_comparator(
        ref, current,
        ComparatorContext(
            index_valve_id=req.passport.passport_id,
            prediction_date=prediction_date,
            reference_study_id=f"echo:{ref.index}" if ref is not None else None,
            current_study_id=f"echo:{current.index}" if current is not None else None,
            source_provenance=source,
            intraprosthetic_ar_confirmed=ar_location,
            fresh=fresh,
            freshness_reason="current study is older than 18 months" if fresh is False else None,
        ),
    )


def comparison_category(model: Mapping[str, Any] | None, comparator: ComparatorResult | Mapping[str, Any] | None,
                        *, model_current: bool = True) -> tuple[str, str]:
    comp = _dict(comparator)
    if not model or not model_current:
        return "comparison_incomplete_model_unavailable", "Comparison incomplete: a current verified model result is unavailable."
    if comp.get("status") not in ("positive", "negative"):
        reasons = ", ".join(comp.get("reason_codes") or []) or "insufficient comparator evidence"
        return "comparison_incomplete_comparator_unavailable", f"Comparison incomplete: the comparator is indeterminate ({reasons})."
    flags = _dict(model.get("messages"))
    if "earlier_assessment" not in flags:
        return "comparison_incomplete_model_flag_unavailable", "Comparison incomplete: the model action flag was not supplied."
    model_flag, rule_flag = bool(flags["earlier_assessment"]), comp.get("status") == "positive"
    if rule_flag and model_flag:
        return "both_flagged", "A current comparator finding and the future-risk flag are both present; they address different time frames."
    if rule_flag and not model_flag:
        return "rule_positive_model_not_flagged", "The current comparator finding remains present; the lower future model risk does not negate it."
    if not rule_flag and model_flag:
        return "rule_negative_model_flagged", "The model adds a future-risk flag although current comparator criteria are not met; this is not proof of an established diagnosis."
    return "neither_flagged", "Neither criterion triggers; this is not a declaration that the valve is normal."


def _add(facts: list[SummaryFact], fact_id: str, label: str, value: Any, category: str,
         provenance: str, unit: str | None = None, measured_on: str | None = None) -> None:
    if value is not None:
        facts.append(SummaryFact(id=fact_id, label=label, value=value, unit=unit,
                                 measured_on=measured_on, provenance=provenance, category=category))


def assemble_patient_summary(request: PredictRequest | Mapping[str, Any], model_response: Mapping[str, Any] | None,
                             comparator: ComparatorResult | Mapping[str, Any] | None = None, *,
                             selected_family: str | None = None, source_provenance: str | None = None,
                             family_comparison: Mapping[str, Any] | None = None) -> PatientSummaryFacts:
    req = request if isinstance(request, PredictRequest) else PredictRequest.model_validate(request)
    response = _dict(model_response)
    source = source_provenance or req.source_kind or "unknown"
    if source not in ("synthetic", "real", "mixed", "unknown"):
        source = "unknown"
    comp_obj = comparator if isinstance(comparator, ComparatorResult) else None
    if comparator is None:
        comp_obj = comparator_for_request(req, source_kind=source)
    comp = comp_obj.to_dict() if comp_obj is not None else _dict(comparator)
    category, category_text = comparison_category(response or None, comp)
    sid = snapshot_hash(req, source)
    facts: list[SummaryFact] = []
    passport = req.passport
    _add(facts, "patient.case_id", "Opaque case identifier", passport.passport_id, "patient", source)
    _add(facts, "patient.source", "Source status", source, "patient", source)
    _add(facts, "patient.assessment_date", "Assessment date", req.prediction_time, "patient", source)
    static = req.static
    if static:
        _add(facts, "patient.age_at_implant", "Age at implant", static.age_at_implant, "patient", source, "years")
        _add(facts, "patient.sex", "Sex", static.sex, "patient", source)
        for field, label in (("diabetes", "Diabetes"), ("dialysis", "Dialysis"),
                             ("atrial_fibrillation", "Atrial fibrillation"),
                             ("bicuspid_native_valve", "Bicuspid native valve"),
                             ("lipid_lowering", "Lipid-lowering treatment")):
            _add(facts, f"patient.{field}", label, getattr(static, field), "patient", source)
    for field, label, unit in (("implant_date", "Implant date", None), ("route", "Valve route", None),
                               ("canonical_model", "Valve model", None), ("design_class", "Valve design", None),
                               ("size_mm", "Valve size", "mm")):
        _add(facts, f"valve.{field}", label, getattr(passport, field), "valve", source, unit)
    implant = parse_date(passport.implant_date)
    assessed = parse_date(req.prediction_time)
    if implant and assessed:
        _add(facts, "valve.age_years", "Valve age", round((assessed - implant).days / 365.25, 2), "valve", source, "years")

    for prefix, values, when in (("reference", comp.get("reference_values") or {}, comp.get("reference_study_date")),
                                 ("current", comp.get("current_values") or {}, comp.get("current_study_date"))):
        _add(facts, f"echo.{prefix}.date", f"{prefix.title()} echo date", when, "echo", source)
        for field, label, unit in (("mean_gradient_mmhg", "Mean gradient", "mmHg"), ("eoa_cm2", "EOA", "cm²"),
                                   ("dvi", "DVI", None), ("intraprosthetic_ar_grade", "Intraprosthetic AR grade", "ordinal"),
                                   ("lvef_pct", "LVEF", "%"), ("svi_ml_m2", "Stroke-volume index", "mL/m²")):
            _add(facts, f"echo.{prefix}.{field}", f"{prefix.title()} {label}", values.get(field), "echo", source, unit, when)
    for field, value in (comp.get("deltas") or {}).items():
        _add(facts, f"echo.change.{field}", field.replace("_", " ").title(), value, "echo", "computed")
    _add(facts, "comparator.status", "Independent comparator assessment", comp.get("status") or "indeterminate", "comparator", "computed")
    _add(facts, "comparator.stage", "Highest demonstrated comparator stage", comp.get("highest_demonstrated_stage"), "comparator", "computed")
    _add(facts, "comparator.reasons", "Comparator reason codes", comp.get("reason_codes") or [], "comparator", "computed")

    returned_family = response.get("model_family") or _dict(response.get("reliability")).get("model_family")
    _add(facts, "model.family", "Returned model family", returned_family, "model", "prediction service")
    _add(facts, "model.version", "Model version", response.get("model_version"), "model", "prediction service")
    _add(facts, "model.risk_12m", "Twelve-month SVD risk", response.get("p_svd_12m"), "model", "prediction service", "probability")
    messages = _dict(response.get("messages"))
    for key, label in (("current_abnormality", "Existing engine current-abnormality flag"),
                       ("earlier_assessment", "Earlier-assessment flag"),
                       ("overdue_surveillance", "Overdue-surveillance flag")):
        _add(facts, f"model.flag.{key}", label, messages.get(key), "model", "prediction service")
    for index, reason in enumerate((_dict(response.get("reliability")).get("reasons") or [])):
        if isinstance(reason, Mapping):
            _add(facts, f"model.reason.{index}", "Model reliability reason", dict(reason), "limitation", "prediction service")
    _add(facts, "comparison.category", "Model-versus-comparator category", category, "comparison", "computed")
    _add(facts, "comparison.interpretation", "Deterministic interpretation", category_text, "comparison", "computed")
    _add(facts, "limitation.validation", "Evidence status", ILLUSTRATIVE_LABEL, "limitation", "fixed")

    for i, lab in enumerate(req.labs):
        _add(facts, f"laboratory.{i}", lab.analyte, lab.value, "laboratory", source, lab.unit or None, lab.date)
    for i, episode in enumerate((req.exposure.episodes if req.exposure else [])):
        _add(facts, f"exposure.{i}", "Antithrombotic exposure",
             {"class": episode.class_, "indication": episode.indication, "start": episode.start, "stop": episode.stop},
             "exposure", source)

    actions: list[AllowedAction] = []
    if messages.get("earlier_assessment"):
        actions.append(AllowedAction(action_id="review_earlier_assessment_flag",
                                     text="Discuss the existing earlier-assessment flag in clinical review.",
                                     evidence_ids=["model.flag.earlier_assessment", "model.risk_12m"]))
    if messages.get("overdue_surveillance"):
        actions.append(AllowedAction(action_id="review_overdue_surveillance_flag",
                                     text="Review the existing overdue-surveillance reminder.",
                                     evidence_ids=["model.flag.overdue_surveillance"]))
    if comp.get("status") == "positive":
        actions.append(AllowedAction(action_id="review_positive_comparator",
                                     text="Review the positive current comparator finding.",
                                     evidence_ids=["comparator.status", "comparator.stage"]))
    if comp.get("status") == "indeterminate":
        actions.append(AllowedAction(action_id="verify_missing_comparator_inputs",
                                     text="Verify the missing or conflicting measurements needed by the comparator.",
                                     evidence_ids=["comparator.status", "comparator.reasons"]))

    include_family = family_comparison is not None
    if include_family:
        _add(facts, "comparison.families", "Validated family comparison", dict(family_comparison),
             "comparison", "prediction service")
    return PatientSummaryFacts(snapshot_id=sid, source_kind=source, selected_family=selected_family,
                               returned_family=returned_family, model_version=response.get("model_version"),
                               comparator_id=comp.get("comparator_id") or "varc3_hvd_comparator_v1",
                               comparison_category=category, facts=facts, allowed_actions=actions,
                               family_comparison_included=include_family)


def validate_findings(draft: ClinicianFindings, facts: PatientSummaryFacts) -> ClinicianFindings:
    fact_ids = {f.id for f in facts.facts}
    action_ids = {a.action_id for a in facts.allowed_actions}
    allowed_numbers = {str(n) for n in (1, 3, 5, 12)}
    for fact in facts.facts:
        if isinstance(fact.value, (int, float)) and not isinstance(fact.value, bool):
            allowed_numbers.add(f"{float(fact.value):g}")
            if fact.unit == "probability":
                allowed_numbers.add(f"{100 * float(fact.value):g}")
    entries = [*draft.findings, *draft.interpretation, *draft.limitations]
    for item in entries:
        if any(eid not in fact_ids for eid in item.evidence_ids):
            raise ValueError("draft cites an unknown evidence ID")
    for item in draft.recommendations:
        if item.action_id not in action_ids:
            raise ValueError("draft selected an ineligible action")
        if any(eid not in fact_ids for eid in item.evidence_ids):
            raise ValueError("recommendation cites an unknown evidence ID")
    text = " ".join([*(item.text for item in entries), *(item.rationale for item in draft.recommendations)])
    prohibited = ("prescribe", "start anticoag", "stop anticoag", "reintervention is indicated",
                  "superior model", "normal valve", "clinician-approved")
    if any(term in text.lower() for term in prohibited):
        raise ValueError("draft exceeds the permitted interpretation/recommendation scope")
    for token in re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", text):
        normal = f"{float(token):g}"
        if normal not in allowed_numbers:
            raise ValueError(f"draft contains an unsupported numeric value: {token}")
    return draft


def deterministic_summary(facts: PatientSummaryFacts) -> str:
    by_id = {f.id: f for f in facts.facts}
    family = facts.returned_family or facts.selected_family or "unavailable"
    risk = by_id.get("model.risk_12m")
    risk_text = f"{100 * float(risk.value):.1f}%" if risk is not None else "unavailable"
    comp = by_id.get("comparator.status")
    interpretation = by_id.get("comparison.interpretation")
    lines = ["## Template summary", "", "**Findings**",
             f"KAIROS {family} twelve-month SVD risk: {risk_text}. "
             f"Independent comparator: {comp.value if comp else 'indeterminate'}.", "", "**Interpretation**",
             str(interpretation.value if interpretation else "The comparison is incomplete."), "",
             "**Recommendations for clinical review**"]
    if facts.allowed_actions:
        lines.extend(f"- {action.text}" for action in facts.allowed_actions)
    else:
        lines.append("- Review the supplied findings in their clinical context; no new action was generated.")
    lines.extend(["", "**Limitations**", f"{ILLUSTRATIVE_LABEL.capitalize()}. The comparator is a current-echo criterion, "
                  "not an independent diagnosis of SVD mechanism. This template does not prescribe treatment or a surveillance interval."])
    return "\n".join(lines)


def export_summary(facts: PatientSummaryFacts, draft_markdown: str | None = None,
                   *, draft_label: str = "Template summary") -> str:
    header = ["# KAIROS Patient Summary", "", f"Snapshot: `{facts.snapshot_id}`",
              f"Source: {facts.source_kind}", f"Model: {facts.returned_family or facts.selected_family or 'unavailable'} / {facts.model_version or 'unavailable'}",
              f"Comparator: {facts.comparator_id}", f"Status: {ILLUSTRATIVE_LABEL}", "", "## Facts", ""]
    for fact in facts.facts:
        unit = f" {fact.unit}" if fact.unit else ""
        when = f" on {fact.measured_on}" if fact.measured_on else ""
        header.append(f"- `{fact.id}` {fact.label}: {fact.value}{unit}{when} ({fact.provenance})")
    header.extend(["", f"## {draft_label}", "", draft_markdown or deterministic_summary(facts)])
    return "\n".join(header) + "\n"
