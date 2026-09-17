"""Span-aware rules extraction on the synthetic fixtures."""
from pathlib import Path

import pytest

from kairos.extraction.rules import extract_rules, mentions_prosthesis

FIX = Path(__file__).parent / "fixtures" / "synthetic_notes"


def _run(name, note_type="operative", when="2016"):
    text = (FIX / name).read_text(encoding="utf-8")
    return text, extract_rules(text, name, note_type, when)


@pytest.mark.parametrize("name,note_type,route,model,size", [
    ("01_savr_op_perimount_ce.txt", "operative", "SAVR", "Perimount", 23),
    ("02_savr_op_trifecta_hashsize.txt", "operative", "SAVR", "Trifecta", 21),
    ("04_tavr_sapien3_ultra_size_field.txt", "procedure", "TAVR", "SAPIEN 3 Ultra", None),
    ("05_tavr_evolut_misspelled_corevalue.txt", "procedure", "TAVR", "Evolut PRO+", None),
    ("17_savr_op_ce_size_after.txt", "operative", "SAVR", "Perimount", 25),
    ("20_tavr_bare_s3_guarded.txt", "procedure", "TAVR", "SAPIEN 3", None),
    ("24_tavr_bare_ultra_resilia.txt", "procedure", "TAVR", "SAPIEN 3 Ultra RESILIA", None),
])
def test_fixture_fields(name, note_type, route, model, size):
    text, r = _run(name, note_type)
    p = r.passport
    assert p.route == route
    assert p.canonical_model == model
    if size is not None:
        assert p.size_mm == size
    assert p.implant_date == "2016"  # implant note dated 2016
    assert p.design_class
    assert p.confidence.canonical_model > 0.5


def test_evidence_spans_point_into_the_note():
    text, r = _run("02_savr_op_trifecta_hashsize.txt")
    fields = {e.field for e in r.passport.evidence}
    assert {"canonical_model", "route", "size_mm"} <= fields
    for e in r.passport.evidence:
        s, t = e.span
        assert text[s:t] == e.text
        assert len(e.text) <= 160
    assert r.passport.market_status == "withdrawn"  # Trifecta


def test_native_and_prosthetic_values_are_separated():
    text, r = _run("07_progress_note_native_and_prosthetic_gradient.txt", "progress", "2018")
    by_cls = {o.native_vs_prosthetic: o for o in r.echo_observations}
    assert by_cls["native"].mean_gradient_mmhg == 52
    assert by_cls["prosthetic"].mean_gradient_mmhg == 13


def test_progress_note_events_and_regurgitation():
    text, r = _run("10_progress_note_severe_regurg_event.txt", "progress", "2021")
    assert "dysfunction" in {e.type for e in r.passport.events}
    pros = [o for o in r.echo_observations if o.native_vs_prosthetic == "prosthetic"]
    assert pros and pros[0].ar_grade == "severe" and pros[0].mean_gradient_mmhg == 34
    assert r.prescreen.mentions_prosthesis


def test_epic_ehr_is_not_a_device():
    text, r = _run("14_progress_note_epic_ehr_reference_only.txt", "progress", "2022")
    assert r.passport.canonical_model is None


def test_prescreen_negative_on_unrelated_text():
    assert not mentions_prosthesis("The patient was seen for a routine dermatology visit. No cardiac history.")
