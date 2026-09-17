"""Patient inputs of the demonstration: examples, form <-> request, staging, what-if."""
from datetime import date

import numpy as np
import pytest

from kairos import demo_patient as dp
from kairos.modelling.predictor import Predictor


@pytest.mark.parametrize("name", list(dp.PRESETS))
def test_every_example_builds_a_valid_request_and_predicts(name, small_bundle, model_cfg):
    request = dp.request_from_form(dp.PRESETS[name]())
    assert request.source_kind == "synthetic" and request.passport.implant_date
    pred = Predictor(small_bundle, model_cfg).predict(request)
    total = (np.array(pred.p_svd_before_death) + pred.p_death_before_svd + np.array(pred.p_alive_intact)
             + pred.p_replaced_non_svd)
    assert np.allclose(total, 1.0, atol=1e-4)


def test_form_request_round_trip():
    form = dp.preset_evolut_treated_thrombosis()
    again = dp.form_from_request(dp.request_from_form(form).model_dump(mode="json", by_alias=True))
    assert again["valve"] == form["valve"]
    assert again["prediction_time"] == form["prediction_time"]
    assert [e["date"] for e in again["echoes"]] == [e["date"] for e in form["echoes"]]
    assert [(e["class"], e["post_suspicion"]) for e in again["episodes"]] == [("SAPT", False), ("FXa", True)]
    assert {lab["analyte"] for lab in again["labs"]} == {"egfr", "ntprobnp"}


def test_frames_round_trip_through_the_editable_tables():
    form = dp.preset_mosaic_dialysis_vka()
    assert dp.lab_records(dp.labs_frame(form["labs"]))[1]["analyte"] == "egfr"
    assert dp.echo_records(dp.echo_frame(form["echoes"]))[0]["mean_gradient_mmhg"] == 12.0
    assert dp.episode_records(dp.episodes_frame(form["episodes"]))[0]["class"] == "VKA"


def test_staging_marks_reference_and_candidate():
    rows, reference = dp.staging_rows(dp.preset_trifecta_gradual())
    assert reference == date(2016, 6, 12)
    assert rows[0]["VARC-3 stage"] == "reference" and rows[-1]["VARC-3 stage"] == "1"
    rows, _ = dp.staging_rows(dp.preset_evolut_treated_thrombosis())
    assert [r["VARC-3 stage"] for r in rows][2] == "2"  # the treated thrombosis episode
    rows, _ = dp.staging_rows(dp.preset_perimount_regurgitation())
    assert rows[-1]["VARC-3 stage"] == "3"


def test_missing_reference_study_is_reported():
    form = dp.preset_blank()
    form["echoes"][0]["date"] = date(2022, 6, 10)  # 9 days after implant: too early for time zero
    rows, reference = dp.staging_rows(form)
    assert reference is None and rows[0]["VARC-3 stage"] == "uncertain"


def test_form_errors_are_listed():
    form = dp.preset_blank()
    form["valve"]["implant_date"] = None
    form["echoes"].append({"date": None, "mean_gradient_mmhg": 12.0})
    form["episodes"].append({"class": "VKA", "start": date(2023, 1, 1), "stop": date(2022, 1, 1)})
    with pytest.raises(dp.FormError) as exc:
        dp.request_from_form(form)
    text = " ".join(exc.value.messages)
    assert "implant date" in text and "Echo row 2" in text and "stop date precedes" in text


def test_hypothetical_next_echo_and_anticoagulant_changes():
    form = dp.preset_sapien_af_stable()
    last = dp.latest_echo(form)["date"]
    hyp = dp.hypothetical_form(form, 12, 25.0, 1.2, 0.35, "mild", "stop anticoagulation")
    assert len(hyp["echoes"]) == len(form["echoes"]) + 1 and hyp["prediction_time"] > last
    assert hyp["episodes"][0]["stop"] == last and form["episodes"][0]["stop"] is None  # original untouched
    started = dp.hypothetical_form(dp.preset_perimount_regurgitation(), 6, 12.0, 1.8, 0.5, "none", "start VKA")
    assert started["episodes"][-1]["class"] == "VKA"
    assert dp.at_latest_echo(form)["prediction_time"] == last


def test_module_coverage_and_extracted_values():
    rows = {r["module"]: r for r in dp.module_coverage(dp.preset_mosaic_dialysis_vka(), ["biomarker_inflammatory"])}
    assert "Phosphate" in rows["Mineral metabolism"]["this patient"]
    assert rows["Inflammatory and molecular"]["in the model"] == "switched off"
    form, notes = dp.apply_passport(dp.preset_blank(), {"canonical_model": "Trifecta", "size_mm": 21, "route": "SAVR",
                                                        "implant_date": "2016"})
    assert form["valve"]["canonical_model"] == "Trifecta" and form["valve"]["size_mm"] == 21
    assert any("year-only" in n for n in notes)
    form, added = dp.add_echo_observations(dp.preset_blank(), [
        {"date": "2023-05-01", "native_vs_prosthetic": "prosthetic", "mean_gradient_mmhg": 14, "eoa_cm2": 1.6},
        {"date": "2023-05-01", "native_vs_prosthetic": "native", "mean_gradient_mmhg": 50}])
    assert added == 1 and form["echoes"][-1]["mean_gradient_mmhg"] == 14
