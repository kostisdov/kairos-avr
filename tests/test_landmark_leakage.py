"""Leakage rules of the landmark builder (milestone M1 unit tests)."""
from datetime import date

import pandas as pd

from kairos.adjudication.framework import EchoPoint
from kairos.modelling.landmark import (
    ExposureRecord,
    build_landmark_dataset,
    exposure_features,
    features_at_landmark,
)


def _tables():
    patients = pd.DataFrame([{"patient_id": "A", "route": "SAVR", "design_class": "externally mounted pericardial",
                              "canonical_model": "Trifecta", "generation": "", "tissue_treatment": "none", "size_mm": 21,
                              "implant_date": date(2015, 1, 1), "implant_year": 2015, "age_at_implant": 70, "sex": "F", "bsa": 1.8,
                              "bmi": 26, "diabetes": False, "egfr0": 70, "dialysis": False, "af": False, "bicuspid": False,
                              "lipid_lowering": True}])
    echoes = pd.DataFrame([
        {"patient_id": "A", "date": date(2015, 3, 1), "mean_gradient": 10, "eoa": 1.8, "dvi": 0.5, "ar_grade": "none", "lvef": 60, "svi": 40},
        {"patient_id": "A", "date": date(2016, 3, 1), "mean_gradient": 12, "eoa": 1.7, "dvi": 0.48, "ar_grade": "none", "lvef": 60, "svi": 40},
        {"patient_id": "A", "date": date(2017, 3, 1), "mean_gradient": 25, "eoa": 1.2, "dvi": 0.35, "ar_grade": "none", "lvef": 58, "svi": 39},
        {"patient_id": "A", "date": date(2018, 3, 1), "mean_gradient": 30, "eoa": 1.1, "dvi": 0.33, "ar_grade": "none", "lvef": 55, "svi": 38},
    ])
    events = pd.DataFrame([{"patient_id": "A", "implant_date": date(2015, 1, 1), "reference_date": date(2015, 3, 1),
                            "svd_candidate_date": date(2017, 3, 1), "svd_adjudicated_date": date(2017, 3, 1),
                            "svd_status": "confirmed", "svd_first_unresolved_date": None,
                            "svd_first_positive_date": date(2017, 3, 1),
                            "death_date": None, "replacement_date": None, "end_followup_date": date(2019, 1, 1),
                            "end_reason": "administrative", "first_thrombosis_date": None}])
    return patients, echoes, pd.DataFrame(columns=["patient_id", "date", "analyte", "value"]), \
        pd.DataFrame(columns=["patient_id", "class", "agent", "indication", "start_date", "stop_date", "source", "post_suspicion"]), events


def test_endpoint_establishing_echo_is_never_a_landmark_or_a_predictor():
    lm = build_landmark_dataset(*_tables())
    dates = sorted(lm["landmark_date"].tolist())
    assert dates == [date(2015, 3, 1), date(2016, 3, 1)]           # nothing on or after the endpoint echo
    row = lm[lm["landmark_date"] == date(2016, 3, 1)].iloc[0]
    assert row["current_gradient"] == 12 and row["n_echoes"] == 2  # the 25 mmHg endpoint echo never enters
    assert row["event"] == 1 and abs(row["time"] - 1.0) < 0.01     # SVD one year after this landmark
    first = lm[lm["landmark_date"] == date(2015, 3, 1)].iloc[0]
    assert first["t_lm"] == 0 and first["delta_gradient"] == 0


def test_patient_meeting_endpoint_before_reference_is_excluded():
    patients, echoes, labs, exposures, events = _tables()
    events.loc[0, "svd_adjudicated_date"] = date(2015, 3, 1)  # endpoint at the reference study itself
    lm = build_landmark_dataset(patients, echoes, labs, exposures, events)
    assert lm.empty


def test_features_only_use_information_on_or_before_the_landmark():
    pts = [EchoPoint(date(2015, 3, 1), 10, 1.8, 0.5, 0, 60, 40, index=0), EchoPoint(date(2016, 3, 1), 12, 1.7, 0.48, 0, 60, 40, index=1),
           EchoPoint(date(2017, 3, 1), 25, 1.2, 0.35, 0, 58, 39, index=2)]
    labs = [(date(2015, 6, 1), "egfr", 70.0), (date(2017, 1, 1), "egfr", 50.0)]
    exposures = [ExposureRecord("VKA", "AF", date(2015, 1, 1), None, False)]
    f = features_at_landmark({"route": "SAVR", "bsa": 1.8}, pts, pts[0], date(2016, 3, 1), labs, exposures, date(2015, 1, 1))
    assert f["current_gradient"] == 12 and f["n_echoes"] == 2 and f["egfr"] == 70.0
    assert abs(f["ac_cum_vka_years"] - 1.16) < 0.02 and f["ac_class_current"] == "VKA"
    assert f["valve_age_years"] > 1.1 and f["overdue_flag"] is False and f["stale_echo"] is False


def test_censoring_at_horizon_and_competing_death():
    patients, echoes, labs, exposures, events = _tables()
    events.loc[0, "svd_adjudicated_date"] = None
    events.loc[0, "death_date"] = date(2018, 9, 1)
    events.loc[0, "end_followup_date"] = date(2018, 9, 1)
    lm = build_landmark_dataset(patients, echoes, labs, exposures, events, horizon_years=5.0)
    assert set(lm["event"]) == {2}
    lm2 = build_landmark_dataset(patients, echoes, labs, exposures, events, horizon_years=1.0)
    first = lm2.sort_values("landmark_date").iloc[0]
    assert first["event"] == 0 and first["time"] == 1.0


def test_exposure_status_past_and_post_suspicion():
    eps = [ExposureRecord("FXa", "suspected_valve_thrombosis", date(2017, 5, 1), date(2017, 11, 1), True)]
    f = exposure_features(eps, date(2018, 1, 1))
    assert f["ac_current_status"] == "past" and f["ac_post_suspicion"] is True and f["ac_class_current"] == "none"


def test_thrombosis_after_follow_up_ends_is_not_an_event():
    from datetime import date

    from kairos.modelling.landmark import _outcome
    out = _outcome(date(2020, 1, 1), None, None, None, date(2021, 1, 1), 5.0, thromb_date=date(2022, 1, 1))
    assert out["event_thromb"] == 0 and abs(out["time_thromb"] - 366 / 365.25) < 1e-9
    seen = _outcome(date(2020, 1, 1), None, None, None, date(2021, 1, 1), 5.0, thromb_date=date(2020, 7, 1))
    assert seen["event_thromb"] == 1
