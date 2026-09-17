"""Landmark builder version 2: label policies, explicit exclusions, NaN hygiene, exposure
coverage, truth refusal and unique-event accounting (detailed design WP-A5)."""
import copy
from datetime import date

import numpy as np
import pandas as pd
import pytest

from kairos.evaluation.support import event_rows, event_summary, unique_event_patients
from kairos.modelling.landmark import (
    FORBIDDEN_PREDICTORS,
    build_landmark,
    exposure_features,
    resolve_svd_label,
)
from kairos.modelling.train import fit_step, landmark_from_cohort
from tests.test_landmark_leakage import _tables


def test_label_policies_for_an_earlier_unresolved_candidate():
    patients, echoes, labs, exposures, events = _tables()
    events.loc[0, "svd_first_unresolved_date"] = date(2015, 9, 1)
    e = events.iloc[0]
    assert resolve_svd_label(e, "primary") == (None, date(2015, 9, 1))
    assert resolve_svd_label(e, "sens_uncertain_positive") == (date(2015, 9, 1), None)
    assert resolve_svd_label(e, "sens_uncertain_negative") == (date(2017, 3, 1), None)
    assert resolve_svd_label(e, "legacy_single_study") == (date(2017, 3, 1), None)
    with pytest.raises(ValueError):
        resolve_svd_label(e, "made_up")

    primary = build_landmark(patients, echoes, labs, exposures, events).rows
    assert primary["landmark_date"].tolist() == [date(2015, 3, 1)]       # nothing on or after the candidate
    assert primary["event"].tolist() == [0] and abs(primary["time"].iloc[0] - 0.5) < 0.01
    pos = build_landmark(patients, echoes, labs, exposures, events, label_policy="sens_uncertain_positive").rows
    assert pos["event"].tolist() == [1]
    neg = build_landmark(patients, echoes, labs, exposures, events, label_policy="sens_uncertain_negative").rows
    assert len(neg) == 2 and set(neg["event"]) == {1}


def test_event_override_rebuilds_eligibility():
    patients, echoes, labs, exposures, events = _tables()
    override = pd.DataFrame({"svd_event_date": [date(2016, 1, 1)]}, index=pd.Index(["A"], name="patient_id"))
    rows = build_landmark(patients, echoes, labs, exposures, events, event_override=override).rows
    assert rows["landmark_date"].tolist() == [date(2015, 3, 1)] and rows["event"].tolist() == [1]


def test_explicit_exclusion_counts():
    patients, echoes, labs, exposures, events = _tables()
    events.loc[0, "svd_adjudicated_date"] = date(2015, 3, 1)
    b = build_landmark(patients, echoes, labs, exposures, events)
    assert b.rows.empty and b.exclusions["endpoint_at_or_before_reference"] == 1 and b.label_policy == "primary"
    patients, echoes, labs, exposures, events = _tables()
    events.loc[0, ["svd_adjudicated_date", "death_date"]] = [None, date(2015, 3, 1)]
    assert build_landmark(patients, echoes, labs, exposures, events).exclusions["death_or_replacement_at_or_before_reference"] == 1


def test_death_and_replacement_stop_landmarks_independently_of_end_date():
    patients, echoes, labs, exposures, events = _tables()
    events.loc[0, ["svd_adjudicated_date", "replacement_date"]] = [None, date(2016, 6, 1)]   # end date stays 2019
    rows = build_landmark(patients, echoes, labs, exposures, events).rows
    assert rows["landmark_date"].max() < date(2016, 6, 1) and set(rows["event"]) == {3}


def test_nan_gradient_is_not_an_adequate_reference():
    patients, echoes, labs, exposures, events = _tables()
    early = {"patient_id": "A", "date": date(2015, 2, 15), "mean_gradient": np.nan, "eoa": 1.8, "dvi": 0.5,
             "ar_grade": np.nan, "lvef": 60, "svi": 40}
    echoes = pd.concat([pd.DataFrame([early]), echoes], ignore_index=True)
    rows = build_landmark(patients, echoes, labs, exposures, events).rows
    assert rows["landmark_date"].min() == date(2015, 3, 1)


def test_truth_columns_are_refused():
    patients, echoes, labs, exposures, events = _tables()
    patients["phenotype_latent"] = "stenotic"
    with pytest.raises(ValueError, match="truth"):
        build_landmark(patients, echoes, labs, exposures, events)


def test_unknown_medication_coverage_is_not_never_anticoagulated():
    unknown = exposure_features([], date(2018, 1, 1), records_available=False)
    assert unknown["ac_records_available"] is False and np.isnan(unknown["ac_class_current"])
    assert np.isnan(unknown["ac_cum_vka_years"])
    verified_none = exposure_features([], date(2018, 1, 1), records_available=True)
    assert verified_none["ac_class_current"] == "none" and verified_none["ac_current_status"] == "never"
    patients, echoes, labs, exposures, events = _tables()
    rows = build_landmark(patients, echoes, labs, exposures, events).rows
    assert (~rows["ac_records_available"].astype(bool)).all() and rows["ac_current_status"].isna().all()


def test_forbidden_predictor_guard(small_cohort, model_cfg):
    lm = landmark_from_cohort(small_cohort, model_cfg)
    assert {"patient_id", "event", "svd_adjudicated_date", "phenotype_latent", "bootstrap_cluster_id"} <= FORBIDDEN_PREDICTORS
    cfg = copy.deepcopy(model_cfg)
    cfg["feature_blocks"]["core_time"] = cfg["feature_blocks"]["core_time"] + ["bootstrap_cluster_id"]
    with pytest.raises(ValueError, match="never be predictors"):
        fit_step(lm, ["core_time"], cfg)


def test_unique_events_ignore_repeated_rows_and_bootstrap_duplicates(small_cohort, model_cfg):
    lm = landmark_from_cohort(small_cohort, model_cfg)
    n_pat, n_rows = unique_event_patients(lm, 1), event_rows(lm, 1)
    assert 0 < n_pat < n_rows
    dup = pd.concat([lm, lm[lm["event"] == 1].assign(bootstrap_cluster_id=lambda d: d["patient_id"] + "#b2")])
    assert unique_event_patients(dup, 1) == n_pat and event_rows(dup, 1) == 2 * n_rows
    s = event_summary(lm, by="route")
    assert sum(s["svd"]["event_patients"].values()) == n_pat
    three = pd.DataFrame({"patient_id": ["a"] * 4 + ["b"] * 4 + ["c"] * 4, "event": [1] * 12})
    assert unique_event_patients(three, 1) == 3 and event_rows(three, 1) == 12
    assert (lm["row_weight_patient_balanced"].groupby(lm["patient_id"]).sum().round(9) == 1).all()
