"""Comparison against current practice: echo-report fields, the calendar comparator, the primary
population and the surveillance-policy simulation."""
import numpy as np
import pandas as pd
import pytest

from kairos.comparators.calendar import CalendarComparator
from kairos.evaluation.surveillance import (
    Policy,
    _load_patients,
    _Measurer,
    base_schedule,
    prevalent_early_hvd,
    rule_status,
    simulate_policy,
    summarise,
)
from kairos.modelling.landmark import FORBIDDEN_PREDICTORS
from kairos.modelling.train import landmark_from_cohort
from kairos.simulation.scenarios import get_scenario


@pytest.fixture(scope="module")
def lm(small_cohort, model_cfg):
    return landmark_from_cohort(small_cohort, model_cfg)


@pytest.fixture(scope="module")
def patients(small_cohort, lm):
    return list(_load_patients(small_cohort, set(lm["patient_id"])).values())


@pytest.fixture(scope="module")
def measure():
    return _Measurer(get_scenario("gradual_stenotic"), seed=7, p_location=0.9)


def test_echoes_carry_regurgitation_location_and_paravalvular_leak(small_cohort):
    e = small_cohort.echoes
    assert {"ar_location", "paravalvular_leak_grade"} <= set(e.columns)
    assert set(e["ar_location"].dropna()) == {"intraprosthetic"}
    assert 0.8 < e["ar_location"].notna().mean() < 0.97
    # one leak grade per patient, stable across that patient's studies
    assert (e.groupby("patient_id")["paravalvular_leak_grade"].nunique() == 1).all()


def test_trajectory_columns_are_truth_and_never_predictors(small_cohort):
    traj = [c for c in small_cohort.truth.columns if c.startswith("traj_")]
    assert len(traj) == 10
    assert set(traj) <= FORBIDDEN_PREDICTORS


def test_calendar_comparator_depends_only_on_route_and_valve_age(lm):
    cal = CalendarComparator(cut_years=5, by_route=True, horizon_years=1.0, min_rows=1).fit(lm)
    p = cal.predict(lm)
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
    key = lm["route"].astype(str) + (lm["valve_age_years"] >= 5).map({True: "o", False: "y"})
    assert (pd.Series(p).groupby(key.to_numpy()).nunique() == 1).all()


def test_primary_population_excludes_prevalent_early_deterioration():
    lm = pd.DataFrame({"delta_gradient": [0.0, 9.9, 10.0, np.nan, 2.0], "delta_ar": [0, 0, 0, np.nan, 1]})
    got = prevalent_early_hvd(lm, {"gradient_rise_mmhg": 10, "ar_increase_grades": 1})
    assert got.tolist() == [False, False, True, False, True]


def test_rule_needs_regurgitation_location_to_rule_out():
    ref = {"mean_gradient_mmhg": 10, "eoa_cm2": 1.8, "dvi": 0.45, "intraprosthetic_ar_grade": 0}
    same = dict(ref)
    assert rule_status(ref, same, location_reported=True) == "negative"
    assert rule_status(ref, same, location_reported=False) == "indeterminate"
    worse = {"mean_gradient_mmhg": 25, "eoa_cm2": 1.3, "dvi": 0.32, "intraprosthetic_ar_grade": 0}
    assert rule_status(ref, worse, location_reported=False) == "positive"


def test_acc_aha_images_surgical_valves_at_five_and_ten_years():
    savr = base_schedule("acc_aha", "SAVR", 0.3, 12.5)
    assert savr[:4].tolist() == [5.0, 10.0, 11.0, 12.0]
    tavr = base_schedule("acc_aha", "TAVR", 0.3, 3.0)
    assert np.allclose(tavr, [1.3, 2.3])
    assert np.allclose(base_schedule("annual", "SAVR", 0.3, 2.5), [1.3, 2.3])


def test_measurements_are_shared_across_policies_on_the_same_day(patients, measure):
    pt = patients[0]
    a, _, la = measure(pt, 2.0)
    b, _, lb = measure(pt, 2.0)
    c, _, _ = measure(pt, 3.0)
    assert (a.mean_gradient, a.eoa, a.dvi, la) == (b.mean_gradient, b.eoa, b.dvi, lb)
    assert (a.mean_gradient, a.eoa) != (c.mean_gradient, c.eoa)


def test_a_silent_model_reproduces_the_base_schedule(patients, measure):
    base = simulate_policy(patients, Policy("annual", "annual"), measure)
    silent = simulate_policy(patients, Policy("g", "annual", True, 0.05), measure,
                             predictor=lambda df: np.zeros(len(df)))
    cols = ["patient_id", "detect", "n_echoes"]
    pd.testing.assert_frame_equal(base[cols], silent[cols])


def test_guidance_only_adds_echoes_and_never_detects_later(patients, measure):
    base = simulate_policy(patients, Policy("acc_aha", "acc_aha"), measure).set_index("patient_id")
    loud = simulate_policy(patients, Policy("g", "acc_aha", True, 0.05, 0.5, 0.1), measure,
                           predictor=lambda df: np.ones(len(df))).set_index("patient_id")
    # a patient the guided policy never flags is imaged at least as often (a detection ends imaging)
    undetected = loud["detect"].isna()
    assert (loud.loc[undetected, "n_echoes"] >= base.loc[undetected, "n_echoes"]).all()
    both = base["detect"].notna() & loud["detect"].notna()
    assert (loud.loc[both, "detect"] <= base.loc[both, "detect"] + 1e-9).all()
    assert loud["alerts"].sum() > 0


def test_summary_reports_lead_time_and_echo_burden(patients, measure):
    base = simulate_policy(patients, Policy("annual", "annual"), measure)
    loud = simulate_policy(patients, Policy("annual+kairos@0.05", "annual", True, 0.05), measure,
                           predictor=lambda df: np.ones(len(df)))
    s = summarise(pd.concat([base, loud], ignore_index=True), n_boot=5)
    row = s["paired"].iloc[0]
    assert row["versus"] == "annual" and row["extra_echoes_per_1000py"] > 0
    assert set(s["by_policy"]["policy"]) == {"annual", "annual+kairos@0.05"}
