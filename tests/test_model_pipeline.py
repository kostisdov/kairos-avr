"""Training, the predictor rules and the module ladder on a small synthetic cohort."""
import numpy as np
import pytest

from kairos import ILLUSTRATIVE_LABEL
from kairos.evaluation.ladder import evaluate_ladder
from kairos.modelling.modules import fit_eligibility, ladder_steps
from kairos.modelling.predictor import EndpointMetError, NoReferenceEchoError, Predictor
from kairos.modelling.train import landmark_from_cohort
from kairos.simulation.generators import generate_cohort
from kairos.simulation.scenarios import get_scenario
from tests.helpers import DETERIORATING, STABLE, cohort_request, passport, request


def test_bundle_summary_and_card(small_bundle):
    card = small_bundle.card()
    assert card["ladder_step"] == "core_plus_both" and card["label"] == ILLUSTRATIVE_LABEL
    assert small_bundle.model.event_counts_["svd"] >= 5 and small_bundle.model.event_counts_["death"] >= 5
    assert card["family"] == "cox" and card["schema_version"] == 2 and card["claimed_routes"]
    assert "valve_age_years" in small_bundle.features and "route" in small_bundle.features


def test_prediction_on_a_synthetic_patient(small_bundle, small_cohort, model_cfg):
    req, ech = cohort_request(small_cohort)
    pred = Predictor(small_bundle, model_cfg).predict(req)
    tot = np.array(pred.p_svd_before_death) + np.array(pred.p_death_before_svd) + np.array(pred.p_alive_intact) + np.array(pred.p_replaced_non_svd)
    assert np.allclose(tot, 1.0, atol=1e-4)
    assert np.all(np.diff(pred.p_svd_before_death) >= -1e-9)
    assert 0 <= pred.p_svd_12m <= pred.p_svd_before_death[0] + 1e-9
    assert pred.reliability.label == ILLUSTRATIVE_LABEL and pred.scenario_set.startswith("gradual_stenotic")
    assert pred.reliability.device_evidence in ("model-level", "class-level")
    assert pred.drivers and all(d.direction in ("up", "down") for d in pred.drivers)


def test_confirmed_endpoint_returns_409_rule(small_bundle, model_cfg):
    pr = Predictor(small_bundle, model_cfg)
    with pytest.raises(EndpointMetError):
        pr.predict(request(DETERIORATING, "2020-01-01"))
    # at the single unconfirmed candidate the prediction proceeds and the finding is flagged
    pred = pr.predict(request(DETERIORATING[:3], "2018-08-01"))
    assert pred.messages.current_abnormality is True


def test_replaced_index_valve_returns_409_rule(small_bundle, model_cfg):
    p = passport(events=[{"type": "redo", "date": "2019-01-01"}])
    with pytest.raises(EndpointMetError):
        Predictor(small_bundle, model_cfg).predict(request(STABLE, "2019-06-01", p=p))


def test_no_reference_echo_is_rejected(small_bundle, model_cfg):
    with pytest.raises(NoReferenceEchoError):
        Predictor(small_bundle, model_cfg).predict(request([("2017-08-01", 10, 1.8, 0.5)], "2018-01-01"))


def test_only_observations_before_prediction_time_are_used(small_bundle, model_cfg):
    pr = Predictor(small_bundle, model_cfg)
    early = pr.predict(request(DETERIORATING, "2017-08-01"))
    assert early.messages.current_abnormality is False  # the 2018 stage-2 echo is after the prediction time


def test_overdue_and_stale_flags(small_bundle, model_cfg):
    pred = Predictor(small_bundle, model_cfg).predict(request(STABLE, "2020-11-01"))
    assert pred.messages.overdue_surveillance is True and pred.reliability.stale_echo is True
    assert "guideline" in pred.message_explanations["overdue_surveillance"]


def test_unmeasured_marker_switches_module_off(model_cfg):
    cohort = generate_cohort(get_scenario("biomarker_information", "unmeasured"), n=200, seed=4)
    av = fit_eligibility(landmark_from_cohort(cohort, model_cfg), model_cfg).card()["modules"]
    assert av["biomarker_lipid"]["available"] is False and "not imputed" in av["biomarker_lipid"]["reason"]
    assert av["biomarker_renal_metabolic"]["available"] is True


def test_ladder_is_fixed(model_cfg):
    names = [n for n, _ in ladder_steps(model_cfg)]
    assert names[:2] == ["reference", "core"] and names[-1] == "core_without_serial_echo"
    assert {"core_plus_biomarkers", "core_plus_anticoagulant", "core_plus_both"} <= set(names)


def test_ladder_evaluation_runs_with_patient_grouping(small_cohort, model_cfg):
    res = evaluate_ladder(small_cohort, model_cfg, n_splits=2, n_boot=5, steps=["reference", "core"])
    r = res.results
    # 2 steps x 3 horizons x 4 states x (pooled, patient-balanced)
    assert set(r["step"]) == {"reference", "core"} and len(r) == 48
    assert set(r["state"]) == {"svd", "death", "replacement", "alive_intact"}
    assert r["brier"].between(0, 1).all() and r["label"].eq("synthetic scenario").all()
    assert (r["support_mode"] == "quick").all() and r["exploratory"].all()   # quick gates never support claims
    assert r["coverage_rows"].between(0, 1).all()
    assert "reference" in res.curves and res.summary["n_patients"] > 0
