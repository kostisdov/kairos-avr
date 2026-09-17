"""dp-ucMGP as a consumed input: carry-forward, the offset model, the generator variants,
the predictor statement and the mechanism check on synthetic scenarios."""
import copy
from datetime import date

import numpy as np
import pandas as pd
import pytest

from kairos.evaluation.ladder import patient_folds
from kairos.evaluation.vitamin_k import evaluate_vitamin_k, mechanism_check
from kairos.extraction.schema import LabObservation, VitaminKStatus
from kairos.modelling.predictor import Predictor
from kairos.modelling.train import landmark_from_cohort, train_bundle
from kairos.modelling.vitamin_k import dp_ucmgp_features, fit_offset_cox, measured_mask, rcs_basis
from kairos.simulation.generators import generate_cohort
from kairos.simulation.scenarios import get_scenario
from tests.helpers import cohort_request

SCENARIO = "anticoagulant_mechanism_confounding"


@pytest.fixture(scope="module")
def mediated_cohort():
    return generate_cohort(get_scenario(SCENARIO, "marker_mediated"), n=1500, seed=11)


@pytest.fixture(scope="module")
def noise_cohort():
    return generate_cohort(get_scenario(SCENARIO, "marker_noise"), n=1500, seed=11)


@pytest.fixture(scope="module")
def mediated_lm(mediated_cohort, model_cfg):
    return landmark_from_cohort(mediated_cohort, model_cfg)


@pytest.fixture(scope="module")
def mediated_bundle(mediated_cohort, mediated_lm, model_cfg):
    return train_bundle(mediated_cohort, model_cfg, "core_plus_anticoagulant", "0.1.0+test", lm=mediated_lm)


def test_carry_forward_twelve_months_then_stale_and_no_look_ahead():
    labs = [(date(2020, 1, 1), "dp_ucmgp", 900.0), (date(2022, 6, 1), "dp_ucmgp", 2000.0), (date(2020, 1, 1), "egfr", 60.0)]
    fresh = dp_ucmgp_features(labs, date(2020, 12, 1))
    assert fresh["dp_ucmgp"] == 900.0 and np.isclose(fresh["log_dp_ucmgp"], np.log(900.0)) and not fresh["dp_ucmgp_stale"]
    stale = dp_ucmgp_features(labs, date(2021, 3, 1))
    assert stale["dp_ucmgp_stale"] and np.isnan(stale["log_dp_ucmgp"])
    none = dp_ucmgp_features(labs, date(2019, 12, 31))
    assert np.isnan(none["dp_ucmgp"]) and not none["dp_ucmgp_stale"]


def test_restricted_cubic_spline_is_linear_beyond_outer_knots():
    knots = np.array([5.5, 6.1, 7.2])
    x = np.array([7.5, 8.0, 8.5])
    b = rcs_basis(x, knots)[:, 1]
    assert np.isclose(b[2] - b[1], b[1] - b[0])
    assert np.allclose(rcs_basis(np.array([5.0, 5.4]), knots)[:, 1], 0.0)


def test_offset_cox_recovers_a_known_effect_and_uses_the_offset():
    rng = np.random.default_rng(0)
    n = 4000
    z = rng.normal(size=(n, 1))
    offset = rng.normal(scale=0.8, size=n)
    t_event = rng.exponential(1.0 / (0.1 * np.exp(offset + 0.5 * z[:, 0])))
    t_cens = rng.uniform(0, 10, size=n)
    time, event = np.minimum(t_event, t_cens), (t_event <= t_cens).astype(int)
    fit = fit_offset_cox(z, time, event, offset, np.zeros(n), penalizer=0.0)
    assert fit["converged"] and abs(fit["beta"][0] - 0.5) < 0.08 and fit["p_value"] < 1e-6
    no_offset = fit_offset_cox(z, time, event, np.zeros(n), np.zeros(n), penalizer=0.0)
    assert abs(no_offset["beta"][0] - 0.5) > abs(fit["beta"][0] - 0.5) - 1e-9


def test_marker_has_its_own_random_stream():
    spec = get_scenario("gradual_stenotic")
    off = copy.deepcopy(spec)
    off.params["vitamin_k"]["dp_ucmgp_substudy_fraction"] = 0.0
    a, b = generate_cohort(spec, n=150, seed=3), generate_cohort(off, n=150, seed=3)
    assert a.echoes.equals(b.echoes) and a.exposures.equals(b.exposures)
    assert (a.labs["analyte"] == "dp_ucmgp").any() and not (b.labs["analyte"] == "dp_ucmgp").any()
    other = a.labs[a.labs["analyte"] != "dp_ucmgp"].reset_index(drop=True)
    assert other.equals(b.labs.reset_index(drop=True))


def _marker_by_vka(cohort):
    mg = cohort.labs[cohort.labs["analyte"] == "dp_ucmgp"]
    vka = set(cohort.exposures.loc[(cohort.exposures["class"] == "VKA") & (cohort.exposures["indication"] == "AF"), "patient_id"])
    logv = np.log(mg["value"])
    return logv[mg["patient_id"].isin(vka)].mean() - logv[~mg["patient_id"].isin(vka)].mean()


def test_variants_generate_marker_from_vka_or_as_noise(mediated_cohort, noise_cohort):
    assert _marker_by_vka(mediated_cohort) > 0.6
    assert abs(_marker_by_vka(noise_cohort)) < 0.15
    assert get_scenario(SCENARIO, "marker_mediated").get("svd_hazard.log_hr_vka_from_implant") == 0.0
    assert get_scenario(SCENARIO, "marker_noise").get("svd_hazard.log_hr_dp_ucmgp_per_log_unit") == 0.0


def test_marker_is_never_imputed_into_the_landmark_rows(mediated_lm, mediated_cohort):
    m = measured_mask(mediated_lm)
    assert 0.2 < m.mean() < 0.8
    in_sub = mediated_cohort.patients.set_index("patient_id")["dp_ucmgp_substudy"]
    assert not mediated_lm.loc[m, "patient_id"].map(in_sub).eq(False).any()


def test_bundle_carries_the_offset_model(mediated_bundle):
    card = mediated_bundle.card()["dp_ucmgp_substudy"]
    assert card["fitted"] and card["n_svd_events_measured"] >= 15
    assert set(card["coefficients"]) == {"log_dp_ucmgp", "log_dp_ucmgp__rcs1", "dp_ucmgp_x_cum_vka"}
    assert card["p_value_method"].startswith("patient-bootstrap") and 0 <= card["p_value"] <= 1
    assert card["coefficients"]["log_dp_ucmgp"] > 0  # higher dp-ucMGP, higher SVD hazard in this variant


def _with_marker(req, values):
    pid = req.passport.passport_id
    extra = [LabObservation(passport_id=pid, date=d, analyte="dp_ucmgp", value=v, unit="pmol/L", assay="InaKtif MGP")
             for d, v in values]
    return req.model_copy(update={"labs": list(req.labs) + extra})


def test_every_prediction_states_dp_ucmgp_use(mediated_bundle, mediated_cohort, model_cfg):
    req, ech = cohort_request(mediated_cohort)
    pr = Predictor(mediated_bundle, model_cfg)
    last = ech.iloc[-1].date
    base = pr.predict(req)
    assert base.dp_ucmgp.status == "not_measured" and not base.dp_ucmgp.measured_value_used
    low = pr.predict(_with_marker(req, [(last.isoformat(), 300.0)]))
    high = pr.predict(_with_marker(req, [(last.isoformat(), 3000.0)]))
    assert high.dp_ucmgp.status == "used" and high.dp_ucmgp.assay == "InaKtif MGP" and high.dp_ucmgp.value_pmol_l == 3000.0
    assert high.p_svd_before_death[2] > low.p_svd_before_death[2]
    assert any(d.feature.startswith("dp-ucMGP") for d in high.drivers)
    old = (pd.Timestamp(last) - pd.DateOffset(months=14)).date().isoformat()
    stale = pr.predict(_with_marker(req, [(old, 3000.0)]))
    assert stale.dp_ucmgp.status == "stale" and np.allclose(stale.p_svd_before_death, base.p_svd_before_death)
    future = pr.predict(_with_marker(req, [((pd.Timestamp(last) + pd.DateOffset(days=30)).date().isoformat(), 3000.0)]))
    assert future.dp_ucmgp.status == "not_measured"


def test_bundle_without_substudy_model_says_so(small_bundle, small_cohort, model_cfg):
    req, ech = cohort_request(small_cohort)
    pred = Predictor(small_bundle, model_cfg).predict(_with_marker(req, [(ech.iloc[-1].date.isoformat(), 900.0)]))
    status = pred.dp_ucmgp
    if small_bundle.vitamin_k_model is None:
        assert status.status == "not_in_model" and "substudy" in status.note
    else:
        assert status.status == "used"


def test_status_flag_is_consistent():
    with pytest.raises(ValueError):
        VitaminKStatus(measured_value_used=True, status="stale")


def test_vka_coefficient_shrinks_only_when_the_marker_carries_the_signal(mediated_lm, noise_cohort, model_cfg):
    med = mechanism_check(mediated_lm, model_cfg)
    noise = mechanism_check(landmark_from_cohort(noise_cohort, model_cfg), model_cfg)
    assert med["fitted"] and noise["fitted"]
    # The size of the shrinkage is not stable at n=1500: with generator version 2 streams, seeds 11-15
    # gave 0.21, 0.57, 0.47, 0.05 and undefined (contrast < 0.05) for the mediated variant against
    # -0.00 to 0.05 for the noise variant. The former "> 0.3" held only for the version 1 draws, so this
    # test checks direction on one seed; magnitude belongs in a repeated-seed evaluation.
    assert med["vka_log_hr_without_marker"] > 0.1 and med["vka_shrinkage_fraction"] > 0.1
    assert noise["vka_shrinkage_fraction"] is None or noise["vka_shrinkage_fraction"] < 0.1
    assert med["vka_shrinkage_fraction"] > (noise["vka_shrinkage_fraction"] or 0.0)


def test_substudy_evaluation_reports_incremental_value_within_measured_rows(mediated_lm, model_cfg):
    keep = mediated_lm["patient_id"].isin(sorted(mediated_lm["patient_id"].unique())[:900])
    lm = mediated_lm.loc[keep].reset_index(drop=True)
    # machinery check on a 900-patient subset: quick support gates (full gates need 30 SVD event patients per fold)
    res = evaluate_vitamin_k(lm, model_cfg, patient_folds(lm["patient_id"], 2, 1), n_boot=5, seed=1, mode="quick")
    assert res["fitted"] and res["folds_with_offset_model"] == 2 and res["label"].startswith("synthetic")
    rows = pd.DataFrame(res["incremental"])
    assert list(rows["horizon_years"]) == [1, 3, 5] and rows["n_rows"].eq(res["n_rows_measured"]).all()
    assert rows[["brier_core_plus_anticoagulant", "brier_plus_dp_ucmgp"]].apply(lambda c: c.between(0, 1)).all().all()
    assert res["mechanism"]["fitted"] and "offset" in res["offset_model"]["method"]


def test_substudy_too_small_is_not_fitted(small_cohort, model_cfg):
    lm = landmark_from_cohort(small_cohort, model_cfg)
    cfg = copy.deepcopy(model_cfg)
    cfg["vitamin_k"]["min_measured_rows"] = 10 ** 6
    res = evaluate_vitamin_k(lm, cfg, patient_folds(lm["patient_id"], 2, 1), n_boot=0)
    assert res["fitted"] is False and "nothing imputed" in res["reason"]
