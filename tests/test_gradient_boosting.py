"""CR-08: the gradient boosting family behind the shared adapter and hazard contract."""
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kairos.modelling.base import UnsupportedFitError, UnsupportedRouteError
from kairos.modelling.cif import STATES, combine_cause_specific, probabilities_at, validate_hazards
from kairos.modelling.gradient_boosting import CauseSpecificGradientBoostingModel
from kairos.modelling.predictor import ModelBundle, Predictor
from kairos.modelling.train import fit_step, fit_vitamin_k, landmark_from_cohort, train_bundle
from tests.helpers import cohort_request

BLOCKS = ["core_static", "core_reference_echo", "core_serial_echo", "core_time"]
HP = {"n_estimators": 40, "learning_rate": 0.1, "max_depth": 2}


@pytest.fixture(scope="module")
def lm(small_cohort, model_cfg):
    return landmark_from_cohort(small_cohort, model_cfg)


@pytest.fixture(scope="module")
def fitted(lm, model_cfg):
    return fit_step(lm, BLOCKS, model_cfg, "quick", "gradient_boosting", HP, keep_training_data=True)


def _X(pipe, model, rows):
    X = pipe.transform(rows)
    for s in model.strata:
        X[s] = rows[s].astype(str).to_numpy()
    return X


def test_contract_closure_and_support(fitted, lm, model_cfg):
    pipe, gb, feats, *_ = fitted
    assert gb.family == "gradient_boosting" and gb.supported_routes == ["SAVR", "TAVR"]
    assert all(("svd", k) in gb.models_ for k in gb.stratum_keys_)          # one model per route and cause
    X = _X(pipe, gb, lm.head(200))
    H = gb.cumulative_hazards(X)
    clean, diag = validate_hazards(H, gb.grid)
    c = combine_cause_specific(H, gb.grid)
    assert np.max(np.abs(sum(c[s] for s in STATES) - 1)) < 1e-10
    p = probabilities_at(c, gb.grid, [1, 3, 5], 1.0)
    assert np.all(np.diff(p["svd"], axis=1) >= -1e-12) and {1.0, 3.0, 5.0} <= set(gb.grid)
    bad = X.head(2).copy()
    bad["route"] = ["unknown", "nan"]
    with pytest.raises(UnsupportedRouteError):
        gb.cumulative_hazards(bad)


def test_hazards_equal_the_library_and_staged_equals_refit(fitted, lm, model_cfg):
    pipe, gb, *_ = fitted
    est = gb.models_[("death", "SAVR")]
    Xtr, _, _ = gb.training_["SAVR"]
    times = est.unique_times_[:60]
    theirs = np.array([f(times) for f in est.predict_cumulative_hazard_function(Xtr[:15])])
    X = _X(pipe, gb, lm)
    rows = np.flatnonzero(gb._keys(X) == "SAVR")[:15]
    ours = gb.cumulative_hazards(X.iloc[rows], times)["death"]
    assert np.max(np.abs(ours - theirs)) < 1e-12
    sub = X.iloc[:150]
    staged = gb.staged_state_probabilities(sub, [20, 40], [1.0, 3.0, 5.0], 1.0)
    _, gb20, *_ = fit_step(lm, BLOCKS, model_cfg, "quick", "gradient_boosting", {**HP, "n_estimators": 20})
    fresh = probabilities_at(combine_cause_specific(gb20.cumulative_hazards(sub), gb20.grid), gb20.grid, [1.0, 3.0, 5.0], 1.0)
    assert all(np.array_equal(staged[20][s], fresh[s]) for s in STATES)


def test_same_information_as_cox_without_scaling_or_splines(lm, model_cfg):
    pc, cox, feats_c, *_ = fit_step(lm, BLOCKS, model_cfg, "quick", "cox")
    pg, gb, feats_g, *_ = fit_step(lm, BLOCKS, model_cfg, "quick", "gradient_boosting", HP)
    assert feats_c == feats_g and pc.medians_ == pg.medians_ and pc.levels_ == pg.levels_
    spline = re.compile(r"__s\d+$")
    assert any(spline.search(c) for c in pc.output_columns_) and not any(spline.search(c) for c in pg.output_columns_)
    raw = pg.transform(lm.head(5))["age_at_implant"].to_numpy()
    assert np.allclose(raw, lm.head(5)["age_at_implant"].to_numpy())       # unscaled values reach the trees


def test_determinism(lm, model_cfg):
    a = fit_step(lm, BLOCKS, model_cfg, "quick", "gradient_boosting", HP)
    b = fit_step(lm, BLOCKS, model_cfg, "quick", "gradient_boosting", HP)
    X = _X(a[0], a[1], lm.head(100))
    assert all(np.array_equal(a[1].cumulative_hazards(X)[c], b[1].cumulative_hazards(X)[c]) for c in ("svd", "death"))


def test_route_level_support_is_explicit():
    rng = np.random.default_rng(1)
    n = 240
    df = pd.DataFrame({"x": rng.normal(size=n), "route": np.where(np.arange(n) % 2 == 0, "SAVR", "TAVR"),
                       "time": rng.uniform(0.1, 5, n), "patient_id": [f"p{i}" for i in range(n)]})
    df["event"] = 2
    df.loc[df.index % 3 == 0, "event"] = 0
    df.loc[[0, 2, 4], "event"] = 1                    # 3 SVD patients, all SAVR
    df.loc[[1, 3, 5, 7, 9, 11], "event"] = 3          # 6 replacement patients, all TAVR
    m = CauseSpecificGradientBoostingModel(n_estimators=10, gates={"fit_min_unique_events": 30, "baseline_min_unique_events": 3,
                                                                   "min_at_risk": 5, "mode": "quick"})
    m.fit(df[["x"]], df["time"], df["event"], ["x"], df[["route"]], patient_ids=df["patient_id"])
    s = m.support_
    assert s["death"]["strata"]["SAVR"]["status"] == "ok"
    assert s["svd"]["strata"]["SAVR"]["status"] == "reduced_baseline_only" and not s["svd"]["strata"]["TAVR"]["supported"]
    assert s["replacement"]["strata"]["TAVR"]["status"] == "reduced_baseline_only"
    with pytest.raises(UnsupportedFitError):
        m.cumulative_hazards(df.loc[[1], ["x", "route"]])
    assert m.supported_routes == []


def test_bundle_round_trip_prediction_and_offset(small_cohort, lm, model_cfg):
    b = train_bundle(small_cohort, model_cfg, "core_plus_anticoagulant", "0.1.0+test", lm=lm, family="gradient_boosting",
                     hyperparameters=HP)
    card = b.card()
    assert card["family"] == "gradient_boosting" and card["hyperparameters"]["n_estimators"] == 40 and not b.model.training_
    req, _ = cohort_request(small_cohort)
    before = Predictor(b, model_cfg).predict(req)
    with tempfile.TemporaryDirectory() as td:
        loaded = ModelBundle.load(b.save(Path(td) / "gb.joblib"))
    after = Predictor(loaded, model_cfg).predict(req)
    assert after.p_svd_before_death == before.p_svd_before_death and after.p_death_before_svd == before.p_death_before_svd
    tot = np.array(after.p_svd_before_death) + after.p_death_before_svd + np.array(after.p_alive_intact) + after.p_replaced_non_svd
    assert np.allclose(tot, 1, atol=1e-9)
    vk = fit_vitamin_k(lm, ["anticoagulant"], [], b.pipeline, b.model, model_cfg)   # offset from the boosted SVD log risk
    assert isinstance(vk.reason, str)


def test_fast_cox_loss_equals_the_library_and_fits_are_unchanged():
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis, survival_loss
    from sksurv.ensemble._coxph_loss import coxph_loss, coxph_negative_gradient
    from sksurv.util import Surv

    from kairos.modelling.gradient_boosting import (
        coxph_loss_fast,
        coxph_negative_gradient_fast,
        fast_cox_loss,
    )

    rng = np.random.default_rng(4)
    for ties in (False, True):
        t = rng.exponential(3, 700)
        t = np.round(t, 1) if ties else t
        e = (rng.uniform(size=700) < 0.3).astype(np.uint8)
        f = rng.normal(size=700)
        assert abs(coxph_loss(e, t, f) - coxph_loss_fast(e, t, f)) < 1e-9
        assert np.max(np.abs(coxph_negative_gradient(e, t, f) - coxph_negative_gradient_fast(e, t, f))) < 1e-12
    X = rng.normal(size=(600, 8)).astype(np.float32)
    y = Surv.from_arrays(rng.uniform(size=600) < 0.4, np.round(rng.exponential(2, 600), 2))
    kw = dict(loss="coxph", n_estimators=25, max_depth=2, min_samples_leaf=10, random_state=0)
    lib = GradientBoostingSurvivalAnalysis(**kw).fit(X, y)
    with fast_cox_loss():
        fast = GradientBoostingSurvivalAnalysis(**kw).fit(X, y)
    assert np.max(np.abs(lib.predict(X) - fast.predict(X))) < 1e-9
    assert survival_loss.coxph_loss is coxph_loss                  # the swap is scoped to the fit
