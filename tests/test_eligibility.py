"""CR-06: marker and module eligibility fitted on training rows only (detailed design WP-A6)."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kairos.evaluation.ladder import evaluate_ladder, patient_folds
from kairos.extraction.schema import LabObservation
from kairos.modelling.features import FeaturePipeline
from kairos.modelling.modules import fit_eligibility, ladder_steps
from kairos.modelling.predictor import ModelBundle, Predictor
from kairos.modelling.train import fit_step, landmark_from_cohort
from kairos.simulation.generators import generate_cohort
from kairos.simulation.scenarios import get_scenario
from tests.helpers import cohort_request


@pytest.fixture(scope="module")
def lm(small_cohort, model_cfg):
    return landmark_from_cohort(small_cohort, model_cfg)


def _blocks(cfg, step):
    return dict(ladder_steps(cfg))[step]


def test_eligible_anchor_does_not_enable_an_absent_companion(lm, model_cfg):
    df = lm.assign(hba1c=np.nan)
    pipe, _, feats, dropped, _, man = fit_step(df, _blocks(model_cfg, "core_plus_renal_metabolic"), model_cfg)
    assert "egfr" in feats and "hba1c" not in feats and not dropped
    assert not any(c.startswith("hba1c") for c in pipe.output_columns_)
    assert man.markers["hba1c"].reason == "absent" and man.modules["biomarker_renal_metabolic"]["status"] == "partial"
    # the generator never measures calcium, PTH or ALP: an eligible phosphate must not bring them in
    pipe, _, feats, _, _, man = fit_step(lm, _blocks(model_cfg, "core_plus_mineral"), model_cfg)
    assert "phosphate" in feats and not {"calcium_corrected", "pth", "alp"} & set(feats)
    assert not any(c.split("__")[0] in ("calcium_corrected", "pth", "alp") for c in pipe.output_columns_)
    assert man.modules["biomarker_mineral"]["excluded"] == {"calcium_corrected": "absent", "pth": "absent", "alp": "absent"}


def test_patient_level_denominator_and_constant_marker(lm, model_cfg):
    pids = sorted(lm["patient_id"].unique())
    frequent = lm.groupby("patient_id").size().sort_values(ascending=False)
    # a marker measured only in the most frequently seen 10% of patients: many rows, few patients
    keep = set(frequent.index[: max(1, len(pids) // 10)])
    df = lm.assign(lpa=np.where(lm["patient_id"].isin(keep), 50.0, np.nan))
    man = fit_eligibility(df, model_cfg)
    m = man.markers["lpa"]
    assert m.patient_fraction < 0.2 <= m.row_fraction or m.patient_fraction < m.row_fraction
    assert not m.eligible and m.reason == "below_cutoff"
    assert man.modules["biomarker_lipid"]["status"] == "unavailable"
    const = fit_eligibility(lm.assign(ntprobnp=400.0), model_cfg)
    assert const.markers["ntprobnp"].reason == "constant" and const.modules["biomarker_cardiac"]["status"] == "unavailable"


def test_pipeline_refuses_to_manufacture_an_absent_marker(lm):
    with pytest.raises(ValueError, match="no non-missing training value"):
        FeaturePipeline(features=["egfr", "pth"]).fit(lm.assign(pth=np.nan))


def test_held_out_fold_cannot_change_training_decisions(small_cohort, model_cfg, lm):
    folds = patient_folds(lm["patient_id"], 2, int(model_cfg["evaluation"]["seed"]))
    mutated = lm.copy()
    held = folds == 1
    mutated.loc[held, "phosphate"] = np.nan
    mutated.loc[held, "egfr"] = 999.0
    mutated.loc[held, "design_class"] = "made-up class"
    kw = dict(n_splits=2, n_boot=0, steps=["core_plus_mineral"])
    a = evaluate_ladder(small_cohort, model_cfg, lm=lm, **kw).fold_fits["core_plus_mineral"]
    b = evaluate_ladder(small_cohort, model_cfg, lm=mutated, **kw).fold_fits["core_plus_mineral"]
    assert 1 in a and 0 in a
    # fold 1 is held out: its fit uses fold 0 rows only and must be identical
    assert a[1] == b[1]
    # fold 0 trains on the mutated rows, so its decisions do change (the test can detect a difference)
    assert a[0]["eligibility"]["modules"]["biomarker_mineral"]["status"] != "unavailable"
    assert b[0]["eligibility"]["modules"]["biomarker_mineral"]["status"] == "unavailable"


def test_equivalent_ladder_steps_are_identified(model_cfg):
    cohort = generate_cohort(get_scenario("biomarker_information", "unmeasured"), n=250, seed=4, namespace="quick")
    res = evaluate_ladder(cohort, model_cfg, n_splits=2, n_boot=0, steps=["core", "core_plus_lipid"])
    r = res.results.drop_duplicates("step").set_index("step")
    assert r.loc["core_plus_lipid", "modules_dropped"] == "biomarker_lipid"
    assert r.loc["core_plus_lipid", "equivalent_to"] == "core" and r.loc["core", "equivalent_to"] == ""
    r = res.results.dropna(subset=["n_event_patients"])
    assert len(r) and (r["n_event_patients"] <= r["n_events"]).all()


def test_bundle_round_trip_reproduces_eligibility_and_predictions(small_bundle, small_cohort, model_cfg):
    req, _ = cohort_request(small_cohort)
    before = Predictor(small_bundle, model_cfg).predict(req)
    with tempfile.TemporaryDirectory() as td:
        loaded = ModelBundle.load(small_bundle.save(Path(td) / "b.joblib"))
    assert loaded.eligibility == small_bundle.eligibility and loaded.eligibility["modules"]
    assert loaded.card()["excluded_features"] == small_bundle.card()["excluded_features"]
    assert {"calcium_corrected", "pth", "alp"} <= set(loaded.excluded_features)
    after = Predictor(loaded, model_cfg).predict(req)
    assert after.p_svd_before_death == before.p_svd_before_death and after.p_svd_12m == before.p_svd_12m
    # a new measurement of an excluded marker is ignored, not silently activated
    extra = req.model_copy(update={"labs": list(req.labs) + [
        LabObservation(passport_id=req.passport.passport_id, date=req.prediction_time, analyte="pth", value=80.0)]})
    assert Predictor(loaded, model_cfg).predict(extra).p_svd_before_death == before.p_svd_before_death
    assert isinstance(pd.DataFrame(loaded.eligibility["markers"]).T, pd.DataFrame)
