"""CR-02 observation-process and onset sensitivity, and the WP-B2 sizing plan guard."""
import copy

import numpy as np
import pandas as pd
import pytest
import yaml

from kairos.evaluation.sensitivity import (
    allocate_events,
    missing_onset_events,
    onset_intervals,
    oracle_events,
    paired_visit_cohorts,
    run_observation_sensitivity,
    summarise,
    visit_intensity_weights,
)
from kairos.evaluation.sizing import (
    load_plan,
    pilot_scenario,
    plan_status,
    write_plan,
)
from kairos.modelling.train import landmark_build_from_cohort, landmark_from_cohort
from kairos.simulation.generators import generate_cohort
from kairos.simulation.scenarios import get_scenario
from kairos.simulation.truth import TRUTH_COLUMNS


@pytest.fixture(scope="module")
def cohort():
    return generate_cohort(get_scenario("gradual_stenotic"), n=500, seed=12, namespace="quick")


def test_intervals_use_only_adjudication_evidence(cohort):
    iv = onset_intervals(cohort.events, cohort.echoes, cohort.patients)
    adj = cohort.events.dropna(subset=["svd_adjudicated_date"]).set_index("patient_id")["svd_adjudicated_date"]
    assert len(iv) == len(adj) and (iv.set_index("patient_id")["interval_hi"] == adj.reindex(iv["patient_id"]).to_numpy()).all()
    d = iv[iv["defensible"]]
    assert len(d) and (d["interval_lo"] < d["interval_hi"]).all()
    echo_dates = set(zip(cohort.echoes["patient_id"], cohort.echoes["date"]))
    assert all((p, lo) in echo_dates for p, lo in zip(d["patient_id"], d["interval_lo"]))   # an observed study, not truth


def test_allocations_stay_inside_interval_and_before_terminal_events(cohort):
    iv = onset_intervals(cohort.events, cohort.echoes, cohort.patients).set_index("patient_id")
    rng = np.random.default_rng(0)
    ends = cohort.events.set_index("patient_id")
    for how in ("left", "midpoint", "right", "uniform"):
        o = allocate_events(cohort.events, iv.reset_index(), how, rng)
        for pid, d in o["svd_event_date"].items():
            r = iv.loc[pid]
            if r["defensible"]:
                assert r["interval_lo"] < d <= r["interval_hi"]
            assert d <= ends.loc[pid, "end_followup_date"]
        lm = landmark_build_from_cohort(cohort, {**_cfg(), }, "primary", event_override=o).rows
        dates = lm.merge(o.reset_index(), on="patient_id")
        assert (dates["landmark_date"] < dates["svd_event_date"]).all()        # eligibility rebuilt per allocation
    right = allocate_events(cohort.events, iv.reset_index(), "right")
    assert (right["svd_event_date"] == iv["interval_hi"].reindex(right.index)).all()


def _cfg():
    from kairos.modelling.modules import load_model_config
    return load_model_config()


def test_oracle_and_missing_onset_never_become_features(cohort):
    cfg = _cfg()
    for o in (oracle_events(cohort.events, cohort.truth), missing_onset_events(cohort.events, cohort.truth)):
        lm = landmark_build_from_cohort(cohort, cfg, "primary", event_override=o).rows
        assert not (TRUTH_COLUMNS & set(lm.columns))
    miss = missing_onset_events(cohort.events, cohort.truth)
    assert set(miss.index) == set(cohort.truth.loc[cohort.truth["missed_crossing"], "patient_id"])
    # the oracle changes the risk set: rows at or after the noise-free crossing disappear
    o = oracle_events(cohort.events, cohort.truth)
    lm = landmark_build_from_cohort(cohort, cfg, "primary", event_override=o).rows
    m = lm.merge(o.reset_index().dropna(), on="patient_id")
    assert (m["landmark_date"] < m["svd_event_date"]).all()


def test_visit_weights_are_fitted_on_training_rows(cohort):
    cfg = _cfg()
    lm = landmark_from_cohort(cohort, cfg)
    pids = lm["patient_id"].unique()
    tr, te = lm[lm["patient_id"].isin(pids[:300])], lm[lm["patient_id"].isin(pids[300:])]
    w, diag = visit_intensity_weights(tr, te)
    assert len(w) == len(te) and np.all(w > 0) and diag["truncation_bounds"][0] <= w.min()
    mutated = te.assign(current_gradient=te["current_gradient"] * 3)
    w2, diag2 = visit_intensity_weights(tr, mutated)
    assert diag2["coefficients"] == diag["coefficients"]          # test rows never change the fitted model


def test_paired_visit_cohorts_and_full_run_are_reproducible():
    spec = get_scenario("gradual_stenotic")
    reg, inf = paired_visit_cohorts(spec, 250, 5)
    assert reg.truth[["patient_id", "threshold_crossing_date"]].equals(inf.truth[["patient_id", "threshold_crossing_date"]])
    assert reg.manifest["five_year_attendance"] >= inf.manifest["five_year_attendance"] - 0.05
    small = generate_cohort(spec, n=300, seed=8, namespace="quick")
    cfg = _cfg()
    a = run_observation_sensitivity(small, cfg, ["reference"], n_splits=2, uniform_draws=1)
    b = run_observation_sensitivity(small, cfg, ["reference"], n_splits=2, uniform_draws=1)
    r = a.results
    assert {"primary", "oracle", "missing_onset", "interval", "visits_regular", "visits_informative"} <= set(r["analysis"])
    assert {"left", "midpoint", "right", "uniform"} <= set(r["allocation"].dropna())
    assert "inverse visit intensity" in set(r["weighting"]) and "patient balanced" in set(r["weighting"])
    assert a.diagnostics["visit_process"]["identical_latent_truth"] and a.diagnostics["n_patients_common"] > 0
    cols = ["analysis", "allocation", "weighting", "horizon_years", "brier", "observed"]
    pd.testing.assert_frame_equal(a.results[cols].reset_index(drop=True), b.results[cols].reset_index(drop=True))
    assert "interval-censored likelihood" in a.label and len(summarise(a))


def test_sizing_plan_is_advisory_and_labels_evidence(tmp_path):
    cfg = _cfg()
    p = pilot_scenario("gradual_stenotic", None, cfg, n=400, dev_seeds=1, n_splits=2, benchmark_step="reference")
    assert p["dev_seeds"] == [9100] and any("pooled Cox" in k for k in p["required_n"])
    path = write_plan([p], cfg, tmp_path / "plan.yaml")
    plan = load_plan(path)
    assert plan["frozen"] is False and "gradual_stenotic" in plan["development"]["scenarios"]
    spec = get_scenario("gradual_stenotic")
    manifest = {"namespace": "full", "key": "gradual_stenotic", "n_requested": 3000, "seed": plan["development"]["seed"],
                "effective_config_hash": spec.effective_config_hash}
    assert plan_status(manifest, plan)["status"] == "unfrozen_plan"
    frozen = copy.deepcopy(plan)
    frozen["frozen"] = True
    frozen["development"]["scenarios"]["gradual_stenotic"]["n"] = 3000
    assert plan_status(manifest, frozen) == {"status": "frozen_match", "evidence": "prespecified", "reason": "matches the frozen plan"}
    mismatch = plan_status({**manifest, "n_requested": 2500}, frozen)
    assert mismatch["status"] == "plan_mismatch" and mismatch["evidence"] == "exploratory" and "2500" in mismatch["reason"]
    assert plan_status({**manifest, "seed": 1}, frozen)["status"] == "plan_mismatch"
    assert plan_status({**manifest, "namespace": "quick"}, None)["status"] == "quick"
    assert plan_status(manifest, None)["status"] == "no_plan"
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["plan_hash"]
