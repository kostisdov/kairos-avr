"""CR-04: support gates, coverage and explicit unsupported statuses."""
import numpy as np
import pandas as pd

from kairos.evaluation.ladder import predict_states
from kairos.evaluation.support import (
    bootstrap_gate,
    common_evaluable_mask,
    coverage,
    gates_for,
    horizon_counts,
    metric_gate,
)
from kairos.modelling.train import fit_step, landmark_from_cohort


def test_metric_gates_apply_per_metric(model_cfg):
    gates = gates_for(model_cfg, "full")
    counts = {"case_patients": 25, "control_patients": 400, "rows_observed_through_horizon": 500}
    assert metric_gate("auc", counts, 0.8, gates).ok
    slope = metric_gate("cal_slope_logit", counts, 0.8, gates)
    assert not slope.ok and slope.gate == "slope_min_events"
    few = metric_gate("brier", {**counts, "case_patients": 3}, 0.8, gates)
    assert few.status == "insufficient_events" and few.exploratory
    cens = metric_gate("brier", counts, 0.01, gates)
    assert cens.status == "censoring_support_failure"
    fu = metric_gate("auc", {**counts, "rows_observed_through_horizon": 2}, 0.8, gates)
    assert fu.status == "insufficient_followup"
    quick = metric_gate("auc", counts, 0.8, gates_for(model_cfg, "quick"))
    assert quick.ok and quick.exploratory                   # quick decisions can never support claims


def test_horizon_counts_use_unique_patients():
    time = np.array([1.0, 1.0, 2.0, 5.0, 5.0, 5.0])
    event = np.array([1, 1, 2, 0, 0, 0])
    pids = np.array(["a", "a", "b", "c", "c", "d"])
    c = horizon_counts(time, event, pids, "svd", 3.0)
    # the competing death (b) is a control with known status, as are c and d
    assert c["case_patients"] == 1 and c["case_rows"] == 2 and c["control_patients"] == 3


def test_bootstrap_gate_coverage_and_common_mask(model_cfg):
    gates = gates_for(model_cfg, "full")
    assert not bootstrap_gate(200, 90, gates).ok and not bootstrap_gate(200, 150, gates).ok
    assert bootstrap_gate(200, 170, gates).ok
    cov = coverage(np.array([True, False, True, False]), ["a", "a", "b", "c"])
    assert cov["row_coverage"] == 0.5 and np.isclose(cov["patient_coverage"], 2 / 3)
    m = common_evaluable_mask([True, True, False], [True, False, True])
    assert list(m) == [True, False, False]


def test_unsupported_rows_get_reasons_not_probabilities(small_cohort, model_cfg):
    lm = landmark_from_cohort(small_cohort, model_cfg)
    pipe, cox, *_ = fit_step(lm, ["core_static", "core_time"], model_cfg, "quick")
    test = lm.head(6).copy()
    test["route"] = ["SAVR", "TAVR", None, "unknown", "SAVR", "TAVR"]
    probs, reasons = predict_states(pipe, cox, test, [1.0, 3.0, 5.0], 1.0)
    assert list(reasons[2:4]) == ["unknown_device_or_route", "unknown_device_or_route"]
    assert np.isnan(probs["svd"][2:4]).all() and np.isfinite(probs["svd"][[0, 1, 4, 5]]).all()
    tot = sum(probs[s] for s in ("svd", "death", "replacement", "alive_intact"))
    assert np.allclose(tot[[0, 1, 4, 5]], 1.0, atol=1e-10)


def test_full_gates_on_a_small_cohort_report_unmet_support(small_cohort, model_cfg):
    lm = landmark_from_cohort(small_cohort, model_cfg)
    _, cox, *_ = fit_step(lm, ["core_static", "core_time"], model_cfg, "full")
    s = cox.support_summary()
    assert s["gates"]["mode"] == "full"
    statuses = {c: v["status"] for c, v in s["causes"].items()}
    assert statuses["death"] == "ok" and statuses["replacement"] != "ok"
    assert all(isinstance(v["event_patients"], int) for v in s["causes"].values())
    assert pd.Series(statuses).isin(["ok", "reduced_baseline_only", "insufficient_events", "fit_failed"]).all()
