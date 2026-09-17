"""Cumulative incidence combination and the evaluation metrics."""
import numpy as np

from kairos.evaluation.metrics import (
    aalen_johansen,
    auc_ipcw,
    brier_ipcw,
    calibration_curve,
    calibration_slope_prob_legacy_v1,
    evaluate_predictions,
)
from kairos.modelling.cif import at_times, combine_cause_specific, probabilities_at


def test_cif_sums_to_one_and_is_monotone():
    grid = np.linspace(0, 5, 261)
    n = 4
    rng = np.random.default_rng(0)
    H = {c: np.cumsum(rng.uniform(0, 0.02, size=(n, len(grid))), axis=1) for c in ("svd", "death", "replacement")}
    for c in H:
        H[c][:, 0] = 0
    cifs = combine_cause_specific(H, grid)
    total = cifs["svd"] + cifs["death"] + cifs["replacement"] + cifs["alive_intact"]
    assert np.allclose(total, 1.0, atol=1e-9)
    for c in ("svd", "death", "replacement"):
        assert np.all(np.diff(cifs[c], axis=1) >= -1e-12)
    assert np.all(np.diff(cifs["alive_intact"], axis=1) <= 1e-12)
    probs = probabilities_at(cifs, grid, [1, 3, 5], 1.0)
    assert probs["svd"].shape == (n, 3)
    assert np.allclose(probs["svd_near_term"], probs["svd"][:, 0])
    assert np.all(np.diff(probs["svd"], axis=1) >= 0)


def test_zero_hazard_gives_alive_intact_one():
    grid = np.linspace(0, 5, 11)
    H = {c: np.zeros((2, 11)) for c in ("svd", "death", "replacement")}
    cifs = combine_cause_specific(H, grid)
    assert np.allclose(cifs["alive_intact"], 1.0) and np.allclose(cifs["svd"], 0.0)
    assert at_times(cifs["alive_intact"], grid, [0.5, 4.9]).shape == (2, 2)


def test_metrics_without_censoring_reduce_to_plain_versions():
    rng = np.random.default_rng(1)
    n = 400
    time = rng.uniform(0.1, 6, n)
    event = np.where(time <= 3, rng.choice([1, 2], size=n, p=[0.6, 0.4]), 0)
    event[time > 3] = 0  # everyone beyond the horizon is simply event-free by 3 years
    y = ((time <= 3) & (event == 1)).astype(float)
    pred = np.clip(y * 0.7 + 0.15 + rng.normal(0, 0.05, n), 0, 1)
    tau = 3.0
    # no censoring before tau (all times > tau are administratively "after"), so weights are 1
    assert abs(brier_ipcw(pred, time, event, 1, tau) - np.mean((y - pred) ** 2)) < 1e-9
    assert abs(aalen_johansen(time, event, 1, tau) - y.mean()) < 1e-9
    assert auc_ipcw(y, time, event, 1, tau) == 1.0
    assert abs(calibration_slope_prob_legacy_v1(y, time, event, 1, tau) - 1.0) < 1e-9
    m = evaluate_predictions(pred, time, event, 1, tau)
    assert m["n_events"] == int(y.sum()) and 0 <= m["brier"] <= 1 and m["ipa"] > 0
    curve = calibration_curve(pred, time, event, 1, tau, n_bins=5)
    assert len(curve) >= 2 and set(curve.columns) >= {"mean_predicted", "observed", "n"}


def test_ipcw_handles_censoring():
    rng = np.random.default_rng(2)
    n = 300
    time = rng.exponential(3, n)
    event = rng.choice([0, 1, 2], size=n, p=[0.3, 0.4, 0.3])
    pred = rng.uniform(0, 0.5, n)
    m = evaluate_predictions(pred, time, event, 1, 2.0)
    assert np.isfinite(m["brier"]) and 0 < m["brier"] < 1 and np.isfinite(m["auc"])


def test_administrative_censoring_at_the_horizon_counts_as_event_free():
    """Landmark rows are capped at the horizon with event 0; at that horizon they are controls."""
    time = np.array([1.0, 2.0, 5.0, 5.0, 5.0, 5.0])
    event = np.array([1, 2, 0, 0, 0, 0])
    pred = np.array([0.9, 0.1, 0.1, 0.1, 0.2, 0.1])
    y = np.array([1, 0, 0, 0, 0, 0], dtype=float)
    assert abs(brier_ipcw(pred, time, event, 1, 5.0) - np.mean((y - pred) ** 2)) < 1e-9
    assert auc_ipcw(pred, time, event, 1, 5.0) == 1.0
    assert np.isfinite(evaluate_predictions(pred, time, event, 1, 5.0)["auc"])
