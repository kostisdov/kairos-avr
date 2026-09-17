"""CR-07: outcome definitions, IPCW weights, logistic recalibration, Brier decomposition and the
patient-cluster bootstrap."""
import numpy as np
import pytest

from kairos.evaluation.metrics import (
    KMCensoring,
    bootstrap_by_patient,
    brier_decomposition,
    evaluate_predictions,
    ipcw_logistic_calibration,
    ipcw_weights,
    outcome,
)


def _expit(x):
    return 1 / (1 + np.exp(-x))


def _logit(p):
    return np.log(p / (1 - p))


def _competing_sample(seed, n=3000, tau=3.0):
    """Constant cause-specific hazards with a covariate on cause 1; true CIF_1(tau) known per subject."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    l1, l2 = 0.08 * np.exp(0.9 * x), np.full(n, 0.12)
    tot = l1 + l2
    t_ev = rng.exponential(1 / tot)
    cause = np.where(rng.uniform(size=n) < l1 / tot, 1, 2)
    c = rng.uniform(0.5, 8.0, size=n)
    time = np.minimum(t_ev, c)
    event = np.where(t_ev <= c, cause, 0)
    true = l1 / tot * (1 - np.exp(-tot * tau))
    return time, event, true, tau


def test_competing_events_are_known_controls_and_early_censoring_is_unknown():
    time = np.array([1.0, 2.0, 0.5, 4.0, 3.0])
    event = np.array([1, 2, 0, 0, 0])
    y, known = outcome(time, event, "svd", 3.0)
    assert list(y) == [1, 0, 0, 0, 0] and list(known) == [True, True, False, True, True]
    ya, _ = outcome(time, event, "alive_intact", 3.0)
    assert list(ya) == [0, 0, 0, 1, 1]


def test_km_left_limit_and_unclipped_weights():
    time = np.array([1.0, 2.0, 3.0, 3.0, 5.0])
    event = np.array([0, 1, 0, 0, 0])
    G = KMCensoring().fit(time, event)
    assert G.G_left([1.0])[0] == 1.0 and np.isclose(G.G_left([1.5])[0], 4 / 5)
    assert np.isclose(G.G_left([3.0])[0], 4 / 5)        # censorings at exactly 3 not subtracted
    wr = ipcw_weights(time, event, 3.0, G, min_censoring_survival=0.9)
    assert np.isclose(wr.w[1], 1 / (4 / 5)) and wr.w[0] == 0 and not wr.support_ok
    assert np.isclose(wr.w[2], 1 / (4 / 5))              # censored exactly at the horizon: observed through it


def test_recalibration_recovers_ideal_and_distorted_calibration():
    ideal, distorted = [], []
    for seed in range(20):
        time, event, true, tau = _competing_sample(seed)
        G = KMCensoring().fit(time, event)
        f = ipcw_logistic_calibration(true, time, event, "svd", tau, G)
        assert f.status == "ok"
        ideal.append((f.intercept_joint, f.slope, f.intercept_offset))
        bent = _expit(0.4 + 2.0 * _logit(true))        # over-dispersed and shifted: slope should be about 0.5
        g = ipcw_logistic_calibration(bent, time, event, "svd", tau, G)
        distorted.append((g.intercept_joint, g.slope))
    a, b, off = np.mean(ideal, axis=0)
    assert abs(a) < 0.1 and abs(b - 1) < 0.1 and abs(off) < 0.1
    assert abs(np.mean(distorted, axis=0)[1] - 0.5) < 0.08


def test_no_events_and_separation_give_missing_estimates_with_reasons():
    time = np.array([4.0] * 50)
    event = np.zeros(50, dtype=int)
    f = ipcw_logistic_calibration(np.full(50, 0.1), time, event, "svd", 3.0)
    assert f.status == "no_events" and np.isnan(f.slope) and np.isnan(f.intercept_offset)
    time = np.r_[np.full(30, 1.0), np.full(30, 4.0)]
    event = np.r_[np.ones(30, dtype=int), np.zeros(30, dtype=int)]
    pred = np.r_[np.full(30, 0.9), np.full(30, 0.1)]      # perfectly separated
    g = ipcw_logistic_calibration(pred, time, event, "svd", 3.0)
    assert g.status == "separation" and np.isnan(g.slope)


def test_brier_decomposition_reconstructs():
    time, event, true, tau = _competing_sample(1, n=2000)
    G = KMCensoring().fit(time, event)
    pred = np.clip(true + np.random.default_rng(2).normal(0, 0.05, len(true)), 0.001, 0.999)
    m = evaluate_predictions(pred, time, event, "svd", tau, G, decomposition=True)
    d = m["decomposition"]
    assert d["status"] == "ok" and d["reconstruction_error"] < 1e-10
    y, _ = outcome(time, event, "svd", tau)
    w = ipcw_weights(time, event, tau, G).w
    assert np.isclose(d["brier_normalised"], np.sum(w * (y - pred) ** 2) / w[w > 0].sum())
    assert np.isclose(d["brier_normalised"] * d["weight_sum_over_n"], m["brier"])
    raw = brier_decomposition(pred, y, np.zeros(len(y)))
    assert raw["status"] == "no_known_status"


def test_all_four_states_evaluate():
    time, event, true, tau = _competing_sample(4, n=1500)
    for state in ("svd", "death", "replacement", "alive_intact"):
        m = evaluate_predictions(np.full(len(time), 0.3), time, event, state, tau)
        assert 0 <= m["brier"] <= 1 and "cal_slope_logit" in m and "obs_minus_pred" in m
    m = evaluate_predictions(np.full(len(time), 0.3), time, event, "replacement", tau)
    assert m["cal_status"] == "no_events" and np.isnan(m["auc"])


def test_patient_bootstrap_keeps_complete_histories_and_counts_failures():
    pids = np.repeat(np.arange(30), 4)
    seen = []

    def fn(idx):
        counts = np.bincount(pids[idx], minlength=30)
        assert np.all(counts % 4 == 0)                   # a drawn patient brings all four rows
        seen.append(len(idx))
        if len(seen) % 5 == 0:
            raise ValueError("degenerate")
        return {"brier": float(len(idx))}

    res = bootstrap_by_patient(fn, pids, n_boot=50, seed=1, keys=("brier",), min_valid=100, min_success_fraction=0.8)
    assert res.failures == {"ValueError": 10} and res.n_valid["brier"] == 40
    assert res.intervals["brier"] is None and np.isnan(res["brier"][0])      # below the gate: no interval
    ok = bootstrap_by_patient(lambda idx: {"brier": 1.0}, pids, n_boot=30, seed=1, keys=("brier",), min_valid=20)
    assert ok.intervals["brier"] == (1.0, 1.0)


@pytest.mark.parametrize("tau", [1.0, 5.0])
def test_administrative_censoring_at_horizon_still_counts_as_event_free(tau):
    time = np.array([0.5, 0.8, tau, tau, tau])
    event = np.array([1, 2, 0, 0, 0])
    y, known = outcome(time, event, "svd", tau)
    assert known.all() and list(y) == [1, 0, 0, 0, 0]
