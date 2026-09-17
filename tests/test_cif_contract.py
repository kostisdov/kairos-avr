"""CR-05: hazard contract, piecewise-constant-hazard integration and the Cox step baseline."""
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import solve_ivp

from kairos.modelling.base import (
    UnsupportedFitError,
    UnsupportedRouteError,
    breslow_step,
    step_eval,
)
from kairos.modelling.cause_specific import CauseSpecificCoxModel
from kairos.modelling.cif import (
    CAUSE_ORDER,
    HazardContractError,
    at_times,
    build_grid,
    combine_cause_specific,
    combine_product_limit_legacy,
    probabilities_at,
)
from kairos.modelling.train import fit_step, landmark_from_cohort


def _H(grid, n=1, **per_cause):
    return {c: np.tile(np.asarray(per_cause.get(c, np.zeros(len(grid))), dtype=float), (n, 1)) for c in CAUSE_ORDER}


def _closure(c):
    return np.max(np.abs(c["svd"] + c["death"] + c["replacement"] + c["alive_intact"] - 1.0))


def test_zero_hazards():
    grid = np.linspace(0, 5, 6)
    c = combine_cause_specific(_H(grid, 3), grid)
    assert np.all(c["alive_intact"] == 1.0) and np.all(c["svd"] == 0.0)


def test_single_cause_is_one_minus_exp():
    grid = np.linspace(0, 5, 11)
    H = 0.37 * grid ** 1.3
    c = combine_cause_specific(_H(grid, svd=H), grid)
    assert np.allclose(c["svd"][0], 1 - np.exp(-H), atol=1e-14) and _closure(c) < 1e-12


def test_constant_competing_hazards_match_the_closed_form():
    lam = {"svd": 0.03, "death": 0.11, "replacement": 0.004}
    grid = build_grid([], [1, 3, 5], 5.0, refine=7)
    c = combine_cause_specific(_H(grid, **{k: v * grid for k, v in lam.items()}), grid)
    tot = sum(lam.values())
    for k, v in lam.items():
        exact = v / tot * (1 - np.exp(-tot * grid))
        assert np.max(np.abs(c[k][0] - exact)) < 1e-13
    assert _closure(c) < 1e-12


def test_large_hazards_zero_first_interval_and_first_knot_mass():
    grid = np.array([0.0, 0.5, 1.0, 2.0])
    big = combine_cause_specific(_H(grid, svd=[0, 0, 40, 80], death=[0, 0, 10, 60]), grid)
    assert np.all(np.isfinite(big["svd"])) and _closure(big) < 1e-12 and big["alive_intact"][0, -1] < 1e-30
    # hazard mass at the first knot is kept (the legacy combination dropped it)
    first = combine_cause_specific(_H(grid, svd=[0, 0.2, 0.2, 0.2]), grid)
    assert np.isclose(first["svd"][0, 1], 1 - np.exp(-0.2))
    # lifelines extrapolated the first baseline value back to time 0; the legacy combination dropped it
    legacy = combine_product_limit_legacy(_H(grid, svd=[0.2, 0.2, 0.2, 0.2]), grid)
    assert legacy["svd"][0, -1] == 0.0


def test_tied_event_times_split_the_interval_mass():
    grid = np.array([0.0, 1.0])
    c = combine_cause_specific(_H(grid, svd=[0, 0.3], death=[0, 0.3]), grid)
    assert np.isclose(c["svd"][0, 1], c["death"][0, 1]) and np.isclose(c["svd"][0, 1], 0.5 * (1 - np.exp(-0.6)))


def test_exact_horizon_boundaries_and_monotonicity():
    grid = build_grid(np.cumsum(np.full(40, 0.123456789)), [1, 3, 5], 5.0)
    assert {1.0, 3.0, 5.0} <= set(grid) and grid[0] == 0
    rng = np.random.default_rng(3)
    H = {k: np.maximum.accumulate(np.cumsum(rng.uniform(0, 0.02, (5, len(grid))), axis=1), axis=1) for k in CAUSE_ORDER}
    for k in H:
        H[k][:, 0] = 0
    c = combine_cause_specific(H, grid)
    assert _closure(c) < 1e-10
    for k in CAUSE_ORDER:
        assert np.all(np.diff(c[k], axis=1) >= -1e-15)
    assert np.all(np.diff(c["alive_intact"], axis=1) <= 1e-15)
    i3 = int(np.flatnonzero(grid == 3.0)[0])
    assert np.array_equal(at_times(c["svd"], grid, [3.0 - 1e-12, 3.0])[:, 0], c["svd"][:, i3])
    p = probabilities_at(c, grid, [1, 3, 5], 1.0)
    assert np.allclose(p["svd_near_term"], p["svd"][:, 0])


def test_contract_violations_are_rejected_not_repaired():
    grid = np.array([0.0, 1.0, 2.0])
    with pytest.raises(HazardContractError, match="non-finite"):
        combine_cause_specific(_H(grid, svd=[0, np.nan, 1]), grid)
    with pytest.raises(HazardContractError, match="decreases"):
        combine_cause_specific(_H(grid, svd=[0, 0.5, 0.4]), grid)
    with pytest.raises(HazardContractError, match="time zero"):
        combine_cause_specific(_H(grid, svd=[0.1, 0.5, 0.6]), grid)
    with pytest.raises(HazardContractError, match="shapes"):
        combine_cause_specific({"svd": np.zeros((1, 3)), "death": np.zeros((2, 3)), "replacement": np.zeros((1, 3))}, grid)
    with pytest.raises(HazardContractError, match="increasing"):
        combine_cause_specific(_H(np.array([0.0, 1.0, 1.0])), np.array([0.0, 1.0, 1.0]))
    tiny = combine_cause_specific(_H(grid, svd=[0, 0.5, 0.5 - 1e-12]), grid)
    assert tiny["_diagnostics"]["max_roundoff_correction"] > 0


def test_discretisation_against_a_numerical_reference():
    """Generator-shaped continuous hazards (Weibull SVD, Gompertz death, exponential replacement),
    integrated exactly at weekly grid points, against an ODE solution: below 1e-4 at the horizons."""
    shape, lam, h0, b, r = 1.8, 0.09, 0.05, 0.085, 0.004
    Hs = {"svd": lambda t: (lam * t) ** shape, "death": lambda t: h0 / b * np.expm1(b * t), "replacement": lambda t: r * t}
    hs = {"svd": lambda t: shape * lam ** shape * t ** (shape - 1), "death": lambda t: h0 * np.exp(b * t),
          "replacement": lambda t: r}
    grid = build_grid(np.arange(1, 261) / 52.0, [1, 3, 5], 5.0)
    c = combine_cause_specific({k: f(grid)[None, :] for k, f in Hs.items()}, grid)

    def rhs(t, y):
        s = y[0]
        rates = [hs[k](t) for k in CAUSE_ORDER]
        return [-sum(rates) * s, *(rt * s for rt in rates)]

    sol = solve_ivp(rhs, (1e-12, 5.0), [1.0, 0, 0, 0], t_eval=[1.0, 3.0, 5.0], rtol=1e-11, atol=1e-13)
    for j, h in enumerate([1.0, 3.0, 5.0]):
        i = int(np.flatnonzero(grid == h)[0])
        ref = dict(zip(("alive_intact", *CAUSE_ORDER), sol.y[:, j]))
        assert max(abs(c[k][0, i] - ref[k]) for k in ref) < 1e-4


# --- Cox step baseline and support -------------------------------------------------------------
def test_breslow_and_step_function():
    knots, vals = breslow_step([1, 2, 2, 3, 4], [1, 1, 0, 1, 0])
    assert np.allclose(knots, [1, 2, 3]) and np.allclose(vals, np.cumsum([1 / 5, 1 / 4, 1 / 2]))
    assert np.allclose(step_eval(knots, vals, [0.5, 1, 1.5, 3, 10]), [0, vals[0], vals[0], vals[2], vals[2]])


def test_cox_step_baseline_matches_lifelines_at_event_times(small_cohort, model_cfg):
    lm = landmark_from_cohort(small_cohort, model_cfg)
    pipe, cox, *_ = fit_step(lm, ["core_static", "core_time"], model_cfg, "quick")
    X = pipe.transform(lm.iloc[:40])
    X["route"] = lm["route"].iloc[:40].astype(str).to_numpy()
    cph = cox.models_["death"]
    t = np.sort(lm.loc[lm["event"] == 2, "time"].unique())[:25]
    ours = cox.cumulative_hazards(X, t)["death"]
    theirs = cph.predict_cumulative_hazard(cox._prepare(X), times=t).to_numpy().T
    assert np.max(np.abs(ours - theirs)) < 1e-10
    assert {1.0, 3.0, 5.0} <= set(cox.grid) and len(cox.grid) > 100


def _toy(n_pat=40, rows=3, events=None, route=("SAVR", "TAVR")):
    rng = np.random.default_rng(0)
    recs = []
    for i in range(n_pat):
        for _ in range(rows):
            recs.append({"patient_id": f"p{i}", "route": route[i % len(route)], "x": rng.normal(),
                         "time": rng.uniform(0.2, 5), "event": 0})
    df = pd.DataFrame(recs)
    for code, pats in (events or {}).items():
        df.loc[df["patient_id"].isin(pats), "event"] = code
    return df


def test_three_event_patients_repeated_in_many_rows_do_not_pass():
    df = _toy(rows=10, events={1: ["p0", "p1", "p2"], 2: [f"p{i}" for i in range(3, 40)], 3: [f"p{i}" for i in range(3, 10)]})
    m = CauseSpecificCoxModel().fit(df[["x"]], df["time"], df["event"], ["x"], df[["route"]], patient_ids=df["patient_id"])
    assert m.support_["svd"]["event_rows"] == 30 and m.support_["svd"]["event_patients"] == 3
    assert m.support_["svd"]["status"] == "insufficient_events"
    assert m.support_["death"]["status"] == "ok" and m.support_["replacement"]["status"] == "reduced_baseline_only"
    X = df[["x", "route"]].head(4)
    with pytest.raises(UnsupportedFitError):          # zero or too few events is never zero future risk
        m.cumulative_hazards(X)
    ok, reasons = m.row_support(X)
    assert not ok.any() and all(r.startswith("insufficient_events") for r in reasons)


def test_missing_or_unseen_route_is_never_substituted():
    df = _toy(events={1: [f"p{i}" for i in range(0, 40, 2)] + [f"p{i}" for i in range(1, 40, 2)][:10],
                      2: [f"p{i}" for i in range(22, 40)], 3: ["p0"]})
    m = CauseSpecificCoxModel(gates={"fit_min_unique_events": 5, "baseline_min_unique_events": 1, "min_at_risk": 2, "mode": "quick"})
    m.fit(df[["x"]], df["time"], df["event"], ["x"], df[["route"]], patient_ids=df["patient_id"])
    X = pd.DataFrame({"x": [0.1, 0.2], "route": ["unknown", "nan"]})
    with pytest.raises(UnsupportedRouteError):
        m.cumulative_hazards(X)
    ok, reasons = m.row_support(X)
    assert list(reasons) == ["unknown_device_or_route", "unknown_device_or_route"]
    assert m.support_["replacement"]["strata"]["TAVR"]["supported"] is False   # p0 is SAVR: no TAVR replacement event
    assert m.supported_routes == ["SAVR"]
