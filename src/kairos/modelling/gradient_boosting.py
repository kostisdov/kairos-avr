"""Cause-specific gradient boosting survival models (CR-08, WP-C2).

One ``sksurv.ensemble.GradientBoostingSurvivalAnalysis(loss="coxph")`` per route and cause (SVD,
death, non-SVD replacement); the other causes are censored at their time. Each route has its own
risk function and its own baseline, which is more flexible than the Cox family (shared
coefficients with route-stratified baselines). Thrombosis stays a secondary outcome, never a fourth
cause.

Hazard contract (:mod:`kairos.modelling.cif`): the baseline is the Breslow estimator on the
training risk scores (the estimator sksurv uses), kept as a right-continuous step function, and
``H_i(t) = H0_route(t) * exp(f(x_i))``. Internal early stopping stays off: its random validation
split would separate a patient's landmark rows.

Event support follows the Cox family per route: covariate model at ``fit_min_unique_events``
unique event patients in the route, covariate-free baseline at ``baseline_min_unique_events``,
otherwise unsupported (prediction raises).

Speed: scikit-survival 0.28 computes the Cox partial likelihood and its gradient with an O(n^2)
loop; fits here swap in exact O(n log n) versions (:func:`fast_cox_loss`), tested equal to the
library's functions and to library-default fits.

Inputs come from the same eligibility and feature pipeline as Cox in ``tree`` mode: the same
imputation, missing indicators and one-hot levels, without standardisation or spline expansion.
"""
from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from kairos.modelling.base import (
    CAUSES,
    SupportMixin,
    breslow_step,
    followup_support_time,
    step_eval,
)
from kairos.modelling.cif import build_grid

DEFAULT_GATES = {"fit_min_unique_events": 30, "baseline_min_unique_events": 5, "min_at_risk": 10}
SUPPORTED = ("ok", "reduced_baseline_only")
COX_LOSS_IMPLEMENTATION = "numpy O(n log n) Breslow partial likelihood, equal to sksurv 0.28 Cython (tested)"


def _risk_set_sums(time: np.ndarray, f: np.ndarray):
    """Sum of exp(f_k) over the Breslow risk set {k: T_k >= T_i} for every row i."""
    order = np.argsort(time, kind="mergesort")
    t_sorted = time[order]
    tail = np.cumsum(np.exp(f[order])[::-1])[::-1]
    out = np.empty(len(time))
    out[order] = tail[np.searchsorted(t_sorted, t_sorted, side="left")]
    return out, order, t_sorted


def coxph_loss_fast(event, time, f) -> float:
    """Negative Cox partial log-likelihood (Breslow ties), as ``sksurv.ensemble._coxph_loss.coxph_loss``."""
    event, time, f = np.asarray(event).astype(bool), np.asarray(time, dtype=float), np.asarray(f, dtype=float)
    rs, _, _ = _risk_set_sums(time, f)
    return float(-np.sum(f[event] - np.log(rs[event])))


def coxph_negative_gradient_fast(event, time, f) -> np.ndarray:
    """Negative gradient of the Cox partial likelihood, as ``coxph_negative_gradient`` (O(n log n), not O(n^2))."""
    event, time, f = np.asarray(event).astype(bool), np.asarray(time, dtype=float), np.asarray(f, dtype=float)
    rs, order, t_sorted = _risk_set_sums(time, f)
    cum = np.cumsum(np.where(event[order], 1.0 / rs[order], 0.0))
    acc = np.empty(len(time))
    acc[order] = cum[np.searchsorted(t_sorted, t_sorted, side="right") - 1]
    return event.astype(float) - np.exp(f) * acc


@contextmanager
def fast_cox_loss():
    """Swap sksurv's quadratic Cox loss and gradient for the equal O(n log n) versions during a fit."""
    from sksurv.ensemble import survival_loss

    saved = survival_loss.coxph_loss, survival_loss.coxph_negative_gradient
    survival_loss.coxph_loss, survival_loss.coxph_negative_gradient = coxph_loss_fast, coxph_negative_gradient_fast
    try:
        yield
    finally:
        survival_loss.coxph_loss, survival_loss.coxph_negative_gradient = saved


@dataclass
class CauseSpecificGradientBoostingModel(SupportMixin):
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 2
    min_samples_leaf: int = 20
    subsample: float = 1.0
    random_state: int = 20260917
    strata: tuple = ("route",)
    max_years: float = 5.0
    refine: int = 1
    horizons: tuple = (1.0, 3.0, 5.0)
    near_term: float = 1.0
    gates: dict = field(default_factory=lambda: dict(DEFAULT_GATES, mode="full"))
    keep_training_data: bool = False
    models_: dict = field(default_factory=dict)            # (cause, stratum) -> estimator
    feature_columns_: dict = field(default_factory=dict)   # stratum -> columns used by that route's models
    input_columns_: list = field(default_factory=list)
    dropped_constant_: dict = field(default_factory=dict)
    stratum_keys_: list = field(default_factory=list)
    event_counts_: dict = field(default_factory=dict)
    support_: dict = field(default_factory=dict)
    baselines_: dict = field(default_factory=dict)         # cause -> stratum -> (knots, values)
    followup_support_: dict = field(default_factory=dict)
    grid_: np.ndarray | None = None
    training_: dict = field(default_factory=dict)          # stratum -> (X, time, event); tuning only

    family = "gradient_boosting"
    adapter_version = "1"

    @property
    def hyperparameters(self) -> dict:
        return {"n_estimators": self.n_estimators, "learning_rate": self.learning_rate, "max_depth": self.max_depth,
                "min_samples_leaf": self.min_samples_leaf, "subsample": self.subsample, "random_state": self.random_state}

    @property
    def grid(self) -> np.ndarray:
        if self.grid_ is None:
            return build_grid([], [*self.horizons, self.near_term], self.max_years, self.refine)
        return self.grid_

    def _estimator(self, n_estimators: int | None = None):
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis

        return GradientBoostingSurvivalAnalysis(loss="coxph", n_estimators=int(n_estimators or self.n_estimators),
                                                learning_rate=float(self.learning_rate), max_depth=int(self.max_depth),
                                                min_samples_leaf=int(self.min_samples_leaf), subsample=float(self.subsample),
                                                random_state=int(self.random_state), n_iter_no_change=None)

    # --- fit ------------------------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, time: pd.Series, event: pd.Series, feature_columns: list[str],
            strata_frame: pd.DataFrame, patient_ids=None) -> CauseSpecificGradientBoostingModel:
        from sksurv.util import Surv

        self.input_columns_ = list(feature_columns)
        keys = self._keys(strata_frame)
        self.stratum_keys_ = sorted(set(keys))
        t = time.to_numpy(dtype=float)
        ev = event.to_numpy()
        pids = np.asarray(patient_ids) if patient_ids is not None else np.arange(len(t))
        fit_min = int(self.gates["fit_min_unique_events"])
        base_min = int(self.gates["baseline_min_unique_events"])
        self.followup_support_ = {k: followup_support_time(t[keys == k], int(self.gates["min_at_risk"])) for k in self.stratum_keys_}
        cand = X[self.input_columns_].astype(float)
        for k in self.stratum_keys_:
            m = keys == k
            sd = cand.loc[m].std(ddof=0)
            self.feature_columns_[k] = [c for c in self.input_columns_ if sd[c] > 1e-9]
            self.dropped_constant_[k] = [c for c in self.input_columns_ if not sd[c] > 1e-9]
            if self.keep_training_data:
                self.training_[k] = (cand.loc[m, self.feature_columns_[k]].to_numpy(dtype=np.float32), t[m], ev[m])
        all_knots: list = []
        for cause, code in CAUSES.items():
            ind = ev == code
            self.event_counts_[cause] = int(ind.sum())
            rec = {"event_patients": int(pd.Series(pids[ind]).nunique()), "event_rows": int(ind.sum()),
                   "gates": {g: self.gates[g] for g in DEFAULT_GATES}, "mode": self.gates.get("mode"), "strata": {}}
            self.baselines_[cause] = {}
            for k in self.stratum_keys_:
                m = keys == k
                n_pat = int(pd.Series(pids[m & ind]).nunique())
                cols = self.feature_columns_[k]
                srec = {"event_patients": n_pat}
                if n_pat >= fit_min and cols:
                    try:
                        Xk = cand.loc[m, cols].to_numpy(dtype=np.float32)
                        est = self._estimator()
                        with warnings.catch_warnings(), fast_cox_loss():
                            warnings.simplefilter("ignore")
                            est.fit(Xk, Surv.from_arrays(event=ind[m], time=t[m]))
                        self.models_[(cause, k)] = est
                        self.baselines_[cause][k] = breslow_step(t[m], ind[m], est.predict(Xk))
                        srec.update(status="ok", supported=True, reason="")
                    except Exception as ex:  # noqa: BLE001 - recorded, never a zero hazard
                        srec.update(status="fit_failed", supported=False, reason=f"{type(ex).__name__}: {ex}"[:200])
                elif n_pat >= base_min:
                    self.baselines_[cause][k] = breslow_step(t[m], ind[m])
                    srec.update(status="reduced_baseline_only", supported=True,
                                reason=(f"{n_pat} event patients in this route: fewer than {fit_min} for a boosted model; "
                                        "covariate-free baseline hazard only"))
                else:
                    srec.update(status="insufficient_events", supported=False,
                                reason=f"{n_pat} event patients in this route: fewer than {base_min}; no estimator")
                rec["strata"][k] = srec
                if k in self.baselines_[cause]:
                    all_knots.extend(np.asarray(self.baselines_[cause][k][0]).tolist())
            statuses = {s["status"] for s in rec["strata"].values()}
            rec["status"] = ("ok" if statuses == {"ok"} else "fit_failed" if "fit_failed" in statuses
                             else "insufficient_events" if statuses == {"insufficient_events"}
                             else "reduced_baseline_only" if statuses <= {"ok", "reduced_baseline_only"} else "partially_supported")
            rec["reason"] = "; ".join(f"{k}: {s['reason']}" for k, s in rec["strata"].items() if s["reason"])
            self.support_[cause] = rec
        self.grid_ = build_grid(all_knots, [*self.horizons, self.near_term], self.max_years, self.refine)
        return self

    # --- prediction ---------------------------------------------------------------------------------
    def _risk(self, X: pd.DataFrame, cause: str, keys: np.ndarray) -> np.ndarray:
        risk = np.ones(len(X))
        for k in set(keys):
            est = self.models_.get((cause, k))
            if est is None:
                continue
            m = keys == k
            risk[m] = np.exp(est.predict(X.loc[m, self.feature_columns_[k]].astype(float).to_numpy(dtype=np.float32)))
        return risk

    def cumulative_hazards(self, X: pd.DataFrame, times: np.ndarray | None = None) -> dict:
        times = self.grid if times is None else np.asarray(times, dtype=float)
        keys = self._check_rows(X)
        out = {}
        for cause in CAUSES:
            risk = self._risk(X, cause, keys)
            H = np.zeros((len(X), len(times)))
            for k in set(keys):
                m = keys == k
                knots, vals = self.baselines_[cause][k]
                H[m] = step_eval(knots, vals, times)[None, :] * risk[m, None]
            out[cause] = H
        return out

    def cumulative_hazard_at(self, X: pd.DataFrame, cause: str, t_rows) -> np.ndarray:
        keys = self._check_rows(X)
        t_rows = np.asarray(t_rows, dtype=float)
        risk = self._risk(X, cause, keys)
        out = np.zeros(len(X))
        for k in set(keys):
            m = keys == k
            knots, vals = self.baselines_[cause][k]
            out[m] = step_eval(knots, vals, t_rows[m]) * risk[m]
        return out

    def log_risk(self, X: pd.DataFrame, cause: str = "svd") -> np.ndarray:
        """Log risk score; zero for rows whose route has no boosted model for ``cause``."""
        keys = self._keys(X)
        known = np.isin(keys, self.stratum_keys_)
        out = np.zeros(len(X))
        if known.any():
            out[known] = np.log(self._risk(X.loc[known], cause, keys[known]))
        return out

    linear_predictor = log_risk

    def has_covariate_model(self, cause: str) -> bool:
        return any(c == cause for c, _ in self.models_)

    def coefficients(self, cause: str = "svd") -> pd.Series:
        return pd.Series(dtype=float)       # no coefficients; explanations are model sensitivities (phase D)

    def contributions(self, X: pd.DataFrame, cause: str = "svd") -> pd.DataFrame:
        return pd.DataFrame(index=X.index)

    def training_summary(self) -> dict:
        return {"family": self.family, "hyperparameters": self.hyperparameters, "cox_loss": COX_LOSS_IMPLEMENTATION,
                "features_by_route": {k: len(v) for k, v in self.feature_columns_.items()},
                "dropped_constant_by_route": self.dropped_constant_, "event_counts": self.event_counts_}

    # --- staged predictions for tuning ---------------------------------------------------------------
    def staged_state_probabilities(self, X: pd.DataFrame, stages: list[int], horizons, near_term: float,
                                   chunk: int = 1000) -> dict:
        """stage (number of trees) -> state -> (n, H) probabilities using the first ``stage`` trees of
        each fitted model, with the Breslow baseline recomputed on the training rows at that stage.
        Needs ``keep_training_data=True`` and supported rows. Equivalent to refitting with
        ``n_estimators=stage``: boosting stages are sequential and deterministic for a fixed random
        state (``tests/test_gradient_boosting.py``)."""
        from kairos.modelling.cif import STATES, combine_cause_specific, probabilities_at

        if not self.training_:
            raise RuntimeError("staged predictions need keep_training_data=True")
        keys = self._check_rows(X)
        stages = sorted({int(s) for s in stages})
        if max(stages) > self.n_estimators:
            raise ValueError("a stage exceeds the number of fitted trees")
        times = self.grid
        # training baselines and test log-risks once per (cause, route, stage)
        base: dict = {}
        risk = {s: {c: np.ones(len(X)) for c in CAUSES} for s in stages}
        for cause, code in CAUSES.items():
            for k in set(keys):
                m = keys == k
                est = self.models_.get((cause, k))
                if est is None:
                    for s in stages:
                        base[(cause, k, s)] = step_eval(*self.baselines_[cause][k], times)
                    continue
                Xtr, ttr, etr = self.training_[k]
                for i, p in enumerate(est.staged_predict(Xtr)):
                    if i + 1 in stages:
                        base[(cause, k, i + 1)] = step_eval(*breslow_step(ttr, etr == code, p), times)
                Xte = X.loc[m, self.feature_columns_[k]].astype(float).to_numpy(dtype=np.float32)
                for i, p in enumerate(est.staged_predict(Xte)):
                    if i + 1 in stages:
                        risk[i + 1][cause][m] = np.exp(p)
        out = {s: {st: np.full((len(X), len(horizons)), np.nan) for st in STATES} for s in stages}
        for s in stages:
            for start in range(0, len(X), chunk):
                sel = np.arange(start, min(start + chunk, len(X)))
                H = {c: np.zeros((len(sel), len(times))) for c in CAUSES}
                for c in CAUSES:
                    for k in set(keys[sel]):
                        mm = keys[sel] == k
                        H[c][mm] = base[(c, k, s)][None, :] * risk[s][c][sel][mm, None]
                pr = probabilities_at(combine_cause_specific(H, times), times, horizons, near_term)
                for st in STATES:
                    out[s][st][sel] = pr[st]
        return out

    def drop_training_data(self) -> None:
        self.training_ = {}
        self.keep_training_data = False
