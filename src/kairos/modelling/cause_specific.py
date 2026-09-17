"""Penalised cause-specific proportional hazards (ridge) for SVD, death and non-SVD
replacement, with route-stratified baseline hazards and landmark time as a covariate.

Implementation: one lifelines ``CoxPHFitter`` per cause on the stacked landmark rows
(landmark supermodel), the other causes treated as censoring for that fit.

Hazard contract (:mod:`kairos.modelling.cif`). The fitted Breslow baseline of each stratum is kept
as a right-continuous step function over its event times; the cumulative hazard of a row is
``H0_stratum(t) * exp(linear predictor)`` evaluated with that step function (lifelines'
``predict_cumulative_hazard`` interpolates linearly between event times, which the contract
forbids). The prediction grid is zero, the horizons and every baseline event time.

Event support (CR-04), counted in unique patients:

* at least ``fit_min_unique_events``: covariate model (status ``ok``);
* at least ``baseline_min_unique_events``: covariate-free route-stratified baseline (Nelson-Aalen,
  status ``reduced_baseline_only``), stated in the support record and never presented as a
  covariate model;
* fewer: status ``insufficient_events``; the cause has no estimator and a prediction raises
  :class:`UnsupportedFitError`, never a zero hazard.

A route stratum with fewer than ``baseline_min_unique_events`` event patients for a cause is
unsupported for that cause. A missing or unseen route raises :class:`UnsupportedRouteError`; no
majority route is substituted.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

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


def support_gates(cfg: dict, mode: str) -> dict:
    sup = cfg.get("support") or {}
    return {**DEFAULT_GATES, **(sup.get(mode) or {}), "mode": mode}


@dataclass
class CauseSpecificCoxModel(SupportMixin):
    penalizer: float = 0.05
    l1_ratio: float = 0.0
    strata: tuple = ("route",)
    max_years: float = 5.0
    refine: int = 1
    horizons: tuple = (1.0, 3.0, 5.0)
    near_term: float = 1.0
    gates: dict = field(default_factory=lambda: dict(DEFAULT_GATES, mode="full"))
    models_: dict = field(default_factory=dict)
    feature_columns_: list = field(default_factory=list)
    strata_levels_: dict = field(default_factory=dict)
    stratum_keys_: list = field(default_factory=list)
    event_counts_: dict = field(default_factory=dict)
    support_: dict = field(default_factory=dict)
    baselines_: dict = field(default_factory=dict)
    followup_support_: dict = field(default_factory=dict)
    dropped_constant_: list = field(default_factory=list)
    train_means_: pd.Series | None = None
    grid_: np.ndarray | None = None

    family = "cox"
    adapter_version = "1"

    @property
    def hyperparameters(self) -> dict:
        return {"penalizer": self.penalizer, "l1_ratio": self.l1_ratio, "strata": list(self.strata)}

    # --- grid and strata -------------------------------------------------------------------
    @property
    def grid(self) -> np.ndarray:
        if self.grid_ is None:
            return build_grid([], [*self.horizons, self.near_term], self.max_years, self.refine)
        return self.grid_

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X[self.feature_columns_].astype(float).copy()
        for s in self.strata:
            df[s] = X[s].astype(str).to_numpy()
        return df

    # --- fit --------------------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, time: pd.Series, event: pd.Series, feature_columns: list[str],
            strata_frame: pd.DataFrame, patient_ids=None) -> CauseSpecificCoxModel:
        # constant columns (e.g. an unused 'other' level) carry no information and break the
        # Newton step; they are excluded here and the exclusion is recorded.
        cand = X[list(feature_columns)].astype(float)
        sd = cand.std(ddof=0)
        self.dropped_constant_ = [c for c in cand.columns if not (sd[c] > 1e-9)]
        self.feature_columns_ = [c for c in cand.columns if sd[c] > 1e-9]
        base = cand[self.feature_columns_].copy()
        self.train_means_ = base.mean()
        keys = self._keys(strata_frame)
        self.stratum_keys_ = sorted(set(keys))
        for s in self.strata:
            vals = strata_frame[s].astype(str)
            self.strata_levels_[s] = sorted(vals.unique().tolist())
            base[s] = vals.to_numpy()
        t = time.to_numpy(dtype=float)
        base["time"] = t
        ev = event.to_numpy()
        pids = np.asarray(patient_ids) if patient_ids is not None else np.arange(len(base))
        fit_min = int(self.gates["fit_min_unique_events"])
        base_min = int(self.gates["baseline_min_unique_events"])
        min_at_risk = int(self.gates["min_at_risk"])
        self.followup_support_ = {k: followup_support_time(t[keys == k], min_at_risk) for k in self.stratum_keys_}
        all_knots: list = []
        for cause, code in CAUSES.items():
            ind = ev == code
            self.event_counts_[cause] = int(ind.sum())
            n_pat = int(pd.Series(pids[ind]).nunique())
            by_stratum = {k: int(pd.Series(pids[ind & (keys == k)]).nunique()) for k in self.stratum_keys_}
            rec = {"event_patients": n_pat, "event_rows": int(ind.sum()), "gates": {k: self.gates[k] for k in DEFAULT_GATES},
                   "mode": self.gates.get("mode"), "strata": {}}
            self.models_[cause] = None
            self.baselines_[cause] = {}
            if n_pat >= fit_min:
                df = base.copy()
                df["e"] = ind.astype(int)
                try:
                    cph = CoxPHFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        cph.fit(df, duration_col="time", event_col="e", strata=list(self.strata) or None,
                                fit_options={"step_size": 0.5})
                    self.models_[cause] = cph
                    rec.update(status="ok", reason="")
                    self.baselines_[cause] = self._baselines_from_lifelines(cph)
                except Exception as ex:  # noqa: BLE001 - recorded, never silently zero
                    rec.update(status="fit_failed", reason=f"{type(ex).__name__}: {ex}"[:200])
            elif n_pat >= base_min:
                rec.update(status="reduced_baseline_only",
                           reason=(f"{n_pat} event patients: fewer than {fit_min} for a covariate model; covariate-free "
                                   "route-stratified baseline hazard only"))
                self.baselines_[cause] = {k: breslow_step(t[keys == k], ind[keys == k]) for k in self.stratum_keys_}
            else:
                rec.update(status="insufficient_events",
                           reason=f"{n_pat} event patients: fewer than {base_min}; no estimator for this cause")
            for k in self.stratum_keys_:
                ok = rec["status"] in SUPPORTED and by_stratum[k] >= base_min
                rec["strata"][k] = {"event_patients": by_stratum[k], "supported": bool(ok),
                                    "reason": "" if ok else (rec["reason"] if rec["status"] not in SUPPORTED else
                                                             f"{by_stratum[k]} event patients in this route: fewer than {base_min}")}
            self.support_[cause] = rec
            for knots, _ in self.baselines_[cause].values():
                all_knots.extend(np.asarray(knots).tolist())
        self.grid_ = build_grid(all_knots, [*self.horizons, self.near_term], self.max_years, self.refine)
        return self

    def _baselines_from_lifelines(self, cph: CoxPHFitter) -> dict:
        out = {}
        b = cph.baseline_cumulative_hazard_
        for col in b.columns:
            key = "|".join(map(str, col)) if isinstance(col, tuple) else (str(col) if self.strata else "all")
            v = b[col].to_numpy(dtype=float)
            idx = b.index.to_numpy(dtype=float)
            jump = np.diff(v, prepend=0.0) > 0
            out[key] = (idx[jump], v[jump])
        return out

    # --- hazards ----------------------------------------------------------------------------
    def _risk(self, X: pd.DataFrame, cause: str) -> np.ndarray:
        cph = self.models_.get(cause)
        if cph is None:
            return np.ones(len(X))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.asarray(cph.predict_partial_hazard(self._prepare(X)), dtype=float).ravel()

    def cumulative_hazards(self, X: pd.DataFrame, times: np.ndarray | None = None) -> dict:
        """cause -> (n, T) cumulative hazards at ``times`` (default: the model grid)."""
        times = self.grid if times is None else np.asarray(times, dtype=float)
        keys = self._check_rows(X)
        out = {}
        for cause in CAUSES:
            risk = self._risk(X, cause)
            H = np.zeros((len(X), len(times)))
            for k in set(keys):
                m = keys == k
                knots, vals = self.baselines_[cause][k]
                H[m] = step_eval(knots, vals, times)[None, :] * risk[m, None]
            out[cause] = H
        return out

    def cumulative_hazard_at(self, X: pd.DataFrame, cause: str, t_rows) -> np.ndarray:
        """Cumulative hazard of ``cause`` for each row at its own time."""
        keys = self._check_rows(X)
        t_rows = np.asarray(t_rows, dtype=float)
        risk = self._risk(X, cause)
        out = np.zeros(len(X))
        for k in set(keys):
            m = keys == k
            knots, vals = self.baselines_[cause][k]
            out[m] = step_eval(knots, vals, t_rows[m]) * risk[m]
        return out

    def linear_predictor(self, X: pd.DataFrame, cause: str = "svd") -> np.ndarray:
        """Log partial hazard; zero for a cause without a covariate model."""
        return np.log(self._risk(X, cause))

    log_risk = linear_predictor

    def has_covariate_model(self, cause: str) -> bool:
        return self.models_.get(cause) is not None

    def training_summary(self) -> dict:
        return {"family": self.family, "hyperparameters": self.hyperparameters, "n_features": len(self.feature_columns_),
                "dropped_constant": self.dropped_constant_, "event_counts": self.event_counts_}

    def coefficients(self, cause: str = "svd") -> pd.Series:
        cph = self.models_.get(cause)
        if cph is None:
            return pd.Series(dtype=float)
        return cph.params_

    def contributions(self, X: pd.DataFrame, cause: str = "svd") -> pd.DataFrame:
        """Per-row contribution of each column to the log partial hazard relative to the
        training mean: coef * (x - mean)."""
        coef = self.coefficients(cause)
        if coef.empty:
            return pd.DataFrame(index=X.index)
        cols = [c for c in coef.index if c in self.feature_columns_]
        x = X[cols].astype(float)
        return (x - self.train_means_[cols]) * coef[cols]
