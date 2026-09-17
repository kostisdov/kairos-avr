"""dp-ucMGP (dephosphorylated-uncarboxylated matrix Gla protein) as a consumed model input.

Specification (docs/kairos_proposal_revised.html, "How dp-ucMGP enters the model"):

* the marker enters the anticoagulant module as log-transformed pmol/L, with the assay
  recorded, through two pre-specified terms on the SVD hazard only: a spline main effect and
  an interaction with cumulative vitamin K antagonist exposure;
* it is measured at time zero and at each annual landmark, carried forward for at most
  twelve months and then marked stale (a stale value is not used);
* it is never imputed across a cohort. The core model with the exposure module is fitted on
  the full cohort; in the substudy subset where dp-ucMGP is measured a second model takes the
  full-cohort SVD linear predictor as an offset and estimates only the dp-ucMGP terms;
* every prediction states whether a measured dp-ucMGP value was used.

Implementation notes. The main effect is a restricted cubic spline (linear tails, knots at
quantiles of the training substudy), which stays stable at the extremes where a B-spline basis
of the same size does not. The interaction is the standardised log marker times cumulative VKA
years (a spline-by-exposure interaction would spend more parameters than a substudy
supports). The offset model is a ridge-penalised stratified Cox partial likelihood with
Breslow ties, fitted by Newton-Raphson, with the ridge penalty on the summed log partial
likelihood as in lifelines, and a Wald test whose covariance comes from resampling patients.
The full-cohort baseline hazards are kept: the dp-ucMGP log hazard ratio is centred so that
the expected number of SVD events in the training substudy (the full-model cumulative hazard
at each row's follow-up time) is unchanged, so the offset model redistributes risk among
measured patients without recalibrating their average level.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

MARKER = "dp_ucmgp"
LOG_FEATURE = "log_dp_ucmgp"
VKA_FEATURE = "ac_cum_vka_years"
UNIT = "pmol/L"
DAYS = 365.25


def dp_ucmgp_features(labs: Iterable[tuple[date, str, float]], landmark: date, carry_forward_months: float = 12.0) -> dict:
    """Latest dp-ucMGP dated on or before ``landmark``; values older than the carry-forward
    window are stale and do not enter the model."""
    latest = None
    for d, analyte, value in labs:
        if analyte != MARKER or d > landmark or value is None or not np.isfinite(value) or value <= 0:
            continue
        if latest is None or d >= latest[0]:
            latest = (d, float(value))
    if latest is None:
        return {MARKER: np.nan, LOG_FEATURE: np.nan, "dp_ucmgp_age_months": np.nan, "dp_ucmgp_stale": False}
    age_months = (landmark - latest[0]).days / DAYS * 12.0
    stale = age_months > carry_forward_months
    return {MARKER: latest[1], LOG_FEATURE: np.nan if stale else math.log(latest[1]),
            "dp_ucmgp_age_months": round(age_months, 2), "dp_ucmgp_stale": bool(stale)}


def measured_mask(df: pd.DataFrame) -> np.ndarray:
    if LOG_FEATURE not in df:
        return np.zeros(len(df), dtype=bool)
    return np.isfinite(pd.to_numeric(df[LOG_FEATURE], errors="coerce").to_numpy(dtype=float))


# --- the two pre-specified terms ---------------------------------------------------------------
RCS_KNOT_QUANTILES = {3: (0.10, 0.50, 0.90), 4: (0.05, 0.35, 0.65, 0.95), 5: (0.05, 0.275, 0.50, 0.725, 0.95)}


def rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Restricted cubic spline (Harrell): x plus k-2 nonlinear terms, linear beyond the outer knots."""
    t = np.asarray(knots, dtype=float)
    scale = (t[-1] - t[0]) ** 2 or 1.0
    cols = [x]
    for j in range(len(t) - 2):
        term = (np.clip(x - t[j], 0, None) ** 3
                - np.clip(x - t[-2], 0, None) ** 3 * (t[-1] - t[j]) / (t[-1] - t[-2])
                + np.clip(x - t[-1], 0, None) ** 3 * (t[-2] - t[j]) / (t[-1] - t[-2]))
        cols.append(term / scale)
    return np.column_stack(cols)


@dataclass
class VitaminKTerms:
    """Design for the dp-ucMGP terms, fitted within training rows only."""
    n_knots: int = 3
    knots_: np.ndarray | None = None
    log_mean_: float = 0.0
    log_sd_: float = 1.0
    lo_: float = 0.0
    hi_: float = 0.0
    col_means_: np.ndarray | None = None
    col_sds_: np.ndarray | None = None
    columns_: list = field(default_factory=list)

    def fit(self, df: pd.DataFrame) -> VitaminKTerms:
        x = pd.to_numeric(df[LOG_FEATURE], errors="coerce").to_numpy(dtype=float)
        self.log_mean_ = float(np.mean(x))
        sd = float(np.std(x))
        self.log_sd_ = sd if sd > 1e-9 else 1.0
        self.lo_, self.hi_ = float(np.min(x)), float(np.max(x))
        self.knots_ = None
        knots = np.quantile(x, RCS_KNOT_QUANTILES.get(self.n_knots, RCS_KNOT_QUANTILES[3]))
        if len(np.unique(x)) > self.n_knots + 2 and np.all(np.diff(knots) > 1e-6):
            self.knots_ = knots
        raw = self._raw(df)
        main = [LOG_FEATURE] + [f"{LOG_FEATURE}__rcs{j + 1}" for j in range(raw.shape[1] - 2)]
        self.columns_ = main + [f"{MARKER}_x_cum_vka"]
        self.col_means_ = raw.mean(axis=0)
        sds = raw.std(axis=0)
        self.col_sds_ = np.where(sds > 1e-9, sds, 1.0)
        return self

    def _raw(self, df: pd.DataFrame) -> np.ndarray:
        x = pd.to_numeric(df[LOG_FEATURE], errors="coerce").to_numpy(dtype=float)
        x = np.clip(np.nan_to_num(x, nan=self.log_mean_), self.lo_, self.hi_)
        z = (x - self.log_mean_) / self.log_sd_
        vka = pd.to_numeric(df[VKA_FEATURE], errors="coerce").fillna(0.0).to_numpy(dtype=float) if VKA_FEATURE in df else np.zeros(len(df))
        main = rcs_basis(x, self.knots_) if self.knots_ is not None else x.reshape(-1, 1)
        return np.column_stack([main, z * vka])

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Standardised design matrix (rows without a usable measurement get zeros; callers mask them)."""
        return (self._raw(df) - self.col_means_) / self.col_sds_

    def frame(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.transform(df), columns=self.columns_, index=df.index)


# --- penalised stratified Cox partial likelihood with an offset -----------------------------------
def _stratum_arrays(time: np.ndarray, event: np.ndarray, strata: np.ndarray):
    out = []
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        order = idx[np.argsort(time[idx], kind="mergesort")]
        t = time[order]
        first = np.searchsorted(t, t, side="left")  # Breslow: tied times share the risk set
        out.append((order, first, event[order].astype(bool)))
    return out


def _loglik_grad_hess(beta: np.ndarray, Z: np.ndarray, offset: np.ndarray, groups) -> tuple[float, np.ndarray, np.ndarray]:
    p = Z.shape[1]
    ll, g, H = 0.0, np.zeros(p), np.zeros((p, p))
    for order, first, ev in groups:
        if not ev.any():
            continue
        z = Z[order]
        eta = offset[order] + z @ beta
        m = eta.max()
        w = np.exp(eta - m)
        s0 = np.cumsum(w[::-1])[::-1][first]
        s1 = np.cumsum((w[:, None] * z)[::-1], axis=0)[::-1][first]
        s2 = np.cumsum((w[:, None, None] * z[:, :, None] * z[:, None, :])[::-1], axis=0)[::-1][first]
        e0, e1, e2 = s0[ev], s1[ev], s2[ev]
        ll += float(np.sum(eta[ev] - m - np.log(e0)))
        mean = e1 / e0[:, None]
        g += np.sum(z[ev] - mean, axis=0)
        H -= np.sum(e2 / e0[:, None, None] - mean[:, :, None] * mean[:, None, :], axis=0)
    return ll, g, H


def fit_offset_cox(Z: np.ndarray, time, event, offset, strata, penalizer: float = 0.05,
                   max_iter: int = 50, tol: float = 1e-8, clusters=None, n_boot: int = 0, seed: int = 0) -> dict:
    """Ridge Cox for columns ``Z`` with a fixed ``offset``; event is a 0/1 indicator."""
    Z = np.asarray(Z, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event).astype(int)
    offset = np.asarray(offset, dtype=float)
    strata = np.asarray(strata).astype(str)
    n, p = Z.shape
    groups = _stratum_arrays(time, event, strata)
    beta = np.zeros(p)
    ll0, _, _ = _loglik_grad_hess(beta, Z, offset, groups)

    def objective(b):
        ll, g, H = _loglik_grad_hess(b, Z, offset, groups)
        return ll - 0.5 * penalizer * b @ b, g - penalizer * b, H - penalizer * np.eye(p), ll

    obj, g, H, ll = objective(beta)
    converged = False
    for _ in range(max_iter):
        step = np.linalg.solve(H, -g)
        t = 1.0
        while t > 1e-6:
            cand = beta + t * step
            c_obj, c_g, c_H, c_ll = objective(cand)
            if c_obj >= obj - 1e-12:
                break
            t *= 0.5
        delta = abs(c_obj - obj)
        beta, obj, g, H, ll = cand, c_obj, c_g, c_H, c_ll
        if delta < tol:
            converged = True
            break
    cov = np.linalg.inv(-H)
    lr = max(0.0, 2.0 * (ll - ll0))
    out = {"beta": beta, "se": np.sqrt(np.clip(np.diag(cov), 0, None)), "loglik": ll, "loglik_offset_only": ll0,
           "lr_stat": lr, "df": p, "p_value": float(stats.chi2.sf(lr, p)), "converged": converged,
           "n": int(n), "n_events": int(event.sum())}
    if clusters is not None and n_boot > 0:
        out.update(_cluster_bootstrap_wald(beta, Z, time, event, offset, strata, np.asarray(clusters), penalizer, n_boot, seed))
    return out


def _cluster_bootstrap_wald(beta, Z, time, event, offset, strata, clusters, penalizer, n_boot, seed) -> dict:
    """Wald test of all dp-ucMGP terms with a covariance from resampling patients (all landmark
    rows of a patient together), because stacked landmark rows are not independent."""
    uniq, inv = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])
        if event[idx].sum() < 5:
            continue
        try:
            b = fit_offset_cox(Z[idx], time[idx], event[idx], offset[idx], strata[idx], penalizer, max_iter=25, tol=1e-6)["beta"]
        except np.linalg.LinAlgError:
            continue
        draws.append(b)
    if len(draws) < 20:
        return {"wald_bootstrap_stat": float("nan"), "p_value_patient_bootstrap": float("nan"), "n_boot_ok": len(draws)}
    V = np.cov(np.asarray(draws), rowvar=False)
    w = float(beta @ np.linalg.pinv(V) @ beta)
    return {"wald_bootstrap_stat": w, "p_value_patient_bootstrap": float(stats.chi2.sf(w, len(beta))), "n_boot_ok": len(draws),
            "se_patient_bootstrap": np.sqrt(np.clip(np.diag(V), 0, None))}


# --- the substudy offset model ----------------------------------------------------------------------
@dataclass
class DpUcMgpOffsetModel:
    fitted: bool
    reason: str = ""
    terms: VitaminKTerms | None = None
    beta: np.ndarray | None = None
    centre: float = 0.0
    summary: dict = field(default_factory=dict)

    def log_hr(self, df: pd.DataFrame) -> np.ndarray:
        """dp-ucMGP log hazard ratio on the SVD hazard; 0 where no usable measurement exists."""
        out = np.zeros(len(df))
        if not self.fitted:
            return out
        m = measured_mask(df)
        if m.any():
            out[m] = self.terms.transform(df.loc[m]) @ self.beta - self.centre
        return out

    def adjust(self, cum_hazards: dict, df: pd.DataFrame) -> dict:
        """Multiply the SVD cumulative hazard by the dp-ucMGP hazard ratio (death and non-SVD
        replacement are untouched)."""
        if not self.fitted:
            return cum_hazards
        out = dict(cum_hazards)
        out["svd"] = np.asarray(cum_hazards["svd"]) * np.exp(self.log_hr(df))[:, None]
        return out

    def card(self) -> dict:
        return {"fitted": self.fitted, "reason": self.reason, **self.summary}


def substudy_counts(lm: pd.DataFrame, svd_code: int = 1) -> dict:
    m = measured_mask(lm)
    return {"n_rows_measured": int(m.sum()), "fraction_rows_measured": round(float(m.mean()) if len(lm) else 0.0, 3),
            "n_patients_measured": int(lm.loc[m, "patient_id"].nunique()) if "patient_id" in lm and m.any() else 0,
            "n_svd_events_measured": int((lm.loc[m, "event"] == svd_code).sum()) if "event" in lm and m.any() else 0}


def fit_dp_ucmgp_offset(lm: pd.DataFrame, full_svd_lp: np.ndarray, cfg: dict, svd_code: int = 1,
                        expected_svd_hazard: np.ndarray | None = None, wald_bootstrap: int | None = None) -> DpUcMgpOffsetModel:
    """Fit the dp-ucMGP terms on measured rows of ``lm`` with the full-cohort SVD linear
    predictor ``full_svd_lp`` (aligned with ``lm``) as an offset. ``expected_svd_hazard`` is the
    full-model SVD cumulative hazard at each row's follow-up time, used for centring."""
    vk = cfg["vitamin_k"]
    counts = substudy_counts(lm, svd_code)
    m = measured_mask(lm)
    if counts["n_rows_measured"] < int(vk["min_measured_rows"]) or counts["n_svd_events_measured"] < int(vk["min_measured_svd_events"]):
        return DpUcMgpOffsetModel(False, reason=(f"dp-ucMGP substudy too small ({counts['n_rows_measured']} measured rows, "
                                                 f"{counts['n_svd_events_measured']} SVD events; need {vk['min_measured_rows']} and "
                                                 f"{vk['min_measured_svd_events']}); the core model reports without it, nothing imputed"),
                                  summary=counts)
    sub = lm.loc[m]
    terms = VitaminKTerms(n_knots=int(vk["spline_knots"])).fit(sub)
    Z = terms.transform(sub)
    strata_cols = list(cfg["model"]["strata"])
    strata = sub[strata_cols].astype(str).agg("|".join, axis=1).to_numpy() if strata_cols else np.zeros(len(sub))
    fit = fit_offset_cox(Z, sub["time"], (sub["event"] == svd_code).astype(int), np.asarray(full_svd_lp)[m], strata,
                         penalizer=float(vk["penalizer"]), clusters=sub["patient_id"].to_numpy() if "patient_id" in sub else None,
                         n_boot=int(vk.get("wald_bootstrap", 0) if wald_bootstrap is None else wald_bootstrap),
                         seed=int(vk.get("seed", 0)))
    lin = Z @ fit["beta"]
    e = np.asarray(expected_svd_hazard, dtype=float)[m] if expected_svd_hazard is not None else np.ones(len(sub))
    e = np.where(np.isfinite(e) & (e > 0), e, 0.0)
    centre = float(np.log(np.sum(e * np.exp(lin)) / np.sum(e))) if e.sum() > 0 else float(np.log(np.mean(np.exp(lin))))
    summary = {**counts, "terms": terms.columns_,
               "coefficients": {c: round(float(b), 4) for c, b in zip(terms.columns_, fit["beta"])},
               "se": {c: round(float(s), 4) for c, s in zip(terms.columns_, fit.get("se_patient_bootstrap", fit["se"]))},
               "df": fit["df"], "converged": fit["converged"],
               "lr_stat_nominal": round(fit["lr_stat"], 3), "p_value_nominal": fit["p_value"],
               "wald_stat_patient_bootstrap": fit.get("wald_bootstrap_stat"), "p_value": fit.get("p_value_patient_bootstrap", fit["p_value"]),
               "p_value_method": ("patient-bootstrap Wald test of all dp-ucMGP terms" if "p_value_patient_bootstrap" in fit
                                  else "nominal likelihood ratio (landmark rows treated as independent)"),
               "log_marker_mean": round(terms.log_mean_, 4), "log_marker_sd": round(terms.log_sd_, 4),
               "rcs_knots_log": None if terms.knots_ is None else [round(float(k), 4) for k in terms.knots_],
               "centre": round(centre, 4),
               "unit": UNIT, "method": "offset: full-cohort SVD linear predictor; ridge Cox for the dp-ucMGP terms only"}
    return DpUcMgpOffsetModel(True, terms=terms, beta=fit["beta"], centre=centre, summary=summary)
