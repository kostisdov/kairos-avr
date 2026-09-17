"""Evaluation metrics for cumulative-incidence predictions under right censoring (CR-07).

Outcome at horizon ``tau`` for a state:

* a cause (SVD, death, non-SVD replacement): ``Y = 1`` if that cause occurred by ``tau``; ``Y = 0``
  if another cause occurred by ``tau`` (competing events stay controls with known status) or the
  subject was observed event-free through ``tau``;
* ``alive_intact``: ``Y = 1`` if observed event-free through ``tau``; ``Y = 0`` for any event by ``tau``.

Status is unknown for subjects censored before ``tau``. Landmark rows are administratively
censored at the horizon; a row censored exactly at ``tau`` is observed through it.

Inverse probability of censoring weights (IPCW): ``1 / G(T-)`` for an event by ``tau``,
``1 / G(tau-)`` for a subject observed through ``tau``, zero otherwise, with
``G(t-) = P(C >= t)`` from the censoring model (Kaplan-Meier on the censoring indicator by default:
it assumes censoring independent of covariates; :class:`CoxCensoring` is the conditional
sensitivity model). Weights are never clipped; a censoring survival below the configured minimum at
``tau`` is reported as a support failure.

Metrics: IPCW Brier score (``sum w (Y - p)^2 / n``), null Brier from the marginal Aalen-Johansen
estimate and IPA; ``obs_minus_pred`` (observed Aalen-Johansen minus mean predicted, descriptive,
probability scale); IPCW logistic recalibration ``logit P(Y=1) = a + b logit(p)`` (joint intercept and
slope, ideal 0 and 1) and the offset intercept (``b`` fixed at 1: calibration-in-the-large on the
log-odds scale); a censoring-aware generalised Brier decomposition that reconstructs the
weight-normalised Brier exactly; time-dependent IPCW AUC (cases have the state by ``tau``, controls
are observed event-free through ``tau``; for ``alive_intact`` the roles swap), supporting evidence
only; patient-cluster bootstrap intervals that record failed resamples.

The IPCW logistic approach follows "Calibration plots for multistate risk predictions models"
(arXiv 2308.13394); competing-risk definitions follow "Validation of prediction models in the
presence of competing risks" (BMJ 2022;377:e069249).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

STATE_CODES = {"svd": 1, "death": 2, "replacement": 3}
EPS_LOGIT = 1e-6
TIME_TOL = 1e-9


def _state(state) -> int | str:
    if isinstance(state, str):
        return state if state == "alive_intact" else STATE_CODES[state]
    return int(state)


# --- censoring models -------------------------------------------------------------------------
class KMCensoring:
    """Kaplan-Meier estimate of the censoring survival G on the censoring indicator (every subject
    with ``T >= c`` is at risk of censoring at ``c``). ``G_left(t) = P(C >= t)``: censorings at exactly
    ``t`` are not subtracted (left limit)."""
    name = "kaplan_meier"

    def fit(self, time, event, X=None) -> KMCensoring:
        time = np.asarray(time, dtype=float)
        cens = np.asarray(event) == 0
        self.knots_ = np.unique(time[cens])
        t_sorted = np.sort(time)
        at_risk = len(time) - np.searchsorted(t_sorted, self.knots_, side="left")
        d = np.bincount(np.searchsorted(self.knots_, time[cens]), minlength=len(self.knots_)).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            factors = np.where(at_risk > 0, 1.0 - d / np.maximum(at_risk, 1), 0.0)
        self.values_ = np.cumprod(factors)
        return self

    def G_left(self, t, X=None) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        idx = np.searchsorted(self.knots_, t, side="left") - 1
        return np.where(idx >= 0, self.values_[np.clip(idx, 0, max(len(self.values_) - 1, 0))] if len(self.values_) else 1.0, 1.0)


class CoxCensoring:
    """Conditional censoring model (sensitivity): Cox model for the censoring indicator with the
    given covariates; ``G_left(t, X) = exp(-H0(t-) exp(lp))`` with a step baseline."""
    name = "cox_conditional"

    def __init__(self, penalizer: float = 0.05):
        self.penalizer = penalizer

    def fit(self, time, event, X: pd.DataFrame) -> CoxCensoring:
        from lifelines import CoxPHFitter

        df = X.astype(float).copy()
        sd = df.std(ddof=0)
        self.columns_ = [c for c in df.columns if sd[c] > 1e-9]
        df = df[self.columns_]
        df["_t"] = np.asarray(time, dtype=float)
        df["_c"] = (np.asarray(event) == 0).astype(int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.cph_ = CoxPHFitter(penalizer=self.penalizer).fit(df, "_t", "_c")
        b = self.cph_.baseline_cumulative_hazard_.iloc[:, 0]
        v = b.to_numpy(dtype=float)
        jump = np.diff(v, prepend=0.0) > 0
        self.knots_, self.values_ = b.index.to_numpy(dtype=float)[jump], v[jump]
        return self

    def G_left(self, t, X: pd.DataFrame) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        idx = np.searchsorted(self.knots_, t, side="left") - 1
        H0 = np.where(idx >= 0, self.values_[np.clip(idx, 0, len(self.values_) - 1)], 0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            risk = np.asarray(self.cph_.predict_partial_hazard(X[self.columns_].astype(float)), dtype=float).ravel()
        return np.exp(-H0 * risk)


def censoring_model(time, event, kind: str = "kaplan_meier", X: pd.DataFrame | None = None):
    if kind == "kaplan_meier":
        return KMCensoring().fit(time, event)
    if kind == "cox_conditional":
        if X is None:
            raise ValueError("the conditional censoring model needs covariates")
        return CoxCensoring().fit(time, event, X)
    raise ValueError(f"unknown censoring model {kind!r}")


# --- outcome and weights ----------------------------------------------------------------------
def event_free_at(time, event, tau: float) -> np.ndarray:
    """Observed event-free through ``tau``: followed beyond it, or censored exactly at it.
    Landmark rows are administratively censored at the horizon; they are controls, not losses."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event)
    return (time > tau) | ((time >= tau - TIME_TOL) & (event == 0))


def outcome(time, event, state, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """(Y, known status) for ``state`` at ``tau``."""
    s = _state(state)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event)
    ev = (time <= tau) & (event != 0)
    free = event_free_at(time, event, tau) & ~ev
    known = ev | free
    y = free.astype(float) if s == "alive_intact" else (ev & (event == s)).astype(float)
    return y, known


@dataclass
class WeightResult:
    w: np.ndarray
    G_at_tau: float
    ess: float
    max_w: float
    p99_w: float
    n_zero: int
    support_ok: bool
    reason: str = ""


def ipcw_weights(time, event, tau: float, G, X=None, min_censoring_survival: float = 0.0) -> WeightResult:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event)
    w = np.zeros(len(time))
    ev = (time <= tau) & (event != 0)
    free = event_free_at(time, event, tau) & ~ev
    g_ev = G.G_left(time[ev], X.loc[ev] if X is not None else None) if ev.any() else np.array([])
    g_tau_rows = G.G_left(np.full(int(free.sum()), tau), X.loc[free] if X is not None else None) if free.any() else np.array([])
    g_tau = float(np.min(g_tau_rows)) if len(g_tau_rows) else float(G.G_left(np.array([tau]))[0]) if X is None else 1.0
    with np.errstate(divide="ignore"):
        w[ev] = np.where(g_ev > 0, 1.0 / np.where(g_ev > 0, g_ev, 1.0), 0.0)
        w[free] = np.where(g_tau_rows > 0, 1.0 / np.where(g_tau_rows > 0, g_tau_rows, 1.0), 0.0)
    pos = w[w > 0]
    ess = float(pos.sum() ** 2 / np.sum(pos ** 2)) if len(pos) else 0.0
    ok = bool(g_tau >= min_censoring_survival and g_tau > 0)
    reason = "" if ok else f"censoring survival at the horizon {g_tau:.3g} below {min_censoring_survival:g}"
    return WeightResult(w=w, G_at_tau=g_tau, ess=ess, max_w=float(pos.max()) if len(pos) else 0.0,
                        p99_w=float(np.percentile(pos, 99)) if len(pos) else 0.0, n_zero=int((w == 0).sum()),
                        support_ok=ok, reason=reason)


# --- marginal estimates and Brier --------------------------------------------------------------
def aalen_johansen(time, event, cause, tau: float) -> float:
    """Marginal probability of ``cause`` by ``tau`` (or of being event-free for ``alive_intact``)."""
    s = _state(cause)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event)
    order = np.argsort(time)
    t, e = time[order], event[order]
    uniq = np.unique(t[(e != 0) & (t <= tau)])
    surv, cifs = 1.0, {1: 0.0, 2: 0.0, 3: 0.0}
    n = len(t)
    for u in uniq:
        at_risk = n - np.searchsorted(t, u, side="left")
        if at_risk <= 0:
            break
        d_all = np.sum((t == u) & (e != 0))
        for c in cifs:
            cifs[c] += surv * np.sum((t == u) & (e == c)) / at_risk
        surv *= 1.0 - d_all / at_risk
    return float(surv) if s == "alive_intact" else float(cifs.get(s, 0.0))


def brier_ipcw(pred, time, event, state, tau: float, G=None, sample_weight=None) -> float:
    pred = np.asarray(pred, dtype=float)
    G = G or KMCensoring().fit(time, event)
    wr = ipcw_weights(time, event, tau, G)
    y, _ = outcome(time, event, state, tau)
    sw = np.ones(len(pred)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    return float(np.sum(sw * wr.w * (y - pred) ** 2) / np.sum(sw))


def brier_decomposition(pred, y, w, n_bins: int = 10) -> dict:
    """Generalised (Stephenson 2008) decomposition of the weight-normalised Brier score:
    BS = reliability - resolution + uncertainty + within-bin variance + within-bin covariance."""
    pred, y, w = (np.asarray(a, dtype=float) for a in (pred, y, w))
    m = w > 0
    if not m.any():
        return {"status": "no_known_status"}
    p, yy, v = pred[m], y[m], w[m] / w[m].sum()
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, max(len(edges) - 2, 0))
    ybar = float(np.sum(v * yy))
    rel = res = wbv = wbc = 0.0
    for k in np.unique(b):
        i = b == k
        vk = v[i].sum()
        pk, yk = np.sum(v[i] * p[i]) / vk, np.sum(v[i] * yy[i]) / vk
        rel += vk * (pk - yk) ** 2
        res += vk * (yk - ybar) ** 2
        wbv += np.sum(v[i] * (p[i] - pk) ** 2)
        wbc += -2.0 * np.sum(v[i] * (p[i] - pk) * (yy[i] - yk))
    unc = ybar * (1.0 - ybar)
    bs = float(np.sum(v * (p - yy) ** 2))
    recon = rel - res + unc + wbv + wbc
    return {"status": "ok", "brier_normalised": bs, "reliability": float(rel), "resolution": float(res),
            "uncertainty": float(unc), "within_bin_variance": float(wbv), "within_bin_covariance": float(wbc),
            "reconstruction_error": float(abs(recon - bs)), "n_bins": int(len(np.unique(b))),
            "weight_sum_over_n": float(w[m].sum() / len(w))}


# --- formal calibration -----------------------------------------------------------------------
def _logit(p, eps=EPS_LOGIT):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _expit(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def weighted_logistic(Z: np.ndarray, y, w, offset=None, max_iter: int = 50, tol: float = 1e-10,
                      max_abs_coef: float = 15.0) -> tuple[np.ndarray | None, str]:
    """Unpenalised weighted logistic regression by Newton-Raphson; (coefficients, status)."""
    Z = np.asarray(Z, dtype=float)
    y, w = np.asarray(y, dtype=float), np.asarray(w, dtype=float)
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, dtype=float)
    beta = np.zeros(Z.shape[1])
    for _ in range(max_iter):
        mu = _expit(Z @ beta + off)
        grad = Z.T @ (w * (y - mu))
        H = (Z * (w * mu * (1 - mu))[:, None]).T @ Z
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            return None, "separation"
        if not np.all(np.isfinite(step)):
            return None, "separation"
        beta = beta + step
        if np.max(np.abs(beta)) > max_abs_coef:
            return None, "separation"
        if np.max(np.abs(step)) < tol:
            return beta, "ok"
    return None, "not_converged"


@dataclass
class CalibrationFit:
    intercept_joint: float = float("nan")
    slope: float = float("nan")
    intercept_offset: float = float("nan")
    status: str = "ok"
    reason: str = ""
    n_events: int = 0
    n_controls: int = 0
    ess: float = 0.0
    eps: float = EPS_LOGIT


def ipcw_logistic_calibration(pred, time, event, state, tau: float, G=None, sample_weight=None,
                              eps: float = EPS_LOGIT, weights: WeightResult | None = None) -> CalibrationFit:
    pred = np.asarray(pred, dtype=float)
    G = G or KMCensoring().fit(time, event)
    wr = weights or ipcw_weights(time, event, tau, G)
    y, _ = outcome(time, event, state, tau)
    sw = np.ones(len(pred)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    w = wr.w * sw
    m = w > 0
    fit = CalibrationFit(n_events=int((y[m] == 1).sum()), n_controls=int((y[m] == 0).sum()), ess=wr.ess, eps=eps)
    if fit.n_events == 0:
        fit.status, fit.reason = "no_events", "no subject with the outcome and known status by the horizon"
        return fit
    if fit.n_controls == 0:
        fit.status, fit.reason = "no_controls", "no subject without the outcome and known status"
        return fit
    x = _logit(pred[m], eps)
    if np.var(x) < 1e-12:
        fit.status, fit.reason = "no_variation", "predictions do not vary; slope undefined"
    else:
        beta, st = weighted_logistic(np.column_stack([np.ones(m.sum()), x]), y[m], w[m])
        if beta is None:
            fit.status, fit.reason = st, f"joint recalibration model: {st}"
        else:
            fit.intercept_joint, fit.slope = float(beta[0]), float(beta[1])
    beta0, st0 = weighted_logistic(np.ones((m.sum(), 1)), y[m], w[m], offset=x)
    if beta0 is not None:
        fit.intercept_offset = float(beta0[0])
    elif fit.status == "ok":
        fit.status, fit.reason = st0, f"offset model: {st0}"
    return fit


def calibration_slope_prob_legacy_v1(pred, time, event, state, tau: float, G=None) -> float:
    """Version-1 metric kept for comparison only: IPCW linear regression slope of Y on the
    predicted probability (probability scale)."""
    pred = np.asarray(pred, dtype=float)
    G = G or KMCensoring().fit(time, event)
    w = ipcw_weights(time, event, tau, G).w
    y, _ = outcome(time, event, state, tau)
    m = w > 0
    if m.sum() < 10 or np.var(pred[m]) < 1e-12:
        return float("nan")
    pm, ym = np.average(pred[m], weights=w[m]), np.average(y[m], weights=w[m])
    var = np.sum(w[m] * (pred[m] - pm) ** 2)
    return float(np.sum(w[m] * (pred[m] - pm) * (y[m] - ym)) / var) if var > 0 else float("nan")


def calibration_curve(pred, time, event, state, tau: float, n_bins: int = 10) -> pd.DataFrame:
    """Binned display (Aalen-Johansen within prediction deciles). A display, not an estimator."""
    pred = np.asarray(pred, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event)
    q = np.quantile(pred, np.linspace(0, 1, n_bins + 1))
    q[0], q[-1] = -np.inf, np.inf
    edges = np.unique(q)
    b = np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for k in range(len(edges) - 1):
        m = b == k
        if m.sum() < 5:
            continue
        rows.append({"bin": k, "n": int(m.sum()), "mean_predicted": float(pred[m].mean()),
                     "observed": aalen_johansen(time[m], event[m], state, tau), "kind": "binned display"})
    return pd.DataFrame(rows)


def formal_calibration_curve(pred, time, event, state, tau: float, G=None, n_points: int = 40) -> pd.DataFrame:
    """IPCW-weighted logistic calibration curve: restricted cubic spline (3 knots) of logit(p)."""
    from kairos.modelling.vitamin_k import rcs_basis

    pred = np.asarray(pred, dtype=float)
    G = G or KMCensoring().fit(time, event)
    wr = ipcw_weights(time, event, tau, G)
    y, _ = outcome(time, event, state, tau)
    m = wr.w > 0
    if m.sum() < 20 or y[m].sum() == 0 or y[m].sum() == m.sum():
        return pd.DataFrame(columns=["predicted", "observed", "kind"])
    x = _logit(pred[m])
    knots = np.quantile(x, [0.1, 0.5, 0.9])
    if len(np.unique(knots)) < 3:
        return pd.DataFrame(columns=["predicted", "observed", "kind"])
    Z = np.column_stack([np.ones(m.sum()), rcs_basis(x, knots)])
    beta, st = weighted_logistic(Z, y[m], wr.w[m])
    if beta is None:
        return pd.DataFrame(columns=["predicted", "observed", "kind"])
    grid_p = np.quantile(pred[m], np.linspace(0.01, 0.99, n_points))
    gx = _logit(grid_p)
    fitted = _expit(np.column_stack([np.ones(len(gx)), rcs_basis(gx, knots)]) @ beta)
    return pd.DataFrame({"predicted": grid_p, "observed": fitted, "kind": "IPCW logistic spline (formal)"})


# --- discrimination -------------------------------------------------------------------------------
def auc_ipcw(pred, time, event, state, tau: float, G=None) -> float:
    pred = np.asarray(pred, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event)
    s = _state(state)
    G = G or KMCensoring().fit(time, event)
    wr = ipcw_weights(time, event, tau, G)
    ev = (time <= tau) & (event != 0)
    free = event_free_at(time, event, tau) & ~ev
    if s == "alive_intact":
        cases, controls, score = free, ev, pred
    else:
        cases, controls, score = ev & (event == s), free, pred
    if cases.sum() == 0 or controls.sum() == 0:
        return float("nan")
    wc, wk = wr.w[cases], wr.w[controls]
    pc, pk = score[cases], score[controls]
    order = np.argsort(pk)
    pk_sorted, wk_sorted = pk[order], wk[order]
    cum = np.concatenate([[0.0], np.cumsum(wk_sorted)])
    lower = cum[np.searchsorted(pk_sorted, pc, side="left")]
    upper = cum[np.searchsorted(pk_sorted, pc, side="right")]
    num = np.sum(wc * (lower + 0.5 * (upper - lower)))
    den = np.sum(wc) * np.sum(wk)
    return float(num / den) if den > 0 else float("nan")


# --- one call per state -------------------------------------------------------------------------
def evaluate_predictions(pred, time, event, state, tau: float, G=None, sample_weight=None,
                         min_censoring_survival: float = 0.0, decomposition: bool = False) -> dict:
    pred = np.asarray(pred, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event)
    G = G or KMCensoring().fit(time, event)
    wr = ipcw_weights(time, event, tau, G, min_censoring_survival=min_censoring_survival)
    y, known = outcome(time, event, state, tau)
    sw = np.ones(len(pred)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    brier = float(np.sum(sw * wr.w * (y - pred) ** 2) / np.sum(sw))
    marginal = aalen_johansen(time, event, state, tau)
    brier_null = float(np.sum(sw * wr.w * (y - marginal) ** 2) / np.sum(sw))
    cal = ipcw_logistic_calibration(pred, time, event, state, tau, G, sample_weight, weights=wr)
    out = {"n": int(len(time)), "n_events": int(np.sum((y == 1) & known)), "n_known": int(known.sum()),
           "observed": marginal, "mean_predicted": float(np.average(pred, weights=sw)),
           "brier": brier, "brier_null": brier_null,
           "ipa": float(1.0 - brier / brier_null) if brier_null > 0 else float("nan"),
           "obs_minus_pred": float(marginal - np.average(pred, weights=sw)),
           "cal_intercept_offset": cal.intercept_offset, "cal_intercept_joint": cal.intercept_joint,
           "cal_slope_logit": cal.slope, "cal_status": cal.status, "cal_reason": cal.reason,
           "cal_slope_prob_legacy_v1": calibration_slope_prob_legacy_v1(pred, time, event, state, tau, G),
           "auc": auc_ipcw(pred, time, event, state, tau, G),
           "G_at_tau": wr.G_at_tau, "ess": wr.ess, "max_weight": wr.max_w, "censoring_support_ok": wr.support_ok,
           "censoring_model": getattr(G, "name", "custom")}
    if decomposition:
        out["decomposition"] = brier_decomposition(pred, y, wr.w * sw)
    return out


# --- bootstrap ---------------------------------------------------------------------------------------
@dataclass
class BootstrapResult:
    intervals: dict = field(default_factory=dict)       # key -> (lo, hi) or None
    n_requested: int = 0
    n_valid: dict = field(default_factory=dict)
    failures: dict = field(default_factory=dict)        # exception type -> count
    kind: str = "fixed_oof"
    min_valid: int = 10
    min_success_fraction: float = 0.0

    def __getitem__(self, key) -> tuple[float, float]:
        iv = self.intervals.get(key)
        return iv if iv is not None else (float("nan"), float("nan"))

    def available(self, key) -> bool:
        return self.intervals.get(key) is not None

    def card(self) -> dict:
        return {"kind": self.kind, "n_requested": self.n_requested, "n_valid": self.n_valid, "failures": self.failures,
                "min_valid": self.min_valid, "min_success_fraction": self.min_success_fraction}


def bootstrap_by_patient(fn, patient_ids, n_boot: int = 200, seed: int = 0, keys=("brier", "auc", "ipa"),
                         min_valid: int = 10, min_success_fraction: float = 0.0, kind: str = "fixed_oof") -> BootstrapResult:
    """fn(row_index_array) -> dict of metrics. Resamples patients with replacement, keeping all
    rows of a patient together (complete histories). An interval is available only with at least
    ``min_valid`` finite replicates and a success fraction of at least ``min_success_fraction``;
    ``kind`` states whether the models were refitted inside the resamples (``full_refit``) or the
    out-of-fold predictions were held fixed (``fixed_oof``)."""
    patient_ids = np.asarray(patient_ids)
    uniq, inverse = np.unique(patient_ids, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.searchsorted(inverse[order], np.arange(len(uniq) + 1))
    groups = [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    samples: dict = {k: [] for k in keys}
    failures: dict = {}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([groups[p] for p in pick]) if len(uniq) else np.array([], dtype=int)
        try:
            r = fn(idx)
        except Exception as ex:  # noqa: BLE001 - a degenerate resample is counted, not hidden
            failures[type(ex).__name__] = failures.get(type(ex).__name__, 0) + 1
            continue
        for k in keys:
            samples[k].append(r.get(k, np.nan))
    res = BootstrapResult(n_requested=n_boot, failures=failures, kind=kind, min_valid=min_valid,
                          min_success_fraction=min_success_fraction)
    for k in keys:
        v = np.asarray(samples[k], dtype=float)
        v = v[np.isfinite(v)]
        res.n_valid[k] = int(len(v))
        ok = len(v) >= max(min_valid, 1) and (n_boot == 0 or len(v) / n_boot >= min_success_fraction)
        res.intervals[k] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if ok else None
    return res
