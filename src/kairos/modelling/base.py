"""Shared primitives for survival model families: step-function baselines, the Breslow estimator
and the structured errors for unsupported fits and routes.

A baseline cumulative hazard is a right-continuous step function: its value at ``t`` is the value
at the last event time on or before ``t``, and zero before the first event time. No
interpolation between event times.
"""
from __future__ import annotations

import numpy as np

CAUSES = {"svd": 1, "death": 2, "replacement": 3}


class UnsupportedFitError(Exception):
    """A cause (or cause within a route) has no supported estimator; there is no prediction."""

    def __init__(self, cause: str, route: str | None, reason: str):
        self.cause, self.route, self.reason = cause, route, reason
        where = f"{cause}" + (f" / route {route}" if route else "")
        super().__init__(f"unsupported fit for {where}: {reason}")


class UnsupportedRouteError(Exception):
    """The request's route is missing or not among the routes the model supports."""

    def __init__(self, route, supported):
        self.route, self.supported = route, list(supported)
        super().__init__(f"route {route!r} is not supported by this model (supported: {self.supported}); "
                         "no route is substituted")


class LegacyBundleError(Exception):
    """A bundle written before the current schema; retrain instead of reinterpreting it."""


def step_eval(knots: np.ndarray, values: np.ndarray, times) -> np.ndarray:
    """Right-continuous step function through (knots, values), zero before the first knot."""
    knots = np.asarray(knots, dtype=float)
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    if len(knots) == 0:
        return np.zeros(times.shape)
    idx = np.searchsorted(knots, times, side="right") - 1
    return np.where(idx >= 0, values[np.clip(idx, 0, len(values) - 1)], 0.0)


def breslow_step(time, event_indicator, log_risk=None) -> tuple[np.ndarray, np.ndarray]:
    """Breslow cumulative baseline hazard: knots at distinct event times, value
    sum_{t_j <= t} d_j / sum_{i: T_i >= t_j} exp(log_risk_i). With ``log_risk`` absent this is the
    Nelson-Aalen estimator (a covariate-free cause-specific baseline)."""
    time = np.asarray(time, dtype=float)
    ev = np.asarray(event_indicator).astype(bool)
    risk = np.exp(np.asarray(log_risk, dtype=float)) if log_risk is not None else np.ones(len(time))
    if not ev.any():
        return np.array([]), np.array([])
    order = np.argsort(time)
    t_sorted, r_sorted = time[order], risk[order]
    # risk set sum for each distinct event time: rows with T >= t_j
    tail = np.cumsum(r_sorted[::-1])[::-1]
    knots = np.unique(time[ev])
    first = np.searchsorted(t_sorted, knots, side="left")
    at_risk = tail[first]
    d = np.bincount(np.searchsorted(knots, time[ev]), minlength=len(knots)).astype(float)
    return knots, np.cumsum(d / at_risk)


def followup_support_time(time, min_at_risk: int) -> float:
    """Largest time with at least ``min_at_risk`` rows still under observation (0 if fewer rows)."""
    t = np.sort(np.asarray(time, dtype=float))[::-1]
    if len(t) < min_at_risk or min_at_risk <= 0:
        return 0.0
    return float(t[min_at_risk - 1])


ADAPTER_SCHEMA_VERSION = 2


def library_versions() -> dict:
    """Versions whose major.minor must match between training and serving."""
    import lifelines
    import sklearn

    out = {"scikit-learn": sklearn.__version__, "lifelines": lifelines.__version__}
    try:
        import sksurv

        out["scikit-survival"] = sksurv.__version__
    except ImportError:  # pragma: no cover - only the Cox family is available
        out["scikit-survival"] = None
    return out


class SupportMixin:
    """Row support, route checks and support summaries shared by the model families.

    Expects ``strata``, ``stratum_keys_``, ``support_`` (cause -> record with ``status``, ``reason``
    and ``strata`` -> {``supported``, ``reason``, ``event_patients``}), ``followup_support_``,
    ``gates``, ``grid_`` and ``family``."""

    def _keys(self, frame):
        import numpy as np

        if not self.strata:
            return np.full(len(frame), "all", dtype=object)
        parts = [frame[s].astype("object").where(frame[s].notna(), None).map(lambda v: "" if v is None else str(v))
                 for s in self.strata]
        return np.asarray(["|".join(t) for t in zip(*parts)], dtype=object)

    @property
    def supported_routes(self) -> list[str]:
        return [k for k in self.stratum_keys_ if all(self.support_[c]["strata"].get(k, {}).get("supported") for c in CAUSES)]

    def row_support(self, frame):
        """(supported mask, reason per row) for rows of ``frame`` (needs the strata columns)."""
        import numpy as np

        keys = self._keys(frame)
        reasons = np.full(len(frame), "", dtype=object)
        for i, k in enumerate(keys):
            if k not in self.stratum_keys_:
                reasons[i] = "unknown_device_or_route"
                continue
            bad = [c for c in CAUSES if not self.support_[c]["strata"][k]["supported"]]
            if bad:
                status = {self.support_[c]["status"] for c in bad}
                reasons[i] = ("fit_failed" if status == {"fit_failed"} else "insufficient_events") + ":" + ",".join(bad)
        return reasons == "", reasons

    def _check_rows(self, X):
        keys = self._keys(X)
        unknown = sorted({k for k in keys if k not in self.stratum_keys_})
        if unknown:
            raise UnsupportedRouteError(unknown[0], self.supported_routes)
        for cause in CAUSES:
            for k in set(keys):
                s = self.support_[cause]["strata"][k]
                if not s["supported"]:
                    raise UnsupportedFitError(cause, k, s["reason"])
        return keys

    def knots(self):
        import numpy as np

        pts = [np.asarray(kn) for per in self.baselines_.values() for kn, _ in per.values()]
        return np.unique(np.concatenate(pts)) if pts else np.array([])

    def support_summary(self) -> dict:
        return {"family": self.family, "adapter_version": getattr(self, "adapter_version", "1"), "gates": dict(self.gates),
                "causes": self.support_, "followup_support_years": self.followup_support_,
                "supported_routes": self.supported_routes, "grid_points": int(len(self.grid))}
