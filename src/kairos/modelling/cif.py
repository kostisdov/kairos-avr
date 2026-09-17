"""Cumulative incidence from cause-specific cumulative hazards (hazard contract, CR-05).

Contract, shared by every model family:

1. an adapter returns, per cause, the cumulative *integrated* hazard ``H_k`` as an array
   ``(n, T)`` on a strictly increasing grid with ``grid[0] == 0`` and ``H_k[:, 0] == 0``;
2. fitted baselines are right-continuous step functions (value at the last event time on or
   before ``t``), never linearly interpolated;
3. the grid contains zero, every requested horizon and every baseline event time up to the
   maximum time (:func:`build_grid`), so each jump sits at the right end of one interval and
   horizons are exact grid points;
4. on each interval, with increments ``dH_k``, ``D = sum_k dH_k`` and ``S_prev`` the probability
   of no event at the interval start (piecewise-constant competing hazards)::

       q = -expm1(-D)
       dF_k = S_prev * q * dH_k / D     (0 when D = 0)
       S_next = S_prev * exp(-D)

This update is exact when the cause-specific hazards are constant within an interval; it does not
claim exact recovery of an arbitrary continuous hazard. ``S + sum_k F_k = 1`` holds algebraically.

Invalid input is rejected (:class:`HazardContractError`): non-finite values, mismatched shapes, an
invalid grid, a non-zero value at time zero, or a decrease larger than the round-off tolerance.
Decreases within the tolerance are clipped and reported. Nothing is capped or renormalised.

The product-limit combination used before 17 September 2026 is kept as
:func:`combine_product_limit_legacy` for the convention comparison only.
"""
from __future__ import annotations

import numpy as np

CAUSE_ORDER = ("svd", "death", "replacement")
STATES = ("svd", "death", "replacement", "alive_intact")
INTEGRATION_VERSION = "pch-expm1/1"
DEFAULT_TOL = 1e-9


class HazardContractError(ValueError):
    """Cumulative hazards that violate the hazard contract."""


def build_grid(knots, horizons, max_years: float, refine: int = 1) -> np.ndarray:
    """Zero, the horizons, ``max_years`` and every knot in (0, max_years]; with ``refine > 1``
    each interval is split into ``refine`` equal parts (the knots and horizons stay grid points)."""
    pts = np.asarray(list(knots), dtype=float)
    pts = pts[(pts > 0) & (pts <= max_years)]
    extra = np.asarray([h for h in horizons if 0 < h <= max_years] + [max_years], dtype=float)
    grid = np.unique(np.concatenate([[0.0], pts, extra]))
    if refine > 1 and len(grid) > 1:
        fine = [grid[:-1, None] + (np.diff(grid)[:, None] * np.arange(refine)[None, :] / refine)]
        grid = np.unique(np.concatenate([np.ravel(fine), grid[-1:]]))
    return grid


def validate_grid(grid) -> np.ndarray:
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or len(grid) < 2:
        raise HazardContractError("time grid must be one-dimensional with at least two points")
    if not np.all(np.isfinite(grid)):
        raise HazardContractError("time grid contains non-finite values")
    if grid[0] != 0.0:
        raise HazardContractError(f"time grid must start at 0, not {grid[0]}")
    if np.any(np.diff(grid) <= 0):
        raise HazardContractError("time grid must be strictly increasing")
    return grid


def validate_hazards(cum_hazards: dict, grid, tol: float = DEFAULT_TOL) -> tuple[dict, dict]:
    """Check ``cum_hazards`` against the contract; returns (clean hazards, diagnostics)."""
    grid = validate_grid(grid)
    missing = [c for c in CAUSE_ORDER if c not in cum_hazards]
    if missing:
        raise HazardContractError(f"cumulative hazards missing for causes {missing}")
    shapes = {c: np.shape(cum_hazards[c]) for c in CAUSE_ORDER}
    first = shapes[CAUSE_ORDER[0]]
    if len(first) != 2 or first[1] != len(grid) or any(s != first for s in shapes.values()):
        raise HazardContractError(f"hazard shapes {shapes} do not match (n, {len(grid)})")
    out, max_fix = {}, 0.0
    for c in CAUSE_ORDER:
        H = np.array(cum_hazards[c], dtype=float)
        if not np.all(np.isfinite(H)):
            raise HazardContractError(f"non-finite cumulative hazard for {c}")
        if np.any(np.abs(H[:, 0]) > tol):
            raise HazardContractError(f"cumulative hazard for {c} is not zero at time zero")
        if np.any(H < -tol):
            raise HazardContractError(f"negative cumulative hazard for {c}")
        d = np.diff(H, axis=1)
        if np.any(d < -tol):
            raise HazardContractError(f"cumulative hazard for {c} decreases by {-d.min():.3g} (> tolerance {tol:g})")
        if d.size and d.min() < 0:
            max_fix = max(max_fix, float(-d.min()))
            H = np.maximum.accumulate(H, axis=1)
        H[:, 0] = 0.0
        out[c] = np.maximum(H, 0.0)
    return out, {"tolerance": tol, "max_roundoff_correction": max_fix, "integration_version": INTEGRATION_VERSION}


def combine_cause_specific(cum_hazards: dict, grid, tol: float = DEFAULT_TOL) -> dict:
    """State probabilities on ``grid``: cause -> F_k (n, T) and ``alive_intact`` -> S (n, T)."""
    H, diag = validate_hazards(cum_hazards, grid, tol)
    dH = {c: np.diff(H[c], axis=1) for c in CAUSE_ORDER}
    D = sum(dH[c] for c in CAUSE_ORDER)
    n = D.shape[0]
    log_s = np.concatenate([np.zeros((n, 1)), -np.cumsum(D, axis=1)], axis=1)
    S = np.exp(log_s)
    q = -np.expm1(-D)
    safe = np.where(D > 0, D, 1.0)
    out = {}
    for c in CAUSE_ORDER:
        dF = S[:, :-1] * q * np.where(D > 0, dH[c] / safe, 0.0)
        out[c] = np.concatenate([np.zeros((n, 1)), np.cumsum(dF, axis=1)], axis=1)
    out["alive_intact"] = S
    out["_diagnostics"] = diag
    return out


def combine_product_limit_legacy(cum_hazards: dict, grid) -> dict:
    """The pre-17-September-2026 combination (discrete product limit with the first increment
    dropped and a 0.999 cap). Kept only to quantify the convention change; never used to predict."""
    n, T = next(iter(cum_hazards.values())).shape
    inc = {}
    for c in CAUSE_ORDER:
        Hc = np.asarray(cum_hazards.get(c, np.zeros((n, T))), dtype=float)
        Hc = np.maximum.accumulate(np.nan_to_num(Hc, nan=0.0), axis=1)
        d = np.diff(Hc, axis=1, prepend=0.0)
        d[:, 0] = 0.0
        inc[c] = np.clip(d, 0.0, None)
    total = sum(inc[c] for c in CAUSE_ORDER)
    over = total > 0.999
    if np.any(over):
        scale = np.where(over, 0.999 / np.maximum(total, 1e-12), 1.0)
        for c in CAUSE_ORDER:
            inc[c] = inc[c] * scale
        total = sum(inc[c] for c in CAUSE_ORDER)
    surv = np.cumprod(1.0 - total, axis=1)
    surv_prev = np.concatenate([np.ones((n, 1)), surv[:, :-1]], axis=1)
    cifs = {c: np.cumsum(surv_prev * inc[c], axis=1) for c in CAUSE_ORDER}
    cifs["alive_intact"] = surv
    return cifs


def at_times(curves: np.ndarray, grid, times, tol: float = 1e-9) -> np.ndarray:
    """Right-continuous evaluation at ``times`` (last grid point <= t, with a tolerance so that a
    horizon that is a grid point up to round-off maps to itself)."""
    grid = np.asarray(grid, dtype=float)
    times = np.atleast_1d(np.asarray(times, dtype=float))
    idx = np.searchsorted(grid, times + tol, side="right") - 1
    idx = np.clip(idx, 0, len(grid) - 1)
    return curves[:, idx]


def probabilities_at(cifs: dict, grid, horizons_years, near_term_years: float = 1.0) -> dict:
    out = {c: at_times(cifs[c], grid, horizons_years) for c in STATES}
    out["svd_near_term"] = at_times(cifs["svd"], grid, [near_term_years])[:, 0]
    return out
