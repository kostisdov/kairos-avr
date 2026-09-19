"""Nested, patient-grouped tuning of the gradient boosting family (CR-08, WP-C3).

Hyperparameters are chosen only on inner folds of the rows given (an outer training set, or the
whole development cohort when a bundle is trained). Every inner fit refits eligibility, the
feature pipeline, rare levels, event support and baselines on the inner training rows. For each
learning rate and depth one model with the largest number of trees is fitted, and smaller tree
counts are scored from its first stages (identical to refitting).

Objective: mean IPCW Brier score over the three causes (SVD, death, non-SVD replacement) and the
three horizons on the inner held-out rows with supported predictions, averaged over inner folds,
unscaled (default decision 3; the most frequent cause, death, therefore weighs most). The scaled
index (IPA) is recorded for information and never used for selection. One shared tuple is chosen
for the whole competing-risk system. Ties within 1e-6 go to the simplest tuple (smaller depth, then
fewer trees, then smaller learning rate).

An inner fold that fails to fit or has no held-out row with supported predictions is skipped for that
tuple and recorded; a tuple is scored on the inner folds that worked. When no tuple can be scored in any
inner fold, the configured default hyperparameters are used with ``status: fallback_default`` (owner
decision of 17 September 2026: synthetic evaluations are not failed for this).
"""
from __future__ import annotations

import itertools
import os
import time

import numpy as np
import pandas as pd

from kairos.evaluation.ladder import patient_folds
from kairos.evaluation.metrics import KMCensoring, aalen_johansen, ipcw_weights, outcome

CAUSES = ("svd", "death", "replacement")


def _grid(cfg: dict, mode: str) -> dict:
    gb = cfg["families"]["gradient_boosting"]
    if os.environ.get("KAIROS_BOOSTING_GRID") == "budget":   # time-boxed exploratory comparison (deviation 46)
        return gb["budget_grid"]
    return gb["quick_grid"] if mode == "quick" else gb["grid"]


def _nanmean(values) -> float:
    """Mean of the finite values; NaN (without a warning) when there are none."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else float("nan")


def _brier(pred, time_, event, state, tau, G):
    w = ipcw_weights(time_, event, tau, G).w
    y, _ = outcome(time_, event, state, tau)
    b = float(np.sum(w * (y - pred) ** 2) / len(pred))
    marg = aalen_johansen(time_, event, state, tau)
    null = float(np.sum(w * (y - marg) ** 2) / len(pred))
    return b, (1.0 - b / null) if null > 0 else float("nan")


def tune_boosting(lm_train: pd.DataFrame, blocks: list[str], cfg: dict, seed: int, mode: str = "full") -> dict:
    from kairos.modelling.train import default_boosting_hyperparameters, fit_step

    gb = cfg["families"]["gradient_boosting"]
    grid = _grid(cfg, mode)
    stages = sorted(int(n) for n in grid["n_estimators"])
    horizons = [float(h) for h in cfg["horizons_years"]]
    near = float(cfg["near_term_horizon_months"]) / 12.0
    inner = int(gb["inner_splits"])
    folds = patient_folds(lm_train["patient_id"], inner, seed)
    fixed = {**default_boosting_hyperparameters(cfg), **gb["fixed"]}
    t0 = time.time()
    if all(len(v) == 1 for v in grid.values()):   # a single candidate needs no inner cross-validation
        selected = {**fixed, "n_estimators": int(grid["n_estimators"][0]), "learning_rate": float(grid["learning_rate"][0]),
                    "max_depth": int(grid["max_depth"][0])}
        return {"selected": selected, "status": "fixed_single_candidate", "table": [], "grid": grid, "inner_splits": 0,
                "reason": "single-candidate grid: no inner tuning", "seed": seed, "mode": mode, "seconds": 0.0}
    scores = {}
    fold_rows = {}
    for lr, depth in itertools.product(grid["learning_rate"], grid["max_depth"]):
        per_stage = {s: [] for s in stages}
        skipped: dict = {}
        for j in range(inner):
            tr, te = lm_train[folds != j], lm_train[folds == j]
            hp = {**fixed, "n_estimators": max(stages), "learning_rate": float(lr), "max_depth": int(depth)}
            try:
                pipe, model, *_ = fit_step(tr, blocks, cfg, mode, "gradient_boosting", hp, keep_training_data=True)
            except Exception as ex:  # noqa: BLE001 - the inner fold is skipped for this tuple, with the reason recorded
                skipped[j] = f"{type(ex).__name__}: {ex}"[:300]
                continue
            X = pipe.transform(te)
            for s in model.strata:
                X[s] = te[s].astype(str).to_numpy()
            ok, _ = model.row_support(X)
            fold_rows[j] = {"rows": int(len(te)), "valid_rows": int(ok.sum())}
            if not ok.any():
                skipped[j] = "no held-out row with supported predictions"
                continue
            probs = model.staged_state_probabilities(X.loc[ok], stages, horizons, near)
            t = te["time"].to_numpy(dtype=float)[ok]
            e = te["event"].to_numpy()[ok]
            G = KMCensoring().fit(t, e)
            for s in stages:
                cell = {}
                for c in CAUSES:
                    for h_i, h in enumerate(horizons):
                        cell[f"{c}@{h:g}"] = _brier(probs[s][c][:, h_i], t, e, c, h, G)
                per_stage[s].append(cell)
            model.drop_training_data()
        for s in stages:
            key = (int(depth), int(s), float(lr))
            cells = per_stage[s]
            if not cells:
                scores[key] = {"valid": False, "reason": "no inner fold could be scored", "skipped_inner_folds": skipped}
                continue
            by = {k: float(np.mean([c[k][0] for c in cells])) for k in cells[0]}
            ipa = {k: _nanmean([c[k][1] for c in cells]) for k in cells[0]}
            scores[key] = {"valid": True, "reason": "", "mean_brier": float(np.mean(list(by.values()))), "brier_by": by,
                           "mean_ipa_information_only": _nanmean(list(ipa.values())),
                           "scored_inner_folds": len(cells), "skipped_inner_folds": skipped}
    valid = {k: v for k, v in scores.items() if v["valid"]}
    table = [{"max_depth": k[0], "n_estimators": k[1], "learning_rate": k[2], **v} for k, v in sorted(scores.items())]
    if not valid:
        return {"selected": dict(fixed), "status": "fallback_default", "table": table, "grid": grid, "inner_splits": inner,
                "reason": "no tuple could be scored in any inner fold; configured default hyperparameters used",
                "seed": seed, "mode": mode, "seconds": round(time.time() - t0, 1)}
    best = min(v["mean_brier"] for v in valid.values())
    chosen = min(k for k, v in valid.items() if v["mean_brier"] <= best + 1e-6)   # simplest among ties
    selected = {**fixed, "max_depth": chosen[0], "n_estimators": chosen[1], "learning_rate": chosen[2]}
    return {"selected": selected, "status": "ok", "objective": gb["objective"], "table": table, "grid": grid,
            "inner_splits": inner, "seed": seed, "mode": mode, "inner_fold_rows": fold_rows,
            "patients": int(lm_train["patient_id"].nunique()), "seconds": round(time.time() - t0, 1)}
