"""The protocol's comparator ladder, and validation beyond random patient splits.

* **Comparator ladder** (study_protocol.md section 5): R1 valve age and type, R2 pre-implant,
  R3 reference echo fitted once, R4 current gradient and its change, KAIROS, KAIROS with the
  protocol's surveillance terms removed (time since the last echo, overdue flag) and, stricter, also
  without the number of echoes and the interval since the previous one. Out-of-fold predictions from
  patient-grouped folds, scored on the common rows every comparator supports, with a paired patient
  bootstrap of the IPCW Brier difference against R4, the comparator that carries the protocol's
  incremental-value claim.
* **Temporal split**: fitted on patients implanted up to ``cut_year``, scored on those implanted
  later, the direction a deployed model faces.
* **Leave one design class out**: each valve design class with enough events is held out in turn
  and scored by a model that never saw it (its level maps to ``other``), a transportability check.

SVD at the configured longest horizon (5 years) is the scored quantity. Everything is synthetic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from kairos.evaluation.ladder import patient_folds, predict_states
from kairos.evaluation.metrics import KMCensoring, evaluate_predictions, ipcw_weights, outcome
from kairos.modelling.train import fit_step

KEYS = ("n", "n_events", "observed", "mean_predicted", "brier", "ipa", "auc", "cal_slope_logit")


def comparators(cfg: dict) -> list[tuple[str, list[str]]]:
    return [(c["name"], list(c["blocks"])) for c in cfg["protocol_comparators"]]


def _horizons(cfg: dict) -> list[float]:
    return [float(h) for h in cfg["horizons_years"]]


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, blocks: list[str], cfg: dict, tau: float) -> np.ndarray:
    """SVD cumulative incidence by ``tau`` for ``test`` from a model fitted on ``train``; NaN where the
    model does not support a row."""
    hz = _horizons(cfg)
    pipe, model, *_ = fit_step(train, blocks, cfg)
    probs, _ = predict_states(pipe, model, test, hz, float(cfg["near_term_horizon_months"]) / 12.0)
    return probs["svd"][:, hz.index(tau)]


def score(p: np.ndarray, rows: pd.DataFrame, tau: float, G=None) -> dict:
    t, e = rows["time"].to_numpy(float), rows["event"].to_numpy()
    m = evaluate_predictions(p, t, e, "svd", tau, G or KMCensoring().fit(t, e))
    return {k: (round(float(m[k]), 4) if isinstance(m[k], float) else m[k]) for k in KEYS}


def _row_losses(p: np.ndarray, rows: pd.DataFrame, tau: float) -> np.ndarray:
    t, e = rows["time"].to_numpy(float), rows["event"].to_numpy()
    w = ipcw_weights(t, e, tau, KMCensoring().fit(t, e)).w
    y, _ = outcome(t, e, "svd", tau)
    return w * (y - p) ** 2


def paired_brier_difference(p_a: np.ndarray, p_b: np.ndarray, rows: pd.DataFrame, tau: float,
                            n_boot: int = 200, seed: int = 20260917) -> dict:
    """Brier(a) - Brier(b) with a patient bootstrap interval (censoring weights fixed on the full rows).
    Negative favours ``a``."""
    diff = _row_losses(p_a, rows, tau) - _row_losses(p_b, rows, tau)
    pids = rows["patient_id"].to_numpy()
    uniq, inv = np.unique(pids, return_inverse=True)
    sums, counts = np.bincount(inv, diff), np.bincount(inv)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        k = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        boots.append(float((k * sums).sum() / (k * counts).sum()))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return {"difference": round(float(diff.mean()), 6), "ci95": [round(float(lo), 6), round(float(hi), 6)],
            "patients": int(len(uniq))}


def comparator_ladder(lm: pd.DataFrame, cfg: dict, n_splits: int = 5, seed: int = 20260916,
                      n_boot: int = 200) -> dict:
    tau = _horizons(cfg)[-1]
    folds = patient_folds(lm["patient_id"], n_splits, seed)
    oof = {}
    for name, blocks in comparators(cfg):
        p = np.full(len(lm), np.nan)
        for k in range(n_splits):
            te = folds == k
            p[te] = fit_predict(lm[folds != k], lm[te], blocks, cfg, tau)
        oof[name] = p
    ok = np.all([np.isfinite(p) for p in oof.values()], axis=0)
    rows = lm[ok].reset_index(drop=True)
    G = KMCensoring().fit(rows["time"].to_numpy(float), rows["event"].to_numpy())
    table = pd.DataFrame([{"comparator": n, **score(p[ok], rows, tau, G)} for n, p in oof.items()])
    base = "R4_current_gradient_and_change"
    paired = {f"{n} vs {base}": paired_brier_difference(oof[n][ok], oof[base][ok], rows, tau, n_boot)
              for n in ("KAIROS_core", "KAIROS_surveillance_blinded", "KAIROS_visit_history_blinded",
                        "R3_reference_echo_once") if n in oof}
    paired["KAIROS_core vs R1_valve_age_and_type"] = paired_brier_difference(
        oof["KAIROS_core"][ok], oof["R1_valve_age_and_type"][ok], rows, tau, n_boot)
    return {"table": table, "paired": paired, "rows_scored": int(ok.sum()), "rows": int(len(lm)), "horizon_years": tau}


def temporal_split(lm: pd.DataFrame, cfg: dict, cut_year: int,
                   names: tuple = ("R1_valve_age_and_type", "R4_current_gradient_and_change", "KAIROS_core")) -> dict:
    tau = _horizons(cfg)[-1]
    year = lm["implant_year"].to_numpy(float)
    train, test = lm[year <= cut_year], lm[year > cut_year].reset_index(drop=True)
    blocks = dict(comparators(cfg))
    preds = {n: fit_predict(train, test, blocks[n], cfg, tau) for n in names}
    ok = np.all([np.isfinite(p) for p in preds.values()], axis=0)
    rows = test[ok].reset_index(drop=True)
    G = KMCensoring().fit(rows["time"].to_numpy(float), rows["event"].to_numpy())
    return {"cut_year": cut_year, "train_patients": int(train["patient_id"].nunique()),
            "test_patients": int(rows["patient_id"].nunique()),
            "table": pd.DataFrame([{"comparator": n, **score(p[ok], rows, tau, G)} for n, p in preds.items()])}


def leave_one_class_out(lm: pd.DataFrame, cfg: dict, min_event_patients: int = 10,
                        names: tuple = ("R1_valve_age_and_type", "KAIROS_core")) -> pd.DataFrame:
    tau = _horizons(cfg)[-1]
    blocks = dict(comparators(cfg))
    svd = (lm["event"] == 1) & (lm["time"] <= tau)
    out = []
    for cls, g in lm.groupby("design_class"):
        n_ev = int(lm.loc[g.index][svd.loc[g.index]]["patient_id"].nunique())
        if n_ev < min_event_patients:
            out.append({"held_out_class": cls, "event_patients": n_ev, "status": f"skipped: fewer than {min_event_patients} event patients"})
            continue
        train, test = lm.drop(index=g.index), g.reset_index(drop=True)
        preds = {n: fit_predict(train, test, blocks[n], cfg, tau) for n in names}
        ok = np.all([np.isfinite(p) for p in preds.values()], axis=0)
        rows = test[ok].reset_index(drop=True)
        G = KMCensoring().fit(rows["time"].to_numpy(float), rows["event"].to_numpy())
        for n, p in preds.items():
            out.append({"held_out_class": cls, "event_patients": n_ev, "status": "ok", "comparator": n,
                        **score(p[ok], rows, tau, G)})
    return pd.DataFrame(out)
