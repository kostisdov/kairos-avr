"""Evaluation of dp-ucMGP consumption in the measured substudy.

Two results, both on synthetic scenarios only:

* **Incremental value.** Patient-grouped cross-validation: in each fold the core model with the
  anticoagulant module is fitted on all training rows, the dp-ucMGP offset model on the measured
  training rows, and both are scored on the measured test rows. Brier score, IPA and
  calibration-in-the-large are compared within that subset, with paired patient-bootstrap
  intervals for the Brier difference.
* **Mechanism check.** On the measured rows, the SVD model with the anticoagulant module is fitted
  without and with the dp-ucMGP terms as ordinary covariates. The VKA contrast (current VKA with
  its cumulative exposure versus a factor Xa inhibitor at the same anticoagulant exposure,
  averaged over current VKA rows, drug terms only) should shrink toward zero when the marker
  carries the signal and stay put when the marker is noise. The shrinkage is a reported result,
  never a tuning target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from kairos.evaluation.metrics import bootstrap_by_patient, evaluate_predictions
from kairos.modelling.cause_specific import CauseSpecificCoxModel
from kairos.modelling.cif import combine_cause_specific, probabilities_at
from kairos.modelling.train import fit_step, make_cox, svd_offset_inputs
from kairos.modelling.vitamin_k import (
    VitaminKTerms,
    fit_dp_ucmgp_offset,
    measured_mask,
    substudy_counts,
)

SVD = 1
VKA_GROUPS = ("ac_class_current", "ac_cum_vka_years")


def _step_blocks(cfg: dict) -> tuple[str, list[str]]:
    need = cfg["vitamin_k"]["requires_block"]
    core = [s for s in cfg["ladder"] if s["name"] == "core"][0]["blocks"]
    return f"core_plus_{need}", list(core) + [need]


def _probs(pipe, cox, lm: pd.DataFrame, cfg: dict, vk=None) -> np.ndarray:
    """SVD probabilities at the horizons; NaN rows where the route is not supported."""
    X = pipe.transform(lm)
    for s in cox.strata:
        X[s] = lm[s].astype(str).to_numpy()
    out = np.full((len(lm), len(cfg["horizons_years"])), np.nan)
    ok, _ = cox.row_support(X)
    if not ok.any():
        return out
    H = cox.cumulative_hazards(X.loc[ok])
    if vk is not None:
        H = vk.adjust(H, lm.loc[ok])
    cifs = combine_cause_specific(H, cox.grid)
    out[ok] = probabilities_at(cifs, cox.grid, list(cfg["horizons_years"]), float(cfg["near_term_horizon_months"]) / 12.0)["svd"]
    return out


def vka_contrast(pipe, cox: CauseSpecificCoxModel, lm: pd.DataFrame) -> float:
    """Mean SVD log-hazard contrast of current VKA versus a factor Xa inhibitor at the same total
    anticoagulant exposure, over current VKA rows, using the drug coefficients only."""
    rows = lm[lm["ac_class_current"].astype(str) == "VKA"]
    coef = cox.coefficients("svd")
    if rows.empty or coef.empty:
        return float("nan")
    cf = rows.copy()
    cf["ac_class_current"] = "FXa"
    cf["ac_cum_vka_years"] = 0.0
    cols = [c for c in coef.index if pipe.group(c) in VKA_GROUPS]
    delta = pipe.transform(rows)[cols].to_numpy() - pipe.transform(cf)[cols].to_numpy()
    return float(np.mean(delta @ coef[cols].to_numpy()))


def mechanism_check(lm: pd.DataFrame, cfg: dict, mode: str = "full") -> dict:
    m = measured_mask(lm)
    sub = lm.loc[m].reset_index(drop=True)
    name, blocks = _step_blocks(cfg)
    pipe, cox, feats, dropped, rare, _ = fit_step(sub, blocks, cfg, mode)
    if cox.models_.get("svd") is None:
        return {"fitted": False, "reason": f"no SVD covariate model in the measured rows ({cox.support_['svd']['reason']})"}
    without = vka_contrast(pipe, cox, sub)
    terms = VitaminKTerms(n_knots=int(cfg["vitamin_k"]["spline_knots"])).fit(sub)
    X = pd.concat([pipe.transform(sub), terms.frame(sub)], axis=1)
    mc = cfg["model"]
    joint: CauseSpecificCoxModel = make_cox(cfg, mode)
    joint.fit(X, sub["time"], sub["event"], list(X.columns), sub[list(mc["strata"])], patient_ids=sub["patient_id"])
    with_marker = vka_contrast(pipe, joint, sub)
    coef = joint.coefficients("svd")
    shrinkage = (1.0 - with_marker / without) if abs(without) > 0.05 else float("nan")
    return {"fitted": True, "step": name, "n_rows": int(len(sub)), "n_vka_rows": int((sub["ac_class_current"].astype(str) == "VKA").sum()),
            "vka_log_hr_without_marker": round(without, 4), "vka_log_hr_with_marker": round(with_marker, 4),
            "vka_shrinkage_fraction": None if not np.isfinite(shrinkage) else round(float(shrinkage), 3),
            "marker_coefficients": {c: round(float(coef[c]), 4) for c in terms.columns_ if c in coef.index},
            "definition": "VKA contrast: current VKA with its cumulative exposure versus factor Xa inhibitor at equal anticoagulant "
                          "exposure, SVD log hazard, drug terms only, mean over current-VKA measured rows",
            "label": "synthetic scenario: illustrative, unvalidated"}


def evaluate_vitamin_k(lm: pd.DataFrame, cfg: dict, folds: np.ndarray, n_boot: int = 30, seed: int = 0,
                       mode: str = "full") -> dict:
    counts = substudy_counts(lm, SVD)
    vk_cfg = cfg["vitamin_k"]
    if counts["n_rows_measured"] < int(vk_cfg["min_measured_rows"]) or counts["n_svd_events_measured"] < int(vk_cfg["min_measured_svd_events"]):
        return {"fitted": False, **counts,
                "reason": "dp-ucMGP substudy below the configured minimum rows or SVD events; nothing imputed, no result claimed"}
    name, blocks = _step_blocks(cfg)
    horizons = list(cfg["horizons_years"])
    m = measured_mask(lm)
    base = np.full((len(lm), len(horizons)), np.nan)
    plus = np.full((len(lm), len(horizons)), np.nan)
    fold_fits = []
    for k in np.unique(folds):
        tr, te = folds != k, (folds == k) & m
        if te.sum() == 0:
            continue
        lm_tr = lm.loc[tr]
        pipe, cox, feats, dropped, rare, _ = fit_step(lm_tr, blocks, cfg, mode)
        if cox.models_.get("svd") is None:
            continue
        lp, expected = svd_offset_inputs(pipe, cox, lm_tr)
        vk = fit_dp_ucmgp_offset(lm_tr, lp, cfg, expected_svd_hazard=expected, wald_bootstrap=0)
        fold_fits.append(vk.fitted)
        test = lm.loc[te]
        base[te] = _probs(pipe, cox, test, cfg)
        plus[te] = _probs(pipe, cox, test, cfg, vk if vk.fitted else None)
    ok = m & np.isfinite(base[:, 0]) & np.isfinite(plus[:, 0])
    if not ok.any():
        return {"fitted": False, **counts, "reason": "no measured test rows with supported predictions"}
    time, event, pids = lm["time"].to_numpy(float), lm["event"].to_numpy(), lm["patient_id"].to_numpy()
    rows = []
    for h_i, h in enumerate(horizons):
        pb, pp, t, e = base[ok, h_i], plus[ok, h_i], time[ok], event[ok]
        mb = evaluate_predictions(pb, t, e, SVD, float(h))
        mp = evaluate_predictions(pp, t, e, SVD, float(h))

        def diff(idx, pb=pb, pp=pp, t=t, e=e, h=float(h)):
            a = evaluate_predictions(pb[idx], t[idx], e[idx], SVD, h)
            b = evaluate_predictions(pp[idx], t[idx], e[idx], SVD, h)
            return {"brier_diff": b["brier"] - a["brier"], "ipa_diff": b["ipa"] - a["ipa"]}

        ci = bootstrap_by_patient(diff, pids[ok], n_boot=n_boot, seed=seed + h_i, keys=("brier_diff", "ipa_diff"))
        rows.append({"horizon_years": h, "n_rows": mb["n"], "n_events": mb["n_events"],
                     "brier_core_plus_anticoagulant": mb["brier"], "brier_plus_dp_ucmgp": mp["brier"],
                     "brier_diff": mp["brier"] - mb["brier"], "brier_diff_lo": ci["brier_diff"][0], "brier_diff_hi": ci["brier_diff"][1],
                     "ipa_core_plus_anticoagulant": mb["ipa"], "ipa_plus_dp_ucmgp": mp["ipa"], "ipa_diff": mp["ipa"] - mb["ipa"],
                     "ipa_diff_lo": ci["ipa_diff"][0], "ipa_diff_hi": ci["ipa_diff"][1],
                     "obs_minus_pred_core_plus_anticoagulant": mb["obs_minus_pred"], "obs_minus_pred_plus_dp_ucmgp": mp["obs_minus_pred"],
                     "label": "synthetic scenario"})
    full_pipe, full_cox, _, _, _, _ = fit_step(lm, blocks, cfg, mode)
    lp, expected = svd_offset_inputs(full_pipe, full_cox, lm)
    full_vk = fit_dp_ucmgp_offset(lm, lp, cfg, expected_svd_hazard=expected)
    return {"fitted": True, "step": name, **counts, "folds_with_offset_model": int(sum(fold_fits)),
            "incremental": rows, "offset_model": full_vk.card(),
            "offset_model_note": ("p_value is a patient-bootstrap Wald test; p_value_nominal treats landmark rows of one patient "
                                  "as independent and is shown for contrast only"),
            "mechanism": mechanism_check(lm, cfg, mode),
            "label": "synthetic scenario: illustrative, unvalidated"}
