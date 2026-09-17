"""The fixed model ladder, evaluated with patient-grouped cross-validation.

Steps: reference (valve age and type), core, core plus each module, core plus biomarkers,
core plus anticoagulant history, core plus both, and core without serial echo. All
preprocessing, including marker and module eligibility and event support, is fitted inside the
training folds and never sees held-out rows; all rows of a patient stay together in every split
and in every bootstrap resample.

Every row gets out-of-fold probabilities for all four states (SVD, death, non-SVD replacement,
alive with the index valve without SVD) or a recorded reason (unsupported route or cause, fit
failure). Metrics are computed on the rows with valid predictions, and coverage says how much of
the intended population that is. Each metric passes its own support gate
(:mod:`kairos.evaluation.support`): AUC and the calibration intercept and slope are not estimated
below their gates; Brier, IPA and observed-minus-predicted stay as descriptive values with an
exploratory status. Performance is pooled over landmark rows (labelled as such); a patient-balanced
weighting (equal total weight per patient) is reported beside it. When the anticoagulant step is
evaluated, the dp-ucMGP substudy is evaluated on the same folds (:mod:`kairos.evaluation.vitamin_k`).
"""
from __future__ import annotations

import hashlib
import json
import time as _time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from kairos.evaluation.metrics import (
    KMCensoring,
    bootstrap_by_patient,
    calibration_curve,
    evaluate_predictions,
    formal_calibration_curve,
)
from kairos.evaluation.support import (
    DESCRIPTIVE,
    ESTIMATED_ONLY_WHEN_SUPPORTED,
    bootstrap_gate,
    coverage,
    gates_for,
    horizon_counts,
    metric_gate,
    unique_event_patients,
)
from kairos.evaluation.vitamin_k import evaluate_vitamin_k
from kairos.modelling.cif import (
    INTEGRATION_VERSION,
    STATES,
    combine_cause_specific,
    probabilities_at,
)
from kairos.modelling.modules import (
    event_count_rule,
    fit_eligibility,
    ladder_steps,
    resolve_features,
)
from kairos.modelling.train import fit_step, landmark_from_cohort, support_mode_for
from kairos.simulation.generators import Cohort

SVD = 1
BOOT_KEYS = ("brier", "ipa", "obs_minus_pred", "auc", "cal_intercept_offset", "cal_slope_logit")
DECOMP_KEYS = ("reliability", "resolution", "uncertainty", "within_bin_variance", "within_bin_covariance",
               "reconstruction_error")
PREDICT_CHUNK = 1000


@dataclass
class LadderResult:
    results: pd.DataFrame
    curves: dict = field(default_factory=dict)              # step -> binned SVD display at the last horizon
    calibration_curves: dict = field(default_factory=dict)  # step -> horizon -> {"binned", "formal"}
    coefficients: dict = field(default_factory=dict)
    availability: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    phenotype: pd.DataFrame = field(default_factory=pd.DataFrame)
    secondary: dict = field(default_factory=dict)
    vitamin_k: dict = field(default_factory=dict)
    fold_fits: dict = field(default_factory=dict)           # step -> fold -> training-only decisions
    oof: dict = field(default_factory=dict)                 # step -> DataFrame of out-of-fold predictions


def patient_folds(patient_ids: pd.Series, n_splits: int, seed: int) -> np.ndarray:
    uniq = np.array(sorted(patient_ids.unique()))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    fold_of = {uniq[i]: k % n_splits for k, i in enumerate(perm)}
    return patient_ids.map(fold_of).to_numpy()


def predict_states(pipe, cox, lm_test: pd.DataFrame, horizons, near: float, chunk: int = PREDICT_CHUNK) -> tuple[dict, np.ndarray]:
    """state -> (n, H) probabilities (NaN where unsupported) and a reason per row ('' when valid)."""
    X = pipe.transform(lm_test)
    for s in cox.strata:
        X[s] = lm_test[s].astype(str).to_numpy()
    out = {s: np.full((len(lm_test), len(horizons)), np.nan) for s in STATES}
    ok, reasons = cox.row_support(X)
    idx = np.flatnonzero(ok)
    for start in range(0, len(idx), chunk):
        sel = idx[start:start + chunk]
        cifs = combine_cause_specific(cox.cumulative_hazards(X.iloc[sel]), cox.grid)
        pr = probabilities_at(cifs, cox.grid, horizons, near)
        for s in STATES:
            out[s][sel] = pr[s]
    return out, reasons


def resolve_step_features(lm: pd.DataFrame, blocks: list[str], cfg: dict) -> list[str]:
    """The feature set a step resolves to on ``lm`` (eligibility and the model-level event rule),
    without fitting a model."""
    feats, _, _ = resolve_features(cfg, blocks, fit_eligibility(lm, cfg))
    feats, _ = event_count_rule(lm, feats, cfg)
    return [f for f in feats if f in lm.columns]


def _checkpoint_key(lm: pd.DataFrame, cohort: Cohort, cfg: dict, family: str, step: str, fold: int, mode: str,
                    label_policy: str, n_splits: int, seed: int) -> str:
    ident = {"tables": cohort.manifest.get("table_hashes"), "cfg": cfg, "family": family, "step": step, "fold": fold,
             "mode": mode, "label_policy": label_policy, "n_splits": n_splits, "seed": seed,
             "rows": hashlib.sha256(lm[["patient_id", "time", "event"]].to_csv(index=False).encode()).hexdigest()}
    return hashlib.sha256(json.dumps(ident, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _gated_metrics(pred, time, event, pids, state, tau, gates, sample_weight=None, decomposition=False) -> dict:
    counts = horizon_counts(time, event, pids, state, tau)
    G = KMCensoring().fit(time, event)
    m = evaluate_predictions(pred, time, event, state, tau, G, sample_weight=sample_weight,
                             min_censoring_survival=float(gates.get("min_censoring_survival", 0.0)),
                             decomposition=decomposition)
    status = {}
    for metric in (*DESCRIPTIVE, *ESTIMATED_ONLY_WHEN_SUPPORTED):
        d = metric_gate(metric, counts, m["G_at_tau"], gates)
        status[metric] = d
        if metric in ESTIMATED_ONLY_WHEN_SUPPORTED and not d.ok:
            m[metric] = float("nan")
    return {"metrics": m, "counts": counts, "status": status}


def evaluate_ladder(cohort: Cohort, cfg: dict, n_splits: int | None = None, n_boot: int | None = None,
                    seed: int | None = None, steps: list[str] | None = None,
                    lm: pd.DataFrame | None = None, label_policy: str = "primary", mode: str | None = None,
                    states: tuple = STATES, patient_balanced: bool = True, fold_of: dict | None = None,
                    extras: bool = True, extra_weighting: tuple | None = None, family: str = "cox",
                    store=None, run_id: str | None = None) -> LadderResult:
    """``fold_of`` fixes the patient -> fold map (common splits across sensitivity analyses);
    ``extras`` adds the secondary thrombosis outcome and the dp-ucMGP substudy; ``extra_weighting``
    is ``(name, fn)`` with ``fn(lm_train, lm_test) -> (weights for lm_test, diagnostics)`` fitted
    inside each training fold and reported as additional exploratory rows. ``family`` selects the
    model family; gradient boosting is tuned on patient-grouped inner folds of each outer training
    set only. With ``store`` and ``run_id`` every outer fold is checkpointed under
    ``metrics/runs/<run_id>/oof/`` and reused when the same fold is requested again."""
    ev = cfg["evaluation"]
    n_splits = int(n_splits or ev["n_splits"])
    n_boot = int(ev["bootstrap"] if n_boot is None else n_boot)
    seed = int(ev["seed"] if seed is None else seed)
    mode = mode or support_mode_for(cohort, cfg)
    gates = gates_for(cfg, mode)
    horizons = [float(h) for h in cfg["horizons_years"]]
    near = float(cfg["near_term_horizon_months"]) / 12.0
    lm = landmark_from_cohort(cohort, cfg, label_policy) if lm is None else lm
    if lm.empty:
        raise ValueError("landmark dataset is empty")
    # descriptive only (whole cohort); every fitted decision below comes from training rows
    availability = {**fit_eligibility(lm, cfg).card(), "scope": "whole cohort, descriptive only; not used for fitting"}
    phenotype = _truth_phenotype(cohort, lm)
    if fold_of is not None:
        folds = lm["patient_id"].map(fold_of).to_numpy()
        if pd.isna(folds).any():
            raise ValueError("fold_of does not cover every patient in the landmark table")
        folds = folds.astype(int)
        n_splits = int(max(fold_of.values())) + 1
    else:
        folds = patient_folds(lm["patient_id"], n_splits, seed)
    all_steps = ladder_steps(cfg)
    if steps:
        all_steps = [(n, b) for n, b in all_steps if n in steps]
    key = cohort.manifest.get("key", "unknown")
    endpoint_version = cohort.manifest.get("endpoint_version")
    rows, curves, cal_curves, coefs, pheno_rows, fold_fits, signatures, oof_frames = [], {}, {}, {}, [], {}, {}, {}
    time = lm["time"].to_numpy(dtype=float)
    event = lm["event"].to_numpy()
    pids = lm["patient_id"].to_numpy()
    balanced = lm["row_weight_patient_balanced"].to_numpy(dtype=float) if "row_weight_patient_balanced" in lm else None
    for name, blocks in all_steps:
        oof = {s: np.full((len(lm), len(horizons)), np.nan) for s in STATES}
        oof_w = np.full(len(lm), np.nan)
        status = np.full(len(lm), "not_predicted", dtype=object)
        dropped_all: list[str] = []
        fold_fits[name] = {}
        for k in range(n_splits):
            tr, te = folds != k, folds == k
            if te.sum() == 0:
                continue
            te_idx = np.flatnonzero(te)
            ck_path = None
            if store is not None and run_id and extra_weighting is None:
                key = _checkpoint_key(lm, cohort, cfg, family, name, k, mode, label_policy, n_splits, seed)
                ck_path = f"runs/{run_id}/oof/{family}/{name}/fold{k}_{key}"
                if store.exists("metrics", f"{ck_path}.json"):
                    saved = store.get_parquet("metrics", f"{ck_path}.parquet")
                    fold_fits[name][k] = {**store.get_json("metrics", f"{ck_path}.json"), "resumed_from_checkpoint": True}
                    for s_ in STATES:
                        oof[s_][te_idx] = saved[[f"{s_}_{i}" for i in range(len(horizons))]].to_numpy()
                    status[te_idx] = saved["status"].fillna("").to_numpy()
                    dropped_all = sorted(set(dropped_all) | set(fold_fits[name][k].get("dropped", [])))
                    continue
            t_fold = _time.time()
            tuning, hp = None, None
            try:
                if family == "gradient_boosting":
                    from kairos.modelling.tuning import tune_boosting

                    tuning = tune_boosting(lm.loc[tr], blocks, cfg, seed + 1000 + k, mode)
                    hp = tuning["selected"]
                pipe, cox, feats, dropped, rare, manifest = fit_step(lm.loc[tr], blocks, cfg, mode, family, hp)
            except Exception as ex:  # noqa: BLE001 - recorded per row, never hidden
                status[te] = f"fit_failed:{type(ex).__name__}"
                fold_fits[name][k] = {"error": f"{type(ex).__name__}: {ex}"[:300], "tuning": tuning}
                continue
            dropped_all = sorted(set(dropped_all) | set(dropped))
            fold_fits[name][k] = {"eligibility": manifest.card(), "features": list(feats), "medians": dict(pipe.medians_),
                                  "levels": {c: list(v) for c, v in pipe.levels_.items()},
                                  "rare_levels": {c: (v if isinstance(v, str) else list(v)) for c, v in rare.items()},
                                  "support": cox.support_summary(), "family": family,
                                  "hyperparameters": cox.hyperparameters, "tuning": tuning, "dropped": list(dropped)}
            pr, reasons = predict_states(pipe, cox, lm.loc[te], horizons, near)
            fold_fits[name][k]["seconds"] = round(_time.time() - t_fold, 1)
            if ck_path is not None:
                frame = pd.DataFrame({f"{s_}_{i}": pr[s_][:, i] for s_ in STATES for i in range(len(horizons))})
                frame["status"] = reasons
                store.put_parquet("metrics", f"{ck_path}.parquet", frame)
                store.put_json("metrics", f"{ck_path}.json", fold_fits[name][k])
            if extra_weighting is not None:
                w_te, w_diag = extra_weighting[1](lm.loc[tr], lm.loc[te])
                oof_w[te_idx] = w_te
                fold_fits[name][k]["weighting"] = w_diag
            for s in STATES:
                oof[s][te_idx] = pr[s]
            status[te_idx] = reasons
        valid = status == ""
        cov = coverage(valid, pids)
        oof_frames[name] = pd.DataFrame({"patient_id": pids, "landmark_date": lm["landmark_date"].to_numpy(), "fold": folds,
                                         "family": family, "time": time, "event": event, "status": status,
                                         **{f"p_{s}_{int(h)}y": oof[s][:, i] for s in STATES for i, h in enumerate(horizons)}})
        full_feats = resolve_step_features(lm, blocks, cfg)
        if family == "cox":
            _, full_cox, _, _, _, _ = fit_step(lm, blocks, cfg, mode)
            coefs[name] = full_cox.coefficients("svd").to_dict()
        sig = tuple(sorted(full_feats))
        equivalent_to = signatures.get(sig, "")
        signatures.setdefault(sig, name)
        tv, ev_, pv = time[valid], event[valid], pids[valid]
        cal_curves[name] = {}
        common = {"scenario": key, "family": family, "step": name, "modules_dropped": ";".join(dropped_all), "equivalent_to": equivalent_to,
                  "label_policy": label_policy, "support_mode": mode, "integration_version": INTEGRATION_VERSION,
                  "endpoint_version": endpoint_version, "coverage_rows": cov["row_coverage"],
                  "coverage_patients": cov["patient_coverage"], "n_rows_intended": cov["rows"],
                  "unsupported_reasons": json.dumps({str(k): int(v) for k, v in pd.Series(status[~valid]).value_counts().items()}),
                  "label": "synthetic scenario"}
        for h_i, h in enumerate(horizons):
            boot = None
            if valid.sum() and n_boot > 0:
                def fn(idx, h=h, h_i=h_i, tv=tv, ev_=ev_, oof=oof, valid=valid):
                    t, e = tv[idx], ev_[idx]
                    G = KMCensoring().fit(t, e)
                    out = {}
                    for s in states:
                        m = evaluate_predictions(oof[s][valid, h_i][idx], t, e, s, h, G)
                        out.update({f"{s}|{k}": m[k] for k in BOOT_KEYS})
                    return out
                boot = bootstrap_by_patient(fn, pv, n_boot=n_boot, seed=seed + h_i,
                                            keys=tuple(f"{s}|{k}" for s in states for k in BOOT_KEYS),
                                            min_valid=int(gates["bootstrap_min_valid"]),
                                            min_success_fraction=float(gates["bootstrap_min_success_fraction"]))
            for s in states:
                if not valid.any():
                    rows.append({**common, "state": s, "horizon_years": h, "weighting": "pooled landmark rows",
                                 "support_status": "no_valid_predictions", "exploratory": True})
                    continue
                pred = oof[s][valid, h_i]
                g = _gated_metrics(pred, tv, ev_, pv, s, h, gates, decomposition=True)
                m, c, st = g["metrics"], g["counts"], g["status"]
                row = {**common, "state": s, "horizon_years": h, "weighting": "pooled landmark rows",
                       "n_rows": m["n"], "n_patients": int(pd.Series(pv).nunique()),
                       "n_events": c["case_rows"], "n_event_patients": c["case_patients"],
                       "n_control_patients": c["control_patients"],
                       "observed": m["observed"], "mean_predicted": m["mean_predicted"],
                       "brier": m["brier"], "brier_null": m["brier_null"], "ipa": m["ipa"],
                       "obs_minus_pred": m["obs_minus_pred"], "cal_intercept_offset": m["cal_intercept_offset"],
                       "cal_intercept_joint": m["cal_intercept_joint"], "cal_slope_logit": m["cal_slope_logit"],
                       "cal_status": m["cal_status"], "cal_slope_prob_legacy_v1": m["cal_slope_prob_legacy_v1"],
                       "auc": m["auc"], "G_at_tau": m["G_at_tau"], "ess": m["ess"], "max_weight": m["max_weight"],
                       "censoring_model": m["censoring_model"],
                       "support_status": st["brier"].status, "support_reason": st["brier"].reason,
                       "status_auc": st["auc"].status, "status_cal_intercept": st["cal_intercept_offset"].status,
                       "status_cal_slope": st["cal_slope_logit"].status,
                       "exploratory": bool(st["brier"].exploratory)}
                dec = m.get("decomposition", {})
                row.update({f"brier_{k}": dec.get(k, float("nan")) for k in DECOMP_KEYS})
                for k in BOOT_KEYS:
                    iv = boot.intervals.get(f"{s}|{k}") if boot is not None else None
                    if k in ESTIMATED_ONLY_WHEN_SUPPORTED and not st[k].ok:
                        iv = None
                    row[f"{k}_lo"], row[f"{k}_hi"] = iv if iv is not None else (float("nan"), float("nan"))
                if boot is not None:
                    bg = bootstrap_gate(n_boot, boot.n_valid.get(f"{s}|brier", 0), gates)
                    row.update({"boot_requested": n_boot, "boot_valid": boot.n_valid.get(f"{s}|brier", 0),
                                "boot_status": bg.status, "boot_failures": json.dumps(boot.failures),
                                "boot_kind": boot.kind})
                rows.append(row)
                if patient_balanced and balanced is not None:
                    gb = _gated_metrics(pred, tv, ev_, pv, s, h, gates, sample_weight=balanced[valid])
                    mb = gb["metrics"]
                    rows.append({**common, "state": s, "horizon_years": h, "weighting": "patient balanced",
                                 "n_rows": mb["n"], "n_patients": int(pd.Series(pv).nunique()),
                                 "n_events": c["case_rows"], "n_event_patients": c["case_patients"],
                                 "n_control_patients": c["control_patients"], "observed": mb["observed"],
                                 "mean_predicted": mb["mean_predicted"], "brier": mb["brier"], "brier_null": mb["brier_null"],
                                 "ipa": mb["ipa"], "obs_minus_pred": mb["obs_minus_pred"],
                                 "cal_intercept_offset": mb["cal_intercept_offset"], "cal_slope_logit": mb["cal_slope_logit"],
                                 "cal_status": mb["cal_status"], "auc": mb["auc"], "G_at_tau": mb["G_at_tau"],
                                 "support_status": gb["status"]["brier"].status, "exploratory": True})
                if extra_weighting is not None and np.isfinite(oof_w[valid]).all():
                    gw = _gated_metrics(pred, tv, ev_, pv, s, h, gates, sample_weight=oof_w[valid])
                    mw = gw["metrics"]
                    rows.append({**common, "state": s, "horizon_years": h, "weighting": extra_weighting[0],
                                 "n_rows": mw["n"], "n_patients": int(pd.Series(pv).nunique()),
                                 "n_events": c["case_rows"], "n_event_patients": c["case_patients"],
                                 "n_control_patients": c["control_patients"], "observed": mw["observed"],
                                 "mean_predicted": mw["mean_predicted"], "brier": mw["brier"], "brier_null": mw["brier_null"],
                                 "ipa": mw["ipa"], "obs_minus_pred": mw["obs_minus_pred"],
                                 "cal_intercept_offset": mw["cal_intercept_offset"], "cal_slope_logit": mw["cal_slope_logit"],
                                 "cal_status": mw["cal_status"], "auc": mw["auc"], "G_at_tau": mw["G_at_tau"],
                                 "support_status": gw["status"]["brier"].status, "exploratory": True})
                if s == "svd":
                    cal_curves[name][h] = {"binned": calibration_curve(pred, tv, ev_, s, h),
                                           "formal": formal_calibration_curve(pred, tv, ev_, s, h)}
                    if h == horizons[-1]:
                        curves[name] = cal_curves[name][h]["binned"]
                    if phenotype is not None:
                        for ph in sorted(pd.Series(phenotype).dropna().unique()):
                            mm = phenotype[valid] == ph
                            if mm.sum() == 0:
                                continue
                            gp = _gated_metrics(pred[mm], tv[mm], ev_[mm], pv[mm], s, h, gates)
                            mp = gp["metrics"]
                            pheno_rows.append({"scenario": key, "step": name, "horizon_years": h, "phenotype": ph,
                                               "n_rows": mp["n"], "n_events": gp["counts"]["case_rows"],
                                               "n_event_patients": gp["counts"]["case_patients"],
                                               "brier": mp["brier"], "ipa": mp["ipa"], "obs_minus_pred": mp["obs_minus_pred"],
                                               "auc": mp["auc"], "support_status": gp["status"]["auc"].status,
                                               "label": "synthetic scenario; latent phenotype from evaluation-only truth"})
    extras = extras and family == "cox"     # the secondary outcome and the dp-ucMGP offset model are Cox analyses
    secondary = secondary_thrombosis(lm, cfg, mode) if extras else {"fitted": False, "reason": "not requested"}
    step_names = [n for n, _ in all_steps]
    vitamin_k = (evaluate_vitamin_k(lm, cfg, folds, n_boot=n_boot, seed=seed, mode=mode)
                 if extras and f"core_plus_{cfg['vitamin_k']['requires_block']}" in step_names
                 else {"fitted": False, "reason": "anticoagulant step not evaluated"})
    summary = {"n_rows": int(len(lm)), "n_patients": int(lm["patient_id"].nunique()),
               "events": {int(k): int(v) for k, v in lm["event"].value_counts().to_dict().items()},
               "event_patients": {int(c): unique_event_patients(lm, c) for c in (1, 2, 3)},
               "n_splits": n_splits, "n_boot": n_boot, "seed": seed, "label_policy": label_policy,
               "support_mode": mode, "gates": gates, "integration_version": INTEGRATION_VERSION, "family": family,
               "run_id": run_id,
               "endpoint_version": endpoint_version,
               "weighting": "pooled over landmark rows (primary); patient-balanced rows beside it"}
    return LadderResult(results=pd.DataFrame(rows), curves=curves, calibration_curves=cal_curves, coefficients=coefs,
                        availability=availability, summary=summary, phenotype=pd.DataFrame(pheno_rows),
                        secondary=secondary, vitamin_k=vitamin_k, fold_fits=fold_fits, oof=oof_frames)


def _truth_phenotype(cohort: Cohort, lm: pd.DataFrame) -> np.ndarray | None:
    """Latent phenotype joined from the evaluation-only truth table (never a landmark column)."""
    truth = getattr(cohort, "truth", None)
    if truth is None or len(truth) == 0 or "phenotype_latent" not in truth:
        return None
    return lm["patient_id"].map(truth.set_index("patient_id")["phenotype_latent"]).to_numpy()


def secondary_thrombosis(lm: pd.DataFrame, cfg: dict, mode: str = "full") -> dict:
    """Thrombosis-related dysfunction as its own outcome with the anticoagulant block, so the
    direction of the anticoagulant association can be compared with the SVD direction."""
    if "event_thromb" not in lm or lm["event_thromb"].sum() < 20:
        return {"fitted": False, "reason": "fewer than 20 thrombosis episodes after landmarks"}
    df = lm.copy()
    df["time"] = df["time_thromb"]
    df["event"] = df["event_thromb"].astype(int)
    try:
        pipe, cox, feats, dropped, rare, _ = fit_step(df, ["core_static", "core_time", "anticoagulant"], cfg, mode)
    except Exception as ex:  # noqa: BLE001
        return {"fitted": False, "reason": f"{type(ex).__name__}: {ex}"[:200]}
    if cox.models_.get("svd") is None:
        return {"fitted": False, "reason": f"no covariate model: {cox.support_['svd']['reason']}"}
    coef = cox.coefficients("svd")
    keep = {k: float(v) for k, v in coef.items() if k.startswith("ac_") or k.startswith("on_antiplatelet")}
    return {"fitted": True, "n_events": int(df["event"].sum()), "coefficients": keep,
            "note": "cause-specific log hazard ratios for first thrombosis episode after the landmark; predictive associations only"}
