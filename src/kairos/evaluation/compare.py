"""Family comparison, promotion rule, candidate freeze and final synthetic replication (CR-08, WP-C3).

* :func:`paired_differences`: the two families' out-of-fold predictions on the rows both can
  predict (never on different surviving subsets), gradient boosting minus Cox for every state and
  horizon, with paired patient-bootstrap intervals (the same resamples for both families). SVD is
  primary; death, replacement and alive-intact are consistency checks. AUC is reported and never
  used to decide.
* :func:`promotion_decision`: the frozen rule (``config/evaluation_plan.yaml: promotion_tolerances``):
  the mean SVD Brier difference across horizons has a 95 % interval below zero; no supported horizon
  is worse by more than the Brier tolerance; the absolute observed-minus-predicted SVD difference
  does not worsen by more than its tolerance; the logistic calibration slope stays estimable. The rule
  is applied to every run (owner decision of 17 September 2026: synthetic data are never refused);
  ``evidence`` says ``prespecified`` only for a full-namespace run under a matching frozen plan and
  ``exploratory`` otherwise. Failure keeps Cox and still publishes the comparison. A pass permits a
  research-demo default change only.
* :func:`freeze_candidate` writes an immutable ``candidate.json``; :func:`evaluate_final` evaluates
  that candidate once on separately seeded synthetic test cohorts (synthetic replication, not
  external validation) and refuses a second run without a stated reason.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from kairos.evaluation.metrics import KMCensoring, bootstrap_by_patient, evaluate_predictions
from kairos.evaluation.support import gates_for, horizon_counts, metric_gate
from kairos.modelling.cif import STATES

DEFAULT_TOLERANCES = {"mean_brier_ci_excludes_zero": True, "max_horizon_brier_worsening": 0.005,
                      "max_abs_obs_minus_pred_worsening": 0.01}
HORIZON_COLS = {1.0: "1y", 3.0: "3y", 5.0: "5y"}


def _pred(oof: pd.DataFrame, state: str, h: float) -> np.ndarray:
    return oof[f"p_{state}_{int(h)}y"].to_numpy(dtype=float)


def paired_differences(oof_cox: pd.DataFrame, oof_gb: pd.DataFrame, time, event, cfg: dict, mode: str,
                       n_boot: int = 200, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event)
    if len(oof_cox) != len(oof_gb) or not (oof_cox["patient_id"].to_numpy() == oof_gb["patient_id"].to_numpy()).all():
        raise ValueError("the two families must be evaluated on the same landmark rows in the same order")
    common = (oof_cox["status"].to_numpy() == "") & (oof_gb["status"].to_numpy() == "")
    pids = oof_cox["patient_id"].to_numpy()[common]
    t, e = time[common], event[common]
    horizons = [float(h) for h in cfg["horizons_years"]]
    gates = gates_for(cfg, mode)
    info = {"rows": int(len(common)), "common_rows": int(common.sum()), "common_patients": int(pd.Series(pids).nunique()),
            "cox_only_rows": int(((oof_cox["status"] == "") & ~common).sum()),
            "gradient_boosting_only_rows": int(((oof_gb["status"] == "") & ~common).sum())}
    if not common.any():
        return pd.DataFrame(), {**info, "status": "no_common_rows"}
    P = {(f, s, h): _pred(o, s, h)[common] for f, o in (("cox", oof_cox), ("gradient_boosting", oof_gb))
         for s in STATES for h in horizons}

    def point(idx=None):
        tt, ee = (t, e) if idx is None else (t[idx], e[idx])
        G = KMCensoring().fit(tt, ee)
        out = {}
        for s in STATES:
            for h in horizons:
                ms = {f: evaluate_predictions(P[(f, s, h)] if idx is None else P[(f, s, h)][idx], tt, ee, s, h, G)
                      for f in ("cox", "gradient_boosting")}
                out[f"{s}|{h}|brier_diff"] = ms["gradient_boosting"]["brier"] - ms["cox"]["brier"]
                out[f"{s}|{h}|abs_obs_minus_pred_diff"] = abs(ms["gradient_boosting"]["obs_minus_pred"]) - abs(ms["cox"]["obs_minus_pred"])
                out[f"{s}|{h}|m"] = ms
            out[f"{s}|mean|brier_diff"] = float(np.mean([out[f"{s}|{h}|brier_diff"] for h in horizons]))
        return out

    est = point()
    keys = tuple(k for k in est if not k.endswith("|m"))
    boot = bootstrap_by_patient(lambda idx: {k: v for k, v in point(idx).items() if not k.endswith("|m")}, pids,
                                n_boot=n_boot, seed=seed, keys=keys, min_valid=int(gates["bootstrap_min_valid"]),
                                min_success_fraction=float(gates["bootstrap_min_success_fraction"]))
    rows = []
    for s in STATES:
        for h in [*horizons, "mean"]:
            row = {"state": s, "horizon_years": h, "brier_diff": est[f"{s}|{h}|brier_diff"],
                   "brier_diff_lo": boot[f"{s}|{h}|brier_diff"][0], "brier_diff_hi": boot[f"{s}|{h}|brier_diff"][1],
                   "boot_valid": boot.n_valid.get(f"{s}|{h}|brier_diff", 0), "boot_requested": n_boot}
            if h != "mean":
                ms = est[f"{s}|{h}|m"]
                counts = horizon_counts(t, e, pids, s, h)
                gate = metric_gate("brier", counts, ms["cox"]["G_at_tau"], gates)
                slope_gate = metric_gate("cal_slope_logit", counts, ms["cox"]["G_at_tau"], gates)
                row.update({"abs_obs_minus_pred_diff": est[f"{s}|{h}|abs_obs_minus_pred_diff"],
                            "abs_obs_minus_pred_diff_lo": boot[f"{s}|{h}|abs_obs_minus_pred_diff"][0],
                            "abs_obs_minus_pred_diff_hi": boot[f"{s}|{h}|abs_obs_minus_pred_diff"][1],
                            "brier_cox": ms["cox"]["brier"], "brier_gradient_boosting": ms["gradient_boosting"]["brier"],
                            "slope_cox": ms["cox"]["cal_slope_logit"], "slope_gradient_boosting": ms["gradient_boosting"]["cal_slope_logit"],
                            "slope_status_gradient_boosting": ms["gradient_boosting"]["cal_status"],
                            "auc_cox": ms["cox"]["auc"], "auc_gradient_boosting": ms["gradient_boosting"]["auc"],
                            "case_patients": counts["case_patients"], "support_status": gate.status,
                            "slope_support_status": slope_gate.status})
            rows.append(row)
    df = pd.DataFrame(rows)
    df["mode"] = mode
    df["label"] = "synthetic scenario; paired out-of-fold comparison on common rows"
    return df, {**info, "status": "ok", "bootstrap": boot.card()}


def promotion_decision(diffs: pd.DataFrame, plan: dict | None, mode: str, plan_match: bool | None = None) -> dict:
    tol = {**DEFAULT_TOLERANCES, **((plan or {}).get("promotion_tolerances") or {})}
    svd = diffs[diffs["state"] == "svd"] if len(diffs) else diffs
    criteria = []
    if len(svd):
        mean = svd[svd["horizon_years"] == "mean"].iloc[0]
        criteria.append({"criterion": "mean SVD Brier improves (95% interval below zero)",
                         "value": mean["brier_diff"], "interval": [mean["brier_diff_lo"], mean["brier_diff_hi"]],
                         "passed": bool(np.isfinite(mean["brier_diff_hi"]) and mean["brier_diff_hi"] < 0)})
        for _, r in svd[svd["horizon_years"] != "mean"].iterrows():
            supported = r["support_status"] == "ok"
            criteria.append({"criterion": f"SVD Brier at {r['horizon_years']:g} y not worse by more than {tol['max_horizon_brier_worsening']}",
                             "value": r["brier_diff"], "supported": supported,
                             "passed": bool((not supported) or r["brier_diff"] <= tol["max_horizon_brier_worsening"])})
            criteria.append({"criterion": f"|observed - predicted| SVD at {r['horizon_years']:g} y not worse by more than "
                                          f"{tol['max_abs_obs_minus_pred_worsening']}",
                             "value": r["abs_obs_minus_pred_diff"], "supported": supported,
                             "passed": bool((not supported) or r["abs_obs_minus_pred_diff"] <= tol["max_abs_obs_minus_pred_worsening"])})
            slope_supported = r["slope_support_status"] == "ok"
            criteria.append({"criterion": f"SVD calibration slope estimable at {r['horizon_years']:g} y",
                             "value": r["slope_gradient_boosting"], "supported": slope_supported,
                             "passed": bool((not slope_supported) or np.isfinite(r["slope_gradient_boosting"]))})
    all_pass = bool(criteria) and all(c["passed"] for c in criteria)
    prespecified = mode == "full" and bool(plan and plan.get("frozen")) and plan_match is not False
    evidence = "prespecified" if prespecified else "exploratory"
    if all_pass:
        decision, reason = "promote_gradient_boosting", "all criteria met: research-demo default may change; no clinical validity"
    else:
        decision, reason = "retain_cox", "at least one criterion not met; the gradient boosting result is still published"
    if not prespecified:
        reason += " (exploratory: quick namespace or no matching frozen plan)"
    return {"decision": decision, "evidence": evidence, "reason": reason, "criteria": criteria, "tolerances": tol, "mode": mode,
            "plan_hash": (plan or {}).get("plan_hash"), "auc_used": False, "decided_at": datetime.now(UTC).isoformat()}


def freeze_candidate(store, run_id: str, scenario_key: str, step: str, decision: dict) -> dict:
    path = f"runs/{run_id}/candidate.json"
    if store.exists("metrics", path):
        raise FileExistsError(f"{path} exists; a frozen candidate is immutable")
    family = "gradient_boosting" if decision.get("decision") == "promote_gradient_boosting" else "cox"
    cand = {"run_id": run_id, "scenario": scenario_key, "step": step, "family": family,
            "hyperparameter_policy": ("tuned on patient-grouped inner folds of the development cohort"
                                      if family == "gradient_boosting" else "prespecified penalty"),
            "decision": decision, "frozen_at": datetime.now(UTC).isoformat()}
    store.put_json("metrics", path, cand)
    return cand


def evaluate_final(store, cfg: dict, run_id: str, plan: dict, dev_cohort, test_cohorts: list, n_boot: int = 200,
                   rerun_reason: str | None = None) -> dict:
    """Train the candidate family and the Cox reference on the development cohort, evaluate both once on
    each separately seeded synthetic test cohort. The caller generates the cohorts from the plan."""
    from kairos.evaluation.ladder import predict_states
    from kairos.modelling.modules import ladder_steps
    from kairos.modelling.train import landmark_from_cohort, train_bundle

    cand_path, done_path = f"runs/{run_id}/candidate.json", f"runs/{run_id}/final/done.json"
    if not store.exists("metrics", cand_path):
        raise FileNotFoundError("no frozen candidate for this run: run freeze-candidate first")
    if store.exists("metrics", done_path) and not rerun_reason:
        raise PermissionError("the final evaluation already ran for this candidate; a rerun needs a stated reason")
    cand = store.get_json("metrics", cand_path)
    families = sorted({cand["family"], "cox"})
    assert cand["step"] in dict(ladder_steps(cfg))
    horizons = [float(h) for h in cfg["horizons_years"]]
    near = float(cfg["near_term_horizon_months"]) / 12.0
    lm_dev = landmark_from_cohort(dev_cohort, cfg)
    bundles = {f: train_bundle(dev_cohort, cfg, cand["step"], "final", lm=lm_dev, mode="full", family=f) for f in families}
    rows = []
    for test in test_cohorts:
        lm = landmark_from_cohort(test, cfg)
        for f, b in bundles.items():
            pr, reasons = predict_states(b.pipeline, b.model, lm, horizons, near)
            ok = reasons == ""
            t, e, pids = lm["time"].to_numpy(float)[ok], lm["event"].to_numpy()[ok], lm["patient_id"].to_numpy()[ok]
            for s in STATES:
                for i, h in enumerate(horizons):
                    p = pr[s][ok, i]
                    m = evaluate_predictions(p, t, e, s, h)
                    ci = bootstrap_by_patient(lambda idx, p=p, s=s, h=h, t=t, e=e: evaluate_predictions(p[idx], t[idx], e[idx], s, h),
                                              pids, n_boot=n_boot, seed=int(test.manifest["seed"]), keys=("brier", "obs_minus_pred"))
                    rows.append({"test_seed": test.manifest["seed"], "family": f, "state": s, "horizon_years": h,
                                 "coverage_rows": float(ok.mean()), "brier": m["brier"], "brier_lo": ci["brier"][0],
                                 "brier_hi": ci["brier"][1], "obs_minus_pred": m["obs_minus_pred"],
                                 "cal_slope_logit": m["cal_slope_logit"], "auc": m["auc"],
                                 "label": "synthetic replication with new seeds from the same generator; not external validation"})
    df = pd.DataFrame(rows)
    store.put_csv("metrics", f"runs/{run_id}/final/final_test.csv", df)
    store.put_json("metrics", done_path, {"candidate": cand, "rerun_reason": rerun_reason, "plan_hash": (plan or {}).get("plan_hash"),
                                          "evidence": "prespecified" if (plan or {}).get("frozen") else "exploratory",
                                          "test_seeds": [c.manifest["seed"] for c in test_cohorts],
                                          "completed_at": datetime.now(UTC).isoformat()})
    return {"candidate": cand, "results": json.loads(df.to_json(orient="records"))}
