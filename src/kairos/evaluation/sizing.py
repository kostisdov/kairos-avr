"""Development-only sizing pilot and the frozen evaluation plan (CR-04, WP-B2).

The pilot generates cohorts with development seeds that are never reused for evaluation, counts
unique event patients per cause and route in the training part of each outer fold and in each
held-out fold, and extrapolates the cohort size each support gate needs (linear in cohort size,
times a safety factor). It proposes sizes; it does not choose them. The plan is written with
``frozen: false``.

The plan is advisory (owner decision of 17 September 2026: the data are synthetic, so evaluation is
never refused). :func:`plan_status` labels every evaluation: ``frozen_match`` (frozen plan, same size,
seed and parameters), ``unfrozen_plan``, ``no_plan``, ``plan_mismatch`` (with the differences) or
``quick``. Only ``frozen_match`` results carry the ``prespecified`` evidence label.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from kairos.evaluation.ladder import patient_folds
from kairos.evaluation.metrics import outcome
from kairos.evaluation.support import gates_for
from kairos.io.config import get_settings
from kairos.modelling.modules import ladder_steps
from kairos.modelling.train import fit_step, landmark_from_cohort
from kairos.simulation.generators import generate_cohort
from kairos.simulation.scenarios import get_scenario

DEV_SEED_BASE = 9100
SAFETY_FACTOR = 1.25
CAUSES = {"svd": 1, "death": 2, "replacement": 3}


def plan_path() -> Path:
    return get_settings().config_dir / "evaluation_plan.yaml"


def _n_needed(n: int, observed: float, gate: int) -> int | None:
    if observed <= 0:
        return None      # no event at the pilot size: extrapolation impossible
    return int(math.ceil(n * gate * SAFETY_FACTOR / observed / 100.0) * 100)


def pilot_scenario(name: str, variant: str | None, cfg: dict, n: int, dev_seeds: int, n_splits: int,
                   benchmark_step: str = "core_plus_both") -> dict:
    gates = gates_for(cfg, "full")
    rows, timings = [], {}
    for k in range(dev_seeds):
        seed = DEV_SEED_BASE + k
        t0 = time.time()
        cohort = generate_cohort(get_scenario(name, variant), n=n, seed=seed, namespace="quick")
        timings.setdefault("generate_seconds", []).append(time.time() - t0)
        lm = landmark_from_cohort(cohort, cfg)
        folds = patient_folds(lm["patient_id"], n_splits, seed)
        for f in range(n_splits):
            tr, te = lm[folds != f], lm[folds == f]
            rec = {"seed": seed, "fold": f, "n": n}
            for cause, code in CAUSES.items():
                rec[f"train_{cause}"] = int(tr.loc[tr["event"] == code, "patient_id"].nunique())
                for route in ("SAVR", "TAVR"):
                    sub = tr[tr["route"] == route]
                    rec[f"train_{cause}_{route}"] = int(sub.loc[sub["event"] == code, "patient_id"].nunique())
                y, known = outcome(te["time"], te["event"], cause, 1.0)
                rec[f"test_{cause}_cases_1y"] = int(te.loc[(y == 1) & known, "patient_id"].nunique())
                y5, known5 = outcome(te["time"], te["event"], cause, 5.0)
                rec[f"test_{cause}_cases_5y"] = int(te.loc[(y5 == 1) & known5, "patient_id"].nunique())
            rows.append(rec)
        if k == 0:
            blocks = dict(ladder_steps(cfg))[benchmark_step]
            t0 = time.time()
            fit_step(lm[folds != 0], blocks, cfg, "full")
            timings["cox_fit_seconds_one_fold"] = time.time() - t0
    df = pd.DataFrame(rows)
    fit_gate, base_gate = int(gates["fit_min_unique_events"]), int(gates["baseline_min_unique_events"])
    metric_gate, slope_gate = int(gates["metric_min_event_patients"]), int(gates["slope_min_events"])
    need: dict = {}
    for cause in CAUSES:
        low = df[f"train_{cause}"].quantile(0.1)
        need[f"{cause}: covariate model (pooled Cox), {fit_gate} in every training fold"] = _n_needed(n, low, fit_gate)
        for route in ("SAVR", "TAVR"):
            r_low = df[f"train_{cause}_{route}"].quantile(0.1)
            need[f"{cause} {route}: route baseline, {base_gate} in every training fold"] = _n_needed(n, r_low, base_gate)
            need[f"{cause} {route}: per-route model (gradient boosting), {fit_gate}"] = _n_needed(n, r_low, fit_gate)
        need[f"{cause}: AUC and intercept at 1 y, {metric_gate} case patients in a held-out fold"] = \
            _n_needed(n, df[f"test_{cause}_cases_1y"].quantile(0.1), metric_gate)
        need[f"{cause}: calibration slope at 5 y, {slope_gate} case patients in a held-out fold"] = \
            _n_needed(n, df[f"test_{cause}_cases_5y"].quantile(0.1), slope_gate)
    cox_keys = [k for k in need if "pooled Cox" in k and not k.startswith("replacement")] + \
               [k for k in need if "route baseline" in k] + [k for k in need if k.startswith("svd:")]
    cox_values = [need[k] for k in cox_keys]
    proposed = None if any(v is None for v in cox_values) else max([n, *cox_values])
    return {"scenario": name, "variant": variant, "pilot_n": n, "dev_seeds": [DEV_SEED_BASE + k for k in range(dev_seeds)],
            "n_splits": n_splits, "counts_10th_percentile": {c: float(df[c].quantile(0.1)) for c in df.columns
                                                              if c.startswith(("train_", "test_"))},
            "required_n": need, "proposed_n_cox_primary": proposed,
            "proposal_rule": ("largest n needed for: pooled SVD and death covariate models, every route baseline, and "
                              "the SVD metric gates; replacement covariate model and per-route boosting needs listed, "
                              "not included"),
            "timings": {k: (float(np.mean(v)) if isinstance(v, list) else float(v)) for k, v in timings.items()},
            "rows": df.to_dict(orient="records")}


def write_plan(pilots: list[dict], cfg: dict, path: Path | None = None, final_test_seeds=(20261001, 20261002)) -> Path:
    path = path or plan_path()
    dev = {}
    for p in pilots:
        key = p["scenario"] if not p["variant"] else f"{p['scenario']}/{p['variant']}"
        spec = get_scenario(p["scenario"], p["variant"])
        dev[key] = {"proposed_n": p["proposed_n_cox_primary"], "effective_config_hash": spec.effective_config_hash,
                    "required_n": p["required_n"], "pilot_n": p["pilot_n"], "timings": p["timings"]}
    body = {"frozen": False,
            "review_note": ("Proposed by the sizing pilot. Review sizes and unmet gates, set n per scenario, then set "
                            "frozen: true. Evaluations run either way and record whether they matched a frozen plan."),
            "generated_at": datetime.now(UTC).isoformat(),
            "safety_factor": SAFETY_FACTOR, "gates": gates_for(cfg, "full"),
            "development": {"seed": int(cfg["evaluation"]["seed"]), "scenarios": dev},
            "final_test": {"seeds": list(final_test_seeds), "note": "generated only after freeze-candidate (phase C)"},
            "promotion_tolerances": {"mean_brier_ci_excludes_zero": True, "max_horizon_brier_worsening": 0.005,
                                     "max_abs_obs_minus_pred_worsening": 0.01}}
    body["plan_hash"] = hashlib.sha256(json.dumps({k: v for k, v in body.items() if k != "generated_at"},
                                                  sort_keys=True, default=str).encode()).hexdigest()[:16]
    path.write_text("# KAIROS frozen evaluation plan (detailed design WP-B2). Values proposed by the sizing pilot are\n"
                    "# assumptions until reviewed; see docs/runbook.md.\n" + yaml.safe_dump(body, sort_keys=False, width=120),
                    encoding="utf-8")
    return path


def load_plan(path: Path | None = None) -> dict | None:
    path = path or plan_path()
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def plan_status(manifest: dict, plan: dict | None) -> dict:
    """Advisory: how an evaluation of this cohort relates to the evaluation plan. Never blocks."""
    if manifest.get("namespace") != "full":
        return {"status": "quick", "evidence": "exploratory", "reason": "quick namespace: smoke test"}
    if not plan:
        return {"status": "no_plan", "evidence": "exploratory", "reason": "no evaluation plan (cli.py size-pilot writes one)"}
    if not plan.get("frozen"):
        return {"status": "unfrozen_plan", "evidence": "exploratory", "reason": "evaluation plan not frozen"}
    entry = plan.get("development", {}).get("scenarios", {}).get(manifest.get("key"))
    if entry is None:
        return {"status": "plan_mismatch", "evidence": "exploratory", "reason": f"scenario {manifest.get('key')} not in the plan"}
    problems = []
    if int(entry.get("n", entry.get("proposed_n") or -1)) != int(manifest.get("n_requested", -2)):
        problems.append(f"cohort n {manifest.get('n_requested')} differs from plan n {entry.get('n', entry.get('proposed_n'))}")
    if int(plan["development"]["seed"]) != int(manifest.get("seed", -1)):
        problems.append(f"cohort seed {manifest.get('seed')} differs from plan seed {plan['development']['seed']}")
    if entry.get("effective_config_hash") != manifest.get("effective_config_hash"):
        problems.append("resolved scenario parameters differ from the plan")
    if problems:
        return {"status": "plan_mismatch", "evidence": "exploratory", "reason": "; ".join(problems)}
    return {"status": "frozen_match", "evidence": "prespecified", "reason": "matches the frozen plan"}


def check_full_evaluation_allowed(manifest: dict, plan: dict | None) -> tuple[bool, str]:
    """Kept for callers: always allowed; the second element is the plan status reason."""
    st = plan_status(manifest, plan)
    return True, f"{st['status']}: {st['reason']}"
