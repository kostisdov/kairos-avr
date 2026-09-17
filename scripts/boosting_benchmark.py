"""Compute benchmark for the gradient boosting comparison (detailed design WP-C0).

    python scripts/boosting_benchmark.py [--scenario gradual_stenotic] [--step core_plus_both] [--tune]

Times one Cox fit and one gradient boosting fit (largest grid point) on a 4/5 training fold of the full
cohort, optionally one complete inner tuning, and projects the full comparison. Timings only; no metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kairos.evaluation.ladder import patient_folds  # noqa: E402
from kairos.io.storage import get_store  # noqa: E402
from kairos.modelling.modules import ladder_steps, load_model_config  # noqa: E402
from kairos.modelling.train import fit_step, landmark_from_cohort  # noqa: E402
from kairos.modelling.tuning import tune_boosting  # noqa: E402
from kairos.simulation.generators import Cohort  # noqa: E402
from kairos.simulation.scenarios import list_scenarios  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="gradual_stenotic")
    ap.add_argument("--step", default="core_plus_both")
    ap.add_argument("--tune", action="store_true")
    args = ap.parse_args(argv)
    store, cfg = get_store(), load_model_config()
    cohort = Cohort.load(store, store.get_json("scenarios", f"{args.scenario}/default/full/latest.json")["prefix"])
    lm = landmark_from_cohort(cohort, cfg)
    tr = lm[patient_folds(lm["patient_id"], 5, 1) != 0]
    blocks = dict(ladder_steps(cfg))[args.step]
    grid = cfg["families"]["gradient_boosting"]["grid"]
    out = {"scenario": args.scenario, "step": args.step, "training_rows": int(len(tr)),
           "training_patients": int(tr["patient_id"].nunique())}
    t = time.time()
    fit_step(tr, blocks, cfg, "full", "cox")
    out["cox_fit_seconds"] = round(time.time() - t, 1)
    hp = {"n_estimators": max(grid["n_estimators"]), "learning_rate": max(grid["learning_rate"]), "max_depth": max(grid["max_depth"])}
    t = time.time()
    _, gb, *_ = fit_step(tr, blocks, cfg, "full", "gradient_boosting", hp)
    out["boosting_fit_seconds_largest_grid_point"] = round(time.time() - t, 1)
    out["boosting_support"] = {c: {k: s["status"] for k, s in v["strata"].items()} for c, v in gb.support_.items()}
    if args.tune:
        t = time.time()
        rec = tune_boosting(tr, blocks, cfg, 1, "full")
        out["tuning_seconds_one_outer_fold"] = round(time.time() - t, 1)
        out["tuning_selected"] = rec["selected"]
    steps = len(cfg["ladder"])
    scen = len(list_scenarios())
    folds = int(cfg["evaluation"]["n_splits"])
    per_fold = out.get("tuning_seconds_one_outer_fold", 3 * 4 * out["boosting_fit_seconds_largest_grid_point"]) + out["boosting_fit_seconds_largest_grid_point"]
    out["projection"] = {"outer_fits": scen * steps * folds,
                         "boosting_hours_at_this_cohort_size": round(scen * steps * folds * per_fold / 3600, 1),
                         "cox_hours_at_this_cohort_size": round(scen * steps * folds * out["cox_fit_seconds"] / 3600, 2),
                         "note": ("single process, excluding bootstrap and prediction; the frozen plan's cohort sizes "
                                  "(5600-16700 patients) scale fit time roughly linearly or worse")}
    print(json.dumps(out, indent=2))
    path = ROOT / "artifacts" / "metrics" / "phaseC" / f"benchmark_{args.scenario}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
