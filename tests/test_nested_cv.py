"""CR-08: nested patient-grouped tuning, checkpoints, the paired comparison and the promotion rule."""
import copy

import numpy as np
import pandas as pd
import pytest

from kairos.evaluation.compare import (
    evaluate_final,
    freeze_candidate,
    paired_differences,
    promotion_decision,
)
from kairos.evaluation.ladder import evaluate_ladder
from kairos.io.storage import LocalStore
from kairos.modelling import train as train_module
from kairos.modelling.train import landmark_from_cohort
from kairos.modelling.tuning import tune_boosting

BLOCKS = ["core_static", "core_time"]


@pytest.fixture(scope="module")
def cfg(model_cfg):
    c = copy.deepcopy(model_cfg)
    c["families"]["gradient_boosting"]["quick_grid"] = {"n_estimators": [10, 20], "learning_rate": [0.1], "max_depth": [1, 2]}
    return c


@pytest.fixture(scope="module")
def lm(small_cohort, cfg):
    return landmark_from_cohort(small_cohort, cfg)


def test_inner_folds_never_split_a_patient(lm, cfg, monkeypatch):
    seen = []
    real = train_module.fit_step

    def spy(rows, *a, **k):
        seen.append(set(rows["patient_id"]))
        return real(rows, *a, **k)

    monkeypatch.setattr(train_module, "fit_step", spy)
    rec = tune_boosting(lm, BLOCKS, cfg, seed=3, mode="quick")
    assert rec["status"] == "ok" and rec["selected"]["n_estimators"] in (10, 20) and rec["selected"]["max_depth"] in (1, 2)
    everyone = set(lm["patient_id"])
    assert len(seen) == 2 * 3                                   # (learning rate, depth) x inner folds
    for train_ids in seen:
        held = everyone - train_ids
        assert held and not (held & train_ids)
    assert all(r["valid"] for r in rec["table"]) and len(rec["table"]) == 4


def test_outer_test_rows_never_change_tuning(small_cohort, lm, cfg):
    from kairos.evaluation.ladder import patient_folds

    folds = patient_folds(lm["patient_id"], 2, int(cfg["evaluation"]["seed"]))
    mutated = lm.copy()
    held = folds == 0            # rows held out from outer fold 0, whose training rows are fold 1
    mutated.loc[held, "valve_age_years"] = mutated.loc[held, "valve_age_years"] + 50
    mutated.loc[held, "design_class"] = "made-up class"
    kw = dict(n_splits=2, n_boot=0, steps=["reference"], family="gradient_boosting", mode="quick", extras=False, states=("svd",))
    a = evaluate_ladder(small_cohort, cfg, lm=lm, **kw).fold_fits["reference"]
    b = evaluate_ladder(small_cohort, cfg, lm=mutated, **kw).fold_fits["reference"]
    assert a[0]["tuning"]["status"] == "ok"
    assert a[0]["tuning"]["table"] == b[0]["tuning"]["table"] and a[0]["hyperparameters"] == b[0]["hyperparameters"]
    # fold 1 trains on the mutated rows; an inner fold without scoreable rows is skipped, never a failed outer fold
    assert "error" not in a[1] and a[1]["tuning"]["status"] in ("ok", "fallback_default")


def test_checkpoints_resume_identically(small_cohort, lm, cfg, tmp_path):
    store = LocalStore(tmp_path)
    kw = dict(n_splits=2, n_boot=0, steps=["reference"], mode="quick", extras=False, store=store, run_id="r1")
    first = evaluate_ladder(small_cohort, cfg, lm=lm, family="cox", **kw)
    again = evaluate_ladder(small_cohort, cfg, lm=lm, family="cox", **kw)
    assert all(f.get("resumed_from_checkpoint") for f in again.fold_fits["reference"].values())
    cols = [c for c in first.oof["reference"].columns if c.startswith("p_")] + ["status"]
    pd.testing.assert_frame_equal(first.oof["reference"][cols], again.oof["reference"][cols])
    other = evaluate_ladder(small_cohort, cfg, lm=lm.iloc[:-5], family="cox", **kw)   # different rows: no reuse
    assert not any(f.get("resumed_from_checkpoint") for f in other.fold_fits["reference"].values())


def test_paired_comparison_on_common_rows_is_exploratory_in_quick_mode(small_cohort, lm, cfg):
    kw = dict(n_splits=2, n_boot=0, steps=["reference"], mode="quick", extras=False)
    cox = evaluate_ladder(small_cohort, cfg, lm=lm, family="cox", **kw).oof["reference"]
    gb = evaluate_ladder(small_cohort, cfg, lm=lm, family="gradient_boosting", **kw).oof["reference"]
    diffs, info = paired_differences(cox, gb, lm["time"], lm["event"], cfg, "quick", n_boot=25, seed=1)
    assert info["common_rows"] <= min((cox["status"] == "").sum(), (gb["status"] == "").sum())
    assert set(diffs["state"]) == {"svd", "death", "replacement", "alive_intact"} and "mean" in set(diffs["horizon_years"])
    d = promotion_decision(diffs, {"frozen": True}, "quick")
    assert d["decision"] in ("promote_gradient_boosting", "retain_cox") and d["evidence"] == "exploratory"
    assert d["auc_used"] is False and d["criteria"]


def _diffs(mean_hi, h_diff=0.0, om=0.0, slope=1.0, support="ok"):
    rows = [{"state": "svd", "horizon_years": "mean", "brier_diff": -0.002, "brier_diff_lo": -0.004, "brier_diff_hi": mean_hi}]
    for h in (1.0, 3.0, 5.0):
        rows.append({"state": "svd", "horizon_years": h, "brier_diff": h_diff, "abs_obs_minus_pred_diff": om,
                     "slope_gradient_boosting": slope, "support_status": support, "slope_support_status": support})
    return pd.DataFrame(rows)


def test_promotion_rule():
    plan = {"frozen": True, "plan_hash": "x", "promotion_tolerances": {"max_horizon_brier_worsening": 0.005,
                                                                      "max_abs_obs_minus_pred_worsening": 0.01}}
    assert promotion_decision(_diffs(-0.001), plan, "full")["decision"] == "promote_gradient_boosting"
    assert promotion_decision(_diffs(0.001), plan, "full")["decision"] == "retain_cox"          # interval includes zero
    assert promotion_decision(_diffs(-0.001, h_diff=0.006), plan, "full")["decision"] == "retain_cox"
    assert promotion_decision(_diffs(-0.001, om=0.02), plan, "full")["decision"] == "retain_cox"
    assert promotion_decision(_diffs(-0.001, slope=float("nan")), plan, "full")["decision"] == "retain_cox"
    assert promotion_decision(_diffs(-0.001, h_diff=0.02, support="insufficient_events"), plan, "full")["decision"] == \
        "promote_gradient_boosting"                        # unsupported horizons cannot block, and cannot select
    unfrozen = promotion_decision(_diffs(-0.001), {**plan, "frozen": False}, "full")
    assert unfrozen["decision"] == "promote_gradient_boosting" and unfrozen["evidence"] == "exploratory"
    assert promotion_decision(_diffs(-0.001), plan, "full")["evidence"] == "prespecified"


def test_candidate_is_immutable_and_final_evaluation_is_guarded(tmp_path, cfg):
    store = LocalStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        evaluate_final(store, cfg, "run-x", {"frozen": True}, None, [])
    cand = freeze_candidate(store, "run-x", "gradual_stenotic", "core", {"decision": "retain_cox"})
    assert cand["family"] == "cox"
    with pytest.raises(FileExistsError):
        freeze_candidate(store, "run-x", "gradual_stenotic", "core", {"decision": "promote_gradient_boosting"})
    store.put_json("metrics", "runs/run-x/final/done.json", {"completed_at": "earlier"})
    with pytest.raises(PermissionError, match="already ran"):
        evaluate_final(store, cfg, "run-x", {"frozen": True}, None, [])


def test_only_the_served_family_replaces_the_active_bundle(tmp_path, monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path

    from kairos.io.config import reset_settings_cache

    monkeypatch.setenv("KAIROS_LOCAL_ARTIFACTS_DIR", str(tmp_path))
    reset_settings_cache()
    try:
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("jobs_cli_test", root / "services" / "jobs" / "cli.py")
        cli = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cli
        spec.loader.exec_module(cli)
        assert cli.main(["scenarios", "--scenario", "gradual_stenotic", "--n", "300", "--namespace", "quick", "--seed", "5"]) == 0
        store = cli.get_store()
        rc = cli.main(["train", "--family", "gradient_boosting", "--namespace", "quick", "--intervals", "0", "--step", "reference"])
        assert rc == 0
        vtag = cli.version_tag()
        assert store.exists("models", f"{vtag}/gradient_boosting/bundle.joblib")
        assert not store.exists("models", "latest/bundle.joblib")     # a non-served family never becomes active
    finally:
        reset_settings_cache()


def test_quick_grid_is_used_in_quick_mode(cfg):
    from kairos.modelling.tuning import _grid

    assert _grid(cfg, "quick")["n_estimators"] == [10, 20] and _grid(cfg, "full")["n_estimators"] == [100, 300]
    assert np.isfinite(len(_grid(cfg, "full")["learning_rate"]))
