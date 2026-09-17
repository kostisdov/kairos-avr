import pandas as pd

from kairos.modelling.modules import load_model_config
from kairos.modelling.tuning import tune_boosting


def test_budget_grid_skips_inner_tuning(monkeypatch):
    monkeypatch.setenv("KAIROS_BOOSTING_GRID", "budget")
    cfg = load_model_config()
    lm = pd.DataFrame({"patient_id": [f"p{i}" for i in range(30)]})
    out = tune_boosting(lm, [], cfg, seed=1)
    assert out["status"] == "fixed_single_candidate"
    assert (out["selected"]["n_estimators"], out["selected"]["learning_rate"], out["selected"]["max_depth"]) == (300, 0.1, 2)
