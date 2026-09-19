"""The protocol comparator ladder and the validation splits."""
import numpy as np
import pandas as pd

from kairos.evaluation.validation_splits import comparators, paired_brier_difference
from kairos.modelling.modules import block_features


def test_every_protocol_comparator_resolves_to_known_features(model_cfg):
    names = [n for n, _ in comparators(model_cfg)]
    assert names[:4] == ["R1_valve_age_and_type", "R2_pre_implant", "R3_reference_echo_once",
                         "R4_current_gradient_and_change"]
    for name, blocks in comparators(model_cfg):
        assert block_features(model_cfg, blocks), name
    blocks = dict(comparators(model_cfg))
    r4 = block_features(model_cfg, blocks["R4_current_gradient_and_change"])
    assert {"current_gradient", "delta_gradient"} <= set(r4) and "current_eoa" not in r4
    r2 = block_features(model_cfg, blocks["R2_pre_implant"])
    assert not {"ieoa", "mismatch_grade", "ref_gradient"} & set(r2)     # pre-implant: nothing from any echo
    blinded = block_features(model_cfg, blocks["KAIROS_visit_history_blinded"])
    assert not {"time_since_last_echo_years", "overdue_flag", "n_echoes", "last_change_interval_years"} & set(blinded)


def test_paired_brier_difference_favours_the_better_prediction():
    rng = np.random.default_rng(0)
    n = 400
    event = (rng.uniform(size=n) < 0.2).astype(int)
    rows = pd.DataFrame({"patient_id": [f"P{i // 2}" for i in range(n)], "event": event,
                         "time": np.where(event == 1, rng.uniform(0.5, 4.5, n), 5.0)})
    good = np.where(event == 1, 0.8, 0.05)
    flat = np.full(n, 0.2)
    d = paired_brier_difference(good, flat, rows, 5.0, n_boot=50)
    assert d["difference"] < 0 and d["ci95"][1] < 0 and d["patients"] == n // 2
    assert paired_brier_difference(flat, flat, rows, 5.0, n_boot=10)["difference"] == 0
