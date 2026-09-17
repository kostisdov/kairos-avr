import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from kairos.evaluation.clinical_comparison import (
    ComparisonInputError,
    decision_net_benefit,
    evaluate_comparator_rows,
    evaluate_paired_rows,
    pair_inputs,
    patient_balanced_weights,
    write_comparison_artifacts,
)

KEYS = ["patient_id", "index_valve_id", "landmark_date"]


def fixtures():
    landmarks = pd.DataFrame([
        {"patient_id": "a", "index_valve_id": "v1", "landmark_date": "2020-01-01", "time": 100, "event": 1, "fold": 0},
        {"patient_id": "b", "index_valve_id": "v2", "landmark_date": "2020-01-01", "time": 400, "event": 0, "fold": 1},
    ])
    predictions = landmarks[KEYS + ["time", "event", "fold"]].copy()
    predictions["p_svd_12m"] = [0.1, 0.01]
    comparator = landmarks[KEYS].copy()
    comparator["status"] = ["positive", "negative"]
    return landmarks, predictions, comparator


def test_keyed_pairing_aligns_after_reordering_and_rejects_duplicates_or_mismatch():
    landmarks, predictions, comparator = fixtures()
    originals = [frame.copy(deep=True) for frame in (landmarks, predictions, comparator)]
    paired = pair_inputs(landmarks, predictions.iloc[::-1], comparator.iloc[::-1], KEYS)
    assert paired.sort_values("patient_id")["p_svd_12m"].tolist() == [0.1, 0.01]
    for actual, original in zip((landmarks, predictions, comparator), originals):
        assert_frame_equal(actual, original)
    with pytest.raises(ComparisonInputError, match="duplicate"):
        pair_inputs(landmarks, pd.concat([predictions, predictions.iloc[[0]]]), comparator, KEYS)
    predictions.loc[0, "event"] = 2
    with pytest.raises(ComparisonInputError, match="mismatched"):
        pair_inputs(landmarks, predictions, comparator, KEYS)


def test_hand_calculated_net_benefit_and_assess_none():
    # One true alert and one false alert at p=0.20 -> (1 - .25) / 2 = .375.
    assert decision_net_benefit([1, 1], [1, 0], [1, 1], 0.2) == pytest.approx(0.375)
    assert decision_net_benefit([0, 0], [1, 0], [1, 1], 0.2) == 0.0


def test_all_negative_comparator_is_reported_as_degenerate():
    landmarks, predictions, comparator = fixtures()
    comparator["status"] = "negative"
    paired = pair_inputs(landmarks, predictions, comparator, KEYS)
    metrics, coverage = evaluate_paired_rows(paired, n_boot=5)
    assert coverage["degenerate_no_rule_positives"] is True
    assert set(metrics["strategy"]) >= {"rule_only", "model_only", "assess_all", "assess_none"}


def test_patient_balancing_gives_each_patient_equal_total_weight():
    ids = np.array(["a", "a", "b"])
    weights = patient_balanced_weights(ids)
    assert weights[ids == "a"].sum() == pytest.approx(weights[ids == "b"].sum())


def test_read_only_adapter_and_artifact_writer_stay_in_new_directory(tmp_path):
    selected = pd.DataFrame([{
        "patient_id": "a", "index_valve_id": "v1", "landmark_date": "2023-01-01",
        "reference_study_id": "r1", "current_study_id": "c1",
        "reference_study_date": "2020-03-01", "current_study_date": "2023-01-01",
        "reference_mean_gradient_mmhg": 10, "current_mean_gradient_mmhg": 20,
        "reference_eoa_cm2": 1.6, "current_eoa_cm2": 1.3,
        "reference_dvi": 0.5, "current_dvi": 0.5,
        "reference_ar_grade": "none", "current_ar_grade": "none",
        "intraprosthetic_ar_confirmed": True, "current_fresh": True,
    }])
    original = selected.copy(deep=True)
    audited = evaluate_comparator_rows(selected)
    assert audited.loc[0, "status"] == "positive"
    assert_frame_equal(selected, original)
    out = tmp_path / "comparison"
    write_comparison_artifacts(out, paired_rows=audited, metrics=pd.DataFrame([{"strategy": "rule_only"}]),
                               coverage={"n_common": 1, "n_total": 1, "n_rule_positive": 1},
                               manifest={"threshold": 0.05})
    assert {p.name for p in out.iterdir()} == {"manifest.json", "comparator_rows.parquet", "coverage.json",
                                               "paired_metrics.csv", "decision_curves.csv", "report.md"}
