from datetime import date

import pytest

from kairos.comparators.varc3_hvd import ComparatorContext, evaluate_varc3_comparator


def echo(gradient=10.0, eoa=1.6, dvi=0.5, ar="none", when=date(2020, 3, 1)):
    return {"date": when, "mean_gradient_mmhg": gradient, "eoa_cm2": eoa, "dvi": dvi,
            "ar_grade": ar, "lvef_pct": 55, "svi_ml_m2": 38}


def context(**kwargs):
    return ComparatorContext(prediction_date=date(2024, 1, 1), intraprosthetic_ar_confirmed=True,
                             source_provenance="synthetic", **kwargs)


@pytest.mark.parametrize("current", [echo(20, 1.3, 0.5, when=date(2023, 1, 1)),
                                      echo(25, 1.6, 0.4, when=date(2023, 1, 1))])
def test_inclusive_moderate_boundaries(current):
    result = evaluate_varc3_comparator(echo(), current, context())
    assert result.status == "positive" and result.highest_demonstrated_stage == 2


@pytest.mark.parametrize("gradient,eoa", [(19.999, 1.3), (20.0, 1.31), (19.0, 1.0)])
def test_gradient_and_area_pair_both_required(gradient, eoa):
    result = evaluate_varc3_comparator(echo(), echo(gradient, eoa, 0.5, when=date(2023, 1, 1)), context())
    assert result.status == "negative"


def test_severe_boundaries_use_unrounded_values():
    ref = echo(10, 1.2, 0.5)
    assert evaluate_varc3_comparator(ref, echo(30, 0.6, 0.3, when=date(2023, 1, 1)), context()).highest_demonstrated_stage == 3
    below = evaluate_varc3_comparator(ref, echo(29.999999, 0.6, 0.3, when=date(2023, 1, 1)), context())
    assert below.highest_demonstrated_stage == 2


def test_independent_ar_branch_survives_missing_stenotic_inputs():
    result = evaluate_varc3_comparator(echo(None, None, None, "mild"),
                                       echo(None, None, None, "moderate", when=date(2023, 1, 1)), context())
    assert result.status == "positive" and result.regurgitant_ge2 is True and result.stenotic_ge2 is None


def test_missing_baseline_or_location_is_unknown_not_negative():
    missing = evaluate_varc3_comparator(echo(None, None, None, None),
                                        echo(None, None, None, "severe", when=date(2023, 1, 1)), context())
    assert missing.status == "indeterminate" and "missing_baseline_ar" in missing.reason_codes
    no_location = evaluate_varc3_comparator(echo(None, None, None, "none"),
                                            echo(None, None, None, "severe", when=date(2023, 1, 1)),
                                            ComparatorContext(prediction_date=date(2024, 1, 1),
                                                              intraprosthetic_ar_confirmed=None))
    assert no_location.status == "indeterminate"


def test_future_or_stale_study_cannot_become_reassuring_negative():
    future = evaluate_varc3_comparator(echo(), echo(11, 1.6, 0.5, when=date(2025, 1, 1)), context())
    assert future.status == "indeterminate" and "future_current_study" in future.reason_codes
    stale = evaluate_varc3_comparator(echo(), echo(11, 1.6, 0.5, when=date(2023, 1, 1)), context(fresh=False))
    assert stale.status == "indeterminate" and "stale_current_study" in stale.reason_codes
