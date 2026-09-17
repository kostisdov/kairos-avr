"""Grouped local sensitivity explanations (WP-D2): model sensitivities, not treatment effects."""
import numpy as np
import pandas as pd

from kairos.modelling.explain import (
    METHOD,
    OWN_REFERENCE_ECHO,
    TRAINING_REFERENCE,
    bundle_p12m,
    grouped_sensitivity,
    grouped_sensitivity_for_bundle,
)
from kairos.modelling.train import landmark_from_cohort


def _row(small_bundle, small_cohort, model_cfg, feature=None):
    lm = landmark_from_cohort(small_cohort, model_cfg)
    lm = lm[lm["route"].astype(str).isin(small_bundle.claimed_routes)]
    if feature is not None:
        lm = lm[lm[feature].notna()]
    return lm.iloc[[len(lm) // 2]][small_bundle.features].reset_index(drop=True)


def test_synthetic_predictor_single_call_and_manual_delta():
    calls = []

    def predict(frame):
        calls.append(len(frame))
        return 0.1 + 0.01 * frame["a"].to_numpy(float) - 0.002 * frame["current_gradient"].to_numpy(float) \
            + 0.003 * frame["delta_gradient"].to_numpy(float)

    row = pd.DataFrame([{"a": 5.0, "b": 1.0, "current_gradient": 30.0, "ref_gradient": 12.0,
                         "delta_gradient": 18.0, "last_change_gradient": 4.0, "gradient_slope": 2.0}])
    groups = {"static": {"reference_kind": TRAINING_REFERENCE, "features": ["a", "b"]},
              "gradient": {"reference_kind": OWN_REFERENCE_ECHO, "from_reference": {"current_gradient": "ref_gradient"},
                           "zero": ["delta_gradient", "last_change_gradient", "gradient_slope"]},
              "absent": {"reference_kind": TRAINING_REFERENCE, "features": ["not_here"]}}
    effects = {e.group: e for e in grouped_sensitivity(predict, row, groups, {"a": 2.0, "b": 0.0})}
    assert calls == [3] and set(effects) == {"static", "gradient"}
    assert np.isclose(effects["static"].delta_probability, 0.01 * (5 - 2))
    assert effects["static"].direction == "increases" and effects["static"].method == METHOD
    manual = (-0.002 * 30 + 0.003 * 18) - (-0.002 * 12 + 0.0)
    assert np.isclose(effects["gradient"].delta_probability, manual)
    assert manual > 0 and effects["gradient"].direction == "increases" and effects["gradient"].reference_kind == OWN_REFERENCE_ECHO


def test_cox_sign_matches_coefficient_and_delta_recomputes(small_bundle, small_cohort, model_cfg):
    b = small_bundle
    assert b.reference_profile and all(f in b.reference_profile for f in b.features)
    coef = b.model.coefficients("svd")
    spline = set(model_cfg["model"]["spline_features"])
    categorical = set(model_cfg["feature_types"]["categorical"]) | set(model_cfg["feature_types"]["binary"])
    predict = bundle_p12m(b)
    checked = 0
    for feat in b.features:
        if feat in spline or feat in categorical or feat == "route" or b.reference_profile.get(feat) is None:
            continue
        cols = [c for c in coef.index if b.pipeline.group(c) == feat]
        if not cols or np.all(np.abs(coef[cols].to_numpy()) < 1e-6):
            continue
        row = _row(b, small_cohort, model_cfg, feat)
        ref = row.copy()
        ref.at[0, feat] = b.reference_profile[feat]
        Xa, Xr = b.pipeline.transform(row), b.pipeline.transform(ref)
        lp_diff = float(sum(coef[c] * (Xa[c].iloc[0] - Xr[c].iloc[0]) for c in cols))
        if abs(lp_diff) < 1e-6:
            continue
        groups = {feat: {"reference_kind": TRAINING_REFERENCE, "features": [feat]}}
        (effect,) = grouped_sensitivity(predict, row, groups, b.reference_profile)
        assert np.sign(effect.delta_probability) == np.sign(lp_diff), feat
        manual = float(predict(row)[0] - predict(ref)[0])
        assert np.isclose(effect.delta_probability, manual, atol=1e-12)
        checked += 1
    assert checked >= 1


def test_bundle_helper_uses_config_groups(small_bundle, small_cohort, model_cfg):
    row = _row(small_bundle, small_cohort, model_cfg)
    effects = grouped_sensitivity_for_bundle(small_bundle, row, model_cfg=model_cfg)
    assert effects and all(e.method == METHOD and e.direction in ("increases", "decreases", "none") for e in effects)
    kinds = {e.group: e.reference_kind for e in effects}
    assert kinds.get("gradient") == OWN_REFERENCE_ECHO and kinds.get("demographics") == TRAINING_REFERENCE
