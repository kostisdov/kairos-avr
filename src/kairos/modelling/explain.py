"""Grouped local sensitivity explanations (WP-D2).

For one patient row, each configured feature group is replaced by a reference value and the
12-month SVD probability is recomputed; the reported effect is ``p(actual) - p(reference)``.
All replacements are scored in a single frame (original row plus one row per group) with one
prediction call, so every family (Cox and gradient boosting) is explained the same way.

These are MODEL SENSITIVITIES, not treatment effects: they describe how this fitted model's
output responds to the patient's inputs, not what would happen to the patient if a value were
changed by intervention.

Reference kinds:

- ``own_reference_echo`` (serial-echo groups): the current measurement is set to the patient's own
  reference study value and its change features to 0, i.e. "what the 12-month SVD probability
  would be if this measurement had not changed since the reference study".
- ``training_reference`` (static, biomarker and exposure groups): training median (numeric) or
  mode (categorical), taken from ``bundle.reference_profile``.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

METHOD = "grouped_local_sensitivity"
OWN_REFERENCE_ECHO = "own_reference_echo"
TRAINING_REFERENCE = "training_reference"
NOTE = "model sensitivities, not treatment effects"


@dataclass
class GroupEffect:
    group: str
    delta_probability: float          # p(actual) - p(reference)
    direction: str                    # "increases" | "decreases" | "none"
    method: str = METHOD
    reference_kind: str = TRAINING_REFERENCE
    features: tuple = ()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["features"] = list(self.features)
        return d


def _direction(delta: float, tol: float) -> str:
    if delta > tol:
        return "increases"
    if delta < -tol:
        return "decreases"
    return "none"


def _replacement(spec, row: pd.Series, reference_profile: dict) -> tuple[str, dict]:
    """(reference kind, {column: replacement value}) restricted to columns present in the row."""
    if isinstance(spec, (list, tuple)):
        spec = {"reference_kind": TRAINING_REFERENCE, "features": list(spec)}
    kind = spec.get("reference_kind", TRAINING_REFERENCE)
    values: dict = {}
    if kind == OWN_REFERENCE_ECHO:
        for target, source in (spec.get("from_reference") or {}).items():
            if target in row.index and source in row.index:
                values[target] = row[source]
        for col in spec.get("zero") or []:
            if col in row.index:
                values[col] = 0.0
    elif kind == TRAINING_REFERENCE:
        for col in spec.get("features") or []:
            if col in row.index and col in reference_profile:
                values[col] = reference_profile[col]
    else:
        raise ValueError(f"unknown reference_kind {kind!r}")
    return kind, values


def grouped_sensitivity(predict_p12m: Callable[[pd.DataFrame], np.ndarray], row: pd.DataFrame,
                        groups: dict, reference_profile: dict | None = None,
                        tol: float = 1e-12) -> list[GroupEffect]:
    """Group effects for the single-row raw feature frame ``row``. ``predict_p12m`` maps a raw
    feature frame to the 12-month SVD probability per row; it is called exactly once, on a frame
    of len(groups)+1 rows (groups with no column in the row are skipped). Effects are sorted by
    absolute delta, largest first."""
    if len(row) != 1:
        raise ValueError("grouped_sensitivity explains exactly one row")
    reference_profile = reference_profile or {}
    base = row.reset_index(drop=True)
    series = base.iloc[0]
    frames, meta = [base], []
    for name, spec in (groups or {}).items():
        kind, values = _replacement(spec, series, reference_profile)
        if not values:
            continue
        r = base.copy()
        for col, v in values.items():
            if r[col].dtype.kind in "biuf" and not isinstance(v, (int, float, np.number, type(None))):
                r[col] = r[col].astype(object)
            r.at[0, col] = np.nan if v is None else v
        frames.append(r)
        meta.append((name, kind, tuple(values)))
    if not meta:
        return []
    frame = pd.concat(frames, ignore_index=True)
    p = np.asarray(predict_p12m(frame), dtype=float).reshape(-1)
    if len(p) != len(frame):
        raise ValueError(f"predict_p12m returned {len(p)} values for {len(frame)} rows")
    out = []
    for i, (name, kind, cols) in enumerate(meta, start=1):
        delta = float(p[0] - p[i])
        out.append(GroupEffect(group=name, delta_probability=delta, direction=_direction(delta, tol),
                               reference_kind=kind, features=cols))
    return sorted(out, key=lambda e: -abs(e.delta_probability))


def bundle_p12m(bundle, near_term_months: float = 12.0) -> Callable[[pd.DataFrame], np.ndarray]:
    """12-month SVD probability of a raw feature frame for either family (the cause-specific
    hazards combined into state probabilities, as in ``Predictor.predict``). The dp-ucMGP
    substudy offset is not applied."""
    from kairos.modelling.cif import combine_cause_specific, probabilities_at

    def predict(frame: pd.DataFrame) -> np.ndarray:
        X = bundle.pipeline.transform(frame)
        for s in bundle.model.strata:
            X[s] = frame[s].astype("object").astype(str).to_numpy()
        H = bundle.model.cumulative_hazards(X)
        cifs = combine_cause_specific(H, bundle.model.grid)
        probs = probabilities_at(cifs, bundle.model.grid, [1.0], float(near_term_months) / 12.0)
        return np.asarray(probs["svd_near_term"], dtype=float)

    return predict


def grouped_sensitivity_for_bundle(predictor_or_bundle, features_row, groups: dict | None = None,
                                   model_cfg: dict | None = None) -> list[GroupEffect]:
    """Grouped sensitivity for a ``ModelBundle`` or ``Predictor`` (either family). ``features_row``
    is a raw landmark feature dict or one-row frame. Groups come from ``groups``, else
    ``model_cfg['explanation_groups']``, else the bundle's config snapshot, else config/model.yaml."""
    bundle = getattr(predictor_or_bundle, "bundle", predictor_or_bundle)
    cfg = model_cfg or getattr(predictor_or_bundle, "cfg", None)
    if groups is None:
        groups = (cfg or {}).get("explanation_groups") or (bundle.config_snapshot or {}).get("explanation_groups")
    if groups is None:
        from kairos.modelling.modules import load_model_config

        cfg = load_model_config()
        groups = cfg.get("explanation_groups") or {}
    near = float((cfg or {}).get("near_term_horizon_months", 12))
    row = features_row if isinstance(features_row, pd.DataFrame) else pd.DataFrame([features_row])
    profile = getattr(bundle, "reference_profile", None) or {}
    return grouped_sensitivity(bundle_p12m(bundle, near), row, groups, profile)
