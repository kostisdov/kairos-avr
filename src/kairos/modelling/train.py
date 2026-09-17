"""Training entry point: fit one ladder step on one synthetic cohort and bundle it."""
from __future__ import annotations

import numpy as np
import pandas as pd

from kairos.evaluation.support import event_summary
from kairos.modelling.cause_specific import CauseSpecificCoxModel, support_gates
from kairos.modelling.features import FeaturePipeline
from kairos.modelling.landmark import FORBIDDEN_PREDICTORS, LandmarkBuild, build_landmark
from kairos.modelling.modules import (
    event_count_rule,
    feature_types,
    fit_eligibility,
    ladder_steps,
    resolve_features,
)
from kairos.modelling.predictor import ModelBundle, make_bundle
from kairos.modelling.vitamin_k import DpUcMgpOffsetModel, fit_dp_ucmgp_offset
from kairos.simulation.generators import Cohort


def landmark_build_from_cohort(cohort: Cohort, cfg: dict, label_policy: str = "primary",
                               event_override: pd.DataFrame | None = None) -> LandmarkBuild:
    """Landmark rows plus exclusion counts. Only the predictor-visible tables are passed on; the
    cohort's truth tables never reach the builder."""
    lc = cfg["landmark"]
    return build_landmark(cohort.patients, cohort.echoes, cohort.labs, cohort.exposures, cohort.events,
                          horizon_years=float(lc["horizon_years"]), stale_months=int(lc["stale_echo_months"]),
                          min_days_between=int(lc["min_days_between_landmarks"]),
                          time_zero_window=tuple(lc["time_zero_window_days"]),
                          dp_ucmgp_carry_months=float(cfg["vitamin_k"]["carry_forward_months"]),
                          label_policy=label_policy, event_override=event_override)


def landmark_from_cohort(cohort: Cohort, cfg: dict, label_policy: str = "primary") -> pd.DataFrame:
    return landmark_build_from_cohort(cohort, cfg, label_policy).rows


def support_mode_for(cohort: Cohort, cfg: dict) -> str:
    """Support gates follow the cohort namespace: quick cohorts get the relaxed quick gates."""
    ns = (cohort.manifest or {}).get("namespace", "full")
    return (cfg.get("support") or {}).get("mode_by_namespace", {}).get(ns, "full")


def make_cox(cfg: dict, mode: str = "full") -> CauseSpecificCoxModel:
    mc = cfg["model"]
    return CauseSpecificCoxModel(penalizer=float(mc["penalizer"]), l1_ratio=float(mc["l1_ratio"]),
                                 strata=tuple(mc["strata"]), max_years=float(mc["max_cif_time_years"]),
                                 refine=int(mc.get("cif_refine", 1)), horizons=tuple(float(h) for h in cfg["horizons_years"]),
                                 near_term=float(cfg["near_term_horizon_months"]) / 12.0, gates=support_gates(cfg, mode))


FAMILIES = ("cox", "gradient_boosting")


def default_boosting_hyperparameters(cfg: dict) -> dict:
    gb = cfg["families"]["gradient_boosting"]
    return {**gb["default_hyperparameters"], **gb["fixed"]}


def make_model(cfg: dict, family: str = "cox", mode: str = "full", hyperparameters: dict | None = None):
    """An unfitted model of ``family`` with the configured gates, horizons and grid settings."""
    if family == "cox":
        return make_cox(cfg, mode)
    if family != "gradient_boosting":
        raise ValueError(f"unknown model family {family!r}; known: {FAMILIES}")
    from kairos.modelling.gradient_boosting import CauseSpecificGradientBoostingModel

    mc = cfg["model"]
    hp = {**default_boosting_hyperparameters(cfg), **(hyperparameters or {})}
    return CauseSpecificGradientBoostingModel(
        n_estimators=int(hp["n_estimators"]), learning_rate=float(hp["learning_rate"]), max_depth=int(hp["max_depth"]),
        min_samples_leaf=int(hp["min_samples_leaf"]), subsample=float(hp["subsample"]), random_state=int(hp["random_state"]),
        strata=tuple(mc["strata"]), max_years=float(mc["max_cif_time_years"]), refine=int(mc.get("cif_refine", 1)),
        horizons=tuple(float(h) for h in cfg["horizons_years"]), near_term=float(cfg["near_term_horizon_months"]) / 12.0,
        gates=support_gates(cfg, mode))


def fit_step(lm: pd.DataFrame, blocks: list[str], cfg: dict, mode: str = "full", family: str = "cox",
             hyperparameters: dict | None = None, keep_training_data: bool = False):
    """Fit eligibility, the pipeline and the three cause-specific models for one ladder step on
    ``lm``. Every decision (eligibility, medians, levels, rare levels, event support) uses these
    rows only, so call it with training rows. ``mode`` selects the support gates (``full`` or
    ``quick``); ``family`` the model family (the pipeline runs in ``tree`` mode for gradient
    boosting: same eligibility, imputation and encodings, no scaling or splines). Returns
    (pipeline, model, features, unavailable modules, rare levels, eligibility manifest)."""
    manifest = fit_eligibility(lm, cfg)
    feats, dropped, _excluded = resolve_features(cfg, blocks, manifest)
    forbidden = sorted(FORBIDDEN_PREDICTORS & set(feats))
    if forbidden:
        raise ValueError(f"features {forbidden} may never be predictors (identifiers, outcomes, endpoint or synthetic truth)")
    feats, rare = event_count_rule(lm, feats, cfg)
    feats = [f for f in feats if f in lm.columns]
    cat, binr, spl = feature_types(cfg)
    mc = cfg["model"]
    pipe = FeaturePipeline(features=feats, categorical=[c for c in cat if c in feats],
                           binary=[b for b in binr if b in feats], spline=[s for s in spl if s in feats],
                           n_knots=int(mc["spline_knots"]), min_frequency=int(mc["onehot_min_frequency"]),
                           rare_levels={k: v for k, v in rare.items() if isinstance(v, list)},
                           mode="tree" if family == "gradient_boosting" else "cox").fit(lm)
    X = pipe.transform(lm)
    model = make_model(cfg, family, mode, hyperparameters)
    if keep_training_data:
        model.keep_training_data = True
    model.fit(X, lm["time"], lm["event"], list(X.columns), lm[list(mc["strata"])], patient_ids=lm["patient_id"])
    return pipe, model, feats, dropped, rare, manifest


def svd_offset_inputs(pipe, cox, lm: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Full-model SVD linear predictor and SVD cumulative hazard at each row's follow-up time
    (NaN for rows whose route is not supported)."""
    X = pipe.transform(lm)
    for s in cox.strata:
        X[s] = lm[s].astype(str).to_numpy()
    ok, _ = cox.row_support(X)
    expected = np.full(len(lm), np.nan)
    if ok.any():
        expected[ok] = cox.cumulative_hazard_at(X.loc[ok], "svd", lm["time"].to_numpy(dtype=float)[ok])
    lp = np.zeros(len(lm))
    if ok.any():
        lp[ok] = cox.log_risk(X.loc[ok], "svd")
    return lp, expected


def fit_vitamin_k(lm: pd.DataFrame, blocks: list[str], dropped: list[str], pipe, cox, cfg: dict) -> DpUcMgpOffsetModel:
    """The dp-ucMGP offset model for a fitted step, or an unfitted model stating why."""
    need = cfg["vitamin_k"]["requires_block"]
    if need not in blocks or need in dropped:
        return DpUcMgpOffsetModel(False, reason=f"ladder step has no {need} module; dp-ucMGP is consumed only inside it")
    if not cox.has_covariate_model("svd"):
        return DpUcMgpOffsetModel(False, reason="no full-cohort SVD covariate model to provide the offset")
    lp, expected = svd_offset_inputs(pipe, cox, lm)
    return fit_dp_ucmgp_offset(lm, lp, cfg, expected_svd_hazard=expected)


def train_bundle(cohort: Cohort, cfg: dict, ladder_step: str, model_version: str,
                 lm: pd.DataFrame | None = None, mode: str | None = None, family: str = "cox",
                 hyperparameters: dict | None = None, seed: int | None = None) -> ModelBundle:
    """Fit one ladder step of one family on the whole cohort and bundle it. Gradient boosting
    hyperparameters are tuned on patient-grouped inner folds of these rows unless given."""
    steps = dict(ladder_steps(cfg))
    if ladder_step not in steps:
        raise KeyError(f"unknown ladder step {ladder_step!r}; known: {list(steps)}")
    lm = landmark_from_cohort(cohort, cfg) if lm is None else lm
    mode = mode or support_mode_for(cohort, cfg)
    tuning_record = None
    if family == "gradient_boosting" and hyperparameters is None:
        from kairos.modelling.tuning import tune_boosting

        tuning_record = tune_boosting(lm, steps[ladder_step], cfg, int(cfg["evaluation"]["seed"] if seed is None else seed), mode)
        hyperparameters = tuning_record["selected"]
    pipe, cox, feats, dropped, rare, manifest = fit_step(lm, steps[ladder_step], cfg, mode, family, hyperparameters)
    _, _, excluded = resolve_features(cfg, steps[ladder_step], manifest)
    vitamin_k = fit_vitamin_k(lm, steps[ladder_step], dropped, pipe, cox, cfg)
    card = manifest.card()
    summary = {"n_rows": int(len(lm)), "n_patients": int(lm["patient_id"].nunique()),
               "events": {k: int(v) for k, v in lm["event"].value_counts().to_dict().items()},
               "event_support": event_summary(lm),
               "model_support": cox.support_summary(),
               "rare_levels": {k: (v if isinstance(v, str) else len(v)) for k, v in rare.items()},
               "cohort_counts": cohort.manifest.get("counts", {})}
    summary["dataset"] = {k: cohort.manifest.get(k) for k in ("schema_version", "generator_version", "endpoint_version",
                                                              "effective_config_hash", "table_hashes", "namespace",
                                                              "n_requested", "n_retained", "seed")}
    cfg_hash = cohort.manifest.get("effective_config_hash") or cohort.manifest.get("config_hash", "")
    scenario_set = f"{cohort.manifest.get('key', 'unknown')}:{cfg_hash}:seed{cohort.manifest.get('seed', '')}"
    model_levels = [lvl for lvl in pipe.levels_.get("canonical_model", []) if lvl != "other"]
    bundle = make_bundle(pipe, cox, feats, ladder_step, scenario_set, model_version, card["modules"], dropped, summary,
                         cfg, model_levels, vitamin_k, eligibility=card, excluded_features=excluded,
                         tuning_record=tuning_record)
    bundle.reference_profile = reference_profile(lm, feats, cfg)   # WP-D2 training_reference values
    return bundle


def reference_profile(lm: pd.DataFrame, feats: list[str], cfg: dict) -> dict:
    """Training reference for grouped sensitivity explanations: median of each numeric feature,
    mode of each categorical or binary feature (None when a feature has no observed value)."""
    cat, binr, _ = feature_types(cfg)
    discrete = set(cat) | set(binr)
    out: dict = {}
    for f in feats:
        if f not in lm.columns:
            continue
        s = lm[f].dropna()
        if s.empty:
            out[f] = None
        elif f in discrete or not pd.api.types.is_numeric_dtype(s):
            v = s.mode().iloc[0]
            out[f] = v.item() if hasattr(v, "item") else v
        else:
            out[f] = float(s.median())
    return out
