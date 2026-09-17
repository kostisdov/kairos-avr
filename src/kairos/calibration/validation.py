"""Structural, fit, independent-target, training-contract and privacy checks (design section 9, G5).

Every report keeps four questions apart: structural correctness, resemblance to fit targets,
generalization to independent (held-out) targets, and usefulness for training a prediction model. A fit
target reproduced by a cohort is calibration evidence, never validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from kairos.calibration.targets import simulate_point

RANGES = {"mean_gradient": (2.0, 150.0), "eoa": (0.2, 5.0), "dvi": (0.05, 1.5), "lvef": (5.0, 90.0), "svi": (5.0, 120.0)}
PATIENT_RANGES = {"age_at_implant": (18.0, 110.0), "bsa": (1.0, 3.0), "bmi": (12.0, 70.0), "egfr0": (1.0, 200.0), "size_mm": (14, 36)}


def structural_checks(cohort) -> dict:
    """Hard violations (schema, keys, chronology, ranges, device support). Zero unexplained violations required."""
    from kairos.simulation.generators import _models_by_class

    v: list[str] = []
    required = {"patients": ["patient_id", "route", "design_class", "canonical_model", "size_mm", "implant_date", "reference_date"],
                "echoes": ["patient_id", "date", "mean_gradient", "eoa", "dvi", "ar_grade"],
                "labs": ["patient_id", "date", "analyte", "value"],
                "exposures": ["patient_id", "class", "start_date", "stop_date"],
                "events": ["patient_id", "svd_adjudicated_date", "death_date", "replacement_date", "end_followup_date"]}
    for table, cols in required.items():
        df = getattr(cohort, table)
        missing = [c for c in cols if c not in df.columns and len(df)]
        if missing:
            v.append(f"{table}: missing columns {missing}")
    p, e = cohort.patients, cohort.events
    if p["patient_id"].duplicated().any():
        v.append("patients: duplicate patient_id")
    if len(e) and e["patient_id"].duplicated().any():
        v.append("events: duplicate patient_id")
    ids = set(p["patient_id"])
    for table in ("echoes", "labs", "exposures", "events"):
        df = getattr(cohort, table)
        if len(df) and not set(df["patient_id"]) <= ids:
            v.append(f"{table}: rows for unknown patients")
    implant = p.set_index("patient_id")["implant_date"]
    if len(cohort.echoes):
        ech = cohort.echoes
        early = [pid for pid, d in zip(ech["patient_id"], ech["date"]) if d < implant[pid]]
        if early:
            v.append(f"echoes: {len(early)} studies before implantation")
        for col, (lo, hi) in RANGES.items():
            if col in ech and ((ech[col] < lo) | (ech[col] > hi)).any():
                v.append(f"echoes.{col}: values outside [{lo}, {hi}]")
    lag = [(r - i).days for r, i in zip(p["reference_date"], p["implant_date"])]
    if any(d < 30 or d > 180 for d in lag):
        v.append("patients: reference study outside 30-180 days after implantation")
    for col, (lo, hi) in PATIENT_RANGES.items():
        if col in p and ((p[col] < lo) | (p[col] > hi)).any():
            v.append(f"patients.{col}: values outside [{lo}, {hi}]")
    sizes = {m["canonical_model"]: set(m["sizes"]) for ms in _models_by_class().values() for m in ms}
    bad_size = [(m, s) for m, s in zip(p["canonical_model"], p["size_mm"]) if m in sizes and int(s) not in sizes[m]]
    if bad_size:
        v.append(f"patients: {len(bad_size)} unsupported device-size combinations")
    if len(e):
        for col in ("svd_adjudicated_date", "death_date", "replacement_date"):
            after = [pid for pid, d, end in zip(e["patient_id"], e[col], e["end_followup_date"]) if d is not None and pd.notna(d) and d > end]
            if after:
                v.append(f"events.{col}: {len(after)} after end of follow-up")
    if len(cohort.exposures):
        x = cohort.exposures
        bad = [1 for s, t in zip(x["start_date"], x["stop_date"]) if t is not None and pd.notna(t) and t < s]
        if bad:
            v.append(f"exposures: {len(bad)} episodes stop before they start")
    return {"passed": not v, "violations": v}


def training_contract_checks(cohort, model_cfg: dict) -> dict:
    """The cohort can be consumed by the landmark builder and trainer (not clinical validity)."""
    from kairos.evaluation.support import event_summary
    from kairos.modelling.landmark import FORBIDDEN_PREDICTORS
    from kairos.modelling.modules import block_features
    from kairos.modelling.train import landmark_build_from_cohort

    out = {"passed": True, "issues": []}
    try:
        build = landmark_build_from_cohort(cohort, model_cfg)
    except Exception as ex:  # noqa: BLE001
        return {"passed": False, "issues": [f"landmark build failed: {type(ex).__name__}: {ex}"[:300]]}
    lm = build.rows
    feats = set(block_features(model_cfg, list(model_cfg["feature_blocks"])))
    if feats & FORBIDDEN_PREDICTORS:
        out["issues"].append(f"forbidden predictors in feature blocks: {sorted(feats & FORBIDDEN_PREDICTORS)}")
    if lm.empty:
        out["issues"].append("no landmark rows")
    else:
        ev = cohort.events.set_index("patient_id")["svd_adjudicated_date"]
        after = [pid for pid, d in zip(lm["patient_id"], lm["landmark_date"]) if pd.notna(ev.get(pid)) and d >= ev.get(pid)]
        if after:
            out["issues"].append(f"{len(after)} landmark rows on or after the endpoint")
        out["event_support"] = event_summary(lm)
        out["landmark_rows"] = int(len(lm))
        out["landmark_patients"] = int(lm["patient_id"].nunique())
    out["exclusions"] = build.exclusions
    out["passed"] = not out["issues"]
    return out


def evaluate_points(cohorts: list, points: list) -> pd.DataFrame:
    """Simulated values for target points over one or more cohorts (replicates), with Monte Carlo error."""
    rows = []
    for p in points:
        vals = []
        for c in cohorts:
            try:
                vals.append(simulate_point(c, p, {})[0])
            except Exception:  # noqa: BLE001
                vals.append(float("nan"))
        v = np.asarray(vals, dtype=float)
        sim = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
        mc = float(np.nanstd(v, ddof=1) / np.sqrt(np.isfinite(v).sum())) if np.isfinite(v).sum() > 1 else float("nan")
        within = None if (p.tolerance is None or not np.isfinite(sim)) else bool(abs(sim - p.observed) <= float(p.tolerance))
        rows.append({"point_id": p.point_id, "target_id": p.target_id, "block": p.block, "kind": p.kind, "observed": p.observed,
                     "simulated": sim, "residual": sim - p.observed, "mc_se": mc, "tolerance": p.tolerance,
                     "within_tolerance": within, "mandatory": p.mandatory})
    return pd.DataFrame(rows)


def independent_validation(holdout_points: list, cohorts: list) -> tuple[str, pd.DataFrame]:
    if not holdout_points:
        return "unavailable", pd.DataFrame()
    res = evaluate_points(cohorts, holdout_points)
    mandatory = res[res["mandatory"].astype(bool)]
    ok = bool(len(mandatory)) and bool(mandatory["within_tolerance"].fillna(False).all())
    return ("passed" if ok else "failed"), res


def privacy_check(cohort, source_persons: pd.DataFrame | None = None, numeric=("age_at_implant", "bsa", "bmi", "egfr0"),
                  categorical=("route", "sex", "canonical_model", "size_mm")) -> dict:
    """Source-neighbour distances and exact rare-combination overlap against source persons, when supplied.
    Synthetic labelling alone is not a privacy guarantee; without source persons the check is not assessed."""
    if source_persons is None or source_persons.empty:
        return {"status": "not_assessed", "reason": "no source person table supplied; required before any shared release"}
    cols = [c for c in numeric if c in cohort.patients and c in source_persons]
    syn = cohort.patients[cols].astype(float).to_numpy()
    src = source_persons[cols].astype(float).to_numpy()
    scale = np.nanstd(src, axis=0)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    d = np.sqrt((((syn[:, None, :] - src[None, :, :]) / scale) ** 2).sum(axis=2))
    nearest = d.min(axis=1)
    cat = [c for c in categorical if c in cohort.patients and c in source_persons]
    src_combo = source_persons[cat].astype(str).agg("|".join, axis=1)
    rare = set(src_combo.value_counts()[lambda s: s < 5].index)
    syn_combo = cohort.patients[cat].astype(str).agg("|".join, axis=1)
    return {"status": "assessed", "nearest_source_distance_min": float(nearest.min()),
            "nearest_source_distance_p05": float(np.percentile(nearest, 5)),
            "share_within_0_1_sd": float((nearest < 0.1).mean()), "rare_source_combinations_reproduced": int(syn_combo.isin(rare).sum()),
            "note": "screening only; disclosure testing appropriate to the release is still required"}
