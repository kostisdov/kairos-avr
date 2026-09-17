"""Feature blocks, module availability and the fixed model ladder (config/model.yaml).

Eligibility (``config/model.yaml: eligibility``) is fitted on training rows only, with training
patients as the denominator. A module whose anchor marker is measured in fewer than
``min_patient_fraction`` of patients is switched off with a recorded reason; within an available
module every other marker is judged on its own. Nothing is imputed into existence: an absent,
sparse or constant marker is excluded, never filled with a median. Canonical model and generation
enter only where the number of patients with an SVD event per level meets the configured minimum;
other levels collapse to ``other``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from kairos.io.config import get_settings


@lru_cache(maxsize=2)
def load_model_config(path: str | None = None) -> dict:
    p = Path(path) if path else get_settings().config_dir / "model.yaml"
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def block_features(cfg: dict, blocks: list[str]) -> list[str]:
    out: list[str] = []
    for b in blocks:
        for f in cfg["feature_blocks"][b]:
            if f not in out:
                out.append(f)
    return out


def ladder_steps(cfg: dict) -> list[tuple[str, list[str]]]:
    return [(s["name"], list(s["blocks"])) for s in cfg["ladder"]]


def feature_types(cfg: dict) -> tuple[list[str], list[str], list[str]]:
    return (list(cfg["feature_types"]["categorical"]), list(cfg["feature_types"]["binary"]),
            list(cfg["model"]["spline_features"]))


ELIGIBILITY_VERSION = "2"
BOOLEAN_ANCHORS = ("ac_records_available",)   # measured means the value is True


@dataclass(frozen=True)
class MarkerEligibility:
    marker: str
    n_patients: int
    n_patients_measured: int
    patient_fraction: float
    row_fraction: float
    n_distinct: int
    eligible: bool
    reason: str          # "", "absent", "below_cutoff", "constant"


@dataclass
class EligibilityManifest:
    """Training-fold decisions about which markers and modules exist. Frozen into the bundle and
    reused unchanged at inference."""
    version: str
    cutoff: float
    n_patients: int
    markers: dict = field(default_factory=dict)      # marker -> MarkerEligibility
    modules: dict = field(default_factory=dict)      # module -> {status, anchors, included, excluded, reason}
    features_absent: list = field(default_factory=list)

    def card(self) -> dict:
        mods = {}
        for m, info in self.modules.items():
            anchor_frac = min((self.markers[a].patient_fraction for a in info["anchors"] if a in self.markers), default=0.0)
            mods[m] = {**info, "available": info["status"] != "unavailable", "fraction": round(anchor_frac, 3)}
        return {"version": self.version, "cutoff": self.cutoff, "denominator": "training patients",
                "n_patients": self.n_patients, "markers": {k: asdict(v) for k, v in self.markers.items()},
                "modules": mods, "features_absent": list(self.features_absent)}


def _measured(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df:
        return pd.Series(False, index=df.index)
    if col in BOOLEAN_ANCHORS:
        return df[col].map(lambda v: v is True or v == 1 or v == "True").astype(bool)
    x = df[col]
    if x.dtype == object:
        num = pd.to_numeric(x, errors="coerce")
        return num.notna() if num.notna().any() else x.notna()
    return pd.to_numeric(x, errors="coerce").notna()


def _marker_eligibility(df: pd.DataFrame, col: str, cutoff: float, check_constant: bool = True) -> MarkerEligibility:
    pids = df["patient_id"] if "patient_id" in df else pd.Series(np.arange(len(df)), index=df.index)
    n_pat = int(pids.nunique())
    m = _measured(df, col)
    n_meas = int(pids[m].nunique())
    pf = n_meas / n_pat if n_pat else 0.0
    rf = float(m.mean()) if len(df) else 0.0
    n_distinct = int(df.loc[m, col].nunique()) if col in df and m.any() else 0
    if n_meas == 0:
        eligible, reason = False, "absent"
    elif pf < cutoff:
        eligible, reason = False, "below_cutoff"
    elif check_constant and col not in BOOLEAN_ANCHORS and n_distinct <= 1:
        eligible, reason = False, "constant"
    else:
        eligible, reason = True, ""
    return MarkerEligibility(col, n_pat, n_meas, round(pf, 4), round(rf, 4), n_distinct, eligible, reason)


def fit_eligibility(lm_train: pd.DataFrame, cfg: dict) -> EligibilityManifest:
    """Marker and module eligibility from the training rows only (patient-level coverage)."""
    ec = cfg["eligibility"]
    cutoff = float(ec["min_patient_fraction"])
    n_pat = int(lm_train["patient_id"].nunique()) if "patient_id" in lm_train else len(lm_train)
    man = EligibilityManifest(version=ELIGIBILITY_VERSION, cutoff=cutoff, n_patients=n_pat)
    judged = []
    for module, anchors in ec["anchors"].items():
        judged += [a for a in anchors if a not in judged]
        judged += [m for m in ec.get("markers", {}).get(module, []) if m not in judged]
    judged += [d for d in ec.get("derived", {}) if d not in judged]
    for col in judged:
        man.markers[col] = _marker_eligibility(lm_train, col, cutoff)
    for module, anchors in ec["anchors"].items():
        block = cfg["feature_blocks"][module]
        missing_anchor = [a for a in anchors if not man.markers[a].eligible]
        if missing_anchor:
            reasons = {a: man.markers[a].reason for a in missing_anchor}
            man.modules[module] = {"status": "unavailable", "anchors": list(anchors), "included": [],
                                   "excluded": {f: "module_unavailable" for f in block},
                                   "reason": (f"anchor {reasons} below {cutoff:.0%} of training patients; "
                                              "module switched off, not imputed")}
            continue
        judged_here = [m for m in ec.get("markers", {}).get(module, []) + list(ec.get("derived", {})) if m in block]
        excluded = {m: man.markers[m].reason for m in judged_here if not man.markers[m].eligible}
        included = [f for f in block if f not in excluded]
        man.modules[module] = {"status": "partial" if excluded else "full", "anchors": list(anchors),
                               "included": included, "excluded": excluded,
                               "reason": "" if not excluded else f"markers excluded individually: {excluded}"}
    all_feats = block_features(cfg, list(cfg["feature_blocks"]))
    man.features_absent = [f for f in all_feats if not _measured(lm_train, f).any()]
    return man


def resolve_features(cfg: dict, blocks: list[str], manifest: EligibilityManifest) -> tuple[list[str], list[str], dict]:
    """Features of ``blocks`` after eligibility: (features, unavailable modules, excluded feature -> reason)."""
    feats: list[str] = []
    dropped: list[str] = []
    excluded: dict = {}
    for b in blocks:
        info = manifest.modules.get(b)
        if info is not None and info["status"] == "unavailable":
            dropped.append(b)
        for f in cfg["feature_blocks"][b]:
            if f in feats:
                continue
            if info is not None and f in info["excluded"]:
                excluded.setdefault(f, info["excluded"][f])
            elif f in manifest.features_absent:
                excluded.setdefault(f, "absent")
            else:
                feats.append(f)
    feats = [f for f in feats if f not in excluded]
    return feats, dropped, excluded


def event_count_rule(df: pd.DataFrame, features: list[str], cfg: dict) -> tuple[list[str], dict]:
    """Apply the minimum-events rule to canonical_model and generation. Events are unique patients
    with an SVD event (repeated landmark rows of one patient count once)."""
    min_ev = int(cfg["model"]["min_events_for_model_level"])
    rare: dict = {}
    feats = list(features)
    for col in ("canonical_model", "generation"):
        if col not in feats or col not in df:
            continue
        if "event" in df:
            ev_rows = df.loc[df["event"] == 1]
            if "patient_id" in df:
                ev_rows = ev_rows.drop_duplicates("patient_id")
            ev = ev_rows[col].astype("object").fillna("missing").astype(str).value_counts()
        else:
            ev = pd.Series(dtype=int)
        ok_levels = [lvl for lvl, c in ev.items() if c >= min_ev]
        if len(ok_levels) < 2:
            feats.remove(col)
            rare[col] = "dropped: fewer than two levels reach the minimum SVD event count (unique patients)"
        else:
            all_levels = df[col].astype("object").fillna("missing").astype(str).unique().tolist()
            rare[col] = [lvl for lvl in all_levels if lvl not in ok_levels]
    return feats, rare


def guideline_interval_months(cfg: dict, jurisdiction: str, route: str, valve_age_years: float) -> tuple[float, str]:
    g = cfg["guideline_surveillance"][jurisdiction]
    if jurisdiction == "ESC_EACTS":
        return float(g["interval_months"].get(route, 12)), g["description"]
    if route == "TAVR":
        return float(g["TAVR"]["interval_months"]), g["description"]
    s = g["SAVR"]
    if valve_age_years >= s["yearly_after_years"]:
        return 12.0, g["description"]
    nxt = [m for m in s["milestones_years"] if m > valve_age_years]
    if nxt:
        return max(1.0, (nxt[0] - valve_age_years) * 12.0 + 12.0), g["description"]
    return 12.0, g["description"]
