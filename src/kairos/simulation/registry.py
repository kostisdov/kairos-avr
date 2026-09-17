"""Versioned parameter registry for distributions that generator version 2.0 hard-coded.

Every value here reproduces the version-2.0 behaviour exactly in ``scenario`` mode, so legacy scenario
cohorts are unchanged (tested by table hash). A compiled calibrated specification overrides leaves
through ``params["generator"]["baseline"]`` and may switch ``params["generator"]["mode"]`` to
``calibrated``, which enables behaviour that changes random draws:

* device era support: the implantation date is drawn before the device, and devices withdrawn before
  that year are not eligible (``device.enforce_era_support``);
* reference echo dependence: gradient, EOA and DVI drawn through a Gaussian copula with a declared,
  positive-definite correlation matrix (``reference_echo.correlation``);
* longitudinal laboratory processes with per-patient random intercepts and slopes on the transformed
  scale (``labs.analytes.<analyte>.patient_intercept_sd`` / ``patient_slope_sd``).

Provenance: every default is ``assumed`` (generator engineering default), never literature-informed,
unless an override supplies its own provenance.
"""
from __future__ import annotations

import copy
import math
import re
from functools import lru_cache

import numpy as np

REGISTRY_VERSION = "1"
MODES = ("scenario", "calibrated")

BASELINE_DEFAULTS: dict = {
    "demographics": {"age_clip": [45.0, 95.0], "bsa_mean": {"F": 1.75, "M": 1.95}, "bsa_sd": 0.18, "bsa_clip": [1.3, 2.6],
                     "bmi_mean": 27.5, "bmi_sd": 4.5, "bmi_clip": [17.0, 45.0]},
    "comorbidity": {"p_diabetes": 0.30, "diabetes_duration_gamma": [2.0, 4.0],
                    "egfr_intercept_at_75": 66.0, "egfr_age_slope_per_year": -0.5, "egfr_sd": 17.0, "egfr_clip": [8.0, 120.0],
                    "dialysis_egfr_threshold": 15.0, "p_dialysis_other": 0.02,
                    "af_age_slope_per_year": 0.01, "p_af_cap": 0.9,
                    "p_bicuspid_under_65": 0.25, "p_bicuspid_65_plus": 0.08, "p_lipid_lowering": 0.7,
                    "p_ntprobnp_high_at_75": 0.35, "ntprobnp_high_age_slope_per_year": 0.01,
                    "phosphate_mean": 3.5, "phosphate_dialysis_shift": 1.5, "phosphate_sd": 0.5, "phosphate_clip": [2.0, 8.0]},
    "device": {"size_center_mm": 24.0, "size_scale": 18.0, "enforce_era_support": False},
    "reference_echo": {"clip_gradient": [4.0, 30.0], "clip_eoa": [0.8, 3.0], "clip_dvi": [0.25, 0.8],
                       "ar_probs": {"TAVR": {"none": 0.6, "trace": 0.3, "mild": 0.1}, "SAVR": {"none": 0.85, "trace": 0.15}},
                       "correlation": None},
    "ventricle": {"lvef_mean": 58.0, "lvef_sd": 8.0, "lvef_clip": [20.0, 75.0], "svi_mean": 40.0, "svi_sd": 8.0,
                  "svi_clip": [18.0, 70.0], "lvef_decline_per_year": 0.4, "svi_decline_per_year": 0.3, "noise_sd": 3.0,
                  "lvef_obs_clip": [15.0, 75.0], "svi_obs_clip": [15.0, 70.0]},
    "labs": {"p_measured_per_visit": 0.7, "analytes": {
        "egfr": {"transform": "identity", "base": {"state": "egfr0"}, "slope_per_year": -1.0, "residual_sd": 4.0, "floor": 5.0},
        "hba1c": {"transform": "identity", "base": {"condition": "diabetes", "true": 7.5, "false": 5.5}, "residual_sd": 0.5},
        "ldl": {"transform": "identity", "base": {"condition": "lipid_lowering", "true": 85.0, "false": 125.0}, "residual_sd": 25.0},
        "phosphate": {"transform": "identity", "base": {"state": "phosphate0"}, "residual_sd": 0.3},
        "lpa": {"transform": "log", "base": {"condition": "lpa_high", "true": 180.0, "false": 35.0}, "residual_sd": 0.35},
        "ntprobnp": {"transform": "log", "base": {"condition": "ntprobnp_high", "true": 1800.0, "false": 350.0}, "residual_sd": 0.5},
        "hscrp": {"transform": "log", "base": {"value": 2.0}, "residual_sd": 0.7},
    }},
    "calendar": {"study_end": "2026-06-30", "time_origin": "implantation"},
}


def deep_merge(base: dict, override: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def generator_settings(params: dict) -> tuple[str, dict]:
    """(mode, resolved baseline registry) for a parameter set."""
    gen = params.get("generator") or {}
    mode = gen.get("mode", "scenario")
    if mode not in MODES:
        raise ValueError(f"generator mode must be one of {MODES}, not {mode!r}")
    registry = deep_merge(BASELINE_DEFAULTS, gen.get("baseline"))
    validate_registry(registry)
    return mode, registry


def validate_registry(reg: dict) -> None:
    for group in ("bsa_clip", "bmi_clip", "age_clip"):
        lo, hi = reg["demographics"][group]
        if lo >= hi:
            raise ValueError(f"demographics.{group} must be increasing")
    for p in ("p_diabetes", "p_dialysis_other", "p_lipid_lowering", "p_bicuspid_under_65", "p_bicuspid_65_plus"):
        if not 0.0 <= float(reg["comorbidity"][p]) <= 1.0:
            raise ValueError(f"comorbidity.{p} must be a probability")
    for route, probs in reg["reference_echo"]["ar_probs"].items():
        if abs(sum(probs.values()) - 1.0) > 1e-9:
            raise ValueError(f"reference_echo.ar_probs.{route} must sum to 1")
    corr = reg["reference_echo"].get("correlation")
    if corr is not None:
        c = np.asarray(corr, dtype=float)
        if c.shape != (3, 3) or not np.allclose(c, c.T) or not np.allclose(np.diag(c), 1.0):
            raise ValueError("reference_echo.correlation must be a symmetric 3x3 matrix with unit diagonal")
        if np.min(np.linalg.eigvalsh(c)) <= 0:
            raise ValueError("reference_echo.correlation must be positive definite")
    for name, a in reg["labs"]["analytes"].items():
        if a["transform"] not in ("identity", "log"):
            raise ValueError(f"analyte {name}: unsupported transform {a['transform']!r}")
        if float(a["residual_sd"]) < 0:
            raise ValueError(f"analyte {name}: residual_sd must be non-negative")


def analyte_base(spec: dict, state: dict) -> float:
    b = spec["base"]
    if "state" in b:
        return float(state[b["state"]])
    if "condition" in b:
        return float(b["true"] if state[b["condition"]] else b["false"])
    return float(b["value"])


def analyte_value(spec: dict, state: dict, years: float, z: float, intercept_re: float = 0.0, slope_re: float = 0.0) -> float:
    """One laboratory value: base (natural scale) moved on the transformed scale by the population slope,
    patient random effects and a residual ``z * residual_sd``; floored where declared."""
    base = analyte_base(spec, state)
    slope = float(spec.get("slope_per_year", 0.0))
    centre = math.log(base) if spec["transform"] == "log" else base
    # adding exact zeros keeps the version-2.0 values bit for bit when no random effect or slope is declared
    moved = centre + intercept_re + (slope + slope_re) * years + float(spec["residual_sd"]) * z
    val = float(np.exp(moved)) if spec["transform"] == "log" else float(moved)
    floor = spec.get("floor")
    return max(float(floor), val) if floor is not None else val


# --- device era support ---------------------------------------------------------------------------------
_YEAR = re.compile(r"(19|20)\d{2}")


@lru_cache(maxsize=1)
def device_withdrawal_years() -> dict:
    """canonical_model -> last year the device can be implanted (withdrawal/discontinuation year stated in
    the device table's market status), or None when no year is stated. First-approval years are not in
    the device table, so earliest-era support is reported as unavailable, never assumed."""
    from kairos.extraction.rules import device_table

    out = {}
    for _, r in device_table().iterrows():
        status = str(r.get("market_status", "")).lower()
        years = [int(m.group(0)) for m in _YEAR.finditer(status)] if ("withdrawn" in status or "discontinued" in status) else []
        out[r["canonical_model"]] = max(years) if years else None
    return out


def device_supported_in_year(model: str, year: int) -> bool:
    last = device_withdrawal_years().get(model)
    return last is None or year <= last


def provenance_table(registry: dict, overrides: dict | None, prefix: str = "") -> list[dict]:
    """Flat list of registry leaves with provenance: ``assumed`` for defaults, the override's label otherwise."""
    rows = []
    for k, v in registry.items():
        path = f"{prefix}.{k}" if prefix else k
        ov = (overrides or {}).get(k) if isinstance(overrides, dict) else None
        if isinstance(v, dict) and not ({"transform", "base"} <= set(v)):
            rows.extend(provenance_table(v, ov if isinstance(ov, dict) else None, path))
        else:
            rows.append({"parameter": f"generator.baseline.{path}", "value": v,
                         "label": "calibrated_or_override" if ov is not None else "assumed",
                         "source": "compiled calibration specification" if ov is not None else "generator 2.0 engineering default"})
    return rows
