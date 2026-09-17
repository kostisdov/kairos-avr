"""Bounded parameter definitions, transforms and priors (design section 6, step 3).

A parameter specification (YAML ``parameters:`` list of :class:`ParameterDef`) names a dotted path into
the scenario parameters (e.g. ``death_hazard.annual_at_age_79`` or
``generator.baseline.comorbidity.p_diabetes``), bounds, a transform (``identity``, ``log`` or ``logit``),
an optional prior on the transformed scale, and whether it is free or fixed. Only a small declared set is
free; everything else stays at reviewed priors or scenario values. The optimizer works on the unit cube of
the transformed bounds.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import yaml

from kairos.calibration.schema import ParameterDef

TRANSFORMS = ("identity", "log", "logit")


def load_parameter_spec(path: str | Path) -> list[ParameterDef]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    defs = [ParameterDef.from_dict(p) for p in raw["parameters"]]
    validate_parameter_spec(defs)
    return defs


def validate_parameter_spec(defs: list[ParameterDef]) -> None:
    names = [d.name for d in defs]
    if len(set(names)) != len(names):
        raise ValueError("parameter names must be unique")
    for d in defs:
        if d.transform not in TRANSFORMS:
            raise ValueError(f"{d.name}: transform must be one of {TRANSFORMS}")
        if not d.lower < d.upper:
            raise ValueError(f"{d.name}: lower bound must be below upper bound")
        if d.transform == "log" and d.lower <= 0:
            raise ValueError(f"{d.name}: log transform needs positive bounds")
        if d.transform == "logit" and not (0 < d.lower and d.upper < 1):
            raise ValueError(f"{d.name}: logit transform needs bounds inside (0, 1)")
        if d.status not in ("free", "fixed"):
            raise ValueError(f"{d.name}: status must be free or fixed")
        if d.status == "fixed" and d.value is None:
            raise ValueError(f"{d.name}: a fixed parameter needs a value")
        if d.value is not None and not d.lower <= d.value <= d.upper:
            raise ValueError(f"{d.name}: value {d.value} outside bounds")


def forward(d: ParameterDef, x: float) -> float:
    if d.transform == "log":
        return math.log(x)
    if d.transform == "logit":
        return math.log(x / (1 - x))
    return float(x)


def inverse(d: ParameterDef, y: float) -> float:
    if d.transform == "log":
        return math.exp(y)
    if d.transform == "logit":
        return 1 / (1 + math.exp(-y))
    return float(y)


def to_unit(defs: list[ParameterDef], values: dict) -> np.ndarray:
    free = [d for d in defs if d.status == "free"]
    return np.array([(forward(d, values[d.name]) - forward(d, d.lower)) / (forward(d, d.upper) - forward(d, d.lower))
                     for d in free], dtype=float)


def from_unit(defs: list[ParameterDef], u: np.ndarray) -> dict:
    """Natural-scale values of every parameter; free ones from the unit vector (clipped to bounds)."""
    out, k = {}, 0
    for d in defs:
        if d.status == "fixed":
            out[d.name] = float(d.value)
            continue
        lo, hi = forward(d, d.lower), forward(d, d.upper)
        out[d.name] = float(min(max(inverse(d, lo + float(np.clip(u[k], 0.0, 1.0)) * (hi - lo)), d.lower), d.upper))
        k += 1
    return out


def prior_penalty(defs: list[ParameterDef], values: dict) -> float:
    total = 0.0
    for d in defs:
        if d.status == "free" and d.prior_mean is not None and d.prior_sd:
            total += ((forward(d, values[d.name]) - forward(d, d.prior_mean)) / float(d.prior_sd)) ** 2
    return float(total)


def starting_values(defs: list[ParameterDef]) -> dict:
    return {d.name: float(d.value if d.value is not None else (d.prior_mean if d.prior_mean is not None
                                                                else inverse(d, (forward(d, d.lower) + forward(d, d.upper)) / 2)))
            for d in defs}


def set_path(params: dict, path: str, value) -> None:
    keys = path.split(".")
    cur = params
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def get_path(params: dict, path: str, default=None):
    cur = params
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def apply_values(params: dict, defs: list[ParameterDef], values: dict) -> dict:
    out = copy.deepcopy(params)
    for d in defs:
        set_path(out, d.path, float(values[d.name]))
    return out
