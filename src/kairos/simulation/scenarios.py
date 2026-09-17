"""Scenario configuration.

Reads ``config/scenarios.yaml``: the literature skeleton built during the data pass (kept
verbatim as the evidence base) plus the ``simulation`` section that parameterises the six
fixed scenarios of the revised proposal. Every simulation parameter is a leaf of the form
``{value, label, source, definition_used}`` with ``label`` in
{literature_informed, assumed, varied}; scenarios override leaves by dotted path.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from kairos.io.config import get_settings

SCENARIO_ORDER = ["gradual_stenotic", "regurgitant_abrupt", "high_competing_mortality",
                  "irregular_surveillance", "biomarker_information",
                  "anticoagulant_mechanism_confounding"]
LABELS = {"literature_informed", "assumed", "varied"}


def scenarios_path() -> Path:
    return get_settings().config_dir / "scenarios.yaml"


@lru_cache(maxsize=4)
def load_scenarios_yaml(path: str | None = None) -> dict:
    p = Path(path) if path else scenarios_path()
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _is_leaf(x: Any) -> bool:
    return isinstance(x, dict) and "value" in x and "label" in x


def _resolve(node: Any, prefix: str, labels: dict) -> Any:
    if _is_leaf(node):
        if node["label"] not in LABELS:
            raise ValueError(f"parameter {prefix} has label {node['label']!r}; expected one of {sorted(LABELS)}")
        labels[prefix] = {k: node.get(k) for k in ("label", "source", "definition_used") if node.get(k) is not None}
        return copy.deepcopy(node["value"])
    if isinstance(node, dict):
        return {k: _resolve(v, f"{prefix}.{k}" if prefix else k, labels) for k, v in node.items()}
    return copy.deepcopy(node)


def _set_path(d: dict, path: str, value: Any) -> None:
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = copy.deepcopy(value)


@dataclass
class ScenarioSpec:
    name: str
    variant: str | None
    description: str
    evaluates: str
    params: dict
    labels: dict = field(default_factory=dict)
    overrides: dict = field(default_factory=dict)
    config_hash: str = ""

    @property
    def key(self) -> str:
        return self.name if not self.variant else f"{self.name}/{self.variant}"

    @property
    def effective_config_hash(self) -> str:
        """Hash of this scenario's resolved parameters (``config_hash`` covers the whole
        simulation section and is therefore the same for every scenario)."""
        return hashlib.sha256(json.dumps(self.params, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def get(self, path: str, default: Any = None) -> Any:
        cur = self.params
        for k in path.split("."):
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur


def simulation_section(path: str | None = None) -> dict:
    y = load_scenarios_yaml(path)
    if "simulation" not in y:
        raise KeyError("config/scenarios.yaml has no simulation section")
    return y["simulation"]


def simulation_config_hash(path: str | None = None) -> str:
    sec = simulation_section(path)
    return hashlib.sha256(json.dumps(sec, sort_keys=True, default=str).encode()).hexdigest()[:12]


def list_scenarios(path: str | None = None) -> list[tuple[str, str | None]]:
    sec = simulation_section(path)
    out = []
    for name in SCENARIO_ORDER:
        sc = sec["scenarios"][name]
        variants = sc.get("variants")
        if variants:
            out.extend((name, v) for v in variants)
        else:
            out.append((name, None))
    return out


def get_scenario(name: str, variant: str | None = None, path: str | None = None) -> ScenarioSpec:
    sec = simulation_section(path)
    if name not in sec["scenarios"]:
        raise KeyError(f"unknown scenario {name!r}; known: {sorted(sec['scenarios'])}")
    sc = sec["scenarios"][name]
    labels: dict = {}
    params = _resolve(sec["common"], "", labels)
    overrides: dict = {}
    for k, v in (sc.get("overrides") or {}).items():
        val = v["value"] if _is_leaf(v) else v
        if _is_leaf(v):
            labels[k] = {kk: v.get(kk) for kk in ("label", "source", "definition_used") if v.get(kk) is not None}
        else:
            labels[k] = {"label": "varied"}
        _set_path(params, k, val)
        overrides[k] = val
    variants = sc.get("variants")
    if variants:
        if variant is None:
            raise ValueError(f"scenario {name} has variants {sorted(variants)}; pick one")
        if variant not in variants:
            raise KeyError(f"unknown variant {variant!r} for {name}")
        for k, v in (variants[variant].get("overrides") or {}).items():
            val = v["value"] if _is_leaf(v) else v
            labels[k] = ({kk: v.get(kk) for kk in ("label", "source", "definition_used") if v.get(kk) is not None}
                         if _is_leaf(v) else {"label": "varied"})
            _set_path(params, k, val)
            overrides[k] = val
        description = variants[variant].get("description", sc.get("description", ""))
    elif variant is not None:
        raise ValueError(f"scenario {name} has no variants")
    else:
        description = sc.get("description", "")
    return ScenarioSpec(name=name, variant=variant, description=str(description).strip(),
                        evaluates=str(sc.get("evaluates", "")).strip(), params=params, labels=labels,
                        overrides=overrides, config_hash=simulation_config_hash(path))


def parameter_table(spec: ScenarioSpec) -> list[dict]:
    """Flat listing of every parameter with its label and source, for manifests and docs."""
    rows = []
    for path, meta in sorted(spec.labels.items()):
        rows.append({"parameter": path, "value": spec.get(path), **meta})
    return rows
