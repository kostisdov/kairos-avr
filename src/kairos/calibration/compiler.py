"""Calibrated specification -> generator inputs (design section 8).

:class:`CompiledGeneratorSpec` offers the attributes the generator reads from a ``ScenarioSpec``
(``name``, ``variant``, ``key``, ``params``, ``labels``, ``get``, ``config_hash``,
``effective_config_hash``) without adding a ``calibrated`` label to the scenario parser's enum. Its
provenance says which parameters were calibrated, fixed at priors, or inherited from the base scenario.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from kairos.calibration.parameters import apply_values, get_path
from kairos.calibration.populations import compile_population_params
from kairos.calibration.schema import ParameterDef, PopulationSpec
from kairos.simulation.registry import deep_merge
from kairos.simulation.scenarios import get_scenario


@dataclass
class CompiledGeneratorSpec:
    name: str
    variant: str | None
    description: str
    evaluates: str
    params: dict
    labels: dict = field(default_factory=dict)
    overrides: dict = field(default_factory=dict)
    config_hash: str = ""
    provenance: dict = field(default_factory=dict)
    generation_kind: str = "calibrated"

    @property
    def key(self) -> str:
        return self.name if not self.variant else f"{self.name}/{self.variant}"

    @property
    def effective_config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.params, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def get(self, path: str, default: Any = None) -> Any:
        return get_path(self.params, path, default)


def compile_spec(population: PopulationSpec, defs: list[ParameterDef], values: dict, bundle_id: str = "uncalibrated",
                 mode: str = "calibrated", generator_baseline: dict | None = None, parameter_set_id: str = "fit") -> CompiledGeneratorSpec:
    base = get_scenario(population.base_scenario, population.base_variant)
    params = deep_merge(base.params, compile_population_params(population))
    params = deep_merge(params, {"generator": {"mode": mode, "baseline": generator_baseline or {}}})
    params = apply_values(params, defs, values)
    labels = copy.deepcopy(base.labels)
    for d in defs:
        labels[d.path] = {"label": "calibrated_fit" if d.status == "free" else "fixed_prior", "source": f"calibration bundle {bundle_id}",
                          "definition_used": d.notes or d.name}
    provenance = {"base_scenario": base.key, "bundle_id": bundle_id, "parameter_set_id": parameter_set_id,
                  "population_id": population.population_id,
                  "free_parameters": [d.name for d in defs if d.status == "free"],
                  "fixed_parameters": [d.name for d in defs if d.status == "fixed"],
                  "assumed_fraction_note": ("every generator parameter not listed here keeps its scenario or registry value, "
                                            "labelled assumed or literature_informed in the manifest")}
    return CompiledGeneratorSpec(name=f"calibrated-{bundle_id}"[:40], variant=parameter_set_id,
                                 description=f"evidence-calibrated specification of {base.key} for population {population.population_id}",
                                 evaluates="calibrated simulation", params=params, labels=labels,
                                 overrides={d.path: values[d.name] for d in defs}, config_hash=base.config_hash,
                                 provenance=provenance)
