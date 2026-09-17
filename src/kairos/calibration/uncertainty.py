"""Parameter uncertainty and labelled sensitivity sets (design section 6, G5).

Two kinds of parameter sets, always labelled separately and never mixed with simulation randomness
(independent generation seeds):

* ``target_perturbation_refit``: each fit target is perturbed by its declared standard error, with *one*
  standard normal draw per target shared by all of that target's curve points (a pointwise interval never
  becomes independent coordinates), then the parameters are refitted from the accepted fit with a small
  budget. Targets without a defensible uncertainty are not perturbed;
* ``declared_sensitivity``: named parameter overrides declared in the parameter specification.

A patient-cluster bootstrap of observed profiles belongs to profiling (:mod:`kairos.calibration.observed`);
its standard errors enter here through the local targets' declared uncertainty.
"""
from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np

from kairos.calibration.fit import FitSettings, fit_parameters


def perturb_points(points: list, rng: np.random.Generator) -> list:
    z_by_target: dict = {}
    out = []
    for p in points:
        if p.obs_se is None:
            out.append(p)
            continue
        z = z_by_target.setdefault(p.target_id, float(rng.normal()))
        out.append(replace(p, observed=float(p.observed + z * p.obs_se)))
    return out


def target_perturbation_refits(points, defs, population, settings: FitSettings, best_values: dict, replicates: int,
                               budget_per_refit: int = 20, seed: int = 0, generator_baseline: dict | None = None) -> list[dict]:
    rng = np.random.default_rng(seed)
    sets = []
    refit_defs = [replace(d, value=float(best_values[d.name])) for d in defs]
    small = replace(settings, initial_design_points=0, multistarts=1, max_objective_evaluations=budget_per_refit, finalists=1)
    for r in range(replicates):
        pts = perturb_points(points, rng)
        res = fit_parameters(pts, refit_defs, population, small, generator_baseline)
        sets.append({"parameter_set_id": f"perturbation_{r:03d}", "type": "target_perturbation_refit", "values": res.best_values,
                     "status": res.status, "evaluations": res.evaluations_used})
    return sets


def declared_sensitivity_sets(spec_raw: dict, best_values: dict) -> list[dict]:
    out = []
    for name, overrides in (spec_raw.get("sensitivity_sets") or {}).items():
        values = copy.deepcopy(best_values)
        unknown = set(overrides) - set(values)
        if unknown:
            raise ValueError(f"sensitivity set {name!r} names unknown parameters {sorted(unknown)}")
        values.update({k: float(v) for k, v in overrides.items()})
        out.append({"parameter_set_id": f"sensitivity_{name}", "type": "declared_sensitivity", "values": values, "status": "declared"})
    return out
