"""Synthetic parameter-recovery targets (work package G4 exit criterion).

Targets are generated from the simulator at *known* parameter values on truth seeds that differ from the
fit and diagnostic seeds, then fitted from deliberately displaced starting values. Recovery of the known
values where they are identifiable, and a visible non-identifiable or failed status where they are not,
tests the calibration machinery. These targets carry ``review_status: approved_synthetic_test`` and
``origin: synthetic_recovery``; they are never evidence about patients.
"""
from __future__ import annotations

import numpy as np

from kairos.calibration.compiler import compile_spec
from kairos.calibration.evidence import EvidenceRegistry
from kairos.calibration.schema import EvidenceTarget, ParameterDef, PopulationSpec
from kairos.calibration.targets import TargetPoint, simulate_point
from kairos.simulation.generators import generate_cohort

DEFAULT_TARGETS = [
    {"id": "death_km_3y", "kind": "km_survival", "variable": "all_cause_death", "origin": "implantation", "horizon": 3.0, "role": "fit",
     "group": "recovery_outcomes"},
    {"id": "death_km_5y", "kind": "km_survival", "variable": "all_cause_death", "origin": "implantation", "horizon": 5.0, "role": "fit",
     "group": "recovery_outcomes"},
    {"id": "svd_cif_5y", "kind": "cumulative_incidence", "variable": "kairos_adjudicated_moderate_or_severe_svd_v2",
     "origin": "reference_echo", "horizon": 5.0, "role": "fit", "group": "recovery_svd"},
    {"id": "diabetes_prop", "kind": "proportion", "variable": "diabetes", "statistic": "proportion:true", "role": "fit",
     "group": "recovery_baseline"},
    {"id": "death_cif_4y_holdout", "kind": "cumulative_incidence", "variable": "all_cause_death", "origin": "reference_echo",
     "horizon": 4.0, "role": "holdout", "group": "recovery_holdout"},
]


def synthetic_recovery_registry(population: PopulationSpec, defs: list[ParameterDef], true_values: dict, n_attempted: int,
                                truth_seeds=(9901, 9902, 9903), target_specs=None, tolerance_se_multiple: float = 3.0,
                                min_tolerance: float = 0.01) -> EvidenceRegistry:
    specs = target_specs or DEFAULT_TARGETS
    spec = compile_spec(population, defs, true_values, "synthetic-truth", "calibrated")
    cohorts = [generate_cohort(spec, n=n_attempted, seed=int(s), namespace="quick") for s in truth_seeds]
    targets = []
    for t in specs:
        view = {"implantation": "enrolled_implantation_origin", "reference_echo": "kairos_reference_eligible"}.get(t.get("origin"),
                                                                                                                    "kairos_reference_eligible")
        point = TargetPoint(t["id"], t["id"], t["group"], t["kind"], t["variable"], t.get("statistic", ""), 0.0, None, None, 1.0, True,
                            t.get("horizon"), view)
        sims = [simulate_point(c, point, {}) for c in cohorts]
        est = float(np.mean([s[0] for s in sims]))
        se = float(np.sqrt(np.mean([s[1] ** 2 for s in sims]) / len(sims)))
        tol = max(min_tolerance, tolerance_se_multiple * se)
        targets.append(EvidenceTarget(
            target_id=f"recovery:{t['id']}", kind=t["kind"], variable=t["variable"], statistic=t.get("statistic", ""), estimate=est,
            uncertainty={"type": "se", "value": se}, sample_size=int(np.mean([s[2] for s in sims])),
            population={"view": view, "population_id": population.population_id},
            time={"origin": t.get("origin", "reference_echo"), "horizon_years": t.get("horizon"), "unit": "years",
                  "competing_events": t["variable"] != "all_cause_death"},
            outcome_definition=t["variable"], study_group_id=t["group"],
            source={"citation": "synthetic recovery truth", "location": f"true values {true_values}", "truth_seeds": list(truth_seeds)},
            review_status="approved_synthetic_test", reviewer="synthetic recovery generator", role=t["role"],
            acceptance={"tolerance": tol, "tolerance_type": "absolute", "weight": 1.0, "mandatory": True},
            origin="synthetic_recovery").to_dict())
    return EvidenceRegistry(targets, [], {"kind": "synthetic_recovery", "true_values": true_values, "truth_seeds": list(truth_seeds)})
