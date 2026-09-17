"""The calibration interfaces named in design section 8, wired over the package modules.

``profile_observed`` and ``validate_evidence`` live in :mod:`observed` and :mod:`evidence`;
``compile_targets`` in :mod:`targets`. This module adds::

    fit_generator(validation, curves, defs, population, cfg, ...) -> CalibrationBundle
    validate_bundle(bundle, holdout_validation, diagnostic_seeds, n) -> dict
    generate_calibrated(bundle, n_attempted, seed, parameter_set_id, ...) -> Cohort
    evaluate_fidelity(cohort, bundle, model_cfg, source_persons=None) -> dict
    resolve_scaffold_generator(standardization_cfg, store) -> callable
"""
from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from kairos.calibration.bundle import COHORT_LABEL, CalibrationBundle, make_bundle_id
from kairos.calibration.compiler import compile_spec
from kairos.calibration.fit import (  # noqa: F401
    FitSettings,
    checkpoint_payload,
    fit_parameters,
    restore_checkpoint,
)
from kairos.calibration.schema import ParameterDef, PopulationSpec, fingerprint
from kairos.calibration.targets import TargetPoint, compile_targets
from kairos.calibration.uncertainty import declared_sensitivity_sets, target_perturbation_refits
from kairos.calibration.validation import (
    evaluate_points,
    independent_validation,
    privacy_check,
    structural_checks,
    training_contract_checks,
)
from kairos.io.config import code_revision
from kairos.simulation.generators import ENDPOINT_VERSION, GENERATOR_VERSION, generate_cohort
from kairos.simulation.registry import REGISTRY_VERSION


class BundleNotAccepted(PermissionError):
    pass


def _point_dicts(points: list[TargetPoint]) -> list[dict]:
    return [asdict(p) for p in points]


def fit_generator(validation, curves: list, defs: list[ParameterDef], population: PopulationSpec, cfg: dict, *,
                  profile=None, settings: FitSettings | None = None, generator_baseline: dict | None = None,
                  parameter_spec_raw: dict | None = None, uncertainty_replicates: int | None = None,
                  uncertainty_budget: int = 20, checkpoint_store=None, extra_scope: list | None = None) -> CalibrationBundle:
    settings = settings or FitSettings.from_config(cfg)
    targets = compile_targets(validation, curves, settings.cap_per_group)
    roles_fp = fingerprint({t["target_id"]: t["role"] for t in validation.targets})
    fingerprints = {"evidence_registry": validation.registry_fingerprint, "roles": roles_fp,
                    "population": fingerprint(population.to_dict()), "parameters": fingerprint([d.to_dict() for d in defs]),
                    "generator_baseline": fingerprint(generator_baseline or {}), "config": fingerprint(cfg),
                    "observed_profile": fingerprint(profile.to_dict()) if profile is not None else None,
                    "settings": fingerprint(asdict(settings)), "code": code_revision()["value"]}
    bundle_id = make_bundle_id({k: v for k, v in fingerprints.items() if k != "code"})
    ckpt_path = f"_checkpoints/{bundle_id}.json"
    checkpoint = None
    if checkpoint_store is not None and checkpoint_store.exists("calibration", ckpt_path):
        import json

        checkpoint = json.loads(checkpoint_store.get_bytes("calibration", ckpt_path))
        if checkpoint.get("roles_fingerprint") != roles_fp:
            raise ValueError("checkpoint was produced under different fit/holdout roles; a warm start cannot change frozen roles")

    def save_ckpt(obj):
        if checkpoint_store is not None:
            import json

            checkpoint_store.put_bytes("calibration", ckpt_path, json.dumps(checkpoint_payload(obj, roles_fp), default=str).encode())

    result = fit_parameters(targets.fit, defs, population, settings, generator_baseline, checkpoint, save_ckpt)

    structural_status, holdout_status, holdout_res = "not_run", "unavailable", pd.DataFrame()
    if result.status != "insufficient_evidence":
        spec = compile_spec(population, defs, result.best_values, bundle_id, "calibrated", generator_baseline)
        cohorts = [generate_cohort(spec, n=settings.attempted_patients_per_evaluation, seed=int(s), namespace="quick")
                   for s in settings.diagnostic_simulation_seeds]
        structural = [structural_checks(c) for c in cohorts]
        structural_status = "passed" if all(s["passed"] for s in structural) else "failed"
        holdout_status, holdout_res = independent_validation(targets.holdout, cohorts)
        if structural_status == "failed" and result.status == "accepted_for_simulation":
            result.status, result.reason = "failed_targets", "structural violations: " + "; ".join(structural[0]["violations"][:3])
    sets = []
    if result.status == "accepted_for_simulation":
        reps = int(cfg.get("calibration", {}).get("source_bootstrap_replicates", 0) if uncertainty_replicates is None else uncertainty_replicates)
        if reps > 0 and targets.fit:
            sets += target_perturbation_refits(targets.fit, defs, population, settings, result.best_values, reps, uncertainty_budget,
                                               settings.optimizer_seed, generator_baseline)
        sets += declared_sensitivity_sets(parameter_spec_raw or {}, result.best_values)
    n_free = sum(1 for d in defs if d.status == "free")
    scope = [f"population {population.population_id} ({population.preset}); {population.scope_statement}".strip("; "),
             "synthetic simulation only: calibration status does not establish clinical predictive validity",
             f"{n_free} parameters fitted; every other generator parameter keeps its scenario or registry value (assumed unless labelled)",
             f"{len(targets.excluded)} evidence targets excluded (unreviewed, incompatible or invalid) and listed with reasons",
             "independent validation " + ("passed on held-out targets" if holdout_status == "passed" else holdout_status),
             *(extra_scope or [])]
    return CalibrationBundle(
        bundle_id=bundle_id, fit_status=result.status, fit_reason=result.reason, structural_status=structural_status,
        independent_validation_status=holdout_status, population=population.to_dict(),
        parameter_definitions=[d.to_dict() for d in defs], fitted_values=result.best_values, input_fingerprints=fingerprints,
        versions={"generator_version": GENERATOR_VERSION, "endpoint_version": ENDPOINT_VERSION, "registry_version": REGISTRY_VERSION,
                  "calibration_schema": "1"},
        seeds={"fit": list(settings.fit_simulation_seeds), "diagnostic": list(settings.diagnostic_simulation_seeds),
               "optimizer": settings.optimizer_seed, "attempted_patients_per_evaluation": settings.attempted_patients_per_evaluation,
               "evaluations_used": result.evaluations_used, "budget": settings.max_objective_evaluations,
               "budget_exhausted": result.budget_exhausted, "seconds": result.seconds},
        targets=[t for t in validation.targets if t["role"] != "excluded"], curves=list(curves or []), excluded_targets=targets.excluded,
        profile_summary=list(profile.summaries) if profile is not None else [], finalists=result.finalists, residuals=result.residuals,
        holdout_residuals=holdout_res, fit_history=result.history, identifiability=result.identifiability, conflicts=result.conflicts,
        uncertainty_sets=sets, generator_baseline=generator_baseline or {}, scope_limitations=scope)


def validate_bundle(bundle: CalibrationBundle, holdout_validation, diagnostic_seeds, n_attempted: int, curves: list | None = None) -> dict:
    """Held-out targets on independent diagnostic seeds, reported separately from the fit."""
    targets = compile_targets(holdout_validation, curves)
    spec = bundle_spec(bundle, "fit")
    cohorts = [generate_cohort(spec, n=n_attempted, seed=int(s), namespace="quick") for s in diagnostic_seeds]
    fit_ids = {t["target_id"] for t in bundle.targets if t["role"] == "fit"}
    leaked = [p.target_id for p in targets.holdout if p.target_id in fit_ids]
    if leaked:
        raise ValueError(f"held-out targets were used for fitting: {leaked[:5]}")
    status, res = independent_validation(targets.holdout, cohorts)
    return {"independent_validation_status": status, "residuals": res, "seeds": list(diagnostic_seeds),
            "note": "held-out comparison; fit-target agreement is not validation"}


def bundle_spec(bundle: CalibrationBundle, parameter_set_id: str = "fit"):
    sets = {s["parameter_set_id"]: s for s in bundle.parameter_sets()}
    if parameter_set_id not in sets:
        raise KeyError(f"unknown parameter set {parameter_set_id!r}; known: {sorted(sets)}")
    defs = [ParameterDef.from_dict(d) for d in bundle.parameter_definitions]
    pop = PopulationSpec.from_dict(bundle.population)
    return compile_spec(pop, defs, sets[parameter_set_id]["values"], bundle.bundle_id, "calibrated", bundle.generator_baseline,
                        parameter_set_id)


def generate_calibrated(bundle: CalibrationBundle, n_attempted: int, seed: int, parameter_set_id: str = "fit",
                        namespace: str = "quick", require_accepted: bool = True, replicate_id: str | None = None):
    if require_accepted and not bundle.accepted:
        raise BundleNotAccepted(f"bundle {bundle.bundle_id} is {bundle.fit_status}: generation requires an accepted bundle "
                                "(generation.require_accepted_bundle)")
    sets = {s["parameter_set_id"]: s for s in bundle.parameter_sets()}
    pset = sets.get(parameter_set_id)
    if pset is None:
        raise KeyError(f"unknown parameter set {parameter_set_id!r}")
    spec = bundle_spec(bundle, parameter_set_id)
    cohort = generate_cohort(spec, n=int(n_attempted), seed=int(seed), namespace=namespace)
    cohort.manifest.update({
        "source_kind": "synthetic", "generation_kind": "calibrated", "calibration_bundle_id": bundle.bundle_id,
        "calibration_bundle_hash": fingerprint(bundle.manifest()), "parameter_set_id": parameter_set_id,
        "parameter_set_type": pset["type"], "population_id": bundle.population.get("population_id"),
        "target_scope": {"fit": [t["target_id"] for t in bundle.targets if t["role"] == "fit"],
                         "holdout": [t["target_id"] for t in bundle.targets if t["role"] == "holdout"]},
        "acceptance": {"fit_status": bundle.fit_status, "structural_status": bundle.structural_status,
                       "independent_validation_status": bundle.independent_validation_status},
        "replicate_id": replicate_id or f"seed{seed}", "fixed_cohort_size_semantics": "attempted",
        "parameter_provenance": spec.provenance, "label": COHORT_LABEL,
        "model_label": "illustrative, unvalidated: model trained on synthetic scenarios"})
    return cohort


def evaluate_fidelity(cohort, bundle: CalibrationBundle, model_cfg: dict, source_persons: pd.DataFrame | None = None) -> dict:
    fit_points = compile_targets(_as_validation(bundle, "fit")).fit
    holdout_points = compile_targets(_as_validation(bundle, "holdout")).holdout
    fit_res = evaluate_points([cohort], fit_points) if fit_points else pd.DataFrame()
    hold_res = evaluate_points([cohort], holdout_points) if holdout_points else pd.DataFrame()
    return {"structure": structural_checks(cohort),
            "fit_targets": {"kind": "calibration fit agreement (not validation)", "rows": fit_res.to_dict(orient="records")},
            "holdout_targets": {"kind": "independent-target comparison" if holdout_points else "unavailable",
                                "rows": hold_res.to_dict(orient="records")},
            "training_contract": training_contract_checks(cohort, model_cfg),
            "privacy": privacy_check(cohort, source_persons),
            "counts_by_view": cohort.manifest.get("counts_by_view"),
            "numerical_support": "single replicate: Monte Carlo error needs replicate cohorts (generate-calibrated --replicates)",
            "label": COHORT_LABEL}


def _as_validation(bundle: CalibrationBundle, role: str):
    from kairos.calibration.evidence import EvidenceValidation

    return EvidenceValidation([t for t in bundle.targets if t["role"] == role], registry_fingerprint="bundle")


def fidelity_markdown(report: dict) -> str:
    lines = ["# Calibrated cohort fidelity", "", f"Label: {report['label']}", "",
             f"- Structure: {'passed' if report['structure']['passed'] else 'failed'} {report['structure']['violations'][:5]}",
             f"- Training contract: {'passed' if report['training_contract']['passed'] else 'failed'} "
             f"{report['training_contract'].get('issues', [])}",
             f"- Privacy: {report['privacy']['status']}", f"- Counts by view: {report['counts_by_view']}", "",
             "## Fit targets (calibration agreement, not validation)", "", "| target | observed | simulated | within tolerance |", "|---|---|---|---|"]
    lines += [f"| {r['point_id']} | {r['observed']:.4g} | {r['simulated']:.4g} | {r['within_tolerance']} |" for r in report["fit_targets"]["rows"]]
    lines += ["", f"## Held-out targets ({report['holdout_targets']['kind']})", ""]
    lines += [f"- {r['point_id']}: observed {r['observed']:.4g}, simulated {r['simulated']:.4g}, within {r['within_tolerance']}"
              for r in report["holdout_targets"]["rows"]]
    return "\n".join(lines) + "\n"


class ScaffoldBackendError(RuntimeError):
    pass


def resolve_scaffold_generator(std_cfg: dict, store):
    """Generator for standardization training-scaffold companion episodes. ``scenario`` uses the pinned
    scenario; ``calibrated_bundle`` needs an explicit, accepted bundle manifest and never falls back."""
    from kairos.calibration.bundle import load_bundle
    from kairos.simulation.scenarios import get_scenario

    d = (std_cfg.get("training_export") or {}).get("defaults") or {}
    backend = d.get("generator_backend", "scenario")
    if backend == "scenario":
        spec = get_scenario(d.get("scenario", "gradual_stenotic"))
        return lambda n, seed: generate_cohort(spec, n=n, seed=seed, namespace="quick")
    if backend != "calibrated_bundle":
        raise ScaffoldBackendError(f"unknown generator_backend {backend!r}")
    ref = d.get("calibration_bundle_manifest")
    if not ref:
        raise ScaffoldBackendError("generator_backend calibrated_bundle needs calibration_bundle_manifest; no fallback generator is used")
    bundle_id = str(ref).rstrip("/").split("/")[-2] if str(ref).endswith("manifest.json") else str(ref)
    try:
        bundle = load_bundle(store, bundle_id)
    except Exception as ex:  # noqa: BLE001
        raise ScaffoldBackendError(f"calibration bundle {bundle_id} cannot be loaded: {type(ex).__name__}: {ex}") from ex
    if d.get("require_accepted_calibration_bundle", True) and not bundle.accepted:
        raise ScaffoldBackendError(f"calibration bundle {bundle_id} is {bundle.fit_status}; an accepted bundle is required")

    def gen(n, seed):
        c = generate_calibrated(bundle, n, seed, "fit", "quick")
        c.manifest.update({"source_kind": "real_with_synthetic_defaults_companion", "purpose": "pipeline_test",
                           "clinical_training_allowed": False})
        return c
    return gen
