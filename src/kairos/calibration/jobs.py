"""Job commands for the calibrated generator (design section 8), registered into ``services/jobs/cli.py``.

    profile-observed      --snapshot <dir> --population <id> [--split-role fit]
    calibrate-generator   --evidence <registry.yaml> ... --population <id> --parameters <yaml> [budget overrides]
                          [--synthetic-recovery name=value,...]
    validate-calibration  --bundle <id> --targets <holdout registry.yaml>
    generate-calibrated   --bundle <id> --n 2500 --replicates 10 --seed 3201 [--namespace quick|full] [--parameter-set fit]
    validate-cohort       --cohort <prefix> --bundle <id>
    train-calibrated      --cohort <prefix> [--family cox|gradient_boosting] [--step ...]   (experimental namespace only)
    evaluate-calibrated   --cohort <prefix> [--families ...] [--quick]                     (experimental namespace only)

Defaults come from ``config/generator_calibration.yaml``. Models trained on calibrated cohorts are written only
under ``models/experimental/`` and never replace the served bundle.
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
import yaml

from kairos.io.config import get_settings
from kairos.io.storage import get_store


def load_calibration_config(path: str | None = None) -> dict:
    p = Path(path) if path else get_settings().config_dir / "generator_calibration.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _population(args, cfg):
    from kairos.calibration.populations import load_populations

    path = args.populations or (cfg.get("inputs") or {}).get("population_specification") or str(
        get_settings().config_dir / "calibration" / "populations.yaml")
    pops = load_populations(path)
    if args.population not in pops:
        raise SystemExit(f"unknown population {args.population!r}; known: {sorted(pops)}")
    return pops[args.population]


def cmd_profile_observed(args) -> int:
    from kairos.calibration.observed import Snapshot, default_policy, profile_observed

    cfg = load_calibration_config(args.config)
    pop = _population(args, cfg)
    policy = default_policy(cfg)
    if args.policy:
        policy.update(yaml.safe_load(Path(args.policy).read_text(encoding="utf-8")))
    snap = Snapshot.load(args.snapshot)
    prof = profile_observed(snap, pop, args.split_role, policy)
    store = get_store()
    path = f"profiles/{snap.manifest.snapshot_id}/{pop.population_id}/{args.split_role}/profile.json"
    store.put_json("calibration", path, prof.to_dict())
    print(json.dumps({"profile": f"calibration/{path}", "persons": prof.persons_in_scope, "summaries": len(prof.summaries),
                      "exclusions": prof.exclusions}, indent=2))
    return 0


def cmd_calibrate_generator(args) -> int:
    from kairos.calibration.api import fit_generator
    from kairos.calibration.bundle import save_bundle
    from kairos.calibration.evidence import EvidenceRegistry, validate_evidence
    from kairos.calibration.fit import FitSettings
    from kairos.calibration.observed import profile_to_targets
    from kairos.calibration.parameters import load_parameter_spec
    from kairos.calibration.recovery import synthetic_recovery_registry
    from kairos.calibration.schema import ObservedProfile

    cfg = load_calibration_config(args.config)
    pop = _population(args, cfg)
    param_path = args.parameters or str(get_settings().config_dir / "calibration" / "parameters_v1.yaml")
    defs = load_parameter_spec(param_path)
    raw_spec = yaml.safe_load(Path(param_path).read_text(encoding="utf-8"))
    if args.free:
        wanted = set(args.free.split(","))
        defs = [replace(d, status="free" if d.name in wanted else "fixed", value=d.value if d.value is not None else d.prior_mean)
                for d in defs]
    registry = EvidenceRegistry()
    extra_scope = []
    for path in args.evidence or []:
        registry = registry.merged(EvidenceRegistry.load(path))
    profile = None
    if args.profile:
        store = get_store()
        profile = ObservedProfile.from_dict(json.loads(store.get_bytes("calibration", args.profile.removeprefix("calibration/"))))
        registry = registry.merged(EvidenceRegistry(profile_to_targets(profile, pop), []))
    settings = FitSettings.from_config(cfg, max_objective_evaluations=args.budget, attempted_patients_per_evaluation=args.n,
                                       initial_design_points=args.design_points, multistarts=args.multistarts,
                                       fit_simulation_seeds=tuple(args.fit_seeds) if args.fit_seeds else None,
                                       diagnostic_simulation_seeds=tuple(args.diagnostic_seeds) if args.diagnostic_seeds else None,
                                       max_seconds=args.max_seconds)
    if args.synthetic_recovery:
        truth = {d.name: d.value for d in defs}
        truth.update({k: float(v) for k, v in (kv.split("=") for kv in args.synthetic_recovery.split(","))})
        defs = [replace(d, evidence_links=[*d.evidence_links, "recovery:"]) for d in defs]
        registry = registry.merged(synthetic_recovery_registry(pop, defs, truth, settings.attempted_patients_per_evaluation))
        extra_scope.append(f"synthetic parameter-recovery test with true values {truth}; not evidence about patients")
    validation = validate_evidence(registry, pop, {**cfg.get("evidence", {}), **cfg.get("calibration", {})})
    for e in validation.errors:
        print(f"evidence error: {e}")
    t0 = time.time()
    bundle = fit_generator(validation, registry.curves, defs, pop, cfg, profile=profile, settings=settings,
                           parameter_spec_raw=raw_spec, uncertainty_replicates=args.uncertainty_replicates,
                           uncertainty_budget=args.uncertainty_budget, checkpoint_store=get_store(), extra_scope=extra_scope)
    manifest = save_bundle(get_store(), bundle)
    print(json.dumps({k: manifest[k] for k in ("bundle_id", "fit_status", "fit_reason", "structural_status",
                                               "independent_validation_status", "targets", "fitted_values", "allowed_purposes")},
                     indent=2, default=str))
    print(f"calibration finished in {time.time() - t0:.0f}s; report: calibration/{bundle.bundle_id}/report.md")
    return 0


def cmd_validate_calibration(args) -> int:
    from kairos.calibration.api import validate_bundle
    from kairos.calibration.bundle import load_bundle
    from kairos.calibration.evidence import EvidenceRegistry, validate_evidence
    from kairos.calibration.schema import PopulationSpec

    cfg = load_calibration_config(args.config)
    store = get_store()
    bundle = load_bundle(store, args.bundle)
    reg = EvidenceRegistry.load(args.targets)
    val = validate_evidence(reg, PopulationSpec.from_dict(bundle.population), {**cfg.get("evidence", {}), **cfg.get("calibration", {})})
    seeds = bundle.seeds["diagnostic"]
    res = validate_bundle(bundle, val, seeds, int(args.n or bundle.seeds["attempted_patients_per_evaluation"]), reg.curves)
    path = f"{bundle.bundle_id}/validation/{reg.fingerprint()}"
    store.put_bytes("calibration", f"{path}/holdout_residuals.csv", res["residuals"].to_csv(index=False).encode())
    store.put_json("calibration", f"{path}/summary.json", {k: v for k, v in res.items() if k != "residuals"})
    print(json.dumps({k: v for k, v in res.items() if k != "residuals"}, indent=2))
    return 0


def cmd_generate_calibrated(args) -> int:
    from kairos.calibration.api import BundleNotAccepted, generate_calibrated
    from kairos.calibration.bundle import calibrated_prefix, load_bundle

    cfg = load_calibration_config(args.config)
    gen = cfg.get("generation", {})
    store = get_store()
    bundle = load_bundle(store, args.bundle)
    n = int(args.n or gen.get("attempted_patients_per_cohort", 2500))
    reps = int(args.replicates or gen.get("cohort_replicates", 10))
    seed0 = int(args.seed or gen.get("first_generation_seed", 3201))
    written = []
    for r in range(reps):
        seed = seed0 + r
        rep_id = f"{args.parameter_set}-seed{seed}"
        prefix = calibrated_prefix(bundle.bundle_id, args.namespace, rep_id)
        if store.exists("scenarios", f"{prefix}/manifest.json"):
            print(f"exists, not regenerated: {prefix}")
            written.append(prefix)
            continue
        try:
            cohort = generate_calibrated(bundle, n, seed, args.parameter_set, args.namespace,
                                         require_accepted=bool(gen.get("require_accepted_bundle", True)), replicate_id=rep_id)
        except BundleNotAccepted as ex:
            print(f"refused: {ex}")
            return 3
        cohort.save(store, prefix)
        written.append(prefix)
        print(f"{prefix}: {cohort.manifest['counts_by_view']}")
    store.put_json("scenarios", f"calibrated/{bundle.bundle_id}/{args.namespace}/replicates.json",
                   {"bundle_id": bundle.bundle_id, "parameter_set": args.parameter_set, "prefixes": written, "n_attempted": n})
    return 0


def cmd_validate_cohort(args) -> int:
    from kairos.calibration.api import evaluate_fidelity, fidelity_markdown
    from kairos.calibration.bundle import load_bundle
    from kairos.modelling.modules import load_model_config
    from kairos.simulation.generators import Cohort

    store = get_store()
    cohort = Cohort.load(store, args.cohort)
    bundle = load_bundle(store, args.bundle or cohort.manifest.get("calibration_bundle_id"))
    source = pd.read_parquet(args.source_persons) if args.source_persons else None
    report = evaluate_fidelity(cohort, bundle, load_model_config(), source)
    store.put_json("scenarios", f"{args.cohort}/diagnostics/fidelity.json", report)
    store.put_bytes("scenarios", f"{args.cohort}/diagnostics/fidelity.md", fidelity_markdown(report).encode())
    print(json.dumps({"structure": report["structure"], "training_contract": {k: report["training_contract"].get(k)
                                                                             for k in ("passed", "issues", "landmark_rows")},
                      "privacy": report["privacy"]["status"]}, indent=2, default=str))
    return 0 if report["structure"]["passed"] else 1


def _calibrated_cohort(store, prefix):
    from kairos.simulation.generators import Cohort

    if not prefix.startswith("calibrated/"):
        raise SystemExit("--cohort must be a calibrated cohort prefix (calibrated/<bundle>/<namespace>/<replicate>)")
    cohort = Cohort.load(store, prefix)
    if cohort.manifest.get("generation_kind") != "calibrated":
        raise SystemExit("the cohort manifest is not a calibrated cohort")
    return cohort


def cmd_train_calibrated(args) -> int:
    import tempfile

    from kairos.io.config import get_settings as gs
    from kairos.modelling.modules import load_model_config
    from kairos.modelling.train import landmark_from_cohort, train_bundle

    store = get_store()
    cfg = load_model_config()
    cohort = _calibrated_cohort(store, args.cohort)
    step = args.step or cfg["deployed_model"]["ladder_step"]
    lm = landmark_from_cohort(cohort, cfg)
    families = ["cox", "gradient_boosting"] if args.family == "both" else [args.family]
    for fam in families:
        b = train_bundle(cohort, cfg, step, gs().model_version(), lm=lm, family=fam)
        b.training_summary["calibration"] = {k: cohort.manifest.get(k) for k in ("calibration_bundle_id", "parameter_set_id",
                                                                                   "parameter_set_type", "population_id", "acceptance", "label")}
        out = f"experimental/{cohort.manifest['calibration_bundle_id']}/{cohort.manifest['replicate_id']}/{fam}"
        with tempfile.TemporaryDirectory() as td:
            store.put_file("models", f"{out}/bundle.joblib", b.save(Path(td) / "bundle.joblib"))
        store.put_json("models", f"{out}/card.json", {**b.card(), "namespace": "experimental", "automatic_promotion": False})
        print(f"models/{out}: {b.training_summary['events']}")
    return 0


def cmd_evaluate_calibrated(args) -> int:
    from kairos.evaluation.ladder import evaluate_ladder
    from kairos.modelling.modules import load_model_config
    from kairos.modelling.train import landmark_from_cohort

    store = get_store()
    cfg = load_model_config()
    cohort = _calibrated_cohort(store, args.cohort)
    lm = landmark_from_cohort(cohort, cfg)
    out = f"experimental/{cohort.manifest['calibration_bundle_id']}/{cohort.manifest['replicate_id']}"
    for fam in args.families:
        res = evaluate_ladder(cohort, cfg, n_splits=3 if args.quick else None,
                              n_boot=int(cfg["evaluation"]["quick_bootstrap"]) if args.quick else None,
                              steps=args.steps, lm=lm, family=fam, mode="quick" if args.quick else None)
        res.results["generation_kind"] = "calibrated"
        res.results["calibration_bundle_id"] = cohort.manifest["calibration_bundle_id"]
        store.put_csv("metrics", f"{out}/{fam}/ladder.csv", res.results)
        print(f"metrics/{out}/{fam}/ladder.csv: {len(res.results)} rows")
    return 0


def register(sub) -> None:
    def common(p):
        p.add_argument("--config", default=None, help="generator calibration config (default config/generator_calibration.yaml)")

    p = sub.add_parser("profile-observed", help="profile a normalized research snapshot (observed evidence only)")
    common(p)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--population", required=True)
    p.add_argument("--populations", default=None)
    p.add_argument("--split-role", default="fit")
    p.add_argument("--policy", default=None, help="YAML with variables and window overrides")
    p.set_defaults(func=cmd_profile_observed)

    c = sub.add_parser("calibrate-generator", help="fit generator parameters to approved, compatible targets")
    common(c)
    c.add_argument("--evidence", nargs="*", default=[])
    c.add_argument("--profile", default=None)
    c.add_argument("--population", required=True)
    c.add_argument("--populations", default=None)
    c.add_argument("--parameters", default=None)
    c.add_argument("--free", default=None, help="comma-separated parameter names to free (others fixed)")
    c.add_argument("--budget", type=int, default=None)
    c.add_argument("--n", type=int, default=None, help="attempted patients per evaluation")
    c.add_argument("--design-points", type=int, default=None)
    c.add_argument("--multistarts", type=int, default=None)
    c.add_argument("--fit-seeds", type=int, nargs="*", default=None)
    c.add_argument("--diagnostic-seeds", type=int, nargs="*", default=None)
    c.add_argument("--max-seconds", type=float, default=None)
    c.add_argument("--uncertainty-replicates", type=int, default=None)
    c.add_argument("--uncertainty-budget", type=int, default=20)
    c.add_argument("--synthetic-recovery", default=None, help="name=value,... true values for a recovery test")
    c.set_defaults(func=cmd_calibrate_generator)

    v = sub.add_parser("validate-calibration", help="held-out target comparison for a bundle")
    common(v)
    v.add_argument("--bundle", required=True)
    v.add_argument("--targets", required=True)
    v.add_argument("--n", type=int, default=None)
    v.set_defaults(func=cmd_validate_calibration)

    g = sub.add_parser("generate-calibrated", help="generate cohorts from an accepted calibration bundle")
    common(g)
    g.add_argument("--bundle", required=True)
    g.add_argument("--n", type=int, default=None)
    g.add_argument("--replicates", type=int, default=None)
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--namespace", choices=["quick", "full"], default="quick")
    g.add_argument("--parameter-set", default="fit")
    g.set_defaults(func=cmd_generate_calibrated)

    f = sub.add_parser("validate-cohort", help="structural, fit, held-out, training-contract and privacy checks")
    common(f)
    f.add_argument("--cohort", required=True)
    f.add_argument("--bundle", default=None)
    f.add_argument("--source-persons", default=None, help="private parquet of source persons for the privacy screen")
    f.set_defaults(func=cmd_validate_cohort)

    t = sub.add_parser("train-calibrated", help="train on a calibrated cohort (experimental namespace only)")
    t.add_argument("--cohort", required=True)
    t.add_argument("--family", choices=["cox", "gradient_boosting", "both"], default="cox")
    t.add_argument("--step", default=None)
    t.set_defaults(func=cmd_train_calibrated)

    e = sub.add_parser("evaluate-calibrated", help="evaluate the ladder on a calibrated cohort (experimental namespace only)")
    e.add_argument("--cohort", required=True)
    e.add_argument("--families", nargs="+", default=["cox"])
    e.add_argument("--steps", nargs="*", default=None)
    e.add_argument("--quick", action="store_true")
    e.set_defaults(func=cmd_evaluate_calibrated)
