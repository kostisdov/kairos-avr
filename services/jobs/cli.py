"""kairos-jobs: scenario generation, training, evaluation, extraction evaluation, reference
upload and database migration. Runs as Azure Container Apps Jobs (manual trigger) and
locally (``python services/jobs/cli.py <command>``); writes to the artefact store
(Blob in Azure, artifacts/ locally).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from kairos import ILLUSTRATIVE_LABEL  # noqa: E402
from kairos.evaluation.compare import (  # noqa: E402
    evaluate_final,
    freeze_candidate,
    paired_differences,
    promotion_decision,
)
from kairos.evaluation.ladder import evaluate_ladder  # noqa: E402
from kairos.evaluation.plots import (  # noqa: E402
    plot_brier_ladder,
    plot_calibration,
    plot_incremental_value,
    plot_phenotype,
    plot_trajectory,
)
from kairos.evaluation.sensitivity import run_observation_sensitivity, summarise  # noqa: E402
from kairos.evaluation.sizing import (  # noqa: E402
    load_plan,
    pilot_scenario,
    plan_status,
    write_plan,
)
from kairos.extraction.rules import extract_rules  # noqa: E402
from kairos.extraction.schema import (  # noqa: E402
    EchoObservation,
    ExposureEpisode,
    ExposureTimeline,
    LabObservation,
    Passport,
    PassportSource,
    PatientStatic,
    PredictRequest,
)
from kairos.io.config import get_settings  # noqa: E402
from kairos.io.storage import copy_tree_to_store, get_store  # noqa: E402
from kairos.modelling import store as model_store  # noqa: E402
from kairos.modelling.modules import load_model_config  # noqa: E402
from kairos.modelling.predictor import Predictor  # noqa: E402
from kairos.modelling.train import landmark_from_cohort, train_bundle  # noqa: E402
from kairos.privacy import scan as privacy_scan  # noqa: E402
from kairos.simulation.generators import Cohort, cohort_prefix, generate_cohort  # noqa: E402
from kairos.simulation.scenarios import get_scenario, list_scenarios  # noqa: E402
from kairos.telemetry import configure_telemetry, get_logger  # noqa: E402

configure_telemetry("kairos-jobs")
log = get_logger("kairos.jobs")
FAMILY_CHOICES = ["cox", "gradient_boosting"]
SMOKE_STAMP = "smoke test: not evidence"


def version_tag() -> str:
    return get_settings().model_version().replace("+", "_")


def _run_record(kind: str):
    try:
        from kairos.io.db import get_repository

        repo = get_repository(get_settings())
    except Exception as ex:  # noqa: BLE001
        log.warning("run record not written (database unavailable: %s)", type(ex).__name__)
        return None, None
    run_id = f"{kind}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    repo.start_run(run_id, kind, {"model_version": get_settings().model_version()})
    return repo, run_id


def _finish(repo, run_id, status="succeeded", manifest=None):
    if repo is not None and run_id is not None:
        try:
            repo.finish_run(run_id, status, manifest)
        except Exception as ex:  # noqa: BLE001
            log.warning("run record not finished: %s", type(ex).__name__)


def key_path(name: str, variant) -> str:
    return f"{name}/{variant or 'default'}"


def latest_prefix(store, name: str, variant, namespace: str = "full") -> str:
    p = f"{key_path(name, variant)}/{namespace}/latest.json"
    if not store.exists("scenarios", p):
        raise FileNotFoundError(f"no {namespace} cohort for {key_path(name, variant)}; run the scenarios job first "
                                f"(scenarios --namespace {namespace})")
    return store.get_json("scenarios", p)["prefix"]


def selected_scenarios(args) -> list[tuple[str, object]]:
    if getattr(args, "all", False) or not getattr(args, "scenario", None):
        return list_scenarios()
    return [(s, v) for s, v in list_scenarios() if s == args.scenario and (args.variant is None or v == args.variant)]


# --- commands -------------------------------------------------------------------------------
def cmd_scenarios(args) -> int:
    store = get_store()
    repo, run_id = _run_record("scenarios")
    written = []
    namespace = args.namespace
    for name, variant in selected_scenarios(args):
        spec = get_scenario(name, variant)
        n = int(args.n or spec.get("n_patients"))
        if namespace == "full" and n != int(spec.get("n_patients")):
            log.error("scenario %s: n=%d differs from the configured n_patients=%s; a smaller or larger cohort belongs "
                      "in the quick namespace (--namespace quick)", spec.key, n, spec.get("n_patients"))
            return 2
        t = time.time()
        cohort = generate_cohort(spec, n=n, seed=args.seed, namespace=namespace)
        prefix = cohort_prefix(spec, args.seed, n, namespace)
        cohort.save(store, prefix)
        store.put_json("scenarios", f"{key_path(name, variant)}/{namespace}/latest.json",
                       {"prefix": prefix, "namespace": namespace, "effective_config_hash": spec.effective_config_hash,
                        "seed": args.seed, "n_requested": n, "generator_version": cohort.manifest["generator_version"],
                        "endpoint_version": cohort.manifest["endpoint_version"],
                        "generated_at": cohort.manifest["generated_at"], "counts": cohort.manifest["counts"]})
        log.info("scenario %s: %s (%.1fs) -> %s", spec.key, cohort.manifest["counts"], time.time() - t, prefix)
        written.append({"key": spec.key, "prefix": prefix, "counts": cohort.manifest["counts"]})
    _finish(repo, run_id, manifest={"written": written})
    print(json.dumps(written, indent=2))
    return 0


def _demo_request(cohort: Cohort, pid: str) -> tuple[PredictRequest, pd.DataFrame]:
    pt = cohort.patients[cohort.patients.patient_id == pid].iloc[0]
    ech = cohort.echoes[cohort.echoes.patient_id == pid].sort_values("date")
    passport = Passport(source=PassportSource(note_ref=f"synthetic:{pid}", note_type="operative", date=pt.implant_date.isoformat()),
                        route=pt.route, canonical_model=pt.canonical_model, design_class=pt.design_class,
                        generation=pt.generation or None, size_mm=int(pt.size_mm), implant_date=pt.implant_date.isoformat())
    obs = [EchoObservation(passport_id=passport.passport_id, date=r.date.isoformat(), mean_gradient_mmhg=float(r.mean_gradient),
                           eoa_cm2=float(r.eoa), dvi=float(r.dvi), ar_grade=r.ar_grade, lvef_pct=float(r.lvef), svi_ml_m2=float(r.svi),
                           native_vs_prosthetic="prosthetic", source="manual") for r in ech.itertuples()]
    exp = cohort.exposures[cohort.exposures.patient_id == pid] if len(cohort.exposures) else pd.DataFrame()
    episodes = []
    if len(exp):
        for _, r in exp.iterrows():
            episodes.append(ExposureEpisode(**{"class": r["class"], "agent": r["agent"], "indication": r["indication"],
                                               "start": r["start_date"].isoformat(),
                                               "stop": r["stop_date"].isoformat() if pd.notna(r["stop_date"]) else None,
                                               "source": r["source"],
                                               "started_within_90d_after_suspicious_echo": bool(r["post_suspicion"])}))
    labs_df = cohort.labs[cohort.labs.patient_id == pid] if len(cohort.labs) else pd.DataFrame()
    labs = [LabObservation(passport_id=passport.passport_id, date=r.date.isoformat(), analyte=r.analyte, value=float(r.value))
            for r in labs_df.itertuples()] if len(labs_df) else []
    static = PatientStatic(age_at_implant=float(pt.age_at_implant), sex=pt.sex, bsa_m2=float(pt.bsa), bmi=float(pt.bmi),
                           diabetes=bool(pt.diabetes), egfr_ml_min=float(pt.egfr0), dialysis=bool(pt.dialysis),
                           atrial_fibrillation=bool(pt.af), bicuspid_native_valve=bool(pt.bicuspid), lipid_lowering=bool(pt.lipid_lowering))
    req = PredictRequest(passport=passport, echo_observations=obs, exposure=ExposureTimeline(passport_id=passport.passport_id, episodes=episodes),
                         labs=labs, static=static, prediction_time=ech.iloc[-1].date.isoformat(), source_kind="synthetic")
    return req, ech


def pick_demo_patient(cohort: Cohort) -> str:
    ev = cohort.events.set_index("patient_id")
    counts = cohort.echoes.groupby("patient_id").size()
    candidates = [p for p, n in counts.items() if n >= 5]
    late = [p for p in candidates if pd.notna(ev.loc[p, "svd_adjudicated_date"])]
    return (late or candidates or list(counts.index))[0]


def trajectory_predictions(predictor: Predictor, req: PredictRequest, ech: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in ech.itertuples():
        sub = req.model_copy(update={"prediction_time": r.date.isoformat()})
        try:
            p = predictor.predict(sub)
        except Exception:  # noqa: BLE001 - endpoint met or before time zero
            continue
        rows.append({"prediction_time": r.date.isoformat(), "p_svd_12m": p.p_svd_12m, "p_svd_5y": p.p_svd_before_death[2],
                     "p_death_5y": p.p_death_before_svd[2], "p_alive_5y": p.p_alive_intact[2], "p_svd_1y": p.p_svd_before_death[0],
                     "p_svd_3y": p.p_svd_before_death[1]})
    return pd.DataFrame(rows)


def cmd_train(args) -> int:
    settings = get_settings()
    store = get_store(settings)
    cfg = load_model_config()
    repo, run_id = _run_record("train")
    scenario = args.scenario or cfg["deployed_model"]["scenario"]
    step = args.step or cfg["deployed_model"]["ladder_step"]
    prefix = latest_prefix(store, scenario, args.variant, args.namespace)
    cohort = Cohort.load(store, prefix)
    lm = landmark_from_cohort(cohort, cfg)
    vtag = version_tag()
    served = cfg["deployed_model"].get("family", "cox")
    families = FAMILY_CHOICES if args.family == "both" else [args.family]
    if run_id is None:
        run_id = f"train-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    stamp = SMOKE_STAMP if args.namespace == "quick" else None
    bundle = None
    trained, support, trained_bundles = {}, {}, {}
    for fam in families:
        t = time.time()
        b = train_bundle(cohort, cfg, step, settings.model_version(), lm=lm, family=fam)
        log.info("trained %s %s on %s: %s (%.1fs)", fam, step, cohort.manifest["key"], b.training_summary["events"], time.time() - t)
        card = b.card()
        if stamp:
            card = {**card, "evidence": stamp}
        with tempfile.TemporaryDirectory() as td:
            local = b.save(Path(td) / "bundle.joblib")
            trained[fam] = model_store.save_run_bundle(store, run_id, fam, local, card)
            # legacy layout, kept for the legacy predict loader; publish via `publish` instead
            store.put_file("models", f"{vtag}/{fam}/bundle.joblib", local)
            store.put_json("models", f"{vtag}/{fam}/card.json", card)
            if fam == served:   # only the served family may replace the legacy latest/ bundle
                log.info("writing legacy models/latest/ for the served family %s (legacy: the predict service "
                         "loads {namespace}/{family}/active.json written by `publish`)", fam)
                for pfx in (vtag, "latest"):
                    store.put_file("models", f"{pfx}/bundle.joblib", local)
                    store.put_json("models", f"{pfx}/card.json", card)
        support[fam] = {"support_mode": card.get("support_mode"), "model_support": card.get("model_support")}
        trained_bundles[fam] = b
        if fam == served:
            bundle = b
    print(json.dumps({"run_id": run_id, "published": False, "families": trained, "support": support,
                      **({"evidence": stamp} if stamp else {}),
                      "next": f"python services/jobs/cli.py publish --family <family> --run-id {run_id} "
                              f"--namespace {args.namespace}"}, indent=2, default=str))
    for fam, b in trained_bundles.items():   # one demo patient per trained family, stored with its run
        if fam != served:
            try:
                pid_f = args.demo_patient or pick_demo_patient(cohort)
                req_f, ech_f = _demo_request(cohort, pid_f)
                preds_f = trajectory_predictions(Predictor(b, cfg), req_f, ech_f)
                store.put_json("models", model_store.run_path(run_id, fam, "demo_patient.json"),
                               {"patient_id": pid_f, "scenario": cohort.manifest["key"],
                                "family": fam, "request": json.loads(req_f.model_dump_json(by_alias=True)),
                                "predictions": preds_f.to_dict(orient="records"), "intervals": None,
                                "label": ILLUSTRATIVE_LABEL, "model_version": settings.model_version(), "run_id": run_id,
                                **({"evidence": stamp} if stamp else {})})
            except Exception as ex:  # noqa: BLE001
                log.warning("demo patient for %s not written: %s", fam, type(ex).__name__)
    if bundle is None:
        log.info("served family %s not trained in this run; the legacy latest/ bundle and demo patient are unchanged", served)
        _finish(repo, run_id, manifest={"scenario": cohort.manifest["key"], "step": step, "families": families,
                                        "run_bundles": trained})
        return 0
    # one-patient demonstration on synthetic dates -------------------------------------------
    predictor = Predictor(bundle, cfg)
    pid = args.demo_patient or pick_demo_patient(cohort)
    req, ech = _demo_request(cohort, pid)
    preds = trajectory_predictions(predictor, req, ech)
    intervals = None
    if args.intervals > 0 and len(preds):
        rng = np.random.default_rng(cfg["evaluation"]["seed"])
        uniq = lm["patient_id"].unique()
        boot = []
        for b in range(args.intervals):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([np.flatnonzero(lm["patient_id"].to_numpy() == p) for p in pick])
            try:
                bb = train_bundle(cohort, cfg, step, settings.model_version(), lm=lm.iloc[idx].reset_index(drop=True))
                boot.append(trajectory_predictions(Predictor(bb, cfg), req, ech).set_index("prediction_time"))
            except Exception as ex:  # noqa: BLE001
                log.warning("bootstrap refit %d failed: %s", b, type(ex).__name__)
        if boot:
            stack = pd.concat(boot, keys=range(len(boot)))
            lo = stack.groupby(level=1).quantile(0.025)
            hi = stack.groupby(level=1).quantile(0.975)
            intervals = pd.DataFrame({f"{c}_lo": lo[c].reindex(preds["prediction_time"]).to_numpy() for c in preds.columns if c != "prediction_time"}
                                     | {f"{c}_hi": hi[c].reindex(preds["prediction_time"]).to_numpy() for c in preds.columns if c != "prediction_time"})
    with tempfile.TemporaryDirectory() as td:
        fig = plot_trajectory(ech, preds, Path(td) / "demo_trajectory.png", title=f"{cohort.manifest['key']} patient {pid}", intervals=intervals)
        for pfx in (vtag, "latest"):
            store.put_file("figures", f"{pfx}/demo_trajectory.png", fig)
    demo = {"patient_id": pid, "scenario": cohort.manifest["key"], "request": json.loads(req.model_dump_json(by_alias=True)),
            "predictions": preds.to_dict(orient="records"), "intervals": intervals.to_dict(orient="records") if intervals is not None else None,
            "label": ILLUSTRATIVE_LABEL, "model_version": settings.model_version(), "family": served, "run_id": run_id,
            **({"evidence": stamp} if stamp else {})}
    store.put_json("models", model_store.run_path(run_id, served, "demo_patient.json"), demo)
    for pfx in (vtag, "latest"):
        store.put_json("models", f"{pfx}/demo_patient.json", demo)
    _finish(repo, run_id, manifest={"scenario": cohort.manifest["key"], "step": step, "model_version": settings.model_version(),
                                    "summary": bundle.training_summary, "run_bundles": trained})
    print(json.dumps(bundle.card(), indent=2, default=str)[:4000])
    print(f"run {run_id}: not published; publish with `cli.py publish --family <family> --run-id {run_id}`")
    return 0


def cmd_publish(args) -> int:
    store = get_store()
    try:
        if args.rollback:
            pointer = model_store.rollback(store, args.namespace, args.family)
        else:
            if not args.run_id:
                log.error("publish needs --run-id (or --rollback)")
                return 2
            pointer = model_store.publish(store, args.namespace, args.family, args.run_id)
    except (FileNotFoundError, LookupError) as ex:
        log.error("%s", ex)
        return 2
    print(json.dumps(pointer, indent=2, default=str))
    return 0


def cmd_evaluate(args) -> int:
    settings = get_settings()
    store = get_store(settings)
    cfg = load_model_config()
    repo, run_id = _run_record("evaluate")
    vtag = version_tag()
    n_splits = 3 if args.quick else None
    n_boot = int(cfg["evaluation"]["quick_bootstrap"]) if args.quick else None
    all_results, all_pheno = [], []
    plan = load_plan()
    families = list(dict.fromkeys(args.families or ["cox"]))
    eval_run = args.run_id or f"eval-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    store.put_json("metrics", f"runs/{eval_run}/run.json",
                   {"run_id": eval_run, "families": families, "namespace": args.namespace, "quick": bool(args.quick),
                    **({"evidence": SMOKE_STAMP} if (args.quick or args.namespace == "quick") else {}),
                    "version": vtag, "plan_hash": (plan or {}).get("plan_hash"), "plan_frozen": bool(plan and plan.get("frozen")),
                    "scenarios": [key_path(n, v) for n, v in selected_scenarios(args)], "started_at": datetime.now(UTC).isoformat()})
    log.info("evaluation run %s: families %s", eval_run, families)
    with tempfile.TemporaryDirectory() as td:
        for name, variant in selected_scenarios(args):
            prefix = latest_prefix(store, name, variant, args.namespace)
            cohort = Cohort.load(store, prefix)
            kp = key_path(name, variant)
            pstat = plan_status(cohort.manifest, plan)
            store.put_json("metrics", f"runs/{eval_run}/{kp}/plan_status.json", pstat)
            if pstat["status"] not in ("quick", "frozen_match"):
                log.warning("evaluating %s without a matching frozen plan (%s): results labelled exploratory", kp, pstat["reason"])
            t = time.time()
            if args.sensitivity == "observation":
                sres = run_observation_sensitivity(cohort, cfg, args.steps or list(cfg["sensitivity"]["steps"]),
                                                   n_splits=n_splits)
                store.put_csv("metrics", f"{vtag}/{kp}/sensitivity_observation.csv", sres.results)
                store.put_csv("metrics", f"{vtag}/{kp}/sensitivity_observation_summary.csv", summarise(sres))
                store.put_json("metrics", f"{vtag}/{kp}/sensitivity_observation.json",
                               {"diagnostics": sres.diagnostics, "label": sres.label, "manifest_key": cohort.manifest["key"]})
                log.info("observation sensitivity %s in %.0fs: %s", kp, time.time() - t, json.dumps(sres.diagnostics["truth"]))
                continue
            lm = landmark_from_cohort(cohort, cfg)
            by_family = {}
            for fam in families:
                t = time.time()
                res = evaluate_ladder(cohort, cfg, n_splits=n_splits, n_boot=n_boot, steps=args.steps, lm=lm, family=fam,
                                      store=store, run_id=eval_run)
                log.info("evaluated %s %s in %.0fs: %d rows", fam, kp, time.time() - t, len(res.results))
                by_family[fam] = res
                _write_ladder(store, cfg, vtag, kp, fam, res, cohort, eval_run, td)
                all_results.append(res.results)
                all_pheno.append(res.phenotype)
            if {"cox", "gradient_boosting"} <= set(by_family):
                _compare_families(store, cfg, vtag, kp, eval_run, by_family, lm, plan, pstat,
                                  int(cfg["evaluation"]["quick_bootstrap"]) if args.quick else int(cfg["evaluation"]["bootstrap"]))
        if all_results:
            allr = pd.concat(all_results, ignore_index=True)
            store.put_csv("metrics", f"{vtag}/ladder_all.csv", allr)
            store.put_csv("metrics", "latest/ladder_all.csv", allr)
            fig = plot_incremental_value(allr, Path(td) / "incremental_value.png", horizon=cfg["horizons_years"][-1])
            store.put_file("figures", f"{vtag}/incremental_value.png", fig)
            store.put_file("figures", "latest/incremental_value.png", fig)
            prim = allr[(allr["state"] == "svd") & (allr["weighting"] == "pooled landmark rows")]
            print(prim.pivot_table(index=["scenario", "family", "step"], columns="horizon_years", values="brier").round(4).to_string())
    _finish(repo, run_id, manifest={"version": vtag, "run": eval_run, "scenarios": [key_path(n, v) for n, v in selected_scenarios(args)]})
    return 0


def _write_ladder(store, cfg, vtag, kp, fam, res, cohort, eval_run, td) -> None:
    out = f"{vtag}/{kp}" if fam == "cox" else f"{vtag}/{kp}/{fam}"
    store.put_csv("metrics", f"{out}/ladder.csv", res.results)
    for step_name, oof in res.oof.items():
        store.put_parquet("metrics", f"{out}/oof/{step_name}.parquet", oof)
        store.put_parquet("metrics", f"runs/{eval_run}/{kp}/oof/{fam}/{step_name}.parquet", oof)
    store.put_json("metrics", f"{out}/ladder.json",
                       {"scenario": cohort.manifest["key"], "manifest": cohort.manifest, "summary": res.summary,
                        "availability": res.availability, "fold_eligibility": {st: {k: v.get("eligibility") for k, v in f.items()}
                                                                               for st, f in res.fold_fits.items()},
                        "coefficients_svd": res.coefficients,
                        "secondary_thrombosis": res.secondary, "dp_ucmgp_substudy": res.vitamin_k,
                        "phenotype": res.phenotype.to_dict(orient="records"),
                        "curves": {k: v.to_dict(orient="records") for k, v in res.curves.items()},
                        "calibration_curves": {st: {str(h): {kind: df.to_dict(orient="records") for kind, df in d.items()}
                                                    for h, d in hs.items()} for st, hs in res.calibration_curves.items()},
                        "fold_support": {st: {k: v.get("support") for k, v in f.items()} for st, f in res.fold_fits.items()},
                        "fold_hyperparameters": {st: {k: {"selected": v.get("hyperparameters"), "seconds": v.get("seconds"),
                                                          "tuning": v.get("tuning")} for k, v in f.items()}
                                                 for st, f in res.fold_fits.items()},
                        "run_id": eval_run, "family": fam,
                        "label": "synthetic scenario: illustrative, unvalidated"})
    if res.vitamin_k.get("incremental"):
        vk_rows = pd.DataFrame(res.vitamin_k["incremental"]).assign(scenario=cohort.manifest["key"])
        store.put_csv("metrics", f"{out}/dp_ucmgp_substudy.csv", vk_rows)
        mech = res.vitamin_k.get("mechanism", {})
        log.info("dp-ucMGP %s: offset p=%s, VKA contrast %s -> %s (shrinkage %s)", kp,
                 res.vitamin_k.get("offset_model", {}).get("p_value"), mech.get("vka_log_hr_without_marker"),
                 mech.get("vka_log_hr_with_marker"), mech.get("vka_shrinkage_fraction"))
    key = cohort.manifest["key"]
    figs = [plot_calibration(res.curves, key, cfg["horizons_years"][-1], Path(td) / f"{out}/calibration.png",
                             steps=["reference", "core", "core_plus_both"]),
            plot_brier_ladder(res.results, key, Path(td) / f"{out}/brier_ladder.png"),
            plot_phenotype(res.phenotype, key, Path(td) / f"{out}/phenotype.png")]
    for f in figs:
        store.put_file("figures", f"{out}/{f.name}", f)


def _compare_families(store, cfg, vtag, kp, eval_run, by_family, lm, plan, pstat, n_boot) -> None:
    mode = by_family["cox"].summary["support_mode"]
    for step_name in by_family["cox"].oof:
        if step_name not in by_family["gradient_boosting"].oof:
            continue
        diffs, info = paired_differences(by_family["cox"].oof[step_name], by_family["gradient_boosting"].oof[step_name],
                                         lm["time"], lm["event"], cfg, mode, n_boot=n_boot, seed=int(cfg["evaluation"]["seed"]))
        decision = (promotion_decision(diffs, plan, mode, plan_match=pstat["status"] == "frozen_match") if len(diffs)
                    else {"decision": "retain_cox", "evidence": "exploratory", "reason": f"no comparison: {info['status']}"})
        decision.update({"step": step_name, "scenario": kp, "comparison": info, "plan_status": pstat})
        for base in (f"{vtag}/{kp}/comparison", f"runs/{eval_run}/{kp}/comparison"):
            if len(diffs):
                store.put_csv("metrics", f"{base}/{step_name}.csv", diffs)
            store.put_json("metrics", f"{base}/{step_name}_decision.json", decision)
        log.info("comparison %s %s: %s (%s)", kp, step_name, decision["decision"], decision["reason"])


def cmd_compare(args) -> int:
    store = get_store()
    cfg = load_model_config()
    plan = load_plan()
    run = store.get_json("metrics", f"runs/{args.run_id}/run.json")
    mode = "quick" if run["namespace"] == "quick" else "full"
    n_boot = int(cfg["evaluation"]["quick_bootstrap"]) if run.get("quick") else int(cfg["evaluation"]["bootstrap"])
    kp = key_path(args.scenario, args.variant)
    base = f"runs/{args.run_id}/{kp}/oof"
    oc = store.get_parquet("metrics", f"{base}/cox/{args.step}.parquet")
    og = store.get_parquet("metrics", f"{base}/gradient_boosting/{args.step}.parquet")
    for o in (oc, og):
        o["status"] = o["status"].fillna("")
    diffs, info = paired_differences(oc, og, oc["time"], oc["event"], cfg, mode, n_boot=n_boot, seed=int(cfg["evaluation"]["seed"]))
    decision = promotion_decision(diffs, plan, mode)
    decision.update({"step": args.step, "scenario": kp, "comparison": info})
    store.put_csv("metrics", f"runs/{args.run_id}/{kp}/comparison/{args.step}.csv", diffs)
    store.put_json("metrics", f"runs/{args.run_id}/{kp}/comparison/{args.step}_decision.json", decision)
    print(json.dumps({k: decision[k] for k in ("decision", "reason")}, indent=2))
    return 0


def cmd_freeze_candidate(args) -> int:
    store = get_store()
    kp = key_path(args.scenario, args.variant)
    path = f"runs/{args.run_id}/{kp}/comparison/{args.step}_decision.json"
    if not store.exists("metrics", path):
        log.error("no comparison decision at %s: evaluate both families first", path)
        return 2
    cand = freeze_candidate(store, args.run_id, kp, args.step, store.get_json("metrics", path))
    print(json.dumps(cand, indent=2, default=str)[:3000])
    return 0


def cmd_evaluate_final(args) -> int:
    store = get_store()
    cfg = load_model_config()
    plan = load_plan()
    cand = store.get_json("metrics", f"runs/{args.run_id}/candidate.json")
    name, _, variant = cand["scenario"].partition("/")
    variant = None if variant in ("", "default") else variant
    dev = Cohort.load(store, latest_prefix(store, name, variant, "full"))
    pstat = plan_status(dev.manifest, plan)
    if pstat["status"] != "frozen_match":
        log.warning("final evaluation without a matching frozen plan (%s): labelled exploratory", pstat["reason"])
    entry = ((plan or {}).get("development", {}).get("scenarios", {}) or {}).get(dev.manifest["key"]) or {}
    n = int(entry.get("n") or dev.manifest.get("n_requested") or len(dev.patients))   # the development cohort's size unless the plan sets one
    seeds = ((plan or {}).get("final_test") or {}).get("seeds") or [20261001, 20261002]
    spec = get_scenario(name, variant)
    tests = [generate_cohort(spec, n=n, seed=int(s), namespace="full") for s in seeds]
    out = evaluate_final(store, cfg, args.run_id, plan, dev, tests, n_boot=int(cfg["evaluation"]["bootstrap"]),
                         rerun_reason=args.rerun_reason)
    print(json.dumps(out["candidate"], indent=2, default=str)[:2000])
    return 0


FIXTURE_EXPECTED = {
    "01_savr_op_perimount_ce.txt": ("SAVR", "Perimount", 23), "02_savr_op_trifecta_hashsize.txt": ("SAVR", "Trifecta", 21),
    "03_savr_op_trifecta_mm_form.txt": ("SAVR", "Trifecta", 21), "04_tavr_sapien3_ultra_size_field.txt": ("TAVR", "SAPIEN 3 Ultra", None),
    "05_tavr_evolut_misspelled_corevalue.txt": ("TAVR", "Evolut PRO+", None), "06_tavr_evolut_r34_shorthand.txt": ("TAVR", None, 34),
    "14_progress_note_epic_ehr_reference_only.txt": (None, "", None), "17_savr_op_ce_size_after.txt": ("SAVR", "Perimount", 25),
    "18_savr_op_ce_mm_form.txt": ("SAVR", "Perimount", None), "19_progress_note_magna_valve_and_sized_magna.txt": (None, "Magna", 21),
    "20_tavr_bare_s3_guarded.txt": ("TAVR", "SAPIEN 3", None), "21_tavr_bare_xt_guarded.txt": ("TAVR", "SAPIEN XT", None),
    "22_progress_note_s3_xt_false_positive_check.txt": (None, "", None), "23_tavr_ultra_near_sapien_proximity.txt": ("TAVR", "SAPIEN 3 Ultra", None),
    "24_tavr_bare_ultra_resilia.txt": ("TAVR", "SAPIEN 3 Ultra RESILIA", None), "25_progress_note_bare_carpentier.txt": (None, "Perimount", None),
}


def cmd_size_pilot(args) -> int:
    """Development-only sizing pilot; writes config/evaluation_plan.yaml unfrozen for review."""
    cfg = load_model_config()
    store = get_store()
    pilots = []
    for name, variant in selected_scenarios(args):
        t = time.time()
        p = pilot_scenario(name, variant, cfg, n=args.n, dev_seeds=args.dev_seeds, n_splits=int(cfg["evaluation"]["n_splits"]))
        pilots.append(p)
        log.info("sizing pilot %s (%.0fs): proposed n %s", key_path(name, variant), time.time() - t, p["proposed_n_cox_primary"])
        store.put_json("metrics", f"sizing_pilot/{key_path(name, variant)}.json", p)
    path = write_plan(pilots, cfg)
    print(f"wrote {path} (frozen: false; review before freezing)")
    return 0


def cmd_extraction_eval(args) -> int:
    """Field-level accuracy on the synthetic fixtures: rules, hosted model, both."""
    from kairos.extraction.llm import LLMExtractor, build_client, merge

    settings = get_settings()
    store = get_store(settings)
    fixtures = ROOT / "tests" / "fixtures" / "synthetic_notes"
    client = build_client(settings, args.api_version) if (args.api_version and settings.openai_endpoint) else None
    llm = LLMExtractor(settings, client=client)
    use_llm = llm.available and not args.rules_only
    rows = []
    for fname, (route, model, size) in FIXTURE_EXPECTED.items():
        text = (fixtures / fname).read_text(encoding="utf-8")
        note_type = "operative" if "_op_" in fname else ("procedure" if "tavr" in fname else "progress")
        rules = extract_rules(text, fname, note_type, "2018")
        variants = {"rules": rules.passport}
        if use_llm:
            try:
                out = llm.extract(text, "synthetic")
                variants["both"] = merge(extract_rules(text, fname, note_type, "2018"), out, text, "2018").passport
                located = sum(1 for e in variants["both"].evidence if e.method == "llm")
                variants["both_llm_evidence_located"] = located
                from kairos.extraction.rules import device_info

                info = device_info(out.canonical_model) if out.canonical_model else {}
                variants["llm"] = Passport(source=PassportSource(note_ref=fname, note_type=note_type, date="2018"),
                                           route=out.route, canonical_model=out.canonical_model, design_class=info.get("design_class"),
                                           size_mm=out.size_mm if out.size_mm and 15 <= out.size_mm <= 36 else None)
            except Exception as ex:  # noqa: BLE001
                log.warning("hosted model failed on %s: %s", fname, type(ex).__name__)
        for method in ("rules", "llm", "both"):
            p = variants.get(method)
            if p is None:
                continue
            rows.append({"fixture": fname, "method": method,
                         "route_correct": None if route is None else (p.route == route),
                         "model_correct": None if model is None else ((p.canonical_model or "") == model),
                         "size_correct": None if size is None else (p.size_mm == size),
                         "llm_quotes_located": variants.get("both_llm_evidence_located") if method == "both" else None})
    df = pd.DataFrame(rows)
    summary = df.groupby("method")[["route_correct", "model_correct", "size_correct"]].mean().round(3)
    print(summary.to_string())
    vtag = version_tag()
    store.put_csv("metrics", f"{vtag}/extraction_fixtures.csv", df)
    store.put_csv("metrics", f"{vtag}/extraction_fixtures_summary.csv", summary.reset_index())
    store.put_csv("metrics", "latest/extraction_fixtures_summary.csv", summary.reset_index())
    return 0


def cmd_llm_check(args) -> int:
    """One synthetic structured-output call per configured deployment (no note text)."""
    from kairos.extraction.llm import LLMExtractor, build_client

    settings = get_settings()
    if not settings.openai_endpoint:
        print("KAIROS_OPENAI_ENDPOINT is not set; nothing to check")
        return 2
    client = build_client(settings, args.api_version) if args.api_version else None
    results = LLMExtractor(settings, client=client).self_check()
    print(json.dumps({"endpoint": settings.openai_endpoint,
                      "api_version": args.api_version or settings.openai_api_version,
                      "results": results}, indent=2, default=str))
    checked = [r for r in results if r["ok"] is not None]
    return 0 if checked and all(r["ok"] for r in checked) else 1


def cmd_upload_reference(args) -> int:
    store = get_store()
    for folder, container in ((ROOT / "data" / "reference", "reference"), (ROOT / "data" / "derived" / "aggregates", "aggregates")):
        if not folder.exists():
            log.warning("%s absent; skipped", folder)
            continue
        findings = privacy_scan(folder, include_forbidden=True)
        if findings:
            for f in findings:
                log.error("privacy finding: %s %s %s", f.rule, f.path, f.detail)
            return 1
        files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in (".csv", ".yaml", ".yml", ".json", ".md")]
        written = copy_tree_to_store(store, container, files, folder)
        log.info("uploaded %d files to %s", len(written), container)
    return 0


def cmd_migrate(args) -> int:
    from kairos.io.db import get_repository

    repo = get_repository(get_settings())
    print("database ready:", repo.engine.url.render_as_string(hide_password=True))
    return 0


def cmd_all(args) -> int:
    rc = cmd_scenarios(args)
    if rc:
        return rc
    args.scenario, args.variant, args.step, args.demo_patient = None, None, None, None
    rc = cmd_train(args)
    if rc:
        return rc
    args.all = True
    return cmd_evaluate(args)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="KAIROS batch jobs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scenarios", help="generate synthetic cohorts")
    s.add_argument("--all", action="store_true")
    s.add_argument("--scenario")
    s.add_argument("--variant")
    s.add_argument("--n", type=int, default=None)
    s.add_argument("--seed", type=int, default=20260916)
    s.add_argument("--namespace", choices=["quick", "full"], default="full",
                   help="full cohorts use the configured size; any other size goes to quick")
    s.set_defaults(func=cmd_scenarios)
    t = sub.add_parser("train", help="train the deployed model bundle")
    t.add_argument("--scenario")
    t.add_argument("--variant")
    t.add_argument("--step")
    t.add_argument("--demo-patient")
    t.add_argument("--intervals", type=int, default=20, help="bootstrap refits for the demo trajectory intervals (0 = none)")
    t.add_argument("--namespace", choices=["quick", "full"], default="full")
    t.add_argument("--family", choices=[*FAMILY_CHOICES, "both"], default="both",
                   help="families to train; only the served family (deployed_model.family) replaces the active bundle")
    t.set_defaults(func=cmd_train)
    e = sub.add_parser("evaluate", help="evaluate the model ladder")
    e.add_argument("--all", action="store_true")
    e.add_argument("--scenario")
    e.add_argument("--variant")
    e.add_argument("--quick", action="store_true")
    e.add_argument("--steps", nargs="*", default=None)
    e.add_argument("--namespace", choices=["quick", "full"], default="full")
    e.add_argument("--families", nargs="+", choices=FAMILY_CHOICES, default=["cox"])
    e.add_argument("--run-id", default=None, help="reuse a run: completed folds are loaded from their checkpoints")
    e.add_argument("--sensitivity", choices=["observation"], default=None,
                   help="run the observation-process and onset sensitivity instead of the ladder")
    e.set_defaults(func=cmd_evaluate)
    c = sub.add_parser("compare", help="paired family comparison and promotion decision for one step of a run")
    c.add_argument("--run-id", required=True)
    c.add_argument("--scenario", required=True)
    c.add_argument("--variant")
    c.add_argument("--step", required=True)
    c.set_defaults(func=cmd_compare)
    fc = sub.add_parser("freeze-candidate", help="write the immutable candidate.json from a comparison decision")
    fc.add_argument("--run-id", required=True)
    fc.add_argument("--scenario", required=True)
    fc.add_argument("--variant")
    fc.add_argument("--step", required=True)
    fc.set_defaults(func=cmd_freeze_candidate)
    ef = sub.add_parser("evaluate-final", help="one-shot evaluation of the frozen candidate on the plan's test seeds")
    ef.add_argument("--run-id", required=True)
    ef.add_argument("--rerun-reason", default=None)
    ef.set_defaults(func=cmd_evaluate_final)
    sp = sub.add_parser("size-pilot", help="development-only sizing pilot; writes an unfrozen evaluation plan")
    sp.add_argument("--all", action="store_true")
    sp.add_argument("--scenario")
    sp.add_argument("--variant")
    sp.add_argument("--n", type=int, default=2000)
    sp.add_argument("--dev-seeds", type=int, default=3)
    sp.set_defaults(func=cmd_size_pilot)
    x = sub.add_parser("extraction-eval", help="field-level extraction accuracy on the synthetic fixtures")
    x.add_argument("--rules-only", action="store_true")
    x.add_argument("--api-version", default=None, help="override KAIROS_OPENAI_API_VERSION, e.g. v1")
    x.set_defaults(func=cmd_extraction_eval)
    k = sub.add_parser("llm-check", help="one synthetic structured-output call per configured model deployment")
    k.add_argument("--api-version", default=None, help="override KAIROS_OPENAI_API_VERSION, e.g. v1")
    k.set_defaults(func=cmd_llm_check)
    u = sub.add_parser("upload-reference", help="upload reference tables and suppressed aggregates")
    u.set_defaults(func=cmd_upload_reference)
    pb = sub.add_parser("publish", help="move a family's active pointer to a trained run, or roll it back")
    pb.add_argument("--family", choices=FAMILY_CHOICES, required=True)
    pb.add_argument("--run-id", default=None)
    pb.add_argument("--namespace", choices=["quick", "full"], default="full")
    pb.add_argument("--rollback", action="store_true", help="restore the previous pointer")
    pb.set_defaults(func=cmd_publish)
    m = sub.add_parser("migrate", help="create database tables")
    m.set_defaults(func=cmd_migrate)
    a = sub.add_parser("all", help="scenarios, train, evaluate")
    a.add_argument("--n", type=int, default=None)
    a.add_argument("--seed", type=int, default=20260916)
    a.add_argument("--quick", action="store_true")
    a.add_argument("--intervals", type=int, default=0)
    a.add_argument("--steps", nargs="*", default=None)
    a.add_argument("--namespace", choices=["quick", "full"], default="full")
    a.set_defaults(func=cmd_all, all=True, scenario=None, variant=None, sensitivity=None, families=["cox"], run_id=None,
                   family="both")
    from kairos.calibration.jobs import register as register_calibration

    register_calibration(sub)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
