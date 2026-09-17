"""Essential tests of docs/kairos_calibrated_generator_design.md (section 10) for kairos.calibration."""
import copy
import json
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from kairos.calibration.api import (
    BundleNotAccepted,
    ScaffoldBackendError,
    fit_generator,
    generate_calibrated,
    resolve_scaffold_generator,
)
from kairos.calibration.bundle import calibrated_prefix, load_bundle, save_bundle
from kairos.calibration.evidence import (
    EvidenceRegistry,
    candidates_from_published_rates,
    study_group_weights,
    validate_curve,
    validate_evidence,
)
from kairos.calibration.fit import FitSettings, target_conflicts
from kairos.calibration.observed import Snapshot, SnapshotRejected, default_policy, profile_observed
from kairos.calibration.parameters import from_unit, load_parameter_spec, to_unit
from kairos.calibration.populations import load_populations, study_view
from kairos.calibration.recovery import synthetic_recovery_registry
from kairos.calibration.schema import EvidenceTarget, SnapshotManifest
from kairos.calibration.targets import TargetPoint, compile_targets
from kairos.io.config import get_settings
from kairos.io.storage import LocalStore
from kairos.km_reconstruct import reconstruct_km
from kairos.simulation.generators import generate_cohort
from kairos.simulation.registry import (
    BASELINE_DEFAULTS,
    device_supported_in_year,
    generator_settings,
)
from kairos.simulation.scenarios import get_scenario

CONFIG = get_settings().config_dir


@pytest.fixture(scope="module")
def pops():
    return load_populations(CONFIG / "calibration" / "populations.yaml")


@pytest.fixture(scope="module")
def defs():
    return load_parameter_spec(CONFIG / "calibration" / "parameters_v1.yaml")


# --- snapshot builders ------------------------------------------------------------------------------------
def _snapshot(n_persons=40, **manifest):
    persons = pd.DataFrame({"person_id": [f"p{i}" for i in range(n_persons)], "index_date": "2018-01-01",
                            "index_date_precision": "day", "route": "SAVR", "value_origin": "observed"})
    obs = []
    for i in range(n_persons):
        obs.append({"person_id": f"p{i}", "record_id": f"r{i}", "variable": "egfr", "value": 60.0 + i % 5, "value_text": None,
                    "unit": "mL/min/1.73m2", "date": "2018-01-20", "date_precision": "day", "value_origin": "observed",
                    "parent_record_ids": "[]", "duplicate_status": "unique", "suppressed": False})
        obs.append({"person_id": f"p{i}", "record_id": f"d{i}", "variable": "diabetes", "value": None, "value_text": "true" if i % 3 == 0 else "false",
                    "unit": "", "date": "2018-01-02", "date_precision": "day", "value_origin": "observed", "parent_record_ids": "[]",
                    "duplicate_status": "unique", "suppressed": False})
    m = {"snapshot_id": "snap-test", "source_kind": "real", "purpose": "research", "published": True, "frozen": True, "lineage_complete": True}
    m.update(manifest)
    part = pd.DataFrame({"person_id": persons["person_id"], "role": ["fit"] * (n_persons - 5) + ["holdout"] * 5})
    return Snapshot(SnapshotManifest(**m), persons, pd.DataFrame(obs), part)


def _policy(**kw):
    p = default_policy({})
    p.update({"baseline_variables": ["egfr"], "categorical_variables": ["diabetes"], "bootstrap_replicates": 20, **kw})
    return p


def _summary(profile, variable, statistic):
    return next(s for s in profile.summaries if s["variable"] == variable and s["statistic"] == statistic)


# --- G1: profiling ------------------------------------------------------------------------------------------
def test_one_patient_with_many_labs_cannot_dominate_baseline(pops):
    snap = _snapshot()
    extra = pd.DataFrame([{"person_id": "p0", "record_id": f"x{k}", "variable": "egfr", "value": 500.0, "value_text": None,
                           "unit": "mL/min/1.73m2", "date": "2018-02-01", "date_precision": "day", "value_origin": "observed",
                           "parent_record_ids": "[]", "duplicate_status": "unique", "suppressed": False} for k in range(200)])
    snap.observations = pd.concat([snap.observations, extra], ignore_index=True)
    prof = profile_observed(snap, pops["local_export_like"], "fit", _policy())
    s = _summary(prof, "egfr", "mean")
    assert s["n_patients"] == 35 and s["value"] < 70          # first eligible result per person; 200 extra rows ignored
    assert s["support"] == "fit_local"


def test_holdout_persons_never_inform_the_fit_profile(pops):
    snap = _snapshot()
    snap.observations.loc[snap.observations["person_id"].isin([f"p{i}" for i in range(35, 40)]) &
                          (snap.observations["variable"] == "egfr"), "value"] = 900.0
    prof = profile_observed(snap, pops["local_export_like"], "fit", _policy())
    assert _summary(prof, "egfr", "mean")["value"] < 70 and prof.exclusions["persons_other_partition"] == 5


def test_synthetic_ancestors_defaults_and_missing_lineage(pops):
    snap = _snapshot()
    derived = pd.DataFrame([{"person_id": "p1", "record_id": "syn1", "variable": "egfr", "value": 10.0, "value_text": None, "unit": "",
                             "date": "2018-01-05", "date_precision": "day", "value_origin": "synthetic_default", "parent_record_ids": "[]",
                             "duplicate_status": "unique", "suppressed": False},
                            {"person_id": "p1", "record_id": "der1", "variable": "egfr", "value": 11.0, "value_text": None, "unit": "",
                             "date": "2018-01-06", "date_precision": "day", "value_origin": "derived",
                             "parent_record_ids": json.dumps(["r1", "syn1"]), "duplicate_status": "unique", "suppressed": False}])
    snap.observations = pd.concat([snap.observations, derived], ignore_index=True)
    prof = profile_observed(snap, pops["local_export_like"], "fit", _policy())
    assert prof.exclusions["records_non_observed_ancestor"] == 1 and prof.exclusions["records_non_observed_origin"] == 1
    missing = _snapshot()
    missing.observations.loc[0, "parent_record_ids"] = json.dumps(["does-not-exist"])
    with pytest.raises(SnapshotRejected, match="ancestor"):
        profile_observed(missing, pops["local_export_like"], "fit", _policy())
    with pytest.raises(SnapshotRejected, match="scaffold"):
        profile_observed(_snapshot(source_kind="real_with_synthetic_defaults"), pops["local_export_like"], "fit", _policy())
    with pytest.raises(SnapshotRejected, match="frozen"):
        profile_observed(_snapshot(frozen=False), pops["local_export_like"], "fit", _policy())


def test_suppressed_counts_and_year_only_dates(pops):
    snap = _snapshot(n_persons=12)
    snap.observations.loc[snap.observations["variable"] == "diabetes", "value_text"] = "<5"
    prof = profile_observed(snap, pops["local_export_like"], "fit", _policy())
    assert prof.exclusions["records_suppressed_not_converted"] == 7            # fit-partition persons only
    assert not any(s["variable"] == "diabetes" for s in prof.summaries)
    assert _summary(prof, "egfr", "mean")["support"] == "descriptive_only"      # 7 persons < 30
    yearly = _snapshot()
    rows = []
    for i in range(40):
        for y in (2018, 2019, 2020):
            rows.append({"person_id": f"p{i}", "record_id": f"y{i}{y}", "variable": "ldl", "value": 100.0 - (y - 2018),
                         "value_text": None, "unit": "mg/dL", "date": str(y), "date_precision": "year", "value_origin": "observed",
                         "parent_record_ids": "[]", "duplicate_status": "unique", "suppressed": False})
    yearly.observations = pd.concat([yearly.observations, pd.DataFrame(rows)], ignore_index=True)
    prof = profile_observed(yearly, pops["local_export_like"], "fit", _policy(longitudinal_variables=["ldl"]))
    stats = {s["statistic"] for s in prof.summaries if s["variable"] == "ldl"}
    assert "slope_per_year" not in stats and {"annual_mean:2018", "annual_mean:2020"} <= stats


# --- G2: evidence -------------------------------------------------------------------------------------------
def _target(**kw):
    base = dict(target_id="t1", kind="cumulative_incidence", variable="all_cause_death", estimate=0.3,
                uncertainty={"type": "se", "value": 0.02}, time={"origin": "implantation", "horizon_years": 5.0},
                outcome_definition="all_cause_death", study_group_id="g1", review_status="approved", role="fit",
                acceptance={"tolerance": 0.05, "tolerance_type": "absolute", "weight": 1.0, "mandatory": True})
    base.update(kw)
    return EvidenceTarget(**base).to_dict()


def test_wrong_unit_origin_endpoint_and_estimands_are_rejected(pops):
    pop = pops["synthetic_recovery"]
    cases = {
        "unit": _target(target_id="u", kind="mean", variable="age_at_implant", unit="months"),
        "origin": _target(target_id="o", time={"origin": "randomisation", "horizon_years": 5.0}),
        "endpoint": _target(target_id="e", variable="kairos_adjudicated_moderate_or_severe_svd_v2", outcome_definition="VARC-3 haemodynamic"),
        "fine_gray": _target(target_id="fg", kind="log_effect", effect_type="subdistribution_log_hr"),
        "one_minus_km": _target(target_id="km", kind="km_survival", variable="non_svd_index_valve_replacement",
                                outcome_definition="non_svd_index_valve_replacement"),
        "unreviewed": _target(target_id="un", review_status="proposed"),
        "no_uncertainty": _target(target_id="nu", uncertainty={"type": "none"}),
        "horizon": _target(target_id="h", time={"origin": "implantation", "horizon_years": 15.0}),
        "ok": _target(target_id="ok"),
    }
    val = validate_evidence(EvidenceRegistry(list(cases.values())), pop)
    by = {t["target_id"]: t for t in val.targets}
    assert by["u"]["compatibility"]["status"] == "unit_mismatch" and by["u"]["role"] == "excluded"
    assert by["o"]["compatibility"]["status"] == "time_origin_mismatch"
    assert by["e"]["compatibility"]["status"] == "endpoint_mismatch" and by["e"]["role"] == "sensitivity"
    assert by["fg"]["compatibility"]["status"] == "estimand_mismatch"
    assert by["km"]["compatibility"]["status"] == "estimand_mismatch"
    assert by["un"]["role"] == "excluded" and by["un"]["compatibility"]["status"] == "unreviewed"
    assert by["nu"]["compatibility"]["status"] == "no_uncertainty" and by["nu"]["role"] == "sensitivity"
    assert by["h"]["compatibility"]["status"] == "extrapolation"
    assert by["ok"]["role"] == "fit" and by["ok"]["compatibility"]["status"] == "compatible"


def test_published_rates_candidates_are_never_approved(pops):
    reg = candidates_from_published_rates(get_settings().reference_dir / "published_rates.csv")
    assert reg.targets and all(t["review_status"] == "proposed" for t in reg.targets)
    val = validate_evidence(reg, pops["external_population_notion_like"])
    assert not val.by_role("fit") and not val.by_role("holdout")


def test_overlapping_publications_and_dense_curves_do_not_multiply_weight():
    a = [_target(target_id="a1", study_group_id="notion"), _target(target_id="a2", study_group_id="notion_followup",
                                                                     overlapping_groups=["notion"])]
    w = study_group_weights(a)
    assert abs(sum(w.values()) - 1.0) < 1e-12                  # two overlapping publications share one study weight
    sparse = _target(target_id="s", study_group_id="gs", curve_id="cs", kind="km_survival")
    dense = _target(target_id="d", study_group_id="gd", curve_id="cd", kind="km_survival")
    curves = [{"target_id": "cs", "time": t, "estimate": 1 - 0.05 * t, "lower": None, "upper": None} for t in (1.0, 5.0)] + \
             [{"target_id": "cd", "time": t, "estimate": 1 - 0.05 * t, "lower": None, "upper": None} for t in np.linspace(0.1, 5, 50)]
    from kairos.calibration.evidence import EvidenceValidation

    ts = compile_targets(EvidenceValidation([sparse, dense]), curves)
    by_target = pd.DataFrame([{"target": p.target_id, "w": p.weight} for p in ts.fit]).groupby("target")["w"].sum()
    assert abs(by_target["s"] - by_target["d"]) < 1e-12 and len(ts.fit) == 52


def test_conflicting_targets_fail_visibly():
    a = TargetPoint("a", "a", "g1", "proportion", "diabetes", "proportion:true", 0.20, 0.01, 0.03, 1.0, True)
    b = TargetPoint("b", "b", "g2", "proportion", "diabetes", "proportion:true", 0.45, 0.01, 0.03, 1.0, True)
    assert target_conflicts([a, b]) and not target_conflicts([a, replace(b, observed=0.24)])


def test_curves_validated_not_repaired_and_cif_cannot_be_reconstructed():
    assert validate_curve([{"time": 1.0, "estimate": 0.9}, {"time": 2.0, "estimate": 0.95}], "km_survival")
    assert validate_curve([{"time": 1.0, "estimate": 0.2}, {"time": 2.0, "estimate": 0.1}], "cumulative_incidence")
    assert not validate_curve([{"time": 1.0, "estimate": 0.9}, {"time": 2.0, "estimate": 0.8}], "km_survival")
    with pytest.raises(ValueError, match="cannot be reconstructed"):
        reconstruct_km([(0, 1.0), (1, 0.9)], [(0, 100)], curve_kind="cumulative_incidence")


# --- G3: generator ---------------------------------------------------------------------------------------------
def test_registry_defaults_reproduce_scenario_mode_exactly():
    spec = get_scenario("gradual_stenotic")
    explicit = copy.deepcopy(spec)
    explicit.params["generator"] = {"mode": "scenario", "baseline": copy.deepcopy(BASELINE_DEFAULTS)}
    a, b = generate_cohort(spec, n=150, seed=4), generate_cohort(explicit, n=150, seed=4)
    for t in ("patients", "echoes", "labs", "exposures", "events"):
        assert a.manifest["table_hashes"][t] == b.manifest["table_hashes"][t], t
    with pytest.raises(ValueError, match="positive definite"):
        generator_settings({"generator": {"baseline": {"reference_echo": {"correlation": [[1, 1.2, 0], [1.2, 1, 0], [0, 0, 1]]}}}})


def test_enrolled_view_includes_pre_reference_deaths_and_post_svd_mortality():
    c = generate_cohort(get_scenario("high_competing_mortality"), n=600, seed=9)
    enr = c.truth_enrolled
    assert len(enr) == 600 and (~enr["reference_eligible"]).sum() == c.manifest["exclusions"]["died_or_replaced_before_reference"]
    assert c.manifest["counts_by_view"] == {"attempted": 600, "enrolled": 600, "reference_eligible": len(c.patients)}
    view = study_view(c, "enrolled_implantation_origin")
    pre = view[~view["reference_eligible"]]
    assert len(pre) and pre["t_death"].notna().sum() + pre["t_replacement"].notna().sum() == len(pre)
    both = view[view["t_svd"].notna() & view["t_death"].notna()]
    assert (both["t_death"] >= both["t_svd"]).all()          # deaths after SVD stay in the all-cause mortality view
    elig = study_view(c, "kairos_reference_eligible")
    assert len(elig) == len(c.patients) and (elig["t_end"] >= 0).all()


def test_calibrated_mode_era_support_copula_and_lab_random_effects():
    spec = copy.deepcopy(get_scenario("gradual_stenotic"))
    spec.params["implant_year_range"] = [2024, 2025]
    spec.params["generator"] = {"mode": "calibrated", "baseline": {
        "device": {"enforce_era_support": True},
        "reference_echo": {"correlation": [[1.0, -0.8, -0.6], [-0.8, 1.0, 0.7], [-0.6, 0.7, 1.0]]},
        "labs": {"analytes": {"egfr": {"patient_slope_sd": 3.0}}}}}
    c = generate_cohort(spec, n=500, seed=11)
    assert all(device_supported_in_year(m, y) for m, y in zip(c.patients["canonical_model"], c.patients["implant_year"]))
    assert not device_supported_in_year("Trifecta", 2024)
    assert c.manifest["generator_mode"] == "calibrated" and c.manifest["era_support"]["enforced"]
    assert c.manifest["era_support"]["design_class_redraws"] >= 0 and "Trifecta" not in set(c.patients["canonical_model"])
    ref = c.echoes[c.echoes["is_reference"]]
    assert np.corrcoef(ref["mean_gradient"], ref["eoa"])[0, 1] < -0.3
    base = generate_cohort(get_scenario("gradual_stenotic"), n=500, seed=11)

    def slope_sd(cohort):
        labs = cohort.labs[cohort.labs["analyte"] == "egfr"].merge(cohort.patients[["patient_id", "implant_date"]], on="patient_id")
        labs["t"] = [(d - i).days / 365.25 for d, i in zip(labs["date"], labs["implant_date"])]
        return np.std([np.polyfit(g["t"], g["value"], 1)[0] for _, g in labs.groupby("patient_id") if len(g) >= 3])

    assert slope_sd(c) > slope_sd(base) + 1.0


def test_changing_surveillance_preserves_latent_draws_in_calibrated_mode():
    spec = copy.deepcopy(get_scenario("gradual_stenotic"))
    spec.params["generator"] = {"mode": "calibrated", "baseline": {}}
    other = copy.deepcopy(spec)
    other.params["surveillance"]["p_missed_visit"] = 0.4
    a, b = generate_cohort(spec, n=200, seed=2), generate_cohort(other, n=200, seed=2)
    cols = ["patient_id", "latent_death_date", "initiation_date", "reference_eligible"]
    pd.testing.assert_frame_equal(a.truth_enrolled[cols], b.truth_enrolled[cols])


# --- G4-G6: fitting, bundles, generation ------------------------------------------------------------------------
def test_parameter_transforms_round_trip(defs):
    free = [replace(d, status="free") for d in defs]
    values = {d.name: (d.lower + d.upper) / 2 for d in free}
    back = from_unit(free, to_unit(free, values))
    assert all(abs(back[k] - v) < 1e-9 for k, v in values.items())


@pytest.fixture(scope="module")
def recovery_bundle(pops, defs, tmp_path_factory):
    pop = pops["synthetic_recovery"]
    d = [replace(x, status="free" if x.name == "death_annual_at_79" else "fixed", evidence_links=["recovery:"]) for x in defs]
    truth = {x.name: x.value for x in d}
    truth["death_annual_at_79"] = 0.16
    specs = [{"id": "death_km_4y", "kind": "km_survival", "variable": "all_cause_death", "origin": "implantation", "horizon": 4.0,
              "role": "fit", "group": "g_out"},
             {"id": "death_cif_3y", "kind": "cumulative_incidence", "variable": "all_cause_death", "origin": "reference_echo",
              "horizon": 3.0, "role": "holdout", "group": "g_hold"}]
    reg = synthetic_recovery_registry(pop, d, truth, n_attempted=900, truth_seeds=(9901, 9902), target_specs=specs,
                                      tolerance_se_multiple=4.0)
    val = validate_evidence(reg, pop)
    settings = FitSettings(initial_design_points=6, multistarts=1, max_objective_evaluations=14, attempted_patients_per_evaluation=900,
                           fit_simulation_seeds=(1201,), diagnostic_simulation_seeds=(2201,), finalists=2)
    bundle = fit_generator(val, [], d, pop, {}, settings=settings, uncertainty_replicates=0,
                           parameter_spec_raw={"sensitivity_sets": {"high": {"death_annual_at_79": 0.2}}})
    store = LocalStore(tmp_path_factory.mktemp("cal"))
    return bundle, store, truth, d


def test_recovery_fit_moves_toward_truth_and_reports_statuses(recovery_bundle, defs):
    bundle, _, truth, d = recovery_bundle
    start = next(x.value for x in defs if x.name == "death_annual_at_79")
    fitted = bundle.fitted_values["death_annual_at_79"]
    assert abs(np.log(fitted) - np.log(truth["death_annual_at_79"])) < abs(np.log(start) - np.log(truth["death_annual_at_79"]))
    assert bundle.fit_status in ("accepted_for_simulation", "failed_targets", "nonidentifiable")
    assert bundle.structural_status == "passed" and bundle.independent_validation_status in ("passed", "failed")
    assert not set(bundle.residuals["target_id"]) & {"recovery:death_cif_3y"}     # the holdout never entered the fit


def test_bundle_is_immutable_checksummed_and_generation_is_guarded(recovery_bundle, pops):
    bundle, store, _, _ = recovery_bundle
    manifest = save_bundle(store, bundle)
    with pytest.raises(FileExistsError):
        save_bundle(store, bundle)
    loaded = load_bundle(store, bundle.bundle_id)
    assert loaded.fitted_values == bundle.fitted_values and manifest["checksums"]
    store.put_bytes("calibration", f"{bundle.bundle_id}/report.md", b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(store, bundle.bundle_id)
    rejected = replace(bundle, fit_status="failed_targets", bundle_id=bundle.bundle_id + "x")
    with pytest.raises(BundleNotAccepted):
        generate_calibrated(rejected, 100, 1)
    accepted = replace(bundle, fit_status="accepted_for_simulation")
    c = generate_calibrated(accepted, 150, 3201, "sensitivity_high" if any(s["parameter_set_id"] == "sensitivity_high"
                                                                             for s in accepted.parameter_sets()) else "fit")
    assert c.manifest["generation_kind"] == "calibrated" and c.manifest["source_kind"] == "synthetic"
    assert "clinical predictive validity unestablished" in c.manifest["label"]
    assert calibrated_prefix("b", "quick", "r1") != calibrated_prefix("b", "full", "r1")


def test_scaffold_backend_never_falls_back(tmp_path):
    store = LocalStore(tmp_path)
    assert callable(resolve_scaffold_generator({"training_export": {"defaults": {"generator_backend": "scenario"}}}, store))
    with pytest.raises(ScaffoldBackendError, match="no fallback"):
        resolve_scaffold_generator({"training_export": {"defaults": {"generator_backend": "calibrated_bundle"}}}, store)
    with pytest.raises(ScaffoldBackendError, match="cannot be loaded"):
        resolve_scaffold_generator({"training_export": {"defaults": {"generator_backend": "calibrated_bundle",
                                                                     "calibration_bundle_manifest": "calibration/cal-missing/manifest.json"}}}, store)


def test_no_evidence_means_insufficient_evidence_not_a_fit(pops, defs):
    pop = pops["external_population_notion_like"]
    reg = candidates_from_published_rates(get_settings().reference_dir / "published_rates.csv")
    val = validate_evidence(reg, pop)
    bundle = fit_generator(val, [], defs, pop, {}, settings=FitSettings(max_objective_evaluations=2, attempted_patients_per_evaluation=50))
    assert bundle.fit_status == "insufficient_evidence" and bundle.structural_status == "not_run"
    assert not bundle.allowed_purposes and len(bundle.excluded_targets) == len(reg.targets)
    assert date.fromisoformat(pop.study_end)
