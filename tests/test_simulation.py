"""Scenario configuration and the synthetic generator."""
import numpy as np

from kairos.simulation.generators import cohort_prefix, generate_cohort
from kairos.simulation.scenarios import (
    LABELS,
    SCENARIO_ORDER,
    get_scenario,
    list_scenarios,
    parameter_table,
    simulation_config_hash,
)


def test_six_scenarios_with_labelled_parameters():
    keys = list_scenarios()
    assert [k for k, _ in keys if _ is None] + sorted({k for k, v in keys if v}) == [s for s in SCENARIO_ORDER if s != "biomarker_information"] + ["biomarker_information"]
    assert len({k for k, _ in keys}) == 6
    for name, variant in keys:
        spec = get_scenario(name, variant)
        rows = parameter_table(spec)
        assert rows and all(r["label"] in LABELS for r in rows)
        assert spec.description and spec.config_hash == simulation_config_hash()


def test_variant_overrides_apply():
    weak = get_scenario("biomarker_information", "weak")
    absent = get_scenario("biomarker_information", "absent")
    assert weak.get("svd_hazard.biomarker_effect_multiplier") == 0.3
    assert absent.get("svd_hazard.biomarker_effect_multiplier") == 0.0
    unmeasured = get_scenario("biomarker_information", "unmeasured")
    assert unmeasured.get("biomarkers.measured_fraction")["lpa"] < 0.05
    hm = get_scenario("high_competing_mortality")
    assert hm.get("death_hazard.multiplier") == 2.5 and hm.get("p_tavr") == 0.9


def test_generator_is_reproducible_and_consistent():
    spec = get_scenario("gradual_stenotic")
    a = generate_cohort(spec, n=120, seed=3)
    b = generate_cohort(spec, n=120, seed=3)
    assert a.patients.equals(b.patients) and a.echoes.equals(b.echoes)
    assert a.manifest["counts"]["patients"] == len(a.patients) <= 120
    assert a.manifest["label"].startswith("synthetic")
    assert set(a.events["patient_id"]) == set(a.patients["patient_id"])
    first = a.echoes.groupby("patient_id")["date"].min()
    ref = a.patients.set_index("patient_id")["reference_date"]
    assert (first == ref.reindex(first.index)).all()
    lag = (a.patients["reference_date"] - a.patients["implant_date"]).map(lambda d: d.days)
    assert lag.between(30, 180).all()
    adj = a.events.dropna(subset=["svd_adjudicated_date"])
    assert (adj["svd_adjudicated_date"] > adj["reference_date"]).all()
    assert cohort_prefix(spec, 3, 120, "quick").startswith("gradual_stenotic/default/quick/")


def test_regurgitant_scenario_produces_regurgitant_phenotypes():
    c = generate_cohort(get_scenario("regurgitant_abrupt"), n=200, seed=5)
    assert (c.truth["phenotype_latent"] == "regurgitant").mean() > 0.3
    d = generate_cohort(get_scenario("gradual_stenotic"), n=200, seed=5)
    assert (d.truth["phenotype_latent"] == "stenotic").all()


def test_anticoagulant_scenario_has_post_suspicion_starts():
    c = generate_cohort(get_scenario("anticoagulant_mechanism_confounding", "marker_mediated"), n=600, seed=11)
    assert c.exposures["post_suspicion"].any()
    assert np.isin(c.exposures["class"].unique(), ["VKA", "FXa", "DTI", "SAPT", "DAPT", "none"]).all()
