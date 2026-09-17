"""Generator version 2: dataset identity, grade mapping, random streams, truth separation and the
shared adjudication policy (detailed design WP-A1 to WP-A4)."""
import copy

import numpy as np
import pandas as pd
import pytest

from kairos.adjudication.framework import (
    EchoPoint,
    adjudicate,
    stage_point,
    thrombosis_windows_from_exposures,
    to_points,
)
from kairos.extraction.schema import EchoObservation, ExposureEpisode
from kairos.grades import REGURG_ORDINAL, regurg_label, regurg_ordinal
from kairos.simulation.generators import cohort_prefix, generate_cohort
from kairos.simulation.scenarios import get_scenario
from kairos.simulation.truth import attendance_by_year, truth_columns_in


def _with(spec, **surveillance):
    s = copy.deepcopy(spec)
    s.params["surveillance"].update(surveillance)
    return s


@pytest.fixture(scope="module")
def base():
    return generate_cohort(get_scenario("gradual_stenotic"), n=400, seed=21)


# --- WP-A1: identity --------------------------------------------------------------------------
def test_prefix_separates_size_and_namespace():
    spec = get_scenario("gradual_stenotic")
    full = cohort_prefix(spec, 20260916)
    assert full.startswith("gradual_stenotic/default/full/")
    assert cohort_prefix(spec, 20260916, 600, "quick") != cohort_prefix(spec, 20260916, 600, "full")
    assert cohort_prefix(spec, 20260916, 600, "full") != full        # same seed, different size
    assert cohort_prefix(spec, 20260916, spec.get("n_patients")) == full
    assert get_scenario("gradual_stenotic").effective_config_hash != get_scenario("regurgitant_abrupt").effective_config_hash
    with pytest.raises(ValueError):
        cohort_prefix(spec, 1, 10, "scratch")


def test_manifest_v2_is_traceable_and_reproducible(base):
    m = base.manifest
    assert m["schema_version"] == 2 and m["generator_version"] == "2.1" and m["endpoint_version"] == "2"
    assert m["code_revision"]["kind"] in ("git", "source_tree") and m["code_revision"]["value"] != "nogit"
    assert m["n_requested"] == 400 and m["n_retained"] == len(base.patients)
    assert m["n_retained"] + m["exclusions"]["died_or_replaced_before_reference"] == 400
    assert set(m["table_hashes"]) == {"patients", "echoes", "labs", "exposures", "events", "truth", "truth_visits", "truth_enrolled"}
    assert all(m["reference_hashes"].values())
    again = generate_cohort(get_scenario("gradual_stenotic"), n=400, seed=21)
    assert again.manifest["table_hashes"] == m["table_hashes"]


# --- WP-A2: one grade mapping -----------------------------------------------------------------
def test_one_regurgitation_mapping_everywhere(base):
    assert REGURG_ORDINAL["trace"] == 0 and REGURG_ORDINAL["mild"] == 1 and REGURG_ORDINAL["severe"] == 3
    assert regurg_ordinal(None) is None and regurg_ordinal(float("nan")) is None and regurg_ordinal("unknown") is None
    assert regurg_label(2) == "moderate"
    assert set(base.echoes["ar_grade"]) <= {"none", "trace", "mild", "moderate", "severe"}
    ref = EchoPoint(date=pd.Timestamp("2016-01-01").date(), mean_gradient=10, eoa=1.8, dvi=0.5, ar_ordinal=regurg_ordinal("mild"))
    fu = EchoPoint(date=pd.Timestamp("2018-01-01").date(), mean_gradient=10, eoa=1.8, dvi=0.5, ar_ordinal=regurg_ordinal("moderate"))
    assert stage_point(fu, ref).stage == "2"   # mild -> moderate is new moderate regurgitation
    obs = EchoObservation(passport_id="x", date="2018-01-01", mean_gradient_mmhg=10, eoa_cm2=1.8, ar_grade="moderate",
                          native_vs_prosthetic="prosthetic", source="manual")
    assert to_points([obs])[0].ar_ordinal == 2


# --- WP-A3: shared adjudication -----------------------------------------------------------------
def test_generator_labels_agree_with_the_live_adjudicator(base):
    ev = base.events.set_index("patient_id")
    checked = 0
    for pid, ech in base.echoes.groupby("patient_id"):
        e = ev.loc[pid]
        if e["svd_status"] == "confirmed_by_terminal_event":
            continue  # the live path has no death date to confirm with
        obs = [EchoObservation(passport_id=pid, date=r.date.isoformat(), mean_gradient_mmhg=float(r.mean_gradient),
                               eoa_cm2=float(r.eoa), dvi=float(r.dvi), ar_grade=r.ar_grade, native_vs_prosthetic="prosthetic",
                               source="manual") for r in ech.sort_values("date").itertuples()]
        pts = to_points(obs)
        exp = base.exposures[base.exposures.patient_id == pid]
        eps = [ExposureEpisode(**{"class": r["class"], "indication": r["indication"], "start": r["start_date"].isoformat(),
                                  "stop": r["stop_date"].isoformat() if pd.notna(r["stop_date"]) else None})
               for _, r in exp.iterrows()]
        ref = pts[0]
        d = adjudicate(pts, ref, thrombosis_windows=thrombosis_windows_from_exposures(eps, pts, ref))
        assert d.date == e["svd_candidate_date"], pid
        assert (d.met and d.confidence == "high") == (e["svd_status"] == "confirmed"), pid
        checked += 1
    assert checked > 300 and (base.events["svd_status"] == "confirmed").any()


def test_unresolved_candidates_are_not_training_events(base):
    ev = base.events
    assert (ev.loc[ev["svd_status"] == "unconfirmed_pending", "svd_adjudicated_date"].isna()).all()
    assert (ev.loc[ev["svd_status"] == "uncertain_transient", "svd_adjudicated_date"].isna()).all()
    confirmed = ev[ev["svd_status"].str.startswith("confirmed")]
    assert (confirmed["svd_adjudicated_date"] == confirmed["svd_candidate_date"]).all()
    assert (confirmed["svd_confirmation_date"] > confirmed["svd_candidate_date"]).all()
    term = ev[ev["svd_status"] == "confirmed_by_terminal_event"]
    assert (term["end_reason"].isin(["death", "replacement"])).all()


# --- WP-A4: truth separation and streams -----------------------------------------------------------
def test_truth_never_in_predictor_tables(base):
    assert truth_columns_in(base.patients, base.echoes, base.labs, base.exposures, base.events) == set()
    assert {"phenotype_latent", "threshold_crossing_date", "initiation_date"} <= set(base.truth.columns)
    t = base.truth.dropna(subset=["threshold_crossing_date", "initiation_date"])
    assert (t["threshold_crossing_date"] >= t["initiation_date"]).all()


def test_attendance_denominator_excludes_dead_or_replaced(base):
    tv = base.truth_visits.merge(base.events[["patient_id", "end_followup_date"]], on="patient_id")
    assert (tv["scheduled_date"] <= tv["end_followup_date"]).all()
    att = attendance_by_year(base.truth_visits)
    assert att["attendance"].between(0, 1).all() and att["eligible_visits"].sum() == (base.truth_visits["visit_index"] > 0).sum()


def test_visit_mechanism_does_not_change_latent_disease():
    spec = get_scenario("gradual_stenotic")
    regular = generate_cohort(spec, n=300, seed=33)
    informative = generate_cohort(_with(spec, p_missed_visit=0.3, informative=True), n=300, seed=33)
    latent = ["patient_id", "phenotype_latent", "initiation_date", "threshold_crossing_date", "n_scheduled_visits",
              "thrombosis_episodes"]
    pd.testing.assert_frame_equal(regular.truth[latent], informative.truth[latent])
    cols = ["patient_id", "scheduled_date", "true_mean_gradient", "true_eoa", "true_dvi", "true_ar_grade"]
    pd.testing.assert_frame_equal(regular.truth_visits[cols], informative.truth_visits[cols])
    common = regular.echoes.merge(informative.echoes, on=["patient_id", "date"], suffixes=("_r", "_i"))
    assert len(common) > 500
    for c in ("mean_gradient", "eoa", "dvi", "ar_grade", "lvef", "svi"):
        assert (common[f"{c}_r"] == common[f"{c}_i"]).all(), c
    assert regular.manifest["five_year_attendance"] > informative.manifest["five_year_attendance"]


def test_forced_visit_loss_delays_detection_without_changing_trajectories():
    spec = get_scenario("gradual_stenotic")
    base = generate_cohort(spec, n=600, seed=44)
    gap = generate_cohort(_with(spec, forced_missed_years=[3.0, 6.0]), n=600, seed=44)
    pd.testing.assert_frame_equal(base.truth[["patient_id", "threshold_crossing_date"]], gap.truth[["patient_id", "threshold_crossing_date"]])
    years = gap.truth_visits.query("visit_index > 0 and years_since_implant >= 3 and years_since_implant < 6")
    assert len(years) > 0 and not years["attended"].any()
    d_base = base.truth["detection_delay_days"].dropna()
    d_gap = gap.truth["detection_delay_days"].dropna()
    assert d_gap.mean() > d_base.mean()
    assert gap.truth["missed_crossing"].sum() >= base.truth["missed_crossing"].sum()
    implant = gap.events.set_index("patient_id")["implant_date"]
    cand = gap.events.dropna(subset=["svd_candidate_date"]).set_index("patient_id")["svd_candidate_date"]
    yrs = np.array([(c - implant[p]).days / 365.25 for p, c in cand.items()])
    assert not ((yrs >= 3.01) & (yrs < 5.99)).any()   # no study, so no candidate, inside the gap
