"""Scoring the synthetic-trained implant-time model on the real extract, exercised on the fabricated
note fixtures (no real patient data)."""
import numpy as np
import pandas as pd
import pytest

from kairos.evaluation.real_extract import implant_time_risk, real_outcomes, score_extract
from kairos.passport import build_passport, prepare_notes
from kairos.simulation.generators import generate_cohort
from kairos.simulation.scenarios import get_scenario
from tests.test_passport import FIXTURE_MANIFEST, FIXTURES_DIR


@pytest.fixture(scope="module")
def extract():
    rows = [{"Profile Key": pid, "Type": typ, "Signed Status": status, "Service Date": year,
             "Notes": (FIXTURES_DIR / fname).read_text(encoding="utf-8")}
            for fname, pid, typ, year, status in FIXTURE_MANIFEST]
    passport, note_rows = build_passport(prepare_notes(pd.DataFrame(rows)))
    return passport, note_rows


def test_valve_in_valve_after_the_index_implant_is_an_event(extract):
    outcomes, excluded = real_outcomes(*extract)
    row = outcomes.set_index("patient_id").loc["SYN_011"]
    assert bool(row["event"]) and row["time"] == 6.0 and row["route"] == "SAVR"
    # single-note fixtures have no later note year and are excluded, and counted
    assert excluded["no_note_after_implant_year"] > 0
    assert set(outcomes["route"]) <= {"SAVR", "TAVR"}


def test_scorecard_is_aggregate_only():
    out = pd.DataFrame({"patient_id": [f"P{i}" for i in range(40)], "route": ["SAVR", "TAVR"] * 20,
                        "time": np.arange(1, 41) % 9 + 1.0, "event": [i % 5 == 0 for i in range(40)]})
    risk = np.where(out["event"], 0.2, 0.05) + np.linspace(0, 0.01, 40)
    card = score_extract(out, risk, n_boot=50)
    assert card["patients_scored"] == 40 and card["events"] == 8
    assert card["c_model"] > 0.9 and len(card["c_model_ci95"]) == 2
    assert "descriptive only" in card["status"]
    assert not any(isinstance(v, (list, dict)) and "P0" in str(v) for v in card.values())


def test_implant_time_risk_uses_only_route_and_design_class(model_cfg):
    cohort = generate_cohort(get_scenario("gradual_stenotic"), n=2500, seed=11)
    pts = pd.DataFrame({"route": ["SAVR", "SAVR", "TAVR"],
                        "design_class": ["stented porcine", "stented porcine", "balloon-expandable intra-annular TAVR"]})
    r = implant_time_risk(cohort, model_cfg, pts)
    assert np.isfinite(r).all() and (r > 0).all() and (r < 1).all()
    assert r[0] == pytest.approx(r[1])
