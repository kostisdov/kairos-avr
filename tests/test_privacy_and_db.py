"""Privacy scan, database round trips and the guideline lookup."""
from pathlib import Path

from kairos.extraction.schema import (
    ExposureEpisode,
    ExposureTimeline,
    Messages,
    Passport,
    PassportSource,
    Prediction,
    Reliability,
)
from kairos.io.config import Settings
from kairos.io.db import get_repository
from kairos.modelling.modules import guideline_interval_months, load_model_config
from kairos.privacy import main as privacy_main
from kairos.privacy import scan


def test_privacy_scan_flags_identifiers_spreadsheets_and_private_dirs(tmp_path: Path):
    (tmp_path / "data" / "reference").mkdir(parents=True)
    (tmp_path / "data" / "reference" / "ok.csv").write_text("canonical_model,n\nTrifecta,21\n", encoding="utf-8")
    assert scan(tmp_path) == []
    fake_id = "Patient" + "_" + "007"  # built at run time so this file never matches the scan itself
    (tmp_path / "data" / "reference" / "bad.csv").write_text(f"patient_id\n{fake_id}\n", encoding="utf-8")
    (tmp_path / "notes.xlsx").write_bytes(b"x")
    (tmp_path / "data" / "derived" / "private").mkdir(parents=True)
    (tmp_path / "data" / "derived" / "private" / "passport.csv").write_text("a\n", encoding="utf-8")
    rules = {f.rule for f in scan(tmp_path)}
    assert {"patient-id", "spreadsheet", "forbidden-path"} <= rules
    assert privacy_main(["--root", str(tmp_path)]) == 1
    assert privacy_main(["--root", str(tmp_path / "data" / "reference")]) == 1


def test_privacy_scan_reads_notebook_outputs(tmp_path: Path):
    import json
    nb_dir = tmp_path / "notebooks"
    nb_dir.mkdir()

    def notebook(text: str) -> str:
        return json.dumps({"cells": [{"cell_type": "code", "source": ["print(rows)"],
                                      "outputs": [{"output_type": "stream", "text": [text]}]}]})

    (nb_dir / "clean.ipynb").write_text(notebook("patients : 2390\n"), encoding="utf-8")
    assert scan(tmp_path) == []
    (nb_dir / "leak.ipynb").write_text(notebook("Profile Key  Service Date\n" + "Patient" + "_" + "012 2019\n"),
                                       encoding="utf-8")
    rules = {f.rule for f in scan(tmp_path)}
    assert {"patient-id", "source-column"} <= rules


def test_repository_round_trip():
    repo = get_repository(Settings(KAIROS_SQLITE_PATH=":memory:"))
    p = Passport(source=PassportSource(note_ref="n", note_type="operative", date="2016-01-01"), route="TAVR", canonical_model="SAPIEN 3")
    repo.save_passport(p, "synthetic", ["rule"], "0.1.0+t")
    repo.save_exposures(ExposureTimeline(passport_id=p.passport_id, episodes=[ExposureEpisode(**{"class": "FXa", "indication": "AF", "start": "2016-01-01"})]))
    pred = Prediction(passport_id=p.passport_id, prediction_time="2017-01-01", p_svd_before_death=[0.01, 0.03, 0.05],
                      p_death_before_svd=[0.05, 0.15, 0.25], p_alive_intact=[0.94, 0.82, 0.70], p_replaced_non_svd=[0, 0, 0],
                      p_svd_12m=0.01, reliability=Reliability(device_evidence="model-level", data_completeness=0.9, stale_echo=False),
                      messages=Messages(current_abnormality=False, earlier_assessment=False, overdue_surveillance=False),
                      model_version="0.1.0+t", scenario_set="s")
    repo.save_prediction(pred, "synthetic")
    assert repo.get_exposures(p.passport_id).episodes[0].class_ == "FXa"
    assert repo.list_predictions(p.passport_id)[0].p_svd_12m == 0.01
    repo.start_run("r1", "train")
    repo.finish_run("r1", "succeeded", {"ok": True})
    assert repo.list_runs()[0]["status"] == "succeeded"


def test_guideline_intervals():
    cfg = load_model_config()
    assert guideline_interval_months(cfg, "ESC_EACTS", "SAVR", 3.0)[0] == 12
    assert guideline_interval_months(cfg, "ACC_AHA", "TAVR", 3.0)[0] == 12
    assert guideline_interval_months(cfg, "ACC_AHA", "SAVR", 11.0)[0] == 12
    assert guideline_interval_months(cfg, "ACC_AHA", "SAVR", 2.0)[0] > 12
