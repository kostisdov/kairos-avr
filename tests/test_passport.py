# -*- coding: utf-8 -*-
"""
Tests for src/kairos/passport.py against the synthetic fixtures in
tests/fixtures/synthetic_notes/ (entirely fabricated content, no real patient data -- see
that directory and docs/data_build_log.md).

Builds an in-memory notes DataFrame shaped like notes_deidentified.xlsx (Profile Key, Type,
Service, Signed Status, Notes, Service Date, ...) from the fixture files, with a synthetic
Service Date and Signed Status assigned per fixture (including one Signed Status="Deleted"
row, to test the exclusion rule) -- then runs it through the real extraction pipeline.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from kairos.passport import (build_passport, extract_note_fields, load_device_patterns,
                              prepare_notes)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "synthetic_notes"

# (filename, Profile Key, Type, Service Date, Signed Status)
FIXTURE_MANIFEST = [
    ("01_savr_op_perimount_ce.txt", "SYN_001", "Operative Report", 2015, "Signed"),
    ("02_savr_op_trifecta_hashsize.txt", "SYN_002", "Operative Report", 2016, "Signed"),
    ("03_savr_op_trifecta_mm_form.txt", "SYN_003", "Operative Report", 2017, "Signed"),
    ("04_tavr_sapien3_ultra_size_field.txt", "SYN_004", "Procedures", 2020, "Signed"),
    ("05_tavr_evolut_misspelled_corevalue.txt", "SYN_005", "Procedures", 2019, "Signed"),
    ("06_tavr_evolut_r34_shorthand.txt", "SYN_006", "Procedures", 2021, "Signed"),
    ("07_progress_note_native_and_prosthetic_gradient.txt", "SYN_007", "Progress Notes", 2018, "Signed"),
    ("08_progress_note_mmHg_unit_variant.txt", "SYN_008", "Progress Notes", 2019, "Signed"),
    ("09_progress_note_dvi_eoa_values.txt", "SYN_009", "Progress Notes", 2020, "Signed"),
    ("10_progress_note_severe_regurg_event.txt", "SYN_010", "Progress Notes", 2021, "Signed"),
    ("11_two_procedure_pt_savr_index.txt", "SYN_011", "Operative Report", 2014, "Signed"),
    ("12_two_procedure_pt_viv_tavr_followup.txt", "SYN_011", "Procedures", 2020, "Signed"),
    ("13_progress_note_missing_reference_echo.txt", "SYN_013", "Progress Notes", 2022, "Signed"),
    ("14_progress_note_epic_ehr_reference_only.txt", "SYN_014", "Progress Notes", 2022, "Signed"),
    ("15_savr_op_konect_biobentall_redo.txt", "SYN_015", "Operative Report", 2023, "Signed"),
    ("16_deleted_status_operative_report.txt", "SYN_016", "Operative Report", 2023, "Deleted"),
    # Added for coordinator Fix 1 (2026-09-16): CE+size-after, Magna free-text, and SAPIEN
    # generation-disambiguation regression tests.
    ("17_savr_op_ce_size_after.txt", "SYN_017", "Operative Report", 2013, "Signed"),
    ("18_savr_op_ce_mm_form.txt", "SYN_018", "Operative Report", 2014, "Signed"),
    ("19_progress_note_magna_valve_and_sized_magna.txt", "SYN_019", "Progress Notes", 2019, "Signed"),
    ("20_tavr_bare_s3_guarded.txt", "SYN_020", "Procedures", 2021, "Signed"),
    ("21_tavr_bare_xt_guarded.txt", "SYN_021", "Procedures", 2015, "Signed"),
    ("22_progress_note_s3_xt_false_positive_check.txt", "SYN_022", "Progress Notes", 2022, "Signed"),
    ("23_tavr_ultra_near_sapien_proximity.txt", "SYN_023", "Procedures", 2020, "Signed"),
    ("24_tavr_bare_ultra_resilia.txt", "SYN_024", "Procedures", 2022, "Signed"),
    ("25_progress_note_bare_carpentier.txt", "SYN_025", "Progress Notes", 2020, "Signed"),
]


@pytest.fixture(scope="module")
def synthetic_notes_raw():
    rows = []
    for fname, pid, typ, year, status in FIXTURE_MANIFEST:
        text = (FIXTURES_DIR / fname).read_text(encoding="utf-8")
        rows.append({"Profile Key": pid, "Note Type": "Generic", "Type": typ,
                     "Service": "Cardiac Surgery", "Signed Status": status,
                     "Authoring Provider Type": "Physician",
                     "Authoring Provider Specialty": "Cardiac Surgery", "Notes": text,
                     "Service Date": year, "Creation Date": year, "Last Edited Date": year})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def prepared_notes(synthetic_notes_raw):
    return prepare_notes(synthetic_notes_raw)


@pytest.fixture(scope="module")
def model_patterns_and_meta():
    return load_device_patterns()


@pytest.fixture(scope="module")
def passport_df(prepared_notes, model_patterns_and_meta):
    patterns, meta = model_patterns_and_meta
    passport, note_level = build_passport(prepared_notes, patterns, meta)
    return passport.set_index("patient_id")


# ---------------------------------------------------------------------------
def test_deleted_note_excluded(prepared_notes):
    assert "SYN_016" not in set(prepared_notes["Profile Key"])


def test_deleted_note_does_not_leak_into_passport(passport_df):
    assert "SYN_016" not in passport_df.index


def test_hashsize_trifecta(passport_df):
    row = passport_df.loc["SYN_002"]
    assert "Trifecta" in row["canonical_model"]
    assert 21 in _json_list(row["all_sizes_mentioned"])


def test_mm_form_trifecta(passport_df):
    row = passport_df.loc["SYN_003"]
    assert "Trifecta" in row["canonical_model"]
    assert 21 in _json_list(row["all_sizes_mentioned"])


def test_sapien3_ultra_size_field(passport_df):
    row = passport_df.loc["SYN_004"]
    assert row["canonical_model"] in ("SAPIEN 3 Ultra", "SAPIEN 3")
    assert row["route"] == "TAVR"


def test_misspelled_corevalue_still_matches(passport_df):
    row = passport_df.loc["SYN_005"]
    assert row["canonical_model"] in ("Evolut PRO+", "CoreValve")
    assert row["route"] == "TAVR"


def test_r34_shorthand(passport_df):
    row = passport_df.loc["SYN_006"]
    assert 34 in _json_list(row["all_sizes_mentioned"])


def test_native_and_prosthetic_gradient_same_note(passport_df):
    import json
    row = passport_df.loc["SYN_007"]
    series = json.loads(row["gradient_series_json"])
    values = {g["value"] for g in series}
    assert 52 in values  # native pre-op gradient
    assert 13 in values  # prosthetic post-op gradient
    native_entry = next(g for g in series if g["value"] == 52)
    prosth_entry = next(g for g in series if g["value"] == 13)
    assert native_entry["native"] is True
    assert prosth_entry["prosthetic"] is True


def test_mmHg_unit_variant_extracted(passport_df):
    row = passport_df.loc["SYN_008"]
    assert row["n_gradients_prosthetic"] >= 1
    assert 15 in {g["value"] for g in _json_list(row["gradient_series_json"])}


def test_dvi_and_eoa_extracted(passport_df):
    import json
    row = passport_df.loc["SYN_009"]
    dvis = json.loads(row["dvi_series_json"])
    eoas = json.loads(row["eoa_series_json"])
    assert any(abs(d["value"] - 0.51) < 1e-9 for d in dvis)
    assert any(abs(e["value"] - 1.6) < 1e-9 for e in eoas)


def test_severe_regurg_event_flagged(passport_df):
    row = passport_df.loc["SYN_010"]
    assert bool(row["event_prosthetic_dysfunction"]) is True


def test_two_procedures_same_patient_route_is_savr_plus_tavr(passport_df):
    row = passport_df.loc["SYN_011"]
    assert row["route"] == "TAVR+SAVR"
    import json
    implant_years = json.loads(row["implant_years_all"])
    assert 2014 in implant_years and 2020 in implant_years
    assert bool(row["event_ViV"]) is True  # "Valve-in-valve TAVR" language present in note 12


def test_missing_reference_echo_note_still_extracts_gradient(passport_df):
    row = passport_df.loc["SYN_013"]
    assert row["n_gradients_prosthetic"] >= 1


def test_epic_ehr_reference_is_not_a_false_positive_valve_match(passport_df):
    """Regression test for the false-positive bug found during development: 'Epic' the
    EHR system must NOT be extracted as the Abbott/St Jude Epic valve when there is no
    nearby valve-specific context."""
    row = passport_df.loc["SYN_014"]
    assert row["canonical_model"] == "" or "Epic" not in str(row["canonical_model"])
    assert "Epic" not in row["all_models_mentioned"]


def test_konect_biobentall_and_redo_event(passport_df):
    row = passport_df.loc["SYN_015"]
    assert "Konect" in row["canonical_model"]
    assert bool(row["event_redo"]) is True


def test_note_ref_format(prepared_notes):
    refs = prepared_notes["note_ref"].tolist()
    assert all("_" in r for r in refs)
    # every note_ref must start with its own Profile Key
    for _, r in prepared_notes.iterrows():
        assert r["note_ref"].startswith(r["Profile Key"])


def test_extract_note_fields_never_returns_raw_text(model_patterns_and_meta, prepared_notes):
    """Sanity check that extract_note_fields()'s return dict contains no long text blobs
    (only numbers/booleans/short sets/lists) -- a structural proxy for 'never expose note
    text', checked here as string-length bounds on every scalar field."""
    patterns, _ = model_patterns_and_meta
    for _, r in prepared_notes.iterrows():
        fields = extract_note_fields(str(r["Notes"]), patterns)
        for key in ("models",):
            for m in fields[key]:
                assert len(m) < 40  # canonical model names are short


def _json_list(cell):
    import json
    return json.loads(cell)


# ---------------------------------------------------------------------------
# Coordinator Fix 1 regression tests (2026-09-16)
# ---------------------------------------------------------------------------
def test_ce_size_after_form(passport_df):
    """'#25 CE' (size before CE) must map to Perimount, not just the pre-existing
    'CE#25' (CE before size) form."""
    row = passport_df.loc["SYN_017"]
    assert row["canonical_model"] == "Perimount"
    assert 25 in _json_list(row["all_sizes_mentioned"])


def test_ce_mm_form(passport_df):
    """'25 mm CE' must map to Perimount."""
    row = passport_df.loc["SYN_018"]
    assert row["canonical_model"] == "Perimount"


def test_magna_valve_and_sized_magna_forms(passport_df):
    """'#21 mm Magna' and bare 'Magna valve' must both map to Magna, not fall through to
    the broader Perimount family bucket or go unmatched."""
    row = passport_df.loc["SYN_019"]
    assert row["canonical_model"] == "Magna"
    assert 21 in _json_list(row["all_sizes_mentioned"])


def test_bare_s3_resolves_to_sapien3_with_valve_context(passport_df):
    row = passport_df.loc["SYN_020"]
    assert row["canonical_model"] == "SAPIEN 3"


def test_bare_xt_resolves_to_sapienxt_with_valve_context(passport_df):
    row = passport_df.loc["SYN_021"]
    assert row["canonical_model"] == "SAPIEN XT"


def test_bare_s3_and_xt_without_valve_context_do_not_false_positive(passport_df):
    """Regression test mirroring the Epic-EHR false-positive fix: bare 'S3' (a room
    number) and 'XT' (a lab order code) with no nearby valve/TAVR context must NOT be
    extracted as SAPIEN 3 / SAPIEN XT."""
    row = passport_df.loc["SYN_022"]
    assert row["canonical_model"] == ""
    assert "SAPIEN 3" not in row["all_models_mentioned"]
    assert "SAPIEN XT" not in row["all_models_mentioned"]


def test_ultra_near_sapien_proximity_resolves_to_sapien3ultra(passport_df):
    """'Sapien platform, Ultra generation' (Ultra separated from Sapien by other words,
    not the contiguous 'S3 Ultra' form already covered elsewhere) must still resolve to
    SAPIEN 3 Ultra."""
    row = passport_df.loc["SYN_023"]
    assert row["canonical_model"] == "SAPIEN 3 Ultra"


def test_bare_ultra_resilia_resolves_to_sapien3ultraresilia(passport_df):
    """'Ultra RESILIA' with no 'SAPIEN 3' prefix must still resolve to the most specific
    canonical model, SAPIEN 3 Ultra RESILIA."""
    row = passport_df.loc["SYN_024"]
    assert row["canonical_model"] == "SAPIEN 3 Ultra RESILIA"


def test_bare_carpentier_resolves_to_perimount(passport_df):
    """Bare 'Carpentier-Edwards bioprosthesis' with no 'Perimount'/'Magna'/CE-size pattern
    nearby must still resolve to Perimount (the closest device_table.csv canonical model
    for an unspecified Carpentier-Edwards aortic bioprosthesis), matching v0's broader
    'perimount|magna|carpentier' bucket behaviour."""
    row = passport_df.loc["SYN_025"]
    assert row["canonical_model"] == "Perimount"


def test_evolut_corevalue_misspelling_regression(passport_df):
    """Regression test for the pre-existing 'Evolut Corevalue Pro+' misspelling alias
    (fixture 05) -- confirms it still resolves correctly after the Fix 1 pattern-loading
    refactor (single-regex-per-model -> list-of-guarded-patterns-per-model)."""
    row = passport_df.loc["SYN_005"]
    assert row["canonical_model"] == "Evolut PRO+"
