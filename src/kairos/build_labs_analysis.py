# -*- coding: utf-8 -*-
"""
Task 6, labs bullet. From labs_deidentified.xlsx:
  1. For warfarin patients (identified via medications_deidentified.xlsx, Simple Generic
     Name containing "warfarin" specifically -- not the broader VKA class, per the task's
     literal wording), the share of INR results in the 2.0-3.0 therapeutic range, per
     patient-year (PRIVATE -- patient-level).
  2. Per-patient-year availability (present/absent count only, no values) of creatinine or
     eGFR, HbA1c, LDL, phosphorus, calcium, NT-proBNP, LVEF (AGGREGATE counts only, <5
     suppressed).

Respiratory-therapy and device-interrogation rows are filtered out first (their Lab
Component Common Name values -- e.g. "RESPIRATORY RATE (POST)", "O2 OXYGEN DEVICE",
"ICD-...", "LEAD1-DEVICE ..." -- were identified by inspecting the field's value_counts()
during development; see docs/data_build_log.md). "TRANSCRIPTION" rows (a free-text/narrative
pseudo-lab-component, 8677 rows in the source file, no LOINC code) are also excluded from
every numeric computation in this module -- String Value on those rows is never read.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from agg_io import suppress, write_aggregate
from kairos.paths import repo_root

DATA_DIR = repo_root()
PRIVATE_DIR = DATA_DIR / "data" / "derived" / "private"
AGG_DIR = DATA_DIR / "data" / "derived" / "aggregates"
LABS_XLSX = DATA_DIR / "labs_deidentified.xlsx"
MEDS_XLSX = DATA_DIR / "medications_deidentified.xlsx"

EXCLUDE_COMPONENT_RE = (
    r"RESPIRATORY RATE|RESPIRATORY SYNCYTIAL|O2 OXYGEN DEVICE|OXYGEN DEVICE|"
    r"^ICD-|^LEAD\d+-DEVICE|^TRANSCRIPTION$"
)

KEY_LAB_PATTERNS = {
    "creatinine_or_eGFR": r"CREATININE|EGFR|^GFR",
    "HbA1c": r"HEMOGLOBIN A1C",
    "LDL": r"^LDL$|LDL CHOLESTEROL",
    "phosphorus": r"PHOSPHORUS",
    "calcium": r"CALCIUM, TOTAL|CALCIUM IONIZED",
    "NT_proBNP": r"NT PRO BNP|PROBNP",
    "LVEF": r"LV EJECTION",
}
INR_PATTERN = r"\bPT INR\b|\bINR\b"


def load_clean_labs():
    labs = pd.read_excel(LABS_XLSX)
    name_upper = labs["Lab Component Common Name"].astype(str).str.upper()
    excluded = name_upper.str.contains(EXCLUDE_COMPONENT_RE, regex=True, na=False)
    print(f"excluding {int(excluded.sum())} / {len(labs)} respiratory-therapy / "
          f"device-interrogation / transcription rows")
    return labs[~excluded].copy(), name_upper[~excluded]


def main():
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    labs, name_upper = load_clean_labs()

    # ---- 1. INR in range, warfarin patients, private ----
    meds = pd.read_excel(MEDS_XLSX)
    warfarin_patients = set(
        meds.loc[meds["Simple Generic Name"].astype(str).str.contains("warfarin", case=False,
                                                                        na=False), "Patient"])
    print(f"{len(warfarin_patients)} patients with >=1 warfarin medication row")

    inr = labs[name_upper.str.contains(INR_PATTERN, regex=True, na=False)].copy()
    inr_wf = inr[inr["Patient"].isin(warfarin_patients)].copy()
    inr_wf = inr_wf[inr_wf["Numeric Value"].notna()]
    inr_wf["in_range_2_3"] = inr_wf["Numeric Value"].between(2.0, 3.0)
    per_patient_year = inr_wf.groupby(["Patient", "Result Date"]).agg(
        n_inr_results=("Numeric Value", "size"),
        n_in_range=("in_range_2_3", "sum"),
    ).reset_index().rename(columns={"Result Date": "year"})
    per_patient_year["pct_in_range_2_3"] = (
        100.0 * per_patient_year["n_in_range"] / per_patient_year["n_inr_results"]).round(1)
    per_patient_year.to_csv(PRIVATE_DIR / "warfarin_inr_time_in_range_by_patient_year.csv",
                             index=False)
    print(f"wrote {len(per_patient_year)} patient-year INR rows -> "
          f"{PRIVATE_DIR / 'warfarin_inr_time_in_range_by_patient_year.csv'}")

    # ---- 2. key-lab per-patient-year availability, aggregate only ----
    avail_rows = []
    for label, pattern in KEY_LAB_PATTERNS.items():
        matched = labs[name_upper.str.contains(pattern, regex=True, na=False)]
        py = matched.groupby(["Patient", "Result Date"]).size().reset_index(name="n_results")
        py["lab"] = label
        avail_rows.append(py.rename(columns={"Result Date": "year"}))
    avail = pd.concat(avail_rows, ignore_index=True) if avail_rows else pd.DataFrame()

    # aggregate: per lab x year, count of DISTINCT PATIENTS with >=1 result that year
    agg = avail.groupby(["lab", "year"])["Patient"].nunique().reset_index(
        name="n_patients_with_ge1_result")
    agg["n_patients_with_ge1_result"] = agg["n_patients_with_ge1_result"].apply(suppress)
    source_note = ("labs_deidentified.xlsx (respiratory-therapy / device-interrogation / "
                   "free-text-transcription rows excluded first)")
    write_aggregate(
        agg, AGG_DIR, "key_lab_availability_by_patient_year.csv", source_note,
        "PATIENT-YEAR AVAILABILITY: count of distinct patients with >=1 result for each "
        "key lab in each year -- NOT lab values themselves.")
    print(f"wrote key-lab availability aggregate -> "
          f"{AGG_DIR / 'key_lab_availability_by_patient_year.csv'}")

    # bonus: overall (all-years) distinct-patient coverage per lab, aggregate
    overall = avail.groupby("lab")["Patient"].nunique().reset_index(
        name="n_patients_with_ge1_result_any_year")
    overall["n_patients_with_ge1_result_any_year"] = overall[
        "n_patients_with_ge1_result_any_year"].apply(suppress)
    write_aggregate(
        overall, AGG_DIR, "key_lab_availability_overall.csv", source_note,
        "Count of distinct patients with >=1 result for each key lab, across all years.")
    print(f"wrote overall key-lab coverage -> {AGG_DIR / 'key_lab_availability_overall.csv'}")


if __name__ == "__main__":
    main()
