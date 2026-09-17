# -*- coding: utf-8 -*-
"""
Task 6, medication bullet. Classifies every row of medications_deidentified.xlsx into the
task-specified drug classes, builds a per-patient-year exposure grid (private) and
aggregate class-prevalence-by-year / exposure-duration tables (committable).

Classification is primarily driven by `Simple Generic Name` (a clean generic-drug-name
field) matched against curated term lists per class, cross-checked against
`Medication Therapeutic Class` (a coded field already present in the source file, e.g.
"ANTIHYPERGLYCEMICS", "ANTICOAGULANTS", "ANTIPLATELET DRUGS") as a secondary signal noted
per row rather than used to override the primary generic-name call.

Year handling: all three source spreadsheets truncate dates to the year (documented
constraint). A medication row is counted as "exposed" for every year from Start Date's year
through the LATER of (End Date, Discontinued Date)'s year inclusive; if neither end-type
date is present, the row is counted as exposed for its Start Date year only (a conservative
choice, documented here rather than assumed silently -- an ongoing/ambiguous-duration
medication is not extrapolated to "still being taken" in later years without evidence).
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
MEDS_XLSX = DATA_DIR / "medications_deidentified.xlsx"

CLASS_TERMS = {
    "VKA": ["warfarin", "acenocoumarol"],
    "DOAC": ["apixaban", "rivaroxaban", "edoxaban", "dabigatran"],
    "antiplatelet": ["aspirin", "clopidogrel", "ticagrelor", "prasugrel", "dipyridamole"],
    "heparin": ["heparin", "enoxaparin", "dalteparin", "fondaparinux", "tinzaparin"],
    "lipid_lowering": ["statin", "atorvastatin", "rosuvastatin", "simvastatin",
                        "pravastatin", "lovastatin", "fluvastatin", "pitavastatin",
                        "ezetimibe", "fenofibrate", "gemfibrozil", "niacin",
                        "icosapent", "alirocumab", "evolocumab"],
    "glucose_lowering": ["metformin", "insulin", "glipizide", "glyburide", "glimepiride",
                          "sitagliptin", "saxagliptin", "linagliptin", "alogliptin",
                          "empagliflozin", "dapagliflozin", "canagliflozin",
                          "liraglutide", "semaglutide", "exenatide", "dulaglutide",
                          "pioglitazone", "rosiglitazone", "acarbose"],
    "mineral_metabolism": ["calcium", "calcitriol", "vitamin d", "ergocalciferol",
                            "cholecalciferol", "paricalcitol", "doxercalciferol",
                            "sevelamer", "lanthanum", "cinacalcet", "etelcalcetide"],
}
# Therapeutic-class fallback: if generic-name matching missed a row but the coded
# therapeutic class is unambiguous for that class, still classify it (documented, not a
# silent override -- see `matched_via` column in the private output).
THERAPEUTIC_CLASS_FALLBACK = {
    "ANTIHYPERGLYCEMICS": "glucose_lowering",
    "ANTICOAGULANTS": None,  # too broad alone (covers VKA+DOAC+heparin) -- generic name required
    "ANTIPLATELET DRUGS": "antiplatelet",
}


def classify_row(generic_name, therapeutic_class):
    g = str(generic_name).lower()
    for cls, terms in CLASS_TERMS.items():
        for t in terms:
            if t in g:
                return cls, "generic_name"
    tc = str(therapeutic_class).upper()
    fb = THERAPEUTIC_CLASS_FALLBACK.get(tc)
    if fb:
        return fb, "therapeutic_class_fallback"
    return None, None


def _year_range(row):
    start = row.get("Start Date")
    end = row.get("End Date")
    disc = row.get("Discontinued Date")
    if pd.isna(start):
        return []
    start = int(start)
    candidates = [int(x) for x in (end, disc) if pd.notna(x)]
    stop = max(candidates) if candidates else start
    stop = max(stop, start)
    stop = min(stop, start + 30)  # sanity guard against a bad/typo date exploding the range
    return list(range(start, stop + 1))


def main():
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    meds = pd.read_excel(MEDS_XLSX)

    meds["drug_class"], meds["matched_via"] = zip(*meds.apply(
        lambda r: classify_row(r["Simple Generic Name"], r["Medication Therapeutic Class"]),
        axis=1))
    classified = meds[meds["drug_class"].notna()].copy()
    print(f"{len(classified)} / {len(meds)} medication rows classified into a tracked class")
    print(classified["drug_class"].value_counts())

    # ---- private: patient x year x class exposure grid ----
    exposure_rows = []
    for _, r in classified.iterrows():
        years = _year_range(r)
        for y in years:
            exposure_rows.append((r["Patient"], y, r["drug_class"]))
    exp = pd.DataFrame(exposure_rows, columns=["Patient", "year", "drug_class"]).drop_duplicates()
    grid = exp.assign(exposed=1).pivot_table(
        index=["Patient", "year"], columns="drug_class", values="exposed",
        fill_value=0, aggfunc="max").reset_index()
    grid.to_csv(PRIVATE_DIR / "medication_exposure_grid.csv", index=False)
    print(f"wrote patient-year exposure grid ({grid.shape[0]} rows) -> "
          f"{PRIVATE_DIR / 'medication_exposure_grid.csv'}")

    # per-patient exposure duration (years) and first/last year per class, also private
    dur = exp.groupby(["Patient", "drug_class"])["year"].agg(
        n_years_exposed="nunique", first_year="min", last_year="max").reset_index()
    dur.to_csv(PRIVATE_DIR / "medication_exposure_durations.csv", index=False)
    print(f"wrote per-patient exposure durations -> "
          f"{PRIVATE_DIR / 'medication_exposure_durations.csv'}")

    # ---- aggregates: class prevalence by year (n patients exposed, <5 suppressed) ----
    prev = exp.groupby(["year", "drug_class"])["Patient"].nunique().reset_index(
        name="n_patients_exposed")
    prev["n_patients_exposed"] = prev["n_patients_exposed"].apply(suppress)
    write_aggregate(
        prev, AGG_DIR, "medication_class_prevalence_by_year.csv",
        "medications_deidentified.xlsx via data/derived/private/medication_exposure_grid.csv (private, not committed)",
        "Count of distinct patients exposed to each drug class in each year (a patient "
        "counts as exposed for every year from a medication row's Start Date through the "
        "later of End Date/Discontinued Date, or Start Date only if neither is present).")

    # aggregate exposure-duration distribution per class (median/IQR-style, cell-safe: only
    # report counts of patients falling in duration bins, suppressed under 5)
    dur2 = dur.copy()
    dur2["duration_bin"] = pd.cut(dur2["n_years_exposed"], bins=[0, 1, 3, 4, 1000],
                                   labels=["1 year", "2-3 years", "4 years", "5+ years"],
                                   include_lowest=True)
    dur_agg = dur2.groupby(["drug_class", "duration_bin"], observed=True)["Patient"].nunique(
        ).reset_index(name="n_patients")
    dur_agg["n_patients"] = dur_agg["n_patients"].apply(suppress)
    write_aggregate(
        dur_agg, AGG_DIR, "medication_exposure_duration_distribution.csv",
        "medications_deidentified.xlsx via data/derived/private/medication_exposure_durations.csv (private, not committed)",
        "Count of distinct patients per (drug_class, duration_bin) pair, duration_bin from "
        "n_years_exposed (distinct calendar years with >=1 exposed day) binned into "
        "1 year / 2-3 years / 4 years / 5+ years.")

    print("wrote aggregates -> data/derived/aggregates/medication_class_prevalence_by_year.csv, "
          "medication_exposure_duration_distribution.csv")


if __name__ == "__main__":
    main()
