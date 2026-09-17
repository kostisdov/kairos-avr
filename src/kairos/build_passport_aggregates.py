# -*- coding: utf-8 -*-
"""
Build data/derived/aggregates/ tables from data/derived/private/passport.csv (Task 6,
second bullet). Every output here is a CELL-LEVEL COUNT/RATE table with NO patient IDs and
NO note text -- safe to commit. Any cell with fewer than 5 patients is replaced with the
string "<5" per NON-NEGOTIABLE RULE 2, applied BEFORE writing (never write the true small
count anywhere in this folder).

Each CSV is a clean single-header-row table (see src/kairos/agg_io.py for why -- a prior
version of this script put a `#`-comment provenance line directly above the CSV header,
which broke plain `pandas.read_csv()`; provenance now lives in a `.meta.yaml` sidecar next
to each CSV instead, per coordinator Fix 4, 2026-09-16).
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from agg_io import suppress, write_aggregate
from kairos.paths import repo_root

DATA_DIR = repo_root()
PRIVATE_DIR = DATA_DIR / "data" / "derived" / "private"
AGG_DIR = DATA_DIR / "data" / "derived" / "aggregates"

ERA_BINS = [(2006, 2012), (2013, 2017), (2018, 2022)]
SOURCE = "data/derived/private/passport.csv (private, not committed)"


def _suppress_df(df, cols):
    for c in cols:
        df[c] = df[c].apply(suppress)
    return df


def era_of(year):
    if year is None or (isinstance(year, float) and pd.isna(year)):
        return None
    year = int(year)
    for lo, hi in ERA_BINS:
        if lo <= year <= hi:
            return f"{lo}-{hi}"
    return "outside 2006-2022"


def main():
    p = pd.read_csv(PRIVATE_DIR / "passport.csv")
    n_patients = len(p)
    written = []

    # 1. patients by route
    by_route = p["route"].fillna("").replace("", "(unspecified)").value_counts().rename_axis(
        "route").reset_index(name="n_patients")
    by_route = _suppress_df(by_route, ["n_patients"])
    written.append(write_aggregate(
        by_route, AGG_DIR, "patients_by_route.csv", SOURCE,
        "Count of patients per route value (SAVR / TAVR / TAVR+SAVR / (hist only) = "
        "route mentioned only in historical/non-implant notes / (unspecified))."))

    # 2. by canonical_model and design_class
    by_model = p.copy()
    by_model["canonical_model"] = by_model["canonical_model"].fillna("").replace("", "(none identified)")
    by_model["design_class"] = by_model["design_class"].fillna("").replace("", "(none identified)")
    by_model_class = by_model.groupby(["canonical_model", "design_class"]).size().reset_index(
        name="n_patients")
    by_model_class = _suppress_df(by_model_class, ["n_patients"])
    written.append(write_aggregate(
        by_model_class, AGG_DIR, "patients_by_canonical_model_and_design_class.csv", SOURCE,
        "Count of patients per (canonical_model, design_class) pair, using each patient's "
        "PRIMARY (most-specific-matched) canonical_model from passport.py."))

    # 3. implant year by era crossed with design_class
    p2 = p.copy()
    p2["era"] = p2["implant_year"].apply(era_of)
    p2["design_class"] = p2["design_class"].fillna("").replace("", "(none identified)")
    era_tab = p2.dropna(subset=["era"]).groupby(["era", "design_class"]).size().reset_index(
        name="n_patients")
    era_tab = _suppress_df(era_tab, ["n_patients"])
    written.append(write_aggregate(
        era_tab, AGG_DIR, "implant_era_by_design_class.csv", SOURCE,
        "Count of patients per (era, design_class) pair, era = first implant_year binned "
        "into 2006-2012 / 2013-2017 / 2018-2022 (per task spec). Patients with no "
        "implant_year, or an implant_year outside 2006-2022, are excluded from this table."))

    # 4. size distribution
    size_dist = p["size_mm"].dropna().astype(int).value_counts().sort_index().rename_axis(
        "size_mm").reset_index(name="n_patients")
    size_dist = _suppress_df(size_dist, ["n_patients"])
    written.append(write_aggregate(
        size_dist, AGG_DIR, "size_distribution.csv", SOURCE,
        "Count of patients per implanted valve size (mm), from each patient's primary "
        "implant-note size mention."))

    # 5. prosthetic-gradient coverage counts
    n_ge1 = int((p["n_gradients_prosthetic"] > 0).sum())
    n_ge2yr = int((p["n_distinct_gradient_years_prosthetic"] >= 2).sum())
    cov = pd.DataFrame([
        {"metric": "patients_with_ge1_prosthetic_gradient", "n_patients": suppress(n_ge1)},
        {"metric": "patients_with_gradients_in_ge2_distinct_years", "n_patients": suppress(n_ge2yr)},
        {"metric": "total_patients", "n_patients": n_patients},
    ])
    written.append(write_aggregate(
        cov, AGG_DIR, "prosthetic_gradient_coverage.csv", SOURCE,
        "Coverage counts for prosthetic (not native) mean-gradient mentions across each "
        "patient's notes."))

    # 6. event-mention counts
    event_cols = ["event_ViV", "event_redo", "event_prosthetic_dysfunction",
                  "event_endocarditis", "event_thrombosis"]
    ev = pd.DataFrame([{"event": c.replace("event_", ""), "n_patients": suppress(int(p[c].sum()))}
                       for c in event_cols])
    written.append(write_aggregate(
        ev, AGG_DIR, "event_mention_counts.csv", SOURCE,
        "Count of patients with >=1 note matching each event-mention regex "
        "(valve-in-valve, redo, prosthetic dysfunction/SVD-like, endocarditis, thrombosis)."))

    # 7. bonus: DVI / EOA / regurgitation series coverage
    def _nonempty(col):
        return p[col].apply(lambda s: len(json.loads(s)) > 0 if isinstance(s, str) else False)
    extra_cov = pd.DataFrame([
        {"field": "dvi_series", "n_patients_with_ge1_value": suppress(int(_nonempty("dvi_series_json").sum()))},
        {"field": "eoa_series", "n_patients_with_ge1_value": suppress(int(_nonempty("eoa_series_json").sum()))},
        {"field": "regurg_series", "n_patients_with_ge1_value": suppress(int(_nonempty("regurg_series_json").sum()))},
    ])
    written.append(write_aggregate(
        extra_cov, AGG_DIR, "dvi_eoa_regurg_coverage.csv", SOURCE,
        "Count of patients with >=1 extracted dimensionless-index / effective-orifice-area "
        "/ regurgitation-grade value anywhere in their notes."))

    print(f"Wrote {len(written)} aggregate CSV+meta.yaml pairs to {AGG_DIR}")
    for csv_path, meta_path in written:
        print(" ", csv_path.name, "+", meta_path.name)


if __name__ == "__main__":
    main()
