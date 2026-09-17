# -*- coding: utf-8 -*-
"""
Task 5 (ClinicalTrials.gov half). Downloads full study JSON for a curated set of NCT
numbers -- found by searching ClinicalTrials.gov API v2 `query.titles` for the trial names
named in the task brief (COMMENCE, Trifecta durability, Trifecta GT, PARTNER II/3, NOTION,
CoreValve US Pivotal, SURTAVI, Evolut Low Risk, Inspiris RESILIA post-market), NOT by
trusting a pre-existing NCT-number list, per the task's explicit instruction -- into
data/raw/ctgov/, then flattens every durability-related outcome measure (SVD, valve
deterioration, reintervention, hemodynamics, thrombosis, valve failure) into
data/reference/ctgov_durability_outcomes.csv.
"""
import csv
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from kairos.paths import repo_root

ROOT = str(repo_root())
RETRIEVED_AT = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# (nct_id, short label, how it was found) -- all found via query.titles searches this
# session, see docs/data_build_log.md for the exact queries run
STUDIES = [
    ("NCT01757665", "COMMENCE (Inspiris RESILIA)"),
    ("NCT01256710", "Trifecta Durability Study"),
    ("NCT03016169", "Trifecta GT Post Market Clinical Follow-up"),
    ("NCT02675114", "PARTNER 3 (low risk, SAPIEN 3)"),
    ("NCT03222128", "PARTNER II S3 Intermediate Risk"),
    ("NCT01057173", "NOTION"),
    ("NCT01240902", "CoreValve US Pivotal (High/Very High Risk)"),
    ("NCT01586910", "SURTAVI"),
    ("NCT02701283", "Evolut Low Risk"),
    ("NCT03666741", "Inspiris RESILIA Durability Registry"),
]

DURABILITY_RE = re.compile(
    r"structural valve deteriorat|\bSVD\b|valve deteriorat|reinterven|re-interven|"
    r"reoperat|explant|valve.?in.?valve|\bViV\b|gradient|effective orifice|hemodynamic|"
    r"haemodynamic|thrombosis|thrombus|valve failure|bioprosthetic valve failure|"
    r"\bBVF\b|\bBVD\b|durability|structural valve|non-structural|regurgitation",
    re.I,
)

# Coordinator Fix 3 (2026-09-16): a NARROWER, precisely-worded boolean check applied to
# every already-pulled row (the 755-row set gathered via the broader DURABILITY_RE pre-
# filter above, which controls what gets pulled from ClinicalTrials.gov at all -- re-
# fetching with NO pre-filter would include every safety/mortality/QoL outcome measure
# from all 10 trials, not just valve-durability-adjacent ones, and was not what was asked).
# True when the outcome title or description concerns: structural valve deterioration,
# hemodynamic valve deterioration, bioprosthetic valve dysfunction or failure,
# reintervention, explant, valve thrombosis, mean gradient, effective orifice area,
# dimensionless index, or regurgitation -- the coordinator's exact category list, quoted in
# the docstring of build() below.
DURABILITY_RELATED_STRICT_RE = re.compile(
    r"structural valve deteriorat|\bSVD\b|"
    r"h[ae]modynamic valve deteriorat|\bHVD\b|"
    r"bioprosthetic valve dysfunction|bioprosthetic valve failure|\bBVD\b|\bBVF\b|"
    r"reintervention|re-intervention|reinterven|"
    r"\bexplant|"
    r"valve thrombosis|"
    r"mean gradient|"
    r"effective orifice area|\bEOA\b|"
    r"dimensionless (?:velocity )?index|\bDVI\b|"
    r"regurgitation",
    re.I,
)


def fetch_study(nct_id, tries=3):
    url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
    for i in range(tries):
        try:
            r = requests.get(url, timeout=45)
            if r.status_code == 200:
                return r.json()
            print(f"  [{r.status_code}] {nct_id}")
            return None
        except Exception as e:
            print(f"  retry {i} {nct_id}: {e}")
            time.sleep(2)
    return None


def flatten_outcomes(study_json, nct_id, label):
    rows = []
    ps = study_json.get("protocolSection", {})
    title = ps.get("identificationModule", {}).get("briefTitle", "")
    arms = ps.get("armsInterventionsModule", {}).get("armGroups", [])
    arm_names = {a.get("label"): a for a in arms}

    results = study_json.get("resultsSection", {})
    outcome_module = results.get("outcomeMeasuresModule", {})
    outcomes = outcome_module.get("outcomeMeasures", [])
    for om in outcomes:
        om_title = om.get("title", "")
        om_desc = om.get("description", "")
        if not DURABILITY_RE.search(om_title + " " + om_desc):
            continue
        durability_related = bool(DURABILITY_RELATED_STRICT_RE.search(om_title + " " + om_desc))
        time_frame = om.get("timeFrame", "")
        unit = om.get("unitOfMeasure", "")
        rtype = om.get("paramType", "")
        # group (arm) definitions and denominators are stored PER OUTCOME MEASURE in the
        # CT.gov API v2 schema (om["groups"], om["denoms"]), not at the module level --
        # confirmed by inspecting the raw JSON (see docs/data_build_log.md)
        group_labels = {g.get("id"): g.get("title") for g in om.get("groups", [])}
        denom_by_group = {}
        for d in om.get("denoms", []):
            for cnt in d.get("counts", []):
                denom_by_group[cnt.get("groupId")] = cnt.get("value")
        classes = om.get("classes", [])
        for cls in classes:
            for cat in cls.get("categories", []):
                cat_title = cat.get("title", "")
                for meas in cat.get("measurements", []):
                    group_id = meas.get("groupId")
                    value = meas.get("value")
                    rows.append(dict(
                        nct_id=nct_id, study_title=title,
                        arm=group_labels.get(group_id, group_id),
                        device=label, outcome_title=om_title,
                        outcome_description=om_desc[:500], time_frame=time_frame,
                        value=value, unit=unit,
                        n_analysed=denom_by_group.get(group_id, ""), result_type=rtype,
                        category=cat_title, durability_related=durability_related,
                    ))
    return rows


def main(refetch=False):
    """refetch=False (default) re-uses the JSON already saved to data/raw/ctgov/ from the
    original run (coordinator Fix 3, 2026-09-16, is a re-processing/re-column-ing task, not
    a re-download -- avoids unnecessary network calls against a public API for data we
    already have). Pass refetch=True to hit the ClinicalTrials.gov API again from scratch."""
    all_rows = []
    for nct_id, label in STUDIES:
        out_path = f"{ROOT}/data/raw/ctgov/{nct_id}.json"
        js = None
        if not refetch and Path(out_path).exists():
            with open(out_path, encoding="utf-8") as f:
                js = json.load(f)
        if js is None:
            js = fetch_study(nct_id)
            if js is None:
                print(f"FAILED to fetch {nct_id} ({label})")
                continue
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(js, f, indent=1)
            time.sleep(0.5)  # be polite, well under any rate limit
        rows = flatten_outcomes(js, nct_id, label)
        n_durability_related = sum(1 for r in rows if r["durability_related"])
        print(f"{nct_id} ({label}): {len(rows)} rows ({n_durability_related} durability_related=True)")
        all_rows.extend(rows)

    cols = ["nct_id", "study_title", "arm", "device", "outcome_title",
            "outcome_description", "time_frame", "value", "unit", "n_analysed",
            "result_type", "category", "durability_related", "retrieved_at"]
    full_path = f"{ROOT}/data/reference/ctgov_durability_outcomes.csv"
    with open(full_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in all_rows:
            r["retrieved_at"] = RETRIEVED_AT
            w.writerow(r)
    print(f"\nTOTAL: wrote {len(all_rows)} rows (full, unfiltered) -> {full_path}")

    filtered_rows = [r for r in all_rows if r["durability_related"]]
    filtered_path = f"{ROOT}/data/reference/ctgov_durability_outcomes_filtered.csv"
    with open(filtered_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(filtered_rows)
    print(f"TOTAL: wrote {len(filtered_rows)} rows (durability_related=True only) -> {filtered_path}")


if __name__ == "__main__":
    main()
