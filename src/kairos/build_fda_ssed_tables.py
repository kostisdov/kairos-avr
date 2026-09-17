# -*- coding: utf-8 -*-
"""
Build data/reference/fda_ssed_hemodynamics.csv and data/reference/fda_ssed_events.csv
(Task 4) by transcribing numbers read directly from the downloaded FDA SSED PDFs in
data/raw/fda_ssed/ (page numbers cited per row; PDFs themselves stay in data/raw/,
per NON-NEGOTIABLE RULE 4). Hand-verified against pdfplumber page.extract_text() output
during development -- see docs/data_build_log.md.

IMPORTANT CORRECTION (documented here and in MANIFEST.csv): the task brief's source list
labelled P180029B.pdf as "Evolut R/PRO"; the SSED's own page 1 confirms it is actually the
LOTUS Edge Valve System (Boston Scientific, approved 2019-04-23). It is used below as
"Lotus Edge" data (and, since its own pivotal trial used CoreValve as an active comparator,
as bonus CoreValve comparator data too). No genuine Evolut R/PRO SSED PDF was located this
session. NOTE (coordinator Fix 3, 2026-09-16): canonical_model is literally "Lotus Edge"
here (the specific product this SSED describes), which differs slightly from
device_table.csv's canonical_model="Lotus" (the broader task-assigned bucket name for this
device family, per the original task brief's own design-class list) -- both refer to the
same underlying product; "Lotus Edge" is the more precise SSED-specific label the
coordinator asked for. design_class is looked up from device_table.csv by canonical_model,
falling back through this alias when the exact literal name isn't a device_table.csv row.
"""
import csv
from datetime import datetime, timezone

import pandas as pd
from kairos.paths import repo_root

ROOT = str(repo_root())
RETRIEVED_AT = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

_device_table = pd.read_csv(f"{ROOT}/data/reference/device_table.csv")
_DESIGN_CLASS_BY_MODEL = dict(zip(_device_table["canonical_model"], _device_table["design_class"]))
# canonical_model values used in this file that are more specific than / an alias of a
# device_table.csv row (see module docstring's Fix-3 note)
_DESIGN_CLASS_ALIAS = {"Lotus Edge": _DESIGN_CLASS_BY_MODEL.get("Lotus", "")}


def _design_class_for(canonical_model):
    return _DESIGN_CLASS_BY_MODEL.get(canonical_model) or _DESIGN_CLASS_ALIAS.get(canonical_model, "")


hemo_rows = []
event_rows = []


def h(document, canonical_model, size_mm, population, time_point, mg_mean, mg_sd,
      pg_mean, pg_sd, eoa_mean, eoa_sd, n, page, definition, confidence):
    hemo_rows.append(dict(document=document, canonical_model=canonical_model,
                           design_class=_design_class_for(canonical_model), size_mm=size_mm,
                           population=population, time_point=time_point,
                           mean_gradient_mmHg_mean=mg_mean, mean_gradient_sd=mg_sd,
                           peak_gradient_mmHg_mean=pg_mean, peak_gradient_sd=pg_sd,
                           eoa_cm2_mean=eoa_mean, eoa_sd=eoa_sd, n=n, page=page,
                           definition=definition, confidence=confidence))


def e(document, canonical_model, event, population, time_point, n_events, denominator, pct,
      follow_up, page, definition, confidence):
    event_rows.append(dict(document=document, canonical_model=canonical_model,
                            design_class=_design_class_for(canonical_model), event=event,
                            population=population, time_point=time_point, n_events=n_events,
                            denominator=denominator, pct=pct, follow_up_duration=follow_up,
                            page=page, definition=definition, confidence=confidence))


SAPIEN3_DEF = ("CEC = Clinical Events Committee adjudicated; VI = Valve Implant population "
               "(received & retained a SAPIEN 3 implant)")

# ---------------------------------------------------------------------------
# SAPIEN 3 (P140031B) -- two overlapping trial cohorts reported in the same SSED:
# PIIS3HR (US high-risk cohort) and S3OUS (outside-US cohort). Overall (all-access-route)
# figures only; the pages read did not break EOA/gradient out by valve size.
# ---------------------------------------------------------------------------
h("P140031B_SAPIEN3_original_SSED.pdf", "SAPIEN 3", "all sizes pooled", "PIIS3HR VI (N=583)",
  "baseline", 45.5, 14.3, 75.8, 22.6, "", "", 583, 25, SAPIEN3_DEF, "high")
h("P140031B_SAPIEN3_original_SSED.pdf", "SAPIEN 3", "all sizes pooled", "PIIS3HR VI (N=583)",
  "30 days", 11.1, 4.5, 21.2, 8.5, "", "", "", 30, SAPIEN3_DEF, "high")
h("P140031B_SAPIEN3_original_SSED.pdf", "SAPIEN 3", "all sizes pooled", "S3OUS VI",
  "baseline", 44.8, 15.4, 77.5, 24.9, 0.6, 0.2, "", 38, SAPIEN3_DEF, "high")
h("P140031B_SAPIEN3_original_SSED.pdf", "SAPIEN 3", "all sizes pooled", "S3OUS VI",
  "30 days", 10.4, 4.1, 21.0, 7.7, 1.5, 0.4, "", 38, SAPIEN3_DEF, "high")
h("P140031B_SAPIEN3_original_SSED.pdf", "SAPIEN 3", "all sizes pooled", "S3OUS VI",
  "1 year", 10.7, 4.1, 21.5, 8.2, 1.4, 0.4, "", 39, SAPIEN3_DEF, "high")

e("P140031B_SAPIEN3_original_SSED.pdf", "SAPIEN 3", "composite: death, stroke, AI>=moderate",
  "S3OUS AT overall", "30 days", 13, 88, 14.8, "30 days", 36,
  "CEC-adjudicated, S3OUS AT (all-treated) population, Table 22", "high")
e("P140031B_SAPIEN3_original_SSED.pdf", "SAPIEN 3", "composite: death, stroke, AI>=moderate",
  "S3OUS AT overall", "1 year", 25, 82, 30.5, "1 year", 36, "CEC-adjudicated, Table 22", "high")
e("P140031B_SAPIEN3_original_SSED.pdf", "SAPIEN 3", "death, all-cause", "S3OUS AT overall",
  "30 days", 8, 102, 7.8, "30 days", 36, "CEC-adjudicated, Table 22", "high")
e("P140031B_SAPIEN3_original_SSED.pdf", "SAPIEN 3", "death, all-cause", "S3OUS AT overall",
  "1 year", 20, 102, 19.6, "1 year", 36, "CEC-adjudicated, Table 22", "high")
e("P140031B_SAPIEN3_original_SSED.pdf", "SAPIEN 3", "death, cardiovascular",
  "S3OUS AT overall", "30 days", 7, 102, 6.9, "30 days", 36, "CEC-adjudicated, Table 22",
  "high")
e("P140031B_SAPIEN3_original_SSED.pdf", "SAPIEN 3", "death, cardiovascular",
  "S3OUS AT overall", "1 year", 9, 102, 8.8, "1 year", 36, "CEC-adjudicated, Table 22",
  "high")
e("P140031B_SAPIEN3_original_SSED.pdf", "SAPIEN 3", "stroke", "S3OUS AT overall", "30 days",
  3, 102, 2.9, "30 days", 36, "CEC-adjudicated, Table 22", "high")
e("P140031B_SAPIEN3_original_SSED.pdf", "SAPIEN 3", "stroke", "S3OUS AT overall", "1 year",
  5, 102, 4.9, "1 year", 36, "CEC-adjudicated, Table 22", "high")
e("P140031B_SAPIEN3_original_SSED.pdf", "SAPIEN 3", "aortic insufficiency >= moderate",
  "S3OUS AT overall", "30 days", 3, 81, 3.7, "30 days", 36, "CEC-adjudicated, Table 22",
  "high")
e("P140031B_SAPIEN3_original_SSED.pdf", "SAPIEN 3", "aortic insufficiency >= moderate",
  "S3OUS AT overall", "1 year", 1, 62, 1.6, "1 year", 36, "CEC-adjudicated, Table 22",
  "high")

# ---------------------------------------------------------------------------
# Inspiris RESILIA / COMMENCE trial (P150048B)
# ---------------------------------------------------------------------------
INSPIRIS_DEF = "COMMENCE trial, N=689 enrolled / 689 implanted (per SSED text), 1-year echo core lab"
for size, mg, mgsd, mgn, eoa, eoasd, eoan in [
    (19, 17.6, 7.8, 16, 1.1, 0.2, 16), (21, 12.6, 4.7, 97, 1.3, 0.3, 97),
    (23, 10.1, 3.8, 158, 1.6, 0.4, 155), (25, 9.6, 5.2, 132, 1.8, 0.5, 131),
    (27, 8.2, 3.5, 69, 2.2, 0.6, 68),
]:
    h("P150048B_InspirisResilia_original_SSED.pdf", "Inspiris RESILIA", size,
      "COMMENCE, evaluable at 1yr", "1 year", mg, mgsd, "", "", eoa, eoasd, mgn, 19,
      INSPIRIS_DEF + " (Table 9; EOA/gradient N differ slightly per column, both reported)",
      "high")
e("P150048B_InspirisResilia_original_SSED.pdf", "Inspiris RESILIA",
  "structural valve deterioration", "female (of 689 enrolled)", "1 year", 0, "", 100.0,
  "1 year", 20, "KM event-free probability, Table 11; text states explicitly 'No cases of "
  "valve thrombosis and structural valve deterioration were observed for either cohort'",
  "high")
e("P150048B_InspirisResilia_original_SSED.pdf", "Inspiris RESILIA",
  "structural valve deterioration", "male (of 689 enrolled)", "1 year", 0, "", 100.0,
  "1 year", 20, "KM event-free probability, Table 11; 0 events in either sex", "high")
e("P150048B_InspirisResilia_original_SSED.pdf", "Inspiris RESILIA", "death (KM event-free %)",
  "female", "1 year", "", "", 99.5, "1 year", 20, "KM event-free probability, Table 11",
  "medium")
e("P150048B_InspirisResilia_original_SSED.pdf", "Inspiris RESILIA", "death (KM event-free %)",
  "male", "1 year", "", "", 96.9, "1 year", 20, "KM event-free probability, Table 11",
  "medium")
e("P150048B_InspirisResilia_original_SSED.pdf", "Inspiris RESILIA", "reoperation "
  "(KM event-free %)", "female", "1 year", "", "", 99.5, "1 year", 20, "Table 11", "medium")
e("P150048B_InspirisResilia_original_SSED.pdf", "Inspiris RESILIA", "reoperation "
  "(KM event-free %)", "male", "1 year", "", "", 99.7, "1 year", 20, "Table 11", "medium")
e("P150048B_InspirisResilia_original_SSED.pdf", "Inspiris RESILIA", "thromboembolism "
  "(KM event-free %)", "female", "1 year", "", "", 97.9, "1 year", 20, "Table 11", "medium")
e("P150048B_InspirisResilia_original_SSED.pdf", "Inspiris RESILIA", "thromboembolism "
  "(KM event-free %)", "male", "1 year", "", "", 95.9, "1 year", 20, "Table 11", "medium")
e("P150048B_InspirisResilia_original_SSED.pdf", "Inspiris RESILIA", "endocarditis "
  "(KM event-free %)", "female", "1 year", "", "", 100.0, "1 year", 20, "Table 11", "medium")
e("P150048B_InspirisResilia_original_SSED.pdf", "Inspiris RESILIA", "endocarditis "
  "(KM event-free %)", "male", "1 year", "", "", 99.0, "1 year", 20, "Table 11", "medium")

# ---------------------------------------------------------------------------
# Avalus (P170006B)
# ---------------------------------------------------------------------------
AVALUS_DEF = "PMA cohort, 864 implanted / 962 enrolled, 904.1 total patient-years, 1-year echo"
for size, mg, mgsd, mgn, eoa, eoasd, eoan in [
    (19, 17.1, 5.0, 27, 1.11, 0.25, 25), (21, 14.5, 4.3, 106, 1.25, 0.25, 99),
    (23, 12.1, 3.8, 205, 1.47, 0.32, 201), (25, 11.7, 4.0, 170, 1.57, 0.31, 167),
    (27, 10.3, 4.2, 43, 1.77, 0.41, 41),
]:
    h("P170006B_Avalus_original_SSED.pdf", "Avalus", size, "PMA cohort, evaluable at 1yr",
      "1 year", mg, mgsd, "", "", eoa, eoasd, mgn, 15,
      AVALUS_DEF + " (Table 10; n given per gradient column, EOA n differs slightly, "
      "both real values from the same table)", "high")
e("P170006B_Avalus_original_SSED.pdf", "Avalus", "endocarditis (late linearized rate)",
  "PMA cohort", "through last contact (late = >30 days post-implant)", 11, "834.2 late "
  "patient-years", 1.3, "904.1 total / 834.2 late patient-years", 15,
  "Late linearized event rate = events / late-patient-years (%); upper confidence bound "
  "2.11% (Greenwood formula) vs ISO 5840:2009 OPC threshold 2.4%", "high")
e("P170006B_Avalus_original_SSED.pdf", "Avalus", "major paravalvular leak", "PMA cohort",
  "through last contact", 0, "834.2 late patient-years", 0.0,
  "904.1 total / 834.2 late patient-years", 15, "Late linearized event rate", "high")

# ---------------------------------------------------------------------------
# LOTUS Edge (P180029B, mislabelled "Evolut R/PRO" in the task brief -- see module
# docstring) -- pivotal trial used CoreValve as an active comparator, so this document
# also yields bonus CoreValve data.
# ---------------------------------------------------------------------------
LOTUS_DEF = ("REPRISE III trial-type non-inferiority design; primary effectiveness endpoint "
             "= composite of all-cause mortality, disabling stroke, and moderate-or-greater "
             "paravalvular regurgitation (core-lab assessed) at 1 year")
e("P180029B_LOTUS_Edge_original_SSED.pdf", "Lotus Edge",
  "composite: death, disabling stroke, PVR>=moderate", "ITT (N=912)", "1 year", 82, 520,
  15.8, "1 year", 33, LOTUS_DEF + ", ITT analysis set", "high")
e("P180029B_LOTUS_Edge_original_SSED.pdf", "CoreValve",
  "composite: death, disabling stroke, PVR>=moderate", "ITT comparator (N=912)", "1 year",
  68, 262, 26.0, "1 year", 33, LOTUS_DEF + ", ITT analysis set (CoreValve = active "
  "comparator arm within the Lotus pivotal trial, sizes 26/29/31mm)", "high")
e("P180029B_LOTUS_Edge_original_SSED.pdf", "Lotus Edge",
  "composite: death, disabling stroke, PVR>=moderate", "Implanted (N=874)", "1 year", 78,
  506, 15.4, "1 year", 33, LOTUS_DEF + ", Implanted analysis set", "high")
e("P180029B_LOTUS_Edge_original_SSED.pdf", "CoreValve",
  "composite: death, disabling stroke, PVR>=moderate", "Implanted comparator (N=874)",
  "1 year", 66, 259, 25.5, "1 year", 33, LOTUS_DEF + ", Implanted analysis set", "high")


def main():
    hcols = ["document", "canonical_model", "design_class", "size_mm", "population",
             "time_point", "mean_gradient_mmHg_mean", "mean_gradient_sd",
             "peak_gradient_mmHg_mean", "peak_gradient_sd", "eoa_cm2_mean", "eoa_sd", "n",
             "page", "definition", "confidence", "retrieved_at"]
    with open(f"{ROOT}/data/reference/fda_ssed_hemodynamics.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=hcols)
        w.writeheader()
        for r in hemo_rows:
            r["retrieved_at"] = RETRIEVED_AT
            w.writerow(r)
    print(f"wrote {len(hemo_rows)} rows -> fda_ssed_hemodynamics.csv")

    ecols = ["document", "canonical_model", "design_class", "event", "population",
             "time_point", "n_events", "denominator", "pct", "follow_up_duration", "page",
             "definition", "confidence", "retrieved_at"]
    with open(f"{ROOT}/data/reference/fda_ssed_events.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ecols)
        w.writeheader()
        for r in event_rows:
            r["retrieved_at"] = RETRIEVED_AT
            w.writerow(r)
    print(f"wrote {len(event_rows)} rows -> fda_ssed_events.csv")


if __name__ == "__main__":
    main()
