# Data plan

Data sources, availability assumptions and preprocessing.
Status tags: **Implemented**, **Demonstrated**, **Planned**, **Unverified**.

---

## 1. Deployment sources

What a hospital already holds. Nothing here requires a new collection instrument.

| Source | Type | Access | Variables |
|---|---|---|---|
| Implant and procedure records | Structured, plus operative report text | Data warehouse; device identifier where recorded | Model, size, route, date, valve-in-valve, concomitant procedures |
| Echo reports | Structured measurements and free text | Echo reporting system; extraction for text | Gradients, velocity, area, dimensionless index, regurgitation, LVEF, stroke volume, date |
| Laboratory | Structured, LOINC | Data warehouse | eGFR trajectory, dialysis, HbA1c, lipids; phosphate, corrected calcium, PTH, alkaline phosphatase; Lp(a) in original units with assay and date; NT-proBNP; hs-CRP; research assays where a substudy exists |
| Medications | Structured, RxNorm | Data warehouse | Antithrombotic class with indication and start, stop, interruption and switch dates; INR results; lipid-lowering, glucose-lowering and mineral-metabolism drugs as exposures |
| Outcomes | Codes, procedure logs, death registry | Data warehouse; national death index | Candidate reinterventions, endocarditis, thrombosis, death and cause, for adjudication |
| Registries | Structured | Research application | External validation, incidence benchmarks |

---

## 2. Availability assumptions

Stated as risks, because each one can invalidate part of the design if it fails.

| Risk | Assumption | What we observed |
|---|---|---|
| **High** | Valve model and size are coded consistently | In our notes they appear as `#21 Trifecta`, `Implant Size: 25` and `Size: Ultra 26mm`. Extraction is required; a device-identifier field cannot be assumed |
| **High** | Serial echo exists for at least 60% of patients at five years | Published follow-up loss is 30 to 35% |
| **High** | Lp(a) is measured | One test among 17 patients with structured results. The module is switched off where absent |
| Medium | Mineral-metabolism results exist outside kidney disease | Phosphorus in 4 of 17; PTH in none |
| Medium | Day-level dates | Available in production, absent from the supplied records |
| Low | Creatinine, HbA1c, lipids and medication history | Present for essentially all patients |

---

## 3. Records supplied for the event

Three de-identified exports from an Epic-based US heart and vascular institute, dates truncated to
the year. **They stay outside version control.** Only code, aggregate counts with small-cell
suppression and figures without patient-level rows are committed.

| File | Rows | Patients | Years | Content |
|---|---|---|---|---|
| `notes_deidentified.xlsx` | 215 | 117 | 2006–2026 | Operative reports, progress notes, TAVR procedure notes, discharge summaries. Thirteen notes carry a deleted status and are excluded, leaving 202 |
| `labs_deidentified.xlsx` | 43,550 | 17 | 2004–2027 | Creatinine or eGFR (17), HbA1c (11), lipids (15), NT-proBNP (10), INR (17), LVEF as a lab component (85 values), phosphorus (4), Lp(a) (1), no PTH |
| `medications_deidentified.xlsx` | 5,807 | 17 | 2001–2027 | Warfarin (6 patients), direct oral anticoagulants (3), antiplatelets (16), statins (16), antidiabetics (16), with start, end and discontinuation years |

Years extend to 2027, beyond the document date. The medication export carries an empty date-shift
column, consistent with de-identification by date shifting; this explanation is inferred from the
file structure and is **Unverified** by the data provider. No date before 2001 exists in any date
column; earlier years inside note text are past medical history.

### What these records can and cannot support

They support extraction, a valve passport, descriptive statistics and the VARC-3 classification
logic. They **cannot** support adjudicated outcomes, landmark timing, exposure lags or model
training: roughly twenty unadjudicated event mentions, year-only dates, and almost no serial echo.
The physician's 20-note review is a pilot extraction audit; a proper validation needs an annotated
sample enriched for hard cases.

---

## 4. Extraction findings — Implemented

Produced by the extractor in this repository and covered by the test suite. Counts use different
denominators, stated in each row.

| Element | Count | Denominator and note |
|---|---|---|
| Implant documented in an operative or procedure note | 97 | Of 117 patients: 53 surgical, 44 transcatheter. A further 15 mention the implant only in history; 5 unspecified |
| Implant year stated explicitly | 86 | Of the 97; 2006 to 2022, two thirds in 2013 to 2018 |
| Valve model named, with generation | 110 | Of 117; 7 with no model identified |
| Label size | 85 | Of 117; mode 23 mm and 26 mm; ten patients at 21 mm or smaller |
| At least one prosthetic mean gradient in text | 46 | Of 117; one patient has gradients in two different years |

**Device and era are confounded in this sample.** All 19 dated Trifecta implants fall in 2013 to
2017, and 24 of 32 SAPIEN implants in 2018 to 2022. Any apparent device effect in these records is
inseparable from the year it was implanted, which is why no device conclusion is drawn from them.

### Extraction traps that produce wrong counts

Three failure modes were identified in these records and are guarded in the extractor and its tests.

1. **`Valve in Valve: No` is a structured field** in transcatheter operative reports. Matching the phrase rather than parsing the field value counts patients as valve-in-valve cases because their report says *No*. The field must be read as a key-value pair.
2. **`Reoperation: No previous surgeries`** behaves the same way.
3. **Native pre-operative gradients sit in the same notes as prosthetic ones.** Of the notes containing a mean gradient, most also contain pre-operative or native language. A series assembled without a context rule is not a trajectory; it is two different valves interleaved.

A fourth trap is worth naming because it is invisible: **Epic is both a St Jude valve and the vendor
of the electronic record**, whose name appears in page furniture throughout. Brand matching without
a valve-context requirement inflates that device's count by an order of magnitude.

---

## 5. Public sources used

No open, downloadable, patient-level dataset links valve identity, serial echo and adjudicated
deterioration. Registries publish curves and tables, never rows. The sources below fill the gaps
around that fact; all are tier 1, meaning downloadable now without registration.

| Source | Use |
|---|---|
| FDA AccessGUDID full release | Device identifiers, brands, models and sizes. Carries **no outcome data of any kind**; design class, leaflet mounting and generation are our own taxonomy layered on top |
| ASE 2024 prosthetic valve guideline | Reference values by model and size, used as context only, never as a substitute for the patient's own reference study |
| NOTION 10-year report | Curves and numbers at risk, for simulation parameters |
| FDA Summaries of Safety and Effectiveness Data | Per-size haemodynamics at protocol time points, device lineage. Deterioration counts in these documents read zero because approval follow-up is far too short |
| ClinicalTrials.gov API v2 | Structured outcomes by trial identifier; definitions cross-check |
| IPDfromKM | Reconstruction of pseudo-individual event times from digitised curves. Output is labelled reconstructed, never real |
| openFDA device event API | Reported counts by brand. Passive surveillance without denominators; a signal check only, never an incidence |

**Durability numbers come from published trial follow-ups and registry reports, not from any
database.** Definitions differ between them — reoperation-based deterioration and echo-based VARC-3
deterioration differ by roughly an order of magnitude — so every parameter carries its definition,
its source and a confidence grade, and parameters are never pooled across definitions without a
stated calibration factor.

---

## 6. Preprocessing

Deduplicate by patient, date and measurement. Normalise units. Parse free text with rules first and
a language-model pass second, with an audit sample. Extreme values are **flagged for source
verification, not rejected**: a gradient above 60 mmHg or an area below 0.6 cm² may be severe
dysfunction rather than a typing error.

Align every echo to the implant date. Define the reference study and the landmark windows. Take the
latest qualifying echo within each window as the index value. Map medication records from RxNorm to
class and assemble a per-patient exposure timeline.

Labels come from adjudication, not from rules alone: the VARC-3 module proposes a stage against the
patient's reference study and reviewers confirm, reclassify or mark uncertain. Event prevalence is
reported by class and by failure phenotype.

---

## 7. Synthetic scenarios

Simulation exists to show whether the model recovers what is in the data and stays quiet about what
is not. Each parameter is labelled literature-informed, assumed, or varied in sensitivity analysis.
Published sub-distribution hazard ratios are not used as cause-specific hazard parameters without
conversion and justification.

A synthetic dataset parameterised from a handful of published hazard ratios cannot establish
representativeness, particularly when the source studies use different endpoints and populations.
Simulation is therefore limited to demonstrating the analytical pipeline and testing robustness
under explicitly varied assumptions: failure-mechanism mix, competing mortality, follow-up intensity
and informativeness, and device effects.

**Synthetic performance must not be interpreted as evidence of clinical accuracy or of
transportability to contemporary valve recipients.** Every synthetic figure is titled as a software
demonstration.

---

## 8. Governance

The three supplied files were provided by the organisers for the event. They stay on team machines,
are excluded from version control by `.gitignore`, and only aggregates appear in the repository.
`scripts/privacy_scan.py` enforces this in continuous integration and before every deployment.

Placeholders such as `[NAME]` and `[DATE]` show the files were de-identified at source; publishing
counts adds no re-identification risk. Production would run under a data use agreement, a data
protection impact assessment and on-premises compute, with the extraction step audited for leakage
of names into structured fields.
