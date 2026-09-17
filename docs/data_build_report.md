# KAIROS data-plumbing build report

Prepared by the data engineer, Docathon Athens, 2026-09-16. Full timestamped detail is in
`docs/data_build_log.md`; this is the consolidated summary the team should read first.

## 1. What was downloaded

All files listed in `data/raw/MANIFEST.csv` (filename, url, sha256, bytes, downloaded_at,
license_or_terms, notes). Summary:

| File | Size | Status |
|---|---|---|
| `gudid_full_release_20260901.zip` (FDA AccessGUDID full release) | 519 MB | OK, 1st attempt |
| `ASE2024_prosthetic_valve_guideline.pdf` | 9.96 MB | OK |
| SAPIEN 3 SSED + 3 supplements (`P140031*.pdf`) | ~2.7 MB total | OK |
| CoreValve/Evolut SSED supplement (`P130021S174B.pdf`) | 0.5 MB | OK |
| Inspiris RESILIA SSED (`P150048B.pdf`) | 0.3 MB | OK |
| Avalus SSED (`P170006B.pdf`) | 0.2 MB | OK |
| INTUITY SSED (`P150036B.pdf`) | 0.2 MB | OK |
| SAPIEN 3 labeling (`P140031S182D.pdf`, pattern check) | 6.75 MB | OK, bonus find |
| Trifecta SSED (`P100029B.pdf`, PMA found via openFDA, not in task brief's URL list) | 0.65 MB | OK |
| Perceval SSED (`P150011B.pdf`, PMA found via openFDA) | 0.30 MB | OK |
| LOTUS Edge SSED (`P180029B.pdf` -- **task brief mislabelled this "Evolut R/PRO"; it is actually LOTUS Edge, Boston Scientific**) | 1.0 MB | OK |
| NOTION, Trifecta-vs-MagnaEase, Trifecta-vs-Perceval, IPDfromKM (PMC articles) | text only | Full text captured via browser render (PDF/HTML blocked, see failures below) |
| Mitroflow SSED (`P060038B.pdf`) | -- | **404** |
| Perimount/Magna Ease SSED (`P860057B.pdf`) | -- | **404** (1986 approval predates electronic SSED archive) |

Total downloaded: ~567 MB across 18 successful fetches. All PDFs remain in `data/raw/`
(never redistributed); only extracted numbers with citation appear in `data/reference/`.

## 2. Derived tables and row counts

| Table | Rows | Notes |
|---|---|---|
| `data/reference/device_table.csv` | 42 canonical models | exceeds 20-model floor; every acceptance-listed size range confirmed present |
| `data/reference/device_aliases.csv` | 68 | includes every free-text form named in the task brief |
| `data/reference/prosthetic_valve_reference_values.csv` | 252 | ASE 2024 Tables A1-A4 (aortic only); 10-cell spot-check all exact matches |
| `data/reference/published_rates.csv` | 52 | NOTION + 2 Trifecta papers, long format |
| `data/reference/fda_ssed_hemodynamics.csv` | 15 | SAPIEN 3, Inspiris RESILIA, Avalus |
| `data/reference/fda_ssed_events.csv` | 26 | SAPIEN 3, Inspiris RESILIA, Avalus, LOTUS Edge (+ bonus CoreValve comparator) |
| `data/reference/ctgov_durability_outcomes.csv` | 755 | 10 trials, durability-related outcome measures only |
| `data/reference/maude_counts_by_brand.csv` | 42 | openFDA MAUDE counts per canonical model; see caveat #11 below re: "+" in Evolut FX+/PRO+ |
| `data/derived/private/passport.csv` / `.parquet` | 117 patients | **private, not committable** |
| `data/derived/aggregates/*.csv` | 11 files | route/model/era/size/gradient-coverage/event/DVI-EOA-regurg/med-class/lab-availability, all `<5`-suppressed |
| `config/scenarios.yaml` | 6 scenarios | sourced where possible, explicit `TODO` elsewhere |

## 3. Acceptance checks

- Device table: >=20 canonical models -- **42, pass**. Every TAVR model with GUDID DIs
  present -- pass, with 3 documented exceptions (ACURATE neo/neo2 absent from this GUDID
  snapshot under any brand/company tried; Myval not FDA-approved so absent as expected).
  Sizes for SAPIEN 3 {20,23,26,29}, Evolut {23,26,29,34}, Magna Ease {19-29}, Trifecta
  {19-29}, Inspiris {19-29} -- **all confirmed programmatically, pass**.
- Reference values: 10-cell spot-check against raw PDF text -- **10/10 exact matches**.
  Coverage improved (coordinator Fix 2, 2026-09-16): 85/252 -> **92/256 rows mapped** to a
  canonical_model (recovered 4 real Perceval rows that a size-format parsing bug had
  silently discarded; added a low-confidence Prima->Prima Plus mapping). Magna and Magna
  Ease confirmed **genuinely absent** from the ASE 2024 guideline's own Table A4 (not a
  mapping bug -- verified by direct search of the raw PDF text). 25 of 42 device_table.csv
  canonical models still have zero reference-value rows, listed in
  `prosthetic_valve_reference_values.meta.yaml`.
- Passport extraction: **39/39** pytest tests pass (was 29/29) against **25** synthetic
  fixtures (was 16) covering every hard case named in the task brief plus 9 added for
  coordinator Fix 1 (CE+size-after, Magna free-text, guarded bare "S3"/"XT", Ultra-near-
  Sapien proximity, bare "Ultra RESILIA", bare "Carpentier", and a negative control mirroring
  the Epic-EHR regression test). Perimount-family (Perimount+Magna+Magna Ease) patient count:
  **16 -> 31**; generic-only SAPIEN (no generation resolved): **1 -> 0**. Full before/after
  in `docs/data_build_log.md`'s "Coordinator-requested corrections" section.
- No pre-2001 dates in any of the three source spreadsheets (earliest = 2001, medications
  Start Date) -- **confirmed**.
- ClinicalTrials.gov durability filter (coordinator Fix 3): `durability_related` column
  added to all 755 rows (file kept in full); **714 True / 41 False** -- investigated and
  confirmed correct against the coordinator's own stated category list (not force-fit to an
  estimated count; see `ctgov_durability_outcomes.meta.yaml` for the investigation).
  `ctgov_durability_outcomes_filtered.csv` (714 rows) added.
- FDA SSED P180029 (LOTUS Edge) relabelling (coordinator Fix 3): `document`, `canonical_model`
  ("Lotus Edge"), and a new `design_class` column (auto-populated for every row in both SSED
  tables) corrected/added -- **confirmed** in the rebuilt CSVs.
- Aggregate formatting (coordinator Fix 4): reproduced and fixed a real bug where a leading
  `#`-comment line broke plain `pandas.read_csv()` on every file in
  `data/derived/aggregates/` (not just the 2 named -- all 11 were affected). All aggregates
  now single-header CSVs with a `.meta.yaml` sidecar; **11/11 verified** by
  `src/kairos/verify_aggregates.py` to parse cleanly into their declared columns with no
  `Patient_` substrings.
- Every committable folder (`data/reference/`, `data/derived/aggregates/`, `config/`,
  `docs/`) scanned for `Patient_\d{3}` / `note_ref`-style patterns -- **zero matches,
  clean**.

## 4. Every 404 / extraction failure / manual follow-up

1. **Mitroflow SSED 404** (`P060038B.pdf`, PMA confirmed to exist via openFDA but no SSED
   PDF found at the expected accessdata.fda.gov path). Manual follow-up: search
   accessdata.fda.gov's PMA supplement list for P060038 directly.
2. **Perimount/Magna Ease SSED 404** (`P860057B.pdf`) -- expected: 1986 original approval
   predates FDA's electronic SSED archive. No action needed; the PMA number itself
   (P860057) is recorded with high confidence via openFDA.
3. **PMC PDFs blocked by Google reCAPTCHA Enterprise** for NOTION, Trifecta-vs-MagnaEase,
   Trifecta-vs-Perceval, IPDfromKM -- direct HTTP GET returns a challenge page, not content
   or a 404. Worked around via interactive browser render (full text captured, saved as
   `*_fulltext.txt`); the literal PDF was never obtained (its own download link resolves to
   a browser-only blob). **No CAPTCHA was solved or bypassed** -- a normal browser render is
   not blocked the way a scripted request is.
4. **Task brief error**: PMA P180029, listed as "Evolut R/PRO" in the task's own SSED URL
   list, is actually the **LOTUS Edge Valve System** (Boston Scientific) per the SSED
   document's own Device Trade Name field. Corrected in the manifest, filename, and
   `device_table.csv`; **no genuine Evolut R/PRO SSED was located this session** -- manual
   follow-up: search accessdata.fda.gov directly for Medtronic Evolut R/PRO's actual PMA
   number (likely a P130021 supplement, since CoreValve/Evolut R shares that base PMA per
   the task's own brief -- worth checking supplement letters there first).
5. **KM curve digitisation not performed**: every KM figure obtained this session (FDA SSED
   figures) is a raster image, not vector paths (confirmed via `pdfplumber` inspection); the
   three PMC papers' figures were never obtained in any form (their PDFs are the same
   reCAPTCHA-blocked resource as #3). `src/kairos/km_reconstruct.py` implements and
   self-tests the Guyot 2012 algorithm on synthetic data; `data/reference/published_curves/`
   is empty pending a team member manually digitising a real figure per
   `docs/km_digitisation_instructions.md` (WebPlotDigitizer steps + exact Python call
   included).
6. **Evolut R/PRO SSED**: see #4 -- genuinely not found this session under the URL given.
7. **ACURATE neo/neo2 and Myval**: zero GUDID device records found this session (documented
   as likely real gaps, not bugs -- see `data/reference/device_table.meta.yaml`
   `known_gaps` section). **Independently corroborated** by the completed openFDA MAUDE
   pull: ACURATE neo shows 0 total reports and ACURATE neo2 shows 2, essentially zero
   footprint in either FDA data source queried this session -- two independent systems
   agreeing makes "genuinely sparse in US FDA data" more likely than "extraction bug in
   one of them."
8. **Physician audit template**: `data/derived/private/extraction_pilot_audit.csv` (20
   notes, 245 field rows) is ready for clinician review but has **not been reviewed by a
   physician yet** -- this is the single most important manual follow-up before trusting
   `passport.csv` for modelling. Instructions (ID-free) are in
   `docs/extraction_pilot_audit_instructions.md`.
9. **VARC-3 stage 0/1 boundary**: the task brief and every source gathered this session give
   exact numeric thresholds for stage 2 and stage 3 only; stage 0 vs 1 is this module's own
   documented interpolation (any detectable worsening vs none), not a sourced numeric
   threshold -- flagged in `src/kairos/varc3.py`'s docstring per the task's instruction to
   say so rather than guess silently.
10. **Scenario config**: `high_thrombosis_TAVR` and `opposite_direction_antithrombotic_effects`
    are almost entirely `TODO` -- no source gathered this session gives usable thrombosis-
    specific incidence or antithrombotic-effect-direction numbers (NOTION reported *zero*
    thrombosis events in both arms, which is the opposite of useful for a high-thrombosis
    stress scenario). Needs a dedicated literature pull.
11. **openFDA query artifact**: in the completed `maude_counts_by_brand.csv`, "Evolut FX+"
    and "Evolut FX" return identical counts (11,305 total / 1,839 SVD-like), as do
    "Evolut PRO+" and "Evolut PRO" (10,607 / 2,672) -- the literal "+" character appears to
    be dropped or treated as a query operator by openFDA's search parser rather than matched
    literally, so the "+"-suffixed query silently degrades to its base-name search. Treat
    each pair as ONE number, not two independently confirmed generation-level counts; every
    other row is unaffected (no other canonical_model name contains "+").

## 5. Exact commands to re-run everything

```bash
# from D:\Dyania, with the packages listed in the task brief installed
# (pip install pdfplumber pymupdf requests pyyaml pyarrow pytest rapidfuzz)

# Task 1 (downloads) -- see docs/data_build_log.md for the exact GUDID URL used
#   (data/raw/gudid, data/raw/fda_ssed, data/raw/papers populated by ad hoc curl/requests
#    calls logged in docs/data_build_log.md; re-download by re-running the fetch() calls
#    documented there, or re-use the already-downloaded files -- they do not expire)

# Task 2
python src/kairos/build_gudid_extract.py        # -> data/raw/gudid/gudid_valve_records.jsonl
python src/kairos/build_device_table.py         # -> data/reference/device_table.csv, device_aliases.csv

# Task 3
python src/kairos/build_reference_values.py     # -> data/reference/prosthetic_valve_reference_values.csv

# Task 4
python src/kairos/build_published_rates.py      # -> data/reference/published_rates.csv
python src/kairos/build_fda_ssed_tables.py      # -> data/reference/fda_ssed_hemodynamics.csv, fda_ssed_events.csv
python src/kairos/km_reconstruct.py             # self-test only (no real curve digitised yet)

# Task 5
python src/kairos/build_ctgov_outcomes.py       # -> data/reference/ctgov_durability_outcomes.csv
python src/kairos/build_openfda_maude_counts.py # -> data/reference/maude_counts_by_brand.csv

# Task 6 (run in this order -- passport.py must run before the aggregates/audit scripts)
python -m src.kairos.passport                   # -> data/derived/private/passport.csv, .parquet
python src/kairos/build_passport_aggregates.py  # -> data/derived/aggregates/*.csv
python src/kairos/build_medication_exposure.py  # -> private exposure grid + aggregate prevalence
python src/kairos/build_labs_analysis.py        # -> private INR-in-range + aggregate lab availability
python src/kairos/build_audit_template.py       # -> private audit CSV + docs instructions

# Task 7
python -m pytest tests/ -v                      # 29 tests, all passing as of this build

# Task 8: config/scenarios.yaml is hand-authored (no build script -- values cite the tables
# built above directly; re-check against updated data/reference/*.csv if those change).
```

## Five-line summary

Downloaded and verified 18 public sources (~567 MB: GUDID full release, ASE 2024 guideline,
10 FDA SSEDs, 4 journal papers) with every hash and license logged in `MANIFEST.csv`, and
corrected one task-brief error (P180029 is LOTUS Edge, not Evolut R/PRO). Built a
42-model device reference table (cross-validated against an independently-run openFDA MAUDE
pull -- both agree ACURATE has near-zero US FDA footprint), a 252-row ASE reference-value
table (10/10 spot-check exact), and 755+52+41 rows of published/FDA/ClinicalTrials.gov
durability numbers, all cited and confidence-flagged. Ran the real local extraction on all
117 patients' notes (202
non-deleted notes), producing a private per-patient passport plus 11 fully de-identified,
small-cell-suppressed aggregate tables -- catching and fixing two real precision bugs
(Epic-EHR false positives; SAPIEN-family over-generalisation) along the way, both now
regression-tested. VARC-3 staging is implemented, cited, and tested (29/29 tests pass); the
scenario-config skeleton is populated wherever real data supports it and left `TODO`
elsewhere rather than guessed. Two manual follow-ups matter most for the team: get a
physician to review `extraction_pilot_audit.csv`, and digitise one real KM curve per
`docs/km_digitisation_instructions.md` once someone has bandwidth.


## Correction 2026-09-16 17:44 UTC: LOTUS classification and alias durability

- `Lotus` (LOTUS Edge Valve System) reclassified from "self-expanding intra-annular TAVR" to "mechanically expanded intra-annular TAVR" in the curated table inside `src/kairos/build_device_table.py`; `device_table.csv`, `fda_ssed_events.csv` and `fda_ssed_hemodynamics.csv` regenerated and now carry the corrected class.
- The free-text alias forms added during the earlier correction round (size-before-CE, bare Carpentier, sized Magna, guarded bare S3 and XT, Sapien-Ultra proximity, bare Ultra RESILIA) had lived only in `device_aliases.csv` and were lost on regeneration; they are now part of the curated list in the build script, so any future rebuild keeps them. `device_aliases.csv` now has 75 rows.
- `passport.py` and the aggregate tables were regenerated; counts unchanged from the corrected run (Perimount 28, Magna 3, Trifecta 21, Trifecta GT 8, SAPIEN 3 17, SAPIEN 3 Ultra 15, 7 patients without a named model). pytest: 39 passed.

## Phase A of the model change design (17 September 2026): endpoint version 2 before/after

Generator 2.0 with endpoint version 2 (`docs/kairos_model_change_detailed_design.md`, WP-A1 to WP-A6).
Full-namespace cohorts (configured size, 2500 requested per scenario, seed 20260916), landmark tables built
under each label policy with `scripts/label_policy_report.py --namespace full`; the CSV is
`artifacts/metrics/phaseA/label_policies_full.csv`. Counts are **unique patients** with an SVD event after
at least one landmark (not landmark rows). Synthetic scenarios: illustrative, unvalidated.

| scenario | legacy single study | primary | uncertain negative | uncertain positive | primary SAVR | primary TAVR | non-SVD replacement | death | 5-year attendance | missed threshold crossings | median detection delay (days) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gradual_stenotic | 144 | 89 | 92 | 145 | 73 | 16 | 33 | 1459 | 0.92 | 94 | 204 |
| regurgitant_abrupt | 222 | 172 | 174 | 223 | 134 | 38 | 33 | 1429 | 0.92 | 94 | 187 |
| high_competing_mortality | 7 | 8 | 8 | 9 | 3 | 5 | 6 | 1969 | 0.91 | 5 | 65 |
| irregular_surveillance | 159 | 103 | 106 | 160 | 84 | 19 | 33 | 1450 | 0.66 | 111 | 210 |
| biomarker_information/meaningful | 167 | 113 | 116 | 168 | 91 | 22 | 33 | 1454 | 0.92 | 102 | 191 |
| biomarker_information/weak | 127 | 89 | 91 | 129 | 72 | 17 | 34 | 1461 | 0.92 | 84 | 215 |
| biomarker_information/absent | 115 | 85 | 87 | 115 | 70 | 15 | 34 | 1462 | 0.92 | 67 | 208 |
| biomarker_information/unmeasured | 167 | 113 | 116 | 168 | 91 | 22 | 33 | 1454 | 0.92 | 102 | 191 |
| anticoagulant_mechanism_confounding/marker_mediated | 202 | 145 | 147 | 206 | 117 | 28 | 33 | 1448 | 0.92 | 107 | 190 |
| anticoagulant_mechanism_confounding/marker_noise | 165 | 114 | 116 | 170 | 93 | 21 | 33 | 1466 | 0.92 | 101 | 191 |

Reading the table:

- Requiring confirmation removes 25 to 38 % of SVD event patients relative to the legacy single-study label.
  Almost all of the loss is unconfirmed last-study candidates (the patient died, was replaced more than 455
  days later or reached the end of follow-up before another study); censoring at an earlier unresolved
  candidate (primary versus uncertain-negative) moves only 2 or 3 patients.
- `biomarker_information/meaningful` and `/unmeasured` have identical event counts: they differ only in which
  laboratory values are measured, and generator 2.0 keeps latent disease in its own random stream.
- Event support for Phase B gates (30 unique events per cause and route, assumed): TAVR SVD reaches 30 only in
  `regurgitant_abrupt`; non-SVD replacement (about 33 patients in total, both routes together) reaches it in
  no scenario; `high_competing_mortality` has 8 SVD event patients. The sizing pilot (WP-B2) has to settle
  cohort sizes before any comparative claim.
- Missed threshold crossings (noise-free trajectory meets stage 2/3 but no adjudicated endpoint) are about as
  frequent as detected endpoints, mostly because death or the end of follow-up comes first. Detection delay is
  measured from the noise-free crossing; it can be negative when noise produces an early confirmed positive.

## Phase B of the model change design (17 September 2026): probability, support, calibration, sensitivity

Synthetic scenarios only: illustrative, unvalidated. Quick-namespace numbers are smoke tests under relaxed gates.

**Integration change (WP-B1).** `scripts/cif_convention_report.py --namespace full` (core_plus_both fitted on each
full cohort, 1000 sampled rows, `artifacts/metrics/phaseB/cif_convention_full.csv`). Largest absolute difference
against the current contract (step baselines, piecewise-constant hazards) over all ten cohorts:

| comparison | state | 1 y mean / max | 3 y mean / max | 5 y mean / max |
|---|---|---|---|---|
| version 1 (interpolated weekly hazards, product limit) | SVD | 0.00003 / 0.0145 | 0.0002 / 0.0068 | 0.0001 / 0.0038 |
| version 1 | death | 0.0016 / 0.0055 | 0.0019 / 0.0027 | 0.0007 / 0.0023 |
| product limit on the new step hazards | SVD | 0.00001 / 0.0067 | 0.00002 / 0.0026 | 0.00004 / 0.0031 |
| product limit on the new step hazards | death | 0.0004 / 0.0019 | 0.0004 / 0.0013 | 0.0003 / 0.0018 |

The discretisation check against an ODE reference (generator-shaped hazards, weekly knots) stays below 1e-4
(`tests/test_cif_contract.py`), so `cif_refine` stays 1.

**Sizing pilot (WP-B2).** `cli.py size-pilot --all --n 2000 --dev-seeds 3` (development seeds 9100-9102, 10th
percentile over 15 folds, safety factor 1.25). Cohort size each gate needs, rounded up to 100:

| scenario | SVD covariate model | SVD route baselines (SAVR / TAVR) | replacement covariate model | SVD AUC at 1 y (20 cases per held-out fold) | SVD slope at 5 y (50 cases) | proposed n |
|---|---|---|---|---|---|---|
| gradual_stenotic | 1200 | 300 / 1100 | 4300 | 12500 | 9100 | 12500 |
| regurgitant_abrupt | 700 | 200 / 600 | 4300 | 5600 | 4700 | 5600 |
| high_competing_mortality | 15000 | not estimable / 6300 | 15000 | not estimable | 312500 | none |
| irregular_surveillance | 1000 | 300 / 1100 | 4300 | 16700 | 7400 | 16700 |
| biomarker_information/meaningful | 900 | 200 / 900 | 4300 | 9700 | 6500 | 9700 |
| biomarker_information/weak | 1100 | 300 / 1300 | 4200 | 9300 | 8000 | 9300 |
| biomarker_information/absent | 1300 | 300 / 1400 | 4200 | 10000 | 9400 | 10000 |
| biomarker_information/unmeasured | 900 | 200 / 900 | 4300 | 9700 | 6500 | 9700 |
| anticoagulant_mechanism_confounding/marker_mediated | 800 | 200 / 800 | 4500 | 6800 | 5900 | 6800 |
| anticoagulant_mechanism_confounding/marker_noise | 900 | 200 / 900 | 4500 | 6300 | 6800 | 6800 |

The binding gate is the 1-year SVD metric support, not the model fit. `high_competing_mortality` has no 1-year SVD
case in any pilot fold at 2000 patients, so no size can be extrapolated; per the design it can only report the unmet
gate or become a separately named assumed variant. One Cox fit of core_plus_both on a 1600-patient training fold took
7 to 11 s. The plan (`config/evaluation_plan.yaml`) is written **unfrozen**; full-namespace evaluation refuses to run
until the owner sets sizes and freezes it.

**Four-state ladder, quick namespace (WP-B2/B3 exit gate).** `cli.py evaluate --all --quick --namespace quick` (600
patients, 3 folds, 30 bootstraps, quick gates): 2640 rows (11 steps x 3 horizons x 4 states x 2 weightings per cohort),
integration `pch-expm1/1`, every row exploratory. Row coverage is 100 % in the two anticoagulant cohorts, 80 % in
seven others and 56 % in `high_competing_mortality`: in some folds one route has no non-SVD replacement event in the training rows, and those
held-out rows are reported as `insufficient_events:replacement` instead of receiving a zero hazard. The Brier
decomposition reconstructs to 3e-16. SVD at 5 years, core_plus_both, gradual_stenotic: 11 event patients, IPCW Brier
0.023, logistic calibration slope 0.51 (1 = ideal), AUC 0.61; non-SVD replacement: 2 event patients, AUC and slope not
estimated (`insufficient_events`). These numbers illustrate the reporting only; no model claim follows from them.

**Observation sensitivity (WP-B4), quick example.** `evaluate --scenario irregular_surveillance --quick --namespace quick
--sensitivity observation` (579 common patients; all 19 adjudicated patients have a defensible onset interval, median
width 417 days). SVD at 3 years, core_plus_both, pooled rows:

| analysis | IPCW Brier | observed (Aalen-Johansen) | SVD event patients |
|---|---|---|---|
| primary (detection) | 0.0143 | 0.014 | 13 |
| interval, right end (= detection) | 0.0143 | 0.014 | 13 |
| interval, midpoint | 0.0211 | 0.021 | 15 |
| interval, left end | 0.0241 | 0.024 | 15 |
| interval, uniform (3 draws, range) | 0.0200 to 0.0206 | 0.020 | 15 |
| oracle (noise-free threshold crossing) | 0.0422 | 0.048 | 33 |
| missing onset (assumption) | 0.0369 | 0.043 | 31 |
| regular visits (paired cohort) | 0.0144 | 0.015 | 14 |
| informative visits, inverse visit-intensity weights | 0.0135 | 0.014 | 13 |

Five-year attendance was 0.76 with regular and 0.64 with informative attendance on identical latent histories; the
visit-intensity weights kept 97 % effective sample size with 1 to 3 % truncated. Detection lagged the noise-free
crossing by a median 168 days, and 28 of 47 crossings were never adjudicated. The sensitivity ranges are assumptions,
not bounds.

## Phase C of the model change design (17 September 2026): gradient boosting family

Synthetic scenarios only: illustrative, unvalidated.

**Dependency (WP-C0).** scikit-survival 0.28.0 resolves against `requirements.lock.txt` without changing any locked
version, for Windows (installed in `.venv`) and for the images' `python:3.11-slim` (Debian bookworm, glibc 2.36;
manylinux_2_28 wheels). Added pins: scikit-survival 0.28.0, ecos 2.0.14, osqp 1.1.3, numexpr 2.14.2, setuptools
84.0.0 (Linux dependency of osqp). The images were not rebuilt in this session.

**Compute benchmark (WP-C0).** `scripts/boosting_benchmark.py --tune`, gradual_stenotic full cohort, core_plus_both,
one training fold (1912 patients, 8549 rows), full grid (100/300 trees, learning rate 0.03/0.1, depth 1/2):

| | library Cox loss | exact O(n log n) loss (deviation 30) |
|---|---|---|
| Cox fit | 4.8 s | 6.8 s (machine busy) |
| boosting fit, 300 trees, depth 2, all route x cause models | 238 s | 31 s |
| inner tuning of one outer fold (4 fits x 3 inner folds) | 1250 s | 216 s |
| projected boosting time, 10 cohorts x 11 steps x 5 folds, single process, 2500 patients | 227 h | 38 h |

Replacing the loss changes no fitted value (predictions agree to 4e-16). At the evaluation plan's proposed sizes
(5600 to 16700 patients) the projection grows further, so the full comparison was **not started**: it needs a compute
budget (parallel folds or an Azure job) from the owner, as the design requires.

**Whole-cohort bundles.** `cli.py train --family both` on the full gradual_stenotic cohort: Cox (served, all causes `ok`)
and gradient boosting (tuned in 243 s on three patient-grouped inner folds: 300 trees, learning rate 0.1, depth 2;
SVD boosted for SAVR, covariate-free baseline for TAVR with fewer than 30 event patients; replacement baseline-only).
The boosting bundle is stored under `models/<version>/gradient_boosting/` and is not served.

**Quick paired comparison (WP-C3 exit).** `evaluate --scenario gradual_stenotic --quick --namespace quick --families cox
gradient_boosting` (600 patients, 3 outer folds, quick grid 30/60 trees x depth 1/2, 30 paired bootstraps), run
`eval-20260917T093954-0c5956`. Boosting predicted every row Cox predicted; comparison on the 2084 common rows of 2617.
core_plus_both, gradient boosting minus Cox (negative favours boosting):

| state | horizon | Brier Cox | Brier boosting | difference (95 % interval) | support |
|---|---|---|---|---|---|
| SVD | 1 y | 0.0020 | 0.0020 | 0.0000 | insufficient_events |
| SVD | 3 y | 0.0115 | 0.0116 | 0.0001 (-0.0005 to 0.0009) | ok (quick gates) |
| SVD | 5 y | 0.0229 | 0.0230 | 0.0001 (-0.0013 to 0.0019) | ok (quick gates) |
| SVD | mean | | | 0.0000 (-0.0006 to 0.0009) | |
| death | 5 y | 0.2023 | 0.2197 | 0.0174 (0.0032 to 0.0273) | ok (quick gates) |

Decision: `retain_cox`, evidence `exploratory` (quick namespace, unfrozen plan). The first attempt of this run failed
outer fold 1 of every boosting step because inner folds without replacement events had no scoreable row; after the
owner's decision of 17 September 2026 (deviation 24) such inner folds are skipped, and all 33 outer fits completed with
tuned hyperparameters. No conclusion about the families follows from a 600-patient smoke run.

## Calibrated generator (design `kairos_calibrated_generator_design.md`, config `generator_calibration.yaml`)

Real-evidence calibration: bundle `cal-de1f3ebaa493`, status `insufficient_evidence`. No standardization snapshot exists
and none of the 48 proposed published-rate candidates has owner review, so all were excluded. Generation from that
bundle is refused. No evidence claim follows.

Synthetic parameter recovery (machinery test only, targets `approved_synthetic_test`): bundle `cal-417fa144a6c1`,
n=2000, budget 60, free `death_annual_at_79` and `p_diabetes`, truth 0.15 / 0.42. Fitted 0.1533 / 0.4108; status
`accepted_for_simulation`, structural checks passed, held-out target passed; 649 s.

| target | truth-seed estimate | calibrated cohort (seed 3201) | within tolerance |
|---|---|---|---|
| death KM 3y | 0.5489 | 0.5437 | yes |
| death KM 5y | 0.3874 | 0.3828 | yes |
| SVD CIF 5y | 0.0183 | 0.0247 | yes |
| diabetes proportion | 0.4063 | 0.4115 | yes |
| death CIF 4y (holdout) | 0.5225 | 0.5317 | yes |

Downstream: `generate-calibrated` produced 2 replicates (1864 and 1870 reference-eligible of 2000); `validate-cohort`
passed structure and the training contract (7308 landmark rows); `train-calibrated` (Cox) and `evaluate-calibrated
--quick` wrote to `models/experimental/` and `metrics/experimental/` only (ladder 264 rows). Privacy is `not_assessed`
because the recovery bundle has no real snapshot.

## Phase D and time-boxed Phase E (17 September 2026)

Phase D: reliability reason codes and the `ErrorDetail` contract; `?family=` and `GET /models` in the predict service;
grouped local sensitivity (`modelling/explain.py`); demo family selector and "Compare families" tab; run-scoped model
store with `publish` / `publish --rollback`; model cards; deviations 39-46. Tests: 262 passed, ruff clean, schemas
regenerated. The privacy scan lists only the local private folders and spreadsheets that stay on this machine.

Time-boxed family comparison (deviation 46; exploratory, biased towards boosting, **not the release comparison**):
full gradual_stenotic cohort, 5 outer folds, budget tuple, 3 steps in parallel processes, about 20 minutes wall time.
Runs `eval-20260917T110855-4b466e` (reference), `-5e5094` (core), `-58679d` (core_plus_both). SVD Brier score:

| step | family | 1 y | 3 y | 5 y |
|---|---|---|---|---|
| reference | Cox | 0.0033 | 0.0196 | 0.0386 |
| reference | boosting | 0.0033 | 0.0196 | 0.0386 |
| core | Cox | 0.0032 | 0.0177 | 0.0354 |
| core | boosting | 0.0032 | 0.0179 | 0.0358 |
| core_plus_both | Cox | 0.0031 | 0.0173 | 0.0350 |
| core_plus_both | boosting | 0.0032 | 0.0180 | 0.0360 |

Boosting did not improve on Cox at any step, even with the bias in its favour. Paired `compare` decision for all three steps: `retain_cox` (exploratory: no frozen plan). The served family stays Cox.
