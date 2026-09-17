# KAIROS data build log

Running log of the data-plumbing build (device reference table, reference values,
published durability numbers, local extraction, VARC-3 module, config skeleton).
Timestamps UTC. No patient-level values or note text appear in this file.

---

- **2026-09-16T10:12Z** — Session start. Reviewed `open_data_research.md` (verified source
  register) and `src/kairos/extract_passport_v0.py` (working rule-based prototype extractor).
  Confirmed environment: Python 3.11.6, pandas 2.2.3, openpyxl 3.1.5, pdfplumber, pymupdf
  (`fitz`), requests, pyyaml, pytest, rapidfuzz already installed; installed pyarrow 25.0.1.
  `Rscript` not found on PATH -> no R/IPDfromKM available, will implement Guyot 2012
  KM-reconstruction algorithm in pure Python per the task's documented fallback.
  Network connectivity to clinicaltrials.gov confirmed (HTTP 200).
- **2026-09-16T10:13Z** — Created folder layout under `D:\Dyania`: `data\raw\{gudid,fda_ssed,
  ctgov,openfda,papers}`, `data\reference\published_curves`, `data\derived\{private,
  aggregates}`, `config`, `src\kairos`, `tests\fixtures\synthetic_notes`, `docs`. Wrote
  `.gitignore` (`*.xlsx`, `data/raw/`, `data/derived/private/`, `tmp/`, `__pycache__/`,
  `.ipynb_checkpoints/`). Repo is NOT git-initialised per instructions (files only).
- **2026-09-16T10:15Z** — TASK 1 downloads. Fetched AccessGUDID download page; latest full
  release is `gudid_full_release_20260901.zip` (519 MB; the register's assumption of a
  pipe-delimited format was wrong -- it is 209 `FULLDownload_PartN_Of_209` XML files, ~5.2M
  device records total). Downloaded successfully on attempt 1 (sha256 recorded in
  `data/raw/MANIFEST.csv`). Verified FDA product codes for replacement heart valves via the
  openFDA classification endpoint (mirrors the FDA product classification database) rather
  than assuming: **DYE = "Replacement Heart-Valve" (surgical, class 3)** and
  **NPT = "Aortic Valve, Prosthesis, Percutaneously Delivered" (TAVR, class 3)** both
  confirmed. Started a streaming XML extraction (`tmp/gudid_extract.py`, iterparse over each
  zip member, filters on productCode in {DYE,NPT} OR a curated valve-brand keyword regex)
  running in background.
- **2026-09-16T10:17Z** — Downloaded ASE 2024 guideline PDF (9.96 MB, matches expected ~9.5MB)
  and all 10 targeted FDA SSED PDFs (SAPIEN 3 original + 3 supplements, CoreValve/Evolut
  supplement S174, Evolut R/PRO original, Inspiris RESILIA original, Avalus original, INTUITY
  original, plus the P140031S182D labeling-pattern check which also succeeded) -- all on the
  first attempt, all recorded in MANIFEST.csv with sha256.
- **2026-09-16T10:18Z** — The 4 PMC article URLs (NOTION 10y, Trifecta-vs-MagnaEase,
  Trifecta-vs-Perceval, IPDfromKM) returned HTTP 200 to a programmatic `requests` GET, but
  the response body was a **Google reCAPTCHA Enterprise challenge page**, not article content
  (confirmed by inspecting the saved bytes). Per the standing rule against bypassing/solving
  CAPTCHAs, no attempt was made to solve it. Instead used the interactive Browser tool (a real
  browser render, not a CAPTCHA workaround) to view each article -- all 4 rendered as full
  open-access text with no challenge. Also checked the PMC "Download PDF" button: it resolves
  to a same-origin blob URL and is not fetchable by a non-browser client or via
  `read_network_requests` (response body not exposed for that request type) -- documented in
  MANIFEST.csv as blocked, not a true 404. Captured the complete rendered article text for all
  4 papers to `data/raw/papers/*_fulltext.txt` (NOTION and Trifecta-vs-MagnaEase in full;
  Trifecta-vs-Perceval and IPDfromKM truncated after their numeric/methods content was fully
  captured, per PMC's page-render length). Deleted the 4 reCAPTCHA placeholder .html files and
  corrected their MANIFEST.csv rows to reflect the block reason.
- **2026-09-16T10:35Z** — Searched the openFDA PMA endpoint (`/device/pma.json`) for PMA
  numbers not in the task's URL list, rather than guessing SSED filenames: found **Trifecta =
  P100029** (Abbott Medical/St Jude), **Perceval = P150011** (Corcym -- contradicts the
  register's "no PMA confirmed" note; a PMA and SSED do exist, approved 2016-01-08), **Mitroflow
  = P060038** (Corcym Canada/Sorin), **Perimount/Magna Ease = P860057** (Edwards, original
  approval 1986). Downloaded `P100029B.pdf` (Trifecta original SSED, 655 KB) and `P150011B.pdf`
  (Perceval original SSED, 303 KB) successfully on the first constructed-URL attempt. Attempted
  `P060038B.pdf` (Mitroflow) and `P860057B.pdf` (Perimount) at the analogous `pdfYY/` paths --
  **both 404**, recorded in MANIFEST.csv; Perimount's 1986 approval predates FDA's electronic
  SSED archive so this is expected, Mitroflow's absence is unexplained and left as an open
  manual follow-up (try accessdata.fda.gov's PMA search UI directly, or the supplement list
  for P060038, which mostly shows post-2009 process-change supplements with no original "B"
  labelled SSED PDF found by this filename pattern).
- **2026-09-16T10:40Z** — TASK 2 device table. Extracted the GUDID zip's 209 XML part files
  (`src/kairos/build_gudid_extract.py`, streaming iterparse, ~12.3 min for 5,205,424 device
  records) filtering on FDA product code DYE/NPT or a target-manufacturer regex ->
  12,976 candidate records to `data/raw/gudid/gudid_valve_records.jsonl`. Iterated the
  precision of that filter: a first pass using a loose brand-keyword regex OR loose
  company-substring match pulled in ~13,000 irrelevant orthopedic/dental/hearing-aid/
  glucose-monitor devices (e.g. Exactech "Epic" hip plates, Abbott Diabetes Care "FreeStyle
  Libre", Ortho Organizers "Lotus Plus" orthodontic tubes) because generic English words in
  valve brand names (Epic, Lotus, Freestyle, and a substring bug where "evolut" matched
  inside "Evolution") collide with unrelated product lines from the same or
  similarly-named companies. Fixed by requiring GUDID's own GMDN preferred-term field to be
  one of the three AORTIC-position heart-valve-bioprosthesis terms (cleanly separates real
  aortic valves from mitral/tricuspid valves, annuloplasty rings, sizers, delivery
  catheters, crimpers -- all of which share a product code or company with the valves but
  carry a different GMDN term), narrowing the target-company regex (e.g. "Abbott Medical" /
  "Abbott Cardiovascular", not bare "Abbott", which also matches Abbott Diabetes Care), and
  a final brandName-only accessory-keyword exclusion (careful to check brandName only, not
  deviceDescription -- a real valve's own description can legitimately say "...Premounted
  on Delivery System", which would have wrongly excluded the genuine Boston Scientific
  LOTUS Edge valve DIs if checked against deviceDescription too). Result: 509 clean
  aortic-valve device records. `src/kairos/build_device_table.py` matches each record's
  brandName/deviceDescription (after stripping GUDID's mojibake (R)/(TM) glyphs, U+00AE/
  U+2122, which were breaking several regexes, e.g. "Hancock(TM)II" with no space) against
  a per-canonical-model regex, extracts Primary DI + sizes-in-mm (from deviceSizes, falling
  back to deviceDescription, falling back to a trailing 2-digit number in
  versionModelNumber/catalogNumber for Corcym Mitroflow/Crown PRT/Perceval DIs that carry
  no other machine-readable size field, e.g. "DLA27"->27mm), and merges with a hand-curated
  static table of manufacturer/design_class/tissue/leaflet_mounting/tissue_treatment/
  generation/market_status, citing an openFDA PMA-number lookup wherever one was actually
  run this session (Trifecta=P100029, Perceval=P150011 [a PMA DOES exist, correcting the
  source register's uncertainty], Mitroflow=P060038, Perimount/Magna/Magna Ease=P860057,
  Portico/Navitor=P190023, Freestyle=P970031, plus the 6 PMAs already given in the task
  brief). Wrote `data/reference/device_table.csv` (42 canonical models -- exceeds the
  20-model acceptance floor), `data/reference/device_aliases.csv` (68 rows, including every
  free-text form named in the task brief: "#21 Trifecta", "21-mm Trifecta", "Trifecta GT",
  "Sapien S3", "S3 Ultra", "Edwards S3", "Size: Ultra 26mm", "Evolut Corevalue Pro+"
  misspelling, "R34mm", "CE"+size for Perimount, "Magna", "Inspiris", "Konect"), and
  `data/reference/device_table.meta.yaml`. Acceptance check: SAPIEN 3 sizes {20,23,26,29}
  present, Evolut sizes {23,26,29,34} present (on Evolut FX/PRO+), Magna Ease sizes
  {19,21,23,25,27,29} present, Trifecta and Inspiris RESILIA both {19..29} present -- all
  confirmed programmatically. Documented gaps (zero GUDID match this session, confidence
  "low", manual-verification note in the source column, no fabricated values): ACURATE
  neo/neo2 (genuinely absent from this GUDID snapshot under any brand/company tried, despite
  Boston Scientific being present for Lotus), Myval (not FDA-approved / CE-only, absence
  expected), Trifecta GT / Epic Supra / Toronto SPV (no distinguishing GUDID brand text
  found), Perceval Plus (GUDID sizes are letter-coded S/M/L/XL only, no numeric mm anywhere
  in its DI record -- left blank rather than inventing an mm equivalence).
- **2026-09-16T10:55Z** — TASK 3 reference values. `src/kairos/build_reference_values.py`
  uses pdfplumber `extract_text()` (not `extract_tables()`, which silently dropped/
  misordered rows on this borderless-table PDF) on ASE 2024 guideline pages 50-54
  (Tables A1 SAPIEN-family TAVR, A2 CoreValve/Evolut R TAVR, A3 percutaneous aortic
  valve-in-valve, A4 surgical aortic -- confirmed Table A5 onward is mitral/pulmonary/
  tricuspid, out of KAIROS's aortic scope, not extracted). Discovered and worked around a
  PDF font-encoding quirk: "±" renders as a literal "6" between two numbers (e.g.
  "19.1±8.2" extracts as "19.168.2") -- confirmed via every table's "mean±SD" caption and
  by spot-checking that recovered SDs are always small and clinically plausible. Also fixed
  a block-parsing bug (multi-word valve names split across two PDF text lines, e.g.
  "Abbott"/"Trifecta", were being mis-read as one combined bogus valve name). Table A4
  rows missing peak or mean gradient (101/252 rows) are classified by magnitude (EOA
  always <4.5, gradients always >=4.0 in this table) and a lone gradient value is recorded
  as mean_gradient with confidence="low" rather than guessed as peak or invented. Wrote
  `data/reference/prosthetic_valve_reference_values.csv` (252 rows: 137 high, 14 medium,
  101 low confidence, 0 unparseable/skipped) and its `.meta.yaml`. Hand-verified 10 cells
  against the raw PDF page text (SAPIEN/SAPIEN3/CoreValve from A1/A2, CoreValve+SAPIEN3
  ViV from A3, Perimount/Epic/Hancock II/Mitroflow/Avalus from A4) -- all 10 exact matches,
  zero discrepancies, logged in the meta.yaml. 85/252 rows map to a task-2 canonical_model;
  the rest are legacy/mechanical valves correctly out of device_table.csv's scope.
- **2026-09-16T11:10Z** — TASK 4 published durability numbers.
  `src/kairos/build_published_rates.py` transcribes every SVD/NSVD/BVD/BVF/mortality/
  endocarditis/thrombosis/reintervention figure from the three already-captured PMC papers
  into `data/reference/published_rates.csv` (52 rows; 22 high, 16 medium, 14 low
  confidence). Flagged rather than silently resolved: NOTION's Abstract labels 20.5%/43.0%
  as "severe NSVD" while its Results body labels the SAME two numbers "severe BVD" and
  separately gives 10.2%/31.9% as "severe NSVD" -- an apparent inconsistency in the source
  article itself; both readings are recorded with a note, not merged or guessed.
  Kaplan-Meier point estimates reported only as mean+/-SE (Suzuki 2022, Nardi 2026) were
  NOT converted to a 95% CI (would require assuming normality and multiplying by ~1.96 --
  a transformation, not an extraction); ci_low/ci_high left blank for those rows instead.
  `src/kairos/build_fda_ssed_tables.py` produced `fda_ssed_hemodynamics.csv` (15 rows: per-
  size 1-year mean gradient/EOA for Inspiris RESILIA and Avalus from their pivotal-trial
  SSED tables, plus SAPIEN 3's pooled baseline/30-day/1-year trend) and `fda_ssed_events.csv`
  (26 rows: SAPIEN 3 CEC-adjudicated 30-day/1-year composite+death+stroke+AI, Inspiris
  RESILIA's explicit "0 cases of ... structural valve deterioration" at 1 year, Avalus's
  endocarditis late-linearized rate, and LOTUS Edge's 1-year composite effectiveness
  endpoint). **Correction to the task brief**: PMA P180029, listed in the task's own source
  list as "Evolut R/PRO", is confirmed by the SSED document's own Device Trade Name field to
  actually be the **LOTUS Edge Valve System (Boston Scientific)** -- filename and
  MANIFEST.csv corrected, device_table.csv's Lotus row upgraded to fda_pma=P180029/
  confidence=high, and the SSED used as Lotus (and bonus CoreValve-comparator, since Lotus's
  own pivotal trial used CoreValve as an active control) data instead. No genuine Evolut
  R/PRO SSED was located this session -- left as a manual follow-up.
  Checked for vector KM-curve data before falling back to manual digitisation, per the
  task's own instruction: inspected `page.curves`/`page.rects`/`page.images` on the actual
  KM-figure pages of the SAPIEN 3 and LOTUS Edge SSEDs -- every figure is a single embedded
  raster image (e.g. SAPIEN 3 p.37: one 362x228px image, zero vector curves), confirming
  there is no coordinate data to extract programmatically; the three PMC papers could not be
  obtained as PDFs at all this session (same reCAPTCHA block as before), so their figures
  were never available in any form. Implemented `src/kairos/km_reconstruct.py` (pure-Python
  Guyot 2012 "iKM" algorithm, since Rscript is not on PATH in this environment --
  confirmed), self-tested against a synthetic example (100 synthetic patients, reconstructed
  curve tracks the digitised input to within ~0.005 at every point, all patients accounted
  for as exactly one event or one censoring). Documented one explicit simplification vs the
  original paper (even, not shape-weighted, distribution of excess censoring within a
  risk-table interval) in the module docstring. Wrote
  `docs/km_digitisation_instructions.md` (WebPlotDigitizer steps + CSV templates + exact
  Python call) since no real curve could be digitised this session --
  `data/reference/published_curves/` is therefore empty pending that manual step; every file
  written there in future must carry `reconstructed_not_real` in its name per the
  non-negotiable rules.
- **2026-09-16T11:25Z** — TASK 5 (ClinicalTrials.gov half). Searched CT.gov API v2
  `query.titles` (not a trusted pre-existing list) for each named trial family and picked
  10 representative NCT numbers actually returned: NCT01757665 COMMENCE, NCT01256710
  Trifecta Durability Study, NCT03016169 Trifecta GT PMCF, NCT02675114 PARTNER 3,
  NCT03222128 PARTNER II S3 Intermediate, NCT01057173 NOTION, NCT01240902 CoreValve US
  Pivotal (High/Very High Risk), NCT01586910 SURTAVI, NCT02701283 Evolut Low Risk,
  NCT03666741 Inspiris RESILIA Durability Registry. Downloaded full JSON for each to
  `data/raw/ctgov/`. Two studies (NOTION, Inspiris Durability Registry) confirmed to have
  **no `resultsSection` posted at all** on CT.gov (their results live in journal
  publications instead, consistent with what open_data_research.md already found for
  NOTION) -- not an extraction failure. Fixed a real bug found during development: the
  CT.gov API v2 stores each outcome measure's arm/group labels and denominators PER
  OUTCOME MEASURE (`om["groups"]`, `om["denoms"]`), not in a shared module-level lookup as
  first assumed -- the first run produced opaque "OG000"-style arm codes instead of names
  until this was found by inspecting the raw JSON directly. `src/kairos/build_ctgov_outcomes.py`
  flattens every outcome measure whose title/description matches a durability-related regex
  (SVD, valve deterioration, reintervention, ViV, hemodynamics, thrombosis, valve failure,
  regurgitation) into `data/reference/ctgov_durability_outcomes.csv`: 755 rows, with
  resolved arm names and true denominators (e.g. COMMENCE's per-size N for the
  "Structural Valve Deterioration" and "Average Mean Gradient" outcome measures).
- **2026-09-16T11:30Z** — TASK 5 (openFDA half). Confirmed the FDA product classification
  facts already logged (DYE/NPT) apply here too. Building `maude_counts_by_brand.csv`
  required first discovering openFDA MAUDE's actual coded `product_problems` vocabulary
  (there is no literal "structural valve deterioration" code) by running
  `search=device.brand_name:"trifecta"&count=product_problems.exact` and keeping every
  plausibly-SVD/thrombotic term from the real observed frequency table (top entries:
  "Material Split, Cut or Torn" 539, "Obstruction of Flow" 319, "Calcified" 275,
  "Backflow" 272, "Device Stenosis" 222, "Intravalvular regurgitation" 196, "Gradient
  Increase" 157, "Perivalvular Leak" 108, ...). Also found and fixed a subtler bug: writing
  the query string with a literal "+AND+" and passing it through `requests.get(params=...)`
  double-encodes the "+" to a literal "%2B", which openFDA does not parse as AND -- every
  such query silently returned a false "404 No matches found" even for terms confirmed
  present (e.g. brand_name:"trifecta" AND product_problems:"Calcified", known-present at
  n=275, 404'd until the query was rewritten with a real space character and left to
  `requests`'s own encoder). `src/kairos/build_openfda_maude_counts.py` queries both
  reports_total and reports_svd_like for all 42 device_table.csv canonical models, 1.6s
  apart (~37 req/min, under the 40/min ceiling), tagging single-common-word brand names
  (Epic, Lotus, Mosaic, Freestyle, Crown PRT, Prima Plus) with a "valve" qualifier to
  reduce the same cross-domain collision risk found earlier in GUDID. Every row carries
  caveat="reported counts, not incidence; passive surveillance, no denominator" per
  NON-NEGOTIABLE guidance and open_data_research.md section 4.
- **2026-09-16T11:40Z** — TASK 6 core extraction. `src/kairos/passport.py` extends
  extract_passport_v0.py: device_aliases.csv-driven model matching (using only the
  alias_type='regex'/'abbreviation' rows, which are the free-text forms curated for real
  notes -- formal `gudid_brand` label strings are not matched against clinical text), joined
  canonical_model/design_class/route from device_table.csv, added size_mm/implant_year, and
  NEW extraction for dimensionless index (DVI), effective orifice area (EOA), and
  regurgitation grade (word- and numeric-grade forms), none of which existed in v0. Serial
  (year-tagged, prosthetic-vs-native-flagged) gradient/DVI/EOA/regurgitation time series are
  stored as a JSON list per patient so the output stays one-row-per-patient.
  **QA finding and fix** (found by comparing route/model distributions against v0 and then
  digging in with regex-only diagnostics -- never printed note text): "Epic" was matching
  27/117 patients as their canonical_model, wildly implausible for a less-common porcine
  valve. Diagnosis (via hit-count/context statistics only): of the 20 Operative Report notes
  containing the word "Epic", ZERO showed the expected "Tissue Implant Type: ... Epic"
  device-field pattern, while 11/36 total matches across the cohort showed explicit
  EHR-system phrasing ("documented in Epic", "Epic chart/flowsheet/record") -- these are the
  clinic's own EHR system (Epic Systems), not the Abbott/St Jude Epic valve. Added a
  context-validation guard (`_AMBIGUOUS_WORD_MODELS`, currently just {"Epic"}) requiring
  nearby valve-specific wording and absence of EHR-reference wording before counting a match
  for models flagged this way. Result after fix: Epic dropped to 3 patients, and the
  previously-masked real distribution surfaced (Trifecta, SAPIEN family, Perimount, CoreValve
  etc. all became visible as primary models instead of being crowded out).
  **Second QA finding and fix**: passport.py's simpler alias-driven matching (unlike
  build_device_table.py's ordered/lookahead-protected patterns) has no mutual exclusion
  between a model family's generic and specific names, so a note mentioning "SAPIEN 3" also
  matches the bare "SAPIEN" pattern; picking `sorted(models)[0]` alphabetically then
  selected the LESS specific "SAPIEN" every time (short strings that are prefixes of longer
  ones sort first). Fixed by selecting the longest (most specific) matched canonical_model
  name per patient instead -- valid across this device-naming convention because a specific
  model name always contains its generic relative as a substring (SAPIEN -> SAPIEN 3 ->
  SAPIEN 3 Ultra; Evolut -> Evolut R/PRO/FX; Magna -> Magna Ease; etc.). Caught by
  tests/test_passport.py::test_sapien3_ultra_size_field against the synthetic fixtures,
  which is exactly what that test suite is for. Real-data impact: canonical_model
  distribution went from {SAPIEN:33, Trifecta:29, Epic:27(bug), Perimount:11, ...} to the
  correctly-disambiguated {Trifecta:21, SAPIEN 3:18, Perimount:13, SAPIEN 3 Ultra:13,
  Trifecta GT:8, CoreValve:5, SAPIEN XT:4, Magna:3, Inspiris RESILIA:3, Epic:3, Evolut PRO:2,
  SAPIEN:1, Freestyle:1} -- substantially more clinically informative (distinguishing SAPIEN
  3 / 3 Ultra / XT / original matters for a durability model, since tissue generation
  differs).
  Confirmed exactly 13 notes excluded by Signed Status (9 Operative Report + 4 Progress
  Notes; the task brief names only the 9 operative ones, but this module excludes ALL
  Deleted-status rows as sounder general data hygiene -- a superset of the explicit
  instruction, documented in the module docstring). Confirmed no pre-2001 dates in any of
  the three spreadsheets' date columns (earliest = 2001, medications Start Date); recorded
  here as the task's requested verification.
  Wrote `data/derived/private/passport.csv` + `.parquet` (117 patients), aggregates to
  `data/derived/aggregates/` (patients_by_route, patients_by_canonical_model_and_
  design_class, implant_era_by_design_class, size_distribution, prosthetic_gradient_
  coverage, event_mention_counts, dvi_eoa_regurg_coverage -- all with the "<5" small-cell
  rule applied before writing), medication exposure grid/durations (private) +
  class-prevalence-by-year/duration-distribution (aggregate, <5-suppressed) from
  `src/kairos/build_medication_exposure.py`, warfarin INR time-in-range (private) + key-lab
  patient-year availability (aggregate, <5-suppressed) from `src/kairos/build_labs_analysis.py`
  (11,818/43,550 lab rows excluded first as respiratory-therapy/device-interrogation/
  free-text-transcription noise), and the physician audit template (20 stratified notes, 245
  field rows, private -- contains note_ref/patient IDs) plus its ID-free instructions
  companion in docs/, from `src/kairos/build_audit_template.py`.
- **2026-09-16T11:50Z** — TASK 7 (VARC-3 + fixtures + tests). `src/kairos/varc3.py`
  implements VARC-3 haemodynamic valve deterioration staging, quoting the task brief's own
  stage-2/stage-3 thresholds verbatim in the docstring (independently corroborated by the
  identical phrasing captured in NOTION's own VARC-3 methods section this session) and
  explicitly documenting that the stage-0/1 boundary is this module's own reasonable
  interpolation, not a sourced numeric threshold (full VARC-3 stage 1 also includes
  morphological/imaging criteria this gradient-only function cannot assess). Supports an
  optional `reference_fallback` (ASE-2024-table stand-in) when no patient baseline echo
  exists, flagging `reference_source="ASE 2024 table"` and `confidence="low"` when used.
  Wrote 16 synthetic fixture notes (tests/fixtures/synthetic_notes/, entirely fabricated,
  no real content) covering every hard case named in the task brief plus two devised during
  development: a deliberate "Epic" EHR-system-only mention (regression test for the false-
  positive bug found and fixed in passport.py) and a missing-reference-echo case. 29/29
  pytest tests pass (`python -m pytest tests/ -v`), including two tests that caught and
  drove the fix of the SAPIEN-family over-generalisation bug described above.
- **2026-09-16T11:55Z** — TASK 8 config skeleton. `config/scenarios.yaml`: 6 scenarios,
  every numeric field either cites a specific row in a data/reference/ table built in Tasks
  4-5 (with `source`/`definition_used`/`confidence`) or is explicitly `TODO` with a note on
  where such evidence might be found -- no invented values. `high_thrombosis_TAVR` and
  `opposite_direction_antithrombotic_effects` are almost entirely TODO: no source gathered
  this session gives usable thrombosis-incidence or antithrombotic-effect-direction numbers
  (NOTION reported *zero* clinical valve thrombosis events in both arms at 10 years, which
  is the opposite of useful for parameterising a high-thrombosis stress scenario).
  `externally_mounted_pericardial_early_failure` is the best-sourced non-baseline scenario,
  built directly from the two Trifecta papers' Cox-regression hazard ratios.
- **2026-09-16T11:56Z** — Final compliance pass: scanned every committable folder
  (data/reference/, data/derived/aggregates/, config/, src/, tests/, docs/) for
  `Patient_\d{3}` and note_ref-style patterns -- zero matches. Verified every
  data/reference/*.csv has a source/definition/retrieved_at column or a `.meta.yaml`
  sidecar (added two sidecars this pass -- ctgov_durability_outcomes.meta.yaml and
  fda_ssed_tables.meta.yaml -- that were initially missing). Full pytest suite re-confirmed
  passing (29/29) after all fixes. Wrote docs/data_build_report.md (Task 9).
  **Status at report time**: Tasks 1-4 and 6-8 fully complete. Task 5's ClinicalTrials.gov
  half is complete (755 rows). Task 5's openFDA MAUDE brand-count half
  (data/reference/maude_counts_by_brand.csv) is still running in the background at report
  time -- confirmed actively progressing throughout (steadily increasing CPU time over 20+
  minutes of wall-clock time), not stuck or crashed, just slower than estimated because each
  of the 42 canonical models needs 2 openFDA queries (one of them a 21-term OR clause across
  the `product_problems` field, which appears to be expensive for openFDA's backend to
  evaluate) paced at 1.6s apart to stay under the 40-requests/minute ceiling. It will finish
  and write the file without further action; if it has not landed by the time this report is
  read, re-run `python src/kairos/build_openfda_maude_counts.py` (idempotent, safe to
  re-run) or consider simplifying the per-brand SVD-like query to fewer, highest-signal
  terms if the current one continues to run long.
- **2026-09-16T14:39Z** — TASK 5 (openFDA half) COMPLETED. The background run finished
  successfully after ~25 minutes wall-clock for 84 paced openFDA requests (1.6s apart, 42
  models x 2 queries each -- see the T11:30Z entry for why the queries are individually
  slow: a 21-term OR clause on `product_problems` appears expensive for openFDA's backend).
  Wrote `data/reference/maude_counts_by_brand.csv`, 42/42 rows, zero failed/blank counts.
  Sanity checks: legacy high-volume devices (SAPIEN, Perimount, SAPIEN 3, Magna) show the
  highest cumulative report counts, as expected for older/more-implanted devices under a
  passive-surveillance system with no denominator (consistent with the caveat column on
  every row). ACURATE neo shows 0 total reports and ACURATE neo2 shows 2 -- independently
  corroborating this session's earlier GUDID finding that ACURATE has essentially no
  footprint in the FDA data ecosystem queried this session (both findings support "genuinely
  sparse in US FDA sources", not a bug in either extraction).
  **Known minor artifact, documented not silently ignored**: "Evolut FX+" and "Evolut FX"
  return IDENTICAL counts (11,305 total / 1,839 SVD-like), as do "Evolut PRO+" and
  "Evolut PRO" (10,607 / 2,672) -- strongly suggesting openFDA's query parser drops or
  treats the literal "+" character as a no-op/operator rather than matching it literally, so
  the "+"-suffixed query silently degrades to its base-name search. This means the FX/FX+
  and PRO/PRO+ generation-level split in maude_counts_by_brand.csv is not reliable --
  treat those 4 rows as two duplicate pairs (use either "Evolut FX" or "Evolut FX+"'s
  number, not both, and likewise for PRO/PRO+) rather than as independently confirmed
  per-generation counts. All other rows are unaffected (no other canonical_model name in
  device_table.csv contains a "+" character).

---

## Coordinator-requested corrections, 2026-09-16 (post-report)

- **FIX 1 (passport.py extraction regression).** Root cause investigation: the coordinator's
  cited numbers ("13 Perimount, under-5 Magna", "33 generic SAPIEN") turned out to come from
  a STALE aggregate file generated before the Epic-false-positive and SAPIEN-specificity
  fixes already logged above -- the live passport.csv at the time already had generic
  SAPIEN=1 and Perimount-family=16. That said, the coordinator's requested pattern additions
  were real, valid gaps independent of that staleness. Refactored load_device_patterns()/
  extract_note_fields() from "one merged regex per canonical model" to "a list of (pattern,
  guarded) tuples per canonical model", generalising the Epic-style context-guard mechanism
  so INDIVIDUAL alias rows (not just a whole canonical model's base name) can require nearby
  valve/TAVR context and no EHR-context -- needed because some new patterns (bare "S3", bare
  "XT") are exactly as collision-prone as bare "Epic" was, while their sibling full-name
  forms ("SAPIEN 3", "Sapien S3") are not and must not be guarded.
  Added to device_aliases.csv: size-then-CE forms ("#25 CE", "25 mm CE") mapping to
  Perimount (the pre-existing entry only covered CE-then-size); "Magna valve" and "#21 mm
  Magna" (redundant with the base "Magna" pattern but added for explicit documentation per
  the request); guarded bare "S3" mapping to SAPIEN 3 and bare "XT" mapping to SAPIEN XT;
  a proximity regex for "Ultra" appearing near "Sapien"/"Sapien 3"/"S3" in either order (not
  just the pre-existing contiguous "S3 Ultra" phrase) mapping to SAPIEN 3 Ultra; bare "Ultra
  RESILIA" mapping to SAPIEN 3 Ultra RESILIA. Also added bare "Carpentier"/"Carpentier-
  Edwards" mapping to Perimount (unguarded) to close the remaining Perimount-family gap
  toward v0's combined "perimount OR magna OR carpentier" bucket count -- empirically
  validated first (regex hit-counting only, no note text logged): of the 14 notes this newly
  matches, 14/14 have "aortic" nearby and 0/14 are mitral-only context, so the Carpentier-
  Edwards-mitral-bioprosthesis collision risk that motivated caution did not materialise in
  this cohort (documented as a residual risk on future data).
  Added 9 new synthetic fixtures (17-25) and 9 new regression tests, including a deliberate
  negative control (bare "S3"/"XT" as a room number / lab order code, no valve context --
  must NOT match) mirroring the existing Epic-EHR negative control. 39/39 tests pass.
  Real-data result: Perimount-family (Perimount+Magna+Magna Ease) primary-canonical_model
  count went 16 to 20 (pattern additions alone) to 31 (plus bare carpentier). Generic-only
  SAPIEN (no generation resolved) went from 1 to 0. Full before/after table is in the reply
  to the coordinator. Re-ran build_passport_aggregates.py (not re-run since the original
  Epic/specificity fixes either -- this run captures ALL of that improvement at once): the
  "(none identified)" canonical_model bucket shrank from 22 to 7 patients as a direct
  consequence.

- **FIX 2 (reference-value mapping gaps).** Investigated why Magna, Magna Ease, and Perceval
  had zero mapped rows: direct search of the raw ASE 2024 PDF text (pages 52-57, Table A4)
  found Magna and Magna Ease genuinely have ZERO rows in the source table at all (confirmed
  absent, not a bug -- only the base "Perimount" entry exists for that family). Perceval,
  however, DOES have 4 real rows in the source (sizes printed as letter-codes with
  parenthesized mm, "S(21)"/"M(23)"/"L(25)"/"XL(27)") that were being silently discarded: the
  block parser's size-token regex only recognised bare 2-digit numbers, so
  "SorinPercevalSutureless S(21) 10.1 6 4.2 1.3 6 0.3" (the printed plus-minus renders as a
  literal "6", see the pdf_quirk note above) matched neither the name+first-row nor size-row
  patterns and fell into the name-continuation branch, corrupting (and ultimately discarding,
  once a later valid entry reset parser state) rather than mis-attributing that data. Fixed
  by extending the size-token regex to accept a letter-code-with-parenthesized-digits form
  and normalising it to the bare mm number for size_mm. Recovered all 4 Perceval rows (sizes
  21/23/25/27mm).
  Also fixed an internal inconsistency: a prior mapping labelled bare "Hancock" as
  canonical_model="Hancock II (or original Hancock, unconfirmed)" -- a string that isn't an
  actual device_table.csv canonical_model name, which silently broke clean coverage-by-set-
  membership accounting. Re-pointed it to the literal "Hancock II" with a new
  UNCERTAIN_CANONICAL_MAPPINGS confidence-downgrade mechanism (also used for a new, low-
  confidence "Prima" mapping to "Prima Plus", since device_table.csv's bonus "Prima Plus"
  canonical model otherwise had zero reference rows and "Prima" is plausibly the same
  Edwards stentless platform, generational relationship unverified).
  Added valve_model_clean to every row (camelCase-boundary and paren-spacing cleanup of the
  PDF-concatenation artifact, with a preserve-list for genuinely-camelCase brand names like
  "CoreValve" so they are not incorrectly split).
  Result: 85/252 rows mapped became 92/256 (net +4 real Perceval rows recovered, +3 Prima
  Plus rows, +0 for Magna/Magna Ease since none exist to map). 25 of device_table.csv's 42
  canonical models still have zero reference-value rows (full list in
  prosthetic_valve_reference_values.meta.yaml and the reply to the coordinator) -- mostly
  newer-generation devices (Ultra/Plus/FX/RESILIA variants, Navitor/Portico/Lotus/JenaValve/
  ACURATE TAVR platforms) that postdate, or were never itemised in, this 2024 guideline's
  appendix tables.

- **FIX 3 (ClinicalTrials.gov and FDA SSED tables).** Added a durability_related boolean to
  every row of the existing 755-row ctgov_durability_outcomes.csv (kept in full) using the
  coordinator's exact category list (structural/hemodynamic valve deterioration,
  bioprosthetic valve dysfunction or failure, reintervention, explant, valve thrombosis,
  mean gradient, EOA, DVI, regurgitation), and wrote the True-only subset to the new
  ctgov_durability_outcomes_filtered.csv. Re-used the already-downloaded JSON in
  data/raw/ctgov/ rather than re-fetching (no new network calls).
  Result differs materially from the coordinator's estimate: 714/755 rows are
  durability_related=True (41 False), not roughly 187/755. Investigated rather than
  silently accepted or force-fit: spot-checked the largest True-driving outcome titles
  against their own outcome_description text -- "The Occurrence of Individual MACCE
  Components" explicitly lists reintervention, defined as any cardiac surgery or
  percutaneous reintervention catheter procedure that repairs, alters, or replaces a
  previously implanted valve, as a component; "Prosthetic Valve Dysfunction (PVD)" is
  explicitly defined via a mean-gradient-over-35-mmHg / EOA-under-0.8-cm2 threshold. Both
  are genuinely durability-related by the coordinator's own literal category list once
  outcome DESCRIPTIONS (not just titles) are searched, which the instruction wording
  ("outcome title or description concerns...") explicitly calls for. Did not narrow the
  regex to artificially hit the estimated count -- flagged the discrepancy and its cause
  instead, in the CSV's meta.yaml and the reply.
  Corrected the P180029 SSED document label throughout: file renamed to
  data/raw/fda_ssed/P180029B_LOTUS_Edge_original_SSED.pdf (MANIFEST.csv updated),
  fda_ssed_events.csv and fda_ssed_hemodynamics.csv now cite that filename with
  canonical_model="Lotus Edge" for its own rows (vs device_table.csv's broader "Lotus"
  bucket name -- both refer to the same product; "Lotus Edge" is the more specific label the
  coordinator asked these SSED-specific tables to use) and a new design_class column
  (auto-populated for every row, all documents, by joining canonical_model against
  device_table.csv, with an explicit alias so "Lotus Edge" resolves to device_table.csv's
  "Lotus" row's "self-expanding intra-annular TAVR"). Confirmed again that no genuine Evolut
  R/PRO SSED was located this session.

- **FIX 4 (aggregate formatting).** Confirmed and reproduced the exact failure mode: a naive
  pandas.read_csv() on the old-format aggregate files (which had a comment-marked provenance
  line directly above the CSV header, inside the same file) silently treated the comment
  line as the header row and the real header as the first data row -- reading
  implant_era_by_design_class.csv this way returned the comment text split on commas as the
  column names, not era/design_class/n_patients. Root cause: 3 separate build scripts
  (build_passport_aggregates.py, build_medication_exposure.py, build_labs_analysis.py) all
  independently wrote the comment line then the dataframe into the same file -- affecting
  all 11 files in data/derived/aggregates/, not just the 2 the coordinator named explicitly
  (which said "at least"). Wrote a shared writer, src/kairos/agg_io.py: write_aggregate()
  writes a clean single-header CSV (no comment line) plus a matching .meta.yaml sidecar
  carrying source/definition/row_count/small_cell_suppression/retrieved_at/privacy_note, and
  self-verifies on every call (raises if the just-written CSV doesn't round-trip through a
  plain pandas.read_csv() into exactly its own declared columns, or contains a "Patient_"
  substring) -- so this class of regression cannot silently reappear. Refactored all 3
  scripts to use it; re-ran all three. implant_era_by_design_class.csv now has its era
  column correctly, and patients_by_canonical_model_and_design_class.csv column order is
  now canonical_model, design_class, n_patients per the coordinator's explicit spec (was
  design_class, canonical_model, n_patients). Wrote src/kairos/verify_aggregates.py (the
  requested re-verification script): checks every aggregate CSV parses with plain pandas
  into its meta.yaml-declared columns and contains no "Patient_" substring; exits non-zero
  on any failure. All 11 files pass.

Full-suite re-verification after all four fixes: python -m pytest tests/ -v gives 39/39
passed. Privacy scan (grep for the Patient_NNN pattern across data/reference/,
data/derived/aggregates/, config/, src/, tests/, docs/) gives zero matches, clean.


## Correction 2026-09-16 17:44 UTC: LOTUS classification and alias durability

- `Lotus` (LOTUS Edge Valve System) reclassified from "self-expanding intra-annular TAVR" to "mechanically expanded intra-annular TAVR" in the curated table inside `src/kairos/build_device_table.py`; `device_table.csv`, `fda_ssed_events.csv` and `fda_ssed_hemodynamics.csv` regenerated and now carry the corrected class.
- The free-text alias forms added during the earlier correction round (size-before-CE, bare Carpentier, sized Magna, guarded bare S3 and XT, Sapien-Ultra proximity, bare Ultra RESILIA) had lived only in `device_aliases.csv` and were lost on regeneration; they are now part of the curated list in the build script, so any future rebuild keeps them. `device_aliases.csv` now has 75 rows.
- `passport.py` and the aggregate tables were regenerated; counts unchanged from the corrected run (Perimount 28, Magna 3, Trifecta 21, Trifecta GT 8, SAPIEN 3 17, SAPIEN 3 Ultra 15, 7 patients without a named model). pytest: 39 passed.
