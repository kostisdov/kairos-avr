# Extraction pilot audit -- instructions for the reviewing physician

## What this is

A pilot accuracy check of the rule-based extraction in `src/kairos/passport.py`. We picked
20 notes, stratified as: 6 SAVR operative reports, 6 TAVR procedure/operative notes,
6 progress notes that contain at least one mean-gradient mention, and 2 notes flagged for
an event mention (valve-in-valve, redo, prosthetic dysfunction, endocarditis, or
thrombosis).

## Where the data file is (and why it isn't here)

The actual audit spreadsheet, `data/derived/private/extraction_pilot_audit.csv`, contains
a `note_ref` column (Profile Key + note index) that is itself a patient identifier, and
therefore lives only under `data/derived/private/` (gitignored, never committed) alongside
the other real patient-level extraction outputs -- never in `docs/`. This instructions file
is the ID-free, committable companion: what to do with that spreadsheet, not the spreadsheet
itself.

## How to review it

The spreadsheet is in long format: one row per (note, extracted field). For each `note_ref`
group:

1. Open the corresponding note in the source EHR/chart system using its `note_ref` (ask the
   data team which patient/note that maps to -- this mapping is intentionally not
   duplicated anywhere outside the private extraction outputs).
2. Read the `field` / `extracted_value` pairs for that note -- these are what the rule-based
   extractor found (e.g. `route_tavr_flag`, `canonical_models_found`, `sizes_found_mm`,
   `mean_gradient_0_mmHg` + `mean_gradient_0_context` [prosthetic/native/unclear],
   `dvi_0`, `eoa_0_cm2`, `regurgitation_0_grade`, `events_flagged`).
3. Fill in `physician_value` with what the note actually says for that field (leave blank
   if the extractor found nothing and that is correct, i.e. a true negative).
4. Fill in `correct` with `Y` / `N` / `partial`.
5. Use `comment` for anything the extractor got structurally wrong (wrong valve model,
   right number but wrong prosthetic/native attribution, missed a size, etc.) -- this
   feedback is what will drive the next iteration of the extraction rules.

## What NOT to do

Please don't paste note text into the `comment` field or anywhere else that might leave
`data/derived/private/`. A short paraphrase ("says 21mm not 23mm") is fine; a verbatim
quote is not needed and should be avoided.
