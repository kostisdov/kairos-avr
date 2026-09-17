# -*- coding: utf-8 -*-
"""
Task 6, audit-template bullet. Builds a physician-review audit template: 20 stratified
notes (6 SAVR operative reports, 6 TAVR procedure/operative notes, 6 progress notes
containing a gradient, 2 notes with an event mention), long format
(note_ref, field, extracted_value, physician_value, correct, comment), physician_value/
correct/comment left blank for manual completion.

Contains note_ref (which embeds Profile Key, a patient identifier) -> written ONLY to
data/derived/private/extraction_pilot_audit.csv. NO note text is written anywhere. A
companion ID-free copy of the *instructions* (not the data) goes to
docs/extraction_pilot_audit_instructions.md.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from kairos.passport import load_notes, load_device_patterns, extract_note_fields
from kairos.paths import repo_root

DATA_DIR = repo_root()
PRIVATE_DIR = DATA_DIR / "data" / "derived" / "private"
DOCS_DIR = DATA_DIR / "docs"

N_PER_STRATUM = {"savr_op": 6, "tavr_op": 6, "progress_gradient": 6, "event_mention": 2}


def main():
    notes = load_notes()
    model_patterns, device_meta = load_device_patterns()

    extracted = []
    for _, r in notes.iterrows():
        text = str(r["Notes"])
        fields = extract_note_fields(text, model_patterns)
        fields.update(note_ref=r["note_ref"], typ=r["Type"], year=int(r["Service Date"]))
        extracted.append(fields)
    ex = pd.DataFrame(extracted)

    savr_op = ex[(ex["typ"] == "Operative Report") & ex["savr"] & ~ex["tavr"]]
    tavr_op = ex[ex["typ"].isin(["Operative Report", "Procedures"]) & ex["tavr"]]
    prog_grad = ex[(ex["typ"] == "Progress Notes") & (ex["gradients"].apply(len) > 0)]
    any_event = ex["events"].apply(lambda d: any(d.values()))
    event_notes = ex[any_event]

    def sample(df, n, seed=42):
        return df.sample(n=min(n, len(df)), random_state=seed) if len(df) else df

    picks = {
        "SAVR operative report": sample(savr_op, N_PER_STRATUM["savr_op"]),
        "TAVR procedure/operative note": sample(tavr_op, N_PER_STRATUM["tavr_op"]),
        "Progress note with gradient": sample(prog_grad, N_PER_STRATUM["progress_gradient"]),
        "Note with event mention": sample(event_notes, N_PER_STRATUM["event_mention"]),
    }
    for label, df in picks.items():
        print(f"{label}: selected {len(df)} / pool {len(df)} "
              f"(pool sizes -- savr_op={len(savr_op)}, tavr_op={len(tavr_op)}, "
              f"prog_grad={len(prog_grad)}, event={len(event_notes)})")

    rows = []
    for stratum, df in picks.items():
        for _, r in df.iterrows():
            nref = r["note_ref"]
            rows.append((nref, "stratum", stratum))
            rows.append((nref, "note_type", r["typ"]))
            rows.append((nref, "service_year", r["year"]))
            rows.append((nref, "route_tavr_flag", r["tavr"]))
            rows.append((nref, "route_savr_flag", r["savr"]))
            rows.append((nref, "canonical_models_found", ";".join(sorted(r["models"])) or "(none)"))
            rows.append((nref, "sizes_found_mm", ";".join(str(s) for s in sorted(r["sizes"])) or "(none)"))
            if r["gradients"]:
                for i, (val, prosth, native) in enumerate(r["gradients"]):
                    ctx = "prosthetic" if prosth and not native else ("native" if native else "unclear")
                    rows.append((nref, f"mean_gradient_{i}_mmHg", val))
                    rows.append((nref, f"mean_gradient_{i}_context", ctx))
            if r["dvis"]:
                for i, (val, prosth, native) in enumerate(r["dvis"]):
                    rows.append((nref, f"dvi_{i}", val))
            if r["eoas"]:
                for i, (val, prosth, native) in enumerate(r["eoas"]):
                    rows.append((nref, f"eoa_{i}_cm2", val))
            if r["regurg"]:
                for i, (grade_ord, grade_word, prosth, native) in enumerate(r["regurg"]):
                    rows.append((nref, f"regurgitation_{i}_grade", grade_word))
            events_true = [k for k, v in r["events"].items() if v]
            rows.append((nref, "events_flagged", ";".join(events_true) or "(none)"))

    audit = pd.DataFrame(rows, columns=["note_ref", "field", "extracted_value"])
    audit["physician_value"] = ""
    audit["correct"] = ""
    audit["comment"] = ""

    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PRIVATE_DIR / "extraction_pilot_audit.csv"
    audit.to_csv(out_path, index=False)
    n_notes = audit["note_ref"].nunique()
    print(f"wrote audit template: {len(audit)} field rows across {n_notes} notes -> {out_path}")

    write_instructions(n_notes)


def write_instructions(n_notes):
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    text = f"""# Extraction pilot audit -- instructions for the reviewing physician

## What this is

A pilot accuracy check of the rule-based extraction in `src/kairos/passport.py`. We picked
{n_notes} notes, stratified as: 6 SAVR operative reports, 6 TAVR procedure/operative notes,
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
"""
    path = DOCS_DIR / "extraction_pilot_audit_instructions.md"
    path.write_text(text, encoding="utf-8")
    print(f"wrote ID-free instructions -> {path}")


if __name__ == "__main__":
    main()
