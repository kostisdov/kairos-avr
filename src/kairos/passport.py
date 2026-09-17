# -*- coding: utf-8 -*-
"""
Per-patient "valve passport" extraction for KAIROS, from the de-identified notes.

Extends src/kairos/extract_passport_v0.py (the working rule-based prototype -- this module
keeps its proven route/model/size/gradient regexes) with:
  - device_aliases.csv-driven canonical-model matching (Task 2's output) instead of a
    hardcoded brand dict
  - canonical_model / design_class / route joined from device_table.csv
  - size_mm and implant_year fields
  - dimensionless index (DVI), effective orifice area (EOA, valve area), and
    regurgitation-grade extraction (new -- v0 did not extract these)
  - full year-tagged time SERIES for gradient/DVI/EOA/regurgitation (not just a max/count),
    stored as a JSON list per patient so a single row per patient still captures serial data
  - event-mention flags: valve-in-valve, redo, prosthetic dysfunction, endocarditis,
    thrombosis

PRIVACY (NON-NEGOTIABLE RULES 1-2): reads notes_deidentified.xlsx directly. Never prints or
writes note text anywhere -- only extracted field values (numbers, booleans, short
categorical strings, and a `note_ref` = "{Profile Key}_{note index}" identifier, which is
itself patient-level and therefore only ever written under data/derived/private/).
Nine operative reports (plus, as a data-hygiene generalisation of that instruction, all
other rows) with Signed Status "Deleted" are excluded from extraction -- 13 rows total (9
Operative Report + 4 Progress Notes), confirmed by direct crosstab against the source file;
see docs/data_build_log.md.

This module is a plain function library, not a script with side effects on import --
call `main()` (or `python -m kairos.passport`) to actually read the spreadsheet and write
outputs.
"""
import json
import os
import re
from pathlib import Path

import pandas as pd

# Repository root: resolved from kairos.paths (KAIROS_REPO_ROOT overrides); /app in containers.
# Override with KAIROS_REPO_ROOT. (Extended 2026-09-16 for the Azure build; behaviour on
# the build machine is unchanged.)
DATA_DIR = Path(os.environ.get("KAIROS_REPO_ROOT", Path(__file__).resolve().parents[2]))
PRIVATE_DIR = DATA_DIR / "data" / "derived" / "private"
REFERENCE_DIR = DATA_DIR / "data" / "reference"
NOTES_XLSX = DATA_DIR / "notes_deidentified.xlsx"

EXCLUDE_SIGNED_STATUS = {"Deleted"}
IMPLANT_NOTE_TYPES = {"Operative Report", "Procedures"}

# ---------------------------------------------------------------------------
# Regex patterns (route/model/size/gradient carried over from extract_passport_v0.py;
# DVI/EOA/regurgitation are new for this module)
# ---------------------------------------------------------------------------
TAVR_PAT = re.compile(
    r"transcatheter aortic valve|\bTAVR\b|\bTAVI\b|sapien|evolut|corevalve|corevalue", re.I)
SAVR_PAT = re.compile(
    r"aortic valve replacement with|\bAVR\b\s*\(?#|sternotomy|bioprosthesis via|"
    r"tissue implant type|replacement type: tissue", re.I)

SIZE_PATS = [
    re.compile(r"#\s?(\d{2})(?:\s?-?\s?mm)?\s?(?:trifecta|magna|perimount|CE\b|inspiris|"
               r"konect|valve|bioprosth)", re.I),
    re.compile(r"(\d{2})\s?-?\s?mm\s+(?:trifecta|magna|perimount|inspiris|edwards|sapien|"
               r"evolut|bioprosthesis|valve)", re.I),
    re.compile(r"implant size:\s*(\d{2})", re.I),
    re.compile(r"size:\s*(?:ultra\s*|R)?(\d{2})\s?mm", re.I),
    re.compile(r"size #\s?(\d{2})", re.I),
    re.compile(r"\(#(\d{2})\)", re.I),
    re.compile(r"\bR\s?(\d{2})\s?mm\b", re.I),  # Evolut-family shorthand, e.g. "R34mm"
]

MG_PATS = [
    re.compile(r"mean gradient (?:is|of|was|=)?\s*(\d{1,3})\s*mm\s?hg", re.I),
    re.compile(r"peak/mean gradients?\s*(?:of|were|was|:)?\s*\d{1,3}/(\d{1,3})", re.I),
    re.compile(r"gradients?:?\s*(?:of\s*)?\d{1,3}/(\d{1,3})\s*mm\s?hg", re.I),
    re.compile(r"\bMG\s*(?:of|is|=)?\s*(\d{1,3})\s*mm", re.I),
    re.compile(r"mean (?:PG|pressure gradient)[^0-9]{0,12}(\d{1,3})", re.I),
]

DVI_PATS = [
    re.compile(r"dimensionless (?:valve )?index(?:\s*\(DVI\))?\s*(?:is|of|was|=|:)?\s*"
               r"(0?\.\d{1,2})", re.I),
    re.compile(r"\bDVI\b\s*(?:is|of|was|=|:)?\s*(0?\.\d{1,2})", re.I),
    re.compile(r"dimensionless velocity index\s*(?:is|of|was|=|:)?\s*(0?\.\d{1,2})", re.I),
]

EOA_PATS = [
    re.compile(r"(?:effective orifice area|EOA)\s*(?:\([^)]{0,20}\))?\s*"
               r"(?:is|of|was|=|:)?\s*(\d\.\d{1,2})\s*cm", re.I),
    re.compile(r"(?:aortic valve area|AVA)\s*(?:is|of|was|=|:)?\s*(\d\.\d{1,2})\s*cm", re.I),
    re.compile(r"valve area\s*(?:is|of|was|=|:)?\s*(\d\.\d{1,2})\s*cm", re.I),
]

_REGURG_WORD = r"(none|trivial|trace|mild(?:-to-moderate)?|moderate(?:-to-severe)?|severe)"
REGURG_PATS_WORD = [
    re.compile(_REGURG_WORD + r"\s*(?:central|paravalvular|transvalvular|intraprosthetic|"
               r"prosthetic|aortic)?\s*(?:regurgitation|insufficiency|\bAR\b)", re.I),
    re.compile(r"(?:regurgitation|insufficiency|\bAR\b)\s*(?:is|was|:|grade)?\s*" +
               _REGURG_WORD, re.I),
]
REGURG_PATS_NUM = [
    re.compile(r"\bAR\b\s*(?:is|was|grade|:)?\s*([1-4])\s?\+", re.I),
    re.compile(r"(?:regurgitation|insufficiency)\s*(?:is|was|grade|:)?\s*([1-4])\s?\+", re.I),
]
_REGURG_ORDINAL = {"none": 0, "trivial": 0, "trace": 1, "mild": 1, "mild-to-moderate": 2,
                    "moderate": 2, "moderate-to-severe": 3, "severe": 4}

EVENT_PATS = {
    "ViV": re.compile(r"valve[- ]in[- ]valve|\bViV\b", re.I),
    "redo": re.compile(r"\bredo\b", re.I),
    "prosthetic_dysfunction": re.compile(
        r"structural valve deterioration|\bSVD\b|prosthetic (?:aortic )?valve "
        r"(?:dysfunction|failure|stenosis|regurgitation)|bioprosthetic (?:AVR|aortic valve|"
        r"valve) (?:stenosis|dysfunction|failure|degeneration)|prosthetic thickening|"
        r"degenerat\w+ (?:bio)?prosth|T82\.0|failed (?:bio)?prosth", re.I),
    "endocarditis": re.compile(r"endocarditis", re.I),
    "thrombosis": re.compile(r"valve thromb|leaflet thromb|\bHALT\b", re.I),
}

_NEGATED_FIELD_PATS = (
    re.compile(r"Valve\s+in\s+Valve\s*:\s*(?:No|None|N/?A)\b", re.I),
    re.compile(r"Reoperation\s*:\s*No previous surgeries", re.I),
    re.compile(r"Reoperation\s*:\s*No\b", re.I),
)


def negated_field_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of structured fields whose recorded value is negative.

    Operative reports carry key-value fields such as "Valve in Valve: No". Matching
    the key as if it were prose counts the patient as having had the very event the
    field denies. Spans are returned rather than the text rewritten, so evidence
    offsets stay valid for callers that report them.
    """
    return [m.span() for pat in _NEGATED_FIELD_PATS for m in pat.finditer(text)]


def in_negated_field(pos: int, spans) -> bool:
    """True when a match at ``pos`` falls inside a negated structured field."""
    return any(start <= pos < end for start, end in spans)


_PROSTH_CTX = re.compile(
    r"prosthetic|sapien|evolut|bioprosth|trifecta|magna|perimount|s/p|status post|"
    r"post[- ]?op|TAVR|AVR", re.I)
_NATIVE_CTX = re.compile(
    r"native|calcified valve|caused by calcified|bicuspid|pre-?op|preoperative", re.I)

# ---------------------------------------------------------------------------
# Generic-word collision guard for canonical model names that are also common English
# words / other clinical-system names. Found by QA during development (see
# docs/data_build_log.md): "Epic" is the Abbott/St Jude porcine valve AND the name of the
# Epic EHR system that these very notes are written in -- checked directly against this
# cohort's real notes (structurally, via regex hit-counts only, never printing note text):
# of 20 Operative Report notes matching the bare word "Epic", ZERO showed the expected
# "Tissue Implant Type: ... Epic" field-label pattern or other valve-adjacent wording,
# while 11 of 36 total matches showed explicit EHR phrasing ("documented in Epic", "Epic
# chart/flowsheet/record"). Rather than silently keep those false positives, models in this
# set are only counted as a genuine device match when the match site also has nearby
# valve-context evidence (mirrors the existing prosthetic/native context-window logic).
#
# The same mechanism is reused (2026-09-16, coordinator-requested Fix 1) for two more
# short/ambiguous SAPIEN-generation tokens: bare "S3" and bare "XT". Unlike "Epic", these
# are not whole canonical-model names -- they are individual device_aliases.csv rows
# (alias_type="regex_guarded") for the "SAPIEN 3" / "SAPIEN XT" canonical models, which also
# have plenty of unambiguous full-name patterns (e.g. "SAPIEN 3", "Sapien S3") that do NOT
# need guarding. GUARDED_BASE_MODELS below marks whole canonical models whose OWN base
# (canonical-name) pattern needs the guard; GUARDED alias rows are marked per-row in
# device_aliases.csv instead, see load_device_patterns().
GUARDED_BASE_MODELS = {"Epic"}
_VALVE_CONTEXT_NEARBY = re.compile(
    r"valve|bioprosth|porcine|pericardial|stent|tissue implant|prosthesis|prosthetic|"
    r"\d{2}\s?mm|supra\b|transcatheter|TAVR|TAVI|implant", re.I)
_EHR_CONTEXT_NEARBY = re.compile(
    r"documented in|reviewed in|per epic|in epic\b|epic (?:chart|flowsheet|record|system|"
    r"note)|chart review", re.I)


def _is_genuine_ambiguous_match(text, start, end, window=100):
    ctx = text[max(0, start - window):end + window]
    if _EHR_CONTEXT_NEARBY.search(ctx):
        return False
    return bool(_VALVE_CONTEXT_NEARBY.search(ctx))


def _context_flags(text, start, end, window=260):
    ctx = text[max(0, start - window):end + 60]
    prosth = bool(_PROSTH_CTX.search(ctx))
    native = bool(_NATIVE_CTX.search(ctx)) and not re.search(r"prosthetic thick", ctx, re.I)
    return prosth, native


# ---------------------------------------------------------------------------
# Device alias matching (Task 2 join)
# ---------------------------------------------------------------------------
def load_device_patterns():
    """{canonical_model: [(compiled_regex, guarded_bool), ...]}, matching against the
    canonical model name itself plus every device_aliases.csv row whose alias_type is
    'regex', 'abbreviation', or 'regex_guarded' -- i.e. the forms actually curated to
    appear in free-text operative notes. `gudid_brand` alias rows (verbose formal
    device-labeling strings, e.g. "CARPENTIER-EDWARDS PERIMOUNT RSR PERICARDIAL
    BIOPROSTHESIS - AORTIC") are NOT used for note matching -- clinicians don't write that
    in a note -- but remain in device_aliases.csv for GUDID/device-table cross-referencing.

    `guarded_bool` marks a pattern that is only accepted as a genuine device match when the
    match site also has nearby valve-context evidence and no nearby EHR/other-system
    context (see _is_genuine_ambiguous_match) -- used for a canonical model's own base name
    when that name is a generic word (GUARDED_BASE_MODELS, e.g. "Epic") and for individual
    alias rows too short/ambiguous to trust unconditionally (alias_type="regex_guarded",
    e.g. bare "S3"/"XT" for the SAPIEN family -- these letters are common enough elsewhere
    that an unconditional match would reintroduce the same class of bug found for "Epic").

    Also returns {canonical_model: {"design_class":..., "route":...}} from device_table.csv.
    """
    device_table = pd.read_csv(REFERENCE_DIR / "device_table.csv")
    aliases = pd.read_csv(REFERENCE_DIR / "device_aliases.csv")

    raw_patterns = {}  # canon -> [(pattern_str, guarded_bool), ...]
    for canon in device_table["canonical_model"].dropna().unique():
        base = re.escape(canon).replace(r"\ ", r"\s+")
        raw_patterns.setdefault(canon, []).append((base, canon in GUARDED_BASE_MODELS))

    for _, row in aliases.iterrows():
        canon = row["canonical_model"]
        if row["alias_type"] in ("regex", "abbreviation", "regex_guarded"):
            raw_patterns.setdefault(canon, []).append(
                (row["alias"], row["alias_type"] == "regex_guarded"))

    compiled = {}
    for canon, pats in raw_patterns.items():
        entries = []
        for p, guarded in pats:
            try:
                entries.append((re.compile(p, re.I), guarded))
            except re.error:
                entries.append((re.compile(re.escape(p), re.I), guarded))
        compiled[canon] = entries

    device_meta = device_table.drop_duplicates("canonical_model").set_index(
        "canonical_model")[["design_class", "route"]].to_dict("index")
    return compiled, device_meta


# ---------------------------------------------------------------------------
# Note-level extraction
# ---------------------------------------------------------------------------
def extract_note_fields(text, model_patterns):
    """Extract every field this module knows how to pull from a SINGLE note's text.
    Returns a dict; never returns or logs the note text itself."""
    found_models = set()
    for canon, pat_list in model_patterns.items():
        for pat, guarded in pat_list:
            if canon in found_models:
                break
            if guarded:
                for m in pat.finditer(text):
                    if _is_genuine_ambiguous_match(text, m.start(), m.end()):
                        found_models.add(canon)
                        break
            elif pat.search(text):
                found_models.add(canon)
    is_tavr = bool(TAVR_PAT.search(text))
    is_savr = bool(SAVR_PAT.search(text))

    sizes = set()
    for pat in SIZE_PATS:
        for m in pat.finditer(text):
            v = int(m.group(1))
            if 17 <= v <= 34:
                sizes.add(v)

    def _numeric_hits(pats, lo, hi, as_float=False):
        hits = []
        for pat in pats:
            for m in pat.finditer(text):
                raw = m.group(1)
                v = float(raw) if as_float else int(raw)
                if lo <= v <= hi:
                    prosth, native = _context_flags(text, m.start(), m.end())
                    hits.append((v, prosth, native))
        return hits

    gradients = _numeric_hits(MG_PATS, 0, 120)
    dvis = _numeric_hits(DVI_PATS, 0.05, 1.2, as_float=True)
    eoas = _numeric_hits(EOA_PATS, 0.2, 6.0, as_float=True)

    regurg = []
    for pat in REGURG_PATS_WORD:
        for m in pat.finditer(text):
            word = m.group(1).lower()
            prosth, native = _context_flags(text, m.start(), m.end())
            regurg.append((_REGURG_ORDINAL.get(word, None), word, prosth, native))
    for pat in REGURG_PATS_NUM:
        for m in pat.finditer(text):
            grade = int(m.group(1))
            prosth, native = _context_flags(text, m.start(), m.end())
            regurg.append((grade, f"{grade}+", prosth, native))

    negated = negated_field_spans(text)
    events = {
        key: any(not in_negated_field(m.start(), negated) for m in pat.finditer(text))
        for key, pat in EVENT_PATS.items()
    }

    return dict(models=found_models, tavr=is_tavr, savr=is_savr, sizes=sizes,
                gradients=gradients, dvis=dvis, eoas=eoas, regurg=regurg, events=events)


# ---------------------------------------------------------------------------
# Patient-level aggregation
# ---------------------------------------------------------------------------
def prepare_notes(raw_notes):
    """Apply the Deleted-status exclusion and note_ref assignment to a raw notes
    DataFrame (columns: Profile Key, Type, Service Date, Notes, Signed Status, ...).
    Factored out from load_notes() so tests can feed a synthetic DataFrame through the
    exact same logic instead of duplicating it (see tests/test_passport.py)."""
    notes = raw_notes[~raw_notes["Signed Status"].isin(EXCLUDE_SIGNED_STATUS)].copy()
    notes = notes.sort_values(["Profile Key", "Service Date"], kind="stable").reset_index(drop=True)
    notes["note_index"] = notes.groupby("Profile Key").cumcount()
    notes["note_ref"] = notes["Profile Key"] + "_" + notes["note_index"].astype(str)
    return notes


def load_notes():
    return prepare_notes(pd.read_excel(NOTES_XLSX))


def build_passport(notes=None, model_patterns=None, device_meta=None):
    if notes is None:
        notes = load_notes()
    if model_patterns is None:
        model_patterns, device_meta = load_device_patterns()

    note_rows = []
    for _, r in notes.iterrows():
        text = str(r["Notes"])
        yr = int(r["Service Date"])
        typ = r["Type"]
        fields = extract_note_fields(text, model_patterns)
        fields.update(pid=r["Profile Key"], yr=yr, typ=typ, note_ref=r["note_ref"])
        note_rows.append(fields)
    df = pd.DataFrame(note_rows)
    df["implant_note"] = df["typ"].isin(IMPLANT_NOTE_TYPES) & (df["tavr"] | df["savr"])

    patients = []
    for pid, g in df.groupby("pid"):
        imp = g[g["implant_note"]]
        implant_years = sorted(imp["yr"].unique().tolist())

        if len(imp):
            has_t = imp["tavr"].any()
            has_s = (imp["savr"] & ~imp["tavr"]).any()
            route = ("TAVR" if has_t else "") + ("+" if has_t and has_s else "") + \
                    ("SAVR" if has_s else "")
        else:
            route = "(hist only)" if (g["tavr"].any() or g["savr"].any()) else ""

        all_models = sorted({m for L in g["models"] for m in L})
        implant_models = sorted({m for L in imp["models"] for m in L}) if len(imp) else []
        # Prefer the MOST SPECIFIC matched name (e.g. "SAPIEN 3" over bare "SAPIEN", which
        # also matches any "SAPIEN 3" mention since passport.py's alias-driven patterns --
        # unlike build_device_table.py's ordered/lookahead patterns -- don't mutually
        # exclude family members). In this device family naming convention, a more specific
        # model name always CONTAINS its more generic relative as a substring (SAPIEN c
        # SAPIEN 3 c SAPIEN 3 Ultra; Evolut c Evolut R/PRO/FX; Magna c Magna Ease; etc.), so
        # picking the longest matched name is a reliable proxy for "most specific" here.
        # Ties broken alphabetically for determinism.
        def _specificity(models):
            return sorted(models, key=lambda m: (-len(m), m))[0] if models else ""
        primary_model = _specificity(implant_models) if implant_models else _specificity(all_models)
        design_class = device_meta.get(primary_model, {}).get("design_class", "") if primary_model else ""

        all_sizes = sorted({s for S in g["sizes"] for s in S})
        implant_sizes = sorted({s for S in imp["sizes"] for s in S}) if len(imp) else []
        size_mm = implant_sizes[0] if implant_sizes else (all_sizes[0] if all_sizes else None)

        def _series(field, note_refs_col="note_ref"):
            out = []
            for _, row in g.iterrows():
                for item in row[field]:
                    if field == "regurg":
                        grade_ord, grade_word, prosth, native = item
                        out.append(dict(year=row["yr"], grade_ordinal=grade_ord,
                                         grade=grade_word, prosthetic=prosth, native=native,
                                         note_ref=row["note_ref"]))
                    else:
                        val, prosth, native = item
                        out.append(dict(year=row["yr"], value=val, prosthetic=prosth,
                                         native=native, note_ref=row["note_ref"]))
            return sorted(out, key=lambda d: d["year"])

        gradient_series = _series("gradients")
        dvi_series = _series("dvis")
        eoa_series = _series("eoas")
        regurg_series = _series("regurg")

        prosth_gradients = [x for x in gradient_series if x["prosthetic"] and not x["native"]]
        gradient_years_prosth = sorted({x["year"] for x in prosth_gradients})

        events = {k: bool(g[g["events"].apply(lambda e: e.get(k, False))].shape[0])
                  for k in EVENT_PATS.keys()}

        patients.append(dict(
            patient_id=pid,
            n_notes=len(g),
            first_note_year=int(g["yr"].min()),
            last_note_year=int(g["yr"].max()),
            route=route,
            implant_year=implant_years[0] if implant_years else None,
            implant_years_all=json.dumps(implant_years),
            canonical_model=primary_model,
            all_models_mentioned=";".join(all_models),
            design_class=design_class,
            size_mm=size_mm,
            all_sizes_mentioned=json.dumps(all_sizes),
            gradient_series_json=json.dumps(gradient_series),
            n_gradients_prosthetic=len(prosth_gradients),
            n_distinct_gradient_years_prosthetic=len(gradient_years_prosth),
            max_gradient_prosthetic=max([x["value"] for x in prosth_gradients], default=None),
            dvi_series_json=json.dumps(dvi_series),
            eoa_series_json=json.dumps(eoa_series),
            regurg_series_json=json.dumps(regurg_series),
            event_ViV=events["ViV"],
            event_redo=events["redo"],
            event_prosthetic_dysfunction=events["prosthetic_dysfunction"],
            event_endocarditis=events["endocarditis"],
            event_thrombosis=events["thrombosis"],
        ))

    passport = pd.DataFrame(patients).sort_values("patient_id").reset_index(drop=True)
    return passport, df


def save_passport(passport, out_dir=PRIVATE_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    passport.to_csv(out_dir / "passport.csv", index=False)
    try:
        passport.to_parquet(out_dir / "passport.parquet", index=False)
    except Exception as ex:
        print(f"WARNING: parquet write failed ({type(ex).__name__}); CSV was written "
              f"successfully. Install/upgrade pyarrow if this persists.")
    return out_dir / "passport.csv", out_dir / "passport.parquet"


def main():
    print("Loading notes (Deleted rows excluded) ...")
    notes = load_notes()
    print(f"  {len(notes)} notes across {notes['Profile Key'].nunique()} patients")
    print("Loading device patterns (device_table.csv + device_aliases.csv) ...")
    model_patterns, device_meta = load_device_patterns()
    print(f"  {len(model_patterns)} canonical models with a matchable pattern")
    print("Extracting ...")
    passport, note_level = build_passport(notes, model_patterns, device_meta)
    csv_path, parquet_path = save_passport(passport)
    print(f"Wrote {len(passport)} patient rows -> {csv_path}")
    print(f"  routes: {passport['route'].value_counts().to_dict()}")
    print(f"  with implant_year: {passport['implant_year'].notna().sum()}")
    print(f"  with >=1 prosthetic gradient: {(passport['n_gradients_prosthetic'] > 0).sum()}")
    return passport, note_level


if __name__ == "__main__":
    main()
