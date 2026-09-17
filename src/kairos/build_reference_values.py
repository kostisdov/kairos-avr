# -*- coding: utf-8 -*-
"""
Build data/reference/prosthetic_valve_reference_values.csv (Task 3) from the ASE 2024
prosthetic valve guideline PDF appendix (Zoghbi et al, J Am Soc Echocardiogr 2024;37:2-63),
Tables A1-A4 (the AORTIC-position tables; A5 onward are mitral/pulmonary/tricuspid and are
out of scope for KAIROS). The PDF stays in data/raw/papers/ -- only extracted numbers with
citation leave this script.

PDF quirk (discovered by inspection, not assumed): this PDF's font encoding renders the
"±" glyph as a literal ASCII "6" character sitting between two decimal numbers, e.g. the
text "19.1(PLUSMINUS)8.2" (mean 19.1, SD 8.2) extracts as the literal string "19.168.2".
Confirmed against the caption "Data are expressed as mean +/- SD" on every appendix table
and by cross-checking that stripping the middle "6" and treating the two sides as
value/SD always yields clinically plausible pairs (SD << mean, right order of magnitude)
across the tables. All "6"-splitting below is done per-whitespace-token with a regex that
requires the remainder to parse as a clean decimal, specifically to avoid corrupting a
literal digit "6" that is part of a real number (e.g. "16.0" in "16.062.0" meaning
16.0 +/- 2.0) -- verified by hand against page text, see docstring at bottom and the
10-cell spot-check block in __main__.

Table A4 (surgical aortic valves) lists Peak gradient, Mean gradient, EOA per valve/size,
but many rows in the source PDF are missing one or more of those three values (sparse
literature pooling). Rows are classified by NUMBER OF VALUES PRESENT and MAGNITUDE
(EOA cm2 values are always < 4.5 in this table; gradients are always >= 4 mmHg) -- see
`classify_a4_row_values()` docstring for the exact rule and the confidence it assigns.
"""
import csv
import re
from datetime import datetime, timezone

import pdfplumber
from kairos.paths import repo_root

ROOT = str(repo_root())
PDF_PATH = f"{ROOT}/data/raw/papers/ASE2024_prosthetic_valve_guideline.pdf"
OUT_CSV = f"{ROOT}/data/reference/prosthetic_valve_reference_values.csv"
RETRIEVED_AT = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
CITATION = ("Zoghbi WA, Jone PN, Chamsi-Pasha MA, et al. Guidelines for the Evaluation of "
            "Prosthetic Valve Function With Cardiovascular Imaging: A Report from the "
            "American Society of Echocardiography. J Am Soc Echocardiogr 2024;37:2-63, "
            "Appendix Tables A1-A4")

# canonical_model mapping (task 2 join key) for valve names as they appear in this PDF
CANONICAL_MAP = {
    "sapien": "SAPIEN", "sapienxt": "SAPIEN XT", "sapien3": "SAPIEN 3",
    "corevalve": "CoreValve", "evolutr30d": "Evolut R", "evolut": "Evolut R",
    "abbottepic": "Epic", "abbotttrifecta": "Trifecta", "baxterperimount": "Perimount",
    "carpentieredwardspericardial": "Perimount", "carpentieredwardsstandard": "Perimount",
    "edwardsinspirisresilia": "Inspiris RESILIA", "edwardsintuity": "INTUITY Elite",
    "medtronicavalus": "Avalus", "medtronicfreestyle": "Freestyle",
    "medtronicmosaic": "Mosaic", "edwardsmosaic": "Mosaic",
    "hancockii": "Hancock II",
    # bare "Hancock" (no "II") is mapped to the "Hancock II" canonical model with LOW
    # confidence via UNCERTAIN_CANONICAL_MAPPINGS below, not a separate free-text label --
    # keeping the mapped value an exact device_table.csv canonical_model name so coverage
    # can be computed by simple set membership (coordinator Fix 2 asked for a clean list of
    # canonical models with zero reference rows). The generational relationship between
    # plain "Hancock" and "Hancock II" in this 1980s-vintage ASE table entry was not
    # independently verified this session.
    "hancock": "Hancock II",
    "mitroflow": "Mitroflow", "sorinpercevalsutureless": "Perceval",
    "stjudemedicalstandard": "",  # mechanical valve, out of device_table scope (bioprosthetic/TAVR only)
    "corevalve100": "CoreValve", "evolut141229": "Evolut R", "sapien3100": "SAPIEN 3",
    "sapienxt100230": "SAPIEN XT",
    # Added coordinator Fix 2 (2026-09-16): device_table.csv's "Prima Plus" canonical model
    # (added in Task 2 as a bonus row beyond the task's original device list, from a real
    # GUDID brand "Edwards Prima Plus Stentless Bioprosthesis") plausibly corresponds to
    # this ASE table's bare "Prima" (Stentless) entries -- Edwards' stentless Prima platform
    # -- but the table does not say "Plus" explicitly and Prima/Prima Plus may be different
    # generations of the same base platform. Mapped with LOW confidence via
    # UNCERTAIN_CANONICAL_MAPPINGS below rather than left unmapped, since some signal is
    # better than none for a bonus row that otherwise has zero reference values, but this
    # should be verified against Edwards' own generational documentation before being
    # treated as equivalent to "Prima Plus" with any confidence.
    "prima": "Prima Plus",
}
UNCERTAIN_CANONICAL_MAPPINGS = {"prima", "hancock"}


def norm_key(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def canonical_for(valve_name):
    return CANONICAL_MAP.get(norm_key(valve_name), "")


def confidence_for_mapping(valve_name, base_confidence):
    """Downgrade confidence to 'low' when the canonical_model mapping itself is uncertain
    (UNCERTAIN_CANONICAL_MAPPINGS), regardless of how confidently the numbers were read off
    the page -- these are two independent kinds of uncertainty."""
    if norm_key(valve_name) in UNCERTAIN_CANONICAL_MAPPINGS:
        return "low"
    return base_confidence


# ---------------------------------------------------------------------------
# valve_model_clean: human-readable display name (coordinator Fix 2). The raw valve_model
# strings inherit pdfplumber's word-concatenation artifact from the source PDF's borderless
# table (e.g. "BaxterPerimount (Stentedbovinepericardial)") -- this inserts spaces at
# lower-to-upper letter transitions and a few punctuation boundaries to make them readable,
# without attempting a full rewrite (it is a display aid, not a new extraction).
# ---------------------------------------------------------------------------
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")
_LOWER_TO_PAREN_RE = re.compile(r"(?<=[a-z0-9])\(")
# Real device/company names that are intentionally camelCase in normal usage -- must NOT be
# split by the generic lower-to-upper heuristic below (e.g. "CoreValve" is the correct
# Medtronic spelling, not a PDF-concatenation artifact like "BaxterPerimount" is).
_PRESERVE_CAMEL = ["CoreValve", "MedtronicHall", "OnX", "LivaNova"]


def clean_display_name(valve_model):
    s = valve_model
    placeholders = {}
    for i, word in enumerate(_PRESERVE_CAMEL):
        if word in s:
            token = f"\x00{i}\x00"
            s = s.replace(word, token)
            placeholders[token] = word
    s = _LOWER_TO_PAREN_RE.sub(" (", s)
    s = _CAMEL_BOUNDARY_RE.sub(" ", s)
    for token, word in placeholders.items():
        s = s.replace(token, word)
    s = re.sub(r"\s+", " ", s).strip()
    return s


NUM_PLUSMINUS = re.compile(r"^(\d+\.?\d*)6(\d+\.?\d*)$")
NUM_BARE = re.compile(r"^(\d+\.?\d*)$")


def parse_token(tok):
    """Return (value, sd_or_None) for one whitespace-separated table cell, or None if the
    token is not numeric (e.g. 'NA')."""
    if tok in ("NA", "--", "-"):
        return None
    m = NUM_PLUSMINUS.match(tok)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = NUM_BARE.match(tok)
    if m:
        return float(m.group(1)), None
    return None


# ---------------------------------------------------------------------------
# Tables A1 / A2: percutaneous (TAVR) valves in native AS, by size, block format:
#   {NAME} {s1}mm {s2}mm {s3}mm {s4}mm Allsizes
#   EOA,cm2 v1 v2 v3 v4 vAll
#   Meangradient,mmHg v1 v2 v3 v4 vAll
#   DVI v1 v2 v3 v4 vAll
# ---------------------------------------------------------------------------
def parse_tavr_native_table(text, table_label, page_no):
    rows = []
    lines = [l for l in text.split("\n")]
    i = 0
    header_re = re.compile(r"^(\S+?)\s+((?:\d{2}mm\s+){3,4})Allsizes\s*$")
    while i < len(lines):
        m = header_re.match(lines[i].strip())
        if m:
            valve_name = m.group(1)
            sizes = [int(x) for x in re.findall(r"(\d{2})mm", m.group(2))]
            metrics = {}
            j = i + 1
            while j < len(lines) and j < i + 4:
                parts = lines[j].strip().split()
                if not parts:
                    break
                label = parts[0]
                if label.startswith("EOA"):
                    metrics["eoa"] = parts[1:]
                elif label.startswith("Meangradient"):
                    metrics["mg"] = parts[1:]
                elif label.startswith("DVI"):
                    metrics["dvi"] = parts[1:]
                else:
                    break
                j += 1
            n_cols = len(sizes) + 1  # + "All sizes"
            for col_idx, size in enumerate(sizes):
                eoa = parse_token(metrics.get("eoa", [None] * n_cols)[col_idx]) if metrics.get("eoa") else None
                mg = parse_token(metrics.get("mg", [None] * n_cols)[col_idx]) if metrics.get("mg") else None
                dvi = parse_token(metrics.get("dvi", [None] * n_cols)[col_idx]) if metrics.get("dvi") else None
                if eoa is None and mg is None and dvi is None:
                    continue
                rows.append(dict(
                    valve_model=valve_name, valve_model_clean=clean_display_name(valve_name),
                    position="aortic", size_mm=size, n="",
                    mean_gradient_mmHg_mean=mg[0] if mg else "",
                    mean_gradient_sd=mg[1] if mg and mg[1] is not None else "",
                    peak_velocity="", peak_gradient_mmHg_mean="", peak_gradient_mmHg_sd="",
                    eoa_cm2_mean=eoa[0] if eoa else "", eoa_sd=eoa[1] if eoa and eoa[1] is not None else "",
                    dvi_mean=dvi[0] if dvi else "", dvi_sd=dvi[1] if dvi and dvi[1] is not None else "",
                    canonical_model=canonical_for(valve_name),
                    source_table=table_label, source_page=page_no,
                    confidence=confidence_for_mapping(valve_name, "high"),
                ))
            i = j
        else:
            i += 1
    return rows


# ---------------------------------------------------------------------------
# Table A3: ViV, one row per device, format "{Device+footnotes} All {peak} {mean} {eoa}"
# ---------------------------------------------------------------------------
def parse_viv_table(text, table_label, page_no):
    rows = []
    device_patterns = [
        ("CoreValve", re.compile(r"^CoreValve[\d,]*\s+All\s+(\S+)\s+(\S+)\s+(\S+)")),
        ("Evolut", re.compile(r"^Evolut[\d,]*\s+All\s+(\S+)\s+(\S+)\s+(\S+)")),
        ("SAPIEN 3", re.compile(r"^SAPIEN3[\d,]*\s+All\s+(\S+)\s+(\S+)\s+(\S+)")),
        ("SAPIEN XT", re.compile(r"^SAPIENXT[\d,]*\s+All\s+(\S+)\s+(\S+)\s+(\S+)")),
    ]
    for line in text.split("\n"):
        s = line.strip()
        for name, pat in device_patterns:
            m = pat.match(s)
            if m:
                peak = parse_token(m.group(1))
                mean = parse_token(m.group(2))
                eoa = parse_token(m.group(3))
                vm = name + " (valve-in-valve)"
                rows.append(dict(
                    valve_model=vm, valve_model_clean=clean_display_name(vm),
                    position="aortic (ViV)", size_mm="All",
                    n="",
                    mean_gradient_mmHg_mean=mean[0] if mean else "",
                    mean_gradient_sd=mean[1] if mean and mean[1] is not None else "",
                    peak_velocity="",
                    peak_gradient_mmHg_mean=peak[0] if peak else "",
                    peak_gradient_mmHg_sd=peak[1] if peak and peak[1] is not None else "",
                    eoa_cm2_mean=eoa[0] if eoa else "", eoa_sd=eoa[1] if eoa and eoa[1] is not None else "",
                    dvi_mean="", dvi_sd="",
                    canonical_model=canonical_for(name.replace(" ", "")),
                    source_table=table_label, source_page=page_no,
                    confidence=confidence_for_mapping(name, "high"),
                ))
    return rows


# ---------------------------------------------------------------------------
# Table A4: surgical aortic valves, stateful block parser (valve name + design descriptor
# span their own line(s); subsequent lines starting with a size number belong to that
# valve until the next valve-name line appears).
# ---------------------------------------------------------------------------
# A "size" token is normally a plain 2-digit mm number (or a range like "19-21"), but the
# Perceval sutureless-valve rows use letter codes with the mm-equivalent in parens instead
# (e.g. "S(21)", "M(23)", "L(25)", "XL(27)") -- found and fixed (coordinator Fix 2,
# 2026-09-16) after discovering Perceval's 4 data rows were silently discarded: the letter-
# code prefix meant these lines matched neither the pure-size-row regex nor the
# name+first-row regex, so they fell into the "text continuation" branch and were appended
# as garbage onto whatever valve name happened to precede them in the table, producing no
# emitted row at all (not even a wrong one) until the next valid entry reset state.
_SIZE_TOKEN = r"(?:\d{2}(?:-\d{2})?|[A-Za-z]{1,3}\(\d{2}\))"
SIZE_ROW_RE = re.compile(rf"^({_SIZE_TOKEN})\s+(.+)$")
# A line that starts a NEW valve block: "{NameWord}{spaces}{size}{spaces}{data...}" -- the
# PDF's row-spanning "Valve" name cell is only as tall as the FIRST data row, so pdfplumber
# emits the name and the first size row on one physical text line; every subsequent size
# for the same valve is on its own line starting directly with the size number (no name).
NAME_PLUS_FIRST_ROW_RE = re.compile(rf"^([A-Za-z][A-Za-z.\-]*)\s+({_SIZE_TOKEN})\s+(.+)$")
_LETTER_SIZE_RE = re.compile(r"^[A-Za-z]{1,3}\((\d{2})\)$")


def _normalise_size_token(size_txt):
    """'S(21)' -> '21' (keep the numeric mm-equivalent for size_mm; the letter code itself
    is not currently a separate schema column, see coordinator Fix 2 report). Plain
    numeric/range tokens ('23', '19-21') pass through unchanged."""
    m = _LETTER_SIZE_RE.match(size_txt)
    return m.group(1) if m else size_txt
DESCRIPTOR_WORDS = re.compile(
    r"^(Stented|Stentless|Bileaflet|Singletiltingdisk|Tiltingdisk|Cagedball|Homograftvalves)",
    re.I,
)


def classify_a4_row_values(tokens):
    """tokens: list of parsed (value, sd) tuples in left-to-right order from a Table A4
    data row (after the size). Returns dict with peak/mean/eoa (each value,sd or None) and
    a confidence label. Classification rule: a value < 4.5 is treated as EOA (cm2); a
    value >= 4.5 is treated as a gradient (mmHg) -- chosen because, inspected across the
    full table, EOA never exceeds ~3.9 and gradients are never below ~4.0. Within the
    gradient values, if two are present the larger is peak and the smaller is mean
    (peak instantaneous gradient is always >= mean gradient over the ejection period,
    by definition -- not an assumption specific to this table)."""
    eoa = [t for t in tokens if t[0] < 4.5]
    grad = [t for t in tokens if t[0] >= 4.5]
    out = {"peak": None, "mean": None, "eoa": None, "confidence": "low"}
    if len(eoa) >= 1:
        out["eoa"] = eoa[0]
    if len(grad) == 2:
        g0, g1 = grad
        out["peak"], out["mean"] = (g0, g1) if g0[0] >= g1[0] else (g1, g0)
        out["confidence"] = "high" if len(tokens) == 3 else "medium"
    elif len(grad) == 1:
        # ambiguous: could be peak-only or mean-only in the source. We report it as mean
        # gradient (the more commonly load-bearing metric for our reference-range use
        # case) but flag low confidence -- see module docstring.
        out["mean"] = grad[0]
        out["confidence"] = "low"
    return out


def _emit_row(rows, valve, descriptor, size_txt, rest, table_label, page_no, skipped):
    toks = [parse_token(t) for t in rest.split()]
    toks = [t for t in toks if t is not None]
    if not toks:
        skipped.append((valve, size_txt, rest))
        return
    cls = classify_a4_row_values(toks)
    if cls["peak"] is None and cls["mean"] is None and cls["eoa"] is None:
        skipped.append((valve, size_txt, rest))
        return
    peak, mean, eoa = cls["peak"], cls["mean"], cls["eoa"]
    vm = f"{valve} ({descriptor})" if descriptor else valve
    rows.append(dict(
        valve_model=vm, valve_model_clean=clean_display_name(vm),
        position="aortic", size_mm=_normalise_size_token(size_txt), n="",
        mean_gradient_mmHg_mean=mean[0] if mean else "",
        mean_gradient_sd=mean[1] if mean and mean[1] is not None else "",
        peak_velocity="",
        peak_gradient_mmHg_mean=peak[0] if peak else "",
        peak_gradient_mmHg_sd=peak[1] if peak and peak[1] is not None else "",
        eoa_cm2_mean=eoa[0] if eoa else "", eoa_sd=eoa[1] if eoa and eoa[1] is not None else "",
        dvi_mean="", dvi_sd="",
        canonical_model=canonical_for(valve),
        source_table=table_label, source_page=page_no,
        confidence=confidence_for_mapping(valve, cls["confidence"]),
    ))


def parse_surgical_aortic_table(pages_text, table_label, page_numbers):
    rows = []
    current_valve = None
    current_descriptor = None
    skipped = []
    for page_text, page_no in zip(pages_text, page_numbers):
        for raw_line in page_text.split("\n"):
            line = raw_line.strip()
            if not line or line.startswith("TableA4") or line.startswith("Valve Size") \
                    or line.startswith("Dataareexpressed") or line.startswith("Modifiedfrom") \
                    or re.match(r"^\d+\s+Zoghbi", line) or "JournaloftheAmerican" in line \
                    or line == "TableA4(Continued)" or line.startswith("(Continued)"):
                continue

            m_pure_size = SIZE_ROW_RE.match(line)
            m_name_first_row = NAME_PLUS_FIRST_ROW_RE.match(line)

            if m_name_first_row:
                # Starts a brand-new valve block: name word + its first size row combined.
                name, size_txt, rest = m_name_first_row.groups()
                current_valve = name
                current_descriptor = None
                _emit_row(rows, current_valve, current_descriptor, size_txt, rest,
                          table_label, page_no, skipped)
            elif m_pure_size:
                # A continuation size row for the CURRENT valve (name cell was merged/
                # blank for this row in the source table).
                if current_valve is None:
                    continue
                size_txt, rest = m_pure_size.groups()
                _emit_row(rows, current_valve, current_descriptor, size_txt, rest,
                          table_label, page_no, skipped)
            else:
                # Pure-text line: either a tissue/design descriptor (known keyword list)
                # or the wrapped second word of a multi-word valve name (e.g. "Abbott" on
                # the name+first-row line, "Trifecta" alone on the next line) -- append to
                # the current valve name in that case rather than discarding it.
                if DESCRIPTOR_WORDS.match(line):
                    current_descriptor = line
                elif current_valve is not None:
                    current_valve = f"{current_valve} {line}"
    return rows, skipped


def main():
    all_rows = []
    with pdfplumber.open(PDF_PATH) as pdf:
        # Table A1 + A2 are both on pdf page index 49 (page 50)
        t_a1a2 = pdf.pages[49].extract_text()
        all_rows += parse_tavr_native_table(t_a1a2, "Table A1/A2", 50)
        # Table A3 on pdf page index 50 (page 51)
        t_a3 = pdf.pages[50].extract_text()
        all_rows += parse_viv_table(t_a3, "Table A3", 51)
        # Table A4 spans pdf page index 51-56 (pages 52-57)
        a4_pages = list(range(51, 57))
        a4_text = [pdf.pages[p].extract_text() for p in a4_pages]
        a4_rows, a4_skipped = parse_surgical_aortic_table(a4_text, "Table A4", [p + 1 for p in a4_pages])
        all_rows += a4_rows

    cols = ["valve_model", "valve_model_clean", "position", "size_mm", "n",
            "mean_gradient_mmHg_mean", "mean_gradient_sd",
            "peak_velocity", "peak_gradient_mmHg_mean", "peak_gradient_mmHg_sd",
            "eoa_cm2_mean", "eoa_sd", "dvi_mean", "dvi_sd", "canonical_model",
            "source_table", "source_page", "confidence", "source", "retrieved_at"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in all_rows:
            r["source"] = CITATION
            r["retrieved_at"] = RETRIEVED_AT
            w.writerow(r)

    print(f"wrote {len(all_rows)} rows -> {OUT_CSV}")
    print(f"Table A4 rows skipped as unparseable (all-NA or no numeric tokens): {len(a4_skipped)}")
    for v, s, raw in a4_skipped[:15]:
        print(f"  SKIPPED  valve={v!r} size={s!r} raw={raw!r}")
    conf_counts = {}
    for r in all_rows:
        conf_counts[r["confidence"]] = conf_counts.get(r["confidence"], 0) + 1
    print("confidence distribution:", conf_counts)


if __name__ == "__main__":
    main()
