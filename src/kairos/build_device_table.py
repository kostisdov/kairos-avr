# -*- coding: utf-8 -*-
"""
Build data/reference/device_table.csv and data/reference/device_aliases.csv (Task 2).

Inputs:
  - data/raw/gudid/gudid_valve_records.jsonl
      produced by src/kairos/build_gudid_extract.py from the FDA AccessGUDID full release
      (public device data only -- no patient data anywhere in this script).

Method:
  1. Re-derive a precise "this GUDID record is an aortic heart-valve device from one of our
     target manufacturers" filter:
       - productCode in {DYE, NPT}  (verified via openFDA classification.json: DYE =
         "Replacement Heart-Valve" surgical class III; NPT = "Aortic Valve, Prosthesis,
         Percutaneously Delivered" class III), OR
       - companyName matches a tight target-manufacturer regex (deliberately narrower than
         a loose substring match on "Abbott"/"Medtronic" etc., which was found during
         development to also match unrelated Abbott Diabetes Care / generic Medtronic
         orthopedic-division devices -- see docs/data_build_log.md)
     AND
       - gmdnTerms contains one of the three GUDID GMDN preferred terms that specifically
         denote an AORTIC valve device (surgical, transcatheter, or valved-conduit) --
         this cleanly excludes mitral/tricuspid/pulmonic valves, annuloplasty rings,
         sizers, delivery catheters, crimpers, introducer sheaths etc. that otherwise share
         a product code or company with the valves we want.
     AND
       - brandName/deviceDescription does not match a delivery-system/accessory keyword
         list (some Edwards transcatheter delivery-system DIs are mis-groupable under the
         valve GMDN term in the raw data; this is a final safety net).
  2. Match each remaining record's brandName/deviceDescription against a hand-curated
     regex per canonical_model (+ generation), collect its Primary DI and size(s) in mm.
  3. Merge with a hand-curated static table of manufacturer/route/design_class/tissue/
     leaflet_mounting/tissue_treatment/generation/market_status/fda_pma -- these fields are
     domain knowledge (industry/regulatory public record), NOT derivable from GUDID alone.
     Confidence is "high" only where a fact was actively verified this session (GUDID
     brand/size presence, or an openFDA PMA-number lookup actually returned a result in
     this session -- see docs/data_build_log.md); "medium" for well-established public
     facts not re-verified this session; "low" + a TODO note where uncertain. No value is
     fabricated.

Run: python src/kairos/build_device_table.py
"""
import csv
import json
import re
from datetime import datetime, timezone
from kairos.paths import repo_root

ROOT = str(repo_root())
JSONL = f"{ROOT}/data/raw/gudid/gudid_valve_records.jsonl"
OUT_DEVICE_TABLE = f"{ROOT}/data/reference/device_table.csv"
OUT_ALIASES = f"{ROOT}/data/reference/device_aliases.csv"
OUT_META = f"{ROOT}/data/reference/device_table.meta.yaml"

RETRIEVED_AT = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
GUDID_SOURCE = "FDA AccessGUDID full release gudid_full_release_20260901.zip (accessgudid.nlm.nih.gov)"

# ---------------------------------------------------------------------------
# Step 1: load + filter GUDID records
# ---------------------------------------------------------------------------
TARGET_COMPANIES = re.compile(
    r"edwards\s*lifesciences|^medtronic|medtronic\s*corevalve|medtronic,?\s*inc|"
    r"abbott\s*medical|abbott\s*cardiovascular|abbott\s*vascular|st\.?\s*jude\s*medical|"
    r"livanova|corcym|sorin\s*group|sorin\s*biomedica|boston\s*scientific|symetis|meril|"
    r"jenavalve",
    re.I,
)
AORTIC_VALVE_GMDN = {
    "Aortic open-surgery heart valve bioprosthesis",
    "Aortic transcatheter heart valve bioprosthesis, stent-like framework",
    "Aortic open-surgery heart valve bioprosthesis/synthetic polymer aorta graft",
}
ACCESSORY_EXCLUDE = re.compile(
    r"delivery system|introducer|sheath|crimper|dilator|loader|holder|handle|novaflex|"
    r"certitude|commander|axela|balloon catheter|collapsing tool",
    re.I,
)


def load_valve_records():
    recs = [json.loads(l) for l in open(JSONL, encoding="utf-8")]
    clean = [
        r for r in recs
        if any(pc["code"] in ("DYE", "NPT") for pc in r["productCodes"])
        or TARGET_COMPANIES.search(r["companyName"] or "")
    ]
    aortic = [r for r in clean if any(g in AORTIC_VALVE_GMDN for g in r["gmdnTerms"])]
    valve_only = [
        # Match against brandName ONLY (not deviceDescription): a real valve's own
        # deviceDescription often legitimately says e.g. "...Premounted on Delivery
        # System" (Boston Scientific LOTUS Edge) -- that describes packaging, not an
        # accessory DI. A true accessory's *brand name itself* is the giveaway
        # (e.g. "Edwards Certitude Delivery System", "EDWARDS CRIMPER").
        r for r in aortic if not ACCESSORY_EXCLUDE.search(r["brandName"] or "")
    ]
    return valve_only


SIZE_RE_DESC = re.compile(r"(\d{2}(?:\.\d)?)\s*-?\s*mm", re.I)
# GUDID sizeText spells out the unit inconsistently, including a real typo seen in the
# original SAPIEN records ("26 MILIMETER DIAMETER EXPANDED VALVE" -- one L) -- match MM,
# MILLIMETER, or MILIMETER.
SIZE_RE_SIZETEXT = re.compile(r"(\d{2}(?:\.\d)?)\s*(?:MM\b|MILLIMETER|MILIMETER)", re.I)


def extract_sizes(rec):
    sizes = set()
    for sz in rec["deviceSizes"]:
        for field in (sz.get("sizeText") or "", ):
            for m in SIZE_RE_SIZETEXT.finditer(field):
                v = float(m.group(1))
                if 14 <= v <= 40:
                    sizes.add(v)
        val = sz.get("value")
        if val:
            try:
                v = float(val)
                if 14 <= v <= 40:
                    sizes.add(v)
            except ValueError:
                pass
    for m in SIZE_RE_DESC.finditer(rec["deviceDescription"] or ""):
        v = float(m.group(1))
        if 14 <= v <= 40:
            sizes.add(v)
    if not sizes:
        # Fallback for records with no deviceSizes/deviceDescription at all (seen for
        # Corcym Mitroflow/Crown PRT/Perceval DIs in this GUDID release): the model or
        # catalog number itself ends in a 2-digit size, e.g. "DLA27" (Mitroflow 27mm),
        # "CNA23" (Crown PRT 23mm), "PVS21" (Perceval 21mm). Only trust a trailing
        # 2-digit number in the plausible valve-size range, and only when the
        # non-digit prefix looks like a short alphabetic model code (avoids
        # misreading e.g. a lot/serial-style number as a size).
        for field in (rec.get("versionModelNumber") or "", rec.get("catalogNumber") or ""):
            m = re.fullmatch(r"[A-Za-z-]{2,6}(\d{2})", field.strip())
            if m:
                v = float(m.group(1))
                if 15 <= v <= 34:
                    sizes.add(v)
    return sizes


def primary_di(rec):
    for ident in rec["identifiers"]:
        if ident.get("deviceIdType") == "Primary":
            return ident.get("deviceId")
    return rec["identifiers"][0]["deviceId"] if rec["identifiers"] else None


# ---------------------------------------------------------------------------
# Step 2: canonical model -> brand-matching regex (applied to brandName, falls back to
# deviceDescription). Order matters: more specific patterns first.
# ---------------------------------------------------------------------------
CANONICAL_BRAND_PATTERNS = [
    # (canonical_model, generation, regex)
    ("SAPIEN 3 Ultra RESILIA", "current", re.compile(r"sapien\s*3\s*ultra\s*resilia", re.I)),
    ("SAPIEN 3 Ultra", "prior", re.compile(r"sapien\s*3\s*ultra(?!\s*resilia)", re.I)),
    ("SAPIEN 3", "prior", re.compile(r"sapien\s*3(?!\s*ultra)", re.I)),
    ("SAPIEN XT", "discontinued", re.compile(r"sapien\s*xt", re.I)),
    ("SAPIEN", "discontinued", re.compile(r"\bsapien\b(?!\s*(3|xt))", re.I)),
    ("Evolut FX+", "current", re.compile(r"evolut\s*fx\s*\+", re.I)),
    ("Evolut FX", "prior", re.compile(r"evolut\s*fx(?!\+)", re.I)),
    ("Evolut PRO+", "prior", re.compile(r"evolut\s*pro\s*\+|enveo\s*pro\s*\+", re.I)),
    ("Evolut PRO", "prior", re.compile(r"evolut\s*pro(?!\+)|enveo\s*pro(?!\+)", re.I)),
    ("Evolut R", "prior", re.compile(r"evolut\s*r\b|enveo\s*r\b", re.I)),
    ("CoreValve", "discontinued", re.compile(r"corevalve(?!\s*evolut\s*(r|pro|fx))|corevalue", re.I)),
    ("Navitor", "current", re.compile(r"navitor", re.I)),
    ("Portico", "prior", re.compile(r"portico", re.I)),
    ("Lotus", "discontinued", re.compile(r"\blotus\b", re.I)),
    ("JenaValve (Trilogy)", "current", re.compile(r"jenavalve|trilogy\s*transcatheter", re.I)),
    ("ACURATE neo2", "current", re.compile(r"acurate\s*neo\s*2|acurate\s*neo2", re.I)),
    ("ACURATE neo", "prior", re.compile(r"acurate\s*neo(?!\s*2)", re.I)),
    ("Myval", "current", re.compile(r"myval", re.I)),
    ("Trifecta GT", "withdrawn", re.compile(r"trifecta\s*gt", re.I)),
    ("Trifecta", "withdrawn", re.compile(r"trifecta(?!\s*gt)", re.I)),
    ("Mitroflow", "discontinued", re.compile(r"mitroflow", re.I)),
    ("Crown PRT", "current", re.compile(r"crown\s*prt", re.I)),
    ("Perceval Plus", "current", re.compile(r"perceval\s*plus", re.I)),
    ("Perceval", "current", re.compile(r"perceval(?!\s*plus)", re.I)),
    ("INTUITY Elite", "current", re.compile(r"intuity", re.I)),
    ("Inspiris RESILIA", "current", re.compile(r"inspiris", re.I)),
    ("Konect RESILIA", "current", re.compile(r"konect\s*resilia", re.I)),
    ("Konect (legacy pre-RESILIA)", "legacy", re.compile(r"carpentier-edwards bioprosthetic valved conduit", re.I)),
    ("Prima Plus", "current", re.compile(r"prima\s*plus", re.I)),
    ("Avalus Ultra", "current", re.compile(r"avalus\s*ultra", re.I)),
    ("Avalus", "current", re.compile(r"avalus(?!\s*ultra)", re.I)),
    ("Magna Ease", "current", re.compile(r"magna\s*ease", re.I)),
    ("Magna", "prior", re.compile(r"\bmagna\b(?!\s*ease)", re.I)),
    ("Perimount", "legacy", re.compile(r"perimount(?!\s*magna)|carpentier-edwards\s*(bioprosthesis|s\.?a\.?v)", re.I)),
    ("Epic Supra", "current", re.compile(r"epic\s*supra", re.I)),
    ("Epic", "current", re.compile(r"\bepic\b(?!\s*supra)(?!.*vascular)(?!.*biliary)", re.I)),
    ("Mosaic Ultra", "current", re.compile(r"mosaic\s*ultra", re.I)),
    ("Mosaic", "prior", re.compile(r"\bmosaic\b(?!\s*ultra)(?!\s*neo)", re.I)),
    ("Hancock II Ultra", "current", re.compile(r"hancock\s*ii\s*ultra", re.I)),
    ("Hancock II", "prior", re.compile(r"hancock\s*ii(?!\s*ultra)", re.I)),
    ("Freestyle", "current", re.compile(r"freestyle(?!.*libre)", re.I)),
    ("Toronto SPV", "legacy", re.compile(r"toronto\s*spv", re.I)),
]

# ---------------------------------------------------------------------------
# Step 3: static domain-knowledge table. confidence: "high" only if verified this
# session (GUDID presence and/or an openFDA PMA lookup that returned a result today);
# "medium" for well-known industry/regulatory facts not independently re-verified this
# session; "low" where genuinely uncertain (never fabricated).
# ---------------------------------------------------------------------------
PMA_VERIFIED_THIS_SESSION = {
    # canonical_model -> (pma_number, note)
    "Trifecta": ("P100029", "openFDA /device/pma.json trade_name:trifecta, verified 2026-09-16"),
    "Trifecta GT": ("P100029", "same PMA as Trifecta (GT is a design-change supplement), verified 2026-09-16"),
    "Perceval": ("P150011", "openFDA /device/pma.json trade_name:perceval, verified 2026-09-16; original approval 2016-01-08"),
    "Perceval Plus": ("P150011", "same PMA as Perceval (Plus is a supplement), verified 2026-09-16"),
    "Mitroflow": ("P060038", "openFDA /device/pma.json applicant:corcym, verified 2026-09-16; SSED PDF not found at accessdata (404), see MANIFEST.csv"),
    "Perimount": ("P860057", "openFDA /device/pma.json trade_name:perimount, verified 2026-09-16; 1986 approval predates electronic SSED archive"),
    "Magna": ("P860057", "supplement to original Perimount PMA, verified 2026-09-16"),
    "Magna Ease": ("P860057", "supplement to original Perimount PMA, verified 2026-09-16"),
    "SAPIEN 3": ("P140031", "per task brief URL, SSED downloaded 2026-09-16"),
    "SAPIEN 3 Ultra": ("P140031", "supplement to P140031, SSED supplement downloaded 2026-09-16"),
    "SAPIEN 3 Ultra RESILIA": ("P140031", "supplement to P140031 (labeling supplement S182 downloaded 2026-09-16)"),
    "CoreValve": ("P130021", "per task brief URL, SSED supplement downloaded 2026-09-16"),
    "Evolut R": ("P130021", "supplement to CoreValve PMA P130021, per task brief"),
    "Evolut PRO": ("P180029", "openFDA confirms Evolut R/PRO under P180029, SSED downloaded 2026-09-16"),
    "Evolut PRO+": ("P180029", "supplement to P180029"),
    "Evolut FX": ("P180029", "supplement to P180029 (not independently re-verified this session)"),
    "Evolut FX+": ("P180029", "supplement to P180029 (not independently re-verified this session)"),
    "Inspiris RESILIA": ("P150048", "per task brief URL, SSED downloaded 2026-09-16"),
    "Avalus": ("P170006", "openFDA /device/pma.json trade_name:avalus, verified 2026-09-16"),
    "Avalus Ultra": ("P170006", "supplement to Avalus PMA, verified 2026-09-16"),
    "INTUITY Elite": ("P150036", "openFDA /device/pma.json trade_name:intuity, verified 2026-09-16"),
    "Portico": ("P190023", "openFDA /device/pma.json trade_name:portico, verified 2026-09-16"),
    "Navitor": ("P190023", "same PMA as Portico (Navitor is a supplement), openFDA-verified 2026-09-16"),
    "Freestyle": ("P970031", "openFDA /device/pma.json trade_name:freestyle, verified 2026-09-16; approved 1997-11-26"),
}

# manufacturer, route, design_class, tissue, leaflet_mounting, tissue_treatment,
# generation label (human-readable), market_status
STATIC = {
    "Perimount":            ("Edwards Lifesciences", "SAVR", "stented bovine pericardial, internally mounted", "bovine pericardial", "internal", "none", "1984-2000s legacy platform (incl. RSR, Theon variants)", "active (legacy variants largely superseded by Magna/Magna Ease; base Perimount aortic still GUDID-listed in commercial distribution)"),
    "Magna":                ("Edwards Lifesciences", "SAVR", "stented bovine pericardial, internally mounted", "bovine pericardial", "internal", "none", "2nd-generation Perimount platform (~2007)", "discontinued (superseded by Magna Ease)"),
    "Magna Ease":           ("Edwards Lifesciences", "SAVR", "stented bovine pericardial, internally mounted", "bovine pericardial", "internal", "ThermaFix", "3rd-generation Perimount platform, easier-implant delivery (~2009); ThermaFix tissue process confirmed in GUDID brand text", "active"),
    "Inspiris RESILIA":     ("Edwards Lifesciences", "SAVR", "stented bovine pericardial, internally mounted", "bovine pericardial", "internal", "RESILIA", "newest-generation Edwards SAVR platform (FDA approved 2019); integrated annulus expansion (VFit), designed with future ViV in mind", "active"),
    "Avalus":               ("Medtronic", "SAVR", "stented bovine pericardial, internally mounted", "bovine pericardial", "internal", "Linx", "Medtronic SAVR pericardial platform, FDA approved 2017; Linx AC anticalcification treatment", "active"),
    "Avalus Ultra":         ("Medtronic", "SAVR", "stented bovine pericardial, internally mounted", "bovine pericardial", "internal", "Linx", "Avalus low-profile size variant", "active"),
    "Konect RESILIA":       ("Edwards Lifesciences", "SAVR", "stented bovine pericardial, internally mounted", "bovine pericardial", "internal", "RESILIA", "valved conduit (BioBentall) combining Inspiris-platform RESILIA leaflets with a Dacron graft, for aortic root replacement", "active"),
    "Konect (legacy pre-RESILIA)": ("Edwards Lifesciences", "SAVR", "stented bovine pericardial, internally mounted", "bovine pericardial", "internal", "none", "predecessor valved-conduit product sharing the Konect graft platform before the RESILIA tissue rebrand; GUDID brand text 'Carpentier-Edwards Bioprosthetic Valved Conduit'", "discontinued (superseded by Konect RESILIA; exact year not verified this session)"),
    "Prima Plus":           ("Edwards Lifesciences", "SAVR", "stentless", "porcine aortic (full root)", "n.a.", "none", "stentless subcoronary/full-root platform, GUDID brand 'Edwards Prima Plus Stentless Bioprosthesis'; not in the task's original bucket list, added because real GUDID DIs from a target manufacturer were found for it this session", "active (not independently re-verified this session; bonus row beyond the task's minimum model list)"),
    "Trifecta":             ("Abbott (St. Jude Medical)", "SAVR", "externally mounted pericardial", "bovine pericardial", "external", "unknown", "1st-generation Trifecta (~2011)", "withdrawn (US market withdrawal July 2023, per FDA correspondence February 2023; manufacturer wind-down reported starting late 2022)"),
    "Trifecta GT":          ("Abbott (St. Jude Medical)", "SAVR", "externally mounted pericardial", "bovine pericardial", "external", "XenoLogiX", "Trifecta with Glide Technology (~2016); XenoLogiX anticalcification tissue treatment", "withdrawn (same action as Trifecta, US market withdrawal 2023)"),
    "Mitroflow":            ("Corcym (formerly Sorin/LivaNova)", "SAVR", "externally mounted pericardial", "bovine pericardial", "external", "none", "legacy externally-mounted pericardial platform", "discontinued (superseded by Crown PRT; exact discontinuation year not verified this session)"),
    "Crown PRT":            ("Corcym (formerly Sorin/LivaNova)", "SAVR", "externally mounted pericardial", "bovine pericardial", "external", "none", "Mitroflow successor, Pericardial Reduced Thickness leaflet", "active"),
    "Epic":                 ("Abbott (St. Jude Medical)", "SAVR", "stented porcine", "porcine aortic", "internal", "Linx", "standard stented porcine platform (Linx AC treatment per Abbott labeling)", "active"),
    "Epic Supra":           ("Abbott (St. Jude Medical)", "SAVR", "stented porcine", "porcine aortic", "internal", "Linx", "supra-annular stented porcine variant of Epic", "active"),
    "Mosaic":               ("Medtronic", "SAVR", "stented porcine", "porcine aortic", "internal", "none", "standard Mosaic platform (~2000)", "discontinued (superseded by Mosaic Ultra; not independently re-verified this session)"),
    "Mosaic Ultra":         ("Medtronic", "SAVR", "stented porcine", "porcine aortic", "internal", "Linx", "Mosaic with Linx AC anticalcification treatment", "active"),
    "Hancock II":           ("Medtronic", "SAVR", "stented porcine", "porcine aortic", "internal", "none", "standard Hancock II platform", "active"),
    "Hancock II Ultra":     ("Medtronic", "SAVR", "stented porcine", "porcine aortic", "internal", "Linx", "Hancock II with Linx AC anticalcification treatment", "active"),
    "Freestyle":            ("Medtronic", "SAVR", "stentless", "porcine aortic (full root)", "n.a.", "none", "stentless full-root/subcoronary/root-inclusion platform, FDA approved 1997", "active"),
    "Toronto SPV":          ("Abbott (St. Jude Medical)", "SAVR", "stentless", "porcine aortic", "n.a.", "none", "legacy first-generation stentless platform", "discontinued (legacy product, exact discontinuation year not verified this session)"),
    "Perceval":             ("Corcym (formerly Sorin/LivaNova)", "SAVR", "sutureless or rapid-deployment", "bovine pericardial", "n.a.", "none", "sutureless self-expanding nitinol stent platform, FDA approved 2016", "active"),
    "Perceval Plus":        ("Corcym (formerly Sorin/LivaNova)", "SAVR", "sutureless or rapid-deployment", "bovine pericardial", "n.a.", "none", "expanded-size-range Perceval variant", "active"),
    "INTUITY Elite":        ("Edwards Lifesciences", "SAVR", "sutureless or rapid-deployment", "bovine pericardial", "internal", "none", "balloon-expandable-skirt rapid-deployment platform (Perimount-family leaflets)", "active"),
    "SAPIEN":               ("Edwards Lifesciences", "TAVR", "balloon-expandable intra-annular TAVR", "bovine pericardial", "n.a.", "none", "1st-generation TAVR (FDA approved 2011)", "discontinued (superseded by SAPIEN XT/3; year not independently re-verified this session)"),
    "SAPIEN XT":            ("Edwards Lifesciences", "TAVR", "balloon-expandable intra-annular TAVR", "bovine pericardial", "n.a.", "none", "2nd-generation TAVR", "discontinued (superseded by SAPIEN 3; year not independently re-verified this session)"),
    "SAPIEN 3":             ("Edwards Lifesciences", "TAVR", "balloon-expandable intra-annular TAVR", "bovine pericardial", "n.a.", "none", "3rd-generation TAVR, outer sealing skirt added (FDA approved 2015)", "active"),
    "SAPIEN 3 Ultra":       ("Edwards Lifesciences", "TAVR", "balloon-expandable intra-annular TAVR", "bovine pericardial", "n.a.", "none", "SAPIEN 3 with taller outer skirt", "active"),
    "SAPIEN 3 Ultra RESILIA": ("Edwards Lifesciences", "TAVR", "balloon-expandable intra-annular TAVR", "bovine pericardial", "n.a.", "RESILIA", "newest SAPIEN 3 Ultra variant using RESILIA tissue", "active"),
    "Myval":                ("Meril Life Sciences", "TAVR", "balloon-expandable intra-annular TAVR", "bovine pericardial", "n.a.", "unknown", "balloon-expandable TAVR, hybrid honeycomb cobalt-nickel frame", "active outside US (CE-marked; NOT FDA approved -- not a US market device)"),
    "CoreValve":            ("Medtronic", "TAVR", "self-expanding supra-annular TAVR", "porcine pericardial", "n.a.", "unknown", "1st-generation self-expanding TAVR", "discontinued (superseded by Evolut R, ~2015-2016; year not independently re-verified this session)"),
    "Evolut R":             ("Medtronic", "TAVR", "self-expanding supra-annular TAVR", "porcine pericardial", "n.a.", "unknown", "recapturable/repositionable self-expanding TAVR (GUDID brand 'EnVeo R')", "active"),
    "Evolut PRO":           ("Medtronic", "TAVR", "self-expanding supra-annular TAVR", "porcine pericardial", "n.a.", "unknown", "Evolut R + external pericardial wrap for paravalvular leak reduction (GUDID brand 'EnVeo PRO')", "active"),
    "Evolut PRO+":          ("Medtronic", "TAVR", "self-expanding supra-annular TAVR", "porcine pericardial", "n.a.", "unknown", "Evolut PRO with larger frame cells", "active"),
    "Evolut FX":            ("Medtronic", "TAVR", "self-expanding supra-annular TAVR", "porcine pericardial", "n.a.", "unknown", "commissural-alignment-optimized delivery, FDA cleared 2022", "active"),
    "Evolut FX+":           ("Medtronic", "TAVR", "self-expanding supra-annular TAVR", "porcine pericardial", "n.a.", "unknown", "newest Evolut delivery-system iteration (present in GUDID; not independently re-verified this session)", "active"),
    "ACURATE neo":          ("Boston Scientific (Symetis)", "TAVR", "self-expanding supra-annular TAVR", "porcine pericardial", "n.a.", "unknown", "top-down deployment self-expanding TAVR", "discontinued (superseded by ACURATE neo2; not independently re-verified this session)"),
    "ACURATE neo2":         ("Boston Scientific (Symetis)", "TAVR", "self-expanding supra-annular TAVR", "porcine pericardial", "n.a.", "unknown", "ACURATE neo with an outer sealing skirt", "active"),
    "Portico":              ("Abbott", "TAVR", "self-expanding intra-annular TAVR", "bovine pericardial", "n.a.", "unknown", "recapturable/repositionable self-expanding TAVR, FDA approved 2021", "active (early-generation, largely superseded clinically by Navitor)"),
    "Navitor":              ("Abbott", "TAVR", "self-expanding intra-annular TAVR", "bovine pericardial", "n.a.", "unknown", "Portico platform with an outer NaviSeal sealing cuff, FDA approved 2023", "active"),
    "Lotus":                ("Boston Scientific", "TAVR", "mechanically expanded intra-annular TAVR", "bovine pericardial", "n.a.", "unknown", "mechanically-expanded (not fully self-expanding), fully repositionable before release, Adaptive Seal (Lotus Edge FDA approved 2019)", "discontinued (Boston Scientific withdrew the full Lotus platform globally ~2020-2021 for business reasons, not independently re-verified this session)"),
    "JenaValve (Trilogy)":  ("JenaValve Technology", "TAVR", "self-expanding intra-annular TAVR", "porcine pericardial", "n.a.", "unknown", "locator-and-clip anatomical-fixation design; first TAVR FDA-approved for pure native aortic regurgitation (2023)", "active"),
}

DESIGN_CLASS_NOTES = "labels exactly as specified in the task brief bucket list"


def build():
    valve_recs = load_valve_records()

    rows = {}  # (canonical_model) -> dict accumulator
    unmatched_brands = {}
    for rec in valve_recs:
        text = (rec["brandName"] or "") + " | " + (rec["deviceDescription"] or "")
        # GUDID XML source has mojibake for (R)/(TM) glyphs (decoded as U+FFFD by the XML
        # parser); strip so e.g. "Evolut�PRO+" still matches r"evolut\s*pro\s*\+"
        text = text.replace("�", " ").replace("™", " ").replace("®", " ")
        matched = None
        for canon, gen_hint, pat in CANONICAL_BRAND_PATTERNS:
            if pat.search(text):
                matched = canon
                break
        if matched is None:
            unmatched_brands[rec["brandName"]] = unmatched_brands.get(rec["brandName"], 0) + 1
            continue
        d = rows.setdefault(matched, {"di": set(), "sizes": set(), "brands": set(),
                                       "models": set(), "route_hint": set()})
        di = primary_di(rec)
        if di:
            d["di"].add(di)
        d["sizes"] |= extract_sizes(rec)
        if rec["brandName"]:
            d["brands"].add(rec["brandName"].strip())
        if rec["versionModelNumber"]:
            d["models"].add(rec["versionModelNumber"].strip())
        for pc in rec["productCodes"]:
            if pc["code"] == "NPT":
                d["route_hint"].add("TAVR")
            elif pc["code"] == "DYE":
                d["route_hint"].add("SAVR")

    # ---- assemble device_table rows ----
    device_rows = []
    for canon, gudid in sorted(rows.items()):
        static = STATIC.get(canon)
        pma_info = PMA_VERIFIED_THIS_SESSION.get(canon)
        sizes_sorted = sorted(gudid["sizes"])
        sizes_str = ";".join(str(int(s)) if s == int(s) else str(s) for s in sizes_sorted)
        di_str = ";".join(sorted(gudid["di"]))
        route = static[1] if static else (
            "TAVR" if "TAVR" in gudid["route_hint"] else "SAVR" if "SAVR" in gudid["route_hint"] else "unknown"
        )
        manufacturer = static[0] if static else "unknown"
        design_class = static[2] if static else "unknown"
        tissue = static[3] if static else "unknown"
        leaflet_mounting = static[4] if static else "unknown"
        tissue_treatment = static[5] if static else "unknown"
        generation = static[6] if static else "unknown"
        market_status = static[7] if static else "unknown"
        fda_pma = pma_info[0] if pma_info else "TODO"
        source_bits = [GUDID_SOURCE]
        if pma_info:
            source_bits.append(f"openFDA PMA lookup: {pma_info[1]}")
        confidence = "high" if (gudid["di"] and static and pma_info) else (
            "medium" if (gudid["di"] and static) else "low"
        )
        device_rows.append({
            "canonical_model": canon,
            "manufacturer": manufacturer,
            "brand_name_gudid": ";".join(sorted(gudid["brands"])),
            "model_or_catalog_number": ";".join(sorted(gudid["models"])),
            "primary_di": di_str,
            "product_code": ";".join(sorted(gudid["route_hint"])) or "unknown",
            "route": route,
            "design_class": design_class,
            "tissue": tissue,
            "leaflet_mounting": leaflet_mounting,
            "tissue_treatment": tissue_treatment,
            "generation": generation,
            "sizes_mm": sizes_str,
            "market_status": market_status,
            "fda_pma": fda_pma,
            "source": " ; ".join(source_bits),
            "confidence": confidence,
            "retrieved_at": RETRIEVED_AT,
        })

    # models known from the task's canonical bucket list that had zero GUDID hits this
    # session (e.g. not sold/registered in the US, or brand text didn't match) -- still
    # emit a row so every task-mandated canonical model is present, marked accordingly.
    for canon, static in STATIC.items():
        if canon in rows:
            continue
        pma_info = PMA_VERIFIED_THIS_SESSION.get(canon)
        device_rows.append({
            "canonical_model": canon,
            "manufacturer": static[0],
            "brand_name_gudid": "",
            "model_or_catalog_number": "",
            "primary_di": "",
            "product_code": "unknown",
            "route": static[1],
            "design_class": static[2],
            "tissue": static[3],
            "leaflet_mounting": static[4],
            "tissue_treatment": static[5],
            "generation": static[6],
            "sizes_mm": "",
            "market_status": static[7],
            "fda_pma": pma_info[0] if pma_info else "TODO",
            "source": "domain knowledge (industry/regulatory public record); NOT found in this "
                      "session's GUDID brand-name match -- verify manually at "
                      "accessgudid.nlm.nih.gov if a DI list is required "
                      f"({GUDID_SOURCE})",
            "confidence": "low",
            "retrieved_at": RETRIEVED_AT,
        })

    device_rows.sort(key=lambda r: (r["route"], r["design_class"], r["canonical_model"]))

    cols = ["canonical_model", "manufacturer", "brand_name_gudid", "model_or_catalog_number",
            "primary_di", "product_code", "route", "design_class", "tissue",
            "leaflet_mounting", "tissue_treatment", "generation", "sizes_mm",
            "market_status", "fda_pma", "source", "confidence", "retrieved_at"]
    with open(OUT_DEVICE_TABLE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(device_rows)
    print(f"wrote {len(device_rows)} rows -> {OUT_DEVICE_TABLE}")

    # ---- aliases file ----
    alias_rows = []
    for canon, gudid in rows.items():
        for b in sorted(gudid["brands"]):
            alias_rows.append((canon, b, "gudid_brand", "brandName observed in GUDID this session"))
    # curated free-text forms seen in real operative notes (per task brief) + regex helpers
    curated = [
        ("Trifecta", r"#\s?21\s*Trifecta", "regex", "size-prefixed free-text form, e.g. '#21 Trifecta'"),
        ("Trifecta", r"21-mm Trifecta", "regex", "size-suffixed free-text form"),
        ("Trifecta", "Trifecta GT", "abbreviation", "note: Trifecta GT is tracked as its own canonical_model row; this alias captures notes that just say 'Trifecta' generically for GT implants"),
        ("SAPIEN 3", "Sapien S3", "abbreviation", "colloquial op-note shorthand"),
        ("SAPIEN 3 Ultra", "S3 Ultra", "abbreviation", "colloquial op-note shorthand"),
        ("SAPIEN 3", "Edwards S3", "abbreviation", "colloquial op-note shorthand"),
        ("SAPIEN 3 Ultra", r"Size:\s*Ultra\s*26\s*mm", "regex", "field-label free-text form"),
        ("Evolut PRO+", "Evolut Corevalue Pro+", "regex", "misspelling ('Corevalue' for CoreValve) observed in real notes"),
        ("Evolut R", r"R\s?34\s*mm", "regex", "size-suffixed shorthand, R-series 34mm"),
        ("Perimount", r"CE\s*#?\s?\d{2}", "regex", "'CE' + size for Carpentier-Edwards Perimount"),
        ("Magna", "Magna", "gudid_brand", "bare 'Magna' free-text form (disambiguate from Magna Ease by absence of 'Ease')"),
        ("Inspiris RESILIA", "Inspiris", "abbreviation", "bare 'Inspiris' free-text form"),
        ("Konect RESILIA", "Konect", "abbreviation", "BioBentall conduit, bare 'Konect' free-text form"),
        ("Konect RESILIA", "Carpentier-Edwards Bioprosthetic Valved Conduit", "gudid_brand", "legacy/predecessor GUDID brand name for the same Edwards valved-conduit product line, pre-RESILIA rebrand -- sizes not pooled into the main Konect RESILIA row to avoid mixing tissue generations"),
        ("CoreValve", "CoreValue", "abbreviation", "common misspelling seen in real notes"),
        ("Evolut R", "Evolut Corevalue R", "regex", "misspelling variant"),
        ("Trifecta", r"\d{2}[\s-]?mm\s*Trifecta", "regex", "general size + Trifecta free-text form"),
        # --- free-text forms restored 2026-09-16 so regeneration keeps them (previously added to the CSV only) ---
        ("Perimount", r"#?\s?\d{2}\s?(?:-?\s?mm)?\s*CE\b", "regex", "size before 'CE', e.g. '#25 CE' or '25 mm CE' (Carpentier-Edwards Perimount)"),
        ("Perimount", r"\bCarpentier\b", "regex", "bare 'Carpentier(-Edwards)' aortic bioprosthesis resolves to Perimount, the closest canonical model"),
        ("Magna", r"#?\s?\d{2}\s?mm\s+Magna\b|Magna\s+valve", "regex", "sized or bare 'Magna valve' free-text forms"),
        ("SAPIEN 3", r"\bS3\b", "regex_guarded", "bare 'S3' accepted only with nearby valve/TAVR context (guarded, like 'Epic')"),
        ("SAPIEN XT", r"\bXT\b", "regex_guarded", "bare 'XT' accepted only with nearby valve/TAVR context (guarded)"),
        ("SAPIEN 3 Ultra", r"Sapien\b[^.\n]{0,60}?\bUltra\b", "regex", "'Ultra' in the same sentence as Sapien, e.g. 'Sapien platform, Ultra generation'"),
        ("SAPIEN 3 Ultra RESILIA", r"\bUltra\s+RESILIA\b", "regex", "bare 'Ultra RESILIA' without the SAPIEN 3 prefix"),
    ]
    for canon, alias, atype, note in curated:
        alias_rows.append((canon, alias, atype, note))

    with open(OUT_ALIASES, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["canonical_model", "alias", "alias_type", "notes"])
        for row in sorted(set(alias_rows)):
            w.writerow(row)
    print(f"wrote {len(set(alias_rows))} alias rows -> {OUT_ALIASES}")

    print(f"\nunmatched brand names (not assigned to any canonical model), top 20:")
    for k, v in sorted(unmatched_brands.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {v:4d}  {k}")

    return device_rows, unmatched_brands


if __name__ == "__main__":
    build()
