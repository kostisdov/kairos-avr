"""Rule-based extraction of a valve passport with evidence spans.

Reuses the compiled patterns, alias tables and context guards of :mod:`kairos.passport`
(the tested rules layer) and adds what the HTTP contract needs: character spans for every
extracted field, per-field confidence, device metadata from ``device_table.csv`` and
per-note echo observations. Nothing here stores or logs note text; the only text that
leaves this module is the short evidence snippet for each span (capped at 160 characters).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache, lru_cache

import pandas as pd

from kairos.extraction.schema import (
    Confidence,
    EchoObservation,
    Evidence,
    Passport,
    PassportEvent,
    PassportSource,
    Prescreen,
    new_passport_id,
)
from kairos.passport import (
    DVI_PATS,
    EOA_PATS,
    EVENT_PATS,
    MG_PATS,
    REFERENCE_DIR,
    REGURG_PATS_NUM,
    REGURG_PATS_WORD,
    SAVR_PAT,
    SIZE_PATS,
    TAVR_PAT,
    _context_flags,
    _is_genuine_ambiguous_match,
    in_negated_field,
    load_device_patterns,
    negated_field_spans,
)

SNIPPET_MAX = 160
EVENT_TYPE_MAP = {"ViV": "valve_in_valve", "redo": "redo",
                  "prosthetic_dysfunction": "dysfunction", "endocarditis": "endocarditis",
                  "thrombosis": "thrombosis"}
IMPLANT_YEAR_PAT = re.compile(
    r"(?:implant\w*|placed|replacement|\bTAVR\b|\bTAVI\b|\bAVR\b|\bSAVR\b)[^.\n]{0,60}?"
    r"\b(?:in|on|since|from)\s+(?:\d{1,2}/\d{1,2}/)?((?:19|20)\d{2})\b", re.I)
LVEF_PAT = re.compile(r"(?:LVEF|ejection fraction|\bEF\b)\s*(?:is|of|=|:)?\s*(?:approximately|about|~)?\s*(\d{2})\s*%", re.I)
PEAK_VEL_PAT = re.compile(r"(?:peak|maximum|max)\s+(?:aortic\s+)?(?:jet\s+)?velocity\s*(?:is|of|=|:)?\s*(\d(?:\.\d{1,2})?)\s*m/s", re.I)
PROSTHESIS_PAT = re.compile(
    r"bioprosth|prosthetic (?:aortic )?valve|aortic valve replacement|\bTAVR\b|\bTAVI\b|\bSAVR\b|"
    r"\bAVR\b|transcatheter aortic|valve[- ]in[- ]valve|tissue (?:valve|implant)", re.I)
_WORD_TO_GRADE = {"none": "none", "trivial": "none", "trace": "trace", "mild": "mild",
                  "mild-to-moderate": "mild", "moderate": "moderate",
                  "moderate-to-severe": "severe", "severe": "severe"}
_NUM_TO_GRADE = {1: "mild", 2: "moderate", 3: "severe", 4: "severe"}


@lru_cache(maxsize=1)
def _patterns():
    return load_device_patterns()


@lru_cache(maxsize=1)
def device_table() -> pd.DataFrame:
    return pd.read_csv(REFERENCE_DIR / "device_table.csv")


def normalise_market_status(raw: str | None) -> str:
    s = (raw or "").strip().lower()
    if s.startswith("withdrawn"):
        return "withdrawn"
    if s.startswith("discontinued"):
        return "discontinued"
    if s.startswith("active"):
        return "active"
    return "unknown"


@cache
def device_info(canonical_model: str | None) -> dict:
    if not canonical_model:
        return {}
    dt = device_table()
    row = dt[dt["canonical_model"] == canonical_model]
    if row.empty:
        return {}
    r = row.iloc[0]
    return {"route": r["route"] if r["route"] in ("SAVR", "TAVR") else "unknown",
            "design_class": r["design_class"],
            "generation": r["generation"] if isinstance(r["generation"], str) else None,
            "market_status": normalise_market_status(r.get("market_status")),
            "tissue_treatment": r.get("tissue_treatment")}


def known_models() -> list[str]:
    return sorted(device_table()["canonical_model"].dropna().unique().tolist())


def _snippet(text: str, start: int, end: int) -> str:
    s = text[start:end]
    return s if len(s) <= SNIPPET_MAX else s[:SNIPPET_MAX]


def _ev(text: str, field: str, start: int, end: int, method: str = "rule") -> Evidence:
    return Evidence(field=field, span=(start, end), text=_snippet(text, start, end), method=method)


@dataclass
class ModelMatch:
    canonical: str
    start: int
    end: int
    guarded: bool


def find_models(text: str) -> list[ModelMatch]:
    patterns, _ = _patterns()
    found: list[ModelMatch] = []
    for canon, pat_list in patterns.items():
        for pat, guarded in pat_list:
            hit = None
            for m in pat.finditer(text):
                if guarded and not _is_genuine_ambiguous_match(text, m.start(), m.end()):
                    continue
                hit = m
                break
            if hit is not None:
                found.append(ModelMatch(canon, hit.start(), hit.end(), guarded))
                break
    return found


def most_specific(matches: list[ModelMatch]) -> ModelMatch | None:
    """Longest canonical name wins (SAPIEN 3 Ultra over SAPIEN 3 over SAPIEN); ties by name."""
    if not matches:
        return None
    return sorted(matches, key=lambda m: (-len(m.canonical), m.canonical))[0]


def mentions_prosthesis(text: str) -> bool:
    return bool(PROSTHESIS_PAT.search(text)) or bool(find_models(text))


def _grade_from_word(word: str) -> str | None:
    return _WORD_TO_GRADE.get(word.lower())


def _classify(prosth: bool, native: bool) -> str:
    if prosth and not native:
        return "prosthetic"
    if native and not prosth:
        return "native"
    return "uncertain"


@dataclass
class RulesResult:
    passport: Passport
    echo_observations: list[EchoObservation]
    prescreen: Prescreen
    warnings: list[str]


def extract_rules(text: str, note_ref: str, note_type: str, note_date: str,
                  passport_id: str | None = None) -> RulesResult:
    warnings: list[str] = []
    evidence: list[Evidence] = []
    matches = find_models(text)
    primary = most_specific(matches)
    canonical = primary.canonical if primary else None
    info = device_info(canonical)
    if primary:
        evidence.append(_ev(text, "canonical_model", primary.start, primary.end))
        for m in matches:
            if m.canonical != canonical:
                evidence.append(_ev(text, "all_models_mentioned", m.start, m.end))
    conf_model = 0.0 if not primary else (0.75 if primary.guarded else 0.9)

    # route ------------------------------------------------------------------------------
    tavr_m = TAVR_PAT.search(text)
    savr_m = SAVR_PAT.search(text)
    implant_note = note_type in ("operative", "procedure")
    route, conf_route = "unknown", 0.0
    if implant_note and (tavr_m or savr_m):
        if tavr_m:
            route, conf_route = "TAVR", 0.9
            evidence.append(_ev(text, "route", tavr_m.start(), tavr_m.end()))
        else:
            route, conf_route = "SAVR", 0.85
            evidence.append(_ev(text, "route", savr_m.start(), savr_m.end()))
    elif info.get("route") in ("SAVR", "TAVR"):
        route, conf_route = info["route"], 0.7
    elif tavr_m or savr_m:
        m = tavr_m or savr_m
        route, conf_route = ("TAVR" if tavr_m else "SAVR"), 0.5
        evidence.append(_ev(text, "route", m.start(), m.end()))
    device_route = info.get("route")
    if device_route in ("SAVR", "TAVR") and route != "unknown" and device_route != route:
        warnings.append(f"route from note pattern ({route}) differs from device table ({device_route}) for {canonical}")

    # size --------------------------------------------------------------------------------
    size_hits: list[tuple[int, int, int]] = []
    for pat in SIZE_PATS:
        for m in pat.finditer(text):
            v = int(m.group(1))
            if 17 <= v <= 34:
                size_hits.append((v, m.start(), m.end()))
    size_mm, conf_size = None, 0.0
    if size_hits:
        if primary:
            anchor = primary.start
            v, s, e = min(size_hits, key=lambda h: abs(h[1] - anchor))
            conf_size = 0.8 if abs(s - anchor) <= 80 else 0.6
        else:
            v, s, e = size_hits[0]
            conf_size = 0.5
        size_mm = v
        evidence.append(_ev(text, "size_mm", s, e))
        if len({h[0] for h in size_hits}) > 1:
            warnings.append("more than one size mentioned; nearest to the device name was chosen")

    # implant date --------------------------------------------------------------------------
    implant_date = None
    if implant_note and route != "unknown":
        implant_date = note_date
    else:
        m = IMPLANT_YEAR_PAT.search(text)
        if m:
            implant_date = m.group(1)
            evidence.append(_ev(text, "implant_date", m.start(1), m.end(1)))

    # events --------------------------------------------------------------------------------
    events = []
    negated = negated_field_spans(text)
    for key, pat in EVENT_PATS.items():
        m = next((c for c in pat.finditer(text) if not in_negated_field(c.start(), negated)), None)
        if m:
            events.append(PassportEvent(type=EVENT_TYPE_MAP[key], date=note_date,
                                        span=(m.start(), m.end())))
            evidence.append(_ev(text, f"event:{EVENT_TYPE_MAP[key]}", m.start(), m.end()))

    passport = Passport(
        passport_id=passport_id or new_passport_id(),
        source=PassportSource(note_ref=note_ref, note_type=note_type, date=note_date),
        route=route, canonical_model=canonical, design_class=info.get("design_class"),
        generation=info.get("generation"), size_mm=size_mm, implant_date=implant_date,
        market_status=info.get("market_status", "unknown"), evidence=evidence,
        confidence=Confidence(route=conf_route, canonical_model=conf_model, size_mm=conf_size),
        events=events)

    echos = extract_echo_observations(text, passport.passport_id, note_date)
    prescreen = Prescreen(mentions_prosthesis=mentions_prosthesis(text), method="rules")
    return RulesResult(passport=passport, echo_observations=echos, prescreen=prescreen,
                       warnings=warnings)


def _numeric_hits(text: str, pats, lo: float, hi: float, as_float: bool):
    hits = []
    for pat in pats:
        for m in pat.finditer(text):
            raw = m.group(1)
            v = float(raw) if as_float else int(raw)
            if lo <= v <= hi:
                prosth, native = _context_flags(text, m.start(), m.end())
                hits.append((v, m.start(1), m.end(1), _classify(prosth, native)))
    return hits


def extract_echo_observations(text: str, passport_id: str, note_date: str) -> list[EchoObservation]:
    """One observation per (prosthetic | native | uncertain) class present in the note."""
    grads = _numeric_hits(text, MG_PATS, 0, 120, False)
    dvis = _numeric_hits(text, DVI_PATS, 0.05, 1.2, True)
    eoas = _numeric_hits(text, EOA_PATS, 0.2, 6.0, True)
    regurg = []
    for pat in REGURG_PATS_WORD:
        for m in pat.finditer(text):
            g = _grade_from_word(m.group(1))
            if g is None:
                continue
            prosth, native = _context_flags(text, m.start(), m.end())
            regurg.append((g, m.start(), m.end(), _classify(prosth, native)))
    for pat in REGURG_PATS_NUM:
        for m in pat.finditer(text):
            g = _NUM_TO_GRADE.get(int(m.group(1)))
            if g is None:
                continue
            prosth, native = _context_flags(text, m.start(), m.end())
            regurg.append((g, m.start(), m.end(), _classify(prosth, native)))
    lvef_hits = [(int(m.group(1)), m.start(1), m.end(1)) for m in LVEF_PAT.finditer(text)
                 if 5 <= int(m.group(1)) <= 90]
    vel_hits = [(float(m.group(1)), m.start(1), m.end(1)) for m in PEAK_VEL_PAT.finditer(text)
                if 0.5 <= float(m.group(1)) <= 8]

    classes = []
    for hits in (grads, dvis, eoas, regurg):
        for h in hits:
            if h[3] not in classes:
                classes.append(h[3])
    if not classes and (lvef_hits or vel_hits):
        classes = ["uncertain"]

    out: list[EchoObservation] = []
    for cls in classes:
        ev: list[Evidence] = []

        def first(hits, field, cls=cls, ev=ev):
            for h in hits:
                if h[3] == cls:
                    ev.append(_ev(text, field, h[1], h[2]))
                    return h[0]
            return None

        mg = first(grads, "mean_gradient_mmhg")
        dvi = first(dvis, "dvi")
        eoa = first(eoas, "eoa_cm2")
        ar = first(regurg, "ar_grade")
        lvef = None
        if lvef_hits and cls != "native":
            lvef = lvef_hits[0][0]
            ev.append(_ev(text, "lvef_pct", lvef_hits[0][1], lvef_hits[0][2]))
        vel = None
        if vel_hits and cls != "native":
            vel = vel_hits[0][0]
            ev.append(_ev(text, "peak_velocity_ms", vel_hits[0][1], vel_hits[0][2]))
        if all(x is None for x in (mg, dvi, eoa, ar, lvef, vel)):
            continue
        out.append(EchoObservation(passport_id=passport_id, date=note_date,
                                   mean_gradient_mmhg=mg, peak_velocity_ms=vel, eoa_cm2=eoa,
                                   dvi=dvi, ar_grade=ar, lvef_pct=lvef,
                                   native_vs_prosthetic=cls, source="rule", evidence=ev))
    return out


def locate_quote(text: str, quote: str | None) -> tuple[int, int] | None:
    """Find a model-supplied quote in the note; used to turn LLM evidence into spans."""
    if not quote:
        return None
    q = quote.strip()
    if not q:
        return None
    i = text.find(q)
    if i < 0:
        i = text.lower().find(q.lower())
    if i < 0:
        return None
    return (i, i + len(q))
