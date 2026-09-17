"""Adjudication framework (design section 2, proposal protocol section 4).

Candidates are identified from echo observations flagged by the VARC-3 module against the
reference study of the same patient. The framework proposes; reviewers decide. Its rules:

* time zero is the first adequate echo 30 to 180 days after implantation (the reference
  study); change-based staging needs that study, so without it every echo is "uncertain";
* a stenotic endpoint needs the gradient criterion with the area or dimensionless-index
  criterion; a regurgitant endpoint needs the regurgitation criterion only;
* confirmation is the next study with a determinable stage, or death / reintervention before
  a second study is possible (:class:`AdjudicationPolicy`); the same rules label the
  synthetic cohorts and decide live eligibility; a single unconfirmed study that is followed by a study back at stage 0 or 1 is
  a transient finding and is classed ``uncertain`` rather than negative;
* a candidate inside a known, medically treated and resolved thrombosis episode is
  attributed to thrombosis and returns the patient to the risk set;
* overlapping mechanisms (an endocarditis or thrombosis mention near the candidate) are
  recorded as ``uncertain`` when they cannot be resolved.

The first candidate that survives these rules with mechanism ``svd`` is the primary
endpoint; its establishing echo is never a predictor of that endpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Iterable

from kairos.extraction.schema import EchoObservation, PassportEvent, parse_date
from kairos.grades import REGURG_ORDINAL, regurg_ordinal
from kairos.varc3 import Echo, VARC3Result, stage_hvd

AR_ORDINAL = REGURG_ORDINAL  # re-export; the mapping lives in kairos.grades
TIME_ZERO_WINDOW_DAYS = (30, 180)
MECHANISM_WINDOW_DAYS = 90


@dataclass
class EchoPoint:
    date: date
    mean_gradient: float | None = None
    eoa: float | None = None
    dvi: float | None = None
    ar_ordinal: int | None = None
    lvef: float | None = None
    svi: float | None = None
    peak_velocity: float | None = None
    index: int = -1

    def to_varc3(self) -> Echo:
        return Echo(mean_gradient_mmHg=self.mean_gradient, eoa_cm2=self.eoa, dvi=self.dvi,
                    regurg_grade=self.ar_ordinal)

    @property
    def adequate(self) -> bool:
        return self.mean_gradient is not None and (self.eoa is not None or self.dvi is not None)


def to_points(observations: Iterable[EchoObservation], prosthetic_only: bool = True) -> list[EchoPoint]:
    pts = []
    for i, o in enumerate(observations):
        if prosthetic_only and o.native_vs_prosthetic != "prosthetic":
            continue
        d = parse_date(o.date)
        if d is None:
            continue
        pts.append(EchoPoint(date=d, mean_gradient=o.mean_gradient_mmhg, eoa=o.eoa_cm2,
                             dvi=o.dvi, ar_ordinal=regurg_ordinal(o.ar_grade),
                             lvef=o.lvef_pct, svi=o.svi_ml_m2, peak_velocity=o.peak_velocity_ms,
                             index=i))
    return sorted(pts, key=lambda p: (p.date, p.index))


def select_reference(points: list[EchoPoint], implant_date: date | None,
                     window_days: tuple[int, int] = TIME_ZERO_WINDOW_DAYS) -> EchoPoint | None:
    """First adequate echo 30 to 180 days after implantation; None if there is none."""
    if implant_date is None:
        return None
    lo, hi = window_days
    for p in points:
        delta = (p.date - implant_date).days
        if lo <= delta <= hi and p.adequate:
            return p
    return None


def stage_point(p: EchoPoint, ref: EchoPoint | None) -> VARC3Result:
    return stage_hvd(p.to_varc3(), ref.to_varc3() if ref else None)


def phenotype_of(result: VARC3Result) -> str:
    d = result.details or {}
    regurg = bool(d.get("new_regurg_moderate_plus") or d.get("new_regurg_severe"))
    grad = (d.get("grad_rise") or 0) >= 10
    if result.stage not in ("2", "3"):
        return "uncertain"
    if regurg and grad:
        return "mixed"
    if regurg:
        return "regurgitant"
    return "stenotic"


@dataclass
class CandidateFlag:
    date: date
    stage: str
    phenotype: str
    reason: str
    index: int
    result: VARC3Result | None = field(repr=False, default=None)


def candidate_flags(points: list[EchoPoint], ref: EchoPoint | None) -> list[CandidateFlag]:
    flags = []
    for p in points:
        if ref is not None and p.date <= ref.date:
            continue
        r = stage_point(p, ref)
        if r.stage in ("2", "3"):
            flags.append(CandidateFlag(p.date, r.stage, phenotype_of(r), r.reason, p.index, r))
    return flags


@dataclass
class EndpointDecision:
    met: bool
    date: date | None = None
    stage: str | None = None
    mechanism: str = "uncertain"
    phenotype: str | None = None
    establishing_index: int | None = None
    confidence: str = "low"
    reason: str = ""
    reference_source: str = "none"
    uncertain_dates: list = field(default_factory=list)
    thrombosis_attributed_dates: list = field(default_factory=list)


@dataclass(frozen=True)
class AdjudicationPolicy:
    """The one adjudication policy for synthetic training labels and live eligibility.

    ``terminal_confirmation_days`` bounds "death or reintervention before a second study is
    possible": a terminal event later than this after the candidate, with no study in between,
    leaves the candidate unconfirmed (label: assumed; one yearly interval plus the three-month
    overdue grace)."""
    version: str = "2"
    confirmation_required: bool = True
    terminal_event_confirms: bool = True
    terminal_confirmation_days: int = 455


DEFAULT_POLICY = AdjudicationPolicy()

STATUSES = ("confirmed", "confirmed_by_terminal_event", "unconfirmed_pending", "uncertain_transient",
            "thrombosis_attributed", "no_candidate", "no_reference")


@dataclass
class AdjudicationRecord:
    status: str
    candidate_date: date | None = None       # first qualifying study of the surviving candidate
    confirmation_date: date | None = None    # confirming study, or the confirming terminal event
    adjudicated_date: date | None = None     # candidate date of a confirmed SVD endpoint (the detection date)
    mechanism: str = "uncertain"             # svd | endocarditis | uncertain
    confidence: str = "low"                  # high | moderate | low
    stage: str | None = None
    phenotype: str | None = None
    establishing_index: int | None = None
    reason: str = ""
    reference_source: str = "none"
    policy_version: str = DEFAULT_POLICY.version
    uncertain_dates: list = field(default_factory=list)
    thrombosis_attributed_dates: list = field(default_factory=list)

    @property
    def first_unresolved_date(self) -> date | None:
        """Earliest candidate that is neither a confirmed SVD endpoint nor thrombosis-attributed:
        transient (uncertain) findings, an unconfirmed candidate, or a candidate with another or
        unresolved mechanism."""
        dates = list(self.uncertain_dates)
        if self.candidate_date is not None and self.adjudicated_date is None:
            dates.append(self.candidate_date)
        return min(dates) if dates else None


def _within(d1: date, d2: date, days: int) -> bool:
    return abs((d1 - d2).days) <= days


def _determinable(stage: str) -> bool:
    return stage in ("0", "1", "2", "3")


def adjudicate_series(points: list[EchoPoint], ref: EchoPoint | None,
                      thrombosis_windows: Iterable[tuple[date, date]] = (),
                      mechanism_events: dict | None = None,
                      death_date: date | None = None,
                      reintervention_date: date | None = None,
                      policy: AdjudicationPolicy = DEFAULT_POLICY) -> AdjudicationRecord:
    """Pure adjudication core. ``mechanism_events`` maps an event type (``endocarditis``,
    ``thrombosis``) to its dates. Candidates are examined in date order; the first one that is
    neither thrombosis-attributed nor transient decides the record."""
    if ref is None:
        return AdjudicationRecord(status="no_reference", reason="no reference study: change-based staging impossible",
                                  policy_version=policy.version)
    flags = candidate_flags(points, ref)
    if not flags:
        return AdjudicationRecord(status="no_candidate", reference_source="patient baseline echo",
                                  reason="no echo reached VARC-3 stage 2 or 3 against the reference study",
                                  policy_version=policy.version)
    ev_dates = mechanism_events or {}
    windows = list(thrombosis_windows)
    terminal = [d for d in (death_date, reintervention_date) if d is not None]
    staged = [(p, stage_point(p, ref).stage) for p in points if p.date > ref.date]
    rec = AdjudicationRecord(status="no_candidate", reference_source="patient baseline echo", policy_version=policy.version)
    for f in flags:
        # 1. resolved, treated thrombosis episode -> intercurrent, back to the risk set
        if any(s <= f.date <= e for s, e in windows):
            rec.thrombosis_attributed_dates.append(f.date)
            continue
        # 2. confirmation by the next study with a determinable stage (or a terminal event before one)
        nxt = next(((p, st) for p, st in staged if p.date > f.date and _determinable(st)), None)
        if nxt is not None and nxt[1] in ("0", "1"):
            rec.uncertain_dates.append(f.date)
            continue
        if nxt is not None:
            status, confirmation, confidence = "confirmed", nxt[0].date, "high"
        else:
            term = min((d for d in terminal if d > f.date), default=None)
            if (policy.terminal_event_confirms and term is not None
                    and (term - f.date).days <= policy.terminal_confirmation_days):
                status, confirmation, confidence = "confirmed_by_terminal_event", term, "high"
            elif policy.confirmation_required:
                status, confirmation, confidence = "unconfirmed_pending", None, "moderate"
            else:
                status, confirmation, confidence = "confirmed", None, "moderate"
        # 3. mechanism overlap
        mechanism = "svd"
        if any(_within(f.date, d, MECHANISM_WINDOW_DAYS) for d in ev_dates.get("endocarditis", [])):
            mechanism = "endocarditis"
        elif any(_within(f.date, d, MECHANISM_WINDOW_DAYS) for d in ev_dates.get("thrombosis", [])):
            mechanism = "uncertain"
        rec.status = status
        rec.candidate_date = f.date
        rec.confirmation_date = confirmation
        rec.adjudicated_date = f.date if (mechanism == "svd" and status.startswith("confirmed")) else None
        rec.mechanism = mechanism
        rec.confidence = confidence if mechanism == "svd" else "low"
        rec.stage = f.stage
        rec.phenotype = f.phenotype
        rec.establishing_index = f.index
        rec.reason = f.reason
        return rec
    rec.status = "uncertain_transient" if rec.uncertain_dates else "thrombosis_attributed"
    rec.reason = ("candidates found but each was attributed to a resolved thrombosis episode "
                  "or classed uncertain (transient finding)")
    return rec


def adjudicate(points: list[EchoPoint], ref: EchoPoint | None,
               events: Iterable[PassportEvent] = (),
               thrombosis_windows: Iterable[tuple[date, date]] = (),
               death_date: date | None = None,
               reintervention_date: date | None = None) -> EndpointDecision:
    """Passport-facing wrapper of :func:`adjudicate_series`. ``met`` is true for an SVD
    candidate that survived the rules; ``confidence == "high"`` marks a confirmed one."""
    ev_dates: dict = {}
    for e in events:
        d = parse_date(e.date) if e.date else None
        if d is not None:
            ev_dates.setdefault(e.type, []).append(d)
    rec = adjudicate_series(points, ref, thrombosis_windows, ev_dates, death_date, reintervention_date)
    return EndpointDecision(met=rec.candidate_date is not None and rec.mechanism == "svd", date=rec.candidate_date,
                            stage=rec.stage, mechanism=rec.mechanism, phenotype=rec.phenotype,
                            establishing_index=rec.establishing_index, confidence=rec.confidence, reason=rec.reason,
                            reference_source=rec.reference_source, uncertain_dates=rec.uncertain_dates,
                            thrombosis_attributed_dates=rec.thrombosis_attributed_dates)


def thrombosis_windows_from_exposures(episodes, echo_points: list[EchoPoint],
                                      ref: EchoPoint | None) -> list[tuple[date, date]]:
    """Windows for exposure episodes with indication suspected_valve_thrombosis that were
    followed by an echo back at stage 0/1 (medically treated and resolved)."""
    windows: list[tuple[date, date]] = []
    if ref is None:
        return windows
    for ep in episodes:
        if ep.indication != "suspected_valve_thrombosis":
            continue
        start = parse_date(ep.start)
        stop = parse_date(ep.stop) if ep.stop else None
        if start is None:
            continue
        after = [p for p in echo_points if p.date >= start]
        resolved = any(stage_point(p, ref).stage in ("0", "1") for p in after)
        if resolved:
            win_start = start - timedelta(days=MECHANISM_WINDOW_DAYS)
            win_end = stop or (start + timedelta(days=365))
            windows.append((win_start, win_end))
    return windows
