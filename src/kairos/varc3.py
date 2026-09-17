# -*- coding: utf-8 -*-
"""
VARC-3 haemodynamic valve deterioration (HVD) staging.

Source: VARC-3 WRITING COMMITTEE; Genereux P, Piazza N, Alu MC, Nazif T, Hahn RT, et al.
Valve Academic Research Consortium 3: updated endpoint definitions for aortic valve clinical
research. Eur Heart J. 2021;42:1825-1857.

Quoted criteria this module implements (these are the two numeric thresholds the task brief
itself specifies, and which independently match the phrasing captured this session from the
NOTION trial paper's own VARC-3 methods section -- see
data/raw/papers/notion_10yr_PMC10984572_fulltext.txt -- corroborating that this is the
correct standard definition, not a paraphrase specific to one trial):

  Stage 2 (moderate HVD): "mean gradient rise of at least 10 mmHg to at least 20 mmHg
  with EOA fall of at least 0.3 cm2 or 25% or DVI fall of at least 0.1 or 20%, or new at
  least moderate intraprosthetic regurgitation."

  Stage 3 (severe HVD): "rise of at least 20 mmHg to at least 30 mmHg with EOA fall of at
  least 0.6 cm2 or 50% or DVI fall of at least 0.2 or 40%, or severe regurgitation."

UNCERTAIN / NOT FULLY SPECIFIED BY THE SOURCE (flagged here rather than guessed, per the
task's explicit instruction): the task brief and our captured secondary sources give exact
combined gradient+EOA/DVI thresholds for stage 2 and stage 3, but NOT a numeric threshold
distinguishing stage 0 (no deterioration) from stage 1 (mild deterioration). The full VARC-3
document defines stage 1 partly via echocardiographic/morphological criteria (leaflet
thickening, restricted motion, new mild regurgitation) that are outside what a
gradient/EOA/DVI/regurgitation-grade-only function can assess. This implementation's stage
0/1 boundary is therefore a REASONABLE, EXPLICITLY-DOCUMENTED INTERPOLATION: stage 1 is
returned when there is a directionally-worsening but sub-stage-2 change (i.e. any gradient
rise, EOA fall, DVI fall, or new-but-sub-moderate regurgitation, however small), and stage 0
when nothing has worsened at all. A user who has access to the full morphological read
(leaflet thickening/calcification/restricted motion on imaging) should override stage 0/1
using that information directly -- this function only sees numbers, not images.

Reference-value fallback: when `reference_echo` is None (no true baseline echo for this
patient/valve), the caller may pass `reference_fallback` (a dict shaped like an echo dict,
typically sourced from data/reference/prosthetic_valve_reference_values.csv for the matching
canonical_model/size). If used, the result's `reference_source` is set to
"ASE 2024 table" (rather than "patient baseline echo") and `confidence` is downgraded,
because population-normal values are a much weaker stand-in for this specific patient's true
post-implant baseline than an actual paired echo would be.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Echo:
    mean_gradient_mmHg: Optional[float] = None
    eoa_cm2: Optional[float] = None
    dvi: Optional[float] = None
    # regurgitation grade as an ordinal: 0=none/trivial, 1=mild, 2=moderate, 3=severe
    regurg_grade: Optional[int] = None


@dataclass
class VARC3Result:
    stage: str  # "0", "1", "2", "3", or "uncertain"
    reason: str
    reference_source: str = "patient baseline echo"
    confidence: str = "high"
    details: dict = field(default_factory=dict)


def stage_hvd(follow_up: Echo, reference_echo: Optional[Echo] = None,
              reference_fallback: Optional[Echo] = None) -> VARC3Result:
    """
    follow_up: the echo being evaluated for deterioration.
    reference_echo: the patient's own true baseline/reference echo (e.g. the earliest
        post-implant or 30-day/3-month echo), if available.
    reference_fallback: an ASE-2024-table (or similar published normal-range) stand-in to
        use ONLY when reference_echo is None. If both are None, the result is "uncertain"
        (per the task's explicit rule: "Return 'uncertain' ... when ... the reference echo
        is absent").

    Returns "uncertain" (per the task's explicit rule) when EOA and DVI are BOTH missing in
    either echo, or when no reference (real or fallback) is available at all.
    """
    reference_source = "patient baseline echo"
    confidence = "high"
    ref = reference_echo
    if ref is None:
        if reference_fallback is None:
            return VARC3Result(stage="uncertain",
                                reason="No reference echo available (neither a patient "
                                       "baseline nor a reference_fallback was supplied).")
        ref = reference_fallback
        reference_source = "ASE 2024 table"
        confidence = "low"

    if ref.mean_gradient_mmHg is None or follow_up.mean_gradient_mmHg is None:
        return VARC3Result(stage="uncertain",
                            reason="Mean gradient missing on the reference and/or "
                                   "follow-up echo; cannot compute a gradient rise.",
                            reference_source=reference_source, confidence=confidence)

    if (ref.eoa_cm2 is None and ref.dvi is None) or \
       (follow_up.eoa_cm2 is None and follow_up.dvi is None):
        return VARC3Result(stage="uncertain",
                            reason="EOA and DVI are both missing on the reference and/or "
                                   "follow-up echo (need at least one of the two to assess "
                                   "the EOA-or-DVI-fall arm of the VARC-3 criteria).",
                            reference_source=reference_source, confidence=confidence)

    grad_rise = follow_up.mean_gradient_mmHg - ref.mean_gradient_mmHg
    grad_final = follow_up.mean_gradient_mmHg

    eoa_fall_abs = eoa_fall_pct = None
    if ref.eoa_cm2 is not None and follow_up.eoa_cm2 is not None and ref.eoa_cm2 > 0:
        eoa_fall_abs = ref.eoa_cm2 - follow_up.eoa_cm2
        eoa_fall_pct = 100.0 * eoa_fall_abs / ref.eoa_cm2

    dvi_fall_abs = dvi_fall_pct = None
    if ref.dvi is not None and follow_up.dvi is not None and ref.dvi > 0:
        dvi_fall_abs = ref.dvi - follow_up.dvi
        dvi_fall_pct = 100.0 * dvi_fall_abs / ref.dvi

    new_regurg_moderate_plus = (follow_up.regurg_grade is not None and follow_up.regurg_grade >= 2
                                 and (ref.regurg_grade is None or ref.regurg_grade < 2))
    new_regurg_severe = (follow_up.regurg_grade is not None and follow_up.regurg_grade >= 3
                          and (ref.regurg_grade is None or ref.regurg_grade < 3))

    details = dict(grad_rise=grad_rise, grad_final=grad_final, eoa_fall_abs=eoa_fall_abs,
                    eoa_fall_pct=eoa_fall_pct, dvi_fall_abs=dvi_fall_abs,
                    dvi_fall_pct=dvi_fall_pct,
                    new_regurg_moderate_plus=new_regurg_moderate_plus,
                    new_regurg_severe=new_regurg_severe)

    def eoa_or_dvi_meets(eoa_abs_thresh, eoa_pct_thresh, dvi_abs_thresh, dvi_pct_thresh):
        eoa_hit = (eoa_fall_abs is not None and eoa_fall_abs >= eoa_abs_thresh) or \
                  (eoa_fall_pct is not None and eoa_fall_pct >= eoa_pct_thresh)
        dvi_hit = (dvi_fall_abs is not None and dvi_fall_abs >= dvi_abs_thresh) or \
                  (dvi_fall_pct is not None and dvi_fall_pct >= dvi_pct_thresh)
        return eoa_hit or dvi_hit

    # Stage 3: rise >=20 to >=30 mmHg AND (EOA fall >=0.6 cm2 or 50%, OR DVI fall >=0.2 or
    # 40%) -- OR severe regurgitation, regardless of gradient/EOA/DVI.
    if new_regurg_severe:
        return VARC3Result(stage="3", reason="New severe intraprosthetic regurgitation.",
                            reference_source=reference_source, confidence=confidence,
                            details=details)
    if grad_rise >= 20 and grad_final >= 30 and eoa_or_dvi_meets(0.6, 50, 0.2, 40):
        return VARC3Result(
            stage="3",
            reason=(f"Mean gradient rose {grad_rise:.1f} mmHg (>=20) to {grad_final:.1f} "
                    f"mmHg (>=30) with EOA fall {eoa_fall_abs}/{eoa_fall_pct}% or DVI fall "
                    f"{dvi_fall_abs}/{dvi_fall_pct}% meeting the stage-3 threshold."),
            reference_source=reference_source, confidence=confidence, details=details)

    # Stage 2: rise >=10 to >=20 mmHg AND (EOA fall >=0.3 cm2 or 25%, OR DVI fall >=0.1 or
    # 20%) -- OR new >=moderate intraprosthetic regurgitation.
    if new_regurg_moderate_plus:
        return VARC3Result(stage="2",
                            reason="New at least moderate intraprosthetic regurgitation.",
                            reference_source=reference_source, confidence=confidence,
                            details=details)
    if grad_rise >= 10 and grad_final >= 20 and eoa_or_dvi_meets(0.3, 25, 0.1, 20):
        return VARC3Result(
            stage="2",
            reason=(f"Mean gradient rose {grad_rise:.1f} mmHg (>=10) to {grad_final:.1f} "
                    f"mmHg (>=20) with EOA fall {eoa_fall_abs}/{eoa_fall_pct}% or DVI fall "
                    f"{dvi_fall_abs}/{dvi_fall_pct}% meeting the stage-2 threshold."),
            reference_source=reference_source, confidence=confidence, details=details)

    # Stage 0 vs 1: see the module docstring's UNCERTAIN/NOT FULLY SPECIFIED note --
    # this boundary is our own reasonable interpolation, not a numeric VARC-3 threshold.
    any_worsening = (grad_rise > 0) or (eoa_fall_abs is not None and eoa_fall_abs > 0) or \
                     (dvi_fall_abs is not None and dvi_fall_abs > 0) or \
                     (follow_up.regurg_grade is not None and ref.regurg_grade is not None and
                      follow_up.regurg_grade > ref.regurg_grade)
    if any_worsening:
        return VARC3Result(
            stage="1",
            reason="Some worsening in gradient/EOA/DVI/regurgitation detected, but not "
                   "meeting the stage-2 combined threshold. Stage 0/1 boundary is this "
                   "module's own interpolation -- see module docstring; the full VARC-3 "
                   "stage-1 definition also includes morphological (imaging) criteria not "
                   "assessed here.",
            reference_source=reference_source, confidence=confidence, details=details)
    return VARC3Result(
        stage="0", reason="No worsening detected in gradient, EOA, DVI, or regurgitation "
                           "grade relative to the reference echo.",
        reference_source=reference_source, confidence=confidence, details=details)
