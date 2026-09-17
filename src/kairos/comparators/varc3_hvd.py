"""Standalone VARC-3-derived current-echo HVD comparator.

This module intentionally does not import :mod:`kairos.varc3` or call ``stage_hvd``.  It
implements the declared comparator independently while accepting the ``EchoPoint`` selected
by the existing 30--180 day KAIROS reference policy (or equivalent mappings/objects).

``negative`` means that the moderate/severe change criteria were not demonstrated; it does
not mean that the valve is normal.  Missing evidence is retained through three-valued logic.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime
from typing import Any, Literal, Mapping

COMPARATOR_ID = "varc3_hvd_comparator_v1"
REFERENCE_POLICY = "kairos_30_180_days"
TriState = bool | None
Status = Literal["positive", "negative", "indeterminate"]

AR_ORDINAL = {"none": 0, "trace": 0, "mild": 1, "moderate": 2, "severe": 3}


@dataclass(frozen=True)
class ComparatorContext:
    """Selection and provenance information supplied by the read-only adapter."""

    index_valve_id: str | None = None
    prediction_date: date | str | None = None
    reference_study_id: str | None = None
    current_study_id: str | None = None
    reference_study_date: date | str | None = None
    current_study_date: date | str | None = None
    source_provenance: str = "unknown"
    intraprosthetic_ar_confirmed: bool | None = None
    fresh: bool | None = True
    freshness_reason: str | None = None
    reference_policy: str = REFERENCE_POLICY


@dataclass
class ComparatorResult:
    comparator_id: str = COMPARATOR_ID
    status: Status = "indeterminate"
    triggered: bool | None = None
    highest_demonstrated_stage: int | None = None
    severity_complete: bool = False
    stenotic_ge2: TriState = None
    stenotic_ge3: TriState = None
    regurgitant_ge2: TriState = None
    regurgitant_ge3: TriState = None
    gradient_only_ge2: TriState = None
    reference_study_id: str | None = None
    current_study_id: str | None = None
    reference_study_date: str | None = None
    current_study_date: str | None = None
    reference_values: dict[str, Any] = field(default_factory=dict)
    current_values: dict[str, Any] = field(default_factory=dict)
    deltas: dict[str, float | None] = field(default_factory=dict)
    branches: dict[str, TriState] = field(default_factory=dict)
    missing_inputs: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    reference_policy: str = REFERENCE_POLICY
    source_provenance: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _value(obj: Any, *names: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
        return None
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _context(value: ComparatorContext | Mapping[str, Any] | Any | None) -> ComparatorContext:
    if value is None:
        return ComparatorContext()
    if isinstance(value, ComparatorContext):
        return value
    raw = dict(value) if isinstance(value, Mapping) else (
        asdict(value) if is_dataclass(value) else vars(value)
    )
    names = ComparatorContext.__dataclass_fields__
    return ComparatorContext(**{k: raw[k] for k in names if k in raw})


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    return text or None


def _date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10]) if value else None
    except (TypeError, ValueError):
        return None


def _number(value: Any, *, positive: bool = False) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v < 0 or (positive and v <= 0):
        return None
    return v


def _ar(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        return AR_ORDINAL.get(value.strip().lower())
    try:
        grade = int(value)
    except (TypeError, ValueError):
        return None
    return grade if 0 <= grade <= 3 else None


def tri_or(*values: TriState) -> TriState:
    """Kleene OR: true wins; false only when every branch is false."""
    if any(v is True for v in values):
        return True
    if values and all(v is False for v in values):
        return False
    return None


def tri_and(*values: TriState) -> TriState:
    """Kleene AND: false wins; true only when every branch is true."""
    if any(v is False for v in values):
        return False
    if values and all(v is True for v in values):
        return True
    return None


def _decrease(reference: float | None, current: float | None, absolute: float,
              percent: float) -> tuple[TriState, float | None, float | None]:
    if reference is None or current is None:
        return None, None, None
    delta = reference - current
    pct = 100.0 * delta / reference
    return bool(delta >= absolute or pct >= percent), delta, pct


def evaluate_varc3_comparator(reference: Any, current: Any,
                              context: ComparatorContext | Mapping[str, Any] | Any | None = None) -> ComparatorResult:
    """Evaluate moderate/severe haemodynamic change using paired current echo data.

    Inputs may be mappings, extraction ``EchoObservation`` objects, or adjudication
    ``EchoPoint`` objects.  The function never selects another reference and never imputes.
    Integrated adapters should pass selection identity, time and freshness in ``context``.
    """
    ctx = _context(context)
    result = ComparatorResult(
        reference_study_id=ctx.reference_study_id,
        current_study_id=ctx.current_study_id,
        reference_study_date=_iso(ctx.reference_study_date or _value(reference, "date")),
        current_study_date=_iso(ctx.current_study_date or _value(current, "date")),
        reference_policy=ctx.reference_policy,
        source_provenance=ctx.source_provenance,
    )
    if reference is None:
        result.missing_inputs.append("reference_study")
        result.reason_codes.append("missing_reference")
        return result
    if current is None:
        result.missing_inputs.append("current_study")
        result.reason_codes.append("missing_current")
        return result

    ref = {
        "mean_gradient_mmhg": _number(_value(reference, "mean_gradient_mmhg", "mean_gradient", "mean_gradient_mmHg")),
        "eoa_cm2": _number(_value(reference, "eoa_cm2", "eoa"), positive=True),
        "dvi": _number(_value(reference, "dvi"), positive=True),
        "intraprosthetic_ar_grade": _ar(_value(reference, "intraprosthetic_ar_grade", "ar_grade", "ar_ordinal", "regurg_grade")),
        "lvef_pct": _number(_value(reference, "lvef_pct", "lvef")),
        "svi_ml_m2": _number(_value(reference, "svi_ml_m2", "svi")),
    }
    cur = {
        "mean_gradient_mmhg": _number(_value(current, "mean_gradient_mmhg", "mean_gradient", "mean_gradient_mmHg")),
        "eoa_cm2": _number(_value(current, "eoa_cm2", "eoa"), positive=True),
        "dvi": _number(_value(current, "dvi"), positive=True),
        "intraprosthetic_ar_grade": _ar(_value(current, "intraprosthetic_ar_grade", "ar_grade", "ar_ordinal", "regurg_grade")),
        "lvef_pct": _number(_value(current, "lvef_pct", "lvef")),
        "svi_ml_m2": _number(_value(current, "svi_ml_m2", "svi")),
    }
    result.reference_values, result.current_values = ref, cur

    # Selection/time guards are intentionally separate from numeric validity.  An invalid
    # integrated pair can never become a reassuring negative.
    rd, cd, pd = _date(result.reference_study_date), _date(result.current_study_date), _date(ctx.prediction_date)
    invalid_selection = False
    if rd is not None and cd is not None and cd < rd:
        result.reason_codes.append("current_before_reference")
        invalid_selection = True
    if cd is not None and pd is not None and cd > pd:
        result.reason_codes.append("future_current_study")
        invalid_selection = True
    if rd is not None and pd is not None and rd > pd:
        result.reason_codes.append("future_reference_study")
        invalid_selection = True
    if ctx.fresh is False:
        result.reason_codes.append("stale_current_study")
        if ctx.freshness_reason:
            result.reason_codes.append("freshness_reason_available")
        invalid_selection = True

    rg, cg = ref["mean_gradient_mmhg"], cur["mean_gradient_mmhg"]
    grad_rise = (cg - rg) if rg is not None and cg is not None else None
    grad2 = None if grad_rise is None else bool(grad_rise >= 10.0 and cg >= 20.0)
    grad3 = None if grad_rise is None else bool(grad_rise >= 20.0 and cg >= 30.0)
    eoa2, eoa_drop, eoa_pct = _decrease(ref["eoa_cm2"], cur["eoa_cm2"], 0.3, 25.0)
    eoa3, _, _ = _decrease(ref["eoa_cm2"], cur["eoa_cm2"], 0.6, 50.0)
    dvi2, dvi_drop, dvi_pct = _decrease(ref["dvi"], cur["dvi"], 0.1, 20.0)
    dvi3, _, _ = _decrease(ref["dvi"], cur["dvi"], 0.2, 40.0)
    stenotic2 = tri_and(grad2, tri_or(eoa2, dvi2))
    stenotic3 = tri_and(grad3, tri_or(eoa3, dvi3))

    ar_ref, ar_cur = ref["intraprosthetic_ar_grade"], cur["intraprosthetic_ar_grade"]
    location_known = ctx.intraprosthetic_ar_confirmed is True
    if not location_known:
        ar2 = ar3 = None
        result.reason_codes.append("intraprosthetic_ar_provenance_unavailable")
    elif ar_ref is None:
        ar2 = ar3 = None
        result.reason_codes.append("missing_baseline_ar")
    elif ar_cur is None:
        ar2 = ar3 = None
        result.reason_codes.append("missing_current_ar")
    else:
        ar_change = ar_cur - ar_ref
        ar2 = bool(ar_cur >= 2 and ar_change >= 1)
        # Severe evidence requires severe current AR plus either new occurrence from none/trace
        # or a rise of at least two ordinal grades.
        ar3 = bool(ar_cur >= 3 and (ar_ref == 0 or ar_change >= 2))

    result.deltas = {
        "mean_gradient_rise_mmhg": grad_rise,
        "eoa_decrease_cm2": eoa_drop,
        "eoa_decrease_percent": eoa_pct,
        "dvi_decrease_absolute": dvi_drop,
        "dvi_decrease_percent": dvi_pct,
        "ar_grade_change": (ar_cur - ar_ref) if ar_ref is not None and ar_cur is not None else None,
    }
    result.branches = {
        "gradient_ge2": grad2, "gradient_ge3": grad3,
        "eoa_ge2": eoa2, "eoa_ge3": eoa3,
        "dvi_ge2": dvi2, "dvi_ge3": dvi3,
        "stenotic_ge2": stenotic2, "stenotic_ge3": stenotic3,
        "regurgitant_ge2": ar2, "regurgitant_ge3": ar3,
    }
    for name, value in (("reference_mean_gradient_mmhg", rg), ("current_mean_gradient_mmhg", cg),
                        ("reference_eoa_cm2", ref["eoa_cm2"]), ("current_eoa_cm2", cur["eoa_cm2"]),
                        ("reference_dvi", ref["dvi"]), ("current_dvi", cur["dvi"])):
        if value is None:
            result.missing_inputs.append(name)
    if ar_ref is None:
        result.missing_inputs.append("reference_intraprosthetic_ar_grade")
    if ar_cur is None:
        result.missing_inputs.append("current_intraprosthetic_ar_grade")

    result.stenotic_ge2, result.stenotic_ge3 = stenotic2, stenotic3
    result.regurgitant_ge2, result.regurgitant_ge3 = ar2, ar3
    result.gradient_only_ge2 = grad2
    moderate = tri_or(stenotic2, ar2)
    severe = tri_or(stenotic3, ar3)
    if invalid_selection:
        result.reason_codes.append("selection_not_evaluable")
        return result
    if severe is True:
        result.status, result.triggered, result.highest_demonstrated_stage = "positive", True, 3
    elif moderate is True:
        result.status, result.triggered, result.highest_demonstrated_stage = "positive", True, 2
    elif moderate is False:
        result.status, result.triggered = "negative", False
        result.reason_codes.append("criteria_not_met")
    else:
        result.reason_codes.append("insufficient_paired_evidence")
    result.severity_complete = severe is not None
    if result.status == "positive":
        result.reason_codes.append(f"stage_{result.highest_demonstrated_stage}_criteria_met")
    return result
