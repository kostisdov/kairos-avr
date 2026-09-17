"""The one intraprosthetic regurgitation grade mapping (generation, staging, landmark features and
the predictor all go through here).

Ordinal scale used by the VARC-3 module: 0 none/trace, 1 mild, 2 moderate, 3 severe. Labels follow
``extraction.schema.ARGrade``; "trivial" is accepted as a synonym of "trace".
"""
from __future__ import annotations

import math

REGURG_ORDINAL = {"none": 0, "trace": 0, "trivial": 0, "mild": 1, "moderate": 2, "severe": 3}
REGURG_LABELS = ("none", "mild", "moderate", "severe")   # canonical label per ordinal


def regurg_ordinal(label) -> int | None:
    """Ordinal for a label; None for a missing or unknown label (never a guessed grade)."""
    if label is None or (isinstance(label, float) and math.isnan(label)):
        return None
    return REGURG_ORDINAL.get(str(label).strip().lower())


def regurg_label(ordinal) -> str | None:
    if ordinal is None or (isinstance(ordinal, float) and math.isnan(ordinal)):
        return None
    o = int(ordinal)
    if not 0 <= o <= 3:
        raise ValueError(f"regurgitation ordinal {ordinal!r} outside 0-3")
    return REGURG_LABELS[o]
