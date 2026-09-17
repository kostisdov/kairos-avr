"""Independent, unfitted clinical comparators.

Comparators are deliberately outside the prediction and adjudication packages.  Their
outputs are audit findings and must never be used as model features or labels.
"""

from kairos.comparators.varc3_hvd import (
    COMPARATOR_ID,
    ComparatorContext,
    ComparatorResult,
    evaluate_varc3_comparator,
)

__all__ = [
    "COMPARATOR_ID",
    "ComparatorContext",
    "ComparatorResult",
    "evaluate_varc3_comparator",
]
