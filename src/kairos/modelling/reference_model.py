"""The reference model of the ladder: valve age and valve type only.

It is the same penalised cause-specific machinery restricted to the ``reference`` block
(valve age in years, route, design class), so every comparison in the ladder is against a
model with the same outcome definition, horizons and evaluation.
"""
from __future__ import annotations

from kairos.modelling.modules import block_features


def reference_features(cfg: dict) -> list[str]:
    return block_features(cfg, ["reference"])
