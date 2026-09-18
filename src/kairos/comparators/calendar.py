"""Calendar comparator: the risk a guideline calendar implicitly assigns.

Guideline surveillance schedules by valve age alone: the 2020 ACC/AHA calendar images a surgical
bioprosthesis at 5 and 10 years, then yearly, so a valve younger than five years is treated as low
risk and an older one as higher risk. This comparator turns that into a probability that can be
scored with the same metrics as the model: the Aalen-Johansen cumulative incidence of SVD by the
horizon within each cell of route x (valve age below or at least ``cut_years``), estimated on
training landmark rows only.

Unlike the VARC-3 threshold rule it never abstains, so a decision curve against current practice
can always be computed. It carries no patient information beyond route and valve age, which is the
point: it is the bar the model has to clear to justify leaving the calendar. The design follows the
calendar comparator of team dyanooumenoi (Dyania Health Hackathon 2026).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from kairos.evaluation.metrics import aalen_johansen

COMPARATOR_ID = "calendar_valve_age_v1"


@dataclass
class CalendarComparator:
    cut_years: float = 5.0
    by_route: bool = True
    horizon_years: float = 1.0
    min_rows: int = 30                     # a cell with fewer training rows falls back to its route (or all rows)
    table_: dict = field(default_factory=dict)
    fallback_: dict = field(default_factory=dict)
    overall_: float | None = None

    def _cells(self, lm: pd.DataFrame) -> pd.Series:
        older = np.where(lm["valve_age_years"].to_numpy(float) >= self.cut_years, f">={self.cut_years:g}y",
                         f"<{self.cut_years:g}y")
        route = lm["route"].astype(str).to_numpy() if self.by_route else np.full(len(lm), "all")
        return pd.Series([f"{r}|{a}" for r, a in zip(route, older)], index=lm.index)

    def _incidence(self, rows: pd.DataFrame) -> float:
        return float(aalen_johansen(rows["time"].to_numpy(float), rows["event"].to_numpy(), "svd", self.horizon_years))

    def fit(self, lm: pd.DataFrame) -> CalendarComparator:
        self.overall_ = self._incidence(lm)
        routes = lm["route"].astype(str) if self.by_route else pd.Series("all", index=lm.index)
        self.fallback_ = {r: (self._incidence(g) if len(g) >= self.min_rows else self.overall_)
                          for r, g in lm.groupby(routes)}
        cells = self._cells(lm)
        self.table_ = {}
        for key, g in lm.groupby(cells):
            route = key.split("|")[0]
            self.table_[key] = {"rows": int(len(g)), "patients": int(g["patient_id"].nunique()),
                                "risk": self._incidence(g) if len(g) >= self.min_rows else self.fallback_.get(route, self.overall_),
                                "fallback": bool(len(g) < self.min_rows)}
        return self

    def predict(self, lm: pd.DataFrame) -> np.ndarray:
        if self.overall_ is None:
            raise RuntimeError("fit the calendar comparator first")
        cells = self._cells(lm)
        routes = lm["route"].astype(str) if self.by_route else pd.Series("all", index=lm.index)
        return np.array([self.table_[c]["risk"] if c in self.table_ else self.fallback_.get(r, self.overall_)
                         for c, r in zip(cells, routes)], dtype=float)

    def card(self) -> dict:
        return {"comparator_id": COMPARATOR_ID, "cut_years": self.cut_years, "by_route": self.by_route,
                "horizon_years": self.horizon_years, "cells": self.table_, "overall": self.overall_}
