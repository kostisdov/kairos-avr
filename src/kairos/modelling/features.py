"""Feature pipeline fitted inside training data only (leakage rule).

Numeric features: median imputation learned on training rows (only for features with at least
one training value; an absent feature raises), a missing indicator where training rows had gaps, standardisation, and a cubic B-spline basis for the features listed
as spline features. Categorical features: one-hot encoding with rare levels (below
``min_frequency`` training rows) collapsed to ``other`` and unseen levels mapped to ``other``.
Binary features: 0/1 with missing mapped to 0 plus an indicator. In ``tree`` mode (gradient
boosting) numeric features keep their raw values and no spline basis is built; eligibility,
imputation, indicators and encodings are identical to the Cox mode.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer


@dataclass
class FeaturePipeline:
    features: list[str]
    categorical: list[str] = field(default_factory=list)
    binary: list[str] = field(default_factory=list)
    spline: list[str] = field(default_factory=list)
    n_knots: int = 4
    min_frequency: int = 15
    rare_levels: dict = field(default_factory=dict)
    medians_: dict = field(default_factory=dict)
    means_: dict = field(default_factory=dict)
    sds_: dict = field(default_factory=dict)
    indicators_: list = field(default_factory=list)
    levels_: dict = field(default_factory=dict)
    splines_: dict = field(default_factory=dict)
    output_columns_: list = field(default_factory=list)
    group_of_: dict = field(default_factory=dict)
    mode: str = "cox"          # "cox": standardise and spline-expand; "tree": raw numeric values, no splines
    fitted: bool = False

    @property
    def numeric(self) -> list[str]:
        return [f for f in self.features if f not in self.categorical and f not in self.binary]

    def fit(self, df: pd.DataFrame) -> FeaturePipeline:
        self.medians_, self.means_, self.sds_, self.indicators_ = {}, {}, {}, []
        self.levels_, self.splines_, self.group_of_ = {}, {}, {}
        cols: list[str] = []
        for f in self.numeric:
            x = pd.to_numeric(df[f], errors="coerce") if f in df else pd.Series(np.nan, index=df.index)
            if not x.notna().any():
                # never manufacture a constant from an absent marker; eligibility must exclude it first
                raise ValueError(f"feature {f!r} has no non-missing training value; exclude it before fitting")
            med = float(np.nanmedian(x))
            self.medians_[f] = med
            xi = x.fillna(med).astype(float)
            if self.mode == "tree":
                self.means_[f], self.sds_[f] = 0.0, 1.0   # trees are invariant to monotone rescaling
            else:
                self.means_[f] = float(xi.mean())
                sd = float(xi.std(ddof=0))
                self.sds_[f] = sd if sd > 1e-9 else 1.0
            if x.isna().any():
                self.indicators_.append(f)
            if self.mode == "cox" and f in self.spline and xi.nunique() > self.n_knots + 2:
                st = SplineTransformer(n_knots=self.n_knots, degree=3, include_bias=False, knots="quantile")
                st.fit(xi.to_numpy().reshape(-1, 1))
                self.splines_[f] = st
                k = st.transform(np.zeros((1, 1))).shape[1]
                for j in range(k):
                    cols.append(f"{f}__s{j}")
                    self.group_of_[f"{f}__s{j}"] = f
            else:
                cols.append(f)
                self.group_of_[f] = f
            if f in self.indicators_:
                cols.append(f"{f}__missing")
                self.group_of_[f"{f}__missing"] = f
        for f in self.binary:
            x = df[f] if f in df else pd.Series(np.nan, index=df.index)
            cols.append(f)
            self.group_of_[f] = f
            if x.isna().any():
                self.indicators_.append(f)
                cols.append(f"{f}__missing")
                self.group_of_[f"{f}__missing"] = f
        for f in self.categorical:
            x = df[f].astype("object") if f in df else pd.Series(np.nan, index=df.index)
            counts = x.fillna("missing").astype(str).value_counts()
            keep = [lvl for lvl, c in counts.items() if c >= self.min_frequency and lvl not in self.rare_levels.get(f, ())]
            if not keep:
                keep = [counts.index[0]] if len(counts) else []
            levels = sorted(keep)
            if len(levels) > 1 or counts.drop(index=levels, errors="ignore").sum() > 0:
                levels = levels + ["other"]
            self.levels_[f] = levels
            for lvl in levels:
                col = f"{f}__{lvl}"
                cols.append(col)
                self.group_of_[col] = f
        self.output_columns_ = cols
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("pipeline not fitted")
        out = {}
        n = len(df)
        for f in self.numeric:
            x = pd.to_numeric(df[f], errors="coerce") if f in df else pd.Series(np.nan, index=df.index)
            miss = x.isna().to_numpy()
            xi = x.fillna(self.medians_[f]).astype(float).to_numpy()
            if f in self.splines_:
                lo, hi = np.nanmin(self.splines_[f].bsplines_[0].t), np.nanmax(self.splines_[f].bsplines_[0].t)
                basis = self.splines_[f].transform(np.clip(xi, lo, hi).reshape(-1, 1))
                for j in range(basis.shape[1]):
                    out[f"{f}__s{j}"] = basis[:, j]
            else:
                out[f] = (xi - self.means_[f]) / self.sds_[f]
            if f in self.indicators_:
                out[f"{f}__missing"] = miss.astype(float)
        for f in self.binary:
            x = df[f] if f in df else pd.Series(np.nan, index=df.index)
            miss = x.isna().to_numpy()
            vals = x.map(lambda v: 1.0 if v in (True, 1, "True", "true", 1.0) else 0.0).to_numpy(dtype=float)
            vals[miss] = 0.0
            out[f] = vals
            if f in self.indicators_:
                out[f"{f}__missing"] = miss.astype(float)
        for f in self.categorical:
            x = (df[f].astype("object") if f in df else pd.Series(np.nan, index=df.index)).fillna("missing").astype(str)
            levels = self.levels_[f]
            mapped = x.where(x.isin(levels), "other" if "other" in levels else levels[0])
            for lvl in levels:
                out[f"{f}__{lvl}"] = (mapped == lvl).to_numpy(dtype=float)
        res = pd.DataFrame(out, index=df.index)
        for c in self.output_columns_:
            if c not in res:
                res[c] = np.zeros(n)
        return res[self.output_columns_]

    def group(self, column: str) -> str:
        return self.group_of_.get(column, column)
