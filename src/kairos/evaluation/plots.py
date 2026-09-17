"""Figures for the evaluation job. Every title carries the words "synthetic scenario"."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TITLE_TAG = "synthetic scenario"


def primary_rows(results: pd.DataFrame, state: str = "svd", family: str = "cox") -> pd.DataFrame:
    """SVD rows pooled over landmark rows (the primary presentation), one family."""
    df = results
    if "family" in df and family in set(df["family"]):
        df = df[df["family"] == family]
    if "state" in df:
        df = df[df["state"] == state]
    if "weighting" in df:
        df = df[df["weighting"] == "pooled landmark rows"]
    return df
FOOTNOTE = "Software demonstration on synthetic data. Illustrative, unvalidated; not clinical evidence."


def _finish(fig, path: Path) -> Path:
    fig.text(0.01, 0.005, FOOTNOTE, fontsize=7, color="0.35")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_calibration(curves: dict, scenario: str, horizon: int, path: Path, steps: list | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, df in curves.items():
        if steps and name not in steps:
            continue
        if df is None or df.empty:
            continue
        ax.plot(df["mean_predicted"], df["observed"], marker="o", label=name)
    lim = 0.05
    for df in curves.values():
        if df is not None and not df.empty:
            lim = max(lim, float(df[["mean_predicted", "observed"]].max().max()) * 1.1)
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="ideal")
    ax.set_xlabel(f"Predicted {horizon}-year SVD before death")
    ax.set_ylabel("Observed (Aalen-Johansen)")
    ax.set_title(f"Calibration, {TITLE_TAG}: {scenario}")
    ax.legend(fontsize=8)
    return _finish(fig, path)


def plot_brier_ladder(results: pd.DataFrame, scenario: str, path: Path) -> Path:
    results = primary_rows(results)
    df = results[results["scenario"] == scenario]
    steps = list(dict.fromkeys(df["step"]))
    horizons = sorted(df["horizon_years"].unique())
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / max(1, len(horizons))
    x = np.arange(len(steps))
    for i, h in enumerate(horizons):
        sub = df[df["horizon_years"] == h].set_index("step").reindex(steps)
        err = np.vstack([sub["brier"] - sub["brier_lo"], sub["brier_hi"] - sub["brier"]])
        ax.bar(x + i * width, sub["brier"], width, yerr=np.nan_to_num(err), capsize=2, label=f"{h} y")
    ax.set_xticks(x + width * (len(horizons) - 1) / 2)
    ax.set_xticklabels(steps, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("IPCW Brier score (lower is better)")
    ax.set_title(f"Brier score by horizon and ladder step, {TITLE_TAG}: {scenario}")
    ax.legend()
    return _finish(fig, path)


def plot_incremental_value(all_results: pd.DataFrame, path: Path, horizon: int = 5) -> Path:
    all_results = primary_rows(all_results)
    df = all_results[all_results["horizon_years"] == horizon]
    scenarios = list(dict.fromkeys(df["scenario"]))
    steps = [s for s in dict.fromkeys(df["step"]) if s.startswith("core_plus")]
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / max(1, len(steps))
    x = np.arange(len(scenarios))
    for i, st in enumerate(steps):
        vals = []
        for sc in scenarios:
            core = df[(df["scenario"] == sc) & (df["step"] == "core")]["brier"]
            this = df[(df["scenario"] == sc) & (df["step"] == st)]["brier"]
            vals.append(float(core.iloc[0] - this.iloc[0]) if len(core) and len(this) else np.nan)
        ax.bar(x + i * width, vals, width, label=st)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x + width * (len(steps) - 1) / 2)
    ax.set_xticklabels(scenarios, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel(f"Brier improvement over core at {horizon} y (positive = better)")
    ax.set_title(f"Incremental value of modules by {TITLE_TAG}")
    ax.legend(fontsize=7)
    return _finish(fig, path)


def plot_phenotype(pheno: pd.DataFrame, scenario: str, path: Path, horizon: int = 5) -> Path:
    df = pheno[(pheno["scenario"] == scenario) & (pheno["horizon_years"] == horizon)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if df.empty:
        ax.text(0.5, 0.5, "no phenotype-specific rows", ha="center")
    else:
        piv = df.pivot_table(index="step", columns="phenotype", values="brier")
        piv.plot(kind="bar", ax=ax)
        ax.set_ylabel(f"IPCW Brier at {horizon} y")
    ax.set_title(f"Phenotype-specific Brier, {TITLE_TAG}: {scenario}")
    return _finish(fig, path)


def plot_trajectory(echoes: pd.DataFrame, predictions: pd.DataFrame, path: Path, title: str = "one synthetic patient",
                    intervals: pd.DataFrame | None = None) -> Path:
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    a1.plot(pd.to_datetime(echoes["date"]), echoes["mean_gradient"], marker="o", label="mean gradient (mmHg)")
    a1.set_ylabel("mean gradient, mmHg")
    if "eoa" in echoes:
        ax2 = a1.twinx()
        ax2.plot(pd.to_datetime(echoes["date"]), echoes["eoa"], marker="s", color="tab:orange", label="EOA (cm2)")
        ax2.set_ylabel("EOA, cm2")
    for d in pd.to_datetime(predictions["prediction_time"]):
        a1.axvline(d, color="0.8", lw=0.8)
    a1.set_title(f"Echo trajectory with landmarks, {TITLE_TAG}: {title}", fontsize=10)
    t = pd.to_datetime(predictions["prediction_time"])
    for col, lab in (("p_svd_5y", "SVD before death, 5 y"), ("p_death_5y", "death before SVD, 5 y"),
                     ("p_alive_5y", "alive with intact valve, 5 y"), ("p_svd_12m", "SVD before death, 12 m")):
        if col in predictions:
            a2.plot(t, predictions[col], marker="o", label=lab)
            if intervals is not None and f"{col}_lo" in intervals:
                a2.fill_between(t, intervals[f"{col}_lo"], intervals[f"{col}_hi"], alpha=0.15)
    a2.set_ylim(0, 1)
    a2.set_ylabel("probability")
    a2.set_title(f"Updated probabilities at each landmark, {TITLE_TAG} (illustrative, unvalidated)", fontsize=10)
    a2.legend(fontsize=8)
    return _finish(fig, path)
