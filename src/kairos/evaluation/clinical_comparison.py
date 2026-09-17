"""Paired, read-only evaluation of KAIROS predictions and the fixed echo comparator.

The functions in this module operate on supplied held-out predictions and landmark outcomes.
They never build a cohort, fit a model, select a threshold, or write outside the caller's
comparison directory.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from kairos.comparators.varc3_hvd import COMPARATOR_ID
from kairos.evaluation.metrics import KMCensoring, bootstrap_by_patient, ipcw_weights, outcome

DEFAULT_THRESHOLD = 0.05
DEFAULT_TAU_DAYS = 365.25


class ComparisonInputError(ValueError):
    """Persisted inputs are duplicated, mismatched, or lack required held-out data."""


@dataclass(frozen=True)
class ComparisonInputs:
    landmarks: str
    predictions: str
    comparator_rows: str


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    if p.suffix.lower() in (".csv", ".tsv"):
        return pd.read_csv(p, sep="\t" if p.suffix.lower() == ".tsv" else ",")
    if p.suffix.lower() in (".json", ".jsonl"):
        return pd.read_json(p, lines=p.suffix.lower() == ".jsonl")
    raise ComparisonInputError(f"unsupported table format: {p.suffix}")


def _require_unique(frame: pd.DataFrame, keys: Sequence[str], name: str) -> None:
    missing = [k for k in keys if k not in frame]
    if missing:
        raise ComparisonInputError(f"{name} is missing key columns: {missing}")
    duplicated = frame.duplicated(list(keys), keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, list(keys)].head(3).to_dict("records")
        raise ComparisonInputError(f"{name} has duplicate comparison keys: {examples}")


def pair_inputs(landmarks: pd.DataFrame, predictions: pd.DataFrame, comparator_rows: pd.DataFrame,
                keys: Sequence[str] = ("patient_id", "index_valve_id", "landmark_date"),
                *, outcome_columns: Sequence[str] = ("time", "event", "fold")) -> pd.DataFrame:
    """Key-align supplied artifacts and reject conflicting outcome or fold values.

    Comparator and prediction artifacts may contain copies of outcome fields.  Copies are
    checked against the landmark source before being dropped; row order is never trusted.
    """
    frames = {"landmarks": landmarks.copy(), "predictions": predictions.copy(),
              "comparator rows": comparator_rows.copy()}
    for name, frame in frames.items():
        _require_unique(frame, keys, name)
    paired = frames["landmarks"].merge(frames["predictions"], on=list(keys), how="inner",
                                        suffixes=("", "__prediction"), validate="one_to_one")
    paired = paired.merge(frames["comparator rows"], on=list(keys), how="inner",
                          suffixes=("", "__comparator"), validate="one_to_one")
    for col in outcome_columns:
        base = col if col in paired else None
        for suffix in ("__prediction", "__comparator"):
            other = f"{col}{suffix}"
            if base and other in paired:
                left, right = paired[base], paired[other]
                equal = left.eq(right) | (left.isna() & right.isna())
                if not bool(equal.all()):
                    raise ComparisonInputError(f"mismatched {col!r} values after keyed pairing")
                paired = paired.drop(columns=other)
    return paired


def decision_net_benefit(action: Iterable[bool | int], y: Iterable[float], ipcw_weight: Iterable[float],
                         threshold: float = DEFAULT_THRESHOLD,
                         population_weight: Iterable[float] | None = None) -> float:
    """Net benefit under the comparator design's explicit weighted formula."""
    if not 0 < threshold < 1:
        raise ValueError("threshold must lie strictly between zero and one")
    a, yy, w = (np.asarray(v, dtype=float) for v in (action, y, ipcw_weight))
    q = np.ones(len(a)) if population_weight is None else np.asarray(population_weight, dtype=float)
    if not (len(a) == len(yy) == len(w) == len(q)):
        raise ValueError("action, outcome and weights must have equal length")
    if np.any(~np.isfinite(q)) or np.any(q < 0) or q.sum() <= 0:
        raise ValueError("population weights must be finite, nonnegative and have positive sum")
    preference = threshold / (1.0 - threshold)
    return float(np.sum(q * w * a * (yy - (1.0 - yy) * preference)) / np.sum(q))


def _decision_metrics(action: np.ndarray, y: np.ndarray, w: np.ndarray, q: np.ndarray) -> dict[str, float | int]:
    known = w > 0
    aw = q * w
    tp = float(np.sum(aw * action * y))
    fp = float(np.sum(aw * action * (1 - y)))
    tn = float(np.sum(aw * (1 - action) * (1 - y)))
    fn = float(np.sum(aw * (1 - action) * y))
    return {
        "n": int(len(action)), "n_known": int(known.sum()), "alert_rate": float(np.average(action, weights=q)),
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "ppv": tp / (tp + fp) if tp + fp else float("nan"),
    }


def patient_balanced_weights(patient_ids: Iterable[Any]) -> np.ndarray:
    ids = pd.Series(list(patient_ids), dtype="object")
    counts = ids.value_counts(dropna=False)
    raw = ids.map(counts).rdiv(1.0).to_numpy(float)
    return raw * len(raw) / raw.sum() if len(raw) else raw


def evaluate_paired_rows(rows: pd.DataFrame, *, patient_col: str = "patient_id",
                         risk_col: str = "p_svd_12m", comparator_status_col: str = "status",
                         time_col: str = "time", event_col: str = "event", threshold: float = DEFAULT_THRESHOLD,
                         tau: float = DEFAULT_TAU_DAYS, n_boot: int = 200, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Evaluate rule/model decisions on one common set, with pooled and patient-balanced rows."""
    required = [patient_col, risk_col, comparator_status_col, time_col, event_col]
    missing = [c for c in required if c not in rows]
    if missing:
        raise ComparisonInputError(f"paired rows lack required columns: {missing}")
    data = rows.copy()
    evaluable = data[comparator_status_col].isin(["positive", "negative"])
    supported = pd.to_numeric(data[risk_col], errors="coerce").notna()
    common = data.loc[evaluable & supported].reset_index(drop=True)
    coverage = {
        "n_total": int(len(data)), "n_common": int(len(common)),
        "n_comparator_evaluable": int(evaluable.sum()), "n_model_supported": int(supported.sum()),
        "n_rule_only": int((evaluable & ~supported).sum()), "n_model_only": int((~evaluable & supported).sum()),
        "n_neither": int((~evaluable & ~supported).sum()),
        "n_rule_positive": int((data.loc[evaluable, comparator_status_col] == "positive").sum()),
    }
    coverage["degenerate_no_rule_positives"] = coverage["n_rule_positive"] == 0
    if common.empty:
        return pd.DataFrame(), coverage

    time = pd.to_numeric(common[time_col], errors="raise").to_numpy(float)
    event = pd.to_numeric(common[event_col], errors="raise").to_numpy(int)
    y, _known = outcome(time, event, "svd", tau)
    censoring = KMCensoring().fit(time, event)
    wr = ipcw_weights(time, event, tau, censoring)
    if not wr.support_ok:
        coverage["weight_support"] = wr.reason
    rule = common[comparator_status_col].eq("positive").to_numpy(float)
    model = common[risk_col].to_numpy(float) >= threshold
    strategies = {"rule_only": rule, "model_only": model.astype(float),
                  "rule_or_model": np.maximum(rule, model), "assess_all": np.ones(len(common)),
                  "assess_none": np.zeros(len(common))}
    records = []
    for population, q in (("pooled_landmark", np.ones(len(common))),
                          ("patient_balanced", patient_balanced_weights(common[patient_col]))):
        nbs = {name: decision_net_benefit(a, y, wr.w, threshold, q) for name, a in strategies.items()}
        rule_stats = _decision_metrics(rule, y, wr.w, q)
        for name, action in strategies.items():
            rec = {"population": population, "strategy": name, "threshold": threshold,
                   "net_benefit": nbs[name], "increment_vs_rule": nbs[name] - nbs["rule_only"],
                   "increment_vs_model": nbs[name] - nbs["model_only"]}
            if name == "rule_only":
                rec.update(rule_stats)
            records.append(rec)

    if n_boot > 0:
        def bootstrap_metrics(index: np.ndarray) -> dict:
            sub = common.iloc[index]
            sy, _ = outcome(sub[time_col].to_numpy(float), sub[event_col].to_numpy(int), "svd", tau)
            swr = ipcw_weights(sub[time_col].to_numpy(float), sub[event_col].to_numpy(int), tau,
                               KMCensoring().fit(sub[time_col], sub[event_col]))
            srule = sub[comparator_status_col].eq("positive").to_numpy(float)
            smodel = (sub[risk_col].to_numpy(float) >= threshold).astype(float)
            q = patient_balanced_weights(sub[patient_col])
            nr = decision_net_benefit(srule, sy, swr.w, threshold, q)
            nm = decision_net_benefit(smodel, sy, swr.w, threshold, q)
            no = decision_net_benefit(np.maximum(srule, smodel), sy, swr.w, threshold, q)
            return {"model_minus_rule": nm - nr, "or_minus_model": no - nm}

        boot = bootstrap_by_patient(bootstrap_metrics, common[patient_col].to_numpy(), n_boot=n_boot, seed=seed,
                                    keys=("model_minus_rule", "or_minus_model"), min_valid=min(10, n_boot),
                                    kind="fixed_oof_paired")
        coverage["bootstrap"] = boot.card() | {"intervals": boot.intervals}
    discordant = common.loc[(rule == 0) & model].copy()
    coverage["rule_negative_model_positive_n"] = int(len(discordant))
    coverage["rule_negative_model_positive_mean_risk"] = (
        float(discordant[risk_col].mean()) if len(discordant) else None
    )
    return pd.DataFrame(records), coverage


def write_comparison_artifacts(output_dir: str | Path, *, paired_rows: pd.DataFrame,
                               metrics: pd.DataFrame, coverage: dict, manifest: dict,
                               disagreements: pd.DataFrame | None = None) -> None:
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    paired_rows.to_parquet(out / "comparator_rows.parquet", index=False)
    (out / "coverage.json").write_text(json.dumps(coverage, indent=2, default=str) + "\n", encoding="utf-8")
    metrics.to_csv(out / "paired_metrics.csv", index=False)
    metrics.to_csv(out / "decision_curves.csv", index=False)
    if disagreements is not None:
        disagreements.to_csv(out / "disagreements.csv", index=False)
    lines = ["# Clinical comparator report", "", "This is a comparison-only artifact. It did not alter KAIROS.", "",
             f"- Comparator: `{COMPARATOR_ID}`", f"- Reference policy: `kairos_30_180_days`",
             f"- Existing decision threshold: {manifest.get('threshold', DEFAULT_THRESHOLD):.1%} (unchanged, assumed)",
             f"- Common evaluable rows: {coverage.get('n_common', 0)} of {coverage.get('n_total', 0)}",
             f"- Rule positives: {coverage.get('n_rule_positive', 0)}",
             "", "The rule is a current-echo finding, not a fitted probability. Brier, calibration and probability AUC: N/A.",
             "Results are synthetic and unvalidated."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def manifest_for_inputs(inputs: ComparisonInputs, *, model_version: str | None = None,
                        threshold: float = DEFAULT_THRESHOLD) -> dict:
    paths = asdict(inputs)
    return {
        "comparator_id": COMPARATOR_ID, "reference_policy": "kairos_30_180_days",
        "model_version": model_version, "threshold": threshold, "threshold_status": "unchanged_assumed",
        "output_scope": "comparison_only", "evidence_status": "synthetic_unvalidated",
        "inputs": {name: {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
                   for name, path in paths.items()},
    }
