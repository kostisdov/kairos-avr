"""Event support accounting and support gates (CR-04).

Event-support rules count unique patients: repeated landmark rows of one patient carry the same
future event and are not independent events. Both numbers are reported. Counts key on the
original ``patient_id``, never on a bootstrap cluster identifier, so resampling a patient twice
does not create a second event patient.

Gates (``config/model.yaml: support``, all assumed) apply independently per metric. A metric that
misses its gate is either not estimated (AUC, calibration intercept, calibration slope) or kept as
a descriptive value with an exploratory status (Brier, IPA, observed minus predicted). Quick-mode
decisions are always exploratory and can never select a winner or promote a model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd

CAUSE_CODES = {"svd": 1, "death": 2, "replacement": 3}
ESTIMATED_ONLY_WHEN_SUPPORTED = ("auc", "cal_intercept_offset", "cal_slope_logit")
DESCRIPTIVE = ("brier", "ipa", "obs_minus_pred")


class SupportStatus(StrEnum):
    OK = "ok"
    INSUFFICIENT_EVENTS = "insufficient_events"
    INSUFFICIENT_FOLLOWUP = "insufficient_followup"
    CENSORING_SUPPORT_FAILURE = "censoring_support_failure"
    FIT_FAILED = "fit_failed"


@dataclass
class SupportDecision:
    status: str
    gate: str
    reason: str = ""
    mode: str = "full"
    counts: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == SupportStatus.OK

    @property
    def exploratory(self) -> bool:
        return self.mode != "full" or not self.ok

    def card(self) -> dict:
        return {**asdict(self), "exploratory": self.exploratory}


def gates_for(cfg: dict, mode: str) -> dict:
    return {**(cfg.get("support") or {}).get(mode, {}), "mode": mode}


def unique_event_patients(lm: pd.DataFrame, cause_code: int, by: str | None = None) -> int | dict:
    """Patients with at least one row carrying ``cause_code`` (optionally per level of ``by``)."""
    rows = lm.loc[lm["event"] == cause_code]
    if by is None:
        return int(rows["patient_id"].nunique())
    return {k: int(v) for k, v in rows.groupby(by)["patient_id"].nunique().items()}


def event_rows(lm: pd.DataFrame, cause_code: int, by: str | None = None) -> int | dict:
    rows = lm.loc[lm["event"] == cause_code]
    if by is None:
        return int(len(rows))
    return {k: int(v) for k, v in rows.groupby(by).size().items()}


def event_summary(lm: pd.DataFrame, by: str | None = None) -> dict:
    """cause -> {event_patients, event_rows} (per ``by`` level when given)."""
    return {cause: {"event_patients": unique_event_patients(lm, code, by), "event_rows": event_rows(lm, code, by)}
            for cause, code in CAUSE_CODES.items()}


def horizon_counts(time, event, patient_ids, state, tau: float) -> dict:
    """Unique patients with the outcome by ``tau`` (cases) and with known non-outcome status
    (controls), plus rows observed event-free through ``tau``."""
    from kairos.evaluation.metrics import outcome

    y, known = outcome(time, event, state, tau)
    pids = np.asarray(patient_ids)
    cases = int(pd.Series(pids[(y == 1) & known]).nunique())
    controls = int(pd.Series(pids[(y == 0) & known]).nunique())
    time = np.asarray(time, dtype=float)
    observed_through = int(np.sum((time >= tau - 1e-9) & ~((time <= tau) & (np.asarray(event) != 0))))
    return {"case_patients": cases, "control_patients": controls, "case_rows": int(((y == 1) & known).sum()),
            "rows_observed_through_horizon": observed_through}


def metric_gate(metric: str, counts: dict, G_at_tau: float, gates: dict) -> SupportDecision:
    mode = gates.get("mode", "full")
    if G_at_tau < float(gates.get("min_censoring_survival", 0.0)):
        return SupportDecision(SupportStatus.CENSORING_SUPPORT_FAILURE, "min_censoring_survival",
                               f"censoring survival {G_at_tau:.3g} at the horizon below {gates['min_censoring_survival']}", mode, counts)
    if counts.get("rows_observed_through_horizon", 0) < int(gates.get("min_at_risk", 0)):
        return SupportDecision(SupportStatus.INSUFFICIENT_FOLLOWUP, "min_at_risk",
                               f"{counts.get('rows_observed_through_horizon', 0)} rows observed through the horizon", mode, counts)
    need_ev = int(gates["slope_min_events"] if metric == "cal_slope_logit" else gates["metric_min_event_patients"])
    gate = "slope_min_events" if metric == "cal_slope_logit" else "metric_min_event_patients"
    if counts["case_patients"] < need_ev:
        return SupportDecision(SupportStatus.INSUFFICIENT_EVENTS, gate,
                               f"{counts['case_patients']} outcome patients, gate {need_ev}", mode, counts)
    if counts["control_patients"] < int(gates["metric_min_control_patients"]):
        return SupportDecision(SupportStatus.INSUFFICIENT_EVENTS, "metric_min_control_patients",
                               f"{counts['control_patients']} control patients, gate {gates['metric_min_control_patients']}", mode, counts)
    return SupportDecision(SupportStatus.OK, gate, "", mode, counts)


def bootstrap_gate(n_requested: int, n_valid: int, gates: dict) -> SupportDecision:
    mode = gates.get("mode", "full")
    need = int(gates["bootstrap_min_valid"])
    frac = float(gates["bootstrap_min_success_fraction"])
    counts = {"n_requested": n_requested, "n_valid": n_valid}
    if n_valid < need or (n_requested and n_valid / n_requested < frac):
        return SupportDecision(SupportStatus.INSUFFICIENT_EVENTS, "bootstrap_min_valid",
                               f"{n_valid} of {n_requested} valid replicates; gate {need} and {frac:.0%}", mode, counts)
    return SupportDecision(SupportStatus.OK, "bootstrap_min_valid", "", mode, counts)


def coverage(valid: np.ndarray, patient_ids) -> dict:
    """Share of intended rows and patients with a valid out-of-fold prediction."""
    valid = np.asarray(valid, dtype=bool)
    pids = pd.Series(np.asarray(patient_ids))
    n_pat = pids.nunique()
    return {"rows": int(len(valid)), "rows_valid": int(valid.sum()),
            "row_coverage": float(valid.mean()) if len(valid) else 0.0,
            "patients": int(n_pat), "patients_with_valid_rows": int(pids[valid].nunique()),
            "patient_coverage": float(pids[valid].nunique() / n_pat) if n_pat else 0.0}


def common_evaluable_mask(*masks) -> np.ndarray:
    """Families or variants are only ever compared on the intersection of their valid rows."""
    out = None
    for m in masks:
        m = np.asarray(m, dtype=bool)
        out = m if out is None else out & m
    return out
