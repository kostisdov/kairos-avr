"""Score a synthetic-trained model on the real notes extract, without training on it.

The real extract cannot feed the KAIROS core model: dates are year-only, age is masked, only a
minority of patients have any prosthetic gradient, and almost none have a dated reference study.
What it can test is the implant-time model: the ``reference`` ladder step (valve age, route and
design class), trained on a synthetic cohort and scored once on the real patients. It is an external
check of whether the synthetic world ranks real valves in the right order, not a validation of
KAIROS.

Outcome, from the notes (label: assumed): the first note year after the implant year that mentions
valve-in-valve, a redo operation or prosthetic dysfunction, i.e. bioprosthetic valve failure or
dysfunction of any cause, not adjudicated SVD. Patients without such a mention are censored at their
last note year. Times are whole years. Only patients with a documented surgical or transcatheter
implant and at least one later note year are scored.

Only aggregates leave this module: counts, concordance and its bootstrap interval. No patient row is
returned by :func:`score_extract` or written by the script.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

from kairos.evaluation.surveillance import _step_blocks
from kairos.modelling.cif import combine_cause_specific, probabilities_at
from kairos.modelling.train import fit_step, landmark_from_cohort

EVENT_FLAGS = ("ViV", "redo", "prosthetic_dysfunction")
OUTCOME_DEFINITION = ("first note year after the implant year mentioning valve-in-valve, redo or prosthetic "
                      "dysfunction; censored at the last note year; whole years")


def real_outcomes(passport: pd.DataFrame, note_rows: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per-patient implant-time features and (time, event); plus exclusion counts."""
    excl = {"route_not_single_documented_implant": 0, "no_implant_year": 0, "no_note_after_implant_year": 0}
    notes = note_rows.groupby("pid")
    rows = []
    for p in passport.itertuples(index=False):
        if p.implant_year is None or pd.isna(p.implant_year):
            excl["no_implant_year"] += 1
            continue
        implant = int(p.implant_year)
        g = notes.get_group(p.patient_id)
        route = index_route(g, implant)
        if route is None:
            excl["route_not_single_documented_implant"] += 1
            continue
        later = g[g["yr"] > implant]
        if later.empty:
            excl["no_note_after_implant_year"] += 1
            continue
        flagged = later[later["events"].apply(lambda e: any(bool(e.get(k)) for k in EVENT_FLAGS))]
        event = not flagged.empty
        end = int(flagged["yr"].min()) if event else int(later["yr"].max())
        rows.append({"patient_id": p.patient_id, "route": route, "design_class": p.design_class or "unknown",
                     "implant_year": implant, "time": float(end - implant), "event": bool(event)})
    return pd.DataFrame(rows, columns=["patient_id", "route", "design_class", "implant_year", "time", "event"]), excl


def index_route(note_rows: pd.DataFrame, implant_year: int) -> str | None:
    """Route of the index valve: the implant notes of the first implant year. A patient whose later
    notes document a second procedure (valve-in-valve, redo) keeps the route of the first; that later
    procedure is the outcome, not the index. Both routes in the same year is ambiguous and returns None."""
    first = note_rows[note_rows["implant_note"] & (note_rows["yr"] == implant_year)]
    tavr, savr = bool(first["tavr"].any()), bool((first["savr"] & ~first["tavr"]).any())
    if tavr == savr:
        return None
    return "TAVR" if tavr else "SAVR"


def implant_time_risk(cohort, cfg: dict, patients: pd.DataFrame, horizon_years: float = 5.0,
                      valve_age_years: float = 0.25) -> np.ndarray:
    """Predicted SVD cumulative incidence by ``horizon_years`` from the ``reference`` step fitted on the
    synthetic ``cohort``, at a landmark ``valve_age_years`` after implantation (the reference study)."""
    lm = landmark_from_cohort(cohort, cfg)
    pipe, model, *_ = fit_step(lm, _step_blocks(cfg, "reference"), cfg)
    rows = patients[["route", "design_class"]].copy()
    rows["valve_age_years"] = valve_age_years
    X = pipe.transform(rows)
    for s in model.strata:
        X[s] = rows[s].astype(str).to_numpy()
    out = np.full(len(rows), np.nan)
    ok, _ = model.row_support(X)
    if ok.any():
        cifs = combine_cause_specific(model.cumulative_hazards(X.loc[ok]), model.grid)
        horizons = [float(h) for h in cfg["horizons_years"]]
        idx = horizons.index(horizon_years) if horizon_years in horizons else len(horizons) - 1
        out[np.flatnonzero(ok)] = probabilities_at(cifs, model.grid, horizons)["svd"][:, idx]
    return out


def _c(time, risk, event) -> float:
    return float(concordance_index(time, -np.asarray(risk, dtype=float), event))


def score_extract(outcomes: pd.DataFrame, risk: np.ndarray, n_boot: int = 1000, seed: int = 20260917) -> dict:
    """Aggregate-only scorecard: Harrell's C for the model and for route alone, with patient-bootstrap
    95% intervals. Year-only times make many pairs tied; ties in time are not comparable."""
    ok = np.isfinite(risk)
    d = outcomes.loc[ok].reset_index(drop=True)
    r = risk[ok]
    t, e = d["time"].to_numpy(float), d["event"].to_numpy(bool)
    route = (d["route"] == "SAVR").to_numpy(float)       # surgical valves have the longer follow-up and more failures
    card = {"patients_scored": int(len(d)), "unsupported_by_model": int((~ok).sum()), "events": int(e.sum()),
            "outcome": OUTCOME_DEFINITION, "median_followup_years": float(np.median(t)) if len(t) else None}
    if e.sum() == 0 or e.all():
        card["status"] = "not_computable: needs both events and non-events"
        return card
    card["c_model"] = _c(t, r, e)
    card["c_route_only"] = _c(t, route, e)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(d), len(d))
        if e[i].any() and not e[i].all():
            bs.append((_c(t[i], r[i], e[i]), _c(t[i], route[i], e[i])))
    bs = np.array(bs)
    if len(bs):
        card["c_model_ci95"] = [round(float(np.percentile(bs[:, 0], q)), 3) for q in (2.5, 97.5)]
        card["c_route_only_ci95"] = [round(float(np.percentile(bs[:, 1], q)), 3) for q in (2.5, 97.5)]
        card["bootstrap_valid"] = int(len(bs))
    card["status"] = "ok" if card["events"] >= 10 else "ok, fewer than 10 events: descriptive only"
    return card
