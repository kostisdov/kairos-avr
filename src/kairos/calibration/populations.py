"""Population specifications and study views.

A calibration never runs without a :class:`PopulationSpec`. Two presets exist: ``local_export_like``
(what is estimable about the supplied export, which may be a selected sample) and
``external_population`` (a named study or registry population). External targets are compared on the
*study view* whose origin and selection match the source:

* ``enrolled_implantation_origin``: every attempted patient from implantation, including deaths and
  replacements before a reference echo could be obtained (``truth_enrolled``);
* ``kairos_reference_eligible``: the KAIROS cohort (reference echo 30 to 180 days after implantation).

Comparing an implantation-origin curve with the reference-eligible cohort would condition on survival
to the reference study; :func:`study_view` never does that silently.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from kairos.calibration.schema import PopulationSpec

DAYS = 365.25
VIEWS = ("enrolled_implantation_origin", "kairos_reference_eligible")


def load_populations(path: str | Path) -> dict[str, PopulationSpec]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {k: PopulationSpec.from_dict({"population_id": k, **v}) for k, v in raw["populations"].items()}


def _years(a, b) -> float | None:
    if a is None or b is None or (isinstance(a, float) and np.isnan(a)) or pd.isna(a) or pd.isna(b):
        return None
    return (a - b).days / DAYS


def study_view(cohort, view: str) -> pd.DataFrame:
    """One row per person of the view with a time origin and event times (years from the origin):
    ``t_death``, ``t_replacement``, ``t_svd`` (adjudicated SVD; reference-eligible persons only),
    ``t_end`` (end of follow-up) and baseline covariates."""
    if view not in VIEWS:
        raise ValueError(f"unknown study view {view!r}; known: {VIEWS}")
    ev = cohort.events.set_index("patient_id") if len(cohort.events) else pd.DataFrame()
    if view == "enrolled_implantation_origin":
        enr = cohort.truth_enrolled
        if enr is None or len(enr) == 0:
            raise ValueError("the implantation-origin view needs truth_enrolled (generator 2.1 or later)")
        rows = []
        for r in enr.itertuples(index=False):
            origin = r.implant_date
            end_reason = r.end_reason
            t_end = _years(r.followup_end_date, origin)
            svd = ev.loc[r.patient_id, "svd_adjudicated_date"] if (r.reference_eligible and r.patient_id in ev.index) else None
            rows.append({"patient_id": r.patient_id, "origin": origin, "route": r.route, "design_class": r.design_class,
                         "age_at_implant": r.age_at_implant, "sex": r.sex, "reference_eligible": bool(r.reference_eligible),
                         "t_end": t_end, "t_death": t_end if end_reason == "death" else None,
                         "t_replacement": t_end if end_reason == "replacement" else None,
                         "t_svd": _years(svd, origin) if svd is not None and pd.notna(svd) else None})
        return pd.DataFrame(rows)
    pts = cohort.patients.set_index("patient_id")
    rows = []
    for pid, e in ev.iterrows():
        origin = e["reference_date"]
        rows.append({"patient_id": pid, "origin": origin, "route": pts.loc[pid, "route"], "design_class": pts.loc[pid, "design_class"],
                     "age_at_implant": pts.loc[pid, "age_at_implant"], "sex": pts.loc[pid, "sex"], "reference_eligible": True,
                     "t_end": _years(e["end_followup_date"], origin),
                     "t_death": _years(e["death_date"], origin) if pd.notna(e["death_date"]) else None,
                     "t_replacement": _years(e["replacement_date"], origin) if pd.notna(e["replacement_date"]) else None,
                     "t_svd": _years(e["svd_adjudicated_date"], origin) if pd.notna(e["svd_adjudicated_date"]) else None})
    return pd.DataFrame(rows)


def compile_population_params(pop: PopulationSpec) -> dict:
    """Scenario parameter overrides implied by the population specification."""
    out: dict = {"implant_year_range": list(pop.implant_years), "followup_years_max": float(pop.max_followup_years),
                 "generator": {"baseline": {"calendar": {"study_end": pop.study_end, "time_origin": "implantation"}}}}
    if pop.routes:
        tavr = float(pop.routes.get("TAVR", 0.0))
        total = sum(float(v) for v in pop.routes.values())
        out["p_tavr"] = tavr / total if total else tavr
    date.fromisoformat(pop.study_end)   # validates the calendar
    return out
