"""Evaluation-only synthetic truth: the column names that must never reach a predictor, and
summaries of the observation process.

Kept free of modelling imports so the landmark builder can enforce the separation without a
dependency on the generator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRUTH_TABLES = ("truth", "truth_visits", "truth_enrolled")

# every column of the truth tables except the join key; none may appear in a predictor table
TRUTH_COLUMNS = frozenset({
    "phenotype_latent", "lpa_high_latent", "dp_ucmgp_latent_log_implant",
    "initiation_date", "initiation_before_end", "threshold_crossing_date", "crossing_before_reference",
    "detection_delay_days", "missed_crossing", "last_negative_date", "onset_interval_lo", "onset_interval_hi",
    "crossing_in_interval", "n_thrombosis_episodes", "thrombosis_episodes", "n_scheduled_visits", "n_attended_visits",
    "visit_index", "scheduled_date", "years_since_implant", "attended", "eligible", "active_thrombosis",
    "true_mean_gradient", "true_eoa", "true_dvi", "true_ar_grade",
    # replayable noise-free trajectory, generator 2.2
    "traj_t0_years", "traj_t_end_years", "traj_t_onset_years", "traj_regurgitant", "traj_slope_mmhg_per_year",
    "traj_eoa_fall_fraction", "traj_ref_gradient", "traj_ref_eoa", "traj_ref_dvi", "traj_ref_ar",
    # enrolled (implantation-origin) view, generator 2.1
    "attempt_index", "planned_reference_date", "reference_eligible", "exclusion_reason", "latent_death_date",
    "latent_replacement_date", "admin_end_date",
    # names used by generator version 1, rejected if an old table is passed in
    "svd_onset_date", "svd_phenotype",
})


def truth_columns_in(*frames: pd.DataFrame) -> set[str]:
    found: set[str] = set()
    for df in frames:
        if df is not None:
            found |= TRUTH_COLUMNS & set(df.columns)
    return found


def attendance_by_year(truth_visits: pd.DataFrame) -> pd.DataFrame:
    """Attendance among scheduled follow-up visits of patients alive with the index valve at that
    visit (the generator schedules no visit after death, replacement or end of follow-up, so a
    death never enters the missing-visit denominator). Year = scheduled visit rounded to whole
    years since implantation; the reference study is excluded."""
    tv = truth_visits[(truth_visits["visit_index"] > 0) & truth_visits["eligible"]]
    if tv.empty:
        return pd.DataFrame(columns=["year", "eligible_visits", "attended", "attendance"])
    year = np.rint(tv["years_since_implant"].to_numpy(dtype=float)).astype(int)
    g = tv.assign(year=year).groupby("year")["attended"].agg(["size", "sum"]).reset_index()
    g.columns = ["year", "eligible_visits", "attended"]
    g["attendance"] = g["attended"] / g["eligible_visits"]
    return g
