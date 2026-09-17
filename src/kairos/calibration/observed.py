"""Observed-data profiling (design section 4.1, work package G1).

Input is a published, frozen **normalized research snapshot** from the standardization service
(``kairos-standardize``, not yet implemented), in this long-format contract:

* ``manifest.json``: :class:`SnapshotManifest` (``source_kind``, ``purpose``, ``published``, ``frozen``,
  table hashes, vocabulary versions, ``lineage_complete``);
* ``persons``: ``person_id``, ``index_date`` (ISO), ``index_date_precision`` (day|month|year),
  ``route``, ``value_origin``;
* ``observations``: ``person_id``, ``record_id``, ``variable``, ``value`` (numeric) or ``value_text``,
  ``unit``, ``date``, ``date_precision``, ``value_origin``, ``parent_record_ids`` (JSON list),
  ``duplicate_status`` (unique|possible_duplicate|exact_duplicate), ``suppressed`` (bool);
* ``partition``: ``person_id`` -> ``fit`` | ``holdout`` (the frozen source split).

Policies (``config/generator_calibration.yaml: evidence`` and ``profiling``):

* a training scaffold (``source_kind=real_with_synthetic_defaults``) or an unpublished, unfrozen or
  lineage-incomplete snapshot is rejected: reference the underlying observed research snapshot;
* a record is admitted only when its origin is ``observed`` or ``derived`` *and every ancestor* is admitted;
  a missing ancestor is an input error, never permission to treat the value as real;
* only persons in the requested partition role contribute (holdout persons never inform fitted profiles);
* one result per person per summary: the first eligible result in the declared baseline window when
  day-precision dates allow it; otherwise an export-level cross-sectional description (never fitted as
  an implantation baseline);
* longitudinal slopes need day-precision dates; year-only records give annual descriptive means only;
* exact duplicates are dropped and possible duplicates excluded (sensitivity: kept);
* suppressed cells are never converted to exact values;
* support: at least ``min_patients_for_local_distribution_fit`` persons for a locally fitted
  distribution and ``min_paired_patients_for_dependency_fit`` for a dependency; below, descriptive only;
* uncertainty: person-cluster bootstrap.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from kairos.calibration.schema import (
    ACCEPTED_ORIGINS,
    SCAFFOLD_SOURCE_KINDS,
    ObservedProfile,
    PopulationSpec,
    SnapshotManifest,
    SummaryStat,
    fingerprint,
)

DAYS = 365.25


class SnapshotRejected(ValueError):
    """The snapshot cannot be used as observed evidence."""


@dataclass
class Snapshot:
    manifest: SnapshotManifest
    persons: pd.DataFrame
    observations: pd.DataFrame
    partition: pd.DataFrame

    @classmethod
    def load(cls, folder: str | Path) -> Snapshot:
        folder = Path(folder)
        manifest = SnapshotManifest.from_dict(json.loads((folder / "manifest.json").read_text(encoding="utf-8")))
        return cls(manifest, pd.read_parquet(folder / "persons.parquet"), pd.read_parquet(folder / "observations.parquet"),
                   pd.read_parquet(folder / "partition.parquet"))

    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "manifest.json").write_text(json.dumps(self.manifest.to_dict(), indent=2), encoding="utf-8")
        self.persons.to_parquet(folder / "persons.parquet", index=False)
        self.observations.to_parquet(folder / "observations.parquet", index=False)
        self.partition.to_parquet(folder / "partition.parquet", index=False)


def default_policy(cfg: dict) -> dict:
    prof = cfg.get("profiling", {})
    ev = cfg.get("evidence", {})
    return {"min_local_fit": int(prof.get("min_patients_for_local_distribution_fit", 30)),
            "min_paired_fit": int(prof.get("min_paired_patients_for_dependency_fit", 50)),
            "require_frozen": bool(ev.get("require_frozen_manifests", True)),
            "reject_scaffold": bool(ev.get("reject_training_scaffold", True)),
            "require_lineage": bool(ev.get("require_transitive_observed_lineage", True)),
            "possible_duplicates": "exclude", "bootstrap_replicates": int(cfg.get("calibration", {}).get("source_bootstrap_replicates", 30)),
            "baseline_window_days": [-30, 180], "baseline_variables": [], "categorical_variables": [],
            "longitudinal_variables": [], "paired_variables": [], "seed": 20260917}


def check_snapshot(manifest: SnapshotManifest, policy: dict) -> None:
    if policy["reject_scaffold"] and (manifest.source_kind in SCAFFOLD_SOURCE_KINDS or manifest.purpose == "pipeline_test"):
        raise SnapshotRejected("training scaffold manifests (synthetic defaults) cannot be profiled as observed evidence; "
                               "reference the underlying observed research snapshot")
    if policy["require_frozen"] and not (manifest.published and manifest.frozen):
        raise SnapshotRejected("the snapshot must be a published, frozen manifest")
    if policy["require_lineage"] and not manifest.lineage_complete:
        raise SnapshotRejected("the snapshot does not declare complete source lineage")


def admitted_records(obs: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Boolean mask of records whose origin and every transitive ancestor are accepted observations."""
    origin = obs.set_index("record_id")["value_origin"].to_dict()
    parents = {r: (json.loads(p) if isinstance(p, str) and p else list(p) if isinstance(p, (list, tuple, np.ndarray)) else [])
               for r, p in zip(obs["record_id"], obs.get("parent_record_ids", pd.Series([None] * len(obs))))}
    memo: dict = {}
    reasons: dict = {"missing_ancestor": 0, "non_observed_origin": 0, "non_observed_ancestor": 0}

    def ok(rid, stack=()):
        if rid in memo:
            return memo[rid]
        if rid not in origin:
            raise SnapshotRejected(f"lineage references unknown record {rid!r}: missing lineage is an input error")
        if rid in stack:
            raise SnapshotRejected(f"lineage cycle at record {rid!r}")
        if origin[rid] not in ACCEPTED_ORIGINS:
            memo[rid] = False
            return False
        memo[rid] = all(ok(p, (*stack, rid)) for p in parents.get(rid, []))
        return memo[rid]

    mask = []
    for rid in obs["record_id"]:
        for p in parents.get(rid, []):
            if p not in origin:
                reasons["missing_ancestor"] += 1
                raise SnapshotRejected(f"record {rid!r} has an ancestor {p!r} absent from the snapshot")
        good = ok(rid)
        if not good:
            reasons["non_observed_origin" if origin[rid] not in ACCEPTED_ORIGINS else "non_observed_ancestor"] += 1
        mask.append(good)
    return pd.Series(mask, index=obs.index), reasons


def _bootstrap_se(values_by_person: pd.Series, fn, B: int, seed: int) -> float | None:
    if B <= 0 or len(values_by_person) < 2:
        return None
    rng = np.random.default_rng(seed)
    arr = values_by_person.to_numpy()
    stats = [fn(arr[rng.integers(0, len(arr), len(arr))]) for _ in range(B)]
    stats = np.asarray([s for s in stats if s is not None and np.isfinite(s)], dtype=float)
    return float(stats.std(ddof=1)) if len(stats) > 1 else None


def _support(n: int, need: int) -> tuple[str, str]:
    if n == 0:
        return "not_estimated", "no eligible person"
    if n < need:
        return "descriptive_only", f"{n} eligible persons < {need}: descriptive only, external prior needed"
    return "fit_local", ""


def profile_observed(snapshot: Snapshot, population: PopulationSpec, split_role: str, policy: dict) -> ObservedProfile:
    check_snapshot(snapshot.manifest, policy)
    persons = snapshot.persons.copy()
    part = snapshot.partition.set_index("person_id")["role"]
    in_split = persons["person_id"].map(part) == split_role
    exclusions = {"persons_other_partition": int((~in_split).sum())}
    persons = persons[in_split & persons["value_origin"].isin(ACCEPTED_ORIGINS)]
    exclusions["persons_non_observed_origin"] = int(in_split.sum() - len(persons))
    if population.routes:
        before = len(persons)
        persons = persons[persons["route"].isin(list(population.routes))]
        exclusions["persons_outside_population_routes"] = before - len(persons)

    obs = snapshot.observations.copy()
    mask, lineage = admitted_records(obs)
    exclusions.update({f"records_{k}": v for k, v in lineage.items()})
    obs = obs[mask & obs["person_id"].isin(persons["person_id"])]
    dup = obs.get("duplicate_status", pd.Series("unique", index=obs.index)).fillna("unique")
    exclusions["records_exact_duplicate"] = int((dup == "exact_duplicate").sum())
    obs = obs[dup != "exact_duplicate"]
    if policy["possible_duplicates"] == "exclude":
        exclusions["records_possible_duplicate"] = int((dup.reindex(obs.index) == "possible_duplicate").sum())
        obs = obs[dup.reindex(obs.index) != "possible_duplicate"]
    supp = obs.get("suppressed", pd.Series(False, index=obs.index)).fillna(False).astype(bool)
    text = obs.get("value_text", pd.Series(None, index=obs.index)).astype(str)
    suppressed = supp | text.str.strip().str.startswith("<")
    exclusions["records_suppressed_not_converted"] = int(suppressed.sum())
    obs = obs[~suppressed]

    idx = persons.set_index("person_id")
    B, seed = int(policy["bootstrap_replicates"]), int(policy["seed"])
    summaries, not_estimated = [], []
    lo, hi = policy["baseline_window_days"]

    def baseline_frame(variable: str) -> tuple[pd.DataFrame, str]:
        o = obs[obs["variable"] == variable].copy()
        if o.empty:
            return o, "baseline"
        o["index_date"] = o["person_id"].map(idx["index_date"])
        o["index_precision"] = o["person_id"].map(idx["index_date_precision"])
        exact = (o["date_precision"] == "day") & (o["index_precision"] == "day")
        if exact.any():
            d = pd.to_datetime(o.loc[exact, "date"]) - pd.to_datetime(o.loc[exact, "index_date"])
            o = o.loc[exact][(d.dt.days >= lo) & (d.dt.days <= hi)]
            o = o.sort_values(["person_id", "date", "record_id"]).groupby("person_id").head(1)
            return o, "baseline"
        # no defensible baseline window: export-level cross-sectional description, one record per person
        o = o.sort_values(["person_id", "date", "record_id"]).groupby("person_id").head(1)
        return o, "cross_sectional"

    for variable in policy["baseline_variables"]:
        o, kind = baseline_frame(variable)
        vals = pd.to_numeric(o.get("value"), errors="coerce").dropna() if len(o) else pd.Series(dtype=float)
        n = int(len(vals))
        status, reason = _support(n, policy["min_local_fit"])
        if kind == "cross_sectional":
            status, reason = "descriptive_only", "no day-precision baseline window: export-level cross-sectional description"
        if n == 0:
            not_estimated.append({"variable": variable, "reason": "no eligible person"})
            continue
        unit = str(o["unit"].iloc[0]) if "unit" in o and len(o) else ""
        for stat, fn in (("mean", np.mean), ("sd", lambda a: np.std(a, ddof=1) if len(a) > 1 else np.nan)):
            summaries.append(SummaryStat(variable, stat, float(fn(vals.to_numpy())), _bootstrap_se(vals, fn, B, seed), n,
                                         int((obs["variable"] == variable).sum()), kind, f"{lo}..{hi} days", unit, status, reason).to_dict())

    for variable in policy["categorical_variables"]:
        o, kind = baseline_frame(variable)
        if o.empty:
            not_estimated.append({"variable": variable, "reason": "no eligible person"})
            continue
        levels = o["value_text"].astype(str)
        n = int(len(levels))
        status, reason = _support(n, policy["min_local_fit"])
        if kind == "cross_sectional":
            status, reason = "descriptive_only", "no day-precision baseline window: export-level cross-sectional description"
        for level in sorted(levels.unique()):
            ind = (levels == level).astype(float)
            summaries.append(SummaryStat(variable, f"proportion:{level}", float(ind.mean()), _bootstrap_se(ind, np.mean, B, seed), n,
                                         int((obs["variable"] == variable).sum()), kind, f"{lo}..{hi} days", "", status, reason).to_dict())

    for a, b in policy["paired_variables"]:
        oa, ka = baseline_frame(a)
        ob, kb = baseline_frame(b)
        pair = pd.merge(oa[["person_id", "value"]], ob[["person_id", "value"]], on="person_id", suffixes=("_a", "_b")).dropna()
        n = int(len(pair))
        status, reason = _support(n, policy["min_paired_fit"])
        if "cross_sectional" in (ka, kb):
            status, reason = "descriptive_only", "no day-precision baseline window for both variables"
        if n < 3:
            not_estimated.append({"variable": f"{a}~{b}", "reason": f"{n} paired persons"})
            continue
        r = float(np.corrcoef(pair["value_a"], pair["value_b"])[0, 1])
        rows = pair[["value_a", "value_b"]].to_numpy()
        rng = np.random.default_rng(seed)
        boots = [np.corrcoef(rows[i].T)[0, 1] for i in (rng.integers(0, n, n) for _ in range(B))] if B else []
        boots = [x for x in boots if np.isfinite(x)]
        summaries.append(SummaryStat(a, f"correlation:{b}", r, float(np.std(boots, ddof=1)) if len(boots) > 1 else None, n, 2 * n,
                                     "dependency", f"{lo}..{hi} days", "", status, reason).to_dict())

    for variable in policy["longitudinal_variables"]:
        o = obs[obs["variable"] == variable].copy()
        o["value"] = pd.to_numeric(o["value"], errors="coerce")
        o = o.dropna(subset=["value"])
        exact = o[o["date_precision"] == "day"]
        slopes = {}
        for pid, g in exact.groupby("person_id"):
            t = (pd.to_datetime(g["date"]) - pd.to_datetime(g["date"]).min()).dt.days.to_numpy() / DAYS
            if len(g) >= 2 and t.max() - t.min() >= 0.5:
                slopes[pid] = float(np.polyfit(t, g["value"].to_numpy(), 1)[0])   # one slope per person: equal person weight
        s = pd.Series(slopes, dtype=float)
        if len(s):
            status, reason = _support(len(s), policy["min_local_fit"])
            summaries.append(SummaryStat(variable, "slope_per_year", float(s.mean()), _bootstrap_se(s, np.mean, B, seed), len(s),
                                         int(len(exact)), "longitudinal", "day-precision dates", "", status, reason).to_dict())
        yearly = o[o["date_precision"] == "year"]
        if len(yearly):
            yearly = yearly.assign(year=yearly["date"].astype(str).str[:4])
            per_person_year = yearly.groupby(["person_id", "year"])["value"].mean()          # a person counts once per year
            by_year = per_person_year.groupby(level="year")
            for year, vals in by_year:
                summaries.append(SummaryStat(variable, f"annual_mean:{year}", float(vals.mean()), None, int(len(vals)), int(len(vals)),
                                             "annual_descriptive", "year-only records", "", "descriptive_only",
                                             "year-only dates: annual description, no slopes or intervals").to_dict())
        if not len(s) and not len(yearly):
            not_estimated.append({"variable": variable, "reason": "no longitudinal records with usable dates"})

    return ObservedProfile(snapshot_id=snapshot.manifest.snapshot_id, population_id=population.population_id, split_role=split_role,
                           policy=policy, persons_in_scope=int(len(persons)), summaries=summaries, exclusions=exclusions,
                           not_estimated=not_estimated, snapshot_fingerprint=fingerprint(snapshot.manifest.to_dict()))


def profile_to_targets(profile: ObservedProfile, population: PopulationSpec, tolerance_sd: float = 2.0) -> list:
    """Evidence targets from a local profile: only ``fit_local`` summaries of the local population may be
    fit targets, and only for the ``local_export_like`` preset; others are descriptive (sensitivity)."""
    from kairos.calibration.schema import EvidenceTarget

    out = []
    for s in profile.summaries:
        stat = s["statistic"]
        kind = "proportion" if stat.startswith("proportion:") else "correlation" if stat.startswith("correlation:") else \
            "mean" if stat == "mean" else "sd" if stat == "sd" else "longitudinal_mean"
        fit_ok = s["support"] == "fit_local" and s["se"] is not None and population.preset == "local_export_like"
        tol = tolerance_sd * s["se"] if s["se"] else None
        out.append(EvidenceTarget(
            target_id=f"local:{profile.snapshot_id}:{s['variable']}:{stat}", kind=kind, variable=s["variable"], statistic=stat,
            estimate=s["value"], uncertainty={"type": "se", "value": s["se"]} if s["se"] else {"type": "none"},
            sample_size=s["n_patients"], population={"preset": population.preset, "view": "kairos_reference_eligible",
                                                     "population_id": population.population_id},
            time={"origin": "implantation", "window": s["window"]}, unit=s["unit"], study_group_id=f"local:{profile.snapshot_id}",
            source={"citation": f"observed snapshot {profile.snapshot_id}", "location": s["kind"],
                    "release_hash": profile.snapshot_fingerprint},
            review_status="approved" if fit_ok else "proposed", reviewer="profile policy" if fit_ok else None,
            role="fit" if fit_ok else "sensitivity",
            compatibility={"status": "compatible" if fit_ok else "descriptive_only", "reason": s["reason"]},
            acceptance={"tolerance": tol, "tolerance_type": "absolute", "weight": 1.0, "mandatory": fit_ok},
            origin="local_profile").to_dict())
    return out


def today() -> str:
    return date.today().isoformat()
