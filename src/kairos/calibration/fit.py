"""Bounded simulated minimum distance with prior regularization (design section 6, G4).

Objective for a parameter vector ``theta``::

    L(theta) = sum_g W_g * D_g / sum_g W_g + lambda * prior_penalty(theta)
    D_g      = sum_i w_i z_i^2 / sum_i w_i          (points i of study group g)
    z_i^2    = (sim_i - obs_i)^2 / (obs_se_i^2 + mc_se_i^2 + mismatch_i^2)

``W_g`` is the group's capped total weight (curve points share their target's weight), so neither a
densely digitized curve nor overlapping publications gain influence. ``sim_i`` is averaged over the
common fit seeds; the same random streams are reused for every candidate. This is a normalized block
distance, not an independent-point likelihood, and its fitted parameters are not a Bayesian posterior.

Procedure: validate targets and parameters; flag conflicting targets; evaluate a Latin hypercube design
plus the declared starting values; refine the best candidates with bounded Powell searches inside the
evaluation budget; re-evaluate finalists on independent diagnostic seeds; check boundaries, local
identifiability (objective change when each free parameter moves) and per-target tolerances; set the
status. An optimizer that terminates is not an accepted calibration.

Statuses: ``accepted_for_simulation`` (every mandatory fit target within tolerance on the diagnostic
seeds, no conflict, every free parameter locally identifiable), ``failed_targets``,
``nonidentifiable``, ``insufficient_evidence`` (no usable fit target or no free parameter linked to one)
and ``incomplete`` (time budget ended before the acceptance procedure).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from kairos.calibration.compiler import compile_spec
from kairos.calibration.parameters import from_unit, prior_penalty, starting_values, to_unit
from kairos.calibration.schema import ParameterDef, PopulationSpec
from kairos.calibration.targets import TargetPoint, simulate_point
from kairos.simulation.generators import generate_cohort


class BudgetExhausted(Exception):
    pass


@dataclass
class FitSettings:
    initial_design_points: int = 32
    multistarts: int = 4
    max_objective_evaluations: int = 200
    attempted_patients_per_evaluation: int = 2500
    fit_simulation_seeds: tuple = (1201, 1202, 1203)
    diagnostic_simulation_seeds: tuple = (2201, 2202, 2203)
    optimizer_seed: int = 20260917
    prior_lambda: float = 1.0
    cap_per_group: float = 1.0
    identifiability_step: float = 0.1       # unit-cube step for the local identifiability check
    identifiability_min_change: float = 0.05  # objective units
    boundary_margin: float = 0.01
    max_seconds: float | None = None
    finalists: int = 3

    @classmethod
    def from_config(cls, cfg: dict, **overrides) -> FitSettings:
        c = cfg.get("calibration", {})
        base = cls(initial_design_points=int(c.get("initial_design_points", 32)), multistarts=int(c.get("multistarts", 4)),
                   max_objective_evaluations=int(c.get("max_objective_evaluations", 200)),
                   attempted_patients_per_evaluation=int(c.get("attempted_patients_per_evaluation", 2500)),
                   fit_simulation_seeds=tuple(c.get("fit_simulation_seeds", (1201, 1202, 1203))),
                   diagnostic_simulation_seeds=tuple(c.get("diagnostic_simulation_seeds", (2201, 2202, 2203))),
                   optimizer_seed=int(c.get("optimizer_seed", 20260917)))
        for k, v in overrides.items():
            if v is not None:
                setattr(base, k, v)
        return base


@dataclass
class Evaluation:
    values: dict
    unit: list
    seeds: tuple
    objective: float
    prior: float
    blocks: dict
    points: list
    seconds: float


class Objective:
    def __init__(self, points: list[TargetPoint], defs: list[ParameterDef], population: PopulationSpec, settings: FitSettings,
                 generator_baseline: dict | None = None, cache: dict | None = None):
        self.points, self.defs, self.population, self.settings = points, defs, population, settings
        self.generator_baseline = generator_baseline or {}
        self.cache = cache if cache is not None else {}
        self.counted = 0
        self.history: list[Evaluation] = []

    def _key(self, values: dict, seeds) -> str:
        return repr((tuple(sorted((k, round(v, 10)) for k, v in values.items())), tuple(seeds)))

    def evaluate(self, values: dict, seeds, count: bool = True) -> Evaluation:
        key = self._key(values, seeds)
        if key in self.cache:
            ev = self.cache[key]
            if count and ev not in self.history:   # warm start: reuse without spending budget, keep for ranking
                self.history.append(ev)
            return ev
        if count:
            if self.counted >= self.settings.max_objective_evaluations:
                raise BudgetExhausted()
            self.counted += 1
        t0 = time.time()
        spec = compile_spec(self.population, self.defs, values, mode="calibrated", generator_baseline=self.generator_baseline)
        per_point = {p.point_id: [] for p in self.points}
        for seed in seeds:
            cohort = generate_cohort(spec, n=self.settings.attempted_patients_per_evaluation, seed=int(seed), namespace="quick")
            cache: dict = {}
            for p in self.points:
                try:
                    per_point[p.point_id].append(simulate_point(cohort, p, cache))
                except Exception as ex:  # noqa: BLE001 - an operator failure makes the point infinitely far, with the reason
                    per_point[p.point_id].append((float("nan"), float("nan"), 0, f"{type(ex).__name__}: {ex}"[:120]))
        rows = []
        for p in self.points:
            res = per_point[p.point_id]
            sims = np.array([r[0] for r in res], dtype=float)
            within = np.array([r[1] for r in res], dtype=float)
            err = next((r[3] for r in res if len(r) > 3), "")
            k = len(sims)
            sim = float(np.nanmean(sims)) if np.isfinite(sims).any() else float("nan")
            between = float(np.nanstd(sims, ddof=1) / np.sqrt(k)) if k > 1 and np.isfinite(sims).sum() > 1 else 0.0
            mc_se = float(max(between, np.sqrt(np.nanmean(within ** 2) / k))) if np.isfinite(within).any() else float("nan")
            obs_se = p.obs_se if p.obs_se is not None else ((p.tolerance or 0.0) / 2.0)
            mismatch = float((p.spec.get("population") or {}).get("mismatch_allowance", 0.0) or 0.0)
            denom = obs_se ** 2 + (mc_se if np.isfinite(mc_se) else 0.0) ** 2 + mismatch ** 2
            z2 = float((sim - p.observed) ** 2 / denom) if (np.isfinite(sim) and denom > 0) else 1e6
            tol_ok = None
            if p.tolerance is not None and np.isfinite(sim):
                tol_ok = bool(abs(sim - p.observed) <= float(p.tolerance))
            rows.append({"point_id": p.point_id, "target_id": p.target_id, "block": p.block, "kind": p.kind, "observed": p.observed,
                         "simulated": sim, "mc_se": mc_se, "obs_se": p.obs_se, "residual": sim - p.observed, "z2": z2,
                         "tolerance": p.tolerance, "within_tolerance": tol_ok, "mandatory": p.mandatory, "weight": p.weight,
                         "operator_error": err})
        df = pd.DataFrame(rows)
        blocks = {}
        num = den = 0.0
        for g, gdf in df.groupby("block"):
            w = gdf["weight"].to_numpy(dtype=float)
            d_g = float(np.sum(w * gdf["z2"]) / w.sum()) if w.sum() > 0 else 0.0
            blocks[g] = {"D": d_g, "W": float(w.sum()), "points": int(len(gdf))}
            num += w.sum() * d_g
            den += w.sum()
        pri = prior_penalty(self.defs, values)
        L = (num / den if den > 0 else 0.0) + self.settings.prior_lambda * pri
        ev = Evaluation(values=dict(values), unit=to_unit(self.defs, values).tolist(), seeds=tuple(seeds), objective=float(L),
                        prior=pri, blocks=blocks, points=rows, seconds=round(time.time() - t0, 2))
        self.cache[key] = ev
        if count:
            self.history.append(ev)
        return ev


def target_conflicts(points: list[TargetPoint]) -> list[dict]:
    """Same quantity (kind, variable, statistic, horizon, view) from different study groups whose estimates
    cannot both be met within their tolerances."""
    out = []
    by_q: dict = {}
    for p in points:
        by_q.setdefault((p.kind, p.variable, p.statistic, p.horizon, p.view), []).append(p)
    for q, members in by_q.items():
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                if a.block == b.block or a.tolerance is None or b.tolerance is None:
                    continue
                if abs(a.observed - b.observed) > float(a.tolerance) + float(b.tolerance):
                    out.append({"quantity": list(map(str, q)), "targets": [a.point_id, b.point_id],
                                "difference": abs(a.observed - b.observed), "combined_tolerance": float(a.tolerance) + float(b.tolerance)})
    return out


@dataclass
class FitResult:
    status: str
    reason: str
    best_values: dict
    finalists: list = field(default_factory=list)
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    residuals: pd.DataFrame = field(default_factory=pd.DataFrame)
    identifiability: dict = field(default_factory=dict)
    conflicts: list = field(default_factory=list)
    evaluations_used: int = 0
    seconds: float = 0.0
    budget_exhausted: bool = False


def fit_parameters(points: list[TargetPoint], defs: list[ParameterDef], population: PopulationSpec, settings: FitSettings,
                   generator_baseline: dict | None = None, checkpoint: dict | None = None, on_checkpoint=None) -> FitResult:
    from scipy.optimize import minimize
    from scipy.stats import qmc

    t_start = time.time()
    free = [d for d in defs if d.status == "free"]
    start = starting_values(defs)
    if not points:
        return FitResult("insufficient_evidence", "no compatible, reviewed fit target", start)
    fit_ids = {p.target_id for p in points}
    linked = [d.name for d in free if any(t.startswith(link) for link in d.evidence_links for t in fit_ids)]
    if free and not linked:
        return FitResult("insufficient_evidence", f"no free parameter is linked (by evidence_links prefix) to a usable fit target: "
                                                  f"{[d.name for d in free]}", start)
    conflicts = target_conflicts(points)
    obj = Objective(points, defs, population, settings, generator_baseline)
    if checkpoint:
        restore_checkpoint(obj, checkpoint)
    fit_seeds = tuple(settings.fit_simulation_seeds)
    exhausted = False

    def timed_out() -> bool:
        return settings.max_seconds is not None and time.time() - t_start > settings.max_seconds

    def f(u):
        if timed_out():
            raise BudgetExhausted()
        return obj.evaluate(from_unit(defs, np.asarray(u)), fit_seeds).objective

    try:
        f(to_unit(defs, start) if free else np.array([]))
        if free:
            design = qmc.LatinHypercube(d=len(free), seed=settings.optimizer_seed).random(settings.initial_design_points)
            for u in design:
                f(u)
            ranked = sorted(obj.history, key=lambda e: e.objective)
            starts = [np.asarray(e.unit) for e in ranked[:settings.multistarts]]
            for i, u0 in enumerate(starts):
                remaining = settings.max_objective_evaluations - obj.counted
                if remaining <= 0:
                    raise BudgetExhausted()
                minimize(f, u0, method="Powell", bounds=[(0.0, 1.0)] * len(free),
                         options={"maxfev": max(1, remaining // (len(starts) - i)), "xtol": 1e-3, "ftol": 1e-4})
    except BudgetExhausted:
        exhausted = True
    if on_checkpoint is not None:
        on_checkpoint(obj)
    if timed_out():
        best = min(obj.history, key=lambda e: e.objective) if obj.history else None
        return FitResult("incomplete", "time budget ended before the acceptance procedure",
                         best.values if best else start, history=_history(obj), evaluations_used=obj.counted,
                         seconds=time.time() - t_start, budget_exhausted=True, conflicts=conflicts)

    # finalists on independent diagnostic seeds (not counted against the fit budget)
    ranked = sorted(obj.history, key=lambda e: e.objective)
    finalists, seen = [], set()
    for e in ranked:
        key = tuple(np.round(e.unit, 4))
        if key in seen:
            continue
        seen.add(key)
        diag = obj.evaluate(e.values, settings.diagnostic_simulation_seeds, count=False)
        finalists.append({"values": e.values, "fit_objective": e.objective, "diagnostic_objective": diag.objective, "evaluation": diag})
        if len(finalists) >= settings.finalists:
            break
    chosen = min(finalists, key=lambda x: x["diagnostic_objective"])
    diag = chosen["evaluation"]
    residuals = pd.DataFrame(diag.points)

    # local identifiability and boundaries
    ident = {"parameters": {}, "boundary_hits": [], "nonidentifiable": []}
    if free:
        u_best = np.asarray(to_unit(defs, chosen["values"]))
        base_L = obj.evaluate(chosen["values"], fit_seeds, count=False).objective
        for i, d in enumerate(free):
            changes = []
            for sgn in (-1, 1):
                u = u_best.copy()
                u[i] = float(np.clip(u[i] + sgn * settings.identifiability_step, 0.0, 1.0))
                if u[i] == u_best[i]:
                    continue
                changes.append(obj.evaluate(from_unit(defs, u), fit_seeds, count=False).objective - base_L)
            flat = bool(changes) and max(abs(c) for c in changes) < settings.identifiability_min_change
            at_bound = bool(u_best[i] < settings.boundary_margin or u_best[i] > 1 - settings.boundary_margin)
            ident["parameters"][d.name] = {"unit_value": float(u_best[i]), "objective_changes": changes, "flat": flat,
                                           "at_bound": at_bound, "evidence_links": d.evidence_links}
            if flat:
                ident["nonidentifiable"].append(d.name)
            if at_bound:
                ident["boundary_hits"].append(d.name)
    mandatory = residuals[residuals["mandatory"].astype(bool)]
    failed = mandatory[mandatory["within_tolerance"].isin([False]) | mandatory["within_tolerance"].isna()]
    if conflicts:
        status, reason = "failed_targets", f"{len(conflicts)} conflicting target pairs cannot be met together"
    elif len(failed):
        status, reason = "failed_targets", f"{len(failed)} mandatory fit targets outside tolerance on diagnostic seeds"
    elif ident["nonidentifiable"]:
        status, reason = "nonidentifiable", f"flat objective for {ident['nonidentifiable']}: fix at priors or add evidence"
    else:
        status, reason = "accepted_for_simulation", "all mandatory fit targets within tolerance on independent diagnostic seeds"
    if exhausted:
        reason += "; evaluation budget reached before the optimizer converged (acceptance procedure completed)"
    return FitResult(status, reason, chosen["values"],
                     finalists=[{k: v for k, v in fz.items() if k != "evaluation"} for fz in finalists],
                     history=_history(obj), residuals=residuals, identifiability=ident, conflicts=conflicts,
                     evaluations_used=obj.counted, seconds=round(time.time() - t_start, 1), budget_exhausted=exhausted)


def checkpoint_payload(obj: Objective, roles_fingerprint: str) -> dict:
    """Serializable evaluations for a warm start; bound to the fit/holdout roles it was produced under."""
    return {"roles_fingerprint": roles_fingerprint,
            "evaluations": [{"values": e.values, "unit": e.unit, "seeds": list(e.seeds), "objective": e.objective, "prior": e.prior,
                             "blocks": e.blocks, "points": e.points, "seconds": e.seconds} for e in obj.cache.values()]}


def restore_checkpoint(obj: Objective, payload: dict, roles_fingerprint: str | None = None) -> None:
    if roles_fingerprint is not None and payload.get("roles_fingerprint") != roles_fingerprint:
        raise ValueError("checkpoint was produced under different fit/holdout roles; a warm start cannot change frozen roles")
    for e in payload.get("evaluations", []):
        ev = Evaluation(values=e["values"], unit=e["unit"], seeds=tuple(e["seeds"]), objective=e["objective"], prior=e["prior"],
                        blocks=e["blocks"], points=e["points"], seconds=e["seconds"])
        obj.cache[obj._key(ev.values, ev.seeds)] = ev


def _history(obj: Objective) -> pd.DataFrame:
    return pd.DataFrame([{"evaluation": i, "objective": e.objective, "prior": e.prior, "seconds": e.seconds,
                          **{f"param:{k}": v for k, v in e.values.items()}} for i, e in enumerate(obj.history)])
