"""Immutable calibration bundles and calibrated cohort storage (design section 7).

Layout (artifact store, private by default)::

    calibration/<bundle_id>/manifest.json, population.json, evidence/targets.parquet, evidence/curves.parquet,
        profile/observed_summary.parquet, parameters/specification.json, parameters/fits.parquet,
        parameters/uncertainty_sets.parquet, diagnostics/target_residuals.parquet,
        diagnostics/holdout_residuals.parquet, diagnostics/identifiability.json, diagnostics/fit_history.parquet,
        report.md
    scenarios/calibrated/<bundle_id>/<quick|full>/<replicate_id>/  (five tables, truth/, diagnostics/)

A bundle is written once; :func:`save_bundle` refuses to overwrite and :func:`load_bundle` verifies
checksums and generator/endpoint compatibility.
"""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from kairos.calibration.schema import FIT_STATUSES, VALIDATION_STATUSES, fingerprint

CONTAINER = "calibration"
BUNDLE_SCHEMA_VERSION = "1"
COHORT_LABEL = "synthetic, evidence-calibrated for specified targets; clinical predictive validity unestablished"


@dataclass
class CalibrationBundle:
    bundle_id: str
    fit_status: str
    fit_reason: str
    structural_status: str
    independent_validation_status: str
    population: dict
    parameter_definitions: list
    fitted_values: dict
    input_fingerprints: dict
    versions: dict
    seeds: dict
    targets: list = field(default_factory=list)
    curves: list = field(default_factory=list)
    excluded_targets: list = field(default_factory=list)
    profile_summary: list = field(default_factory=list)
    finalists: list = field(default_factory=list)
    residuals: pd.DataFrame = field(default_factory=pd.DataFrame)
    holdout_residuals: pd.DataFrame = field(default_factory=pd.DataFrame)
    fit_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    identifiability: dict = field(default_factory=dict)
    conflicts: list = field(default_factory=list)
    uncertainty_sets: list = field(default_factory=list)
    generator_baseline: dict = field(default_factory=dict)
    scope_limitations: list = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    schema_version: str = BUNDLE_SCHEMA_VERSION

    def __post_init__(self):
        if self.fit_status not in FIT_STATUSES:
            raise ValueError(f"fit_status must be one of {FIT_STATUSES}")
        if self.independent_validation_status not in VALIDATION_STATUSES:
            raise ValueError(f"independent_validation_status must be one of {VALIDATION_STATUSES}")

    @property
    def accepted(self) -> bool:
        return self.fit_status == "accepted_for_simulation"

    @property
    def allowed_purposes(self) -> list[str]:
        if self.accepted:
            return ["calibrated_research_simulation", "sensitivity_stress", "standardization_training_scaffold"]
        return []

    def parameter_sets(self) -> list[dict]:
        return [{"parameter_set_id": "fit", "type": "accepted_fit" if self.accepted else "unaccepted_fit",
                 "values": self.fitted_values}] + list(self.uncertainty_sets)

    def manifest(self) -> dict:
        free = [d for d in self.parameter_definitions if d["status"] == "free"]
        return {"bundle_id": self.bundle_id, "schema_version": self.schema_version, "created_at": self.created_at,
                "fit_status": self.fit_status, "fit_reason": self.fit_reason, "structural_status": self.structural_status,
                "independent_validation_status": self.independent_validation_status,
                "external_claim_allowed": self.accepted and self.independent_validation_status == "passed",
                "allowed_purposes": self.allowed_purposes, "population_id": self.population.get("population_id"),
                "input_fingerprints": self.input_fingerprints, "versions": self.versions, "seeds": self.seeds,
                "fitted_values": self.fitted_values,
                "parameters": {"free": [d["name"] for d in free], "fixed": [d["name"] for d in self.parameter_definitions if d["status"] == "fixed"]},
                "targets": {"fit": sum(1 for t in self.targets if t["role"] == "fit"),
                            "holdout": sum(1 for t in self.targets if t["role"] == "holdout"),
                            "sensitivity": sum(1 for t in self.targets if t["role"] == "sensitivity"),
                            "excluded": len(self.excluded_targets)},
                "identifiability": {k: self.identifiability.get(k) for k in ("nonidentifiable", "boundary_hits")},
                "conflicts": len(self.conflicts), "parameter_sets": [s["parameter_set_id"] for s in self.parameter_sets()],
                "scope_limitations": self.scope_limitations,
                "label": "calibration bundle for synthetic simulation; not a clinical model and not externally validated"
                         if self.independent_validation_status != "passed" else
                         "calibration bundle for synthetic simulation; held-out targets passed (synthetic evidence only)"}


def make_bundle_id(fingerprints: dict) -> str:
    return "cal-" + fingerprint(fingerprints)[:12]


def _put_df(store, path: str, df: pd.DataFrame) -> str:
    df = df.copy()
    for c in df.columns:
        if df[c].map(lambda v: isinstance(v, (dict, list))).any():
            df[c] = df[c].map(lambda v: json.dumps(v, default=str) if isinstance(v, (dict, list)) else v)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    data = buf.getvalue()
    store.put_bytes(CONTAINER, path, data)
    return hashlib.sha256(data).hexdigest()


def _put_json(store, path: str, obj) -> str:
    data = json.dumps(obj, indent=2, default=str).encode()
    store.put_bytes(CONTAINER, path, data)
    return hashlib.sha256(data).hexdigest()


def report_markdown(b: CalibrationBundle) -> str:
    m = b.manifest()
    lines = [f"# Calibration bundle {b.bundle_id}", "",
             f"- Fit status: **{b.fit_status}** ({b.fit_reason})",
             f"- Structural status: {b.structural_status}",
             f"- Independent validation: {b.independent_validation_status}",
             f"- Population: {b.population.get('population_id')} ({b.population.get('preset')})",
             f"- Allowed purposes: {', '.join(m['allowed_purposes']) or 'none (not accepted)'}",
             f"- Targets: {m['targets']}", "",
             "Four separate questions: structure, fit-target resemblance, independent-target generalization, training usefulness.",
             "Fit targets reproduced here are calibration evidence, not validation.", "", "## Fitted and fixed parameters", "",
             "| parameter | value | status | evidence |", "|---|---|---|---|"]
    for d in b.parameter_definitions:
        lines.append(f"| {d['name']} | {b.fitted_values.get(d['name'])} | {d['status']} | {', '.join(d.get('evidence_links') or []) or 'none'} |")
    if len(b.residuals):
        lines += ["", "## Fit-target residuals (diagnostic seeds)", "", "| target | observed | simulated | tolerance | within |",
                  "|---|---|---|---|---|"]
        for r in b.residuals.itertuples(index=False):
            lines.append(f"| {r.point_id} | {r.observed:.4g} | {r.simulated:.4g} | {r.tolerance} | {r.within_tolerance} |")
    if len(b.holdout_residuals):
        lines += ["", "## Held-out targets", "", "| target | observed | simulated | tolerance | within |", "|---|---|---|---|---|"]
        for r in b.holdout_residuals.itertuples(index=False):
            lines.append(f"| {r.point_id} | {r.observed:.4g} | {r.simulated:.4g} | {r.tolerance} | {r.within_tolerance} |")
    if b.excluded_targets:
        lines += ["", "## Excluded targets", ""] + [f"- {t['target_id']}: {t['compatibility'].get('reason')}" for t in b.excluded_targets[:200]]
    if b.identifiability:
        lines += ["", "## Identifiability", "", f"- Nonidentifiable: {b.identifiability.get('nonidentifiable')}",
                  f"- Boundary hits: {b.identifiability.get('boundary_hits')}"]
    lines += ["", "## Scope limitations", ""] + [f"- {s}" for s in b.scope_limitations]
    return "\n".join(lines) + "\n"


def save_bundle(store, b: CalibrationBundle) -> dict:
    base = b.bundle_id
    if store.exists(CONTAINER, f"{base}/manifest.json"):
        raise FileExistsError(f"calibration bundle {base} exists; bundles are immutable")
    sums = {
        "population.json": _put_json(store, f"{base}/population.json", b.population),
        "evidence/targets.parquet": _put_df(store, f"{base}/evidence/targets.parquet", pd.DataFrame(b.targets + b.excluded_targets)),
        "evidence/curves.parquet": _put_df(store, f"{base}/evidence/curves.parquet", pd.DataFrame(b.curves)),
        "profile/observed_summary.parquet": _put_df(store, f"{base}/profile/observed_summary.parquet", pd.DataFrame(b.profile_summary)),
        "parameters/specification.json": _put_json(store, f"{base}/parameters/specification.json",
                                                   {"parameters": b.parameter_definitions, "generator_baseline": b.generator_baseline}),
        "parameters/fits.parquet": _put_df(store, f"{base}/parameters/fits.parquet",
                                           pd.DataFrame([{"parameter_set_id": "fit", **b.fitted_values}] +
                                                        [{"parameter_set_id": f"finalist_{i}", **fz["values"]} for i, fz in enumerate(b.finalists)])),
        "parameters/uncertainty_sets.parquet": _put_df(store, f"{base}/parameters/uncertainty_sets.parquet",
                                                       pd.DataFrame([{"parameter_set_id": s["parameter_set_id"], "type": s["type"],
                                                                      "status": s.get("status"), **s["values"]} for s in b.uncertainty_sets])),
        "diagnostics/target_residuals.parquet": _put_df(store, f"{base}/diagnostics/target_residuals.parquet", b.residuals),
        "diagnostics/holdout_residuals.parquet": _put_df(store, f"{base}/diagnostics/holdout_residuals.parquet", b.holdout_residuals),
        "diagnostics/fit_history.parquet": _put_df(store, f"{base}/diagnostics/fit_history.parquet", b.fit_history),
        "diagnostics/identifiability.json": _put_json(store, f"{base}/diagnostics/identifiability.json",
                                                      {**b.identifiability, "conflicts": b.conflicts}),
    }
    report = report_markdown(b).encode()
    store.put_bytes(CONTAINER, f"{base}/report.md", report)
    sums["report.md"] = hashlib.sha256(report).hexdigest()
    manifest = {**b.manifest(), "checksums": sums}
    _put_json(store, f"{base}/manifest.json", manifest)
    return manifest


def load_bundle(store, bundle_id: str) -> CalibrationBundle:
    from kairos.simulation.generators import ENDPOINT_VERSION, GENERATOR_VERSION

    base = bundle_id
    manifest = json.loads(store.get_bytes(CONTAINER, f"{base}/manifest.json"))
    for path, digest in manifest["checksums"].items():
        if hashlib.sha256(store.get_bytes(CONTAINER, f"{base}/{path}")).hexdigest() != digest:
            raise ValueError(f"calibration bundle {bundle_id}: checksum mismatch for {path}")
    if manifest["versions"].get("endpoint_version") != ENDPOINT_VERSION:
        raise ValueError(f"bundle endpoint version {manifest['versions'].get('endpoint_version')} differs from {ENDPOINT_VERSION}")
    if str(manifest["versions"].get("generator_version", "")).split(".")[0] != GENERATOR_VERSION.split(".")[0]:
        raise ValueError(f"bundle generator version {manifest['versions'].get('generator_version')} is incompatible with {GENERATOR_VERSION}")

    def df(path):
        return pd.read_parquet(io.BytesIO(store.get_bytes(CONTAINER, f"{base}/{path}")))

    spec = json.loads(store.get_bytes(CONTAINER, f"{base}/parameters/specification.json"))
    sets = df("parameters/uncertainty_sets.parquet")
    names = [d["name"] for d in spec["parameters"]]
    uncertainty = [{"parameter_set_id": r["parameter_set_id"], "type": r["type"], "status": r.get("status"),
                    "values": {n: float(r[n]) for n in names if n in r}} for _, r in sets.iterrows()]
    targets = df("evidence/targets.parquet").to_dict(orient="records")
    for t in targets:
        for k in ("uncertainty", "population", "time", "source", "compatibility", "acceptance", "overlapping_groups"):
            if isinstance(t.get(k), str):
                t[k] = json.loads(t[k])
    ident = json.loads(store.get_bytes(CONTAINER, f"{base}/diagnostics/identifiability.json"))
    return CalibrationBundle(
        bundle_id=manifest["bundle_id"], fit_status=manifest["fit_status"], fit_reason=manifest["fit_reason"],
        structural_status=manifest["structural_status"], independent_validation_status=manifest["independent_validation_status"],
        population=json.loads(store.get_bytes(CONTAINER, f"{base}/population.json")), parameter_definitions=spec["parameters"],
        fitted_values=manifest["fitted_values"], input_fingerprints=manifest["input_fingerprints"], versions=manifest["versions"],
        seeds=manifest["seeds"], targets=[t for t in targets if t["role"] != "excluded"],
        excluded_targets=[t for t in targets if t["role"] == "excluded"], residuals=df("diagnostics/target_residuals.parquet"),
        holdout_residuals=df("diagnostics/holdout_residuals.parquet"), fit_history=df("diagnostics/fit_history.parquet"),
        identifiability={k: v for k, v in ident.items() if k != "conflicts"}, conflicts=ident.get("conflicts", []),
        uncertainty_sets=uncertainty, generator_baseline=spec.get("generator_baseline", {}),
        scope_limitations=manifest.get("scope_limitations", []), created_at=manifest["created_at"],
        curves=df("evidence/curves.parquet").to_dict(orient="records"))


def calibrated_prefix(bundle_id: str, namespace: str, replicate_id: str) -> str:
    """Storage prefix for calibrated cohorts (never a legacy scenario prefix)."""
    if namespace not in ("quick", "full"):
        raise ValueError("namespace must be quick or full")
    return f"calibrated/{bundle_id}/{namespace}/{replicate_id}"
