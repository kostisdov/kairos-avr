"""Run-scoped model store and active-bundle pointers (detailed design WP-D3, fixes F26).

Layout in the ``models`` container::

    runs/{run_id}/{family}/bundle.joblib, card.json, demo_patient.json
    {namespace}/{family}/active.json      {run_id, family, path, sha256, previous, published_at}

``train`` writes run artefacts only; ``publish`` moves the active pointer and keeps the previous pointer so
``publish --rollback`` can restore it. The predict service loads every family that has an active pointer.
"""
from __future__ import annotations

import hashlib
import tempfile
from datetime import UTC, datetime
from pathlib import Path

FAMILIES = ("cox", "gradient_boosting")


def run_path(run_id: str, family: str, name: str = "bundle.joblib") -> str:
    return f"runs/{run_id}/{family}/{name}"


def active_path(namespace: str, family: str) -> str:
    return f"{namespace}/{family}/active.json"


def save_run_bundle(store, run_id: str, family: str, local_bundle: Path, card: dict) -> dict:
    """Store a trained bundle under its run; returns the would-be pointer (not published)."""
    data = Path(local_bundle).read_bytes()
    path = run_path(run_id, family)
    store.put_bytes("models", path, data)
    store.put_json("models", run_path(run_id, family, "card.json"), card)
    return {"run_id": run_id, "family": family, "path": path, "sha256": hashlib.sha256(data).hexdigest()}


def read_active(store, namespace: str, family: str) -> dict | None:
    p = active_path(namespace, family)
    return store.get_json("models", p) if store.exists("models", p) else None


def active_families(store, namespace: str) -> list[str]:
    return [f for f in FAMILIES if read_active(store, namespace, f) is not None]


def publish(store, namespace: str, family: str, run_id: str) -> dict:
    """Point ``{namespace}/{family}/active.json`` at a finished run; the old pointer becomes ``previous``."""
    path = run_path(run_id, family)
    if not store.exists("models", path):
        raise FileNotFoundError(f"models/{path} not found; train the run first")
    digest = hashlib.sha256(store.get_bytes("models", path)).hexdigest()
    old = read_active(store, namespace, family)
    if old is not None:
        old = {k: v for k, v in old.items() if k != "previous"}
    pointer = {"run_id": run_id, "family": family, "path": path, "sha256": digest, "previous": old,
               "published_at": datetime.now(UTC).isoformat()}
    store.put_json("models", active_path(namespace, family), pointer)
    return pointer


def rollback(store, namespace: str, family: str) -> dict:
    cur = read_active(store, namespace, family)
    if cur is None or not cur.get("previous"):
        raise LookupError(f"no previous pointer to roll back to for {namespace}/{family}")
    prev = cur["previous"]
    pointer = {**prev, "previous": {k: v for k, v in cur.items() if k != "previous"},
               "published_at": datetime.now(UTC).isoformat(), "rolled_back": True}
    store.put_json("models", active_path(namespace, family), pointer)
    return pointer


def load_active_bundle(store, namespace: str, family: str):
    """Load the active bundle for a family, verifying its checksum; None when nothing is published."""
    from kairos.modelling.predictor import ModelBundle

    ptr = read_active(store, namespace, family)
    if ptr is None:
        return None
    data = store.get_bytes("models", ptr["path"])
    if hashlib.sha256(data).hexdigest() != ptr["sha256"]:
        raise ValueError(f"checksum mismatch for models/{ptr['path']}")
    local = Path(tempfile.gettempdir()) / f"kairos_active_{namespace}_{family}.joblib"
    local.write_bytes(data)
    return ModelBundle.load(local)
