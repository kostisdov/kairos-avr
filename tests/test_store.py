"""Run-scoped model store: save, publish, rollback, checksum verification (WP-D3)."""
from __future__ import annotations

import hashlib

import pytest

from kairos.io.storage import LocalStore
from kairos.modelling import store as ms


@pytest.fixture()
def store(tmp_path):
    return LocalStore(tmp_path / "artifacts")


def _bundle(tmp_path, name: str, payload: bytes):
    p = tmp_path / name
    p.write_bytes(payload)
    return p


def test_save_run_bundle_writes_bundle_and_card(store, tmp_path):
    ptr = ms.save_run_bundle(store, "r1", "cox", _bundle(tmp_path, "a.joblib", b"bundle-one"), {"family": "cox"})
    assert ptr["path"] == "runs/r1/cox/bundle.joblib"
    assert ptr["sha256"] == hashlib.sha256(b"bundle-one").hexdigest()
    assert store.get_bytes("models", "runs/r1/cox/bundle.joblib") == b"bundle-one"
    assert store.get_json("models", "runs/r1/cox/card.json") == {"family": "cox"}
    assert ms.read_active(store, "full", "cox") is None   # saving never publishes
    assert ms.active_families(store, "full") == []


def test_publish_keeps_previous_and_rollback_restores(store, tmp_path):
    ms.save_run_bundle(store, "r1", "cox", _bundle(tmp_path, "a.joblib", b"one"), {})
    ms.save_run_bundle(store, "r2", "cox", _bundle(tmp_path, "b.joblib", b"two"), {})
    first = ms.publish(store, "full", "cox", "r1")
    assert first["previous"] is None
    second = ms.publish(store, "full", "cox", "r2")
    assert second["run_id"] == "r2"
    assert second["previous"]["run_id"] == "r1"
    assert "previous" not in second["previous"]
    assert ms.active_families(store, "full") == ["cox"]
    assert ms.read_active(store, "quick", "cox") is None

    back = ms.rollback(store, "full", "cox")
    assert back["run_id"] == "r1" and back["rolled_back"] is True
    assert back["previous"]["run_id"] == "r2"
    assert ms.read_active(store, "full", "cox")["run_id"] == "r1"


def test_rollback_without_previous_raises(store, tmp_path):
    with pytest.raises(LookupError):
        ms.rollback(store, "full", "cox")
    ms.save_run_bundle(store, "r1", "cox", _bundle(tmp_path, "a.joblib", b"one"), {})
    ms.publish(store, "full", "cox", "r1")
    with pytest.raises(LookupError):
        ms.rollback(store, "full", "cox")


def test_publish_missing_run_raises(store):
    with pytest.raises(FileNotFoundError):
        ms.publish(store, "full", "gradient_boosting", "nope")


def test_load_active_bundle_checks_checksum(store, tmp_path, monkeypatch):
    from kairos.modelling import predictor

    monkeypatch.setattr(predictor.ModelBundle, "load", classmethod(lambda cls, p: ("loaded", p.read_bytes())))
    assert ms.load_active_bundle(store, "full", "cox") is None
    ms.save_run_bundle(store, "r1", "cox", _bundle(tmp_path, "a.joblib", b"good"), {})
    ms.publish(store, "full", "cox", "r1")
    assert ms.load_active_bundle(store, "full", "cox") == ("loaded", b"good")

    store.put_bytes("models", "runs/r1/cox/bundle.joblib", b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        ms.load_active_bundle(store, "full", "cox")
