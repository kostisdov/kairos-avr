"""Artefact store: Azure Blob in Azure, the local filesystem elsewhere.

Containers mirror config/app.yaml: scenarios, models, metrics, figures, reference,
aggregates, raw. Nothing under ``raw`` is ever written by code in this repository; it exists
for the owner's private staging only (design rule 0.2).
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from kairos.io.config import Settings, azure_credential, get_settings

CONTAINERS = ("scenarios", "models", "metrics", "figures", "reference", "aggregates", "raw", "calibration")


class ArtifactStore:
    def put_bytes(self, container: str, path: str, data: bytes) -> None:
        raise NotImplementedError

    def get_bytes(self, container: str, path: str) -> bytes:
        raise NotImplementedError

    def exists(self, container: str, path: str) -> bool:
        raise NotImplementedError

    def list(self, container: str, prefix: str = "") -> list[str]:
        raise NotImplementedError

    def delete(self, container: str, path: str) -> None:
        raise NotImplementedError

    # convenience -----------------------------------------------------------------
    def put_text(self, container: str, path: str, text: str) -> None:
        self.put_bytes(container, path, text.encode("utf-8"))

    def get_text(self, container: str, path: str) -> str:
        return self.get_bytes(container, path).decode("utf-8")

    def put_json(self, container: str, path: str, obj) -> None:
        self.put_text(container, path, json.dumps(obj, indent=2, default=str))

    def get_json(self, container: str, path: str):
        return json.loads(self.get_text(container, path))

    def put_file(self, container: str, path: str, local: Path) -> None:
        self.put_bytes(container, path, Path(local).read_bytes())

    def get_file(self, container: str, path: str, local: Path) -> Path:
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(self.get_bytes(container, path))
        return local

    def put_parquet(self, container: str, path: str, df: pd.DataFrame) -> None:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        self.put_bytes(container, path, buf.getvalue())

    def get_parquet(self, container: str, path: str) -> pd.DataFrame:
        return pd.read_parquet(io.BytesIO(self.get_bytes(container, path)))

    def put_csv(self, container: str, path: str, df: pd.DataFrame) -> None:
        self.put_text(container, path, df.to_csv(index=False))


class LocalStore(ArtifactStore):
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, container: str, path: str) -> Path:
        if container not in CONTAINERS:
            raise ValueError(f"unknown container {container!r}")
        return self.root / container / path

    def put_bytes(self, container, path, data):
        p = self._p(container, path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def get_bytes(self, container, path):
        return self._p(container, path).read_bytes()

    def exists(self, container, path):
        return self._p(container, path).exists()

    def list(self, container, prefix=""):
        base = self.root / container
        if not base.exists():
            return []
        out = []
        for p in base.rglob("*"):
            if p.is_file():
                rel = p.relative_to(base).as_posix()
                if rel.startswith(prefix):
                    out.append(rel)
        return sorted(out)

    def delete(self, container, path):
        p = self._p(container, path)
        if p.exists():
            p.unlink()


class BlobStore(ArtifactStore):
    def __init__(self, account_url: str, credential=None):
        from azure.storage.blob import BlobServiceClient

        self.client = BlobServiceClient(account_url=account_url,
                                        credential=credential or azure_credential())

    def _blob(self, container: str, path: str):
        if container not in CONTAINERS:
            raise ValueError(f"unknown container {container!r}")
        return self.client.get_blob_client(container=container, blob=path)

    def put_bytes(self, container, path, data):
        self._blob(container, path).upload_blob(data, overwrite=True)

    def get_bytes(self, container, path):
        return self._blob(container, path).download_blob().readall()

    def exists(self, container, path):
        return self._blob(container, path).exists()

    def list(self, container, prefix=""):
        cc = self.client.get_container_client(container)
        return sorted(b.name for b in cc.list_blobs(name_starts_with=prefix or None))

    def delete(self, container, path):
        b = self._blob(container, path)
        if b.exists():
            b.delete_blob()


def get_store(settings: Settings | None = None) -> ArtifactStore:
    s = settings or get_settings()
    if s.storage_account_url:
        return BlobStore(s.storage_account_url, azure_credential(s))
    return LocalStore(s.artifacts_dir)


def copy_tree_to_store(store: ArtifactStore, container: str, files: Iterable[Path],
                       base: Path, prefix: str = "") -> list[str]:
    """Upload a set of files (already vetted as committable) under a prefix."""
    written = []
    for f in files:
        rel = Path(f).relative_to(base).as_posix()
        dest = f"{prefix}{rel}" if prefix else rel
        store.put_file(container, dest, Path(f))
        written.append(dest)
    return written
