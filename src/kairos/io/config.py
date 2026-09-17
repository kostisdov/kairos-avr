"""Settings for every KAIROS service.

All values come from environment variables (see config/app.yaml for the contract). The
real-notes switch ``ALLOW_REAL_NOTES_TO_LLM`` is read here and only here; no request can
change it (design rule 0.3).
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from kairos import __version__

REPO_ROOT = Path(os.environ.get("KAIROS_REPO_ROOT", Path(__file__).resolve().parents[3]))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore", case_sensitive=False)

    kairos_env: str = Field("local", alias="KAIROS_ENV")
    allow_real_notes_to_llm: bool = Field(True, alias="ALLOW_REAL_NOTES_TO_LLM")

    storage_account_url: str = Field("", alias="KAIROS_STORAGE_ACCOUNT_URL")
    local_artifacts_dir: str = Field("artifacts", alias="KAIROS_LOCAL_ARTIFACTS_DIR")

    pg_host: str = Field("", alias="KAIROS_PG_HOST")
    pg_db: str = Field("kairos", alias="KAIROS_PG_DB")
    pg_user: str = Field("", alias="KAIROS_PG_USER")
    pg_auth: Literal["entra", "password"] = Field("entra", alias="KAIROS_PG_AUTH")
    pg_password: str = Field("", alias="KAIROS_PG_PASSWORD")
    sqlite_path: str = Field("", alias="KAIROS_SQLITE_PATH")

    openai_endpoint: str = Field("", alias="KAIROS_OPENAI_ENDPOINT")
    # dated version (classic deployments path) or "v1" (version-free /openai/v1/ path)
    openai_api_version: str = Field("2025-04-01-preview", alias="KAIROS_OPENAI_API_VERSION")
    openai_extract_deployment: str = Field("kairos-extract", alias="KAIROS_OPENAI_EXTRACT_DEPLOYMENT")
    openai_prescreen_deployment: str = Field("", alias="KAIROS_OPENAI_PRESCREEN_DEPLOYMENT")
    openai_adjudicate_deployment: str = Field("", alias="KAIROS_OPENAI_ADJUDICATE_DEPLOYMENT")
    openai_summary_deployment: str = Field("", alias="KAIROS_OPENAI_SUMMARY_DEPLOYMENT")
    # reasoning effort per role; an empty value marks a non-reasoning model (temperature 0 is sent)
    openai_extract_reasoning_effort: str = Field("low", alias="KAIROS_OPENAI_EXTRACT_REASONING_EFFORT")
    openai_prescreen_reasoning_effort: str = Field("", alias="KAIROS_OPENAI_PRESCREEN_REASONING_EFFORT")
    openai_adjudicate_reasoning_effort: str = Field("medium", alias="KAIROS_OPENAI_ADJUDICATE_REASONING_EFFORT")
    openai_summary_reasoning_effort: str = Field("low", alias="KAIROS_OPENAI_SUMMARY_REASONING_EFFORT")
    summary_max_completion_tokens: int = Field(4000, alias="KAIROS_SUMMARY_MAX_COMPLETION_TOKENS", ge=256, le=32000)
    openai_api_key: str = Field("", alias="KAIROS_OPENAI_API_KEY")  # local development only

    keyvault_url: str = Field("", alias="KAIROS_KEYVAULT_URL")
    extract_url: str = Field("http://127.0.0.1:8001", alias="KAIROS_EXTRACT_URL")
    predict_url: str = Field("http://127.0.0.1:8002", alias="KAIROS_PREDICT_URL")
    model_blob_prefix: str = Field("latest", alias="KAIROS_MODEL_BLOB_PREFIX")
    git_sha: str = Field("", alias="KAIROS_GIT_SHA")
    azure_client_id: str = Field("", alias="AZURE_CLIENT_ID")
    appinsights_connection_string: str = Field("", alias="APPLICATIONINSIGHTS_CONNECTION_STRING")

    @property
    def repo_root(self) -> Path:
        return REPO_ROOT

    @property
    def reference_dir(self) -> Path:
        return REPO_ROOT / "data" / "reference"

    @property
    def config_dir(self) -> Path:
        return REPO_ROOT / "config"

    @property
    def artifacts_dir(self) -> Path:
        p = Path(self.local_artifacts_dir)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def llm_configured(self) -> bool:
        return bool(self.openai_endpoint and self.openai_extract_deployment)

    @property
    def summary_llm_configured(self) -> bool:
        return bool(self.openai_endpoint and self.openai_summary_deployment)

    def model_version(self) -> str:
        sha = self.git_sha if self.git_sha and self.git_sha != "nogit" else ""
        return f"{__version__}+{sha or revision_tag()}"


def detect_git_sha() -> str:
    sha = os.environ.get("KAIROS_GIT_SHA", "")
    if sha and sha != "nogit":  # the image build arg defaults to the placeholder "nogit"
        return sha
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:  # noqa: BLE001 - git is optional on the build machine
        pass
    return "nogit"


@lru_cache(maxsize=1)
def source_tree_hash() -> str:
    """sha256 over the relative path and bytes of the package source and the configuration.

    Used as the code revision when no git revision exists, so "nogit" is never mistaken for a
    unique version. Only ``src/kairos`` and ``config`` enter, which every image contains."""
    h = hashlib.sha256()
    files = sorted([*(REPO_ROOT / "src" / "kairos").rglob("*.py"), *(REPO_ROOT / "config").glob("*.yaml")])
    for p in files:
        h.update(p.relative_to(REPO_ROOT).as_posix().encode())
        h.update(b"\0")
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
        h.update(b"\0")
    return h.hexdigest()[:12]


def code_revision() -> dict:
    sha = detect_git_sha()
    if sha != "nogit":
        return {"kind": "git", "value": sha}
    return {"kind": "source_tree", "value": source_tree_hash()}


def revision_tag() -> str:
    rev = code_revision()
    return rev["value"] if rev["kind"] == "git" else f"src-{rev['value']}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()


def azure_credential(settings: Settings | None = None):
    """DefaultAzureCredential, preferring the user-assigned managed identity in Azure."""
    from azure.identity import DefaultAzureCredential

    s = settings or get_settings()
    kwargs = {}
    if s.azure_client_id:
        kwargs["managed_identity_client_id"] = s.azure_client_id
    return DefaultAzureCredential(exclude_interactive_browser_credential=True, **kwargs)
