"""Repository root resolution.

The data-build scripts originally hard-coded the absolute path of the machine they
were written on, which made ``make quick`` and every build script fail anywhere
else. The root is resolved from this file's location and can be overridden with
``KAIROS_REPO_ROOT``, matching the contract already used by ``kairos.io.config``.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = ["repo_root", "REPO_ROOT"]


def repo_root() -> Path:
    """Absolute path of the repository root."""
    override = os.environ.get("KAIROS_REPO_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[2]


REPO_ROOT = repo_root()
