"""Shared fixtures. Environment is pinned before any settings object is created."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("KAIROS_SQLITE_PATH", ":memory:")
os.environ.setdefault("KAIROS_LOCAL_ARTIFACTS_DIR", tempfile.mkdtemp(prefix="kairos-test-"))
os.environ.setdefault("KAIROS_OPENAI_ENDPOINT", "")
os.environ.setdefault("KAIROS_STORAGE_ACCOUNT_URL", "")
os.environ.setdefault("KAIROS_PG_HOST", "")
os.environ.setdefault("APPLICATIONINSIGHTS_CONNECTION_STRING", "")

from kairos.io.config import reset_settings_cache  # noqa: E402
from kairos.modelling.modules import load_model_config  # noqa: E402
from kairos.modelling.train import train_bundle  # noqa: E402
from kairos.simulation.generators import generate_cohort  # noqa: E402
from kairos.simulation.scenarios import get_scenario  # noqa: E402

reset_settings_cache()


@pytest.fixture(scope="session")
def model_cfg():
    return load_model_config()


@pytest.fixture(scope="session")
def small_cohort():
    # quick namespace: relaxed support gates for a 350-patient fixture (never evidence)
    return generate_cohort(get_scenario("gradual_stenotic"), n=350, seed=7, namespace="quick")


@pytest.fixture(scope="session")
def small_bundle(small_cohort, model_cfg):
    return train_bundle(small_cohort, model_cfg, "core_plus_both", "0.1.0+test")
