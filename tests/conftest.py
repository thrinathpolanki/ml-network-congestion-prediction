import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config  # noqa: E402
from src.data.generator import generate_dataset  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return get_config()


@pytest.fixture(scope="session")
def small_df(cfg):
    """3 sites, 7 days, clean (no injected issues) - fast."""
    return generate_dataset(cfg, n_sites=3, start="2026-01-05", end="2026-01-12", seed=1, inject_issues=False)
