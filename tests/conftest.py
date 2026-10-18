"""Shared fixtures: a session-wide toy-data directory and isolated settings objects."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from irbench.data.toy import ToySpec, generate_toy_dataset
from irbench.settings import Settings, get_settings
from irbench.utils import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
SMALL_TOY = ToySpec(
    landmarks=4,
    queries_per_landmark=2,
    easy_per_landmark=4,
    hard_per_landmark=2,
    junk_per_landmark=1,
    negatives=12,
)


@pytest.fixture(autouse=True)
def _seed() -> Iterator[None]:
    seed_everything(0)
    get_settings.cache_clear()
    yield


@pytest.fixture(scope="session")
def toy_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """DATA_DIR containing toy_a and toy_b (small) in the official on-disk formats."""
    root = tmp_path_factory.mktemp("data")
    for name in ("toy_a", "toy_b"):
        generate_toy_dataset(name, root, seed=0, spec=SMALL_TOY)
    return root


def make_settings(
    data_dir: Path, results_dir: Path, config_dir: Path = CONFIG_DIR, **extra: object
) -> Settings:
    """Settings that never touch /data, /results or network services."""
    values: dict[str, object] = {
        "data_dir": data_dir,
        "results_dir": results_dir,
        "config_dir": config_dir,
        "readme_path": results_dir / "README.md",
        "qdrant_url": ":memory:",
        "redis_url": "redis://localhost:1/0",
        "auto_download": False,
        "device": "cpu",
        "batch_size": 4,
        "log_level": "WARNING",
    }
    values.update(extra)
    return Settings(**values)


@pytest.fixture
def unit_vectors() -> np.ndarray:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 32)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)
