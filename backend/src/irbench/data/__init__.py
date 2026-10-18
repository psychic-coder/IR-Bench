"""Dataset registry: download/generate, parse, verify and (optionally) subsample."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from irbench.data.base import Protocol, Query, RetrievalDataset, select_subset
from irbench.data.download import DatasetUnavailableError, ensure_dataset_files
from irbench.data.oxford_paris import load_oxford_paris
from irbench.data.toy import TOY_DATASETS, generate_toy_dataset

HELDOUT = {"oxford5k": "paris6k", "paris6k": "oxford5k", "toy_a": "toy_b", "toy_b": "toy_a"}
DISPLAY_NAMES = {"oxford5k": "ROxford5k", "paris6k": "RParis6k", "toy_a": "Toy-A", "toy_b": "Toy-B"}


def _load_real(name: str, data_dir: Path, auto_download: bool) -> RetrievalDataset:
    ensure_dataset_files(name, data_dir, auto_download=auto_download)
    gnd = data_dir / "revisitop" / ("gnd_roxford5k.pkl" if name == "oxford5k" else "gnd_rparis6k.pkl")
    return load_oxford_paris(name, data_dir / name, gnd, HELDOUT[name])


def _load_toy(name: str, data_dir: Path, auto_download: bool, seed: int = 0) -> RetrievalDataset:
    root = generate_toy_dataset(name, data_dir, seed=seed)
    return load_oxford_paris(name, root, data_dir / "revisitop" / f"gnd_r{name}.pkl", HELDOUT[name])


LOADERS: dict[str, Callable[[str, Path, bool], RetrievalDataset]] = {
    "oxford5k": _load_real,
    "paris6k": _load_real,
    **{n: _load_toy for n in TOY_DATASETS},
}


def available_datasets() -> list[str]:
    return sorted(LOADERS)


def load_dataset(
    name: str,
    data_dir: Path,
    subset: int | None = None,
    seed: int = 42,
    auto_download: bool = True,
) -> RetrievalDataset:
    """Load (downloading on first use) and optionally subsample a dataset.

    Raises:
        DatasetUnavailableError: when files are missing and cannot be downloaded; the
            message lists the manual steps.
    """
    if name not in LOADERS:
        raise KeyError(f"unknown dataset '{name}'. Available: {available_datasets()}")
    ds = LOADERS[name](name, data_dir, auto_download)
    if subset:
        ds = select_subset(ds, subset, seed)
    return ds


__all__ = [
    "DISPLAY_NAMES",
    "HELDOUT",
    "DatasetUnavailableError",
    "Protocol",
    "Query",
    "RetrievalDataset",
    "available_datasets",
    "load_dataset",
]
