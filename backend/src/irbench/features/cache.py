"""Disk cache of extracted descriptors (``.npy`` + JSON metadata) keyed by a config hash."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from irbench.features.base import FeatureExtractor
from irbench.features.preprocessing import crop_bbox, load_image
from irbench.logging_utils import get_logger
from irbench.utils import ensure_dir, stable_hash

log = get_logger(__name__)


@dataclass(frozen=True)
class ImageItem:
    """An image to describe: identifier, file path and optional ``[x1, y1, x2, y2]`` crop."""

    id: str
    path: str
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class FeatureSet:
    """Descriptors for an ordered list of images plus extraction timing metadata."""

    ids: list[str]
    vectors: np.ndarray
    meta: dict[str, Any]


class FeatureCache:
    """Content-addressed store under ``<root>/<key>.npy`` and ``<root>/<key>.json``."""

    def __init__(self, root: Path) -> None:
        self.root = ensure_dir(Path(root))

    def _paths(self, key: str) -> tuple[Path, Path]:
        return self.root / f"{key}.npy", self.root / f"{key}.json"

    def load(self, key: str) -> FeatureSet | None:
        npy, meta_path = self._paths(key)
        if not (npy.exists() and meta_path.exists()):
            return None
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        vectors = np.load(npy)
        ids = meta.pop("ids")
        if len(ids) != vectors.shape[0]:
            log.warning("feature cache entry is inconsistent, ignoring", extra={"key": key})
            return None
        return FeatureSet(ids=ids, vectors=vectors.astype(np.float32, copy=False), meta=meta)

    def save(self, key: str, fs: FeatureSet) -> None:
        npy, meta_path = self._paths(key)
        tmp_npy = npy.with_name(npy.stem + ".tmp.npy")
        np.save(tmp_npy, fs.vectors.astype(np.float32))
        os.replace(tmp_npy, npy)
        tmp_meta = meta_path.with_suffix(".json.tmp")
        with open(tmp_meta, "w", encoding="utf-8") as fh:
            json.dump({**fs.meta, "ids": fs.ids}, fh)
        os.replace(tmp_meta, meta_path)


def feature_cache_key(extractor: FeatureExtractor, items: Sequence[ImageItem], tag: str, device: str) -> str:
    """Hash of extractor config + exact image list (ids and crops) + device + tag."""
    return stable_hash(
        {
            "extractor": extractor.config(),
            "items": stable_hash([(it.id, it.bbox) for it in items], length=32),
            "tag": tag,
            "device": device,
            "v": 2,
        },
        length=24,
    )


def extract_items(
    extractor: FeatureExtractor,
    items: Sequence[ImageItem],
    cache: FeatureCache | None,
    tag: str,
    device: str,
    per_item_timing: bool = False,
    warmup: int = 0,
    loader: Callable[[str], Image.Image] = load_image,
    show_progress: bool = True,
    progress_cb: Callable[[int, int], None] | None = None,
) -> FeatureSet:
    """Describe ``items`` with ``extractor``, using the cache when possible.

    With ``per_item_timing`` every image is processed alone (batch of one, as a query
    would be at serving time) and its wall-clock extraction time is recorded; the first
    ``warmup`` items are processed once untimed beforehand. Otherwise images are processed
    in chunks of ``extractor.batch_size`` and only the total time is recorded. Disk
    decoding is excluded from timings; resizing/normalisation and the forward pass are
    included. ``progress_cb(done, total)`` is called after every chunk (used by the API's
    readiness endpoint).
    """
    key = feature_cache_key(extractor, items, tag, device)
    if cache is not None:
        hit = cache.load(key)
        if hit is not None:
            hit.meta["cache_hit"] = True
            return hit

    vectors = np.zeros((len(items), extractor.dim), dtype=np.float32)
    per_item_ms: list[float] = []
    total_ms = 0.0
    desc = f"extract {extractor.name}/{extractor.pooling} [{tag}]"
    if per_item_timing:
        for it in items[: max(0, warmup)]:
            extractor.extract([crop_bbox(loader(it.path), it.bbox)])
        for i, it in enumerate(tqdm(items, desc=desc, disable=not show_progress, leave=False)):
            img = crop_bbox(loader(it.path), it.bbox)
            start = time.perf_counter()
            vectors[i] = extractor.extract([img])[0]
            ms = (time.perf_counter() - start) * 1000.0
            per_item_ms.append(ms)
            total_ms += ms
            if progress_cb is not None:
                progress_cb(i + 1, len(items))
    else:
        chunk = max(1, extractor.batch_size * 4)
        progress = tqdm(total=len(items), desc=desc, disable=not show_progress, leave=False)
        for start_idx in range(0, len(items), chunk):
            batch_items = items[start_idx : start_idx + chunk]
            imgs = [crop_bbox(loader(it.path), it.bbox) for it in batch_items]
            start = time.perf_counter()
            vectors[start_idx : start_idx + len(batch_items)] = extractor.extract(imgs)
            total_ms += (time.perf_counter() - start) * 1000.0
            progress.update(len(batch_items))
            if progress_cb is not None:
                progress_cb(min(start_idx + chunk, len(items)), len(items))
        progress.close()

    meta: dict[str, Any] = {
        "extractor_config": extractor.config(),
        "tag": tag,
        "device": device,
        "n": len(items),
        "dim": int(extractor.dim),
        "total_ms": total_ms,
        "ms_per_image": (total_ms / len(items)) if items else None,
        "per_item_ms": per_item_ms if per_item_timing else None,
        "cache_hit": False,
    }
    fs = FeatureSet(ids=[it.id for it in items], vectors=vectors, meta=meta)
    if cache is not None:
        cache.save(key, fs)
        fs.meta = {**meta}
    return fs
