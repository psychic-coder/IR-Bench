"""Seeded distractor subsampling (default 10k) from any folder of images.

The default source is the first revisitop1m archive (downloaded on demand). Any images
placed under ``DISTRACTOR_DIR`` (e.g. the full Flickr100k set or all of R1M) are used as
the pool; a seeded shuffle picks the subset, and undecodable files are skipped and counted.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from irbench.features.cache import ImageItem
from irbench.features.preprocessing import verify_image
from irbench.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class DistractorSet:
    """Ordered distractor images; prefixes of ``items`` give nested subsets."""

    items: list[ImageItem]
    pool_size: int
    skipped_corrupt: int
    source: str
    notes: list[str] = field(default_factory=list)

    def prefix(self, n: int) -> list[ImageItem]:
        return self.items[: min(n, len(self.items))]


def load_distractors(
    folder: Path, max_size: int, seed: int, exclude_ids: set[str] | None = None
) -> DistractorSet:
    """Pick up to ``max_size`` decodable images from ``folder`` with a seeded shuffle.

    Ids are paths relative to ``folder`` (with extension); callers namespace them as
    ``distractors/<relative path>`` so they never collide with dataset image ids.
    """
    exclude_ids = exclude_ids or set()
    if not folder.exists():
        return DistractorSet([], 0, 0, str(folder), notes=[f"{folder} does not exist"])
    pool = sorted(
        p for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.is_file()
    )
    pool = [p for p in pool if p.stem not in exclude_ids]
    rng = random.Random(seed)
    rng.shuffle(pool)
    chosen: list[ImageItem] = []
    corrupt = 0
    for p in pool:
        if len(chosen) >= max_size:
            break
        if not verify_image(p):
            corrupt += 1
            continue
        rel = p.relative_to(folder).as_posix()
        chosen.append(ImageItem(id=rel, path=str(p)))
    notes = []
    if len(chosen) < max_size:
        notes.append(f"requested {max_size} distractors, only {len(chosen)} available in {folder}")
        log.warning("not enough distractor images", extra={"requested": max_size, "available": len(chosen)})
    return DistractorSet(chosen, len(pool), corrupt, str(folder), notes)
