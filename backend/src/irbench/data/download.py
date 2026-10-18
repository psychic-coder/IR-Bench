"""Automatic dataset download with existence checks, caching and manual-placement fallback.

Archive URLs are the ones used by the official revisitop download script
(``filipradenovic/revisitop/python/download.py``) for images and revisited ground truth;
the original-protocol ground-truth archive names come from the Oxford/Paris pages. Every
URL can be overridden with an environment variable. When a download fails the caller gets
:class:`DatasetUnavailableError` whose message names the page to open and the folder to
put the files in; the next run continues from local files.
"""

from __future__ import annotations

import os
import tarfile
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

from irbench.logging_utils import get_logger
from irbench.utils import ensure_dir

log = get_logger(__name__)

OXFORD_PAGE = "https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/"
PARIS_PAGE = "https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/"
REVISITOP_PAGE = "http://cmp.felk.cvut.cz/revisitop/"


class DatasetUnavailableError(RuntimeError):
    """Raised when data is missing and cannot be downloaded; message has manual steps."""


@dataclass(frozen=True)
class Resource:
    """One downloadable file."""

    key: str
    filename: str
    default_url: str
    env_var: str
    folder: str  # relative to DATA_DIR
    page: str
    kind: str  # "images" | "gt" | "gnd" | "distractors"
    description: str

    @property
    def url(self) -> str:
        return os.environ.get(self.env_var) or self.default_url


RESOURCES: dict[str, list[Resource]] = {
    "oxford5k": [
        Resource(
            "oxford_images",
            "oxbuild_images-v1.tgz",
            "https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images-v1.tgz",
            "OXFORD_IMAGES_URL",
            "oxford5k",
            OXFORD_PAGE,
            "images",
            "Oxford Buildings images (5,063 JPEGs)",
        ),
        Resource(
            "oxford_gt",
            "gt_files_170407.tgz",
            "https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/gt_files_170407.tgz",
            "OXFORD_GT_URL",
            "oxford5k",
            OXFORD_PAGE,
            "gt",
            "Oxford original ground truth (*_good/_ok/_junk/_query.txt)",
        ),
    ],
    "paris6k": [
        Resource(
            "paris_images_1",
            "paris_1-v1.tgz",
            "https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_1-v1.tgz",
            "PARIS_IMAGES_1_URL",
            "paris6k",
            PARIS_PAGE,
            "images",
            "Paris images, part 1",
        ),
        Resource(
            "paris_images_2",
            "paris_2-v1.tgz",
            "https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_2-v1.tgz",
            "PARIS_IMAGES_2_URL",
            "paris6k",
            PARIS_PAGE,
            "images",
            "Paris images, part 2",
        ),
        Resource(
            "paris_gt",
            "paris_120310.tgz",
            "https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_120310.tgz",
            "PARIS_GT_URL",
            "paris6k",
            PARIS_PAGE,
            "gt",
            "Paris original ground truth (*_good/_ok/_junk/_query.txt)",
        ),
    ],
    "revisitop": [
        Resource(
            "roxford_gnd",
            "gnd_roxford5k.pkl",
            "http://cmp.felk.cvut.cz/revisitop/data/datasets/roxford5k/gnd_roxford5k.pkl",
            "ROXFORD_GND_URL",
            "revisitop",
            REVISITOP_PAGE,
            "gnd",
            "Revisited Oxford ground truth (Easy/Medium/Hard)",
        ),
        Resource(
            "rparis_gnd",
            "gnd_rparis6k.pkl",
            "http://cmp.felk.cvut.cz/revisitop/data/datasets/rparis6k/gnd_rparis6k.pkl",
            "RPARIS_GND_URL",
            "revisitop",
            REVISITOP_PAGE,
            "gnd",
            "Revisited Paris ground truth (Easy/Medium/Hard)",
        ),
    ],
}

GND_FOR = {"oxford5k": "gnd_roxford5k.pkl", "paris6k": "gnd_rparis6k.pkl"}
EXPECTED_IMAGES = {"oxford5k": 5063, "paris6k": 6392}
REVISITOP1M_URL = "http://ptak.felk.cvut.cz/revisitop/revisitop1m/jpg/revisitop1m.{index}.tar.gz"


def _manual_message(resource: Resource, data_dir: Path, reason: str) -> str:
    return (
        f"Could not obtain {resource.description} ({resource.filename}): {reason}\n"
        f"  1. Open {resource.page}\n"
        f"  2. Download '{resource.filename}' (or the equivalent archive / extracted files)\n"
        f"  3. Place it in {data_dir / resource.folder} (host path ./data/{resource.folder})\n"
        f"  4. Re-run; archives found there are extracted automatically and the run continues "
        f"from local files.\n"
        f"  (Override the URL with the environment variable {resource.env_var} if it moved.)"
    )


def download_file(url: str, dest: Path, timeout: float = 60.0) -> Path:
    """Stream ``url`` to ``dest`` via a ``.part`` file (atomic rename on success)."""
    ensure_dir(dest.parent)
    part = dest.with_name(dest.name + ".part")
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0)) or None
        with open(part, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))
    os.replace(part, dest)
    return dest


def extract_flat(archive: Path, folder: Path) -> int:
    """Extract ``.jpg`` files into ``folder/jpg`` and ``.txt`` files into ``folder/gt``.

    Directory structure inside the archive is discarded (only basenames are used), which
    also makes extraction immune to path-traversal members. Returns the number of files
    written.
    """
    jpg_dir = ensure_dir(folder / "jpg")
    gt_dir = ensure_dir(folder / "gt")
    written = 0
    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            base = os.path.basename(member.name)
            if not base or base.startswith("."):
                continue
            lower = base.lower()
            if lower.endswith((".jpg", ".jpeg")):
                target = jpg_dir / base
            elif lower.endswith(".txt"):
                target = gt_dir / base
            else:
                continue
            src = tar.extractfile(member)
            if src is None:
                continue
            with open(target, "wb") as out:
                out.write(src.read())
            written += 1
    return written


def _marker(folder: Path, filename: str) -> Path:
    return folder / f".done-{filename}"


def _already_present(resource: Resource, folder: Path, dataset: str | None) -> bool:
    if resource.kind == "images" and dataset is not None:
        n = sum(1 for _ in folder.rglob("*.jpg"))
        return n >= int(0.95 * EXPECTED_IMAGES.get(dataset, 1))
    if resource.kind == "gt":
        return any(folder.rglob("*_query.txt"))
    if resource.kind == "gnd":
        return (folder / resource.filename).exists()
    return False


def ensure_resource(resource: Resource, data_dir: Path, auto_download: bool, dataset: str | None) -> None:
    """Make one resource available locally (existing files -> local archive -> download)."""
    folder = ensure_dir(data_dir / resource.folder)
    marker = _marker(folder, resource.filename)
    if marker.exists():
        return
    target = folder / resource.filename
    if resource.kind != "gnd" and target.exists():
        log.info("extracting local archive", extra={"archive": str(target)})
        extract_flat(target, folder)
        marker.touch()
        return
    if _already_present(resource, folder, dataset):
        marker.touch()
        return
    if not auto_download:
        raise DatasetUnavailableError(_manual_message(resource, data_dir, "AUTO_DOWNLOAD is disabled"))
    try:
        log.info("downloading", extra={"url": resource.url, "dest": str(target)})
        download_file(resource.url, target)
    except Exception as exc:
        raise DatasetUnavailableError(
            _manual_message(resource, data_dir, f"download failed ({exc})")
        ) from exc
    if resource.kind != "gnd":
        try:
            extract_flat(target, folder)
        except (tarfile.TarError, OSError) as exc:
            raise DatasetUnavailableError(
                _manual_message(resource, data_dir, f"archive is corrupt ({exc})")
            ) from exc
        if os.environ.get("KEEP_ARCHIVES", "false").lower() not in {"1", "true", "yes"}:
            target.unlink(missing_ok=True)
    marker.touch()


def ensure_dataset_files(dataset: str, data_dir: Path, auto_download: bool = True) -> None:
    """Ensure images, original ground truth and revisited ground truth for a dataset.

    All failures are collected so a single run reports every missing piece at once.
    """
    if dataset not in RESOURCES:
        raise KeyError(f"no download recipe for dataset '{dataset}'")
    errors: list[str] = []
    gnd = [r for r in RESOURCES["revisitop"] if r.filename == GND_FOR[dataset]]
    for resource in [*RESOURCES[dataset], *gnd]:
        try:
            ensure_resource(resource, data_dir, auto_download, dataset)
        except DatasetUnavailableError as exc:
            errors.append(str(exc))
    if errors:
        raise DatasetUnavailableError("\n\n".join(errors))


def ensure_distractors(data_dir: Path, folder: Path, n_archives: int, auto_download: bool) -> None:
    """Download ``n_archives`` revisitop1m archives (best effort) into ``folder/revisitop1m``.

    Failures are logged, not raised: the scale test then runs with whatever distractor
    images exist locally (and reports the achieved size).
    """
    if n_archives <= 0:
        return
    target_dir = ensure_dir(folder / "revisitop1m")
    for index in range(1, n_archives + 1):
        filename = f"revisitop1m.{index}.tar.gz"
        marker = _marker(target_dir, filename)
        if marker.exists():
            continue
        archive = target_dir / filename
        try:
            if not archive.exists():
                if not auto_download:
                    log.warning(
                        "distractor archive missing and AUTO_DOWNLOAD disabled", extra={"file": filename}
                    )
                    continue
                url = os.environ.get("REVISITOP1M_URL_TEMPLATE", REVISITOP1M_URL).format(index=index)
                download_file(url, archive, timeout=120)
            extract_flat(archive, target_dir)
            archive.unlink(missing_ok=True)
            marker.touch()
        except Exception as exc:
            log.warning(
                "distractor download failed; place images manually",
                extra={
                    "file": filename,
                    "error": str(exc),
                    "page": REVISITOP_PAGE,
                    "folder": str(folder),
                },
            )
