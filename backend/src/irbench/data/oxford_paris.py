"""Parsers for Oxford5k / Paris6k (original protocol) and ROxford / RParis (revisited).

Original ground truth: one ``<landmark>_<n>_{query,good,ok,junk}.txt`` quadruple per
query. The query file holds ``<image> x1 y1 x2 y2`` (Oxford prefixes the image name with
``oxc1_``). Positives are good + ok; junk images are ignored during scoring.

Revisited ground truth (``gnd_r*.pkl``): ``imlist`` (database), ``qimlist`` (queries) and
per-query ``gnd`` dicts with ``bbx``, ``easy``, ``hard`` and ``junk`` index arrays.
"""

from __future__ import annotations

import json
import pickle
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from irbench.data.base import Protocol, Query, RetrievalDataset, drop_missing
from irbench.features.preprocessing import verify_image
from irbench.logging_utils import get_logger

log = get_logger(__name__)

QUERY_FILE_RE = re.compile(r"^(?P<stem>.+)_query\.txt$")


def scan_images(root: Path) -> dict[str, str]:
    """Map file stem -> path for every ``.jpg`` below ``root`` (first one wins on clashes)."""
    paths: dict[str, str] = {}
    for p in sorted(root.rglob("*.jpg")):
        if "cache" in p.parts:
            continue
        paths.setdefault(p.stem, str(p))
    return paths


def find_corrupt(root: Path, paths: dict[str, str], workers: int = 8) -> list[str]:
    """Ids whose files fail to decode; cached in ``root/.verify.json`` keyed by pool size."""
    cache_file = root / ".verify.json"
    if cache_file.exists():
        try:
            with open(cache_file, encoding="utf-8") as fh:
                cached = json.load(fh)
            if cached.get("n") == len(paths):
                return list(cached["corrupt"])
        except (OSError, ValueError, KeyError):
            pass
    ids = sorted(paths)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        ok = list(pool.map(lambda i: verify_image(paths[i]), ids))
    corrupt = [i for i, good in zip(ids, ok, strict=True) if not good]
    try:
        with open(cache_file, "w", encoding="utf-8") as fh:
            json.dump({"n": len(paths), "corrupt": corrupt}, fh)
    except OSError:
        pass
    if corrupt:
        log.warning("corrupt images skipped", extra={"root": str(root), "count": len(corrupt)})
    return corrupt


def _read_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return [line.strip() for line in fh if line.strip()]


def _strip_prefix(name: str) -> str:
    name = name.strip()
    if name.startswith("oxc1_"):
        name = name[len("oxc1_") :]
    if name.lower().endswith(".jpg"):
        name = name[:-4]
    return name


def parse_original(gt_files: list[Path], db_ids: list[str], paths: dict[str, str]) -> Protocol:
    """Build the original protocol from ``*_query.txt`` files (siblings give good/ok/junk)."""
    row = {i: r for r, i in enumerate(db_ids)}
    queries: list[Query] = []
    gts: list[dict[str, list[int]]] = []
    for qfile in sorted(gt_files, key=lambda p: p.name):
        m = QUERY_FILE_RE.match(qfile.name)
        if not m:
            continue
        stem = m.group("stem")
        parts = _read_lines(qfile)[0].split()
        qid = _strip_prefix(parts[0])
        bbox = tuple(float(v) for v in parts[1:5]) if len(parts) >= 5 else None
        lists: dict[str, list[str]] = {}
        for kind in ("good", "ok", "junk"):
            f = qfile.with_name(f"{stem}_{kind}.txt")
            lists[kind] = [_strip_prefix(x) for x in _read_lines(f)] if f.exists() else []
        pos = [row[i] for i in lists["good"] + lists["ok"] if i in row]
        junk = [row[i] for i in lists["junk"] if i in row]
        queries.append(Query(id=qid, path=paths.get(qid, ""), bbox=bbox))  # type: ignore[arg-type]
        gts.append({"pos": sorted(set(pos)), "junk": sorted(set(junk))})
    return Protocol("original", list(db_ids), queries, gts)


def parse_revisited(gnd_file: Path, paths: dict[str, str]) -> Protocol:
    """Build the revisited protocol from a ``gnd_r*.pkl`` file."""
    with open(gnd_file, "rb") as fh:
        cfg = pickle.load(fh)
    imlist = [str(x) for x in cfg["imlist"]]
    qimlist = [str(x) for x in cfg["qimlist"]]
    queries: list[Query] = []
    gts: list[dict[str, list[int]]] = []
    for qid, g in zip(qimlist, cfg["gnd"], strict=True):
        bbx = g.get("bbx")
        bbox = tuple(float(v) for v in bbx) if bbx is not None and len(bbx) == 4 else None
        queries.append(Query(id=qid, path=paths.get(qid, ""), bbox=bbox))  # type: ignore[arg-type]
        gts.append({k: sorted(int(v) for v in g.get(k, [])) for k in ("easy", "hard", "junk")})
    return Protocol("revisited", imlist, queries, gts)


def load_oxford_paris(
    name: str, root: Path, gnd_file: Path | None, heldout: str | None, verify: bool = True
) -> RetrievalDataset:
    """Load a dataset folder that follows the Oxford/Paris layout (also used by toy data)."""
    paths = scan_images(root)
    if not paths:
        raise FileNotFoundError(f"no .jpg images found under {root}")
    corrupt = find_corrupt(root, paths) if verify else []
    available = set(paths) - set(corrupt)

    protocols: dict[str, Protocol] = {}
    gt_files = [p for p in root.rglob("*_query.txt")]
    if gt_files:
        protocols["original"] = parse_original(gt_files, sorted(paths), paths)
    if gnd_file is not None and gnd_file.exists():
        protocols["revisited"] = parse_revisited(gnd_file, paths)
    if not protocols:
        raise FileNotFoundError(f"no ground truth found for {name} (looked under {root} and {gnd_file})")

    referenced: set[str] = set()
    stats: dict[str, object] = {"n_corrupt": len(corrupt), "corrupt_ids": corrupt}
    for pname, proto in list(protocols.items()):
        referenced |= set(proto.db_ids) | {q.id for q in proto.queries}
        cleaned, drop_stats = drop_missing(proto, available)
        protocols[pname] = cleaned
        stats[f"{pname}_db_dropped"] = drop_stats["db_dropped"]
        stats[f"{pname}_queries_dropped"] = drop_stats["queries_dropped"]
    missing = sorted((referenced - set(paths)) | (referenced & set(corrupt)))
    stats["n_missing_referenced"] = len(missing)
    if missing:
        log.warning(
            "missing or corrupt images referenced by ground truth",
            extra={"dataset": name, "count": len(missing)},
        )
    return RetrievalDataset(
        name=name,
        root=root,
        paths={k: v for k, v in paths.items() if k in available},
        protocols=protocols,
        heldout=heldout,
        missing=missing,
        stats=stats,
    )
