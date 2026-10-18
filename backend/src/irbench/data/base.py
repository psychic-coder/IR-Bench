"""Dataset containers shared by all retrieval datasets.

A :class:`RetrievalDataset` owns a pool of images (``id -> path``) and one
:class:`Protocol` per evaluation protocol. Each protocol has its own database order,
query list (with bounding boxes) and ground truth expressed as *row indices into that
protocol's database*:

* original protocol : ``{"pos": [...], "junk": [...]}``           (pos = good + ok)
* revisited protocol: ``{"easy": [...], "hard": [...], "junk": [...]}``
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from irbench.features.cache import ImageItem


@dataclass
class Query:
    """One query: image id (file stem), path and ``[x1, y1, x2, y2]`` crop."""

    id: str
    path: str
    bbox: tuple[float, float, float, float] | None


@dataclass
class Protocol:
    """Database, queries and ground truth for one protocol of one dataset."""

    name: str  # "original" | "revisited"
    db_ids: list[str]
    queries: list[Query]
    gt: list[dict[str, list[int]]]

    def db_items(self, paths: dict[str, str]) -> list[ImageItem]:
        return [ImageItem(id=i, path=paths[i]) for i in self.db_ids]

    def query_items(self) -> list[ImageItem]:
        return [ImageItem(id=q.id, path=q.path, bbox=q.bbox) for q in self.queries]

    def positives(self, qi: int, setup: str | None = None) -> list[int]:
        """Positive rows for query ``qi`` (``setup`` = easy|medium|hard for revisited)."""
        g = self.gt[qi]
        if self.name == "original":
            return list(g["pos"])
        setup = setup or "medium"
        if setup == "easy":
            return list(g["easy"])
        if setup == "medium":
            return list(g["easy"]) + list(g["hard"])
        if setup == "hard":
            return list(g["hard"])
        raise ValueError(f"unknown revisited setup '{setup}'")

    def junk(self, qi: int, setup: str | None = None) -> list[int]:
        """Rows ignored when scoring query ``qi``."""
        g = self.gt[qi]
        if self.name == "original":
            return list(g["junk"])
        setup = setup or "medium"
        if setup == "easy":
            return list(g["junk"]) + list(g["hard"])
        if setup == "medium":
            return list(g["junk"])
        if setup == "hard":
            return list(g["junk"]) + list(g["easy"])
        raise ValueError(f"unknown revisited setup '{setup}'")


@dataclass
class RetrievalDataset:
    """A dataset with its image pool and evaluation protocols."""

    name: str
    root: Path
    paths: dict[str, str]
    protocols: dict[str, Protocol]
    heldout: str | None = None
    missing: list[str] = field(default_factory=list)
    subset: int | None = None
    stats: dict[str, Any] = field(default_factory=dict)

    def all_db_items(self) -> list[ImageItem]:
        """Union of every protocol's database, sorted by id (feature extraction order)."""
        ids = sorted({i for p in self.protocols.values() for i in p.db_ids})
        return [ImageItem(id=i, path=self.paths[i]) for i in ids]

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "heldout": self.heldout,
            "subset": self.subset,
            "n_images_pool": len(self.paths),
            "n_missing_or_corrupt": len(self.missing),
            "missing_examples": self.missing[:20],
            "protocols": {
                k: {"n_db": len(p.db_ids), "n_queries": len(p.queries)} for k, p in self.protocols.items()
            },
            **self.stats,
        }


def drop_missing(protocol: Protocol, available: set[str]) -> tuple[Protocol, dict[str, int]]:
    """Remove unavailable database images and queries; remap ground-truth rows."""
    keep_rows = [r for r, i in enumerate(protocol.db_ids) if i in available]
    remap = {old: new for new, old in enumerate(keep_rows)}
    db_ids = [protocol.db_ids[r] for r in keep_rows]
    queries: list[Query] = []
    gts: list[dict[str, list[int]]] = []
    dropped_q = 0
    for q, g in zip(protocol.queries, protocol.gt, strict=True):
        if q.id not in available:
            dropped_q += 1
            continue
        queries.append(q)
        gts.append({k: [remap[r] for r in v if r in remap] for k, v in g.items()})
    stats = {
        "db_dropped": len(protocol.db_ids) - len(db_ids),
        "queries_dropped": dropped_q,
    }
    return Protocol(protocol.name, db_ids, queries, gts), stats


def select_subset(
    dataset: RetrievalDataset, size: int, seed: int, relevant_share: float = 0.6
) -> RetrievalDataset:
    """Seeded smoke-test subset with roughly ``size`` database images.

    Queries are visited in a seeded random order; a query is kept while the union of its
    relevant images (all positives + junk, across protocols) still fits in
    ``relevant_share * size``. The remaining budget is filled with random other images.
    Ground truth is then restricted to the chosen images (queries left with no positive
    are kept but score NaN and are excluded from the mean, as in the official code).
    """
    rng = random.Random(seed)
    budget = max(1, int(size * relevant_share))
    rel_by_query: dict[str, set[str]] = {}
    for proto in dataset.protocols.values():
        for q, g in zip(proto.queries, proto.gt, strict=True):
            ids = {proto.db_ids[r] for rows in g.values() for r in rows}
            rel_by_query.setdefault(q.id, set()).update(ids)
    query_ids = sorted(rel_by_query)
    rng.shuffle(query_ids)
    chosen: set[str] = set()
    chosen_queries: set[str] = set()
    for qid in query_ids:
        rel = rel_by_query[qid]
        if not chosen_queries and len(rel) > budget:
            rel = set(rng.sample(sorted(rel), budget))
        if len(chosen | rel) <= budget or not chosen_queries:
            chosen |= rel
            chosen_queries.add(qid)
    pool = sorted({i for p in dataset.protocols.values() for i in p.db_ids} - chosen)
    fill = max(0, size - len(chosen))
    chosen |= set(rng.sample(pool, min(fill, len(pool))))

    protocols: dict[str, Protocol] = {}
    for name, proto in dataset.protocols.items():
        db_ids = [i for i in proto.db_ids if i in chosen]
        row = {i: r for r, i in enumerate(db_ids)}
        queries: list[Query] = []
        gts: list[dict[str, list[int]]] = []
        for q, g in zip(proto.queries, proto.gt, strict=True):
            if q.id not in chosen_queries:
                continue
            queries.append(q)
            gts.append({k: [row[proto.db_ids[r]] for r in v if proto.db_ids[r] in row] for k, v in g.items()})
        protocols[name] = Protocol(proto.name, db_ids, queries, gts)
    return RetrievalDataset(
        name=dataset.name,
        root=dataset.root,
        paths=dataset.paths,
        protocols=protocols,
        heldout=dataset.heldout,
        missing=dataset.missing,
        subset=size,
        stats={**dataset.stats, "subset_seed": seed},
    )
