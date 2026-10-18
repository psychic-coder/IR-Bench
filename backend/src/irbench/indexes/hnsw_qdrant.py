"""HNSW graph index served by Qdrant.

The index talks to a Qdrant server (``url``) or, for tests, to qdrant-client's embedded
``":memory:"`` mode (``location``). Embedded mode performs exact search, so it validates
the plumbing but not HNSW behaviour.
"""

from __future__ import annotations

import contextlib
import time
import uuid
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

from irbench.indexes.base import MemoryReport, SearchResult, VectorIndex
from irbench.logging_utils import get_logger

log = get_logger(__name__)

UPSERT_BATCH = 512
QUERY_BATCH = 64


class QdrantHNSWIndex(VectorIndex):
    """HNSW with configurable ``m`` (graph degree), ``ef_construct`` and search-time ``ef``.

    Collections are created with dot-product distance (vectors are L2-normalised), a single
    segment, and an indexing threshold of 1 KB so the HNSW graph is always built (Qdrant
    otherwise keeps small collections as plain, exhaustively-scanned segments).
    ``build`` waits until every vector is HNSW-indexed before returning, so ``build_time``
    covers upload + graph construction.
    """

    kind = "hnsw"
    search_param_names = ("ef",)

    def __init__(
        self,
        m: int = 16,
        ef_construct: int = 128,
        ef: int = 64,
        url: str | None = None,
        location: str | None = None,
        collection: str | None = None,
        reuse: bool = False,
        keep: bool = False,
        timeout_s: float = 900.0,
        **params: Any,
    ) -> None:
        super().__init__(m=int(m), ef_construct=int(ef_construct), ef=int(ef), **params)
        if not url and not location:
            raise ValueError("QdrantHNSWIndex needs either url or location")
        self._client = QdrantClient(url=url, timeout=120) if url else QdrantClient(location=location)
        self._embedded = url is None
        self.collection = collection or f"irb_{uuid.uuid4().hex[:12]}"
        self.reuse = reuse
        self.keep = keep
        self.timeout_s = timeout_s

    def _collection_matches(self, n: int, d: int) -> bool:
        try:
            if not self._client.collection_exists(self.collection):
                return False
            info = self._client.get_collection(self.collection)
        except Exception:
            return False
        params = info.config.params.vectors
        size = getattr(params, "size", None)
        return (
            info.points_count == n
            and size == d
            and info.config.hnsw_config.m == self.params["m"]
            and info.config.hnsw_config.ef_construct == self.params["ef_construct"]
        )

    def build(self, vectors: np.ndarray, train_vectors: np.ndarray | None = None) -> None:
        x = self._check_vectors(vectors)
        self.n, self.d = x.shape
        start = time.perf_counter()
        if self.reuse and self._collection_matches(self.n, self.d):
            log.info("reusing existing qdrant collection", extra={"collection": self.collection})
            self._wait_indexed()
            self._build_time_s = time.perf_counter() - start
            return
        if self._client.collection_exists(self.collection):
            self._client.delete_collection(self.collection)
        self._client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(size=self.d, distance=models.Distance.DOT),
            hnsw_config=models.HnswConfigDiff(
                m=self.params["m"], ef_construct=self.params["ef_construct"], full_scan_threshold=10
            ),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1, default_segment_number=1),
        )
        for lo in range(0, self.n, UPSERT_BATCH):
            hi = min(lo + UPSERT_BATCH, self.n)
            self._client.upsert(
                collection_name=self.collection,
                points=models.Batch(ids=list(range(lo, hi)), vectors=x[lo:hi].tolist()),
                wait=True,
            )
        self._wait_indexed()
        self._build_time_s = time.perf_counter() - start

    def _wait_indexed(self) -> None:
        if self._embedded:
            return
        deadline = time.monotonic() + self.timeout_s
        while True:
            info = self._client.get_collection(self.collection)
            indexed = info.indexed_vectors_count or 0
            if info.status == models.CollectionStatus.GREEN and indexed >= (info.points_count or 0):
                return
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"qdrant collection {self.collection} not indexed after {self.timeout_s}s "
                    f"({indexed}/{info.points_count} vectors indexed, status={info.status})"
                )
            time.sleep(0.25)

    def search(self, queries: np.ndarray, k: int) -> SearchResult:
        q = self._check_queries(queries, self.d)
        k = max(1, min(int(k), self.n))
        params = models.SearchParams(hnsw_ef=self.params["ef"], exact=False)
        scores = np.full((q.shape[0], k), -np.inf, dtype=np.float32)
        idx = np.full((q.shape[0], k), -1, dtype=np.int64)
        if q.shape[0] == 1:
            batches = [
                self._client.query_points(
                    collection_name=self.collection,
                    query=q[0].tolist(),
                    limit=k,
                    search_params=params,
                    with_payload=False,
                ).points
            ]
        else:
            batches = []
            for lo in range(0, q.shape[0], QUERY_BATCH):
                reqs = [
                    models.QueryRequest(query=row.tolist(), limit=k, params=params, with_payload=False)
                    for row in q[lo : lo + QUERY_BATCH]
                ]
                batches.extend(
                    r.points
                    for r in self._client.query_batch_points(collection_name=self.collection, requests=reqs)
                )
        for i, points in enumerate(batches):
            for j, p in enumerate(points[:k]):
                idx[i, j] = int(p.id)
                scores[i, j] = float(p.score)
        return SearchResult(scores=scores, indices=idx)

    def memory_report(self) -> MemoryReport:
        m = self.params["m"]
        vectors = self.n * self.d * 4
        level0_links = self.n * 2 * m * 4
        upper_links = int(round(self.n * (m / max(m - 1, 1)) * 4))
        total = vectors + level0_links + upper_links
        return MemoryReport(
            total_bytes=total,
            n_vectors=self.n,
            bytes_per_vector=total / max(self.n, 1),
            encoding_bytes_per_vector=float(self.d * 4),
            breakdown={
                "vectors_float32": vectors,
                "graph_links_level0": level0_links,
                "graph_links_upper_levels": upper_links,
            },
            method="estimate",
            serialized_bytes=None,
            notes=(
                "lower-bound estimate: float32 vectors + 2M uint32 links per node on level 0 + "
                "expected M/(M-1) links per node on upper levels; excludes Qdrant id-tracker, "
                "payload and segment overhead"
            ),
        )

    def close(self) -> None:
        if not self.keep:
            try:
                if self._client.collection_exists(self.collection):
                    self._client.delete_collection(self.collection)
            except Exception as exc:  # pragma: no cover - best effort cleanup
                log.warning("failed to delete qdrant collection", extra={"error": str(exc)})
        with contextlib.suppress(Exception):
            self._client.close()
