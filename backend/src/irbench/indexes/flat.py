"""Exact brute-force inner-product search (FAISS ``IndexFlatIP``)."""

from __future__ import annotations

import time

import faiss
import numpy as np

from irbench.indexes.base import MemoryReport, SearchResult, VectorIndex


class FlatIPIndex(VectorIndex):
    """Exhaustive search; the accuracy reference for every approximate index."""

    kind = "flat"

    def __init__(self, **params: object) -> None:
        super().__init__(**params)
        self._index: faiss.IndexFlatIP | None = None

    def build(self, vectors: np.ndarray, train_vectors: np.ndarray | None = None) -> None:
        x = self._check_vectors(vectors)
        start = time.perf_counter()
        self.n, self.d = x.shape
        self._index = faiss.IndexFlatIP(self.d)
        self._index.add(x)
        self._build_time_s = time.perf_counter() - start

    def search(self, queries: np.ndarray, k: int) -> SearchResult:
        if self._index is None:
            raise RuntimeError("index not built")
        q = self._check_queries(queries, self.d)
        k = max(1, min(int(k), self.n))
        scores, idx = self._index.search(q, k)
        return SearchResult(scores=scores, indices=idx.astype(np.int64))

    def memory_report(self) -> MemoryReport:
        if self._index is None:
            raise RuntimeError("index not built")
        vec_bytes = self.n * self.d * 4
        serialized = int(faiss.serialize_index(self._index).nbytes)
        return MemoryReport(
            total_bytes=vec_bytes,
            n_vectors=self.n,
            bytes_per_vector=vec_bytes / self.n,
            encoding_bytes_per_vector=float(self.d * 4),
            breakdown={"vectors_float32": vec_bytes},
            method="exact",
            serialized_bytes=serialized,
            notes="n*d*4 bytes; ids are implicit row numbers",
        )
