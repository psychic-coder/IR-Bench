"""Inverted-file index with product quantisation (FAISS ``IndexIVFPQ``, inner product)."""

from __future__ import annotations

import math
import time
from typing import Any

import faiss
import numpy as np

from irbench.indexes.base import MemoryReport, SearchResult, VectorIndex

MIN_POINTS_PER_CENTROID = 39  # FAISS warns below this many training points per centroid


class IVFPQIndex(VectorIndex):
    """IVF coarse quantiser (``nlist`` cells) + PQ codes (``m`` sub-vectors x ``nbits``).

    Parameters are clamped to what the training set supports and the effective values
    are reported: ``nlist <= n_train / 39`` and ``2**nbits <= n_train`` (k-means cannot
    have more centroids than points). ``m`` must divide the dimensionality.
    """

    kind = "ivfpq"
    search_param_names = ("nprobe",)

    def __init__(
        self, nlist: int = 64, m: int = 16, nbits: int = 8, nprobe: int = 8, seed: int = 42, **params: Any
    ) -> None:
        super().__init__(
            nlist=int(nlist), m=int(m), nbits=int(nbits), nprobe=int(nprobe), seed=int(seed), **params
        )
        self._index: faiss.IndexIVFPQ | None = None
        self._effective: dict[str, Any] = {}

    def build(self, vectors: np.ndarray, train_vectors: np.ndarray | None = None) -> None:
        x = self._check_vectors(vectors)
        train = x if train_vectors is None else self._check_vectors(train_vectors)
        self.n, self.d = x.shape
        if train.shape[1] != self.d:
            raise ValueError("training vectors have a different dimensionality")
        m = self.params["m"]
        if self.d % m != 0:
            raise ValueError(f"PQ m={m} must divide the dimensionality d={self.d}")
        n_train = train.shape[0]
        nlist = max(1, min(self.params["nlist"], n_train // MIN_POINTS_PER_CENTROID or 1))
        nbits = max(1, min(self.params["nbits"], int(math.floor(math.log2(max(2, n_train))))))
        start = time.perf_counter()
        quantizer = faiss.IndexFlatIP(self.d)
        index = faiss.IndexIVFPQ(quantizer, self.d, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
        index.cp.seed = self.params["seed"]
        index.pq.cp.seed = self.params["seed"]
        # parameters are already clamped above; silence FAISS' small-training-set warning
        index.cp.min_points_per_centroid = 1
        index.pq.cp.min_points_per_centroid = 1
        index.train(train)
        index.add(x)
        self._build_time_s = time.perf_counter() - start
        self._index = index
        self._quantizer = quantizer  # keep a Python reference alive for the C++ object
        self._effective = {"nlist": nlist, "m": m, "nbits": nbits, "n_train": n_train}
        self.set_search_params(nprobe=self.params["nprobe"])

    def set_search_params(self, **params: Any) -> None:
        super().set_search_params(**params)
        if self._index is not None and "nprobe" in params:
            self._index.nprobe = max(1, min(int(params["nprobe"]), self._index.nlist))

    def effective_params(self) -> dict[str, Any]:
        out = dict(self.params)
        out.update(self._effective)
        if self._index is not None:
            out["nprobe"] = int(self._index.nprobe)
        return out

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
        pq = self._index.pq
        code_size = int(pq.code_size)
        codes = self.n * code_size
        ids = self.n * 8
        centroids = int(self._index.nlist) * self.d * 4
        codebooks = int(pq.M) * int(pq.ksub) * int(pq.dsub) * 4
        total = codes + ids + centroids + codebooks
        serialized = int(faiss.serialize_index(self._index).nbytes)
        return MemoryReport(
            total_bytes=total,
            n_vectors=self.n,
            bytes_per_vector=total / self.n,
            encoding_bytes_per_vector=float(code_size),
            breakdown={
                "pq_codes": codes,
                "inverted_list_ids_int64": ids,
                "coarse_centroids_float32": centroids,
                "pq_codebooks_float32": codebooks,
            },
            method="exact",
            serialized_bytes=serialized,
            notes="codes n*ceil(m*nbits/8) + ids n*8 + centroids nlist*d*4 + codebooks 2^nbits*d*4",
        )
