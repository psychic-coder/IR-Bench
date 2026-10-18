"""Vector-index interface: ``build``, ``search``, ``memory_bytes``, ``build_time``."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

import numpy as np


@dataclass
class MemoryReport:
    """Memory accounting for an index.

    ``encoding_bytes_per_vector`` is the size of one stored vector representation (raw
    float32 for flat/HNSW, the PQ code for IVF-PQ). ``bytes_per_vector`` amortises the
    *whole* index (codes, ids, codebooks, centroids, graph links) over ``n`` vectors.
    ``method`` is ``"exact"`` when derived from the data structure layout (and cross-checked
    against a serialisation when available) and ``"estimate"`` when the index lives in an
    external service whose internal overhead is not observable (Qdrant).
    """

    total_bytes: int
    n_vectors: int
    bytes_per_vector: float
    encoding_bytes_per_vector: float
    breakdown: dict[str, int]
    method: str
    serialized_bytes: int | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResult:
    """Top-k scores and integer row indices (``-1`` marks an empty slot)."""

    scores: np.ndarray
    indices: np.ndarray
    extra: dict[str, Any] = field(default_factory=dict)


class VectorIndex(ABC):
    """Inner-product index over L2-normalised float32 vectors (so IP == cosine)."""

    kind: ClassVar[str] = "base"
    search_param_names: ClassVar[tuple[str, ...]] = ()

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = dict(params)
        self.n: int = 0
        self.d: int = 0
        self._build_time_s: float | None = None

    @property
    def build_time(self) -> float | None:
        """Seconds spent in :meth:`build` (training + adding + graph construction)."""
        return self._build_time_s

    @abstractmethod
    def build(self, vectors: np.ndarray, train_vectors: np.ndarray | None = None) -> None:
        """Index ``vectors`` (``(n, d)`` float32); row ``i`` gets id ``i``.

        ``train_vectors`` is used by indexes that learn codebooks/centroids (IVF-PQ);
        when ``None`` they train on ``vectors`` themselves.
        """

    @abstractmethod
    def search(self, queries: np.ndarray, k: int) -> SearchResult:
        """Return the ``k`` highest inner-product rows for each query."""

    @abstractmethod
    def memory_report(self) -> MemoryReport:
        """Detailed memory accounting."""

    def memory_bytes(self) -> int:
        """Total index size in bytes."""
        return self.memory_report().total_bytes

    def set_search_params(self, **params: Any) -> None:
        """Change search-time parameters (``ef``, ``nprobe``) without rebuilding."""
        unknown = set(params) - set(self.search_param_names)
        if unknown:
            raise ValueError(f"{self.kind} has no search parameters {sorted(unknown)}")
        self.params.update(params)

    def effective_params(self) -> dict[str, Any]:
        """Parameters actually used (after clamping to the data size)."""
        return dict(self.params)

    def close(self) -> None:  # noqa: B027 - optional hook, deliberately not abstract
        """Release external resources (no-op for in-process indexes)."""

    @staticmethod
    def _check_vectors(vectors: np.ndarray) -> np.ndarray:
        arr = np.ascontiguousarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            raise ValueError("vectors must be a non-empty (n, d) matrix")
        return arr

    @staticmethod
    def _check_queries(queries: np.ndarray, d: int) -> np.ndarray:
        arr = np.ascontiguousarray(queries, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] != d:
            raise ValueError(f"query dimension {arr.shape[1]} != index dimension {d}")
        return arr
