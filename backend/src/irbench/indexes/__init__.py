"""Vector indexes: exact (FAISS flat), HNSW (Qdrant), IVF-PQ (FAISS)."""

from __future__ import annotations

from typing import Any

from irbench.indexes.base import MemoryReport, SearchResult, VectorIndex
from irbench.indexes.flat import FlatIPIndex
from irbench.indexes.hnsw_qdrant import QdrantHNSWIndex
from irbench.indexes.ivfpq import IVFPQIndex

INDEXES: dict[str, type[VectorIndex]] = {
    FlatIPIndex.kind: FlatIPIndex,
    QdrantHNSWIndex.kind: QdrantHNSWIndex,
    IVFPQIndex.kind: IVFPQIndex,
}


def register_index(cls: type[VectorIndex]) -> type[VectorIndex]:
    """Class decorator adding an index type to the registry under ``cls.kind``."""
    INDEXES[cls.kind] = cls
    return cls


def create_index(kind: str, **params: Any) -> VectorIndex:
    """Instantiate an index by kind (``flat`` | ``hnsw`` | ``ivfpq``)."""
    if kind not in INDEXES:
        raise KeyError(f"unknown index '{kind}'. Registered: {sorted(INDEXES)}")
    return INDEXES[kind](**params)


__all__ = [
    "INDEXES",
    "FlatIPIndex",
    "IVFPQIndex",
    "MemoryReport",
    "QdrantHNSWIndex",
    "SearchResult",
    "VectorIndex",
    "create_index",
    "register_index",
]
