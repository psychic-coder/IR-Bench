"""Re-ranker interface: refine an initial ranked list for one query."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from PIL import Image

from irbench.indexes.base import VectorIndex


@dataclass
class RerankContext:
    """Everything a re-ranker may need for one query.

    ``db_vectors`` are the post-processed database descriptors (row ``i`` == index id ``i``).
    ``query_image`` and ``load_db_image`` are only needed by spatial verification.
    """

    query_vector: np.ndarray
    db_vectors: np.ndarray
    index: VectorIndex
    depth: int
    query_image: Image.Image | None = None
    load_db_image: Callable[[int], Image.Image] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Ranking:
    """A ranked list: ``indices[j]`` is the database row at rank ``j``."""

    scores: np.ndarray
    indices: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)

    def valid(self) -> Ranking:
        """Drop empty (-1) slots."""
        mask = self.indices >= 0
        return Ranking(scores=self.scores[mask], indices=self.indices[mask], info=self.info)


class Reranker(ABC):
    """Base class for re-rankers."""

    name: ClassVar[str] = "base"

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)

    @abstractmethod
    def rerank(self, ctx: RerankContext, ranking: Ranking) -> Ranking:
        """Return a refined ranking (may re-query the index)."""

    def config(self) -> dict[str, Any]:
        return {"name": self.name, **self.params}


class RerankChain(Reranker):
    """Apply several re-rankers in order (e.g. AQE then spatial verification)."""

    name = "chain"

    def __init__(self, stages: list[Reranker]) -> None:
        super().__init__()
        self.stages = stages

    def rerank(self, ctx: RerankContext, ranking: Ranking) -> Ranking:
        for stage in self.stages:
            ranking = stage.rerank(ctx, ranking)
        return ranking

    def config(self) -> dict[str, Any]:
        return {"name": "+".join(s.name for s in self.stages), "stages": [s.config() for s in self.stages]}
