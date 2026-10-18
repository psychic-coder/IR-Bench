"""Query expansion: average QE (AQE) and alpha-weighted QE (alpha-QE)."""

from __future__ import annotations

from typing import Any

import numpy as np

from irbench.postprocess import l2_normalize
from irbench.rerank.base import Ranking, RerankContext, Reranker


class AverageQueryExpansion(Reranker):
    """AQE (Chum et al. 2007): ``q' = l2n(q + sum_{i<=n} x_i)``, then re-query the index."""

    name = "aqe"

    def __init__(self, top_n: int = 3, **params: Any) -> None:
        super().__init__(top_n=int(top_n), **params)

    def _weights(self, scores: np.ndarray) -> np.ndarray:
        return np.ones_like(scores, dtype=np.float32)

    def rerank(self, ctx: RerankContext, ranking: Ranking) -> Ranking:
        valid = ranking.valid()
        n = min(self.params["top_n"], valid.indices.size)
        if n == 0:
            return ranking
        top = valid.indices[:n]
        w = self._weights(valid.scores[:n].astype(np.float32))
        expanded = ctx.query_vector.astype(np.float32) + (w[:, None] * ctx.db_vectors[top]).sum(axis=0)
        q2 = l2_normalize(expanded[None, :])
        res = ctx.index.search(q2, ctx.depth)
        info = {**ranking.info, f"{self.name}_expanded_with": [int(i) for i in top]}
        return Ranking(scores=res.scores[0], indices=res.indices[0], info=info)


class AlphaQueryExpansion(AverageQueryExpansion):
    """alpha-QE (Radenovic et al. 2018): neighbours weighted by ``max(sim, 0) ** alpha``."""

    name = "alpha_qe"

    def __init__(self, top_n: int = 3, alpha: float = 3.0, **params: Any) -> None:
        super().__init__(top_n=top_n, alpha=float(alpha), **params)

    def _weights(self, scores: np.ndarray) -> np.ndarray:
        return np.power(np.clip(scores, 0.0, None), self.params["alpha"]).astype(np.float32)
