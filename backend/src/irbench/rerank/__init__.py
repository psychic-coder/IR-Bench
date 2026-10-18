"""Re-rankers: AQE, alpha-QE, ORB+RANSAC spatial verification, and chains of them."""

from __future__ import annotations

from typing import Any

from irbench.rerank.base import Ranking, RerankChain, RerankContext, Reranker
from irbench.rerank.qe import AlphaQueryExpansion, AverageQueryExpansion
from irbench.rerank.spatial import KeypointCache, SpatialVerification

RERANKERS: dict[str, type[Reranker]] = {
    AverageQueryExpansion.name: AverageQueryExpansion,
    AlphaQueryExpansion.name: AlphaQueryExpansion,
    SpatialVerification.name: SpatialVerification,
}


def register_reranker(cls: type[Reranker]) -> type[Reranker]:
    """Class decorator adding a re-ranker to the registry under ``cls.name``."""
    RERANKERS[cls.name] = cls
    return cls


def create_reranker(
    spec: str | None, params: dict[str, dict[str, Any]] | None = None, **shared: Any
) -> Reranker | None:
    """Build a re-ranker from a spec such as ``"none"``, ``"aqe"`` or ``"aqe+sv"``.

    ``params`` maps stage name -> keyword arguments; ``shared`` kwargs (e.g. a keypoint
    cache) are passed to stages that accept them.
    """
    if spec is None or spec in {"", "none"}:
        return None
    params = params or {}
    stages: list[Reranker] = []
    for name in spec.split("+"):
        name = name.strip()
        if name not in RERANKERS:
            raise KeyError(f"unknown re-ranker '{name}'. Registered: {sorted(RERANKERS)}")
        kwargs = dict(params.get(name, {}))
        if name == SpatialVerification.name and "keypoint_cache" in shared:
            kwargs["keypoint_cache"] = shared["keypoint_cache"]
        if name == SpatialVerification.name and "seed" in shared:
            kwargs.setdefault("seed", shared["seed"])
        stages.append(RERANKERS[name](**kwargs))
    return stages[0] if len(stages) == 1 else RerankChain(stages)


def rerank_needs_images(spec: str | None) -> bool:
    """True if the spec includes spatial verification."""
    return bool(spec) and SpatialVerification.name in str(spec).split("+")


__all__ = [
    "RERANKERS",
    "AlphaQueryExpansion",
    "AverageQueryExpansion",
    "KeypointCache",
    "Ranking",
    "RerankChain",
    "RerankContext",
    "Reranker",
    "SpatialVerification",
    "create_reranker",
    "register_reranker",
    "rerank_needs_images",
]
