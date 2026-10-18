"""Query expansion and spatial verification behaviour."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from irbench.features.preprocessing import load_image
from irbench.indexes import create_index
from irbench.rerank import (
    AlphaQueryExpansion,
    KeypointCache,
    Ranking,
    RerankChain,
    RerankContext,
    create_reranker,
    rerank_needs_images,
)
from irbench.rerank.spatial import SpatialVerification


def _ctx(db: np.ndarray, q: np.ndarray, depth: int = 10, **kw: object) -> tuple[RerankContext, Ranking]:
    index = create_index("flat")
    index.build(db)
    res = index.search(q[None, :], depth)
    ctx = RerankContext(query_vector=q, db_vectors=db, index=index, depth=depth, **kw)
    return ctx, Ranking(res.scores[0], res.indices[0])


def test_aqe_pulls_in_neighbours_of_neighbours() -> None:
    # 1-D chain on the unit circle: q is close to a, a is close to b; AQE should raise b
    angles = np.array([0.0, 0.25, 0.5, 1.6, 3.0])
    db = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    q = np.array([np.cos(-0.2), np.sin(-0.2)], dtype=np.float32)
    ctx, ranking = _ctx(db, q, depth=5)
    out = create_reranker("aqe", {"aqe": {"top_n": 2}}).rerank(ctx, ranking)
    assert set(out.indices[:3].tolist()) == {0, 1, 2}
    assert out.scores[2] > ranking.scores[2]  # the third item is scored against the expanded query


def test_alpha_qe_weights() -> None:
    w = AlphaQueryExpansion(top_n=3, alpha=3.0)._weights(np.array([1.0, 0.5, -0.2], dtype=np.float32))
    assert np.allclose(w, [1.0, 0.125, 0.0])


def test_reranker_factory() -> None:
    assert create_reranker("none") is None
    chain = create_reranker("aqe+sv", {"sv": {"top_k": 5}})
    assert isinstance(chain, RerankChain) and [s.name for s in chain.stages] == ["aqe", "sv"]
    assert rerank_needs_images("aqe+sv") and not rerank_needs_images("aqe")
    with pytest.raises(KeyError):
        create_reranker("diffusion-ish")


def test_spatial_verification_promotes_geometric_match(toy_data_dir: Path) -> None:
    jpgs = sorted((toy_data_dir / "toy_a" / "jpg").glob("*.jpg"))
    target = load_image(jpgs[0])
    # query: a crop of the target image (same scene geometry)
    w, h = target.size
    query = target.crop((w // 6, h // 6, w - w // 6, h - h // 6))
    paths = [jpgs[-1], jpgs[-2], jpgs[0], jpgs[-3]]  # target sits at rank 3 initially
    db = np.eye(4, dtype=np.float32)
    q = np.array([0.6, 0.5, 0.4, 0.3], dtype=np.float32)
    q /= np.linalg.norm(q)
    cache = KeypointCache()
    ctx, ranking = _ctx(db, q, depth=4, query_image=query, load_db_image=lambda r: load_image(paths[r]))
    sv = SpatialVerification(top_k=4, min_inliers=10, keypoint_cache=cache)
    out = sv.rerank(ctx, ranking)
    assert out.indices[0] == 2
    assert out.info["sv_inliers"][2] >= 10
    assert out.scores[0] > 1.0
    assert len(cache) == 4  # database keypoints were cached


def test_spatial_verification_requires_images() -> None:
    db = np.eye(3, dtype=np.float32)
    ctx, ranking = _ctx(db, db[0])
    with pytest.raises(ValueError):
        SpatialVerification().rerank(ctx, ranking)


def test_count_inliers_on_unrelated_images_is_small() -> None:
    rng = np.random.default_rng(0)
    a = Image.fromarray(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
    b = Image.fromarray(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
    sv = SpatialVerification(min_inliers=12)
    assert sv.count_inliers(sv.describe(a), sv.describe(b)) < 12
