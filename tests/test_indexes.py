"""Index shapes, exact-neighbour recovery on toy data, and memory accounting."""

from __future__ import annotations

import numpy as np
import pytest

from irbench.indexes import create_index


def _clustered(n_clusters: int = 8, per: int = 40, d: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, d))
    x = np.repeat(centers, per, axis=0) + 0.05 * rng.standard_normal((n_clusters * per, d))
    x = x.astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _exact_topk(db: np.ndarray, q: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-(q @ db.T), axis=1)[:, :k]


@pytest.mark.parametrize(
    ("kind", "params"),
    [
        ("flat", {}),
        ("hnsw", {"location": ":memory:", "m": 16, "ef_construct": 64, "ef": 64}),
        ("ivfpq", {"nlist": 4, "m": 8, "nbits": 8, "nprobe": 4}),
    ],
)
def test_shapes_and_self_retrieval(kind: str, params: dict) -> None:
    db = _clustered()
    index = create_index(kind, **params)
    index.build(db)
    res = index.search(db[:10], 5)
    assert res.scores.shape == (10, 5) and res.indices.shape == (10, 5)
    assert res.indices.dtype == np.int64
    if kind != "ivfpq":
        assert np.array_equal(res.indices[:, 0], np.arange(10))
    index.close()


def test_flat_recovers_exact_neighbours(unit_vectors: np.ndarray) -> None:
    index = create_index("flat")
    index.build(unit_vectors)
    q = unit_vectors[:20] + 0.01
    res = index.search(q, 10)
    assert np.array_equal(res.indices, _exact_topk(unit_vectors, q.astype(np.float32), 10))
    assert np.all(np.diff(res.scores, axis=1) <= 1e-6)


def test_hnsw_embedded_recovers_exact_neighbours(unit_vectors: np.ndarray) -> None:
    index = create_index("hnsw", location=":memory:", m=16, ef_construct=64, ef=128)
    index.build(unit_vectors)
    q = unit_vectors[5:15]
    res = index.search(q, 10)
    assert np.array_equal(res.indices, _exact_topk(unit_vectors, q, 10))
    batch = index.search(q, 10)  # multi-query path
    assert np.array_equal(batch.indices, res.indices)
    index.close()


def test_ivfpq_with_all_cells_finds_true_cluster() -> None:
    db = _clustered()
    index = create_index("ivfpq", nlist=4, m=16, nbits=8, nprobe=4, seed=1)
    index.build(db)
    res = index.search(db[::40], 10)  # one query per cluster
    clusters = res.indices // 40
    assert np.all(clusters == np.arange(8)[:, None])


def test_ivfpq_clamps_parameters_and_reports_them() -> None:
    db = _clustered(n_clusters=2, per=50)  # 100 vectors
    index = create_index("ivfpq", nlist=64, m=8, nbits=8, nprobe=128)
    index.build(db)
    eff = index.effective_params()
    assert eff["nlist"] == 100 // 39
    assert eff["nbits"] == 6  # 2**6 = 64 <= 100 < 128
    assert eff["nprobe"] == eff["nlist"]


def test_ivfpq_rejects_m_not_dividing_d() -> None:
    with pytest.raises(ValueError):
        create_index("ivfpq", nlist=2, m=7, nbits=4).build(_clustered())


def test_pq_memory_accounting() -> None:
    db = _clustered(n_clusters=8, per=64, d=64)  # n = 512
    n, d = db.shape
    m, nbits, nlist = 16, 8, 8
    index = create_index("ivfpq", nlist=nlist, m=m, nbits=nbits, nprobe=2)
    index.build(db)
    mem = index.memory_report()
    code_bytes = m * nbits // 8
    assert mem.encoding_bytes_per_vector == code_bytes
    assert mem.breakdown["pq_codes"] == n * code_bytes
    assert mem.breakdown["inverted_list_ids_int64"] == n * 8
    assert mem.breakdown["coarse_centroids_float32"] == nlist * d * 4
    assert mem.breakdown["pq_codebooks_float32"] == (2**nbits) * d * 4
    assert mem.total_bytes == sum(mem.breakdown.values())
    assert mem.bytes_per_vector == pytest.approx(mem.total_bytes / n)
    # the analytic total must agree with FAISS' own serialisation to within the header size
    assert mem.serialized_bytes is not None
    assert abs(mem.serialized_bytes - mem.total_bytes) / mem.total_bytes < 0.05


def test_flat_memory_accounting(unit_vectors: np.ndarray) -> None:
    index = create_index("flat")
    index.build(unit_vectors)
    mem = index.memory_report()
    assert mem.total_bytes == unit_vectors.size * 4
    assert mem.encoding_bytes_per_vector == unit_vectors.shape[1] * 4
    assert mem.serialized_bytes is not None and mem.serialized_bytes >= mem.total_bytes


def test_hnsw_memory_estimate_formula(unit_vectors: np.ndarray) -> None:
    index = create_index("hnsw", location=":memory:", m=8)
    index.build(unit_vectors)
    mem = index.memory_report()
    n, d = unit_vectors.shape
    assert mem.method == "estimate"
    assert mem.breakdown["vectors_float32"] == n * d * 4
    assert mem.breakdown["graph_links_level0"] == n * 2 * 8 * 4
    index.close()


def test_search_params_change_without_rebuild() -> None:
    db = _clustered()
    index = create_index("ivfpq", nlist=4, m=8, nbits=8, nprobe=1)
    index.build(db)
    index.set_search_params(nprobe=4)
    assert index.effective_params()["nprobe"] == 4
    with pytest.raises(ValueError):
        index.set_search_params(ef=3)


def test_dimension_mismatch_is_rejected(unit_vectors: np.ndarray) -> None:
    index = create_index("flat")
    index.build(unit_vectors)
    with pytest.raises(ValueError):
        index.search(np.zeros((1, 7), dtype=np.float32), 3)
