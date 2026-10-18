"""L2 normalisation and held-out PCA-whitening."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from irbench.postprocess import PCAWhitening, l2_normalize


def test_l2_normalize_rows() -> None:
    x = np.array([[3.0, 4.0], [0.0, 0.0]])
    out = l2_normalize(x)
    assert np.allclose(out[0], [0.6, 0.8])
    assert np.allclose(out[1], [0.0, 0.0])  # zero vectors stay zero (no NaN)
    assert out.dtype == np.float32


def test_whitening_decorrelates_training_data() -> None:
    rng = np.random.default_rng(0)
    mix = rng.standard_normal((16, 16))
    X = rng.standard_normal((2000, 16)) @ mix  # strongly correlated features
    pca = PCAWhitening(dim=None, whiten=True).fit(X)
    Y = (X - pca.mean_) @ pca.projection_.T  # whitened, before the final L2
    cov = np.cov(Y, rowvar=False)
    assert np.allclose(cov, np.eye(16), atol=0.05)


def test_dimension_reduction_and_rank_clamp() -> None:
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 128)).astype(np.float32)
    assert PCAWhitening(dim=16).fit(X).transform(X).shape == (50, 16)
    big = PCAWhitening(dim=100).fit(X)
    assert big.effective_dim_ == 49  # rank of 50 centred samples
    assert np.allclose(np.linalg.norm(big.transform(X[:3]), axis=1), 1.0, atol=1e-5)


def test_plain_pca_keeps_variance_order() -> None:
    rng = np.random.default_rng(2)
    X = rng.standard_normal((500, 8)) * np.array([10, 5, 1, 1, 1, 1, 1, 1])
    pca = PCAWhitening(dim=2, whiten=False).fit(X)
    assert pca.eigenvalues_[0] > pca.eigenvalues_[1]
    total = float(np.var(X, axis=0, ddof=1).sum())
    assert 0.8 < pca.explained_variance_ratio(total) <= 1.0


def test_save_load_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    X = rng.standard_normal((100, 12))
    pca = PCAWhitening(dim=6).fit(X)
    path = tmp_path / "pca.npz"
    pca.save(path)
    loaded = PCAWhitening.load(path)
    assert np.allclose(loaded.transform(X), pca.transform(X))
    assert loaded.describe() == pca.describe()


def test_transform_before_fit_raises() -> None:
    with pytest.raises(RuntimeError):
        PCAWhitening().transform(np.zeros((1, 3)))
