"""Descriptor post-processing: L2 normalisation and (held-out) PCA-whitening."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation (returns float32)."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, eps)).astype(np.float32)


@dataclass
class PCAWhitening:
    """PCA projection with optional whitening and dimensionality reduction.

    Fit on a *held-out* collection (the other dataset), never on the evaluation database
    or queries. ``transform`` computes ``l2n(P (x - mu))`` with
    ``P = diag(lambda_1..k ^ -power) U_k^T`` when whitening (``power = 0.5`` is classic
    whitening) or ``P = U_k^T`` for plain PCA.
    """

    dim: int | None = None
    whiten: bool = True
    power: float = 0.5
    eps: float = 1e-6
    mean_: np.ndarray | None = None
    projection_: np.ndarray | None = None
    eigenvalues_: np.ndarray | None = None
    effective_dim_: int | None = None

    def fit(self, X: np.ndarray) -> PCAWhitening:
        """Estimate mean, principal axes and eigenvalues from ``X`` (``(n, d)``)."""
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        if n < 2:
            raise ValueError("PCA needs at least two training vectors")
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        eig = (S**2) / (n - 1)
        rank = int(np.sum(eig > eig.max() * 1e-10)) if eig.size else 0
        k = min(self.dim or d, rank, d)
        if k < 1:
            raise ValueError("training data is degenerate (rank 0)")
        self.effective_dim_ = k
        eig_k = eig[:k]
        components = Vt[:k]
        if self.whiten:
            floor = self.eps * float(eig_k.max())
            scale = np.power(eig_k + floor, -self.power)
            proj = components * scale[:, None]
        else:
            proj = components
        self.projection_ = proj.astype(np.float32)
        self.mean_ = self.mean_.astype(np.float32)
        self.eigenvalues_ = eig_k.astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project, (whiten), and L2-normalise."""
        if self.projection_ is None or self.mean_ is None:
            raise RuntimeError("PCAWhitening.transform called before fit")
        Y = (np.asarray(X, dtype=np.float32) - self.mean_) @ self.projection_.T
        return l2_normalize(Y)

    def explained_variance_ratio(self, total_variance: float | None = None) -> float | None:
        """Fraction of training variance kept (needs the training total variance)."""
        if self.eigenvalues_ is None or not total_variance:
            return None
        return float(self.eigenvalues_.sum() / total_variance)

    def save(self, path: Path) -> None:
        if self.projection_ is None or self.mean_ is None or self.eigenvalues_ is None:
            raise RuntimeError("cannot save an unfitted PCAWhitening")
        np.savez(
            path,
            mean=self.mean_,
            projection=self.projection_,
            eigenvalues=self.eigenvalues_,
            meta=np.array(
                [self.dim or -1, int(self.whiten), self.power, self.eps, self.effective_dim_ or -1]
            ),
        )

    @classmethod
    def load(cls, path: Path) -> PCAWhitening:
        data = np.load(path)
        dim, whiten, power, eps, eff = data["meta"].tolist()
        obj = cls(dim=None if dim < 0 else int(dim), whiten=bool(whiten), power=float(power), eps=float(eps))
        obj.mean_ = data["mean"]
        obj.projection_ = data["projection"]
        obj.eigenvalues_ = data["eigenvalues"]
        obj.effective_dim_ = None if eff < 0 else int(eff)
        return obj

    def describe(self) -> dict[str, Any]:
        return {
            "dim_requested": self.dim,
            "dim_effective": self.effective_dim_,
            "whiten": self.whiten,
            "power": self.power,
        }
