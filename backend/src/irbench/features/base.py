"""Feature-extractor interface and registry.

Every extractor exposes ``extract(images) -> np.ndarray`` returning L2-normalised,
float32 global descriptors of shape ``(len(images), dim)``. Extractors register
themselves with :func:`register_extractor`; :func:`create_extractor` builds them by name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ExtractorInfo:
    """Static metadata about a registered extractor."""

    name: str
    family: str  # "classical" | "cnn" | "transformer"
    poolings: tuple[str, ...]
    default_pooling: str
    supports_multiscale: bool
    description: str


class FeatureExtractor(ABC):
    """Base class: subclasses implement :meth:`_extract_batch`."""

    name: ClassVar[str] = "base"
    family: ClassVar[str] = "base"
    poolings: ClassVar[tuple[str, ...]] = ("none",)
    default_pooling: ClassVar[str] = "none"
    supports_multiscale: ClassVar[bool] = False
    description: ClassVar[str] = ""

    def __init__(
        self,
        pooling: str | None = None,
        gem_p: float = 3.0,
        image_size: int | None = None,
        scales: Sequence[float] = (1.0,),
        device: str = "cpu",
        batch_size: int = 16,
        **_: Any,
    ) -> None:
        self.pooling = pooling or self.default_pooling
        if self.pooling not in self.poolings:
            raise ValueError(
                f"pooling '{self.pooling}' is not valid for '{self.name}'; valid: {self.poolings}"
            )
        self.gem_p = float(gem_p)
        self.image_size = image_size
        self.scales = tuple(float(s) for s in scales)
        if not self.scales:
            raise ValueError("scales must not be empty")
        if self.scales != (1.0,) and not self.supports_multiscale:
            raise ValueError(f"'{self.name}' does not support multi-scale extraction")
        self.device = device
        self.batch_size = max(1, int(batch_size))

    # --------------------------------------------------------------------- API
    @property
    @abstractmethod
    def dim(self) -> int:
        """Output dimensionality."""

    def config(self) -> dict[str, Any]:
        """Everything that changes the produced descriptors (used for cache keys)."""
        cfg: dict[str, Any] = {
            "extractor": self.name,
            "pooling": self.pooling,
            "image_size": self.image_size,
            "scales": list(self.scales),
        }
        if "gem" in self.pooling:
            cfg["gem_p"] = self.gem_p
        cfg.update(self._extra_config())
        return cfg

    def _extra_config(self) -> dict[str, Any]:
        return {}

    def extract(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Return ``(N, dim)`` float32 L2-normalised descriptors for RGB PIL images."""
        if len(images) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        acc: np.ndarray | None = None
        for scale in self.scales:
            feats = self._extract_batch(images, scale).astype(np.float32, copy=False)
            feats = _l2n(feats)
            acc = feats if acc is None else acc + feats
        assert acc is not None
        return _l2n(acc / len(self.scales)).astype(np.float32)

    @abstractmethod
    def _extract_batch(self, images: Sequence[Image.Image], scale: float) -> np.ndarray:
        """Compute raw pooled descriptors at one scale."""

    def warmup(self) -> None:
        """Run one dummy forward pass (model load + kernel compilation)."""
        self.extract([Image.new("RGB", (224, 224), color=(127, 127, 127))])


def _l2n(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


EXTRACTORS: dict[str, type[FeatureExtractor]] = {}


def register_extractor(cls: type[FeatureExtractor]) -> type[FeatureExtractor]:
    """Class decorator adding an extractor to the registry under ``cls.name``."""
    if cls.name in EXTRACTORS:
        raise ValueError(f"extractor '{cls.name}' already registered")
    EXTRACTORS[cls.name] = cls
    return cls


def extractor_info(name: str) -> ExtractorInfo:
    """Static metadata for a registered extractor."""
    cls = get_extractor_class(name)
    return ExtractorInfo(
        name=cls.name,
        family=cls.family,
        poolings=cls.poolings,
        default_pooling=cls.default_pooling,
        supports_multiscale=cls.supports_multiscale,
        description=cls.description,
    )


def get_extractor_class(name: str) -> type[FeatureExtractor]:
    _ensure_builtin_registered()
    if name not in EXTRACTORS:
        raise KeyError(f"unknown extractor '{name}'. Registered: {sorted(EXTRACTORS)}")
    return EXTRACTORS[name]


def create_extractor(name: str, **kwargs: Any) -> FeatureExtractor:
    """Instantiate a registered extractor by name."""
    return get_extractor_class(name)(**kwargs)


def list_extractors() -> list[ExtractorInfo]:
    _ensure_builtin_registered()
    return [extractor_info(n) for n in sorted(EXTRACTORS)]


def _ensure_builtin_registered() -> None:
    # Import side effects register the built-in extractors exactly once.
    from irbench.features import classical, cnn, transformer  # noqa: F401
