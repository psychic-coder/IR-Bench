"""Shared machinery for PyTorch extractors: deterministic preprocessing and shape-bucketed batching."""

from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import torch
from PIL import Image

from irbench.features.base import FeatureExtractor
from irbench.features.preprocessing import RESAMPLE, resize_max_side

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TorchExtractor(FeatureExtractor):
    """Base class for torch backbones.

    Images are resized deterministically (bicubic, longer side = ``image_size * scale``,
    aspect preserved, optionally snapped to a patch multiple), normalised, and batched
    *by identical tensor shape* so that variable-size inputs never need padding (padding
    would leak zeros into GAP/GeM/R-MAC statistics).
    """

    mean: ClassVar[tuple[float, float, float]] = IMAGENET_MEAN
    std: ClassVar[tuple[float, float, float]] = IMAGENET_STD
    patch_multiple: ClassVar[int] = 1
    fixed_size: ClassVar[tuple[int, int] | None] = None
    default_image_size: ClassVar[int] = 512

    def __init__(self, pretrained: bool = True, **kwargs: Any) -> None:
        kwargs.setdefault("image_size", self.default_image_size)
        super().__init__(**kwargs)
        self.pretrained = bool(pretrained)
        self.model = self._build_model().to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self._mean = torch.tensor(self.mean, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor(self.std, dtype=torch.float32).view(3, 1, 1)

    @abstractmethod
    def _build_model(self) -> torch.nn.Module:
        """Construct the backbone (with or without pretrained weights)."""

    @abstractmethod
    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Map a normalised ``(B, 3, H, W)`` batch to pooled ``(B, D)`` descriptors."""

    def _extra_config(self) -> dict[str, Any]:
        return {"pretrained": self.pretrained}

    def _prepare(self, img: Image.Image, scale: float) -> torch.Tensor:
        if self.fixed_size is not None:
            resized = img.resize(self.fixed_size, RESAMPLE)
        else:
            size = (
                None if self.image_size is None else max(self.patch_multiple, round(self.image_size * scale))
            )
            if size is None and scale != 1.0:
                size = max(self.patch_multiple, round(max(img.size) * scale))
            resized = resize_max_side(img, size, multiple_of=self.patch_multiple)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr.transpose(2, 0, 1).copy())
        return (tensor - self._mean) / self._std

    def _extract_batch(self, images: Sequence[Image.Image], scale: float) -> np.ndarray:
        tensors = [self._prepare(img, scale) for img in images]
        buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for i, t in enumerate(tensors):
            buckets[tuple(t.shape)].append(i)
        out = np.empty((len(images), self.dim), dtype=np.float32)
        with torch.inference_mode():
            for indices in buckets.values():
                for start in range(0, len(indices), self.batch_size):
                    chunk = indices[start : start + self.batch_size]
                    batch = torch.stack([tensors[i] for i in chunk]).to(self.device)
                    feats = self._forward(batch).float().cpu().numpy()
                    out[chunk] = feats
            if self.device == "cuda":
                torch.cuda.synchronize()
        return out
