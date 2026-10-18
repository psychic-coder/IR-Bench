"""Classical hand-crafted global descriptors: HOG and HSV colour histograms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
from PIL import Image
from skimage.feature import hog

from irbench.features.base import FeatureExtractor, register_extractor
from irbench.features.pooling import CLASSICAL_POOLINGS
from irbench.features.preprocessing import RESAMPLE, resize_max_side


@register_extractor
class HOGExtractor(FeatureExtractor):
    """Histogram of Oriented Gradients on a fixed-size grayscale rendition of the image.

    The image is resized (aspect ratio *not* preserved) to ``hog_size x hog_size`` so
    every image yields the same cell grid; the concatenated block-normalised cell
    histograms form the global descriptor.
    """

    name = "hog"
    family = "classical"
    poolings = CLASSICAL_POOLINGS
    default_pooling = "none"
    description = "HOG on a fixed grayscale grid (9 orientations, 16px cells, 2x2 blocks)"

    def __init__(
        self,
        hog_size: int = 128,
        orientations: int = 9,
        pixels_per_cell: int = 16,
        cells_per_block: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.hog_size = int(hog_size)
        self.orientations = int(orientations)
        self.pixels_per_cell = int(pixels_per_cell)
        self.cells_per_block = int(cells_per_block)
        cells = self.hog_size // self.pixels_per_cell
        blocks = cells - self.cells_per_block + 1
        if blocks < 1:
            raise ValueError("hog_size too small for the cell/block configuration")
        self._dim = blocks * blocks * self.cells_per_block**2 * self.orientations

    @property
    def dim(self) -> int:
        return self._dim

    def _extra_config(self) -> dict[str, Any]:
        return {
            "hog_size": self.hog_size,
            "orientations": self.orientations,
            "pixels_per_cell": self.pixels_per_cell,
            "cells_per_block": self.cells_per_block,
        }

    def _extract_batch(self, images: Sequence[Image.Image], scale: float) -> np.ndarray:
        out = np.empty((len(images), self._dim), dtype=np.float32)
        for i, img in enumerate(images):
            gray = img.convert("L").resize((self.hog_size, self.hog_size), RESAMPLE)
            arr = np.asarray(gray, dtype=np.float32) / 255.0
            out[i] = hog(
                arr,
                orientations=self.orientations,
                pixels_per_cell=(self.pixels_per_cell, self.pixels_per_cell),
                cells_per_block=(self.cells_per_block, self.cells_per_block),
                block_norm="L2-Hys",
                feature_vector=True,
            )
        return out


@register_extractor
class HSVHistogramExtractor(FeatureExtractor):
    """Joint 3-D HSV colour histogram with Hellinger (square-root) normalisation."""

    name = "hsv_hist"
    family = "classical"
    poolings = CLASSICAL_POOLINGS
    default_pooling = "none"
    description = "Joint HSV colour histogram (8x8x8 bins), Hellinger-normalised"

    def __init__(self, bins: Sequence[int] = (8, 8, 8), work_size: int = 256, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.bins = tuple(int(b) for b in bins)
        if len(self.bins) != 3:
            raise ValueError("bins must have three entries (H, S, V)")
        self.work_size = int(work_size)

    @property
    def dim(self) -> int:
        return int(np.prod(self.bins))

    def _extra_config(self) -> dict[str, Any]:
        return {"bins": list(self.bins), "work_size": self.work_size}

    def _extract_batch(self, images: Sequence[Image.Image], scale: float) -> np.ndarray:
        out = np.empty((len(images), self.dim), dtype=np.float32)
        for i, img in enumerate(images):
            small = resize_max_side(img, self.work_size)
            hsv = cv2.cvtColor(np.asarray(small, dtype=np.uint8), cv2.COLOR_RGB2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, list(self.bins), [0, 180, 0, 256, 0, 256])
            hist = hist.astype(np.float32).ravel()
            hist /= max(float(hist.sum()), 1e-12)
            out[i] = np.sqrt(hist)
        return out
