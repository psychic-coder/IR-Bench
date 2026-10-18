"""Spatial verification: ORB keypoints + RANSAC geometric model on the top-k shortlist."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

import cv2
import numpy as np
from PIL import Image

from irbench.features.preprocessing import resize_max_side
from irbench.rerank.base import Ranking, RerankContext, Reranker


class KeypointCache:
    """Thread-safe LRU cache of ORB keypoints/descriptors per database row."""

    def __init__(self, max_items: int = 20_000) -> None:
        self.max_items = max_items
        self._data: OrderedDict[Any, tuple[np.ndarray, np.ndarray | None]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Any) -> tuple[np.ndarray, np.ndarray | None] | None:
        with self._lock:
            value = self._data.get(key)
            if value is not None:
                self._data.move_to_end(key)
            return value

    def put(self, key: Any, value: tuple[np.ndarray, np.ndarray | None]) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.max_items:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)


class SpatialVerification(Reranker):
    """Re-score the top-``top_k`` candidates by RANSAC inlier count.

    For each candidate, ORB descriptors are matched to the query's with a Lowe ratio test,
    then an affine (default) or homography model is fitted with RANSAC. Candidates with at
    least ``min_inliers`` inliers are moved ahead of the rest (sorted by inlier count, ties
    broken by the original similarity) and receive score ``1 + inliers``; all other
    results keep their original order and score.
    """

    name = "sv"

    def __init__(
        self,
        top_k: int = 20,
        min_inliers: int = 12,
        n_features: int = 1000,
        ratio: float = 0.8,
        reproj_threshold: float = 8.0,
        max_side: int = 640,
        model: str = "affine",
        seed: int = 42,
        keypoint_cache: KeypointCache | None = None,
        **params: Any,
    ) -> None:
        if model not in {"affine", "homography"}:
            raise ValueError("model must be 'affine' or 'homography'")
        super().__init__(
            top_k=int(top_k),
            min_inliers=int(min_inliers),
            n_features=int(n_features),
            ratio=float(ratio),
            reproj_threshold=float(reproj_threshold),
            max_side=int(max_side),
            model=model,
            seed=int(seed),
            **params,
        )
        self.cache = keypoint_cache if keypoint_cache is not None else KeypointCache()
        self._orb_local = threading.local()

    def config(self) -> dict[str, Any]:
        return {"name": self.name, **self.params}

    def _orb(self) -> cv2.ORB:
        orb = getattr(self._orb_local, "orb", None)
        if orb is None:
            orb = cv2.ORB_create(nfeatures=self.params["n_features"])
            self._orb_local.orb = orb
        return orb

    def describe(self, img: Image.Image) -> tuple[np.ndarray, np.ndarray | None]:
        """ORB keypoint coordinates ``(N, 2)`` and descriptors ``(N, 32)`` (or ``None``)."""
        small = resize_max_side(img, self.params["max_side"])
        gray = cv2.cvtColor(np.asarray(small, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
        kps, desc = self._orb().detectAndCompute(gray, None)
        pts = np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 2)
        return pts, desc

    def count_inliers(
        self, q: tuple[np.ndarray, np.ndarray | None], c: tuple[np.ndarray, np.ndarray | None]
    ) -> int:
        """RANSAC inliers between two ORB feature sets."""
        q_pts, q_desc = q
        c_pts, c_desc = c
        if q_desc is None or c_desc is None or len(q_desc) < 2 or len(c_desc) < 2:
            return 0
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        pairs = matcher.knnMatch(q_desc, c_desc, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < self.params["ratio"] * p[1].distance]
        min_matches = 3 if self.params["model"] == "affine" else 4
        if len(good) < min_matches:
            return 0
        src = q_pts[[m.queryIdx for m in good]].reshape(-1, 1, 2)
        dst = c_pts[[m.trainIdx for m in good]].reshape(-1, 1, 2)
        cv2.setRNGSeed(self.params["seed"])
        if self.params["model"] == "affine":
            _, mask = cv2.estimateAffine2D(
                src,
                dst,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.params["reproj_threshold"],
                maxIters=2000,
                confidence=0.99,
            )
        else:
            _, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.params["reproj_threshold"])
        return int(mask.sum()) if mask is not None else 0

    def rerank(self, ctx: RerankContext, ranking: Ranking) -> Ranking:
        if ctx.query_image is None or ctx.load_db_image is None:
            raise ValueError("spatial verification needs the query image and a database image loader")
        valid = ranking.valid()
        k = min(self.params["top_k"], valid.indices.size)
        if k == 0:
            return ranking
        q_feat = self.describe(ctx.query_image)
        cache_ns = ctx.extra.get("keypoint_namespace", "default")
        key_fn = ctx.extra.get("keypoint_key") or (lambda r: (cache_ns, r))
        inliers = np.zeros(k, dtype=np.int64)
        for j in range(k):
            row = int(valid.indices[j])
            key = key_fn(row)
            c_feat = self.cache.get(key)
            if c_feat is None:
                c_feat = self.describe(ctx.load_db_image(row))
                self.cache.put(key, c_feat)
            inliers[j] = self.count_inliers(q_feat, c_feat)

        head_idx = valid.indices[:k]
        head_scores = valid.scores[:k]
        verified = np.where(inliers >= self.params["min_inliers"])[0]
        unverified = np.where(inliers < self.params["min_inliers"])[0]
        # verified: inliers desc, then original score desc (stable on original rank)
        verified = verified[np.lexsort((-head_scores[verified], -inliers[verified]))]
        order = np.concatenate([verified, unverified])
        new_scores = head_scores.astype(np.float32).copy()
        new_scores[verified] = 1.0 + inliers[verified].astype(np.float32)
        scores = np.concatenate([new_scores[order], valid.scores[k:]])
        indices = np.concatenate([head_idx[order], valid.indices[k:]])
        info = {
            **ranking.info,
            "sv_inliers": {int(head_idx[j]): int(inliers[j]) for j in range(k)},
            "sv_verified": int(verified.size),
        }
        return Ranking(scores=scores, indices=indices, info=info)
