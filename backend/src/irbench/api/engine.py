"""Serving engine: dataset, lazily built feature spaces and indexes, warm-up and search.

Thread model: FastAPI runs the sync endpoints in a thread pool. Building a feature space
or an index is guarded by a per-key lock (concurrent requests for the same config wait
for one build); model inference is serialised per extractor; index reads are concurrent.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from irbench.api.cache import EmbeddingCache
from irbench.api.errors import APIError
from irbench.api.metrics import ApiMetrics
from irbench.data import HELDOUT, DatasetUnavailableError, RetrievalDataset, available_datasets, load_dataset
from irbench.data.oxford_paris import scan_images
from irbench.features import FeatureExtractor, create_extractor, extractor_info
from irbench.features.cache import FeatureCache, extract_items
from irbench.features.preprocessing import load_image, make_thumbnail
from irbench.indexes import VectorIndex, create_index
from irbench.logging_utils import get_logger
from irbench.postprocess import PCAWhitening
from irbench.rerank import KeypointCache, Ranking, RerankContext, create_reranker, rerank_needs_images
from irbench.settings import ServingBackbone, ServingConfig, Settings
from irbench.utils import ensure_dir, resolve_device, seed_everything, stable_hash

log = get_logger(__name__)


@dataclass
class FeatureSpace:
    """Database descriptors for one (backbone, pooling) pair, ready to index."""

    key: str
    backbone: str
    pooling: str
    extractor: FeatureExtractor
    db: np.ndarray
    ids: list[str]
    paths: list[str]
    pca: PCAWhitening | None
    infer_lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class PrebuildStatus:
    backbone: str
    pooling: str
    index: str
    status: str = "pending"
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "backbone": self.backbone,
            "pooling": self.pooling,
            "index": self.index,
            "status": self.status,
            "error": self.error,
        }


class RetrievalEngine:
    """Owns everything needed to answer ``/search``."""

    def __init__(
        self, settings: Settings, serving: ServingConfig, cache: EmbeddingCache, metrics: ApiMetrics
    ) -> None:
        self.settings = settings
        self.serving = serving
        self.cache = cache
        self.metrics = metrics
        self.device = resolve_device(settings.device)
        self.feature_cache = FeatureCache(settings.feature_cache_dir)
        self.keypoints = KeypointCache(max_items=20_000)
        self.dataset: RetrievalDataset | None = None
        self.ready = False
        self.stage = "starting"
        self.message = "service starting"
        self.error: str | None = None
        self.progress: dict[str, int] | None = None
        self.prebuilt = [PrebuildStatus(p.backbone, p.pooling, p.index) for p in serving.prebuild]
        self._spaces: dict[str, FeatureSpace] = {}
        self._indexes: dict[tuple[str, str], VectorIndex] = {}
        self._build_locks: defaultdict[Any, threading.Lock] = defaultdict(threading.Lock)
        self._registry_lock = threading.Lock()
        self._warmup_thread: threading.Thread | None = None
        self._image_index: dict[str, dict[str, str]] = {}
        self._heldout: RetrievalDataset | None = None

    # ------------------------------------------------------------------ status
    def _set(self, stage: str, message: str, progress: dict[str, int] | None = None) -> None:
        self.stage, self.message, self.progress = stage, message, progress
        log.info("warmup", extra={"stage": stage, "detail": message})

    def status(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "stage": self.stage,
            "message": self.message,
            "progress": self.progress,
            "error": self.error,
            "built": [p.as_dict() for p in self.prebuilt],
            "dataset": self.serving.dataset,
            "n_images": len(self._db_ids()) if self.dataset else None,
        }

    # ------------------------------------------------------------------ warm-up
    def start_warmup(self, only: tuple[str, str, str] | None = None) -> bool:
        """Start (or extend) warm-up in a background thread; False if one is running."""
        if self._warmup_thread is not None and self._warmup_thread.is_alive():
            return False
        if only is not None:
            b, p, i = only
            if not any((s.backbone, s.pooling, s.index) == only for s in self.prebuilt):
                self.prebuilt.append(PrebuildStatus(b, p, i))
        self._warmup_thread = threading.Thread(target=self.warmup, name="warmup", daemon=True)
        self._warmup_thread.start()
        return True

    def warmup(self) -> None:
        """Load the dataset, then build every prebuild configuration."""
        try:
            seed_everything(self.settings.seed)
            self.error = None
            if self.dataset is None:
                self._set("dataset", f"loading dataset {self.serving.dataset} (downloads on first run)")
                self.dataset = load_dataset(
                    self.serving.dataset,
                    self.settings.data_dir,
                    subset=self.settings.api_dataset_subset,
                    seed=self.settings.seed,
                    auto_download=self.settings.auto_download,
                )
            for pb in self.prebuilt:
                if pb.status == "ready":
                    continue
                pb.status, pb.error = "building", None
                try:
                    self.validate(pb.backbone, pb.pooling, pb.index, "none", self.serving.default_k)
                    self.get_index(self.get_space(pb.backbone, pb.pooling), pb.index)
                    pb.status = "ready"
                except Exception as exc:
                    pb.status, pb.error = "failed", f"{type(exc).__name__}: {exc}"
                    log.error("prebuild failed", extra={"config": pb.as_dict()})
            ok = [p for p in self.prebuilt if p.status == "ready"]
            if self.prebuilt and not ok:
                self.error = "every prebuild configuration failed; see 'built' for errors"
                self._set("error", self.error)
                return
            self.ready = True
            self.metrics.ready.set(1)
            self._set("ready", f"serving {len(self._db_ids())} images from {self.serving.dataset}")
        except DatasetUnavailableError as exc:
            self.error = str(exc)
            self._set("error", "dataset unavailable - see 'error' for manual download steps")
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            log.exception("warm-up failed")
            self._set("error", "warm-up failed")

    # ------------------------------------------------------------------ config validation
    def backbone(self, name: str) -> ServingBackbone:
        for b in self.serving.backbones:
            if b.name == name:
                return b
        raise APIError(
            422,
            "invalid_config",
            f"unknown backbone '{name}'",
            {"valid": [b.name for b in self.serving.backbones]},
        )

    def validate(
        self, backbone: str, pooling: str | None, index: str, rerank: str, k: int
    ) -> tuple[str, str, str, str, int]:
        bb = self.backbone(backbone)
        pooling = pooling or bb.default_pooling
        if pooling not in bb.poolings:
            raise APIError(
                422,
                "invalid_config",
                f"pooling '{pooling}' not available for {backbone}",
                {"valid": bb.poolings},
            )
        if index not in self.serving.indexes:
            raise APIError(422, "invalid_config", f"unknown index '{index}'", {"valid": self.serving.indexes})
        if rerank not in self.serving.rerankers:
            raise APIError(
                422, "invalid_config", f"unknown reranker '{rerank}'", {"valid": self.serving.rerankers}
            )
        if not 1 <= k <= self.serving.max_k:
            raise APIError(422, "invalid_config", f"k must be between 1 and {self.serving.max_k}")
        return backbone, pooling, index, rerank, k

    # ------------------------------------------------------------------ feature spaces / indexes
    def _protocol(self) -> Any:
        assert self.dataset is not None
        if self.serving.protocol in self.dataset.protocols:
            return self.dataset.protocols[self.serving.protocol]
        return next(iter(self.dataset.protocols.values()))

    def _db_ids(self) -> list[str]:
        return self._protocol().db_ids

    def _make_extractor(self, bb: ServingBackbone, pooling: str) -> FeatureExtractor:
        info = extractor_info(bb.name)
        kwargs: dict[str, Any] = {
            "pooling": pooling,
            "device": self.device,
            "batch_size": self.settings.batch_size,
        }
        if bb.image_size is not None:
            kwargs["image_size"] = bb.image_size
        if info.family != "classical":
            kwargs["pretrained"] = bb.pretrained
        return create_extractor(bb.name, **kwargs)

    def get_space(self, backbone: str, pooling: str) -> FeatureSpace:
        """Return (building if needed) the feature space for a backbone/pooling pair."""
        if self.dataset is None:
            raise APIError(503, "not_ready", "dataset not loaded yet")
        bb = self.backbone(backbone)
        key = stable_hash(
            {
                "b": bb.model_dump(),
                "p": pooling,
                "ds": self.serving.dataset,
                "pr": self.serving.protocol,
                "s": self.settings.api_dataset_subset,
                "pca": self.serving.pca_whitening,
            }
        )
        if key in self._spaces:
            return self._spaces[key]
        with self._build_locks[("space", key)]:
            if key in self._spaces:
                return self._spaces[key]
            ext = self._make_extractor(bb, pooling)
            proto = self._protocol()
            items = proto.db_items(self.dataset.paths)
            subset = self.settings.api_dataset_subset or "full"
            label = f"{backbone}/{pooling}"

            def progress(done: int, total: int) -> None:
                self._set(
                    "extracting", f"extracting {label} database descriptors", {"done": done, "total": total}
                )

            fs = extract_items(
                ext,
                items,
                self.feature_cache,
                tag=f"{self.serving.dataset}:api:{proto.name}:{subset}",
                device=self.device,
                show_progress=False,
                progress_cb=progress,
            )
            db = fs.vectors
            pca = None
            if self.serving.pca_whitening:
                held_name = HELDOUT[self.serving.dataset]
                if self._heldout is None:
                    self._set("dataset", f"loading held-out dataset {held_name} for PCA-whitening")
                    self._heldout = load_dataset(
                        held_name,
                        self.settings.data_dir,
                        subset=self.settings.api_dataset_subset,
                        seed=self.settings.seed,
                        auto_download=self.settings.auto_download,
                    )
                h_items = self._heldout.all_db_items()
                h_fs = extract_items(
                    ext,
                    h_items,
                    self.feature_cache,
                    tag=f"{held_name}:api-heldout:{subset}",
                    device=self.device,
                    show_progress=False,
                    progress_cb=progress,
                )
                pca = PCAWhitening(dim=None, whiten=True).fit(h_fs.vectors)
                db = pca.transform(db)
            space = FeatureSpace(
                key=key,
                backbone=backbone,
                pooling=pooling,
                extractor=ext,
                db=db,
                ids=[f"{self.serving.dataset}/{i}" for i in proto.db_ids],
                paths=[self.dataset.paths[i] for i in proto.db_ids],
                pca=pca,
            )
            with self._registry_lock:
                self._spaces[key] = space
            return space

    def index_params(self, index: str) -> dict[str, Any]:
        return dict(getattr(self.serving.index_defaults, index))

    def get_index(self, space: FeatureSpace, index: str) -> VectorIndex:
        """Return (building if needed) an index over a feature space."""
        key = (space.key, index)
        if key in self._indexes:
            return self._indexes[key]
        with self._build_locks[("index", key)]:
            if key in self._indexes:
                return self._indexes[key]
            params = self.index_params(index)
            self._set("indexing", f"building {index} index for {space.backbone}/{space.pooling}")
            if index == "hnsw":
                url = self.settings.qdrant_url
                loc = {"location": ":memory:"} if url == ":memory:" else {"url": url}
                params.update(
                    loc, collection=f"api_{space.key[:12]}_{stable_hash(params)[:6]}", reuse=True, keep=True
                )
            if index == "ivfpq":
                params.setdefault("seed", self.settings.seed)
            idx = create_index(index, **params)
            try:
                idx.build(space.db)
            except ValueError as exc:
                raise APIError(
                    422, "invalid_config", f"cannot build {index} for {space.backbone}: {exc}"
                ) from exc
            mem = idx.memory_report()
            labels = {"backbone": space.backbone, "pooling": space.pooling, "index": index}
            self.metrics.index_vectors.labels(**labels).set(mem.n_vectors)
            self.metrics.index_bytes.labels(**labels).set(mem.total_bytes)
            with self._registry_lock:
                self._indexes[key] = idx
            if self.ready:
                self._set("ready", f"serving {len(self._db_ids())} images from {self.serving.dataset}")
            return idx

    # ------------------------------------------------------------------ search
    def search(
        self,
        image: Image.Image,
        image_hash: str,
        backbone: str,
        pooling: str | None,
        index: str,
        rerank: str,
        k: int,
        bbox: list[float] | None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        if not self.ready:
            raise APIError(503, "not_ready", "backend is warming up", self.status())
        backbone, pooling, index, rerank, k = self.validate(backbone, pooling, index, rerank, k)
        space = self.get_space(backbone, pooling)
        vindex = self.get_index(space, index)

        cache_key = self.cache.key(image_hash, space.key)
        t0 = time.perf_counter()
        vec = self.cache.get(cache_key) if use_cache else None
        cache_state = (
            "disabled" if not (use_cache and self.cache.enabled) else ("hit" if vec is not None else "miss")
        )
        if vec is None or vec.shape[0] != space.db.shape[1]:
            with space.infer_lock:
                raw = space.extractor.extract([image])[0]
            vec = space.pca.transform(raw[None, :])[0] if space.pca is not None else raw
            if use_cache:
                self.cache.set(cache_key, vec)
            if cache_state == "hit":
                cache_state = "miss"
        t1 = time.perf_counter()

        rr_params = self.serving.rerank_defaults.model_dump()
        reranker = create_reranker(rerank, rr_params, keypoint_cache=self.keypoints, seed=self.settings.seed)
        depth = k
        if rerank_needs_images(rerank):
            depth = max(k, int(rr_params["sv"].get("top_k", 20)))
        depth = min(depth, space.db.shape[0])
        res = vindex.search(vec[None, :].astype(np.float32), depth)
        t2 = time.perf_counter()
        ranking = Ranking(scores=res.scores[0], indices=res.indices[0])
        if reranker is not None:
            ctx = RerankContext(
                query_vector=vec.astype(np.float32),
                db_vectors=space.db,
                index=vindex,
                depth=depth,
                query_image=image,
                load_db_image=lambda r: load_image(space.paths[r]),
                extra={"keypoint_key": lambda r: space.paths[r]},
            )
            ranking = reranker.rerank(ctx, ranking)
        ranking = ranking.valid()
        t3 = time.perf_counter()

        inliers = ranking.info.get("sv_inliers", {})
        results = []
        for rank, (row, score) in enumerate(
            zip(ranking.indices[:k], ranking.scores[:k], strict=False), start=1
        ):
            iid = space.ids[int(row)]
            results.append(
                {
                    "rank": rank,
                    "id": iid,
                    "score": float(score),
                    "image_url": f"/images/{iid}",
                    "thumbnail_url": f"/thumbnails/{iid}",
                    "inliers": inliers.get(int(row)),
                }
            )
        lat = {
            "extract": (t1 - t0) * 1000.0,
            "search": (t2 - t1) * 1000.0,
            "rerank": (t3 - t2) * 1000.0,
            "total": (t3 - t0) * 1000.0,
        }
        for stage, ms in lat.items():
            self.metrics.stage_latency.labels(
                stage=stage, backbone=backbone, index=index, rerank=rerank
            ).observe(ms)
        self.metrics.cache.labels(result=cache_state).inc()
        mem = vindex.memory_report()
        return {
            "query_hash": image_hash,
            "dataset": self.serving.dataset,
            "config": {
                "backbone": backbone,
                "pooling": pooling,
                "index": index,
                "index_params": {k: v for k, v in vindex.effective_params().items() if k != "seed"},
                "rerank": rerank,
                "k": k,
                "bbox": bbox,
                "pca_whitening": space.pca is not None,
            },
            "results": results,
            "latency_ms": lat,
            "cache": cache_state,
            "cache_hit": cache_state == "hit",
            "index": {
                "type": index,
                "n_vectors": mem.n_vectors,
                "dim": int(space.db.shape[1]),
                "memory_bytes": mem.total_bytes,
                "bytes_per_vector": mem.bytes_per_vector,
                "memory_method": mem.method,
            },
        }

    # ------------------------------------------------------------------ images
    def sample_images(self, n: int = 12) -> list[str]:
        """A deterministic, evenly spaced sample of database image ids (empty before load)."""
        if self.dataset is None:
            return []
        ids = self._db_ids()
        step = max(1, len(ids) // n)
        return [f"{self.serving.dataset}/{i}" for i in ids[::step][:n]]

    def resolve_image(self, image_id: str) -> Path:
        """Map a global id (``<dataset>/<image>`` or ``distractors/<rel>``) to a file path."""
        if "/" not in image_id:
            raise APIError(404, "not_found", f"image '{image_id}' not found")
        prefix, rest = image_id.split("/", 1)
        if prefix == "distractors":
            for base in (self.settings.resolved_distractor_dir, self.settings.data_dir / "toy_distractors"):
                cand = (base / rest).resolve()
                if cand.is_file() and cand.is_relative_to(base.resolve()):
                    return cand
            raise APIError(404, "not_found", f"image '{image_id}' not found")
        if prefix not in available_datasets():
            raise APIError(404, "not_found", f"unknown dataset '{prefix}'")
        if prefix not in self._image_index:
            root = self.settings.data_dir / prefix
            self._image_index[prefix] = scan_images(root) if root.exists() else {}
        path = self._image_index[prefix].get(rest)
        if path is None:
            raise APIError(404, "not_found", f"image '{image_id}' not found")
        return Path(path)

    def thumbnail(self, image_id: str, size: int) -> Path:
        src = self.resolve_image(image_id)
        digest = hashlib.sha1(f"{image_id}:{size}".encode()).hexdigest()
        out = ensure_dir(self.settings.thumbnail_dir) / f"{digest}.jpg"
        if not out.exists():
            thumb = make_thumbnail(load_image(src), size)
            tmp = out.with_suffix(".tmp.jpg")
            thumb.save(tmp, "JPEG", quality=85)
            tmp.replace(out)
        return out

    def close(self) -> None:
        for idx in self._indexes.values():
            idx.close()
