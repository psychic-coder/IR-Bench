"""The benchmark harness: one function maps a run configuration to metrics.

``(extractor, pooling, post-processing, index, re-ranker, protocol, dataset) -> metrics``

Expensive artefacts are shared between runs: descriptors are cached on disk (keyed by
config hash), PCA models and exact neighbour lists are memoised, and the last built
index is reused when only search-time parameters (``ef``, ``nprobe``) change, so sweeps
measure the parameter rather than rebuild noise.
"""

from __future__ import annotations

import gc
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from irbench.bench.config import ABLATIONS, BenchmarkConfig, ExperimentCfg, RunConfig, expand_experiment
from irbench.bench.store import ResultsStore
from irbench.data import HELDOUT, DatasetUnavailableError, RetrievalDataset, load_dataset
from irbench.data.distractors import DistractorSet, load_distractors
from irbench.data.download import ensure_distractors
from irbench.data.toy import TOY_DATASETS, generate_toy_distractors
from irbench.eval.metrics import RECALL_KS, ann_recall, evaluate_protocol
from irbench.features import FeatureExtractor, create_extractor, extractor_info
from irbench.features.cache import FeatureCache, FeatureSet, ImageItem, extract_items
from irbench.features.preprocessing import crop_bbox, load_image
from irbench.indexes import VectorIndex, create_index
from irbench.logging_utils import get_logger
from irbench.postprocess import PCAWhitening
from irbench.rerank import KeypointCache, Ranking, RerankContext, create_reranker, rerank_needs_images
from irbench.settings import Settings
from irbench.utils import (
    configure_threads,
    hardware_info,
    latency_stats,
    library_versions,
    resolve_device,
    seed_everything,
    stable_hash,
    utc_now_iso,
)

log = get_logger(__name__)

RESULT_SCHEMA_VERSION = 3
SEARCH_PARAMS = {"hnsw": ("ef",), "ivfpq": ("nprobe",), "flat": ()}
RANKING_KEEP = 50


@dataclass
class PlannedRun:
    """A unique (run config, dataset, protocol) triple and the experiments that use it."""

    run_id: str
    config: RunConfig
    dataset: str
    protocol: str
    experiments: list[tuple[str, str]]  # (experiment name, ablation letter)


class SkipRunError(Exception):
    """Raised when a configuration is invalid for this backbone/dataset (recorded as skipped)."""


class BenchmarkRunner:
    """Executes a :class:`BenchmarkConfig` and stores one record per planned run."""

    def __init__(
        self,
        cfg: BenchmarkConfig,
        settings: Settings,
        results_dir: Path | None = None,
        resume: bool = True,
        show_progress: bool = True,
    ) -> None:
        self.cfg = cfg
        self.settings = settings
        pref = settings.device if settings.device != "auto" else cfg.device
        self.device = resolve_device(pref)
        configure_threads(settings.num_threads)
        self.store = ResultsStore(results_dir or settings.results_dir)
        if not resume:
            self.store.reset()
        self.show_progress = show_progress
        self.feature_cache = FeatureCache(settings.feature_cache_dir)
        self.keypoints = KeypointCache(max_items=60_000)
        self._datasets: dict[str, RetrievalDataset] = {}
        self._extractor: tuple[str, FeatureExtractor] | None = None
        self._features: OrderedDict[str, FeatureSet] = OrderedDict()
        self._pca: dict[str, PCAWhitening] = {}
        self._exact: dict[str, np.ndarray] = {}
        self._index: tuple[str, VectorIndex] | None = None
        self._distractors: dict[str, DistractorSet] = {}

    # ------------------------------------------------------------------ planning
    def plan(self) -> list[PlannedRun]:
        """Expand every experiment; identical runs shared by several experiments run once."""
        planned: dict[str, PlannedRun] = {}
        for exp in self.cfg.experiments:
            for rc in expand_experiment(self.cfg.defaults, exp):
                for ds in self._datasets_for(exp):
                    for proto in exp.protocols or self.cfg.protocols:
                        rid = self.run_id(rc, ds, proto)
                        if rid not in planned:
                            planned[rid] = PlannedRun(rid, rc, ds, proto, [])
                        tag = (exp.name, exp.ablation)
                        if tag not in planned[rid].experiments:
                            planned[rid].experiments.append(tag)
        return sorted(planned.values(), key=self._order_key)

    def _datasets_for(self, exp: ExperimentCfg) -> list[str]:
        return list(exp.datasets or self.cfg.datasets)

    def run_id(self, rc: RunConfig, dataset: str, protocol: str) -> str:
        return stable_hash(
            {
                "config": rc.model_dump(),
                "dataset": dataset,
                "protocol": protocol,
                "subset": self.cfg.subset,
                "eval_depth": self.cfg.eval_depth,
                "ann_recall_k": self.cfg.ann_recall_k,
                "seed": self.cfg.seed,
                "device": self.device,
                "schema": RESULT_SCHEMA_VERSION,
            }
        )

    @staticmethod
    def _order_key(p: PlannedRun) -> tuple:
        rc = p.config
        build_params = {k: v for k, v in rc.index.params.items() if k not in SEARCH_PARAMS[rc.index.type]}
        search_params = {k: v for k, v in rc.index.params.items() if k in SEARCH_PARAMS[rc.index.type]}
        return (
            p.dataset,
            stable_hash(rc.extractor.model_dump()),
            rc.pooling.name or "",
            rc.pooling.p,
            p.protocol,
            stable_hash(rc.postprocess.model_dump()),
            rc.distractors,
            rc.index.type,
            stable_hash(build_params),
            str(sorted(search_params.items())),
            rc.rerank.name,
        )

    # ------------------------------------------------------------------ execution
    def run_all(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Run the whole plan (resuming finished runs) and return (records, meta)."""
        start = time.time()
        plan = self.plan()
        done = self.store.load()
        log.info(
            "benchmark plan", extra={"runs": len(plan), "already_done": sum(p.run_id in done for p in plan)}
        )
        records: list[dict[str, Any]] = []
        for i, p in enumerate(plan, 1):
            if p.run_id in done and done[p.run_id].get("status") in {"ok", "skipped"}:
                rec = done[p.run_id]
            else:
                log.info(
                    "run",
                    extra={
                        "i": i,
                        "n": len(plan),
                        "dataset": p.dataset,
                        "protocol": p.protocol,
                        "label": p.config.label(extractor_info(p.config.extractor.name).default_pooling),
                    },
                )
                rec = self.execute(p)
                self.store.append(rec)
            rec = {**rec, "experiments": [{"name": n, "ablation": a} for n, a in p.experiments]}
            records.append(rec)
        self._release_index()
        meta = self.meta(time.time() - start)
        return records, meta

    def execute(self, p: PlannedRun) -> dict[str, Any]:
        """Run one planned configuration; never raises except for unavailable datasets."""
        base: dict[str, Any] = {
            "run_id": p.run_id,
            "dataset": p.dataset,
            "protocol": p.protocol,
            "config": p.config.model_dump(),
            "config_hash": stable_hash(p.config.model_dump()),
            "label": p.config.label(),
            "extractor": p.config.extractor.name,
            "pooling": p.config.pooling.name,
            "index": p.config.index.type,
            "rerank": p.config.rerank.name,
            "created_at": utc_now_iso(),
        }
        try:
            info = extractor_info(p.config.extractor.name)
            base["family"] = info.family
            base["pooling"] = p.config.pooling.name or info.default_pooling
            base["label"] = p.config.label(info.default_pooling)
            seed_everything(self.cfg.seed)
            base.update(self._execute(p))
            base["status"] = "ok"
        except SkipRunError as exc:
            base.update(status="skipped", reason=str(exc))
            log.info("skipped", extra={"label": base["label"], "reason": str(exc)})
        except DatasetUnavailableError:
            raise
        except Exception as exc:
            base.update(
                status="failed", reason=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()
            )
            log.error("run failed", extra={"label": base["label"], "error": str(exc)})
        return base

    # ------------------------------------------------------------------ helpers
    def dataset(self, name: str) -> RetrievalDataset:
        if name not in self._datasets:
            self._datasets[name] = load_dataset(
                name,
                self.settings.data_dir,
                subset=self.cfg.subset,
                seed=self.cfg.seed,
                auto_download=self.settings.auto_download,
            )
        return self._datasets[name]

    def extractor(self, rc: RunConfig) -> FeatureExtractor:
        info = extractor_info(rc.extractor.name)
        pooling = rc.pooling.name or info.default_pooling
        if pooling not in info.poolings:
            raise SkipRunError(
                f"pooling '{pooling}' not applicable to {rc.extractor.name} (valid: {info.poolings})"
            )
        if rc.extractor.scales != [1.0] and not info.supports_multiscale:
            raise SkipRunError(f"{rc.extractor.name} does not support multi-scale extraction")
        kwargs: dict[str, Any] = dict(
            pooling=pooling,
            gem_p=rc.pooling.p,
            scales=rc.extractor.scales,
            device=self.device,
            batch_size=self.settings.batch_size or self.cfg.batch_size,
            **rc.extractor.params,
        )
        if rc.extractor.image_size is not None:
            kwargs["image_size"] = rc.extractor.image_size
        if info.family != "classical":
            kwargs["pretrained"] = rc.extractor.pretrained
        key = stable_hash({"name": rc.extractor.name, **{k: v for k, v in kwargs.items()}})
        if self._extractor is None or self._extractor[0] != key:
            self._extractor = None
            gc.collect()
            self._extractor = (key, create_extractor(rc.extractor.name, **kwargs))
        return self._extractor[1]

    def features(
        self, ext: FeatureExtractor, items: list[ImageItem], tag: str, per_item_timing: bool = False
    ) -> FeatureSet:
        key = stable_hash({"ext": ext.config(), "tag": tag, "items": [(i.id, i.bbox) for i in items]})
        if key in self._features:
            self._features.move_to_end(key)
            return self._features[key]
        fs = extract_items(
            ext,
            items,
            self.feature_cache,
            tag=tag,
            device=self.device,
            per_item_timing=per_item_timing,
            warmup=self.cfg.warmup_queries if per_item_timing else 0,
            show_progress=self.show_progress,
        )
        self._features[key] = fs
        while len(self._features) > 8:
            self._features.popitem(last=False)
        return fs

    def distractor_set(self, dataset: str) -> DistractorSet:
        toy = dataset in TOY_DATASETS
        key = "toy" if toy else "real"
        if key in self._distractors:
            return self._distractors[key]
        dcfg = self.cfg.distractors
        if toy and dcfg.synthetic_for_toy:
            folder = self.settings.data_dir / "toy_distractors"
            generate_toy_distractors(folder, dcfg.max_size, seed=self.cfg.seed)
        else:
            folder = Path(dcfg.dir) if dcfg.dir else self.settings.resolved_distractor_dir
            ensure_distractors(
                self.settings.data_dir, folder, dcfg.download_archives, self.settings.auto_download
            )
        exclude = set()
        for ds in self._datasets.values():
            exclude |= set(ds.paths)
        self._distractors[key] = load_distractors(folder, dcfg.max_size, self.cfg.seed, exclude_ids=exclude)
        return self._distractors[key]

    def _release_index(self) -> None:
        if self._index is not None:
            self._index[1].close()
            self._index = None

    def _get_index(self, rc: RunConfig, space_key: str, db: np.ndarray) -> tuple[VectorIndex, bool]:
        kind = rc.index.type
        search = {k: v for k, v in rc.index.params.items() if k in SEARCH_PARAMS[kind]}
        build = {k: v for k, v in rc.index.params.items() if k not in SEARCH_PARAMS[kind]}
        build_key = stable_hash({"space": space_key, "kind": kind, "build": build})
        reused = False
        if self._index is not None and self._index[0] == build_key:
            index = self._index[1]
            reused = True
        else:
            self._release_index()
            params: dict[str, Any] = dict(build)
            if kind == "hnsw":
                url = self.settings.qdrant_url
                params.update({"location": ":memory:"} if url == ":memory:" else {"url": url})
                params["collection"] = f"bench_{build_key[:16]}"
            if kind == "ivfpq":
                params.setdefault("seed", self.cfg.seed)
            index = create_index(kind, **params)
            index.build(db)
            self._index = (build_key, index)
        if search:
            index.set_search_params(**search)
        return index, reused

    def _exact_topk(self, space_key: str, db: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
        key = f"{space_key}:{k}"
        if key not in self._exact:
            sims = queries @ db.T
            k = min(k, db.shape[0])
            part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            order = np.take_along_axis(sims, part, axis=1).argsort(axis=1)[:, ::-1]
            self._exact[key] = np.take_along_axis(part, order, axis=1)
        return self._exact[key]

    # ------------------------------------------------------------------ the harness
    def _execute(self, p: PlannedRun) -> dict[str, Any]:
        rc, cfg = p.config, self.cfg
        ds = self.dataset(p.dataset)
        if p.protocol not in ds.protocols:
            raise SkipRunError(f"protocol '{p.protocol}' not available for {p.dataset}")
        proto = ds.protocols[p.protocol]
        if not proto.queries:
            raise SkipRunError("no queries left after subsetting / missing-image removal")
        ext = self.extractor(rc)
        subset_tag = f"subset{cfg.subset}" if cfg.subset else "full"

        # ---- descriptors
        db_fs = self.features(ext, ds.all_db_items(), f"{p.dataset}:db:{subset_tag}")
        q_fs = self.features(ext, proto.query_items(), f"{p.dataset}:{p.protocol}:queries:{subset_tag}", True)
        row_of = {iid: r for r, iid in enumerate(db_fs.ids)}
        db = db_fs.vectors[[row_of[i] for i in proto.db_ids]]
        ids = [f"{p.dataset}/{i}" for i in proto.db_ids]
        paths = [ds.paths[i] for i in proto.db_ids]
        queries = q_fs.vectors
        n_distractors = 0
        distractor_notes: list[str] = []
        if rc.distractors:
            dset = self.distractor_set(p.dataset)
            items = dset.prefix(rc.distractors)
            distractor_notes = list(dset.notes)
            if len(items) < rc.distractors:
                raise SkipRunError(
                    f"requested {rc.distractors} distractors but only {len(items)} are available "
                    f"(see docs: place images in {dset.source})"
                )
            d_fs = self.features(ext, items, f"distractors:{dset.source}:{len(items)}")
            db = np.vstack([db, d_fs.vectors])
            ids += [f"distractors/{it.id}" for it in items]
            paths += [it.path for it in items]
            n_distractors = len(items)

        # ---- post-processing (PCA / whitening learned on the held-out dataset only)
        pca_info: dict[str, Any] | None = None
        if rc.postprocess.pca != "none":
            held_name = ds.heldout or HELDOUT.get(p.dataset)
            if held_name is None:
                raise SkipRunError("no held-out dataset configured for PCA")
            held = self.dataset(held_name)
            h_fs = self.features(ext, held.all_db_items(), f"{held_name}:db:{subset_tag}")
            pkey = stable_hash(
                {"ext": ext.config(), "held": held_name, "pp": rc.postprocess.model_dump(), "s": subset_tag}
            )
            if pkey not in self._pca:
                self._pca[pkey] = PCAWhitening(
                    dim=rc.postprocess.dim, whiten=rc.postprocess.pca == "whiten"
                ).fit(h_fs.vectors)
            pca = self._pca[pkey]
            db = pca.transform(db)
            queries = pca.transform(queries)
            total_var = float(np.var(h_fs.vectors, axis=0, ddof=1).sum())
            pca_info = {
                **pca.describe(),
                "fit_on": held_name,
                "n_fit": int(h_fs.vectors.shape[0]),
                "explained_variance_ratio": pca.explained_variance_ratio(total_var),
            }
        space_key = stable_hash(
            {
                "ext": ext.config(),
                "ds": p.dataset,
                "proto": p.protocol,
                "subset": subset_tag,
                "pp": rc.postprocess.model_dump(),
                "nd": n_distractors,
            }
        )

        # ---- index
        try:
            index, reused = self._get_index(rc, space_key, db)
        except ValueError as exc:
            raise SkipRunError(f"index configuration invalid for this data: {exc}") from exc
        n = db.shape[0]
        depth = min(cfg.eval_depth, n)
        reranker = create_reranker(
            rc.rerank.name, rc.rerank.params, keypoint_cache=self.keypoints, seed=cfg.seed
        )
        needs_images = rerank_needs_images(rc.rerank.name)
        q_images = (
            [crop_bbox(load_image(q.path), q.bbox) for q in proto.queries]
            if needs_images
            else [None] * len(queries)
        )

        def ctx_for(i: int) -> RerankContext:
            return RerankContext(
                query_vector=queries[i],
                db_vectors=db,
                index=index,
                depth=depth,
                query_image=q_images[i],
                load_db_image=lambda r: load_image(paths[r]),
                extra={"keypoint_key": lambda r: paths[r]},
            )

        def run_query(i: int) -> tuple[Ranking, float, float]:
            t0 = time.perf_counter()
            res = index.search(queries[i : i + 1], depth)
            t1 = time.perf_counter()
            ranking = Ranking(scores=res.scores[0], indices=res.indices[0])
            if reranker is not None:
                ranking = reranker.rerank(ctx_for(i), ranking)
            t2 = time.perf_counter()
            return ranking.valid(), (t1 - t0) * 1000.0, (t2 - t1) * 1000.0

        nq = len(queries)
        for i in range(min(cfg.warmup_queries, nq)):  # warm-up, excluded from timing
            run_query(i)
        if needs_images:  # database ORB features are an offline artefact: precompute them
            for i in range(nq):
                run_query(i)

        rankings: list[np.ndarray] = []
        scores_kept: list[np.ndarray] = []
        search_ms: list[float] = []
        rerank_ms: list[float] = []
        for i in range(nq):
            ranking, s_ms, r_ms = run_query(i)
            rankings.append(ranking.indices)
            scores_kept.append(ranking.scores[:RANKING_KEEP])
            search_ms.append(s_ms)
            rerank_ms.append(r_ms)
        extract_ms = list(q_fs.meta.get("per_item_ms") or [])
        if len(extract_ms) != nq:
            extract_ms = [float(q_fs.meta.get("ms_per_image") or 0.0)] * nq
        total_ms = [e + s + r for e, s, r in zip(extract_ms, search_ms, rerank_ms, strict=True)]

        # ---- ANN quality: neighbour recall against exact search at small k
        ann: dict[str, Any] = {}
        if index.kind != "flat":
            k = min(cfg.ann_recall_k, n)
            exact = self._exact_topk(space_key, db, queries, k)
            approx = np.full((nq, k), -1, dtype=np.int64)
            ann_ms: list[float] = []
            for i in range(min(cfg.warmup_queries, nq)):
                index.search(queries[i : i + 1], k)
            for i in range(nq):
                t0 = time.perf_counter()
                res = index.search(queries[i : i + 1], k)
                ann_ms.append((time.perf_counter() - t0) * 1000.0)
                approx[i, : res.indices.shape[1]] = res.indices[0, :k]
            ann = {
                "ann_recall@1": ann_recall(approx, exact, 1),
                f"ann_recall@{k}": ann_recall(approx, exact, k),
                "ann_recall_k": k,
                "ann_search_ms": latency_stats(ann_ms),
            }

        # ---- batch throughput of the index alone
        t0 = time.perf_counter()
        index.search(queries, depth)
        batch_s = time.perf_counter() - t0

        metrics = evaluate_protocol(rankings, proto, RECALL_KS)
        mem = index.memory_report().to_dict()
        headline = self._headline(metrics, p.protocol)
        self.store.save_rankings(
            p.run_id,
            {
                "query_ids": [q.id for q in proto.queries],
                "top_ids": [[ids[int(j)] for j in r[:RANKING_KEEP]] for r in rankings],
                "top_scores": [[float(s) for s in sc] for sc in scores_kept],
            },
        )
        return {
            "headline_map": headline,
            "metrics": {setup: {k: v for k, v in m.items() if k != "ap"} for setup, m in metrics.items()},
            "per_query_ap": {setup: m["ap"] for setup, m in metrics.items()},
            "query_ids": [q.id for q in proto.queries],
            "latency_ms": {
                "extract": latency_stats(extract_ms),
                "search": latency_stats(search_ms),
                "rerank": latency_stats(rerank_ms),
                "total": latency_stats(total_ms),
            },
            "throughput_qps": {
                "end_to_end_single_stream": (1000.0 / float(np.mean(total_ms))) if total_ms else None,
                "search_batch": (nq / batch_s) if batch_s > 0 else None,
            },
            **ann,
            "memory": mem,
            "vector_dim": int(db.shape[1]),
            "index_params_effective": index.effective_params(),
            "index_reused_from_previous_run": reused,
            "build_time_s": index.build_time,
            "extract_ms_per_image_db": db_fs.meta.get("ms_per_image"),
            "extract_db_from_cache": bool(db_fs.meta.get("cache_hit")),
            "n_db": len(proto.db_ids),
            "n_distractors": n_distractors,
            "n_index": n,
            "n_queries": nq,
            "eval_depth": depth,
            "pca": pca_info,
            "distractor_notes": distractor_notes,
            "device": self.device,
            "extractor_config": ext.config(),
        }

    def _headline(self, metrics: dict[str, Any], protocol: str) -> float | None:
        if protocol == "original":
            return metrics["all"]["map"]
        setup = self.cfg.headline.setup if self.cfg.headline.setup != "all" else "medium"
        return metrics[setup]["map"]

    # ------------------------------------------------------------------ meta
    def meta(self, wall_s: float) -> dict[str, Any]:
        datasets = {name: ds.describe() for name, ds in self._datasets.items()}
        distractors = {
            k: {
                "source": v.source,
                "available": len(v.items),
                "pool_size": v.pool_size,
                "skipped_corrupt": v.skipped_corrupt,
                "notes": v.notes,
            }
            for k, v in self._distractors.items()
        }
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "benchmark": self.cfg.name,
            "config_hash": self.cfg.config_hash(),
            "config": self.cfg.model_dump(),
            "seed": self.cfg.seed,
            "subset": self.cfg.subset,
            "device": self.device,
            "hardware": hardware_info(self.device),
            "libraries": library_versions(),
            "datasets": datasets,
            "distractors": distractors,
            "headline": self.cfg.headline.model_dump(),
            "ablations": ABLATIONS,
            "generated_at": utc_now_iso(),
            "wall_time_s": wall_s,
            "timing_notes": (
                "Warm-up queries are excluded. extract = per-query forward pass incl. resize/normalise "
                "(disk decode excluded); search = index query at eval_depth; rerank = query expansion "
                "re-query and/or spatial verification with precomputed database ORB features."
            ),
        }
