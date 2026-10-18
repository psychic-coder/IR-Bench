"""FastAPI application: search, configs, images, benchmark results, readiness and metrics."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Body, FastAPI, File, Form, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from irbench import __version__
from irbench.api.cache import EmbeddingCache
from irbench.api.engine import RetrievalEngine
from irbench.api.errors import APIError, error_payload, install_error_handlers
from irbench.api.metrics import ApiMetrics
from irbench.api.schemas import (
    ConfigsResponse,
    ErrorResponse,
    HealthResponse,
    ReadyResponse,
    SearchResponse,
    WarmupRequest,
    WarmupResponse,
)
from irbench.features import extractor_info
from irbench.features.preprocessing import crop_bbox, decode_image_bytes
from irbench.logging_utils import configure_logging, get_logger
from irbench.settings import Settings, get_settings
from irbench.utils import sha256_bytes

log = get_logger("irbench.api")

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/gif",
    "image/tiff",
    "application/octet-stream",
}
ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
    415: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}


def _parse_bbox(raw: str | None) -> list[float] | None:
    if raw is None or not raw.strip():
        return None
    try:
        vals = [float(v) for v in raw.replace(" ", "").split(",")]
    except ValueError as exc:
        raise APIError(422, "invalid_bbox", "bbox must be 'x1,y1,x2,y2' numbers") from exc
    if len(vals) != 4 or vals[2] <= vals[0] or vals[3] <= vals[1] or min(vals) < 0:
        raise APIError(422, "invalid_bbox", "bbox must be 'x1,y1,x2,y2' with x2>x1, y2>y1, all >= 0")
    return vals


def create_app(
    settings: Settings | None = None,
    cache: EmbeddingCache | None = None,
    start_warmup: bool = True,
) -> FastAPI:
    """Application factory (tests inject settings and a fake cache)."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    serving = settings.serving()
    metrics = ApiMetrics()
    if cache is None:
        cache = EmbeddingCache(url=settings.redis_url, ttl_seconds=settings.cache_ttl_seconds)
    engine = RetrievalEngine(settings, serving, cache, metrics)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if start_warmup and not settings.skip_warmup:
            engine.start_warmup()
        yield
        engine.close()

    app = FastAPI(
        title="Image Retrieval Benchmark Engine API",
        version=__version__,
        description="Search Oxford/Paris landmarks with classical, CNN and transformer descriptors.",
        lifespan=lifespan,
    )
    app.state.engine = engine
    app.state.settings = settings
    app.state.metrics = metrics
    install_error_handlers(app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def observe(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        request.state.request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        length = request.headers.get("content-length")
        if (
            request.method == "POST"
            and length
            and length.isdigit()
            and int(length) > settings.max_upload_bytes + 64_000
        ):
            return JSONResponse(
                error_payload(
                    request, "payload_too_large", f"request body exceeds {settings.max_upload_mb} MB"
                ),
                status_code=413,
            )
        start = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - start) * 1000.0
        route = getattr(request.scope.get("route"), "path", "unmatched")
        metrics.requests.labels(route=route, method=request.method, status=str(response.status_code)).inc()
        metrics.request_latency.labels(route=route, method=request.method).observe(ms)
        response.headers["X-Request-ID"] = request.state.request_id
        if route not in {"/health", "/metrics", "/ready"}:
            log.info(
                "request",
                extra={
                    "request_id": request.state.request_id,
                    "method": request.method,
                    "route": route,
                    "status": response.status_code,
                    "duration_ms": round(ms, 2),
                },
            )
        return response

    # ------------------------------------------------------------------ search
    @app.post("/search", response_model=SearchResponse, responses=ERROR_RESPONSES, tags=["search"])
    def search(
        image: UploadFile = File(..., description="query image (JPEG/PNG/WebP/BMP/GIF/TIFF)"),
        backbone: str | None = Form(None, description="backbone name; default = first in api.yaml"),
        pooling: str | None = Form(None, description="pooling; default = backbone default"),
        index: Literal["flat", "hnsw", "ivfpq"] = Form("flat"),
        rerank: str = Form("none", description="none | aqe | alpha_qe | sv | aqe+sv"),
        k: int = Form(serving.default_k, ge=1, le=1000),
        bbox: str | None = Form(
            None, description="optional crop 'x1,y1,x2,y2' in pixels (protocol-style query box)"
        ),
        use_cache: bool = Form(True),
    ) -> dict:
        ctype = (image.content_type or "application/octet-stream").lower()
        if ctype not in ALLOWED_CONTENT_TYPES:
            raise APIError(
                415, "unsupported_media_type", f"content type '{ctype}' is not an accepted image type"
            )
        data = image.file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise APIError(413, "payload_too_large", f"image exceeds {settings.max_upload_mb} MB")
        if not data:
            raise APIError(400, "invalid_image", "empty upload")
        try:
            img = decode_image_bytes(data, settings.max_image_pixels)
        except ValueError as exc:
            raise APIError(400, "invalid_image", str(exc)) from exc
        box = _parse_bbox(bbox)
        if box is not None:
            if box[0] >= img.width or box[1] >= img.height:
                raise APIError(422, "invalid_bbox", "bbox lies outside the image")
            img = crop_bbox(img, box)
        image_hash = sha256_bytes(data + (repr(box).encode() if box else b""))
        backbone = backbone or serving.backbones[0].name
        return engine.search(img, image_hash, backbone, pooling, index, rerank, k, box, use_cache)

    # ------------------------------------------------------------------ configuration
    @app.get("/configs", response_model=ConfigsResponse, tags=["meta"])
    def configs() -> dict:
        backbones = []
        for b in serving.backbones:
            info = extractor_info(b.name)
            backbones.append(
                {
                    "name": b.name,
                    "family": info.family,
                    "poolings": b.poolings,
                    "default_pooling": b.default_pooling,
                    "image_size": b.image_size,
                    "description": info.description,
                }
            )
        return {
            "dataset": serving.dataset,
            "protocol": serving.protocol,
            "pca_whitening": serving.pca_whitening,
            "backbones": backbones,
            "indexes": [{"name": i, "params": engine.index_params(i)} for i in serving.indexes],
            "rerankers": serving.rerankers,
            "default_k": serving.default_k,
            "max_k": serving.max_k,
            "prebuilt": [p.as_dict() for p in engine.prebuilt],
            "sample_images": engine.sample_images(),
        }

    # ------------------------------------------------------------------ images
    @app.get("/images/{image_id:path}", responses=ERROR_RESPONSES, tags=["images"])
    def get_image(image_id: str) -> FileResponse:
        path = engine.resolve_image(image_id)
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/thumbnails/{image_id:path}", responses=ERROR_RESPONSES, tags=["images"])
    def get_thumbnail(image_id: str, size: int = Query(256, ge=32, le=1024)) -> FileResponse:
        path = engine.thumbnail(image_id, size)
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})

    # ------------------------------------------------------------------ benchmark artefacts
    def _results_file(name: str, what: str) -> FileResponse:
        path = settings.results_dir / name
        if not path.exists():
            raise APIError(
                404, "no_results", f"No results generated yet: {what} not found. Run `make bench-fast`."
            )
        return FileResponse(path, media_type="application/json", headers={"Cache-Control": "no-cache"})

    @app.get("/benchmark", responses=ERROR_RESPONSES, tags=["benchmark"])
    def benchmark() -> FileResponse:
        return _results_file("results.json", "results/results.json")

    @app.get("/hard-queries", responses=ERROR_RESPONSES, tags=["benchmark"])
    def hard_queries() -> FileResponse:
        return _results_file("hard_queries.json", "results/hard_queries.json")

    # ------------------------------------------------------------------ operations
    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/ready", response_model=ReadyResponse, responses={503: {"model": ReadyResponse}}, tags=["ops"])
    def ready() -> JSONResponse:
        status = engine.status()
        return JSONResponse(ReadyResponse(**status).model_dump(), status_code=200 if status["ready"] else 503)

    @app.post("/warmup", response_model=WarmupResponse, status_code=202, tags=["ops"])
    def warmup(req: WarmupRequest | None = Body(None)) -> dict:
        only = None
        if req is not None and req.backbone:
            bb = engine.backbone(req.backbone)
            only = (req.backbone, req.pooling or bb.default_pooling, req.index or "flat")
            engine.validate(only[0], only[1], only[2], "none", serving.default_k)
        started = engine.start_warmup(only)
        return {
            "accepted": started,
            "message": "warm-up started" if started else "warm-up already running; poll /ready",
        }

    @app.get("/metrics", tags=["ops"])
    def prometheus_metrics() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app
