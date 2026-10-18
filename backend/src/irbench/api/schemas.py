"""Pydantic v2 request/response models (the source of the frontend's zod schemas)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str = Field(description="Machine-readable error code, e.g. 'invalid_image', 'not_ready'")
    message: str
    details: dict[str, Any] | None = None
    request_id: str | None = None


class ErrorResponse(BaseModel):
    """Every non-2xx response has this shape."""

    error: ErrorDetail


class SearchResultItem(BaseModel):
    rank: int
    id: str = Field(description="Global image id '<dataset>/<image>'")
    score: float
    image_url: str
    thumbnail_url: str
    inliers: int | None = Field(None, description="RANSAC inliers when spatial verification ran")


class StageLatency(BaseModel):
    extract: float
    search: float
    rerank: float
    total: float


class SearchConfigEcho(BaseModel):
    backbone: str
    pooling: str
    index: str
    index_params: dict[str, Any]
    rerank: str
    k: int
    bbox: list[float] | None = None
    pca_whitening: bool


class IndexInfo(BaseModel):
    type: str
    n_vectors: int
    dim: int
    memory_bytes: int
    bytes_per_vector: float
    memory_method: str


class SearchResponse(BaseModel):
    query_hash: str
    dataset: str
    config: SearchConfigEcho
    results: list[SearchResultItem]
    latency_ms: StageLatency
    cache: Literal["hit", "miss", "disabled"]
    cache_hit: bool
    index: IndexInfo


class BackboneInfo(BaseModel):
    name: str
    family: str
    poolings: list[str]
    default_pooling: str
    image_size: int | None
    description: str


class IndexOption(BaseModel):
    name: str
    params: dict[str, Any]


class PrebuiltConfig(BaseModel):
    backbone: str
    pooling: str
    index: str
    status: Literal["pending", "building", "ready", "failed"]
    error: str | None = None


class ConfigsResponse(BaseModel):
    dataset: str
    protocol: str
    pca_whitening: bool
    backbones: list[BackboneInfo]
    indexes: list[IndexOption]
    rerankers: list[str]
    default_k: int
    max_k: int
    prebuilt: list[PrebuiltConfig]
    sample_images: list[str] = Field(
        default_factory=list, description="A few database image ids (example queries, load-test seeds)"
    )


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str


class ReadyResponse(BaseModel):
    ready: bool
    stage: str
    message: str
    progress: dict[str, int] | None = None
    error: str | None = None
    built: list[PrebuiltConfig] = Field(default_factory=list)
    dataset: str
    n_images: int | None = None


class WarmupRequest(BaseModel):
    backbone: str | None = None
    pooling: str | None = None
    index: Literal["flat", "hnsw", "ivfpq"] | None = None


class WarmupResponse(BaseModel):
    accepted: bool
    message: str
