"""Single typed settings object for the whole project.

Environment variables configure paths, services and runtime knobs; YAML files under
``configs/`` describe the benchmark grid (``fast.yaml`` / ``full.yaml``) and the serving
configuration (``api.yaml``). :func:`get_settings` returns one cached :class:`Settings`
instance that also exposes the parsed serving configuration.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import AliasChoices, BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServingIndexDefaults(BaseModel):
    """Default parameters for each index type when used by the API."""

    flat: dict[str, Any] = Field(default_factory=dict)
    hnsw: dict[str, Any] = Field(default_factory=lambda: {"m": 16, "ef_construct": 128, "ef": 64})
    ivfpq: dict[str, Any] = Field(default_factory=lambda: {"nlist": 64, "m": 16, "nbits": 8, "nprobe": 8})


class ServingRerankDefaults(BaseModel):
    """Default re-ranker parameters used by the API."""

    aqe: dict[str, Any] = Field(default_factory=lambda: {"top_n": 3})
    alpha_qe: dict[str, Any] = Field(default_factory=lambda: {"top_n": 3, "alpha": 3.0})
    sv: dict[str, Any] = Field(default_factory=lambda: {"top_k": 20, "min_inliers": 12})


class ServingBackbone(BaseModel):
    """A backbone that the API is allowed to serve."""

    name: str
    image_size: int | None = None
    pretrained: bool = True
    poolings: list[str]
    default_pooling: str


class ServingPrebuild(BaseModel):
    """A (backbone, pooling, index) triple built during warm-up."""

    backbone: str
    pooling: str
    index: Literal["flat", "hnsw", "ivfpq"] = "flat"


class ServingConfig(BaseModel):
    """Contents of ``configs/api.yaml``."""

    dataset: str = "roxford5k"
    protocol: Literal["revisited", "original"] = "revisited"
    pca_whitening: bool = False
    backbones: list[ServingBackbone]
    indexes: list[Literal["flat", "hnsw", "ivfpq"]] = Field(default_factory=lambda: ["flat", "hnsw", "ivfpq"])
    rerankers: list[str] = Field(default_factory=lambda: ["none", "aqe", "alpha_qe", "sv", "aqe+sv"])
    prebuild: list[ServingPrebuild] = Field(default_factory=list)
    index_defaults: ServingIndexDefaults = Field(default_factory=ServingIndexDefaults)
    rerank_defaults: ServingRerankDefaults = Field(default_factory=ServingRerankDefaults)
    max_k: int = 100
    default_k: int = 20


class Settings(BaseSettings):
    """Process-wide settings, populated from environment variables."""

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True, case_sensitive=False)

    data_dir: Path = Field(Path("/data"), validation_alias=AliasChoices("DATA_DIR", "data_dir"))
    results_dir: Path = Field(Path("/results"), validation_alias=AliasChoices("RESULTS_DIR", "results_dir"))
    config_dir: Path = Field(Path("/app/configs"), validation_alias=AliasChoices("CONFIG_DIR", "config_dir"))
    readme_path: Path = Field(
        Path("/workspace/README.md"), validation_alias=AliasChoices("README_PATH", "readme_path")
    )
    seed: int = Field(42, validation_alias=AliasChoices("SEED", "seed"))
    device: Literal["auto", "cpu", "cuda"] = Field("auto", validation_alias=AliasChoices("DEVICE", "device"))
    dataset_subset: int | None = Field(
        None, validation_alias=AliasChoices("DATASET_SUBSET", "dataset_subset")
    )
    auto_download: bool = Field(True, validation_alias=AliasChoices("AUTO_DOWNLOAD", "auto_download"))
    distractor_dir: Path | None = Field(
        None, validation_alias=AliasChoices("DISTRACTOR_DIR", "distractor_dir")
    )
    batch_size: int = Field(16, validation_alias=AliasChoices("BATCH_SIZE", "batch_size"))
    num_threads: int | None = Field(None, validation_alias=AliasChoices("NUM_THREADS", "num_threads"))
    redis_url: str = Field("redis://redis:6379/0", validation_alias=AliasChoices("REDIS_URL", "redis_url"))
    qdrant_url: str = Field("http://qdrant:6333", validation_alias=AliasChoices("QDRANT_URL", "qdrant_url"))
    api_config: str = Field("api.yaml", validation_alias=AliasChoices("API_CONFIG", "api_config"))
    api_dataset: str | None = Field(None, validation_alias=AliasChoices("API_DATASET", "api_dataset"))
    api_dataset_subset: int | None = Field(
        None, validation_alias=AliasChoices("API_DATASET_SUBSET", "api_dataset_subset")
    )
    cors_origins: str = Field(
        "http://localhost:3000", validation_alias=AliasChoices("CORS_ORIGINS", "cors_origins")
    )
    max_upload_mb: float = Field(10.0, validation_alias=AliasChoices("MAX_UPLOAD_MB", "max_upload_mb"))
    max_image_pixels: int = Field(
        40_000_000, validation_alias=AliasChoices("MAX_IMAGE_PIXELS", "max_image_pixels")
    )
    cache_ttl_seconds: int = Field(
        86_400, validation_alias=AliasChoices("CACHE_TTL_SECONDS", "cache_ttl_seconds")
    )
    log_level: str = Field("INFO", validation_alias=AliasChoices("LOG_LEVEL", "log_level"))
    skip_warmup: bool = Field(False, validation_alias=AliasChoices("SKIP_WARMUP", "skip_warmup"))

    @field_validator("dataset_subset", "api_dataset_subset", "num_threads", mode="before")
    @classmethod
    def _empty_is_none(cls, value: Any) -> Any:
        """Treat empty strings (e.g. ``DATASET_SUBSET=``) and 0 as 'not set'."""
        if value is None:
            return None
        if isinstance(value, str) and value.strip() in {"", "0", "none", "None"}:
            return None
        if isinstance(value, int) and value <= 0:
            return None
        return value

    @field_validator("distractor_dir", mode="before")
    @classmethod
    def _empty_path_is_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    # ----------------------------------------------------------------- derived paths
    @property
    def cache_dir(self) -> Path:
        """Root of all derived caches (features, PCA, thumbnails, keypoints)."""
        return self.data_dir / "cache"

    @property
    def feature_cache_dir(self) -> Path:
        return self.cache_dir / "features"

    @property
    def thumbnail_dir(self) -> Path:
        return self.cache_dir / "thumbnails"

    @property
    def resolved_distractor_dir(self) -> Path:
        return self.distractor_dir or (self.data_dir / "distractors")

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    def serving(self) -> ServingConfig:
        """Parse and return the serving configuration (``configs/<API_CONFIG>``)."""
        return load_serving_config(self.config_dir / self.api_config, self.api_dataset)


def load_serving_config(path: Path, dataset_override: str | None = None) -> ServingConfig:
    """Load ``api.yaml`` into a :class:`ServingConfig`."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = ServingConfig.model_validate(raw)
    if dataset_override:
        cfg = cfg.model_copy(update={"dataset": dataset_override})
    return cfg


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached settings object."""
    return Settings()
