"""Redis-backed embedding cache keyed on (image SHA-256, embedding config).

The cache is an optimisation, never a dependency: if Redis is unreachable the service
keeps answering (``cache: "disabled"``) and retries the connection after a back-off.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import redis

from irbench.logging_utils import get_logger

log = get_logger(__name__)


class EmbeddingCache:
    """Stores float32 vectors as raw bytes with a TTL."""

    def __init__(self, client: Any | None = None, url: str | None = None, ttl_seconds: int = 86_400) -> None:
        self.ttl = ttl_seconds
        self._client = client
        self._url = url
        self._down_until = 0.0
        self._lock = threading.Lock()
        if self._client is None and url:
            self._client = redis.Redis.from_url(url, socket_connect_timeout=1.0, socket_timeout=1.0)

    @property
    def enabled(self) -> bool:
        return self._client is not None and time.monotonic() >= self._down_until

    def _fail(self, exc: Exception) -> None:
        with self._lock:
            if time.monotonic() >= self._down_until:
                log.warning("redis unavailable; embedding cache disabled for 30s", extra={"error": str(exc)})
            self._down_until = time.monotonic() + 30.0

    @staticmethod
    def key(image_hash: str, config_key: str) -> str:
        return f"irbench:emb:{config_key}:{image_hash}"

    def get(self, key: str) -> np.ndarray | None:
        if not self.enabled:
            return None
        try:
            raw = self._client.get(key)  # type: ignore[union-attr]
        except redis.RedisError as exc:
            self._fail(exc)
            return None
        if raw is None:
            return None
        return np.frombuffer(raw, dtype=np.float32).copy()

    def set(self, key: str, vector: np.ndarray) -> None:
        if not self.enabled:
            return
        try:
            self._client.set(key, np.ascontiguousarray(vector, dtype=np.float32).tobytes(), ex=self.ttl)  # type: ignore[union-attr]
        except redis.RedisError as exc:
            self._fail(exc)

    def ping(self) -> bool:
        if self._client is None:
            return False
        try:
            return bool(self._client.ping())
        except redis.RedisError:
            return False
