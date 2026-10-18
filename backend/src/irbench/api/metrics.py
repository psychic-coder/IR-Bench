"""Prometheus metrics exposed at ``GET /metrics``."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

LATENCY_BUCKETS_MS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 30000)


class ApiMetrics:
    """A private registry per app instance (keeps tests isolated)."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry(auto_describe=True)
        self.requests = Counter(
            "irbench_http_requests_total",
            "HTTP requests by route, method and status",
            ["route", "method", "status"],
            registry=self.registry,
        )
        self.request_latency = Histogram(
            "irbench_http_request_duration_ms",
            "HTTP request latency in milliseconds",
            ["route", "method"],
            buckets=LATENCY_BUCKETS_MS,
            registry=self.registry,
        )
        self.stage_latency = Histogram(
            "irbench_search_stage_duration_ms",
            "Per-stage /search latency in milliseconds",
            ["stage", "backbone", "index", "rerank"],
            buckets=LATENCY_BUCKETS_MS,
            registry=self.registry,
        )
        self.cache = Counter(
            "irbench_embedding_cache_total",
            "Embedding cache lookups by result",
            ["result"],
            registry=self.registry,
        )
        self.ready = Gauge(
            "irbench_ready", "1 when warm-up finished and /search is served", registry=self.registry
        )
        self.index_vectors = Gauge(
            "irbench_index_vectors",
            "Vectors per built index",
            ["backbone", "pooling", "index"],
            registry=self.registry,
        )
        self.index_bytes = Gauge(
            "irbench_index_memory_bytes",
            "Memory per built index",
            ["backbone", "pooling", "index"],
            registry=self.registry,
        )
