"""API smoke tests against toy data, embedded Qdrant and fakeredis."""

from __future__ import annotations

import io
import json
from pathlib import Path

import fakeredis
import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image

from irbench.api.cache import EmbeddingCache
from irbench.api.main import create_app

from .conftest import make_settings

API_CFG = {
    "dataset": "toy_a",
    "protocol": "revisited",
    "backbones": [
        {"name": "hsv_hist", "poolings": ["none"], "default_pooling": "none"},
        {
            "name": "resnet50",
            "image_size": 64,
            "pretrained": False,
            "poolings": ["gem", "gap"],
            "default_pooling": "gem",
        },
    ],
    "prebuild": [{"backbone": "hsv_hist", "pooling": "none", "index": "flat"}],
    "index_defaults": {
        "hnsw": {"m": 8, "ef_construct": 32, "ef": 16},
        "ivfpq": {"nlist": 2, "m": 16, "nbits": 4, "nprobe": 2},
    },
    "rerank_defaults": {"sv": {"top_k": 5, "min_inliers": 8}},
    "default_k": 5,
    "max_k": 50,
}


@pytest.fixture
def client(toy_data_dir: Path, tmp_path: Path) -> TestClient:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "api.yaml").write_text(yaml.safe_dump(API_CFG))
    settings = make_settings(toy_data_dir, tmp_path / "results", config_dir=cfg_dir, max_upload_mb=1.0)
    cache = EmbeddingCache(client=fakeredis.FakeRedis(), ttl_seconds=60)
    app = create_app(settings, cache=cache, start_warmup=False)
    return TestClient(app)


def _jpeg(path: Path | None = None, size: tuple[int, int] = (160, 120)) -> bytes:
    img = (
        Image.open(path).convert("RGB")
        if path
        else Image.fromarray(np.random.default_rng(0).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    )
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return buf.getvalue()


def _query_bytes(toy_data_dir: Path) -> bytes:
    return (sorted((toy_data_dir / "toy_a" / "jpg").glob("*.jpg"))[0]).read_bytes()


def _search(client: TestClient, data: bytes, **form: str):
    return client.post("/search", files={"image": ("q.jpg", data, "image/jpeg")}, data=form)


def test_health_ready_and_not_ready_search(client: TestClient, toy_data_dir: Path) -> None:
    assert client.get("/health").json() == {"status": "ok", "version": "1.0.0"}
    r = client.get("/ready")
    assert r.status_code == 503 and r.json()["ready"] is False
    res = _search(client, _query_bytes(toy_data_dir))
    assert res.status_code == 503
    assert res.json()["error"]["code"] == "not_ready"


def test_search_flow_and_cache(client: TestClient, toy_data_dir: Path) -> None:
    client.app.state.engine.warmup()
    ready = client.get("/ready").json()
    assert ready["ready"] is True and ready["built"][0]["status"] == "ready"
    data = _query_bytes(toy_data_dir)
    first = _search(client, data, backbone="hsv_hist", k="3")
    assert first.status_code == 200, first.text
    body = first.json()
    assert len(body["results"]) == 3 and body["cache"] == "miss" and body["cache_hit"] is False
    assert set(body["latency_ms"]) == {"extract", "search", "rerank", "total"}
    assert body["results"][0]["thumbnail_url"].startswith("/thumbnails/toy_a/")
    second = _search(client, data, backbone="hsv_hist", k="3").json()
    assert second["cache"] == "hit" and second["cache_hit"] is True
    assert [x["id"] for x in second["results"]] == [x["id"] for x in body["results"]]

    # lazily built config: CNN + IVF-PQ + AQE + spatial verification + bbox crop
    lazy = _search(
        client,
        data,
        backbone="resnet50",
        pooling="gap",
        index="ivfpq",
        rerank="aqe+sv",
        k="4",
        bbox="10,10,150,110",
    )
    assert lazy.status_code == 200, lazy.text
    lb = lazy.json()
    assert lb["config"]["bbox"] == [10.0, 10.0, 150.0, 110.0]
    assert lb["config"]["index_params"]["nlist"] >= 1
    hnsw = _search(client, data, backbone="hsv_hist", index="hnsw", k="2")
    assert hnsw.status_code == 200 and hnsw.json()["index"]["memory_method"] == "estimate"

    metrics = client.get("/metrics").text
    assert "irbench_search_stage_duration_ms_bucket" in metrics
    assert 'irbench_embedding_cache_total{result="hit"}' in metrics
    assert "irbench_ready 1.0" in metrics


def test_input_validation(client: TestClient, toy_data_dir: Path) -> None:
    client.app.state.engine.warmup()
    data = _query_bytes(toy_data_dir)
    bad = _search(client, b"definitely not an image")
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "invalid_image"
    txt = client.post("/search", files={"image": ("a.txt", b"hello", "text/plain")})
    assert txt.status_code == 415
    big = _search(client, b"\xff" * (1024 * 1024 + 10))
    assert big.status_code == 413 and big.json()["error"]["code"] == "payload_too_large"
    unknown = _search(client, data, backbone="vgg16")
    assert unknown.status_code == 422 and unknown.json()["error"]["code"] == "invalid_config"
    pooling = _search(client, data, backbone="resnet50", pooling="cls")
    assert pooling.status_code == 422
    box = _search(client, data, bbox="5,5,1,1")
    assert box.status_code == 422 and box.json()["error"]["code"] == "invalid_bbox"
    k = _search(client, data, k="0")
    assert k.status_code == 422 and k.json()["error"]["code"] == "validation_error"
    for r in (bad, txt, big, unknown, box, k):
        assert set(r.json()["error"]) == {"code", "message", "details", "request_id"}


def test_configs_images_and_thumbnails(client: TestClient) -> None:
    client.app.state.engine.warmup()
    cfg = client.get("/configs").json()
    assert cfg["dataset"] == "toy_a"
    assert [b["name"] for b in cfg["backbones"]] == ["hsv_hist", "resnet50"]
    assert cfg["backbones"][0]["family"] == "classical"
    assert {i["name"] for i in cfg["indexes"]} == {"flat", "hnsw", "ivfpq"}
    sample = cfg["sample_images"][0]
    img = client.get(f"/images/{sample}")
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    thumb = client.get(f"/thumbnails/{sample}?size=64")
    assert thumb.status_code == 200
    assert max(Image.open(io.BytesIO(thumb.content)).size) == 64
    assert client.get("/images/toy_a/nope").status_code == 404
    assert client.get("/images/distractors/../../etc/passwd").status_code == 404
    assert client.get("/images/unknown_ds/x").json()["error"]["code"] == "not_found"


def test_benchmark_and_hard_queries_endpoints(client: TestClient) -> None:
    missing = client.get("/benchmark")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "no_results"
    assert "No results generated yet" in missing.json()["error"]["message"]
    results = client.app.state.settings.results_dir
    results.mkdir(parents=True, exist_ok=True)
    (results / "results.json").write_text(json.dumps({"meta": {}, "runs": [], "ablations": []}))
    (results / "hard_queries.json").write_text(json.dumps({"datasets": {}}))
    assert client.get("/benchmark").json()["runs"] == []
    assert client.get("/hard-queries").json() == {"datasets": {}}


def test_warmup_endpoint(client: TestClient) -> None:
    r = client.post("/warmup", json={"backbone": "resnet50", "pooling": "gem", "index": "flat"})
    assert r.status_code == 202 and r.json()["accepted"] is True
    client.app.state.engine._warmup_thread.join(timeout=300)
    statuses = {(b["backbone"], b["index"]): b["status"] for b in client.get("/ready").json()["built"]}
    assert statuses[("resnet50", "flat")] == "ready"
    bad = client.post("/warmup", json={"backbone": "nope"})
    assert bad.status_code == 422


def test_cache_degrades_gracefully_without_redis(toy_data_dir: Path, tmp_path: Path) -> None:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "api.yaml").write_text(yaml.safe_dump(API_CFG))
    settings = make_settings(toy_data_dir, tmp_path / "results", config_dir=cfg_dir)
    app = create_app(settings, start_warmup=False)  # redis://localhost:1 is unreachable
    client = TestClient(app)
    app.state.engine.warmup()
    res = _search(client, _query_bytes(toy_data_dir))
    assert res.status_code == 200
    assert res.json()["cache"] in {"miss", "disabled"}
    again = _search(client, _query_bytes(toy_data_dir))
    assert again.json()["cache"] == "disabled"


def test_cors_headers(client: TestClient) -> None:
    r = client.options(
        "/configs", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"}
    )
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"
    r2 = client.options(
        "/configs", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"}
    )
    assert r2.headers.get("access-control-allow-origin") is None
