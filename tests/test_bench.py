"""End-to-end harness run on toy data: every artefact is produced from measurements."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from irbench.bench import BenchmarkRunner, load_benchmark_config
from irbench.bench.config import BenchmarkConfig, ExperimentCfg, expand_experiment
from irbench.bench.report import write_report

from .conftest import CONFIG_DIR, make_settings

MINI = {
    "name": "mini",
    "seed": 3,
    "datasets": ["toy_a"],
    "protocols": ["revisited"],
    "eval_depth": 1000,
    "ann_recall_k": 5,
    "warmup_queries": 1,
    "device": "cpu",
    "hard_query_ap_threshold": 0.99,
    "distractors": {"max_size": 10, "synthetic_for_toy": True},
    "defaults": {
        "extractor": {"name": "hsv_hist"},
        "rerank": {"name": "none", "params": {"sv": {"top_k": 5, "min_inliers": 8}}},
    },
    "experiments": [
        {
            "name": "families",
            "ablation": "i",
            "protocols": ["revisited", "original"],
            "runs": [
                {"extractor": {"name": "hog"}},
                {"extractor": {"name": "hsv_hist"}},
                {"extractor": {"name": "resnet50", "image_size": 64, "pretrained": False}},
            ],
        },
        {
            "name": "pooling",
            "ablation": "b",
            "base": {"extractor": {"name": "resnet50", "image_size": 64, "pretrained": False}},
            "grid": {"pooling.name": ["gap", "gem", "cls"]},
        },
        {
            "name": "pca",
            "ablation": "g",
            "grid": {"postprocess": [{"pca": "none"}, {"pca": "whiten", "dim": 16}]},
        },
        {
            "name": "hnsw",
            "ablation": "c",
            "base": {"index": {"type": "hnsw", "params": {"m": 8}}},
            "grid": {"index.params.ef": [2, 16]},
        },
        {
            "name": "ivf",
            "ablation": "d",
            "base": {"index": {"type": "ivfpq", "params": {"nlist": 2, "m": 16, "nbits": 4}}},
            "grid": {"index.params.nprobe": [1, 2]},
        },
        {
            "name": "pq",
            "ablation": "a",
            "runs": [
                {"index": {"type": "flat"}},
                {"index": {"type": "ivfpq", "params": {"nlist": 1, "m": 8, "nbits": 4, "nprobe": 1}}},
            ],
        },
        {"name": "rerank", "ablation": "f", "grid": {"rerank.name": ["none", "aqe", "sv"]}},
        {"name": "scale", "ablation": "h", "grid": {"distractors": [0, 10, 500]}},
        {
            "name": "ms",
            "ablation": "j",
            "base": {"extractor": {"name": "resnet50", "image_size": 64, "pretrained": False}},
            "grid": {"extractor.scales": [[1.0], [1.0, 0.7]]},
        },
    ],
}


@pytest.fixture
def mini_config(tmp_path: Path) -> Path:
    path = tmp_path / "mini.yaml"
    path.write_text(yaml.safe_dump(MINI))
    return path


def test_grid_expansion_dedupes_and_merges() -> None:
    exp = ExperimentCfg(
        name="x",
        ablation="c",
        base={"index": {"type": "hnsw", "params": {"m": 8}}},
        runs=[{}, {"pooling": {"name": "gem"}}],
        grid={"index.params.ef": [1, 2, 2]},
    )
    runs = expand_experiment({"extractor": {"name": "resnet50"}}, exp)
    assert len(runs) == 4
    assert {r.index.params["ef"] for r in runs} == {1, 2}
    assert all(r.index.params["m"] == 8 for r in runs)


def test_invalid_ablation_letter_is_rejected() -> None:
    with pytest.raises(ValueError):
        ExperimentCfg(name="x", ablation="z")


@pytest.mark.parametrize("name", ["fast.yaml", "full.yaml", "ci.yaml"])
def test_shipped_configs_load(name: str) -> None:
    cfg = load_benchmark_config(CONFIG_DIR / name)
    assert isinstance(cfg, BenchmarkConfig)
    assert {e.ablation for e in cfg.experiments} >= {"a", "b", "c", "d", "f", "g", "h", "i", "j"}


def test_dataset_subset_env_overrides_file() -> None:
    assert load_benchmark_config(CONFIG_DIR / "fast.yaml", subset_override=300).subset == 300


def test_end_to_end_harness(toy_data_dir: Path, tmp_path: Path, mini_config: Path) -> None:
    results = tmp_path / "results"
    settings = make_settings(toy_data_dir, results)
    cfg = load_benchmark_config(mini_config)
    runner = BenchmarkRunner(cfg, settings, results, resume=True, show_progress=False)
    records, meta = runner.run_all()
    by_status: dict[str, list] = {}
    for r in records:
        by_status.setdefault(r["status"], []).append(r)
    assert not by_status.get("failed"), [r["reason"] for r in by_status.get("failed", [])]
    skipped = {r["label"]: r["reason"] for r in by_status.get("skipped", [])}
    assert any("cls" in label for label in skipped)  # cls pooling is invalid for a CNN
    assert any("+500d" in label for label in skipped)  # only 10 distractors exist

    ok = by_status["ok"]
    assert {r["protocol"] for r in ok} == {"revisited", "original"}
    one = next(r for r in ok if r["index"] == "flat" and r["protocol"] == "revisited")
    assert set(one["metrics"]) == {"easy", "medium", "hard"}
    for stage in ("extract", "search", "rerank", "total"):
        assert one["latency_ms"][stage]["p95"] is not None
    assert one["memory"]["bytes_per_vector"] > 0
    hnsw = [r for r in ok if r["index"] == "hnsw"]
    assert hnsw and all(0.0 <= r["ann_recall@5"] <= 1.0 for r in hnsw)
    assert any(r["index_reused_from_previous_run"] for r in hnsw)  # ef sweep reuses the build
    pca = next(r for r in ok if r["config"]["postprocess"]["pca"] == "whiten")
    assert pca["pca"]["fit_on"] == "toy_b" and pca["vector_dim"] == 16
    scale = [r for r in ok if r["n_distractors"] == 10]
    assert scale and scale[0]["n_index"] == scale[0]["n_db"] + 10

    paths = write_report(
        records, meta, results, runner.dataset, cfg.hard_query_ap_threshold, cfg.hard_query_top
    )
    data = json.loads(paths["json"].read_text())
    assert data["meta"]["seed"] == 3 and "hardware" in data["meta"] and "libraries" in data["meta"]
    assert {t["key"] for t in data["ablations"]} >= {
        "a",
        "b",
        "b_sweep",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
    }
    for t in data["ablations"]:
        assert (results / t["plot"]).exists()
    csv_lines = paths["csv"].read_text().splitlines()
    assert len(csv_lines) == len(records) + 1
    summary = paths["summary"].read_text()
    assert "## Headline" in summary and "plots/e_pareto_map_vs_p95.png" in summary
    hard = json.loads(paths["hard"].read_text())
    assert "toy_a" in hard["datasets"]
    q = hard["datasets"]["toy_a"]["queries"][0]
    assert q["returned"] and {x["label"] for x in q["returned"]} <= {
        "positive",
        "junk",
        "negative",
        "distractor",
    }

    # resume: a second run executes nothing and returns identical records
    runner2 = BenchmarkRunner(cfg, settings, results, resume=True, show_progress=False)
    records2, _ = runner2.run_all()
    assert [r["run_id"] for r in records2] == [r["run_id"] for r in records]
    assert [r.get("headline_map") for r in records2] == [r.get("headline_map") for r in records]


def test_same_seed_same_metrics(toy_data_dir: Path, tmp_path: Path) -> None:
    cfg_dict = {**MINI, "experiments": [MINI["experiments"][0]]}
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg_dict))
    cfg = load_benchmark_config(path)
    out = []
    for i in range(2):
        results = tmp_path / f"r{i}"
        runner = BenchmarkRunner(cfg, make_settings(toy_data_dir, results), results, show_progress=False)
        out.append([r.get("headline_map") for r in runner.run_all()[0]])
    assert out[0] == out[1]
