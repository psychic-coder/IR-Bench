"""Benchmark configuration (YAML) and grid expansion into concrete runs."""

from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from irbench.utils import stable_hash

ABLATIONS = {
    "a": "IVF-PQ: mAP lost per 10x memory saved",
    "b": "GeM vs GAP (vs R-MAC) per backbone, and GeM p sweep",
    "c": "HNSW recall vs ef",
    "d": "IVF-PQ recall vs nprobe",
    "e": "mAP vs p95 latency Pareto frontier",
    "f": "Re-ranking: AQE / alpha-QE / spatial verification",
    "g": "PCA-whitening and output dimension",
    "h": "Scale test: distractors",
    "i": "Classical vs CNN vs Transformer",
    "j": "Multi-scale extraction",
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExtractorCfg(_Strict):
    name: str
    image_size: int | None = None
    scales: list[float] = Field(default_factory=lambda: [1.0])
    pretrained: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


class PoolingCfg(_Strict):
    name: str | None = None  # None -> extractor default
    p: float = 3.0


class PostprocessCfg(_Strict):
    pca: Literal["none", "pca", "whiten"] = "none"
    dim: int | None = None  # None = full dimensionality


class IndexCfg(_Strict):
    type: Literal["flat", "hnsw", "ivfpq"] = "flat"
    params: dict[str, Any] = Field(default_factory=dict)


class RerankCfg(_Strict):
    name: str = "none"  # none | aqe | alpha_qe | sv | aqe+sv | ...
    params: dict[str, dict[str, Any]] = Field(default_factory=dict)


class RunConfig(_Strict):
    """One point of the benchmark grid (dataset/protocol are attached by the runner)."""

    extractor: ExtractorCfg
    pooling: PoolingCfg = Field(default_factory=PoolingCfg)
    postprocess: PostprocessCfg = Field(default_factory=PostprocessCfg)
    index: IndexCfg = Field(default_factory=IndexCfg)
    rerank: RerankCfg = Field(default_factory=RerankCfg)
    distractors: int = 0

    def label(self, resolved_pooling: str | None = None) -> str:
        """Compact human-readable description used in tables and plots.

        ``resolved_pooling`` replaces a ``null`` pooling with the extractor default.
        """
        ext = self.extractor
        pooling = self.pooling.name or resolved_pooling or "default"
        parts = [ext.name] if pooling == "none" else [ext.name, pooling]
        if "gem" in pooling and self.pooling.p != 3.0:
            parts[-1] += f"(p={self.pooling.p:g})"
        if ext.scales != [1.0]:
            parts.append("ms[" + ",".join(f"{s:g}" for s in ext.scales) + "]")
        if self.postprocess.pca != "none":
            parts.append(f"{self.postprocess.pca}{self.postprocess.dim or 'full'}")
        idx = self.index.type
        if self.index.params:
            idx += "(" + ",".join(f"{k}={v}" for k, v in sorted(self.index.params.items())) + ")"
        parts.append(idx)
        if self.rerank.name != "none":
            parts.append(self.rerank.name)
        if self.distractors:
            parts.append(f"+{self.distractors}d")
        return " | ".join(parts)


class DistractorCfg(_Strict):
    dir: str | None = None  # default: DISTRACTOR_DIR or DATA_DIR/distractors
    download_archives: int = 0  # revisitop1m archives to fetch (each ~10k images)
    max_size: int = 10_000
    synthetic_for_toy: bool = True  # toy datasets get generated distractors


class HeadlineCfg(_Strict):
    protocol: Literal["revisited", "original"] = "revisited"
    setup: Literal["easy", "medium", "hard", "all"] = "medium"


class ExperimentCfg(_Strict):
    name: str
    ablation: str
    title: str | None = None
    datasets: list[str] | None = None
    protocols: list[Literal["original", "revisited"]] | None = None
    base: dict[str, Any] = Field(default_factory=dict)
    grid: dict[str, list[Any]] = Field(default_factory=dict)
    runs: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> ExperimentCfg:
        if self.ablation not in ABLATIONS:
            raise ValueError(f"ablation must be one of {sorted(ABLATIONS)}")
        return self


class BenchmarkConfig(_Strict):
    """Top-level contents of ``configs/fast.yaml`` / ``configs/full.yaml``."""

    name: str
    seed: int = 42
    datasets: list[str] = Field(default_factory=lambda: ["oxford5k", "paris6k"])
    protocols: list[Literal["original", "revisited"]] = Field(default_factory=lambda: ["revisited"])
    subset: int | None = None
    eval_depth: int = 1000
    ann_recall_k: int = 10
    warmup_queries: int = 5
    batch_size: int = 16
    device: Literal["auto", "cpu", "cuda"] = "auto"
    hard_query_ap_threshold: float = 0.3
    hard_query_top: int = 20
    headline: HeadlineCfg = Field(default_factory=HeadlineCfg)
    distractors: DistractorCfg = Field(default_factory=DistractorCfg)
    defaults: dict[str, Any]
    experiments: list[ExperimentCfg]

    def config_hash(self) -> str:
        return stable_hash(self.model_dump())


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge (``override`` wins; lists are replaced, not concatenated)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def set_dotted(d: dict[str, Any], dotted: str, value: Any) -> dict[str, Any]:
    """Return a copy of ``d`` with ``a.b.c`` set to ``value``."""
    out = copy.deepcopy(d)
    node = out
    keys = dotted.split(".")
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = copy.deepcopy(value)
    return out


def expand_experiment(defaults: dict[str, Any], exp: ExperimentCfg) -> list[RunConfig]:
    """Expand ``base`` + (``runs`` x ``grid``) into concrete :class:`RunConfig` objects."""
    base = deep_merge(defaults, exp.base)
    variants = [deep_merge(base, r) for r in exp.runs] or [base]
    keys = list(exp.grid)
    combos = list(itertools.product(*(exp.grid[k] for k in keys))) if keys else [()]
    out: list[RunConfig] = []
    seen: set[str] = set()
    for variant in variants:
        for combo in combos:
            cfg = variant
            for k, v in zip(keys, combo, strict=True):
                cfg = set_dotted(cfg, k, v)
            rc = RunConfig.model_validate(cfg)
            h = stable_hash(rc.model_dump())
            if h not in seen:
                seen.add(h)
                out.append(rc)
    return out


def load_benchmark_config(path: Path, subset_override: int | None = None) -> BenchmarkConfig:
    """Parse a benchmark YAML; ``DATASET_SUBSET`` (``subset_override``) wins over the file."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = BenchmarkConfig.model_validate(raw)
    if subset_override:
        cfg = cfg.model_copy(update={"subset": subset_override})
    for exp in cfg.experiments:  # validate every run eagerly so typos fail fast
        expand_experiment(cfg.defaults, exp)
    return cfg
