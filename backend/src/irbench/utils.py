"""Shared helpers: seeding, hashing, device selection, timing, hardware reporting."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

TRACKED_LIBRARIES = (
    "numpy",
    "torch",
    "torchvision",
    "transformers",
    "faiss-cpu",
    "faiss-gpu",
    "qdrant-client",
    "opencv-python-headless",
    "scikit-image",
    "pillow",
    "fastapi",
    "pydantic",
    "matplotlib",
)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, OpenCV and (if installed) PyTorch; enable deterministic kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import cv2

        cv2.setRNGSeed(seed)
    except ImportError:  # pragma: no cover - cv2 is a hard dependency in practice
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # pragma: no cover
        pass


def stable_hash(obj: Any, length: int = 16) -> str:
    """Deterministic short hash of a JSON-serialisable object (key order independent)."""
    payload = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of raw bytes (used for upload cache keys)."""
    return hashlib.sha256(data).hexdigest()


def resolve_device(preference: str = "auto") -> str:
    """Return ``"cuda"`` when requested/available, otherwise ``"cpu"``."""
    if preference == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:  # pragma: no cover
        pass
    return "cpu"


def configure_threads(num_threads: int | None) -> None:
    """Pin the number of intra-op threads for torch and FAISS (reproducible timings)."""
    if not num_threads:
        return
    try:
        import torch

        torch.set_num_threads(num_threads)
    except ImportError:  # pragma: no cover
        pass
    try:
        import faiss

        faiss.omp_set_num_threads(num_threads)
    except ImportError:  # pragma: no cover
        pass


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    """Context manager measuring wall-clock milliseconds into ``result["ms"]``."""
    result: dict[str, float] = {"ms": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["ms"] = (time.perf_counter() - start) * 1000.0


def latency_stats(values_ms: Sequence[float]) -> dict[str, float | None]:
    """p50/p95/p99/mean of a latency sample in milliseconds (``None`` when empty)."""
    if len(values_ms) == 0:
        return {"p50": None, "p95": None, "p99": None, "mean": None, "n": 0}
    arr = np.asarray(values_ms, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(arr.mean()),
        "n": int(arr.size),
    }


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def library_versions() -> dict[str, str]:
    """Installed versions of the libraries that influence results."""
    out: dict[str, str] = {}
    for name in TRACKED_LIBRARIES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def hardware_info(device: str) -> dict[str, Any]:
    """Describe the machine the benchmark ran on."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": _cpu_model(),
        "cpu_logical_cores": os.cpu_count(),
        "device": device,
    }
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 2)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        info["torch_threads"] = torch.get_num_threads()
        if device == "cuda" and torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
    except ImportError:  # pragma: no cover
        pass
    return info


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def ensure_dir(path: Path) -> Path:
    """``mkdir -p`` that returns the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_json(path: Path, obj: Any) -> None:
    """Write JSON atomically (tmp file + rename) so readers never see partial files."""
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    os.replace(tmp, path)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def nan_to_none(value: float | None) -> float | None:
    """Convert NaN/inf to ``None`` for JSON output."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None
