"""Crash-safe incremental storage of benchmark runs (JSONL) and per-run rankings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from irbench.utils import atomic_write_json, ensure_dir


class ResultsStore:
    """``<results>/runs.jsonl`` holds one JSON record per completed run (latest wins);
    ``<results>/rankings/<run_id>.json`` holds the top of every query's ranked list."""

    def __init__(self, root: Path) -> None:
        self.root = ensure_dir(Path(root))
        self.runs_file = self.root / "runs.jsonl"
        self.rankings_dir = ensure_dir(self.root / "rankings")

    def load(self) -> dict[str, dict[str, Any]]:
        """All stored records keyed by ``run_id`` (later lines override earlier ones)."""
        out: dict[str, dict[str, Any]] = {}
        if not self.runs_file.exists():
            return out
        with open(self.runs_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue  # a torn final line from an interrupted run
                out[rec["run_id"]] = rec
        return out

    def append(self, record: dict[str, Any]) -> None:
        with open(self.runs_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
            fh.flush()

    def save_rankings(self, run_id: str, payload: dict[str, Any]) -> None:
        atomic_write_json(self.rankings_dir / f"{run_id}.json", payload)

    def load_rankings(self, run_id: str) -> dict[str, Any] | None:
        path = self.rankings_dir / f"{run_id}.json"
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def reset(self) -> None:
        """Forget previous runs (``--no-resume``)."""
        if self.runs_file.exists():
            self.runs_file.unlink()
        for f in self.rankings_dir.glob("*.json"):
            f.unlink()
