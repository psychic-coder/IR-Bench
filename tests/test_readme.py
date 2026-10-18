"""README auto-generated block: placeholder without results, summary with results."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from irbench.readme import END, NO_RESULTS, START, update_readme

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def readme(tmp_path: Path) -> Path:
    path = tmp_path / "README.md"
    path.write_text(f"# Title\n\n## Results\n\n{START}\nold content\n{END}\n\n## After\n")
    return path


def test_no_results_placeholder(readme: Path, tmp_path: Path) -> None:
    changed = update_readme(readme, tmp_path / "summary.md", tmp_path / "results.json")
    text = readme.read_text()
    assert changed and NO_RESULTS in text and "old content" not in text
    assert text.endswith("## After\n")


def test_results_without_successful_runs_still_say_no_results(readme: Path, tmp_path: Path) -> None:
    (tmp_path / "summary.md").write_text("# Benchmark results\n| a |\n")
    (tmp_path / "results.json").write_text(json.dumps({"runs": [{"status": "failed"}]}))
    update_readme(readme, tmp_path / "summary.md", tmp_path / "results.json")
    assert NO_RESULTS in readme.read_text()


def test_summary_is_inserted_with_fixed_links(readme: Path, tmp_path: Path) -> None:
    (tmp_path / "summary.md").write_text("# Benchmark results: `x`\n\n## (e) Pareto\n\n![p](plots/e.png)\n")
    (tmp_path / "results.json").write_text(json.dumps({"runs": [{"status": "ok"}]}))
    update_readme(readme, tmp_path / "summary.md", tmp_path / "results.json")
    text = readme.read_text()
    assert "### Benchmark results: `x`" in text and "#### (e) Pareto" in text
    assert "](results/plots/e.png)" in text
    assert NO_RESULTS not in text
    # idempotent
    assert update_readme(readme, tmp_path / "summary.md", tmp_path / "results.json") is False


def test_missing_markers_raise(tmp_path: Path) -> None:
    path = tmp_path / "README.md"
    path.write_text("# no markers\n")
    with pytest.raises(ValueError):
        update_readme(path, tmp_path / "s.md", tmp_path / "r.json")


@pytest.mark.skipif(not (REPO_ROOT / "README.md").exists(), reason="README.md is not inside the test image")
def test_repository_readme_has_markers() -> None:
    text = (REPO_ROOT / "README.md").read_text()
    assert START in text and END in text
