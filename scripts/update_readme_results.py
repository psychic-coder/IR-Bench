#!/usr/bin/env python3
"""Fill README.md's auto-generated results block from results/summary.md.

Usage:
    python scripts/update_readme_results.py [--readme README.md] [--results-dir results]

If results/summary.md is missing, or results/results.json has no successful run, the block
says "No results generated yet" instead of showing numbers. Runs with only the Python
standard library (the repository's backend/src is added to sys.path).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from irbench.readme import update_readme  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--readme", default=str(ROOT / "README.md"))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    args = parser.parse_args()
    results = Path(args.results_dir)
    changed = update_readme(Path(args.readme), results / "summary.md", results / "results.json")
    print("README.md results block updated" if changed else "README.md results block already up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
