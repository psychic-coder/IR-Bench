#!/usr/bin/env python3
"""Run a benchmark configuration and write every artefact to RESULTS_DIR.

Usage:
    python scripts/run_benchmark.py --config fast.yaml            # DATASET_SUBSET=300 for smoke mode
    python scripts/run_benchmark.py --config full.yaml --no-resume

Equivalent to ``irbench bench``; see ``irbench bench --help`` for all flags.
"""

from __future__ import annotations

import sys

from irbench.cli import main

if __name__ == "__main__":
    sys.exit(main(["bench", *sys.argv[1:]]))
