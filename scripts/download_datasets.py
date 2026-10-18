#!/usr/bin/env python3
"""Download (or verify manually placed) Oxford5k, Paris6k and revisited ground truth.

Usage (inside the backend container or any environment with irbench installed):
    python scripts/download_datasets.py [--datasets oxford5k paris6k] [--distractor-archives 1]

Exit code 2 means something could not be downloaded; the printed message names the page
to open and the ./data/<folder> to place the files in. Re-running continues from local files.
"""

from __future__ import annotations

import sys

from irbench.cli import main

if __name__ == "__main__":
    sys.exit(main(["download", *sys.argv[1:]]))
