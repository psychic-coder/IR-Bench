#!/usr/bin/env python3
"""Generate the synthetic toy datasets (toy_a, toy_b) under DATA_DIR.

They are written in the exact Oxford/Paris on-disk formats (flat vs nested JPEG folders,
original *_query/_good/_ok/_junk.txt files and a revisited gnd_r*.pkl) and are used by the
tests, CI and `make bench-ci`. Their numbers say nothing about real retrieval quality.
"""

from __future__ import annotations

import sys

from irbench.cli import main

if __name__ == "__main__":
    sys.exit(main(["toy", *sys.argv[1:]]))
