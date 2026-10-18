#!/usr/bin/env python3
"""Poll GET /ready until the API reports ready (standard library only).

Usage:
    python scripts/wait_for_ready.py [--url http://localhost:8000] [--timeout 1800]

Exits 0 when ready, 1 on timeout or when the API reports a warm-up error (the error text,
e.g. manual dataset download instructions, is printed).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    deadline = time.monotonic() + args.timeout
    last = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{args.url}/ready", timeout=10) as resp:
                body = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body = json.loads(exc.read() or b"{}")
        except (urllib.error.URLError, ConnectionError, TimeoutError, ValueError) as exc:
            body = {"stage": "unreachable", "message": str(exc)}
        if body.get("ready"):
            print(f"ready: {body.get('message')}")
            return 0
        if body.get("stage") == "error":
            print(f"warm-up failed:\n{body.get('error')}", file=sys.stderr)
            return 1
        line = f"{body.get('stage')}: {body.get('message')} {body.get('progress') or ''}"
        if line != last:
            print(line, flush=True)
            last = line
        time.sleep(args.interval)
    print("timed out waiting for /ready", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
