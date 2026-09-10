# -*- coding: utf-8 -*-
"""Clone the Shiu et al. model at a pinned commit.

The repository is not vendored: it contains FlyWire connectome data under CC BY-NC 4.0,
and redistribution would spread the NC restriction over the entire testbed (DATA.md, section 1).

The connectome data lives inside the clone (86 MB parquet for v630, 100 MB for v783) —
the external archive its Readme mentions does not need to be downloaded.

Run:  python scripts/setup_model.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

URL = "https://github.com/philshiu/Drosophila_brain_model.git"
PIN = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "Drosophila_brain_model"


def run(args: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(
        args, cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    if not DST.exists():
        print("Cloning %s -> %s" % (URL, DST.name))
        run(["git", "clone", URL, str(DST)])
    else:
        print("%s already present" % DST.name)

    head = run(["git", "rev-parse", "HEAD"], cwd=DST)
    if head != PIN:
        print("HEAD %s != pinned %s; switching" % (head[:12], PIN[:12]))
        run(["git", "fetch", "origin"], cwd=DST)
        run(["git", "checkout", PIN], cwd=DST)
        head = run(["git", "rev-parse", "HEAD"], cwd=DST)

    dirty = run(["git", "status", "--porcelain"], cwd=DST)
    print("HEAD: %s" % head)
    print("Local changes: %s" % ("NONE (as expected)" if not dirty else dirty))

    missing = [
        n for n in ("model.py", "utils.py", "2023_03_23_connectivity_630_final.parquet")
        if not (DST / n).exists()
    ]
    if missing:
        print("Missing files: %s" % ", ".join(missing))
        return 1
    print("Model is ready to run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
