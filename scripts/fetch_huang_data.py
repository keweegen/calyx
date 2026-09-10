# -*- coding: utf-8 -*-
"""Download the Huang et al. 2024 archives from Zenodo and verify checksums.

Dataset: DOI 10.5281/zenodo.10998457, CC BY 4.0. Not stored in the repository (DATA.md).

Source quirk: Zenodo's file endpoint often returns 504 while the metadata endpoint
works fine. So we download the whole thing to a temporary file, verify the size
against the API, and only then rename it. Resuming (HTTP Range) is unsuitable for
this source: on a 504 the file gets an HTML error page written into it, and resuming
would append data on top of that.

Run:  python scripts/fetch_huang_data.py [Figure3 Figure4 ...]
      python scripts/fetch_huang_data.py --all
With no arguments, downloads what's needed for reference H1b: Figure3 and Figure4.
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

RECORD = "10998457"
API = "https://zenodo.org/api/records/%s" % RECORD
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CHECKSUMS = ROOT / "results" / "huang_reference" / "checksums.txt"
DEFAULT = ["Figure3", "Figure4"]
ATTEMPTS = 25


def known_checksums() -> dict[str, tuple[int, str]]:
    """{'Figure3.zip': (size, sha256)} from the frozen checksums file."""
    out: dict[str, tuple[int, str]] = {}
    if not CHECKSUMS.exists():
        return out
    for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 3:
            out[parts[0]] = (int(parts[1]), parts[2].lower())
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def remote_index() -> dict[str, tuple[int, str]]:
    """{'Figure3.zip': (size, url)} from the record metadata."""
    with urllib.request.urlopen(API, timeout=60) as r:
        rec = json.load(r)
    return {f["key"]: (f["size"], f["links"]["self"]) for f in rec["files"]}


def download(url: str, dst: Path, want_size: int) -> bool:
    tmp = dst.with_suffix(dst.suffix + ".part")
    tmp.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as fh:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
    except Exception as exc:                      # 504s and drops are routine
        print("      %s" % exc)
        tmp.unlink(missing_ok=True)
        return False
    got = tmp.stat().st_size
    if got != want_size:
        print("      size %d, expected %d" % (got, want_size))
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(dst)
    return True


def main(argv: list[str]) -> int:
    names = DEFAULT
    if "--all" in argv:
        names = ["Figure%d" % i for i in range(1, 6)]
    elif len(argv) > 1:
        names = [a for a in argv[1:] if not a.startswith("-")]

    DATA.mkdir(parents=True, exist_ok=True)
    known = known_checksums()
    remote = remote_index()
    failed = []

    for stem in names:
        key = stem if stem.endswith(".zip") else stem + ".zip"
        if key not in remote:
            print("%s: not in the Zenodo record" % key)
            failed.append(key)
            continue
        want_size, url = remote[key]
        dst = DATA / key

        if dst.exists() and dst.stat().st_size == want_size:
            print("%s: already present (%d bytes)" % (key, want_size))
        else:
            print("%s: downloading %.0f MB" % (key, want_size / 1048576))
            for attempt in range(1, ATTEMPTS + 1):
                if download(url, dst, want_size):
                    break
                print("   attempt %d of %d failed" % (attempt, ATTEMPTS))
            else:
                print("%s: did not download in %d attempts" % (key, ATTEMPTS))
                failed.append(key)
                continue

        if key in known:
            size, digest = known[key]
            actual = sha256(dst)
            if actual == digest and dst.stat().st_size == size:
                print("   SHA-256 matches")
            else:
                print("   SHA-256 MISMATCH: %s (expected %s)" % (actual, digest))
                failed.append(key)
        else:
            print("   SHA-256 %s (no reference on file, add it to %s)" % (sha256(dst), CHECKSUMS.name))

    if failed:
        print("\nNot obtained or not verified: %s" % ", ".join(failed))
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
