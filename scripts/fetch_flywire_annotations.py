# -*- coding: utf-8 -*-
"""Download FlyWire v630 cell-type annotations and verify checksums.

Source: the Schlegel et al. annotation repository [18],
https://github.com/flyconnectome/flywire_annotations, tag `v1.1.0`
(commit df6bb136f5b3d91c3992df4e8de2642329e2a384).

Why this tag specifically. Branch `main` and tags from `v2.0.0` on give `root_id`
for FlyWire materialization **783**, while the model [2] is built on **630**; the
two must not be mixed — identifiers from different materializations don't match.
Tag `v1.1.0` is the last one where `root_id` refers to 630. This is the annotation
version from the Schlegel et al. preprint; the Nature (2024) version is tag `v2.1.0`
on 783. The discrepancy is recorded in DATA.md.

The second file is hemibrain metadata [9], exported from neuPrint. Its `instance`
field is used to build the "type -> mushroom body compartment" map
(see build_mb_subcircuit.py); the FlyWire annotations themselves have no compartments.

Licenses: FlyWire annotations — CC BY-NC 4.0 (the source repository has no LICENSE
file; we treat it under FlyWire's terms, like the rest of the connectome data);
hemibrain — CC BY 4.0. None of this is committed to git (DATA.md).

Run:  python scripts/fetch_flywire_annotations.py
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

COMMIT = "df6bb136f5b3d91c3992df4e8de2642329e2a384"
BASE = "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/%s/supplemental_files/" % COMMIT
FILES = [
    "Supplemental_file1_annotations.tsv",
    "Supplemental_file4_hemibrain_meta.csv",
]

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "flywire_annotations"
CHECKSUMS = ROOT / "results" / "mb_subcircuit" / "checksums.txt"
ATTEMPTS = 5


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def known_checksums() -> dict[str, tuple[int, str]]:
    """{'file': (size, sha256)} from the frozen checksums file."""
    out: dict[str, tuple[int, str]] = {}
    if not CHECKSUMS.exists():
        return out
    for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 3 and not line.startswith("#"):
            out[parts[0]] = (int(parts[1]), parts[2].lower())
    return out


def download(url: str, dst: Path) -> bool:
    tmp = dst.with_suffix(dst.suffix + ".part")
    tmp.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as fh:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
    except Exception as exc:
        print("      %s" % exc)
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(dst)
    return True


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    known = known_checksums()
    failed = []

    for name in FILES:
        dst = DEST / name
        if not dst.exists():
            print("%s: downloading" % name)
            for attempt in range(1, ATTEMPTS + 1):
                if download(BASE + name, dst):
                    break
                print("   attempt %d of %d failed" % (attempt, ATTEMPTS))
            else:
                print("%s: did not download" % name)
                failed.append(name)
                continue
        else:
            print("%s: already present (%d bytes)" % (name, dst.stat().st_size))

        actual = sha256(dst)
        if name in known:
            size, digest = known[name]
            if actual == digest and dst.stat().st_size == size:
                print("   SHA-256 matches")
            else:
                print("   SHA-256 MISMATCH: %s (expected %s)" % (actual, digest))
                failed.append(name)
        else:
            print("   SHA-256 %s (no reference on file, add it to %s)" % (actual, CHECKSUMS.name))

    if failed:
        print("\nNot obtained or not verified: %s" % ", ".join(failed))
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
