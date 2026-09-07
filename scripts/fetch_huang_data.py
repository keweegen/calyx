# -*- coding: utf-8 -*-
"""Скачать архивы Huang et al. 2024 с Zenodo и сверить контрольные суммы.

Набор: DOI 10.5281/zenodo.10998457, CC BY 4.0. В репозитории не хранится (DATA.md).

Особенность источника: файловый эндпоинт Zenodo часто отвечает 504 при работающих
метаданных. Поэтому качаем целиком во временный файл, сверяем размер из API и
только потом переименовываем. Докачка (HTTP Range) для этого источника непригодна:
на 504 в файл пишется HTML-страница ошибки, и продолжение дописывает данные поверх.

Запуск:  python scripts/fetch_huang_data.py [Figure3 Figure4 ...]
         python scripts/fetch_huang_data.py --all
Без аргументов качает то, что нужно для эталона H1b: Figure3 и Figure4.
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
    """{'Figure3.zip': (size, sha256)} из зафиксированного файла сумм."""
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
    """{'Figure3.zip': (size, url)} из метаданных записи."""
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
    except Exception as exc:                      # 504 и обрывы — обычное дело
        print("      %s" % exc)
        tmp.unlink(missing_ok=True)
        return False
    got = tmp.stat().st_size
    if got != want_size:
        print("      размер %d, ожидался %d" % (got, want_size))
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
            print("%s: нет в записи Zenodo" % key)
            failed.append(key)
            continue
        want_size, url = remote[key]
        dst = DATA / key

        if dst.exists() and dst.stat().st_size == want_size:
            print("%s: уже на месте (%d байт)" % (key, want_size))
        else:
            print("%s: качаем %.0f МБ" % (key, want_size / 1048576))
            for attempt in range(1, ATTEMPTS + 1):
                if download(url, dst, want_size):
                    break
                print("   попытка %d из %d не удалась" % (attempt, ATTEMPTS))
            else:
                print("%s: не скачался за %d попыток" % (key, ATTEMPTS))
                failed.append(key)
                continue

        if key in known:
            size, digest = known[key]
            actual = sha256(dst)
            if actual == digest and dst.stat().st_size == size:
                print("   SHA-256 совпадает")
            else:
                print("   SHA-256 НЕ СОВПАДАЕТ: %s (ожидался %s)" % (actual, digest))
                failed.append(key)
        else:
            print("   SHA-256 %s (эталона нет, записать в %s)" % (sha256(dst), CHECKSUMS.name))

    if failed:
        print("\nНе получено или не сверено: %s" % ", ".join(failed))
        return 1
    print("\nГотово.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
