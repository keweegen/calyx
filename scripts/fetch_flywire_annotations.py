# -*- coding: utf-8 -*-
"""Скачать разметку типов клеток FlyWire v630 и сверить контрольные суммы.

Источник: репозиторий аннотаций Schlegel et al. [18],
https://github.com/flyconnectome/flywire_annotations, тег `v1.1.0`
(коммит df6bb136f5b3d91c3992df4e8de2642329e2a384).

Почему именно этот тег. Ветка `main` и теги от `v2.0.0` дают `root_id` для
материализации FlyWire **783**, а модель [2] построена на **630**; смешивать
нельзя — идентификаторы разных материализаций не совпадают. Тег `v1.1.0` —
последний, где `root_id` относится к 630. Это версия аннотаций из препринта
Schlegel et al.; версия из Nature (2024) — тег `v2.1.0` на 783. Расхождение
зафиксировано в DATA.md.

Второй файл — метаданные hemibrain [9], выгруженные из neuPrint. Из его поля
`instance` строится карта «тип → компартмент грибовидного тела»
(см. build_mb_subcircuit.py); в самих аннотациях FlyWire компартментов нет.

Лицензии: аннотации FlyWire — CC BY-NC 4.0 (в репозитории-источнике файла
LICENSE нет; относим к условиям FlyWire, как остальные данные коннектома);
hemibrain — CC BY 4.0. В git ничего из этого не кладётся (DATA.md).

Запуск:  python scripts/fetch_flywire_annotations.py
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
    """{'файл': (размер, sha256)} из зафиксированного файла сумм."""
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
            print("%s: качаем" % name)
            for attempt in range(1, ATTEMPTS + 1):
                if download(BASE + name, dst):
                    break
                print("   попытка %d из %d не удалась" % (attempt, ATTEMPTS))
            else:
                print("%s: не скачался" % name)
                failed.append(name)
                continue
        else:
            print("%s: уже на месте (%d байт)" % (name, dst.stat().st_size))

        actual = sha256(dst)
        if name in known:
            size, digest = known[name]
            if actual == digest and dst.stat().st_size == size:
                print("   SHA-256 совпадает")
            else:
                print("   SHA-256 НЕ СОВПАДАЕТ: %s (ожидался %s)" % (actual, digest))
                failed.append(name)
        else:
            print("   SHA-256 %s (эталона нет, записать в %s)" % (actual, CHECKSUMS.name))

    if failed:
        print("\nНе получено или не сверено: %s" % ", ".join(failed))
        return 1
    print("\nГотово.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
