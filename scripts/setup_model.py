# -*- coding: utf-8 -*-
"""Клонировать модель Shiu et al. на зафиксированный коммит.

Репозиторий не вендорится: он содержит данные коннектома FlyWire под CC BY-NC 4.0,
и редистрибуция распространила бы ограничение NC на весь стенд (DATA.md, раздел 1).

Данные коннектома лежат внутри клона (86 МБ parquet для v630, 100 МБ для v783) —
внешний архив, о котором говорит его Readme, скачивать не нужно.

Запуск:  python scripts/setup_model.py
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
        print("Клонируем %s -> %s" % (URL, DST.name))
        run(["git", "clone", URL, str(DST)])
    else:
        print("%s уже есть" % DST.name)

    head = run(["git", "rev-parse", "HEAD"], cwd=DST)
    if head != PIN:
        print("HEAD %s != зафиксированного %s; переключаемся" % (head[:12], PIN[:12]))
        run(["git", "fetch", "origin"], cwd=DST)
        run(["git", "checkout", PIN], cwd=DST)
        head = run(["git", "rev-parse", "HEAD"], cwd=DST)

    dirty = run(["git", "status", "--porcelain"], cwd=DST)
    print("HEAD: %s" % head)
    print("Локальные правки: %s" % ("НЕТ (как и должно быть)" if not dirty else dirty))

    missing = [
        n for n in ("model.py", "utils.py", "2023_03_23_connectivity_630_final.parquet")
        if not (DST / n).exists()
    ]
    if missing:
        print("Не хватает файлов: %s" % ", ".join(missing))
        return 1
    print("Модель готова к запуску.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
