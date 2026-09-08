# -*- coding: utf-8 -*-
"""Заморозить предрегистрацию ступени V1b: хэш спецификации, байтовая копия
и хэш конфига стенда.

Порядок обязателен (спецификация, раздел 3, правило v0.11): текст критериев и
допусков замораживается ДО первого прогона ступени в любом режиме, включая
разведочный. После заморозки критерии не редактируются; если критерий оказался
не о той величине, заводится новая ступень с новым именем.

Запуск:  python scripts/freeze_v1b.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SPEC = Path(r"E:\Neuro\Simulation\whitepaper\experiment-spec-h1-h3.md")
OUT = HERE / "results" / "v1b"
VERSION = "v0.14"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def main() -> int:
    if not SPEC.exists():
        print("спецификация не найдена: %s" % SPEC, file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)

    frozen = OUT / ("experiment-spec-h1-h3.%s.frozen.md" % VERSION)
    if frozen.exists():
        print("замороженная копия уже существует: %s" % frozen.name, file=sys.stderr)
        print("повторная заморозка запрещена правилом v0.11", file=sys.stderr)
        return 1
    shutil.copy2(SPEC, frozen)

    spec_hash = sha256(SPEC)
    size = SPEC.stat().st_size

    # Конфиг стенда: всё, что задаёт прогон, кроме двух калибруемых параметров.
    # Их значения появляются после калибровки и хэшируются отдельно вместе с
    # картой сетки (V1b-4.6).
    cfg = {
        "stage": "V1b",
        "spec_version": VERSION,
        "substrate": "FlyWire v630, подсхема mb_subcircuit",
        "n_neurons": 6087,
        "n_edges": 555544,
        "substitutions": {
            "dan_mask": {"edges_removed": 49316,
                         "targets": ["Kenyon_Cell", "MBON"], "source": "DAN"},
            "graded_apl": {"form": "g_apl*max(0, v - v_0)",
                           "membrane": "из [2], без порога, сброса и рефрактерности"},
            "odor_input": {"source": "[39] Hallem & Carlson 2006",
                           "transform": "[42] Olsen 2010, R_max=165, sigma=12, n=1.5, m=10.63",
                           "sides": "both", "glomeruli": 24, "upn": 115},
        },
        "protocol": {"pulse_ms": 1000, "window_ms": 4000, "trials": 6,
                     "mbon_pulse_ms": 5000, "reference_window_ms": 2000},
        "panel": {"n": 14, "calibration": 6, "evaluation": 8},
        "kc_response": {"spikes_per_trial": 1, "min_trials": 3, "n_kc": 5177},
        "criteria": {
            "f_band": [0.03, 0.10], "f_max": 0.10,
            "overlap_separation_min": 0.25, "overlap_ceiling": 0.40,
            "mbon_floor_hz": 2.0, "mbon_ceiling_hz": 67.0,
            "mbon_md_min": 0.19, "mbon_md_types": 5,
            "spikes_ab_target": 2.2,
        },
        "grid": {"scale": "2^k, k=-6..2", "g_apl": "{0} + g_ref*10^(k/2), k=-4..4",
                 "g_ref": "1/(v_th - v_0) = 1/7 mV^-1"},
    }
    cfg_path = OUT / "config_v1b.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    cfg_hash = sha256(cfg_path)

    text = """# Предрегистрация ступени V1b: хэши, зафиксированные до первого прогона.
# Считаются до прогона ступени в любом режиме, включая разведочный
# (спецификация, правило v0.11). Сами файлы хэшей в спецификацию не входят.
#
# Дата заморозки: {date}
# Источник: whitepaper/experiment-spec-h1-h3.md, версия {version}.
experiment-spec-h1-h3.md {size} {spec}
config_v1b.json {cfg_size} {cfg}
#
# Байтовая копия замороженной спецификации лежит рядом:
# {frozen}. Она нужна потому, что рабочий каталог whitepaper/
# под git не заведён и исходник .md существует в одном экземпляре; без копии
# хэш ссылался бы на файл, который может быть переписан.
#
# Живой документ после прогона дополняется результатом и разбором, поэтому его
# текущий хэш этому значению не равен и равным быть не должен.
# Предрегистрацией является замороженная копия рядом.
#
# Два калибруемых параметра режима B (pn_kc_scale и g_apl) в этот конфиг не
# входят: их значения появляются после калибровки и хэшируются отдельно вместе
# с полной картой по сетке (спецификация, V1b-4.6).
#
# Дымовые прогоны кода, выполненные до заморозки и объявленные в отчёте:
# сборка сети, импульс запаха, шесть предъявлений на одном запахе и шесть
# пустых предъявлений (0 спайков, спонтанная частота 0,0000 Гц). Критерии на
# оценочном наборе до заморозки не вычислялись.
""".format(date=date.today().isoformat(), version=VERSION, size=size,
           spec=spec_hash, cfg_size=cfg_path.stat().st_size, cfg=cfg_hash,
           frozen=frozen.name)

    (OUT / "spec_sha256.txt").write_text(text, encoding="utf-8")
    print("заморожено:")
    print("  спецификация %s  %d байт  %s" % (VERSION, size, spec_hash))
    print("  конфиг                      %s" % cfg_hash)
    print("  копия: %s" % frozen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
