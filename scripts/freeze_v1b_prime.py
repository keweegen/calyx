# -*- coding: utf-8 -*-
"""Freeze the pre-registration of stage V1b′: spec hash, byte copy,
and testbed config hash.

The order is mandatory (spec, section 3, rule v0.11): the text of the criteria
and tolerances is frozen BEFORE the first run of the stage in any mode, including
exploratory. After freezing the criteria are not edited; if a criterion turns out
to be about the wrong quantity, a new stage is started under a new name.

V1b′ is such a stage. V1b was closed with outcome NOT-TESTABLE due to an
execution error: the admissibility function implemented V1b-4.5 only halfway.
The differences between V1b′ and V1b are listed in spec section 3ж and duplicated
in the config below; the thresholds, grid, seeds and point-selection rule do
not change.

Run:  python scripts/freeze_v1b_prime.py
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
OUT = HERE / "results" / "v1b_prime"
VERSION = "v0.15"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def main() -> int:
    if not SPEC.exists():
        print("spec not found: %s" % SPEC, file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)

    frozen = OUT / ("experiment-spec-h1-h3.%s.frozen.md" % VERSION)
    if frozen.exists():
        print("frozen copy already exists: %s" % frozen.name, file=sys.stderr)
        print("re-freezing is forbidden by rule v0.11", file=sys.stderr)
        return 1
    shutil.copy2(SPEC, frozen)

    spec_hash = sha256(SPEC)
    size = SPEC.stat().st_size

    cfg = {
        "stage": "V1b'",
        "spec_version": VERSION,
        "spec_section": "3ж",
        "supersedes": {
            "stage": "V1b", "spec_version": "v0.14",
            "outcome": "NOT-TESTABLE",
            "reason": "ошибка исполнения: допустимость реализована без "
                      "ограничения MBON на C (V1b-4.5)",
            "report": "results/v1b/report_v1b_closure.md",
        },
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
        "protocol": {
            "P14": {"pulse_ms": 1000, "window_ms": 4000, "trials": 6},
            "P14_M": {"pulse_ms": 5000, "window_ms": 5000, "trials": 6,
                      "window_start": "t_on", "subtract_baseline": False,
                      "aggregation": "среднее по клеткам типа, внутри клетки — "
                                     "среднее по пробам"},
            "reference_window_ms": 2000,
        },
        "seeds": {
            "calibration_P14_C": 20260908,
            "calibration_P14M_C": 20260958,
            "evaluation_P14_all14": 20261008,
            "evaluation_P14M_all14": 20261108,
            "empty_presentation": 20261208,
            "permutation_null": 20261308,
            "note": "пробы получают базу плюс 1..6; отрезки не пересекаются; "
                    "прогон при g_APL = 0 идёт на зёрнах 20261008 (парная сверка)",
        },
        "panel": {"n": 14, "calibration": 6, "evaluation": 8},
        "kc_response": {"spikes_per_trial": 1, "min_trials": 3, "n_kc": 5177},
        "criteria": {
            "f_band": [0.03, 0.10], "f_max": 0.10,
            "overlap_separation_min": 0.25, "overlap_ceiling": 0.40,
            "mbon_floor_hz": 2.0, "mbon_ceiling_hz": 67.0,
            "mbon_md_min": 0.19, "mbon_md_types": 5,
            "spikes_ab_target": 2.2,
        },
        "admissibility": {
            "candidate": "среднее f(o) в [0,03; 0,10] и max f(o) <= 0,10 на C",
            "admissible": "кандидат И V1b-3.3 И V1b-3.4 И V1b-3.5 на C "
                          "при тайминге P14-M",
            "mbon_evaluated_for": "только кандидаты (конъюнкция с коротким замыканием)",
            "not_computed_state": "запрещено в завершённом прогоне; наличие в "
                                  "итоговых артефактах есть ошибка исполнения",
            "md_undefined": "если max_o r_t(o) + min_o r_t(o) = 0, MD_t не "
                            "определена и считается не выполнившей V1b-3.5",
        },
        "grid": {
            "stage1_scale": "2^k, k=-6..2",
            "stage1_g_apl": "{0} + g_ref*10^(k/2), k=-4..4",
            "g_ref": "1/(v_th - v_0) = 1/7 mV^-1",
            "stage2_bbox_over": "кандидаты стадии 1 (отклонение от V1b-4.4, "
                                "обоснование в разделе 3ж)",
            "stage2_steps": "2^(1/4) по масштабу, 10^(1/8) по гейну",
            "stage2_expansion_clipped": False,
            "stage2_zero_gain_row": "входит тогда и только тогда, когда среди "
                                    "кандидатов стадии 1 есть точка с g_apl = 0",
            "shared_nodes_recomputed": True,
        },
        "selection": {
            "pool": "допустимые точки стадии 2",
            "tie_breaks": ["|s_ab - 2.2|",
                           "max d(p), d = расстояние Чебышёва до недопустимой "
                           "точки или узла вне сетки; узлы вне сетки недопустимы",
                           "min g_apl", "min pn_kc_scale"],
        },
        "outcomes": ["PASS", "FAIL-EVAL", "FAIL-CAL-KC", "FAIL-CAL-MBON",
                     "NOT-TESTABLE"],
        "expected_outcome": "FAIL-CAL-MBON",
        "preconditions": {
            "spec_conformance_test": "test_spec_conformance.py, все проверки "
                                     "проходят (V1b'-0)",
            "pilot_regression": "карта доли на 447 узлах совпадает с пилотом "
                                "побитово (V1b'-1а)",
        },
    }
    cfg_path = OUT / "config_v1b_prime.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    cfg_hash = sha256(cfg_path)

    text = """# Предрегистрация ступени V1b′: хэши, зафиксированные до первого прогона.
# Считаются до прогона ступени в любом режиме, включая разведочный
# (спецификация, правило v0.11). Сами файлы хэшей в спецификацию не входят.
#
# Дата заморозки: {date}
# Источник: whitepaper/experiment-spec-h1-h3.md, версия {version}, раздел 3ж.
experiment-spec-h1-h3.md {size} {spec}
config_v1b_prime.json {cfg_size} {cfg}
#
# Байтовая копия замороженной спецификации лежит рядом:
# {frozen}. Она нужна потому, что рабочий каталог whitepaper/
# под git не заведён и исходник .md существует в одном экземпляре; без копии
# хэш ссылался бы на файл, который может быть переписан.
#
# ПРОИСХОЖДЕНИЕ. Ступень V1b закрыта исходом NOT-TESTABLE по ошибке исполнения
# (отчёт results/v1b/report_v1b_closure.md). Спецификация V1b′ составлена ПОСЛЕ
# просмотра калибровочной сетки пилота V1b — 447 точек, 57 кандидатов, форма
# допустимой по доле области, положение выбранной точки в углу сетки — и ПОСЛЕ
# диагностики механизма: шесть точек полосы при тайминге M, прогон при
# g_apl = 0, снятие рёбер APL→MBON, веса торможения по ролям. Всё это было
# известно автору в момент написания раздела 3ж. Пороги, сетка, зёрна и правило
# выбора точки относительно V1b не менялись.
#
# СЛЕПОТА НАБОРА E СОХРАНЕНА. Ни одна из трёх эталонных пар критерия перекрытия
# не вычислялась; набор E как прогон ступени не запускался. Объявлено одно
# частичное расходование: запах pentyl acetate при тайминге M на зёрнах
# 20261009 и 20261010, 2 пробы, в дымовой проверке кода (отчёт закрытия V1b,
# раздел 6). Тайминга P14, на котором определены критерии доли и перекрытия,
# это не касается.
#
# ОЖИДАЕМЫЙ ИСХОД ОБЪЯВЛЕН ДО ПРОГОНА: FAIL-CAL-MBON. Основание — диагностика
# отчёта закрытия V1b: в шести точках допустимой по доле полосы при тайминге M
# молчат все 97 MBON подсхемы; снятие рёбер APL→MBON поднимает лучший тип T38
# до 0,350 Гц при поле 2 Гц. Ступень прогоняется, а не закрывается
# рассуждением: диагностика сделана на одном запахе при двух пробах, измерены
# шесть точек из 57, и нижний конец полосы не измерен.
#
# Два калибруемых параметра режима B (pn_kc_scale и g_apl) в этот конфиг не
# входят: их значения появляются после калибровки и хэшируются отдельно вместе
# с полной картой по сетке (V1b-4.6).
#
# ПРЕДУСЛОВИЯ ПРОГОНА. До первого прогона в любом режиме: тест соответствия
# кода спецификации (test_spec_conformance.py) проходит целиком; карта доли
# отвечающих на 447 узлах совпадает с картой пилота побитово.
""".format(date=date.today().isoformat(), version=VERSION, size=size,
           spec=spec_hash, cfg_size=cfg_path.stat().st_size, cfg=cfg_hash,
           frozen=frozen.name)

    (OUT / "spec_sha256.txt").write_text(text, encoding="utf-8")
    print("frozen:")
    print("  spec %s  %d bytes  %s" % (VERSION, size, spec_hash))
    print("  config                       %s" % cfg_hash)
    print("  copy: %s" % frozen)
    print()
    print("expected outcome declared before the run: %s" % cfg["expected_outcome"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
