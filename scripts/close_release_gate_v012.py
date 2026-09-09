# -*- coding: utf-8 -*-
"""Закрытие ворот выпуска v0.12: исходы G1 и G2 против объявленных условий.

Ворота записаны 9 сентября 2026 в results/v1c/release_gate_v012.json, хэш
185cf07d9b3176bd, ДО выполнения обеих проверок. Здесь записаны их исходы.
Скрипт проверяет целостность ворот перед записью: условие, изменённое после
результата, обесценило бы обе проверки.

Запуск:  python scripts/close_release_gate_v012.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "v1c"
GATE_HASH_PREFIX = "185cf07d9b3176bd"

STATUS = {
    "artefact_kind": "исходы ворот выпуска v0.12",
    "date": "2026-09-10",
    "gate": {"file": "release_gate_v012.json",
             "sha256_prefix": GATE_HASH_PREFIX,
             "written": "2026-09-09, до выполнения обеих проверок"},

    "G1": {
        "condition": "закрыть Figure S3 работы Liu et al. 2022",
        "outcome": "FOUND",
        "how": "опубликованная версия недостижима — Europe PMC отвечает "
               "«Article with id PMC8825683 is not open access one», blob-"
               "эндпоинт PMC отдаёт заглушку. Легенда получена из препринта "
               "bioRxiv 10.1101/2021.08.19.456845v1, где фигура имеет номер S4; "
               "PDF скачан и текст извлечён лично",
        "what_it_says": "грибовидное тело входит ОДНОЙ агрегатной точкой с "
                        "усами: среднее и стандартное отклонение амплитуды ВПСП "
                        "из популяционных записей Turner 2008 против плотности, "
                        "усреднённой по всем 10 739 связям hemibrain. "
                        "«highly consistent with those for PN-LHN connections, "
                        "although modestly lower than predicted»",
        "consequence": "роль Liu ОСЛАБЛЕНА, а не усилена: правило поддержано на "
                       "уровне популяции и молчит на уровне подтипов, где лежит "
                       "наш вопрос. Подтип-разрешённое сравнение в этой фигуре "
                       "невозможно по построению — физиология в ней "
                       "популяционная",
        "detail": "поле figure_S3_closed в kc_classification_result.json",
    },

    "G2": {
        "condition": "закрыть поиск численных значений порога генерации спайка "
                     "для αβ, α′β′ и γ",
        "outcome": "PARTIALLY-FOUND",
        "found": [
            {"quantity": "расстояние от потенциала покоя до порога",
             "subtype": "αβ core",
             "value_mV": 8.2,
             "source": "Groschner L.N., Hak L.C.W., Bogacz R., DasGupta S., "
                       "Miesenböck G. 2018, Cell 173:894-905.e13, "
                       "doi:10.1016/j.cell.2018.03.075, PMID 29706545, "
                       "PMC5947940",
             "quote": "During stimulus trains, the membrane potentials of abc "
                      "KCs in wild-type flies climbed in a stepwise fashion "
                      "from resting potential to spike threshold, bridging the "
                      "average potential difference of 8.2 mV by integrating "
                      "4-30 synaptic quanta",
             "verified_by_me": True},
            {"quantity": "тот же зазор у α′β′ — КАЧЕСТВЕННО, числа нет",
             "subtype": "α′β′",
             "source": "Groschner et al. 2018, тот же абзац",
             "quote": "By contrast, just one or two EPSPs could close the "
                      "narrow voltage gap between the resting potentials and "
                      "spike thresholds of a'b' KCs, obviating the need for "
                      "extensive synaptic integration",
             "verified_by_me": True},
        ],
        "not_found": [
            "абсолютные значения порога в мВ по трём подтипам: ни в одной "
            "работе не приведены в тексте или таблице",
            "дополнительные материалы Inada et al. 2017 получены "
            "(kazamalab.riken.jp, 5,3 МБ) — численных значений порога, "
            "потенциала покоя и входного сопротивления по подтипам в них НЕТ; "
            "Figure S4 приложения даёт наклон другой зависимости и тоже только "
            "графиком",
            "Chen, Huang et al. 2026, Curr Biol, doi:10.1016/j.cub.2026.02.028 "
            "— единственная работа, где порог, потенциал покоя, сопротивление и "
            "постоянная времени измерены раздельно по ВСЕМ ТРЁМ подтипам "
            "(Figure S2, n = 7/5, 4/12, 22/10), но числа даны только точками на "
            "графиках; сравнение подтипов между собой авторы не проводили — их "
            "ось сравнения сон против депривации",
            "Turner et al. 2008 — разбивка по подтипам есть только для ширины "
            "настройки и числа спайков за ответ (α′β′ 4,9 ± 3,0 против αβ "
            "2,2 ± 1,2, P = 0,007); порог, потенциал покоя и сопротивление даны "
            "единым пулом: Vspike − Vrest = 21,5 ± 5,6 мВ, n = 17",
            "Greenin-Whitehead et al. 2025, J Physiol — намеренно работали "
            "только с γ, чтобы не смешивать подтипы; сравнения между подтипами "
            "нет",
        ],
        "not_checked": ["Gruntman & Turner 2013", "Honegger et al. 2011",
                        "Murthy et al. 2008", "Lin et al. 2014",
                        "Papadopoulou et al. 2011", "Hige et al. 2015",
                        "Cohn, Morantte & Ruta 2015", "Abdelrahman et al.",
                        "транскриптомные работы на предмет функциональной "
                        "валидации"],
    },

    "what_G2_changes": {
        "second_independent_line_for_M1": "Groschner 2018 — другая лаборатория, "
            "другая задача (принятие решений, не обоняние), другой протокол — "
            "даёт то же направление, что Inada 2017: зазор «покой-порог» у α′β′ "
            "УЗКИЙ и закрывается одним-двумя ВПСП, тогда как у αβ core он "
            "составляет 8,2 мВ и требует интеграции 4-30 квантов. Направление "
            "M1 подтверждено дважды, независимо.",
        "still_no_quantitative_replacement": "числа для α′β′ нет ни в одной "
            "работе. Сверх того две опубликованные величины одного и того же "
            "по смыслу зазора для клеток Кеньона расходятся между работами в "
            "2,6 раза: 8,2 мВ у Groschner для αβ core против 21,5 ± 5,6 мВ "
            "пулом у Turner. Разные препараты и протоколы, и это само по себе "
            "показывает, что параметризовать модель по литературе нельзя.",
    },

    "gate_result": "ОБА УСЛОВИЯ ЗАКРЫТЫ. Выпуск консолидированной ревизии v0.12 "
                   "разрешён. Разрешение механизмов M2 и M3 в ворота не "
                   "входило и условием выпуска не является.",
}


def main() -> int:
    gate = OUT / "release_gate_v012.json"
    if not gate.exists():
        print("ворота не найдены: %s" % gate, file=sys.stderr)
        return 1
    h = hashlib.sha256(gate.read_bytes()).hexdigest()
    if not h.startswith(GATE_HASH_PREFIX):
        print("ворота изменились после записи: %s против %s"
              % (h[:16], GATE_HASH_PREFIX), file=sys.stderr)
        return 1
    p = OUT / "release_gate_v012_status.json"
    p.write_text(json.dumps(STATUS, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    print("ворота целы: %s" % h[:16])
    print("G1: %s" % STATUS["G1"]["outcome"])
    print("G2: %s" % STATUS["G2"]["outcome"])
    print()
    print(STATUS["what_G2_changes"]["second_independent_line_for_M1"])
    print()
    print(STATUS["what_G2_changes"]["still_no_quantitative_replacement"])
    print()
    print(STATUS["gate_result"])
    print("\nзаписано: %s" % p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
