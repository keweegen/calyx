# -*- coding: utf-8 -*-
"""Record stamp 0 of stage V1c: hash of the spec draft and a byte copy.

Stamp 0 is not the stage's pre-registration. It is the stage's design, written
down in formulas and without any numbers that pre-hash measurements could
supply, and hashed BEFORE those measurements run. Rule D (spec, section 1a)
allows diagnostics between the closure of one stage and the freeze of the next
only when stamp 0 already exists: otherwise what was seen could shift the
criteria and grid boundaries toward the observed value without formally
violating pre-registration.

V1c's pre-registration will be a later version of the document, in which the
numbers have been substituted from the formulas of section 3з and the
"Provenance" and "Expected outcome" sections have been filled in. It will get
its own hash from a different script. The difference between stamp 0 and that
version is bounded by condition D3 and listed in "Provenance".

Re-recording stamp 0 is forbidden: the point of the stamp is that it predates
the measurements, and a rewritten stamp does not have that property.

Run:  python scripts/stamp0_v1c.py
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
OUT = HERE / "results" / "v1c"
VERSION = "v0.16"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def main() -> int:
    if not SPEC.exists():
        print("spec not found: %s" % SPEC, file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)

    frozen = OUT / ("experiment-spec-h1-h3.%s.stamp0.md" % VERSION)
    if frozen.exists():
        print("stamp 0 already recorded: %s" % frozen.name, file=sys.stderr)
        print("re-recording is forbidden by rule D1", file=sys.stderr)
        return 1
    shutil.copy2(SPEC, frozen)

    spec_hash = sha256(SPEC)
    size = SPEC.stat().st_size

    # Stamp 0's config holds the stage's design, not its parameters: by
    # construction, none of the numbers V1c-E6 measurements could supply appear here.
    cfg = {
        "stage": "V1c",
        "artefact_kind": "штамп 0 (правило Д1), не предрегистрация",
        "spec_version": VERSION,
        "spec_section": "3з",
        "rule": "Д (раздел 1а)",
        "follows": {
            "stage": "V1b'", "spec_version": "v0.15",
            "outcome": "FAIL-CAL-MBON",
            "report": "results/v1b_prime/report_v1b_prime.md",
            "spec_hash_prefix": "f638fbb510ac35de",
        },
        "knob": {
            "name": "kc_mbon_scale",
            "acts_on": "вес каждого ребра KC->MBON подсхемы",
            "identity_at": "s = 1 тождественно конфигурации V1b'",
            "conditions": [
                "один глобальный скаляр: не по типу, не по компартменту, "
                "не по запаху, не по полушарию, не по классу KC",
                "критерии калибровки только на наивном состоянии",
                "значение фиксируется до H1 и входит в хэш конфига H1",
                "H1 прогоняется минимум в трёх точках полосы s",
                "верхняя граница: одиночный спайк KC не доводит ни один MBON "
                "множества T38 до порога",
            ],
        },
        "grid_s": {
            "s_min": 1,
            "s_max_formula": "(v_th - v_0) / (A * max_m w_max(m)), m в MBON "
                             "типов T38",
            "epsp_peak_formula": "dV_peak(w) = A * w, "
                                 "A = (1/(rho-1)) * (rho^(-1/(rho-1)) - "
                                 "rho^(-rho/(rho-1))), rho = t_mbr / tau",
            "epsp_peak_time": "t* = tau * t_mbr * ln(rho) / (t_mbr - tau)",
            "why_A": "модель [2] токовая: вес есть приращение g, а не v, "
                     "поэтому амплитуда одиночного ВПСП по напряжению меньше w",
            "threshold_strict": "условие спайка v > v_th строгое, поэтому "
                                "s_max достигается и входит в сетку",
            "nodes": "s_k = 10^(k/N_dec), k = 0..K, K = max k при s_k < s_max, "
                     "плюс узел s_max",
            "n_per_decade": 4,
        },
        "grid_other": {
            "points": "57 кандидатов V1b' (артефакт замороженной ступени)",
            "full_2d_grid_repeated": False,
            "fraction_recomputed_at_each_3d_point": True,
            "why": "MBON->APL даёт 2,3 % входа APL, инвариантность "
                   "разреженности к s не установлена",
        },
        "band_and_h1": {
            "band": "значения s, при которых хотя бы одна из 57 точек проходит "
                    "полную конъюнкцию доли и V1b-3.3/3.4/3.5 на C",
            "h1_points": "минимум полосы, максимум полосы, узел ближайший к "
                         "sqrt(s_lo * s_hi); при равенстве меньший узел",
            "short_band": "если узлов меньше трёх - все, число объявляется "
                          "ограничением, узлы не добавляются",
        },
        "outcomes": ["PASS", "FAIL-EVAL", "FAIL-CAL-KC",
                     "FAIL-CAL-MBON-FLOOR", "FAIL-CAL-MBON-CEIL",
                     "NOT-TESTABLE"],
        "expected_outcome": "в штамп 0 не входит (E5 заполняется после "
                            "разрешённых измерений)",
        "testability_check_before_freeze": "если у какого-либо типа T38 все "
                                           "клетки имеют нулевую сумму весов "
                                           "KC->m, ступень NOT-TESTABLE по "
                                           "построению и не прогоняется",
        "pre_hash_measurements_allowed": {
            "E6.1": "структурные веса KC->MBON, симулятор не запускается",
            "E6.2": "диагностика зазора окна доли KC и окна частот MBON при s=1",
            "E6.4": "регрессионный контроль, параметры не меняются",
        },
        "pre_hash_measurements_forbidden": {
            "E6.3-C": "подпороговый потенциал MBON, суммарный ток KC на MBON, "
                      "число одновременных спайков KC до порога - в любой "
                      "конфигурации, включая уже прогнанные точки V1b' при s=1; "
                      "перенесено внутрь V1c как некритериальный выход",
        },
        "knob_axis_for_D2": {
            "parameter": "kc_mbon_scale",
            "forbidden_state_variables": [
                "ток и проводимость синапсов KC->MBON",
                "подпороговый мембранный потенциал MBON и расстояние до порога",
                "число одновременных спайков KC, доводящее MBON до порога",
            ],
            "not_forbidden": "частота спайков MBON - выход ступени, а не "
                             "переменная, на которую ручка действует "
                             "непосредственно",
        },
        "unchanged": [
            "пороги 2 Гц, 67 Гц, 0,19 и правило 5 из 6",
            "полоса доли отвечающих KC и её потолок",
            "панель ступени, разведение наборов C и E, слепота E",
            "таблица зёрен", "протокол P14-M", "источник PN-паттернов",
            "определение ответа KC", "критерий V1a и его нулевое распределение",
        ],
    }
    cfg_path = OUT / "stamp0_v1c.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    cfg_hash = sha256(cfg_path)

    text = """# Штамп 0 ступени V1c: хэши, зафиксированные ДО до-хэшевых измерений.
# Правило Д1 (спецификация, раздел 1а). Это НЕ предрегистрация ступени:
# предрегистрацией станет более поздняя версия с подставленными числами и
# собственным хэшем.
#
# Дата записи: {date}
# Источник: whitepaper/experiment-spec-h1-h3.md, версия {version}, раздел 3з.
experiment-spec-h1-h3.md {size} {spec}
stamp0_v1c.json {cfg_size} {cfg}
#
# Байтовая копия лежит рядом: {frozen}.
# Она нужна потому, что рабочий каталог whitepaper/ под git не заведён и
# исходник .md существует в одном экземпляре; без копии хэш ссылался бы на
# файл, который может быть переписан. Смысл штампа 0 в том, что он старше
# измерений, поэтому копия обязательна и повторная запись запрещена.
#
# ЧТО ЗАПИСАНО ФОРМУЛАМИ И БЕЗ ЧИСЕЛ: критерии и допуски (наследуются от V1b');
# определение третьей ручки kc_mbon_scale и пять её условий; границы сетки по s
# формулами, включая вывод верхней границы из синаптической модели [2];
# шаг сетки; правило полосы и трёх точек H1; исходы, включая раздельные
# FAIL-CAL-MBON-FLOOR и FAIL-CAL-MBON-CEIL; проверка тестируемости до заморозки;
# закрытый перечень разрешённых до-хэшевых измерений и их списков выходов.
#
# ЧТО ЗАПРЕЩЕНО ДО ЗАМОРОЗКИ V1c: измерение подпорогового потенциала MBON,
# суммарного тока KC на MBON и числа одновременных спайков KC до порога.
# Основание: в подпороговом режиме отклик линеен по s, и такое измерение даёт
# значение ручки в других единицах с точностью порядка шага сетки.
#
# ИЗВЕСТНО ДО ШТАМПА 0. Правило Д1 к более ранним измерениям неприменимо.
# Автору штампа 0 известны: исход V1b' и его заголовочные величины (максимум по
# кандидатам от минимума по типам R_t = 0,000000 Гц при поле 2 Гц; лучшая
# частота одного типа 0,3306 Гц; 25 кандидатов из 57 без единого спайка MBON;
# потолок не нарушен нигде) и диагностика механизма отчёта закрытия V1b
# (снятие 101 ребра APL->MBON поднимает лучший тип до 0,350 Гц, три типа из
# шести на нуле; медианный вес узла на MBON 1,4 мВ против 2,4 мВ на KC;
# доля MBON во входе APL 2,3 %). Ни одна из этих величин не есть значение s.
#
# ОЖИДАЕМЫЙ ИСХОД в штамп 0 не входит: он формулируется после разрешённых
# измерений и записывается в финальную версию с указанием, какие измерения его
# мотивировали (условие Д3).
""".format(date=date.today().isoformat(), version=VERSION, size=size,
           spec=spec_hash, cfg_size=cfg_path.stat().st_size, cfg=cfg_hash,
           frozen=frozen.name)

    (OUT / "stamp0_sha256.txt").write_text(text, encoding="utf-8")
    print("stamp 0 recorded:")
    print("  spec %s  %d bytes  %s" % (VERSION, size, spec_hash))
    print("  stamp config                 %s" % cfg_hash)
    print("  copy: %s" % frozen)
    print()
    print("from this point measurements V1c-E6.1, E6.2 and E6.4 are allowed;")
    print("measurement E6.3-C is forbidden until the stage is frozen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
