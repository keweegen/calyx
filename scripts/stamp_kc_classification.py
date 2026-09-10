# -*- coding: utf-8 -*-
"""Classification stamp: how to read the α′β′ literature BEFORE it is read.

Why. After stage V1c closed, a question remains open: is the silence of
Kenyon cells of subtype α′β′ a property of the functional regime, or an
artifact of the model family. The answer depends on literature that, at the
time this stamp is recorded, has not yet been read. A classification rule
recorded AFTER reading is the choice of whichever reading is convenient;
recorded BEFORE, it is pre-registration. The validation ladder is built on
this distinction, and it applies to the literature question the same way it
applies to a run.

What's here. Three branches with their consequences, the order of evidence
collection, and one explicit prohibition — all dictated by the owner of
methodological decisions on September 9, 2026, before the literature search
was started and before its result was obtained. The script writes the
artefact and its hash and refuses to overwrite a stamp already recorded.

Run:  python scripts/stamp_kc_classification.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "v1c"

STAMP = {
    "artefact_kind": "штамп классификации, записан до получения ответа литературы",
    "date": "2026-09-09",
    "stage_closed": "V1c, исход FAIL-CAL-MBON-FLOOR",
    "question": "молчание клеток Кеньона подтипа α′β′ в семействе [2] — "
                "биологически правдоподобный функциональный режим или артефакт "
                "семейства модели",

    "measured_before_this_stamp": {
        "source": "results/v1c/kc_subtype_balance.json, скрипт "
                  "v1c_kc_subtype_diag.py; симулятор не запускался",
        "uni_pn_mV_per_cell": {"KCab": 18.66, "KCa'b'": 6.52, "KCg": 18.99},
        "inhibition_over_excitation": {"KCab": 0.0420, "KCa'b'": 0.1399,
                                       "KCg": 0.0348},
        "ratio_excitation_worst_over_best": 2.91,
        "ratio_balance_worst_over_best": 4.02,
        "model_fact": "модель [2] задаёт один набор параметров нейрона на все "
                      "клетки: порог, постоянная времени мембраны и "
                      "сопротивление у всех одни и те же, поэтому дисбаланс не "
                      "компенсируется по построению модели",
    },

    "order_of_evidence": [
        "1. Факт in vivo: отвечают ли KCα′β′ на запахи вообще и насколько их "
        "рекрутирование отличается от KCαβ и KCγ.",
        "2. Клеточная физиология: есть ли данные о различиях внутренней "
        "возбудимости между подтипами KC — входное сопротивление, порог спайка, "
        "постоянная времени мембраны, потенциал покоя.",
        "3. Модели: используют ли опубликованные модели грибовидного тела "
        "одинаковую динамику всех KC или уже вводят зависящие от подтипа "
        "возбудимость либо масштаб входа.",
        "4. Только после этого — классифицировать исход V1c по развилкам ниже.",
    ],

    "branches": [
        {
            "id": "A",
            "finding": "α′β′ действительно почти не рекрутируются при подобных "
                       "запахах",
            "consequence": "V1c остаётся сильным отрицательным результатом "
                           "данного функционального режима; нового семейства "
                           "не заводится",
        },
        {
            "id": "B",
            "finding": "α′β′ нормально отвечают, и физиология компенсирует "
                       "слабый вход PN — компенсация показана данными",
            "consequence": "найден конкретный дефект модели [2]; появляется "
                           "обоснованное новое семейство",
        },
        {
            "id": "C",
            "finding": "α′β′ отвечают, но доказанной внутренней компенсации нет",
            "consequence": "диагноз модели остаётся вероятным, механизм "
                           "компенсации неизвестен; условные параметры вводить "
                           "нельзя",
        },
    ],

    "prohibition": {
        "rule": "Измеренные отношения 2,91 и 4,02 суть СТРУКТУРНЫЕ величины, а "
                "не оценка требуемой возбудимости. Они не превращаются в "
                "параметр модели, из них не выводится ручка, и ни одна ступень "
                "не строится на их значении.",
        "ground": "Это тот же класс ошибки, что и линейная экстраполяция "
                  "подпорогового пика в единицы `s`, отменённая при закрытии "
                  "V1c: отношение двух измеренных чисел не есть оценка третьей "
                  "величины, стоящей между ними по смыслу. Знак приближённого "
                  "равенства такого перехода не обосновывает.",
    },

    "if_branch_B_or_C": {
        "what_the_next_stage_is_not": "не «ещё одна попытка пройти пол отклика "
                                      "MBON» и не поиск коэффициента по сетке",
        "hypothesis_to_test": "Uniform KC intrinsic dynamics are insufficient "
                              "to reproduce subtype recruitment implied by the "
                              "connectome and observed physiology.",
        "what_the_stage_asks": "может ли НЕЗАВИСИМО ОБОСНОВАННАЯ зависящая от "
                               "подтипа динамика восстановить рекрутирование "
                               "α′β′ при сохранении разреженности остальных "
                               "подтипов и без разрушения воспроизводимости "
                               "V1a-S и V1b′",
        "independently_justified": "параметры берутся из опубликованных "
                                   "измерений физиологии подтипов, а не "
                                   "подбираются под исход; источник каждого "
                                   "числа называется в штампе 0 новой ступени",
    },

    "declared_by": "владелец методологических решений, 9 сентября 2026, до "
                   "запуска поиска по литературе и до получения его результата",
}


def main() -> int:
    p = OUT / "kc_classification_stamp.json"
    h = OUT / "kc_classification_stamp_sha256.txt"
    if p.exists():
        print("stamp already recorded: %s\noverwriting is forbidden: it declares "
              "the rule BEFORE the answer is obtained, and rewriting it afterward "
              "would mean picking the rule to fit the answer." % p, file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(STAMP, ensure_ascii=False, indent=2), encoding="utf-8")
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    h.write_text(
        "# Штамп классификации исхода V1c по литературе об α′β′.\n"
        "# Записан ДО запуска поиска и ДО получения его результата: правило\n"
        "# классификации, записанное после ответа, есть выбор удобного\n"
        "# прочтения, а не классификация.\n#\n"
        "# Дата записи: 2026-09-09\n"
        "kc_classification_stamp.json %d %s\n" % (p.stat().st_size, digest),
        encoding="utf-8")
    print("stamp recorded: %s\nhash: %s" % (p, digest[:16]))
    print("\nbranches:")
    for b in STAMP["branches"]:
        print("  %s. %s\n     -> %s" % (b["id"], b["finding"], b["consequence"]))
    print("\nprohibition: %s" % STAMP["prohibition"]["rule"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
