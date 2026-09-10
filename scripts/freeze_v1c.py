# -*- coding: utf-8 -*-
"""Freeze the pre-registration of stage V1c: hashes, config and byte copy.

This is the stage freeze, not stamp 0. Stamp 0 was recorded earlier by
scripts/stamp0_v1c.py and sits alongside it: it proves that the stage's design
predates the pre-hash measurements. What's frozen here is the version in which
the numbers have been substituted from stamp 0's formulas, the "Provenance" and
"Expected outcome" sections have been filled in, and both open fields of
stamp 0 are closed.

What gets hashed: the spec in full; the stage config with the five canonical
grid nodes; the detector table that the spec references by number.
All three files are protected from line-ending normalization in .gitattributes.

Freeze rule (spec section 1): once the hash is recorded, the stage's criteria
and tolerances are not edited, and runs against the old version are not
reinterpreted. Re-recording is forbidden.

Run:  python scripts/freeze_v1c.py
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
VERSION = "v0.17"

# Canonical grid nodes: decimal strings from the spec, V1c-E3.2.
# As strings, not numbers: the config reads them as strings, the nodes are not
# recomputed at run time, and the order of arithmetic at the boundaries does not
# affect the strictness of condition (5)'s inequality.
NODES = ["1", "1.6836", "1.7783", "3.1623", "5.3875"]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    if not SPEC.exists():
        print("spec not found: %s" % SPEC, file=sys.stderr)
        return 1
    stamp0 = OUT / "stamp0_sha256.txt"
    if not stamp0.exists():
        print("stamp 0 not found: %s" % stamp0, file=sys.stderr)
        print("freezing without stamp 0 violates rule D1", file=sys.stderr)
        return 1

    det_path = OUT / "detector_table.json"
    weights_path = OUT / "weights_kc_mbon.json"
    for p in (det_path, weights_path):
        if not p.exists():
            print("missing pre-hash measurement artefact: %s" % p, file=sys.stderr)
            return 1

    det = json.loads(det_path.read_text(encoding="utf-8"))
    if det["nodes"] != NODES:
        print("detector table nodes do not match the canonical ones: %s"
              % det["nodes"], file=sys.stderr)
        return 1
    if not det["condition5_holds_on_all_nodes"]:
        print("condition (5) is violated at some node - freezing forbidden",
              file=sys.stderr)
        return 1

    frozen = OUT / ("experiment-spec-h1-h3.%s.frozen.md" % VERSION)
    if frozen.exists():
        print("pre-registration already frozen: %s" % frozen.name,
              file=sys.stderr)
        print("re-freezing is forbidden by the rule in section 1",
              file=sys.stderr)
        return 1

    w = json.loads(weights_path.read_text(encoding="utf-8"))

    cfg = {
        "stage": "V1c",
        "artefact_kind": "предрегистрация ступени",
        "spec_version": VERSION,
        "spec_sections": ["3з", "3и"],
        "derived_from_stamp0": {
            "spec_version": "v0.16",
            "spec_hash_prefix": "ed2952902a681c7b",
            "spec_size_bytes": 246115,
            "copy": "experiment-spec-h1-h3.v0.16.stamp0.md",
            "difference_limited_by": "условие Д3; перечислено шестью позициями "
                                     "раздела «Происхождение»",
        },
        "knob": {"name": "kc_mbon_scale",
                 "acts_on": "вес каждого ребра KC->MBON подсхемы",
                 "identity_at": "s = 1 тождественно конфигурации V1b'"},
        "grid_s": {
            "nodes_canonical": NODES,
            "n_nodes": len(NODES),
            "s_min": "1",
            "s_pop_reportonly": "1.6836",
            "s_max": "5.3875",
            "n_per_decade": 4,
            "rounding": "узлы формулы 10^(k/4) - по ближайшему, 4 знака; "
                        "граничные узлы s_max и s_pop - вниз",
            "A": w["epsp"]["A"],
            "theta_mV": w["model_constants"]["theta_mV"],
            "w_max_mV_T38": w["s_max"]["w_max_mV_T38"],
            "w_max_mV_all_mbon": w["s_max"]["w_max_mV_all_mbon"],
        },
        "grid_other": {"points": "57 кандидатов V1b'",
                       "n_3d_points": 57 * len(NODES),
                       "fraction_recomputed_at_each_3d_point": True},
        "outcomes": ["PASS-FULL", "PASS-T38", "FAIL-EVAL", "FAIL-CAL-KC",
                     "FAIL-CAL-MBON-FLOOR", "FAIL-CAL-MBON-CEIL",
                     "NOT-TESTABLE"],
        "outcome_priority": "при одновременном выполнении определений "
                            "FAIL-CAL-MBON-CEIL и FAIL-CAL-MBON-FLOOR "
                            "записывается CEIL",
        "expected_outcome": "FAIL-CAL-MBON-CEIL (категорией; положение полосы "
                            "не вычисляется и не записывается)",
        "non_criterial_outputs": {
            "detector_table": "results/v1c/detector_table.json",
            "threshold_map": "results/v1c/threshold_map.parquet и "
                             "threshold_map_summary.json",
            "reproduction_check_at_s1": "побитовая сверка с артефактами V1b' "
                                        "по величинам, которые V1b' заморозил",
        },
        "pre_hash_measurements_done": {
            "E6.1": "results/v1c/weights_kc_mbon.json",
            "E6.2": "results/diag_window/window_gap.json",
            "E6.4": "results/v1c/regression_trains.json",
        },
        "unchanged_from_V1b_prime": [
            "пороги 2 Гц, 67 Гц, 0,19 и правило 5 из 6",
            "полоса доли отвечающих KC и её потолок",
            "панель ступени, разведение наборов C и E, слепота E",
            "таблица зёрен", "протокол P14-M", "источник PN-паттернов",
            "определение ответа KC",
        ],
    }
    cfg_path = OUT / "config_v1c.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    shutil.copy2(SPEC, frozen)
    spec_hash, size = sha256(SPEC), SPEC.stat().st_size
    cfg_hash = sha256(cfg_path)
    det_hash = sha256(det_path)

    text = """# Предрегистрация ступени V1c: хэши, зафиксированные ДО прогона.
# Правило раздела 1 спецификации: после записи хэша критерии и допуски ступени
# не редактируются, прогоны по старой версии не переинтерпретируются.
#
# Дата записи: {date}
# Источник: whitepaper/experiment-spec-h1-h3.md, версия {version},
# разделы 3з (предрегистрация) и 3и (некритериальные выходы).
experiment-spec-h1-h3.md {size} {spec}
config_v1c.json {cfg_size} {cfg}
detector_table.json {det_size} {det}
#
# Байтовая копия спецификации лежит рядом: {frozen}.
# Она нужна потому, что рабочий каталог whitepaper/ под git не заведён и
# исходник .md существует в одном экземпляре.
#
# ОТКУДА ПОЛУЧЕНА. Из штампа 0 ступени V1c: версия v0.16, 246 115 байт,
# хэш ed2952902a681c7b, копия experiment-spec-h1-h3.v0.16.stamp0.md,
# записанная ДО всех до-хэшевых измерений перечня V1c-E6. Разность между
# штампом 0 и этой версией ограничена условием Д3 и перечислена шестью
# позициями раздела «Происхождение».
#
# СЕТКА ПО s: {nodes} - пять узлов, канонические десятичные строки.
# Граничные узлы округлены вниз, поэтому на них неравенство условия (5)
# и неравенство таблицы детекторов остаются строгими.
# Трёхмерных точек: 57 кандидатов V1b' x 5 узлов = 285.
#
# ОЖИДАЕМЫЙ ИСХОД: FAIL-CAL-MBON-CEIL, категорией. Положение полосы, значение s
# пересечения пола или потолка и достаточность границы s_max не вычисляются и
# не записываются; величины журнала E6.2 в единицы s не пересчитываются.
#
# ЗАПРЕТ СНЯТ. С момента этой заморозки измерение подпорогового потенциала MBON
# (пункт C внешнего ревью) разрешено и обязательно как некритериальный выход
# V1c-E7.2: оно уже не может повлиять на конструкцию ступени.
""".format(date=date.today().isoformat(), version=VERSION, size=size,
           spec=spec_hash, cfg_size=cfg_path.stat().st_size, cfg=cfg_hash,
           det_size=det_path.stat().st_size, det=det_hash,
           frozen=frozen.name, nodes="; ".join(NODES))

    (OUT / "spec_sha256.txt").write_text(text, encoding="utf-8")

    print("V1c pre-registration frozen:")
    print("  spec %s  %d bytes  %s" % (VERSION, size, spec_hash))
    print("  stage config                   %s" % cfg_hash)
    print("  detector table                 %s" % det_hash)
    print("  copy: %s" % frozen)
    print()
    print("grid nodes: %s" % ", ".join(NODES))
    print("3D points: %d" % (57 * len(NODES)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
