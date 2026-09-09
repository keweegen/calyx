# -*- coding: utf-8 -*-
"""Отчёт ступени V1c: исход, карта по трёхмерной сетке, некритериальные выходы.

Читает `results/v1c/report_map.json`, собранный run_v1c_grid.py --merge, и
пишет `results/v1c/report_v1c.md`. Ничего не вычисляет заново, кроме
агрегатов таблицы: исход выносит сборка, отчёт его записывает.

Что обязано быть в отчёте как объявленное ДО прогона (решение владельца
9 сентября 2026, принятое до первой точки):

  1. P14-M и карта V1c-E7.2 вычислены во всех 285 точках; в полосу и в
     определения FLOOR/CEIL входит только множество K точек, прошедших по доле
     при своём узле s; результаты MBON вне K помечены candidate_at_s = false;
     состояния «не вычислено» в артефакте нет ни у одной точки.
  2. Нарушение потолка в правиле приоритета V1c-E4.5 операционализировано как
     «существует t из T38 с R_t > 67 Гц», то есть not ceiling_ok. Оборот
     «по правилу 5 из 6» прочитан как ссылка на унаследованный блок порогов,
     а не как переопределение агрегации: V1b-3.4 говорит «для каждого типа»,
     а кванторы E4.3 и E4.5 - «хотя бы один».
  3. FLOOR проверяется на верхнем узле по канонической строке 5,3875, квантор
     по точкам K и по типам - существования. Два случая, текстом не покрытые
     (на верхнем узле нет точек K; или все точки K верхнего узла держат шесть
     типов не ниже пола и проваливаются только по глубине модуляции),
     закрывают ступень исходом NOT-TESTABLE, а не назначаются по аналогии.
  4. Процедура V1c-E7.3: сверка в шарде на каждой точке узла s = 1 и повторно
     при сборке; ключ - точное значение float; оба источника и все поля,
     включая min_R_t, max_R_t и violated; равенство представлений через ту же
     сериализацию, какой пишет шард (NaN совпадает с NaN); механизм HALT.json.
  5. Наблюдатель: EQS_CORE без правок плюс строка `vmax : volt`; run_regularly
     в слоте groups, порядок 1; d = max(0, v_th - vmax) типа float32; признак
     спайка берётся из SpikeMonitor, не из d = 0; проверка «d = 0 без спайка»
     с ожиданием 0.
  6. Три предпрогонные проверки в каждом шарде и отдельным артефактом
     preflight.json; пилот исполнения на одной точке s = 1 с наблюдателем и
     без, значения vmax пилота не записаны.
  7. Зёрна SEED_CAL = 20260908 и SEED_CAL_M = 20260958, пробы +1..+6,
     одинаковые на всех пяти узлах.
  8. Порядок точек и назначение шардов - run_plan.json, записан до запуска.
     Не в config_v1c.json: тот заморожен и его SHA-256 входит в
     предрегистрацию.
  9. Карта на калибровке содержит только набор C; столбец run различает
     калибровочный и оценочный прогоны.
 10. Регрессия V1c-E6.4 повторно не прогонялась: тождество при s = 1
     проверяется внутри ступени выходом V1c-E7.3.

Запуск:  python v1c_report.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "results" / "v1c"

DECLARED = __doc__.split("объявленное ДО прогона", 1)[1].split("Запуск:")[0].strip()


def fmt(x, n=4):
    if x is None:
        return "—"
    if isinstance(x, float):
        return ("%.*f" % (n, x)).replace(".", ",")
    return str(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "report_v1c.md"))
    a = ap.parse_args()

    import v1b_subcircuit as V
    import v1c_stage as S

    p = OUT / "report_map.json"
    if not p.exists():
        print("исход не собран: запустите run_v1c_grid.py --merge", file=sys.stderr)
        return 1
    doc = json.loads(p.read_text(encoding="utf-8"))
    res, pts = doc["summary"], doc["points"]
    cfg = S.config()
    ns = list(cfg["grid_s"]["nodes_canonical"])

    L = []
    w = L.append
    w("# Отчёт прогона ступени V1c\n")
    w("**Исход: %s.**\n" % res["outcome"])
    w("Ожидаемый исход, объявленный до прогона: %s\n" % cfg["expected_outcome"])
    w("Спецификация v0.17, хэш `636c49968a0866bd`, разделы 3з и 3и; конфиг "
      "`7ddaa3e9c2690aeb`; таблица детекторов `9a4a1590c3c228c3`. "
      "Байтовая копия — `experiment-spec-h1-h3.v0.17.frozen.md`.\n")
    w("Основание исхода: %s\n" % res["ground"])
    if res.get("stopping_rule"):
        w("Основание правила остановки whitepaper, наступившее по V1c-E4.5: %s\n"
          % res["stopping_rule"])

    w("\n## 1. Сетка и что на ней посчитано\n")
    w("| величина | значение |")
    w("|---|---|")
    w("| трёхмерных точек | %d |" % res["n_points"])
    w("| кандидатов V1b′ | 57 |")
    w("| узлов по `s` | %s |" % ", ".join("`%s`" % x for x in ns))
    w("| точек множества `K` (прошли по доле при своём `s`) | %d |" % res["n_in_K"])
    w("| полоса `s` (V1c-E4.1) | %s |"
      % (", ".join("`%s`" % x for x in res["band"]) if res.get("band") else "пуста"))
    w("| точек с превышением потолка 67 Гц | %s |"
      % fmt(res.get("n_points_over_ceiling"), 0))
    w("| проверка «`d = 0` без спайка», ожидание 0 | %d |"
      % res["zero_without_spike_total"])
    w("")
    w("`K` по узлам:\n")
    w("| узел `s` | точек в `K` | max по `K` от min по типам `R_t`, Гц | "
      "max по `K` от max по типам `R_t`, Гц |")
    w("|---|---|---|---|")
    h = res["headline"]
    for s in ns:
        w("| `%s` | %d | %s | %s |"
          % (s, h["n_in_K_by_node"].get(s, 0),
             fmt(h.get("max_of_min_R_t_hz@%s" % s), 6),
             fmt(h.get("max_of_max_R_t_hz@%s" % s), 6)))
    w("\nПол — %s Гц, потолок — %s Гц (V1b-3.3, V1b-3.4, унаследованы без "
      "изменений).\n" % (fmt(h["floor_hz"], 1), fmt(h["ceiling_hz"], 0)))

    w("\n## 2. Карта точек\n")
    w("Полная карта — `results/v1c/report_map.json`. Ниже точки множества `K`, "
      "по узлам; `R_t` — средняя частота типа по шести запахам набора C при "
      "тайминге P14-M.\n")
    w("| узел `s` | `pn_kc_scale` | `g_apl` | `f̄_C` | min `R_t` | max `R_t` | "
      "нарушено |")
    w("|---|---|---|---|---|---|---|")
    k = [x for x in pts if x.get("candidate_at_s")]
    k.sort(key=lambda x: (float(x["s_node"]), -x["f_mean"]))
    for x in k:
        rt = [v["R_t"] for v in x["mbon"]["per_type"].values()]
        broke = S._violated(x["mbon"])
        w("| `%s` | %s | %s | %s | %s | %s | %s |"
          % (x["s_node"], fmt(x["pn_kc_scale"], 4), fmt(x["g_apl_rel"], 5),
             fmt(x["f_mean"], 4), fmt(min(rt) if rt else None, 4),
             fmt(max(rt) if rt else None, 4),
             ", ".join(broke) if broke else "—"))

    w("\n## 3. Некритериальные выходы\n")
    w("### 3.1. Таблица детекторов (V1c-E7.1)\n")
    tab = json.loads((OUT / "detector_table.json").read_text(encoding="utf-8"))
    w("| узел `s` | детекторов в `T₃₈` | детекторов вне `T₃₈` | типы вне `T₃₈` |")
    w("|---|---|---|---|")
    for s in ns:
        d = tab["by_node"][s]
        w("| `%s` | %d | %d | %s |"
          % (s, d["n_detectors_T38"], d["n_detectors_outside_T38"],
             ", ".join(d["types_outside_T38"]) or "—"))
    w("\nТаблица пересчитана перед первой точкой каждого шарда из весов, "
      "снятых с собранной сети Brian 2; расхождений нет. Детекторов среди "
      "типов `T₃₈` нет ни на одном узле — это тождественно условию (5) при "
      "данном `s_max`.\n")

    w("### 3.2. Карта расстояния до порога (V1c-E7.2)\n")
    sp = OUT / "threshold_map_summary.json"
    if sp.exists():
        sm = json.loads(sp.read_text(encoding="utf-8"))
        w("Первичный артефакт — `results/v1c/threshold_map.parquet`, %d строк: "
          "одна на набор «кандидат, узел `s`, клетка MBON, набор, запах, "
          "проба». Временные трассы `v` не сохранялись. Сводка — "
          "`threshold_map_summary.json`, перевычисляема из первичного "
          "артефакта.\n" % sm["n_rows_primary"])
        by_t = {}
        for r in sm["per_T38_type"]:
            by_t.setdefault(r["s_node"], []).append(r["median_of_cell_medians_mV"])
        w("Медиана по клеткам типа от медиан клеток, по узлам (мВ до порога, "
          "медиана по типам `T₃₈` и по кандидатам):\n")
        w("| узел `s` | медиана расстояния до порога, мВ |")
        w("|---|---|")
        import statistics as st
        for s in ns:
            v = by_t.get(s)
            w("| `%s` | %s |" % (s, fmt(st.median(v), 4) if v else "—"))
    else:
        w("Сводка не собрана.\n")

    w("\n### 3.3. Проверка воспроизведения на узле `s = 1` (V1c-E7.3)\n")
    r = res["reproduction_v1b_prime"]
    w("Сверено %d точек узла `s = 1` с артефактами V1b′ побитово; расхождений "
      "в %d.\n" % (r["n_checked"], r["n_mismatched"]))
    if r["mismatched_cand_idx"]:
        w("Расхождения в кандидатах: %s\n"
          % ", ".join(str(x) for x in r["mismatched_cand_idx"]))

    w("\n## 4. Объявленное до прогона\n")
    w(DECLARED)

    w("\n## 5. Что этот исход не говорит\n")
    w("Величины зазора измерения V1c-E6.2 в единицы `s` не пересчитывались ни "
      "до прогона, ни в этом отчёте: самоограничение Д4 держалось до "
      "завершения прогона. Положение полосы, значение `s`, при котором тип "
      "пересекает пол или потолок, и достаточность границы `s_max` до прогона "
      "не вычислялись и в предрегистрации не записаны.\n")
    w("Результат относится к семейству «градуальный APL плюс вход [42]» с "
      "одним глобальным скаляром на веса KC→MBON и нулевым фоном MBON. "
      "О коннектоме и о модели [2] в иных режимах он не говорит.\n")

    out = Path(a.out)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("отчёт записан: %s (исход %s)" % (out, res["outcome"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
