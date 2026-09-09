# -*- coding: utf-8 -*-
"""Тест соответствия кода замороженной спецификации ступени V1b.

Зачем. Калибровка V1b была прогнана кодом, который реализовал ограничение
допустимости V1b-4.5 не целиком: проверялась только доля отвечающих клеток
Кеньона, а требование «а также V1b-3.3, V1b-3.4 и V1b-3.5 по запахам C» не
вычислялось ни в одной точке сетки. Ошибку не поймал никто, потому что
соответствие кода спецификации ничем не проверялось: заявленная методология
(правило как модуль, критерий провала до запуска, хэш конфига) молчаливо
предполагала, что код делает то, что записано, и этого предположения никто не
тестировал.

Этот файл закрывает дыру. На каждое численное или процедурное требование
раздела 3е спецификации - свой тест, в докстроке которого записана та фраза
спецификации, которую тест проверяет. Тест обязан ловить ошибку V1b-4.5: пока
код не исправлен, соответствующая проверка ДОЛЖНА проваливаться. Набор,
который проходит целиком на дефектном коде, бесполезен.

Прогонов симулятора здесь нет: всё проверяется на синтетических данных и на
константах, поэтому набор идёт секунды и может стоять перед каждым прогоном
ступени.

Запуск:  python test_spec_conformance.py
         python test_spec_conformance.py --verbose
Код возврата 0 - все проверки прошли, 1 - есть провалившиеся.
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CHECKS = []


def check(code: str, requirement: str):
    """Зарегистрировать проверку. code - номер требования спецификации."""
    def deco(fn):
        CHECKS.append((code, requirement, fn))
        return fn
    return deco


class Fail(AssertionError):
    pass


# ограничение MBON на C, вычисленное и пройденное: точка допустима только
# вместе с ним (V1b-4.5)
MBON_OK = {"floor_ok": True, "ceiling_ok": True, "md_ok": True, "pass": True}


def eq(got, want, what: str):
    if isinstance(want, float) or isinstance(got, float):
        ok = abs(float(got) - float(want)) <= 1e-9 * max(1.0, abs(float(want)))
    else:
        ok = got == want
    if not ok:
        raise Fail("%s: получено %r, спецификация требует %r" % (what, got, want))


def true(cond, what: str):
    if not cond:
        raise Fail(what)


# --- синтетика вместо прогонов -----------------------------------------------
def fake_result(V, neurons, counts_by_role: dict, n_trials: int = 6) -> dict:
    """Результат прогона с заданным числом спайков по ролям."""
    apl = set(neurons.loc[neurons.mb_role == "APL", "root_id"])
    core = sorted(set(neurons.root_id) - apl)
    role = dict(zip(neurons.root_id, neurons.mb_role))
    c = np.zeros((len(core), n_trials), dtype=np.int32)
    for k, i in enumerate(core):
        c[k, :] = counts_by_role.get(role[i], 0)
    return {"core_ids": core, "counts": c, "counts_by_window": [c, c],
            "windows": [(0.0, 4.0), (0.0, 2.0)],
            "seeds": list(range(n_trials))}


# --- отклик клеток Кеньона ---------------------------------------------------
@check("V1b-2.1", "KC отвечает на пробу, если число её спайков в окне >= 1; "
                  "единица фиксирована и подбору не подлежит")
def t_response_threshold(V, neurons, con):
    eq(V.KC_SPIKE_THRESHOLD, 1, "порог спайков на пробу")


@check("V1b-2.2", "Запах предъявляется 6 раз; KC отвечает на запах, если "
                  "ответила не менее чем на 3 пробах из 6")
def t_trials(V, neurons, con):
    eq(V.N_TRIALS, 6, "число предъявлений")
    eq(V.MIN_TRIALS, 3, "минимум проб для ответа на запах")
    # ровно 3 из 6 - ответ; ровно 2 из 6 - нет
    apl = set(neurons.loc[neurons.mb_role == "APL", "root_id"])
    core = sorted(set(neurons.root_id) - apl)
    role = dict(zip(neurons.root_id, neurons.mb_role))
    kc_idx = [k for k, i in enumerate(core) if role[i] == "Kenyon_Cell"]
    for n_hit, want_resp in ((3, True), (2, False)):
        c = np.zeros((len(core), 6), dtype=np.int32)
        c[kc_idx[0], :n_hit] = 1
        res = {"core_ids": core, "counts": c, "counts_by_window": [c],
               "windows": [(0.0, 4.0)], "seeds": list(range(6))}
        f = V.kc_fraction(res, neurons)["f"]
        eq(f > 0, want_resp, "ответ при %d пробах из 6" % n_hit)


@check("V1b-2.3", "f(o) = N_resp(o)/N_KC, где N_KC = 5 177 - все нейроны типа "
                  "Kenyon_Cell подсхемы, оба полушария, без отбора")
def t_denominator(V, neurons, con):
    n_kc = int((neurons.mb_role == "Kenyon_Cell").sum())
    eq(n_kc, 5177, "число KC в подсхеме")
    res = fake_result(V, neurons, {"Kenyon_Cell": 1})
    kf = V.kc_fraction(res, neurons)
    eq(kf["n_kc_denominator"], 5177, "знаменатель доли отвечающих")
    eq(kf["f"], 1.0, "доля при ответе всех KC")
    # отчётные знаменатели
    cov = V.kc_coverage(neurons, con)
    rep = V.kc_fraction_report(res, neurons, cov)
    eq(rep["n_denominator_all"], 5177, "знаменатель, все KC")
    eq(rep["n_denominator_with_upn"], 4820, "знаменатель, KC с входом от uPN")
    eq(rep["n_right"], 2597, "KC правого полушария")
    eq(rep["n_left"], 2580, "KC левого полушария")


@check("V1b-2.4", "Для порога k из {1, 2, 3, 5} спайков на пробу и правил "
                  "агрегации {>=1 из 6; >=3 из 6; 6 из 6}")
def t_sensitivity(V, neurons, con):
    res = fake_result(V, neurons, {"Kenyon_Cell": 2})
    keys = set(V.kc_fraction(res, neurons)["sensitivity"])
    want = {"k=%d,%s" % (k, lab) for k in (1, 2, 3, 5)
            for lab in (">=1/6", ">=3/6", "6/6")}
    eq(keys, want, "набор ячеек таблицы чувствительности")
    tab = V.sensitivity_table({"o1": res, "o2": res}, neurons)
    eq(set(tab), want, "набор ячеек таблицы по панели")
    true(all({"mean", "max"} == set(v) for v in tab.values()),
         "в каждой ячейке печатаются среднее и максимум по панели")


@check("V1b-2.5", "S_P(o) = (1 - (sum r_i/N)^2 / (sum r_i^2/N)) / (1 - 1/N), "
                  "r_i - среднее по 6 пробам число спайков KC i в окне")
def t_sparseness(V, neurons, con):
    n_kc = int((neurons.mb_role == "Kenyon_Cell").sum())
    # при равных r_i отношение (sum r/N)^2 / (sum r^2/N) равно 1, значит S_P = 0:
    # ноль - плотный код, единица - предельно разреженный
    res = fake_result(V, neurons, {"Kenyon_Cell": 3})
    eq(V.sparseness_treves_rolls(res, neurons), 0.0, "S_P при равных r_i")
    # одна клетка несёт всё: предельно разреженный код, S_P = 1
    apl = set(neurons.loc[neurons.mb_role == "APL", "root_id"])
    core = sorted(set(neurons.root_id) - apl)
    role = dict(zip(neurons.root_id, neurons.mb_role))
    kc_idx = [k for k, i in enumerate(core) if role[i] == "Kenyon_Cell"]
    c = np.zeros((len(core), 6), dtype=np.int32)
    c[kc_idx[0], :] = 6
    res1 = {"core_ids": core, "counts": c, "counts_by_window": [c],
            "windows": [(0.0, 4.0)], "seeds": list(range(6))}
    r = np.zeros(n_kc); r[0] = 6.0
    want = (1.0 - (r.sum() / n_kc) ** 2 / ((r ** 2).sum() / n_kc)) / (1.0 - 1.0 / n_kc)
    eq(V.sparseness_treves_rolls(res1, neurons), want, "S_P при одной активной KC")


# --- отклик MBON -------------------------------------------------------------
@check("V1b-3.1", "T38 - шесть типов MBON, записанных в [38]: MBON11, MBON12, "
                  "MBON13, MBON14, MBON17, MBON18; тайминг M - предъявление 5 с, "
                  "средняя частота в окне предъявления")
def t_t38(V, neurons, con):
    eq(list(V.T38), ["MBON11", "MBON12", "MBON13", "MBON14", "MBON17", "MBON18"],
       "множество T38")
    eq(V.M_PULSE_MS, 5000, "длительность предъявления при тайминге M")
    eq(V.M_WINDOW_MS, 5000, "окно измерения при тайминге M равно окну предъявления")
    typ = set(neurons.loc[neurons.mb_role == "MBON", "hemibrain_type"]
              .astype("string").dropna())
    missing = [t for t in V.T38 if t not in typ]
    true(not missing, "типы T38 отсутствуют в подсхеме: %s" % missing)
    # частота типа - среднее по клеткам типа и по пробам, делённое на окно
    res = fake_result(V, neurons, {"MBON": 10})
    rates = V.mbon_type_rates({"o": res}, neurons, V.M_WINDOW_MS)
    eq(float(rates.loc["MBON11", "o"]), 10.0 / (V.M_WINDOW_MS / 1000.0),
       "частота типа MBON11 при 10 спайках в окне 5 с")


@check("V1b-3.1", "Для группы избегания MBON01-MBON06 эталона нет, пороги к её "
                  "типам не применяются; для типов вне T38 порогов нет")
def t_avoid_no_thresholds(V, neurons, con):
    eq(list(V.AVOID), ["MBON0%d" % k for k in range(1, 7)], "группа избегания")
    res = fake_result(V, neurons, {"MBON": 0})
    rep = V.mbon_report({"o": res}, neurons, V.M_WINDOW_MS)
    true(set(V.AVOID) & set(rep["avoid_group_report_only"]),
         "группа избегания печатается отчётно")
    true(not (set(V.AVOID) & set(rep["T38"]["per_type"])),
         "типы группы избегания не должны попадать в пороговую часть")
    true(rep["other_types_report_only"],
         "типы вне T38 и вне группы избегания печатаются отчётно")


@check("V1b-3.3", "R_t = mean_o r_t(o); требуется R_t >= 2 Гц для всех шести типов")
def t_floor(V, neurons, con):
    eq(V.MBON_FLOOR_HZ, 2.0, "порог пола")
    lo = pd.DataFrame(np.full((6, 14), 1.99), index=V.T38,
                      columns=V.PANEL_ALL)
    eq(V.check_mbon(lo)["floor_ok"], False, "пол при 1,99 Гц у всех типов")
    hi = lo + 0.02
    eq(V.check_mbon(hi)["floor_ok"], True, "пол при 2,01 Гц у всех типов")
    one_low = hi.copy(); one_low.iloc[0, :] = 1.0
    eq(V.check_mbon(one_low)["floor_ok"], False,
       "пол требуется для ВСЕХ шести типов, не для большинства")


@check("V1b-3.4", "Для каждого типа t из T38: R_t <= 67 Гц")
def t_ceiling(V, neurons, con):
    eq(V.MBON_CEIL_HZ, 67.0, "порог потолка")
    r = pd.DataFrame(np.full((6, 14), 66.9), index=V.T38, columns=V.PANEL_ALL)
    eq(V.check_mbon(r)["ceiling_ok"], True, "потолок при 66,9 Гц")
    r2 = r + 0.2
    eq(V.check_mbon(r2)["ceiling_ok"], False, "потолок при 67,1 Гц")


@check("V1b-3.5", "MD_t = (max-min)/(max+min) по 14 запахам; MD_t >= 0,19 не "
                  "менее чем у 5 типов из 6; тип с R_t < 2 Гц не участвует")
def t_md(V, neurons, con):
    eq(V.MBON_MD_MIN, 0.19, "порог глубины модуляции")
    eq(V.MBON_MD_TYPES, 5, "сколько типов должны его пройти")
    base = np.full((6, 14), 10.0)
    base[:, 0] = 10.0 * (1 + 0.25) / (1 - 0.25)   # MD = 0,25, выше порога
    r = pd.DataFrame(base, index=V.T38, columns=V.PANEL_ALL)
    res = V.check_mbon(r)
    eq(res["n_md_ok"], 6, "MD выше порога засчитывается у всех шести типов")
    # правило сравнения нестрогое: считаются типы с MD >= 0,19, а не > 0,19.
    # Сравнивать с сконструированным 0,19 нельзя - величина не представима в
    # двоичной плавающей точке; сверяем агрегат с теми MD, что вернул сам код.
    near = np.full((6, 14), 10.0)
    for k in range(6):
        near[k, 0] = 10.0 * (1 + 0.185 + 0.002 * k) / (1 - 0.185 - 0.002 * k)
    rn = pd.DataFrame(near, index=V.T38, columns=V.PANEL_ALL)
    got = V.check_mbon(rn)
    want = sum(1 for d in got["per_type"].values() if d["MD"] >= V.MBON_MD_MIN)
    eq(got["n_md_ok"], want, "агрегат считает типы с MD >= порога (нестрогое)")
    # два типа без модуляции: 4 из 6 - критерий не выполнен
    flat = r.copy(); flat.iloc[:2, 0] = 10.0
    eq(V.check_mbon(flat)["md_ok"], False, "MD у 4 типов из 6")
    # один тип без модуляции: 5 из 6 - выполнен
    flat1 = r.copy(); flat1.iloc[:1, 0] = 10.0
    eq(V.check_mbon(flat1)["md_ok"], True, "MD у 5 типов из 6")
    # тип, провалившийся по полу, из подсчёта MD исключается
    low = r.copy(); low.iloc[0, :] = 1.0
    eq(V.check_mbon(low)["n_md_ok"], 5, "тип с R_t < 2 Гц исключён из подсчёта MD")


@check("V1b-3.5", "Медиана MD_t по всем 2 002 пятизапаховым подмножествам панели")
def t_md_subsets(V, neurons, con):
    eq(len(V.PANEL_ALL), 14, "размер панели ступени")
    eq(len(list(combinations(range(14), 5))), 2002, "число пятизапаховых подмножеств")
    r = pd.DataFrame(np.random.default_rng(0).uniform(1, 20, (6, 14)),
                     index=V.T38, columns=V.PANEL_ALL)
    md = V.md_subset_medians(r, V.T38)
    eq(md["n_subsets"], 2002, "число подмножеств в отчётной величине")
    eq(md["k"], 5, "размер подмножества")
    eq(set(md["per_type"]), set(V.T38), "медиана считается по каждому типу T38")


# --- калибровка --------------------------------------------------------------
@check("V1b-4.1", "C - 2-heptanone, isopentyl acetate, hexanal, "
                  "6-methyl-5-hepten-2-one, diethyl succinate, methyl octanoate; "
                  "E - pentyl acetate, butyl acetate, ethyl lactate, 1-octen-3-ol, "
                  "pentanal, benzaldehyde, alpha-humulene, ethyl octanoate")
def t_panel_split(V, neurons, con):
    eq(list(V.PANEL_CAL), ["2-heptanone", "isopentyl acetate", "hexanal",
                           "6-methyl-5-hepten-2-one", "diethyl succinate",
                           "methyl octanoate"], "калибровочный набор C")
    eq(list(V.PANEL_EVAL), ["pentyl acetate", "butyl acetate", "ethyl lactate",
                            "1-octen-3-ol", "pentanal", "benzaldehyde",
                            "alpha-humulene", "ethyl octanoate"],
       "оценочный набор E")
    eq(len(V.PANEL_ALL), 14, "панель ступени")
    true(not (set(V.PANEL_CAL) & set(V.PANEL_EVAL)), "C и E не пересекаются")
    panel = pd.read_csv(V.PANEL, sep="\t", index_col=0)
    missing = [o for o in V.PANEL_ALL if o not in panel.index]
    true(not missing, "запахи панели отсутствуют в источнике входа: %s" % missing)
    # три эталонных вещества критерия перекрытия входят в E
    for a, b in V.REF_PAIRS:
        true(a in V.PANEL_EVAL and b in V.PANEL_EVAL,
             "эталонная пара (%s, %s) должна лежать внутри E" % (a, b))


@check("V1b-4.1", "Зёрна предъявлений для E не пересекаются с зёрнами калибровки")
def t_seeds_disjoint(V, neurons, con):
    span = lambda b: {b + i for i in range(1, V.N_TRIALS + 1)}
    bases = {"SEED_CAL": V.SEED_CAL, "SEED_CAL_M": V.SEED_CAL_M,
             "SEED_EVAL": V.SEED_EVAL, "SEED_EVAL_M": V.SEED_EVAL_M,
             "SEED_EMPTY": V.SEED_EMPTY, "SEED_PERM": V.SEED_PERM}
    names = list(bases)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            true(not (span(bases[a]) & span(bases[b])),
                 "отрезки зёрен %s и %s пересекаются" % (a, b))


@check("V1b-4.3", "g_ref = 1/(v_th - v_0) = 1/7 мВ^-1")
def t_gref(V, neurons, con):
    eq(V.g_ref_value(), 1.0 / 7.0, "g_ref")


@check("V1b-4.4", "Стадия 1: pn_kc_scale из {2^k, k = -6..2} (9 значений), "
                  "g_apl из {0} + {10^(k/2), k = -4..4} (10 значений) - 90 точек")
def t_grid1(V, neurons, con):
    g = V.grid_stage1()
    eq(len(g), 90, "число точек стадии 1")
    scales = sorted({s for s, _ in g})
    gains = sorted({x for _, x in g})
    eq(scales, [2.0 ** k for k in range(-6, 3)], "масштабы стадии 1")
    eq(len(gains), 10, "число значений гейна")
    eq(gains[0], 0.0, "ноль торможения входит в сетку стадии 1")
    eq(gains[1:], [10.0 ** (k / 2.0) for k in range(-4, 5)], "гейны стадии 1")


@check("V1b-4.4", "Стадия 2: ограничивающий прямоугольник допустимых точек "
                  "стадии 1, расширенный на один шаг стадии 1 в каждую сторону; "
                  "шаги 2^(1/4) по масштабу и 10^(1/8) по g_apl")
def t_grid2(V, neurons, con):
    s1 = [{"pn_kc_scale": 2.0, "g_apl_rel": 10.0 ** -0.5,
           "f_mean": 0.05, "f_max": 0.06},
          {"pn_kc_scale": 4.0, "g_apl_rel": 1.0, "f_mean": 0.05, "f_max": 0.06}]
    g = V.grid_stage2(s1)
    scales = sorted({s for s, _ in g})
    gains = sorted({x for _, x in g})
    eq(scales[0], 1.0, "нижняя граница масштаба: min/2 по шагу стадии 1")
    eq(scales[-1], 8.0, "верхняя граница масштаба: max*2 по шагу стадии 1")
    for a, b in zip(scales, scales[1:]):
        eq(b / a, 2.0 ** 0.25, "шаг сетки по масштабу")
    eq(gains[0], 10.0 ** -1.0, "нижняя граница гейна: min/10^(1/2)")
    eq(gains[-1], 10.0 ** 0.5, "верхняя граница гейна: max*10^(1/2)")
    for a, b in zip(gains, gains[1:]):
        eq(b / a, 10.0 ** 0.125, "шаг сетки по гейну")


@check("V1b-4.5", "Точка допустима, если на C выполнены: среднее f(o) в "
                  "[0,03; 0,10] и max f(o) <= 0,10, А ТАКЖЕ V1b-3.3, V1b-3.4 "
                  "и V1b-3.5 по запахам C")
def t_admissible(V, neurons, con):
    eq(tuple(V.F_BAND), (0.03, 0.10), "полоса средней доли отвечающих")
    eq(V.F_MAX, 0.10, "потолок максимальной доли")
    ok = {"f_mean": 0.05, "f_max": 0.06, "mbon": MBON_OK}
    eq(V.admissible(ok), True, "точка внутри полосы и с пройденным MBON")
    eq(V.admissible({"f_mean": 0.02, "f_max": 0.03, "mbon": MBON_OK}), False,
       "среднее ниже полосы")
    eq(V.admissible({"f_mean": 0.05, "f_max": 0.11, "mbon": MBON_OK}), False,
       "максимум выше потолка")
    # ограничение MBON: точка, проходящая по доле, но проваленная по любому из
    # трёх подограничений, допустимой быть НЕ должна
    for key, label in (("floor_ok", "V1b-3.3"), ("ceiling_ok", "V1b-3.4"),
                       ("md_ok", "V1b-3.5")):
        bad = dict(ok, mbon=dict(MBON_OK, **{key: False}))
        true(V.admissible(bad) is False,
             "точка, прошедшая по доле, но провалившая %s на C, признана "
             "допустимой" % label)
    # и главное: «не вычислено» не равно «прошла». Именно смешение этих двух
    # состояний сделало калибровку V1b не соответствующей спецификации.
    true(V.admissible({"f_mean": 0.05, "f_max": 0.06}) is False,
         "точка, у которой ограничение MBON не вычислялось, признана допустимой")
    true(V.passes_mbon({"f_mean": 0.05, "f_max": 0.06}) is None,
         "невычисленное ограничение MBON обязано отличаться от вычисленного")
    # ограничение по доле остаётся доступным отдельно: по нему идёт отсев перед
    # дорогим прогоном MBON
    eq(V.passes_fraction({"f_mean": 0.05, "f_max": 0.06}), True,
       "ограничение по доле как отдельная функция")


@check("V1b-4.6", "Среди допустимых точек СТАДИИ 2 берётся минимум |f_C - 0,05|")
def t_choose_stage2(V, neurons, con):
    """Точка стадии 1 не должна выигрывать выбор у точки стадии 2."""
    pts = [{"pn_kc_scale": 1.0, "g_apl_rel": 0.1, "f_mean": 0.050, "f_max": 0.06,
            "mbon": MBON_OK, "s_ab": 2.2, "stage": 1},
           {"pn_kc_scale": 2.0, "g_apl_rel": 0.2, "f_mean": 0.052, "f_max": 0.06,
            "mbon": MBON_OK, "s_ab": 2.6, "stage": 2}]
    best = V.choose_point(pts)
    eq(best["pn_kc_scale"], 2.0,
       "выбор идёт среди допустимых точек стадии 2; точка стадии 1 выиграла")


@check("V1b-4.6", "Лексикографически: |s - 2,2|, затем расстояние Чебышёва до "
                  "недопустимой точки или края, затем min g_apl, "
                  "затем min pn_kc_scale")
def t_choose(V, neurons, con):
    eq(V.SPIKES_AB_TARGET, 2.2, "эталон спайков за ответ")
    pts = [{"pn_kc_scale": 1.0, "g_apl_rel": 0.1, "f_mean": 0.049, "f_max": 0.06,
            "mbon": MBON_OK, "s_ab": 3.0, "stage": 2},
           {"pn_kc_scale": 2.0, "g_apl_rel": 0.2, "f_mean": 0.051, "f_max": 0.06,
            "mbon": MBON_OK, "s_ab": 2.3, "stage": 2}]
    best = V.choose_point(pts)
    eq(best["pn_kc_scale"], 2.0, "внутри полосы выбор идёт по |s - 2,2|")
    # вне полосы - по |f - 0,05|
    out = [{"pn_kc_scale": 1.0, "g_apl_rel": 0.1, "f_mean": 0.035, "f_max": 0.06,
            "mbon": MBON_OK, "s_ab": 2.2, "stage": 2},
           {"pn_kc_scale": 2.0, "g_apl_rel": 0.2, "f_mean": 0.042, "f_max": 0.06,
            "mbon": MBON_OK, "s_ab": 9.9, "stage": 2}]
    eq(V.choose_point(out)["pn_kc_scale"], 2.0, "вне полосы выбор идёт по |f - 0,05|")
    # точка с неопределённым s ранжируется ниже всех определённых
    nan = [{"pn_kc_scale": 1.0, "g_apl_rel": 0.1, "f_mean": 0.05, "f_max": 0.06,
            "mbon": MBON_OK, "s_ab": float("nan"), "stage": 2},
           {"pn_kc_scale": 2.0, "g_apl_rel": 0.2, "f_mean": 0.05, "f_max": 0.06,
            "mbon": MBON_OK, "s_ab": 9.9, "stage": 2}]
    eq(V.choose_point(nan)["pn_kc_scale"], 2.0,
       "точка с неопределённым s ранжируется ниже точки с определённым")
    # тай-брейк (3) и (4): при равном |s - 2,2| выигрывает меньший g_apl,
    # при равном g_apl - меньший pn_kc_scale
    tie34 = [{"pn_kc_scale": 3.0, "g_apl_rel": 0.2, "f_mean": 0.05, "f_max": 0.06,
              "mbon": MBON_OK, "s_ab": 2.2, "stage": 2},
             {"pn_kc_scale": 1.0, "g_apl_rel": 0.1, "f_mean": 0.05, "f_max": 0.06,
              "mbon": MBON_OK, "s_ab": 2.2, "stage": 2}]
    eq(V.choose_point(tie34)["pn_kc_scale"], 1.0, "тай-брейки по g_apl и масштабу")
    # тай-брейк (2), стоящий ВЫШЕ них: при равном |s - 2,2| выигрывает точка,
    # дальше отстоящая от недопустимой области или края сетки
    tie2 = [{"pn_kc_scale": 1.0, "g_apl_rel": 0.1, "f_mean": 0.05, "f_max": 0.06,
             "mbon": MBON_OK, "s_ab": 2.2, "stage": 2, "chebyshev_to_edge": 1},
            {"pn_kc_scale": 2.0, "g_apl_rel": 0.2, "f_mean": 0.05, "f_max": 0.06,
             "mbon": MBON_OK, "s_ab": 2.2, "stage": 2, "chebyshev_to_edge": 4}]
    eq(V.choose_point(tie2)["pn_kc_scale"], 2.0,
       "тай-брейк по расстоянию Чебышёва до края не реализован")


@check("V1b′-3.5а", "Если max_o r_t(o) + min_o r_t(o) = 0, величина MD_t "
                     "не определена и считается НЕ выполнившей V1b-3.5")
def t_md_undefined(V, neurons, con):
    r = pd.DataFrame(np.full((6, 14), 10.0), index=V.T38, columns=V.PANEL_ALL)
    r.iloc[:, 0] = 20.0                       # у всех типов MD определена и велика
    r.iloc[0, :] = 0.0                        # молчащий тип: max + min = 0
    res = V.check_mbon(r)
    md = res["per_type"][V.T38[0]]["MD"]
    true(md != md, "MD молчащего типа обязана быть не определена, получено %r" % md)
    eq(res["n_md_ok"], 5, "молчащий тип в подсчёте MD не засчитывается")
    eq(res["floor_ok"], False, "молчащий тип проваливает и пол")


@check("V1b′-3.1а", "Число предъявлений набора P14-M - 6; окно измерения "
                     "равно окну предъявления и начинается в t_on")
def t_p14m_protocol(V, neurons, con):
    eq(V.M_PULSE_MS, V.M_WINDOW_MS, "окно измерения равно окну предъявления")
    eq(V.M_PULSE_MS, 5000, "длительность предъявления")
    import inspect
    src = inspect.getsource(V.eval_p14m)
    true("M_PULSE_MS" in src and "M_WINDOW_MS" in src,
         "прогон P14-M обязан идти на тайминге M")
    true("SEED_EVAL_M" in inspect.signature(V.eval_p14m).parameters
         or "SEED_EVAL_M" in src, "прогон-вердикт идёт на своих зёрнах")


@check("V1b′-4.4", "Строка g_apl = 0 входит в сетку стадии 2 тогда и только "
                    "тогда, когда среди кандидатов стадии 1 есть точка с "
                    "нулевым гейном")
def t_zero_gain_row(V, neurons, con):
    ok = {"f_mean": 0.05, "f_max": 0.06}
    without = V.grid_stage2([dict(ok, pn_kc_scale=2.0, g_apl_rel=10.0 ** -0.5)])
    true(0.0 not in {g for _, g in without},
         "нулевой гейн попал в сетку без кандидата с нулевым гейном")
    with_zero = V.grid_stage2([dict(ok, pn_kc_scale=2.0, g_apl_rel=0.0),
                               dict(ok, pn_kc_scale=2.0, g_apl_rel=10.0 ** -0.5)])
    true(0.0 in {g for _, g in with_zero},
         "нулевой гейн не попал в сетку при кандидате с нулевым гейном")


@check("V1b′-4.4", "Расширение на шаг стадии 1 не обрезается диапазоном "
                    "стадии 1: тонкая сетка законно выходит за грубую")
def t_expansion_not_clipped(V, neurons, con):
    s1_max = max(s for s, _ in V.grid_stage1())
    g = V.grid_stage2([{"pn_kc_scale": s1_max, "g_apl_rel": 1.0,
                        "f_mean": 0.05, "f_max": 0.06}])
    eq(max(s for s, _ in g), s1_max * 2.0,
       "верхняя граница масштаба стадии 2 при кандидате на краю грубой сетки")


@check("V1b′-4.4", "Прямоугольник стадии 2 строится по КАНДИДАТАМ стадии 1, "
                    "а не по допустимым точкам")
def t_bbox_over_candidates(V, neurons, con):
    # кандидат без вычисленного ограничения MBON: допустимым не является,
    # но прямоугольник по нему строиться обязан
    cand = [{"pn_kc_scale": 2.0, "g_apl_rel": 1.0, "f_mean": 0.05, "f_max": 0.06}]
    eq(V.admissible(cand[0]), False, "кандидат без MBON допустимым не является")
    true(len(V.grid_stage2(cand)) > 0,
         "сетка стадии 2 пуста: прямоугольник построен по допустимым, а не по "
         "кандидатам")


@check("V1b′-4.6", "d(p) = min по q из D расстояния Чебышёва, D - недопустимые "
                    "точки сетки И все узлы вне сетки; d >= 1 для любой точки")
def t_chebyshev(V, neurons, con):
    # сетка 5x5, допустимы все. Ближайшие узлы вне сетки лежат на индексах -1
    # и 5, поэтому центральная точка (индекс 2) отстоит на 3, а угловая - на 1.
    pts = [{"pn_kc_scale": float(i), "g_apl_rel": float(j),
            "f_mean": 0.05, "f_max": 0.06, "mbon": MBON_OK}
           for i in range(5) for j in range(5)]
    m = V.chebyshev_margins(pts)
    eq(m[(2.0, 2.0)], 3, "расстояние центральной точки сетки 5x5")
    eq(m[(0.0, 0.0)], 1, "расстояние угловой точки: узлы вне сетки недопустимы")
    true(all(v >= 1 for v in m.values()), "d >= 1 для любой точки")
    # недопустимый сосед сокращает расстояние
    pts2 = [dict(p) for p in pts]
    for p in pts2:
        if (p["pn_kc_scale"], p["g_apl_rel"]) == (3.0, 2.0):
            p["mbon"] = dict(MBON_OK, floor_ok=False, pass_=False)
    m2 = V.chebyshev_margins(pts2)
    eq(m2[(2.0, 2.0)], 1, "недопустимый сосед сокращает расстояние до 1")


@check("V1b-4.9", "Множество клеток - KC с меткой hemibrain_type на KCab "
                  "(1 771 клетка); окно эталона [t_on; t_on + 2 с]; учитываются "
                  "пары «клетка - запах», отвечающие по V1b-2.1-2.2")
def t_spikes_per_response(V, neurons, con):
    kc = neurons[neurons.mb_role == "Kenyon_Cell"].hemibrain_type.astype(str)
    eq(int(kc.str.startswith("KCab").sum()), 1771, "число KCab в подсхеме")
    apl = set(neurons.loc[neurons.mb_role == "APL", "root_id"])
    core = sorted(set(neurons.root_id) - apl)
    m = V._kc_mask(core, neurons, "KCab")
    eq(int(m.sum()), 1771, "маска подтипа KCab")
    # одна отвечающая клетка: 4 пробы со спайком в окне эталона по 3 спайка
    idx = int(np.nonzero(m)[0][0])
    full = np.zeros((len(core), 6), dtype=np.int32); full[idx, :] = 1
    ref = np.zeros((len(core), 6), dtype=np.int32); ref[idx, :4] = 3
    res = {"core_ids": core, "counts": full, "counts_by_window": [full, ref],
           "windows": [(0.0, 4.0), (0.0, 2.0)], "seeds": list(range(6))}
    eq(V.spikes_per_response({"o": res}, neurons, "KCab", 1), 3.0,
       "счёт пары - среднее по предъявлениям, где в окне эталона есть спайк")
    # клетка, не отвечающая по V1b-2.2 (2 пробы из 6), в счёт не входит
    full2 = np.zeros((len(core), 6), dtype=np.int32); full2[idx, :2] = 1
    res2 = dict(res, counts=full2, counts_by_window=[full2, ref])
    true(np.isnan(V.spikes_per_response({"o": res2}, neurons, "KCab", 1)),
         "неотвечающая клетка не должна учитываться; величина не определена")


@check("V1b-3д", "«Запись» - случайная равномерная подвыборка 120 KC, число "
                 "подвыборок 24; разделение >= 0,25, потолок 0,40")
def t_overlap_params(V, neurons, con):
    eq(V.N_SUBSAMPLE, 120, "размер подвыборки «записи»")
    eq(V.N_DRAWS, 24, "число подвыборок")
    eq(V.OVERLAP_SEP_MIN, 0.25, "порог разделения")
    eq(V.OVERLAP_CEIL, 0.40, "потолок для химически различных пар")
    r = {V.REF_PAIRS[0]: 0.70, V.REF_PAIRS[1]: 0.15, V.REF_PAIRS[2]: 0.11}
    ov = V.check_overlap(r)
    eq(ov["pass"], True, "эталонные величины [40] должны проходить критерий")
    eq(round(ov["separation"], 4), 0.55, "разделение на эталонных величинах")
    bad = {V.REF_PAIRS[0]: 0.70, V.REF_PAIRS[1]: 0.50, V.REF_PAIRS[2]: 0.11}
    eq(V.check_overlap(bad)["ceiling_ok"], False, "потолок 0,40 нарушен")


@check("V1b-4.7", "Критерий перекрытия на E проверяет независимую величину - "
                  "это единственный из четырёх критериев, который ступень "
                  "проверяет вслепую")
def t_blindness(V, neurons, con):
    """Слепота E исполняема: при неполном наборе E критерии не вычисляются."""
    import inspect
    src = inspect.getsource(V.eval_p14)
    true("not_computed" in src,
         "eval_p14 обязана отказываться считать критерии, когда набор E "
         "в прогон не входит целиком")
    true("PANEL_EVAL" in src, "принадлежность к E проверяется явно")


@check("v0.11-а", "При выключенных подменах субстрат совпадает с V1a; маска "
                  "снимает 49 316 рёбер")
def t_regression(V, neurons, con):
    r = V.regression_substrate(neurons, con)
    eq(r["n_masked"], 49316, "число снятых маской рёбер")
    eq(r["mask_count_ok"], True, "счёт снятых рёбер")
    eq(r["substitutions_off_identical"], True, "субстрат при выключенных подменах")


@check("v0.11-б", "Разность конфигураций между режимами равна ровно объявленному "
                  "множеству ключей; хэши с исключением объявленных ключей равны")
def t_config_hash(V, neurons, con):
    off = {"dan_mask": False, "graded_apl": False, "odor_input": False,
           "pn_kc_scale": 1.0, "g_apl_rel": 0.0, "substrate": "flywire_630"}
    on = dict(off, dan_mask=True, graded_apl=True, odor_input=True,
              pn_kc_scale=8.0, g_apl_rel=3.16228)
    eq(V.config_hash(off, V.DECLARED_KEYS), V.config_hash(on, V.DECLARED_KEYS),
       "хэши конфигов с исключением объявленных ключей")
    # необъявленный ключ обязан ломать равенство
    sneaky = dict(on, w_syn_mV=0.3)
    true(V.config_hash(off, V.DECLARED_KEYS) != V.config_hash(sneaky, V.DECLARED_KEYS),
         "изменение НЕобъявленного ключа обязано ломать равенство хэшей")


# --- предрегистрация ступени V1c (разделы 3з и 3и) ---------------------------
#
# Спецификация V1c требует «тест соответствия кода спецификации перед прогоном»
# и относит к нему проверку согласованности таблицы детекторов и структурный
# тест масштабирования (V1c-E7.1). Проверки ниже - их часть, исполнимая без
# запуска симулятора.

V1C_NODES = ["1", "1.6836", "1.7783", "3.1623", "5.3875"]


def near(got, want, tol, what):
    """Равенство с явным допуском: eq() держит жёсткие 1e-9."""
    if not abs(float(got) - float(want)) <= tol:
        raise Fail("%s: получено %.12g, спецификация требует %.12g "
                   "(допуск %.3g)" % (what, got, want, tol))


def _v1c_artifacts():
    import json
    d = HERE / "results" / "v1c"
    w = json.loads((d / "weights_kc_mbon.json").read_text(encoding="utf-8"))
    t = json.loads((d / "detector_table.json").read_text(encoding="utf-8"))
    return w, t


@check("V1c-E3.2", "Узлы сетки финальной версии - {1; 1,6836; 1,7783; 3,1623; "
                   "5,3875}, пять узлов; список входит в хэш и не пополняется")
def t_v1c_nodes(V, neurons, con):
    """V1c-E3.2  Пять канонических узлов сетки по s, строками спецификации."""
    import v1c_detectors as D
    eq(list(D.NODES), V1C_NODES, "канонические узлы сетки")
    _, t = _v1c_artifacts()
    eq(list(t["nodes"]), V1C_NODES, "узлы таблицы детекторов")


@check("V1c-E3.1", "s_max = (v_th - v_0) / (A * max_m w_max(m)) по клеткам "
                   "шести типов T38; A - чистое число из t_mbr и tau")
def t_v1c_smax(V, neurons, con):
    """V1c-E3.1  s_max и s_pop пересчитываются из констант и весов."""
    import math
    w, _ = _v1c_artifacts()
    c = w["model_constants"]
    rho = c["t_mbr_ms"] / c["tau_ms"]
    a = (1.0 / (rho - 1.0)) * (rho ** (-1.0 / (rho - 1.0))
                               - rho ** (-rho / (rho - 1.0)))
    near(a, w["epsp"]["A"], 1e-12, "множитель A")
    near(a, 0.157490, 1e-6, "A с точностью записи спецификации")
    t_star = (c["tau_ms"] * c["t_mbr_ms"] * math.log(rho)
              / (c["t_mbr_ms"] - c["tau_ms"]))
    near(t_star, 9.242, 5e-4, "время пика одиночного ВПСП")
    theta = c["v_th_mV"] - c["v_0_mV"]
    near(theta, 7.0, 1e-9, "порог θ")
    near(theta / (a * w["s_max"]["w_max_mV_T38"]), 5.387542964, 1e-6, "s_max")
    near(theta / (a * w["s_max"]["w_max_mV_all_mbon"]), 1.683607176, 1e-6,
         "s_pop")


@check("V1c-E3.2", "Граничные узлы округляются вниз, так что на них "
                   "неравенство условия (5) остаётся строгим")
def t_v1c_rounding(V, neurons, con):
    """V1c-E3.2  Округление вниз оставляет строгими граничные неравенства."""
    w, _ = _v1c_artifacts()
    a = w["epsp"]["A"]
    theta = w["model_constants"]["theta_mV"]
    for node, w_max, who in ((5.3875, w["s_max"]["w_max_mV_T38"], "s_max/T38"),
                             (1.6836, w["s_max"]["w_max_mV_all_mbon"],
                              "s_pop/все MBON")):
        if not node * a * w_max < theta:
            raise Fail("на узле %s неравенство не строгое: %.10f >= %.1f"
                       % (who, node * a * w_max, theta))


@check("V1c-E2.2", "Условие (5): одиночный спайк одной клетки Кеньона не "
                   "доводит ни один MBON множества T38 до порога")
def t_v1c_condition5(V, neurons, con):
    """V1c-E2.2  Ни на одном узле сетки нет детекторов среди типов T38."""
    _, t = _v1c_artifacts()
    for node in V1C_NODES:
        eq(t["by_node"][node]["n_detectors_T38"], 0,
           "детекторов в T38 на узле %s" % node)
    if not t["condition5_holds_on_all_nodes"]:
        raise Fail("артефакт не подтверждает условие (5)")


@check("V1c-E7.1", "Клетка m есть детектор на узле s_k тогда и только тогда, "
                   "когда s_k · A · w_max(m) > θ; неравенство строгое")
def t_v1c_detectors(V, neurons, con):
    """V1c-E7.1  Таблица детекторов пересчитывается из весов и совпадает."""
    import v1c_detectors as D
    w, t = _v1c_artifacts()
    a, theta = w["epsp"]["A"], w["model_constants"]["theta_mV"]
    for node in V1C_NODES:
        want = sorted(int(k) for k, d in w["per_mbon"].items()
                      if float(node) * a * d["w_max_mV"] > theta)
        got = sorted(c["mbon_id"] for c in t["by_node"][node]["detectors"])
        eq(got, want, "детекторы на узле %s" % node)
    fresh = D.build()
    eq(fresh["by_node"], t["by_node"], "таблица, пересчитанная скриптом")


@check("V1c-E5", "Если у какого-либо типа T38 все клетки имеют нулевую сумму "
                 "весов KC->m, ступень непроверяема по построению")
def t_v1c_testability(V, neurons, con):
    """V1c-E5  Все шесть типов T38 имеют ненулевой вход KC."""
    w, _ = _v1c_artifacts()
    eq(w["testability_V1c_E5"]["types_with_zero_total_weight"], [],
       "типы T38 с нулевым входом KC")
    for t38 in V.T38:
        d = w["by_type_T38"][t38]
        if d["n_cells"] == 0 or d["sum_w_mV_total"] <= 0:
            raise Fail("тип %s не имеет входа KC" % t38)


@check("V1c-E7.1", "Проверка согласованности: таблица пересчитывается из "
                   "весов, загруженных тем же загрузчиком")
def t_v1c_weights_consistent(V, neurons, con):
    """V1c-E7.1  Веса KC->MBON артефакта сходятся с подсхемой и маской."""
    w, _ = _v1c_artifacts()
    eq(w["substrate"]["dan_mask_touches_kc_mbon"], False,
       "маска подмен не касается рёбер KC->MBON")
    eq(w["substrate"]["n_edges_kc_mbon_before_mask"],
       w["substrate"]["n_edges_kc_mbon_after_mask"],
       "число рёбер KC->MBON до и после маски")
    eq(sum(d["n_edges_kc"] for d in w["per_mbon"].values()),
       w["all_mbon"]["n_edges_kc"], "сумма рёбер по клеткам")
    near(sum(d["sum_w_mV"] for d in w["per_mbon"].values()),
         w["all_mbon"]["sum_w_mV_total"], 1e-6, "сумма весов по клеткам")
    eq(max(d["w_max_mV"] for d in w["per_mbon"].values()),
       w["s_max"]["w_max_mV_all_mbon"], "максимум веса по всем клеткам")
    # структурный тест масштабирования на загруженной сети принадлежит
    # прогонщику ступени и добавляется вместе с ним: здесь проверяется то,
    # что проверяемо без запуска симулятора

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    import v1b_subcircuit as V
    neurons, con = V.load_substrate()

    print("Соответствие кода спецификации: V1b/V1b' и предрегистрация V1c "
          "(%d проверок)" % len(CHECKS))
    print("=" * 78)
    failed = []
    for code, requirement, fn in CHECKS:
        try:
            fn(V, neurons, con)
            status, detail = "ОК  ", ""
        except Fail as e:
            status, detail = "ПРОВАЛ", str(e)
            failed.append((code, requirement, detail))
        except Exception as e:  # ошибка самого теста - тоже провал
            status = "ОШИБКА"
            detail = "%s: %s" % (type(e).__name__, e)
            failed.append((code, requirement, detail))
        print("%-6s %-10s %s" % (status, code, fn.__doc__ or requirement[:58]))
        if detail:
            print("       -> %s" % detail)
        if a.verbose:
            print("       требование: %s" % requirement)

    print("=" * 78)
    if failed:
        print("провалено %d из %d:" % (len(failed), len(CHECKS)))
        for code, requirement, detail in failed:
            print("  %s — %s" % (code, requirement))
            print("     %s" % detail)
        return 1
    print("все %d проверок прошли" % len(CHECKS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
