# -*- coding: utf-8 -*-
"""Диагностика зазора: окно доли отвечающих KC против окна частот MBON.

Вне ступени. Ни один критерий здесь не вычисляется как критерий и ни один
параметр по результатам не выбирается: назначение - придать количественную
форму несовместимости, установленной прогоном V1b' (исход FAIL-CAL-MBON).
Множитель KC->MBON не трогается: это ручка V1c, её первое измерение обязано
пройти под хэшем.

Две оси меряются по-разному, и это не небрежность, а свойство критериев.

  доля отвечающих KC   определена при тайминге P14 по правилу V1b-2.1-2.3:
                       не менее одного спайка не менее чем в трёх пробах из
                       шести. На двух пробах правило недостижимо и f тождественно
                       равна нулю, поэтому доля НЕ пересчитывается здесь, а
                       читается из артефактов калибровки V1b' - там она
                       посчитана шестью пробами на зёрнах SEED_CAL при обоих
                       нужных масштабах и 21 значении g_apl.
  частота MBON R_t     определена при тайминге M как среднее по пробам и по
                       клеткам типа (V1b-3.1). Правила «не менее чем в половине
                       проб» у неё нет, поэтому две пробы её оценивают, и
                       прогоняется здесь только она.

Набор запахов - калибровочный C: слепота набора E не расходуется (V1b-4.7).
Зёрна SEED_CAL_M + 1..2 - те же самые первые две пробы, которые ступень
использовала для ограничения допустимости, поэтому величины сопоставимы с
кандидатами напрямую.

Артефакты пишутся в results/diag_window: каталоги закрытых ступеней V1b и
V1b' служат базой сравнения и не перезаписываются.

Запуск (нужно окружение MSVC, иначе cython не соберётся):
    python  v1b_window_diag.py --map                    # без прогона, мгновенно
    msvc_run.bat v1b_window_diag.py --scan --shard 0 --of 8   # и так для 0..7
    msvc_run.bat v1b_window_diag.py --scan --grid refine --shard 0 --of 8
    python  v1b_window_diag.py --merge
    msvc_run.bat v1b_window_diag.py --verify            # сверка с артефактом ступени
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PRIME = HERE / "results" / "v1b_prime"      # источник карты доли, только чтение
OUT = HERE / "results" / "diag_window"      # свой каталог, ничего не перезаписывает

# Два масштаба: 8 - верхний угол сетки стадии 2 и выбранная точка пилота,
# 4 - вдвое ниже, единственный другой масштаб, у которого окно доли целиком
# лежит внутри сетки.
SCALES = (8.0, 4.0)

# Десять значений g_apl от нуля до максимума сетки стадии 2. Все, кроме нуля,
# лежат на узлах сетки стадии 2 (шаг 10^(1/8)); ноль - узел стадии 1. Узлы
# выбраны так, чтобы диапазон покрывался примерно равномерно по log g и чтобы
# в него попали точки, для которых ступень уже посчитала MBON шестью пробами.
G_VALUES = (0.0, 0.01, 0.031623, 0.1, 0.17783,
            0.31623, 0.56234, 1.0, 1.7783, 3.16228)

# Уточнение разрешения (правило Е2). Не фиксированная лестница узлов, а
# механическая дихотомия, объявленная в журнале измерения
# results/diag_window/journal_e6_2.md ДО запуска: пока нижний конец вилки равен
# нулю или минимум R_t на нём ниже пола - деление пополам линейно; затем деление
# пополам по логарифму, пока отношение концов вилки не станет 1,25 или меньше.
# Предел объявлен и жёсткий. Узлы основного прохода не пересчитываются.
REFINE_MAX_NODES = 8            # на масштаб, предел из журнала
REFINE_LOG_RATIO = 1.25         # остановка шага 2

N_TRIALS_DIAG = 2
# Точка сверки не задаётся константой, а выбирается как кандидат ступени с
# наибольшей ненулевой частотой: сверка на точке, где все шесть типов молчат,
# прошла бы и при сломанном стенде.
VERIFY_PICK = "кандидат V1b' с максимальной R_t по T38"


# --- ось доли отвечающих KC: чтение артефактов, без прогона -------------------

def _load(p: Path) -> list:
    if not p.exists():
        raise SystemExit("нет артефакта: %s" % p)
    return json.loads(p.read_text(encoding="utf-8"))


def fraction_map(V) -> dict:
    """Окно доли отвечающих KC по оси g_apl, из артефактов калибровки V1b'.

    Точка входит в окно, если выполнена ПЕРВАЯ половина V1b-4.5 целиком:
    f_mean в полосе И f_max не выше потолка. Это та же функция passes_fraction,
    которой пользовалась ступень, - здесь она не вычисляет критерий заново, а
    применяется к уже записанным величинам.
    """
    pts = _load(PRIME / "calibration_stage1.json") + \
          _load(PRIME / "calibration_stage2.json")
    out = {"source": ["results/v1b_prime/calibration_stage1.json",
                      "results/v1b_prime/calibration_stage2.json"],
           "rule": "V1b-2.1-2.3 при тайминге P14: >=1 спайка в >=3 пробах из 6",
           "band": list(V.F_BAND), "ceiling": V.F_MAX,
           "seed_base": V.SEED_CAL, "n_trials": V.N_TRIALS, "by_scale": {}}
    dup_checked = 0
    for sc in SCALES:
        rows = [p for p in pts if abs(p["pn_kc_scale"] - sc) < 1e-9]
        # Сетки стадий пересекаются: узел, попавший в обе, записан дважды.
        # Дубликат снимается, но перед этим величины сверяются - совпадение
        # обязано быть побитовым, потому что зёрна и протокол у стадий одни.
        by_g = {}
        for p in rows:
            k = round(p["g_apl_rel"], 10)
            if k in by_g:
                if (by_g[k]["f_mean"], by_g[k]["f_max"]) != (p["f_mean"], p["f_max"]):
                    raise SystemExit(
                        "узел (scale %g, g %g) записан стадиями по-разному: "
                        "%r против %r - недетерминизм, разбирать до отчёта"
                        % (sc, p["g_apl_rel"],
                           (by_g[k]["f_mean"], by_g[k]["f_max"]),
                           (p["f_mean"], p["f_max"])))
                dup_checked += 1
                # предпочтение точке стадии 2: у неё считался MBON
                if "mbon" in p and "mbon" not in by_g[k]:
                    by_g[k] = p
                continue
            by_g[k] = p
        rows = sorted(by_g.values(), key=lambda p: p["g_apl_rel"])
        curve = [{"g_apl_rel": p["g_apl_rel"], "f_mean": p["f_mean"],
                  "f_max": p["f_max"], "in_window": bool(V.passes_fraction(p)),
                  "mbon_computed_by_stage": "mbon" in p} for p in rows]
        win = [c["g_apl_rel"] for c in curve if c["in_window"]]
        out["by_scale"]["%g" % sc] = {
            "n_grid_points": len(curve),
            "g_min_grid": curve[0]["g_apl_rel"] if curve else None,
            "g_max_grid": curve[-1]["g_apl_rel"] if curve else None,
            "window_g_lo": min(win) if win else None,
            "window_g_hi": max(win) if win else None,
            "n_in_window": len(win),
            "curve": curve}
    out["duplicate_nodes_checked"] = dup_checked
    return out


# --- ось частот MBON: прогон при тайминге M -----------------------------------

def _rates_at(V, neurons, con, sc: float, g: float, seeds: list[int]) -> dict:
    """Величины MBON и KC в одной точке при тайминге M, набор C."""
    role = dict(zip(neurons.root_id, neurons.mb_role))
    n_kc_total = int(sum(1 for r in role.values() if r == "Kenyon_Cell"))
    by = {}
    kc_spk = mb_spk = 0
    kc_resp = np.zeros(0)
    for o in V.PANEL_CAL:
        res = V.run_odor(neurons, con, o, pn_kc_scale=sc,
                         g_apl=g * V.g_ref_value(), seeds=seeds,
                         pulse_ms=V.M_PULSE_MS, window_ms=V.M_WINDOW_MS)
        by[o] = res
        ids = res["core_ids"]
        kc = np.array([role[i] == "Kenyon_Cell" for i in ids])
        mb = np.array([role[i] == "MBON" for i in ids])
        kc_spk += int(res["counts"][kc].sum())
        mb_spk += int(res["counts"][mb].sum())
        r = ((res["counts"][kc] >= 1).sum(axis=1) >= 1)
        kc_resp = r if kc_resp.size == 0 else (kc_resp | r)

    rates = V.mbon_type_rates(by, neurons, V.M_WINDOW_MS)
    chk = V.check_mbon(rates)
    per = chk["per_type"]
    rt = {t: per[t]["R_t"] for t in chk["types_present"]}
    n_mbon = int(sum(1 for r in role.values() if r == "MBON"))
    active = set()
    for o, res in by.items():
        ids = res["core_ids"]
        for k, i in enumerate(ids):
            if role[i] == "MBON" and res["counts"][k].sum() > 0:
                active.add(i)
    return {"pn_kc_scale": sc, "g_apl_rel": g,
            "R_t_T38": rt,
            "R_t_min": min(rt.values()) if rt else None,
            "R_t_max": max(rt.values()) if rt else None,
            "n_types_at_zero": int(sum(1 for v in rt.values() if v == 0.0)),
            "floor_ok": bool(chk["floor_ok"]), "ceiling_ok": bool(chk["ceiling_ok"]),
            "corridor_ok": bool(chk["floor_ok"] and chk["ceiling_ok"]),
            "types_missing": chk["missing"],
            "kc_spikes_6_odors": kc_spk, "mbon_spikes_6_odors": mb_spk,
            "mbon_active_cells": len(active), "mbon_cells_total": n_mbon,
            # диагностическая доля, НЕ критерий: правило V1b-2.1-2.3 требует
            # трёх проб из шести и на двух пробах неприменимо; здесь - «хотя бы
            # один спайк хотя бы в одной из двух проб хотя бы на одном запахе C»
            "kc_responding_ge1_of2_any_odor_DIAG":
                float(kc_resp.sum()) / n_kc_total,
            "n_kc_denominator": n_kc_total}


def shard_path(i: int, n: int, grid: str = "main") -> Path:
    stem = "mbon_scan" if grid == "main" else "mbon_scan_refine"
    return OUT / ("%s.shard%02d_of%02d.json" % (stem, i, n))


def scan(V, neurons, con, shard: int, of: int, which: str = "main") -> int:
    grid = [(sc, g) for sc in SCALES for g in G_VALUES]
    mine = [p for k, p in enumerate(grid) if k % of == shard]
    path = shard_path(shard, of, which)
    done = _load(path) if path.exists() else []
    seen = {(round(p["pn_kc_scale"], 10), round(p["g_apl_rel"], 10)) for p in done}
    todo = [p for p in mine if (round(p[0], 10), round(p[1], 10)) not in seen]
    seeds = [V.SEED_CAL_M + i for i in range(1, N_TRIALS_DIAG + 1)]
    print("проход %s, шард %d из %d: точек %d, посчитано %d, к прогону %d"
          % (which, shard, of, len(mine), len(done), len(todo)), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for k, (sc, g) in enumerate(todo, 1):
        t0 = time.time()
        row = _rates_at(V, neurons, con, sc, g, seeds)
        row["wall_s"] = round(time.time() - t0, 1)
        row["seeds"] = seeds
        row["grid_pass"] = which
        done.append(row)
        path.write_text(json.dumps(done, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        print("  [%d/%d] scale %-4g g %-9.5g | KC %9d, MBON %7d, живых %3d/%d "
              "| min R_t %8.4f, max R_t %9.4f | %.0f c"
              % (k, len(todo), sc, g, row["kc_spikes_6_odors"],
                 row["mbon_spikes_6_odors"], row["mbon_active_cells"],
                 row["mbon_cells_total"], row["R_t_min"], row["R_t_max"],
                 row["wall_s"]), flush=True)
    print("шард %d готов" % shard, flush=True)
    return 0


def refine(V, neurons, con, sc: float) -> int:
    """Уточнение разрешения по правилу Е2 для одного масштаба.

    Дихотомия последовательна: следующий узел определяется предыдущим, поэтому
    шардами не бьётся. Правило и предел объявлены в журнале измерения ДО
    запуска; здесь они только исполняются.
    """
    seeds = [V.SEED_CAL_M + i for i in range(1, N_TRIALS_DIAG + 1)]
    path = OUT / ("mbon_scan_refine.scale%g.json" % sc)
    done = _load(path) if path.exists() else []

    # все известные узлы этого масштаба: основной проход плюс уже уточнённые
    known = []
    for q in sorted(OUT.glob("mbon_scan.shard*.json")):
        known += [r for r in _load(q) if abs(r["pn_kc_scale"] - sc) < 1e-9]
    known += done
    known = {round(r["g_apl_rel"], 12): r for r in known}

    def measure(g: float) -> dict:
        k = round(g, 12)
        if k in known:
            return known[k]
        t0 = time.time()
        row = _rates_at(V, neurons, con, sc, g, seeds)
        row["wall_s"] = round(time.time() - t0, 1)
        row["seeds"] = seeds
        row["grid_pass"] = "refine"
        done.append(row)
        known[k] = row
        path.write_text(json.dumps(done, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        print("  добавлен узел g = %.9g | KC %9d, MBON %7d, живых %3d "
              "| min R_t %8.4f, max R_t %9.4f | %.0f c"
              % (g, row["kc_spikes_6_odors"], row["mbon_spikes_6_odors"],
                 row["mbon_active_cells"], row["R_t_min"], row["R_t_max"],
                 row["wall_s"]), flush=True)
        return row

    added = len(done)
    curve = sorted(known.values(), key=lambda r: r["g_apl_rel"])
    floor = V.MBON_FLOOR_HZ
    above = [r for r in curve if r["R_t_min"] >= floor]

    if not above:
        # Вырожденный случай журнала: пол не пересекается, потому что минимум
        # ниже него уже в наиболее благоприятной точке оси g = 0. Пересекать
        # нечего; добавляются два узла как проверка монотонности.
        best = curve[0]
        print("масштаб %g: пола не пересекает - минимум R_t = %.4f Гц уже при "
              "g = %.4g. Уточнение вырождено, добавляются два узла проверки "
              "монотонности." % (sc, best["R_t_min"], best["g_apl_rel"]))
        for g in (0.0025, 0.005):
            if added >= REFINE_MAX_NODES:
                break
            measure(g)
            added += 1
        return 0

    g_lo = max(r["g_apl_rel"] for r in above)
    hi = [r for r in curve if r["g_apl_rel"] > g_lo]
    if not hi:
        print("масштаб %g: пол не нарушается ни в одном узле - уточнять нечего" % sc)
        return 0
    g_hi = min(r["g_apl_rel"] for r in hi)
    print("масштаб %g: стартовая вилка (%.9g; %.9g), предел %d узлов"
          % (sc, g_lo, g_hi, REFINE_MAX_NODES))

    while added < REFINE_MAX_NODES:
        lo_row = known[round(g_lo, 12)]
        if g_lo == 0.0 or lo_row["R_t_min"] < floor:
            g_new = 0.5 * (g_lo + g_hi)          # шаг 1: линейно
        else:
            if g_hi / g_lo <= REFINE_LOG_RATIO:  # шаг 2 завершён
                break
            g_new = math.sqrt(g_lo * g_hi)       # шаг 2: по логарифму
        row = measure(g_new)
        added += 1
        if row["R_t_min"] >= floor:
            g_lo = g_new
        else:
            g_hi = g_new

    print("масштаб %g: финальная вилка (%.9g; %.9g), добавлено узлов %d"
          % (sc, g_lo, g_hi, added))
    return 0


def fraction_at(V, neurons, con, sc: float, gs: list) -> int:
    """Доля отвечающих KC в объявленных узлах: тайминг P14, шесть проб.

    Протокол тот же, что у источника выхода 1 (артефакты калибровки V1b'):
    та же функция ступени, те же зёрна, то же правило «не менее трёх проб из
    шести». Узлы объявлены в журнале измерения до запуска.
    """
    path = OUT / ("fraction_extra.scale%g.json" % sc)
    done = _load(path) if path.exists() else []
    seen = {round(r["g_apl_rel"], 12) for r in done}
    OUT.mkdir(parents=True, exist_ok=True)
    for g in gs:
        if round(g, 12) in seen:
            continue
        t0 = time.time()
        pt = V.evaluate_point(neurons, con, V.PANEL_CAL, pn_kc_scale=sc,
                              g_apl_rel=g, seed_base=V.SEED_CAL)
        pt.pop("_by_odor")
        pt["wall_s"] = round(time.time() - t0, 1)
        pt["grid_pass"] = "refine_fraction"
        pt["passes_fraction"] = bool(V.passes_fraction(pt))
        done.append(pt)
        path.write_text(json.dumps(done, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        print("  g = %.9g | f_mean %.4f, f_max %.4f, в полосе: %s | %.0f c"
              % (g, pt["f_mean"], pt["f_max"],
                 "да" if pt["passes_fraction"] else "нет", pt["wall_s"]),
              flush=True)
    return 0


# --- сведение и пересечение порога --------------------------------------------

def _interp_log(x0, y0, x1, y1, y) -> float:
    """Точка пересечения уровня y при линейной интерполяции по log10 x и log10 y."""
    lx0, lx1 = math.log10(x0), math.log10(x1)
    ly0, ly1 = math.log10(y0), math.log10(y1)
    return 10 ** (lx0 + (lx1 - lx0) * (math.log10(y) - ly0) / (ly1 - ly0))


def _f_at(curve: list, g: float) -> dict:
    """Доля отвечающих KC при данном g: узел сетки или интерполяция по log g."""
    pos = [c for c in curve if c["g_apl_rel"] > 0]
    for c in pos:
        if abs(c["g_apl_rel"] - g) < 1e-9:
            return {"f_mean": c["f_mean"], "kind": "узел сетки"}
    lo = [c for c in pos if c["g_apl_rel"] < g]
    hi = [c for c in pos if c["g_apl_rel"] > g]
    if not lo or not hi:
        return {"f_mean": None, "kind": "вне сетки"}
    a, b = lo[-1], hi[0]
    la, lb = math.log10(a["g_apl_rel"]), math.log10(b["g_apl_rel"])
    w = (math.log10(g) - la) / (lb - la)
    return {"f_mean": a["f_mean"] + w * (b["f_mean"] - a["f_mean"]),
            "kind": "интерполяция по log g между %.5g и %.5g"
                    % (a["g_apl_rel"], b["g_apl_rel"]),
            "bracket_f": [a["f_mean"], b["f_mean"]],
            "bracket_g": [a["g_apl_rel"], b["g_apl_rel"]]}


def merge(V) -> dict:
    rows = []
    for p in sorted(OUT.glob("mbon_scan.shard*.json")):
        rows += _load(p)
    refined = []
    for p in sorted(OUT.glob("mbon_scan_refine.*.json")):
        refined += _load(p)
    # Узел, посчитанный основным проходом, уплотнением не заменяется: порядок
    # решений сохраняется в артефакте, а не затирается.
    seen = {(round(r["pn_kc_scale"], 10), round(r["g_apl_rel"], 12)) for r in rows}
    rows += [r for r in refined
             if (round(r["pn_kc_scale"], 10), round(r["g_apl_rel"], 12)) not in seen]
    if not rows:
        raise SystemExit("шардов нет: сначала --scan")
    fm = fraction_map(V)
    # Узлы доли, добавленные по второй записи Е2: тот же протокол, что у
    # источника выхода 1, поэтому они входят в ту же кривую с пометкой прохода.
    for q in sorted(OUT.glob("fraction_extra.*.json")):
        for r in _load(q):
            key = "%g" % r["pn_kc_scale"]
            if key not in fm["by_scale"]:
                continue
            cur = fm["by_scale"][key]["curve"]
            if any(abs(c["g_apl_rel"] - r["g_apl_rel"]) < 1e-12 for c in cur):
                continue
            cur.append({"g_apl_rel": r["g_apl_rel"], "f_mean": r["f_mean"],
                        "f_max": r["f_max"],
                        "in_window": bool(V.passes_fraction(r)),
                        "mbon_computed_by_stage": False,
                        "grid_pass": "refine_fraction"})
            cur.sort(key=lambda c: c["g_apl_rel"])
    out = {"status": "диагностика вне ступени: критерии не вычисляются как "
                     "критерии, параметры не выбираются, множитель KC->MBON "
                     "не трогается",
           "odors": V.PANEL_CAL, "n_trials_mbon": N_TRIALS_DIAG,
           "timing_mbon": "M", "pulse_ms": V.M_PULSE_MS,
           "grid_main": list(G_VALUES),
           "refine_rule": {"max_nodes_per_scale": REFINE_MAX_NODES,
                           "log_stop_ratio": REFINE_LOG_RATIO,
                           "journal": "results/diag_window/journal_e6_2.md"},
           "n_nodes_main": len([r for r in rows
                                if r.get("grid_pass", "main") == "main"]),
           "floor_hz": V.MBON_FLOOR_HZ, "ceiling_hz": V.MBON_CEIL_HZ,
           "fraction_axis": fm, "by_scale": {}}
    for sc in SCALES:
        key = "%g" % sc
        curve = sorted([r for r in rows if abs(r["pn_kc_scale"] - sc) < 1e-9],
                       key=lambda r: r["g_apl_rel"])
        fw = fm["by_scale"][key]
        d = {"mbon_curve": curve,
             "fraction_window_g": [fw["window_g_lo"], fw["window_g_hi"]],
             "corridor_g": None, "crossing": None}
        cor = [r["g_apl_rel"] for r in curve if r["corridor_ok"]]
        if cor:
            d["corridor_g"] = [min(cor), max(cor)]
        # Производные величины правила Е4: считаются арифметически из уже
        # записанных наблюдаемых, новых прогонов не требуют, критериями быть
        # не могут. Коридор пуст на сетке тогда и только тогда, когда g67 >= g2.
        over = [r["g_apl_rel"] for r in curve
                if r["R_t_max"] is not None and r["R_t_max"] > V.MBON_CEIL_HZ]
        under = [r["g_apl_rel"] for r in curve
                 if r["R_t_min"] is not None and r["R_t_min"] < V.MBON_FLOOR_HZ]
        d["derived_E4"] = {
            "g_67": max(over) if over else None,
            "g_2": min(under) if under else None,
            "corridor_empty_on_grid":
                bool(over and under and max(over) >= min(under)),
            "note": "производные величины (правило Е4), критериями не являются"}
        # пересечение пола: наибольшее g, при котором min_t R_t ещё не ниже пола
        above = [r for r in curve if r["R_t_min"] is not None
                 and r["R_t_min"] >= V.MBON_FLOOR_HZ]
        below = [r for r in curve if r["R_t_min"] is not None
                 and r["R_t_min"] < V.MBON_FLOOR_HZ]
        if above and below:
            a = max(above, key=lambda r: r["g_apl_rel"])
            nxt = [r for r in below if r["g_apl_rel"] > a["g_apl_rel"]]
            b = min(nxt, key=lambda r: r["g_apl_rel"]) if nxt else None
            cr = {"bracket_g": [a["g_apl_rel"], b["g_apl_rel"] if b else None],
                  "bracket_min_R_t": [a["R_t_min"], b["R_t_min"] if b else None]}
            if b and a["g_apl_rel"] > 0 and a["R_t_min"] > 0 and b["R_t_min"] > 0:
                cr["g_star"] = _interp_log(a["g_apl_rel"], a["R_t_min"],
                                           b["g_apl_rel"], b["R_t_min"],
                                           V.MBON_FLOOR_HZ)
                cr["g_star_kind"] = ("интерполяция по log g и log R_t между "
                                     "узлами; оценка, не измерение")
                cr["f_at_g_star"] = _f_at(fw["curve"], cr["g_star"])
            else:
                cr["g_star"] = None
                cr["g_star_kind"] = ("интерполяция невозможна: нулевая частота "
                                     "или ноль на оси g; только вилка")
            cr["f_at_bracket"] = [
                _f_at(fw["curve"], a["g_apl_rel"])["f_mean"]
                if a["g_apl_rel"] > 0 else None,
                _f_at(fw["curve"], b["g_apl_rel"])["f_mean"] if b else None]
            d["crossing"] = cr
        # зазор: во сколько раз надо ослабить торможение против окна доли
        if d["crossing"] and d["crossing"].get("g_star") and fw["window_g_lo"]:
            d["gap_factor_g"] = fw["window_g_lo"] / d["crossing"]["g_star"]
        out["by_scale"][key] = d
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "window_gap.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("записано: %s" % p)
    return out


def print_summary(V, out: dict) -> None:
    for sc in SCALES:
        d = out["by_scale"]["%g" % sc]
        print("\n=== масштаб PN->KC = %g ===" % sc)
        print("%-10s %10s %10s %10s %9s | %s"
              % ("g/g_ref", "KC спайков", "MBON спайк", "живых MBON",
                 "min R_t", "коридор [2; 67]"))
        for r in d["mbon_curve"]:
            print("%-10.5g %10d %10d %10d %9.4f | %s"
                  % (r["g_apl_rel"], r["kc_spikes_6_odors"],
                     r["mbon_spikes_6_odors"], r["mbon_active_cells"],
                     r["R_t_min"], "да" if r["corridor_ok"] else "нет"))
        fw = d["fraction_window_g"]
        print("окно доли KC (P14, 6 проб, из артефактов ступени): "
              "g в [%s; %s]" % (fw[0], fw[1]))
        print("окно коридора MBON (M, 2 пробы, узлы этого прогона): %s"
              % ("g в [%s; %s]" % tuple(d["corridor_g"]) if d["corridor_g"]
                 else "пусто ни в одном узле"))
        cr = d["crossing"]
        if cr:
            print("пол %.1f Гц пересекается между g = %s и g = %s "
                  "(min R_t %.4f -> %.4f)"
                  % (out["floor_hz"], cr["bracket_g"][0], cr["bracket_g"][1],
                     cr["bracket_min_R_t"][0], cr["bracket_min_R_t"][1] or 0.0))
            if cr.get("g_star"):
                f = cr["f_at_g_star"]
                print("оценка точки пересечения g* = %.5g; доля отвечающих KC "
                      "там f = %s (%s)"
                      % (cr["g_star"],
                         "%.4f" % f["f_mean"] if f["f_mean"] is not None else "нет",
                         f["kind"]))
            print("доля отвечающих KC на концах вилки: %s"
                  % " -> ".join("%.4f" % v if v is not None else "нет"
                                for v in cr["f_at_bracket"]))
        e4 = d.get("derived_E4") or {}
        print("производные Е4: g_67 = %s, g_2 = %s -> коридор на сетке %s"
              % (e4.get("g_67"), e4.get("g_2"),
                 "ПУСТ" if e4.get("corridor_empty_on_grid") else "не пуст"))
        if d.get("gap_factor_g"):
            print("зазор по оси торможения: нижний край окна доли выше точки "
                  "пересечения пола в %.1f раза" % d["gap_factor_g"])


def verify(V, neurons, con) -> int:
    """Сверка стенда диагностики со ступенью: тот же расчёт, что делала V1b'.

    Точка - кандидат ступени, поэтому её величины MBON записаны в артефакте
    калибровки. Расхождение означает недетерминизм и разбирается до того, как
    числа диагностики попадут в отчёт.
    """
    rows = [p for p in _load(PRIME / "calibration_stage2.json") if "mbon" in p]
    if not rows:
        raise SystemExit("в артефакте нет ни одной точки с посчитанным MBON")
    pick = max(rows, key=lambda p: max(v["R_t"] for v in
                                       p["mbon"]["per_type"].values()))
    # Прогон ведётся на значениях ИЗ АРТЕФАКТА полной точности, а не на
    # округлённых: иначе сравнивались бы разные точки сетки.
    sc, g = pick["pn_kc_scale"], pick["g_apl_rel"]
    ref = pick["mbon"]["per_type"]
    if max(v["R_t"] for v in ref.values()) == 0.0:
        print("ПРЕДУПРЕЖДЕНИЕ: у лучшего кандидата все типы молчат, "
              "сверка вырождена", flush=True)
    m = V.eval_p14m(neurons, con, pn_kc_scale=sc, g_apl_rel=g,
                    seed_base=V.SEED_CAL_M, odors=V.PANEL_CAL)
    got = m["T38"]["per_type"]
    bad = 0
    print("сверка точки scale %g, g %g: R_t по типам, ступень против диагностики"
          % (sc, g))
    for t in sorted(set(ref) | set(got)):
        a = ref.get(t, {}).get("R_t")
        b = got.get(t, {}).get("R_t")
        ok = a is not None and b is not None and abs(a - b) < 1e-12
        bad += 0 if ok else 1
        print("  %-8s ступень %.6f, диагностика %.6f  %s"
              % (t, a if a is not None else float("nan"),
                 b if b is not None else float("nan"),
                 "" if ok else "<-- РАСХОЖДЕНИЕ"))
    print("расхождений: %d" % bad)
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", action="store_true", help="окно доли KC, без прогона")
    ap.add_argument("--scan", action="store_true", help="ось MBON, прогон")
    ap.add_argument("--refine", type=float, default=None,
                    help="уточнение разрешения по правилу Е2 для масштаба")
    ap.add_argument("--fraction-at", nargs="+", type=float, default=None,
                    metavar="G", help="доля KC в объявленных узлах (P14, 6 проб)")
    ap.add_argument("--scale", type=float, default=8.0)
    ap.add_argument("--merge", action="store_true", help="свести шарды и сосчитать")
    ap.add_argument("--verify", action="store_true", help="сверка со ступенью")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--codegen", default="cython", choices=["numpy", "cython"])
    a = ap.parse_args()
    if not any((a.map, a.scan, a.merge, a.verify, a.refine,
                a.fraction_at)):
        ap.error("нечего делать: --map, --scan, --refine, --fraction-at, "
                 "--merge или --verify")
    if a.refine is not None and a.refine not in SCALES:
        ap.error("масштаб %g не объявлен в измерении" % a.refine)

    if a.scan or a.verify or a.refine is not None or a.fraction_at:
        from brian2 import prefs
        prefs.codegen.target = a.codegen
        if a.codegen == "cython":
            tag = ("r%g" % a.refine) if a.refine is not None else ("w%02d" % a.shard)
            d = HERE / ".cython_cache" / ("wdiag_%s" % tag)
            d.mkdir(parents=True, exist_ok=True)
            prefs.codegen.runtime.cython.cache_dir = str(d)
    import v1b_subcircuit as V

    if a.map:
        fm = fraction_map(V)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "fraction_window.json").write_text(
            json.dumps(fm, ensure_ascii=False, indent=2), encoding="utf-8")
        for sc in SCALES:
            d = fm["by_scale"]["%g" % sc]
            print("масштаб %g: сетка g от %s до %s, %d узлов; окно доли "
                  "f в [%.2f; %.2f] и f_max <= %.2f -> g в [%s; %s], узлов %d"
                  % (sc, d["g_min_grid"], d["g_max_grid"], d["n_grid_points"],
                     V.F_BAND[0], V.F_BAND[1], V.F_MAX,
                     d["window_g_lo"], d["window_g_hi"], d["n_in_window"]))
    if a.scan or a.verify or a.refine is not None or a.fraction_at:
        neurons, con = V.load_substrate()
        if a.verify:
            return verify(V, neurons, con)
        if a.fraction_at:
            return fraction_at(V, neurons, con, a.scale, a.fraction_at)
        if a.refine is not None:
            return refine(V, neurons, con, a.refine)
        return scan(V, neurons, con, a.shard, a.of)
    if a.merge:
        print_summary(V, merge(V))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
