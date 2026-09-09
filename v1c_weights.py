# -*- coding: utf-8 -*-
"""Измерение V1c-E6.1: структурные веса KC->MBON в подсхеме.

Разрешено правилом Д (спецификация, раздел 1а) и перечислено в штампе 0
ступени V1c (раздел 3з, V1c-E6.1). Штамп 0 записан ДО этого измерения:
results/v1c/stamp0_sha256.txt. Симулятор не запускается, ни один параметр не
меняется, ни одна переменная состояния не регистрируется - читаются таблица
связей подсхемы и константы модели [2].

Что считается:
  - на каждый MBON подсхемы: число рёбер KC->m, число синапсов, сумма весов,
    максимальный вес одного ребра, квантили распределения весов 10/50/90 %;
  - агрегаты на каждый из шести типов T38 и на все MBON;
  - число A и время пика одиночного ВПСП по формулам V1c-E3.1;
  - верхняя граница сетки s_max по формуле V1c-E3.1;
  - проверка тестируемости V1c-E5: тип T38 с нулевой суммой весов у всех
    своих клеток делает ступень непроверяемой по построению.

Список выходов записан в штампе 0 до запуска и здесь не расширяется.

Запуск:  python v1c_weights.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "v1c"
STAMP0 = OUT / "stamp0_sha256.txt"


def epsp_peak_factor(t_mbr_ms: float, tau_ms: float) -> tuple[float, float]:
    """Множитель и время пика одиночного ВПСП (V1c-E3.1).

    Модель [2] токовая: спайк добавляет вес w к переменной g, а мембрана
    интегрирует g. Из dv/dt = (v_0 - v + g)/t_mbr и dg/dt = -g/tau при покое
    одиночный спайк даёт отклонение
        u(t) = w/(1 - rho) * (exp(-t/tau) - exp(-t/t_mbr)),  rho = t_mbr/tau,
    пик которого достигается в t* и равен A*w.
    """
    rho = t_mbr_ms / tau_ms
    if abs(rho - 1.0) < 1e-12:
        raise SystemExit("вырожденный случай t_mbr = tau: формула пика иная")
    a = (1.0 / (rho - 1.0)) * (rho ** (-1.0 / (rho - 1.0))
                               - rho ** (-rho / (rho - 1.0)))
    t_star = tau_ms * t_mbr_ms * math.log(rho) / (t_mbr_ms - tau_ms)
    return float(a), float(t_star)


def main() -> int:
    if not STAMP0.exists():
        raise SystemExit(
            "штамп 0 не найден: %s\nправило Д1 запрещает это измерение до "
            "записи штампа 0" % STAMP0)

    import v1b_subcircuit as V
    from model import default_params as dp

    neurons, con = V.load_substrate()
    role = dict(zip(neurons.root_id, neurons.mb_role))
    typ = dict(zip(neurons.root_id,
                   neurons.hemibrain_type.astype("string").fillna(V.UNTYPED)))

    # Та же маска подмен, что у ступени: рёбра DAN->KC и DAN->MBON снимаются.
    # На рёбра KC->MBON она влиять не должна, и это проверяется, а не
    # предполагается.
    con_masked, n_masked = V.apply_dan_mask(con, role)

    def kc_mbon(c: pd.DataFrame) -> pd.DataFrame:
        pre = c.Presynaptic_ID.map(role)
        post = c.Postsynaptic_ID.map(role)
        return c[(pre == "Kenyon_Cell") & (post == "MBON")]

    e_all, e_masked = kc_mbon(con), kc_mbon(con_masked)
    mask_touches_kc_mbon = len(e_all) != len(e_masked)
    e = e_masked

    w_syn_mV = float(dp["w_syn"] / (0.001 * 1.0))
    t_mbr_ms = float(dp["t_mbr"] / (0.001 * 1.0))
    tau_ms = float(dp["tau"] / (0.001 * 1.0))
    v_th_mV = float(dp["v_th"] / (0.001 * 1.0))
    v_0_mV = float(dp["v_0"] / (0.001 * 1.0))
    theta_mV = v_th_mV - v_0_mV

    a_peak, t_star_ms = epsp_peak_factor(t_mbr_ms, tau_ms)

    # вес ребра при s = 1, в мВ приращения переменной g
    e = e.assign(w_mV=e["Excitatory x Connectivity"].to_numpy() * w_syn_mV)

    mbon_ids = sorted(i for i, r in role.items() if r == "MBON")
    per_mbon = {}
    for m in mbon_ids:
        sub = e[e.Postsynaptic_ID == m]
        w = sub["w_mV"].to_numpy(dtype=float)
        per_mbon[str(m)] = {
            "hemibrain_type": str(typ[m]),
            "n_edges_kc": int(len(sub)),
            "n_synapses_kc": int(sub["Connectivity"].abs().sum()) if len(sub) else 0,
            "sum_w_mV": float(w.sum()) if len(w) else 0.0,
            "w_max_mV": float(w.max()) if len(w) else 0.0,
            "w_q10_mV": float(np.quantile(w, 0.10)) if len(w) else None,
            "w_q50_mV": float(np.quantile(w, 0.50)) if len(w) else None,
            "w_q90_mV": float(np.quantile(w, 0.90)) if len(w) else None,
        }

    def aggregate(ids: list[int]) -> dict:
        w = e[e.Postsynaptic_ID.isin(ids)]["w_mV"].to_numpy(dtype=float)
        sums = [per_mbon[str(i)]["sum_w_mV"] for i in ids]
        maxs = [per_mbon[str(i)]["w_max_mV"] for i in ids]
        return {"n_cells": len(ids),
                "n_edges_kc": int(sum(per_mbon[str(i)]["n_edges_kc"] for i in ids)),
                "n_synapses_kc": int(sum(per_mbon[str(i)]["n_synapses_kc"]
                                         for i in ids)),
                "sum_w_mV_total": float(np.sum(sums)),
                "sum_w_mV_per_cell": [float(x) for x in sums],
                "sum_w_mV_min_cell": float(np.min(sums)) if sums else 0.0,
                "w_max_mV": float(np.max(maxs)) if maxs else 0.0,
                "w_q10_mV": float(np.quantile(w, 0.10)) if len(w) else None,
                "w_q50_mV": float(np.quantile(w, 0.50)) if len(w) else None,
                "w_q90_mV": float(np.quantile(w, 0.90)) if len(w) else None}

    by_type = {}
    for t in V.T38:
        ids = [m for m in mbon_ids if typ[m] == t]
        by_type[t] = aggregate(ids)
    all_mbon = aggregate(mbon_ids)

    # V1c-E5: тип, у всех клеток которого сумма весов нулевая, делает ступень
    # непроверяемой по построению - никакое s не изменит нуля.
    untestable = [t for t, d in by_type.items()
                  if d["n_cells"] == 0 or d["sum_w_mV_total"] == 0.0]

    # V1c-E3.1: верхняя граница сетки по s. Максимум берётся по клеткам шести
    # типов T38; величина по всем MBON печатается отчётно и границы не задаёт.
    w_max_t38 = max(d["w_max_mV"] for d in by_type.values())
    s_max_t38 = theta_mV / (a_peak * w_max_t38) if w_max_t38 > 0 else None
    s_max_all = (theta_mV / (a_peak * all_mbon["w_max_mV"])
                 if all_mbon["w_max_mV"] > 0 else None)

    # V1c-E3.2: узлы сетки. Порождаются формулой штампа 0, не выбором.
    n_dec = 4
    nodes = []
    if s_max_t38:
        k = 0
        while 10.0 ** (k / n_dec) < s_max_t38:
            nodes.append(10.0 ** (k / n_dec))
            k += 1
        nodes.append(float(s_max_t38))

    out = {
        "measurement": "V1c-E6.1",
        "permitted_by": "правило Д, штамп 0 записан до измерения",
        "stamp0": STAMP0.name,
        "simulator_run": False,
        "substrate": {"n_mbon": len(mbon_ids),
                      "n_edges_kc_mbon_before_mask": int(len(e_all)),
                      "n_edges_kc_mbon_after_mask": int(len(e_masked)),
                      "dan_mask_removed_edges": int(n_masked),
                      "dan_mask_touches_kc_mbon": bool(mask_touches_kc_mbon)},
        "model_constants": {"w_syn_mV": w_syn_mV, "t_mbr_ms": t_mbr_ms,
                            "tau_ms": tau_ms, "v_th_mV": v_th_mV,
                            "v_0_mV": v_0_mV, "theta_mV": theta_mV},
        "epsp": {"A": a_peak, "t_peak_ms": t_star_ms,
                 "note": "пиковое отклонение мембраны от одиночного спайка "
                         "весом w равно A*w; модель токовая"},
        "by_type_T38": by_type,
        "all_mbon": all_mbon,
        "per_mbon": per_mbon,
        "testability_V1c_E5": {"types_with_zero_total_weight": untestable,
                               "stage_testable": len(untestable) == 0},
        "s_max": {"w_max_mV_T38": w_max_t38, "s_max_T38": s_max_t38,
                  "w_max_mV_all_mbon": all_mbon["w_max_mV"],
                  "s_max_all_mbon_reportonly": s_max_all,
                  "formula": "(v_th - v_0) / (A * max_m w_max(m))"},
        "grid_s": {"n_per_decade": n_dec, "n_nodes": len(nodes),
                   "nodes": nodes},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "weights_kc_mbon.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Измерение V1c-E6.1: структурные веса KC->MBON (симулятор не запускался)")
    print("рёбер KC->MBON: %d; маска подмен их не касается: %s"
          % (len(e), "да" if not mask_touches_kc_mbon else "НЕТ, разбирать"))
    print("\n%-8s %6s %7s %9s %11s %11s %9s"
          % ("тип", "клеток", "рёбер", "синапсов", "Σw, мВ", "мин Σw/кл", "w_max, мВ"))
    for t, d in by_type.items():
        print("%-8s %6d %7d %9d %11.1f %11.1f %9.4f"
              % (t, d["n_cells"], d["n_edges_kc"], d["n_synapses_kc"],
                 d["sum_w_mV_total"], d["sum_w_mV_min_cell"], d["w_max_mV"]))
    print("%-8s %6d %7d %9d %11.1f %11.1f %9.4f"
          % ("все MBON", all_mbon["n_cells"], all_mbon["n_edges_kc"],
             all_mbon["n_synapses_kc"], all_mbon["sum_w_mV_total"],
             all_mbon["sum_w_mV_min_cell"], all_mbon["w_max_mV"]))

    print("\nОдиночный ВПСП: A = %.6f, пик в %.3f мс, порог θ = %.1f мВ"
          % (a_peak, t_star_ms, theta_mV))
    print("s_max по T38 = θ / (A · w_max) = %.1f / (%.6f · %.4f) = %.2f"
          % (theta_mV, a_peak, w_max_t38, s_max_t38))
    print("отчётно, по всем MBON: w_max = %.4f мВ, s_max = %.2f"
          % (all_mbon["w_max_mV"], s_max_all))
    print("\nсетка по s: %d узлов, %d на декаду, от %.4g до %.4g"
          % (len(nodes), n_dec, nodes[0], nodes[-1]))
    print("  " + ", ".join("%.4g" % x for x in nodes))
    print("\nтестируемость V1c-E5: %s"
          % ("все шесть типов T38 имеют ненулевой вход KC - ступень проверяема"
             if not untestable else
             "НЕПРОВЕРЯЕМА по построению для типов: " + ", ".join(untestable)))
    print("\nзаписано: %s" % p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
