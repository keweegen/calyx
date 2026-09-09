# -*- coding: utf-8 -*-
"""Таблица детекторов ступени V1c: некритериальный выход и проверка согласованности.

Что это. На каждом узле сетки по `s` — список клеток MBON, для которых
одиночный спайк одной клетки Кеньона доводит клетку до порога:

    клетка m есть детектор на узле s_k   <=>   s_k · A · w_max(m) > θ,

неравенство строгое; `A` — множитель пикового отклонения одиночного ВПСП,
`θ = v_th − v_0`, `w_max(m)` — вес самого сильного одиночного ребра KC→m при
`s = 1`, взятый по клетке, а не по типу.

Статус величины. Это арифметика над наблюдаемыми, записанными измерением
V1c-E6.1 (`w_max(m)` на каждый из 97 MBON) и над константами модели.
По правилу Е4 такая величина есть производная и критерием быть не может;
таблица объявлена некритериальным выходом ступени. Симулятор не запускается,
ни один параметр не меняется, новая ось не вводится — правило Д2 соблюдено
тождественно.

Канонические значения узлов — десятичные строки спецификации, а не результат
перевычисления формул во время исполнения. Граничные узлы `s_pop` и `s_max`
округлены вниз, поэтому задающие их клетки на своих граничных узлах
детекторами не являются и строгость неравенства не зависит от порядка
арифметики.

Запуск:
    python v1c_detectors.py                 # построить таблицу
    python v1c_detectors.py --check         # сверить с таблицей спецификации
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SUB = HERE / "data" / "mb_subcircuit"
OUT = HERE / "results" / "v1c"
WEIGHTS = OUT / "weights_kc_mbon.json"

T38 = ["MBON11", "MBON12", "MBON13", "MBON14", "MBON17", "MBON18"]

# Канонические узлы сетки: строки спецификации, раздел 3з, V1c-E3.2.
# 1 и 10^(k/4) - по формуле, с округлением по ближайшему; s_pop и s_max -
# по формуле V1c-E3.1, с округлением вниз.
NODES = ("1", "1.6836", "1.7783", "3.1623", "5.3875")
NODE_KIND = {"1": "s_min, формула E3.1",
             "1.6836": "s_pop, формула E3.1 по всем MBON, округление вниз",
             "1.7783": "10^(1/4), формула E3.2",
             "3.1623": "10^(2/4), формула E3.2",
             "5.3875": "s_max, формула E3.1 по T38, округление вниз"}


def build() -> dict:
    w = json.loads(WEIGHTS.read_text(encoding="utf-8"))
    a_peak = float(w["epsp"]["A"])
    theta = float(w["model_constants"]["theta_mV"])

    neurons = pd.read_csv(SUB / "neurons.csv")
    side = dict(zip(neurons.root_id, neurons.side.astype("string")))
    comp = dict(zip(neurons.root_id, neurons.compartment.astype("string")))

    # число синапсов самого сильного ребра: читается из той же таблицы связей,
    # из которой измерение E6.1 взяло w_max
    con = pd.read_parquet(SUB / "connectivity.parquet")
    role = dict(zip(neurons.root_id, neurons.mb_role))
    pre = con.Presynaptic_ID.map(role)
    post = con.Postsynaptic_ID.map(role)
    e = con[(pre == "Kenyon_Cell") & (post == "MBON")]
    w_syn = float(w["model_constants"]["w_syn_mV"])
    e = e.assign(w_mV=e["Excitatory x Connectivity"].to_numpy() * w_syn)
    top = (e.sort_values("w_mV", ascending=False)
             .drop_duplicates("Postsynaptic_ID")
             .set_index("Postsynaptic_ID"))

    cells = []
    for mid, d in w["per_mbon"].items():
        m = int(mid)
        wm = float(d["w_max_mV"])
        row = {"mbon_id": m, "hemibrain_type": d["hemibrain_type"],
               "side": str(side.get(m)), "compartment": str(comp.get(m)),
               "in_T38": d["hemibrain_type"] in T38,
               "n_edges_kc": d["n_edges_kc"], "w_max_mV": wm,
               "no_kc_input": d["n_edges_kc"] == 0}
        row["n_synapses_max_edge"] = (
            int(abs(top.loc[m, "Connectivity"])) if m in top.index else 0)
        for s in NODES:
            row["detector_at_" + s] = bool(float(s) * a_peak * wm > theta)
        first = [s for s in NODES if row["detector_at_" + s]]
        row["first_detector_node"] = first[0] if first else None
        cells.append(row)
    cells.sort(key=lambda r: -r["w_max_mV"])

    by_node = {}
    for s in NODES:
        det = [c for c in cells if c["detector_at_" + s]]
        by_node[s] = {
            "node_kind": NODE_KIND[s],
            "n_detectors_T38": sum(1 for c in det if c["in_T38"]),
            "n_detectors_outside_T38": sum(1 for c in det if not c["in_T38"]),
            "types_outside_T38": sorted({c["hemibrain_type"] for c in det
                                         if not c["in_T38"]}),
            "detectors": [{"mbon_id": c["mbon_id"],
                           "hemibrain_type": c["hemibrain_type"],
                           "side": c["side"], "compartment": c["compartment"],
                           "in_T38": c["in_T38"],
                           "w_max_mV": c["w_max_mV"],
                           "n_synapses_max_edge": c["n_synapses_max_edge"],
                           "ratio_to_theta": round(
                               float(s) * a_peak * c["w_max_mV"] / theta, 6)}
                          for c in det]}

    return {"output": "таблица детекторов ступени V1c",
            "status": "некритериальный выход; производная величина по правилу "
                      "Е4, критерием быть не может",
            "source": "results/v1c/weights_kc_mbon.json (измерение V1c-E6.1)",
            "definition": "детектор <=> s_k · A · w_max(m) > θ, строго",
            "A": a_peak, "theta_mV": theta,
            "nodes": list(NODES),
            "n_cells_total": len(cells),
            "n_cells_without_kc_input": sum(1 for c in cells
                                            if c["no_kc_input"]),
            "condition5_holds_on_all_nodes": all(
                by_node[s]["n_detectors_T38"] == 0 for s in NODES),
            "by_node": by_node,
            "cells": cells}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="сверить с записанной таблицей вместо перезаписи")
    a = ap.parse_args()

    t = build()
    p = OUT / "detector_table.json"

    if a.check:
        if not p.exists():
            raise SystemExit("таблица не записана: %s" % p)
        old = json.loads(p.read_text(encoding="utf-8"))
        diff = [k for k in ("by_node", "cells", "A", "theta_mV", "nodes")
                if old.get(k) != t[k]]
        print("сверка таблицы детекторов: %s"
              % ("совпадает" if not diff
                 else "РАСХОЖДЕНИЕ в полях " + ", ".join(diff)))
        return 0 if not diff else 1

    OUT.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(t, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Таблица детекторов ступени V1c (некритериальный выход)")
    print("A = %.6f, θ = %.1f мВ, клеток %d, из них без входа KC %d"
          % (t["A"], t["theta_mV"], t["n_cells_total"],
             t["n_cells_without_kc_input"]))
    print("\n%-9s %8s %10s   %s" % ("узел s", "в T38", "вне T38",
                                    "типы вне T38"))
    for s in NODES:
        d = t["by_node"][s]
        print("%-9s %8d %10d   %s"
              % (s, d["n_detectors_T38"], d["n_detectors_outside_T38"],
                 ", ".join(d["types_outside_T38"]) or "-"))
    print("\nусловие (5) на всех узлах: %s"
          % ("выполнено - в T38 детекторов нет"
             if t["condition5_holds_on_all_nodes"] else "НАРУШЕНО"))
    print("\nдетекторы верхнего узла %s:" % NODES[-1])
    for c in t["by_node"][NODES[-1]]["detectors"]:
        print("  %d  %-8s %-6s %-5s w_max %7.4f мВ, синапсов %3d, s·A·w/θ = %.4f"
              % (c["mbon_id"], c["hemibrain_type"], c["side"],
                 c["compartment"], c["w_max_mV"], c["n_synapses_max_edge"],
                 c["ratio_to_theta"]))
    print("\nзаписано: %s" % p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
