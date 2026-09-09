# -*- coding: utf-8 -*-
"""Почему молчат KCα′β′: баланс возбуждения и торможения по подтипам KC.

Что это. Диагностика ПОСЛЕ закрытия ступени V1c. Ступень закрыта исходом
FAIL-CAL-MBON-FLOOR, и его определили два типа MBON — MBON13 (α′2) и MBON17
(α′3m), — которые получают 100 % входа от клеток Кеньона подтипа α′β′, а те в
этом семействе почти не отвечают. Отчёт ступени называет это фактом; здесь
измеряется его структурная причина.

Статус измерения. Симулятор не запускается. Читается замороженная таблица
связей тем же загрузчиком и с той же маской подмен, какими пользуется ступень,
и веса вычисляются той же формулой `Excitatory x Connectivity` × `w_syn`.
Никакой параметр не меняется, никакая переменная состояния не регистрируется.
Ступень V1c закрыта, следующая не заведена, поэтому правило Д ограничений не
накладывает; ограничение Д2 выполнено тождественно.

Что вычисляется, объявлено до запуска:
  - число клеток каждого подтипа KC в подсхеме;
  - рёбра и суммарный вес входа PN→KC на клетку, отдельно по всем PN и по
    унигломерулярным (запах подаётся только на них);
  - доля клеток подтипа вовсе без унигломерулярного входа;
  - рёбра и суммарный вес торможения APL→KC на клетку при g_apl = 1;
  - отношение торможения к возбуждению на клетку при g_apl = g_ref.

Чего измерение НЕ говорит. Оно не говорит, сколько входов приходит
одновременно: до порога доводит совпадение, а не сумма. Оно не говорит, что
модель неверна: чтобы это утверждать, нужны данные о живой мухе — отвечают ли
α′β′ и компенсируют ли они меньший вход большей возбудимостью. Модель [2]
задаёт ОДИН набор параметров нейрона на все клетки, поэтому всякое различие
подтипов по внутренней возбудимости в ней отсутствует по построению; следствие
этого измерения — величина, которую такая компенсация должна была бы иметь.

Запуск:  python v1c_kc_subtype_diag.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "results" / "v1c"

SUBTYPES = ("KCab", "KCa'b'", "KCg")


def kc_subtype(t: str) -> str | None:
    for p in SUBTYPES:
        if t.startswith(p):
            return p
    return None


def main() -> int:
    import v1b_subcircuit as V
    from model import default_params as dp

    w_syn = float(dp["w_syn"] / (0.001 * 1.0))
    g_ref = V.g_ref_value()
    neurons, con = V.load_substrate()
    role = dict(zip(neurons.root_id, neurons.mb_role))
    typ = dict(zip(neurons.root_id, neurons.hemibrain_type.astype(str)))
    cls = dict(zip(neurons.root_id, neurons.cell_sub_class.astype(str)))
    kept, n_masked = V.apply_dan_mask(con, role)

    kc = {i: kc_subtype(typ.get(i, "")) for i, r in role.items()
          if r == "Kenyon_Cell"}
    kc = {i: v for i, v in kc.items() if v}
    n_by = {s: sum(1 for v in kc.values() if v == s) for s in SUBTYPES}

    pre = kept.Presynaptic_ID.map(role)
    post = kept.Postsynaptic_ID.map(role)

    e = kept[(pre == "PN") & (post == "Kenyon_Cell")].copy()
    e["w"] = e["Excitatory x Connectivity"].to_numpy() * w_syn
    e["sub"] = e.Postsynaptic_ID.map(kc)
    e["pn"] = e.Presynaptic_ID.map(cls)
    e = e[e["sub"].notna()]
    uni = e[e.pn == "uniglomerular"]

    a = kept[(pre == "APL") & (post == "Kenyon_Cell")].copy()
    a["w"] = np.abs(a["Connectivity"].to_numpy()) * w_syn
    a["sub"] = a.Postsynaptic_ID.map(kc)
    a = a[a["sub"].notna()]

    with_uni = set(uni.Postsynaptic_ID)
    rows = {}
    for s in SUBTYPES:
        n = n_by[s]
        ge, gu, ga = e[e["sub"] == s], uni[uni["sub"] == s], a[a["sub"] == s]
        ids = [i for i, v in kc.items() if v == s]
        ex = float(gu.w.sum()) / n
        inh = float(ga.w.sum()) / n * g_ref
        rows[s] = {
            "n_cells": n,
            "pn_edges": int(len(ge)), "pn_edges_per_cell": len(ge) / n,
            "pn_sum_mV": float(ge.w.sum()), "pn_mV_per_cell": float(ge.w.sum()) / n,
            "uni_edges": int(len(gu)), "uni_edges_per_cell": len(gu) / n,
            "uni_sum_mV": float(gu.w.sum()), "uni_mV_per_cell": ex,
            "uni_share_of_pn_weight": float(gu.w.sum()) / float(ge.w.sum()),
            "cells_without_uni_input": sum(1 for i in ids if i not in with_uni),
            "apl_edges": int(len(ga)), "apl_edges_per_cell": len(ga) / n,
            "apl_sum_mV_at_g1": float(ga.w.sum()),
            "apl_mV_per_cell_at_gref": inh,
            "inhibition_over_excitation": inh / ex,
        }

    ratios = {s: rows[s]["inhibition_over_excitation"] for s in SUBTYPES}
    worst = max(ratios, key=ratios.get)
    best = min(ratios, key=ratios.get)
    doc = {
        "measurement": "баланс возбуждения и торможения по подтипам клеток Кеньона",
        "status": "диагностика после закрытия ступени V1c; симулятор не запускался",
        "why": "исход V1c определили два типа MBON, получающие 100 % входа от "
               "KCα′β′; здесь измеряется структурная причина молчания α′β′",
        "source": "замороженная таблица связей подсхемы, тот же загрузчик и та же "
                  "маска подмен, что у ступени",
        "n_edges_masked": int(n_masked),
        "w_syn_mV": w_syn, "g_ref": g_ref,
        "n_kc_labelled": len(kc),
        "by_subtype": rows,
        "headline": {
            "uni_mV_per_cell_ratio_best_over_worst":
                rows[best]["uni_mV_per_cell"] / rows[worst]["uni_mV_per_cell"],
            "inhibition_over_excitation_worst": worst,
            "inhibition_over_excitation_ratio_worst_over_best":
                ratios[worst] / ratios[best],
        },
        "not_measured": [
            "число одновременно активных входов: до порога доводит совпадение, "
            "а не сумма",
            "внутренняя возбудимость подтипов: модель [2] задаёт один набор "
            "параметров нейрона на все клетки, различия в ней отсутствуют по "
            "построению",
            "соответствие живой мухе: требует данных о доле отвечающих KC по "
            "подтипам in vivo",
        ],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "kc_subtype_balance.json"
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Баланс возбуждения и торможения по подтипам клеток Кеньона")
    print("=" * 78)
    print("%-9s %7s %11s %11s %11s %11s" % ("подтип", "клеток", "уни мВ/кл",
                                            "APL мВ/кл", "торм/возб", "без уни"))
    for s in SUBTYPES:
        r = rows[s]
        print("%-9s %7d %11.2f %11.4f %11.4f %7d (%.1f%%)"
              % (s, r["n_cells"], r["uni_mV_per_cell"],
                 r["apl_mV_per_cell_at_gref"], r["inhibition_over_excitation"],
                 r["cells_without_uni_input"],
                 100.0 * r["cells_without_uni_input"] / r["n_cells"]))
    print("=" * 78)
    print("Возбуждающий вход на клетку у %s меньше, чем у %s, в %.2f раза."
          % (worst, best, doc["headline"]["uni_mV_per_cell_ratio_best_over_worst"]))
    print("Отношение торможения к возбуждению у %s хуже, чем у %s, в %.2f раза."
          % (worst, best,
             doc["headline"]["inhibition_over_excitation_ratio_worst_over_best"]))
    print("Порог, постоянная времени мембраны и сопротивление в модели [2] у всех "
          "клеток одни и те же,")
    print("поэтому этот дисбаланс ничем не компенсируется — по построению модели, "
          "а не по данным.")
    print("\nзаписано: %s" % p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
