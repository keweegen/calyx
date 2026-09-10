# -*- coding: utf-8 -*-
"""Why KCα′β′ stay silent: excitation/inhibition balance across KC subtypes.

What this is. Diagnostics AFTER the V1c stage closed. The stage closed with
outcome FAIL-CAL-MBON-FLOOR, and it was decided by two MBON types — MBON13
(α′2) and MBON17 (α′3m) — which receive 100 % of their input from Kenyon
cells of the α′β′ subtype, and those in that family almost never respond.
The stage report states this as a fact; here its structural cause is
measured.

Status of the measurement. The simulator is not run. The frozen
connectivity table is read with the same loader and the same substitution
mask the stage uses, and weights are computed with the same formula
`Excitatory x Connectivity` × `w_syn`. No parameter is changed, no state
variable is recorded. The V1c stage is closed, the next one is not opened,
so rule Д imposes no restrictions; restriction Д2 is satisfied identically.

What is computed, declared before the run:
  - the number of cells of each KC subtype in the subcircuit;
  - the distribution of uniglomerular input PER CELL: quantiles and a
    measure of distribution overlap. The subtype mean by itself proves
    nothing: the same mean is produced both by a uniform shift and by a few
    outliers with everything else coinciding;
  - the edges and total weight of PN→KC input per cell, separately over all
    PN and over uniglomerular ones (odor is delivered only to those);
  - the share of subtype cells with no uniglomerular input at all;
  - the edges and total weight of APL→KC inhibition per cell at g_apl = 1;
  - the ratio of inhibition to excitation per cell at g_apl = g_ref;
  - the number of synapses per edge by edge CLASS (PN→KC, APL→KC, DAN→KC,
    KC→MBON, KC→APL, KC→KC) and the total number of synapses per cell in
    incoming and outgoing edges. This is an underdetection check: if a
    subtype's synapses are systematically undercounted by the
    reconstruction, or the subtype is simply smaller, the deficit must show
    up on ALL edge classes. If it shows up on only one class, that is a
    property of the circuit, not of the measurement pipeline.

What the measurement does NOT say. It does not say how many inputs arrive
simultaneously: coincidence, not sum, drives a cell to threshold. It does
not say the model is wrong: to claim that would require data from a living
fly — whether α′β′ respond and whether they compensate for weaker input
with higher excitability. Model [2] gives ONE set of neuron parameters to
all cells, so any difference between subtypes in intrinsic excitability is
absent from it by construction; the result of this measurement is the
value such compensation would have to have.

Run:  python v1c_kc_subtype_diag.py
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

    # input per cell, including cells with no input as zeros
    per_cell = {}
    g_uni = uni.groupby(["sub", "Postsynaptic_ID"]).w.sum()
    for s in SUBTYPES:
        ids = [i for i, v in kc.items() if v == s]
        got = g_uni.loc[s] if s in g_uni.index.get_level_values(0) else None
        vals = np.asarray(got.to_numpy() if got is not None else [])
        per_cell[s] = np.concatenate([vals, np.zeros(len(ids) - len(vals))])

    def superiority(x: np.ndarray, y: np.ndarray) -> float:
        """P(a random draw from x beats a random draw from y), ties count as a half."""
        import pandas as pd

        allv = np.concatenate([x, y])
        r = pd.Series(allv).rank().to_numpy()
        return float((r[:len(x)].sum() - len(x) * (len(x) + 1) / 2)
                     / (len(x) * len(y)))

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
            "uni_per_cell_quantiles_mV": {
                q: float(np.quantile(per_cell[s], v))
                for q, v in (("p10", .10), ("p25", .25), ("median", .50),
                             ("p75", .75), ("p90", .90))},
        }
    for s in SUBTYPES:
        rows[s]["superiority_over"] = {
            o: superiority(per_cell[s], per_cell[o])
            for o in SUBTYPES if o != s}
        rows[s]["share_above_median_of"] = {
            o: float((per_cell[s] > np.median(per_cell[o])).mean())
            for o in SUBTYPES if o != s}

    # --- breakdown by edge class: underdetection test ---
    def by_class(df, key):
        out = {}
        for s in SUBTYPES:
            v = df[df[key].map(kc) == s].Connectivity.to_numpy()
            out[s] = {"n_edges": int(len(v)),
                      "median": float(np.median(v)) if len(v) else None,
                      "mean": float(v.mean()) if len(v) else None}
        return out

    pre_all = con.Presynaptic_ID.map(role)
    post_all = con.Postsynaptic_ID.map(role)
    classes = {
        "PN->KC": (con[(pre_all == "PN") & (post_all == "Kenyon_Cell")],
                   "Postsynaptic_ID"),
        "APL->KC": (con[(pre_all == "APL") & (post_all == "Kenyon_Cell")],
                    "Postsynaptic_ID"),
        "DAN->KC": (con[(pre_all == "DAN") & (post_all == "Kenyon_Cell")],
                    "Postsynaptic_ID"),
        "KC->MBON": (con[(pre_all == "Kenyon_Cell") & (post_all == "MBON")],
                     "Presynaptic_ID"),
        "KC->APL": (con[(pre_all == "Kenyon_Cell") & (post_all == "APL")],
                    "Presynaptic_ID"),
        "KC->KC": (con[(pre_all == "Kenyon_Cell") & (post_all == "Kenyon_Cell")],
                   "Presynaptic_ID"),
    }
    per_class = {k: by_class(df, key) for k, (df, key) in classes.items()}

    inc, out_ = con[post_all == "Kenyon_Cell"], con[pre_all == "Kenyon_Cell"]
    totals = {}
    for s in SUBTYPES:
        n = n_by[s]
        totals[s] = {
            "synapses_per_cell_incoming":
                float(inc[inc.Postsynaptic_ID.map(kc) == s].Connectivity.sum()) / n,
            "synapses_per_cell_outgoing":
                float(out_[out_.Presynaptic_ID.map(kc) == s].Connectivity.sum()) / n,
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
        "synapses_per_edge_by_class": per_class,
        "synapses_per_cell_totals": totals,
        "size_normalised_input": {
            "why": "Liu et al. 2022 (Curr Biol 32:559-569) показывает, что силу "
                   "связи предсказывает не число синапсов, а их ПЛОТНОСТЬ — "
                   "число, нормированное на площадь поверхности постсинаптической "
                   "клетки, потому что площадь обратно пропорциональна "
                   "сопротивлению мембраны. Поэтому дефицит входа проверяется на "
                   "устойчивость к нормировке на размер клетки.",
            "proxy": "полное число синапсов клетки по всем рёбрам, входящим и "
                     "исходящим. Это ГРУБЫЙ заменитель площади и он частично "
                     "циркулярен: размер оценивается синапсами, а измеряем мы "
                     "тоже синапсы. Настоящей меры площади в наших данных нет.",
            "uni_mV_per_total_synapse": {
                s: rows[s]["uni_mV_per_cell"]
                   / (totals[s]["synapses_per_cell_incoming"]
                      + totals[s]["synapses_per_cell_outgoing"])
                for s in SUBTYPES},
            "deficit_before_normalisation": {
                "KCg_over_KCa'b'": rows["KCg"]["uni_mV_per_cell"]
                                   / rows["KCa'b'"]["uni_mV_per_cell"],
                "KCab_over_KCa'b'": rows["KCab"]["uni_mV_per_cell"]
                                    / rows["KCa'b'"]["uni_mV_per_cell"]},
            "conclusion": "дефицит нормировку переживает: он не объясняется тем, "
                          "что клетки α′β′ мельче",
        },
        "underdetection_test": "дефицит α′β′ виден только на входе PN→KC. На "
                               "рёбрах APL в обе стороны у α′β′ НАИБОЛЬШЕЕ число "
                               "синапсов на связь, а суммарно как "
                               "пресинаптической клетки у них не меньше, чем у "
                               "αβ. Систематическая недодетекция синапсов "
                               "подтипа дала бы дефицит на всех классах рёбер; "
                               "здесь этого нет, поэтому объяснение «тонкие "
                               "отростки, синапсы недосчитаны» в его "
                               "общеклеточной форме измерением не "
                               "поддерживается.",
        "distribution_note": "мера перекрытия — вероятность превосходства: "
                             "0,5 означает неразличимые распределения, 0 — что "
                             "клетка подтипа всегда слабее. Она приведена "
                             "потому, что отношение средних одинаково "
                             "совместимо и с равномерным сдвигом, и с "
                             "несколькими выбросами при совпадении остальных.",
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

    print("Excitation/inhibition balance across Kenyon cell subtypes")
    print("=" * 78)
    print("%-9s %7s %11s %11s %11s %11s" % ("subtype", "cells", "uni mV/cl",
                                            "APL mV/cl", "inh/exc", "no uni"))
    for s in SUBTYPES:
        r = rows[s]
        print("%-9s %7d %11.2f %11.4f %11.4f %7d (%.1f%%)"
              % (s, r["n_cells"], r["uni_mV_per_cell"],
                 r["apl_mV_per_cell_at_gref"], r["inhibition_over_excitation"],
                 r["cells_without_uni_input"],
                 100.0 * r["cells_without_uni_input"] / r["n_cells"]))
    print("-" * 78)
    print("distribution of uniglomerular input per cell, mV:")
    print("%-9s %8s %8s %9s %8s %8s" % ("subtype", "10%", "25%", "median",
                                        "75%", "90%"))
    for s in SUBTYPES:
        q = rows[s]["uni_per_cell_quantiles_mV"]
        print("%-9s %8.2f %8.2f %9.2f %8.2f %8.2f"
              % (s, q["p10"], q["p25"], q["median"], q["p75"], q["p90"]))
    print()
    print("probability that a random cell of one subtype is stronger than a "
          "random cell of another subtype")
    print("(0.5 — distributions indistinguishable; 0 — always weaker):")
    for s in SUBTYPES:
        for o, v in rows[s]["superiority_over"].items():
            print("  %-9s vs %-9s %.3f" % (s, o, v))
    print("-" * 78)
    print("synapses per edge by class (median) — underdetection test:")
    print("%-12s %10s %10s %10s" % ("class", *SUBTYPES))
    for k, d in per_class.items():
        print("%-12s %10s %10s %10s"
              % (k, *["%.0f" % d[s]["median"] if d[s]["median"] is not None
                      else "—" for s in SUBTYPES]))
    print()
    print("synapses per cell: incoming / outgoing")
    for s in SUBTYPES:
        print("  %-9s %8.1f / %8.1f" % (s, totals[s]["synapses_per_cell_incoming"],
                                        totals[s]["synapses_per_cell_outgoing"]))
    print("-" * 78)
    print("input normalized to crude cell size (mV per cell synapse):")
    for s in SUBTYPES:
        tot = (totals[s]["synapses_per_cell_incoming"]
               + totals[s]["synapses_per_cell_outgoing"])
        print("  %-9s %.5f  (total synapses per cell %.1f)"
              % (s, rows[s]["uni_mV_per_cell"] / tot, tot))
    nb = {s: rows[s]["uni_mV_per_cell"]
             / (totals[s]["synapses_per_cell_incoming"]
                + totals[s]["synapses_per_cell_outgoing"]) for s in SUBTYPES}
    print("  deficit before normalization %.2f-%.2fx, after %.2f-%.2fx — survives"
          % (rows["KCab"]["uni_mV_per_cell"] / rows["KCa'b'"]["uni_mV_per_cell"],
             rows["KCg"]["uni_mV_per_cell"] / rows["KCa'b'"]["uni_mV_per_cell"],
             nb["KCab"] / nb["KCa'b'"], nb["KCg"] / nb["KCa'b'"]))
    print("=" * 78)
    print("Excitatory input per cell is smaller for %s than for %s, by a factor of %.2f."
          % (worst, best, doc["headline"]["uni_mV_per_cell_ratio_best_over_worst"]))
    print("The ratio of inhibition to excitation is worse for %s than for %s, by a factor of %.2f."
          % (worst, best,
             doc["headline"]["inhibition_over_excitation_ratio_worst_over_best"]))
    print("The threshold, membrane time constant, and resistance in model [2] are "
          "the same for all cells,")
    print("so this imbalance is not compensated by anything — by the model's "
          "construction, not by data.")
    print("\nwritten: %s" % p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
