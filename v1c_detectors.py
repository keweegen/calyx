# -*- coding: utf-8 -*-
"""Detector table of the V1c stage: non-criterion output and consistency check.

What this is. At every grid node over `s` — a list of MBON cells for which a
single spike of one Kenyon cell drives the cell to threshold:

    cell m is a detector at node s_k   <=>   s_k · A · w_max(m) > θ,

the inequality is strict; `A` is the peak-deflection factor of a single
EPSP, `θ = v_th − v_0`, `w_max(m)` is the weight of the strongest single
KC→m edge at `s = 1`, taken per cell, not per type.

Status of the value. This is arithmetic over observables recorded by
measurement V1c-E6.1 (`w_max(m)` for each of the 97 MBON) and over the model
constants. Under rule Е4 such a value is derived and cannot be a criterion;
the table is declared a non-criterion output of the stage. The simulator is
not run, no parameter is changed, no new axis is introduced — rule Д2 is
satisfied identically.

The canonical node values are the decimal strings of the spec, not the
result of recomputing formulas at run time. The boundary nodes `s_pop` and
`s_max` are rounded down, so the cells that define them are not detectors at
their own boundary nodes, and the strictness of the inequality does not
depend on arithmetic order.

Run:
    python v1c_detectors.py                 # build the table
    python v1c_detectors.py --check         # check against the recorded table
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

# Canonical grid nodes: spec strings, section 3з, V1c-E3.2.
# 1 and 10^(k/4) - by formula, rounded to nearest; s_pop and s_max - by
# formula V1c-E3.1, rounded down.
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

    # number of synapses of the strongest edge: read from the same
    # connectivity table from which measurement E6.1 took w_max
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
                    help="verify against the recorded table instead of overwriting it")
    a = ap.parse_args()

    t = build()
    p = OUT / "detector_table.json"

    if a.check:
        if not p.exists():
            raise SystemExit("table not recorded: %s" % p)
        old = json.loads(p.read_text(encoding="utf-8"))
        diff = [k for k in ("by_node", "cells", "A", "theta_mV", "nodes")
                if old.get(k) != t[k]]
        print("detector table check: %s"
              % ("matches" if not diff
                 else "DISCREPANCY in fields " + ", ".join(diff)))
        return 0 if not diff else 1

    OUT.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(t, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Detector table of the V1c stage (non-criterion output)")
    print("A = %.6f, θ = %.1f mV, cells %d, of which without KC input %d"
          % (t["A"], t["theta_mV"], t["n_cells_total"],
             t["n_cells_without_kc_input"]))
    print("\n%-9s %8s %10s   %s" % ("node s", "in T38", "outside T38",
                                    "types outside T38"))
    for s in NODES:
        d = t["by_node"][s]
        print("%-9s %8d %10d   %s"
              % (s, d["n_detectors_T38"], d["n_detectors_outside_T38"],
                 ", ".join(d["types_outside_T38"]) or "-"))
    print("\ncondition (5) on all nodes: %s"
          % ("holds - no detectors in T38"
             if t["condition5_holds_on_all_nodes"] else "VIOLATED"))
    print("\ndetectors at the top node %s:" % NODES[-1])
    for c in t["by_node"][NODES[-1]]["detectors"]:
        print("  %d  %-8s %-6s %-5s w_max %7.4f mV, synapses %3d, s·A·w/θ = %.4f"
              % (c["mbon_id"], c["hemibrain_type"], c["side"],
                 c["compartment"], c["w_max_mV"], c["n_synapses_max_edge"],
                 c["ratio_to_theta"]))
    print("\nwritten: %s" % p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
