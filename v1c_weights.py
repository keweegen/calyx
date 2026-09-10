# -*- coding: utf-8 -*-
"""Measurement V1c-E6.1: structural KC->MBON weights in the subcircuit.

Permitted by rule Д (spec, section 1a) and listed in stamp 0 of the V1c
stage (section 3з, V1c-E6.1). Stamp 0 was recorded BEFORE this measurement:
results/v1c/stamp0_sha256.txt. The simulator is not run, no parameter is
changed, no state variable is recorded - the subcircuit connectivity table
and the model constants [2] are read.

What is computed:
  - for each MBON of the subcircuit: the number of KC->m edges, the number
    of synapses, the sum of weights, the maximum single-edge weight, the
    10/50/90 % quantiles of the weight distribution;
  - aggregates for each of the six T38 types and for all MBON;
  - the number A and the peak time of a single EPSP by the V1c-E3.1
    formulas;
  - the grid's upper bound s_max by the V1c-E3.1 formula;
  - the V1c-E5 testability check: a T38 type with zero total weight across
    all its cells makes the stage untestable by construction.

The list of outputs is recorded in stamp 0 before the run and is not
extended here.

Run:  python v1c_weights.py
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
    """Factor and peak time of a single EPSP (V1c-E3.1).

    Model [2] is current-based: a spike adds weight w to variable g, and the
    membrane integrates g. From dv/dt = (v_0 - v + g)/t_mbr and
    dg/dt = -g/tau at rest, a single spike gives a deflection
        u(t) = w/(1 - rho) * (exp(-t/tau) - exp(-t/t_mbr)),  rho = t_mbr/tau,
    whose peak is reached at t* and equals A*w.
    """
    rho = t_mbr_ms / tau_ms
    if abs(rho - 1.0) < 1e-12:
        raise SystemExit("degenerate case t_mbr = tau: the peak formula is different")
    a = (1.0 / (rho - 1.0)) * (rho ** (-1.0 / (rho - 1.0))
                               - rho ** (-rho / (rho - 1.0)))
    t_star = tau_ms * t_mbr_ms * math.log(rho) / (t_mbr_ms - tau_ms)
    return float(a), float(t_star)


def main() -> int:
    if not STAMP0.exists():
        raise SystemExit(
            "stamp 0 not found: %s\nrule Д1 forbids this measurement before "
            "stamp 0 is recorded" % STAMP0)

    import v1b_subcircuit as V
    from model import default_params as dp

    neurons, con = V.load_substrate()
    role = dict(zip(neurons.root_id, neurons.mb_role))
    typ = dict(zip(neurons.root_id,
                   neurons.hemibrain_type.astype("string").fillna(V.UNTYPED)))

    # The same substitution mask as the stage's: DAN->KC and DAN->MBON edges
    # are removed. It must not affect KC->MBON edges, and this is checked,
    # not assumed.
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

    # edge weight at s = 1, in mV of increment to variable g
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

    # V1c-E5: a type whose cells all have zero total weight makes the stage
    # untestable by construction - no s will change a zero.
    untestable = [t for t, d in by_type.items()
                  if d["n_cells"] == 0 or d["sum_w_mV_total"] == 0.0]

    # V1c-E3.1: the grid's upper bound over s. The maximum is taken over the
    # cells of the six T38 types; the value over all MBON is printed for the
    # record and does not set the bound.
    w_max_t38 = max(d["w_max_mV"] for d in by_type.values())
    s_max_t38 = theta_mV / (a_peak * w_max_t38) if w_max_t38 > 0 else None
    s_max_all = (theta_mV / (a_peak * all_mbon["w_max_mV"])
                 if all_mbon["w_max_mV"] > 0 else None)

    # V1c-E3.2: grid nodes. Generated by the stamp 0 formula, not by choice.
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

    print("Measurement V1c-E6.1: structural KC->MBON weights (simulator not run)")
    print("KC->MBON edges: %d; the substitution mask does not touch them: %s"
          % (len(e), "yes" if not mask_touches_kc_mbon else "NO, look into it"))
    print("\n%-8s %6s %7s %9s %11s %11s %9s"
          % ("type", "cells", "edges", "synapses", "Σw, mV", "min Σw/cell", "w_max, mV"))
    for t, d in by_type.items():
        print("%-8s %6d %7d %9d %11.1f %11.1f %9.4f"
              % (t, d["n_cells"], d["n_edges_kc"], d["n_synapses_kc"],
                 d["sum_w_mV_total"], d["sum_w_mV_min_cell"], d["w_max_mV"]))
    print("%-8s %6d %7d %9d %11.1f %11.1f %9.4f"
          % ("all MBON", all_mbon["n_cells"], all_mbon["n_edges_kc"],
             all_mbon["n_synapses_kc"], all_mbon["sum_w_mV_total"],
             all_mbon["sum_w_mV_min_cell"], all_mbon["w_max_mV"]))

    print("\nSingle EPSP: A = %.6f, peak at %.3f ms, threshold θ = %.1f mV"
          % (a_peak, t_star_ms, theta_mV))
    print("s_max over T38 = θ / (A · w_max) = %.1f / (%.6f · %.4f) = %.2f"
          % (theta_mV, a_peak, w_max_t38, s_max_t38))
    print("for the record, over all MBON: w_max = %.4f mV, s_max = %.2f"
          % (all_mbon["w_max_mV"], s_max_all))
    print("\ngrid over s: %d nodes, %d per decade, from %.4g to %.4g"
          % (len(nodes), n_dec, nodes[0], nodes[-1]))
    print("  " + ", ".join("%.4g" % x for x in nodes))
    print("\ntestability V1c-E5: %s"
          % ("all six T38 types have nonzero KC input - the stage is testable"
             if not untestable else
             "UNTESTABLE by construction for types: " + ", ".join(untestable)))
    print("\nwritten: %s" % p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
