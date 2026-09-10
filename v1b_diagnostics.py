# -*- coding: utf-8 -*-
# Output strings are deliberately Russian: this script must reproduce the
# committed report byte-for-byte.
"""Mechanism diagnostics for stage V1b: why MBONs are silent in the admissible band.

Outside the stage. No criterion is computed here and no parameter is chosen
from the results: the purpose is to separate the causes of MBON silence
found after calibrating regime B. Run at the chosen calibration point
(pn_kc_scale = 8, g_apl = 3,1623 g_ref) and at the points of the admissible
band known from the calibration artifacts.

One of the checks removes the APL->MBON edges - a substitution that is not
in the stage. It is introduced only to separate causes and is marked as
such; its result weakens the alternative explanation (APL inhibition on
MBON), not the desired one.

The odor set is the calibration one: the evaluation set E stays blind until
the stage run (V1b-4.7).

Run:  msvc_run.bat v1b_diagnostics.py --all
      msvc_run.bat v1b_diagnostics.py --band --cause --weights --edges
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# the chosen V1b calibration point and the admissible-band points leading to it
POINT = (8.0, 3.16228)
BAND = [(3.3636, 0.74989), (4.0, 1.0), (4.7568, 1.3335),
        (5.6569, 1.7783), (6.7272, 2.3714), (8.0, 3.16228)]
ODOR = "2-heptanone"          # calibration set; E's blindness is not spent
N_TRIALS_DIAG = 2


def _apl_to_mbon(con, role, apl_ids):
    return con.Presynaptic_ID.isin(apl_ids) & (
        con.Postsynaptic_ID.map(role) == "MBON")


def band_scan(V, neurons, con, role, seeds) -> list:
    """MBON response along the admissible band at timing M."""
    out = []
    print("Отклик MBON вдоль допустимой полосы (тайминг M, %s, %d пробы)"
          % (ODOR, len(seeds)))
    print("%-8s %-9s | %10s %12s | %s"
          % ("scale", "g/g_ref", "KC спайков", "MBON спайков", "доля отв. KC"))
    for sc, g in BAND:
        res = V.run_odor(neurons, con, ODOR, pn_kc_scale=sc,
                         g_apl=g * V.g_ref_value(), seeds=seeds,
                         pulse_ms=V.M_PULSE_MS, window_ms=V.M_WINDOW_MS)
        ids = res["core_ids"]
        mb = np.array([role[i] == "MBON" for i in ids])
        kc = np.array([role[i] == "Kenyon_Cell" for i in ids])
        row = {"pn_kc_scale": sc, "g_apl_rel": g,
               "kc_spikes": int(res["counts"][kc].sum()),
               "mbon_spikes": int(res["counts"][mb].sum()),
               "mbon_active_cells": int((res["counts"][mb].sum(axis=1) > 0).sum()),
               "kc_fraction_any_trial":
                   float(((res["counts"][kc] >= 1).sum(axis=1) >= 1).mean())}
        out.append(row)
        print("%-8.4g %-9.5g | %10d %12d | %.4f"
              % (sc, g, row["kc_spikes"], row["mbon_spikes"],
                 row["kc_fraction_any_trial"]), flush=True)
    return out


def cause_split(V, neurons, con, role, apl_ids, seeds) -> dict:
    """Three conditions at the chosen point: separate inhibition from a scarce KC input."""
    drop = _apl_to_mbon(con, role, apl_ids)
    sc, g = POINT
    conds = [("как есть", con, g),
             ("без APL->MBON", con[~drop].copy(), g),
             ("g_apl = 0", con, 0.0)]
    print("\nРазделение причин в выбранной точке scale %.4g "
          "(снимается рёбер APL->MBON: %d)" % (sc, int(drop.sum())))
    print("%-16s %12s %13s %10s | R_t по T38, Гц" % ("условие", "KC спайков",
                                                     "MBON спайков", "активных"))
    out = {"n_edges_apl_to_mbon": int(drop.sum()), "conditions": {}}
    for label, c, gg in conds:
        by = {ODOR: V.run_odor(neurons, c, ODOR, pn_kc_scale=sc,
                               g_apl=gg * V.g_ref_value(), seeds=seeds,
                               pulse_ms=V.M_PULSE_MS, window_ms=V.M_WINDOW_MS)}
        res = by[ODOR]
        ids = res["core_ids"]
        mb = np.array([role[i] == "MBON" for i in ids])
        kc = np.array([role[i] == "Kenyon_Cell" for i in ids])
        rates = V.mbon_type_rates(by, neurons, V.M_WINDOW_MS)
        rt = {t: float(rates.loc[t, ODOR]) for t in V.T38 if t in rates.index}
        allt = rates.loc[:, ODOR]
        d = {"g_apl_rel": gg, "kc_spikes": int(res["counts"][kc].sum()),
             "mbon_spikes": int(res["counts"][mb].sum()),
             "mbon_active_cells": int((res["counts"][mb].sum(axis=1) > 0).sum()),
             "mbon_cells_total": int(mb.sum()),
             "mean_rate_all_mbon_hz":
                 float(res["counts"][mb].mean() / (V.M_WINDOW_MS / 1000.0)),
             "R_t_T38": rt,
             "best_type_overall": [str(allt.idxmax()), float(allt.max())]}
        out["conditions"][label] = d
        print("%-16s %12d %13d %10d | %s"
              % (label, d["kc_spikes"], d["mbon_spikes"], d["mbon_active_cells"],
                 ", ".join("%s %.3f" % (t, v) for t, v in rt.items())), flush=True)
    print("пол V1b-3.3 = %.1f Гц у всех шести типов" % V.MBON_FLOOR_HZ)
    return out


def inhibition_weights(V, neurons, con, role, apl_ids) -> dict:
    """Total graded-node weight per cell, by role."""
    from model import default_params as dp
    g_abs = POINT[1] * V.g_ref_value()
    out_edges = con[con.Presynaptic_ID.isin(apl_ids)]
    w_syn_mV = float(dp["w_syn"] / (0.001 * 1.0))
    print("\nСуммарный вес градуального узла на клетку при g_apl = %.5g g_ref"
          % POINT[1])
    print("%-14s %8s %10s %10s %12s %12s"
          % ("мишень", "клеток", "син. мед.", "син. макс", "w мед., мВ", "w макс, мВ"))
    res = {}
    for r in ("Kenyon_Cell", "MBON", "DAN", "PN"):
        e = out_edges[out_edges.Postsynaptic_ID.map(role) == r]
        if not len(e):
            continue
        per = e.groupby("Postsynaptic_ID")["Connectivity"].apply(
            lambda s: float(np.abs(s).sum()))
        w = per * w_syn_mV * g_abs
        res[r] = {"n_cells": int(len(per)), "syn_median": float(per.median()),
                  "syn_max": float(per.max()), "w_median_mV": float(w.median()),
                  "w_max_mV": float(w.max())}
        print("%-14s %8d %10.0f %10.0f %12.1f %12.1f"
              % (r, len(per), per.median(), per.max(), w.median(), w.max()))
    return res


def edge_counts(con, role) -> dict:
    """Edges that determine separability of the third knob (the KC->MBON multiplier).

    If the graded node APL has no input from MBON, the KC->MBON multiplier on
    KC has no effect and calibrates one-dimensionally; otherwise the link
    closes through APL.
    """
    pre, post = con.Presynaptic_ID.map(role), con.Postsynaptic_ID.map(role)
    pairs = [("MBON", "APL"), ("Kenyon_Cell", "APL"), ("PN", "APL"),
             ("APL", "MBON"), ("APL", "Kenyon_Cell"), ("Kenyon_Cell", "MBON")]
    print("\nРёбра, существенные для сепарабельности множителя KC->MBON")
    out = {}
    into_apl = 0
    for a, b in pairs:
        e = con[(pre == a) & (post == b)]
        syn = int(e["Connectivity"].abs().sum())
        out["%s->%s" % (a, b)] = {"n_edges": int(len(e)), "n_synapses": syn}
        if b == "APL":
            into_apl += syn
        print("   %-12s -> %-12s рёбер %6d, синапсов %8d" % (a, b, len(e), syn))
    share = out["MBON->APL"]["n_synapses"] / into_apl if into_apl else float("nan")
    out["mbon_share_of_apl_input"] = float(share)
    print("   доля MBON во входе APL: %.4f — связь не нулевая, но слабая;"
          % share)
    print("   в разреженном режиме MBON молчат, то есть их вклад в APL равен нулю")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", action="store_true")
    ap.add_argument("--cause", action="store_true")
    ap.add_argument("--weights", action="store_true")
    ap.add_argument("--edges", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--codegen", default="cython", choices=["numpy", "cython"])
    a = ap.parse_args()
    if a.all:
        a.band = a.cause = a.weights = a.edges = True
    if not any((a.band, a.cause, a.weights, a.edges)):
        ap.error("нечего делать: укажите --all или отдельные проверки")

    from brian2 import prefs
    prefs.codegen.target = a.codegen
    import v1b_subcircuit as V

    neurons, con = V.load_substrate()
    role = dict(zip(neurons.root_id, neurons.mb_role))
    apl_ids = set(neurons.loc[neurons.mb_role == "APL", "root_id"])
    seeds = [V.SEED_EVAL + i for i in range(1, N_TRIALS_DIAG + 1)]

    out = {"point": {"pn_kc_scale": POINT[0], "g_apl_rel": POINT[1]},
           "odor": ODOR, "n_trials": N_TRIALS_DIAG, "seeds": seeds,
           "timing": "M", "pulse_ms": V.M_PULSE_MS,
           "status": "диагностика вне ступени: критерии не вычисляются, "
                     "параметры не выбираются"}
    if a.band:
        out["band_scan"] = band_scan(V, neurons, con, role, seeds)
    if a.cause:
        out["cause_split"] = cause_split(V, neurons, con, role, apl_ids, seeds)
    if a.weights:
        out["inhibition_weights"] = inhibition_weights(V, neurons, con, role, apl_ids)
    if a.edges:
        out["edge_counts"] = edge_counts(con, role)

    V.OUT.mkdir(parents=True, exist_ok=True)
    p = V.OUT / "diagnostics_mbon_silence.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nзаписано: %s" % p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
