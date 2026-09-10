# -*- coding: utf-8 -*-
"""Cost of running the actual mushroom-body subcircuit.

Replaces the synthetic estimate in `bench_mb_size.py`: there the network was built from
literature benchmarks (2,000 KC, 34 MBON, ~100 DAN, dense KC→MBON layer); here the
subcircuit is taken from the FlyWire v630 connectome
(`build_mb_subcircuit.py`), with its actual composition and connectivity.

Configuration — the one in which steps 1 and 2 will run: the subcircuit is autonomous,
input is delivered as Poissonian excitation of uniglomerular PNs, external input is cut off.

Run:  .venv/Scripts/python.exe bench_mb_subcircuit.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE / "Drosophila_brain_model"
SUB = HERE / "data" / "mb_subcircuit"
OUT = HERE / "results" / "mb_subcircuit"
sys.path.insert(0, str(REPO))

T_SIM_MS = 1000
RATE_HZ = 100
SEED_UPN30 = 20260907


def main() -> int:
    from brian2 import (NeuronGroup, Synapses, PoissonInput, SpikeMonitor,
                        Network, prefs, ms, Hz, mV)
    from model import default_params as dp

    neurons = pd.read_csv(SUB / "neurons.csv")
    comp = pd.read_csv(SUB / "completeness.csv", index_col=0)
    con = pd.read_parquet(SUB / "connectivity.parquet")
    ids = list(comp.index.astype("int64"))
    print("codegen backend: %r" % prefs["codegen.target"])
    print("subcircuit: %d neurons, %d edges, %d synapses"
          % (len(ids), len(con), int(con.Connectivity.sum())))

    idx = {f: k for k, f in enumerate(ids)}
    upn = neurons[(neurons.mb_role == "PN")
                  & (neurons.cell_sub_class == "uniglomerular")
                  & (neurons.side == "right")].root_id.astype("int64").tolist()

    t_build = time.time()
    neu = NeuronGroup(len(ids), model=dp["eqs"], method="linear",
                      threshold=dp["eq_th"], reset=dp["eq_rst"],
                      refractory="rfc", name="mb", namespace=dp)
    neu.v = dp["v_0"]; neu.g = 0; neu.rfc = dp["t_rfc"]
    syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=dp["t_dly"], name="mb_syn")
    syn.connect(i=con.Presynaptic_Index.to_numpy(), j=con.Postsynaptic_Index.to_numpy())
    syn.w = con["Excitatory x Connectivity"].to_numpy() * dp["w_syn"]

    pois = []
    for f in upn:
        i = idx[f]
        p = PoissonInput(target=neu[i], target_var="v", N=1, rate=RATE_HZ * Hz,
                         weight=dp["w_syn"] * dp["f_poi"])
        neu[i].rfc = 0 * ms
        pois.append(p)
    mon = SpikeMonitor(neu)
    net = Network(neu, syn, mon, *pois)
    build_s = time.time() - t_build

    t0 = time.time()
    net.run(T_SIM_MS * ms)
    wall = time.time() - t0

    res = {
        "n_neurons": len(ids), "n_edges": int(len(con)),
        "n_synapses": int(con.Connectivity.sum()),
        "n_pn_stimulated": len(upn), "rate_hz": RATE_HZ,
        "t_sim_ms": T_SIM_MS, "build_s": round(build_s, 2),
        "run_s": round(wall, 2), "n_spikes": int(mon.num_spikes),
    }
    print("network construction: %.2f s" % build_s)
    print("1 s of model time: %.2f s, spikes %d" % (wall, mon.num_spikes))
    n_runs = 30 * 6
    print("estimate for %d runs of 1 s sequentially: %.0f s (%.1f min)"
          % (n_runs, wall * n_runs, wall * n_runs / 60))
    print("with 8 processes: %.1f min" % (wall * n_runs / 8 / 60))
    res["estimate_180_runs_8proc_min"] = round(wall * n_runs / 8 / 60, 1)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bench.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
