# -*- coding: utf-8 -*-
"""Benchmark of the codegen backend: numpy vs cython.

Measures network construction and simulation of 1 s of model time separately, in a
single process, without joblib — to compare pure cost, not the scheduler.

Two scales:
  full — the full model [2], 127,400 neurons (reference for the ladder's stages);
  sub  — mushroom-body subcircuit, 6,087 neurons (working configuration for the steps).

The cache of compiled inserts is placed on E:, not in the profile on C:.

Run:  .venv/Scripts/python.exe bench_backend.py numpy full
         .venv/Scripts/python.exe bench_backend.py cython full
         .venv/Scripts/python.exe bench_backend.py cython sub
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE / "Drosophila_brain_model"
SUB = HERE / "data" / "mb_subcircuit"
OUT = HERE / "results" / "bench_backend"
sys.path.insert(0, str(REPO))

TARGET = sys.argv[1] if len(sys.argv) > 1 else "numpy"
SCOPE = sys.argv[2] if len(sys.argv) > 2 else "full"
T_SIM_MS = 1000
RATE_HZ = 100
SEED_ODOR = 20260907 + 1

# Cython cache — on E:, to avoid writing to the profile on C: (owner's constraint).
CACHE = HERE / ".cython_cache"
CACHE.mkdir(exist_ok=True)
os.environ.setdefault("BRIAN2_CACHE_DIR", str(CACHE))

from brian2 import prefs  # noqa: E402  import after env

prefs.codegen.target = TARGET
if TARGET == "cython":
    prefs.codegen.runtime.cython.cache_dir = str(CACHE)
    # compile on a single thread: parallel workers used to stall on it before
    prefs.codegen.runtime.cython.multiprocess_safe = True

from brian2 import (NeuronGroup, Synapses, PoissonInput, SpikeMonitor,  # noqa: E402
                    Network, ms, mV, Hz)
from model import default_params as dp  # noqa: E402
import numpy as np  # noqa: E402


def load(scope: str):
    """Returns (completeness index, connectivity df, list of IDs to stimulate)."""
    if scope == "full":
        comp = pd.read_csv(REPO / "2023_03_23_completeness_630_final.csv", index_col=0)
        con = pd.read_parquet(REPO / "2023_03_23_connectivity_630_final.parquet")
    else:
        comp = pd.read_csv(SUB / "completeness.csv", index_col=0)
        con = pd.read_parquet(SUB / "connectivity.parquet")
    neurons = pd.read_csv(SUB / "neurons.csv")
    upn = neurons[(neurons.mb_role == "PN")
                  & (neurons.cell_sub_class == "uniglomerular")
                  & (neurons.side == "right")].root_id.astype("int64").to_numpy()
    rng = np.random.default_rng(SEED_ODOR)
    exc = sorted(rng.choice(upn, size=30, replace=False).tolist())
    return comp, con, exc


def main() -> int:
    print("backend %s, scope %s" % (TARGET, SCOPE))
    comp, con, exc = load(SCOPE)
    ids = list(comp.index.astype("int64"))
    idx = {f: k for k, f in enumerate(ids)}
    print("network: %d neurons, %d edges" % (len(ids), len(con)))

    t0 = time.time()
    neu = NeuronGroup(len(ids), model=dp["eqs"], method="linear",
                      threshold=dp["eq_th"], reset=dp["eq_rst"],
                      refractory="rfc", name="net", namespace=dp)
    neu.v = dp["v_0"]; neu.g = 0; neu.rfc = dp["t_rfc"]
    syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=dp["t_dly"], name="net_syn")
    syn.connect(i=con.Presynaptic_Index.to_numpy(), j=con.Postsynaptic_Index.to_numpy())
    syn.w = con["Excitatory x Connectivity"].to_numpy() * dp["w_syn"]
    pois = []
    for f in exc:
        if f not in idx:
            continue
        i = idx[f]
        pois.append(PoissonInput(target=neu[i], target_var="v", N=1, rate=RATE_HZ * Hz,
                                 weight=dp["w_syn"] * dp["f_poi"]))
        neu[i].rfc = 0 * ms
    mon = SpikeMonitor(neu)
    net = Network(neu, syn, mon, *pois)
    build_s = time.time() - t0

    t1 = time.time()
    net.run(T_SIM_MS * ms)
    run_s = time.time() - t1

    res = {"target": TARGET, "scope": SCOPE, "n_neurons": len(ids),
           "n_edges": int(len(con)), "n_stimulated": len(pois),
           "build_s": round(build_s, 2), "run_s": round(run_s, 2),
           "n_spikes": int(mon.num_spikes)}
    print("build %.2f s | simulation of 1 s model time %.2f s | spikes %d"
          % (build_s, run_s, mon.num_spikes))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ("%s_%s.json" % (SCOPE, TARGET))).write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
