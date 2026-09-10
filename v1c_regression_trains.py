# -*- coding: utf-8 -*-
"""Regression control V1c-E6.4: identity of spike trains to the V1a-S core.

What is checked. Spec, section 3, criterion (a): "Testbed V1b with all
step-1 substitutions disabled reproduces the frozen V1a-S v0.9 artifacts
bitwise: 0 discrepancies on 555,544 edges, 56,020 out of 56,020 spike trains
across all 33 conditions. Tolerance zero." The structural part of this
criterion (edges, mask, config hashes) has been run and passes:
v1b_subcircuit.py --regression. Train identity has never been run; this
script runs it.

Stamp 0 of the V1c stage names this measurement V1c-E6.4 and permits it
before the freeze: "Passes Д2: the parameters are not changed at all." No
kc_mbon_scale value is set here; the simulator runs in a configuration
identical to V1a: substitutions off, graded APL off, input — replayed
spikes of external partners from the full-model recording [2].

What exactly is regressed. The core equations are taken from
v1b_subcircuit.EQS_CORE, i.e. from the very text the stage computes with.
They differ from [2] by the term (- inh) and the variable inh, which is
identically zero when graded APL is off. The regression checks that this
difference is numerically immaterial: the summation order, the integration
method, and the constants give the same spike train down to the last step
of the time grid.

The comparison set repeats V1a: the core is every subcircuit neuron except
PN (5,602 cells), 10 trials, 33 conditions, i.e. 56,020 trains per
condition. The reference is the frozen artifacts
results/v1a/runs/<condition>.parquet, full-model recordings restricted to
the core. Tolerance zero: time-grid step numbers are compared, not times
with a tolerance.

Run:
    msvc_run.bat v1c_regression_trains.py                 # all 33 conditions
    msvc_run.bat v1c_regression_trains.py --cond cal01_100Hz --trials 2
    python v1c_regression_trains.py --merge               # merge shards
    msvc_run.bat v1c_regression_trains.py --shard 0 --of 8
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
V1A = HERE / "results" / "v1a" / "runs"
OUT = HERE / "results" / "v1c"
# External input is taken from the FULL connectome, as in V1a: external
# means any presynaptic partner of a core neuron that is not in the core,
# including ones lying outside the subcircuit. The subcircuit connectivity
# will not do for this - it contains only internal edges, and a run on it
# would be checking a different network.
PATH_CON = HERE / "Drosophila_brain_model" / "2023_03_23_connectivity_630_final.parquet"

N_TRIALS = 10
T_RUN_MS = 1000
N_TRAINS_PER_COND = 56020


def conditions() -> list[str]:
    """The 33 V1a-S v0.9 conditions, in the order of the artifact names."""
    names = sorted(p.stem for p in V1A.glob("*.parquet")
                   if not p.stem.endswith("_replay"))
    if not names:
        raise SystemExit("no V1a artefacts found: %s" % V1A)
    return names


def core_ids(neurons: pd.DataFrame) -> list[int]:
    """The V1a core: every subcircuit neuron except PN."""
    return sorted(neurons[neurons.mb_role != "PN"].root_id.astype("int64").tolist())


def run_condition(name: str, con: pd.DataFrame,
                  ids: list[int], n_trials: int, codegen: str,
                  cache: str) -> dict:
    """Run one condition with the V1b testbed core and check trains against the reference."""
    from brian2 import (NeuronGroup, Synapses, SpikeGeneratorGroup, SpikeMonitor,
                        Network, prefs, ms, mV, second, defaultclock)
    import v1b_subcircuit as V
    from model import default_params as dp

    prefs.codegen.target = codegen
    if codegen == "cython" and cache:
        prefs.codegen.runtime.cython.cache_dir = cache
    dt_s = float(defaultclock.dt)

    full = pd.read_parquet(V1A / ("%s.parquet" % name))

    core = set(ids)
    onto = con[con.Postsynaptic_ID.isin(core)]
    inner = onto[onto.Presynaptic_ID.isin(core)]
    outer = onto[~onto.Presynaptic_ID.isin(core)]
    ext_ids = sorted(set(outer.Presynaptic_ID.astype("int64")))

    ci = {f: k for k, f in enumerate(ids)}
    ei = {f: k for k, f in enumerate(ext_ids)}
    in_i = inner.Presynaptic_ID.map(ci).to_numpy()
    in_j = inner.Postsynaptic_ID.map(ci).to_numpy()
    in_w = inner["Excitatory x Connectivity"].to_numpy() * dp["w_syn"]
    ex_i = outer.Presynaptic_ID.map(ei).to_numpy()
    ex_j = outer.Postsynaptic_ID.map(ci).to_numpy()
    ex_w = outer["Excitatory x Connectivity"].to_numpy() * dp["w_syn"]

    idx = np.asarray(ids, dtype=np.int64)
    n_exact = n_total = 0
    worst = 0
    mismatch: list[dict] = []
    t0 = time.time()

    for trial in range(n_trials):
        spk = full[full.trial == trial]
        s = spk[spk.flywire_id.isin(ei)]
        gi = s.flywire_id.map(ei).to_numpy()
        gt = s["t"].to_numpy() * second

        # The core is built with the V1b stage equations (EQS_CORE), not a
        # copy of [2]: the point of the regression is that the term (- inh)
        # changes nothing when inh = 0.
        neu = NeuronGroup(len(ids), model=V.EQS_CORE, method="linear",
                          threshold=dp["eq_th"], reset=dp["eq_rst"],
                          refractory="rfc", name="core", namespace=dp)
        neu.v = dp["v_0"]
        neu.g = 0 * mV
        neu.inh = 0 * mV
        neu.rfc = dp["t_rfc"]
        syn = Synapses(neu, neu, "w : volt", on_pre="g += w",
                       delay=dp["t_dly"], name="core_syn")
        syn.connect(i=in_i, j=in_j)
        syn.w = in_w
        gen = SpikeGeneratorGroup(len(ext_ids), gi, gt, name="ext")
        syn_e = Synapses(gen, neu, "w : volt", on_pre="g += w",
                         delay=dp["t_dly"], name="ext_syn")
        syn_e.connect(i=ex_i, j=ex_j)
        syn_e.w = ex_w
        mon = SpikeMonitor(neu)
        net = Network(neu, syn, gen, syn_e, mon)
        net.run(T_RUN_MS * ms)

        got = {idx[b]: np.sort(np.round(np.asarray(t) / dt_s).astype(np.int64))
               for b, t in mon.spike_trains().items() if len(t)}

        ref_tr = spk[spk.flywire_id.isin(core)]
        ref = {int(k): np.sort(np.round(v.to_numpy() / dt_s).astype(np.int64))
               for k, v in ref_tr.groupby("flywire_id")["t"]}

        for nid in ids:
            x = ref.get(nid, np.empty(0, dtype=np.int64))
            y = got.get(nid, np.empty(0, dtype=np.int64))
            n_total += 1
            if len(x) == len(y) and np.array_equal(x, y):
                n_exact += 1
            else:
                worst = max(worst, abs(len(x) - len(y)))
                if len(mismatch) < 20:
                    mismatch.append({"trial": trial, "flywire_id": int(nid),
                                     "n_ref": int(len(x)), "n_got": int(len(y))})

    return {"condition": name, "n_trials": n_trials,
            "n_ext_partners": len(ext_ids), "n_inner_edges": int(len(inner)),
            "n_outer_edges": int(len(outer)),
            "n_compared": n_total, "n_exact": n_exact,
            "max_spike_count_diff": int(worst),
            "identical": bool(n_exact == n_total),
            "mismatch_head": mismatch,
            "wall_s": round(time.time() - t0, 1)}


def merge() -> int:
    rows: list[dict] = []
    for p in sorted(OUT.glob("regression_trains.shard*.json")):
        rows.extend(json.loads(p.read_text(encoding="utf-8"))["conditions"])
    if not rows:
        raise SystemExit("no shards to merge in %s" % OUT)
    return report(rows, merged=True)


def report(rows: list[dict], merged: bool = False) -> int:
    rows = sorted(rows, key=lambda r: r["condition"])
    n_cond = len(rows)
    n_cmp = sum(r["n_compared"] for r in rows)
    n_ex = sum(r["n_exact"] for r in rows)
    ok = all(r["identical"] for r in rows)
    full_scope = (n_cond == 33
                  and all(r["n_compared"] == N_TRAINS_PER_COND for r in rows))

    print("\n%-18s %9s %9s %9s %s" % ("condition", "compared", "matched",
                                      "max Δ", "result"))
    for r in rows:
        print("%-18s %9d %9d %9d %s"
              % (r["condition"], r["n_compared"], r["n_exact"],
                 r["max_spike_count_diff"],
                 "identical" if r["identical"] else "DISCREPANCY"))
    print("-" * 60)
    print("conditions %d, trains %d, matched %d" % (n_cond, n_cmp, n_ex))
    print("scope of criterion (a): %s"
          % ("full - 33 conditions of 56,020 trains" if full_scope
             else "PARTIAL, criterion (a) not closed"))
    print("result: %s" % ("TRAIN IDENTITY, tolerance zero, 0 discrepancies"
                        if ok else "THERE ARE DISCREPANCIES - investigate before the V1c run"))

    out = {"measurement": "V1c-E6.4",
           "permitted_by": "правило Д2: параметры ступени не меняются",
           "criterion": "спецификация, раздел 3, критерий (а)",
           "reference": "results/v1a/runs (замороженные артефакты V1a-S v0.9)",
           "tolerance": "нулевой: сравниваются номера шагов сетки времени",
           "n_conditions": n_cond, "n_compared": n_cmp, "n_exact": n_ex,
           "scope_complete": bool(full_scope),
           "identical": bool(ok), "merged": bool(merged),
           "conditions": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "regression_trains.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("written: %s" % p)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="", help="one condition instead of all")
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--of", type=int, default=0)
    ap.add_argument("--codegen", default="cython", choices=["numpy", "cython"])
    ap.add_argument("--cache", default="")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()

    if a.merge:
        return merge()

    import v1b_subcircuit as V
    neurons, _ = V.load_substrate()
    ids = core_ids(neurons)
    con = pd.read_parquet(PATH_CON)
    names = [a.cond] if a.cond else conditions()
    if a.of:
        names = [n for k, n in enumerate(names) if k % a.of == a.shard]

    print("Regression V1c-E6.4: identity of trains to the V1a-S core")
    print("core %d neurons, trials %d, conditions %d, tolerance zero"
          % (len(ids), a.trials, len(names)))
    if a.trials != N_TRIALS or (not a.cond and not a.of
                                and len(names) != 33):
        print("WARNING: scope reduced, criterion (a) is not closed by this "
              "run")

    rows = []
    for name in names:
        r = run_condition(name, con, ids, a.trials, a.codegen, a.cache)
        rows.append(r)
        print("  %-18s %6d/%-6d  %5.0f s  %s"
              % (name, r["n_exact"], r["n_compared"], r["wall_s"],
                 "identical" if r["identical"] else "DISCREPANCY"))

    if a.of:
        OUT.mkdir(parents=True, exist_ok=True)
        p = OUT / ("regression_trains.shard%02d_of%02d.json" % (a.shard, a.of))
        p.write_text(json.dumps({"conditions": rows}, ensure_ascii=False,
                                indent=2), encoding="utf-8")
        print("shard written: %s" % p)
        return 0
    return report(rows)


if __name__ == "__main__":
    raise SystemExit(main())
