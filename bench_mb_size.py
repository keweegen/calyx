# -*- coding: utf-8 -*-
"""Is the Cython backend needed for a network the size of step 1?

Builds an LIF network at the scale of the mushroom-body subcircuit (about 2000 KC per
hemisphere, 34 MBON, ~100 DAN per [8], [9]) with the same dynamics as in model [2], and
measures the simulation time for 1 s of model time on the numpy backend.

For comparison, the same measurement on a whole-brain-scale network is not made: it
already exists from the V0 runs (about 57 s for 5 parallel trials of 1000 ms).

Run:  .venv/Scripts/python.exe bench_mb_size.py
"""
import time
from brian2 import (
    NeuronGroup, Synapses, PoissonInput, SpikeMonitor, Network,
    prefs, mV, ms, Hz, defaultclock, second,
)

print("codegen.target =", repr(prefs["codegen.target"]))

# Dynamics constants — as in model.py of repository [2]
V_0, V_RST, V_TH = -52 * mV, -52 * mV, -45 * mV
T_MBR, TAU, T_RFC, T_DLY = 20 * ms, 5 * ms, 2.2 * ms, 1.8 * ms
W_SYN = 0.275 * mV

EQS = """
dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
dg/dt = -g / tau              : volt (unless refractory)
rfc                           : second
"""

# Subcircuit sizes: KC, MBON, DAN
N_KC, N_MBON, N_DAN = 2000, 34, 100
N = N_KC + N_MBON + N_DAN


def build_and_run(t_sim=1 * second, seed_conn=0):
    ns = {"v_0": V_0, "t_mbr": T_MBR, "tau": TAU}
    neu = NeuronGroup(
        N, EQS, method="exact", threshold="v > v_th",
        reset="v = v_rst; g = 0 * mV", refractory="rfc",
        namespace={**ns, "v_th": V_TH, "v_rst": V_RST},
    )
    neu.v = V_0
    neu.g = 0 * mV
    neu.rfc = T_RFC

    syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=T_DLY)
    # KC -> MBON: dense layer, where plasticity lives in step 1
    syn.connect(condition="i < %d and j >= %d and j < %d" % (N_KC, N_KC, N_KC + N_MBON))
    # MBON -> DAN: step 2 feedback
    syn.connect(condition="i >= %d and i < %d and j >= %d"
                % (N_KC, N_KC + N_MBON, N_KC + N_MBON))
    syn.w = W_SYN

    inp = PoissonInput(neu[:N_KC], "g", 1, 20 * Hz, weight=250 * W_SYN)
    mon = SpikeMonitor(neu)
    net = Network(neu, syn, inp, mon)

    t0 = time.time()
    net.run(t_sim)
    dt = time.time() - t0
    return dt, len(syn), mon.num_spikes


print("network: %d neurons (KC %d, MBON %d, DAN %d)" % (N, N_KC, N_MBON, N_DAN))
wall, n_syn, n_spk = build_and_run()
print("synapses: %d" % n_syn)
print("spikes over 1 s of model time: %d" % n_spk)
print()
print("simulation time for 1 s of model time: %.2f s" % wall)
print("wall time / model time:                %.1fx" % wall)
print()
n_runs = 30 * 6           # 30 repeats per condition, order of six conditions/controls
print("estimate for %d runs of 1 s sequentially: %.0f s (%.1f min)"
      % (n_runs, wall * n_runs, wall * n_runs / 60))
print("with 8 parallel processes:                    %.0f s (%.1f min)"
      % (wall * n_runs / 8, wall * n_runs / 8 / 60))
