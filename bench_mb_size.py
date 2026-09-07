# -*- coding: utf-8 -*-
"""Нужен ли Cython-бэкенд для сети размера шага 1?

Строит LIF-сеть масштаба подсхемы грибовидного тела (около 2000 KC на полушарие,
34 MBON, ~100 DAN по [8], [9]) с той же динамикой, что в модели [2], и меряет
время симуляции 1 с модельного времени на бэкенде numpy.

Для сравнения — тот же замер на сети масштаба полного мозга не делается: он уже
есть из прогонов V0 (около 57 с на 5 параллельных трайлов по 1000 мс).

Запуск:  .venv/Scripts/python.exe bench_mb_size.py
"""
import time
from brian2 import (
    NeuronGroup, Synapses, PoissonInput, SpikeMonitor, Network,
    prefs, mV, ms, Hz, defaultclock, second,
)

print("codegen.target =", repr(prefs["codegen.target"]))

# Константы динамики — как в model.py репозитория [2]
V_0, V_RST, V_TH = -52 * mV, -52 * mV, -45 * mV
T_MBR, TAU, T_RFC, T_DLY = 20 * ms, 5 * ms, 2.2 * ms, 1.8 * ms
W_SYN = 0.275 * mV

EQS = """
dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
dg/dt = -g / tau              : volt (unless refractory)
rfc                           : second
"""

# Размеры подсхемы: KC, MBON, DAN
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
    # KC -> MBON: плотный слой, в котором в шаге 1 живёт пластичность
    syn.connect(condition="i < %d and j >= %d and j < %d" % (N_KC, N_KC, N_KC + N_MBON))
    # MBON -> DAN: обратная связь шага 2
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


print("сеть: %d нейронов (KC %d, MBON %d, DAN %d)" % (N, N_KC, N_MBON, N_DAN))
wall, n_syn, n_spk = build_and_run()
print("синапсов: %d" % n_syn)
print("спайков за 1 с модельного времени: %d" % n_spk)
print()
print("время симуляции 1 с модельного времени: %.2f с" % wall)
print("реальное время / модельное:            %.1fx" % wall)
print()
n_runs = 30 * 6           # 30 повторов на условие, порядок шести условий/контролей
print("оценка для %d прогонов по 1 с последовательно: %.0f с (%.1f мин)"
      % (n_runs, wall * n_runs, wall * n_runs / 60))
print("при 8 параллельных процессах:                 %.0f с (%.1f мин)"
      % (wall * n_runs / 8, wall * n_runs / 8 / 60))
