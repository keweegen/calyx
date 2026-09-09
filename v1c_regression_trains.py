# -*- coding: utf-8 -*-
"""Регрессионный контроль V1c-E6.4: тождество спайковых поездов ядру V1a-S.

Что проверяется. Спецификация, раздел 3, критерий (а): «Стенд V1b со всеми
отключёнными подменами шага 1 воспроизводит замороженные артефакты V1a-S v0.9
побитово: 0 расхождений на 555 544 рёбрах, 56 020 из 56 020 спайковых поездов
на всех 33 условиях. Допуск нулевой.» Структурная часть этого критерия
(рёбра, маска, хэши конфигов) прогонялась и проходит: v1b_subcircuit.py
--regression. Тождество поездов не прогонялось ни разу; этот скрипт его
прогоняет.

Штамп 0 ступени V1c называет это измерение V1c-E6.4 и разрешает его до
заморозки: «Проходит Д2: параметры не меняются вовсе.» Ни одного значения
kc_mbon_scale здесь не задаётся, симулятор исполняется в конфигурации,
тождественной V1a: подмены выключены, градуальный APL выключен, вход —
воспроизведённые спайки внешних партнёров из записи полной модели [2].

Что именно регрессируется. Уравнения ядра берутся из v1b_subcircuit.EQS_CORE,
то есть из того самого текста, по которому считает ступень. Они отличаются от
[2] членом (- inh) и переменной inh, которая при выключенном градуальном APL
тождественно равна нулю. Регрессия проверяет, что это отличие численно
безразлично: порядок суммирования, метод интегрирования и константы дают тот
же спайковый поезд до последнего шага сетки времени.

Множество сравнения повторяет V1a: ядро — все нейроны подсхемы, кроме PN
(5 602 клетки), 10 трайлов, 33 условия, то есть 56 020 поездов на условие.
Эталон — замороженные артефакты results/v1a/runs/<условие>.parquet, записи
полной модели, ограниченные ядром. Допуск нулевой: сравниваются номера шагов
сетки времени, а не времена с допуском.

Запуск:
    msvc_run.bat v1c_regression_trains.py                 # все 33 условия
    msvc_run.bat v1c_regression_trains.py --cond cal01_100Hz --trials 2
    python v1c_regression_trains.py --merge               # свести шарды
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
# Внешний вход берётся из ПОЛНОГО коннектома, как в V1a: внешним считается
# всякий пресинаптический партнёр нейрона ядра, не входящий в ядро, в том
# числе лежащий вне подсхемы. Связность подсхемы для этого не годится - она
# содержит только внутренние рёбра, и прогон на ней проверял бы другую сеть.
PATH_CON = HERE / "Drosophila_brain_model" / "2023_03_23_connectivity_630_final.parquet"

N_TRIALS = 10
T_RUN_MS = 1000
N_TRAINS_PER_COND = 56020


def conditions() -> list[str]:
    """33 условия V1a-S v0.9 в порядке имён артефактов."""
    names = sorted(p.stem for p in V1A.glob("*.parquet")
                   if not p.stem.endswith("_replay"))
    if not names:
        raise SystemExit("не найдены артефакты V1a: %s" % V1A)
    return names


def core_ids(neurons: pd.DataFrame) -> list[int]:
    """Ядро V1a: все нейроны подсхемы, кроме PN."""
    return sorted(neurons[neurons.mb_role != "PN"].root_id.astype("int64").tolist())


def run_condition(name: str, con: pd.DataFrame,
                  ids: list[int], n_trials: int, codegen: str,
                  cache: str) -> dict:
    """Прогнать одно условие ядром стенда V1b и сверить поезда с эталоном."""
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

        # Ядро строится уравнениями ступени V1b (EQS_CORE), а не копией [2]:
        # смысл регрессии в том, что член (- inh) при inh = 0 ничего не меняет.
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
        raise SystemExit("нет шардов для сведения в %s" % OUT)
    return report(rows, merged=True)


def report(rows: list[dict], merged: bool = False) -> int:
    rows = sorted(rows, key=lambda r: r["condition"])
    n_cond = len(rows)
    n_cmp = sum(r["n_compared"] for r in rows)
    n_ex = sum(r["n_exact"] for r in rows)
    ok = all(r["identical"] for r in rows)
    full_scope = (n_cond == 33
                  and all(r["n_compared"] == N_TRAINS_PER_COND for r in rows))

    print("\n%-18s %9s %9s %9s %s" % ("условие", "сравнено", "совпало",
                                      "макс.Δ", "итог"))
    for r in rows:
        print("%-18s %9d %9d %9d %s"
              % (r["condition"], r["n_compared"], r["n_exact"],
                 r["max_spike_count_diff"],
                 "тождество" if r["identical"] else "РАСХОЖДЕНИЕ"))
    print("-" * 60)
    print("условий %d, поездов %d, совпало %d" % (n_cond, n_cmp, n_ex))
    print("объём критерия (а): %s"
          % ("полный - 33 условия по 56 020 поездов" if full_scope
             else "ЧАСТИЧНЫЙ, критерий (а) не закрыт"))
    print("итог: %s" % ("ТОЖДЕСТВО ПОЕЗДОВ, допуск нулевой, расхождений 0"
                        if ok else "ЕСТЬ РАСХОЖДЕНИЯ - разбирать до прогона V1c"))

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
    print("записано: %s" % p)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="", help="одно условие вместо всех")
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

    print("Регрессия V1c-E6.4: тождество поездов ядру V1a-S")
    print("ядро %d нейронов, трайлов %d, условий %d, допуск нулевой"
          % (len(ids), a.trials, len(names)))
    if a.trials != N_TRIALS or (not a.cond and not a.of
                                and len(names) != 33):
        print("ВНИМАНИЕ: объём сокращён, критерий (а) этим прогоном не "
              "закрывается")

    rows = []
    for name in names:
        r = run_condition(name, con, ids, a.trials, a.codegen, a.cache)
        rows.append(r)
        print("  %-18s %6d/%-6d  %5.0f с  %s"
              % (name, r["n_exact"], r["n_compared"], r["wall_s"],
                 "тождество" if r["identical"] else "РАСХОЖДЕНИЕ"))

    if a.of:
        OUT.mkdir(parents=True, exist_ok=True)
        p = OUT / ("regression_trains.shard%02d_of%02d.json" % (a.shard, a.of))
        p.write_text(json.dumps({"conditions": rows}, ensure_ascii=False,
                                indent=2), encoding="utf-8")
        print("шард записан: %s" % p)
        return 0
    return report(rows)


if __name__ == "__main__":
    raise SystemExit(main())
