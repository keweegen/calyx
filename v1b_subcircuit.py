# -*- coding: utf-8 -*-
"""Ступень V1b: подсхема грибовидного тела после подмен шага 1.

Три подмены, каждая включается отдельным флагом конфига, чтобы регрессионный
контроль (спецификация, раздел 3, критерий V1b (а)) мог выключить их все и
получить конфигурацию V1a:

  dan_mask   маска быстрых синапсов DAN: снимаются рёбра DAN->KC и DAN->MBON,
             то есть обе стороны пластичного синапса, где дофамин учёлся бы
             дважды. DAN->APL, DAN->DAN и DAN->PN остаются: правило шага 1 их
             не моделирует.
  graded_apl APL выводится из спайкового режима. Уравнение мембраны из [2]
             без порога, сброса и рефрактерности; выход g_apl*max(0, v - v_0)
             передаётся каждый шаг интегрирования по рёбрам APL->KC и APL->MBON
             с весами коннектома (суммируемая синаптическая переменная).
  odor_input вход - паттерны запахов вместо пуассоновской стимуляции выборок PN.
             Частоты по гломерулам из results/odor_panel/pn_rates_by_odor.tsv
             (источник [39], преобразование [42]); подаются на унигломерулярные
             PN обоих полушарий симметрично.

Калибруются ровно два параметра режима B: pn_kc_scale и g_apl (спецификация,
раздел 3е, V1b-4.3). Всё остальное - режим A из [2].

Запуск:  python v1b_subcircuit.py --self-test
         python v1b_subcircuit.py --odor "pentyl acetate" --scale 1.0 --gapl 1.0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE / "Drosophila_brain_model"
sys.path.insert(0, str(REPO))

SUB = HERE / "data" / "mb_subcircuit"
PANEL = HERE / "results" / "odor_panel" / "pn_rates_by_odor.tsv"
OUT = HERE / "results" / "v1b"

# Окно измерения по [40]: импульс 1 с, амплитуда - средняя частота за 4 с от
# начала импульса (спецификация, раздел 3е, V1b-2.1).
T_ON_MS, T_PULSE_MS, T_WINDOW_MS = 200, 1000, 4000
T_RUN_MS = T_ON_MS + T_WINDOW_MS

# Адаптивное усечение: прогон идёт до затухания, а не весь измерительный
# интервал. QUIET_MS - окно, в котором не должно быть ни одного спайка,
# чтобы прогон считался завершённым. Счётчики тождественны полному прогону:
# при нулевом фоне сеть без входа не спайкует.
SETTLE_MS, QUIET_MS = 300, 200

KC_SPIKE_THRESHOLD = 1      # V1b-2.1: ответ на пробу - хотя бы один спайк
N_TRIALS = 6                # V1b-2.2: шесть предъявлений
MIN_TRIALS = 3              # V1b-2.2: ответ не менее чем на половине


def load_substrate() -> tuple[pd.DataFrame, pd.DataFrame]:
    neurons = pd.read_csv(SUB / "neurons.csv")
    neurons = neurons.assign(
        gl=neurons.hemibrain_type.astype(str).str.split("_").str[0])
    con = pd.read_parquet(SUB / "connectivity.parquet")
    return neurons, con


def apply_dan_mask(con: pd.DataFrame, role: dict) -> tuple[pd.DataFrame, int]:
    """Снять рёбра DAN->KC и DAN->MBON. Возвращает оставшиеся рёбра и число снятых."""
    pre = con.Presynaptic_ID.map(role)
    post = con.Postsynaptic_ID.map(role)
    drop = (pre == "DAN") & post.isin(["Kenyon_Cell", "MBON"])
    return con[~drop].copy(), int(drop.sum())


def odor_rates(odor: str, neurons: pd.DataFrame) -> dict[int, float]:
    """Частоты пуассоновского входа на унигломерулярные PN обоих полушарий."""
    pn = pd.read_csv(PANEL, sep="\t", index_col=0)
    if odor not in pn.index:
        raise SystemExit("запах %r не в панели; доступны: %s"
                         % (odor, ", ".join(pn.index)))
    by_gl = pn.loc[odor].to_dict()
    upn = neurons[(neurons.mb_role == "PN")
                  & (neurons.cell_sub_class == "uniglomerular")]
    return {int(r.root_id): float(by_gl[r.gl])
            for r in upn.itertuples() if r.gl in by_gl}


def build(neurons: pd.DataFrame, con: pd.DataFrame, *, pn_kc_scale: float,
          g_apl: float, dan_mask: bool, graded_apl: bool,
          rates: dict[int, float] | None,
          pulse_ms: int = T_PULSE_MS, run_ms: int = T_RUN_MS):
    """Собрать сеть Brian 2. Возвращает (Network, SpikeMonitor, порядок id)."""
    from brian2 import (NeuronGroup, Synapses, PoissonGroup, SpikeMonitor,
                        Network, TimedArray, Hz, ms, mV)
    from model import default_params as dp

    role = dict(zip(neurons.root_id, neurons.mb_role))
    n_masked = 0
    if dan_mask:
        con, n_masked = apply_dan_mask(con, role)

    apl_ids = sorted(neurons.loc[neurons.mb_role == "APL", "root_id"])
    if graded_apl:
        core_ids = sorted(set(neurons.root_id) - set(apl_ids))
    else:
        core_ids = sorted(neurons.root_id)
        apl_ids = []

    ci = {f: k for k, f in enumerate(core_ids)}
    ai = {f: k for k, f in enumerate(apl_ids)}

    # Уравнения ядра: [2] плюс член градуального торможения. При выключенном
    # градуальном APL член остаётся тождественным нулю, поэтому уравнение
    # численно совпадает с [2] и регрессия к V1a не нарушается.
    eqs_core = """
        dv/dt = (v_0 - v + g - inh) / t_mbr : volt (unless refractory)
        dg/dt = -g / tau                    : volt (unless refractory)
        inh                                 : volt
        rfc                                 : second
    """
    neu = NeuronGroup(len(core_ids), model=eqs_core, method="linear",
                      threshold=dp["eq_th"], reset=dp["eq_rst"],
                      refractory="rfc", name="core", namespace=dp)
    neu.v = dp["v_0"]
    neu.g = 0 * mV
    neu.inh = 0 * mV
    neu.rfc = dp["t_rfc"]

    objs = [neu]

    # рёбра внутри ядра
    e = con[con.Presynaptic_ID.isin(ci) & con.Postsynaptic_ID.isin(ci)]
    w = e["Excitatory x Connectivity"].to_numpy() * dp["w_syn"]
    if pn_kc_scale != 1.0:
        pre_role = e.Presynaptic_ID.map(role).to_numpy()
        post_role = e.Postsynaptic_ID.map(role).to_numpy()
        sel = (pre_role == "PN") & (post_role == "Kenyon_Cell")
        # не np.where: он срывает единицы Brian 2 и веса приходят безразмерными
        mult = np.ones(len(e))
        mult[sel] = pn_kc_scale
        w = w * mult
    syn = Synapses(neu, neu, "w : volt", on_pre="g += w",
                   delay=dp["t_dly"], name="core_syn")
    syn.connect(i=e.Presynaptic_ID.map(ci).to_numpy(),
                j=e.Postsynaptic_ID.map(ci).to_numpy())
    syn.w = w
    objs.append(syn)

    apl = None
    if graded_apl:
        # Градуальный узел: та же мембрана, без порога, сброса и рефрактерности.
        eqs_apl = """
            dv/dt = (v_0 - v + g) / t_mbr : volt
            dg/dt = -g / tau              : volt
            out = clip((v - v_0) / mV, 0, 1e9) : 1
        """
        apl = NeuronGroup(len(apl_ids), model=eqs_apl, method="linear",
                          name="apl", namespace=dp)
        apl.v = dp["v_0"]
        apl.g = 0 * mV
        objs.append(apl)

        # вход на APL - обычные спайковые синапсы
        into = con[con.Presynaptic_ID.isin(ci) & con.Postsynaptic_ID.isin(ai)]
        s_in = Synapses(neu, apl, "w : volt", on_pre="g += w",
                        delay=dp["t_dly"], name="apl_in")
        s_in.connect(i=into.Presynaptic_ID.map(ci).to_numpy(),
                     j=into.Postsynaptic_ID.map(ai).to_numpy())
        s_in.w = into["Excitatory x Connectivity"].to_numpy() * dp["w_syn"]
        objs.append(s_in)

        # выход APL - суммируемая переменная, передаётся каждый шаг
        out = con[con.Presynaptic_ID.isin(ai) & con.Postsynaptic_ID.isin(ci)]
        s_out = Synapses(apl, neu,
                         """w : volt
                            inh_post = w * out_pre : volt (summed)""",
                         name="apl_out", namespace=dp)
        s_out.connect(i=out.Presynaptic_ID.map(ai).to_numpy(),
                      j=out.Postsynaptic_ID.map(ci).to_numpy())
        # знак торможения задаётся членом (-inh) в уравнении ядра, поэтому вес
        # берётся по модулю числа синапсов; g_apl - калибруемый коэффициент
        s_out.w = np.abs(out["Connectivity"].to_numpy()) * dp["w_syn"] * g_apl
        objs.append(s_out)
        n_apl_in, n_apl_out = len(into), len(out)
    else:
        n_apl_in = n_apl_out = 0

    n_stim = 0
    if rates:
        # Импульс запаха 1 с, начиная с T_ON_MS (протокол [40]). Пуассоновская
        # группа с временным профилем: до и после импульса частота нулевая.
        # PoissonInput постоянен во времени и для импульса не годится.
        stim_ids = [f for f, r in rates.items() if f in ci and r > 0]
        n_stim = len(stim_ids)
        if n_stim:
            step_ms = 50
            n_steps = int(np.ceil(run_ms / step_ms))
            prof = np.zeros((n_steps, n_stim))
            on0, on1 = T_ON_MS // step_ms, (T_ON_MS + pulse_ms) // step_ms
            prof[on0:on1, :] = np.array([rates[f] for f in stim_ids])
            ta = TimedArray(prof * Hz, dt=step_ms * ms, name="odor_profile")
            src = PoissonGroup(n_stim, rates="odor_profile(t, i)", name="odor",
                               namespace={"odor_profile": ta})
            s_stim = Synapses(src, neu, on_pre="g += w_stim", name="odor_syn",
                              namespace={"w_stim": dp["f_poi"] * dp["w_syn"]})
            s_stim.connect(i=np.arange(n_stim),
                           j=np.array([ci[f] for f in stim_ids]))
            objs.extend([src, s_stim])
            # у стимулируемых нейронов рефрактерность снимается, как в model.poi
            for f in stim_ids:
                neu.rfc[ci[f]] = 0 * ms

    mon = SpikeMonitor(neu)
    objs.append(mon)
    net = Network(*objs)
    stats = {"n_core": len(core_ids), "n_apl": len(apl_ids),
             "n_edges_core": len(e), "n_masked": n_masked,
             "n_apl_in": n_apl_in, "n_apl_out": n_apl_out,
             "n_poisson": n_stim}
    return net, mon, core_ids, stats


def measure(mon, core_ids: list[int], neurons: pd.DataFrame,
            window_ms: float = T_WINDOW_MS) -> pd.DataFrame:
    """Средняя частота каждого нейрона в окне измерения, спайк/с."""
    from brian2 import second
    t0, t1 = T_ON_MS / 1000.0, (T_ON_MS + window_ms) / 1000.0
    counts = np.zeros(len(core_ids))
    for idx, ts in mon.spike_trains().items():
        t = np.asarray(ts / second)
        counts[idx] = ((t >= t0) & (t < t1)).sum()
    role = dict(zip(neurons.root_id, neurons.mb_role))
    return pd.DataFrame({"root_id": core_ids,
                         "n_spikes": counts.astype(int),
                         "rate_hz": counts / (window_ms / 1000.0),
                         "role": [role[i] for i in core_ids]})



def run_odor(neurons: pd.DataFrame, con: pd.DataFrame, odor: str, *,
             pn_kc_scale: float, g_apl: float, seeds: list[int],
             dan_mask: bool = True, graded_apl: bool = True,
             pulse_ms: int = T_PULSE_MS, window_ms: int = T_WINDOW_MS,
             extra_windows: tuple = ()) -> dict:
    """Шесть предъявлений одного запаха. Возвращает счёт спайков по пробам.

    Сеть строится один раз; между пробами состояние восстанавливается и
    меняется только зерно генератора, поэтому пробы различаются лишь
    реализацией пуассоновского входа (V1b-2.2).
    """
    from brian2 import ms, seed as b2seed, device

    rates = odor_rates(odor, neurons)
    run_ms = T_ON_MS + window_ms
    net, mon, core_ids, st = build(
        neurons, con, pn_kc_scale=pn_kc_scale, g_apl=g_apl,
        dan_mask=dan_mask, graded_apl=graded_apl, rates=rates,
        pulse_ms=pulse_ms, run_ms=run_ms)
    net.store("init")

    wins = [(0.0, window_ms / 1000.0)] + [(a / 1000.0, b / 1000.0)
                                          for a, b in extra_windows]
    counts = [np.zeros((len(core_ids), len(seeds)), dtype=np.int32) for _ in wins]
    truncated_at = []
    for k, sd in enumerate(seeds):
        net.restore("init")
        b2seed(sd)
        # до конца импульса плюс запас, дальше - пока в последнем окне есть спайки
        done = min(T_ON_MS + pulse_ms + SETTLE_MS, run_ms)
        net.run(done * ms)
        while done < run_ms:
            prev = int(mon.num_spikes)
            step = min(QUIET_MS, run_ms - done)
            net.run(step * ms)
            done += step
            if int(mon.num_spikes) == prev:
                break
        truncated_at.append(done)
        for idx, ts in mon.spike_trains().items():
            # имя не t: локальная t протекает в пространство имён Brian и
            # конфликтует с его внутренней переменной времени
            spk = np.asarray(ts) - T_ON_MS / 1000.0
            for wi, (a, b) in enumerate(wins):
                counts[wi][idx, k] = ((spk >= a) & (spk < b)).sum()
    return {"core_ids": core_ids, "counts": counts[0],
            "counts_by_window": counts, "windows": wins,
            "stats": st, "odor": odor, "seeds": list(seeds),
            "pulse_ms": pulse_ms, "window_ms": window_ms,
            "simulated_ms": truncated_at, "full_ms": run_ms}


def kc_fraction(res: dict, neurons: pd.DataFrame) -> dict:
    """Доля отвечающих KC по правилу V1b-2.1-2.3 плюс таблица чувствительности."""
    role = dict(zip(neurons.root_id, neurons.mb_role))
    is_kc = np.array([role[i] == "Kenyon_Cell" for i in res["core_ids"]])
    c = res["counts"][is_kc]
    n_kc_total = int(sum(1 for r in role.values() if r == "Kenyon_Cell"))

    def frac(k: int, min_trials: int) -> float:
        return float((( c >= k).sum(axis=1) >= min_trials).sum()) / n_kc_total

    out = {"n_kc_denominator": n_kc_total,
           "f": frac(KC_SPIKE_THRESHOLD, MIN_TRIALS),
           "sensitivity": {}}
    for k in (1, 2, 3, 5):
        for mt, lab in ((1, ">=1/6"), (MIN_TRIALS, ">=3/6"), (len(res["seeds"]), "6/6")):
            out["sensitivity"]["k=%d,%s" % (k, lab)] = round(frac(k, mt), 5)
    return out


# --- наборы запахов (спецификация, V1b-4.1) ----------------------------------
PANEL_CAL = ["2-heptanone", "isopentyl acetate", "hexanal",
             "6-methyl-5-hepten-2-one", "diethyl succinate", "methyl octanoate"]
PANEL_EVAL = ["pentyl acetate", "butyl acetate", "ethyl lactate", "1-octen-3-ol",
              "pentanal", "benzaldehyde", "alpha-humulene", "ethyl octanoate"]
REF_PAIRS = [("pentyl acetate", "butyl acetate"),
             ("pentyl acetate", "ethyl lactate"),
             ("butyl acetate", "ethyl lactate")]

# множество T38: шесть типов MBON, записанных в [38] (спецификация, V1b-3.1)
T38 = ["MBON11", "MBON12", "MBON13", "MBON14", "MBON17", "MBON18"]
SEED_CAL, SEED_EVAL = 20260908, 20260908 + 100

# пороги критериев (спецификация, раздел 3е)
F_BAND, F_MAX = (0.03, 0.10), 0.10
OVERLAP_SEP_MIN, OVERLAP_CEIL = 0.25, 0.40
MBON_FLOOR_HZ, MBON_CEIL_HZ, MBON_MD_MIN, MBON_MD_TYPES = 2.0, 67.0, 0.19, 5
SPIKES_AB_TARGET = 2.2          # V1b-4.9, [46]
N_SUBSAMPLE, N_DRAWS = 120, 24  # V1b-3д: подвыборка «записи»


def run_panel(neurons, con, odors, *, pn_kc_scale, g_apl, seed_base,
              pulse_ms=T_PULSE_MS, window_ms=T_WINDOW_MS, extra_windows=(),
              dan_mask=True, graded_apl=True) -> dict:
    """Прогнать набор запахов одним и тем же стендом."""
    seeds = [seed_base + i for i in range(1, N_TRIALS + 1)]
    out = {}
    for o in odors:
        out[o] = run_odor(neurons, con, o, pn_kc_scale=pn_kc_scale, g_apl=g_apl,
                          seeds=seeds, dan_mask=dan_mask, graded_apl=graded_apl,
                          pulse_ms=pulse_ms, window_ms=window_ms,
                          extra_windows=extra_windows)
    return out


def _kc_mask(core_ids, neurons, prefix: str | None = None) -> np.ndarray:
    role = dict(zip(neurons.root_id, neurons.mb_role))
    typ = dict(zip(neurons.root_id, neurons.hemibrain_type.astype(str)))
    if prefix is None:
        return np.array([role[i] == "Kenyon_Cell" for i in core_ids])
    return np.array([role[i] == "Kenyon_Cell" and typ[i].startswith(prefix)
                     for i in core_ids])


def overlap_pearson(by_odor: dict, neurons: pd.DataFrame,
                    rng_seed: int = 20260908) -> dict:
    """Перекрытие ансамблей KC по V1b-3д: Пирсон внутри подвыборки «записи»."""
    odors = list(by_odor)
    ids = by_odor[odors[0]]["core_ids"]
    kc = _kc_mask(ids, neurons)
    # вектор запаха - среднее по пробам число спайков каждой KC
    vec = {o: by_odor[o]["counts"][kc].mean(axis=1) for o in odors}
    n_kc = int(kc.sum())
    rng = np.random.default_rng(rng_seed)
    draws = [rng.choice(n_kc, size=min(N_SUBSAMPLE, n_kc), replace=False)
             for _ in range(N_DRAWS)]

    r = {}
    for a in range(len(odors)):
        for b in range(a + 1, len(odors)):
            oa, ob = odors[a], odors[b]
            vals = []
            for d in draws:
                x, y = vec[oa][d], vec[ob][d]
                if x.std() == 0 or y.std() == 0:
                    continue
                vals.append(float(np.corrcoef(x, y)[0, 1]))
            r[(oa, ob)] = float(np.mean(vals)) if vals else float("nan")
    return r


def check_overlap(r: dict) -> dict:
    """Критерий перекрытия: порядок трёх эталонных пар, разделение, потолок."""
    g = lambda a, b: r.get((a, b), r.get((b, a), float("nan")))
    pab = g(*REF_PAIRS[0])
    pel, bel = g(*REF_PAIRS[1]), g(*REF_PAIRS[2])
    order = pab > pel and pab > bel
    sep = pab - max(pel, bel)
    ceil_ok = max(pel, bel) <= OVERLAP_CEIL
    vals = [v for v in r.values() if v == v]
    return {"r_PA_BA": pab, "r_PA_EL": pel, "r_BA_EL": bel,
            "order_ok": bool(order), "separation": float(sep),
            "separation_ok": bool(sep >= OVERLAP_SEP_MIN),
            "ceiling_ok": bool(ceil_ok),
            "median_all_pairs": float(np.median(vals)) if vals else float("nan"),
            "p90_all_pairs": float(np.percentile(vals, 90)) if vals else float("nan"),
            "pass": bool(order and sep >= OVERLAP_SEP_MIN and ceil_ok)}


def mbon_type_rates(by_odor: dict, neurons: pd.DataFrame,
                    window_ms: float) -> pd.DataFrame:
    """Средняя частота типа MBON по запахам, спайк/с (V1b-3.1)."""
    ids = by_odor[list(by_odor)[0]]["core_ids"]
    typ = dict(zip(neurons.root_id, neurons.hemibrain_type.astype(str)))
    role = dict(zip(neurons.root_id, neurons.mb_role))
    rows = {}
    for o, res in by_odor.items():
        per = res["counts"].mean(axis=1) / (window_ms / 1000.0)
        d = {}
        for k, i in enumerate(ids):
            if role[i] == "MBON":
                d.setdefault(typ[i], []).append(per[k])
        rows[o] = {t: float(np.mean(v)) for t, v in d.items()}
    return pd.DataFrame(rows)


def check_mbon(rates: pd.DataFrame) -> dict:
    """Пол, потолок и глубина модуляции по T38 (V1b-3.3-3.5)."""
    present = [t for t in T38 if t in rates.index]
    res = {"types_present": present, "missing": [t for t in T38 if t not in rates.index]}
    per_type = {}
    for t in present:
        v = rates.loc[t].to_numpy(dtype=float)
        hi, lo = float(v.max()), float(v.min())
        per_type[t] = {"R_t": float(v.mean()), "max": hi, "min": lo,
                       "MD": (hi - lo) / (hi + lo) if (hi + lo) > 0 else float("nan")}
    res["per_type"] = per_type
    floors = [v["R_t"] >= MBON_FLOOR_HZ for v in per_type.values()]
    ceils = [v["R_t"] <= MBON_CEIL_HZ for v in per_type.values()]
    mds = [v["MD"] >= MBON_MD_MIN for t, v in per_type.items()
           if per_type[t]["R_t"] >= MBON_FLOOR_HZ]
    res.update({"floor_ok": bool(all(floors)), "ceiling_ok": bool(all(ceils)),
                "n_md_ok": int(sum(mds)), "md_ok": bool(sum(mds) >= MBON_MD_TYPES),
                "pass": bool(all(floors) and all(ceils) and sum(mds) >= MBON_MD_TYPES)})
    return res


def spikes_per_response(by_odor: dict, neurons: pd.DataFrame,
                        prefix: str, window_idx: int = 1) -> float:
    """Спайков за ответ у подтипа KC в окне эталона (V1b-4.9).

    window_idx указывает на окно [0; 2 с] из extra_windows; учитываются пары
    «клетка-запах», отвечающие по правилу V1b-2.1-2.2, и внутри них только
    предъявления, на которых в окне эталона есть спайк.
    """
    ids = by_odor[list(by_odor)[0]]["core_ids"]
    m = _kc_mask(ids, neurons, prefix)
    vals = []
    for res in by_odor.values():
        full = res["counts"][m]                        # окно 4 с, правило ответа
        ref = res["counts_by_window"][window_idx][m]   # окно эталона 2 с
        responder = (full >= KC_SPIKE_THRESHOLD).sum(axis=1) >= MIN_TRIALS
        for row in np.nonzero(responder)[0]:
            hit = ref[row][ref[row] >= 1]
            if hit.size:
                vals.append(float(hit.mean()))
    return float(np.mean(vals)) if vals else float("nan")


# --- сетка калибровки (спецификация, V1b-4.4-4.6) -----------------------------
def g_ref_value() -> float:
    """g_ref = 1/(v_th - v_0) в 1/мВ, из констант [2]."""
    from model import default_params as dp
    return 1.0 / float((dp["v_th"] - dp["v_0"]) / (0.001 * 1.0))


def grid_stage1() -> list[tuple[float, float]]:
    scales = [2.0 ** k for k in range(-6, 3)]                 # 9 значений
    gains = [0.0] + [10.0 ** (k / 2.0) for k in range(-4, 5)]  # 10 значений
    return [(sc, g) for sc in scales for g in gains]


def grid_stage2(stage1: list[dict]) -> list[tuple[float, float]]:
    """Уточняющая сетка (V1b-4.4).

    Ограничивающий прямоугольник допустимых точек стадии 1, расширенный на один
    шаг стадии 1 в каждую сторону; шаги 2^(1/4) по масштабу и 10^(1/8) по g_apl.
    Границы задаются правилом, записанным до прогона, поэтому выход за диапазон
    стадии 1 законен и режимом C не является.
    """
    ok = [p for p in stage1 if admissible(p)]
    if not ok:
        return []
    sc = [p["pn_kc_scale"] for p in ok]
    gs = [p["g_apl_rel"] for p in ok if p["g_apl_rel"] > 0]
    s_lo, s_hi = min(sc) / 2.0, max(sc) * 2.0          # шаг стадии 1 по масштабу
    if gs:
        g_lo, g_hi = min(gs) / (10 ** 0.5), max(gs) * (10 ** 0.5)
    else:
        g_lo, g_hi = 0.0, 10 ** -2.0

    out_s, x = [], s_lo
    while x <= s_hi * 1.0001:
        out_s.append(x)
        x *= 2 ** 0.25
    out_g, y = [], g_lo
    if g_lo <= 0:
        out_g.append(0.0)
        y = 10 ** -2.0
    while y <= g_hi * 1.0001:
        out_g.append(y)
        y *= 10 ** 0.125
    # ноль торможения остаётся в сетке: он есть в стадии 1 и его исключение
    # сузило бы пространство после просмотра результата
    if 0.0 not in out_g and any(p["g_apl_rel"] == 0 for p in ok):
        out_g.insert(0, 0.0)
    return [(a, b) for a in out_s for b in out_g]


def evaluate_point(neurons, con, odors, *, pn_kc_scale, g_apl_rel, seed_base) -> dict:
    """Величины в точке сетки на заданном наборе запахов."""
    g_abs = g_apl_rel * g_ref_value()
    by = run_panel(neurons, con, odors, pn_kc_scale=pn_kc_scale, g_apl=g_abs,
                   seed_base=seed_base, extra_windows=((0, 2000),))
    f = {o: kc_fraction(by[o], neurons)["f"] for o in odors}
    vals = list(f.values())
    return {"pn_kc_scale": pn_kc_scale, "g_apl_rel": g_apl_rel,
            "f_by_odor": f, "f_mean": float(np.mean(vals)),
            "f_max": float(np.max(vals)),
            "s_ab": spikes_per_response(by, neurons, "KCab"),
            "s_apbp": spikes_per_response(by, neurons, "KCa'b'"),
            "s_g": spikes_per_response(by, neurons, "KCg"),
            "_by_odor": by}


def admissible(pt: dict) -> bool:
    """V1b-4.5: точка допустима по доле отвечающих на калибровочном наборе."""
    return (F_BAND[0] <= pt["f_mean"] <= F_BAND[1]) and pt["f_max"] <= F_MAX


def choose_point(points: list[dict]) -> dict | None:
    """V1b-4.6: лексикографический выбор среди допустимых точек."""
    ok = [p for p in points if admissible(p)]
    if not ok:
        return None
    band = [p for p in ok if 0.045 <= p["f_mean"] <= 0.055]
    if not band:
        return min(ok, key=lambda p: abs(p["f_mean"] - 0.05))
    defined = [p for p in band if p["s_ab"] == p["s_ab"]]
    pool = defined or band
    best = min(p["s_ab"] - SPIKES_AB_TARGET if p["s_ab"] == p["s_ab"] else float("inf")
               for p in pool)
    key = lambda p: (abs(p["s_ab"] - SPIKES_AB_TARGET) if p["s_ab"] == p["s_ab"]
                     else float("inf"), p["g_apl_rel"], p["pn_kc_scale"])
    return min(pool, key=key)


# --- регрессионный контроль (спецификация, v0.11, критерий (а) и (б)) ---------
def regression_substrate(neurons: pd.DataFrame, con: pd.DataFrame) -> dict:
    """При выключенных подменах субстрат обязан совпасть с V1a побитово."""
    role = dict(zip(neurons.root_id, neurons.mb_role))
    kept, n_masked = apply_dan_mask(con, role)
    off = con  # подмены выключены: маска не применяется
    same_edges = len(off) == len(con) and n_masked > 0
    return {"n_edges_total": int(len(con)),
            "n_edges_with_mask": int(len(kept)),
            "n_masked": int(n_masked),
            "substitutions_off_identical": bool(same_edges),
            "expected_masked": 49316,
            "mask_count_ok": bool(n_masked == 49316)}


def config_hash(cfg: dict, exclude: tuple = ()) -> str:
    """Хэш конфига с исключением объявленных ключей (критерий (б))."""
    d = {k: v for k, v in sorted(cfg.items()) if k not in exclude}
    return hashlib.sha256(json.dumps(d, sort_keys=True,
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


DECLARED_KEYS = ("dan_mask", "graded_apl", "odor_input", "pn_kc_scale", "g_apl_rel")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--odor", default="pentyl acetate")
    ap.add_argument("--scale", type=float, default=1.0, help="pn_kc_scale")
    ap.add_argument("--gapl", type=float, default=1.0, help="g_apl в единицах g_ref")
    ap.add_argument("--no-dan-mask", action="store_true")
    ap.add_argument("--no-graded-apl", action="store_true")
    ap.add_argument("--codegen", default="numpy", choices=["numpy", "cython"],
                    help="генератор кода Brian 2; cython требует окружения MSVC")
    ap.add_argument("--cache", default="",
                    help="каталог кэша cython (свой на процесс при параллельном запуске)")
    ap.add_argument("--evaluate", action="store_true",
                    help="оценочный набор: все четыре критерия в заданной точке")
    ap.add_argument("--grid", type=int, default=0,
                    help="калибровка: сколько точек стадии 1 прогнать (0 - не запускать)")
    ap.add_argument("--regression", action="store_true",
                    help="контроль (а) и (б): субстрат и хэши конфигов")
    ap.add_argument("--trials", action="store_true",
                    help="полный протокол: 6 предъявлений и доля отвечающих KC")
    ap.add_argument("--self-test", action="store_true",
                    help="короткий прогон: собрать сеть и проверить, что она считает")
    a = ap.parse_args()

    from brian2 import prefs, ms
    prefs.codegen.target = a.codegen
    if a.codegen == "cython" and a.cache:
        prefs.codegen.runtime.cython.cache_dir = a.cache
    from model import default_params as dp

    neurons, con = load_substrate()
    rates = odor_rates(a.odor, neurons)
    # g_ref = 1/(v_th - v_0): гейн, при котором APL при деполяризации, равной
    # порогу KC, даёт одну синаптическую единицу за такт (V1b-4.3)
    g_ref = 1.0 / float((dp["v_th"] - dp["v_0"]) / (0.001 * 1.0))  # в 1/мВ
    g_abs = a.gapl * g_ref

    if a.regression:
        r = regression_substrate(neurons, con)
        cfg_off = {"dan_mask": False, "graded_apl": False, "odor_input": False,
                   "pn_kc_scale": 1.0, "g_apl_rel": 0.0, "substrate": "flywire_630",
                   "w_syn_mV": 0.275, "dt_ms": 0.1}
        cfg_on = dict(cfg_off, dan_mask=True, graded_apl=True, odor_input=True,
                      pn_kc_scale=a.scale, g_apl_rel=a.gapl)
        h_off = config_hash(cfg_off, DECLARED_KEYS)
        h_on = config_hash(cfg_on, DECLARED_KEYS)
        print("контроль (а), субстрат:")
        print("   рёбер всего %d, снимается маской %d (ожидается 49 316: %s)"
              % (r["n_edges_total"], r["n_masked"], "да" if r["mask_count_ok"] else "НЕТ"))
        print("   при выключенных подменах субстрат тождественен: %s"
              % ("да" if r["substitutions_off_identical"] else "НЕТ"))
        print("контроль (б), конфиги с исключением объявленных ключей:")
        print("   выключено: %s" % h_off[:16])
        print("   включено:  %s" % h_on[:16])
        print("   равны: %s" % ("да" if h_off == h_on else "НЕТ"))
        print()
        print("Тождество спайковых поездов на 33 условиях проверяется прогоном")
        print("v1a_subcircuit.py против замороженных артефактов results/v1a/runs.")
        return 0 if (r["mask_count_ok"] and h_off == h_on) else 1

    if a.evaluate:
        pt = evaluate_point(neurons, con, PANEL_EVAL, pn_kc_scale=a.scale,
                            g_apl_rel=a.gapl, seed_base=SEED_EVAL)
        by = pt.pop("_by_odor")
        r = overlap_pearson(by, neurons)
        ov = check_overlap(r)
        print("оценочный набор, %d запахов, pn_kc_scale %.4g, g_apl %.4g g_ref"
              % (len(PANEL_EVAL), a.scale, a.gapl))
        print("доля отвечающих KC: среднее %.4f, максимум %.4f  (полоса %s, потолок %.2f)"
              % (pt["f_mean"], pt["f_max"], F_BAND, F_MAX))
        print("перекрытие: r(PA,BA) %.3f, r(PA,EL) %.3f, r(BA,EL) %.3f"
              % (ov["r_PA_BA"], ov["r_PA_EL"], ov["r_BA_EL"]))
        print("   порядок %s, разделение %.3f (>= %.2f) %s, потолок %s"
              % (ov["order_ok"], ov["separation"], OVERLAP_SEP_MIN,
                 ov["separation_ok"], ov["ceiling_ok"]))
        print("спайков за ответ: KCab %.2f (эталон %.1f), KCa'b' %.2f, KCg %.2f"
              % (pt["s_ab"], SPIKES_AB_TARGET, pt["s_apbp"], pt["s_g"]))
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "evaluate.json").write_text(
            json.dumps({**pt, "overlap": {str(k): v for k, v in r.items()},
                        "overlap_check": ov}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        return 0

    if a.grid:
        pts = grid_stage1()[:a.grid]
        print("стадия 1: %d точек из %d" % (len(pts), len(grid_stage1())))
        done = []
        for k, (sc, g) in enumerate(pts, 1):
            pt = evaluate_point(neurons, con, PANEL_CAL, pn_kc_scale=sc,
                                g_apl_rel=g, seed_base=SEED_CAL)
            pt.pop("_by_odor")
            done.append(pt)
            print("  [%d/%d] scale %.4g, g %.4g -> f̄ %.4f, f_max %.4f, s_ab %.2f%s"
                  % (k, len(pts), sc, g, pt["f_mean"], pt["f_max"], pt["s_ab"],
                     "  ДОПУСТИМА" if admissible(pt) else ""))
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "calibration_stage1.json").write_text(
            json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")
        best = choose_point(done)
        print()
        print("выбрана: %s" % ({k: best[k] for k in ("pn_kc_scale", "g_apl_rel",
                                                     "f_mean", "s_ab")}
                               if best else "нет допустимых точек (исход FAIL-CAL)"))
        return 0

    if a.trials:
        seeds = [20260908 + i for i in range(1, N_TRIALS + 1)]
        res = run_odor(neurons, con, a.odor, pn_kc_scale=a.scale, g_apl=g_abs,
                       seeds=seeds, dan_mask=not a.no_dan_mask,
                       graded_apl=not a.no_graded_apl)
        kf = kc_fraction(res, neurons)
        print("запах %r, pn_kc_scale %.4g, g_apl %.4g g_ref, зёрна %s"
              % (a.odor, a.scale, a.gapl, seeds))
        print("доля отвечающих KC (>=1 спайк, >=3 из 6): %.4f  (знаменатель %d)"
              % (kf["f"], kf["n_kc_denominator"]))
        print("таблица чувствительности:")
        for key in sorted(kf["sensitivity"]):
            print("   %-14s %.4f" % (key, kf["sensitivity"][key]))
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / ("kc_fraction_%s.json" % a.odor.replace(" ", "_"))).write_text(
            json.dumps({"odor": a.odor, "pn_kc_scale": a.scale,
                        "g_apl_in_gref": a.gapl, "seeds": seeds, **kf},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    net, mon, core_ids, st = build(
        neurons, con, pn_kc_scale=a.scale, g_apl=g_abs,
        dan_mask=not a.no_dan_mask, graded_apl=not a.no_graded_apl,
        rates=rates)

    print("подсхема: ядро %d, APL %d (градуальный: %s)"
          % (st["n_core"], st["n_apl"], not a.no_graded_apl))
    print("рёбра ядра %d, снято маской DAN %d, вход APL %d, выход APL %d"
          % (st["n_edges_core"], st["n_masked"], st["n_apl_in"], st["n_apl_out"]))
    print("стимулируемых PN %d, запах %r, pn_kc_scale %.4g, g_apl %.4g (%.4g g_ref)"
          % (st["n_poisson"], a.odor, a.scale, g_abs, a.gapl))

    dur = (T_ON_MS + T_PULSE_MS + 200) if a.self_test else T_RUN_MS
    net.run(dur * ms, report="text", report_period=30 * 1000 * ms)

    df = measure(mon, core_ids, neurons,
                 window_ms=dur - T_ON_MS)
    kc = df[df.role == "Kenyon_Cell"]
    resp = int((kc.n_spikes >= KC_SPIKE_THRESHOLD).sum())
    print()
    print("спайков всего: %d" % int(df.n_spikes.sum()))
    print("KC с >=1 спайком: %d из %d (%.1f %%)"
          % (resp, len(kc), 100.0 * resp / len(kc)))
    for r in ("MBON", "DAN", "PN"):
        s = df[df.role == r]
        print("%-4s: активных %d из %d, средняя частота активных %.1f Гц"
              % (r, int((s.n_spikes > 0).sum()), len(s),
                 s.loc[s.n_spikes > 0, "rate_hz"].mean() if (s.n_spikes > 0).any() else 0.0))

    if not a.self_test:
        OUT.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT / ("rates_%s.tsv" % a.odor.replace(" ", "_")),
                  sep="\t", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
