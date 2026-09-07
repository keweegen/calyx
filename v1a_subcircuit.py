# -*- coding: utf-8 -*-
"""V1a лестницы валидации: подсхема грибовидного тела против полной модели [2].

Три части, все по спецификации v0.9, разделы 3а, 3г и 6.

1. Тождество субстрата. Рёбра подсхемы и их веса совпадают с рёбрами полной
   модели, ограниченными на то же множество нейронов. Допуск нулевой.

2. Тождество отклика при воспроизведённом внешнем входе. Полная модель [2]
   прогоняется со стимуляцией PN и записывает спайки всех нейронов. Затем
   прогоняется ядро подсхемы (KC, MBON, DAN, APL), и каждому его нейрону
   подаются спайки всех внешних пресинаптических партнёров, взятые из записи
   полной модели, с весами из того же коннектома. Проекционные нейроны тоже
   воспроизводятся, а не стимулируются. Подсхема при этом детерминирована;
   порог — не менее 99 % нейронов ядра с совпавшим спайковым поездом с
   точностью до шага интегрирования, у остальных расхождение не более
   одного спайка.

3. Проверки декодера. Первая — согласованность знака. Вторая — индекс на
   наивных входах в допуске 0,01. Третья не выполняется: она зависит от
   источника PN-паттернов, отложенного до V1b (спецификация, раздел 3а).

Запуск:  .venv/Scripts/python.exe v1a_subcircuit.py [--quick]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE / "Drosophila_brain_model"
SUB = HERE / "data" / "mb_subcircuit"
OUT = HERE / "results" / "v1a"
RUNS = OUT / "runs"
sys.path.insert(0, str(REPO))

PATH_COMP = REPO / "2023_03_23_completeness_630_final.csv"
PATH_CON = REPO / "2023_03_23_connectivity_630_final.parquet"

QUICK = "--quick" in sys.argv
N_RUN = 3 if QUICK else 10
T_RUN_MS = 1000

# Наборы запахов и критерий второй проверки декодера — спецификация, раздел 3г.
# Пятнадцать выборок по 30 унигломерулярных ALPN правого полушария: десять
# калибровочных, пять оценочных. Зёрна объявлены до прогона; выборки odor1..odor4
# прогона v0.8 не переиспользуются — они уже наблюдались.
FREQS = [20, 100]
SEED_UPN30 = 20260907
N_UPN30 = 30
CALIB_ODORS = ["cal%02d" % k for k in range(1, 11)]
EVAL_ODORS = ["evl%02d" % k for k in range(1, 6)]
ODOR_SEEDS = ({o: SEED_UPN30 + 100 + k for k, o in enumerate(CALIB_ODORS, 1)}
              | {o: SEED_UPN30 + 200 + k for k, o in enumerate(EVAL_ODORS, 1)})

CONDITIONS = (
    [("uPN_right", 20), ("uPN_right", 100), ("silence", 0)]
    + [(o, f) for o in CALIB_ODORS + EVAL_ODORS for f in FREQS]
    if not QUICK else
    [(o, 100) for o in CALIB_ODORS[:3] + EVAL_ODORS[:2]] + [("silence", 0)]
)

# Пороги критерия. Константа 0,01 унаследована из v0.7 (четверть от эффекта 0,038
# по [29]) и не пересматривалась; изменилась величина, которую она ограничивает.
NULL_LOCATION_TOL = 0.01       # условие 1: |mu_null|
ZBAR_TOL = 2.0                 # условие 2: |z| среднего пяти оценочных запахов
NULL_SCALE_TOL = 0.0095        # условие 3: sigma_null, четверть от 0,038
EFFECT_SIZE = 0.038            # сдвиг индекса при депрессии одного типа на 80 % [29]

PASS_FRACTION = 0.99           # порог доли совпавших поездов
PASS_MAX_EXTRA_SPIKES = 1

# Группы декодера по таблице 1 [30]; вес типа = 1/N внутри группы.
AVOID = ["MBON%02d" % i for i in (1, 2, 3, 4, 5, 6)]
APPROACH = ["MBON%02d" % i for i in (9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)]


# --- часть 1: тождество субстрата --------------------------------------------
def check_substrate(neurons: pd.DataFrame) -> dict:
    con = pd.read_parquet(PATH_CON)
    sel = set(neurons.root_id.astype("int64"))
    want = con[con.Presynaptic_ID.isin(sel) & con.Postsynaptic_ID.isin(sel)]
    got = pd.read_parquet(SUB / "connectivity.parquet")

    key = ["Presynaptic_ID", "Postsynaptic_ID"]
    a = want.set_index(key)["Excitatory x Connectivity"].sort_index()
    b = got.set_index(key)["Excitatory x Connectivity"].sort_index()
    same_index = a.index.equals(b.index)
    n_diff = int((~a.eq(b)).sum()) if same_index else -1

    # перенумерация: индексы подсхемы должны быть биекцией на порядок completeness
    comp = pd.read_csv(SUB / "completeness.csv", index_col=0)
    idx = {fid: k for k, fid in enumerate(comp.index.astype("int64"))}
    bad_i = int((got.Presynaptic_ID.map(idx) != got.Presynaptic_Index).sum())
    bad_j = int((got.Postsynaptic_ID.map(idx) != got.Postsynaptic_Index).sum())

    res = {
        "n_edges_expected": int(len(want)),
        "n_edges_got": int(len(got)),
        "edge_sets_equal": bool(same_index),
        "n_weight_mismatch": n_diff,
        "n_bad_pre_index": bad_i,
        "n_bad_post_index": bad_j,
    }
    res["pass"] = bool(same_index and n_diff == 0 and bad_i == 0 and bad_j == 0
                       and len(want) == len(got))
    return res


# --- вход PN ------------------------------------------------------------------
def pn_sets(neurons: pd.DataFrame) -> dict[str, list[int]]:
    """Наборы входов. Выборки одного размера, зёрна объявлены до прогона."""
    pn = neurons[(neurons.mb_role == "PN") & (neurons.cell_sub_class == "uniglomerular")]
    right = np.array(sorted(pn[pn.side == "right"].root_id.astype("int64").tolist()))
    sets: dict[str, list[int]] = {"uPN_right": right.tolist(), "silence": []}
    for odor, seed in ODOR_SEEDS.items():
        rng = np.random.default_rng(seed)
        sets[odor] = sorted(rng.choice(right, size=N_UPN30, replace=False).tolist())
    return sets


# --- часть 2: прогоны ---------------------------------------------------------
def n_proc_by_memory() -> int:
    """Число воркеров по доступной памяти (та же причина, что в v0_sugar.py)."""
    import ctypes, os

    class MS(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]
    try:
        m = MS(); m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        avail = m.ullAvailPhys / 2**30
    except Exception:
        return 4
    return int(min(max(1, (avail - 4.0) // 2.5), os.cpu_count() or 1))


def run_full(name: str, exc: list[int], rate_hz: int) -> Path:
    """Прогон полной модели [2] кодом форка без изменений."""
    from brian2 import Hz, ms
    from model import run_exp, default_params

    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / ("%s.parquet" % name)
    if path.exists():
        print("   %s: уже посчитан" % name)
        return path
    params = dict(default_params)
    params["t_run"] = T_RUN_MS * ms
    params["n_run"] = N_RUN
    params["r_poi"] = rate_hz * Hz
    run_exp(exp_name=name, neu_exc=exc, path_res=str(RUNS),
            path_comp=str(PATH_COMP), path_con=str(PATH_CON),
            params=params, n_proc=n_proc_by_memory())
    return path


def run_replay(core_ids: list[int], full_spikes: pd.DataFrame) -> pd.DataFrame:
    """Ядро подсхемы с воспроизведённым внешним входом. Возвращает спайки.

    Внешним считается всякий пресинаптический партнёр нейрона ядра, не входящий
    в ядро, — в том числе PN. Его спайки берутся из записи полной модели того же
    трайла, поэтому случайности в прогоне не остаётся.
    """
    from brian2 import (NeuronGroup, Synapses, SpikeGeneratorGroup, SpikeMonitor,
                        Network, ms, mV, second, defaultclock)
    from model import default_params as dp

    con = pd.read_parquet(PATH_CON)
    core = set(core_ids)
    onto_core = con[con.Postsynaptic_ID.isin(core)]
    inner = onto_core[onto_core.Presynaptic_ID.isin(core)]
    outer = onto_core[~onto_core.Presynaptic_ID.isin(core)]
    ext_ids = sorted(set(outer.Presynaptic_ID.astype("int64")))
    print("   ядро %d нейронов, внутренних рёбер %d, внешних партнёров %d, внешних рёбер %d"
          % (len(core_ids), len(inner), len(ext_ids), len(outer)))

    ci = {f: k for k, f in enumerate(core_ids)}
    ei = {f: k for k, f in enumerate(ext_ids)}
    in_i = inner.Presynaptic_ID.map(ci).to_numpy()
    in_j = inner.Postsynaptic_ID.map(ci).to_numpy()
    in_w = inner["Excitatory x Connectivity"].to_numpy() * dp["w_syn"]
    ex_i = outer.Presynaptic_ID.map(ei).to_numpy()
    ex_j = outer.Postsynaptic_ID.map(ci).to_numpy()
    ex_w = outer["Excitatory x Connectivity"].to_numpy() * dp["w_syn"]

    rows = []
    for trial, spk in full_spikes.groupby("trial"):
        s = spk[spk.flywire_id.isin(ei)]
        gi = s.flywire_id.map(ei).to_numpy()
        gt = s["t"].to_numpy() * second

        neu = NeuronGroup(len(core_ids), model=dp["eqs"], method="linear",
                          threshold=dp["eq_th"], reset=dp["eq_rst"],
                          refractory="rfc", name="core", namespace=dp)
        neu.v = dp["v_0"]; neu.g = 0; neu.rfc = dp["t_rfc"]
        syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=dp["t_dly"], name="core_syn")
        syn.connect(i=in_i, j=in_j); syn.w = in_w
        gen = SpikeGeneratorGroup(len(ext_ids), gi, gt, name="ext")
        syn_e = Synapses(gen, neu, "w : volt", on_pre="g += w", delay=dp["t_dly"], name="ext_syn")
        syn_e.connect(i=ex_i, j=ex_j); syn_e.w = ex_w
        mon = SpikeMonitor(neu)
        net = Network(neu, syn, gen, syn_e, mon)
        net.run(T_RUN_MS * ms)

        for bi, ts in mon.spike_trains().items():
            if len(ts):
                rows.append(pd.DataFrame({"t": np.asarray(ts), "trial": trial,
                                          "flywire_id": core_ids[bi]}))
        print("      трайл %d: %d спайков" % (trial, mon.num_spikes))
    return (pd.concat(rows, ignore_index=True) if rows
            else pd.DataFrame(columns=["t", "trial", "flywire_id"]))


def compare(full: pd.DataFrame, repl: pd.DataFrame, core_ids: list[int], dt_s: float) -> dict:
    """Сравнение спайковых поездов ядра: полная модель против подсхемы."""
    core = set(core_ids)
    f = full[full.flywire_id.isin(core)]
    # трайлы берутся из конфига, а не из данных: при контроле тишины спайков нет
    # вовсе, и «нет расхождений» должно означать сравнение, а не пустой цикл
    trials = list(range(N_RUN))
    n_exact = n_total = 0
    worst = 0
    for tr in trials:
        a = f[f.trial == tr]; b = repl[repl.trial == tr]
        ga = {k: np.sort(np.round(v.to_numpy() / dt_s).astype(np.int64))
              for k, v in a.groupby("flywire_id")["t"]}
        gb = {k: np.sort(np.round(v.to_numpy() / dt_s).astype(np.int64))
              for k, v in b.groupby("flywire_id")["t"]}
        for nid in core:
            x = ga.get(nid, np.empty(0, dtype=np.int64))
            y = gb.get(nid, np.empty(0, dtype=np.int64))
            n_total += 1
            if len(x) == len(y) and np.array_equal(x, y):
                n_exact += 1
            else:
                worst = max(worst, abs(len(x) - len(y)))
    frac = n_exact / n_total if n_total else 0.0
    return {"n_compared": n_total, "n_exact": n_exact, "fraction_exact": frac,
            "max_spike_count_diff": int(worst),
            "pass": bool(frac >= PASS_FRACTION and worst <= PASS_MAX_EXTRA_SPIKES)}


# --- часть 3: декодер ---------------------------------------------------------
def type_rates(spikes: pd.DataFrame, neurons: pd.DataFrame, n_run: int, t_s: float) -> pd.Series:
    """Средняя частота по типу MBON (Гц), усреднённая по нейронам и трайлам."""
    mbon = neurons[neurons.mb_role == "MBON"][["root_id", "hemibrain_type"]]
    s = spikes.merge(mbon, left_on="flywire_id", right_on="root_id", how="inner")
    cnt = s.groupby(["hemibrain_type", "flywire_id"]).size()
    per_neuron = cnt / (n_run * t_s)
    sizes = mbon.groupby("hemibrain_type").size()
    total = per_neuron.groupby("hemibrain_type").sum()
    return (total / sizes).reindex(sizes.index).fillna(0.0)


def index_from_rates(rates: pd.Series, norm: pd.Series) -> float | None:
    """Индекс запаха по разделу 6 спецификации; вес типа = 1/N внутри группы.

    Тип, у которого нормировочная константа не определена (частота по
    калибровочному набору равна нулю), из группы исключается и в N не входит
    (правка v0.8, раздел 3в). Если в группе не осталось типов, индекс не
    определён.
    """
    def group(types: list[str]) -> float | None:
        usable = [t for t in types if t in rates.index and norm.get(t, 0) > 0]
        if not usable:
            return None
        return float(np.sum([rates[t] / norm[t] for t in usable]) / len(usable))
    av, ap = group(AVOID), group(APPROACH)
    if av is None or ap is None or av + ap == 0:
        return None
    return (av - ap) / (av + ap)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    neurons = pd.read_csv(SUB / "neurons.csv")
    stats = json.loads((HERE / "results" / "mb_subcircuit" / "subcircuit_stats.json")
                       .read_text(encoding="utf-8"))
    report: dict = {"config_sha256": stats["config_sha256"], "spec": "v0.9",
                    "n_run": N_RUN, "t_run_ms": T_RUN_MS, "quick": QUICK}

    print("V1a, конфиг подсхемы %s" % stats["config_sha256"][:16])

    print("\n[1] тождество субстрата")
    report["substrate"] = check_substrate(neurons)
    for k, v in report["substrate"].items():
        print("   %-22s %s" % (k, v))

    core_ids = sorted(neurons[neurons.mb_role != "PN"].root_id.astype("int64").tolist())
    sets = pn_sets(neurons)
    print("\nнаборы входов: " + ", ".join(
        "%s %d" % (k, len(v)) for k, v in sets.items() if v))
    ov = set(sets[CALIB_ODORS[0]]) & set(sets[EVAL_ODORS[0]])
    print("запахов: %d калибровочных, %d оценочных по %d PN; пересечение %s и %s: %d"
          % (len(CALIB_ODORS), len(EVAL_ODORS), N_UPN30,
             CALIB_ODORS[0], EVAL_ODORS[0], len(ov)))

    from brian2 import defaultclock
    dt_s = float(defaultclock.dt)

    print("\n[2] прогоны и сравнение поездов")
    report["conditions"] = {}
    spikes_by_cond: dict[str, pd.DataFrame] = {}
    for cond, rate in CONDITIONS:
        name = "%s_%dHz" % (cond, rate)
        print("  %s" % name)
        t0 = time.time()
        path = run_full(name, sets[cond], rate)
        full = pd.read_parquet(path)
        repl_path = RUNS / ("%s_replay.parquet" % name)
        repl = run_replay(core_ids, full)
        repl.to_parquet(repl_path, compression="brotli")
        # декодер читает ответы подсхемы, а не полной модели
        spikes_by_cond[name] = repl
        res = compare(full, repl, core_ids, dt_s)
        res["wall_s"] = round(time.time() - t0, 1)
        report["conditions"][name] = res
        print("   совпало %d из %d (%.4f), макс. расхождение спайков %d -> %s"
              % (res["n_exact"], res["n_compared"], res["fraction_exact"],
                 res["max_spike_count_diff"], "ПРОШЛА" if res["pass"] else "НЕ ПРОШЛА"))

    print("\n[3] декодер")
    report["decoder"] = decoder_checks(spikes_by_cond, neurons)
    for k, v in report["decoder"].items():
        if k == "check2_by_freq":
            for f, d in v.items():
                print("   --- %s ---" % f)
                print("       нулевое распределение (10 калибровочных, «оставь один»): "
                      "mu %+.5f, sigma %.5f | медиана %+.5f, MAD %.5f"
                      % (d["mu_null"], d["sigma_null"], d["median_null"], d["mad_null"]))
                print("       индексы 5 оценочных: %s"
                      % ", ".join("%+.4f" % v for v in d["evaluation_indices"]))
                print("       z каждого: %s | z среднего: %+.3f"
                      % (", ".join("%+.2f" % z for z in d["z_each"]), d["zbar"]))
                print("       разделимость 0,038/sigma = %s" % d["separation_effect_over_sigma"])
                print("       условие 1 |mu|<=%.3f: %s | условие 2 |z|<=%.1f: %s | "
                      "условие 3 sigma<=%.4f: %s -> %s"
                      % (NULL_LOCATION_TOL, d["cond1_location_pass"], ZBAR_TOL,
                         d["cond2_zbar_pass"], NULL_SCALE_TOL, d["cond3_scale_pass"],
                         "ПРОШЛА" if d["pass"] else "НЕ ПРОШЛА"))
        elif k == "protocol":
            print("   %-26s калибровка %d, оценка %d запахов"
                  % (k, len(v["calibration_odors"]), len(v["evaluation_odors"])))
        else:
            print("   %-26s %s" % (k, v))

    ok_sub = report["substrate"]["pass"]
    ok_resp = all(c["pass"] for c in report["conditions"].values())
    ok_dec = report["decoder"]["pass"]
    report["v1a_pass"] = bool(ok_sub and ok_resp and ok_dec)
    (OUT / "v1a_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nV1a: субстрат %s, отклик %s, декодер %s -> %s"
          % (ok_sub, ok_resp, ok_dec, "ПРОЙДЕНА" if report["v1a_pass"] else "НЕ ПРОЙДЕНА"))
    return 0


def decoder_checks(spikes: dict[str, pd.DataFrame], neurons: pd.DataFrame) -> dict:
    """Первая и вторая проверки декодера раздела 6 спецификации.

    Вторая проверка выполняется по процедуре раздела 3г: десять калибровочных
    запахов задают нормировочные константы и, методом «оставь один», нулевое
    распределение; пять отложенных проверяются против него. Три условия —
    положение нуля, согласованность и разделимость — должны выполниться на
    обеих объявленных частотах.
    """
    out: dict = {}
    t_s = T_RUN_MS / 1000.0

    # первая проверка: согласованность знака. Активируем группы искусственно.
    types = sorted(set(neurons[neurons.mb_role == "MBON"].hemibrain_type.dropna()))
    one = pd.Series(1.0, index=types)
    only_av = pd.Series([1.0 if t in AVOID else 0.0 for t in types], index=types)
    only_ap = pd.Series([1.0 if t in APPROACH else 0.0 for t in types], index=types)
    i_av = index_from_rates(only_av, one)
    i_ap = index_from_rates(only_ap, one)
    i_base = index_from_rates(one, one)
    out["sign_avoid_only"] = i_av
    out["sign_approach_only"] = i_ap
    out["index_at_baseline"] = i_base
    sign_ok = (i_av == 1.0 and i_ap == -1.0 and abs(i_base) < 1e-12)
    out["check1_sign_pass"] = bool(sign_ok)

    # Вторая проверка — процедура раздела 3г, зафиксирована до получения
    # оценочных данных. Нулевое распределение оценивается по калибровочному
    # набору методом «оставь один»: константы для запаха i пересчитываются по
    # остальным девяти, и его индекс считается тем же кодом, что оценочные.
    out["protocol"] = {
        "calibration_odors": CALIB_ODORS, "evaluation_odors": EVAL_ODORS,
        "seeds": ODOR_SEEDS, "n_pn": N_UPN30,
        "null_location_estimator": "mean", "null_scale_estimator": "sd(ddof=1)",
        "thresholds": {"null_location": NULL_LOCATION_TOL, "zbar": ZBAR_TOL,
                       "null_scale": NULL_SCALE_TOL},
        "effect_size_ref": EFFECT_SIZE,
    }
    out["check2_by_freq"] = {}
    for f in FREQS:
        cal_keys = ["%s_%dHz" % (o, f) for o in CALIB_ODORS]
        evl_keys = ["%s_%dHz" % (o, f) for o in EVAL_ODORS]
        if not all(k in spikes for k in cal_keys + evl_keys):
            continue
        cal_rates = [type_rates(spikes[k], neurons, N_RUN, t_s) for k in cal_keys]
        evl_rates = [type_rates(spikes[k], neurons, N_RUN, t_s) for k in evl_keys]
        norm_full = pd.concat(cal_rates, axis=1).mean(axis=1)

        null = []
        for i in range(len(cal_rates)):
            others = [r for j, r in enumerate(cal_rates) if j != i]
            null.append(index_from_rates(cal_rates[i],
                                         pd.concat(others, axis=1).mean(axis=1)))
        null_a = np.array([v for v in null if v is not None], dtype=float)
        mu = float(null_a.mean())
        sd = float(null_a.std(ddof=1))
        med = float(np.median(null_a))
        mad = float(1.4826 * np.median(np.abs(null_a - med)))

        evl = [index_from_rates(r, norm_full) for r in evl_rates]
        evl_a = np.array([v for v in evl if v is not None], dtype=float)
        zbar = float((evl_a.mean() - mu) / (sd / np.sqrt(len(evl_a)))) if sd > 0 else None
        z_each = [float((v - mu) / sd) for v in evl_a] if sd > 0 else []

        c1 = bool(abs(mu) <= NULL_LOCATION_TOL)
        c2 = bool(zbar is not None and abs(zbar) <= ZBAR_TOL)
        c3 = bool(sd <= NULL_SCALE_TOL)
        out["check2_by_freq"]["%dHz" % f] = {
            "n_types_with_defined_norm": int((norm_full > 0).sum()),
            "null_indices": [None if v is None else round(v, 6) for v in null],
            "evaluation_indices": [None if v is None else round(v, 6) for v in evl],
            "mu_null": round(mu, 6), "sigma_null": round(sd, 6),
            "median_null": round(med, 6), "mad_null": round(mad, 6),
            "zbar": None if zbar is None else round(zbar, 4),
            "z_each": [round(z, 4) for z in z_each],
            "separation_effect_over_sigma": round(EFFECT_SIZE / sd, 2) if sd > 0 else None,
            "cond1_location_pass": c1, "cond2_zbar_pass": c2, "cond3_scale_pass": c3,
            "pass": bool(c1 and c2 and c3),
        }
    checks = out["check2_by_freq"]
    out["check2_naive_pass"] = bool(checks) and all(v["pass"] for v in checks.values())
    if not checks:
        out["check2_note"] = "нет полного набора условий ни на одной частоте"
    out["check3_note"] = ("не выполняется: зависит от источника PN-паттернов, "
                          "отложенного до V1b (спецификация, раздел 3а)")
    out["pass"] = bool(sign_ok and out["check2_naive_pass"])
    return out


if __name__ == "__main__":
    raise SystemExit(main())
