# -*- coding: utf-8 -*-
"""V1a of the validation ladder: mushroom-body subcircuit against the full model [2].

Three parts, all per specification v0.9, sections 3а, 3г and 6.

1. Substrate identity. Subcircuit edges and their weights match the edges of the
   full model restricted to the same set of neurons. The tolerance is zero.

2. Response identity under replayed external input. The full model [2] is run
   with PN stimulation and records the spikes of all neurons. Then the
   subcircuit core (KC, MBON, DAN, APL) is run, and each of its neurons receives
   the spikes of all external presynaptic partners, taken from the full model's
   recording, with weights from the same connectome. Projection neurons are also
   replayed, not stimulated. The subcircuit is thereby deterministic; the
   threshold is at least 99% of core neurons with a spike train matching to the
   integration step, and for the rest a discrepancy of at most one spike.

3. Decoder checks. The first is sign consistency. The second is the index on
   naive inputs within a tolerance of 0.01. The third is not run: it depends on
   the source of PN patterns, deferred to V1b (specification, section 3а).

Run:  .venv/Scripts/python.exe v1a_subcircuit.py [--quick]
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

# Odor sets and the second decoder check criterion — specification, section 3г.
# Fifteen draws of 30 uniglomerular ALPN of the right hemisphere: ten
# calibration, five evaluation. Seeds are declared before the run; the odor1..odor4
# draws of the v0.8 run are not reused — they have already been observed.
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

# Criterion thresholds. The constant 0.01 is inherited from v0.7 (a quarter of the
# effect size 0.038 per [29]) and was not revised; what changed is the quantity it bounds.
NULL_LOCATION_TOL = 0.01       # condition 1: |mu_null|
ZBAR_TOL = 2.0                 # condition 2: |z| of the mean of five evaluation odors
NULL_SCALE_TOL = 0.0095        # condition 3: sigma_null, a quarter of 0.038
EFFECT_SIZE = 0.038            # index shift from an 80% depression of one type [29]

PASS_FRACTION = 0.99           # threshold on the fraction of matching trains
PASS_MAX_EXTRA_SPIKES = 1

# Decoder groups per table 1 [30]; type weight = 1/N within the group.
AVOID = ["MBON%02d" % i for i in (1, 2, 3, 4, 5, 6)]
APPROACH = ["MBON%02d" % i for i in (9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)]


# --- part 1: substrate identity --------------------------------------------
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

    # reindexing: subcircuit indices must be a bijection onto the completeness order
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


# --- PN input ------------------------------------------------------------------
def pn_sets(neurons: pd.DataFrame) -> dict[str, list[int]]:
    """Input sets. Draws of one size, seeds declared before the run."""
    pn = neurons[(neurons.mb_role == "PN") & (neurons.cell_sub_class == "uniglomerular")]
    right = np.array(sorted(pn[pn.side == "right"].root_id.astype("int64").tolist()))
    sets: dict[str, list[int]] = {"uPN_right": right.tolist(), "silence": []}
    for odor, seed in ODOR_SEEDS.items():
        rng = np.random.default_rng(seed)
        sets[odor] = sorted(rng.choice(right, size=N_UPN30, replace=False).tolist())
    return sets


# --- part 2: runs ---------------------------------------------------------
def n_proc_by_memory() -> int:
    """Number of workers based on available memory (same reason as in v0_sugar.py)."""
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
    """Run of the full model [2] with the fork's code, unchanged."""
    from brian2 import Hz, ms
    from model import run_exp, default_params

    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / ("%s.parquet" % name)
    if path.exists():
        print("   %s: already computed" % name)
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
    """Subcircuit core with replayed external input. Returns spikes.

    Any presynaptic partner of a core neuron that is not itself in the core is
    considered external — PN included. Its spikes are taken from the full
    model's recording of the same trial, so no randomness remains in the run.
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
    print("   core %d neurons, internal edges %d, external partners %d, external edges %d"
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
        print("      trial %d: %d spikes" % (trial, mon.num_spikes))
    return (pd.concat(rows, ignore_index=True) if rows
            else pd.DataFrame(columns=["t", "trial", "flywire_id"]))


def compare(full: pd.DataFrame, repl: pd.DataFrame, core_ids: list[int], dt_s: float) -> dict:
    """Comparison of core spike trains: full model against subcircuit."""
    core = set(core_ids)
    f = full[full.flywire_id.isin(core)]
    # trials are taken from the config, not from the data: in the silence
    # control there are no spikes at all, and "no discrepancies" must mean a
    # comparison, not an empty loop
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


# --- part 3: decoder ---------------------------------------------------------
def type_rates(spikes: pd.DataFrame, neurons: pd.DataFrame, n_run: int, t_s: float) -> pd.Series:
    """Mean rate by MBON type (Hz), averaged over neurons and trials."""
    mbon = neurons[neurons.mb_role == "MBON"][["root_id", "hemibrain_type"]]
    s = spikes.merge(mbon, left_on="flywire_id", right_on="root_id", how="inner")
    cnt = s.groupby(["hemibrain_type", "flywire_id"]).size()
    per_neuron = cnt / (n_run * t_s)
    sizes = mbon.groupby("hemibrain_type").size()
    total = per_neuron.groupby("hemibrain_type").sum()
    return (total / sizes).reindex(sizes.index).fillna(0.0)


def index_from_rates(rates: pd.Series, norm: pd.Series) -> float | None:
    """Odor index per specification section 6; type weight = 1/N within the group.

    A type whose normalizing constant is undefined (rate on the calibration set
    is zero) is excluded from the group and does not count toward N (v0.8
    revision, section 3в). If no types remain in the group, the index is
    undefined.
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

    print("V1a, subcircuit config %s" % stats["config_sha256"][:16])

    print("\n[1] substrate identity")
    report["substrate"] = check_substrate(neurons)
    for k, v in report["substrate"].items():
        print("   %-22s %s" % (k, v))

    core_ids = sorted(neurons[neurons.mb_role != "PN"].root_id.astype("int64").tolist())
    sets = pn_sets(neurons)
    print("\ninput sets: " + ", ".join(
        "%s %d" % (k, len(v)) for k, v in sets.items() if v))
    ov = set(sets[CALIB_ODORS[0]]) & set(sets[EVAL_ODORS[0]])
    print("odors: %d calibration, %d evaluation of %d PN each; overlap of %s and %s: %d"
          % (len(CALIB_ODORS), len(EVAL_ODORS), N_UPN30,
             CALIB_ODORS[0], EVAL_ODORS[0], len(ov)))

    from brian2 import defaultclock
    dt_s = float(defaultclock.dt)

    print("\n[2] runs and train comparison")
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
        # the decoder reads the subcircuit's responses, not the full model's
        spikes_by_cond[name] = repl
        res = compare(full, repl, core_ids, dt_s)
        res["wall_s"] = round(time.time() - t0, 1)
        report["conditions"][name] = res
        print("   matched %d of %d (%.4f), max spike-count discrepancy %d -> %s"
              % (res["n_exact"], res["n_compared"], res["fraction_exact"],
                 res["max_spike_count_diff"], "PASSED" if res["pass"] else "FAILED"))

    print("\n[3] decoder")
    report["decoder"] = decoder_checks(spikes_by_cond, neurons)
    for k, v in report["decoder"].items():
        if k == "check2_by_freq":
            for f, d in v.items():
                print("   --- %s ---" % f)
                print("       null distribution (10 calibration, leave-one-out): "
                      "mu %+.5f, sigma %.5f | median %+.5f, MAD %.5f"
                      % (d["mu_null"], d["sigma_null"], d["median_null"], d["mad_null"]))
                print("       indices of 5 evaluation odors: %s"
                      % ", ".join("%+.4f" % v for v in d["evaluation_indices"]))
                print("       z of each: %s | z of mean: %+.3f"
                      % (", ".join("%+.2f" % z for z in d["z_each"]), d["zbar"]))
                print("       separation 0.038/sigma = %s" % d["separation_effect_over_sigma"])
                print("       condition 1 |mu|<=%.3f: %s | condition 2 |z|<=%.1f: %s | "
                      "condition 3 sigma<=%.4f: %s -> %s"
                      % (NULL_LOCATION_TOL, d["cond1_location_pass"], ZBAR_TOL,
                         d["cond2_zbar_pass"], NULL_SCALE_TOL, d["cond3_scale_pass"],
                         "PASSED" if d["pass"] else "FAILED"))
        elif k == "protocol":
            print("   %-26s calibration %d, evaluation %d odors"
                  % (k, len(v["calibration_odors"]), len(v["evaluation_odors"])))
        else:
            print("   %-26s %s" % (k, v))

    ok_sub = report["substrate"]["pass"]
    ok_resp = all(c["pass"] for c in report["conditions"].values())
    ok_dec = report["decoder"]["pass"]
    report["v1a_pass"] = bool(ok_sub and ok_resp and ok_dec)
    (OUT / "v1a_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nV1a: substrate %s, response %s, decoder %s -> %s"
          % (ok_sub, ok_resp, ok_dec, "PASSED" if report["v1a_pass"] else "FAILED"))
    return 0


def decoder_checks(spikes: dict[str, pd.DataFrame], neurons: pd.DataFrame) -> dict:
    """First and second decoder checks of specification section 6.

    The second check is run by the procedure of section 3г: ten calibration
    odors set the normalizing constants and, by leave-one-out, the null
    distribution; five held-out odors are checked against it. Three
    conditions — null location, consistency, and separation — must hold at
    both declared frequencies.
    """
    out: dict = {}
    t_s = T_RUN_MS / 1000.0

    # first check: sign consistency. Groups are activated artificially.
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

    # The second check is the procedure of section 3г, fixed before the
    # evaluation data were obtained. The null distribution is estimated on the
    # calibration set by leave-one-out: the constants for odor i are
    # recomputed from the other nine, and its index is computed by the same
    # code as the evaluation odors.
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
        out["check2_note"] = "no complete set of conditions at any frequency"
    out["check3_note"] = ("not run: depends on the source of PN patterns, "
                          "deferred to V1b (specification, section 3а)")
    out["pass"] = bool(sign_ok and out["check2_naive_pass"])
    return out


if __name__ == "__main__":
    raise SystemExit(main())
