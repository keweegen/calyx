# -*- coding: utf-8 -*-
"""V0 of the validation ladder: reproduction of the published result of Shiu et al. [2].

Claim under test (fig. 1 of the paper): activation of the sugar sensory neurons (GRN)
excites the motor neuron MN9, which drives proboscis extension, and the MN9 rate
increases with input frequency.

This is stage V0: it checks fork correctness and agreement with the published model
before plasticity is introduced into the testbed. Weights are not changed here.

Run:  .venv/Scripts/python.exe v0_sugar.py [--quick]
"""
import sys, time, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE / "Drosophila_brain_model"
sys.path.insert(0, str(REPO))

from brian2 import Hz, ms
from model import run_exp, default_params
import utils as utl

CONFIG = {
    "path_comp": str(REPO / "2023_03_23_completeness_630_final.csv"),
    "path_con": str(REPO / "2023_03_23_connectivity_630_final.parquet"),
}

# FlyWire v630. Sugar GRN and MN9 IDs are from figures.ipynb of repository [2].
NEU_SUGAR = [
    720575940624963786, 720575940630233916, 720575940637568838,
    720575940638202345, 720575940617000768, 720575940630797113,
    720575940632889389, 720575940621754367, 720575940621502051,
    720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543,
    720575940632425919, 720575940633143833, 720575940612670570,
    720575940628853239, 720575940629176663, 720575940611875570,
]
ID_MN9 = 720575940660219265

QUICK = "--quick" in sys.argv
FREQS = [20, 100, 200] if QUICK else [20, 60, 100, 140, 180, 200]
N_RUN = 5 if QUICK else 30
T_RUN = 1000 * ms

# Peak memory per worker when building the full network (127,400 neurons),
# measured on this machine: working set ~1.95 GB, peak at synapse resize ~2.2 GB.
GB_PER_WORKER = 2.5
RAM_RESERVE_GB = 4.0


def n_proc_by_memory() -> int:
    """Number of workers based on available memory, not core count.

    run_exp with n_proc=-1 spawns one process per core, and each builds its own
    copy of the network. On 12 cores and 32 GB this gives MemoryError: bad allocation
    during synapse resize (measured: peak 31.6 GB of 31.8 GB). The limit removes the cause.
    """
    import ctypes

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
        return 4  # unknown platform: conservative

    import os
    by_mem = int(max(1, (avail - RAM_RESERVE_GB) // GB_PER_WORKER))
    n = min(by_mem, os.cpu_count() or 1)
    print("memory: %.1f GB available -> n_proc=%d (%.1f GB/worker, %.1f GB reserved)"
          % (avail, n, GB_PER_WORKER, RAM_RESERVE_GB))
    return n


def main() -> int:
    out = HERE / "results" / ("v0_quick" if QUICK else "v0_sugar")
    out.mkdir(parents=True, exist_ok=True)

    params = dict(default_params)
    params["t_run"] = T_RUN
    params["n_run"] = N_RUN

    n_proc = n_proc_by_memory()
    print("V0: sugar GRN -> MN9, freqs=%s, n_run=%d, t_run=%s" % (FREQS, N_RUN, T_RUN))
    t0 = time.time()
    for f in FREQS:
        params["r_poi"] = f * Hz
        run_exp(
            exp_name="sugarR_%dHz" % f,
            neu_exc=NEU_SUGAR,
            path_res=str(out),
            params=params,
            n_proc=n_proc,
            **CONFIG,
        )
    print("simulations done in %.0f s" % (time.time() - t0))

    paths = [str(out / ("sugarR_%dHz.parquet" % f)) for f in FREQS]
    df_spike = utl.load_exps(paths)
    df_rate, df_rate_std = utl.get_rate(df_spike, t_run=T_RUN, n_run=N_RUN)
    df_rate = df_rate.fillna(0)
    df_rate_std = df_rate_std.fillna(0)

    df_rate.to_csv(out / "rate.csv")
    df_rate_std.to_csv(out / "rate_std.csv")

    print("\nMN9 (%d) firing rate by sugar GRN input frequency:" % ID_MN9)
    rows = []
    for f in FREQS:
        col = "sugarR_%dHz" % f
        r = float(df_rate.loc[ID_MN9, col]) if ID_MN9 in df_rate.index else 0.0
        s = float(df_rate_std.loc[ID_MN9, col]) if ID_MN9 in df_rate_std.index else 0.0
        rows.append({"input_hz": f, "mn9_hz": r, "mn9_std": s})
        print("  %3d Hz input  ->  MN9 %6.2f +- %.2f Hz" % (f, r, s))

    (out / "mn9.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    rates = [r["mn9_hz"] for r in rows]
    monotonic = all(b >= a for a, b in zip(rates, rates[1:]))
    responds = rates[-1] > 0
    print("\nMN9 responds at highest input: %s" % responds)
    print("MN9 rate non-decreasing in input frequency: %s" % monotonic)
    print("V0 %s" % ("PASS" if (responds and monotonic) else "CHECK — see rates above"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
