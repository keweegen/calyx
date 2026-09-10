# -*- coding: utf-8 -*-
"""Diagnosing TerminatedWorkerError: tracking free memory during a real run_exp.

The parent launches a child process with the real run_exp and, every 0.5 s, samples
available physical memory. If the memory-shortage hypothesis is correct, the minimum
available memory at n_proc=12 should approach zero, while at a small n_proc it should not.

Run:  .venv/Scripts/python.exe diag_mem.py <n_run> <n_proc>
"""
import sys, time, ctypes, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE / "Drosophila_brain_model"


class MS(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def avail_gb():
    m = MS(); m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2**30, m.ullTotalPhys / 2**30


CHILD = r'''
import sys
from pathlib import Path
REPO = Path(sys.argv[1])
sys.path.insert(0, str(REPO))
from brian2 import Hz, ms
from model import run_exp, default_params
NEU = [720575940624963786,720575940630233916,720575940637568838,720575940638202345,
       720575940617000768,720575940630797113,720575940632889389,720575940621754367,
       720575940621502051,720575940640649691,720575940639332736,720575940616885538,
       720575940639198653,720575940620900446,720575940617937543,720575940632425919,
       720575940633143833,720575940612670570,720575940628853239,720575940629176663,
       720575940611875570]
p = dict(default_params)
p["t_run"] = 1000 * ms
p["n_run"] = int(sys.argv[2])
p["r_poi"] = 20 * Hz
run_exp(exp_name="diag", neu_exc=NEU,
        path_res=str(Path(sys.argv[4]) / "results" / "diag"),
        path_comp=str(REPO / "2023_03_23_completeness_630_final.csv"),
        path_con=str(REPO / "2023_03_23_connectivity_630_final.parquet"),
        params=p, n_proc=int(sys.argv[3]), force_overwrite=True)
'''


def main():
    n_run = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    n_proc = int(sys.argv[2]) if len(sys.argv) > 2 else -1

    total = avail_gb()[1]
    print("n_run=%d  n_proc=%d  total RAM %.1f GB" % (n_run, n_proc, total))

    (HERE / "results" / "diag").mkdir(parents=True, exist_ok=True)
    child = subprocess.Popen(
        [sys.executable, "-c", CHILD, str(REPO), str(n_run), str(n_proc), str(HERE)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

    lo = total
    t0 = time.time()
    while child.poll() is None:
        a, _ = avail_gb()
        lo = min(lo, a)
        time.sleep(0.5)
    dt = time.time() - t0
    err = child.stderr.read()

    print("exit code:      %d after %.0f s" % (child.returncode, dt))
    print("min available:  %.2f GB of %.1f GB  (used at peak: %.2f GB)" % (lo, total, total - lo))
    if child.returncode != 0:
        tail = [l for l in err.splitlines() if l.strip()][-3:]
        print("child error tail:")
        for l in tail:
            print("   ", l[:160])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
