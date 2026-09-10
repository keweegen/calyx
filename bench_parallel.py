# -*- coding: utf-8 -*-
"""Wall-clock timing on one real run_exp condition: 10 trials, 12 workers.

The single-process measurement (bench_backend.py) does not answer the question that
matters in practice: runs go out to 12 processes, and there the bottleneck is memory
bandwidth, not the CPU. This script measures what a run actually costs.

Run:  .venv/Scripts/python.exe bench_parallel.py          (backend from brian_preferences)
         msvc_run.bat bench_parallel.py                      (same, but with a compiler available)
"""
import sys, time, shutil
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE / "Drosophila_brain_model"
sys.path.insert(0, str(REPO))
from brian2 import prefs, Hz, ms
print("codegen.target =", repr(prefs["codegen.target"]))
from model import run_exp, default_params

n = pd.read_csv(HERE / "data" / "mb_subcircuit" / "neurons.csv")
upn = n[(n.mb_role == "PN") & (n.cell_sub_class == "uniglomerular")
        & (n.side == "right")].root_id.astype("int64").to_numpy()
exc = sorted(np.random.default_rng(20260907 + 1).choice(upn, size=30, replace=False).tolist())

out = HERE / "results" / "bench_backend" / "parallel"
shutil.rmtree(out, ignore_errors=True)
out.mkdir(parents=True, exist_ok=True)

import importlib.util as _u
_s = _u.spec_from_file_location("v1a", HERE / "v1a_subcircuit.py")
_m = _u.module_from_spec(_s); sys.modules["v1a"] = _m; _s.loader.exec_module(_m)
n_proc = _m.n_proc_by_memory()
print("workers:", n_proc)
p = dict(default_params); p["t_run"] = 1000 * ms; p["n_run"] = 10; p["r_poi"] = 20 * Hz
t0 = time.time()
run_exp(exp_name="probe", neu_exc=exc, path_res=str(out),
        path_comp=str(REPO / "2023_03_23_completeness_630_final.csv"),
        path_con=str(REPO / "2023_03_23_connectivity_630_final.parquet"),
        params=p, n_proc=n_proc)
print("wall time per condition (10 trials, 12 workers): %.0f s" % (time.time() - t0))
