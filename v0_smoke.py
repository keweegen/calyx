# -*- coding: utf-8 -*-
"""V0, дымовой тест: собирается ли модель Shiu et al. на текущем стеке.

Проверяется только исправность реализации, не научный результат:
короткий прогон, мало повторов, известные входные нейроны из example.ipynb.
Полное воспроизведение — в v0_sugar.py.
"""
import sys, time
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

# сахарные сенсорные нейроны, список из example.ipynb (FlyWire v630)
NEU_SUGAR = [
    720575940624963786, 720575940630233916, 720575940637568838,
    720575940638202345, 720575940617000768, 720575940630797113,
    720575940632889389, 720575940621754367, 720575940621502051,
    720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543,
    720575940632425919, 720575940633143833, 720575940612670570,
    720575940628853239, 720575940629176663, 720575940611875570,
]

def main() -> int:
    params = dict(default_params)
    params["t_run"] = 100 * ms   # вместо 1000 мс
    params["n_run"] = 2          # вместо 30
    params["r_poi"] = 150 * Hz

    out = HERE / "results" / "smoke"
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    run_exp(
        exp_name="smoke_sugar",
        neu_exc=NEU_SUGAR,
        path_res=str(out),
        params=params,
        n_proc=1,
        force_overwrite=True,
        **CONFIG,
    )
    dt = time.time() - t0
    print("run_exp finished in %.1f s" % dt)

    df = utl.load_exps([str(out / "smoke_sugar.parquet")])
    print("spike rows:", len(df))
    rate, std = utl.get_rate(df, t_run=params["t_run"], n_run=params["n_run"])
    top = rate.sort_values("smoke_sugar", ascending=False).head(10)
    print(top)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
