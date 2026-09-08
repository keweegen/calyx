# -*- coding: utf-8 -*-
"""Калибровка ступени V1b: сетка стадии 1, шардами по независимым процессам.

Стоимость одной точки - 6 калибровочных запахов по 6 предъявлений. Три вещи
делают полную сетку минутами вместо часов, и все три измерены, а не предположены:

  адаптивное усечение   при нулевом фоне модели [2] сеть без входа не спайкует,
                        поэтому прогон идёт до затухания, а не весь измерительный
                        интервал. Счётчики тождественны полному прогону:
                        проверено, 0 расхождений на 18 255 счётчиках.
  cython                генератор кода Brian 2; 3.1x против numpy на этой
                        подсхеме, что совпадает с бенчем V1a (bench_backend.py).
  шарды                 подсхема - 6 087 нейронов, около 0,5 ГБ на процесс, тогда
                        как в V1a параллелизм упирался в память полномозговой
                        модели.

Почему шарды, а не multiprocessing.Pool: на Windows venv порождает детей через
sys._base_executable, и Pool в этой конфигурации встал в дедлок - воркеры
загрузились и висели на 0 % ЦП. Независимые процессы этого класса проблем не
имеют, отлаживаются по одному и переживают падение соседа.

Каждому процессу нужен свой каталог кэша cython, иначе компиляции гонятся за
одни и те же файлы.

Прогон продолжаемый: точки, уже записанные в шард, при перезапуске пропускаются.

Запуск (нужно окружение MSVC, иначе cython не соберётся):
    msvc_run.bat run_v1b_grid.py --shard 0 --of 8      # и так для 0..7
    msvc_run.bat run_v1b_grid.py --merge
    python run_v1b_grid.py --shard 0 --of 1 --codegen numpy   # без MSVC, медленнее
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "v1b"
def merged_path(stage: int) -> Path:
    return OUT / ("calibration_stage%d.json" % stage)


def shard_path(i: int, n: int, stage: int = 1) -> Path:
    return OUT / ("calibration_stage%d.shard%02d_of%02d.json" % (stage, i, n))


def load(p: Path) -> list:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def run_shard(a) -> int:
    from brian2 import prefs
    prefs.codegen.target = a.codegen
    if a.codegen == "cython":
        d = HERE / ".cython_cache" / ("grid_w%02d" % a.shard)
        d.mkdir(parents=True, exist_ok=True)
        prefs.codegen.runtime.cython.cache_dir = str(d)
    import v1b_subcircuit as V

    grid = _grid_for(V, a.stage)
    mine = [p for k, p in enumerate(grid) if k % a.of == a.shard]
    path = shard_path(a.shard, a.of, a.stage)
    done = load(path)
    seen = {(round(p["pn_kc_scale"], 10), round(p["g_apl_rel"], 10)) for p in done}
    todo = [p for p in mine if (round(p[0], 10), round(p[1], 10)) not in seen]
    if a.limit:
        todo = todo[:a.limit]

    print("стадия %d, шард %d из %d: точек %d, посчитано %d, к прогону %d, codegen %s"
          % (a.stage, a.shard, a.of, len(mine), len(done), len(todo), a.codegen),
          flush=True)
    neurons, con = V.load_substrate()
    for k, (sc, g) in enumerate(todo, 1):
        t0 = time.time()
        pt = V.evaluate_point(neurons, con, V.PANEL_CAL, pn_kc_scale=sc,
                              g_apl_rel=g, seed_base=V.SEED_CAL)
        pt.pop("_by_odor")
        pt["wall_s"] = round(time.time() - t0, 1)
        done.append(pt)
        path.write_text(json.dumps(done, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        print("  [%d/%d] scale %.4g g %.4g -> f %.4f, f_max %.4f, s_ab %.2f, %.0f c%s"
              % (k, len(todo), sc, g, pt["f_mean"], pt["f_max"], pt["s_ab"],
                 pt["wall_s"], "  ДОПУСТИМА" if V.admissible(pt) else ""), flush=True)
    print("шард %d готов" % a.shard, flush=True)
    return 0


def _grid_for(V, stage: int):
    if stage == 1:
        return V.grid_stage1()
    prev = merged_path(1)
    if not prev.exists():
        raise SystemExit("стадия 1 не собрана: запустите --merge --stage 1")
    return V.grid_stage2(json.loads(prev.read_text(encoding="utf-8")))


def merge(a) -> int:
    import v1b_subcircuit as V
    pts, seen = [], set()
    for p in sorted(OUT.glob("calibration_stage%d.shard*.json" % a.stage)):
        for pt in load(p):
            key = (round(pt["pn_kc_scale"], 10), round(pt["g_apl_rel"], 10))
            if key not in seen:
                seen.add(key)
                pts.append(pt)
    merged_path(a.stage).write_text(
        json.dumps(pts, ensure_ascii=False, indent=2), encoding="utf-8")
    grid = _grid_for(V, a.stage)
    n_ok = sum(1 for p in pts if V.admissible(p))
    print("собрано точек: %d из %d, допустимых %d" % (len(pts), len(grid), n_ok))
    if len(pts) < len(grid):
        print("сетка не полна — выбор точки преждевременен")
        return 0
    pool = pts
    if a.stage == 2:
        # выбор идёт по объединению грубой и уточняющей сеток (V1b-4.6)
        pool = json.loads(merged_path(1).read_text(encoding="utf-8")) + pts
    best = V.choose_point(pool)
    if best is None:
        print("допустимых точек нет — исход FAIL-CAL (спецификация, V1b-4.8)")
    else:
        print("выбрана: pn_kc_scale %.5g, g_apl %.5g g_ref, f %.4f, s_ab %.2f"
              % (best["pn_kc_scale"], best["g_apl_rel"], best["f_mean"], best["s_ab"]))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--codegen", default="cython", choices=["numpy", "cython"])
    ap.add_argument("--limit", type=int, default=0, help="прогнать только N точек")
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2])
    ap.add_argument("--merge", action="store_true", help="собрать шарды и выбрать точку")
    a = ap.parse_args()

    sys.path.insert(0, str(HERE))
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if not (OUT / "spec_sha256.txt").exists():
        print("предрегистрация не заморожена: запустите scripts/freeze_v1b.py",
              file=sys.stderr)
        return 1
    return merge(a) if a.merge else run_shard(a)


if __name__ == "__main__":
    raise SystemExit(main())
