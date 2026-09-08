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
# Ступень V1b′ (спецификация v0.15, раздел 3ж). Артефакты пишутся в свой
# каталог; каталог закрытой ступени V1b служит только базой сравнения.
OUT = HERE / "results" / "v1b_prime"
PILOT = HERE / "results" / "v1b"
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
        pt["stage"] = a.stage
        # Допустимость - конъюнкция (V1b-4.5), поэтому ограничение MBON на C
        # считается только для точек, прошедших по доле: прогон при тайминге
        # P14-M вчетверо дороже прогона доли, а недопустимая по доле точка
        # допустимой стать не может. Ключа mbon у непрошедших точек нет, и это
        # означает «не вычислялось», а не «прошло».
        if V.passes_fraction(pt):
            m = V.eval_p14m(neurons, con, pn_kc_scale=sc, g_apl_rel=g,
                            seed_base=V.SEED_CAL_M, odors=V.PANEL_CAL)
            t = m["T38"]
            pt["mbon"] = {"floor_ok": t["floor_ok"], "ceiling_ok": t["ceiling_ok"],
                          "md_ok": t["md_ok"], "n_md_ok": t["n_md_ok"],
                          "pass": t["pass"], "per_type": t["per_type"],
                          "missing": t["missing"]}
        pt["wall_s"] = round(time.time() - t0, 1)
        done.append(pt)
        path.write_text(json.dumps(done, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        if "mbon" not in pt:
            tail = ""
        elif V.admissible(pt):
            tail = "  ДОПУСТИМА"
        else:
            t = pt["mbon"]
            broke = [n for n, v in (("V1b-3.3", t["floor_ok"]),
                                    ("V1b-3.4", t["ceiling_ok"]),
                                    ("V1b-3.5", t["md_ok"])) if not v]
            tail = "  по доле прошла, MBON нет: " + ", ".join(broke)
        print("  [%d/%d] scale %.4g g %.4g -> f %.4f, f_max %.4f, s_ab %.2f, %.0f c%s"
              % (k, len(todo), sc, g, pt["f_mean"], pt["f_max"], pt["s_ab"],
                 pt["wall_s"], tail), flush=True)
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
    n_frac = sum(1 for p in pts if V.passes_fraction(p))
    n_ok = sum(1 for p in pts if V.admissible(p))
    n_unknown = sum(1 for p in pts
                    if V.passes_fraction(p) and V.passes_mbon(p) is None)
    print("собрано точек: %d из %d" % (len(pts), len(grid)))
    print("   прошли по доле: %d; из них допустимы по V1b-4.5: %d" % (n_frac, n_ok))
    if len(pts) < len(grid):
        print("сетка не полна — выбор точки преждевременен")
        return 0
    if n_unknown:
        # «не вычислено» - не «не прошло». Объявлять исход по невычисленному
        # ограничению значит повторить ту самую ошибку, из-за которой V1b
        # закрыта: сначала досчитать, потом выносить исход.
        print("   у %d точек ограничение MBON не вычислялось: исход не выносится,"
              % n_unknown)
        print("   сетку нужно дожать прогоном с ограничением V1b-4.5")
        return 1
    # V1b-4.6: выбор среди допустимых точек стадии 2, не по объединению стадий
    best = V.choose_point([p for p in pts if p.get("stage", a.stage) == a.stage])
    if best is None:
        if n_frac == 0:
            print("ни одна точка не прошла по доле — исход FAIL-CAL-KC")
        else:
            print("по доле прошли %d точек, по V1b-4.5 не прошла ни одна — "
                  "исход FAIL-CAL-MBON" % n_frac)
            worst = [(min((d["R_t"] for d in p["mbon"]["per_type"].values()),
                          default=float("nan")), p)
                     for p in pts if "mbon" in p]
            worst = [w for w in worst if w[0] == w[0]]
            if worst:
                lo, p = max(worst, key=lambda w: w[0])
                print("   лучшее, что даёт семейство на полу: min по типам R_t = "
                      "%.4f Гц (порог %.1f) в точке scale %.4g, g %.5g"
                      % (lo, V.MBON_FLOOR_HZ, p["pn_kc_scale"], p["g_apl_rel"]))
    else:
        print("выбрана: pn_kc_scale %.5g, g_apl %.5g g_ref, f %.4f, s_ab %.2f"
              % (best["pn_kc_scale"], best["g_apl_rel"], best["f_mean"], best["s_ab"]))
    return 0


def check_preconditions() -> int:
    """Предусловия прогона ступени: V1b'-0 и V1b'-1а.

    V1b'-0: тест соответствия кода спецификации проходит целиком. Провал -
    ошибка исполнения, ступень не начинается.
    V1b'-1а: карта доли отвечающих на узлах, уже посчитанных пилотом, совпадает
    с ней побитово. Зёрна калибровки те же, поэтому расхождение означает
    недетерминизм и разбирается до калибровки.
    """
    import subprocess
    r = subprocess.run([sys.executable, str(HERE / "test_spec_conformance.py")],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        print("V1b'-0: тест соответствия кода спецификации ПРОВАЛЕН, "
              "ступень не начинается", file=sys.stderr)
        print(r.stdout[-3000:], file=sys.stderr)
        return 1
    print("V1b'-0: тест соответствия кода спецификации пройден", flush=True)

    checked = 0
    for stage in (1, 2):
        p_new, p_old = merged_path(stage), PILOT / ("calibration_stage%d.json" % stage)
        if not (p_new.exists() and p_old.exists()):
            continue
        old = {(round(x["pn_kc_scale"], 10), round(x["g_apl_rel"], 10)): x
               for x in load(p_old)}
        for x in load(p_new):
            key = (round(x["pn_kc_scale"], 10), round(x["g_apl_rel"], 10))
            y = old.get(key)
            if y is None:
                continue
            checked += 1
            for f in ("f_mean", "f_max"):
                if x[f] != y[f]:
                    print("V1b'-1а: расхождение с пилотом в точке %s, поле %s: "
                          "%r против %r — недетерминизм, разбирать до калибровки"
                          % (key, f, x[f], y[f]), file=sys.stderr)
                    return 1
    print("V1b'-1а: сверка с пилотом — %d общих точек, расхождений нет"
          % checked, flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--codegen", default="cython", choices=["numpy", "cython"])
    ap.add_argument("--limit", type=int, default=0, help="прогнать только N точек")
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2])
    ap.add_argument("--merge", action="store_true", help="собрать шарды и выбрать точку")
    ap.add_argument("--skip-preconditions", action="store_true",
                    help="не проверять предусловия V1b'-0 и V1b'-1а; только для "
                         "отладки, подтверждающим прогон при этом не является")
    a = ap.parse_args()

    sys.path.insert(0, str(HERE))
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("CALYX_V1B_OUT", str(OUT))
    if not (OUT / "spec_sha256.txt").exists():
        print("предрегистрация не заморожена: запустите "
              "scripts/freeze_v1b_prime.py", file=sys.stderr)
        return 1
    if not a.skip_preconditions and check_preconditions() != 0:
        return 1
    return merge(a) if a.merge else run_shard(a)


if __name__ == "__main__":
    raise SystemExit(main())
