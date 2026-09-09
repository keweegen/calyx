# -*- coding: utf-8 -*-
"""Прогонщик ступени V1c: третья ручка kc_mbon_scale на 285 трёхмерных точках.

Ступень предрегистрирована спецификацией v0.17 (хэш 636c49968a0866bd, разделы
3з и 3и) вместе с конфигом config_v1c.json и таблицей детекторов
detector_table.json. Прогонщик исполняет её и ничего в ней не решает.

Что делает прогон, всё - буквой замороженного текста:

  сетка          57 кандидатов V1b' на пять узлов s, узлы читаются из конфига
                 КАК СТРОКИ и во время исполнения не перевычисляются;
  порядок        сначала все 57 точек узла s = 1, потом остальные 228; это
                 закрывает сверку V1c-E7.3 за первый час, а не за третий;
  доля           пересчитывается в каждой трёхмерной точке, а не наследуется
                 (V1c-E3.3): обратная связь MBON->APL даёт 2,3 % входа APL, и
                 инвариантность разреженности к s не установлена;
  MBON           P14-M считается во ВСЕХ 285 точках, а не только в прошедших по
                 доле: этого требует некритериальный выход V1c-E7.2 («для
                 каждой трёхмерной точки сетки», около миллиона строк =
                 285 x 97 x 36). В критерий входят только точки с
                 candidate_at_s = true;
  карта          расстояние до порога по-предъявленно, без временных трасс
                 (V1c-E7.2);
  сверка         на каждой точке узла s = 1 - побитовое сравнение с артефактами
                 V1b' (V1c-E7.3); расхождение пишет HALT.json и закрывает
                 ступень исходом NOT-TESTABLE.

Три предпрогонные проверки исполняет каждый шард перед своей первой точкой:
тест соответствия кода спецификации, пересчёт таблицы детекторов по весам
своего загрузчика и структурный тест масштабирования на собранной сети. Смысл
последних двух - «симулятор видит не те веса, по которым вычислен s_max», а это
свойство процесса, а не текста, поэтому проверка идёт в каждом процессе.

Запуск:
    python run_v1c_grid.py --plan                     # план точек и шардов
    python run_v1c_grid.py --preflight                # три проверки, артефакт
    python run_v1c_grid.py --pilot --codegen numpy    # пилот исполнения
    run_shard_v1c.bat 0 8                             # и так для 0..7
    python run_v1c_grid.py --merge
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "v1c"
PRERUN = OUT / "prerun"
sys.path.insert(0, str(HERE))

HASHED = ("experiment-spec-h1-h3.v0.17.frozen.md", "config_v1c.json",
          "detector_table.json", "weights_kc_mbon.json")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def artefact_hashes() -> dict:
    return {n: sha(OUT / n) for n in HASHED}


def shard_path(i: int, n: int) -> Path:
    return OUT / ("v1c_grid.shard%02d_of%02d.json" % (i, n))


def map_path(i: int, n: int) -> Path:
    return OUT / ("threshold_map.shard%02d_of%02d.parquet" % (i, n))


def load_shard(p: Path) -> dict:
    if not p.exists():
        return {"header": {}, "points": []}
    return json.loads(p.read_text(encoding="utf-8"))


# --- план прогона -------------------------------------------------------------
def write_plan(n_shards: int) -> int:
    """Порядок точек и назначение шардов, записанные ДО запуска.

    Не в config_v1c.json: тот заморожен вместе со спецификацией, его SHA-256
    записан в spec_sha256.txt как часть предрегистрации, и дописать в него
    что-либо значило бы сломать хэш. План - отдельный артефакт, объявляемый в
    отчёте наравне с прочим объявленным до прогона.
    """
    import v1c_stage as S

    g = S.grid_3d()
    ns = S.nodes()
    # Назначение шардов - по рангу точки ВНУТРИ её узла, со сдвигом на узел.
    # Наивное «шард = индекс mod 8» на этом порядке вырождается: хвост идёт
    # четвёрками узлов, 4 делит 8, и каждому шарду достаётся ровно один узел из
    # четырёх. Тогда падение одного шарда теряет целый узел сетки, а исход по
    # неполной сетке не выносится вовсе. Сдвиг на 3 (взаимно просто с 8)
    # раскладывает 57 точек каждого узла по всем восьми шардам.
    rank = {}
    plan = []
    for k, (i, c, s_node) in enumerate(g):
        r = rank.get(s_node, 0)
        rank[s_node] = r + 1
        shard = (r + 3 * ns.index(s_node)) % n_shards
        plan.append({"idx": k, "shard": shard, "cand_idx": i, "s_node": s_node,
                     "pn_kc_scale": c["pn_kc_scale"],
                     "g_apl_rel": c["g_apl_rel"]})
    doc = {"stage": "V1c", "artefact_kind": "план прогона, объявлен до запуска",
           "spec_hash_prefix": "636c49968a0866bd",
           "n_points": len(plan), "n_shards": n_shards,
           "order": "сначала все 57 кандидатов на узле s = 1, затем 228 точек "
                    "внешним циклом по кандидату и внутренним по четырём "
                    "оставшимся узлам",
           "shard_assignment": "по рангу точки внутри её узла со сдвигом 3 на "
                               "номер узла, по модулю %d: каждый шард получает "
                               "7-8 точек КАЖДОГО узла" % n_shards,
           "shard_assignment_ground": "наивное «индекс mod %d» на этом порядке "
                                      "вырождается: хвост идёт четвёрками "
                                      "узлов, 4 делит %d, и шарду достаётся "
                                      "ровно один узел из четырёх; падение "
                                      "шарда теряло бы целый узел сетки, а "
                                      "исход по неполной сетке не выносится"
                                      % (n_shards, n_shards),
           "order_ground": "V1c-E7.3 останавливает прогон на первом расхождении "
                           "с V1b'; блок s = 1 первым закрывает все 57 сверок "
                           "за первый час прогона",
           "order_does_not_affect_numbers": "сеть строится заново на каждую "
                                            "точку, зерно ставится на каждую "
                                            "пробу, кэш cython у каждого шарда "
                                            "свой; общие узлы стадий V1b', "
                                            "посчитанные независимыми "
                                            "процессами, совпали побитово",
           "points_per_shard": {str(k): sum(1 for p in plan if p["shard"] == k)
                                for k in range(n_shards)},
           "points_per_shard_by_node": {
               str(k): {n: sum(1 for p in plan
                               if p["shard"] == k and p["s_node"] == n)
                        for n in ns}
               for k in range(n_shards)},
           "hashes": artefact_hashes(), "plan": plan}
    p = OUT / "run_plan.json"
    if p.exists():
        print("план уже записан: %s. Перезапись запрещена: порядок объявляется "
              "ДО запуска, и переписать его позже значило бы объявить его "
              "задним числом." % p, file=sys.stderr)
        return 1
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    h = sha(p)
    (OUT / "run_plan_sha256.txt").write_text(
        "# Порядок 285 точек и назначение шардов ступени V1c, объявленные ДО\n"
        "# запуска прогона. Хэш записан здесь, а не в spec_sha256.txt: тот\n"
        "# заморожен вместе со спецификацией, и дописать в него что-либо\n"
        "# значило бы сломать предрегистрацию.\n#\n"
        "# Дата записи: 2026-09-09\n"
        "run_plan.json %d %s\n" % (p.stat().st_size, h), encoding="utf-8")
    print("план записан: %s\n%d точек, %d шардов, по шардам: %s\nхэш: %s"
          % (p, len(plan), n_shards,
             ", ".join("%s:%d" % kv for kv in doc["points_per_shard"].items()),
             h[:16]))
    return 0


# --- предпрогонные проверки ---------------------------------------------------
def run_preflight(neurons=None, con=None, write: bool = True) -> dict:
    import v1b_subcircuit as V
    import v1c_stage as S

    if neurons is None:
        neurons, con = V.load_substrate()
    res = S.preflight(neurons, con)
    res["hashes"] = artefact_hashes()
    res["fingerprint"] = hashlib.sha256(
        json.dumps(res, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "preflight.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print("предпрогонные проверки записаны: %s (отпечаток %s)"
              % (OUT / "preflight.json", res["fingerprint"]))
    return res


# --- одна трёхмерная точка ----------------------------------------------------
def evaluate_3d(V, S, neurons, con, cand_idx: int, cand: dict, s_node: str):
    """Одна точка: доля при P14, отклик MBON при P14-M, карта расстояния.

    Возвращает (точка, кадр карты, число аномалий «d = 0 без спайка»).
    """
    s = float(s_node)
    pt = V.evaluate_point(neurons, con, V.PANEL_CAL,
                          pn_kc_scale=cand["pn_kc_scale"],
                          g_apl_rel=cand["g_apl_rel"], seed_base=V.SEED_CAL,
                          kc_mbon_scale=s)
    pt.pop("_by_odor")
    pt["cand_idx"] = cand_idx
    pt["s_node"] = s_node
    # ограничение по доле пересчитано в этой трёхмерной точке, а не унаследовано
    pt["candidate_at_s"] = bool(V.passes_fraction(pt))

    m = V.eval_p14m(neurons, con, pn_kc_scale=cand["pn_kc_scale"],
                    g_apl_rel=cand["g_apl_rel"], seed_base=V.SEED_CAL_M,
                    odors=V.PANEL_CAL, kc_mbon_scale=s, record_vmax=True,
                    return_by_odor=True)
    by = m.pop("_by_odor")
    t = m["T38"]
    pt["mbon"] = {"floor_ok": t["floor_ok"], "ceiling_ok": t["ceiling_ok"],
                  "md_ok": t["md_ok"], "n_md_ok": t["n_md_ok"],
                  "pass": t["pass"], "per_type": t["per_type"],
                  "missing": t["missing"]}
    df = S.threshold_rows(by, neurons, cand_idx=cand_idx, cand=cand,
                          s_node=s_node, panel="C", run="cal")
    return pt, df, S.zero_without_spike(by, neurons)


# --- пилот исполнения ---------------------------------------------------------
def run_pilot(a) -> int:
    """Одна точка s = 1 с наблюдателем и без; сверка поездов и полей E7.3.

    Пилот первой точкой прогона не считается: он ничего не записывает в карту
    ступени и ни одной величины расстояния до порога не сохраняет. Его
    назначение - поймать неинертность наблюдателя за двенадцать минут, а не
    через час прогона, и сделать это на уровне спайковых ПОЕЗДОВ - того, чего
    V1c-E7.3 дать не может: у V1b' заморожены агрегаты, а не поезда.
    Сравниваются индексы клеток и времена спайков всех 72 предъявлений обоих
    таймингов на наборе C, затем счётчики в окне измерения.
    """
    import numpy as np
    import v1b_subcircuit as V
    import v1c_stage as S

    neurons, con = V.load_substrate()
    run_preflight(neurons, con)

    cand = S.candidates()[0]
    kw = dict(pn_kc_scale=cand["pn_kc_scale"],
              g_apl_rel=cand["g_apl_rel"] * V.g_ref_value())
    print("пилот: кандидат 0, pn_kc_scale %r, g_apl_rel %r"
          % (cand["pn_kc_scale"], cand["g_apl_rel"]), flush=True)

    trains, counts, n_spikes = {}, {}, {}
    for tag, rec in (("без наблюдателя", False), ("с наблюдателем", True)):
        t0 = time.time()
        got_t, got_c, tot = {}, {}, 0
        for timing, pulse, win, seed in (("P14", V.T_PULSE_MS, V.T_WINDOW_MS,
                                          V.SEED_CAL),
                                         ("P14-M", V.M_PULSE_MS, V.M_WINDOW_MS,
                                          V.SEED_CAL_M)):
            seeds = [seed + i for i in range(1, V.N_TRIALS + 1)]
            for o in V.PANEL_CAL:
                r = V.run_odor(neurons, con, o, pn_kc_scale=kw["pn_kc_scale"],
                               g_apl=kw["g_apl_rel"], seeds=seeds,
                               pulse_ms=pulse, window_ms=win,
                               kc_mbon_scale=1.0, record_vmax=rec,
                               return_trains=True)
                got_c[(timing, o)] = r["counts"]
                for k, (ii, tt) in enumerate(r["trains"]):
                    got_t[(timing, o, k)] = (ii, tt)
                    tot += len(ii)
        counts[rec], trains[rec], n_spikes[rec] = got_c, got_t, tot
        print("  %s: %.0f с, предъявлений %d, спайков %d"
              % (tag, time.time() - t0, len(got_t), tot), flush=True)

    keys = sorted(trains[False], key=str)
    same_t = (sorted(trains[True], key=str) == keys
              and all(np.array_equal(trains[False][k][0], trains[True][k][0])
                      and np.array_equal(trains[False][k][1], trains[True][k][1])
                      for k in keys))
    same_c = all(np.array_equal(counts[False][k], counts[True][k])
                 for k in counts[False])
    same = same_t and same_c
    n_pres = len(keys)
    print("пилот: поезда на %d предъявлениях (%d спайков) — %s; счётчики — %s"
          % (n_pres, n_spikes[False],
             "совпадают побитово" if same_t else "РАСХОДЯТСЯ",
             "совпадают" if same_c else "РАСХОДЯТСЯ"), flush=True)

    doc = {"check": "пилот исполнения перед прогоном V1c",
           "status": "пройден" if same else "ПРОВАЛЕН",
           "ground": "инертность наблюдателя vmax на реальном пути ступени",
           "candidate": {"cand_idx": 0, "pn_kc_scale": cand["pn_kc_scale"],
                         "g_apl_rel": cand["g_apl_rel"]},
           "n_presentations_compared": n_pres,
           "n_spikes_compared": n_spikes[False],
           "timings": ["P14", "P14-M"], "panel": "C",
           "spike_trains_identical": bool(same_t),
           "spike_counts_identical": bool(same_c),
           "compared": "индексы клеток и времена спайков каждого предъявления, "
                       "затем счётчики в окне измерения",
           "vmax_values_written": False,
           "note": "значения vmax пилота не записываются: объявленная до "
                   "прогона часть отчёта не содержит ничего о расстоянии до "
                   "порога",
           "hashes": artefact_hashes()}
    PRERUN.mkdir(parents=True, exist_ok=True)
    (PRERUN / "pilot_s1.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print("записано: %s" % (PRERUN / "pilot_s1.json"))
    return 0 if same else 1


# --- шард ---------------------------------------------------------------------
def run_shard(a) -> int:
    import pandas as pd
    from brian2 import prefs
    prefs.codegen.target = a.codegen
    if a.codegen == "cython":
        d = HERE / ".cython_cache" / ("v1c_w%02d" % a.shard)
        d.mkdir(parents=True, exist_ok=True)
        prefs.codegen.runtime.cython.cache_dir = str(d)

    import v1b_subcircuit as V
    import v1c_stage as S

    if S.halted():
        print("прогон остановлен сигналом %s: %s"
              % (S.HALT, S.halted()["reason"]), file=sys.stderr)
        return 1

    neurons, con = V.load_substrate()
    pre = run_preflight(neurons, con, write=False)

    # Порядок и нарезка берутся из плана: он объявлен до запуска и захэширован,
    # а пересчёт нарезки в прогонщике позволил бы ей молча разойтись с планом.
    plan_p = OUT / "run_plan.json"
    if not plan_p.exists():
        print("план прогона не записан: запустите --plan", file=sys.stderr)
        return 1
    plan = json.loads(plan_p.read_text(encoding="utf-8"))
    if plan["n_shards"] != a.of:
        print("план записан на %d шардов, запрошено %d"
              % (plan["n_shards"], a.of), file=sys.stderr)
        return 1
    cands = S.candidates()
    mine = [(e["idx"], (e["cand_idx"], cands[e["cand_idx"]], e["s_node"]))
            for e in plan["plan"] if e["shard"] == a.shard]
    path, mpath = shard_path(a.shard, a.of), map_path(a.shard, a.of)
    doc = load_shard(path)
    doc["header"] = {"shard": a.shard, "of": a.of, "codegen": a.codegen,
                     "preflight_fingerprint": pre["fingerprint"],
                     "hashes": pre["hashes"],
                     "mbon_computed_at_every_point": True,
                     "map_written_at_every_point": True,
                     "run_plan_sha256": sha(OUT / "run_plan.json")}
    done = {(p["cand_idx"], p["s_node"]) for p in doc["points"]}
    todo = [(k, t) for k, t in mine if (t[0], t[2]) not in done]
    if a.limit:
        todo = todo[:a.limit]
    print("шард %d из %d: точек %d, посчитано %d, к прогону %d, codegen %s"
          % (a.shard, a.of, len(mine), len(done), len(todo), a.codegen),
          flush=True)

    frames = [pd.read_parquet(mpath)] if mpath.exists() else []
    for n, (k, (i, c, s_node)) in enumerate(todo, 1):
        if S.halted():
            print("сигнал остановки получен, шард прекращает работу",
                  file=sys.stderr)
            return 1
        t0 = time.time()
        pt, df, n_anom = evaluate_3d(V, S, neurons, con, i, c, s_node)
        pt["zero_without_spike"] = n_anom
        pt["wall_s"] = round(time.time() - t0, 1)

        if s_node == "1":
            diffs = S.reproduction_check(pt)
            pt["reproduction_v1b_prime"] = "совпадение" if not diffs else diffs
            if diffs:
                S.halt("V1c-E7.3: расхождение с артефактами V1b' на узле s = 1",
                       {"cand_idx": i, "pn_kc_scale": c["pn_kc_scale"],
                        "g_apl_rel": c["g_apl_rel"], "fields": diffs})
                print("V1c-E7.3 ПРОВАЛЕНА в точке %d: %s"
                      % (i, "; ".join(diffs[:5])), file=sys.stderr)
                return 1

        doc["points"].append(pt)
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        frames.append(df)
        pd.concat(frames, ignore_index=True).to_parquet(mpath, index=False)

        m = pt["mbon"]
        tail = ("кандидат при s" if pt["candidate_at_s"] else "по доле не прошла")
        if pt["candidate_at_s"]:
            broke = [nm for nm, ok in (("3.3", m["floor_ok"]),
                                       ("3.4", m["ceiling_ok"]),
                                       ("3.5", m["md_ok"])) if not ok]
            tail += ", ДОПУСТИМА" if not broke else ", нарушено " + ",".join(broke)
        print("  [%d/%d] точка %d: кандидат %d, s %s -> f %.4f, min R_t %.4f, "
              "max R_t %.4f, %.0f c; %s"
              % (n, len(todo), k, i, s_node, pt["f_mean"],
                 min((v["R_t"] for v in m["per_type"].values()), default=float("nan")),
                 max((v["R_t"] for v in m["per_type"].values()), default=float("nan")),
                 pt["wall_s"], tail), flush=True)
    print("шард %d готов" % a.shard, flush=True)
    return 0


# --- сборка -------------------------------------------------------------------
def merge(a) -> int:
    import pandas as pd
    import v1b_subcircuit as V
    import v1c_stage as S

    h = S.halted()
    if h:
        print("прогон остановлен сигналом %s: %s" % (S.HALT, h["reason"]),
              file=sys.stderr)
        print("исход ступени — %s, основание: %s" % (h["outcome"], h["ground"]),
              file=sys.stderr)
        return 1

    pts, seen = [], set()
    heads = []
    for p in sorted(OUT.glob("v1c_grid.shard*.json")):
        d = load_shard(p)
        heads.append(d.get("header", {}))
        for pt in d["points"]:
            key = (pt["cand_idx"], pt["s_node"])
            if key not in seen:
                seen.add(key)
                pts.append(pt)
    pts.sort(key=lambda p: (p["cand_idx"], float(p["s_node"])))
    grid = S.grid_3d()
    print("собрано точек: %d из %d" % (len(pts), len(grid)))

    fps = {h.get("preflight_fingerprint") for h in heads if h}
    print("отпечатков предпрогонных проверок среди шардов: %d %s"
          % (len(fps), sorted(x for x in fps if x)))
    if len(fps) > 1:
        print("шарды прогонялись при разных предпрогонных проверках — "
              "это ошибка исполнения", file=sys.stderr)
        S.halt("отпечатки предпрогонных проверок шардов различаются", sorted(fps))

    (OUT / "v1c_grid.json").write_text(
        json.dumps(pts, ensure_ascii=False, indent=2), encoding="utf-8")

    # V1c-E7.3 повторно, над собранным артефактом: запись для отчёта
    repro, bad = [], []
    for pt in pts:
        if pt["s_node"] != "1":
            continue
        d = S.reproduction_check(pt)
        repro.append({"cand_idx": pt["cand_idx"], "diffs": d})
        if d:
            bad.append(pt["cand_idx"])
    print("V1c-E7.3 при сборке: сверено %d точек узла s = 1, расхождений в %d"
          % (len(repro), len(bad)))

    # карта расстояния до порога
    frames = [pd.read_parquet(p) for p in sorted(OUT.glob("threshold_map.shard*.parquet"))]
    summ = None
    if frames:
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(["cand_idx", "s_node", "mbon_id", "run", "odor",
                                 "trial"])
        df.to_parquet(OUT / "threshold_map.parquet", index=False)
        print("карта расстояния до порога: %d строк -> %s"
              % (len(df), OUT / "threshold_map.parquet"))
        summ = S.threshold_summary(df)
        (OUT / "threshold_map_summary.json").write_text(
            json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")

    if len(pts) < len(grid):
        print("сетка не полна — исход не выносится")
        return 0
    if any(p.get("mbon") is None for p in pts):
        print("у части точек отклик MBON не вычислялся: исход не выносится",
              file=sys.stderr)
        return 1

    res = S.outcome(pts)
    res["reproduction_v1b_prime"] = {
        "n_checked": len(repro), "n_mismatched": len(bad),
        "mismatched_cand_idx": bad}
    res["zero_without_spike_total"] = sum(p.get("zero_without_spike", 0) for p in pts)
    res["hashes"] = artefact_hashes()
    if bad:
        # V1c-E7.3: расхождение с V1b' на узле s = 1 закрывает ступень
        # исходом NOT-TESTABLE, а не FAIL: оно есть утверждение об исполнении,
        # а не о семействе
        res["outcome_before_reproduction_check"] = res["outcome"]
        res["outcome"] = "NOT-TESTABLE"
        res["ground"] = ("ошибка исполнения: проверка воспроизведения V1c-E7.3 "
                         "не прошла в %d точках узла s = 1" % len(bad))
        res["stopping_rule"] = "не выводится: исход ступени не получен"
    res["headline"] = _headline(V, pts)
    (OUT / "report_map.json").write_text(
        json.dumps({"summary": res, "points": pts}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    print("\nисход ступени V1c: %s" % res["outcome"])
    print("  основание: %s" % res["ground"])
    print("  точек, прошедших по доле при своём s: %d из %d"
          % (res["n_in_K"], res["n_points"]))
    print("  полоса: %s" % (res.get("band") if res.get("band") else "пуста"))
    print("  проверка «d = 0 без спайка»: %d при ожидании 0"
          % res["zero_without_spike_total"])
    for k, v in res["headline"].items():
        print("  %s: %s" % (k, v))
    return 0


def _headline(V, pts: list[dict]) -> dict:
    """Заголовочные величины исхода, считаемые по точкам множества K."""
    k = [p for p in pts if p.get("candidate_at_s")]
    out = {"n_in_K_by_node": {}}
    for s in sorted({p["s_node"] for p in pts}, key=float):
        ks = [p for p in k if p["s_node"] == s]
        out["n_in_K_by_node"][s] = len(ks)
        if not ks:
            continue
        rts = [[v["R_t"] for v in p["mbon"]["per_type"].values()] for p in ks]
        mins = [min(r) for r in rts if r]
        maxs = [max(r) for r in rts if r]
        if not mins:
            continue
        out["max_of_min_R_t_hz@%s" % s] = round(max(mins), 6)
        out["max_of_max_R_t_hz@%s" % s] = round(max(maxs), 6)
    out["floor_hz"], out["ceiling_hz"] = V.MBON_FLOOR_HZ, V.MBON_CEIL_HZ
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=8)
    ap.add_argument("--codegen", default="cython", choices=["numpy", "cython"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ["CALYX_V1B_OUT"] = str(OUT)
    if not (OUT / "spec_sha256.txt").exists():
        print("предрегистрация не заморожена", file=sys.stderr)
        return 1
    if a.plan:
        return write_plan(a.of)
    if a.preflight:
        run_preflight()
        return 0
    if a.pilot:
        from brian2 import prefs
        prefs.codegen.target = a.codegen
        return run_pilot(a)
    return merge(a) if a.merge else run_shard(a)


if __name__ == "__main__":
    raise SystemExit(main())
