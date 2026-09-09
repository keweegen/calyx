# -*- coding: utf-8 -*-
"""Ступень V1c: предпрогонные проверки, некритериальные выходы, исход.

Что здесь. Библиотека ступени V1c (спецификация v0.17, хэш 636c49968a0866bd,
разделы 3з и 3и). Прогонщик - run_v1c_grid.py; здесь то, что он вызывает.

Три предпрогонные проверки (V1c-E7.1, «перед первой точкой прогона»):

  тест соответствия      test_spec_conformance.py целиком, унаследованный
                         пункт V1b'-0;
  таблица детекторов     пересчитывается из весов, загруженных ТЕМ ЖЕ
                         загрузчиком, которым пользуется симулятор, то есть
                         снимается с рёбер собранной сети Brian 2, а не
                         вычисляется по таблице связей заново;
  структурный тест       на каждом узле s_k сумма весов рёбер KC->MBON равна
  масштабирования        s_k, умноженному на сумму при s = 1, с относительным
                         допуском 1e-12; сумма весов всех прочих рёбер равна
                         сумме при s = 1 точно.

Провал любой из трёх - исход NOT-TESTABLE по основанию «ошибка исполнения»,
и прогон не начинается.

Два некритериальных выхода вычисляются здесь же: карта расстояния до порога
(V1c-E7.2) и проверка воспроизведения на узле s = 1 (V1c-E7.3). Ни один из них
не участвует ни в одном критерии; их провал закрывает ступень NOT-TESTABLE,
а не FAIL.

Канонические узлы сетки - десятичные строки конфига, а не результат
перевычисления формул во время исполнения: граничные узлы округлены вниз, и
перевычисление вернуло бы значения, при которых строгость неравенства условия
(5) зависела бы от порядка арифметики.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "v1c"
PRIME = HERE / "results" / "v1b_prime"
CONFIG = OUT / "config_v1c.json"
DETECTORS = OUT / "detector_table.json"
WEIGHTS = OUT / "weights_kc_mbon.json"


class NotTestable(RuntimeError):
    """Ошибка исполнения: ступень закрывается исходом NOT-TESTABLE."""


# --- конфиг ступени ----------------------------------------------------------
def config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def nodes() -> list[str]:
    """Пять канонических узлов сетки по s, строками (V1c-E3.2).

    Читаются как строки и во время исполнения не перевычисляются. Порядок -
    порядок конфига, он же порядок возрастания.
    """
    ns = list(config()["grid_s"]["nodes_canonical"])
    if ns != ["1", "1.6836", "1.7783", "3.1623", "5.3875"]:
        raise NotTestable("узлы сетки конфига не совпадают со спецификацией: %r" % ns)
    return ns


def candidates() -> list[dict]:
    """57 кандидатов V1b' - точки, прошедшие ограничение по доле (V1c-E3.3)."""
    rm = json.loads((PRIME / "report_map.json").read_text(encoding="utf-8"))
    cs = rm["candidates"]
    if len(cs) != 57:
        raise NotTestable("кандидатов V1b' не 57, а %d" % len(cs))
    return cs


def grid_3d() -> list[tuple[int, dict, str]]:
    """285 трёхмерных точек: (индекс кандидата, кандидат, узел s).

    Порядок объявлен до прогона: СНАЧАЛА все 57 кандидатов на узле s = 1, потом
    остальные 228 точек внешним циклом по кандидату и внутренним по четырём
    оставшимся узлам. Основание - V1c-E7.3: она останавливает прогон на первом
    расхождении с V1b', и блок s = 1 закрывает все 57 сверок примерно за 42
    минуты на восьми шардах, а не через три часа.

    На числа порядок не влияет: сеть строится заново на каждую точку и
    пересеивается на каждую пробу, а кэш cython у каждого шарда свой. Эмпирика -
    общие узлы стадий V1b', посчитанные независимыми процессами, совпали
    побитово.
    """
    ns = nodes()
    cs = list(enumerate(candidates()))
    head = [(i, c, ns[0]) for i, c in cs]
    tail = [(i, c, s) for i, c in cs for s in ns[1:]]
    return head + tail


# --- веса собранной сети -----------------------------------------------------
def loaded_kc_mbon_weights(neurons, con, s: str = "1") -> dict:
    """Веса рёбер KC->MBON, снятые с собранной сети Brian 2.

    Именно «тем же загрузчиком, которым пользуется симулятор»: сеть строится
    функцией build() ступени, и веса читаются с объекта Synapses, а не
    вычисляются по таблице связей повторно. Возвращает суммы по классам рёбер
    и максимум одиночного ребра на каждую клетку MBON, в мВ.
    """
    import v1b_subcircuit as V
    from brian2 import mV
    from brian2.utils.logger import BrianLogger

    # Проверка строит сеть по разу на узел и не запускает её. Brian при сборке
    # мусора считает такие объекты «не включёнными в сеть», хотя они были в ней:
    # предупреждение здесь ложное и глушится по имени, а не целиком.
    BrianLogger.suppress_name("unused_brian_object")

    # торможение берётся ненулевым, иначе «сумма весов прочих рёбер» на
    # синапсах APL была бы тождественным нулём и о неизменности не сообщала бы
    net, _, core_ids, _ = V.build(
        neurons, con, pn_kc_scale=1.0, g_apl=V.g_ref_value(), dan_mask=True,
        graded_apl=True, rates=None, kc_mbon_scale=float(s))
    syn = net["core_syn"]
    w = np.asarray(syn.w / mV, dtype=np.float64)
    ids = np.asarray(core_ids)
    pre, post = ids[np.asarray(syn.i)], ids[np.asarray(syn.j)]
    role = dict(zip(neurons.root_id, neurons.mb_role))
    r_pre = np.array([role[i] for i in pre])
    r_post = np.array([role[i] for i in post])
    sel = (r_pre == "Kenyon_Cell") & (r_post == "MBON")

    w_max, n_edges = {}, {}
    for m in sorted({int(i) for i in post[sel]}):
        ww = w[sel][post[sel] == m]
        w_max[m] = float(ww.max())
        n_edges[m] = int(len(ww))
    # рёбра входа и выхода градуального узла - тоже «прочие рёбра»: ручка s
    # обязана не трогать и их
    w_apl = 0.0
    n_apl = 0
    for name in ("apl_in", "apl_out"):
        obj = net[name] if name in [o.name for o in net.objects] else None
        if obj is not None:
            ww = np.asarray(obj.w / mV, dtype=np.float64)
            w_apl += float(ww.sum())
            n_apl += int(len(ww))
    return {"sum_kc_mbon_mV": float(w[sel].sum()),
            "sum_other_mV": float(w[~sel].sum()) + w_apl,
            "sum_other_core_mV": float(w[~sel].sum()),
            "sum_apl_mV": w_apl,
            "n_edges_kc_mbon": int(sel.sum()),
            "n_edges_other": int((~sel).sum()) + n_apl,
            "w_max_mV": w_max, "n_edges_by_mbon": n_edges}


# --- проверка 1: согласованность таблицы детекторов (V1c-E7.1) ---------------
def check_detector_table(neurons, con) -> dict:
    """Таблица детекторов пересчитывается из весов собранной сети.

    Сверяются: A и θ - из констант модели; w_max(m) на каждую клетку с
    относительным допуском 1e-9; класс каждой клетки на каждом узле - точно.
    Появление детектора среди типов T38 означало бы нарушение условия (5).
    """
    from model import default_params as dp

    tab = json.loads(DETECTORS.read_text(encoding="utf-8"))
    ns = nodes()
    if list(tab["nodes"]) != ns:
        raise NotTestable("узлы таблицы детекторов не совпадают с конфигом")

    rho = float(dp["t_mbr"] / dp["tau"])
    a_model = (1.0 / (rho - 1.0)) * (rho ** (-1.0 / (rho - 1.0))
                                     - rho ** (-rho / (rho - 1.0)))
    theta_model = float((dp["v_th"] - dp["v_0"]) / (0.001 * 1.0))
    if abs(a_model - tab["A"]) > 1e-12:
        raise NotTestable("A таблицы %.15g против A констант модели %.15g"
                          % (tab["A"], a_model))
    if abs(theta_model - tab["theta_mV"]) > 1e-12:
        raise NotTestable("θ таблицы %.15g против θ констант модели %.15g"
                          % (tab["theta_mV"], theta_model))
    a, theta = tab["A"], tab["theta_mV"]

    got = loaded_kc_mbon_weights(neurons, con, "1")
    w_max = got["w_max_mV"]
    cells = {int(c["mbon_id"]): c for c in tab["cells"]}
    n_with_input = sum(1 for c in tab["cells"] if not c["no_kc_input"])
    if len(w_max) != n_with_input:
        raise NotTestable("клеток MBON с входом KC на сети %d, в таблице %d"
                          % (len(w_max), n_with_input))

    for m, c in cells.items():
        want = float(c["w_max_mV"])
        if c["no_kc_input"]:
            if m in w_max:
                raise NotTestable("клетка %d объявлена без входа KC, но на сети "
                                  "у неё %d рёбер" % (m, got["n_edges_by_mbon"][m]))
            continue
        if m not in w_max:
            raise NotTestable("клетка %d есть в таблице, но не на сети" % m)
        if abs(w_max[m] - want) > 1e-9 * max(abs(want), 1e-30):
            raise NotTestable("w_max клетки %d: сеть %.15g, таблица %.15g"
                              % (m, w_max[m], want))

    # класс каждой клетки на каждом узле - точно
    for s in ns:
        want = sorted(int(d["mbon_id"]) for d in tab["by_node"][s]["detectors"])
        have = sorted(m for m, wm in w_max.items() if float(s) * a * wm > theta)
        if have != want:
            raise NotTestable("детекторы на узле %s: сеть %r, таблица %r"
                              % (s, have, want))
        in_t38 = [m for m in have if cells[m]["in_T38"]]
        if in_t38:
            raise NotTestable("на узле %s детектор среди типов T38: %r — "
                              "условие (5) нарушено" % (s, in_t38))
    return {"check": "V1c-E7.1, согласованность таблицы детекторов",
            "status": "пройдена", "A": a, "theta_mV": theta,
            "n_mbon_in_table": len(cells), "n_mbon_with_kc_input": len(w_max),
            "n_detectors_by_node": {s: len(tab["by_node"][s]["detectors"])
                                    for s in ns},
            "detectors_in_T38_any_node": 0}


# --- проверка 2: структурный тест масштабирования (V1c-E7.1) -----------------
def check_structural_scaling(neurons, con) -> dict:
    """Сумма весов KC->MBON равна s_k * сумме при s = 1; прочие - точно равны.

    Допуск на масштабируемую сумму - относительный 1e-12; на неизменяемую -
    точное равенство. Проверка идёт по весам собранной сети: она отвечает на
    вопрос, видит ли симулятор те веса, по которым вычислена граница s_max.
    """
    base = loaded_kc_mbon_weights(neurons, con, "1")
    rows = []
    for s in nodes():
        got = base if s == "1" else loaded_kc_mbon_weights(neurons, con, s)
        want = float(s) * base["sum_kc_mbon_mV"]
        rel = abs(got["sum_kc_mbon_mV"] - want) / max(abs(want), 1e-30)
        if rel > 1e-12:
            raise NotTestable("узел %s: сумма KC->MBON %.17g, ожидается %.17g, "
                              "относительное расхождение %.3g" % (s, got["sum_kc_mbon_mV"],
                                                                  want, rel))
        if got["sum_other_mV"] != base["sum_other_mV"]:
            raise NotTestable("узел %s: сумма весов прочих рёбер изменилась: "
                              "%.17g против %.17g" % (s, got["sum_other_mV"],
                                                      base["sum_other_mV"]))
        if got["n_edges_kc_mbon"] != base["n_edges_kc_mbon"] \
                or got["n_edges_other"] != base["n_edges_other"]:
            raise NotTestable("узел %s: изменилось число рёбер" % s)
        rows.append({"node": s, "sum_kc_mbon_mV": got["sum_kc_mbon_mV"],
                     "rel_error": rel, "sum_other_mV": got["sum_other_mV"]})
    return {"check": "V1c-E7.1, структурный тест масштабирования",
            "status": "пройдена", "tol_rel_scaled": 1e-12,
            "tol_other": "точное равенство",
            "n_edges_kc_mbon": base["n_edges_kc_mbon"],
            "n_edges_other": base["n_edges_other"], "by_node": rows}


# --- проверка 3: тест соответствия кода спецификации (V1b'-0) ---------------
def check_conformance() -> dict:
    r = subprocess.run([sys.executable, str(HERE / "test_spec_conformance.py")],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        raise NotTestable("тест соответствия кода спецификации провален:\n%s"
                          % r.stdout[-3000:])
    tail = [l for l in r.stdout.splitlines() if l.startswith("все ")]
    return {"check": "V1b'-0, тест соответствия кода спецификации",
            "status": "пройден", "summary": tail[-1] if tail else ""}


def check_run_plan() -> dict:
    """План прогона существует, совпадает со своим хэшем и покрывает сетку.

    План объявляет порядок 285 точек и назначение шардов ДО запуска. Проверка
    существует потому, что «объявлено до прогона» обязано быть механически
    проверяемым: иначе порядок можно переписать задним числом, а расхождение
    плана с сеткой оставило бы часть точек непосчитанной, и исход был бы вынесен
    по неполной сетке.
    """
    import hashlib

    p = OUT / "run_plan.json"
    h = OUT / "run_plan_sha256.txt"
    if not (p.exists() and h.exists()):
        raise NotTestable("план прогона или его хэш не записаны: запустите "
                          "run_v1c_grid.py --plan")
    want = [l.split()[-1] for l in h.read_text(encoding="utf-8").splitlines()
            if l.startswith("run_plan.json")]
    got = hashlib.sha256(p.read_bytes()).hexdigest()
    if not want or got != want[0]:
        raise NotTestable("план прогона не совпадает со своим хэшем: %s против %s"
                          % (got[:16], (want or ["—"])[0][:16]))
    plan = json.loads(p.read_text(encoding="utf-8"))
    grid = [(i, s) for i, _, s in grid_3d()]
    got_pts = [(e["cand_idx"], e["s_node"]) for e in plan["plan"]]
    if got_pts != grid:
        raise NotTestable("порядок плана не совпадает с сеткой ступени: %d точек "
                          "плана против %d точек сетки" % (len(got_pts), len(grid)))
    if len(set(got_pts)) != len(grid):
        raise NotTestable("в плане есть повторяющиеся точки")
    per = {}
    for e in plan["plan"]:
        per.setdefault(e["shard"], set()).add(e["s_node"])
    missing = {k: sorted(set(nodes()) - v) for k, v in per.items() if len(v) < 5}
    if missing:
        raise NotTestable("шардам не хватает узлов: %r — падение шарда потеряло "
                          "бы целый узел сетки" % missing)
    return {"check": "план прогона объявлен до запуска и покрывает сетку",
            "status": "пройдена", "sha256_prefix": got[:16],
            "n_points": len(got_pts), "n_shards": plan["n_shards"],
            "points_per_shard": plan["points_per_shard"]}


def preflight(neurons, con, *, verbose: bool = True) -> dict:
    """Все проверки перед первой точкой прогона. Провал - NotTestable."""
    res = [check_conformance(),
           check_run_plan(),
           check_detector_table(neurons, con),
           check_structural_scaling(neurons, con)]
    if verbose:
        for r in res:
            print("предпрогонная проверка: %s — %s" % (r["check"], r["status"]),
                  flush=True)
    return {"preflight": res}


# --- некритериальный выход V1c-E7.2: карта расстояния до порога -------------
def _mbon_index(neurons, core_ids) -> tuple[np.ndarray, list[int], list[str]]:
    role = dict(zip(neurons.root_id, neurons.mb_role))
    typ = dict(zip(neurons.root_id,
                   neurons.hemibrain_type.astype("string").fillna("<без типа>")))
    m = np.array([role[i] == "MBON" for i in core_ids])
    ids = [i for i, keep in zip(core_ids, m) if keep]
    return m, ids, [str(typ[i]) for i in ids]


def threshold_rows(by_odor: dict, neurons, *, cand_idx: int, cand: dict,
                   s_node: str, panel: str, run: str = "cal") -> pd.DataFrame:
    """Строки карты расстояния до порога: клетка x запах x проба (V1c-E7.2).

        d = v_th − max_t v(t),

    максимум по всем шагам симулятора внутри окна, v читается до сброса. d = 0
    тогда и только тогда, когда клетка дала спайк в этом окне; поэтому нулю
    приравнивается измеренное значение, а не отбрасывается знак.

    Временные трассы v не сохраняются, агрегаций в первичном артефакте нет.
    """
    from model import default_params as dp
    v_th = float(dp["v_th"] / (0.001 * 1.0))

    no_kc = set(_cells_without_kc_input())
    first = by_odor[list(by_odor)[0]]
    mask, ids, types = _mbon_index(neurons, first["core_ids"])
    n_cells = len(ids)

    frames = []
    for odor, res in by_odor.items():
        if "vmax_mV" not in res:
            raise NotTestable("прогон запаха %r выполнен без наблюдателя vmax: "
                              "карта V1c-E7.2 не может быть записана" % odor)
        vm = res["vmax_mV"][mask]                 # клетки x пробы
        sp = res["counts"][mask] > 0              # спайк в окне предъявления
        n_trials = vm.shape[1]
        d = np.where(sp, 0.0, np.maximum(0.0, v_th - vm)).astype(np.float32)
        frames.append(pd.DataFrame({
            "cand_idx": np.int16(cand_idx),
            "pn_kc_scale": float(cand["pn_kc_scale"]),
            "g_apl_rel": float(cand["g_apl_rel"]),
            "s_node": s_node,
            "mbon_id": np.repeat(np.asarray(ids, dtype=np.int64), n_trials),
            "hemibrain_type": np.repeat(np.asarray(types, dtype=object), n_trials),
            "panel": panel,
            "run": run,
            "odor": odor,
            "trial": np.tile(np.arange(1, n_trials + 1, dtype=np.int8), n_cells),
            "d_peak_mV": d.reshape(-1),
            "spiked": sp.reshape(-1),
            "no_kc_input": np.repeat(
                np.array([i in no_kc for i in ids], dtype=bool), n_trials),
        }))
    # Строковые столбцы остаются строками, а не категориями: у точек разные
    # узлы s, склейка кадров с несовпадающими наборами категорий молча даёт
    # object, и тип столбца в артефакте зависел бы от порядка склейки. Parquet
    # словарное кодирование делает сам.
    return pd.concat(frames, ignore_index=True)


def zero_without_spike(by_odor: dict, neurons) -> int:
    """Число предъявлений MBON, где измеренное v_th - vmax <= 0 без спайка.

    Мера этого события нулевая: порог строгий (v > v_th), поэтому равенство пика
    порогу спайка не даёт, но и расстояния не оставляет. Величина печатается как
    проверка исполнения с объявленным до прогона ожиданием 0; она не критерий.
    """
    from model import default_params as dp
    v_th = float(dp["v_th"] / (0.001 * 1.0))
    first = by_odor[list(by_odor)[0]]
    mask, _, _ = _mbon_index(neurons, first["core_ids"])
    n = 0
    for res in by_odor.values():
        sp = res["counts"][mask] > 0
        d = v_th - res["vmax_mV"][mask]
        n += int(((~sp) & (d <= 0)).sum())
    return n


def _cells_without_kc_input() -> list[int]:
    w = json.loads(WEIGHTS.read_text(encoding="utf-8"))
    return [int(k) for k, d in w["per_mbon"].items() if d["n_edges_kc"] == 0]


def threshold_summary(df: pd.DataFrame) -> dict:
    """Сводка карты: квантили по предъявлениям и медианы по типам T38.

    Вычисляется из первичного артефакта и перевычисляема из него: сводка не
    заменяет карту и в критериях не участвует.
    """
    import v1b_subcircuit as V

    g = df.groupby(["cand_idx", "s_node", "mbon_id"], observed=True)["d_peak_mV"]
    per_cell = g.agg(min="min", q10=lambda x: float(np.quantile(x, 0.10)),
                     median="median",
                     q90=lambda x: float(np.quantile(x, 0.90)), max="max")
    per_cell["frac_zero"] = df.groupby(
        ["cand_idx", "s_node", "mbon_id"], observed=True)["spiked"].mean()
    per_cell = per_cell.reset_index()

    types = df[["mbon_id", "hemibrain_type"]].drop_duplicates()
    per_cell = per_cell.merge(types, on="mbon_id", how="left")
    t38 = per_cell[per_cell.hemibrain_type.isin(V.T38)]
    by_type = (t38.groupby(["cand_idx", "s_node", "hemibrain_type"],
                           observed=True)["median"]
               .median().reset_index()
               .rename(columns={"median": "median_of_cell_medians_mV"}))

    return {"output": "V1c-E7.2, сводка карты расстояния до порога",
            "status": "некритериальный выход; в критериях не участвует",
            "n_rows_primary": int(len(df)),
            "per_cell": per_cell.to_dict(orient="records"),
            "per_T38_type": by_type.to_dict(orient="records")}


# --- некритериальный выход V1c-E7.3: воспроизведение на узле s = 1 ----------
def _prime_calibration() -> dict:
    out = {}
    for stage in (1, 2):
        p = PRIME / ("calibration_stage%d.json" % stage)
        for x in json.loads(p.read_text(encoding="utf-8")):
            out.setdefault((repr(x["pn_kc_scale"]), repr(x["g_apl_rel"])), x)
    return out


def _prime_candidates() -> dict:
    rm = json.loads((PRIME / "report_map.json").read_text(encoding="utf-8"))
    return {(repr(c["pn_kc_scale"]), repr(c["g_apl_rel"])): c
            for c in rm["candidates"]}


def _same(a, b) -> bool:
    """Побитовое равенство: для float - равенство представлений, NaN = NaN."""
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a.hex() == b.hex() if not (math.isnan(a) or math.isnan(b)) else False
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def as_written(pt: dict) -> dict:
    """Точка в том виде, в каком её пишет шард.

    V1c-E7.3 требует равенства представлений, а не равенства объектов в памяти,
    поэтому сверка идёт по значениям, прошедшим через тот же json.dumps, каким
    шард пишет свой файл, и обратную загрузку.
    """
    return json.loads(json.dumps(pt, ensure_ascii=False))


def reproduction_check(pt: dict) -> list[str]:
    """Сверка точки узла s = 1 с артефактами V1b'. Возвращает список расхождений.

    Сравнивается по фактической схеме артефактов V1b' (V1c-E7.3): из карт
    калибровки - f_mean, f_max, f_by_odor, s_ab, s_apbp, s_g; из report_map -
    R_t, MD, min_R_t, max_R_t, violated. Допуск побитовый: умножение веса на
    s = 1 в арифметике IEEE тождественно, поэтому более слабый допуск лишь
    скрыл бы недетерминизм.
    """
    pt = as_written(pt)
    key = (repr(pt["pn_kc_scale"]), repr(pt["g_apl_rel"]))
    diffs = []
    old = _prime_calibration().get(key)
    if old is None:
        return ["точки %r нет в картах калибровки V1b'" % (key,)]
    for f in ("f_mean", "f_max", "s_ab", "s_apbp", "s_g"):
        if not _same(pt[f], old[f]):
            diffs.append("%s: V1c %r, V1b' %r" % (f, pt[f], old[f]))
    if not _same(pt["f_by_odor"], old["f_by_odor"]):
        for o in old["f_by_odor"]:
            if not _same(pt["f_by_odor"].get(o), old["f_by_odor"][o]):
                diffs.append("f_by_odor[%s]: V1c %r, V1b' %r"
                             % (o, pt["f_by_odor"].get(o), old["f_by_odor"][o]))

    cand = _prime_candidates().get(key)
    if cand is None:
        return diffs + ["точки %r нет среди кандидатов V1b'" % (key,)]
    m = pt.get("mbon")
    if m is None:
        return diffs + ["в точке узла s = 1 отклик MBON не вычислялся"]
    per = m["per_type"]
    for t, want in cand["R_t"].items():
        if not _same(per.get(t, {}).get("R_t"), want):
            diffs.append("R_t[%s]: V1c %r, V1b' %r"
                         % (t, per.get(t, {}).get("R_t"), want))
    for t, want in cand["MD"].items():
        got = per.get(t, {}).get("MD")
        if not _same(got, want):
            diffs.append("MD[%s]: V1c %r, V1b' %r" % (t, got, want))
    rt = [v["R_t"] for v in per.values()]
    if rt:
        if not _same(float(min(rt)), cand["min_R_t"]):
            diffs.append("min_R_t: V1c %r, V1b' %r" % (float(min(rt)), cand["min_R_t"]))
        if not _same(float(max(rt)), cand["max_R_t"]):
            diffs.append("max_R_t: V1c %r, V1b' %r" % (float(max(rt)), cand["max_R_t"]))
    viol = _violated(m)
    if not _same(viol, list(cand["violated"])):
        diffs.append("violated: V1c %r, V1b' %r" % (viol, cand["violated"]))
    return diffs


def _violated(m: dict) -> list[str]:
    return [n for n, ok in (("V1b-3.3", m["floor_ok"]),
                            ("V1b-3.4", m["ceiling_ok"]),
                            ("V1b-3.5", m["md_ok"])) if not ok]


# --- сигнал аварийной остановки ----------------------------------------------
HALT = OUT / "HALT.json"


def halt(reason: str, detail) -> None:
    """Записать сигнал остановки: шарды проверяют его перед каждой точкой.

    V1c-E7.3 говорит «прогон останавливается на первом обнаруженном
    расхождении». При восьми независимых процессах остановить можно только свой
    шард, поэтому первый обнаруживший пишет сигнал, а остальные его читают.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    if not HALT.exists():
        HALT.write_text(json.dumps(
            {"outcome": "NOT-TESTABLE", "ground": "ошибка исполнения",
             "reason": reason, "detail": detail}, ensure_ascii=False, indent=2),
            encoding="utf-8")


def halted() -> dict | None:
    return json.loads(HALT.read_text(encoding="utf-8")) if HALT.exists() else None


# --- исход ступени (V1c-E4.1, E4.3, E4.4, E4.5) ------------------------------
def outcome(points: list[dict]) -> dict:
    """Исход калибровочной части ступени по замороженным определениям.

    Порядок вычисления объявлен до прогона:

      K = множество трёхмерных точек, прошедших ограничение по доле на C при
          СВОЁМ узле s. K пусто -> FAIL-CAL-KC (V1c-E4.5, первое предложение).
      полоса = узлы, на которых есть точка K с полной конъюнкцией (V1c-E4.1).
          Непуста -> исход не выносится здесь: он требует оценки на наборе E,
          а слепота E расходуется только прогоном оценки.
      полоса пуста и в K есть точка с превышением потолка -> CEIL
          (V1c-E4.5, правило приоритета).
      иначе на верхнем узле s_max есть точка K с типом ниже пола -> FLOOR.
      иначе -> NOT-TESTABLE: определения E4.3 и E4.5 наблюдённый случай не
          покрывают, и присвоение исхода по аналогии было бы трактовкой после
          прогона.

    Нарушение потолка операционализировано как `not ceiling_ok`, то есть
    «существует t из T38 с R_t > 67 Гц»: буква V1b-3.4 говорит «для каждого
    типа», а кванторы E4.3 и E4.5 - «хотя бы один».
    """
    import v1b_subcircuit as V

    ns = nodes()
    s_pop = config()["grid_s"]["s_pop_reportonly"]
    k = [p for p in points if p.get("candidate_at_s")]
    base = {"n_points": len(points), "n_in_K": len(k),
            "K_definition": "точки, прошедшие ограничение по доле на C при своём узле s",
            "ceiling_rule": "нарушение = существует t из T38 с R_t > 67 Гц",
            "floor_node": ns[-1], "s_pop": s_pop}
    if not k:
        return dict(base, outcome="FAIL-CAL-KC",
                    ground="ни одна трёхмерная точка не удовлетворяет "
                           "ограничению по доле",
                    stopping_rule="первое основание в форме «нет ни одной "
                                  "трёхмерной точки в полосе разреженности»")

    band = sorted({p["s_node"] for p in k if V.admissible(p)}, key=float)
    if band:
        regime = "FULL" if float(band[0]) <= float(s_pop) else "T38"
        return dict(base, outcome="BAND-NONEMPTY", band=band,
                    band_regime=regime,
                    ground="полоса непуста; исход PASS-FULL / PASS-T38 / "
                           "FAIL-EVAL определяется прогоном оценки на наборе E",
                    stopping_rule="не наступило; это расхождение с "
                                  "объявленным ожиданием")

    ceil = [p for p in k if not p["mbon"]["ceiling_ok"]]
    if ceil:
        return dict(base, outcome="FAIL-CAL-MBON-CEIL", band=[],
                    n_points_over_ceiling=len(ceil),
                    ground="полоса пуста, и хотя бы в одной точке K хотя бы "
                           "один тип T38 выше потолка 67 Гц",
                    stopping_rule="первое основание: скаляра нет ни при каком "
                                  "s, разрыв замкнут сверху, и продолжение "
                                  "сетки за s_max его не откроет")

    top = [p for p in k if p["s_node"] == ns[-1]]
    if not top:
        return dict(base, outcome="NOT-TESTABLE", band=[],
                    ground="ошибка исполнения: на верхнем узле %s ни одна точка "
                           "не прошла по доле, а K непусто на других узлах; "
                           "определения исходов E4.3 и E4.5 этот случай не "
                           "покрывают" % ns[-1],
                    stopping_rule="не выводится: исход не назначается по аналогии")

    below = [p for p in top
             if any(v["R_t"] < V.MBON_FLOOR_HZ for v in p["mbon"]["per_type"].values())]
    if below:
        return dict(base, outcome="FAIL-CAL-MBON-FLOOR", band=[],
                    n_points_top_node=len(top), n_top_below_floor=len(below),
                    ground="полоса пуста, потолок не нарушен ни в одной точке K, "
                           "и на верхнем узле есть точка K, где хотя бы один тип "
                           "T38 ниже пола 2 Гц",
                    stopping_rule="второе основание: внутри области "
                                  "популяционного кода скаляра нет, а коридор, "
                                  "если он есть, лежит только за верхней границей")

    return dict(base, outcome="NOT-TESTABLE", band=[], n_points_top_node=len(top),
                ground="ошибка исполнения: полоса пуста, потолок не нарушен, и на "
                       "верхнем узле все точки K держат все шесть типов не ниже "
                       "пола — провал только по глубине модуляции; определения "
                       "исходов E4.3 и E4.5 этот случай не покрывают",
                stopping_rule="не выводится: исход не назначается по аналогии")
