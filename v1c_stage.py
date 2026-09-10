# -*- coding: utf-8 -*-
"""V1c stage: pre-run checks, non-criterion outputs, outcome.

What this is. The V1c stage library (spec v0.17, hash 636c49968a0866bd,
sections 3з and 3и). The runner is run_v1c_grid.py; this is what it calls.

Three pre-run checks (V1c-E7.1, "before the first run point"):

  conformance test        test_spec_conformance.py in full, the inherited
                          V1b'-0 item;
  detector table          recomputed from weights loaded by THE SAME
                          loader the simulator uses, i.e. taken from the
                          edges of the assembled Brian 2 network, not
                          computed from the connectivity table again;
  structural scaling      at every node s_k the sum of KC->MBON edge
  test                    weights equals s_k times the sum at s = 1, with a
                          relative tolerance of 1e-12; the sum of all other
                          edge weights equals the sum at s = 1 exactly.

Failure of any of the three gives outcome NOT-TESTABLE on the ground
"execution error", and the run does not start.

Two non-criterion outputs are computed here as well: the distance-to-threshold
map (V1c-E7.2) and the reproduction check at node s = 1 (V1c-E7.3). Neither
participates in any criterion; their failure closes the stage NOT-TESTABLE,
not FAIL.

The canonical grid nodes are the decimal strings from the config, not the
result of recomputing the formulas at run time: the boundary nodes are
rounded down, and recomputing them would return values for which the
strictness of inequality (5) would depend on arithmetic order.
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
    """Execution error: the stage closes with outcome NOT-TESTABLE."""


# --- stage config -------------------------------------------------------------
def config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def nodes() -> list[str]:
    """The five canonical grid nodes over s, as strings (V1c-E3.2).

    Read as strings and not recomputed at run time. The order is the
    config's order, which is also ascending order.
    """
    ns = list(config()["grid_s"]["nodes_canonical"])
    if ns != ["1", "1.6836", "1.7783", "3.1623", "5.3875"]:
        raise NotTestable("config grid nodes do not match the specification: %r" % ns)
    return ns


def candidates() -> list[dict]:
    """The 57 V1b' candidates - points that passed the share constraint (V1c-E3.3)."""
    rm = json.loads((PRIME / "report_map.json").read_text(encoding="utf-8"))
    cs = rm["candidates"]
    if len(cs) != 57:
        raise NotTestable("V1b' candidates are not 57, but %d" % len(cs))
    return cs


def grid_3d() -> list[tuple[int, dict, str]]:
    """285 three-dimensional points: (candidate index, candidate, node s).

    The order is declared before the run: FIRST all 57 candidates at node
    s = 1, then the remaining 228 points by an outer loop over candidate and
    an inner loop over the four remaining nodes. The reason is V1c-E7.3: it
    stops the run at the first divergence from V1b', and the s = 1 block
    closes all 57 checks in about 42 minutes on eight shards, rather than
    three hours.

    Order does not affect the numbers: the network is rebuilt for every
    point and reseeded for every trial, and each shard has its own cython
    cache. Empirically, the shared V1b' stage nodes, computed by independent
    processes, matched bitwise.
    """
    ns = nodes()
    cs = list(enumerate(candidates()))
    head = [(i, c, ns[0]) for i, c in cs]
    tail = [(i, c, s) for i, c in cs for s in ns[1:]]
    return head + tail


# --- weights of the assembled network ------------------------------------------
def loaded_kc_mbon_weights(neurons, con, s: str = "1") -> dict:
    """KC->MBON edge weights, taken from the assembled Brian 2 network.

    Specifically "the same loader the simulator uses": the network is built
    by the stage's build() function, and the weights are read off the
    Synapses object rather than recomputed from the connectivity table.
    Returns the sums by edge class and the maximum single edge for each
    MBON cell, in mV.
    """
    import v1b_subcircuit as V
    from brian2 import mV
    from brian2.utils.logger import BrianLogger

    # The check builds the network once per node and does not run it. During
    # garbage collection Brian considers such objects "not included in the
    # network", even though they were: the warning here is a false positive
    # and is suppressed by name, not wholesale.
    BrianLogger.suppress_name("unused_brian_object")

    # inhibition is taken nonzero, otherwise the "sum of other edge weights"
    # on APL synapses would be identically zero and would say nothing about
    # invariance
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
    # the input and output edges of the graded node are "other edges" too:
    # the s knob must not touch them either
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


# --- check 1: detector table consistency (V1c-E7.1) --------------------------
def check_detector_table(neurons, con) -> dict:
    """The detector table is recomputed from the assembled network's weights.

    Checked: A and θ - against the model constants; w_max(m) for each cell
    with a relative tolerance of 1e-9; each cell's class at each node -
    exactly. A detector appearing among the T38 types would mean condition
    (5) is violated.
    """
    from model import default_params as dp

    tab = json.loads(DETECTORS.read_text(encoding="utf-8"))
    ns = nodes()
    if list(tab["nodes"]) != ns:
        raise NotTestable("detector table nodes do not match the config")

    rho = float(dp["t_mbr"] / dp["tau"])
    a_model = (1.0 / (rho - 1.0)) * (rho ** (-1.0 / (rho - 1.0))
                                     - rho ** (-rho / (rho - 1.0)))
    theta_model = float((dp["v_th"] - dp["v_0"]) / (0.001 * 1.0))
    if abs(a_model - tab["A"]) > 1e-12:
        raise NotTestable("table A %.15g vs model-constant A %.15g"
                          % (tab["A"], a_model))
    if abs(theta_model - tab["theta_mV"]) > 1e-12:
        raise NotTestable("table θ %.15g vs model-constant θ %.15g"
                          % (tab["theta_mV"], theta_model))
    a, theta = tab["A"], tab["theta_mV"]

    got = loaded_kc_mbon_weights(neurons, con, "1")
    w_max = got["w_max_mV"]
    cells = {int(c["mbon_id"]): c for c in tab["cells"]}
    n_with_input = sum(1 for c in tab["cells"] if not c["no_kc_input"])
    if len(w_max) != n_with_input:
        raise NotTestable("MBON cells with KC input in the network %d, in the table %d"
                          % (len(w_max), n_with_input))

    for m, c in cells.items():
        want = float(c["w_max_mV"])
        if c["no_kc_input"]:
            if m in w_max:
                raise NotTestable("cell %d is declared with no KC input, but in the "
                                  "network it has %d edges" % (m, got["n_edges_by_mbon"][m]))
            continue
        if m not in w_max:
            raise NotTestable("cell %d is in the table but not in the network" % m)
        if abs(w_max[m] - want) > 1e-9 * max(abs(want), 1e-30):
            raise NotTestable("w_max of cell %d: network %.15g, table %.15g"
                              % (m, w_max[m], want))

    # each cell's class at each node - exactly
    for s in ns:
        want = sorted(int(d["mbon_id"]) for d in tab["by_node"][s]["detectors"])
        have = sorted(m for m, wm in w_max.items() if float(s) * a * wm > theta)
        if have != want:
            raise NotTestable("detectors at node %s: network %r, table %r"
                              % (s, have, want))
        in_t38 = [m for m in have if cells[m]["in_T38"]]
        if in_t38:
            raise NotTestable("at node %s a detector among T38 types: %r - "
                              "condition (5) violated" % (s, in_t38))
    return {"check": "V1c-E7.1, согласованность таблицы детекторов",
            "status": "пройдена", "A": a, "theta_mV": theta,
            "n_mbon_in_table": len(cells), "n_mbon_with_kc_input": len(w_max),
            "n_detectors_by_node": {s: len(tab["by_node"][s]["detectors"])
                                    for s in ns},
            "detectors_in_T38_any_node": 0}


# --- check 2: structural scaling test (V1c-E7.1) ------------------------------
def check_structural_scaling(neurons, con) -> dict:
    """Sum of KC->MBON weights equals s_k * the sum at s = 1; others are exactly equal.

    The tolerance on the scaled sum is a relative 1e-12; on the invariant
    sum, exact equality. The check runs over the assembled network's
    weights: it answers whether the simulator sees the same weights from
    which the s_max bound was computed.
    """
    base = loaded_kc_mbon_weights(neurons, con, "1")
    rows = []
    for s in nodes():
        got = base if s == "1" else loaded_kc_mbon_weights(neurons, con, s)
        want = float(s) * base["sum_kc_mbon_mV"]
        rel = abs(got["sum_kc_mbon_mV"] - want) / max(abs(want), 1e-30)
        if rel > 1e-12:
            raise NotTestable("node %s: sum KC->MBON %.17g, expected %.17g, "
                              "relative discrepancy %.3g" % (s, got["sum_kc_mbon_mV"],
                                                                  want, rel))
        if got["sum_other_mV"] != base["sum_other_mV"]:
            raise NotTestable("node %s: sum of weights of other edges changed: "
                              "%.17g vs %.17g" % (s, got["sum_other_mV"],
                                                      base["sum_other_mV"]))
        if got["n_edges_kc_mbon"] != base["n_edges_kc_mbon"] \
                or got["n_edges_other"] != base["n_edges_other"]:
            raise NotTestable("node %s: edge count changed" % s)
        rows.append({"node": s, "sum_kc_mbon_mV": got["sum_kc_mbon_mV"],
                     "rel_error": rel, "sum_other_mV": got["sum_other_mV"]})
    return {"check": "V1c-E7.1, структурный тест масштабирования",
            "status": "пройдена", "tol_rel_scaled": 1e-12,
            "tol_other": "точное равенство",
            "n_edges_kc_mbon": base["n_edges_kc_mbon"],
            "n_edges_other": base["n_edges_other"], "by_node": rows}


# --- check 3: spec-conformance test of the code (V1b'-0) ---------------------
def check_conformance() -> dict:
    r = subprocess.run([sys.executable, str(HERE / "test_spec_conformance.py")],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        raise NotTestable("the code-to-specification conformance test failed:\n%s"
                          % r.stdout[-3000:])
    tail = [l for l in r.stdout.splitlines() if l.startswith("все ")]
    return {"check": "V1b'-0, тест соответствия кода спецификации",
            "status": "пройден", "summary": tail[-1] if tail else ""}


def check_run_plan() -> dict:
    """The run plan exists, matches its own hash, and covers the grid.

    The plan declares the order of the 285 points and the shard assignment
    BEFORE the run starts. This check exists because "declared before the
    run" must be mechanically verifiable: otherwise the order could be
    rewritten after the fact, and a mismatch between the plan and the grid
    would leave part of the points uncomputed, with the outcome then being
    decided on an incomplete grid.
    """
    import hashlib

    p = OUT / "run_plan.json"
    h = OUT / "run_plan_sha256.txt"
    if not (p.exists() and h.exists()):
        raise NotTestable("the run plan or its hash is not recorded: run "
                          "run_v1c_grid.py --plan")
    want = [l.split()[-1] for l in h.read_text(encoding="utf-8").splitlines()
            if l.startswith("run_plan.json")]
    got = hashlib.sha256(p.read_bytes()).hexdigest()
    if not want or got != want[0]:
        raise NotTestable("the run plan does not match its own hash: %s vs %s"
                          % (got[:16], (want or ["—"])[0][:16]))
    plan = json.loads(p.read_text(encoding="utf-8"))
    grid = [(i, s) for i, _, s in grid_3d()]
    got_pts = [(e["cand_idx"], e["s_node"]) for e in plan["plan"]]
    if got_pts != grid:
        raise NotTestable("the plan order does not match the stage grid: %d plan "
                          "points vs %d grid points" % (len(got_pts), len(grid)))
    if len(set(got_pts)) != len(grid):
        raise NotTestable("the plan has duplicate points")
    per = {}
    for e in plan["plan"]:
        per.setdefault(e["shard"], set()).add(e["s_node"])
    missing = {k: sorted(set(nodes()) - v) for k, v in per.items() if len(v) < 5}
    if missing:
        raise NotTestable("shards are missing nodes: %r - losing a shard would "
                          "lose an entire grid node" % missing)
    return {"check": "план прогона объявлен до запуска и покрывает сетку",
            "status": "пройдена", "sha256_prefix": got[:16],
            "n_points": len(got_pts), "n_shards": plan["n_shards"],
            "points_per_shard": plan["points_per_shard"]}


def preflight(neurons, con, *, verbose: bool = True) -> dict:
    """All checks before the first run point. Failure - NotTestable."""
    res = [check_conformance(),
           check_run_plan(),
           check_detector_table(neurons, con),
           check_structural_scaling(neurons, con)]
    if verbose:
        for r in res:
            print("pre-run check: %s — %s" % (r["check"], r["status"]),
                  flush=True)
    return {"preflight": res}


# --- non-criterion output V1c-E7.2: distance-to-threshold map ----------------
def _mbon_index(neurons, core_ids) -> tuple[np.ndarray, list[int], list[str]]:
    role = dict(zip(neurons.root_id, neurons.mb_role))
    typ = dict(zip(neurons.root_id,
                   neurons.hemibrain_type.astype("string").fillna("<без типа>")))
    m = np.array([role[i] == "MBON" for i in core_ids])
    ids = [i for i, keep in zip(core_ids, m) if keep]
    return m, ids, [str(typ[i]) for i in ids]


def threshold_rows(by_odor: dict, neurons, *, cand_idx: int, cand: dict,
                   s_node: str, panel: str, run: str = "cal") -> pd.DataFrame:
    """Rows of the distance-to-threshold map: cell x odor x trial (V1c-E7.2).

        d = v_th − max_t v(t),

    the maximum over all simulator steps inside the window, v read before
    reset. d = 0 if and only if the cell spiked in that window; hence the
    measured value is set equal to zero rather than having its sign
    dropped.

    The v time traces are not saved; there are no aggregations in the
    primary artifact.
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
            raise NotTestable("the run for odor %r was done without the vmax observer: "
                              "the V1c-E7.2 map cannot be recorded" % odor)
        vm = res["vmax_mV"][mask]                 # cells x trials
        sp = res["counts"][mask] > 0              # spike within the presentation window
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
    # String columns stay strings, not categories: points have different s
    # nodes, and concatenating frames with mismatched category sets silently
    # yields object, and the artifact's column type would depend on
    # concatenation order. Parquet does dictionary encoding on its own.
    return pd.concat(frames, ignore_index=True)


def zero_without_spike(by_odor: dict, neurons) -> int:
    """Number of MBON presentations where measured v_th - vmax <= 0 without a spike.

    The measure of this event is zero: the threshold is strict (v > v_th),
    so a peak equal to the threshold does not produce a spike, yet leaves no
    distance either. The value is printed as an execution check with an
    expectation of 0 declared before the run; it is not a criterion.
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
    """Map summary: quantiles over presentations and medians over T38 types.

    Computed from the primary artifact and recomputable from it: the
    summary does not replace the map and does not participate in any
    criterion.
    """
    import v1b_subcircuit as V

    g = df.groupby(["cand_idx", "s_node", "mbon_id"], observed=True)["d_peak_mV"]
    per_cell = g.agg(min="min", q10=lambda x: float(np.quantile(x, 0.10)),
                     median="median",
                     q90=lambda x: float(np.quantile(x, 0.90)), max="max")
    # "the share of presentations with d = 0" is computed from d itself, not
    # from the spike flag: the two quantities are identical by construction,
    # but the spec names d specifically, and the agreement between the two
    # counts is checked separately by the zero_without_spike counter, with
    # an expectation of 0 declared beforehand
    per_cell["frac_zero"] = df.assign(_z=(df["d_peak_mV"] == 0)).groupby(
        ["cand_idx", "s_node", "mbon_id"], observed=True)["_z"].mean()
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


# --- non-criterion output V1c-E7.3: reproduction at node s = 1 ---------------
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
    """Bitwise equality: for float - equality of representations, NaN = NaN."""
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
    """The point in the form the shard writes it.

    V1c-E7.3 requires equality of representations, not equality of objects
    in memory, so the check runs over values that have gone through the
    same json.dumps the shard uses to write its file, and back through
    loading.
    """
    return json.loads(json.dumps(pt, ensure_ascii=False))


def reproduction_check(pt: dict) -> list[str]:
    """Check the node s = 1 point against the V1b' artifacts. Returns a list of discrepancies.

    Compared against the actual schema of the V1b' artifacts (V1c-E7.3):
    from the calibration maps - f_mean, f_max, f_by_odor, s_ab, s_apbp, s_g;
    from report_map - R_t, MD, min_R_t, max_R_t, violated. The tolerance is
    bitwise: multiplying a weight by s = 1 is an identity in IEEE
    arithmetic, so a weaker tolerance would only hide nondeterminism.
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


# --- halt signal ---------------------------------------------------------------
HALT = OUT / "HALT.json"


def halt(reason: str, detail) -> None:
    """Write the halt signal: shards check it before every point.

    V1c-E7.3 says "the run stops at the first detected discrepancy". With
    eight independent processes, each can only stop its own shard, so the
    first one to detect it writes the signal, and the rest read it.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    if not HALT.exists():
        HALT.write_text(json.dumps(
            {"outcome": "NOT-TESTABLE", "ground": "ошибка исполнения",
             "reason": reason, "detail": detail}, ensure_ascii=False, indent=2),
            encoding="utf-8")


def halted() -> dict | None:
    return json.loads(HALT.read_text(encoding="utf-8")) if HALT.exists() else None


# --- stage outcome (V1c-E4.1, E4.3, E4.4, E4.5) --------------------------------
def outcome(points: list[dict]) -> dict:
    """Outcome of the calibration part of the stage, per the frozen definitions.

    The computation order is declared before the run:

      K = the set of three-dimensional points that passed the share
          constraint on C at THEIR OWN node s. K empty -> FAIL-CAL-KC
          (V1c-E4.5, first sentence).
      band = the nodes at which there is a K point with the full
          conjunction (V1c-E4.1). Nonempty -> the outcome is not decided
          here: it requires evaluation on set E, and the blinding of E is
          spent only by the evaluation run.
      band empty and there is a K point that exceeds the ceiling -> CEIL
          (V1c-E4.5, priority rule).
      otherwise, at the top node s_max there is a K point with a type below
          the floor -> FLOOR.
      otherwise -> NOT-TESTABLE: definitions E4.3 and E4.5 do not cover the
          observed case, and assigning an outcome by analogy would be an
          interpretation made after the run.

    Ceiling violation is operationalized as `not ceiling_ok`, i.e.
    "there exists t from T38 with R_t > 67 Hz": the letter of V1b-3.4 says
    "for every type", while the quantifiers of E4.3 and E4.5 say "at least
    one".
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
    base["n_points_over_ceiling"] = len(ceil)
    base["max_R_t_over_K_hz"] = max(
        (max(v["R_t"] for v in p["mbon"]["per_type"].values())
         for p in k if p["mbon"]["per_type"]), default=float("nan"))
    base["max_R_t_over_all_points_hz"] = max(
        (max(v["R_t"] for v in p["mbon"]["per_type"].values())
         for p in points if p["mbon"]["per_type"]), default=float("nan"))
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
