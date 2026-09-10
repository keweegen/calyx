# -*- coding: utf-8 -*-
"""V1c stage runner: the third knob kc_mbon_scale over 285 three-dimensional points.

The stage is pre-registered by specification v0.17 (hash 636c49968a0866bd,
sections 3з and 3и) together with the config_v1c.json config and the
detector_table.json detector table. The runner executes it and decides
nothing in it.

Everything the run does, to the letter of the frozen text:

  grid           57 V1b' candidates over five nodes s, nodes are read from
                 the config AS STRINGS and are not recomputed at runtime;
  order          first all 57 points of node s = 1, then the remaining 228;
                 this closes the V1c-E7.3 check within the first hour, not
                 the third;
  fraction       recomputed at every three-dimensional point, not inherited
                 (V1c-E3.3): the MBON->APL feedback gives 2.3% of APL's
                 input, and invariance of sparseness to s is not established;
  MBON           P14-M is computed at ALL 285 points, not only at those that
                 passed on the fraction: this is required by the
                 non-criterial output V1c-E7.2 ("for every three-dimensional
                 grid point", about a million rows = 285 x 97 x 36). Only
                 points with candidate_at_s = true enter the criterion;
  map            distance to threshold per presentation, without time traces
                 (V1c-E7.2);
  reproduction   at every point of node s = 1 - a bitwise comparison against
                 the V1b' artifacts (V1c-E7.3); a mismatch writes HALT.json
                 and closes the stage with outcome NOT-TESTABLE.

Every shard executes three pre-run checks before its first point: the test
of code conformance to the specification, a recomputation of the detector
table from its own loader's weights, and the structural scaling test on the
assembled network. The point of the latter two is "the simulator sees
different weights than the ones s_max was computed from" - a property of the
process, not of the text, so the check runs in every process.

Run:
    python run_v1c_grid.py --plan                     # plan of points and shards
    python run_v1c_grid.py --preflight                # three checks, artifact
    python run_v1c_grid.py --pilot --codegen numpy    # execution pilot
    run_shard_v1c.bat 0 8                             # and so on for 0..7
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


# --- run plan -------------------------------------------------------------------
def write_plan(n_shards: int) -> int:
    """The order of points and the shard assignment, recorded BEFORE the run.

    Not in config_v1c.json: that one is frozen together with the
    specification, its SHA-256 is recorded in spec_sha256.txt as part of
    the pre-registration, and appending anything to it would mean breaking
    the hash. The plan is a separate artifact, declared in the report on
    par with everything else declared before the run.
    """
    import v1c_stage as S

    g = S.grid_3d()
    ns = S.nodes()
    # Shard assignment - by the point's rank WITHIN its node, with an offset
    # by node. The naive "shard = index mod 8" degenerates on this order:
    # the tail runs in groups of four nodes, 4 divides 8, and each shard
    # gets exactly one node out of four. Then losing one shard loses an
    # entire grid node, and no outcome is rendered on an incomplete grid at
    # all. An offset of 3 (coprime with 8) spreads each node's 57 points
    # across all eight shards.
    rank = {}
    plan = []
    for k, (i, c, s_node) in enumerate(g):
        r = rank.get(s_node, 0)
        rank[s_node] = r + 1
        shard = (r + 3 * ns.index(s_node)) % n_shards
        plan.append({"idx": k, "shard": shard, "cand_idx": i, "s_node": s_node,
                     "pn_kc_scale": c["pn_kc_scale"],
                     "g_apl_rel": c["g_apl_rel"]})
    doc = {"stage": "V1c", "artefact_kind": "run plan, declared before the run",
           "spec_hash_prefix": "636c49968a0866bd",
           "n_points": len(plan), "n_shards": n_shards,
           "order": "first all 57 candidates at node s = 1, then 228 points "
                    "in an outer loop over the candidate and an inner loop "
                    "over the four remaining nodes",
           "shard_assignment": "by the point's rank within its node with an "
                               "offset of 3 on the node number, modulo %d: "
                               "each shard gets 7-8 points of EVERY node"
                               % n_shards,
           "shard_assignment_ground": "the naive \"index mod %d\" degenerates "
                                      "on this order: the tail runs in "
                                      "groups of four nodes, 4 divides %d, "
                                      "and a shard gets exactly one node out "
                                      "of four; losing a shard would lose an "
                                      "entire grid node, and no outcome is "
                                      "rendered on an incomplete grid"
                                      % (n_shards, n_shards),
           "order_ground": "V1c-E7.3 stops the run at the first mismatch "
                           "with V1b'; the s = 1 block closes all 57 checks "
                           "first, within the first hour of the run",
           "order_does_not_affect_numbers": "the network is rebuilt fresh "
                                            "for every point, the seed is "
                                            "set for every trial, each shard "
                                            "has its own cython cache; the "
                                            "shared V1b' stage nodes, "
                                            "computed by independent "
                                            "processes, matched bitwise",
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
        print("plan already recorded: %s. Overwriting is forbidden: the "
              "order is declared BEFORE the run, and rewriting it later "
              "would mean declaring it after the fact." % p, file=sys.stderr)
        return 1
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    h = sha(p)
    (OUT / "run_plan_sha256.txt").write_text(
        "# The order of 285 points and the shard assignment of stage V1c,\n"
        "# declared BEFORE the run starts. The hash is recorded here, not\n"
        "# in spec_sha256.txt: that one is frozen together with the\n"
        "# specification, and appending anything to it would mean breaking\n"
        "# the pre-registration.\n#\n"
        "# Date recorded: 2026-09-09\n"
        "run_plan.json %d %s\n" % (p.stat().st_size, h), encoding="utf-8")
    print("plan recorded: %s\n%d points, %d shards, per shard: %s\nhash: %s"
          % (p, len(plan), n_shards,
             ", ".join("%s:%d" % kv for kv in doc["points_per_shard"].items()),
             h[:16]))
    return 0


# --- pre-run checks ---------------------------------------------------------
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
        print("pre-run checks recorded: %s (fingerprint %s)"
              % (OUT / "preflight.json", res["fingerprint"]))
    return res


# --- one three-dimensional point -----------------------------------------------
def evaluate_3d(V, S, neurons, con, cand_idx: int, cand: dict, s_node: str):
    """One point: the fraction at P14, the MBON response at P14-M, the threshold map.

    Returns (point, map frame, number of "d = 0 without a spike" anomalies).
    """
    s = float(s_node)
    pt = V.evaluate_point(neurons, con, V.PANEL_CAL,
                          pn_kc_scale=cand["pn_kc_scale"],
                          g_apl_rel=cand["g_apl_rel"], seed_base=V.SEED_CAL,
                          kc_mbon_scale=s)
    pt.pop("_by_odor")
    pt["cand_idx"] = cand_idx
    pt["s_node"] = s_node
    # the fraction constraint is recomputed at this three-dimensional point,
    # not inherited
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


# --- execution pilot -------------------------------------------------------------
def run_pilot(a) -> int:
    """One point s = 1 with and without the observer; comparing trains and E7.3 fields.

    The pilot is not counted as the run's first point: it writes nothing
    into the stage map and saves not a single distance-to-threshold value.
    Its purpose is to catch observer non-inertness within twelve minutes,
    not an hour into the run, and to do so at the level of spike TRAINS -
    something V1c-E7.3 cannot provide: V1b' has aggregates frozen, not
    trains. Cell indices and spike times of all 72 presentations of both
    timings on set C are compared, then the counts in the measurement window.
    """
    import numpy as np
    import v1b_subcircuit as V
    import v1c_stage as S

    neurons, con = V.load_substrate()
    run_preflight(neurons, con)

    cand = S.candidates()[0]
    kw = dict(pn_kc_scale=cand["pn_kc_scale"],
              g_apl_rel=cand["g_apl_rel"] * V.g_ref_value())
    print("pilot: candidate 0, pn_kc_scale %r, g_apl_rel %r"
          % (cand["pn_kc_scale"], cand["g_apl_rel"]), flush=True)

    trains, counts, n_spikes = {}, {}, {}
    for tag, rec in (("without observer", False), ("with observer", True)):
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
        print("  %s: %.0f s, presentations %d, spikes %d"
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
    print("pilot: trains on %d presentations (%d spikes) — %s; counts — %s"
          % (n_pres, n_spikes[False],
             "match bitwise" if same_t else "DIVERGE",
             "match" if same_c else "DIVERGE"), flush=True)

    doc = {"check": "execution pilot before the V1c run",
           "status": "passed" if same else "FAILED",
           "ground": "vmax observer inertness on the stage's real path",
           "candidate": {"cand_idx": 0, "pn_kc_scale": cand["pn_kc_scale"],
                         "g_apl_rel": cand["g_apl_rel"]},
           "n_presentations_compared": n_pres,
           "n_spikes_compared": n_spikes[False],
           "timings": ["P14", "P14-M"], "panel": "C",
           "spike_trains_identical": bool(same_t),
           "spike_counts_identical": bool(same_c),
           "compared": "cell indices and spike times of every presentation, "
                       "then the counts in the measurement window",
           "vmax_values_written": False,
           "note": "the pilot's vmax values are not recorded: the part of "
                   "the report declared before the run contains nothing "
                   "about distance to threshold",
           "hashes": artefact_hashes()}
    PRERUN.mkdir(parents=True, exist_ok=True)
    (PRERUN / "pilot_s1.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print("recorded: %s" % (PRERUN / "pilot_s1.json"))
    return 0 if same else 1


# --- shard -----------------------------------------------------------------------
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
        print("run stopped by signal %s: %s"
              % (S.HALT, S.halted()["reason"]), file=sys.stderr)
        return 1

    neurons, con = V.load_substrate()
    pre = run_preflight(neurons, con, write=False)

    # The order and the split are taken from the plan: it is declared before
    # the run and hashed, and recomputing the split in the runner would let
    # it silently diverge from the plan.
    plan_p = OUT / "run_plan.json"
    if not plan_p.exists():
        print("run plan not recorded: run --plan", file=sys.stderr)
        return 1
    plan = json.loads(plan_p.read_text(encoding="utf-8"))
    if plan["n_shards"] != a.of:
        print("plan recorded for %d shards, %d requested"
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
    print("shard %d of %d: points %d, computed %d, to run %d, codegen %s"
          % (a.shard, a.of, len(mine), len(done), len(todo), a.codegen),
          flush=True)

    frames = [pd.read_parquet(mpath)] if mpath.exists() else []
    for n, (k, (i, c, s_node)) in enumerate(todo, 1):
        if S.halted():
            print("stop signal received, shard is halting",
                  file=sys.stderr)
            return 1
        t0 = time.time()
        pt, df, n_anom = evaluate_3d(V, S, neurons, con, i, c, s_node)
        pt["zero_without_spike"] = n_anom
        pt["wall_s"] = round(time.time() - t0, 1)

        if s_node == "1":
            diffs = S.reproduction_check(pt)
            pt["reproduction_v1b_prime"] = "match" if not diffs else diffs
            if diffs:
                S.halt("V1c-E7.3: mismatch with the V1b' artifacts at node s = 1",
                       {"cand_idx": i, "pn_kc_scale": c["pn_kc_scale"],
                        "g_apl_rel": c["g_apl_rel"], "fields": diffs})
                print("V1c-E7.3 FAILED at point %d: %s"
                      % (i, "; ".join(diffs[:5])), file=sys.stderr)
                return 1

        doc["points"].append(pt)
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        frames.append(df)
        pd.concat(frames, ignore_index=True).to_parquet(mpath, index=False)

        m = pt["mbon"]
        tail = ("candidate at s" if pt["candidate_at_s"] else "did not pass on the fraction")
        if pt["candidate_at_s"]:
            broke = [nm for nm, ok in (("3.3", m["floor_ok"]),
                                       ("3.4", m["ceiling_ok"]),
                                       ("3.5", m["md_ok"])) if not ok]
            tail += ", ADMISSIBLE" if not broke else ", violated " + ",".join(broke)
        print("  [%d/%d] point %d: candidate %d, s %s -> f %.4f, min R_t %.4f, "
              "max R_t %.4f, %.0f s; %s"
              % (n, len(todo), k, i, s_node, pt["f_mean"],
                 min((v["R_t"] for v in m["per_type"].values()), default=float("nan")),
                 max((v["R_t"] for v in m["per_type"].values()), default=float("nan")),
                 pt["wall_s"], tail), flush=True)
    print("shard %d ready" % a.shard, flush=True)
    return 0


# --- merge -----------------------------------------------------------------------
def merge(a) -> int:
    import pandas as pd
    import v1b_subcircuit as V
    import v1c_stage as S

    h = S.halted()
    if h:
        print("run stopped by signal %s: %s" % (S.HALT, h["reason"]),
              file=sys.stderr)
        print("stage outcome — %s, ground: %s" % (h["outcome"], h["ground"]),
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
    print("points collected: %d of %d" % (len(pts), len(grid)))

    fps = {h.get("preflight_fingerprint") for h in heads if h}
    print("pre-run check fingerprints among shards: %d %s"
          % (len(fps), sorted(x for x in fps if x)))
    if len(fps) > 1:
        print("shards were run under different pre-run checks — "
              "this is an execution error", file=sys.stderr)
        S.halt("shards' pre-run check fingerprints differ", sorted(fps))

    (OUT / "v1c_grid.json").write_text(
        json.dumps(pts, ensure_ascii=False, indent=2), encoding="utf-8")

    # V1c-E7.3 again, over the assembled artifact: recorded for the report
    repro, bad = [], []
    for pt in pts:
        if pt["s_node"] != "1":
            continue
        d = S.reproduction_check(pt)
        repro.append({"cand_idx": pt["cand_idx"], "diffs": d})
        if d:
            bad.append(pt["cand_idx"])
    print("V1c-E7.3 at merge: %d points of node s = 1 checked, mismatches in %d"
          % (len(repro), len(bad)))

    # threshold distance map
    frames = [pd.read_parquet(p) for p in sorted(OUT.glob("threshold_map.shard*.parquet"))]
    summ = None
    if frames:
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(["cand_idx", "s_node", "mbon_id", "run", "odor",
                                 "trial"])
        df.to_parquet(OUT / "threshold_map.parquet", index=False)
        print("threshold distance map: %d rows -> %s"
              % (len(df), OUT / "threshold_map.parquet"))
        summ = S.threshold_summary(df)
        (OUT / "threshold_map_summary.json").write_text(
            json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")

    if len(pts) < len(grid):
        print("grid is not complete — no outcome is rendered")
        return 0
    if any(p.get("mbon") is None for p in pts):
        print("MBON response was not computed for some points: no outcome is rendered",
              file=sys.stderr)
        return 1

    res = S.outcome(pts)
    res["reproduction_v1b_prime"] = {
        "n_checked": len(repro), "n_mismatched": len(bad),
        "mismatched_cand_idx": bad}
    res["zero_without_spike_total"] = sum(p.get("zero_without_spike", 0) for p in pts)
    res["hashes"] = artefact_hashes()
    if bad:
        # V1c-E7.3: a mismatch with V1b' at node s = 1 closes the stage with
        # outcome NOT-TESTABLE, not FAIL: it is a statement about execution,
        # not about the model family
        res["outcome_before_reproduction_check"] = res["outcome"]
        res["outcome"] = "NOT-TESTABLE"
        res["ground"] = ("execution error: the V1c-E7.3 reproduction check "
                         "did not pass at %d points of node s = 1" % len(bad))
        res["stopping_rule"] = "not derived: no stage outcome was obtained"
    res["headline"] = _headline(V, pts)
    (OUT / "report_map.json").write_text(
        json.dumps({"summary": res, "points": pts}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    print("\nstage V1c outcome: %s" % res["outcome"])
    print("  ground: %s" % res["ground"])
    print("  points passing on the fraction at their own s: %d of %d"
          % (res["n_in_K"], res["n_points"]))
    print("  band: %s" % (res.get("band") if res.get("band") else "empty"))
    print("  \"d = 0 without a spike\" check: %d, expected 0"
          % res["zero_without_spike_total"])
    for k, v in res["headline"].items():
        print("  %s: %s" % (k, v))
    return 0


def _headline(V, pts: list[dict]) -> dict:
    """Headline outcome values, computed over the points of set K."""
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
        print("pre-registration is not frozen", file=sys.stderr)
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
