# -*- coding: utf-8 -*-
"""Calibration of stage V1b: the stage 1 grid, sharded across independent processes.

The cost of one point is 6 calibration odors times 6 presentations. Three
things make the full grid take minutes instead of hours, and all three are
measured, not assumed:

  adaptive truncation    at the zero background of model [2] a network with
                        no input does not spike, so the run proceeds until
                        decay, not the whole measurement interval. Counts
                        are identical to the full run: verified, 0
                        mismatches on 18,255 counts.
  cython                 the Brian 2 code generator; 3.1x versus numpy on
                        this subcircuit, matching the V1a benchmark
                        (bench_backend.py).
  shards                 the subcircuit is 6,087 neurons, about 0.5 GB per
                        process, whereas in V1a parallelism ran into the
                        memory of the whole-brain model.

Why shards, not multiprocessing.Pool: on Windows venv spawns children via
sys._base_executable, and Pool deadlocked in this configuration - workers
came up and hung at 0% CPU. Independent processes do not have this class of
problem, debug one at a time, and survive a neighbor's crash.

Every process needs its own cython cache directory, otherwise compilations
race each other for the same files.

The run is resumable: points already recorded in a shard are skipped on restart.

Run (needs the MSVC environment, otherwise cython won't build):
    msvc_run.bat run_v1b_grid.py --shard 0 --of 8      # and so on for 0..7
    msvc_run.bat run_v1b_grid.py --merge
    python run_v1b_grid.py --shard 0 --of 1 --codegen numpy   # without MSVC, slower
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Stage V1b′ (specification v0.15, section 3ж). Artifacts are written to its
# own directory; the closed stage V1b's directory serves only as the
# comparison baseline.
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

    print("stage %d, shard %d of %d: points %d, computed %d, to run %d, codegen %s"
          % (a.stage, a.shard, a.of, len(mine), len(done), len(todo), a.codegen),
          flush=True)
    neurons, con = V.load_substrate()
    for k, (sc, g) in enumerate(todo, 1):
        t0 = time.time()
        pt = V.evaluate_point(neurons, con, V.PANEL_CAL, pn_kc_scale=sc,
                              g_apl_rel=g, seed_base=V.SEED_CAL)
        pt.pop("_by_odor")
        pt["stage"] = a.stage
        # Admissibility is a conjunction (V1b-4.5), so the MBON constraint on
        # C is computed only for points that passed on the fraction: a run
        # at timing P14-M costs four times more than a fraction run, and a
        # point inadmissible on the fraction cannot become admissible.
        # Points that did not pass have no mbon key, and that means "not
        # computed", not "passed".
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
            tail = "  ADMISSIBLE"
        else:
            t = pt["mbon"]
            broke = [n for n, v in (("V1b-3.3", t["floor_ok"]),
                                    ("V1b-3.4", t["ceiling_ok"]),
                                    ("V1b-3.5", t["md_ok"])) if not v]
            tail = "  passed on the fraction, no MBON: " + ", ".join(broke)
        print("  [%d/%d] scale %.4g g %.4g -> f %.4f, f_max %.4f, s_ab %.2f, %.0f s%s"
              % (k, len(todo), sc, g, pt["f_mean"], pt["f_max"], pt["s_ab"],
                 pt["wall_s"], tail), flush=True)
    print("shard %d ready" % a.shard, flush=True)
    return 0


def _grid_for(V, stage: int):
    if stage == 1:
        return V.grid_stage1()
    prev = merged_path(1)
    if not prev.exists():
        raise SystemExit("stage 1 is not assembled: run --merge --stage 1")
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
    print("points collected: %d of %d" % (len(pts), len(grid)))
    print("   passed on the fraction: %d; of these admissible under V1b-4.5: %d" % (n_frac, n_ok))
    if len(pts) < len(grid):
        print("grid is not complete — choosing a point is premature")
        return 0
    if n_unknown:
        # "not computed" is not "did not pass". Rendering an outcome on an
        # uncomputed constraint means repeating the very error that closed
        # V1b: finish computing first, then render the outcome.
        print("   %d points have an uncomputed MBON constraint: no outcome is rendered,"
              % n_unknown)
        print("   the grid needs to be finished with a run under the V1b-4.5 constraint")
        return 1
    # V1b-4.6: the selection is among stage 2's admissible points, not over the union of stages
    best = V.choose_point([p for p in pts if p.get("stage", a.stage) == a.stage])
    if best is None:
        if n_frac == 0:
            print("not a single point passed on the fraction — outcome FAIL-CAL-KC")
        else:
            print("%d points passed on the fraction, not one passed under V1b-4.5 — "
                  "outcome FAIL-CAL-MBON" % n_frac)
            worst = [(min((d["R_t"] for d in p["mbon"]["per_type"].values()),
                          default=float("nan")), p)
                     for p in pts if "mbon" in p]
            worst = [w for w in worst if w[0] == w[0]]
            if worst:
                lo, p = max(worst, key=lambda w: w[0])
                print("   the best the family gives on the floor: min over types R_t = "
                      "%.4f Hz (threshold %.1f) at point scale %.4g, g %.5g"
                      % (lo, V.MBON_FLOOR_HZ, p["pn_kc_scale"], p["g_apl_rel"]))
    else:
        print("chosen: pn_kc_scale %.5g, g_apl %.5g g_ref, f %.4f, s_ab %.2f"
              % (best["pn_kc_scale"], best["g_apl_rel"], best["f_mean"], best["s_ab"]))
    return 0


def check_preconditions() -> int:
    """Preconditions for running the stage: V1b'-0 and V1b'-1а.

    V1b'-0: the code conformance to specification test passes in full.
    A failure is an execution error, the stage does not start.
    V1b'-1а: the responding-fraction map at the nodes already computed by
    the pilot matches it bitwise. The calibration seeds are the same, so a
    mismatch means non-determinism and is investigated before calibration.
    """
    import subprocess
    r = subprocess.run([sys.executable, str(HERE / "test_spec_conformance.py")],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        print("V1b'-0: the code conformance to specification test FAILED, "
              "the stage does not start", file=sys.stderr)
        print(r.stdout[-3000:], file=sys.stderr)
        return 1
    print("V1b'-0: the code conformance to specification test passed", flush=True)

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
                    print("V1b'-1а: mismatch with the pilot at point %s, field %s: "
                          "%r versus %r — non-determinism, investigate before calibration"
                          % (key, f, x[f], y[f]), file=sys.stderr)
                    return 1
    print("V1b'-1а: comparison with the pilot — %d shared points, no mismatches"
          % checked, flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--codegen", default="cython", choices=["numpy", "cython"])
    ap.add_argument("--limit", type=int, default=0, help="run only N points")
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2])
    ap.add_argument("--merge", action="store_true", help="assemble shards and select the point")
    ap.add_argument("--skip-preconditions", action="store_true",
                    help="skip checking preconditions V1b'-0 and V1b'-1а; debugging "
                         "only, does not count as a confirming run")
    a = ap.parse_args()

    sys.path.insert(0, str(HERE))
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("CALYX_V1B_OUT", str(OUT))
    if not (OUT / "spec_sha256.txt").exists():
        print("pre-registration is not frozen: run "
              "scripts/freeze_v1b_prime.py", file=sys.stderr)
        return 1
    if not a.skip_preconditions and check_preconditions() != 0:
        return 1
    return merge(a) if a.merge else run_shard(a)


if __name__ == "__main__":
    raise SystemExit(main())
