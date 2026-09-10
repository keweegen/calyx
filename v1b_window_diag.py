# -*- coding: utf-8 -*-
"""Gap diagnostic: the KC responding-fraction window versus the MBON rate window.

Outside the stage. Not a single criterion is computed here as a criterion,
and not a single parameter is chosen from the results: the purpose is to
give quantitative shape to the incompatibility established by the V1b' run
(outcome FAIL-CAL-MBON). The KC->MBON multiplier is not touched: that is the
V1c knob, and its first measurement must pass under a hash.

The two axes are measured differently, and that is not carelessness but a
property of the criteria.

  KC responding fraction   defined at timing P14 by rule V1b-2.1-2.3: at
                       least one spike in at least three trials out of six.
                       At two trials the rule is unreachable and f is
                       identically zero, so the fraction is NOT recomputed
                       here but read from the V1b' calibration artifacts -
                       there it was computed over six trials on the SEED_CAL
                       seeds at both needed scales and 21 g_apl values.
  MBON rate R_t        defined at timing M as the mean over trials and over
                       cells of the type (V1b-3.1). It has no "at least half
                       the trials" rule, so two trials do evaluate it, and
                       only it is run here.

Odor set - calibration set C: the blindness of set E is not spent (V1b-4.7).
Seeds SEED_CAL_M + 1..2 are the very same first two trials the stage used
for the admissibility constraint, so the values are directly comparable to
the candidates.

Artifacts are written to results/diag_window: the closed stages' V1b and
V1b' directories serve as the comparison baseline and are not overwritten.

Run (needs the MSVC environment, otherwise cython won't build):
    python  v1b_window_diag.py --map                    # no run, instant
    msvc_run.bat v1b_window_diag.py --scan --shard 0 --of 8   # and so on for 0..7
    msvc_run.bat v1b_window_diag.py --scan --grid refine --shard 0 --of 8
    python  v1b_window_diag.py --merge
    msvc_run.bat v1b_window_diag.py --verify            # comparison with the stage artifact
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PRIME = HERE / "results" / "v1b_prime"      # source of the fraction map, read-only
OUT = HERE / "results" / "diag_window"      # own directory, overwrites nothing

# Two scales: 8 - the upper corner of the stage 2 grid and the pilot's
# chosen point, 4 - half as much, the only other scale whose fraction
# window lies entirely inside the grid.
SCALES = (8.0, 4.0)

# Ten g_apl values from zero to the maximum of the stage 2 grid. All but
# zero lie on the stage 2 grid nodes (step 10^(1/8)); zero is a stage 1
# node. The nodes are chosen so the range is covered roughly evenly in log
# g and so that it includes points for which the stage has already computed
# MBON over six trials.
G_VALUES = (0.0, 0.01, 0.031623, 0.1, 0.17783,
            0.31623, 0.56234, 1.0, 1.7783, 3.16228)

# Resolution refinement (rule Е2). Not a fixed ladder of nodes but a
# mechanical dichotomy, declared in the measurement journal
# results/diag_window/journal_e6_2.md BEFORE the run: while the lower end
# of the bracket equals zero or the minimum R_t there is below the floor -
# linear bisection; then bisection by logarithm, until the ratio of the
# bracket ends is 1.25 or less. The limit is declared and hard. Main-pass
# nodes are not recomputed.
REFINE_MAX_NODES = 8            # per scale, limit from the journal
REFINE_LOG_RATIO = 1.25         # stop condition for step 2

N_TRIALS_DIAG = 2
# The verification point is not given as a constant but chosen as the
# stage candidate with the largest nonzero rate: a comparison at a point
# where all six types are silent would pass even with a broken testbed.
VERIFY_PICK = "V1b' candidate with the maximum R_t over T38"


# --- KC responding-fraction axis: reading artifacts, no run --------------------

def _load(p: Path) -> list:
    if not p.exists():
        raise SystemExit("no artefact: %s" % p)
    return json.loads(p.read_text(encoding="utf-8"))


def fraction_map(V) -> dict:
    """KC responding-fraction window along the g_apl axis, from V1b' calibration artifacts.

    A point enters the window if the FIRST half of V1b-4.5 holds in full:
    f_mean in the band AND f_max no higher than the ceiling. This is the
    same passes_fraction function the stage used - here it does not
    recompute the criterion but is applied to already-recorded values.
    """
    pts = _load(PRIME / "calibration_stage1.json") + \
          _load(PRIME / "calibration_stage2.json")
    out = {"source": ["results/v1b_prime/calibration_stage1.json",
                      "results/v1b_prime/calibration_stage2.json"],
           "rule": "V1b-2.1-2.3 under timing P14: >=1 spike in >=3 trials out of 6",
           "band": list(V.F_BAND), "ceiling": V.F_MAX,
           "seed_base": V.SEED_CAL, "n_trials": V.N_TRIALS, "by_scale": {}}
    dup_checked = 0
    for sc in SCALES:
        rows = [p for p in pts if abs(p["pn_kc_scale"] - sc) < 1e-9]
        # The stage grids overlap: a node landing in both is recorded
        # twice. The duplicate is dropped, but the values are compared
        # first - the match must be bitwise, because both stages share the
        # same seeds and protocol.
        by_g = {}
        for p in rows:
            k = round(p["g_apl_rel"], 10)
            if k in by_g:
                if (by_g[k]["f_mean"], by_g[k]["f_max"]) != (p["f_mean"], p["f_max"]):
                    raise SystemExit(
                        "grid node (scale %g, g %g) recorded differently by the stages: "
                        "%r vs %r - nondeterminism, resolve before the report"
                        % (sc, p["g_apl_rel"],
                           (by_g[k]["f_mean"], by_g[k]["f_max"]),
                           (p["f_mean"], p["f_max"])))
                dup_checked += 1
                # prefer the stage 2 point: it has MBON computed
                if "mbon" in p and "mbon" not in by_g[k]:
                    by_g[k] = p
                continue
            by_g[k] = p
        rows = sorted(by_g.values(), key=lambda p: p["g_apl_rel"])
        curve = [{"g_apl_rel": p["g_apl_rel"], "f_mean": p["f_mean"],
                  "f_max": p["f_max"], "in_window": bool(V.passes_fraction(p)),
                  "mbon_computed_by_stage": "mbon" in p} for p in rows]
        win = [c["g_apl_rel"] for c in curve if c["in_window"]]
        out["by_scale"]["%g" % sc] = {
            "n_grid_points": len(curve),
            "g_min_grid": curve[0]["g_apl_rel"] if curve else None,
            "g_max_grid": curve[-1]["g_apl_rel"] if curve else None,
            "window_g_lo": min(win) if win else None,
            "window_g_hi": max(win) if win else None,
            "n_in_window": len(win),
            "curve": curve}
    out["duplicate_nodes_checked"] = dup_checked
    return out


# --- MBON rate axis: run at timing M --------------------------------------------

def _rates_at(V, neurons, con, sc: float, g: float, seeds: list[int]) -> dict:
    """MBON and KC values at one point under timing M, set C."""
    role = dict(zip(neurons.root_id, neurons.mb_role))
    n_kc_total = int(sum(1 for r in role.values() if r == "Kenyon_Cell"))
    by = {}
    kc_spk = mb_spk = 0
    kc_resp = np.zeros(0)
    for o in V.PANEL_CAL:
        res = V.run_odor(neurons, con, o, pn_kc_scale=sc,
                         g_apl=g * V.g_ref_value(), seeds=seeds,
                         pulse_ms=V.M_PULSE_MS, window_ms=V.M_WINDOW_MS)
        by[o] = res
        ids = res["core_ids"]
        kc = np.array([role[i] == "Kenyon_Cell" for i in ids])
        mb = np.array([role[i] == "MBON" for i in ids])
        kc_spk += int(res["counts"][kc].sum())
        mb_spk += int(res["counts"][mb].sum())
        r = ((res["counts"][kc] >= 1).sum(axis=1) >= 1)
        kc_resp = r if kc_resp.size == 0 else (kc_resp | r)

    rates = V.mbon_type_rates(by, neurons, V.M_WINDOW_MS)
    chk = V.check_mbon(rates)
    per = chk["per_type"]
    rt = {t: per[t]["R_t"] for t in chk["types_present"]}
    n_mbon = int(sum(1 for r in role.values() if r == "MBON"))
    active = set()
    for o, res in by.items():
        ids = res["core_ids"]
        for k, i in enumerate(ids):
            if role[i] == "MBON" and res["counts"][k].sum() > 0:
                active.add(i)
    return {"pn_kc_scale": sc, "g_apl_rel": g,
            "R_t_T38": rt,
            "R_t_min": min(rt.values()) if rt else None,
            "R_t_max": max(rt.values()) if rt else None,
            "n_types_at_zero": int(sum(1 for v in rt.values() if v == 0.0)),
            "floor_ok": bool(chk["floor_ok"]), "ceiling_ok": bool(chk["ceiling_ok"]),
            "corridor_ok": bool(chk["floor_ok"] and chk["ceiling_ok"]),
            "types_missing": chk["missing"],
            "kc_spikes_6_odors": kc_spk, "mbon_spikes_6_odors": mb_spk,
            "mbon_active_cells": len(active), "mbon_cells_total": n_mbon,
            # a diagnostic fraction, NOT a criterion: rule V1b-2.1-2.3
            # requires three trials out of six and is inapplicable at two
            # trials; here it is "at least one spike in at least one of the
            # two trials on at least one odor of C"
            "kc_responding_ge1_of2_any_odor_DIAG":
                float(kc_resp.sum()) / n_kc_total,
            "n_kc_denominator": n_kc_total}


def shard_path(i: int, n: int, grid: str = "main") -> Path:
    stem = "mbon_scan" if grid == "main" else "mbon_scan_refine"
    return OUT / ("%s.shard%02d_of%02d.json" % (stem, i, n))


def scan(V, neurons, con, shard: int, of: int, which: str = "main") -> int:
    grid = [(sc, g) for sc in SCALES for g in G_VALUES]
    mine = [p for k, p in enumerate(grid) if k % of == shard]
    path = shard_path(shard, of, which)
    done = _load(path) if path.exists() else []
    seen = {(round(p["pn_kc_scale"], 10), round(p["g_apl_rel"], 10)) for p in done}
    todo = [p for p in mine if (round(p[0], 10), round(p[1], 10)) not in seen]
    seeds = [V.SEED_CAL_M + i for i in range(1, N_TRIALS_DIAG + 1)]
    print("pass %s, shard %d of %d: points %d, computed %d, to run %d"
          % (which, shard, of, len(mine), len(done), len(todo)), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for k, (sc, g) in enumerate(todo, 1):
        t0 = time.time()
        row = _rates_at(V, neurons, con, sc, g, seeds)
        row["wall_s"] = round(time.time() - t0, 1)
        row["seeds"] = seeds
        row["grid_pass"] = which
        done.append(row)
        path.write_text(json.dumps(done, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        print("  [%d/%d] scale %-4g g %-9.5g | KC %9d, MBON %7d, active %3d/%d "
              "| min R_t %8.4f, max R_t %9.4f | %.0f s"
              % (k, len(todo), sc, g, row["kc_spikes_6_odors"],
                 row["mbon_spikes_6_odors"], row["mbon_active_cells"],
                 row["mbon_cells_total"], row["R_t_min"], row["R_t_max"],
                 row["wall_s"]), flush=True)
    print("shard %d ready" % shard, flush=True)
    return 0


def refine(V, neurons, con, sc: float) -> int:
    """Resolution refinement per rule Е2 for one scale.

    The dichotomy is sequential: the next node is determined by the
    previous one, so it is not split across shards. The rule and the limit
    are declared in the measurement journal BEFORE the run; here they are
    only executed.
    """
    seeds = [V.SEED_CAL_M + i for i in range(1, N_TRIALS_DIAG + 1)]
    path = OUT / ("mbon_scan_refine.scale%g.json" % sc)
    done = _load(path) if path.exists() else []

    # all known nodes of this scale: the main pass plus what is already refined
    known = []
    for q in sorted(OUT.glob("mbon_scan.shard*.json")):
        known += [r for r in _load(q) if abs(r["pn_kc_scale"] - sc) < 1e-9]
    known += done
    known = {round(r["g_apl_rel"], 12): r for r in known}

    def measure(g: float) -> dict:
        k = round(g, 12)
        if k in known:
            return known[k]
        t0 = time.time()
        row = _rates_at(V, neurons, con, sc, g, seeds)
        row["wall_s"] = round(time.time() - t0, 1)
        row["seeds"] = seeds
        row["grid_pass"] = "refine"
        done.append(row)
        known[k] = row
        path.write_text(json.dumps(done, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        print("  added node g = %.9g | KC %9d, MBON %7d, active %3d "
              "| min R_t %8.4f, max R_t %9.4f | %.0f s"
              % (g, row["kc_spikes_6_odors"], row["mbon_spikes_6_odors"],
                 row["mbon_active_cells"], row["R_t_min"], row["R_t_max"],
                 row["wall_s"]), flush=True)
        return row

    added = len(done)
    curve = sorted(known.values(), key=lambda r: r["g_apl_rel"])
    floor = V.MBON_FLOOR_HZ
    above = [r for r in curve if r["R_t_min"] >= floor]

    if not above:
        # Degenerate journal case: the floor is not crossed, because the
        # minimum is already below it at the most favorable point of the
        # g = 0 axis. There is nothing to cross; two nodes are added as a
        # monotonicity check.
        best = curve[0]
        print("scale %g: does not cross the floor - minimum R_t = %.4f Hz "
              "already at g = %.4g. Refinement is degenerate, two "
              "monotonicity-check nodes are added." % (sc, best["R_t_min"], best["g_apl_rel"]))
        for g in (0.0025, 0.005):
            if added >= REFINE_MAX_NODES:
                break
            measure(g)
            added += 1
        return 0

    g_lo = max(r["g_apl_rel"] for r in above)
    hi = [r for r in curve if r["g_apl_rel"] > g_lo]
    if not hi:
        print("scale %g: the floor is not violated at any node - nothing to refine" % sc)
        return 0
    g_hi = min(r["g_apl_rel"] for r in hi)
    print("scale %g: starting bracket (%.9g; %.9g), limit %d nodes"
          % (sc, g_lo, g_hi, REFINE_MAX_NODES))

    while added < REFINE_MAX_NODES:
        lo_row = known[round(g_lo, 12)]
        if g_lo == 0.0 or lo_row["R_t_min"] < floor:
            g_new = 0.5 * (g_lo + g_hi)          # step 1: linear
        else:
            if g_hi / g_lo <= REFINE_LOG_RATIO:  # step 2 complete
                break
            g_new = math.sqrt(g_lo * g_hi)       # step 2: by logarithm
        row = measure(g_new)
        added += 1
        if row["R_t_min"] >= floor:
            g_lo = g_new
        else:
            g_hi = g_new

    print("scale %g: final bracket (%.9g; %.9g), nodes added %d"
          % (sc, g_lo, g_hi, added))
    return 0


def fraction_at(V, neurons, con, sc: float, gs: list) -> int:
    """KC responding fraction at declared nodes: timing P14, six trials.

    The protocol is the same as output 1's source (V1b' calibration
    artifacts): the same stage function, the same seeds, the same "at
    least three trials out of six" rule. The nodes are declared in the
    measurement journal before the run.
    """
    path = OUT / ("fraction_extra.scale%g.json" % sc)
    done = _load(path) if path.exists() else []
    seen = {round(r["g_apl_rel"], 12) for r in done}
    OUT.mkdir(parents=True, exist_ok=True)
    for g in gs:
        if round(g, 12) in seen:
            continue
        t0 = time.time()
        pt = V.evaluate_point(neurons, con, V.PANEL_CAL, pn_kc_scale=sc,
                              g_apl_rel=g, seed_base=V.SEED_CAL)
        pt.pop("_by_odor")
        pt["wall_s"] = round(time.time() - t0, 1)
        pt["grid_pass"] = "refine_fraction"
        pt["passes_fraction"] = bool(V.passes_fraction(pt))
        done.append(pt)
        path.write_text(json.dumps(done, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        print("  g = %.9g | f_mean %.4f, f_max %.4f, in band: %s | %.0f s"
              % (g, pt["f_mean"], pt["f_max"],
                 "yes" if pt["passes_fraction"] else "no", pt["wall_s"]),
              flush=True)
    return 0


# --- assembly and threshold crossing ---------------------------------------------

def _interp_log(x0, y0, x1, y1, y) -> float:
    """The point where level y crosses, by linear interpolation in log10 x and log10 y."""
    lx0, lx1 = math.log10(x0), math.log10(x1)
    ly0, ly1 = math.log10(y0), math.log10(y1)
    return 10 ** (lx0 + (lx1 - lx0) * (math.log10(y) - ly0) / (ly1 - ly0))


def _f_at(curve: list, g: float) -> dict:
    """KC responding fraction at a given g: a grid node or interpolation in log g."""
    pos = [c for c in curve if c["g_apl_rel"] > 0]
    for c in pos:
        if abs(c["g_apl_rel"] - g) < 1e-9:
            return {"f_mean": c["f_mean"], "kind": "grid node"}
    lo = [c for c in pos if c["g_apl_rel"] < g]
    hi = [c for c in pos if c["g_apl_rel"] > g]
    if not lo or not hi:
        return {"f_mean": None, "kind": "outside the grid"}
    a, b = lo[-1], hi[0]
    la, lb = math.log10(a["g_apl_rel"]), math.log10(b["g_apl_rel"])
    w = (math.log10(g) - la) / (lb - la)
    return {"f_mean": a["f_mean"] + w * (b["f_mean"] - a["f_mean"]),
            "kind": "interpolation in log g between %.5g and %.5g"
                    % (a["g_apl_rel"], b["g_apl_rel"]),
            "bracket_f": [a["f_mean"], b["f_mean"]],
            "bracket_g": [a["g_apl_rel"], b["g_apl_rel"]]}


def merge(V) -> dict:
    rows = []
    for p in sorted(OUT.glob("mbon_scan.shard*.json")):
        rows += _load(p)
    refined = []
    for p in sorted(OUT.glob("mbon_scan_refine.*.json")):
        refined += _load(p)
    # A node computed by the main pass is not replaced by refinement: the
    # order of decisions is preserved in the artifact, not overwritten.
    seen = {(round(r["pn_kc_scale"], 10), round(r["g_apl_rel"], 12)) for r in rows}
    rows += [r for r in refined
             if (round(r["pn_kc_scale"], 10), round(r["g_apl_rel"], 12)) not in seen]
    if not rows:
        raise SystemExit("no shards: run --scan first")
    fm = fraction_map(V)
    # Fraction nodes added under Е2's second record: the same protocol as
    # output 1's source, so they enter the same curve with a pass label.
    for q in sorted(OUT.glob("fraction_extra.*.json")):
        for r in _load(q):
            key = "%g" % r["pn_kc_scale"]
            if key not in fm["by_scale"]:
                continue
            cur = fm["by_scale"][key]["curve"]
            if any(abs(c["g_apl_rel"] - r["g_apl_rel"]) < 1e-12 for c in cur):
                continue
            cur.append({"g_apl_rel": r["g_apl_rel"], "f_mean": r["f_mean"],
                        "f_max": r["f_max"],
                        "in_window": bool(V.passes_fraction(r)),
                        "mbon_computed_by_stage": False,
                        "grid_pass": "refine_fraction"})
            cur.sort(key=lambda c: c["g_apl_rel"])
    out = {"status": "diagnostic outside the stage: criteria are not "
                     "computed as criteria, parameters are not chosen, the "
                     "KC->MBON multiplier is not touched",
           "odors": V.PANEL_CAL, "n_trials_mbon": N_TRIALS_DIAG,
           "timing_mbon": "M", "pulse_ms": V.M_PULSE_MS,
           "grid_main": list(G_VALUES),
           "refine_rule": {"max_nodes_per_scale": REFINE_MAX_NODES,
                           "log_stop_ratio": REFINE_LOG_RATIO,
                           "journal": "results/diag_window/journal_e6_2.md"},
           "n_nodes_main": len([r for r in rows
                                if r.get("grid_pass", "main") == "main"]),
           "floor_hz": V.MBON_FLOOR_HZ, "ceiling_hz": V.MBON_CEIL_HZ,
           "fraction_axis": fm, "by_scale": {}}
    for sc in SCALES:
        key = "%g" % sc
        curve = sorted([r for r in rows if abs(r["pn_kc_scale"] - sc) < 1e-9],
                       key=lambda r: r["g_apl_rel"])
        fw = fm["by_scale"][key]
        d = {"mbon_curve": curve,
             "fraction_window_g": [fw["window_g_lo"], fw["window_g_hi"]],
             "corridor_g": None, "crossing": None}
        cor = [r["g_apl_rel"] for r in curve if r["corridor_ok"]]
        if cor:
            d["corridor_g"] = [min(cor), max(cor)]
        # Derived quantities of rule Е4: computed arithmetically from
        # already-recorded observables, require no new runs, cannot be
        # criteria. The corridor is empty on the grid if and only if g67 >= g2.
        over = [r["g_apl_rel"] for r in curve
                if r["R_t_max"] is not None and r["R_t_max"] > V.MBON_CEIL_HZ]
        under = [r["g_apl_rel"] for r in curve
                 if r["R_t_min"] is not None and r["R_t_min"] < V.MBON_FLOOR_HZ]
        d["derived_E4"] = {
            "g_67": max(over) if over else None,
            "g_2": min(under) if under else None,
            "corridor_empty_on_grid":
                bool(over and under and max(over) >= min(under)),
            "note": "derived quantities (rule Е4), are not criteria"}
        # floor crossing: the largest g at which min_t R_t is still not below the floor
        above = [r for r in curve if r["R_t_min"] is not None
                 and r["R_t_min"] >= V.MBON_FLOOR_HZ]
        below = [r for r in curve if r["R_t_min"] is not None
                 and r["R_t_min"] < V.MBON_FLOOR_HZ]
        if above and below:
            a = max(above, key=lambda r: r["g_apl_rel"])
            nxt = [r for r in below if r["g_apl_rel"] > a["g_apl_rel"]]
            b = min(nxt, key=lambda r: r["g_apl_rel"]) if nxt else None
            cr = {"bracket_g": [a["g_apl_rel"], b["g_apl_rel"] if b else None],
                  "bracket_min_R_t": [a["R_t_min"], b["R_t_min"] if b else None]}
            if b and a["g_apl_rel"] > 0 and a["R_t_min"] > 0 and b["R_t_min"] > 0:
                cr["g_star"] = _interp_log(a["g_apl_rel"], a["R_t_min"],
                                           b["g_apl_rel"], b["R_t_min"],
                                           V.MBON_FLOOR_HZ)
                cr["g_star_kind"] = ("interpolation in log g and log R_t "
                                     "between nodes; an estimate, not a "
                                     "measurement")
                cr["f_at_g_star"] = _f_at(fw["curve"], cr["g_star"])
            else:
                cr["g_star"] = None
                cr["g_star_kind"] = ("interpolation impossible: zero rate "
                                     "or zero on the g axis; bracket only")
            cr["f_at_bracket"] = [
                _f_at(fw["curve"], a["g_apl_rel"])["f_mean"]
                if a["g_apl_rel"] > 0 else None,
                _f_at(fw["curve"], b["g_apl_rel"])["f_mean"] if b else None]
            d["crossing"] = cr
        # gap: by what factor inhibition must be relaxed relative to the fraction window
        if d["crossing"] and d["crossing"].get("g_star") and fw["window_g_lo"]:
            d["gap_factor_g"] = fw["window_g_lo"] / d["crossing"]["g_star"]
        out["by_scale"][key] = d
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "window_gap.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("recorded: %s" % p)
    return out


def print_summary(V, out: dict) -> None:
    for sc in SCALES:
        d = out["by_scale"]["%g" % sc]
        print("\n=== scale PN->KC = %g ===" % sc)
        print("%-10s %10s %10s %10s %9s | %s"
              % ("g/g_ref", "KC spikes", "MBON spike", "active MBON",
                 "min R_t", "corridor [2; 67]"))
        for r in d["mbon_curve"]:
            print("%-10.5g %10d %10d %10d %9.4f | %s"
                  % (r["g_apl_rel"], r["kc_spikes_6_odors"],
                     r["mbon_spikes_6_odors"], r["mbon_active_cells"],
                     r["R_t_min"], "yes" if r["corridor_ok"] else "no"))
        fw = d["fraction_window_g"]
        print("KC fraction window (P14, 6 trials, from stage artifacts): "
              "g in [%s; %s]" % (fw[0], fw[1]))
        print("MBON corridor window (M, 2 trials, this run's nodes): %s"
              % ("g in [%s; %s]" % tuple(d["corridor_g"]) if d["corridor_g"]
                 else "empty at every node"))
        cr = d["crossing"]
        if cr:
            print("the %.1f Hz floor is crossed between g = %s and g = %s "
                  "(min R_t %.4f -> %.4f)"
                  % (out["floor_hz"], cr["bracket_g"][0], cr["bracket_g"][1],
                     cr["bracket_min_R_t"][0], cr["bracket_min_R_t"][1] or 0.0))
            if cr.get("g_star"):
                f = cr["f_at_g_star"]
                print("estimated crossing point g* = %.5g; KC responding "
                      "fraction there f = %s (%s)"
                      % (cr["g_star"],
                         "%.4f" % f["f_mean"] if f["f_mean"] is not None else "none",
                         f["kind"]))
            print("KC responding fraction at the bracket ends: %s"
                  % " -> ".join("%.4f" % v if v is not None else "none"
                                for v in cr["f_at_bracket"]))
        e4 = d.get("derived_E4") or {}
        print("derived Е4: g_67 = %s, g_2 = %s -> corridor on the grid %s"
              % (e4.get("g_67"), e4.get("g_2"),
                 "EMPTY" if e4.get("corridor_empty_on_grid") else "not empty"))
        if d.get("gap_factor_g"):
            print("gap along the inhibition axis: the lower edge of the "
                  "fraction window is %.1f times above the floor crossing point"
                  % d["gap_factor_g"])


def verify(V, neurons, con) -> int:
    """Compare the diagnostic testbed with the stage: the same computation V1b' did.

    The point is a stage candidate, so its MBON values are recorded in the
    calibration artifact. A mismatch means non-determinism and is
    investigated before the diagnostic numbers go into the report.
    """
    rows = [p for p in _load(PRIME / "calibration_stage2.json") if "mbon" in p]
    if not rows:
        raise SystemExit("the artefact has no point with a computed MBON")
    pick = max(rows, key=lambda p: max(v["R_t"] for v in
                                       p["mbon"]["per_type"].values()))
    # The run uses the FULL-PRECISION values FROM THE ARTIFACT, not rounded
    # ones: otherwise different grid points would be compared.
    sc, g = pick["pn_kc_scale"], pick["g_apl_rel"]
    ref = pick["mbon"]["per_type"]
    if max(v["R_t"] for v in ref.values()) == 0.0:
        print("WARNING: all types are silent for the best candidate, "
              "the comparison is degenerate", flush=True)
    m = V.eval_p14m(neurons, con, pn_kc_scale=sc, g_apl_rel=g,
                    seed_base=V.SEED_CAL_M, odors=V.PANEL_CAL)
    got = m["T38"]["per_type"]
    bad = 0
    print("comparing point scale %g, g %g: R_t by type, stage versus diagnostic"
          % (sc, g))
    for t in sorted(set(ref) | set(got)):
        a = ref.get(t, {}).get("R_t")
        b = got.get(t, {}).get("R_t")
        ok = a is not None and b is not None and abs(a - b) < 1e-12
        bad += 0 if ok else 1
        print("  %-8s stage %.6f, diagnostic %.6f  %s"
              % (t, a if a is not None else float("nan"),
                 b if b is not None else float("nan"),
                 "" if ok else "<-- MISMATCH"))
    print("mismatches: %d" % bad)
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", action="store_true", help="KC-fraction window, no run")
    ap.add_argument("--scan", action="store_true", help="MBON axis, run")
    ap.add_argument("--refine", type=float, default=None,
                    help="resolution refinement per rule Е2 for the scale")
    ap.add_argument("--fraction-at", nargs="+", type=float, default=None,
                    metavar="G", help="KC fraction at declared nodes (P14, 6 trials)")
    ap.add_argument("--scale", type=float, default=8.0)
    ap.add_argument("--merge", action="store_true", help="merge shards and compute")
    ap.add_argument("--verify", action="store_true", help="verification against the stage")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--codegen", default="cython", choices=["numpy", "cython"])
    a = ap.parse_args()
    if not any((a.map, a.scan, a.merge, a.verify, a.refine,
                a.fraction_at)):
        ap.error("nothing to do: --map, --scan, --refine, --fraction-at, "
                 "--merge or --verify")
    if a.refine is not None and a.refine not in SCALES:
        ap.error("scale %g is not declared in the measurement" % a.refine)

    if a.scan or a.verify or a.refine is not None or a.fraction_at:
        from brian2 import prefs
        prefs.codegen.target = a.codegen
        if a.codegen == "cython":
            tag = ("r%g" % a.refine) if a.refine is not None else ("w%02d" % a.shard)
            d = HERE / ".cython_cache" / ("wdiag_%s" % tag)
            d.mkdir(parents=True, exist_ok=True)
            prefs.codegen.runtime.cython.cache_dir = str(d)
    import v1b_subcircuit as V

    if a.map:
        fm = fraction_map(V)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "fraction_window.json").write_text(
            json.dumps(fm, ensure_ascii=False, indent=2), encoding="utf-8")
        for sc in SCALES:
            d = fm["by_scale"]["%g" % sc]
            print("scale %g: grid g from %s to %s, %d nodes; fraction "
                  "window f in [%.2f; %.2f] and f_max <= %.2f -> g in [%s; %s], nodes %d"
                  % (sc, d["g_min_grid"], d["g_max_grid"], d["n_grid_points"],
                     V.F_BAND[0], V.F_BAND[1], V.F_MAX,
                     d["window_g_lo"], d["window_g_hi"], d["n_in_window"]))
    if a.scan or a.verify or a.refine is not None or a.fraction_at:
        neurons, con = V.load_substrate()
        if a.verify:
            return verify(V, neurons, con)
        if a.fraction_at:
            return fraction_at(V, neurons, con, a.scale, a.fraction_at)
        if a.refine is not None:
            return refine(V, neurons, con, a.refine)
        return scan(V, neurons, con, a.shard, a.of)
    if a.merge:
        print_summary(V, merge(V))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
