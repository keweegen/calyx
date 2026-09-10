# -*- coding: utf-8 -*-
"""Test of code conformance to the frozen specification of stage V1b.

Why. The V1b calibration was run by code that implemented the V1b-4.5
admissibility constraint incompletely: only the fraction of responding Kenyon
cells was checked, while the requirement "plus V1b-3.3, V1b-3.4 and V1b-3.5 on
the C odors" was not computed at a single grid point. No one caught the
error, because nothing checked the code's conformance to the specification:
the stated methodology (rule as module, fail criterion set before the run,
config hash) silently assumed the code does what is written, and no one
tested that assumption.

This file closes the gap. Every numerical or procedural requirement of
section 3е of the specification gets its own test, whose docstring records
the exact phrase of the specification that the test checks. The test set is
required to catch the V1b-4.5 error: until the code is fixed, the
corresponding check MUST fail. A suite that passes in full on defective code
is useless.

There are no simulator runs here: everything is checked on synthetic data
and on constants, so the suite runs in seconds and can stand before every
run of the stage.

Run:     python test_spec_conformance.py
         python test_spec_conformance.py --verbose
Exit code 0 - all checks passed, 1 - some failed.
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CHECKS = []


def check(code: str, requirement: str):
    """Register a check. code - the specification requirement number."""
    def deco(fn):
        CHECKS.append((code, requirement, fn))
        return fn
    return deco


class Fail(AssertionError):
    pass


# MBON constraint on C, computed and passed: a point is admissible only
# together with it (V1b-4.5)
MBON_OK = {"floor_ok": True, "ceiling_ok": True, "md_ok": True, "pass": True}


def eq(got, want, what: str):
    if isinstance(want, float) or isinstance(got, float):
        ok = abs(float(got) - float(want)) <= 1e-9 * max(1.0, abs(float(want)))
    else:
        ok = got == want
    if not ok:
        raise Fail("%s: got %r, specification requires %r" % (what, got, want))


def true(cond, what: str):
    if not cond:
        raise Fail(what)


# --- synthetic data in place of runs -----------------------------------------
def fake_result(V, neurons, counts_by_role: dict, n_trials: int = 6) -> dict:
    """Run result with a given spike count per role."""
    apl = set(neurons.loc[neurons.mb_role == "APL", "root_id"])
    core = sorted(set(neurons.root_id) - apl)
    role = dict(zip(neurons.root_id, neurons.mb_role))
    c = np.zeros((len(core), n_trials), dtype=np.int32)
    for k, i in enumerate(core):
        c[k, :] = counts_by_role.get(role[i], 0)
    return {"core_ids": core, "counts": c, "counts_by_window": [c, c],
            "windows": [(0.0, 4.0), (0.0, 2.0)],
            "seeds": list(range(n_trials))}


# --- Kenyon cell response ------------------------------------------------------
@check("V1b-2.1", "KC responds to a trial if its spike count in the window "
                  ">= 1; the unit is fixed and not subject to tuning")
def t_response_threshold(V, neurons, con):
    eq(V.KC_SPIKE_THRESHOLD, 1, "spike threshold per trial")


@check("V1b-2.2", "An odor is presented 6 times; KC responds to the odor if "
                  "it responded on at least 3 out of 6 trials")
def t_trials(V, neurons, con):
    eq(V.N_TRIALS, 6, "number of presentations")
    eq(V.MIN_TRIALS, 3, "minimum trials to respond to an odor")
    # exactly 3 out of 6 - a response; exactly 2 out of 6 - not
    apl = set(neurons.loc[neurons.mb_role == "APL", "root_id"])
    core = sorted(set(neurons.root_id) - apl)
    role = dict(zip(neurons.root_id, neurons.mb_role))
    kc_idx = [k for k, i in enumerate(core) if role[i] == "Kenyon_Cell"]
    for n_hit, want_resp in ((3, True), (2, False)):
        c = np.zeros((len(core), 6), dtype=np.int32)
        c[kc_idx[0], :n_hit] = 1
        res = {"core_ids": core, "counts": c, "counts_by_window": [c],
               "windows": [(0.0, 4.0)], "seeds": list(range(6))}
        f = V.kc_fraction(res, neurons)["f"]
        eq(f > 0, want_resp, "response at %d trials out of 6" % n_hit)


@check("V1b-2.3", "f(o) = N_resp(o)/N_KC, where N_KC = 5,177 - all neurons "
                  "of type Kenyon_Cell of the subcircuit, both hemispheres, "
                  "no selection")
def t_denominator(V, neurons, con):
    n_kc = int((neurons.mb_role == "Kenyon_Cell").sum())
    eq(n_kc, 5177, "number of KC in the subcircuit")
    res = fake_result(V, neurons, {"Kenyon_Cell": 1})
    kf = V.kc_fraction(res, neurons)
    eq(kf["n_kc_denominator"], 5177, "denominator of the responding fraction")
    eq(kf["f"], 1.0, "fraction when all KC respond")
    # reporting denominators
    cov = V.kc_coverage(neurons, con)
    rep = V.kc_fraction_report(res, neurons, cov)
    eq(rep["n_denominator_all"], 5177, "denominator, all KC")
    eq(rep["n_denominator_with_upn"], 4820, "denominator, KC with input from uPN")
    eq(rep["n_right"], 2597, "KC of the right hemisphere")
    eq(rep["n_left"], 2580, "KC of the left hemisphere")


@check("V1b-2.4", "For threshold k from {1, 2, 3, 5} spikes per trial and "
                  "aggregation rules {>=1 out of 6; >=3 out of 6; 6 out of 6}")
def t_sensitivity(V, neurons, con):
    res = fake_result(V, neurons, {"Kenyon_Cell": 2})
    keys = set(V.kc_fraction(res, neurons)["sensitivity"])
    want = {"k=%d,%s" % (k, lab) for k in (1, 2, 3, 5)
            for lab in (">=1/6", ">=3/6", "6/6")}
    eq(keys, want, "set of sensitivity table cells")
    tab = V.sensitivity_table({"o1": res, "o2": res}, neurons)
    eq(set(tab), want, "set of table cells across the panel")
    true(all({"mean", "max"} == set(v) for v in tab.values()),
         "each cell must print the mean and max across the panel")


@check("V1b-2.5", "S_P(o) = (1 - (sum r_i/N)^2 / (sum r_i^2/N)) / (1 - 1/N), "
                  "r_i - the mean over 6 trials of the spike count of KC i "
                  "in the window")
def t_sparseness(V, neurons, con):
    n_kc = int((neurons.mb_role == "Kenyon_Cell").sum())
    # with equal r_i the ratio (sum r/N)^2 / (sum r^2/N) equals 1, so S_P = 0:
    # zero - dense code, one - maximally sparse
    res = fake_result(V, neurons, {"Kenyon_Cell": 3})
    eq(V.sparseness_treves_rolls(res, neurons), 0.0, "S_P with equal r_i")
    # one cell carries everything: maximally sparse code, S_P = 1
    apl = set(neurons.loc[neurons.mb_role == "APL", "root_id"])
    core = sorted(set(neurons.root_id) - apl)
    role = dict(zip(neurons.root_id, neurons.mb_role))
    kc_idx = [k for k, i in enumerate(core) if role[i] == "Kenyon_Cell"]
    c = np.zeros((len(core), 6), dtype=np.int32)
    c[kc_idx[0], :] = 6
    res1 = {"core_ids": core, "counts": c, "counts_by_window": [c],
            "windows": [(0.0, 4.0)], "seeds": list(range(6))}
    r = np.zeros(n_kc); r[0] = 6.0
    want = (1.0 - (r.sum() / n_kc) ** 2 / ((r ** 2).sum() / n_kc)) / (1.0 - 1.0 / n_kc)
    eq(V.sparseness_treves_rolls(res1, neurons), want, "S_P with one active KC")


# --- MBON response --------------------------------------------------------------
@check("V1b-3.1", "T38 - six MBON types recorded in [38]: MBON11, MBON12, "
                  "MBON13, MBON14, MBON17, MBON18; timing M - a 5 s "
                  "presentation, mean rate in the presentation window")
def t_t38(V, neurons, con):
    eq(list(V.T38), ["MBON11", "MBON12", "MBON13", "MBON14", "MBON17", "MBON18"],
       "the T38 set")
    eq(V.M_PULSE_MS, 5000, "presentation duration under timing M")
    eq(V.M_WINDOW_MS, 5000, "measurement window under timing M equals the "
       "presentation window")
    typ = set(neurons.loc[neurons.mb_role == "MBON", "hemibrain_type"]
              .astype("string").dropna())
    missing = [t for t in V.T38 if t not in typ]
    true(not missing, "T38 types missing from the subcircuit: %s" % missing)
    # type rate - mean across cells of the type and across trials, divided by the window
    res = fake_result(V, neurons, {"MBON": 10})
    rates = V.mbon_type_rates({"o": res}, neurons, V.M_WINDOW_MS)
    eq(float(rates.loc["MBON11", "o"]), 10.0 / (V.M_WINDOW_MS / 1000.0),
       "MBON11 type rate at 10 spikes in a 5 s window")


@check("V1b-3.1", "For the avoidance group MBON01-MBON06 there is no "
                  "reference, thresholds are not applied to its types; for "
                  "types outside T38 there are no thresholds")
def t_avoid_no_thresholds(V, neurons, con):
    eq(list(V.AVOID), ["MBON0%d" % k for k in range(1, 7)], "avoidance group")
    res = fake_result(V, neurons, {"MBON": 0})
    rep = V.mbon_report({"o": res}, neurons, V.M_WINDOW_MS)
    true(set(V.AVOID) & set(rep["avoid_group_report_only"]),
         "avoidance group must be printed for reporting")
    true(not (set(V.AVOID) & set(rep["T38"]["per_type"])),
         "avoidance group types must not fall into the threshold part")
    true(rep["other_types_report_only"],
         "types outside T38 and outside the avoidance group are printed for reporting")


@check("V1b-3.3", "R_t = mean_o r_t(o); R_t >= 2 Hz is required for all six types")
def t_floor(V, neurons, con):
    eq(V.MBON_FLOOR_HZ, 2.0, "floor threshold")
    lo = pd.DataFrame(np.full((6, 14), 1.99), index=V.T38,
                      columns=V.PANEL_ALL)
    eq(V.check_mbon(lo)["floor_ok"], False, "floor at 1.99 Hz for all types")
    hi = lo + 0.02
    eq(V.check_mbon(hi)["floor_ok"], True, "floor at 2.01 Hz for all types")
    one_low = hi.copy(); one_low.iloc[0, :] = 1.0
    eq(V.check_mbon(one_low)["floor_ok"], False,
       "floor is required for ALL six types, not for a majority")


@check("V1b-3.4", "For each type t in T38: R_t <= 67 Hz")
def t_ceiling(V, neurons, con):
    eq(V.MBON_CEIL_HZ, 67.0, "ceiling threshold")
    r = pd.DataFrame(np.full((6, 14), 66.9), index=V.T38, columns=V.PANEL_ALL)
    eq(V.check_mbon(r)["ceiling_ok"], True, "ceiling at 66.9 Hz")
    r2 = r + 0.2
    eq(V.check_mbon(r2)["ceiling_ok"], False, "ceiling at 67.1 Hz")


@check("V1b-3.5", "MD_t = (max-min)/(max+min) over 14 odors; MD_t >= 0.19 for "
                  "at least 5 out of 6 types; a type with R_t < 2 Hz does "
                  "not participate")
def t_md(V, neurons, con):
    eq(V.MBON_MD_MIN, 0.19, "modulation depth threshold")
    eq(V.MBON_MD_TYPES, 5, "how many types must pass it")
    base = np.full((6, 14), 10.0)
    base[:, 0] = 10.0 * (1 + 0.25) / (1 - 0.25)   # MD = 0.25, above threshold
    r = pd.DataFrame(base, index=V.T38, columns=V.PANEL_ALL)
    res = V.check_mbon(r)
    eq(res["n_md_ok"], 6, "MD above threshold counted for all six types")
    # the comparison rule is non-strict: types with MD >= 0.19 count, not > 0.19.
    # Comparing against a constructed 0.19 is impossible - the value is not
    # representable in binary floating point; we check the aggregate against
    # the MD values the code itself returned.
    near = np.full((6, 14), 10.0)
    for k in range(6):
        near[k, 0] = 10.0 * (1 + 0.185 + 0.002 * k) / (1 - 0.185 - 0.002 * k)
    rn = pd.DataFrame(near, index=V.T38, columns=V.PANEL_ALL)
    got = V.check_mbon(rn)
    want = sum(1 for d in got["per_type"].values() if d["MD"] >= V.MBON_MD_MIN)
    eq(got["n_md_ok"], want, "aggregate counts types with MD >= threshold (non-strict)")
    # two types without modulation: 4 out of 6 - criterion not met
    flat = r.copy(); flat.iloc[:2, 0] = 10.0
    eq(V.check_mbon(flat)["md_ok"], False, "MD for 4 out of 6 types")
    # one type without modulation: 5 out of 6 - met
    flat1 = r.copy(); flat1.iloc[:1, 0] = 10.0
    eq(V.check_mbon(flat1)["md_ok"], True, "MD for 5 out of 6 types")
    # a type that failed the floor is excluded from the MD count
    low = r.copy(); low.iloc[0, :] = 1.0
    eq(V.check_mbon(low)["n_md_ok"], 5, "type with R_t < 2 Hz excluded from the MD count")


@check("V1b-3.5", "Median of MD_t over all 2,002 five-odor subsets of the panel")
def t_md_subsets(V, neurons, con):
    eq(len(V.PANEL_ALL), 14, "panel size of the stage")
    eq(len(list(combinations(range(14), 5))), 2002, "number of five-odor subsets")
    r = pd.DataFrame(np.random.default_rng(0).uniform(1, 20, (6, 14)),
                     index=V.T38, columns=V.PANEL_ALL)
    md = V.md_subset_medians(r, V.T38)
    eq(md["n_subsets"], 2002, "number of subsets in the reported quantity")
    eq(md["k"], 5, "subset size")
    eq(set(md["per_type"]), set(V.T38), "median is computed for each T38 type")


# --- calibration ----------------------------------------------------------------
@check("V1b-4.1", "C - 2-heptanone, isopentyl acetate, hexanal, "
                  "6-methyl-5-hepten-2-one, diethyl succinate, methyl octanoate; "
                  "E - pentyl acetate, butyl acetate, ethyl lactate, 1-octen-3-ol, "
                  "pentanal, benzaldehyde, alpha-humulene, ethyl octanoate")
def t_panel_split(V, neurons, con):
    eq(list(V.PANEL_CAL), ["2-heptanone", "isopentyl acetate", "hexanal",
                           "6-methyl-5-hepten-2-one", "diethyl succinate",
                           "methyl octanoate"], "calibration set C")
    eq(list(V.PANEL_EVAL), ["pentyl acetate", "butyl acetate", "ethyl lactate",
                            "1-octen-3-ol", "pentanal", "benzaldehyde",
                            "alpha-humulene", "ethyl octanoate"],
       "evaluation set E")
    eq(len(V.PANEL_ALL), 14, "panel of the stage")
    true(not (set(V.PANEL_CAL) & set(V.PANEL_EVAL)), "C and E must not overlap")
    panel = pd.read_csv(V.PANEL, sep="\t", index_col=0)
    missing = [o for o in V.PANEL_ALL if o not in panel.index]
    true(not missing, "panel odors missing from the input source: %s" % missing)
    # three reference substances of the overlap criterion belong to E
    for a, b in V.REF_PAIRS:
        true(a in V.PANEL_EVAL and b in V.PANEL_EVAL,
             "reference pair (%s, %s) must lie within E" % (a, b))


@check("V1b-4.1", "Presentation seeds for E do not overlap with calibration seeds")
def t_seeds_disjoint(V, neurons, con):
    span = lambda b: {b + i for i in range(1, V.N_TRIALS + 1)}
    bases = {"SEED_CAL": V.SEED_CAL, "SEED_CAL_M": V.SEED_CAL_M,
             "SEED_EVAL": V.SEED_EVAL, "SEED_EVAL_M": V.SEED_EVAL_M,
             "SEED_EMPTY": V.SEED_EMPTY, "SEED_PERM": V.SEED_PERM}
    names = list(bases)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            true(not (span(bases[a]) & span(bases[b])),
                 "seed spans %s and %s overlap" % (a, b))


@check("V1b-4.3", "g_ref = 1/(v_th - v_0) = 1/7 mV^-1")
def t_gref(V, neurons, con):
    eq(V.g_ref_value(), 1.0 / 7.0, "g_ref")


@check("V1b-4.4", "Stage 1: pn_kc_scale from {2^k, k = -6..2} (9 values), "
                  "g_apl from {0} + {10^(k/2), k = -4..4} (10 values) - 90 points")
def t_grid1(V, neurons, con):
    g = V.grid_stage1()
    eq(len(g), 90, "number of stage 1 points")
    scales = sorted({s for s, _ in g})
    gains = sorted({x for _, x in g})
    eq(scales, [2.0 ** k for k in range(-6, 3)], "stage 1 scales")
    eq(len(gains), 10, "number of gain values")
    eq(gains[0], 0.0, "zero inhibition is included in the stage 1 grid")
    eq(gains[1:], [10.0 ** (k / 2.0) for k in range(-4, 5)], "stage 1 gains")


@check("V1b-4.4", "Stage 2: the bounding box of stage 1's admissible points, "
                  "expanded by one stage 1 step on each side; steps of "
                  "2^(1/4) in scale and 10^(1/8) in g_apl")
def t_grid2(V, neurons, con):
    s1 = [{"pn_kc_scale": 2.0, "g_apl_rel": 10.0 ** -0.5,
           "f_mean": 0.05, "f_max": 0.06},
          {"pn_kc_scale": 4.0, "g_apl_rel": 1.0, "f_mean": 0.05, "f_max": 0.06}]
    g = V.grid_stage2(s1)
    scales = sorted({s for s, _ in g})
    gains = sorted({x for _, x in g})
    eq(scales[0], 1.0, "lower scale bound: min/2 by the stage 1 step")
    eq(scales[-1], 8.0, "upper scale bound: max*2 by the stage 1 step")
    for a, b in zip(scales, scales[1:]):
        eq(b / a, 2.0 ** 0.25, "grid step in scale")
    eq(gains[0], 10.0 ** -1.0, "lower gain bound: min/10^(1/2)")
    eq(gains[-1], 10.0 ** 0.5, "upper gain bound: max*10^(1/2)")
    for a, b in zip(gains, gains[1:]):
        eq(b / a, 10.0 ** 0.125, "grid step in gain")


@check("V1b-4.5", "A point is admissible if on C the following hold: mean "
                  "f(o) in [0.03; 0.10] and max f(o) <= 0.10, PLUS V1b-3.3, "
                  "V1b-3.4 and V1b-3.5 on the C odors")
def t_admissible(V, neurons, con):
    eq(tuple(V.F_BAND), (0.03, 0.10), "band of the mean responding fraction")
    eq(V.F_MAX, 0.10, "ceiling of the max fraction")
    ok = {"f_mean": 0.05, "f_max": 0.06, "mbon": MBON_OK}
    eq(V.admissible(ok), True, "point inside the band and with MBON passed")
    eq(V.admissible({"f_mean": 0.02, "f_max": 0.03, "mbon": MBON_OK}), False,
       "mean below the band")
    eq(V.admissible({"f_mean": 0.05, "f_max": 0.11, "mbon": MBON_OK}), False,
       "max above the ceiling")
    # MBON constraint: a point that passes on the fraction but fails any of
    # the three sub-constraints must NOT be admissible
    for key, label in (("floor_ok", "V1b-3.3"), ("ceiling_ok", "V1b-3.4"),
                       ("md_ok", "V1b-3.5")):
        bad = dict(ok, mbon=dict(MBON_OK, **{key: False}))
        true(V.admissible(bad) is False,
             "a point that passed on the fraction but failed %s on C was "
             "found admissible" % label)
    # and the main point: "not computed" is not equal to "passed". It was
    # exactly the conflation of these two states that made the V1b
    # calibration not conform to the specification.
    true(V.admissible({"f_mean": 0.05, "f_max": 0.06}) is False,
         "a point whose MBON constraint was not computed was found admissible")
    true(V.passes_mbon({"f_mean": 0.05, "f_max": 0.06}) is None,
         "an uncomputed MBON constraint must differ from a computed one")
    # the fraction constraint remains available separately: it is the filter
    # before the expensive MBON run
    eq(V.passes_fraction({"f_mean": 0.05, "f_max": 0.06}), True,
       "fraction constraint as a separate function")


@check("V1b-4.6", "Among the admissible points of STAGE 2, the minimum of "
                  "|f_C - 0.05| is taken")
def t_choose_stage2(V, neurons, con):
    """A stage 1 point must not win the selection over a stage 2 point."""
    pts = [{"pn_kc_scale": 1.0, "g_apl_rel": 0.1, "f_mean": 0.050, "f_max": 0.06,
            "mbon": MBON_OK, "s_ab": 2.2, "stage": 1},
           {"pn_kc_scale": 2.0, "g_apl_rel": 0.2, "f_mean": 0.052, "f_max": 0.06,
            "mbon": MBON_OK, "s_ab": 2.6, "stage": 2}]
    best = V.choose_point(pts)
    eq(best["pn_kc_scale"], 2.0,
       "the selection is made among stage 2 admissible points; a stage 1 point won")


@check("V1b-4.6", "Lexicographically: |s - 2.2|, then the Chebyshev distance "
                  "to an inadmissible point or the edge, then min g_apl, "
                  "then min pn_kc_scale")
def t_choose(V, neurons, con):
    eq(V.SPIKES_AB_TARGET, 2.2, "reference spikes per response")
    pts = [{"pn_kc_scale": 1.0, "g_apl_rel": 0.1, "f_mean": 0.049, "f_max": 0.06,
            "mbon": MBON_OK, "s_ab": 3.0, "stage": 2},
           {"pn_kc_scale": 2.0, "g_apl_rel": 0.2, "f_mean": 0.051, "f_max": 0.06,
            "mbon": MBON_OK, "s_ab": 2.3, "stage": 2}]
    best = V.choose_point(pts)
    eq(best["pn_kc_scale"], 2.0, "inside the band the selection goes by |s - 2.2|")
    # outside the band - by |f - 0.05|
    out = [{"pn_kc_scale": 1.0, "g_apl_rel": 0.1, "f_mean": 0.035, "f_max": 0.06,
            "mbon": MBON_OK, "s_ab": 2.2, "stage": 2},
           {"pn_kc_scale": 2.0, "g_apl_rel": 0.2, "f_mean": 0.042, "f_max": 0.06,
            "mbon": MBON_OK, "s_ab": 9.9, "stage": 2}]
    eq(V.choose_point(out)["pn_kc_scale"], 2.0,
       "outside the band the selection goes by |f - 0.05|")
    # a point with undefined s ranks below all defined ones
    nan = [{"pn_kc_scale": 1.0, "g_apl_rel": 0.1, "f_mean": 0.05, "f_max": 0.06,
            "mbon": MBON_OK, "s_ab": float("nan"), "stage": 2},
           {"pn_kc_scale": 2.0, "g_apl_rel": 0.2, "f_mean": 0.05, "f_max": 0.06,
            "mbon": MBON_OK, "s_ab": 9.9, "stage": 2}]
    eq(V.choose_point(nan)["pn_kc_scale"], 2.0,
       "a point with undefined s ranks below a point with a defined one")
    # tie-break (3) and (4): at equal |s - 2.2| the smaller g_apl wins,
    # at equal g_apl - the smaller pn_kc_scale
    tie34 = [{"pn_kc_scale": 3.0, "g_apl_rel": 0.2, "f_mean": 0.05, "f_max": 0.06,
              "mbon": MBON_OK, "s_ab": 2.2, "stage": 2},
             {"pn_kc_scale": 1.0, "g_apl_rel": 0.1, "f_mean": 0.05, "f_max": 0.06,
              "mbon": MBON_OK, "s_ab": 2.2, "stage": 2}]
    eq(V.choose_point(tie34)["pn_kc_scale"], 1.0, "tie-breaks by g_apl and scale")
    # tie-break (2), which ranks ABOVE them: at equal |s - 2.2| the point
    # farther from the inadmissible region or the grid edge wins
    tie2 = [{"pn_kc_scale": 1.0, "g_apl_rel": 0.1, "f_mean": 0.05, "f_max": 0.06,
             "mbon": MBON_OK, "s_ab": 2.2, "stage": 2, "chebyshev_to_edge": 1},
            {"pn_kc_scale": 2.0, "g_apl_rel": 0.2, "f_mean": 0.05, "f_max": 0.06,
             "mbon": MBON_OK, "s_ab": 2.2, "stage": 2, "chebyshev_to_edge": 4}]
    eq(V.choose_point(tie2)["pn_kc_scale"], 2.0,
       "tie-break by Chebyshev distance to the edge not implemented")


@check("V1b′-3.5а", "If max_o r_t(o) + min_o r_t(o) = 0, the value MD_t is "
                     "undefined and is counted as NOT meeting V1b-3.5")
def t_md_undefined(V, neurons, con):
    r = pd.DataFrame(np.full((6, 14), 10.0), index=V.T38, columns=V.PANEL_ALL)
    r.iloc[:, 0] = 20.0                       # for all types MD is defined and large
    r.iloc[0, :] = 0.0                        # silent type: max + min = 0
    res = V.check_mbon(r)
    md = res["per_type"][V.T38[0]]["MD"]
    true(md != md, "MD of the silent type must be undefined, got %r" % md)
    eq(res["n_md_ok"], 5, "silent type not counted in the MD tally")
    eq(res["floor_ok"], False, "silent type also fails the floor")


@check("V1b′-3.1а", "Number of presentations of the P14-M set - 6; the "
                     "measurement window equals the presentation window and "
                     "starts at t_on")
def t_p14m_protocol(V, neurons, con):
    eq(V.M_PULSE_MS, V.M_WINDOW_MS, "measurement window equals the presentation window")
    eq(V.M_PULSE_MS, 5000, "presentation duration")
    import inspect
    src = inspect.getsource(V.eval_p14m)
    true("M_PULSE_MS" in src and "M_WINDOW_MS" in src,
         "the P14-M run must go on timing M")
    true("SEED_EVAL_M" in inspect.signature(V.eval_p14m).parameters
         or "SEED_EVAL_M" in src, "the verdict run goes on its own seeds")


@check("V1b′-4.4", "The row g_apl = 0 is included in the stage 2 grid if "
                    "and only if there is a point with zero gain among the "
                    "stage 1 candidates")
def t_zero_gain_row(V, neurons, con):
    ok = {"f_mean": 0.05, "f_max": 0.06}
    without = V.grid_stage2([dict(ok, pn_kc_scale=2.0, g_apl_rel=10.0 ** -0.5)])
    true(0.0 not in {g for _, g in without},
         "zero gain got into the grid without a zero-gain candidate")
    with_zero = V.grid_stage2([dict(ok, pn_kc_scale=2.0, g_apl_rel=0.0),
                               dict(ok, pn_kc_scale=2.0, g_apl_rel=10.0 ** -0.5)])
    true(0.0 in {g for _, g in with_zero},
         "zero gain did not get into the grid with a zero-gain candidate")


@check("V1b′-4.4", "The expansion by one stage 1 step is not clipped by the "
                    "stage 1 range: the fine grid legitimately extends past "
                    "the coarse one")
def t_expansion_not_clipped(V, neurons, con):
    s1_max = max(s for s, _ in V.grid_stage1())
    g = V.grid_stage2([{"pn_kc_scale": s1_max, "g_apl_rel": 1.0,
                        "f_mean": 0.05, "f_max": 0.06}])
    eq(max(s for s, _ in g), s1_max * 2.0,
       "upper scale bound of stage 2 with a candidate on the edge of the coarse grid")


@check("V1b′-4.4", "The stage 2 bounding box is built from stage 1 "
                    "CANDIDATES, not from admissible points")
def t_bbox_over_candidates(V, neurons, con):
    # a candidate without a computed MBON constraint: it is not admissible,
    # but the bounding box must still be built from it
    cand = [{"pn_kc_scale": 2.0, "g_apl_rel": 1.0, "f_mean": 0.05, "f_max": 0.06}]
    eq(V.admissible(cand[0]), False, "a candidate without MBON is not admissible")
    true(len(V.grid_stage2(cand)) > 0,
         "stage 2 grid is empty: the bounding box was built from admissible "
         "points, not from candidates")


@check("V1b′-4.6", "d(p) = min over q in D of the Chebyshev distance, D - "
                    "inadmissible grid points AND all nodes outside the "
                    "grid; d >= 1 for any point")
def t_chebyshev(V, neurons, con):
    # a 5x5 grid, all admissible. The nearest nodes outside the grid lie at
    # indices -1 and 5, so the central point (index 2) is at distance 3, and
    # the corner point - at 1.
    pts = [{"pn_kc_scale": float(i), "g_apl_rel": float(j),
            "f_mean": 0.05, "f_max": 0.06, "mbon": MBON_OK}
           for i in range(5) for j in range(5)]
    m = V.chebyshev_margins(pts)
    eq(m[(2.0, 2.0)], 3, "distance of the central point of a 5x5 grid")
    eq(m[(0.0, 0.0)], 1, "distance of a corner point: nodes outside the grid are inadmissible")
    true(all(v >= 1 for v in m.values()), "d >= 1 for any point")
    # an inadmissible neighbor shortens the distance
    pts2 = [dict(p) for p in pts]
    for p in pts2:
        if (p["pn_kc_scale"], p["g_apl_rel"]) == (3.0, 2.0):
            p["mbon"] = dict(MBON_OK, floor_ok=False, pass_=False)
    m2 = V.chebyshev_margins(pts2)
    eq(m2[(2.0, 2.0)], 1, "an inadmissible neighbor shortens the distance to 1")


@check("V1b-4.9", "The cell set is KC labeled hemibrain_type KCab (1,771 "
                  "cells); the reference window [t_on; t_on + 2 s]; "
                  "cell-odor pairs responding under V1b-2.1-2.2 are counted")
def t_spikes_per_response(V, neurons, con):
    kc = neurons[neurons.mb_role == "Kenyon_Cell"].hemibrain_type.astype(str)
    eq(int(kc.str.startswith("KCab").sum()), 1771, "number of KCab in the subcircuit")
    apl = set(neurons.loc[neurons.mb_role == "APL", "root_id"])
    core = sorted(set(neurons.root_id) - apl)
    m = V._kc_mask(core, neurons, "KCab")
    eq(int(m.sum()), 1771, "KCab subtype mask")
    # one responding cell: 4 trials with a spike in the reference window of 3 spikes each
    idx = int(np.nonzero(m)[0][0])
    full = np.zeros((len(core), 6), dtype=np.int32); full[idx, :] = 1
    ref = np.zeros((len(core), 6), dtype=np.int32); ref[idx, :4] = 3
    res = {"core_ids": core, "counts": full, "counts_by_window": [full, ref],
           "windows": [(0.0, 4.0), (0.0, 2.0)], "seeds": list(range(6))}
    eq(V.spikes_per_response({"o": res}, neurons, "KCab", 1), 3.0,
       "pair count - the mean over presentations where there is a spike in "
       "the reference window")
    # a cell not responding under V1b-2.2 (2 out of 6 trials) is not counted
    full2 = np.zeros((len(core), 6), dtype=np.int32); full2[idx, :2] = 1
    res2 = dict(res, counts=full2, counts_by_window=[full2, ref])
    true(np.isnan(V.spikes_per_response({"o": res2}, neurons, "KCab", 1)),
         "a non-responding cell must not be counted; the value is undefined")


@check("V1b-3д", "\"Recording\" - a random uniform subsample of 120 KC, 24 "
                 "subsamples; separation >= 0.25, ceiling 0.40")
def t_overlap_params(V, neurons, con):
    eq(V.N_SUBSAMPLE, 120, "\"recording\" subsample size")
    eq(V.N_DRAWS, 24, "number of subsamples")
    eq(V.OVERLAP_SEP_MIN, 0.25, "separation threshold")
    eq(V.OVERLAP_CEIL, 0.40, "ceiling for chemically distinct pairs")
    r = {V.REF_PAIRS[0]: 0.70, V.REF_PAIRS[1]: 0.15, V.REF_PAIRS[2]: 0.11}
    ov = V.check_overlap(r)
    eq(ov["pass"], True, "reference values [40] must pass the criterion")
    eq(round(ov["separation"], 4), 0.55, "separation at the reference values")
    bad = {V.REF_PAIRS[0]: 0.70, V.REF_PAIRS[1]: 0.50, V.REF_PAIRS[2]: 0.11}
    eq(V.check_overlap(bad)["ceiling_ok"], False, "ceiling 0.40 violated")


@check("V1b-4.7", "The overlap criterion on E checks an independent "
                  "quantity - it is the only one of the four criteria that "
                  "the stage checks blind")
def t_blindness(V, neurons, con):
    """The blindness of E is enforceable: with an incomplete E set the criteria are not computed."""
    import inspect
    src = inspect.getsource(V.eval_p14)
    true("not_computed" in src,
         "eval_p14 must refuse to compute the criteria when the E set is "
         "not entirely part of the run")
    true("PANEL_EVAL" in src, "membership in E is checked explicitly")


@check("v0.11-а", "With substitutions off, the substrate matches V1a; the "
                  "mask removes 49,316 edges")
def t_regression(V, neurons, con):
    r = V.regression_substrate(neurons, con)
    eq(r["n_masked"], 49316, "number of edges removed by the mask")
    eq(r["mask_count_ok"], True, "count of removed edges")
    eq(r["substitutions_off_identical"], True, "substrate with substitutions off")


@check("v0.11-б", "The difference between mode configs equals exactly the "
                  "declared key set; hashes with the declared keys excluded "
                  "are equal")
def t_config_hash(V, neurons, con):
    off = {"dan_mask": False, "graded_apl": False, "odor_input": False,
           "pn_kc_scale": 1.0, "g_apl_rel": 0.0, "substrate": "flywire_630"}
    on = dict(off, dan_mask=True, graded_apl=True, odor_input=True,
              pn_kc_scale=8.0, g_apl_rel=3.16228)
    eq(V.config_hash(off, V.DECLARED_KEYS), V.config_hash(on, V.DECLARED_KEYS),
       "config hashes with the declared keys excluded")
    # an undeclared key must break the equality
    sneaky = dict(on, w_syn_mV=0.3)
    true(V.config_hash(off, V.DECLARED_KEYS) != V.config_hash(sneaky, V.DECLARED_KEYS),
         "changing an UNDECLARED key must break hash equality")


# --- pre-registration of stage V1c (sections 3з and 3и) -----------------------
#
# The V1c specification requires a "test of code conformance to the
# specification before the run" and assigns to it the detector table
# consistency check and the structural scaling test (V1c-E7.1). The checks
# below are the part of it that is executable without running the simulator.

V1C_NODES = ["1", "1.6836", "1.7783", "3.1623", "5.3875"]


def near(got, want, tol, what):
    """Equality with an explicit tolerance: eq() holds a strict 1e-9."""
    if not abs(float(got) - float(want)) <= tol:
        raise Fail("%s: got %.12g, specification requires %.12g "
                   "(tolerance %.3g)" % (what, got, want, tol))


def _v1c_artifacts():
    import json
    d = HERE / "results" / "v1c"
    w = json.loads((d / "weights_kc_mbon.json").read_text(encoding="utf-8"))
    t = json.loads((d / "detector_table.json").read_text(encoding="utf-8"))
    return w, t


@check("V1c-E3.2", "The final version's grid nodes are {1; 1.6836; 1.7783; "
                   "3.1623; 5.3875}, five nodes; the list enters the hash "
                   "and is not extended")
def t_v1c_nodes(V, neurons, con):
    """V1c-E3.2  The five canonical grid nodes by s, as specification strings."""
    import v1c_detectors as D
    eq(list(D.NODES), V1C_NODES, "canonical grid nodes")
    _, t = _v1c_artifacts()
    eq(list(t["nodes"]), V1C_NODES, "detector table nodes")


@check("V1c-E3.1", "s_max = (v_th - v_0) / (A * max_m w_max(m)) over cells "
                   "of the six T38 types; A - a dimensionless number from "
                   "t_mbr and tau")
def t_v1c_smax(V, neurons, con):
    """V1c-E3.1  s_max and s_pop are recomputed from the constants and the weights."""
    import math
    w, _ = _v1c_artifacts()
    c = w["model_constants"]
    rho = c["t_mbr_ms"] / c["tau_ms"]
    a = (1.0 / (rho - 1.0)) * (rho ** (-1.0 / (rho - 1.0))
                               - rho ** (-rho / (rho - 1.0)))
    near(a, w["epsp"]["A"], 1e-12, "multiplier A")
    near(a, 0.157490, 1e-6, "A to the precision the specification records it")
    t_star = (c["tau_ms"] * c["t_mbr_ms"] * math.log(rho)
              / (c["t_mbr_ms"] - c["tau_ms"]))
    near(t_star, 9.242, 5e-4, "peak time of a single EPSP")
    theta = c["v_th_mV"] - c["v_0_mV"]
    near(theta, 7.0, 1e-9, "threshold θ")
    near(theta / (a * w["s_max"]["w_max_mV_T38"]), 5.387542964, 1e-6, "s_max")
    near(theta / (a * w["s_max"]["w_max_mV_all_mbon"]), 1.683607176, 1e-6,
         "s_pop")


@check("V1c-E3.2", "Boundary nodes are rounded down, so that at them the "
                   "inequality of condition (5) remains strict")
def t_v1c_rounding(V, neurons, con):
    """V1c-E3.2  Rounding down keeps the boundary inequalities strict."""
    w, _ = _v1c_artifacts()
    a = w["epsp"]["A"]
    theta = w["model_constants"]["theta_mV"]
    for node, w_max, who in ((5.3875, w["s_max"]["w_max_mV_T38"], "s_max/T38"),
                             (1.6836, w["s_max"]["w_max_mV_all_mbon"],
                              "s_pop/all MBON")):
        if not node * a * w_max < theta:
            raise Fail("at node %s the inequality is not strict: %.10f >= %.1f"
                       % (who, node * a * w_max, theta))


@check("V1c-E2.2", "Condition (5): a single spike of one Kenyon cell does "
                   "not bring any MBON of the T38 set to threshold")
def t_v1c_condition5(V, neurons, con):
    """V1c-E2.2  No grid node has detectors among the T38 types."""
    _, t = _v1c_artifacts()
    for node in V1C_NODES:
        eq(t["by_node"][node]["n_detectors_T38"], 0,
           "detectors in T38 at node %s" % node)
    if not t["condition5_holds_on_all_nodes"]:
        raise Fail("artifact does not confirm condition (5)")


@check("V1c-E7.1", "Cell m is a detector at node s_k if and only if "
                   "s_k · A · w_max(m) > θ; the inequality is strict")
def t_v1c_detectors(V, neurons, con):
    """V1c-E7.1  The detector table is recomputed from the weights and matches."""
    import v1c_detectors as D
    w, t = _v1c_artifacts()
    a, theta = w["epsp"]["A"], w["model_constants"]["theta_mV"]
    for node in V1C_NODES:
        want = sorted(int(k) for k, d in w["per_mbon"].items()
                      if float(node) * a * d["w_max_mV"] > theta)
        got = sorted(c["mbon_id"] for c in t["by_node"][node]["detectors"])
        eq(got, want, "detectors at node %s" % node)
    fresh = D.build()
    eq(fresh["by_node"], t["by_node"], "table recomputed by the script")


@check("V1c-E5", "If for any T38 type all cells have zero total KC->m "
                 "weight, the stage is untestable by construction")
def t_v1c_testability(V, neurons, con):
    """V1c-E5  All six T38 types have nonzero KC input."""
    w, _ = _v1c_artifacts()
    eq(w["testability_V1c_E5"]["types_with_zero_total_weight"], [],
       "T38 types with zero KC input")
    for t38 in V.T38:
        d = w["by_type_T38"][t38]
        if d["n_cells"] == 0 or d["sum_w_mV_total"] <= 0:
            raise Fail("type %s has no KC input" % t38)


@check("V1c-E7.1", "Consistency check: the table is recomputed from weights "
                   "loaded by the same loader")
def t_v1c_weights_consistent(V, neurons, con):
    """V1c-E7.1  The artifact's KC->MBON weights agree with the subcircuit and the mask."""
    w, _ = _v1c_artifacts()
    eq(w["substrate"]["dan_mask_touches_kc_mbon"], False,
       "substitution mask does not touch KC->MBON edges")
    eq(w["substrate"]["n_edges_kc_mbon_before_mask"],
       w["substrate"]["n_edges_kc_mbon_after_mask"],
       "number of KC->MBON edges before and after the mask")
    eq(sum(d["n_edges_kc"] for d in w["per_mbon"].values()),
       w["all_mbon"]["n_edges_kc"], "sum of edges over cells")
    near(sum(d["sum_w_mV"] for d in w["per_mbon"].values()),
         w["all_mbon"]["sum_w_mV_total"], 1e-6, "sum of weights over cells")
    eq(max(d["w_max_mV"] for d in w["per_mbon"].values()),
       w["s_max"]["w_max_mV_all_mbon"], "max weight over all cells")
    # the structural scaling test on the built network belongs to the stage
    # runner (v1c_stage.check_structural_scaling): it requires an assembled
    # Brian 2 network, and this suite must run in seconds and stand before
    # every shard. What is checked here is what is checkable without the
    # simulator.


# --- V1c stage runner ---------------------------------------------------------
@check("V1c-E2.1", "s multiplies the weight of every KC->MBON edge of the "
                   "subcircuit; other edges are not affected")
def t_v1c_knob_edge_sets(V, neurons, con):
    """V1c-E2.1  The PN->KC and KC->MBON edge sets do not overlap."""
    role = dict(zip(neurons.root_id, neurons.mb_role))
    kept, _ = V.apply_dan_mask(con, role)
    apl = set(neurons.loc[neurons.mb_role == "APL", "root_id"])
    core = set(neurons.root_id) - apl
    e = kept[kept.Presynaptic_ID.isin(core) & kept.Postsynaptic_ID.isin(core)]
    pre = e.Presynaptic_ID.map(role).to_numpy()
    post = e.Postsynaptic_ID.map(role).to_numpy()
    sel_pn = (pre == "PN") & (post == "Kenyon_Cell")
    sel_kc = (pre == "Kenyon_Cell") & (post == "MBON")
    true(not bool((sel_pn & sel_kc).any()),
         "PN->KC and KC->MBON edges overlapped: the order of assigning "
         "multipliers would become significant")
    true(bool(sel_pn.any()) and bool(sel_kc.any()),
         "one of the edge sets is empty")
    w = _v1c_artifacts()[0]
    eq(int(sel_kc.sum()), w["all_mbon"]["n_edges_kc"],
       "KC->MBON edges in the core versus the V1c-E6.1 measurement")


@check("V1c-E3.3", "The stage is run on 57 V1b' candidates multiplied by "
                   "the grid over s; 285 three-dimensional points")
def t_v1c_grid3d(V, neurons, con):
    """V1c-E3.3  285 three-dimensional points: 57 candidates times five nodes."""
    import v1c_stage as S
    g = S.grid_3d()
    eq(len(g), 285, "number of three-dimensional points")
    eq(len(S.candidates()), 57, "number of V1b' candidates")
    from collections import Counter
    eq(sorted(Counter(i for i, _, _ in g).values()), [5] * 57,
       "nodes per candidate")
    eq(sorted(Counter(s for _, _, s in g).values()), [57] * 5,
       "candidates per node")
    # every candidate must be a point that passed the fraction constraint
    for _, c, _ in g:
        if not V.passes_fraction(c):
            raise Fail("candidate %r does not pass the fraction constraint"
                       % ((c["pn_kc_scale"], c["g_apl_rel"]),))


@check("V1c-E3.2", "The run config reads nodes as strings, and nodes are "
                   "not recomputed at runtime")
def t_v1c_nodes_from_config(V, neurons, con):
    """V1c-E3.2  Nodes are taken as strings from the config, not from formulas."""
    import v1c_stage as S
    eq(S.nodes(), V1C_NODES, "nodes read by the runner from the config")
    for s in S.nodes():
        if not isinstance(s, str):
            raise Fail("node %r is not a string" % (s,))
    # boundary nodes are rounded down: the strict inequality must hold even
    # after converting the string to a number
    w, _ = _v1c_artifacts()
    a, theta = w["epsp"]["A"], w["model_constants"]["theta_mV"]
    for s, wm in (("5.3875", w["s_max"]["w_max_mV_T38"]),
                  ("1.6836", w["s_max"]["w_max_mV_all_mbon"])):
        if not float(s) * a * wm < theta:
            raise Fail("node %s: the inequality is not strict after float(string)" % s)


@check("V1c-E7.2", "d = v_th - max_t v(t); d >= 0, and d = 0 if and only if "
                   "the cell fired a spike in this window")
def t_v1c_threshold_map_definition(V, neurons, con):
    """V1c-E7.2  Distance to threshold: definition, sign and the spike flag."""
    import v1c_stage as S
    from model import default_params as dp
    v_th = float(dp["v_th"] / (0.001 * 1.0))

    apl = set(neurons.loc[neurons.mb_role == "APL", "root_id"])
    core = sorted(set(neurons.root_id) - apl)
    role = dict(zip(neurons.root_id, neurons.mb_role))
    n_tr = 6
    counts = np.zeros((len(core), n_tr), dtype=np.int32)
    vmax = np.full((len(core), n_tr), v_th - 1.5)
    mb = [k for k, i in enumerate(core) if role[i] == "MBON"]
    counts[mb[0], 0] = 3                      # spiking cell
    vmax[mb[0], 0] = v_th + 0.4               # its peak is above threshold
    vmax[mb[1], 1] = v_th - 0.25              # subthreshold, closer to threshold
    res = {"core_ids": core, "counts": counts, "vmax_mV": vmax,
           "counts_by_window": [counts], "windows": [(0.0, 5.0)],
           "seeds": list(range(n_tr))}
    df = S.threshold_rows({"o1": res}, neurons, cand_idx=0,
                          cand={"pn_kc_scale": 1.0, "g_apl_rel": 0.0},
                          s_node="1", panel="C")
    eq(len(df), 97 * n_tr, "rows for one odor: MBON cells times trials")
    true(bool((df.d_peak_mV >= 0).all()), "d >= 0 in all rows")
    z = df[df.d_peak_mV == 0]
    eq(int(len(z)), 1, "rows with d = 0")
    true(bool(z.spiked.iloc[0]), "the row with d = 0 must be the one that spiked")
    eq(int(df.spiked.sum()), 1, "rows with the spike flag")
    sub = df[(df.mbon_id == core[mb[1]]) & (df.trial == 2)]
    eq(float(sub.d_peak_mV.iloc[0]), 0.25, "d of the subthreshold cell")
    eq(str(df.d_peak_mV.dtype), "float32", "type of field d_peak_mV")
    eq(str(df.spiked.dtype), "bool", "type of field spiked")
    for col in ("cand_idx", "pn_kc_scale", "g_apl_rel", "s_node", "mbon_id",
                "hemibrain_type", "panel", "odor", "trial", "d_peak_mV",
                "spiked", "no_kc_input"):
        true(col in df.columns, "field %s missing from the map" % col)


@check("V1c-E7.3", "The tolerance is bitwise: for integer values and lists "
                   "- exact equality, for floating-point numbers - exact "
                   "equality of representations")
def t_v1c_bitwise(V, neurons, con):
    """V1c-E7.3  Bitwise comparison: NaN matches NaN, a close value does not."""
    import v1c_stage as S
    true(S._same(1.0, 1.0), "1.0 did not match 1.0")
    true(not S._same(1.0, 1.0 + 2.220446049250313e-16), "close numbers matched")
    true(S._same(float("nan"), float("nan")), "NaN did not match NaN")
    true(not S._same(float("nan"), 0.0), "NaN matched zero")
    true(not S._same(0.0, -0.0), "zeros of different sign matched")
    true(S._same([1, 2], [1, 2]) and not S._same([1, 2], [2, 1]),
         "list comparison is not bitwise")
    true(S._same({"a": 1.5}, {"a": 1.5}) and not S._same({"a": 1.5}, {"b": 1.5}),
         "dict comparison is not bitwise")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    import v1b_subcircuit as V
    neurons, con = V.load_substrate()

    print("Code conformance to specification: V1b/V1b' and V1c "
          "pre-registration (%d checks)" % len(CHECKS))
    print("=" * 78)
    failed = []
    for code, requirement, fn in CHECKS:
        try:
            fn(V, neurons, con)
            status, detail = "OK  ", ""
        except Fail as e:
            status, detail = "FAIL", str(e)
            failed.append((code, requirement, detail))
        except Exception as e:  # an error in the test itself is also a failure
            status = "ERROR"
            detail = "%s: %s" % (type(e).__name__, e)
            failed.append((code, requirement, detail))
        print("%-6s %-10s %s" % (status, code, fn.__doc__ or requirement[:58]))
        if detail:
            print("       -> %s" % detail)
        if a.verbose:
            print("       requirement: %s" % requirement)

    print("=" * 78)
    if failed:
        print("failed %d out of %d:" % (len(failed), len(CHECKS)))
        for code, requirement, detail in failed:
            print("  %s — %s" % (code, requirement))
            print("     %s" % detail)
        return 1
    print("all %d checks passed" % len(CHECKS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
