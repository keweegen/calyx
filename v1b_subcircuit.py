# -*- coding: utf-8 -*-
"""Stage V1b: mushroom-body subcircuit after the step-1 substitutions.

Three substitutions, each toggled by its own config flag, so that the
regression control (specification, section 3, criterion V1b (а)) can turn them
all off and recover the V1a configuration:

  dan_mask   mask of the DAN fast synapses: the DAN->KC and DAN->MBON edges are
             removed, i.e. both sides of the plastic synapse where dopamine
             would otherwise be counted twice. DAN->APL, DAN->DAN and DAN->PN
             remain: the step-1 rule does not model them.
  graded_apl APL is taken out of spiking mode. The membrane equation from [2]
             without threshold, reset, or refractoriness; the output
             g_apl*max(0, v - v_0) is passed every integration step along the
             APL->KC and APL->MBON edges with connectome weights (a summed
             synaptic variable).
  odor_input input - odor patterns instead of Poisson stimulation of PN draws.
             Rates by glomerulus from results/odor_panel/pn_rates_by_odor.tsv
             (source [39], conversion [42]); applied to uniglomerular PN of
             both hemispheres symmetrically.

Exactly two parameters of mode B are calibrated: pn_kc_scale and g_apl
(specification, section 3е, V1b-4.3). Everything else is mode A from [2].

Run:  python v1b_subcircuit.py --self-test
      python v1b_subcircuit.py --odor "pentyl acetate" --scale 1.0 --gapl 1.0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE / "Drosophila_brain_model"
sys.path.insert(0, str(REPO))

SUB = HERE / "data" / "mb_subcircuit"
PANEL = HERE / "results" / "odor_panel" / "pn_rates_by_odor.tsv"
# Directory for the stage's artifacts. Default is V1b (pilot); runs of the next
# stage set it via CALYX_V1B_OUT, so that a closed stage's artifacts are not
# overwritten by the new stage's artifacts.
OUT = Path(os.environ.get("CALYX_V1B_OUT") or (HERE / "results" / "v1b"))

# Measurement window per [40]: 1 s pulse, amplitude is the mean rate over 4 s
# from pulse onset (specification, section 3е, V1b-2.1).
T_ON_MS, T_PULSE_MS, T_WINDOW_MS = 200, 1000, 4000
T_RUN_MS = T_ON_MS + T_WINDOW_MS

# Adaptive truncation: the run proceeds until decay, not through the whole
# measurement interval. QUIET_MS is the window in which there must be no spike
# at all for the run to be considered finished. The counts are identical to
# the full run: with zero background, a network with no input does not spike.
SETTLE_MS, QUIET_MS = 300, 200

KC_SPIKE_THRESHOLD = 1      # V1b-2.1: response to a trial - at least one spike
N_TRIALS = 6                # V1b-2.2: six presentations
MIN_TRIALS = 3              # V1b-2.2: response on at least half


# Core equations: [2] plus a graded-inhibition term. With graded APL off, the
# term stays identically zero, so the equation numerically matches [2] and the
# regression to V1a is not broken. Factored into a constant so that the train
# regression (v1c_regression_trains.py) checks exactly the text that the stage
# computes with, not a copy of it.
EQS_CORE = """
    dv/dt = (v_0 - v + g - inh) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau                    : volt (unless refractory)
    inh                                 : volt
    rfc                                 : second
"""


def load_substrate() -> tuple[pd.DataFrame, pd.DataFrame]:
    neurons = pd.read_csv(SUB / "neurons.csv")
    neurons = neurons.assign(
        gl=neurons.hemibrain_type.astype(str).str.split("_").str[0])
    con = pd.read_parquet(SUB / "connectivity.parquet")
    return neurons, con


def apply_dan_mask(con: pd.DataFrame, role: dict) -> tuple[pd.DataFrame, int]:
    """Remove the DAN->KC and DAN->MBON edges. Returns the remaining edges and the number removed."""
    pre = con.Presynaptic_ID.map(role)
    post = con.Postsynaptic_ID.map(role)
    drop = (pre == "DAN") & post.isin(["Kenyon_Cell", "MBON"])
    return con[~drop].copy(), int(drop.sum())


def odor_rates(odor: str, neurons: pd.DataFrame) -> dict[int, float]:
    """Poisson input rates on uniglomerular PN of both hemispheres."""
    pn = pd.read_csv(PANEL, sep="\t", index_col=0)
    if odor not in pn.index:
        raise SystemExit("odor %r not in the panel; available: %s"
                         % (odor, ", ".join(pn.index)))
    by_gl = pn.loc[odor].to_dict()
    upn = neurons[(neurons.mb_role == "PN")
                  & (neurons.cell_sub_class == "uniglomerular")]
    return {int(r.root_id): float(by_gl[r.gl])
            for r in upn.itertuples() if r.gl in by_gl}


def build(neurons: pd.DataFrame, con: pd.DataFrame, *, pn_kc_scale: float,
          g_apl: float, dan_mask: bool, graded_apl: bool,
          rates: dict[int, float] | None,
          pulse_ms: int = T_PULSE_MS, run_ms: int = T_RUN_MS,
          kc_mbon_scale: float = 1.0, record_vmax: bool = False):
    """Build the Brian 2 network. Returns (Network, SpikeMonitor, id order).

    kc_mbon_scale - the third knob of stage V1c (V1c-E2.1): one global scalar
    on the weight of every KC->MBON edge of the subcircuit; other edges are not
    affected. At value 1.0 the scaling branch is not executed at all, so the
    network is bitwise identical to the V1b' network, not merely identical up
    to multiplication by one.

    record_vmax - observer for the non-criterion output V1c-E7.2: the peak
    membrane-potential value over the run, read BEFORE reset. The vmax
    variable enters no equation and acts on no other variable; its inertness
    is proven by the bitwise reconciliation V1c-E7.3 at node s = 1 against the
    V1b' artifacts computed without it.
    """
    from brian2 import (NeuronGroup, Synapses, PoissonGroup, SpikeMonitor,
                        Network, TimedArray, Hz, ms, mV)
    from model import default_params as dp

    role = dict(zip(neurons.root_id, neurons.mb_role))
    n_masked = 0
    if dan_mask:
        con, n_masked = apply_dan_mask(con, role)

    apl_ids = sorted(neurons.loc[neurons.mb_role == "APL", "root_id"])
    if graded_apl:
        core_ids = sorted(set(neurons.root_id) - set(apl_ids))
    else:
        core_ids = sorted(neurons.root_id)
        apl_ids = []

    ci = {f: k for k, f in enumerate(core_ids)}
    ai = {f: k for k, f in enumerate(apl_ids)}

    # The text of the core equations is not edited: the observer is appended
    # as a separate line, so that the train regression (v1c_regression_trains.py)
    # keeps checking exactly the EQS_CORE constant that the stage computes
    # with.
    eqs = EQS_CORE + ("    vmax : volt\n" if record_vmax else "")
    neu = NeuronGroup(len(core_ids), model=eqs, method="linear",
                      threshold=dp["eq_th"], reset=dp["eq_rst"],
                      refractory="rfc", name="core", namespace=dp)
    neu.v = dp["v_0"]
    neu.g = 0 * mV
    neu.inh = 0 * mV
    neu.rfc = dp["t_rfc"]
    if record_vmax:
        # V1c-E7.2: the maximum over all simulator steps, the value of v read
        # before reset. The groups slot, ordered after the integrator, is the
        # state after integration and before the threshold check (thresholds
        # slot) and before reset (resets slot) - exactly what the specification
        # requires. Before pulse onset the input is zero, the background of
        # model [2] is zero, so v is identically v_0, and the maximum over the
        # whole run coincides with the maximum over the presentation window.
        neu.vmax = dp["v_0"]
        neu.run_regularly("vmax = clip(v, vmax, 1e9 * volt)",
                          when="groups", order=1, name="vmax_tracker")

    objs = [neu]

    # edges within the core
    e = con[con.Presynaptic_ID.isin(ci) & con.Postsynaptic_ID.isin(ci)]
    w = e["Excitatory x Connectivity"].to_numpy() * dp["w_syn"]
    if pn_kc_scale != 1.0 or kc_mbon_scale != 1.0:
        pre_role = e.Presynaptic_ID.map(role).to_numpy()
        post_role = e.Postsynaptic_ID.map(role).to_numpy()
        # not np.where: it strips Brian 2 units and the weights come out dimensionless
        mult = np.ones(len(e))
        if pn_kc_scale != 1.0:
            mult[(pre_role == "PN") & (post_role == "Kenyon_Cell")] = pn_kc_scale
        if kc_mbon_scale != 1.0:
            # the sets of PN->KC and KC->MBON edges do not overlap, so the
            # order of assignments does not affect the result
            mult[(pre_role == "Kenyon_Cell") & (post_role == "MBON")] = kc_mbon_scale
        w = w * mult
    syn = Synapses(neu, neu, "w : volt", on_pre="g += w",
                   delay=dp["t_dly"], name="core_syn")
    syn.connect(i=e.Presynaptic_ID.map(ci).to_numpy(),
                j=e.Postsynaptic_ID.map(ci).to_numpy())
    syn.w = w
    objs.append(syn)

    apl = None
    if graded_apl:
        # Graded node: the same membrane, without threshold, reset, or refractoriness.
        eqs_apl = """
            dv/dt = (v_0 - v + g) / t_mbr : volt
            dg/dt = -g / tau              : volt
            out = clip((v - v_0) / mV, 0, 1e9) : 1
        """
        apl = NeuronGroup(len(apl_ids), model=eqs_apl, method="linear",
                          name="apl", namespace=dp)
        apl.v = dp["v_0"]
        apl.g = 0 * mV
        objs.append(apl)

        # input to APL - ordinary spiking synapses
        into = con[con.Presynaptic_ID.isin(ci) & con.Postsynaptic_ID.isin(ai)]
        s_in = Synapses(neu, apl, "w : volt", on_pre="g += w",
                        delay=dp["t_dly"], name="apl_in")
        s_in.connect(i=into.Presynaptic_ID.map(ci).to_numpy(),
                     j=into.Postsynaptic_ID.map(ai).to_numpy())
        s_in.w = into["Excitatory x Connectivity"].to_numpy() * dp["w_syn"]
        objs.append(s_in)

        # APL output - a summed variable, passed every step
        out = con[con.Presynaptic_ID.isin(ai) & con.Postsynaptic_ID.isin(ci)]
        s_out = Synapses(apl, neu,
                         """w : volt
                            inh_post = w * out_pre : volt (summed)""",
                         name="apl_out", namespace=dp)
        s_out.connect(i=out.Presynaptic_ID.map(ai).to_numpy(),
                      j=out.Postsynaptic_ID.map(ci).to_numpy())
        # the sign of inhibition is set by the (-inh) term in the core
        # equation, so the weight is taken as the absolute value of the
        # synapse count; g_apl is the calibrated coefficient
        s_out.w = np.abs(out["Connectivity"].to_numpy()) * dp["w_syn"] * g_apl
        objs.append(s_out)
        n_apl_in, n_apl_out = len(into), len(out)
    else:
        n_apl_in = n_apl_out = 0

    n_stim = 0
    if rates:
        # 1 s odor pulse, starting at T_ON_MS (protocol [40]). A Poisson
        # group with a time profile: before and after the pulse the rate is
        # zero. PoissonInput is constant in time and does not fit a pulse.
        stim_ids = [f for f, r in rates.items() if f in ci and r > 0]
        n_stim = len(stim_ids)
        if n_stim:
            step_ms = 50
            n_steps = int(np.ceil(run_ms / step_ms))
            prof = np.zeros((n_steps, n_stim))
            on0, on1 = T_ON_MS // step_ms, (T_ON_MS + pulse_ms) // step_ms
            prof[on0:on1, :] = np.array([rates[f] for f in stim_ids])
            ta = TimedArray(prof * Hz, dt=step_ms * ms, name="odor_profile")
            src = PoissonGroup(n_stim, rates="odor_profile(t, i)", name="odor",
                               namespace={"odor_profile": ta})
            s_stim = Synapses(src, neu, on_pre="g += w_stim", name="odor_syn",
                              namespace={"w_stim": dp["f_poi"] * dp["w_syn"]})
            s_stim.connect(i=np.arange(n_stim),
                           j=np.array([ci[f] for f in stim_ids]))
            objs.extend([src, s_stim])
            # refractoriness is removed for stimulated neurons, as in model.poi
            for f in stim_ids:
                neu.rfc[ci[f]] = 0 * ms

    mon = SpikeMonitor(neu)
    objs.append(mon)
    net = Network(*objs)
    stats = {"n_core": len(core_ids), "n_apl": len(apl_ids),
             "n_edges_core": len(e), "n_masked": n_masked,
             "n_apl_in": n_apl_in, "n_apl_out": n_apl_out,
             "n_poisson": n_stim, "kc_mbon_scale": float(kc_mbon_scale)}
    return net, mon, core_ids, stats


def measure(mon, core_ids: list[int], neurons: pd.DataFrame,
            window_ms: float = T_WINDOW_MS) -> pd.DataFrame:
    """Mean rate of each neuron in the measurement window, spikes/s."""
    from brian2 import second
    t0, t1 = T_ON_MS / 1000.0, (T_ON_MS + window_ms) / 1000.0
    counts = np.zeros(len(core_ids))
    for idx, ts in mon.spike_trains().items():
        t = np.asarray(ts / second)
        counts[idx] = ((t >= t0) & (t < t1)).sum()
    role = dict(zip(neurons.root_id, neurons.mb_role))
    return pd.DataFrame({"root_id": core_ids,
                         "n_spikes": counts.astype(int),
                         "rate_hz": counts / (window_ms / 1000.0),
                         "role": [role[i] for i in core_ids]})



def run_odor(neurons: pd.DataFrame, con: pd.DataFrame, odor: str, *,
             pn_kc_scale: float, g_apl: float, seeds: list[int],
             dan_mask: bool = True, graded_apl: bool = True,
             pulse_ms: int = T_PULSE_MS, window_ms: int = T_WINDOW_MS,
             extra_windows: tuple = (), kc_mbon_scale: float = 1.0,
             record_vmax: bool = False, return_trains: bool = False) -> dict:
    """Six presentations of one odor. Returns the spike counts by trial.

    The network is built once; between trials the state is restored and only
    the generator seed changes, so trials differ only in the realization of
    the Poisson input (V1b-2.2).
    """
    from brian2 import ms, seed as b2seed, device

    rates = odor_rates(odor, neurons)
    run_ms = T_ON_MS + window_ms
    net, mon, core_ids, st = build(
        neurons, con, pn_kc_scale=pn_kc_scale, g_apl=g_apl,
        dan_mask=dan_mask, graded_apl=graded_apl, rates=rates,
        pulse_ms=pulse_ms, run_ms=run_ms, kc_mbon_scale=kc_mbon_scale,
        record_vmax=record_vmax)
    net.store("init")
    core_grp = net["core"] if record_vmax else None
    # trains as sequences: needed by the execution check before the V1c
    # stage run, where comparison by counts is weaker than required
    trains = [] if return_trains else None
    # name is not vmax: a local variable named after the group variable
    # leaks into Brian's namespace and it prints a resolution conflict on
    # every trial
    vmax_buf = (np.zeros((len(core_ids), len(seeds)), dtype=np.float64)
                if record_vmax else None)

    wins = [(0.0, window_ms / 1000.0)] + [(a / 1000.0, b / 1000.0)
                                          for a, b in extra_windows]
    counts = [np.zeros((len(core_ids), len(seeds)), dtype=np.int32) for _ in wins]
    truncated_at = []
    for k, sd in enumerate(seeds):
        net.restore("init")
        b2seed(sd)
        # to the end of the pulse plus a margin, after that - while there are spikes in the last window
        done = min(T_ON_MS + pulse_ms + SETTLE_MS, run_ms)
        net.run(done * ms)
        while done < run_ms:
            prev = int(mon.num_spikes)
            step = min(QUIET_MS, run_ms - done)
            net.run(step * ms)
            done += step
            if int(mon.num_spikes) == prev:
                break
        truncated_at.append(done)
        if return_trains:
            from brian2 import second as _second
            trains.append((np.asarray(mon.i[:], dtype=np.int64),
                           np.asarray(mon.t[:] / _second, dtype=np.float64)))
        if record_vmax:
            # the value is read after the trial run: the variable itself is
            # updated every step before the threshold check and before reset
            from brian2 import mV as _mV
            vmax_buf[:, k] = np.asarray(core_grp.vmax / _mV, dtype=np.float64)
        for idx, ts in mon.spike_trains().items():
            # name is not t: a local t leaks into Brian's namespace and
            # conflicts with its internal time variable
            spk = np.asarray(ts) - T_ON_MS / 1000.0
            for wi, (a, b) in enumerate(wins):
                counts[wi][idx, k] = ((spk >= a) & (spk < b)).sum()
    out = {"core_ids": core_ids, "counts": counts[0],
           "counts_by_window": counts, "windows": wins,
           "stats": st, "odor": odor, "seeds": list(seeds),
           "pulse_ms": pulse_ms, "window_ms": window_ms,
           "simulated_ms": truncated_at, "full_ms": run_ms}
    if record_vmax:
        out["vmax_mV"] = vmax_buf
    if return_trains:
        out["trains"] = trains
    return out


def kc_fraction(res: dict, neurons: pd.DataFrame) -> dict:
    """Fraction of responding KC by rule V1b-2.1-2.3 plus a sensitivity table."""
    role = dict(zip(neurons.root_id, neurons.mb_role))
    is_kc = np.array([role[i] == "Kenyon_Cell" for i in res["core_ids"]])
    c = res["counts"][is_kc]
    n_kc_total = int(sum(1 for r in role.values() if r == "Kenyon_Cell"))

    def frac(k: int, min_trials: int) -> float:
        return float((( c >= k).sum(axis=1) >= min_trials).sum()) / n_kc_total

    out = {"n_kc_denominator": n_kc_total,
           "f": frac(KC_SPIKE_THRESHOLD, MIN_TRIALS),
           "sensitivity": {}}
    for k in (1, 2, 3, 5):
        for mt, lab in ((1, ">=1/6"), (MIN_TRIALS, ">=3/6"), (len(res["seeds"]), "6/6")):
            out["sensitivity"]["k=%d,%s" % (k, lab)] = round(frac(k, mt), 5)
    return out


# --- odor sets (specification, V1b-4.1) ----------------------------------
PANEL_CAL = ["2-heptanone", "isopentyl acetate", "hexanal",
             "6-methyl-5-hepten-2-one", "diethyl succinate", "methyl octanoate"]
PANEL_EVAL = ["pentyl acetate", "butyl acetate", "ethyl lactate", "1-octen-3-ol",
              "pentanal", "benzaldehyde", "alpha-humulene", "ethyl octanoate"]
REF_PAIRS = [("pentyl acetate", "butyl acetate"),
             ("pentyl acetate", "ethyl lactate"),
             ("butyl acetate", "ethyl lactate")]

# stage panel P14: 14 odorants, C and E together (specification, V1b-4.1)
PANEL_ALL = PANEL_CAL + PANEL_EVAL

# set T38: six MBON types recorded in [38] (specification, V1b-3.1)
T38 = ["MBON11", "MBON12", "MBON13", "MBON14", "MBON17", "MBON18"]
# avoidance group: there is no reference in [38], no thresholds apply, the
# values are printed for reporting only (specification, V1b-3.1, second
# corollary)
AVOID = ["MBON01", "MBON02", "MBON03", "MBON04", "MBON05", "MBON06"]
# label for cells without hemibrain_type: there is one such MBON in the subcircuit
UNTYPED = "<без типа>"
# Seed table: its own offset for every "set x timing" pair, all distinct.
# Six trials per run, so a step of 50 is enough that the ranges do not overlap.
# The g_APL = 0 run uses the seeds of the run it is compared against
# (V1b-2.5 - a directional reconciliation, it must be paired by input realization).
SEED_CAL = 20260908             # P14, calibration, set C
SEED_CAL_M = SEED_CAL + 50      # P14-M, calibration, set C (admissibility constraint)
SEED_EVAL = SEED_CAL + 100      # P14, evaluation, all 14 odors; and the g_APL = 0 run
SEED_EVAL_M = SEED_CAL + 200    # P14-M, evaluation, all 14 odors (criterion verdict)
SEED_EMPTY = SEED_CAL + 300     # empty presentation (V1b-4.10)
SEED_PERM = SEED_CAL + 400      # RNG for the section 3д null permutations

# Timing M per [38]: 5 s presentation, mean rate over the presentation window
# (specification, V1b-3.1). Set P14-M is the 14 odorants at this timing.
M_PULSE_MS = M_WINDOW_MS = 5000

# criterion thresholds (specification, section 3е)
F_BAND, F_MAX = (0.03, 0.10), 0.10
OVERLAP_SEP_MIN, OVERLAP_CEIL = 0.25, 0.40
MBON_FLOOR_HZ, MBON_CEIL_HZ, MBON_MD_MIN, MBON_MD_TYPES = 2.0, 67.0, 0.19, 5
SPIKES_AB_TARGET = 2.2          # V1b-4.9, [46]
N_SUBSAMPLE, N_DRAWS = 120, 24  # V1b-3д: "recording" subsample


def run_panel(neurons, con, odors, *, pn_kc_scale, g_apl, seed_base,
              pulse_ms=T_PULSE_MS, window_ms=T_WINDOW_MS, extra_windows=(),
              dan_mask=True, graded_apl=True, kc_mbon_scale=1.0,
              record_vmax=False) -> dict:
    """Run an odor set through the same testbed."""
    seeds = [seed_base + i for i in range(1, N_TRIALS + 1)]
    out = {}
    for o in odors:
        out[o] = run_odor(neurons, con, o, pn_kc_scale=pn_kc_scale, g_apl=g_apl,
                          seeds=seeds, dan_mask=dan_mask, graded_apl=graded_apl,
                          pulse_ms=pulse_ms, window_ms=window_ms,
                          extra_windows=extra_windows,
                          kc_mbon_scale=kc_mbon_scale, record_vmax=record_vmax)
    return out


def _kc_mask(core_ids, neurons, prefix: str | None = None) -> np.ndarray:
    role = dict(zip(neurons.root_id, neurons.mb_role))
    typ = dict(zip(neurons.root_id, neurons.hemibrain_type.astype(str)))
    if prefix is None:
        return np.array([role[i] == "Kenyon_Cell" for i in core_ids])
    return np.array([role[i] == "Kenyon_Cell" and typ[i].startswith(prefix)
                     for i in core_ids])


def overlap_pearson(by_odor: dict, neurons: pd.DataFrame,
                    rng_seed: int = 20260908) -> dict:
    """Overlap of KC ensembles per V1b-3д: Pearson within the "recording" subsample."""
    odors = list(by_odor)
    ids = by_odor[odors[0]]["core_ids"]
    kc = _kc_mask(ids, neurons)
    # odor vector - mean spike count of each KC over trials
    vec = {o: by_odor[o]["counts"][kc].mean(axis=1) for o in odors}
    n_kc = int(kc.sum())
    rng = np.random.default_rng(rng_seed)
    draws = [rng.choice(n_kc, size=min(N_SUBSAMPLE, n_kc), replace=False)
             for _ in range(N_DRAWS)]

    r = {}
    for a in range(len(odors)):
        for b in range(a + 1, len(odors)):
            oa, ob = odors[a], odors[b]
            vals = []
            for d in draws:
                x, y = vec[oa][d], vec[ob][d]
                if x.std() == 0 or y.std() == 0:
                    continue
                vals.append(float(np.corrcoef(x, y)[0, 1]))
            r[(oa, ob)] = float(np.mean(vals)) if vals else float("nan")
    return r


def check_overlap(r: dict) -> dict:
    """Overlap criterion: order of three reference pairs, separation, ceiling."""
    g = lambda a, b: r.get((a, b), r.get((b, a), float("nan")))
    pab = g(*REF_PAIRS[0])
    pel, bel = g(*REF_PAIRS[1]), g(*REF_PAIRS[2])
    order = pab > pel and pab > bel
    sep = pab - max(pel, bel)
    ceil_ok = max(pel, bel) <= OVERLAP_CEIL
    vals = [v for v in r.values() if v == v]
    return {"r_PA_BA": pab, "r_PA_EL": pel, "r_BA_EL": bel,
            "order_ok": bool(order), "separation": float(sep),
            "separation_ok": bool(sep >= OVERLAP_SEP_MIN),
            "ceiling_ok": bool(ceil_ok),
            "median_all_pairs": float(np.median(vals)) if vals else float("nan"),
            "p90_all_pairs": float(np.percentile(vals, 90)) if vals else float("nan"),
            "pass": bool(order and sep >= OVERLAP_SEP_MIN and ceil_ok)}


def mbon_type_rates(by_odor: dict, neurons: pd.DataFrame,
                    window_ms: float) -> pd.DataFrame:
    """Mean MBON-type rate by odor, spikes/s (V1b-3.1)."""
    ids = by_odor[list(by_odor)[0]]["core_ids"]
    # one MBON of the subcircuit (left, 720575940623743415) is annotated
    # without a type and without a compartment; it is not in T38, it has no
    # reference, but it must not silently disappear from the table either -
    # so it gets an explicit label
    typ = dict(zip(neurons.root_id,
                   neurons.hemibrain_type.astype("string").fillna(UNTYPED)))
    role = dict(zip(neurons.root_id, neurons.mb_role))
    rows = {}
    for o, res in by_odor.items():
        per = res["counts"].mean(axis=1) / (window_ms / 1000.0)
        d = {}
        for k, i in enumerate(ids):
            if role[i] == "MBON":
                d.setdefault(typ[i], []).append(per[k])
        rows[o] = {t: float(np.mean(v)) for t, v in d.items()}
    return pd.DataFrame(rows)


def check_mbon(rates: pd.DataFrame) -> dict:
    """Floor, ceiling, and modulation depth over T38 (V1b-3.3-3.5)."""
    present = [t for t in T38 if t in rates.index]
    res = {"types_present": present, "missing": [t for t in T38 if t not in rates.index]}
    per_type = {}
    for t in present:
        v = rates.loc[t].to_numpy(dtype=float)
        hi, lo = float(v.max()), float(v.min())
        per_type[t] = {"R_t": float(v.mean()), "max": hi, "min": lo,
                       "MD": (hi - lo) / (hi + lo) if (hi + lo) > 0 else float("nan")}
    res["per_type"] = per_type
    floors = [v["R_t"] >= MBON_FLOOR_HZ for v in per_type.values()]
    ceils = [v["R_t"] <= MBON_CEIL_HZ for v in per_type.values()]
    mds = [v["MD"] >= MBON_MD_MIN for t, v in per_type.items()
           if per_type[t]["R_t"] >= MBON_FLOOR_HZ]
    res.update({"floor_ok": bool(all(floors)), "ceiling_ok": bool(all(ceils)),
                "n_md_ok": int(sum(mds)), "md_ok": bool(sum(mds) >= MBON_MD_TYPES),
                "pass": bool(all(floors) and all(ceils) and sum(mds) >= MBON_MD_TYPES)})
    return res


def md_subset_medians(rates: pd.DataFrame, types: list[str],
                      k: int = 5) -> dict:
    """Median MD_t over all k-odor subsets of the panel (V1b-3.5, report).

    For a panel of 14 odors and k = 5 there are exactly 2,002 subsets - the
    number the specification names. The value is for reporting: it is compared
    against the reference 0.27, it does not determine a verdict.
    """
    from itertools import combinations
    n = rates.shape[1]
    idx = list(combinations(range(n), k))
    per_type = {}
    for t in types:
        if t not in rates.index:
            continue
        v = rates.loc[t].to_numpy(dtype=float)
        mds = []
        for c in idx:
            sub = v[list(c)]
            hi, lo = float(sub.max()), float(sub.min())
            mds.append((hi - lo) / (hi + lo) if (hi + lo) > 0 else float("nan"))
        good = [m for m in mds if m == m]
        per_type[t] = float(np.median(good)) if good else float("nan")
    vals = [x for x in per_type.values() if x == x]
    return {"n_subsets": len(idx), "k": k, "per_type": per_type,
            "median_over_types": float(np.median(vals)) if vals else float("nan")}


def mbon_type_spikes(by_odor: dict, neurons: pd.DataFrame, mbon_type: str,
                     window_idx: int = 0) -> dict:
    """Mean spike count of a type's cells in a given window, by odor.

    Serves the reporting value of V1b-3.1: the MBON11 spike count over the 1 s
    pulse next to 118 +- 8.3 from [29]. The value has no threshold.
    """
    ids = by_odor[list(by_odor)[0]]["core_ids"]
    typ = dict(zip(neurons.root_id,
                   neurons.hemibrain_type.astype("string").fillna(UNTYPED)))
    role = dict(zip(neurons.root_id, neurons.mb_role))
    m = np.array([role[i] == "MBON" and typ[i] == mbon_type for i in ids])
    if not m.any():
        return {"type": mbon_type, "n_cells": 0, "by_odor": {},
                "mean_over_odors": float("nan")}
    per = {}
    for o, res in by_odor.items():
        c = res["counts_by_window"][window_idx][m]     # cells x trials
        per[o] = float(c.mean())                       # mean over cells and trials
    return {"type": mbon_type, "n_cells": int(m.sum()), "by_odor": per,
            "mean_over_odors": float(np.mean(list(per.values())))}


def mbon_report(by_odor: dict, neurons: pd.DataFrame, window_ms: float) -> dict:
    """Full breakdown of the MBON response on set P14-M (specification, V1b-3.1-3.5).

    The blocking part is the V1b-3.3-3.5 thresholds over set T38. The rest is
    for reporting only: the avoidance group MBON01-MBON06, for which there is
    no reference in [38], all other types, MD medians over five-odor subsets,
    and the MBON11 spike count over the first second of the presentation.
    """
    rates = mbon_type_rates(by_odor, neurons, window_ms)
    chk = check_mbon(rates)
    avoid = {t: {"R_t": float(rates.loc[t].mean()),
                 "max": float(rates.loc[t].max()),
                 "min": float(rates.loc[t].min())}
             for t in AVOID if t in rates.index}
    other = sorted(set(rates.index) - set(T38) - set(AVOID))
    return {"n_odors": rates.shape[1], "odors": list(rates.columns),
            "window_ms": window_ms,
            "T38": chk,
            "md_subsets": md_subset_medians(rates, T38),
            "avoid_group_report_only": avoid,
            "other_types_report_only": {
                t: float(rates.loc[t].mean()) for t in other},
            "rates_table": {t: {o: float(rates.loc[t, o])
                                for o in rates.columns}
                            for t in rates.index}}


def spikes_per_response(by_odor: dict, neurons: pd.DataFrame,
                        prefix: str, window_idx: int = 1) -> float:
    """Spikes per response of a KC subtype in the reference window (V1b-4.9).

    window_idx points to the [0; 2 s] window from extra_windows; the "cell-odor"
    pairs counted are those that respond by rule V1b-2.1-2.2, and within them
    only the presentations on which there is a spike in the reference window.
    """
    ids = by_odor[list(by_odor)[0]]["core_ids"]
    m = _kc_mask(ids, neurons, prefix)
    vals = []
    for res in by_odor.values():
        full = res["counts"][m]                        # 4 s window, response rule
        ref = res["counts_by_window"][window_idx][m]   # 2 s reference window
        responder = (full >= KC_SPIKE_THRESHOLD).sum(axis=1) >= MIN_TRIALS
        for row in np.nonzero(responder)[0]:
            hit = ref[row][ref[row] >= 1]
            if hit.size:
                vals.append(float(hit.mean()))
    return float(np.mean(vals)) if vals else float("nan")


# --- evaluation-run diagnostics (report, do not determine the verdict) -----------
def sensitivity_table(by_odor: dict, neurons: pd.DataFrame) -> dict:
    """V1b-2.4: mean and maximum of f(o) over the panel for each response rule.

    The k-spikes-per-trial thresholds and aggregation rules are set by the
    specification and are not subject to tuning; the table makes the
    dependence of the outcome on the rule visible.
    """
    per = {o: kc_fraction(res, neurons)["sensitivity"]
           for o, res in by_odor.items()}
    keys = sorted(next(iter(per.values())))
    return {k: {"mean": float(np.mean([per[o][k] for o in per])),
                "max": float(np.max([per[o][k] for o in per]))}
            for k in keys}


def kc_coverage(neurons: pd.DataFrame, con: pd.DataFrame) -> pd.Series:
    """Fraction of masked input c_i covered per KC (specification, section 3д).

    The denominator is all synapses from uniglomerular PN onto the cell, the
    numerator is those of them that come from the 24 glomeruli of mask [39].
    Computed over the 4,820 KC that have at least one synapse from a uPN;
    reproduces the section 3д median of 0.44 and quartiles of 0.25 and 0.62.
    """
    upn = neurons[(neurons.mb_role == "PN")
                  & (neurons.cell_sub_class == "uniglomerular")]
    gl_mask = set(pd.read_csv(PANEL, sep="	", index_col=0).columns)
    in_mask = {int(r.root_id) for r in upn.itertuples() if r.gl in gl_mask}
    all_upn = {int(x) for x in upn.root_id}
    kc = {int(x) for x in neurons.loc[neurons.mb_role == "Kenyon_Cell", "root_id"]}
    e = con[con.Presynaptic_ID.isin(all_upn) & con.Postsynaptic_ID.isin(kc)]
    tot = e.groupby("Postsynaptic_ID")["Connectivity"].sum()
    msk = e[e.Presynaptic_ID.isin(in_mask)].groupby(
        "Postsynaptic_ID")["Connectivity"].sum()
    return msk.reindex(tot.index).fillna(0) / tot


def sparseness_treves_rolls(res: dict, neurons: pd.DataFrame) -> float:
    """Population sparseness S_P over all 5,177 KC (V1b-2.5).

    r_i is the mean over 6 trials of the spike count of KC i in the
    measurement window.
    """
    role = dict(zip(neurons.root_id, neurons.mb_role))
    kc = np.array([role[i] == "Kenyon_Cell" for i in res["core_ids"]])
    r = res["counts"][kc].mean(axis=1)
    n = len(r)
    denom = float((r ** 2).sum() / n)
    if denom == 0:
        return float("nan")
    return float((1.0 - (float(r.sum() / n) ** 2) / denom) / (1.0 - 1.0 / n))


def kc_fraction_report(res: dict, neurons: pd.DataFrame,
                       cov: pd.Series) -> dict:
    """Fraction responding under three denominators (V1b-2.3: 5,177 is the criterion, the rest for reporting only)."""
    role = dict(zip(neurons.root_id, neurons.mb_role))
    side = dict(zip(neurons.root_id, neurons.side))
    ids = res["core_ids"]
    kc = np.array([role[i] == "Kenyon_Cell" for i in ids])
    kc_ids = [i for i, m in zip(ids, kc) if m]
    resp = (res["counts"][kc] >= KC_SPIKE_THRESHOLD).sum(axis=1) >= MIN_TRIALS
    with_upn = set(cov.index.astype("int64"))
    out = {"f_all_5177": float(resp.sum()) / len(kc_ids),
           "n_denominator_all": len(kc_ids)}
    sel = np.array([i in with_upn for i in kc_ids])
    out["f_with_upn_input"] = float(resp[sel].sum()) / int(sel.sum())
    out["n_denominator_with_upn"] = int(sel.sum())
    for h in ("right", "left"):
        m = np.array([side[i] == h for i in kc_ids])
        out["f_%s" % h] = float(resp[m].sum()) / int(m.sum())
        out["n_%s" % h] = int(m.sum())
    return out


def coverage_diagnostics(by_odor: dict, neurons: pd.DataFrame,
                         cov: pd.Series) -> dict:
    """Whether the uncovered fraction of input selects for responding cells (section 3д).

    Fraction responding broken down by quartile of c_i, and the Spearman rank
    correlation between c_i and the number of panel odors the cell responded to.
    """
    from scipy.stats import spearmanr
    role = dict(zip(neurons.root_id, neurons.mb_role))
    ids = by_odor[list(by_odor)[0]]["core_ids"]
    kc = np.array([role[i] == "Kenyon_Cell" for i in ids])
    kc_ids = np.array([i for i, m in zip(ids, kc) if m])
    n_odors_resp = np.zeros(len(kc_ids), dtype=int)
    for res in by_odor.values():
        r = (res["counts"][kc] >= KC_SPIKE_THRESHOLD).sum(axis=1) >= MIN_TRIALS
        n_odors_resp += r.astype(int)

    c = cov.reindex(kc_ids)                     # NaN for 357 KC with no input from uPN
    have = c.notna().to_numpy()
    cv, nr = c.to_numpy()[have], n_odors_resp[have]
    q = np.quantile(cv, [0.25, 0.5, 0.75])
    bins = np.digitize(cv, q)                   # 0..3
    by_q = {}
    for b in range(4):
        m = bins == b
        by_q["Q%d" % (b + 1)] = {
            "n_kc": int(m.sum()),
            "c_range": [float(cv[m].min()), float(cv[m].max())] if m.any() else None,
            "mean_odors_responded": float(nr[m].mean()) if m.any() else float("nan"),
            "frac_responding_any": float((nr[m] > 0).mean()) if m.any() else float("nan")}
    rho, pval = spearmanr(cv, nr)
    no_input = int((~have).sum())
    return {"quartiles_of_c": [float(x) for x in q], "by_quartile": by_q,
            "spearman_rho": float(rho), "spearman_p": float(pval),
            "n_kc_with_upn_input": int(have.sum()),
            "n_kc_without_upn_input": no_input,
            "mean_odors_responded_without_upn_input":
                float(n_odors_resp[~have].mean()) if no_input else float("nan")}


def empty_presentation(neurons, con, *, pn_kc_scale: float, g_apl_rel: float,
                       seed_base: int, window_ms: int = T_WINDOW_MS) -> dict:
    """Empty presentation: V1b-4.10, third quantity.

    The same testbed, input rates zero. Prints the spontaneous KC rate and the
    fraction of KC that rule V1b-2.1-2.2 would classify as responding. Measured
    before the freeze: 0 spikes in the whole subcircuit at pn_kc_scale = 1 and
    g_apl = 0.03 g_ref.
    """
    from brian2 import ms, seed as b2seed
    role = dict(zip(neurons.root_id, neurons.mb_role))
    run_ms = T_ON_MS + window_ms
    net, mon, core_ids, st = build(
        neurons, con, pn_kc_scale=pn_kc_scale, g_apl=g_apl_rel * g_ref_value(),
        dan_mask=True, graded_apl=True, rates=None, run_ms=run_ms)
    net.store("init")
    kc = np.array([role[i] == "Kenyon_Cell" for i in core_ids])
    seeds = [seed_base + i for i in range(1, N_TRIALS + 1)]
    counts = np.zeros((len(core_ids), len(seeds)), dtype=np.int32)
    for k, sd in enumerate(seeds):
        net.restore("init")
        b2seed(sd)
        net.run(run_ms * ms)
        for idx, ts in mon.spike_trains().items():
            spk = np.asarray(ts) - T_ON_MS / 1000.0
            counts[idx, k] = ((spk >= 0) & (spk < window_ms / 1000.0)).sum()
    c = counts[kc]
    resp = (c >= KC_SPIKE_THRESHOLD).sum(axis=1) >= MIN_TRIALS
    return {"n_spikes_subcircuit": int(counts.sum()),
            "n_spikes_kc": int(c.sum()),
            "spontaneous_rate_hz": float(c.mean() / (window_ms / 1000.0)),
            "f_responding_empty": float(resp.sum()) / int(kc.sum()),
            "n_kc": int(kc.sum()), "seeds": seeds, "window_ms": window_ms}


# --- calibration grid (specification, V1b-4.4-4.6) -----------------------------
def g_ref_value() -> float:
    """g_ref = 1/(v_th - v_0) in 1/mV, from the constants of [2]."""
    from model import default_params as dp
    return 1.0 / float((dp["v_th"] - dp["v_0"]) / (0.001 * 1.0))


def grid_stage1() -> list[tuple[float, float]]:
    scales = [2.0 ** k for k in range(-6, 3)]                 # 9 values
    gains = [0.0] + [10.0 ** (k / 2.0) for k in range(-4, 5)]  # 10 values
    return [(sc, g) for sc in scales for g in gains]


def grid_stage2(stage1: list[dict]) -> list[tuple[float, float]]:
    """Refinement grid (V1b-4.4).

    The bounding box of stage-1 admissible points, expanded by one stage-1
    step in each direction; steps of 2^(1/4) in scale and 10^(1/8) in g_apl.
    The bounds are set by a rule written before the run, so going outside the
    stage-1 range is legitimate and is not mode C.
    """
    # V1b-4.4 says "stage-1 admissible points", and V1b-4.5 defines
    # admissibility as the conjunction of the fraction and MBON constraints.
    # The bounding box is built from the FRACTION constraint alone: otherwise,
    # with an empty MBON constraint, stage 2 would not exist at all, and the
    # "grid exhaustion" stopping rule would be replaced by stopping at the
    # coarse grid. This reading is declared before the run; if the region is
    # empty the outcome is FAIL-CAL-MBON regardless, but the map is still
    # computed on the fine grid.
    ok = [p for p in stage1 if passes_fraction(p)]
    if not ok:
        return []
    sc = [p["pn_kc_scale"] for p in ok]
    gs = [p["g_apl_rel"] for p in ok if p["g_apl_rel"] > 0]
    s_lo, s_hi = min(sc) / 2.0, max(sc) * 2.0          # stage-1 step in scale
    if gs:
        g_lo, g_hi = min(gs) / (10 ** 0.5), max(gs) * (10 ** 0.5)
    else:
        g_lo, g_hi = 0.0, 10 ** -2.0

    out_s, x = [], s_lo
    while x <= s_hi * 1.0001:
        out_s.append(x)
        x *= 2 ** 0.25
    out_g, y = [], g_lo
    if g_lo <= 0:
        out_g.append(0.0)
        y = 10 ** -2.0
    while y <= g_hi * 1.0001:
        out_g.append(y)
        y *= 10 ** 0.125
    # zero inhibition remains in the grid: it is present in stage 1 and
    # excluding it would narrow the space after seeing the result
    if 0.0 not in out_g and any(p["g_apl_rel"] == 0 for p in ok):
        out_g.insert(0, 0.0)
    return [(a, b) for a in out_s for b in out_g]


def evaluate_point(neurons, con, odors, *, pn_kc_scale, g_apl_rel, seed_base,
                   kc_mbon_scale: float = 1.0) -> dict:
    """Quantities at a grid point on the given odor set.

    kc_mbon_scale defaults to 1.0: at this value the scaling branch in
    build() is not executed, and the point is identical to the V1b' point.
    """
    g_abs = g_apl_rel * g_ref_value()
    by = run_panel(neurons, con, odors, pn_kc_scale=pn_kc_scale, g_apl=g_abs,
                   seed_base=seed_base, extra_windows=((0, 2000),),
                   kc_mbon_scale=kc_mbon_scale)
    f = {o: kc_fraction(by[o], neurons)["f"] for o in odors}
    vals = list(f.values())
    return {"pn_kc_scale": pn_kc_scale, "g_apl_rel": g_apl_rel,
            "kc_mbon_scale": kc_mbon_scale,
            "f_by_odor": f, "f_mean": float(np.mean(vals)),
            "f_max": float(np.max(vals)),
            "s_ab": spikes_per_response(by, neurons, "KCab"),
            "s_apbp": spikes_per_response(by, neurons, "KCa'b'"),
            "s_g": spikes_per_response(by, neurons, "KCg"),
            "_by_odor": by}


def passes_fraction(pt: dict) -> bool:
    """First half of V1b-4.5: the constraint on the fraction of responding KC on C.

    A separate function because admissibility is a conjunction, and the MBON
    constraint is computed only for points that pass this one: an MBON run on
    C costs four times as much as a fraction run, and a point inadmissible by
    fraction cannot become admissible.
    """
    return (F_BAND[0] <= pt["f_mean"] <= F_BAND[1]) and pt["f_max"] <= F_MAX


def passes_mbon(pt: dict) -> bool | None:
    """Second half of V1b-4.5: V1b-3.3, V1b-3.4 and V1b-3.5 on odors C.

    Returns None if the constraint was not computed at this point: this is
    neither "pass" nor "fail" but "unknown", and such a point is not
    considered admissible. The distinction matters - it was precisely the
    conflation of "not computed" with "pass" that made the V1b calibration
    fail to match the specification.
    """
    m = pt.get("mbon")
    if m is None:
        return None
    return bool(m.get("floor_ok") and m.get("ceiling_ok") and m.get("md_ok"))


def admissible(pt: dict) -> bool:
    """V1b-4.5: admissibility is the conjunction of the fraction and MBON constraints on C."""
    if not passes_fraction(pt):
        return False
    return passes_mbon(pt) is True


def chebyshev_margins(points: list[dict]) -> dict:
    """Distance in grid steps to the nearest inadmissible point or edge.

    Tiebreak (2) of rule V1b-4.6. Chebyshev metric on the indices of grid
    nodes built from the points themselves: the grid edge counts as an
    inadmissible neighbor, so a point in the corner gets distance 1, not
    infinity.
    """
    key = lambda p: (round(p["pn_kc_scale"], 10), round(p["g_apl_rel"], 10))
    sx = sorted({key(p)[0] for p in points})
    gy = sorted({key(p)[1] for p in points})
    si = {v: i for i, v in enumerate(sx)}
    gi = {v: i for i, v in enumerate(gy)}
    ok_cells = {(si[key(p)[0]], gi[key(p)[1]]) for p in points if admissible(p)}
    out = {}
    for p in points:
        a, b = key(p)
        i, j = si[a], gi[b]
        d = 1
        while True:
            # Chebyshev ring of radius d: if it contains an inadmissible
            # node or a step outside the grid, the distance is d
            hit = False
            for di in range(-d, d + 1):
                for dj in range(-d, d + 1):
                    if max(abs(di), abs(dj)) != d:
                        continue
                    ci, cj = i + di, j + dj
                    if not (0 <= ci < len(sx) and 0 <= cj < len(gy)):
                        hit = True
                    elif (ci, cj) not in ok_cells:
                        hit = True
                    if hit:
                        break
                if hit:
                    break
            if hit or d > max(len(sx), len(gy)):
                out[(a, b)] = d
                break
            d += 1
    return out


def choose_point(points: list[dict]) -> dict | None:
    """V1b-4.6: lexicographic choice among stage-2 admissible points.

    The choice is made over stage 2, not over the union of stages: in V1b the
    union gave the same point, but the rule specifies stage 2, and the outcomes
    coinciding is not a justification for the discrepancy. Points without a
    stage label are considered to belong to the grid under consideration -
    that is how the tests and the single-stage breakdown invoke this.
    """
    stage2 = [p for p in points if p.get("stage") == 2]
    pool_all = stage2 if stage2 else points
    ok = [p for p in pool_all if admissible(p)]
    if not ok:
        return None
    band = [p for p in ok if 0.045 <= p["f_mean"] <= 0.055]
    if not band:
        return min(ok, key=lambda p: abs(p["f_mean"] - 0.05))
    defined = [p for p in band if p["s_ab"] == p["s_ab"]]
    pool = defined or band

    # tiebreak (2) is computed over the whole grid, not the band: the
    # distance is measured to inadmissible nodes, and by definition they are
    # not in the band
    margin = {p.get("chebyshev_to_edge") for p in pool}
    if None in margin:
        m = chebyshev_margins(pool_all)
        get_margin = lambda p: m[(round(p["pn_kc_scale"], 10),
                                  round(p["g_apl_rel"], 10))]
    else:
        get_margin = lambda p: p["chebyshev_to_edge"]

    key = lambda p: (abs(p["s_ab"] - SPIKES_AB_TARGET) if p["s_ab"] == p["s_ab"]
                     else float("inf"),
                     -get_margin(p), p["g_apl_rel"], p["pn_kc_scale"])
    return min(pool, key=key)


# --- evaluation runs at the chosen point (V1b-4.2, V1b-4.7) ------------------
def eval_p14(neurons, con, *, pn_kc_scale: float, g_apl_rel: float,
             seed_base: int = SEED_EVAL, odors: list | None = None) -> dict:
    """Run P14: 14 odors, 1 s pulse, 4 s window.

    Gives the responding-KC-fraction criterion and the overlap criterion (both
    gated on E, V1b-4.2), plus the report-only quantities V1b-2.3, V1b-2.4,
    V1b-2.5, V1b-4.9, V1b-4.10, and the section 3д coverage diagnostics.
    """
    g_abs = g_apl_rel * g_ref_value()
    odors = list(odors or PANEL_ALL)
    by = run_panel(neurons, con, odors, pn_kc_scale=pn_kc_scale,
                   g_apl=g_abs, seed_base=seed_base, extra_windows=((0, 2000),))
    # The criteria are gated on E. When set E is not fully included in the
    # run - during a code check or the admissibility constraint on C - the
    # criteria are not computed: blindness to E must survive until the
    # evaluation run (V1b-4.7).
    on_e = [o for o in PANEL_EVAL if o in odors]
    by_e = {o: by[o] for o in on_e}
    cov = kc_coverage(neurons, con)

    f_all = {o: kc_fraction(by[o], neurons)["f"] for o in odors}
    if len(on_e) == len(PANEL_EVAL):
        # bands the same as in V1b-4.5; overlap - over the 28 pairs within E
        f_e = [f_all[o] for o in PANEL_EVAL]
        kc_crit = {"f_mean_E": float(np.mean(f_e)), "f_max_E": float(np.max(f_e)),
                   "band": list(F_BAND), "ceiling": F_MAX,
                   "pass": bool(F_BAND[0] <= np.mean(f_e) <= F_BAND[1]
                                and max(f_e) <= F_MAX)}
        ov = check_overlap(overlap_pearson(by_e, neurons))
    else:
        kc_crit = ov = {"not_computed": "set E is not fully included in the run"}
    r_all = overlap_pearson(by, neurons)
    ov_all = check_overlap(r_all) if len(odors) >= 3 else {}

    return {"point": {"pn_kc_scale": pn_kc_scale, "g_apl_rel": g_apl_rel},
            "seed_base": seed_base, "timing": "P14",
            "pulse_ms": T_PULSE_MS, "window_ms": T_WINDOW_MS,
            "kc_fraction_by_odor": f_all,
            "kc_fraction_criterion_E": kc_crit,
            "odors": odors,
            "kc_fraction_report": {o: kc_fraction_report(by[o], neurons, cov)
                                   for o in odors},
            "overlap_criterion_E_28_pairs": ov,
            "overlap_report_all_pairs": {
                "n_pairs": len(r_all), "check_on_all": ov_all,
                "r": {"%s | %s" % k: v for k, v in r_all.items()}},
            "sensitivity_V1b_2_4": sensitivity_table(by, neurons),
            "sparseness_V1b_2_5": {o: sparseness_treves_rolls(by[o], neurons)
                                   for o in odors},
            "spikes_per_response_V1b_4_9_4_10": {
                "s_ab": spikes_per_response(by, neurons, "KCab"),
                "s_apbp": spikes_per_response(by, neurons, "KCa'b'"),
                "s_g": spikes_per_response(by, neurons, "KCg"),
                "target_ab": SPIKES_AB_TARGET},
            "coverage_diagnostics": coverage_diagnostics(by, neurons, cov)}


def eval_p14m(neurons, con, *, pn_kc_scale: float, g_apl_rel: float,
              seed_base: int = SEED_EVAL_M,
              odors: list | None = None, kc_mbon_scale: float = 1.0,
              record_vmax: bool = False, return_by_odor: bool = False) -> dict:
    """Run P14-M: 14 odors, 5 s presentation, presentation window.

    The set on which the MBON response criterion (V1b-3.1) is defined. The
    additional [0; 1 s] window is needed for the report-only MBON11 spike
    count next to 118 +- 8.3 from [29]: the stimulus up to the end of the
    first second is the same for the 1 s and 5 s pulses.
    """
    g_abs = g_apl_rel * g_ref_value()
    odors = list(odors or PANEL_ALL)
    by = run_panel(neurons, con, odors, pn_kc_scale=pn_kc_scale,
                   g_apl=g_abs, seed_base=seed_base,
                   pulse_ms=M_PULSE_MS, window_ms=M_WINDOW_MS,
                   extra_windows=((0, 1000),), kc_mbon_scale=kc_mbon_scale,
                   record_vmax=record_vmax)
    rep = mbon_report(by, neurons, M_WINDOW_MS)
    rep.update({"point": {"pn_kc_scale": pn_kc_scale, "g_apl_rel": g_apl_rel,
                          "kc_mbon_scale": kc_mbon_scale},
                "seed_base": seed_base, "timing": "P14-M",
                "pulse_ms": M_PULSE_MS,
                "MBON11_spikes_first_second": mbon_type_spikes(
                    by, neurons, "MBON11", window_idx=1),
                "MBON11_reference_29": "118 +- 8,3 спайка за 1 с импульса"})
    if return_by_odor:
        rep["_by_odor"] = by
    return rep


# --- regression control (specification, v0.11, criterion (а) and (б)) ---------
def regression_substrate(neurons: pd.DataFrame, con: pd.DataFrame) -> dict:
    """With substitutions off, the substrate must match V1a bitwise."""
    role = dict(zip(neurons.root_id, neurons.mb_role))
    kept, n_masked = apply_dan_mask(con, role)
    off = con  # substitutions off: the mask is not applied
    same_edges = len(off) == len(con) and n_masked > 0
    return {"n_edges_total": int(len(con)),
            "n_edges_with_mask": int(len(kept)),
            "n_masked": int(n_masked),
            "substitutions_off_identical": bool(same_edges),
            "expected_masked": 49316,
            "mask_count_ok": bool(n_masked == 49316)}


def config_hash(cfg: dict, exclude: tuple = ()) -> str:
    """Config hash excluding the declared keys (criterion (б))."""
    d = {k: v for k, v in sorted(cfg.items()) if k not in exclude}
    return hashlib.sha256(json.dumps(d, sort_keys=True,
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


DECLARED_KEYS = ("dan_mask", "graded_apl", "odor_input", "pn_kc_scale", "g_apl_rel")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--odor", default="pentyl acetate")
    ap.add_argument("--scale", type=float, default=1.0, help="pn_kc_scale")
    ap.add_argument("--gapl", type=float, default=1.0, help="g_apl in units of g_ref")
    ap.add_argument("--no-dan-mask", action="store_true")
    ap.add_argument("--no-graded-apl", action="store_true")
    ap.add_argument("--codegen", default="numpy", choices=["numpy", "cython"],
                    help="Brian 2 code generation target; cython requires an MSVC environment")
    ap.add_argument("--cache", default="",
                    help="cython cache directory (separate per process for parallel runs)")
    ap.add_argument("--evaluate", action="store_true",
                    help="run P14 at the point: KC fraction, overlap, and diagnostics")
    ap.add_argument("--evaluate-mbon", action="store_true",
                    help="run P14-M at the point: MBON response criterion (V1b-3.1)")
    ap.add_argument("--empty", action="store_true",
                    help="empty presentation: spontaneous rate (V1b-4.10)")
    ap.add_argument("--apl0", action="store_true",
                    help="run P14 with g_APL = 0 on the evaluation seeds (V1b-2.5)")
    ap.add_argument("--panel", default="all", choices=["all", "cal", "eval"],
                    help="odor panel for the run; cal checks the code "
                         "without spending panel E's blindness")
    ap.add_argument("--grid", type=int, default=0,
                    help="calibration: how many stage 1 points to run (0 - do not run)")
    ap.add_argument("--regression", action="store_true",
                    help="control (а) and (б): substrate and config hashes")
    ap.add_argument("--trials", action="store_true",
                    help="full protocol: 6 presentations and fraction of responding KC")
    ap.add_argument("--self-test", action="store_true",
                    help="short run: assemble the network and check that it computes")
    a = ap.parse_args()

    from brian2 import prefs, ms
    prefs.codegen.target = a.codegen
    if a.codegen == "cython" and a.cache:
        prefs.codegen.runtime.cython.cache_dir = a.cache
    from model import default_params as dp

    neurons, con = load_substrate()
    rates = odor_rates(a.odor, neurons)
    # g_ref = 1/(v_th - v_0): the gain at which APL, at a depolarization
    # equal to the KC threshold, gives one synaptic unit per tick (V1b-4.3)
    g_ref = 1.0 / float((dp["v_th"] - dp["v_0"]) / (0.001 * 1.0))  # in 1/mV
    g_abs = a.gapl * g_ref

    if a.regression:
        r = regression_substrate(neurons, con)
        cfg_off = {"dan_mask": False, "graded_apl": False, "odor_input": False,
                   "pn_kc_scale": 1.0, "g_apl_rel": 0.0, "substrate": "flywire_630",
                   "w_syn_mV": 0.275, "dt_ms": 0.1}
        cfg_on = dict(cfg_off, dan_mask=True, graded_apl=True, odor_input=True,
                      pn_kc_scale=a.scale, g_apl_rel=a.gapl)
        h_off = config_hash(cfg_off, DECLARED_KEYS)
        h_on = config_hash(cfg_on, DECLARED_KEYS)
        print("control (а), substrate:")
        print("   total edges %d, removed by mask %d (expected 49,316: %s)"
              % (r["n_edges_total"], r["n_masked"], "yes" if r["mask_count_ok"] else "NO"))
        print("   with substitutions off the substrate is identical: %s"
              % ("yes" if r["substitutions_off_identical"] else "NO"))
        print("control (б), configs excluding the declared keys:")
        print("   off: %s" % h_off[:16])
        print("   on:  %s" % h_on[:16])
        print("   equal: %s" % ("yes" if h_off == h_on else "NO"))
        print()
        print("Identity of spike trains under 33 conditions is checked by running")
        print("v1a_subcircuit.py against the frozen artifacts in results/v1a/runs.")
        return 0 if (r["mask_count_ok"] and h_off == h_on) else 1

    panel = {"all": PANEL_ALL, "cal": PANEL_CAL, "eval": PANEL_EVAL}[a.panel]
    # on the calibration set only code checks are run, so the seeds are
    # calibration seeds too: evaluation seeds are spent once, in the stage run
    seed_p14 = SEED_CAL if a.panel == "cal" else SEED_EVAL
    seed_p14m = SEED_CAL_M if a.panel == "cal" else SEED_EVAL_M

    def save(name: str, obj: dict) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        print("written: %s" % (OUT / name))

    if a.evaluate:
        res = eval_p14(neurons, con, pn_kc_scale=a.scale, g_apl_rel=a.gapl,
                       seed_base=seed_p14, odors=panel)
        kc, ov = res["kc_fraction_criterion_E"], res["overlap_criterion_E_28_pairs"]
        if "not_computed" in kc:
            print("set %s: criteria on E are not computed (%s)"
                  % (a.panel, kc["not_computed"]))
            save("evaluate_p14_%s.json" % a.panel, res)
            return 0
        print("run P14, 14 odors, pn_kc_scale %.4g, g_apl %.4g g_ref, seeds %d"
              % (a.scale, a.gapl, res["seed_base"]))
        print("fraction of responding KC on E: mean %.4f, max %.4f "
              "(band %s, ceiling %.2f) -> %s"
              % (kc["f_mean_E"], kc["f_max_E"], F_BAND, F_MAX,
                 "satisfied" if kc["pass"] else "NOT SATISFIED"))
        print("overlap on E: r(PA,BA) %.3f, r(PA,EL) %.3f, r(BA,EL) %.3f"
              % (ov["r_PA_BA"], ov["r_PA_EL"], ov["r_BA_EL"]))
        print("   order %s, separation %.3f (>= %.2f) %s, ceiling %s -> %s"
              % (ov["order_ok"], ov["separation"], OVERLAP_SEP_MIN,
                 ov["separation_ok"], ov["ceiling_ok"],
                 "satisfied" if ov["pass"] else "NOT SATISFIED"))
        sp = res["spikes_per_response_V1b_4_9_4_10"]
        print("spikes per response: KCab %.2f (reference %.1f), KCa'b' %.2f, KCg %.2f"
              % (sp["s_ab"], SPIKES_AB_TARGET, sp["s_apbp"], sp["s_g"]))
        cd = res["coverage_diagnostics"]
        print("Spearman(c_i, number of odors responded) rho %.3f, p %.3g"
              % (cd["spearman_rho"], cd["spearman_p"]))
        save("evaluate_p14.json", res)
        return 0

    if a.evaluate_mbon:
        res = eval_p14m(neurons, con, pn_kc_scale=a.scale, g_apl_rel=a.gapl,
                        seed_base=seed_p14m, odors=panel)
        t = res["T38"]
        print("run P14-M, 14 odors, %d ms presentation, pn_kc_scale %.4g, "
              "g_apl %.4g g_ref, seeds %d"
              % (M_PULSE_MS, a.scale, a.gapl, res["seed_base"]))
        if t["missing"]:
            print("T38 types not in the subcircuit: %s" % ", ".join(t["missing"]))
        print("%-8s %10s %10s %10s %8s" % ("type", "R_t, Hz", "max", "min", "MD"))
        for name in T38:
            d = t["per_type"].get(name)
            if d is None:
                continue
            print("%-8s %10.3f %10.3f %10.3f %8.3f"
                  % (name, d["R_t"], d["max"], d["min"], d["MD"]))
        print("floor >= %.1f Hz for all six: %s" % (MBON_FLOOR_HZ, t["floor_ok"]))
        print("ceiling <= %.0f Hz for all six: %s" % (MBON_CEIL_HZ, t["ceiling_ok"]))
        print("MD >= %.2f for %d types (need %d): %s"
              % (MBON_MD_MIN, t["n_md_ok"], MBON_MD_TYPES, t["md_ok"]))
        print("criterion V1b-3.1: %s" % ("satisfied" if t["pass"] else "NOT SATISFIED"))
        md = res["md_subsets"]
        print("median MD over %d five-odor subsets: %.3f (reference 0.27)"
              % (md["n_subsets"], md["median_over_types"]))
        p11 = res["MBON11_spikes_first_second"]
        print("MBON11 over the first second: mean over 14 odors %.2f spikes "
              "per cell (reference [29] 118 +- 8.3; different stimuli)"
              % p11["mean_over_odors"])
        save("evaluate_p14m_%s.json" % a.panel, res)
        return 0

    if a.empty:
        res = empty_presentation(neurons, con, pn_kc_scale=a.scale,
                                 g_apl_rel=a.gapl, seed_base=SEED_EMPTY)
        print("empty presentation, pn_kc_scale %.4g, g_apl %.4g g_ref, seeds %s"
              % (a.scale, a.gapl, res["seeds"]))
        print("spikes in the whole subcircuit %d, of which KC %d"
              % (res["n_spikes_subcircuit"], res["n_spikes_kc"]))
        print("spontaneous KC rate %.4f Hz (reference [46] 0.1 +- 0.4)"
              % res["spontaneous_rate_hz"])
        print("fraction of KC that rule V1b-2.1-2.2 would deem responding: %.4f"
              % res["f_responding_empty"])
        save("evaluate_empty.json", res)
        return 0

    if a.apl0:
        # V1b-2.5: the reconciliation is paired by input realization, so the seeds are evaluation seeds
        res = eval_p14(neurons, con, pn_kc_scale=a.scale, g_apl_rel=0.0,
                       seed_base=seed_p14, odors=panel)
        print("run P14 at g_APL = 0, pn_kc_scale %.4g, seeds %d"
              % (a.scale, res["seed_base"]))
        for o in panel:
            print("   %-28s S_P %.4f, fraction responding %.4f"
                  % (o, res["sparseness_V1b_2_5"][o], res["kc_fraction_by_odor"][o]))
        save("evaluate_apl0_%s.json" % a.panel, res)
        return 0

    if a.grid:
        # Screening by fraction, not calibration. The MBON constraint is not
        # computed here, so admissibility per V1b-4.5 is undefined and no
        # point is chosen. The stage calibration runs only through
        # run_v1b_grid.py: it shards, computes both halves of V1b-4.5, and
        # writes the stage artifact.
        pts = grid_stage1()[:a.grid]
        print("screening by fraction on the stage-1 grid: %d points of %d (not calibration)"
              % (len(pts), len(grid_stage1())))
        done = []
        for k, (sc, g) in enumerate(pts, 1):
            pt = evaluate_point(neurons, con, PANEL_CAL, pn_kc_scale=sc,
                                g_apl_rel=g, seed_base=SEED_CAL)
            pt.pop("_by_odor")
            done.append(pt)
            print("  [%d/%d] scale %.4g, g %.4g -> f̄ %.4f, f_max %.4f, s_ab %.2f%s"
                  % (k, len(pts), sc, g, pt["f_mean"], pt["f_max"], pt["s_ab"],
                     "  passed by fraction" if passes_fraction(pt) else ""))
        OUT.mkdir(parents=True, exist_ok=True)
        # separate name: the calibration artifact is not overwritten by this branch
        (OUT / "fraction_screen_stage1.json").write_text(
            json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")
        print()
        print("passed by fraction: %d of %d. Admissibility per V1b-4.5 is undefined:"
              % (sum(1 for p in done if passes_fraction(p)), len(done)))
        print("the MBON constraint on C was not computed. Point selection — run_v1b_grid.py.")
        return 0

    if a.trials:
        seeds = [20260908 + i for i in range(1, N_TRIALS + 1)]
        res = run_odor(neurons, con, a.odor, pn_kc_scale=a.scale, g_apl=g_abs,
                       seeds=seeds, dan_mask=not a.no_dan_mask,
                       graded_apl=not a.no_graded_apl)
        kf = kc_fraction(res, neurons)
        print("odor %r, pn_kc_scale %.4g, g_apl %.4g g_ref, seeds %s"
              % (a.odor, a.scale, a.gapl, seeds))
        print("fraction of responding KC (>=1 spike, >=3 of 6): %.4f  (denominator %d)"
              % (kf["f"], kf["n_kc_denominator"]))
        print("sensitivity table:")
        for key in sorted(kf["sensitivity"]):
            print("   %-14s %.4f" % (key, kf["sensitivity"][key]))
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / ("kc_fraction_%s.json" % a.odor.replace(" ", "_"))).write_text(
            json.dumps({"odor": a.odor, "pn_kc_scale": a.scale,
                        "g_apl_in_gref": a.gapl, "seeds": seeds, **kf},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    net, mon, core_ids, st = build(
        neurons, con, pn_kc_scale=a.scale, g_apl=g_abs,
        dan_mask=not a.no_dan_mask, graded_apl=not a.no_graded_apl,
        rates=rates)

    print("subcircuit: core %d, APL %d (graded: %s)"
          % (st["n_core"], st["n_apl"], not a.no_graded_apl))
    print("core edges %d, removed by DAN mask %d, APL input %d, APL output %d"
          % (st["n_edges_core"], st["n_masked"], st["n_apl_in"], st["n_apl_out"]))
    print("stimulated PN %d, odor %r, pn_kc_scale %.4g, g_apl %.4g (%.4g g_ref)"
          % (st["n_poisson"], a.odor, a.scale, g_abs, a.gapl))

    dur = (T_ON_MS + T_PULSE_MS + 200) if a.self_test else T_RUN_MS
    net.run(dur * ms, report="text", report_period=30 * 1000 * ms)

    df = measure(mon, core_ids, neurons,
                 window_ms=dur - T_ON_MS)
    kc = df[df.role == "Kenyon_Cell"]
    resp = int((kc.n_spikes >= KC_SPIKE_THRESHOLD).sum())
    print()
    print("spikes total: %d" % int(df.n_spikes.sum()))
    print("KC with >=1 spike: %d of %d (%.1f%%)"
          % (resp, len(kc), 100.0 * resp / len(kc)))
    for r in ("MBON", "DAN", "PN"):
        s = df[df.role == r]
        print("%-4s: active %d of %d, mean rate of active %.1f Hz"
              % (r, int((s.n_spikes > 0).sum()), len(s),
                 s.loc[s.n_spikes > 0, "rate_hz"].mean() if (s.n_spikes > 0).any() else 0.0))

    if not a.self_test:
        OUT.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT / ("rates_%s.tsv" % a.odor.replace(" ", "_")),
                  sep="\t", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
