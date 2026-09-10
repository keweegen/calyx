# -*- coding: utf-8 -*-
"""Download the Hallem & Carlson 2006 odorant data and build the PN input for V1b.

Source: the `DoOR.data` package (rOpenSci), https://github.com/ropensci/DoOR.data,
branch master. It reproduces the published matrices of individual studies
as CSV; the Hallem & Carlson 2006 [39] data live in the `Hallem.2006.EN` column
of the per-receptor files, the first row (`CAS == "SFR"`) is spontaneous activity.
Study metadata is in `door_dataset_info.csv`: electrophysiology, units of
"spikes", spontaneous activity subtracted, concentration 10^-2, solvent
paraffin oil (16 water-soluble compounds — in water).

Why Hallem and not the DoOR consensus matrix. Decided at V1b pre-registration.
The consensus covers more glomeruli in the union over the database (47 vs. 24, 89%
of PN->KC synapses vs. 45%), but its coverage there **depends on the odorant** (45-80%
across the panel), while the intersection of measured glomeruli across all odorants
of the panel equals exactly the Hallem set. So the entire gain from the consensus
consists of odor-dependent coverage, and that introduces an artifact into the
ensemble-overlap criterion, biased toward passing the stage. The constant Hallem
mask introduces an artifact of known sign, biased toward failure. The latter was
chosen. The consensus remains a sensitivity branch; it is reported and does not
change the verdict.

ORN -> PN transformation: the divisive normalization of Olsen, Bhandawat & Wilson 2010
[40], equations 2, 4 and 5:
    PN_g = R_max * ORN_g^n / (ORN_g^n + sigma^n + s^n),   s = m * (sum_g ORN_g) / 190
Constants from the paper, mode A: R_max = 165 spikes/s, sigma = 12 spikes/s, n = 1.5,
m = 10.63 (glomerulus VM7). Negative input (inhibition below the spontaneous
level) gives zero PN response — a rule from the paper itself; in the model [2] the
baseline rate is zero, so no other representation is possible.

Stage panel: odorants present simultaneously in Hallem 2006 and in the panels
of Honegger 2011 [25] and Campbell 2013 [41] — the works the V1b criterion numbers
are taken from. Verified: of the 17 chemically defined compounds in these
panels, 14 are in Hallem (missing 3-octanol, 4-methylcyclohexanol and
1-hepten-3-ol; the last is absent from every DoOR dataset).

DoOR.data license — MIT; the original Hallem & Carlson 2006 data — Cell,
cited by DOI 10.1016/j.cell.2006.01.050. Data are not committed to git (DATA.md).

Run:  python scripts/fetch_odor_panel.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data" / "odor_panel"
OUT = HERE / "results" / "odor_panel"
SUB = HERE / "data" / "mb_subcircuit"

RAW = "https://raw.githubusercontent.com/ropensci/DoOR.data/master/data/"
API = "https://api.github.com/repos/ropensci/DoOR.data/contents/data"

HC = "Hallem.2006.EN"

# Olsen 2010, mode A
R_MAX, SIGMA, EXPONENT, M_COEF, LFP_DIV = 165.0, 12.0, 1.5, 10.63, 190.0

# panel: Hallem 2006 ∩ (Honegger 2011 ∪ Campbell 2013), by CAS
PANEL_CAS = {
    "110-43-0": "2-heptanone",
    "110-93-0": "6-methyl-5-hepten-2-one",
    "6753-98-6": "alpha-humulene",
    "100-52-7": "benzaldehyde",
    "97-64-3": "ethyl lactate",
    "106-32-1": "ethyl octanoate",
    "66-25-1": "hexanal",
    "123-92-2": "isopentyl acetate",
    "111-11-5": "methyl octanoate",
    "123-25-1": "diethyl succinate",
    "110-62-3": "pentanal",
    "628-63-7": "pentyl acetate",
    "123-86-4": "butyl acetate",
    "3391-86-4": "1-octen-3-ol",
}


def fetch(name: str) -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    p = DATA / name
    if not p.exists():
        urllib.request.urlretrieve(RAW + name, p)
    return p


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(fetch(name), sep=";")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def expand(gl: str) -> set[str]:
    """hemibrain splits VC3 into VC3l/VC3m; multiglomerular entries are not distributed."""
    if gl == "VC3":
        return {"VC3l", "VC3m"}
    return set() if "+" in gl else {gl}


def olsen(orn: pd.Series) -> pd.Series:
    """ORN (spikes/s, evoked response) -> PN (spikes/s). Olsen 2010, eq. 2, 4, 5."""
    r = orn.clip(lower=0.0)
    s = M_COEF * r.sum() / LFP_DIV
    num = r ** EXPONENT
    return R_MAX * num / (num + SIGMA ** EXPONENT + s ** EXPONENT)


def main() -> int:
    names = [x["name"] for x in json.load(urllib.request.urlopen(API))
             if x["name"].endswith(".csv")]
    receptors = [n for n in names
                 if not n.startswith("door") and n not in ("ORs.csv", "odor.csv")]

    evoked, sfr = {}, {}
    for f in receptors:
        df = read(f)
        if HC not in df.columns:
            continue
        is_sfr = df["CAS"].astype(str).str.upper().eq("SFR")
        v = df.loc[is_sfr, HC]
        sfr[f[:-4]] = float(v.iloc[0]) if len(v) and pd.notna(v.iloc[0]) else 0.0
        body = df.loc[~is_sfr, ["InChIKey", HC]].dropna()
        evoked[f[:-4]] = body.set_index("InChIKey")[HC]

    mat = pd.DataFrame(evoked)
    if mat.shape != (110, 24):
        print("Hallem matrix is not 110x24, but %dx%d — the source has changed" % mat.shape,
              file=sys.stderr)
        return 1

    mapping = read("door_mappings.csv")[["receptor", "glomerulus"]].dropna().astype(str)
    mapping = mapping[(mapping.glomerulus != "?") & (mapping.receptor != "?")]
    rec2gl = dict(zip(mapping.receptor, mapping.glomerulus))

    neurons = pd.read_csv(SUB / "neurons.csv")
    neurons = neurons.assign(gl=neurons.hemibrain_type.astype(str).str.split("_").str[0])
    # Both hemispheres. A glomerulus's response is assigned to all of its uPN
    # regardless of side: the "receptor -> glomerulus" mapping does not distinguish
    # side, and an odor normally excites both antennal lobes. Unilateral stimulation
    # would make half the components of the overlap vector (section 3д) agree on
    # zero and would raise the correlation for every pair at once.
    upn = neurons[(neurons.mb_role == "PN")
                  & (neurons.cell_sub_class == "uniglomerular")]

    # "glomerulus -> receptor" mapping: VC3 has two hemibrain glomeruli (VC3l, VC3m)
    # for one receptor Or35a, and both get its response. The reverse mapping here
    # would be wrong: a "receptor -> glomerulus" dict would keep an arbitrary one.
    rec_of_gl: dict[str, str] = {}
    for r in mat.columns:
        for gl in sorted(expand(rec2gl.get(r, ""))):
            if gl in set(upn.gl):
                rec_of_gl[gl] = r
    mask_gl = sorted(rec_of_gl)

    odor = read("odor.csv")
    odor["CAS"] = odor["CAS"].astype(str).str.strip()
    key_by_cas = dict(zip(odor.CAS, odor.InChIKey))

    rows = {}
    for cas, name in PANEL_CAS.items():
        key = key_by_cas.get(cas)
        if key not in mat.index:
            print("odorant %s (%s) is absent from the Hallem matrix" % (name, cas),
                  file=sys.stderr)
            return 1
        # response over the mask glomeruli; the normalization sum is over all 24 receptors
        orn_all = mat.loc[key]
        pn_all = olsen(orn_all)
        rows[name] = {gl: pn_all[rec_of_gl[gl]] for gl in mask_gl}
    pn = pd.DataFrame(rows).T.reindex(columns=mask_gl)

    # input coverage
    con = pd.read_parquet(SUB / "connectivity.parquet")
    kc = set(neurons.loc[neurons.mb_role == "Kenyon_Cell", "root_id"])
    e = con[con.Postsynaptic_ID.isin(kc) & con.Presynaptic_ID.isin(set(upn.root_id))]
    e = e.assign(gl=e.Presynaptic_ID.map(dict(zip(upn.root_id, upn.gl))))
    syn_total = int(e["Connectivity"].sum())
    syn_mask = int(e.loc[e.gl.isin(mask_gl), "Connectivity"].sum())

    per_kc = e.groupby("Postsynaptic_ID")["Connectivity"].sum()
    per_kc_mask = (e[e.gl.isin(mask_gl)].groupby("Postsynaptic_ID")["Connectivity"]
                   .sum().reindex(per_kc.index).fillna(0))
    c_i = per_kc_mask / per_kc

    OUT.mkdir(parents=True, exist_ok=True)
    pn.to_csv(OUT / "pn_rates_by_odor.tsv", sep="\t", float_format="%.4f")
    c_i.rename("c_i").to_csv(OUT / "kc_input_coverage.tsv", sep="\t", float_format="%.4f")

    stats = {
        "source": "Hallem & Carlson 2006 via DoOR.data (ropensci), column %s" % HC,
        "matrix": {"odorants": int(mat.shape[0]), "receptors": int(mat.shape[1]),
                   "missing_cells": int(mat.isna().sum().sum())},
        "olsen_2010": {"R_max": R_MAX, "sigma": SIGMA, "exponent": EXPONENT,
                       "m": M_COEF, "lfp_divisor": LFP_DIV},
        "panel": {"n_odorants": int(pn.shape[0]), "n_pairs": int(pn.shape[0] * (pn.shape[0] - 1) / 2)},
        "mask": {
            "glomeruli": len(mask_gl),
            "glomeruli_in_subcircuit": int(upn.gl.nunique()),
            "upn_covered": int(upn.gl.isin(mask_gl).sum()),
            "upn_total": int(len(upn)),
            "synapses_covered": syn_mask,
            "synapses_total": syn_total,
            "synapse_fraction": round(syn_mask / syn_total, 4),
        },
        "kc_input_coverage": {
            "median": round(float(c_i.median()), 4),
            "q25": round(float(c_i.quantile(0.25)), 4),
            "q75": round(float(c_i.quantile(0.75)), 4),
            "below_0.1": int((c_i < 0.1).sum()),
            "n_kc": int(len(c_i)),
        },
        "checksums": {p.name: sha256(p) for p in sorted(DATA.glob("*.csv"))},
    }
    (OUT / "panel_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Hallem matrix: %d odorants x %d receptors, %d missing"
          % (mat.shape[0], mat.shape[1], int(mat.isna().sum().sum())))
    print("panel: %d odorants, %d pairs" % (pn.shape[0], pn.shape[0] * (pn.shape[0] - 1) // 2))
    print("mask: %d glomeruli out of %d, %d uPN out of %d, %.0f %% of PN->KC synapses"
          % (len(mask_gl), upn.gl.nunique(), int(upn.gl.isin(mask_gl).sum()),
             len(upn), 100 * syn_mask / syn_total))
    print("KC input coverage: median %.2f, quartiles %.2f/%.2f, below 0.1 for %d of %d"
          % (c_i.median(), c_i.quantile(.25), c_i.quantile(.75),
             (c_i < .1).sum(), len(c_i)))
    print("PN rates: %.1f-%.1f spikes/s" % (pn.min().min(), pn.max().max()))
    print("written: %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
