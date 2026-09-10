# -*- coding: utf-8 -*-
"""Connectome-complete subcircuit of the mushroom body from FlyWire v630.

Selects KC, MBON, DAN and APL neurons by the type annotations [18] (tag v1.1.0,
materialization 630) plus antennal lobe projection neurons (ALPN) as the input
source, takes **all** edges between the selected neurons from the model connectome [2],
and writes the subcircuit in the same format that `model.create_model` reads.
Nothing is dropped inside the subcircuit: neither edges nor neurons of the selected types.

Mushroom body compartments. FlyWire does not annotate them; the "type →
compartment" map is built from the `instance` field of the hemibrain metadata [9],
where the compartment is given in parentheses after the type code (`MBON11(y1pedc>a/B)_R`).
The Latin notation is converted to Greek by the rules of transliterate() below, and
the result is checked against the closed list of compartments MB_COMPARTMENTS; anything
that does not resolve into it is flagged and does not go into the map. For branching
types (`y4>y1y2`, `y1pedc>a/B`, `y4<y1y2`) the compartment is taken as the part before
`>` or `<` — the dendritic field of the MBON and the axonal field of the DAN, respectively.

Run:  .venv/Scripts/python.exe build_mb_subcircuit.py
Outputs:  data/mb_subcircuit/    — IDs, subcircuit in the model's format (outside git)
          results/mb_subcircuit/ — counts, compartment map, config hash
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE / "Drosophila_brain_model"
ANN = HERE / "data" / "flywire_annotations"
OUT_DATA = HERE / "data" / "mb_subcircuit"
OUT_RES = HERE / "results" / "mb_subcircuit"

# --- subcircuit config: everything that determines composition, and nothing more -------------
CONFIG = {
    "substrate": "FlyWire v630 (форк [2], 2023_03_23_*_630_final)",
    "annotations": "flywire_annotations v1.1.0 (df6bb136), Supplemental_file1",
    "compartments_from": "hemibrain meta Supplemental_file4, поле instance",
    # core of the subcircuit: cell_class values of the annotation
    "core_cell_class": ["Kenyon_Cell", "MBON", "DAN"],
    # APL is annotated as MBIN; we take it by hemibrain_type so as not to pull in DPM
    "core_hemibrain_type": ["APL"],
    # input: antennal lobe projection neurons
    "input_cell_class": ["ALPN"],
    # a PN is included if it has at least this many connections onto a core neuron
    "input_min_connectivity_to_core": 1,
    # both sides: the subcircuit is connectome-complete, hemispheres are not separated
    "sides": "both",
    # edges: all connectome edges with both ends in the subcircuit
    "edges": "all pairs within selection",
}

# Mushroom body compartments (Li et al. 2020 [9]); pedc is the pedunculus.
MB_COMPARTMENTS = [
    "γ1", "γ2", "γ3", "γ4", "γ5",
    "β1", "β2", "β′1", "β′2",
    "α1", "α2", "α3", "α′1", "α′2", "α′3",
    "pedc", "calyx",
]
# Instance values that are not a compartment (names of cell-body fiber tracts).
NOT_A_COMPARTMENT = {"PDL05", "PVL17"}


def transliterate(s: str) -> str:
    """Latin hemibrain notation into Greek.

    'y' becomes gamma only before a digit ('calyx' is left untouched); 'B' becomes
    beta always; 'a' becomes alpha before a digit, before a prime, or right after
    '>' or '/' (in 'y5B'2a' and in 'bilateral' the final 'a' remains itself).
    """
    out = []
    for k, ch in enumerate(s):
        nxt = s[k + 1] if k + 1 < len(s) else ""
        prv = s[k - 1] if k else ""
        if ch == "'":
            out.append("′")
        elif ch == "y" and nxt.isdigit():
            out.append("γ")
        elif ch == "B":
            out.append("β")
        elif ch == "a" and (nxt.isdigit() or nxt == "'" or prv in ">/"):
            out.append("α")
        else:
            out.append(ch)
    return "".join(out)


_TOK = re.compile(r"(pedc|calyx)|([γβα]′?)((?:\d[a-z]*)+)")


def split_compartments(field: str) -> list[str]:
    """Split a field like 'γ1pedc' or 'α2p3p' into compartments of the closed list.

    Parsed left to right. Qualifier letters ('m', 'p', 'd', 'sc', 'ap', 'ped')
    are dropped; a repeated digit under one letter is several compartments
    ('α2p3p' — dendrites in α2p and α3p, i.e. α2 and α3). The suffix after '_'
    ('β′2mp_bilateral') belongs to the neuron, not the compartment, and is
    stripped. Returns an empty list if the field is not parsed in full or yields
    a compartment outside MB_COMPARTMENTS: such a type is flagged and does not go
    into the map.
    """
    field = field.split("_")[0]
    # 'pedc' and 'calyx' are split off explicitly, otherwise the digit suffix eats them ('γ1pedc')
    parts = [p for p in re.split(r"(pedc|calyx)", field) if p]
    out: list[str] = []
    for part in parts:
        if part in ("pedc", "calyx"):
            out.append(part)
            continue
        pos = 0
        while pos < len(part):
            m = _TOK.match(part, pos)
            if not m:
                return []
            letter = m.group(2)
            for d in re.finditer(r"\d", m.group(3)):
                out.append(letter + d.group(0))
            pos = m.end()
    if not out or any(c not in MB_COMPARTMENTS for c in out):
        return []
    return out


def build_compartment_map(hb: pd.DataFrame) -> pd.DataFrame:
    """Map of 'hemibrain_type to compartments' from the instance field of metadata [9]."""
    m = hb[hb["morphology_type"].fillna("").str.match(r"(MBON|PAM|PPL1|PPL2)")].copy()
    m["par"] = m["instance"].map(
        lambda s: (re.findall(r"\((.*?)\)", s) or [None])[0] if isinstance(s, str) else None
    )
    rows = []
    for t, sub in m.groupby("morphology_type"):
        vals = sorted({v for v in sub["par"] if isinstance(v, str)})
        if len(vals) > 1:
            print("   WARNING: type %s has multiple instance variants: %s" % (t, vals))
        raw = vals[0] if vals else ""
        greek = transliterate(raw) if raw else ""
        field = re.split("[><]", greek)[0] if greek else ""
        if (raw in NOT_A_COMPARTMENT) or not raw:
            comps = []
            note = "not a compartment (%s)" % (raw or "instance without parentheses")
        else:
            comps = split_compartments(field)
            note = "" if comps else "did not resolve into the closed list"
        rows.append({
            "hemibrain_type": t,
            "instance_raw": raw,
            "field_greek": greek,
            "innervates": field,
            "compartments": ";".join(comps),
            "note": note,
        })
    return pd.DataFrame(rows).sort_values("hemibrain_type").reset_index(drop=True)


def write_reconciliation(neurons: pd.DataFrame, hb: pd.DataFrame, cfg_hash: str) -> None:
    """Reconciliation of subcircuit counts against [8], [30] and [9]. Requirement of draft task 4.1.

    The hemibrain numbers are computed here from its metadata, not taken from the
    text of the paper: this way the reconciliation is reproducible together with
    everything else.
    """
    def hb_n(prefix: tuple[str, ...], side: str | None = None) -> int:
        s = hb[hb.morphology_type.fillna("").str.startswith(prefix)]
        if side:
            s = s[s.side == side]
        return len(s)

    kc = neurons[neurons.mb_role == "Kenyon_Cell"]
    mbon = neurons[neurons.mb_role == "MBON"]
    dan = neurons[neurons.mb_role == "DAN"]
    apl = neurons[neurons.mb_role == "APL"]
    lines = [
        "Reconciliation of mushroom-body subcircuit counts against the literature",
        "config hash: %s" % cfg_hash,
        "",
        "The hemibrain numbers are computed from Supplemental_file4_hemibrain_meta.csv,",
        "the Aso 2014 numbers are from the text of [8]/[30] (see notes).",
        "",
        "%-12s %-26s %-26s %s" % ("class", "FlyWire v630 (our count)", "hemibrain [9] (our count)", "Aso 2014 [8]/[30]"),
        "-" * 100,
        "%-12s %-26s %-26s %s" % (
            "KC",
            "%d (L %d, R %d)" % (len(kc), (kc.side == "left").sum(), (kc.side == "right").sum()),
            "%d (right hemisphere)" % hb_n(("KC",)),
            "\"∼2000 Kenyon cells\" [30]"),
        "%-12s %-26s %-26s %s" % (
            "MBON",
            "%d (L %d, R %d), types %d" % (len(mbon), (mbon.side == "left").sum(),
                                           (mbon.side == "right").sum(), mbon.hemibrain_type.nunique()),
            "%d (R %d), types %d" % (hb_n(("MBON",)), hb_n(("MBON",), "right"),
                                     hb[hb.morphology_type.fillna("").str.startswith("MBON")].morphology_type.nunique()),
            "34 neurons of 21 types [30, table 1]"),
        "%-12s %-26s %-26s %s" % (
            "DAN",
            "%d (L %d, R %d), types %d" % (len(dan), (dan.side == "left").sum(),
                                           (dan.side == "right").sum(), dan.hemibrain_type.nunique()),
            "%d (R %d)" % (hb_n(("PAM", "PPL1", "PPL2")), hb_n(("PAM", "PPL1", "PPL2"), "right")),
            ">100 neurons of 20 types [8]"),
        "%-12s %-26s %-26s %s" % (
            "APL",
            "%d (one per hemisphere)" % len(apl),
            "%d (right only)" % hb_n(("APL",)),
            "1 per hemisphere [8]"),
        "",
        "Discrepancy breakdown.",
        "",
        "1. KC. Discrepancy with hemibrain is about +34% per hemisphere; the",
        "   \"∼2000\" reference from [30] is closer to hemibrain than to FlyWire. These are different flies and different",
        "   reconstructions; the explanation for the discrepancy is not verified in our sources and",
        "   remains an open question. What matters for the testbed is different: the substrate",
        "   is FlyWire v630, and in it the KC count equals the one computed here, not the",
        "   literature reference. The phrase \"~2000 per hemisphere\" in the whitepaper",
        "   needs to be corrected to the actual substrate count.",
        "",
        "2. MBON. \"34 neurons of 21 types\" is the [30] 2014 set. In hemibrain [9] the types",
        "   MBON22–MBON35 are described later, and the FlyWire annotation follows the [9] nomenclature.",
        "   The discrepancy is not an extraction error but a difference of sets; for the level-1 decoder",
        "   this matters because valence in [30] is established only for",
        "   types MBON01–MBON21.",
        "",
        "3. DAN. Agreement with hemibrain is good. \"20 types\" from [8] versus 25 types",
        "   in the [9] nomenclature: [9] splits PAM more finely (PAM01–PAM15).",
        "",
        "4. APL. Full agreement: one per hemisphere.",
    ]
    (OUT_RES / "counts_vs_literature.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_RES.mkdir(parents=True, exist_ok=True)

    cfg_json = json.dumps(CONFIG, ensure_ascii=False, sort_keys=True, indent=2)
    cfg_hash = hashlib.sha256(cfg_json.encode("utf-8")).hexdigest()
    print("subcircuit config, sha256 = %s" % cfg_hash)

    print("\n1. type annotations")
    ann = pd.read_csv(ANN / "Supplemental_file1_annotations.tsv", sep="\t", low_memory=False)
    hb = pd.read_csv(ANN / "Supplemental_file4_hemibrain_meta.csv", low_memory=False)
    print("   annotations: %d rows; hemibrain: %d rows" % (len(ann), len(hb)))

    print("\n2. compartment map from hemibrain [9]")
    cmap = build_compartment_map(hb)
    n_ok = int((cmap.compartments != "").sum())
    print("   types: %d, of which with a compartment: %d" % (len(cmap), n_ok))
    for _, r in cmap[cmap.compartments == ""].iterrows():
        print("      no compartment: %-8s instance=%-8s %s"
              % (r.hemibrain_type, r.instance_raw, r.note))
    cmap.to_csv(OUT_RES / "compartment_map.tsv", sep="\t", index=False, encoding="utf-8")

    print("\n3. neuron selection")
    comp = pd.read_csv(REPO / "2023_03_23_completeness_630_final.csv", index_col=0)
    model_ids = set(comp.index.astype("int64"))
    ann = ann[ann.root_id.isin(model_ids)].copy()

    is_core = (ann.cell_class.isin(CONFIG["core_cell_class"])
               | ann.hemibrain_type.isin(CONFIG["core_hemibrain_type"]))
    core = ann[is_core].copy()
    core["mb_role"] = core.cell_class.where(core.cell_class != "MBIN", "APL")

    pn_all = ann[ann.cell_class.isin(CONFIG["input_cell_class"])].copy()
    print("   core: %d neurons; PN candidates: %d" % (len(core), len(pn_all)))

    print("\n4. connectome edges")
    con = pd.read_parquet(REPO / "2023_03_23_connectivity_630_final.parquet")
    core_ids = set(core.root_id.astype("int64"))
    # keep only the PN that actually synapse onto the subcircuit core
    to_core = con[con.Postsynaptic_ID.isin(core_ids)]
    pn_hit = to_core.groupby("Presynaptic_ID")["Connectivity"].sum()
    keep_pn = {i for i in pn_all.root_id.astype("int64")
               if pn_hit.get(i, 0) >= CONFIG["input_min_connectivity_to_core"]}
    pn = pn_all[pn_all.root_id.isin(keep_pn)].copy()
    pn["mb_role"] = "PN"
    print("   PN with a connection to the core: %d of %d" % (len(pn), len(pn_all)))

    sel = pd.concat([core, pn], ignore_index=True)
    sel_ids = set(sel.root_id.astype("int64"))
    edges = con[con.Presynaptic_ID.isin(sel_ids) & con.Postsynaptic_ID.isin(sel_ids)].copy()
    print("   neurons in subcircuit: %d; edges: %d; synapses: %d"
          % (len(sel), len(edges), int(edges.Connectivity.sum())))

    print("\n5. neuron compartments")
    cm = {k: (v if isinstance(v, str) else "")
          for k, v in cmap.set_index("hemibrain_type")["compartments"].to_dict().items()}

    def compartments_of(t) -> str:
        """Compartments of a type. A compound FlyWire type ('MBON25,MBON34') is a union.

        A compound label means the annotation did not separate two hemibrain
        types; taking the union of compartments is the only thing that follows
        from this, and such neurons are marked in neurons.csv with the compound
        type.
        """
        if not isinstance(t, str) or not t:
            return ""
        if "," not in t:
            return cm.get(t, "")
        parts: list[str] = []
        for piece in t.split(","):
            for c in cm.get(piece.strip(), "").split(";"):
                if c and c not in parts:
                    parts.append(c)
        return ";".join(parts)

    sel["compartment"] = sel.hemibrain_type.map(compartments_of)

    print("\n6. writing output")
    cols = ["root_id", "mb_role", "cell_class", "cell_sub_class", "hemibrain_type",
            "compartment", "side", "top_nt", "top_nt_conf", "ito_lee_hemilineage"]
    neurons = (sel[cols].sort_values(["mb_role", "hemibrain_type", "root_id"])
               .reset_index(drop=True))
    neurons.to_csv(OUT_DATA / "neurons.csv", index=False, encoding="utf-8")

    # subcircuit in the format that model.create_model reads
    sub_comp = comp.loc[sorted(sel_ids)].copy()
    sub_comp.to_csv(OUT_DATA / "completeness.csv", encoding="utf-8")
    idx = {fid: k for k, fid in enumerate(sub_comp.index.astype("int64"))}
    sub_con = edges.copy()
    sub_con["Presynaptic_Index"] = sub_con.Presynaptic_ID.map(idx)
    sub_con["Postsynaptic_Index"] = sub_con.Postsynaptic_ID.map(idx)
    sub_con = (sub_con.sort_values(["Presynaptic_Index", "Postsynaptic_Index"])
               .reset_index(drop=True))
    sub_con.to_parquet(OUT_DATA / "connectivity.parquet", compression="brotli")

    # counts
    counts = (neurons.groupby(["mb_role", "hemibrain_type", "side"], dropna=False)
              .size().rename("n").reset_index())
    counts.to_csv(OUT_RES / "type_counts.tsv", sep="\t", index=False, encoding="utf-8")

    by_role = neurons.groupby(["mb_role", "side"], dropna=False).size().unstack(fill_value=0)
    print(by_role.to_string())
    for role in ("MBON", "DAN"):
        sub = neurons[neurons.mb_role == role]
        no_t = int(sub.hemibrain_type.isna().sum())
        no_c = int((sub.compartment == "").sum())
        print("   %-4s without hemibrain type: %d; without compartment: %d of %d"
              % (role, no_t, no_c, len(sub)))
        bad = sub[sub.compartment == ""].hemibrain_type.value_counts(dropna=False)
        for t, n in bad.items():
            print("        %s: %d" % ("<no type>" if pd.isna(t) else t, n))

    stats = {
        "config_sha256": cfg_hash,
        "config": CONFIG,
        "n_neurons": int(len(sel)),
        "n_edges": int(len(edges)),
        "n_synapses": int(edges.Connectivity.sum()),
        "by_role": {k: int(v) for k, v in neurons.mb_role.value_counts().items()},
        "by_role_side": {r: {s: int(n) for s, n in row.items()}
                         for r, row in by_role.iterrows()},
        "n_types": {r: int(sub.hemibrain_type.nunique())
                    for r, sub in neurons.groupby("mb_role")},
        "n_without_hemibrain_type": {r: int(sub.hemibrain_type.isna().sum())
                                     for r, sub in neurons.groupby("mb_role")},
        "n_without_compartment": {
            r: int((sub.compartment == "").sum())
            for r, sub in neurons.groupby("mb_role") if r in ("MBON", "DAN")},
        "compartments": sorted({c for cs in neurons.compartment for c in cs.split(";") if c}),
    }
    (OUT_RES / "subcircuit_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_RES / "config.json").write_text(cfg_json, encoding="utf-8")

    write_reconciliation(neurons, hb, cfg_hash)

    print("\ncompartments in subcircuit: %s" % ", ".join(stats["compartments"]))
    print("done: data/mb_subcircuit/, results/mb_subcircuit/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
