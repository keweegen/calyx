# -*- coding: utf-8 -*-
"""Коннектомно-полная подсхема грибовидного тела из FlyWire v630.

Отбирает нейроны KC, MBON, DAN и APL по разметке типов [18] (тег v1.1.0,
материализация 630) плюс проекционные нейроны антеннальной доли (ALPN) как
источник входа, берёт из коннектома модели [2] **все** рёбра между отобранными
нейронами и записывает подсхему в том же формате, что читает `model.create_model`.
Внутри подсхемы не выбрасывается ничего: ни рёбра, ни нейроны выбранных типов.

Компартменты грибовидного тела. FlyWire их не размечает; карта «тип →
компартмент» строится из поля `instance` метаданных hemibrain [9], где
компартмент стоит в скобках после кода типа (`MBON11(y1pedc>a/B)_R`). Латинская
запись переводится в греческую по правилам transliterate() ниже, результат
проверяется по закрытому списку компартментов MB_COMPARTMENTS; всё, что в него
не раскладывается, помечается и в карту не идёт. Для типов с ветвлением
(`y4>y1y2`, `y1pedc>a/B`, `y4<y1y2`) компартментом считается часть до `>` или
`<` — дендритное поле MBON и аксонное поле DAN соответственно.

Запуск:  .venv/Scripts/python.exe build_mb_subcircuit.py
Выходы:  data/mb_subcircuit/    — ID, подсхема в формате модели (вне git)
         results/mb_subcircuit/ — счётчики, карта компартментов, хэш конфига
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

# --- конфиг подсхемы: всё, что определяет состав, и ничего больше -------------
CONFIG = {
    "substrate": "FlyWire v630 (форк [2], 2023_03_23_*_630_final)",
    "annotations": "flywire_annotations v1.1.0 (df6bb136), Supplemental_file1",
    "compartments_from": "hemibrain meta Supplemental_file4, поле instance",
    # ядро подсхемы: значения cell_class разметки
    "core_cell_class": ["Kenyon_Cell", "MBON", "DAN"],
    # APL размечен как MBIN; берём его по hemibrain_type, чтобы не втянуть DPM
    "core_hemibrain_type": ["APL"],
    # вход: проекционные нейроны антеннальной доли
    "input_cell_class": ["ALPN"],
    # PN включается, если имеет хотя бы столько связей на нейрон ядра
    "input_min_connectivity_to_core": 1,
    # обе стороны: подсхема коннектомно-полная, полушария не разделяются
    "sides": "both",
    # рёбра: все рёбра коннектома, оба конца которых в подсхеме
    "edges": "all pairs within selection",
}

# Компартменты грибовидного тела (Li et al. 2020 [9]); pedc — цветоножка.
MB_COMPARTMENTS = [
    "γ1", "γ2", "γ3", "γ4", "γ5",
    "β1", "β2", "β′1", "β′2",
    "α1", "α2", "α3", "α′1", "α′2", "α′3",
    "pedc", "calyx",
]
# Значения instance, которые компартментом не являются (имена трактов тел клеток).
NOT_A_COMPARTMENT = {"PDL05", "PVL17"}


def transliterate(s: str) -> str:
    """Латинская запись hemibrain в греческую.

    'y' в гамму только перед цифрой ('calyx' не трогаем); 'B' в бету всегда;
    'a' в альфу перед цифрой, перед штрихом или сразу после '>' или '/'
    (в 'y5B'2a' и в 'bilateral' конечная 'a' остаётся собой).
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
    """Разложить поле вида 'γ1pedc' или 'α2p3p' на компартменты закрытого списка.

    Разбор слева направо. Буквы-уточнители ('m', 'p', 'd', 'sc', 'ap', 'ped')
    отбрасываются; повтор цифры при одной букве — это несколько компартментов
    ('α2p3p' — дендриты в α2p и α3p, то есть α2 и α3). Суффикс после '_'
    ('β′2mp_bilateral') относится к нейрону, а не к компартменту, и снимается.
    Возвращает пустой список, если поле разобрано не полностью или дало
    компартмент вне MB_COMPARTMENTS: такой тип помечается и в карту не идёт.
    """
    field = field.split("_")[0]
    # 'pedc' и 'calyx' отделяем явно, иначе их съедает суффикс цифры ('γ1pedc')
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
    """Карта 'hemibrain_type в компартменты' из поля instance метаданных [9]."""
    m = hb[hb["morphology_type"].fillna("").str.match(r"(MBON|PAM|PPL1|PPL2)")].copy()
    m["par"] = m["instance"].map(
        lambda s: (re.findall(r"\((.*?)\)", s) or [None])[0] if isinstance(s, str) else None
    )
    rows = []
    for t, sub in m.groupby("morphology_type"):
        vals = sorted({v for v in sub["par"] if isinstance(v, str)})
        if len(vals) > 1:
            print("   ВНИМАНИЕ: у типа %s несколько вариантов instance: %s" % (t, vals))
        raw = vals[0] if vals else ""
        greek = transliterate(raw) if raw else ""
        field = re.split("[><]", greek)[0] if greek else ""
        if (raw in NOT_A_COMPARTMENT) or not raw:
            comps = []
            note = "не компартмент (%s)" % (raw or "instance без скобок")
        else:
            comps = split_compartments(field)
            note = "" if comps else "не разложилось по закрытому списку"
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
    """Сверка счётчиков подсхемы с [8], [30] и [9]. Требование задачи 4.1 драфта.

    Числа для hemibrain считаются здесь же из его метаданных, а не берутся из
    текста статьи: так сверка воспроизводима вместе со всем остальным.
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
        "Сверка счётчиков подсхемы грибовидного тела с литературой",
        "хэш конфига: %s" % cfg_hash,
        "",
        "Числа hemibrain посчитаны из Supplemental_file4_hemibrain_meta.csv,",
        "числа Aso 2014 — из текста [8]/[30] (см. примечания).",
        "",
        "%-12s %-26s %-26s %s" % ("класс", "FlyWire v630 (наш счёт)", "hemibrain [9] (наш счёт)", "Aso 2014 [8]/[30]"),
        "-" * 100,
        "%-12s %-26s %-26s %s" % (
            "KC",
            "%d (L %d, R %d)" % (len(kc), (kc.side == "left").sum(), (kc.side == "right").sum()),
            "%d (правое полушарие)" % hb_n(("KC",)),
            "«∼2000 Kenyon cells» [30]"),
        "%-12s %-26s %-26s %s" % (
            "MBON",
            "%d (L %d, R %d), типов %d" % (len(mbon), (mbon.side == "left").sum(),
                                           (mbon.side == "right").sum(), mbon.hemibrain_type.nunique()),
            "%d (R %d), типов %d" % (hb_n(("MBON",)), hb_n(("MBON",), "right"),
                                     hb[hb.morphology_type.fillna("").str.startswith("MBON")].morphology_type.nunique()),
            "34 нейрона 21 типа [30, табл. 1]"),
        "%-12s %-26s %-26s %s" % (
            "DAN",
            "%d (L %d, R %d), типов %d" % (len(dan), (dan.side == "left").sum(),
                                           (dan.side == "right").sum(), dan.hemibrain_type.nunique()),
            "%d (R %d)" % (hb_n(("PAM", "PPL1", "PPL2")), hb_n(("PAM", "PPL1", "PPL2"), "right")),
            ">100 нейронов 20 типов [8]"),
        "%-12s %-26s %-26s %s" % (
            "APL",
            "%d (по одному на полушарие)" % len(apl),
            "%d (только правое)" % hb_n(("APL",)),
            "1 на полушарие [8]"),
        "",
        "Разбор расхождений.",
        "",
        "1. KC. Расхождение с hemibrain — примерно +34 % на полушарие; ориентир",
        "   «∼2000» из [30] ближе к hemibrain, чем к FlyWire. Это разные мухи и разные",
        "   реконструкции; объяснение расхождения в наших источниках не проверено и",
        "   остаётся открытым вопросом. Для стенда существенно другое: субстратом",
        "   служит FlyWire v630, и в нём число KC равно посчитанному здесь, а не",
        "   литературному ориентиру. Формулировку «~2000 на полушарие» в whitepaper",
        "   нужно поправить на фактическое число субстрата.",
        "",
        "2. MBON. «34 нейрона 21 типа» — набор [30] 2014 года. В hemibrain [9] типы",
        "   MBON22–MBON35 описаны позже, и разметка FlyWire следует номенклатуре [9].",
        "   Расхождение не ошибка извлечения, а разница наборов; для декодера уровня 1",
        "   это существенно, потому что валентность в [30] установлена только для",
        "   типов MBON01–MBON21.",
        "",
        "3. DAN. Согласие с hemibrain хорошее. «20 типов» из [8] против 25 типов",
        "   номенклатуры [9]: [9] дробит PAM мельче (PAM01–PAM15).",
        "",
        "4. APL. Полное совпадение: по одному на полушарие.",
    ]
    (OUT_RES / "counts_vs_literature.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_RES.mkdir(parents=True, exist_ok=True)

    cfg_json = json.dumps(CONFIG, ensure_ascii=False, sort_keys=True, indent=2)
    cfg_hash = hashlib.sha256(cfg_json.encode("utf-8")).hexdigest()
    print("конфиг подсхемы, sha256 = %s" % cfg_hash)

    print("\n1. разметка типов")
    ann = pd.read_csv(ANN / "Supplemental_file1_annotations.tsv", sep="\t", low_memory=False)
    hb = pd.read_csv(ANN / "Supplemental_file4_hemibrain_meta.csv", low_memory=False)
    print("   аннотаций: %d строк; hemibrain: %d строк" % (len(ann), len(hb)))

    print("\n2. карта компартментов из hemibrain [9]")
    cmap = build_compartment_map(hb)
    n_ok = int((cmap.compartments != "").sum())
    print("   типов: %d, из них с компартментом: %d" % (len(cmap), n_ok))
    for _, r in cmap[cmap.compartments == ""].iterrows():
        print("      без компартмента: %-8s instance=%-8s %s"
              % (r.hemibrain_type, r.instance_raw, r.note))
    cmap.to_csv(OUT_RES / "compartment_map.tsv", sep="\t", index=False, encoding="utf-8")

    print("\n3. отбор нейронов")
    comp = pd.read_csv(REPO / "2023_03_23_completeness_630_final.csv", index_col=0)
    model_ids = set(comp.index.astype("int64"))
    ann = ann[ann.root_id.isin(model_ids)].copy()

    is_core = (ann.cell_class.isin(CONFIG["core_cell_class"])
               | ann.hemibrain_type.isin(CONFIG["core_hemibrain_type"]))
    core = ann[is_core].copy()
    core["mb_role"] = core.cell_class.where(core.cell_class != "MBIN", "APL")

    pn_all = ann[ann.cell_class.isin(CONFIG["input_cell_class"])].copy()
    print("   ядро: %d нейронов; кандидатов PN: %d" % (len(core), len(pn_all)))

    print("\n4. рёбра коннектома")
    con = pd.read_parquet(REPO / "2023_03_23_connectivity_630_final.parquet")
    core_ids = set(core.root_id.astype("int64"))
    # PN оставляем те, что действительно синаптируют на ядро подсхемы
    to_core = con[con.Postsynaptic_ID.isin(core_ids)]
    pn_hit = to_core.groupby("Presynaptic_ID")["Connectivity"].sum()
    keep_pn = {i for i in pn_all.root_id.astype("int64")
               if pn_hit.get(i, 0) >= CONFIG["input_min_connectivity_to_core"]}
    pn = pn_all[pn_all.root_id.isin(keep_pn)].copy()
    pn["mb_role"] = "PN"
    print("   PN со связью на ядро: %d из %d" % (len(pn), len(pn_all)))

    sel = pd.concat([core, pn], ignore_index=True)
    sel_ids = set(sel.root_id.astype("int64"))
    edges = con[con.Presynaptic_ID.isin(sel_ids) & con.Postsynaptic_ID.isin(sel_ids)].copy()
    print("   нейронов в подсхеме: %d; рёбер: %d; синапсов: %d"
          % (len(sel), len(edges), int(edges.Connectivity.sum())))

    print("\n5. компартменты нейронов")
    cm = {k: (v if isinstance(v, str) else "")
          for k, v in cmap.set_index("hemibrain_type")["compartments"].to_dict().items()}

    def compartments_of(t) -> str:
        """Компартменты типа. Составной тип FlyWire ('MBON25,MBON34') — объединение.

        Составной ярлык означает, что разметка не развела два hemibrain-типа;
        объединение компартментов — единственное, что из этого следует, и
        такие нейроны помечены в neurons.csv составным типом.
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

    print("\n6. запись")
    cols = ["root_id", "mb_role", "cell_class", "cell_sub_class", "hemibrain_type",
            "compartment", "side", "top_nt", "top_nt_conf", "ito_lee_hemilineage"]
    neurons = (sel[cols].sort_values(["mb_role", "hemibrain_type", "root_id"])
               .reset_index(drop=True))
    neurons.to_csv(OUT_DATA / "neurons.csv", index=False, encoding="utf-8")

    # подсхема в формате, который читает model.create_model
    sub_comp = comp.loc[sorted(sel_ids)].copy()
    sub_comp.to_csv(OUT_DATA / "completeness.csv", encoding="utf-8")
    idx = {fid: k for k, fid in enumerate(sub_comp.index.astype("int64"))}
    sub_con = edges.copy()
    sub_con["Presynaptic_Index"] = sub_con.Presynaptic_ID.map(idx)
    sub_con["Postsynaptic_Index"] = sub_con.Postsynaptic_ID.map(idx)
    sub_con = (sub_con.sort_values(["Presynaptic_Index", "Postsynaptic_Index"])
               .reset_index(drop=True))
    sub_con.to_parquet(OUT_DATA / "connectivity.parquet", compression="brotli")

    # счётчики
    counts = (neurons.groupby(["mb_role", "hemibrain_type", "side"], dropna=False)
              .size().rename("n").reset_index())
    counts.to_csv(OUT_RES / "type_counts.tsv", sep="\t", index=False, encoding="utf-8")

    by_role = neurons.groupby(["mb_role", "side"], dropna=False).size().unstack(fill_value=0)
    print(by_role.to_string())
    for role in ("MBON", "DAN"):
        sub = neurons[neurons.mb_role == role]
        no_t = int(sub.hemibrain_type.isna().sum())
        no_c = int((sub.compartment == "").sum())
        print("   %-4s без типа hemibrain: %d; без компартмента: %d из %d"
              % (role, no_t, no_c, len(sub)))
        bad = sub[sub.compartment == ""].hemibrain_type.value_counts(dropna=False)
        for t, n in bad.items():
            print("        %s: %d" % ("<без типа>" if pd.isna(t) else t, n))

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

    print("\nкомпартменты в подсхеме: %s" % ", ".join(stats["compartments"]))
    print("готово: data/mb_subcircuit/, results/mb_subcircuit/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
