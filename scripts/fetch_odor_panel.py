# -*- coding: utf-8 -*-
"""Скачать одорантные данные Hallem & Carlson 2006 и построить вход PN для V1b.

Источник: пакет `DoOR.data` (rOpenSci), https://github.com/ropensci/DoOR.data,
ветка master. Он воспроизводит опубликованные матрицы отдельных исследований
как CSV; данные Hallem & Carlson 2006 [39] лежат в колонке `Hallem.2006.EN`
файлов по рецепторам, первая строка (`CAS == "SFR"`) — спонтанная активность.
Метаданные исследования — `door_dataset_info.csv`: электрофизиология, единицы
«spikes», спонтанная активность вычтена, концентрация 10^-2, растворитель
парафиновое масло (16 водорастворимых веществ — в воде).

Почему Hallem, а не консенсусная матрица DoOR. Решено при предрегистрации V1b.
Консенсус покрывает больше гломерул в объединении по базе (47 против 24, 89 %
синапсов PN→KC против 45 %), но покрытие там **зависит от одоранта** (45–80 %
по панели), а пересечение измеренных гломерул по всем одорантам панели равно
ровно набору Hallem. Значит весь выигрыш консенсуса состоит из запах-зависимого
покрытия, а оно вносит артефакт в критерий перекрытия ансамблей, направленный
в сторону прохождения ступени. Постоянная маска Hallem вносит артефакт
известного знака, направленный в сторону провала. Выбран второй.
Консенсус остаётся ветвью чувствительности; она отчётная и вердикт не меняет.

Преобразование ORN → PN: дивизивная нормировка Olsen, Bhandawat & Wilson 2010
[40], уравнения 2, 4 и 5:
    PN_g = R_max * ORN_g^n / (ORN_g^n + sigma^n + s^n),   s = m * (sum_g ORN_g) / 190
Константы из статьи, режим A: R_max = 165 спайк/с, sigma = 12 спайк/с, n = 1,5,
m = 10,63 (гломерула VM7). Отрицательный вход (торможение ниже спонтанного
уровня) даёт нулевой отклик PN — правило самой статьи; в модели [2] фоновая
частота равна нулю, поэтому иное представление невозможно.

Панель ступени: одоранты, присутствующие одновременно в Hallem 2006 и в
панелях Honegger 2011 [25] и Campbell 2013 [41] — тех работах, откуда взяты
числа критериев V1b. Проверено: из 17 химически определённых веществ этих
панелей в Hallem есть 14 (нет 3-октанола, 4-метилциклогексанола и
1-гептен-3-ола; последнего нет ни в одном датасете DoOR).

Лицензия DoOR.data — MIT; исходные данные Hallem & Carlson 2006 — Cell,
цитируются по DOI 10.1016/j.cell.2006.01.050. В git данные не кладутся (DATA.md).

Запуск:  python scripts/fetch_odor_panel.py
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

# Olsen 2010, режим A
R_MAX, SIGMA, EXPONENT, M_COEF, LFP_DIV = 165.0, 12.0, 1.5, 10.63, 190.0

# панель: Hallem 2006 ∩ (Honegger 2011 ∪ Campbell 2013), по CAS
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
    """hemibrain делит VC3 на VC3l/VC3m; мультигломерулярные записи не разносим."""
    if gl == "VC3":
        return {"VC3l", "VC3m"}
    return set() if "+" in gl else {gl}


def olsen(orn: pd.Series) -> pd.Series:
    """ORN (спайк/с, вызванный отклик) -> PN (спайк/с). Olsen 2010, ур. 2, 4, 5."""
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
        print("матрица Hallem не 110x24, а %dx%d — источник изменился" % mat.shape,
              file=sys.stderr)
        return 1

    mapping = read("door_mappings.csv")[["receptor", "glomerulus"]].dropna().astype(str)
    mapping = mapping[(mapping.glomerulus != "?") & (mapping.receptor != "?")]
    rec2gl = dict(zip(mapping.receptor, mapping.glomerulus))

    neurons = pd.read_csv(SUB / "neurons.csv")
    neurons = neurons.assign(gl=neurons.hemibrain_type.astype(str).str.split("_").str[0])
    # Оба полушария. Отклик гломерулы присваивается всем её uPN независимо от
    # стороны: отображение «рецептор → гломерула» стороны не различает, а запах
    # в норме возбуждает обе антеннальные доли. Односторонняя стимуляция сделала
    # бы половину компонент вектора перекрытия (раздел 3д) согласованными нулями
    # и подняла бы корреляцию у всех пар разом.
    upn = neurons[(neurons.mb_role == "PN")
                  & (neurons.cell_sub_class == "uniglomerular")]

    # отображение «гломерула -> рецептор»: у VC3 две hemibrain-гломерулы (VC3l, VC3m)
    # на один рецептор Or35a, и обе получают его отклик. Обратное отображение здесь
    # некорректно: словарь «рецептор -> гломерула» оставил бы произвольную одну.
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
            print("одорант %s (%s) отсутствует в матрице Hallem" % (name, cas),
                  file=sys.stderr)
            return 1
        # отклик по гломерулам маски; сумма для нормировки — по всем 24 рецепторам
        orn_all = mat.loc[key]
        pn_all = olsen(orn_all)
        rows[name] = {gl: pn_all[rec_of_gl[gl]] for gl in mask_gl}
    pn = pd.DataFrame(rows).T.reindex(columns=mask_gl)

    # покрытие входа
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

    print("матрица Hallem: %d одорантов x %d рецепторов, пропусков %d"
          % (mat.shape[0], mat.shape[1], int(mat.isna().sum().sum())))
    print("панель: %d одорантов, %d пар" % (pn.shape[0], pn.shape[0] * (pn.shape[0] - 1) // 2))
    print("маска: %d гломерул из %d, %d uPN из %d, %.0f %% синапсов PN->KC"
          % (len(mask_gl), upn.gl.nunique(), int(upn.gl.isin(mask_gl).sum()),
             len(upn), 100 * syn_mask / syn_total))
    print("покрытие входа KC: медиана %.2f, квартили %.2f/%.2f, ниже 0,1 у %d из %d"
          % (c_i.median(), c_i.quantile(.25), c_i.quantile(.75),
             (c_i < .1).sum(), len(c_i)))
    print("частоты PN: %.1f-%.1f спайк/с" % (pn.min().min(), pn.max().max()))
    print("записано: %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
