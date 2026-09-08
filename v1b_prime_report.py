# -*- coding: utf-8 -*-
"""Отчёт ступени V1b′: исход и карта по сетке.

При исходе FAIL-CAL-MBON спецификация (раздел 3ж, V1b′-4.8) требует привести
для каждого кандидата `f̄_C`, `max f`, `s_C^{αβ}`, `R_t` для шести типов T₃₈,
`MD_t` и перечень нарушенных подограничений, а также заголовочную величину -
максимум по кандидатам от минимума по типам `R_t`, то есть лучшее, что даёт
семейство на полу при достигнутой разреженности.

Отчётные величины V1b′-5а печатаются здесь же: число кандидатов и допустимых
точек по стадиям и, если выбранная точка существует, её расстояние до края.

Запуск:  python v1b_prime_report.py            # таблица и сводка
         python v1b_prime_report.py --json     # плюс артефакт map.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OUT = HERE / "results" / "v1b_prime"
PILOT = HERE / "results" / "v1b"


def load(stage: int) -> list:
    p = OUT / ("calibration_stage%d.json" % stage)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def key(p: dict) -> tuple:
    return round(p["pn_kc_scale"], 10), round(p["g_apl_rel"], 10)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="записать map.json")
    a = ap.parse_args()

    import v1b_subcircuit as V

    s1, s2 = load(1), load(2)
    if not (s1 and s2):
        print("сетка не собрана: запустите run_v1b_grid.py --merge", file=sys.stderr)
        return 1

    print("Ступень V1b′, спецификация v0.15, раздел 3ж")
    print("=" * 100)
    per_stage = {}
    for n, pts in ((1, s1), (2, s2)):
        cand = [p for p in pts if V.passes_fraction(p)]
        adm = [p for p in pts if V.admissible(p)]
        unk = [p for p in cand if V.passes_mbon(p) is None]
        per_stage[n] = {"n_points": len(pts), "n_candidates": len(cand),
                        "n_admissible": len(adm), "n_not_evaluated": len(unk)}
        print("стадия %d: точек %d, кандидатов %d, допустимых %d"
              % (n, len(pts), len(cand), len(adm)))
        if unk:
            # V1b′-4.5: в завершённом прогоне это состояние запрещено
            print("   ОШИБКА ИСПОЛНЕНИЯ: у %d кандидатов ограничение MBON не "
                  "вычислено" % len(unk))
            return 1

    cand = {}
    for p in s1 + s2:
        if V.passes_fraction(p):
            cand.setdefault(key(p), p)
    print("различных кандидатов по обеим стадиям: %d" % len(cand))

    # выбор точки идёт по стадии 2 (V1b-4.6)
    best = V.choose_point([dict(p, stage=2) for p in s2])
    print()
    if best is not None:
        outcome = "PASS или FAIL-EVAL — решается оценкой на E"
        print("выбрана точка: scale %.5g, g %.5g g_ref"
              % (best["pn_kc_scale"], best["g_apl_rel"]))
    elif not cand:
        outcome = "FAIL-CAL-KC"
    else:
        outcome = "FAIL-CAL-MBON"
    print("ИСХОД: %s" % outcome)

    rows = []
    for kk in sorted(cand):
        p = cand[kk]
        m = p["mbon"]
        rt = {t: d["R_t"] for t, d in m["per_type"].items()}
        md = {t: d["MD"] for t, d in m["per_type"].items()}
        broke = [n for n, v in (("V1b-3.3", m["floor_ok"]),
                                ("V1b-3.4", m["ceiling_ok"]),
                                ("V1b-3.5", m["md_ok"])) if not v]
        rows.append({"pn_kc_scale": p["pn_kc_scale"], "g_apl_rel": p["g_apl_rel"],
                     "f_mean": p["f_mean"], "f_max": p["f_max"],
                     "s_ab": p["s_ab"], "R_t": rt, "MD": md,
                     "min_R_t": min(rt.values()), "max_R_t": max(rt.values()),
                     "violated": broke})

    print()
    print("Карта кандидатов (V1b′-4.8). Порог пола %.1f Гц у ВСЕХ шести типов, "
          "потолок %.0f Гц." % (V.MBON_FLOOR_HZ, V.MBON_CEIL_HZ))
    hdr = "%-8s %-9s %7s %7s %6s %9s %9s  %s" % (
        "scale", "g/g_ref", "f", "f_max", "s_ab", "min R_t", "max R_t", "нарушено")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: (-r["max_R_t"], r["pn_kc_scale"])):
        print("%-8.4g %-9.5g %7.4f %7.4f %6.2f %9.4f %9.4f  %s"
              % (r["pn_kc_scale"], r["g_apl_rel"], r["f_mean"], r["f_max"],
                 r["s_ab"], r["min_R_t"], r["max_R_t"], ", ".join(r["violated"])))

    head = max(r["min_R_t"] for r in rows) if rows else float("nan")
    best_type = max(r["max_R_t"] for r in rows) if rows else float("nan")
    silent = sum(1 for r in rows if r["max_R_t"] == 0)
    print()
    print("Заголовочная величина исхода: max по кандидатам от min по типам R_t "
          "= %.6f Гц при поле %.1f Гц." % (head, V.MBON_FLOOR_HZ))
    print("Лучшая частота одного типа среди всех кандидатов: %.4f Гц — "
          "недобор до пола в %.0f раз." % (best_type, V.MBON_FLOOR_HZ / best_type
                                           if best_type else float("inf")))
    print("Кандидатов, где ни один из 97 MBON не дал ни одного спайка: %d из %d."
          % (silent, len(rows)))
    n_floor = sum(1 for r in rows if "V1b-3.3" in r["violated"])
    n_ceil = sum(1 for r in rows if "V1b-3.4" in r["violated"])
    n_md = sum(1 for r in rows if "V1b-3.5" in r["violated"])
    print("Нарушают пол V1b-3.3: %d; потолок V1b-3.4: %d; различение V1b-3.5: %d."
          % (n_floor, n_ceil, n_md))

    if a.json:
        art = {"stage": "V1b'", "spec_version": "v0.15", "outcome": outcome,
               "per_stage": per_stage, "n_candidates_distinct": len(cand),
               "headline_max_of_min_R_t_hz": head,
               "best_single_type_R_t_hz": best_type,
               "n_candidates_fully_silent": silent,
               "violations": {"V1b-3.3": n_floor, "V1b-3.4": n_ceil,
                              "V1b-3.5": n_md},
               "floor_hz": V.MBON_FLOOR_HZ, "ceiling_hz": V.MBON_CEIL_HZ,
               "candidates": rows}
        p = OUT / "report_map.json"
        p.write_text(json.dumps(art, ensure_ascii=False, indent=2),
                     encoding="utf-8")
        print("\nзаписано: %s" % p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
