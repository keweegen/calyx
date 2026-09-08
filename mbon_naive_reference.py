"""Naive MBON odour responses from Huang et al. 2024 public data (Zenodo 10998457),
Figure2 Panel c: 6 MBON types x 5 odours x 12 flies.

These are the reference values behind the V1b criterion "MBON respond to odour"
(specification, section 3, row V1b). Rates are *evoked*: the 0-5 s baseline is
subtracted from the 5-10 s odour window, the same convention huang_reference.py
uses for the Figure3 table. Evoked is the right comparison for the testbed
because the model [2] has no spontaneous activity - its firing rate is the
evoked component by construction, so raw during-odour rates would be compared
against a quantity the model cannot produce.

Odour codes in the file names: VIN = apple cider vinegar, EtA = 1% ethyl
acetate, OCT = 1% 3-octanol, BEN_L / BEN_H = 0.3% and 3% benzaldehyde.

Run:  python mbon_naive_reference.py
"""
from __future__ import annotations

import collections
import io
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio

HERE = Path(__file__).resolve().parent
DATA = os.environ.get("CALYX_DATA") or str(HERE / "data")
OUT = HERE / "results" / "huang_reference" / "mbon_naive_figure2.txt"

ODOR, BASE = (5.0, 10.0), (0.0, 5.0)


def load(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    m = sio.loadmat(io.BytesIO(zf.read(name)))
    rate = float(m["imagingRate"].ravel()[0])
    return np.asarray(m["DetectedSpikes"]).ravel().astype(float) / rate


def rates(t: np.ndarray) -> tuple[float, float]:
    r = lambda a, b: float(((t >= a) & (t < b)).sum()) / (b - a)
    return r(*ODOR) - r(*BASE), r(*BASE)


def main() -> int:
    z = zipfile.ZipFile(os.path.join(DATA, "Figure2.zip"))
    names = [n for n in z.namelist()
             if n.lower().endswith(".mat") and not n.startswith("__MACOSX")
             and "/Panel c/" in n and "/MBON-" in n]

    evoked = collections.defaultdict(lambda: collections.defaultdict(list))
    base = collections.defaultdict(list)
    for n in names:
        cell = n.split("/")[-2]
        stem = n.split("/")[-1][:-4]
        odour = re.search(r"Fly\d+_(.+)$", stem).group(1)
        ev, bl = rates(load(z, n))
        evoked[cell][odour].append(ev)
        base[cell].append(bl)

    odours = sorted({o for c in evoked for o in evoked[c]})
    lines = ["Huang et al. 2024, Figure2 Panel c: naive MBON responses",
             "evoked rate = mean over %g-%g s minus mean over %g-%g s, in Hz"
             % (ODOR + BASE),
             "%d types, %d odours, %d recordings" % (len(evoked), len(odours), len(names)),
             "",
             "%-12s %8s | %s | %6s %4s"
             % ("type", "baseline", "  ".join("%7s" % o for o in odours), "MD", "n")]

    per_type_mean, per_cell_odour, singles, mds = {}, [], [], {}
    for c in sorted(evoked):
        means = {o: float(np.mean(evoked[c][o])) for o in odours}
        hi, lo = max(means.values()), min(means.values())
        md = (hi - lo) / (hi + lo)
        mds[c] = md
        per_type_mean[c] = float(np.mean(list(means.values())))
        per_cell_odour.extend(means.values())
        singles.append(max(max(v) for v in evoked[c].values()))
        lines.append("%-12s %8.1f | %s | %6.2f %4d"
                     % (c, np.mean(base[c]),
                        "  ".join("%7.1f" % means[o] for o in odours),
                        md, len(evoked[c][odours[0]])))

    lines += [
        "",
        "Quantities used by the V1b criterion (all evoked, Hz):",
        "  floor, min over types of the odour-averaged rate : %.1f  (%s)"
        % (min(per_type_mean.values()),
           min(per_type_mean, key=per_type_mean.get)),
        "  min over (type, odour) cell means                : %.1f" % min(per_cell_odour),
        "  ceiling, max over (type, odour) cell means       : %.1f" % max(per_cell_odour),
        "  ceiling, max single recording                    : %.1f" % max(singles),
        "  modulation depth (max-min)/(max+min) by type     : %.2f-%.2f, min %.2f (%s)"
        % (min(mds.values()), max(mds.values()), min(mds.values()),
           min(mds, key=mds.get)),
        "  baseline by type (not reproducible in the model) : %.1f-%.1f"
        % (min(np.mean(base[c]) for c in base), max(np.mean(base[c]) for c in base)),
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nwritten: %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
