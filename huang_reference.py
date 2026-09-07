"""Compute the CS+/CS- reference table from Huang et al. 2024 public data
(Zenodo 10998457), Figure3 (1-h protocol) and Figure4 (24-h protocol).

Odour window verified empirically as 5-10 s in a 15-s trace (see h1b.py).
"""
import os, zipfile, io, re, collections
from pathlib import Path
import numpy as np
import scipy.io as sio

# Archives are not stored in the repository (see DATA.md); fetch them with
# scripts/fetch_huang_data.py, or point CALYX_DATA at an existing copy.
DATA = os.environ.get('CALYX_DATA') or str(Path(__file__).resolve().parent / 'data')
ODOR, BASE = (5.0, 10.0), (0.0, 5.0)


def load(zf, name):
    m = sio.loadmat(io.BytesIO(zf.read(name)))
    rate = float(m['imagingRate'].ravel()[0])
    t = np.asarray(m['DetectedSpikes']).ravel().astype(float) / rate
    n = len(np.asarray(m['t']).ravel())
    return t, n / rate


def evoked(t):
    r = lambda a, b: float(((t >= a) & (t < b)).sum()) / (b - a)
    return r(*ODOR) - r(*BASE), r(*BASE)


def collect(zippath, pathfilter, label):
    z = zipfile.ZipFile(zippath)
    names = [n for n in z.namelist()
             if n.lower().endswith('.mat') and not n.startswith('__MACOSX')
             and pathfilter(n)]
    out = collections.defaultdict(dict)
    base = collections.defaultdict(dict)
    lens = set()
    for n in names:
        b = n.split('/')[-1][:-4]
        cell = b.split('_')[0]
        fly = int(re.search(r'Fly(\d+)', b).group(1))
        cs = 'CS+' if 'CS+' in b else 'CS-'
        sess = b.split('_')[-1]
        t, dur = load(z, n)
        lens.add(dur)
        ev, bl = evoked(t)
        out[(cell, fly)][(sess, cs)] = ev
        base[(cell, fly)][(sess, cs)] = bl
    return out, base, lens


def report(title, data, sessions):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    cells = sorted({k[0] for k in data})
    for cell in cells:
        flies = sorted(f for (c, f) in data if c == cell)
        print("\n  %s   (n = %d flies)" % (cell, len(flies)))
        print("    %-6s | %-17s | %-17s | %s"
              % ("sess", "CS+ evoked, Hz", "CS- evoked, Hz", "bias vs Pre, Hz  [95% CI]"))
        for s in sessions:
            plus = np.array([data[(cell, f)][(s, 'CS+')] for f in flies])
            minus = np.array([data[(cell, f)][(s, 'CS-')] for f in flies])
            pre = np.array([data[(cell, f)][('Pre', 'CS+')] - data[(cell, f)][('Pre', 'CS-')]
                            for f in flies])
            bias = (plus - minus) - pre
            sem = lambda v: v.std(ddof=1) / np.sqrt(len(v))
            ci = 1.96 * sem(bias)
            print("    %-6s | %7.2f +- %-6.2f | %7.2f +- %-6.2f | %7.2f  [%7.2f, %7.2f]"
                  % (s, plus.mean(), sem(plus), minus.mean(), sem(minus),
                     bias.mean(), bias.mean() - ci, bias.mean() + ci))
        # normalised: CS+ evoked response as fraction of its own Pre value
        pre_plus = np.array([data[(cell, f)][('Pre', 'CS+')] for f in flies])
        print("    normalised CS+ response (fraction of that fly's Pre value):")
        for s in sessions[1:]:
            v = np.array([data[(cell, f)][(s, 'CS+')] for f in flies]) / pre_plus
            sem = v.std(ddof=1) / np.sqrt(len(v))
            print("      %-6s %6.3f +- %.3f   -> change %+.1f %%  [95%% CI %+.1f, %+.1f]"
                  % (s, v.mean(), sem, 100 * (v.mean() - 1),
                     100 * (v.mean() - 1.96 * sem - 1), 100 * (v.mean() + 1.96 * sem - 1)))


d, b, lens = collect(DATA + '/Figure3.zip', lambda n: '/Panel f/' in n, 'MBON')
print("Figure3 Panel f (MBONs, 1-h protocol, attractive pair ACV / 1%% EtA); trace %s s" % sorted(lens))
report("MBON, Figure 3 (Pre / Mid / 5 min / 1 h)", d, ['Pre', 'Mid', '5min', '1hr'])

d, b, lens = collect(DATA + '/Figure3.zip', lambda n: '/Panel e/' in n, 'DAN')
report("PPL1-DAN, Figure 3 (Pre / Mid / 5 min / 1 h)", d, ['Pre', 'Mid', '5min', '1hr'])

for pair in ['Attractive', 'Repulsive']:
    d, b, lens = collect(DATA + '/Figure4.zip',
                         lambda n, p=pair: ('LTM_%s odor pair' % p) in n, 'LTM')
    report("MBON-a3, Figure 4, %s odour pair (Pre..24 h)" % pair.lower(), d,
           ['Pre', 'Mid', '5min', '1hr', '3hr', '24hr'])
