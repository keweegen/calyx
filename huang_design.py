import sys, zipfile, collections, re, io
import numpy as np
import scipy.io as sio

path = sys.argv[1]
z = zipfile.ZipFile(path)
names = [n for n in z.namelist()
         if n.lower().endswith('.mat') and not n.startswith('__MACOSX')]

rows = []
for n in names:
    base = n.split('/')[-1][:-4]
    d = '/'.join(n.split('/')[:-1])
    cell = base.split('_')[0]
    fly = (re.search(r'Fly(\d+)', base) or [None, None])[1]
    cs = 'CS+' if 'CS+' in base else ('CS-' if 'CS-' in base else '?')
    sess = base.split('_')[-1]
    rows.append((d, cell, fly, cs, sess))

# design matrix per panel
bypanel = collections.defaultdict(list)
for r in rows:
    panel = r[0].split('/')[1] if len(r[0].split('/')) > 1 else r[0]
    bypanel[panel].append(r)

for panel in sorted(bypanel):
    rs = bypanel[panel]
    print("\n### PANEL %s  (n=%d files)" % (panel, len(rs)))
    cells = sorted(set(r[1] for r in rs))
    sessions = sorted(set(r[4] for r in rs))
    css = sorted(set(r[3] for r in rs))
    print("  cell types (%d): %s" % (len(cells), cells))
    print("  sessions (%d): %s" % (len(sessions), sessions))
    print("  CS labels: %s" % css)
    print("  %-16s %-6s %s" % ("cell", "flies", "session x CS counts"))
    for c in cells:
        sub = [r for r in rs if r[1] == c]
        flies = sorted(set(r[2] for r in sub))
        grid = collections.Counter((r[4], r[3]) for r in sub)
        cells_txt = " ".join("%s/%s=%d" % (s, x, grid[(s, x)])
                             for s in sessions for x in css if grid[(s, x)])
        print("  %-16s %-6d %s" % (c, len(flies), cells_txt))
        # are the same flies present in every session?
        persess = collections.defaultdict(set)
        for r in sub:
            persess[r[4]].add(r[2])
        allsame = len(set(frozenset(v) for v in persess.values())) == 1
        print("      same flies across all sessions: %s  (%s)"
              % (allsame, {k: len(v) for k, v in sorted(persess.items())}))
