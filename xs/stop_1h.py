"""Nov 3 2026: user idea after CVNA: a 15m vector stop should not fire on a small break when the 1h vector is still bullish.
Variants of the 15m track's pre-sticky exit (the +2% sticky-1h stop and the +5% lock are unchanged; 1h track unchanged; entries are the vector's):
 L   live: exit on the first 15m flip down
 A   1h-confirm: exit on a 15m flip down ONLY when the 1h vector is also bearish (+ hard stop at -H on the prior close)
     A3: H = 3%   A5: H = 5% (neighbour)   Ainf: no hard stop (8% book breaker only)
 G   grace window: 15m flip ignored for the first N 15m bars after entry unless loss >= 3%   G4 (1h)   G8 (2h, neighbour)
Declared before results. Pass rule as always: >= 70% of 40 matched runs beat live in BOTH periods, neighbour >= 60%.
Trade set is rebuilt per variant (a longer hold blocks the next entry on that ticker); parity check: variant L must equal the live set."""
import sys, os, glob, zlib, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from collections import deque
HERE = '/home/claude/fartpie/xs'; sys.path.insert(0, HERE); _a = sys.argv; sys.argv = sys.argv[:1] + ['x']
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0]); sys.argv = _a
def run2(H, reg, s1, mode='L', hard=.03, grace=0):
    o, c, hi, d, m = H.open.values, H.close.values, H.high.values, H.dt.values, H.m.values
    g = [j for j in range(1, len(H)) if 570 <= m[j] < 960]; T = []; pos = False; rearm = None
    KR = [reg[j - 1] for j in g]; KS = [s1[j - 1] for j in g]
    for p in range(1, len(g)):
        j = g[p]
        if not pos:
            if KR[p] == 1 and KR[p - 1] != 1: pos, e, px, b1, peak, h1, pe = True, j, o[j], False, o[j], False, p
            continue
        if j == e: continue
        k = g[p - 1]
        if k >= e: peak = max(peak, hi[k])
        b1 = b1 or KS[p] == 1; now = c[j - 1] / px - 1 >= .02 and b1; h1 = h1 or now
        cur = c[j - 1] / px - 1
        if h1: ex = KS[p] == -1
        else:
            ex = KR[p] == -1
            if ex and mode == 'A': ex = KS[p] == -1
            if ex and mode == 'G' and p - pe <= grace: ex = False
            if not ex and mode in ('A', 'G') and hard is not None and cur <= -hard: ex = True
        pg = peak / px - 1; lock = False
        if pg >= .05 and cur <= .5 * pg: ex = True
        if ex: T.append((d[e], d[j], o[j] / px - 1 - C2)); pos = False
    return T
if __name__ == '__main__':
    VAR = {'L': dict(), 'A3': dict(mode='A', hard=.03), 'A5': dict(mode='A', hard=.05), 'Ainf': dict(mode='A', hard=None),
           'G4': dict(mode='G', grace=4, hard=.03), 'G8': dict(mode='G', grace=8, hard=.03)}
    Xo = pd.read_parquet('/tmp/claude-0/ml/noise_X.parquet'); tks = sorted(Xo.tk.unique()); rows = []
    for tk in tks:
        f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
        if not f: continue
        H = load(f[0]); reg0, s10 = states(H, vec)
        for nm, kw in VAR.items():
            rows += [(tk, nm, False, pd.Timestamp(a), pd.Timestamp(b), r) for a, b, r in run2(H, reg0, s10, **kw)]
        for a, b, r in run(H, reg0, s10, True)[0]: rows.append((tk, 'H1', True, pd.Timestamp(a), pd.Timestamp(b), r))
        print(tk, flush=True)
    V = pd.DataFrame(rows, columns=['tk', 'var', 'h1', 'entry', 'exit', 'ret']); V.to_parquet('/tmp/claude-0/ml/stop1h_V.parquet')
