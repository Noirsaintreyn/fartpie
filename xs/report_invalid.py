"""IN-TRADE bad-sign exit on V15a: after entry, exit at next bar open if a 15m bar CLOSES beyond the flip bar's far extreme (INV=extreme) or its midpoint (INV=mid) against the trade (ICT-style invalidation), optionally only for low/mid narrative scores. Else vector exit (or k15 stop base). Nov 11 2026."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
src = open('/home/claude/fartpie/xs/report_v15_nostop.py').read(); NS = {'__name__': 'x'}; exec(src[:src.index("def run(")], NS)
B, o, h, l, c, dy, reg, bm, adr = (NS[k] for k in ('B', 'o', 'h', 'l', 'c', 'dy', 'reg', 'bm', 'adr'))
F = pd.read_pickle(SC + 'stopscore.pkl'); SCORE = {(k[0], k[1], k[2]): s for k, s in F.score.items()}
def run(stopk, inv, maxscore, cost=.5):
    rows = []; i = 1
    while i < len(B) - 1:
        if dy[i] != dy[i - 1] or reg[i] == reg[i - 1] or reg[i] == 0 or not (585 <= bm[i] + 15 <= 930) or dy[i + 1] != dy[i]: i += 1; continue
        dr = int(reg[i]); e = o[i + 1]; a = adr.get(pd.Timestamp(dy[i]), np.nan); S = e - dr * stopk * a if stopk else None; j = i + 1; ex = None
        key = (pd.Timestamp(dy[i]), int(bm[i] + 15), dr); sc = SCORE.get(key, 3)
        lvl = (l[i] if inv == 'extreme' else (h[i] + l[i]) / 2) if dr > 0 else (h[i] if inv == 'extreme' else (h[i] + l[i]) / 2)
        use = inv is not None and sc <= maxscore
        while j < len(B) and dy[j] == dy[i]:
            if S is not None and np.isfinite(S) and ((dr > 0 and l[j] <= S) or (dr < 0 and h[j] >= S)): ex = S; break
            if use and j + 1 < len(B) and dy[j + 1] == dy[i] and ((dr > 0 and c[j] < lvl) or (dr < 0 and c[j] > lvl)): ex = o[j + 1]; break
            if j + 1 < len(B) and dy[j + 1] == dy[i] and reg[j] == -dr and reg[j - 1] != reg[j]: ex = o[j + 1]; break
            j += 1
        brk = ex is not None
        if ex is None: ex = c[j - 1]
        kk = min(j if brk else j - 1, len(B) - 1); wr = min(((l[q] - e) if dr > 0 else (e - h[q])) for q in range(i + 1, kk + 1)); wr = max(wr, -abs(e - S)) if (S is not None and np.isfinite(S) and ex == S) else wr
        rows.append((key, dr * (ex - e) - cost, min(wr, 0) - cost)); i = max(j, i + 1)
    return pd.DataFrame(rows, columns=['key', 'pts', 'mae'])
def tstat(x): x = np.asarray(x, float); return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 8 and x.std() > 0 else np.nan
print(f'{"variant":44} {"n":>5} {"mean":>6} {"win":>4} {"worst":>6} {"A (t)":>13} {"B (t)":>13} {"sum":>7}')
for lab, args in (('k15 stop, no invalidation (base)', (.15, None, 9)), ('k15 + invalidation(extreme), all scores', (.15, 'extreme', 9)), ('k15 + invalidation(extreme), score<=4', (.15, 'extreme', 4)), ('k15 + invalidation(extreme), score<=2', (.15, 'extreme', 2)),
                  ('k15 + invalidation(mid), all', (.15, 'mid', 9)), ('k15 + invalidation(mid), score<=4', (.15, 'mid', 4)), ('k15 + invalidation(mid), score<=2', (.15, 'mid', 2)),
                  ('k25 stop base', (.25, None, 9)), ('k25 + invalidation(extreme), score<=4', (.25, 'extreme', 4)), ('k25 + invalidation(mid), score<=4', (.25, 'mid', 4)), ('k25 + invalidation(extreme), all', (.25, 'extreme', 9))):
    R = run(*args); R['day'] = [k[0] for k in R.key]; y = R.pts; a, b = R[R.day < '2022-01-01'].pts, R[R.day >= '2022-01-01'].pts
    print(f'{lab:44} {len(R):5d} {y.mean():+6.1f} {np.mean(y>0):4.0%} {y.min():+6.0f} {a.mean():+6.1f} ({tstat(a):+.1f}) {b.mean():+6.1f} ({tstat(b):+.1f}) {y.sum():+7.0f}', flush=True)
