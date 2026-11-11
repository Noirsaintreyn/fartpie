"""NQ vector flips on 15m RTH bars with PURE vector exits (no 0.15 ADR stop) vs the same entries with the 0.15*ADR20 stop. Points per contract, 2014-2026. Nov 11 2026."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/fartpie/xs'); _a = sys.argv; sys.argv = sys.argv[:1] + ['x']; NSL = {}
exec(open('/home/claude/fartpie/xs/signal_lab.py').read().split("if sys.argv[1] == '1':")[0], NSL); sys.argv = _a; vec = NSL['vec']
X = pd.read_parquet('/tmp/claude-0/ml/nq5_real.parquet'); X = X[X.rth].copy(); X['day'] = pd.to_datetime(X.day).dt.normalize(); print(X.columns.tolist()[:12])
X['b'] = X.m // 15 * 15 if 'm' in X else 0
B = X.groupby(['day', 'b']).agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last')).reset_index().sort_values(['day', 'b']).reset_index(drop=True)
B['reg'] = np.asarray(vec(B[['open', 'high', 'low', 'close']])); rng = B.groupby('day').apply(lambda g: g.high.max() - g.low.min()); adr = rng.shift(1).rolling(20).mean()
o, h, l, c = B.open.values, B.high.values, B.low.values, B.close.values; dy = B.day.values; reg = B.reg.values; bm = B.b.values
def run(stopk, cost=.5):
    rows = []; i = 1
    while i < len(B) - 1:
        if dy[i] != dy[i - 1] or reg[i] == reg[i - 1] or reg[i] == 0 or not (585 <= bm[i] + 15 <= 930) or dy[i + 1] != dy[i]: i += 1; continue
        dr = int(reg[i]); e = o[i + 1]; a = adr.get(pd.Timestamp(dy[i]), np.nan); S = (e - dr * min(stopk[1] * a, stopk[2]) if isinstance(stopk, tuple) else e - dr * stopk * a) if stopk else None; j = i + 1; ex = None
        while j < len(B) and dy[j] == dy[i]:
            if S is not None and np.isfinite(S) and ((dr > 0 and l[j] <= S) or (dr < 0 and h[j] >= S)): ex = S; break
            if j + 1 < len(B) and dy[j + 1] == dy[i] and reg[j] == -dr and reg[j - 1] != reg[j]: ex = o[j + 1]; break
            j += 1
        brk = ex is not None
        if ex is None: ex = c[j - 1]
        kk = min(j if brk else j - 1, len(B) - 1); wr = min(((l[q] - e) if dr > 0 else (e - h[q])) for q in range(i + 1, kk + 1)); wr = max(wr, -abs(e - S)) if (S is not None and np.isfinite(S) and ex == S) else wr; rows.append((pd.Timestamp(dy[i]), dr * (ex - e) - cost, min(wr, 0) - cost, int(bm[i] + 15), dr)); i = max(j, i + 1)
    return pd.DataFrame(rows, columns=['day', 'pts', 'mae', 'm', 'dir'])
def tstat(x): x = np.asarray(x, float); return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 8 and x.std() > 0 else np.nan
for nm, k in (('with 0.15 ADR stop', .15), ('PURE vector exit (no stop)', None), ('wide 0.5 ADR catastrophe stop', .5)):
    T = run(k); a = T[T.day < '2018-01-01'].pts; b = T[(T.day >= '2018-01-01') & (T.day < '2022-01-01')].pts; d = T[T.day >= '2022-01-01'].pts
    print(f'{nm:30} n {len(T)} pts/trade {T.pts.mean():+6.1f} win {np.mean(T.pts>0):3.0%} t {tstat(T.pts):+5.1f} worst {T.pts.min():+.0f} | 2014-17 {a.mean():+6.1f} ({tstat(a):+.1f}) | 2018-21 {b.mean():+6.1f} ({tstat(b):+.1f}) | 2022-26 {d.mean():+6.1f} ({tstat(d):+.1f}) | pts/day {T.pts.sum()/T.day.nunique():+.1f}')

import pickle; pickle.dump({nm: run(k) for nm, k in (('pure', None), ('cat05', .5), ('stop15', .15), ('cap100', ('c', .5, 100.0)), ('p100', ('c', 99.0, 100.0)))}, open('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/v15_exits.pkl', 'wb'))
