"""506d: context sizing. Hypotheses (declared before looking): (A) agreement - flips that several options models tag at once carry more edge, untagged less; (B) persistence - when the last 30 closed flips across the universe lost on average, the next ones are worse (own-edge regime).
Multipliers fixed a priori: A = 0.5 untagged / 1 one tag / 1.5 two+ tags; B = 0.5 when trailing-30 mean < 0. Base sizing = cushion 6x (cap 30000/K), K=5, 14 perp names, ALL flips.
Split: train cohorts start 2019-21, test 2022+. Metric: P(+900 before floor/daily) over 365d."""
import sys
exec(open('xs/report_506_perps.py').read().split("if __name__")[0])
PERP = {'NVDA', 'MU', 'AMD', 'GOOGL', 'TSLA', 'COIN', 'META', 'AVGO', 'AAPL', 'HOOD', 'ORCL', 'PLTR', 'AMZN', 'MSFT'}
T = SETS['ALL flips'].copy(); T = T[T.tk.isin(PERP)].copy()
tg = J[J.tag.isin(['PCV', 'PCZ', 'SKW', 'CHM'])].groupby(['tk', 'entry', 'h1']).tag.nunique().rename('ntag').reset_index()
T = T.merge(tg, on=['tk', 'entry', 'h1'], how='left'); T['ntag'] = T.ntag.fillna(0).astype(int)
U = SETS['ALL flips'].sort_values('exit'); ex = U.exit.values; cr = np.cumsum(U.ret.values); n = len(U)
def trail(ts):
    j = np.searchsorted(ex, np.datetime64(ts), side='left')
    return np.nan if j < 60 else (cr[j - 1] - cr[j - 31]) / 30
T['trail'] = [trail(t) for t in T.entry]; T['yr'] = T.entry.dt.year
T['era'] = np.where(T.yr <= 2021, 'train', 'test')
print('edge bp by ntag:'); print(T.groupby(['era', 'ntag']).ret.agg(n='size', bp=lambda x: x.mean() * 1e4).round(1).to_string())
T['bad'] = T.trail < 0
print('edge bp by trailing-30 sign:'); print(T.groupby(['era', 'bad']).ret.agg(n='size', bp=lambda x: x.mean() * 1e4).round(1).to_string())
def coh2(X, start, mult, base, K, horizon=365):
    X = X[(X.entry >= start) & (X.entry < start + pd.Timedelta(days=horizon))]
    ev = sorted([(r.entry, 0, i) for i, r in enumerate(X.itertuples())] + [(r.exit, 1, i) for i, r in enumerate(X.itertuples())])
    rets = X.ret.values; hold = ((X.exit - X.entry).dt.total_seconds() / 86400).values; m = mult(X); sz = {}; op = set(); bal = 10000.0; cur = None; ds = bal
    pn = lambda i: sz[i] * (rets[i] - 2 * COM - SWAP * hold[i])
    for ts, k, i in ev:
        d = ts.normalize()
        if d != cur: cur = d; ds = bal + sum(min(0, pn(j)) for j in op)
        if k == 0:
            if len(op) >= K: continue
            sz[i] = min(30000 / K, base(bal) * m[i]); op.add(i)
        else:
            if i not in sz or i not in op: continue
            op.discard(i); bal += pn(i)
            if bal >= 10900: return 'P', (ts - start).days
        eq = bal + sum(min(0, pn(j)) for j in op)
        if eq < 9700 or eq < ds - 300: return 'F', (ts - start).days
    return 'N', None
MULTS = {'flat': lambda X: np.ones(len(X)),
         'A agree': lambda X: np.where(X.ntag.values == 0, .5, np.where(X.ntag.values == 1, 1, 1.5)),
         'B persist': lambda X: np.where(X.bad.values, .5, 1.0),
         'A+B': lambda X: np.where(X.ntag.values == 0, .5, np.where(X.ntag.values == 1, 1, 1.5)) * np.where(X.bad.values, .5, 1.0)}
rows = []; Ts = T.sort_values('entry').reset_index(drop=True)
for era, (a, b) in (('train 2019-21', ('2019-02-01', '2021-12-31')), ('test 2022-25', ('2022-01-03', '2025-10-20'))):
    days = pd.date_range(a, b, freq='B')[::3]
    for lab, mf in MULTS.items():
        for c, K in ((6, 5), (10, 5)):
            R = [coh2(Ts, d, mf, (lambda bl, c=c: c * (bl - 9700)), K) for d in days]
            p = [r for r in R if r[0] == 'P']; f = [r for r in R if r[0] == 'F']
            rows.append(dict(era=era, ctx=lab, cushion=f'{c}x{K}', n=len(R), passed=round(100 * len(p) / len(R)), failed=round(100 * len(f) / len(R)), p_before_f=round(100 * len(p) / max(1, len(p) + len(f))),
                             pass90=round(100 * np.mean([1 if (r[0] == 'P' and r[1] <= 90) else 0 for r in R])), med_days=float(np.median([r[1] for r in p])) if p else None)); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv('xs/report_506d_context_size.csv', index=False)
