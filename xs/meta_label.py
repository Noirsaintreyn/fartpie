"""Meta-labeling (Oct 21 2026): can a model tell good vector trades from bad at entry?
Trades: point-in-time S&P, ratio >= 1.5 (xs/ext/wide_T.parquet, live rules). Features known at entry. Walk-forward by year:
train on trades that EXITED before Jan 1 of the test year, test on that year (2020-2026; 2019 is training only).
Uses: (a) skip low scores, (b) tie-break by score instead of options ratio, (c) size by score. Same 6-slot book + 8% breaker."""
import os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
src = open('xs/slot_tests.py').read().split('P = [(np.datetime64')[0]
src = src.replace("PX = []", "PX = []\nif os.getenv('NOPX', '1') == '1': PX = [pd.Series(np.nan, index=T.index)]\nfor tk, g in ([] if os.getenv('NOPX', '1') == '1' else T.groupby('tk')):", 1).replace("for tk, g in T.groupby('tk'):\n    f = glob", "    f = glob", 1) if False else src
exec(src)                                                # T with rs20, mom60, vol20, rnd, px; CL closes; sim()
T = T.sort_values('entry').reset_index(drop=True)
d = T.entry.dt.normalize()
HI20 = (CL / CL.rolling(20).max() - 1).shift(1); GAP = None
def look(M): return np.array([M[t].asof(x) if t in M else np.nan for t, x in zip(T.tk, d)], float)
T['dist20h'] = look(HI20)
D = pd.read_parquet('etf_daily/SPY.parquet'); D['d'] = pd.to_datetime(D.datetime).dt.tz_localize(None).dt.normalize(); spy = D.drop_duplicates('d').set_index('d').adjclose
vix = pd.read_parquet('etf_daily/VIX.parquet').close; vix.index = pd.to_datetime(vix.index).tz_localize(None).normalize() if getattr(vix.index, 'tz', None) else pd.to_datetime(vix.index).normalize()
T['spy50'] = (spy / spy.rolling(50).mean() - 1).shift(1).reindex(d, method='ffill').values
T['spy20'] = (spy / spy.shift(20) - 1).shift(1).reindex(d, method='ffill').values
T['vix'] = vix.shift(1).reindex(d, method='ffill').values
T['tod'] = T.entry.dt.hour * 60 + T.entry.dt.minute; T['dow'] = T.entry.dt.dayofweek; T['h1f'] = T.h1.astype(int)
T['nday'] = T.groupby(d).tk.transform('size')                                   # flips across the universe that day
T['rrank'] = T.groupby(d).ratio.rank(pct=True)
# last closed trade in the same name (exit before this entry)
prev = []
for tk, g in T.groupby('tk'):
    ex = g.exit.values; rt = g.ret.values; out = []
    for e in g.entry.values:
        k = np.where(ex < e)[0]; out.append(rt[k[np.argmax(ex[k])]] if len(k) else np.nan)
    prev.append(pd.Series(out, index=g.index))
T['prevret'] = pd.concat(prev)
F = ['ratio', 'rrank', 'rs20', 'mom60', 'vol20', 'dist20h', 'spy50', 'spy20', 'vix', 'tod', 'dow', 'h1f', 'nday', 'prevret']
res = {}
for lab, y in (('win', (T.ret > 0).astype(int)), ('runner', (T.ret > .05).astype(int))):
    sc = pd.Series(np.nan, index=T.index); aucs = []
    for yr in range(2020, 2027):
        a = pd.Timestamp(f'{yr}-01-01'); tr = T.exit < a; te = (T.entry >= a) & (T.entry < a + pd.DateOffset(years=1))
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=.05, max_leaf_nodes=15, min_samples_leaf=60, l2_regularization=1.0, random_state=0)
        m.fit(T.loc[tr, F], y[tr]); p = m.predict_proba(T.loc[te, F])[:, 1]; sc[te] = p
        aucs.append((yr, roc_auc_score(y[te], p) if y[te].nunique() > 1 else np.nan, te.sum()))
    T['s_' + lab] = sc; res[lab] = aucs
    print(f'{lab}: base rate {y.mean():.1%}  out-of-sample AUC by year: ' + '  '.join(f'{yr} {au:.3f}' for yr, au, n in aucs), flush=True)
    q = pd.qcut(sc.dropna(), 5, labels=False); g = T.loc[q.index].groupby(q).ret.agg(['mean', 'count']); g['mean'] *= 100
    print('   avg trade % by score quintile (low -> high): ' + '  '.join(f'{v:+.2f}' for v in g['mean']), flush=True)
# book tests 2020-2026 (scored years)
Z = T[T.s_win.notna()].copy()
for lab in ('win', 'runner'):
    Z[f'k_{lab}'] = Z[f's_{lab}']
    Z[f'q_{lab}'] = Z.groupby(Z.entry.dt.year)[f's_{lab}'].rank(pct=True)          # rank within its own test year (no look-ahead across years)
def sim2(Zs, a, b, key='ratio', skip=None, size=None, n=6, brk=.08):
    Zs = Zs[(Zs.entry >= a) & (Zs.entry < b)]
    if skip: Zs = Zs[Zs[skip[0]] >= skip[1]]
    Zs = Zs.sort_values(['entry', key, 'h1', 'tk'], ascending=[True, False, True, True], kind='stable')
    from collections import deque
    op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0
    W = (1 / n) * np.clip(0.5 + Zs[size].values, .5, 1.5) if size else np.full(len(Zs), 1 / n)
    for e, x, r, w in zip(Zs.entry.values, Zs.exit.values, Zs.ret.values, W):
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr, ww in done: real += np.log1p(rr * ww); hist.append((xt, real)); win.append((xt, real))
        op = [o for o in op if o > e]
        if len(op) >= n: continue
        if brk and hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [h for _, h in win]) < np.log1p(-brk): continue
        op.append(x); pend.append((x, r, w))
    for xt, rr, ww in sorted(pend, key=lambda z: z[0]): real += np.log1p(rr * ww); hist.append((xt, real))
    s = pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index()
    eq = np.exp(s.groupby(s.index.normalize()).last()); yy = (eq.index[-1] - eq.index[0]).days / 365.25
    return eq.iloc[-1] ** (1 / yy) - 1, (eq / eq.cummax() - 1).min()
PP = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2020-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2020-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
print(f'\n{"book (6 slots, 8% breaker), scored years only":<48}{"2020-21":>16}{"2022-26":>16}{"2020-26":>16}{"2026 YTD":>16}')
def show(nm, **kw): print(f'  {nm:<46}' + ''.join(f'{c:>+9.1%} /{dd:>5.0%}' for c, dd in (sim2(Z, a, b, **kw) for a, b in PP)), flush=True)
show('live: tie-break by options ratio')
for lab in ('win', 'runner'):
    print(f'  -- model target: {lab}')
    show(f'tie-break by model score', key=f'k_{lab}')
    show(f'skip bottom 20% scores', skip=(f'q_{lab}', .2))
    show(f'skip bottom 40% scores', skip=(f'q_{lab}', .4))
    show(f'size 0.5x-1.5x by score', size=f'q_{lab}')
show('placebo: random tie-break', key='rnd')
