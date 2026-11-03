"""Nov 2 2026: stock-level jump eligibility. Options-ratio universe unchanged; on top, each month a walk-forward model scores every
name by its jump propensity and only higher-scored names may take flips that month. DECLARED BEFORE RESULTS:
 target (primary)    number of big UP days (close-to-close > 2 x 20d sigma) the stock has in the NEXT month (Poisson GBM)
 target (secondary)  the stock's next-month summed trade return (regression)
 features (known at the month start): big-up / big-down day frequency 60d and 250d, 20d realized vol and its ratio to 120d,
   own Hawkes-style decayed count of big-up days, mean options ratio last month, trailing 1y avg trade and flip count of that name
 rule: drop the bottom 20% (primary) / 33% / 50% of names each month by score. Pass: >= 70% of 40 matched runs beat live in BOTH
 2020-21 and 2022-26 (walk-forward by year), neighbour >= 60%."""
import glob, zlib, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from collections import deque
import lightgbm as lgb
X = pd.read_parquet('/tmp/claude-0/ml/noise_X.parquet').sort_values('entry').reset_index(drop=True)
X['mo'] = X.entry.dt.to_period('M')
W = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet'); W['mo'] = W.entry.dt.to_period('M')
RM = W.groupby(['tk', 'mo']).ratio.mean()
rows = []
for tk in X.tk.unique():
    H = pd.read_parquet(glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')[0], columns=['datetime', 'close'])
    H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); m = H.dt.dt.hour * 60 + H.dt.dt.minute
    c = H[(m >= 570) & (m < 960)].groupby(H.dt.dt.normalize()).close.last(); r = c.pct_change(); sd = r.rolling(20).std().shift(1)
    up = (r > 2 * sd).astype(float); dn = (r < -2 * sd).astype(float); hk = up.ewm(alpha=1 - np.exp(-1 / 10), adjust=False).mean()
    D = pd.DataFrame({'up': up, 'dn': dn, 'r': r, 'hk': hk}); D['mo'] = D.index.to_period('M')
    for mo, g in D.groupby('mo'):
        prev = D[D.index < g.index[0]]
        if len(prev) < 250: continue
        rv20, rv120 = prev.r.tail(20).std(), prev.r.tail(120).std()
        T = X[(X.tk == tk) & (X.exit < g.index[0]) & (X.exit >= g.index[0] - pd.Timedelta(days=365))]
        rows.append(dict(tk=tk, mo=mo, f_up60=prev.up.tail(60).mean(), f_up250=prev.up.tail(250).mean(), f_dn60=prev.dn.tail(60).mean(),
                         f_dn250=prev.dn.tail(250).mean(), rv20=rv20, rvr=rv20 / rv120, hk=prev.hk.iloc[-1], ratio=RM.get((tk, mo - 1), np.nan),
                         trec=T.ret.mean() if len(T) >= 5 else np.nan, tn=len(T), y_up=g.up.sum()))
M = pd.DataFrame(rows); M['y_pnl'] = M.set_index(['tk', 'mo']).index.map(X.groupby(['tk', 'mo']).ret.sum()).fillna(0)
F = ['f_up60', 'f_up250', 'f_dn60', 'f_dn250', 'rv20', 'rvr', 'hk', 'ratio', 'trec', 'tn']; M['yr'] = M.mo.dt.year
for tgt, obj in (('y_up', 'poisson'), ('y_pnl', 'regression')):
    M['p_' + tgt] = np.nan
    for Y in range(2020, 2027):
        tr = M.mo < pd.Period(f'{Y}-01', 'M') - 1; te = M.yr == Y
        m = lgb.LGBMRegressor(objective=obj, n_estimators=300, learning_rate=.02, num_leaves=8, min_child_samples=60, subsample=.8, subsample_freq=1, reg_lambda=5, verbose=-1)
        m.fit(M.loc[tr, F], M.loc[tr, tgt]); M.loc[te, 'p_' + tgt] = m.predict(M.loc[te, F])
O = M[M.yr >= 2020]
print('stock-month predictability (rank corr, 2020-21 / 2022-26):')
for tgt in ('y_up', 'y_pnl'):
    print(f'  {tgt}: ' + ' / '.join(f'{g["p_" + tgt].corr(g[tgt], method="spearman"):+.3f}' for _, g in O.groupby(O.yr >= 2022)) +
          '   (also vs next-month trade P&L: ' + ' / '.join(f'{g["p_" + tgt].corr(g.y_pnl, method="spearman"):+.3f}' for _, g in O.groupby(O.yr >= 2022)) + ')', flush=True)
X = X[X.entry >= '2020-01-01'].reset_index(drop=True)
HSH = np.array([zlib.crc32(f'{a}{b}'.encode()) for a, b in zip(X.tk, X.entry.astype('int64'))], dtype=np.uint64)
DROPS = [((HSH * 2654435761 + i * 40503) % 1000) < 50 for i in range(1, 41)]
IDX = pd.bdate_range('2020-01-01', X.exit.max())
P2 = [(np.datetime64('2020-01-01'), np.datetime64('2022-01-01')), (np.datetime64('2022-01-01'), np.datetime64('2027-01-01'))]
def stat2(s):
    eq = s.groupby(s.index.normalize()).last().reindex(IDX, method='ffill').fillna(0); out = []
    for a, b in P2:
        e = eq[(eq.index >= a) & (eq.index < b)]; e = np.exp(e - e.iloc[0]); yr = (e.index[-1] - e.index[0]).days / 365.25
        c = e.iloc[-1] ** (1 / yr) - 1; dd = (e / e.cummax() - 1).min(); out += [c, dd, c / -dd]
    return out
def sim(E, Xx, R, K, TK):
    o = np.lexsort((TK, -K, E)); op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0
    for e, x, r in zip(E[o], Xx[o], R[o]):
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr in done: real += np.log1p(rr / 6); hist.append((xt, real)); win.append((xt, real))
        op = [q for q in op if q > e]
        if len(op) >= 6: continue
        if hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [h for _, h in win]) < np.log1p(-.08): continue
        op.append(x); pend.append((x, r))
    for xt, rr in sorted(pend, key=lambda z: z[0]): real += np.log1p(rr / 6); hist.append((xt, real))
    return pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index()
def ev(keep):
    k = np.asarray(keep, bool)
    return np.array([stat2(sim(X.entry.values[k & ~dl], X.exit.values[k & ~dl], X.ret.values[k & ~dl], X.ratio.values[k & ~dl].astype(float), X.tk.values[k & ~dl])) for dl in DROPS])
def line(nm, a, b=None):
    m = np.median(a, axis=0); s = f'  {nm:<44} ret {m[0]:+.1%} / {m[3]:+.1%}   DD {m[1]:.0%} / {m[4]:.0%}   ret/DD {m[2]:.2f} / {m[5]:.2f}'
    if b is not None:
        w1, w2 = (a[:, 2] > b[:, 2]).mean(), (a[:, 5] > b[:, 5]).mean(); s += f'   beats live {w1:.0%} / {w2:.0%}' + ('  PASS' if w1 >= .7 and w2 >= .7 else '')
    print(s, flush=True)
B0 = ev(np.ones(len(X), bool)); print('\nbook, 40 matched runs, 2020-21 / 2022-26'); line('live (options-ratio universe only)', B0)
top = X.groupby(X.entry.dt.year <= 2021).ret.transform(lambda s: s >= s.quantile(.95))
for tgt, nm in (('y_up', 'jump model'), ('y_pnl', 'next-month P&L model')):
    sc = M.set_index(['tk', 'mo'])['p_' + tgt]; pr = M.assign(p=M['p_' + tgt]).groupby('mo').p.rank(pct=True); PR = pd.Series(pr.values, index=pd.MultiIndex.from_frame(M[['tk', 'mo']]))
    s = pd.Series(list(zip(X.tk, X.mo))).map(PR).values
    for q in (.2, .33, .5):
        keep = ~(s <= q)
        print(f'  {nm}: drop bottom {q:.0%} of names -> removes {(~keep).mean():.0%} of flips, {(top.values & ~keep).sum() / top.sum():.0%} of top-5% winners')
        line(f'    {nm}, drop bottom {q:.0%} of names', ev(keep), B0)
