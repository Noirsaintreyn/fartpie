"""Nov 2 2026: the vector run ON volatility instead of price - is the stock's volatility trending up or down when the price vector flips bull?
 vrv   vector on the stock's daily log realized variance (15m RTH + overnight), state at the prior close (1 = vol expanding regime)
 vgap  vector on the 3-day average log |overnight gap| (news-volatility regime)
 vhar  vector on the HAR-style smoothed vol (5-day mean log RV)
Series fed to the vector as close; open = prior value, high/low = max/min of the two. Avg trade by state (book's trades), then as tie-break /
filter, 40 matched runs."""
import os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/fartpie/xs/stress_budget.py').read().split("rng = np.random.default_rng(43)")[0].replace("print(", "(lambda *a, **k: None)("))
Ls = Ls.reset_index(drop=True); Ls['d'] = Ls.entry.dt.normalize(); F = []
def vec_series(s):
    s = s.dropna(); o = s.shift().fillna(s); X = pd.DataFrame({'dt': s.index, 'open': o.values, 'high': np.maximum(o, s).values, 'low': np.minimum(o, s).values, 'close': s.values, 'volume': 1.0})
    return pd.Series(LS.vec(X).reg.values, index=s.index)
for tk in Ls.tk.unique():
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    if not f: continue
    H = pd.read_parquet(f[0]); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 570) & (m < 960)].drop_duplicates('dt').sort_values('dt'); H['d'] = H.dt.dt.normalize()
    r = np.log(H.close).diff(); r[H.d != H.d.shift()] = np.nan
    cl = H.groupby('d').close.last(); op = H.groupby('d').open.first(); on = np.log(op / cl.shift())
    rv = r.pow(2).groupby(H.d).sum() + on.fillna(0) ** 2; lrv = np.log(rv.clip(lower=1e-8))
    X = pd.DataFrame({'vrv': vec_series(lrv), 'vgap': vec_series(np.log(on.abs().clip(lower=1e-5)).rolling(3).mean()), 'vhar': vec_series(lrv.rolling(5).mean())})
    X = X.shift(1); X['tk'] = tk; F.append(X.reset_index().rename(columns={'index': 'd'}))       # known at the prior close
F = pd.concat(F); F['d'] = pd.to_datetime(F.d); Ls = Ls.merge(F, on=['tk', 'd'], how='left'); K = ['vrv', 'vgap', 'vhar']
TK = taken(Ls).merge(Ls[['tk', 'entry'] + K], on=['tk', 'entry']); TK['h'] = np.where(TK.entry.dt.year <= 2021, '2019-21', '2022-26')
print('A) trades the book takes: avg trade % (n) by the volatility-vector state at entry   2019-21 / 2022-26')
for c in K:
    t = TK.groupby([c, 'h']).ret.agg(['mean', 'size']).unstack('h')
    print(f'  {c}: ' + '   '.join(f'{"vol expanding" if k == 1 else "vol contracting" if k == -1 else k}: {t.loc[k, ("mean", "2019-21")] * 100:+.2f}% ({int(t.loc[k, ("size", "2019-21")])}) / {t.loc[k, ("mean", "2022-26")] * 100:+.2f}% ({int(t.loc[k, ("size", "2022-26")])})' for k in t.index if k in (1, -1)), flush=True)
pr = lambda s: s.groupby(Ls.entry.dt.year).rank(pct=True)
rng = np.random.default_rng(109); DR = [rng.random(len(Ls)) < .05 if i else np.zeros(len(Ls), bool) for i in range(41)]
def ev(Z0): return np.array([[v_ for cd in stat(dret(simw(Z0[~dl[Z0.index]])).cumsum()) for v_ in (cd[0], cd[1], cd[0] / -cd[1])] for dl in [pd.Series(d, Ls.index) for d in DR]])
def line(nm, a, b=None):
    m = np.median(a[1:], axis=0); s = f'  {nm:<36} ret {m[0]:+.1%} / {m[3]:+.1%}   DD {m[1]:.0%} / {m[4]:.0%}   ret/DD {m[2]:.2f} / {m[5]:.2f}'
    if b is not None: s += f'   beats live {(a[1:, 2] > b[1:, 2]).mean():.0%} / {(a[1:, 5] > b[1:, 5]).mean():.0%}'
    print(s, flush=True)
print('\nB) 40 matched runs, 2019-21 / 2022-26'); B0 = ev(Ls); line('live', B0)
for c in K:
    for sgn, nm in ((1, 'expanding'), (-1, 'contracting')):
        Z = Ls.copy(); Z['ratio'] = (pr(Ls.ratio) + (Ls[c] == sgn) * 10).values; line(f'{c} {nm} first', ev(Z), B0)
        line(f'skip when {c} {"contracting" if sgn == 1 else "expanding"}', ev(Ls[Ls[c] != -sgn]), B0)
