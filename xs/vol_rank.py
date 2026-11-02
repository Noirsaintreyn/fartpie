"""Nov 2 2026: rank flips by forecast volatility EXPANSION (the book's profit is ~70-80% from >2-sigma days). All known at the prior close.
 har     HAR-RV (Corsi 2009) on log realized variance (15m RTH returns + overnight), pooled, walk-forward by year -> forecast vs the stock's 1y median RV
 harrs   HAR with realized semivariances (Patton & Sheppard 2015): good (RS+) / bad (RS-) volatility split
 harj    HAR with the jump part (RV - bipower variation; Andersen, Bollerslev & Diebold 2007)
 gjr     GJR-GARCH(1,1) per stock (Glosten, Jagannathan & Runkle 1993), refit each Jan 1 on the prior 3 years -> next-day sigma vs long-run sigma
 rsplus  share of good volatility RS+/RV, last 5 days        jumpy  number of >2-sigma days in the last 60
A) avg trade by quintile for the trades the book takes; how often the trade's holding period contains a >2-sigma up day
B) as the slot tie-break (alone and blended with the ratio), 40 matched runs."""
import os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from arch import arch_model
exec(open('/home/claude/fartpie/xs/stress_budget.py').read().split("rng = np.random.default_rng(43)")[0].replace("print(", "(lambda *a, **k: None)("))
Ls = Ls.reset_index(drop=True); DD = []
for tk in Ls.tk.unique():
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    if not f: continue
    H = pd.read_parquet(f[0]); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 570) & (m < 960)].drop_duplicates('dt').sort_values('dt'); H['d'] = H.dt.dt.normalize()
    r = np.log(H.close).diff(); r[H.d != H.d.shift()] = np.nan                                  # intraday 15m returns
    g = pd.DataFrame({'r': r, 'd': H.d}).dropna()
    cl = H.groupby('d').close.last(); op = H.groupby('d').open.first(); on = np.log(op / cl.shift())
    rv = g.groupby('d').r.apply(lambda x: (x ** 2).sum()) + on.reindex(g.d.unique()).fillna(0) ** 2
    bv = g.groupby('d').r.apply(lambda x: (np.pi / 2) * (x.abs() * x.abs().shift()).sum())
    rsp = g.groupby('d').r.apply(lambda x: (x[x > 0] ** 2).sum()); rsn = g.groupby('d').r.apply(lambda x: (x[x < 0] ** 2).sum())
    ret = np.log(cl).diff()
    X = pd.DataFrame({'rv': rv, 'bv': bv, 'rsp': rsp, 'rsn': rsn, 'ret': ret}).dropna(subset=['rv']); X['tk'] = tk
    s20 = X.ret.rolling(20).std().shift(1); X['big'] = (X.ret.abs() > 2 * s20); X['bigup'] = X.ret > 2 * s20
    lr = np.log(X.rv.clip(lower=1e-8)); X['lr'] = lr; X['lw'] = lr.rolling(5).mean(); X['lm'] = lr.rolling(22).mean()
    X['lrsp'] = np.log(X.rsp.clip(lower=1e-9)); X['lrsn'] = np.log(X.rsn.clip(lower=1e-9)); X['jmp'] = np.log((X.rv - X.bv).clip(lower=1e-9))
    X['y'] = lr.shift(-1); X['base'] = lr.rolling(252, min_periods=120).median()
    X['rsplus'] = (X.rsp.rolling(5).sum() / X.rv.rolling(5).sum()); X['jumpy'] = X.big.rolling(60).sum()
    gj = pd.Series(np.nan, index=X.index); rr = X.ret.dropna() * 100
    for Y in range(2019, 2027):
        tr = rr[(rr.index < pd.Timestamp(f'{Y}-01-01')) & (rr.index >= pd.Timestamp(f'{Y - 3}-01-01'))]
        if len(tr) < 400: continue
        try: p = arch_model(tr, mean='Constant', vol='GARCH', p=1, o=1, q=1, dist='t').fit(disp='off').params
        except Exception: continue
        mu, w, a, gm, b = p['mu'], p['omega'], p['alpha[1]'], p['gamma[1]'], p['beta[1]']; lrv = w / max(1 - a - gm / 2 - b, 1e-3)
        s2 = tr.var(); seq = rr[rr.index >= pd.Timestamp(f'{Y - 1}-10-01')]
        for d_, x in seq.items():
            if d_ >= pd.Timestamp(f'{Y}-01-01') and d_ < pd.Timestamp(f'{Y + 1}-01-01'): gj[d_] = np.sqrt(s2 / lrv)    # sigma for day d_ known at d_-1 close
            e = x - mu; s2 = w + (a + gm * (e < 0)) * e * e + b * s2
    X['gjr'] = gj; DD.append(X); print(tk, end=' ', flush=True)
DD = pd.concat(DD).reset_index().rename(columns={'index': 'd'})
F = ['lr', 'lw', 'lm']; DD['yr'] = DD.d.dt.year
for nm, cols in (('har', F), ('harrs', ['lrsp', 'lrsn', 'lw', 'lm']), ('harj', F + ['jmp'])):
    DD[nm] = np.nan
    for Y in range(2019, 2027):
        tr = (DD.yr < Y) & DD[cols + ['y']].notna().all(axis=1); te = (DD.yr == Y) & DD[cols].notna().all(axis=1)
        if tr.sum() < 1000: continue
        A_ = np.c_[np.ones(tr.sum()), DD.loc[tr, cols].values]; beta = np.linalg.lstsq(A_, DD.loc[tr, 'y'].values, rcond=None)[0]
        DD.loc[te, nm] = np.c_[np.ones(te.sum()), DD.loc[te, cols].values] @ beta - DD.loc[te, 'base']    # forecast vs normal = expected expansion
DD = DD.sort_values(['tk', 'd']); K = ['har', 'harrs', 'harj', 'gjr', 'rsplus', 'jumpy']
for c in K: DD[c] = DD.groupby('tk')[c].shift(0)                                       # har* use day t to forecast t+1: shift to entry day below
Ls['d'] = Ls.entry.dt.normalize()
key = DD[['tk', 'd'] + K].copy(); key['d'] = key.groupby('tk').d.shift(-1)              # day-t forecast attaches to the next session (entry day)
key.loc[:, 'gjr'] = DD.gjr.values                                                      # gjr already indexed to the day it applies to
key['d'] = np.where(key.d.isna(), DD.d, key.d); key['d'] = pd.to_datetime(key.d)
gjk = DD[['tk', 'd', 'gjr']]
Ls = Ls.merge(key.drop(columns='gjr'), on=['tk', 'd'], how='left').merge(gjk, on=['tk', 'd'], how='left')
TK = taken(Ls).merge(Ls[['tk', 'entry'] + K], on=['tk', 'entry']); TK['h'] = np.where(TK.entry.dt.year <= 2021, '2019-21', '2022-26')
print('\n\nA) trades the book takes: avg trade % by quintile (q1 low forecast vol expansion ... q5 high)')
for c in K:
    q = pd.qcut(TK[c].rank(method='first'), 5, labels=False) + 1; t = (TK.groupby([q, 'h']).ret.mean() * 100).unstack('h')
    print(f'  {c:<7} ' + '  '.join(f'q{i}: {t.loc[i, "2019-21"]:+.2f}/{t.loc[i, "2022-26"]:+.2f}' for i in t.index) + f'   (cover {TK[c].notna().mean():.0%})', flush=True)
pr = lambda s: s.groupby(Ls.entry.dt.year).rank(pct=True)
rng = np.random.default_rng(73); DR = [rng.random(len(Ls)) < .05 if i else np.zeros(len(Ls), bool) for i in range(41)]
def ev(key_):
    Z0 = Ls.copy(); Z0['ratio'] = key_.fillna(key_.median()).values
    return np.array([[v_ for cd in stat(dret(simw(Z0[~dl])).cumsum()) for v_ in (cd[0], cd[1], cd[0] / -cd[1])] for dl in DR])
def line(nm, a, b=None):
    m = np.median(a[1:], axis=0); s = f'  {nm:<28} ret {m[0]:+.1%} / {m[3]:+.1%}   DD {m[1]:.0%} / {m[4]:.0%}   ret/DD {m[2]:.2f} / {m[5]:.2f}'
    if b is not None: s += f'   beats live {(a[1:, 2] > b[1:, 2]).mean():.0%} / {(a[1:, 5] > b[1:, 5]).mean():.0%}'
    print(s, flush=True)
print('\nB) slot tie-break, 40 matched runs, 2019-21 / 2022-26'); B0 = ev(Ls.ratio); line('live (ratio)', B0)
for c in K:
    line(f'{c} high first', ev(Ls[c]), B0); line(f'ratio + {c}', ev(pr(Ls.ratio) + pr(Ls[c])), B0)
