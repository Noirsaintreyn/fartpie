"""Nov 2 2026: informed-options-trading rankers with free data (opt_features: ATM call / put IV per stock, 2019-02+, ~half of trades covered).
 civ_piv5 / civ_piv21   change in call IV minus change in put IV over 5 / 21 trading days (An, Ang, Bali & Cakici 2014: rising call IV = informed buying)
 ivprem                 log(ATM IV / HAR-forecast realized vol) - options pricing more movement than history predicts
 ivprem_noearn          same, only when no earnings report within the next 10 trading days (otherwise neutral) - 'someone knows' proxy
All known at the prior close. Avg trade by quintile (book's trades), then as the slot tie-break (alone / + ratio), 40 matched runs."""
import os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/fartpie/xs/stress_budget.py').read().split("rng = np.random.default_rng(43)")[0].replace("print(", "(lambda *a, **k: None)("))
Ls = Ls.reset_index(drop=True); Ls['d'] = Ls.entry.dt.normalize()
DD = []
for tk in Ls.tk.unique():
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    if not f: continue
    H = pd.read_parquet(f[0]); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 570) & (m < 960)].drop_duplicates('dt').sort_values('dt'); H['d'] = H.dt.dt.normalize()
    r = np.log(H.close).diff(); r[H.d != H.d.shift()] = np.nan; g = pd.DataFrame({'r': r, 'd': H.d}).dropna()
    cl = H.groupby('d').close.last(); op = H.groupby('d').open.first(); on = np.log(op / cl.shift())
    rv = (g.groupby('d').r.apply(lambda x: (x ** 2).sum()) + on.reindex(g.d.unique()).fillna(0) ** 2).clip(lower=1e-8)
    X = pd.DataFrame({'lr': np.log(rv)}); X['lw'] = X.lr.rolling(5).mean(); X['lm'] = X.lr.rolling(22).mean(); X['y'] = X.lr.shift(-1); X['tk'] = tk; DD.append(X)
DD = pd.concat(DD).reset_index().rename(columns={'index': 'd'}); DD['yr'] = DD.d.dt.year; DD['fc'] = np.nan; cols = ['lr', 'lw', 'lm']
for Y in range(2019, 2027):
    tr = (DD.yr < Y) & DD[cols + ['y']].notna().all(axis=1); te = (DD.yr == Y) & DD[cols].notna().all(axis=1)
    if tr.sum() < 1000: continue
    b = np.linalg.lstsq(np.c_[np.ones(tr.sum()), DD.loc[tr, cols].values], DD.loc[tr, 'y'].values, rcond=None)[0]
    DD.loc[te, 'fc'] = np.c_[np.ones(te.sum()), DD.loc[te, cols].values] @ b
DD['harvol'] = np.sqrt(252 * np.exp(DD.fc))                                           # annualized vol forecast for the NEXT session
DD = DD.sort_values(['tk', 'd']); DD['d_next'] = DD.groupby('tk').d.shift(-1)
O = pd.read_csv('/mnt/user-data/uploads/fartpie/xs/ext/opt_features.csv', parse_dates=['date']).rename(columns={'sym': 'tk', 'date': 'd'}).sort_values(['tk', 'd'])
O['civ'] = O.iv_c50; O['piv'] = O.iv_p50; O['iv'] = (O.iv_c50 + O.iv_p50) / 2
for n in (5, 21): O[f'civ_piv{n}'] = O.groupby('tk').civ.diff(n) - O.groupby('tk').piv.diff(n)
O = O[['tk', 'd', 'iv', 'civ_piv5', 'civ_piv21']]
def asof(Z, F, cols, on_col):                                                         # attach the latest value known at the prior close
    out = []
    for tk, g in Z.groupby('tk'):
        f = F[F.tk == tk].dropna(subset=[on_col]).sort_values(on_col)
        if not len(f): out.append(pd.DataFrame(index=g.index, columns=cols)); continue
        k = np.searchsorted(f[on_col].values, (g.d - pd.Timedelta(days=0)).values, side='left') - 1   # strictly before the entry day
        v = f[cols].values; res = np.where((k >= 0)[:, None], v[np.clip(k, 0, None)], np.nan); out.append(pd.DataFrame(res, index=g.index, columns=cols))
    return pd.concat(out).sort_index()
Ls[['iv', 'civ_piv5', 'civ_piv21']] = asof(Ls, O, ['iv', 'civ_piv5', 'civ_piv21'], 'd').astype(float).values
hv = DD[['tk', 'd_next', 'harvol']].dropna().rename(columns={'d_next': 'd'}); hv['d'] = hv.d + pd.Timedelta(hours=1)  # forecast made at d-1 close for day d
Ls['harvol'] = asof(Ls, hv, ['harvol'], 'd').astype(float).values.ravel()
Ls['ivprem'] = np.log(Ls.iv / Ls.harvol)
src2 = open('/home/claude/fartpie/xs/pre_earn_flip.py').read(); L = Ls; exec(src2[src2.index('k = np.full'):src2.index("L['k'] = k")]); Ls['k'] = k
Ls['ivprem_noearn'] = np.where(Ls.k > 10, Ls.ivprem, np.nan)
K = ['civ_piv5', 'civ_piv21', 'ivprem', 'ivprem_noearn']
TK = taken(Ls).merge(Ls[['tk', 'entry'] + K], on=['tk', 'entry']); TK['h'] = np.where(TK.entry.dt.year <= 2021, '2019-21', '2022-26')
print('A) trades the book takes: avg trade % by quintile (q1 low ... q5 high), covered trades only')
for c in K:
    T2 = TK[TK[c].notna()]; q = pd.qcut(T2[c].rank(method='first'), 5, labels=False) + 1; t = (T2.groupby([q, 'h']).ret.mean() * 100).unstack('h')
    print(f'  {c:<14} ' + '  '.join(f'q{i}: {t.loc[i, "2019-21"]:+.2f}/{t.loc[i, "2022-26"]:+.2f}' for i in t.index) + f'   (cover {TK[c].notna().mean():.0%})', flush=True)
pr = lambda s: s.groupby(Ls.entry.dt.year).rank(pct=True)
rng = np.random.default_rng(79); DR = [rng.random(len(Ls)) < .05 if i else np.zeros(len(Ls), bool) for i in range(41)]
def ev(key_):
    Z0 = Ls.copy(); Z0['ratio'] = key_.fillna(key_.median()).values
    return np.array([[v_ for cd in stat(dret(simw(Z0[~dl])).cumsum()) for v_ in (cd[0], cd[1], cd[0] / -cd[1])] for dl in DR])
def line(nm, a, b=None):
    m = np.median(a[1:], axis=0); s = f'  {nm:<30} ret {m[0]:+.1%} / {m[3]:+.1%}   DD {m[1]:.0%} / {m[4]:.0%}   ret/DD {m[2]:.2f} / {m[5]:.2f}'
    if b is not None: s += f'   beats live {(a[1:, 2] > b[1:, 2]).mean():.0%} / {(a[1:, 5] > b[1:, 5]).mean():.0%}'
    print(s, flush=True)
print('\nB) slot tie-break, 40 matched runs, 2019-21 / 2022-26'); B0 = ev(Ls.ratio); line('live (ratio)', B0)
for c in K:
    line(f'{c} high first', ev(Ls[c]), B0); line(f'ratio + {c}', ev(pr(Ls.ratio) + pr(Ls[c]).fillna(.5)), B0)
