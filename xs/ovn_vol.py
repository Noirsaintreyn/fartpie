"""Nov 2 2026: the book earns mostly overnight -> rank flips by OVERNIGHT behaviour (all known at the prior close):
 on_share   share of total variance that came from overnight gaps, last 60 days (news-driven names)
 on_har     HAR forecast of log overnight squared gap (1d / 5d / 22d), pooled walk-forward, vs the stock's 1y median (expected overnight expansion)
 on_mom     mean overnight return, last 20 days (Lou, Polk & Skouras 2019: overnight returns persist)
 on_abs5    mean |overnight gap| last 5 days vs last 60 days
Avg trade by quintile (book's trades), then as tie-break alone / + ratio, 40 matched runs."""
import os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/fartpie/xs/stress_budget.py').read().split("rng = np.random.default_rng(43)")[0].replace("print(", "(lambda *a, **k: None)("))
Ls = Ls.reset_index(drop=True); Ls['d'] = Ls.entry.dt.normalize(); F = []
for tk in Ls.tk.unique():
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    if not f: continue
    H = pd.read_parquet(f[0]); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = H.dt.dt.hour * 60 + H.dt.dt.minute; R = H[(m >= 570) & (m < 960)].drop_duplicates('dt').sort_values('dt')
    D = R.groupby(R.dt.dt.normalize()).agg(o=('open', 'first'), c=('close', 'last'))
    on = np.log(D.o / D.c.shift()); it = np.log(D.c / D.o)
    X = pd.DataFrame({'on2': on ** 2, 'it2': it ** 2, 'on': on})
    X['on_share'] = X.on2.rolling(60).sum() / (X.on2.rolling(60).sum() + X.it2.rolling(60).sum())
    lo = np.log(X.on2.clip(lower=1e-8)); X['l1'], X['l5'], X['l22'] = lo, lo.rolling(5).mean(), lo.rolling(22).mean(); X['y'] = lo.shift(-1); X['base'] = lo.rolling(252, min_periods=120).median()
    X['on_mom'] = X.on.rolling(20).mean(); X['on_abs5'] = X.on.abs().rolling(5).mean() / X.on.abs().rolling(60).mean(); X['tk'] = tk
    F.append(X.reset_index().rename(columns={'index': 'd', 'dt': 'd'}))
F = pd.concat(F); F['yr'] = F.d.dt.year; F['on_har'] = np.nan; cols = ['l1', 'l5', 'l22']
for Y in range(2019, 2027):
    tr = (F.yr < Y) & F[cols + ['y']].notna().all(axis=1); te = (F.yr == Y) & F[cols].notna().all(axis=1)
    if tr.sum() < 1000: continue
    b = np.linalg.lstsq(np.c_[np.ones(tr.sum()), F.loc[tr, cols].values], F.loc[tr, 'y'].values, rcond=None)[0]
    F.loc[te, 'on_har'] = np.c_[np.ones(te.sum()), F.loc[te, cols].values] @ b - F.loc[te, 'base']
K = ['on_share', 'on_har', 'on_mom', 'on_abs5']
F = F.sort_values(['tk', 'd']); F['d_next'] = F.groupby('tk').d.shift(-1)              # values at close of d apply to entries on the next session
G = F.dropna(subset=['d_next'])[['tk', 'd_next'] + K].rename(columns={'d_next': 'd'})
Ls = Ls.merge(G, on=['tk', 'd'], how='left')
TK = taken(Ls).merge(Ls[['tk', 'entry'] + K], on=['tk', 'entry']); TK['h'] = np.where(TK.entry.dt.year <= 2021, '2019-21', '2022-26')
print('A) trades the book takes: avg trade % by quintile (q1 low ... q5 high)')
for c in K:
    T2 = TK[TK[c].notna()]; q = pd.qcut(T2[c].rank(method='first'), 5, labels=False) + 1; t = (T2.groupby([q, 'h']).ret.mean() * 100).unstack('h')
    print(f'  {c:<9} ' + '  '.join(f'q{i}: {t.loc[i, "2019-21"]:+.2f}/{t.loc[i, "2022-26"]:+.2f}' for i in t.index) + f'   (cover {TK[c].notna().mean():.0%})', flush=True)
pr = lambda s: s.groupby(Ls.entry.dt.year).rank(pct=True)
rng = np.random.default_rng(103); DR = [rng.random(len(Ls)) < .05 if i else np.zeros(len(Ls), bool) for i in range(41)]
def ev(key_):
    Z0 = Ls.copy(); Z0['ratio'] = key_.fillna(key_.median()).values
    return np.array([[v_ for cd in stat(dret(simw(Z0[~dl])).cumsum()) for v_ in (cd[0], cd[1], cd[0] / -cd[1])] for dl in DR])
def line(nm, a, b=None):
    m = np.median(a[1:], axis=0); s = f'  {nm:<22} ret {m[0]:+.1%} / {m[3]:+.1%}   DD {m[1]:.0%} / {m[4]:.0%}   ret/DD {m[2]:.2f} / {m[5]:.2f}'
    if b is not None: s += f'   beats live {(a[1:, 2] > b[1:, 2]).mean():.0%} / {(a[1:, 5] > b[1:, 5]).mean():.0%}'
    print(s, flush=True)
print('\nB) tie-break, 40 matched runs, 2019-21 / 2022-26'); B0 = ev(Ls.ratio); line('live (ratio)', B0)
for c in K: line(f'{c} high first', ev(Ls[c]), B0); line(f'ratio + {c}', ev(pr(Ls.ratio) + pr(Ls[c]).fillna(.5)), B0)
