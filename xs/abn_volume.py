"""Nov 2 2026: free stock-side attention - abnormal daily share volume (Gervais, Kaniel & Mingelgrin 2001 high-volume return premium).
 av1  yesterday's RTH share volume / its prior 50-day average      av5  last 5 days' average / prior 50-day average
Known at the prior close. Avg trade by quintile (book's trades), then as tie-break alone / + ratio, 40 matched runs."""
import os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/fartpie/xs/stress_budget.py').read().split("rng = np.random.default_rng(43)")[0].replace("print(", "(lambda *a, **k: None)("))
Ls = Ls.reset_index(drop=True); Ls['d'] = Ls.entry.dt.normalize(); V = []
for tk in Ls.tk.unique():
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    if not f: continue
    H = pd.read_parquet(f[0], columns=['datetime', 'volume']); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = H.dt.dt.hour * 60 + H.dt.dt.minute; v = H[(m >= 570) & (m < 960)].groupby(H.dt.dt.normalize()).volume.sum()
    base = v.rolling(50).mean().shift(1)
    X = pd.DataFrame({'av1': v / base, 'av5': v.rolling(5).mean() / base.shift(4)}).shift(1); X['tk'] = tk; V.append(X.reset_index().rename(columns={'dt': 'd', 'index': 'd'}))
V = pd.concat(V); V['d'] = pd.to_datetime(V.d); Ls = Ls.merge(V, on=['tk', 'd'], how='left')
TK = taken(Ls).merge(Ls[['tk', 'entry', 'av1', 'av5']], on=['tk', 'entry']); TK['h'] = np.where(TK.entry.dt.year <= 2021, '2019-21', '2022-26')
for c in ('av1', 'av5'):
    q = pd.qcut(TK[c].rank(method='first'), 5, labels=False) + 1; t = (TK.groupby([q, 'h']).ret.mean() * 100).unstack('h')
    print(f'  {c}: ' + '  '.join(f'q{i}: {t.loc[i, "2019-21"]:+.2f}/{t.loc[i, "2022-26"]:+.2f}' for i in t.index), flush=True)
pr = lambda s: s.groupby(Ls.entry.dt.year).rank(pct=True)
rng = np.random.default_rng(83); DR = [rng.random(len(Ls)) < .05 if i else np.zeros(len(Ls), bool) for i in range(41)]
def ev(key_):
    Z0 = Ls.copy(); Z0['ratio'] = key_.fillna(key_.median()).values
    return np.array([[v_ for cd in stat(dret(simw(Z0[~dl])).cumsum()) for v_ in (cd[0], cd[1], cd[0] / -cd[1])] for dl in DR])
def line(nm, a, b=None):
    m = np.median(a[1:], axis=0); s = f'  {nm:<22} ret {m[0]:+.1%} / {m[3]:+.1%}   ret/DD {m[2]:.2f} / {m[5]:.2f}'
    if b is not None: s += f'   beats live {(a[1:, 2] > b[1:, 2]).mean():.0%} / {(a[1:, 5] > b[1:, 5]).mean():.0%}'
    print(s, flush=True)
B0 = ev(Ls.ratio); line('live (ratio)', B0)
for c in ('av1', 'av5'): line(f'{c} high first', ev(Ls[c]), B0); line(f'ratio + {c}', ev(pr(Ls.ratio) + pr(Ls[c])), B0)
