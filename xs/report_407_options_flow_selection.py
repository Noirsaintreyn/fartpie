"""Nov 23 2026 report 407 (stage 2 of the stock-analysis model): option flow / positioning features as STOCK SELECTORS, 21-trading-day forward excess return vs SPY (open t+1 -> open t+22), monthly decision.
Data: gex/<TK>.csv (net_gex, atm_iv, call_vol, put_vol; 27 names), oi_daily/<TK>.csv (put/call volume and OI, opening-position counts p_dpos/c_dpos; 35 names).  Eligible (tk, month) pairs = PIT S&P names from report 406.
DECLARED directions: P1 put share of 5d option volume (LOW better); P2 opening-position imbalance 5d = sum(c_dpos - p_dpos)/sum(c_dpos + p_dpos) (HIGH better); P3 put share of open interest (LOW better); P4 own net-GEX z vs prior 126d (LOWER better); P5 ATM IV minus 20d realised vol (HIGHER better).
Reference layers on the same footing: F (fundamentals percentile), ratio, V (vector bull), R (vol relief).  Metric: monthly Spearman IC (min 8 names), mean IC and t, train 2019-21 | test 2018H2+2022-26; PASS = declared-direction sign in both halves and pooled |t| >= 2.
Stage 3 (not run here): vanna / charm / vega aggregated from the theta chains."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; X = '/mnt/user-data/uploads/fartpie/xs/ext/'
T = pd.read_parquet(SP + 'stock406.parquet')[['tk', 't', 'Fp', 'ratio', 'V', 'R', 'half']]
D1 = pd.concat([pd.read_parquet(SP + 'lse_daily_54.parquet'), pd.read_parquet(SP + 'lse_daily_pool.parquet')]).drop_duplicates(['symbol', 'ts']); D1['day'] = pd.to_datetime(D1.ts).dt.normalize(); PX = {tk: g.sort_values('day').set_index('day') for tk, g in D1.groupby('symbol')}
MD = pd.read_parquet(SP + 'macro_daily.parquet'); MD['day'] = pd.to_datetime(MD.ts).dt.normalize(); spy = MD[MD.symbol == 'SPY'].set_index('day').sort_index().open
def fwd(tk, t):
    g = PX[tk]; 
    if t not in g.index: return np.nan
    p = g.index.get_loc(t)
    if p + 23 >= len(g): return np.nan
    a, b = g.index[p + 1], g.index[p + 22]; return g.open.iloc[p + 22] / g.open.iloc[p + 1] - 1 - (spy.get(b, np.nan) / spy.get(a, np.nan) - 1)
T['y'] = [fwd(a, b) for a, b in zip(T.tk, T.t)]
G = {}; O = {}
import os
for f in os.listdir(X + 'gex'): G[f[:-4]] = pd.read_csv(X + 'gex/' + f, parse_dates=['date']).set_index('date').sort_index()
for f in os.listdir(X + 'oi_daily'): O[f[:-4]] = pd.read_csv(X + 'oi_daily/' + f, parse_dates=['date']).set_index('date').sort_index()
rows = []
for tk, t in zip(T.tk, T.t):
    r = dict(tk=tk, t=t)
    o = O.get(tk)
    if o is not None and t in o.index:
        w = o.loc[:t].tail(5); r['P1'] = w.put_vol.sum() / (w.put_vol.sum() + w.call_vol.sum()); r['P2'] = (w.c_dpos.sum() - w.p_dpos.sum()) / max(w.c_dpos.sum() + w.p_dpos.sum(), 1); r['P3'] = o.put_oi.loc[t] / (o.put_oi.loc[t] + o.call_oi.loc[t])
    g = G.get(tk)
    if g is not None and t in g.index:
        h = g.loc[:t].tail(127); prior = h.net_gex.iloc[:-1]
        if len(prior) > 60 and prior.std() > 0: r['P4'] = (h.net_gex.iloc[-1] - prior.mean()) / prior.std()
        c = PX[tk].close.loc[:t].tail(21); rv = np.log(c).diff().std() * np.sqrt(252) if len(c) > 15 else np.nan; r['P5'] = g.atm_iv.loc[t] - rv
    rows.append(r)
F = T.merge(pd.DataFrame(rows), on=['tk', 't']); F.to_parquet(SP + 'opt407.parquet')
DIR = {'P1': -1, 'P2': 1, 'P3': -1, 'P4': -1, 'P5': 1, 'Fp': 1, 'ratio': 1, 'V': 1, 'R': 1}
NAME = {'P1': 'P1 put share of 5d volume', 'P2': 'P2 opening-position imbalance', 'P3': 'P3 put share of OI', 'P4': 'P4 own net-GEX z', 'P5': 'P5 IV - 20d RV', 'Fp': 'F fundamentals', 'ratio': 'ratio (attention)', 'V': 'V vector bull', 'R': 'R vol relief'}
print(f"{'signal (declared dir)':34s}{'months':>7s}{'names/mo':>9s} | IC train (t) | IC test (t) | pooled t | top-bot tercile excess pp train | test")
for k, dr in DIR.items():
    ic = {}; tb = {}
    for t, g in F.groupby('t'):
        g = g[g[k].notna() & g.y.notna()]
        if len(g) < 8 or g[k].nunique() < 3: continue
        ic[t] = spearmanr(g[k], g.y)[0] * dr; q = g[k].rank(pct=True); tb[t] = (g.y[q >= 2 / 3].mean() - g.y[q <= 1 / 3].mean()) * dr * 100 if (q >= 2 / 3).any() and (q <= 1 / 3).any() else np.nan
    ic = pd.Series(ic); tb = pd.Series(tb); h = lambda s, n: s[[((d >= pd.Timestamp('2019-01-01') and d < pd.Timestamp('2022-01-01')) if n == 'train' else (d < pd.Timestamp('2019-01-01') or d >= pd.Timestamp('2022-01-01'))) for d in s.index]]
    f = lambda s: (s.mean(), s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 3 else np.nan)
    a, b, c = f(h(ic, 'train')), f(h(ic, 'test')), f(ic); okk = np.sign(a[0]) == np.sign(b[0]) and a[0] > 0 and abs(c[1]) >= 2
    print(f"{NAME[k]:34s}{len(ic):7d}{F[F[k].notna()].groupby('t').size().mean():9.1f} | {a[0]:+.3f} ({a[1]:+.1f}) | {b[0]:+.3f} ({b[1]:+.1f}) | {c[1]:+.1f} | {h(tb,'train').mean():+5.2f} | {h(tb,'test').mean():+5.2f}  {'PASS' if okk else ''}")
