"""Oct 22 2026: earnings guard with a second opinion. Before an earnings gap, if the trade is up >= X%, sell at the close UNLESS an
earnings model (known before the report) also says 'up'. Models, decided before seeing results:
 beat     : last quarter's EPS surprise > 0          streak : beat the last 3 quarters in a row
 react    : last report's gap-day move was up          nospike: option volume over the last 3 days < 1.5x its 252d median
 calm     : the stock did NOT run up > 10% in the 10 days before the report (run-ups tend to sell the news)
Also prints the gap reaction by each model for all vector-long reports (no book) to see if any model reads the gap at all."""
from collections import deque
import os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('xs/slot_tests.py').read().split('P = [(np.datetime64')[0])
exec('def sim3' + open('xs/equity_models.py').read().split('def sim3')[1].split('STRAT_MED =')[0]); STRAT_MED = 1
exec(open('xs/exit_lab2.py').read().split('BARS = {}')[1].split('VARS =')[0].replace('def bars', 'BARS = {}\ndef bars', 1))
T = T.sort_values('entry').reset_index(drop=True)
TD = pd.DatetimeIndex(sorted(pd.to_datetime(pd.read_parquet('etf_daily/SPY.parquet').datetime).dt.tz_localize(None).dt.normalize().unique()))
R = pd.read_csv('xs/ext/optvol_ratio_sp.csv', parse_dates=['date']).pivot_table(index='date', columns='tk', values='r').sort_index()
SPK = (R.rolling(3).mean() / R.rolling(252, min_periods=60).median().shift(1)).shift(1)   # through the PRIOR close (decision at 15:45)
EVS = {}
for f in glob.glob('xs/ext/earn/*.csv'):
    tk = f.split('/')[-1][:-4]
    if tk.startswith('_'): continue
    try: e = pd.read_csv(f); t = pd.to_datetime(e.iloc[:, 0], utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    except Exception: continue
    G = [d if tt.hour < 12 else TD[min(TD.searchsorted(d, 'right'), len(TD) - 1)] for d, tt in zip(t.dt.normalize(), t)]
    x = pd.DataFrame({'G': pd.DatetimeIndex(G), 'surp': pd.to_numeric(e['Surprise(%)'], errors='coerce')}).dropna(subset=['G']).drop_duplicates('G').sort_values('G').reset_index(drop=True)
    EVS[tk] = x
def model_votes(tk, G, B):
    x = EVS[tk]; past = x[x.G < G]; done = past.dropna(subset=['surp'])
    C = CL[tk] if tk in CL.columns else None
    pre = TD[TD.searchsorted(G) - 1]
    v = {}
    v['beat'] = bool(len(done) and done.surp.iloc[-1] > 0)
    v['streak'] = bool(len(done) >= 3 and (done.surp.iloc[-3:] > 0).all())
    if C is not None and len(past):
        g0 = past.G.iloc[-1]; p0 = C[C.index < g0]; p1 = C[C.index <= g0]
        v['react'] = bool(len(p0) and len(p1) and p1.iloc[-1] > p0.iloc[-1])
        c10 = C[C.index <= pre]; v['calm'] = bool(len(c10) > 11 and c10.iloc[-1] / c10.iloc[-11] - 1 <= .10)
    else: v['react'] = v['calm'] = False
    s = SPK[tk].asof(pre) if tk in SPK.columns else np.nan; v['nospike'] = bool(not (s >= 1.5))
    v['reactdn'] = not v['react']; v['nospike_reactdn'] = v['nospike'] or v['reactdn']
    return v
MODELS = ['beat', 'streak', 'react', 'nospike', 'calm', 'reactdn', 'nospike_reactdn']
ROWS = []; ALT = {}
VARS = [(X, m) for X in (.03, .05) for m in ['none'] + MODELS]
for k in VARS: ALT[k] = (T.ret.values.copy(), T.exit.values.copy())
for tk, g in T.groupby('tk'):
    B = bars(tk)
    if B is None or tk not in EVS: continue
    dt = B.dt.values; o, c = B.open.values, B.close.values; day = B.dt.dt.normalize().values; gset = set(EVS[tk].G.values.astype('datetime64[ns]'))
    for i, e, x in zip(g.index, g.entry.values, g.exit.values):
        a = np.searchsorted(dt, e); b = np.searchsorted(dt, x)
        if a >= len(dt) or dt[a] != e or b >= len(dt) or b <= a: continue
        p0 = o[a]; base = o[b] / p0 - 1
        ends = [j for j in range(a, b) if day[j + 1] != day[j] and np.datetime64(day[j + 1]) in gset]
        if not ends: continue
        j = ends[0]; up = c[j] / p0 - 1; G = pd.Timestamp(day[j + 1]); v = model_votes(tk, G, B)
        nxt = np.searchsorted(day, np.datetime64(G)); gap = o[nxt] / c[j] - 1 if nxt < len(o) else np.nan
        ROWS.append(dict(tk=tk, G=G, up=up, gap=gap, rest=base - up, **v))
        for k in VARS:
            X, m = k
            if up >= X and (m == 'none' or not v[m]):
                r_, x_ = ALT[k]; r_[i] += up - base; x_[i] = dt[j]
E = pd.DataFrame(ROWS); E['per'] = np.where(E.G.dt.year >= 2022, '22-26', '19-21')
print(f'{len(E)} vector trades held into an earnings gap ({(E.up >= .03).sum()} up >= 3%, {(E.up >= .05).sum()} up >= 5% at the close before)\n')
print('gap % (close before -> open after) by model vote, all vector trades into earnings:')
for m in MODELS:
    gg = E.groupby([m, 'per']).gap.agg(['mean', 'count']); gg['mean'] *= 100; print(f'  {m:<8}', '  '.join(f'{"yes" if k[0] else "no "} {k[1]}: {r["mean"]:+.2f}% (n {int(r["count"])})' for k, r in gg.iterrows()))
print('\n  same, only trades up >= 3% before the report:')
F = E[E.up >= .03]
for m in MODELS:
    gg = F.groupby([m, 'per']).gap.agg(['mean', 'count']); gg['mean'] *= 100; print(f'  {m:<8}', '  '.join(f'{"yes" if k[0] else "no "} {k[1]}: {r["mean"]:+.2f}% (n {int(r["count"])})' for k, r in gg.iterrows()))
P = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
def show(nm, Z): print(f'  {nm:<48}' + ''.join(f'{cc:>+9.1%} /{dd:>5.0%}' for cc, dd in (sim3(Z, a, b) for a, b in P)), flush=True)
print(f'\n{"guard: sell before earnings if up >= X, unless":<50}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}')
show('live (never sell)', T)
for k in VARS:
    Z = T.copy(); Z['ret'], Z['exit'] = ALT[k]; show(f'+{k[0]:.0%}, ' + ('no second model' if k[1] == 'none' else f'unless {k[1]} says up'), Z)
