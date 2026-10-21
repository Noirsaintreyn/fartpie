"""Oct 21 2026: earnings on the full-S&P screener (ratio >= 1.5, live rules). Yahoo earnings dates + EPS surprise (xs/ext/earn).
Gap day G = report day if before noon ET, else the next trading day (that open carries the reaction).
 E1 skip entries when G falls within the next K trading days   E2 sell at the close before G (frees the slot)
 E3 post-earnings drift: after a beat (surprise > 0 / > 5%), flips in the next 20 trading days get priority; or skip flips after a miss"""
from collections import deque
import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('xs/slot_tests.py').read().split('P = [(np.datetime64')[0])
exec('def sim3' + open('xs/equity_models.py').read().split('def sim3')[1].split('STRAT_MED =')[0]); STRAT_MED = 1
exec(open('xs/exit_lab2.py').read().split('BARS = {}')[1].split('VARS =')[0].replace('def bars', 'BARS = {}\ndef bars', 1))
T = T.sort_values('entry').reset_index(drop=True)
TD = pd.DatetimeIndex(sorted(pd.read_parquet('etf_daily/SPY.parquet').datetime.pipe(pd.to_datetime).dt.tz_localize(None).dt.normalize().unique()))
EV = {}
for f in glob.glob('xs/ext/earn/*.csv'):
    tk = f.split('/')[-1][:-4]
    if tk.startswith('_'): continue
    try: e = pd.read_csv(f)
    except Exception: continue
    if not len(e): continue
    t = pd.to_datetime(e.iloc[:, 0], utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    D = t.dt.normalize(); G = [d if tt.hour < 12 else TD[min(TD.searchsorted(d, 'right'), len(TD) - 1)] for d, tt in zip(D, t)]
    EV[tk] = pd.DataFrame({'G': pd.DatetimeIndex(G), 'surp': pd.to_numeric(e['Surprise(%)'], errors='coerce')}).dropna(subset=['G']).sort_values('G')
print('names with earnings:', len(EV), 'of', T.tk.nunique())
def tdays(a, b): return TD.searchsorted(b) - TD.searchsorted(a)            # trading days from a to b
nextG, crossG, lastG, lastS = [], [], [], []
for tk, e, x in zip(T.tk, T.entry, T.exit):
    E_ = EV.get(tk)
    if E_ is None: nextG.append(np.nan); crossG.append(pd.NaT); lastG.append(np.nan); lastS.append(np.nan); continue
    g = E_.G.values; en = np.datetime64(e.normalize())
    k = np.searchsorted(g, en, 'right') if e.hour * 60 + e.minute > 570 else np.searchsorted(g, en)   # next gap strictly after this entry bar
    ng = g[k] if k < len(g) else None
    nextG.append(tdays(e.normalize(), pd.Timestamp(ng)) if ng is not None else np.nan)
    crossG.append(pd.Timestamp(ng) if ng is not None and pd.Timestamp(ng) + pd.Timedelta('9h30min') <= x else pd.NaT)
    j = k - 1
    lastG.append(tdays(pd.Timestamp(g[j]), e.normalize()) if j >= 0 else np.nan); lastS.append(E_.surp.values[j] if j >= 0 else np.nan)
T['nextG'] = nextG; T['crossG'] = crossG; T['sinceG'] = lastG; T['lsurp'] = lastS
T['per'] = np.where(T.entry.dt.year >= 2022, '22-26', '19-21'); cr = T.crossG.notna()
print('\navg trade % (per period):'); print((T.groupby([cr.rename('holds through earnings'), 'per']).ret.agg(['mean', 'count']).assign(mean=lambda z: z['mean'] * 100)).round(3).to_string())
post = (T.sinceG <= 20)
print((T[post].groupby([np.sign(T.lsurp[post]).rename('last surprise sign'), 'per']).ret.agg(['mean', 'count']).assign(mean=lambda z: z['mean'] * 100)).round(3).to_string())
# E2 exit before earnings: sell at the last bar close before the gap day
r2, x2 = T.ret.values.copy(), T.exit.values.copy()
for i in np.where(cr)[0]:
    B = bars(T.tk[i])
    if B is None: continue
    dt = B.dt.values; a = np.searchsorted(dt, T.entry.values[i]); b = np.searchsorted(dt, T.exit.values[i]); gdt = np.datetime64(T.crossG[i] + pd.Timedelta('9h30min'))
    j = np.searchsorted(dt, gdt) - 1
    if a < len(dt) and j > a and b < len(dt):
        p0 = B.open.values[a]; r2[i] += B.close.values[j] / p0 - B.open.values[b] / p0; x2[i] = dt[j]
P = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
def show(nm, Z, **kw): print(f'  {nm:<48}' + ''.join(f'{cc:>+9.1%} /{dd:>5.0%}' for cc, dd in (sim3(Z, a, b, **kw) for a, b in P)), flush=True)
print(f'\n{"earnings (S&P, ratio >= 1.5, live rules)":<50}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}')
show('live', T)
for K in (1, 2, 3, 5): show(f'E1 skip entries with earnings in next {K} days', T[~(T.nextG <= K)])
Z = T.copy(); Z['ret'], Z['exit'] = r2, x2; show('E2 sell at the close before earnings', Z)
Z = T.copy(); Z['ret'], Z['exit'] = r2, x2; show('E1(2d)+E2', Z[~(Z.nextG <= 2)])
T['pead'] = np.where(post & (T.lsurp > 0), 1, 0) + T.ratio / 100; show('E3 tie-break: recent beat first', T, key='pead')
T['pead5'] = np.where(post & (T.lsurp > 5), 1, 0) + T.ratio / 100; show('E3 tie-break: recent beat > 5% first', T, key='pead5')
show('E3 skip flips within 20d after a miss', T[~(post & (T.lsurp < 0))])
