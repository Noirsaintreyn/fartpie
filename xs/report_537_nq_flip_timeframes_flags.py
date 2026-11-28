"""Report 537 (Nov 28 2026): Rey: "my nq prop model, using nq negative gamma flips, test it with PCV SKW CHM and whatever else for more trades".
DECLARED BEFORE RESULTS. Model = NQ vector flips (state machine, 1:1 stop-at-vector / target same distance, cost 1 pt, one trade at a time, 09:35-15:30, same engine as reports 230/231 vx_trades.vx) on 15m / 30m / 1H / 2H bars. Features = QQQ prior-close (shifted one day) own-252d ranks top third: PCV (put/call vol), SKW (skew), CHM (charm), VAN (vanna), DEX, IVH (ATM IV30), plus TERM inverted (short IV > long IV) and NEG (net gamma < 0).
Arms per timeframe: ALL; NEG (the model); NFLX = number of flags among PCV/SKW/CHM/VAN/DEX/IVH/TERM; ADDEDK = flip on a NON-neg day with >=K flags (K=2,3) -> the extra trades; WIDE = NEG | (not NEG & >=2 flags); TRIPLE = NEG & >=2 flags (selection).
Question: which timeframe/arm gives MORE trades than NEG-1H while keeping edge (same sign, t>=2 both eras, n>=30 per era; many cells so t~3 is the real bar). TRAIN 2019-21 | TEST 2018 + 2022-26. Net pts/trade, day-clustered t, trades/yr.
"""
import sys, re, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs'); base = open('xs/vx_trades.py').read()
Q = pd.read_csv('/mnt/user-data/uploads/fartpie/data/theta_feat2/QQQ.csv', parse_dates=['date']).set_index('date').sort_index()
rk = lambda s: s.rolling(252, min_periods=60).apply(lambda a: (a[:-1] < a[-1]).mean() if len(a) > 1 else np.nan, raw=True)
F = pd.DataFrame(index=Q.index); F['gex'] = Q.gex; F['term'] = Q.term
for c, nm in (('pc_vol', 'PCV'), ('skew', 'SKW'), ('charm', 'CHM'), ('vanna', 'VAN'), ('dex', 'DEX'), ('iv30', 'IVH')): F[nm] = rk(Q[c]) >= 2 / 3
F = F.shift(1); F.index = F.index.normalize(); F = F.reset_index().rename(columns={'date': 'day'})
def cell(S):
    if len(S) < 5: return f'n{len(S):4d}  (few)       '
    dm = S.groupby('day').pts.sum(); t = dm.mean() / (dm.std() / np.sqrt(len(dm))) if len(dm) > 2 and dm.std() > 0 else np.nan; return f'n{len(S):4d} {S.pts.mean():+6.1f} (t{t:+.1f}) w{(S.pts>0).mean()*100:2.0f}%'
for freq in ('15min', '30min', '60min', '120min'):
    ns = {}; exec(base.replace("'60min'", f"'{freq}'").split("if __name__")[0], ns); T = ns['vx'](); T['yr'] = T.day.dt.year
    T = T.merge(F, on='day', how='left'); T = T[T.gex.notna()].copy(); T['NEG'] = T.gex < 0; T['TERMI'] = T.term < 0
    fl = ['PCV', 'SKW', 'CHM', 'VAN', 'DEX', 'IVH', 'TERMI']; T['nfl'] = T[fl].fillna(False).astype(int).sum(axis=1); tr = T.yr.between(2019, 2021); te = ~tr; yrs = T.yr.nunique()
    arms = [('ALL', T.NEG | ~T.NEG), ('NEG (the model)', T.NEG), ('NEG & >=2 flags', T.NEG & (T.nfl >= 2)), ('NEG & <2 flags', T.NEG & (T.nfl < 2)), ('ADDED not-NEG & >=2', ~T.NEG & (T.nfl >= 2)), ('ADDED not-NEG & >=3', ~T.NEG & (T.nfl >= 3)), ('WIDE NEG|>=2', T.NEG | (T.nfl >= 2)), ('WIDE NEG|>=3', T.NEG | (T.nfl >= 3))] + [(f, T[f].fillna(False).astype(bool)) for f in fl]
    print(f'\n=== {freq}: {len(T)} trades ({len(T)/yrs:.0f}/yr) | net pts/trade TRAIN 2019-21 | TEST 2018+2022-26 | /yr')
    for lab, m in arms: print(f'{lab:22} | {cell(T[m & tr])} | {cell(T[m & te])} | {m.sum()/yrs:5.0f}')
