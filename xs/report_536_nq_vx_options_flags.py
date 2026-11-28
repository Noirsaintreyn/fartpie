"""Report 536 (Nov 28 2026): Rey: add PCV / skew / charm and the other option features to the NQ neg-gamma vector leg "for more trades".
DECLARED BEFORE RESULTS. Leg = the 1H 1:1 vector trades of reports 230/231 (vx_trades.vx(): entry next 5m open after the 1H flip, stop = vector level, target 1:1, flat 15:55, cost 1 pt, one at a time, 09:35-15:30). Features = QQQ daily option features (data/theta_feat2/QQQ.csv, built from OI x EOD) ranked on own trailing 252d and shifted ONE trading day (prior close known): NEG net gamma<0; PCV put/call volume rank top third; SKW skew rank top third; CHM charm rank top third; VAN vanna top third; IVH ATM IV rank top third. nfl = #flags among NEG/PCV/SKW/CHM.
Arms: ALL; NEG (230's gate); each flag alone; ANY (>=1 of NEG/PCV/SKW/CHM); NFL2 (>=2); NFL0 (none, control). Question 1 (more trades): does widening from NEG to ANY keep per-trade edge? Question 2: do the extra trades (ANY minus NEG) carry edge on their own?
Split: TRAIN 2019-21 | TEST 2018+2022-26 (Nov 17). Metric: net pts/trade after 1 pt cost, day-clustered t, win%, trades/yr. PASS for an arm: same sign and |t|>=2 both eras with n>=30, and the ADDED trades (arm minus NEG) positive in both eras. ~10 arms -> read t~3 as strong.
"""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs'); from vx_trades import vx
T = vx(); T['yr'] = T.day.dt.year
Q = pd.read_csv('/mnt/user-data/uploads/fartpie/data/theta_feat2/QQQ.csv', parse_dates=['date']).set_index('date').sort_index()
rk = lambda s: s.rolling(252, min_periods=60).apply(lambda a: (a[:-1] < a[-1]).mean() if len(a) > 1 else np.nan, raw=True)
F = pd.DataFrame(index=Q.index); F['gex'] = Q.gex
for c in ('pc_vol', 'skew', 'charm', 'vanna', 'iv30'): F['r_' + c] = rk(Q[c])
F = F.shift(1); F.index = F.index.normalize(); T = T.merge(F.reset_index().rename(columns={'date': 'day'}), on='day', how='left'); T = T[T.gex.notna()].copy()
T['NEG'] = T.gex < 0; T['PCV'] = T.r_pc_vol >= 2 / 3; T['SKW'] = T.r_skew >= 2 / 3; T['CHM'] = T.r_charm >= 2 / 3; T['VAN'] = T.r_vanna >= 2 / 3; T['IVH'] = T.r_iv30 >= 2 / 3
T['nfl'] = T.NEG.astype(int) + T.PCV.astype(int) + T.SKW.astype(int) + T.CHM.astype(int); T['ANY'] = T.nfl >= 1; T['NFL2'] = T.nfl >= 2; T['NFL0'] = T.nfl == 0
yrs = T.yr.nunique(); tr = T.yr.between(2019, 2021); te = ~tr
def cell(S):
    if len(S) < 5: return f'n{len(S):4d}  (few)'
    dm = S.groupby('day').pts.sum(); t = dm.mean() / (dm.std() / np.sqrt(len(dm))) if len(dm) > 2 and dm.std() > 0 else np.nan; return f'n{len(S):4d} {S.pts.mean():+6.1f} (t{t:+.1f}) win{(S.pts>0).mean()*100:3.0f}%'
print(f'VX trades with QQQ features: {len(T)} ({len(T)/yrs:.0f}/yr), years {T.yr.min()}-{T.yr.max()}\nnet pts/trade | TRAIN 2019-21 | TEST 2018+2022-26 | trades/yr')
arms = [('ALL', T.NEG | ~T.NEG), ('NEG', T.NEG), ('POS (not NEG)', ~T.NEG), ('PCV', T.PCV), ('SKW', T.SKW), ('CHM', T.CHM), ('VAN', T.VAN), ('IVH', T.IVH), ('ANY(N/P/S/C)', T.ANY), ('NFL2+', T.NFL2), ('NFL0 none', T.NFL0), ('ADDED = ANY & not NEG', T.ANY & ~T.NEG)]
for lab, m in arms: print(f'{lab:22} | {cell(T[m & tr])} | {cell(T[m & te])} | {m.sum()/yrs:5.0f}')
print('\nlongs only:')
for lab, m in arms[:2] + arms[8:9] + arms[-1:]: print(f'{lab:22} | {cell(T[m & tr & (T.dir == 1)])} | {cell(T[m & te & (T.dir == 1)])}')
print('shorts only:')
for lab, m in arms[:2] + arms[8:9] + arms[-1:]: print(f'{lab:22} | {cell(T[m & tr & (T.dir == -1)])} | {cell(T[m & te & (T.dir == -1)])}')
