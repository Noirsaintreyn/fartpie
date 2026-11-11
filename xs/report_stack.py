"""STACK: hedging (greeks) -> map (AMT/TPO) -> environment (ICT); orderflow not needed. Nov 11 2026.
Layer H (what hedging flows push): short-gamma regime (GEX<0 = trend-friendly) + vanna flow aligned (IV falling & net vanna>0 => dealers buy; IV rising & vanna>0 => dealers sell; 1-day IV change from the prior row). 0-2.
Layer M (where/how price reacted): price beyond developing POC, price beyond prior value area (in trade dir), initial-balance break with trade. 0-3.
Layer E (environment): sweep with, structure shift with, displacement with, FVG with. 0-4.
Layer satisfied: H>=1, M>=2, E>=3 (declared before results). Tier = number of layers satisfied (0-3). Also single-layer comparisons. Both eras."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; U = '/mnt/user-data/uploads/fartpie/'
g1 = pd.read_csv(U + 'options_greeks_daily_QQQ.csv', parse_dates=['date']).set_index('date'); g2 = pd.read_csv(U + 'options_greeks_daily_v2_QQQ.csv', parse_dates=['date']).set_index('date')
OP = pd.DataFrame({'gex': g1.call_gex - g1.put_gex, 'vanna': g2.net_vanna, 'charm': g2.net_charm, 'iv1': g1.atm_iv / g1.atm_iv.shift(1) - 1}).shift(1)
def tstat(x): x = np.asarray(x, float); return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 8 and x.std() > 0 else np.nan
def line(lab, s):
    a, b = s[s.era == 'A'].pts, s[s.era == 'B'].pts
    ok = len(a) >= 30 and len(b) >= 30 and np.sign(a.mean()) == np.sign(b.mean()) and abs(tstat(a)) >= 2 and abs(tstat(b)) >= 2
    print(f' {lab:30} n {len(s):4d} win {np.mean(s.pts>0) if len(s) else 0:3.0%} mean {s.pts.mean() if len(s) else 0:+6.1f} | A n{len(a):3d} {a.mean() if len(a) else 0:+6.1f} ({tstat(a):+.1f}) | B n{len(b):3d} {b.mean() if len(b) else 0:+6.1f} ({tstat(b):+.1f}) {"PASS" if ok else ""}')
for nm in ('V15a', 'RA'):
    N = pd.read_pickle(SC + f'narr_{nm}.pkl').set_index(['day', 'm']); N = N[~N.index.duplicated()]
    A = pd.read_pickle(SC + f'amt_{nm}.pkl').set_index(['day', 'm']); A = A[~A.index.duplicated()]
    R = N[['pts', 'dir', 'era', 'n_with', 'mss', 'disp', 'fvg_with', 'score']].join(A[['dev_poc', 'px_val', 'ib_break', 'open_val']], how='inner').reset_index()
    for c in OP.columns: R[c] = OP[c].reindex(R.day).values
    R = R.dropna(subset=['gex', 'vanna', 'iv1']).copy()
    R['vflow'] = R.dir * np.sign(R.vanna) * -np.sign(R.iv1)        # +1 = vanna/IV hedging flow supports the trade direction
    R['H'] = (R.gex < 0).astype(int) + (R.vflow == 1); R['M'] = (R.dev_poc == 1).astype(int) + (R.px_val == 1) + (R.ib_break == 1); R['E'] = (R.n_with >= 1).astype(int) + (R.mss == 1) + (R.disp == 1) + (R.fvg_with == 1)
    R['lH'], R['lM'], R['lE'] = R.H >= 1, R.M >= 2, R.E >= 3; R['tier'] = R.lH.astype(int) + R.lM + R.lE
    print(f'\n######## {nm}: {len(R)} trades, mean {R.pts.mean():+.1f}')
    print('- vanna flow aligned (+1 = hedging flow with trade) :'); [line(f'   {v:+d}', R[R.vflow == v]) for v in (1, -1)]
    print('- layer H (hedging) count 0/1/2:'); [line(f'   H={v}', R[R.H == v]) for v in (0, 1, 2)]
    print('- layer M (map) count 0..3:'); [line(f'   M={v}', R[R.M == v]) for v in (0, 1, 2, 3)]
    print('- layer E (environment) count 0..4:'); [line(f'   E={v}', R[R.E == v]) for v in (0, 1, 2, 3, 4)]
    print('- layers satisfied (H>=1, M>=2, E>=3):'); [line(f'   tier {v}', R[R.tier == v]) for v in (0, 1, 2, 3)]
    for lab, m in (('only H satisfied', R.lH & ~R.lM & ~R.lE), ('only M satisfied', ~R.lH & R.lM & ~R.lE), ('only E satisfied', ~R.lH & ~R.lM & R.lE), ('M & E (no hedging)', ~R.lH & R.lM & R.lE), ('H & M & E', R.lH & R.lM & R.lE), ('H & M (no E)', R.lH & R.lM & ~R.lE), ('H & E (no M)', R.lH & ~R.lM & R.lE)): line('   ' + lab, R[m])
    R.to_pickle(SC + f'stack_{nm}.pkl')
