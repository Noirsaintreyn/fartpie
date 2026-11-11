"""OPTIONS data (QQQ daily greeks 2011-2025 + NQ gamma walls) as regime / direction support for V15a and Rule A. Nov 11 2026.
All features from the PREVIOUS trading day's row (EOD data is not known intraday). Directional ones are aligned (+1 = supports trade); regime ones are bucketed in terciles cut on 2014-21.
Features: net GEX (call-put) sign/tercile, GEX z (vs 60d), put/call volume ratio, put/call OI ratio, ATM IV level + 5d change, 25d skew, net DEX, net vanna, net charm (signs, aligned for dex/vanna/charm), distance of entry price to NQ call/put wall (aligned)."""
import pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; U = '/mnt/user-data/uploads/fartpie/'
g1 = pd.read_csv(U + 'options_greeks_daily_QQQ.csv', parse_dates=['date']).set_index('date'); g2 = pd.read_csv(U + 'options_greeks_daily_v2_QQQ.csv', parse_dates=['date']).set_index('date')
w = pd.read_csv(U + 'gex_walls_nq.csv', parse_dates=['date']).set_index('date')
O = g1.join(g2[['net_vanna', 'net_charm']]); O['gex'] = O.call_gex - O.put_gex; O['gexz'] = (O.gex - O.gex.rolling(60).mean()) / O.gex.rolling(60).std()
O['pcv'] = O.put_volume / O.call_volume; O['pco'] = O.put_oi / O.call_oi; O['iv'] = O.atm_iv; O['ivchg'] = O.atm_iv / O.atm_iv.shift(5) - 1; O['dex'] = O.call_dex + O.put_dex
O = O[['gex', 'gexz', 'pcv', 'pco', 'iv', 'ivchg', 'skew_25d', 'dex', 'net_vanna', 'net_charm']].shift(1)      # previous trading day's row
W = w.shift(1)
O2, N = pickle.load(open(SC + 's6.pkl', 'rb')); L = dict(zip(N, O2['V2']))
AM = {nm: pd.read_pickle(SC + f'amt_{nm}.pkl').set_index(['day', 'm']) for nm in ('V15a', 'RA')}
def tstat(x): x = np.asarray(x, float); return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 8 and x.std() > 0 else np.nan
def line(lab, s, fl):
    a, b = s[s.era == 'A'].pts, s[s.era == 'B'].pts
    ok = len(a) >= 30 and len(b) >= 30 and np.sign(a.mean()) == np.sign(b.mean()) and abs(tstat(a)) >= 2 and abs(tstat(b)) >= 2
    print(f' {lab:36} n {len(s):4d} win {np.mean(s.pts>0) if len(s) else 0:3.0%} mean {s.pts.mean() if len(s) else 0:+6.1f} | A n{len(a):3d} {a.mean() if len(a) else 0:+6.1f} ({tstat(a):+.1f}) | B n{len(b):3d} {b.mean() if len(b) else 0:+6.1f} ({tstat(b):+.1f}) {"PASS" if ok else ""}')
    if ok: fl.append(lab)
X = pd.read_parquet('/tmp/claude-0/ml/nq5_real.parquet'); X = X[X.rth]; X['day'] = pd.to_datetime(X.day).dt.normalize(); OPEN = X.groupby('day').open.first()
for nm in ('V15a', 'RA'):
    T = L[nm][0]; T = T[T.sz > 0].copy(); T['day'] = pd.to_datetime(T.day); R = T.set_index(['day', 'm']); R = R[~R.index.duplicated()].join(AM[nm][['dev_poc', 'amt']], how='left').reset_index()
    R['era'] = np.where(R.day < '2022-01-01', 'A', 'B')
    for c in O.columns: R[c] = O[c].reindex(R.day).values
    R['cw'] = W.call_wall_nq.reindex(R.day).values; R['pw'] = W.put_wall_nq.reindex(R.day).values; R['op'] = OPEN.reindex(R.day).values
    R['d_call'] = R.dir * np.sign(R.cw - R.op); R['d_put'] = R.dir * np.sign(R.op - R.pw)    # trading toward the call wall (long below it / short above) etc
    R = R.dropna(subset=['gex', 'iv']); fl = []
    print(f'\n######## {nm}: {len(R)} trades, mean {R.pts.mean():+.1f}')
    A_ = R[R.era == 'A']
    print('- net GEX sign'); [line(f'   GEX {nm_}', R[m], fl) for nm_, m in (('positive', R.gex > 0), ('negative', R.gex <= 0))]
    for col, lab in (('gexz', 'GEX z-score (60d)'), ('pcv', 'put/call volume'), ('pco', 'put/call OI'), ('iv', 'ATM IV level'), ('ivchg', 'ATM IV 5d change'), ('skew_25d', '25d skew')):
        q = A_[col].quantile([1/3, 2/3]).values; b = pd.cut(R[col], [-1e18, q[0], q[1], 1e18], labels=['low', 'mid', 'high']); print(f'- {lab} (terciles from 2014-21)'); [line(f'   {v}', R[b == v], fl) for v in ('low', 'mid', 'high')]
    for col, lab in (('dex', 'net DEX'), ('net_vanna', 'net vanna'), ('net_charm', 'net charm')):
        R[col + '_al'] = R.dir * np.sign(R[col]); print(f'- {lab} sign aligned (+1 = positive exposure and long / negative and short)'); [line(f'   {v:+d}', R[R[col + "_al"] == v], fl) for v in (1, -1)]
    print('- dealer positioning x trend: positive GEX & price beyond developing POC / negative GEX & beyond'); [line(f'   {nm_}', R[m], fl) for nm_, m in (('GEX+ & beyond dev POC', (R.gex > 0) & (R.dev_poc == 1)), ('GEX- & beyond dev POC', (R.gex <= 0) & (R.dev_poc == 1)), ('GEX+ & not beyond', (R.gex > 0) & (R.dev_poc != 1)), ('GEX- & not beyond', (R.gex <= 0) & (R.dev_poc != 1)))]
    print('- NQ gamma wall: trading toward the call wall (+1) / put wall'); [line(f'   toward call wall {v:+d}', R[R.d_call == v], fl) for v in (1, -1)]
    print(f'>> PASS buckets: {fl}'); R.to_pickle(SC + f'opt_{nm}.pkl')
