"""Nov 2 2026: live book since open (Oct 13) vs the night's best candidate models replayed on the SAME live trades.
Replayable without unseen signals: size tilt (weights only) and drop-low-vol (trades removed, slot left in cash - the real version
would refill it with another flip we cannot see). Open positions marked at the site snapshot's last price.
Low-vol thresholds = 20th / 66.7th percentile of 20d realized vol across 2026 backtest flips (fixed before looking at live)."""
import json, glob, numpy as np, pandas as pd
L = '/mnt/user-data/uploads/fartpie/live'
led = json.load(open(f'{L}/ledger.json')); snap = json.load(open(f'{L}/site_snapshot.json'))
last = {}
for p in snap['positions']: last[(p['base'], '1h' if p['name'].endswith('1h') else '15m')] = float(p['last'])
def daily(path):
    H = pd.read_parquet(path); dtc = 'datetime' if 'datetime' in H else H.columns[0]
    H['dt'] = pd.to_datetime(H[dtc], utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); m = H.dt.dt.hour * 60 + H.dt.dt.minute
    return H[(m >= 570) & (m < 960)].groupby(H.dt.dt.normalize()).close.last()
X = pd.read_parquet('/tmp/claude-0/ml/noise_X.parquet'); X26 = X[X.entry.dt.year == 2026]; rv26 = []
for tk, g in X26.groupby('tk'):
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    if not f: continue
    s = daily(f[0]).pct_change().rolling(20).std().shift(1); rv26 += [s.asof(d) for d in g.d]
lo, hi = np.nanpercentile(rv26, 20), np.nanpercentile(rv26, 100 * 2 / 3)
rows = []
for k, v in led.items():
    if k == '__asof__': continue
    t0 = pd.Timestamp(v['t0']); s = daily(f"{L}/{v['tk']}_15m.parquet").pct_change().rolling(20).std().shift(1); rv = s.asof(t0.normalize())
    xp = v.get('xpx') or last.get((v['tk'], v['track']))
    rows.append(dict(key=k, tk=v['tk'], track=v['track'], t0=t0, status=v['status'], ret=(xp / v['px'] - 1 - 2e-4) if xp else np.nan, rv=rv))
T = pd.DataFrame(rows).sort_values('t0'); T['grp'] = np.where(T.rv <= lo, 'low', np.where(T.rv > hi, 'high', 'mid'))
print(f"live trades since open: {len(T)} ({(T.status == 'open').sum()} open, marked at the site's last price); vol groups: {T.grp.value_counts().to_dict()}")
def book(w):
    return np.expm1(np.nansum(np.log1p(T.ret * w)))
W0 = np.full(len(T), 1 / 6); Wt = np.where(T.grp == 'high', 1.25, np.where(T.grp == 'low', .75, 1)) / 6; Wd = np.where(T.grp == 'low', 0, 1) / 6
print(f"\n  site says: book {snap['history']['book']:+.2f}%   SPY {snap['history']['spy']:+.2f}%   QQQ {snap['history']['qqq']:+.2f}%  (since {snap['history']['since']})")
for nm, w in (('live (equal 1/6)', W0), ('size tilt 1.25 / 1 / 0.75', Wt), ('drop low-vol flips (slot left in cash)', Wd)):
    print(f'  {nm:<40} {book(w):+.2%}')
print('\n  by vol group, avg trade: ' + '  '.join(f'{g}: {T.ret[T.grp == g].mean():+.2%} (n {(T.grp == g).sum()})' for g in ('low', 'mid', 'high')))
print(T[['key', 'grp', 'ret']].to_string(index=False, formatters={'ret': lambda x: f'{x:+.2%}'}))
