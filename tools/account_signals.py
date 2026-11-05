"""ACCOUNT-START SIGNALS - when to buy a new prop eval, known at the prior close (Nov 5 2026 backtest, xs/nq_accounts3.py).
Evals started 2022-2025 on the NQ Rule A system, Lucid 50K: share that ended in a payout when the signal was ON vs OFF (held in both 2022-23 and 2024-25):
  VOL-HI  QQQ 20-day Yang-Zhang vol in the top third of its last year (252-day percentile >= 2/3)   96% vs 42%
  VXN     VXN close at or above its 1-year (252-day) median                                         87% vs 39%
  COLD    NQ Rule A system's last 20 trades (1 lot, NQ pts) net negative                            93% vs 53%
  DIP     QQQ 20-day z-score below -1                                                               81% vs 56%
Evals bought with every signal OFF still averaged +$2,647, so this is a green light for WHEN to add (tight cash, accounts beyond Lucid's 5), not a reason to sit flat.
COLD reads live_data/nq_rule_a_trades.csv (backtest log; append live trades with --add so it stays current).
  python tools/account_signals.py                      today's read (uses the last COMPLETE daily bar)
  python tools/account_signals.py --json               JSON for the desktop widget (cached 30 min in ~/.lap_acct_signals.json)
  python tools/account_signals.py --add 2026-11-05 74  log a Rule A trade: session date, net NQ points (1 lot, after the 1 pt cost)
  python tools/account_signals.py --history 2022-01-01 2025-12-31   share of sessions each signal was on (check against the backtest)"""
import json, os, sys, time, datetime as dt
import numpy as np, pandas as pd
from zoneinfo import ZoneInfo
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); LOG = os.path.join(ROOT, 'live_data', 'nq_rule_a_trades.csv')
CACHE = os.path.expanduser('~/.lap_acct_signals.json'); ET = ZoneInfo('America/New_York')
STATS = {'VOL-HI': ('QQQ vol in top third of its year', .96, .42), 'VXN': ('VXN at or above 1-year median', .87, .39),
         'COLD': ('system cold: last 20 trades net < 0', .93, .53), 'DIP': ('QQQ 20-day z below -1', .81, .56)}
def data(start):
    import yfinance as yf
    q = yf.download('QQQ', start=start, interval='1d', progress=False, auto_adjust=False); q.columns = [c[0] if isinstance(c, tuple) else c for c in q.columns]
    y = yf.download(['QQQ', '^VXN'], start=start, interval='1d', progress=False, auto_adjust=True)['Close'].ffill()
    q.index = q.index.tz_localize(None).normalize() if q.index.tz is not None else q.index.normalize(); y.index = y.index.tz_localize(None).normalize() if y.index.tz is not None else y.index.normalize()
    now = dt.datetime.now(ET)
    if now.hour * 60 + now.minute < 16 * 60 + 15:                                      # today's bar is not a close yet
        today = pd.Timestamp(now.date()); q = q[q.index < today]; y = y[y.index < today]
    return q, y
def series(q, y):
    o, h, l, c = q.Open, q.High, q.Low, q.Close; n = 20; kz = .34 / (1.34 + (n + 1) / (n - 1))
    yz = np.log(o / c.shift(1)).rolling(n).var() + kz * np.log(c / o).rolling(n).var() + (1 - kz) * (np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)).rolling(n).mean()
    yzp = yz.dropna().rolling(252, min_periods=120).rank(pct=True)
    vx = y['^VXN']; vxm = vx.rolling(252, min_periods=120).median(); qa = y['QQQ']; z = (qa - qa.rolling(20).mean()) / qa.rolling(20).std()
    return yz, yzp, vx, vxm, z
def trades():
    if not os.path.exists(LOG): return pd.DataFrame(columns=['day', 'net_pts'])
    L = pd.read_csv(LOG, parse_dates=['day']); return L.sort_values('day')
def read(asof, yz, yzp, vx, vxm, z, L, live=True):
    a = lambda s: s[s.index <= asof].iloc[-1] if (s.index <= asof).any() else np.nan
    p, v, m, zz = a(yzp), a(vx), a(vxm), a(z); T = L[L.day <= asof]; last20 = T.net_pts.iloc[-20:].sum() if len(T) >= 20 else np.nan
    stale = live and (len(T) == 0 or (asof - T.day.iloc[-1]).days > 30)       # live only: the log has not been kept up
    out = [dict(key='VOL-HI', on=bool(p >= 2 / 3), value=f'{p:.0%} pct (20d YZ vol {np.sqrt(a(yz) * 252):.0%}/yr)'),
           dict(key='VXN', on=bool(v >= m), value=f'{v:.1f} vs median {m:.1f}'),
           dict(key='COLD', on=bool(last20 < 0) if not stale else False, stale=bool(stale), value=(f'last 20 {last20:+.0f} pts' if np.isfinite(last20) else 'n/a') + (f' (log ends {T.day.iloc[-1]:%b %d}: add trades)' if stale and len(T) else '')),
           dict(key='DIP', on=bool(zz < -1), value=f'z {zz:+.2f}')]
    for s in out: s['label'], s['payout_on'], s['payout_off'] = STATS[s['key']]
    return out
def nxt(d):
    d = d + pd.Timedelta(days=1)
    while d.dayofweek >= 5: d += pd.Timedelta(days=1)
    return d
def now_json():
    if os.path.exists(CACHE):
        try:
            c = json.load(open(CACHE))
            if time.time() - c.get('ts', 0) < 1800: return c
        except Exception: pass
    q, y = data((pd.Timestamp.today() - pd.Timedelta(days=3 * 365)).strftime('%Y-%m-%d')); S = series(q, y); asof = min(q.index[-1], y.index[-1])
    sig = read(asof, *S, trades()); on = [s['key'] for s in sig if s['on']]
    out = dict(ts=time.time(), asof=f'{asof:%Y-%m-%d}', for_session=f'{nxt(asof):%Y-%m-%d}', green=bool(on), on=on, signals=sig)
    try: json.dump(out, open(CACHE, 'w'))
    except Exception: pass
    return out
if __name__ == '__main__':
    a = sys.argv[1:]
    if a[:1] == ['--add']:
        L = trades(); L = pd.concat([L, pd.DataFrame([{'day': pd.Timestamp(a[1]), 'net_pts': float(a[2])}])], ignore_index=True); L['day'] = pd.to_datetime(L.day).dt.date
        L.to_csv(LOG, index=False); os.path.exists(CACHE) and os.remove(CACHE); print(f'logged {a[1]} {float(a[2]):+.1f} pts; last 20 net {L.net_pts.iloc[-20:].sum():+.1f}'); sys.exit()
    if a[:1] == ['--history']:
        q, y = data('2015-06-01'); S = series(q, y); L = trades(); days = q.index[(q.index >= a[1]) & (q.index <= a[2])]; on = {k: [] for k in STATS}
        for d in days:
            prev = q.index[q.index < d]
            if not len(prev): continue
            for s in read(prev[-1], *S, L, live=False): on[s['key']].append(s['on'])
        print(f'{len(days)} sessions {a[1]} -> {a[2]}: ' + ', '.join(f'{k} on {np.mean(v):.0%}' for k, v in on.items())); sys.exit()
    r = now_json()
    if a[:1] == ['--json']: print(json.dumps(r)); sys.exit()
    print(f"Account-start read for {r['for_session']} (prior close {r['asof']}): {'GREEN - ' + ', '.join(r['on']) if r['green'] else 'no signal on'}")
    for s in r['signals']: print(f"  {'ON ' if s['on'] else 'off'} {s['key']:<7} {s['value']:<38} evals paid {s['payout_on']:.0%} when on vs {s['payout_off']:.0%} off")
