"""Nov 5 2026: EVERYTHING IN NQ POINTS PER DAY (1 contract, after a 1 pt round-trip cost). Goal set by Rey: 7.5 - 20 pts a day on average, at least 50 pts a month.
DECLARED BEFORE RESULTS. QQQ 5m bars 2018 - Oct 2026 converted to NQ points (QQQ move x NQ / QQQ at the prior close); RTH 15m vector; entry at the next 5m open; stop checked first.
  A  Rule A (the current prop system's trades, 75 / 50, flat 12:00) - from live_data/nq_rule_a_trades.csv
  B  every AM flip 09:45-11:30, <= 2 a day, 75 / 50, flat 12:00
  C  every flip 09:45-15:30, <= 4 a day, 75 / 50, flat 15:55
  D  every flip 09:45-15:30, hold until the vector flips back (stop 50), flat 15:55   (always-in-the-trend style)
  E  C, but stop for the day at +20 pts or -40 pts
  F  every flip 09:45-15:30, 20 / 20, <= 4 a day, stop for the day at +20 / -40     (built for the 7.5-20 a day target)
For B-F: coin-flip direction on the same entries (median of 10). Reported per trading day (every session counts, flat days = 0): mean, % of days >= +7.5, % days < 0,
per month: mean, % of months >= 50 pts, worst month; by period 2018-21 / 2022-23 / 2024-25 / 2026."""
import numpy as np, pandas as pd, sys, os, glob, warnings; warnings.filterwarnings('ignore')
HERE = '/home/claude/fartpie/xs'; sys.path.insert(0, HERE); _a = sys.argv; sys.argv = sys.argv[:1] + ['x']
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0]); sys.argv = _a
import yfinance as yf
H = load(glob.glob('/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/QQQ_5m_*.parquet')[0]); H = H[(H.m >= 570) & (H.m < 960)].reset_index(drop=True); H['day'] = H.dt.dt.normalize()
B = H.set_index('dt')[['open', 'high', 'low', 'close']].groupby(pd.Grouper(freq='15min')).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna().reset_index()
B['reg'] = np.asarray(vec(B[['open', 'high', 'low', 'close']])); B['end'] = B.dt + pd.Timedelta('15min'); B['me'] = B.end.dt.hour * 60 + B.end.dt.minute
Y = yf.download(['NQ=F', 'QQQ'], start='2017-12-01', interval='1d', progress=False, auto_adjust=False)['Close'].ffill(); Y.index = Y.index.normalize()
K = (Y['NQ=F'] / Y['QQQ']).shift(1)                                                           # NQ points per QQQ dollar, prior close
t5, o5, h5, l5, c5, m5, d5 = H.dt.values, H.open.values, H.high.values, H.low.values, H.close.values, H.m.values, H.day.values
k5 = K.reindex(pd.DatetimeIndex(d5)).ffill().values
pos5 = pd.Series(np.arange(len(H)), index=H.dt.values)
reg15 = pd.Series(B.reg.values, index=B.end.values)                                          # vector state known at each 15m close
FL = []                                                                                       # (entry 5m index, dir, minute of signal)
r = B.reg.values
for k in range(1, len(B)):
    if r[k] != r[k - 1] and r[k - 1] != 0 and 585 <= B.me.values[k] <= 930:
        j = pos5.get(B.end.values[k])
        if j is not None and d5[j] == B.end.values[k].astype('datetime64[D]'): FL.append((int(j), int(r[k]), int(B.me.values[k])))
FL = pd.DataFrame(FL, columns=['j', 'dir', 'me']); FL['day'] = d5[FL.j.values]
print(f'15m RTH flips 2018-2026: {len(FL)} ({len(FL) / H.day.nunique():.2f} a day); AM 09:45-11:30: {(FL.me <= 690).sum()}', flush=True)
def sim(j, d, T, S, flat, vexit):
    e = o5[j]; kk = k5[j]
    for q in range(j, len(H)):
        if d5[q] != d5[j] or m5[q] >= flat: return (c5[q - 1] - e) * d * kk, q - 1
        lo = ((l5[q] if d > 0 else h5[q]) - e) * d * kk; hi = ((h5[q] if d > 0 else l5[q]) - e) * d * kk
        if lo <= -S: return -S, q
        if T and hi >= T: return T, q
        if vexit and m5[q] % 15 == 10:                                                          # a 15m bar closes at the end of this 5m bar
            st = reg15.get(t5[q] + np.timedelta64(5, 'm'))
            if st is not None and st == -d and q + 1 < len(H) and d5[q + 1] == d5[j]: return (o5[q + 1] - e) * d * kk, q
    return None
def run(X, T, S, flat, nmax, vexit=False, cap=None, floor=None):
    out = []
    for day, g in X.groupby('day', sort=True):
        busy = -1; n = 0; dp = 0.0
        for j, d in zip(g.j.values, g.dir.values):
            if n >= nmax or j <= busy or (cap is not None and (dp >= cap or dp <= floor)): continue
            res = sim(j, d, T, S, flat, vexit)
            if res is None: continue
            dp += res[0] - 1.0; busy = res[1]; n += 1
        out.append((day, dp, n))
    return pd.DataFrame(out, columns=['day', 'pts', 'n']).set_index('day').pts
ALLD = pd.DatetimeIndex(sorted(H.day.unique()))
def stats(P):
    P = P.reindex(ALLD).fillna(0.0); M = P.groupby(P.index.to_period('M')).sum()
    per = np.select([P.index < '2022-01-01', P.index < '2024-01-01', P.index < '2026-01-01'], ['18-21', '22-23', '24-25'], '2026')
    return dict(day=P.mean(), ge=(P >= 7.5).mean(), neg=(P < 0).mean(), mo=M.mean(), mo50=(M >= 50).mean(), worst=M.min(), per={p: P[per == p].mean() for p in ('18-21', '22-23', '24-25', '2026')},
                yr=P.groupby(P.index.year).sum())
L = pd.read_csv('/home/claude/fartpie/live_data/nq_rule_a_trades.csv', parse_dates=['day']); PA = L.groupby('day').net_pts.sum()
AM = FL[FL.me <= 690]; rng = np.random.default_rng(3)
V = {'A Rule A (current system)': (None, None), 'B all AM flips 75/50': (AM, dict(T=75, S=50, flat=720, nmax=2)), 'C all-day flips 75/50': (FL, dict(T=75, S=50, flat=955, nmax=4)),
     'D all-day flips, vector exit': (FL, dict(T=None, S=50, flat=955, nmax=4, vexit=True)), 'E C + day stop +20 / -40': (FL, dict(T=75, S=50, flat=955, nmax=4, cap=20, floor=-40)),
     'F 20/20, day stop +20 / -40': (FL, dict(T=20, S=20, flat=955, nmax=4, cap=20, floor=-40))}
print(f'\n{"NQ points, 1 contract, after costs":<32} {"pts/day":>8} {"days>=7.5":>10} {"days<0":>7} {"pts/month":>10} {"months>=50":>11} {"worst month":>12} | per day 18-21 / 22-23 / 24-25 / 2026 | coin-flip pts/day', flush=True)
for nm, (X, kw) in V.items():
    P = PA if X is None else run(X, **kw); s = stats(P)
    cf = '' if X is None else f'{np.median([stats(run(X.assign(dir=rng.choice([-1, 1], len(X))), **kw))["day"] for _ in range(10)]):+.2f}'
    print(f'  {nm:<30} {s["day"]:>+8.2f} {s["ge"]:>10.0%} {s["neg"]:>7.0%} {s["mo"]:>+10.1f} {s["mo50"]:>11.0%} {s["worst"]:>+12.0f} | ' + ' / '.join(f'{v:+.2f}' for v in s['per'].values()) + f' | {cf}', flush=True)
    print(f'      by year: ' + ' '.join(f'{y} {v:+.0f}' for y, v in s['yr'].items()), flush=True)
