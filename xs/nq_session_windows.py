"""Nov 5 2026: can the vector reach 50 NQ points a month? More trades from the REST OF THE DAY. DECLARED BEFORE RESULTS.
QQQ 5m 2018 - Oct 2026 -> 15m vector (signal_lab.vec), every flip (no Rule A filter), entry at the next 5m open, 75 / 50 NQ points (QQQ move x NQ/QQQ prior-close ratio),
stop checked first inside a bar, 1 pt cost, <= 2 trades a day per window:
  AM   flips of bars closing 09:45-11:30, flat 12:00 (the current window, unfiltered)
  MID  11:45-13:30, flat 15:55        PM  13:45-15:30, flat 15:55
  ALL  AM + MID + PM together (<= 3 a day)
Report NQ points per month by year (1 contract), trades per month, and coin-flip direction for each window."""
import numpy as np, pandas as pd, sys, os, glob, warnings; warnings.filterwarnings('ignore')
HERE = '/home/claude/fartpie/xs'; sys.path.insert(0, HERE); _a = sys.argv; sys.argv = sys.argv[:1] + ['x']
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0]); sys.argv = _a
import yfinance as yf
H = load(glob.glob('/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/QQQ_5m_*.parquet')[0]).reset_index(drop=True)
o5, h5, l5, c5, m5 = H.open.values, H.high.values, H.low.values, H.close.values, H.m.values; d5 = H.dt.dt.normalize().values; t5 = H.dt.values
x = H.set_index('dt')[['open', 'high', 'low', 'close']]
g = x.groupby([x.index.normalize(), pd.Grouper(freq='15min', origin='start_day', offset='4h')]).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
g.index = g.index.get_level_values(1); g = g.reset_index(); g.columns = ['dt', 'open', 'high', 'low', 'close']; g['reg'] = np.asarray(vec(g[['open', 'high', 'low', 'close']]))
g['end'] = g.dt + pd.Timedelta('15min'); me = (g.end.dt.hour * 60 + g.end.dt.minute).values; r = g.reg.values
nq = yf.download('NQ=F', start='2017-12-01', interval='1d', progress=False, auto_adjust=False)['Close'].squeeze(); nq.index = nq.index.normalize()
qd = H[H.m < 960].groupby(H.dt.dt.normalize()).close.last(); K = (nq.reindex(qd.index, method='ffill') / qd).shift(1)
W = {'AM': (585, 690, 720), 'MID': (705, 810, 955), 'PM': (825, 930, 955)}
C = []
for k in range(2, len(g)):
    if r[k] == r[k - 1] or r[k - 1] == 0: continue
    for w, (a, b, flat) in W.items():
        if a <= me[k] <= b:
            j = np.searchsorted(t5, g.end.values[k])
            if j < len(t5) and (t5[j] - g.end.values[k]) <= np.timedelta64(5, 'm'): C.append((pd.Timestamp(d5[j]), j, int(r[k]), w, flat))
C = pd.DataFrame(C, columns=['day', 'j', 'dir', 'w', 'flat']); C['k'] = K.reindex(C.day).values; C = C[np.isfinite(C.k)]
def sim(j, d, kk, flat):
    e = o5[j]
    for q in range(j, len(t5)):
        if d5[q] != d5[j] or m5[q] >= flat: return (c5[q - 1] - e) * d * kk - 1.0, q - 1
        if ((l5[q] if d > 0 else h5[q]) - e) * d * kk <= -50: return -51.0, q
        if ((h5[q] if d > 0 else l5[q]) - e) * d * kk >= 75: return 74.0, q
    return None
def run(X, cap):
    out = []
    for day, gg in X.sort_values('j').groupby('day'):
        busy, n = -1, 0
        for j, d, kk, fl, w in zip(gg.j.values, gg.dir.values, gg.k.values, gg.flat.values, gg.w.values):
            if n >= cap or j <= busy: continue
            res = sim(j, d, kk, fl)
            if res: out.append((day, w, res[0])); busy = res[1]; n += 1
    return pd.DataFrame(out, columns=['day', 'w', 'net'])
rng = np.random.default_rng(3); months = (C.day.max() - C.day.min()).days / 30.44
SETS = {'AM': (C[C.w == 'AM'], 2), 'MID': (C[C.w == 'MID'], 2), 'PM': (C[C.w == 'PM'], 2), 'ALL': (C, 3)}
for nm, (X, cap) in SETS.items():
    R = run(X, cap); cf = np.median([run(X.assign(dir=rng.choice([-1, 1], len(X))), cap).net.sum() / months for _ in range(10)])
    y = R.groupby(R.day.dt.year).net.agg(['sum', 'size']); mo = pd.Series({yy: (9.3 if yy == 2026 else 12) for yy in y.index})
    print(f'{nm:<4} trades/month {len(R) / months:4.1f} | avg {R.net.mean():+5.1f} pts/trade win {R.net.gt(0).mean():.0%} | {R.net.sum() / months:+5.1f} pts/month (coin-flip {cf:+5.1f}) | by year pts/month: '
          + ' '.join(f'{yy} {s / mo[yy]:+5.0f}' for yy, s in y['sum'].items()) + f' | years >= 50/month: {(y["sum"] / mo >= 50).sum()} of {len(y)}', flush=True)
