"""Kalman filter (local linear trend: price level + velocity, constant-velocity model) as an entry gate.
 stock: velocity of the 15m Kalman trend at the flip bar, in ATR per bar, for a fast (q/r 1e-2) and a slow (1e-4) filter
 market: SPY daily Kalman velocity > 0 (a smoother alternative to the 50-day average)
Live trades (xs/ext/ahx 'base+sticky'), ratio >= 2.5, 6 slots. Stage 1 TK..: features; stage 2: buckets + sims."""
import sys, glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'xs')
def kalman_vel(y, q):
    x = np.array([y[0], 0.0]); P = np.eye(2); F = np.array([[1, 1], [0, 1]]); Q = q * np.array([[1 / 3, 1 / 2], [1 / 2, 1]]); v = np.empty(len(y))
    for i, z in enumerate(y):
        x = F @ x; P = F @ P @ F.T + Q
        S = P[0, 0] + 1.0; K = P[:, 0] / S; x = x + K * (z - x[0]); P = P - np.outer(K, P[0, :]); v[i] = x[1]
    return v
if sys.argv[1] == '1':
    fs = {os.path.basename(f).split('_15m_')[0]: f for f in sorted(glob.glob('lse_ohlcv_cache/*_15m_2018-01-01_*.parquet'))}
    for tk in sys.argv[2:]:
        H = pd.read_parquet(fs[tk]); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
        m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 240) & (m < 960) & (H.volume > 0) & (H.dt.dt.dayofweek < 5)].drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
        tr = np.maximum(H.high - H.low, np.maximum((H.high - H.close.shift()).abs(), (H.low - H.close.shift()).abs())); atr = tr.rolling(14).mean().values
        y = H.close.values / H.close.values[0] * 100; sc = H.close.values[0] / 100           # scale-free filtering (r = 1 in index points)
        F = pd.DataFrame({'entry': H.dt.shift(-1)})
        for nm, q in (('vf', 1e-2), ('vs', 1e-4)): F[nm] = kalman_vel(y, q) * sc / atr
        F.dropna().assign(tk=tk).to_parquet(f'xs/ext/kalx/{tk}.parquet'); print(tk, end=' ', flush=True)
else:
    exec(open('xs/ctx_sim.py').read().split('T = pd.concat')[0])
    T = pd.concat(pd.read_parquet(f) for f in glob.glob('xs/ext/ahx/*.parquet')); T = T[T['var'] == 'base+sticky']
    T['entry'] = pd.to_datetime(T.entry); T['exit'] = pd.to_datetime(T.exit)
    T = T[(T.entry >= '2019-01-01') & (T.entry <= T.tk.map(last) + pd.Timedelta(days=30))]
    T['ratio'] = [RAT[t].asof(d.normalize()) for t, d in zip(T.tk, T.entry)]; T = T[T.ratio >= 2.5]
    T = T.merge(pd.concat(pd.read_parquet(f) for f in glob.glob('xs/ext/kalx/*.parquet')), on=['tk', 'entry'], how='left')
    D = pd.read_parquet('etf_daily/SPY.parquet'); D['d'] = pd.to_datetime(D.datetime).dt.tz_localize(None).dt.normalize(); c = D.drop_duplicates('d').set_index('d').adjclose
    for nm, q in (('spk_f', 1e-2), ('spk_s', 1e-3)):
        y = (c / c.iloc[0] * 100).values; ok = pd.Series(kalman_vel(y, q) > 0, index=c.index).shift(1)
        T[nm] = ok.reindex(T.entry.dt.normalize(), method='ffill').fillna(True).values.astype(bool)
    L = T[~T.h1]
    for f in ('vf', 'vs'):
        q = pd.qcut(L[f], 5); g = L.groupby([q, L.entry.dt.year >= 2022]).ret.mean().unstack() * 100
        print(f'\n15m flips: avg trade % by stock Kalman velocity ({"fast" if f == "vf" else "slow"}), quintiles (2019-21 | 2022-26):'); print(g.round(2).to_string())
    def sim(Z, n=6):
        Z = Z.sort_values(['entry', 'ratio', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable'); op, ev = [], []
        for e, x, r in zip(Z.entry.values, Z.exit.values, Z.ret.values):
            op = [o for o in op if o > e]
            if len(op) < n: op.append(x); ev.append((x, r / n))
        return pd.Series([v for _, v in ev], index=pd.DatetimeIndex([t for t, _ in ev])).sort_index()
    def st(s):
        e = (1 + s).groupby(s.index.normalize()).prod().cumprod(); y = (e.index[-1] - e.index[0]).days / 365.25
        return e.iloc[-1] ** (1 / y) - 1, (e / e.cummax() - 1).min()
    P = (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))
    print(f'\n{"6 slots, ratio >= 2.5, sticky":<40}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}')
    def show(nm, Z):
        res = [st(sim(Z[(Z.entry >= a) & (Z.entry < b)])) for a, b in P]
        print(f'  {nm:<38}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in res), flush=True)
    show('live', T)
    for f, nm in (('vf', 'fast'), ('vs', 'slow')):
        show(f'buy only if stock Kalman ({nm}) rising', T[T.h1 | (T[f] > 0)])
        show(f'skip if stock Kalman ({nm}) top 20%', T[T.h1 | ~(T[f] > L[f].quantile(.8))])
    show('only when SPY Kalman (fast) rising', T[T.spk_f]); show('only when SPY Kalman (slow) rising', T[T.spk_s])
