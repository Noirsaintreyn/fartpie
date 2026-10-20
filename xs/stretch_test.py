"""Skip stretched entries (bought-the-top flips like NVDA Oct 19). Live model trades = xs/ext/ahx 'base+sticky', ratio >= 2.5, 6 slots.
Entry features known at the close of the bar before the entry open:
  risk = (entry - vector stop) / ATR      run = (close - lowest low of the last 26 bars) / ATR      z = price vs session VWAP in VWAP sigmas
Stage 1 TK..: features -> xs/ext/stretch/<TK>.parquet.  Stage 2: sims (+ SPY / QQQ 50-day filter)."""
import sys, glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'xs')
from vector_pine import run_state_machine
if sys.argv[1] == '1':
    fs = {os.path.basename(f).split('_15m_')[0]: f for f in sorted(glob.glob('lse_ohlcv_cache/*_15m_2018-01-01_*.parquet'))}
    for tk in sys.argv[2:]:
        H = pd.read_parquet(fs[tk]); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
        m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 240) & (m < 960) & (H.volume > 0) & (H.dt.dt.dayofweek < 5)].drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
        reg, lv, atr, _ = run_state_machine(H[['open', 'high', 'low', 'close']].assign(datetime=H.dt)); atr = np.asarray(atr, float)
        stop = np.asarray(lv, float) - 0.35 * atr
        v = H.volume.clip(lower=1); tp = (H.high + H.low + H.close) / 3; d = H.dt.dt.normalize()
        cv = v.groupby(d).cumsum(); vw = (tp * v).groupby(d).cumsum() / cv; sd = np.sqrt(((v * tp ** 2).groupby(d).cumsum() / cv - vw ** 2).clip(lower=0))
        F = pd.DataFrame({'dt': H.dt, 'nxt_open': H.open.shift(-1), 'stop': stop, 'atr': atr, 'z': ((H.close - vw) / sd.replace(0, np.nan)).clip(-6, 6),
                          'run': (H.close - H.low.rolling(26).min()) / atr})
        F['known'] = H.dt.shift(-1)                                          # features at bar j-1 close -> used for an entry at bar j's open
        F['risk'] = (F.nxt_open - F.stop) / F.atr
        F[['known', 'risk', 'run', 'z']].dropna(subset=['known']).assign(tk=tk).to_parquet(f'xs/ext/stretch/{tk}.parquet'); print(tk, end=' ', flush=True)
else:
    exec(open('xs/ctx_sim.py').read().split('T = pd.concat')[0])                  # RAT, last
    T = pd.concat(pd.read_parquet(f) for f in glob.glob('xs/ext/ahx/*.parquet')); T = T[T['var'] == 'base+sticky']
    T['entry'] = pd.to_datetime(T.entry); T['exit'] = pd.to_datetime(T.exit)
    T = T[(T.entry >= '2019-01-01') & (T.entry <= T.tk.map(last) + pd.Timedelta(days=30))]
    T['ratio'] = [RAT[t].asof(d.normalize()) for t, d in zip(T.tk, T.entry)]; T = T[T.ratio >= (1.0 if sys.argv[1] == '3' else 2.5)]
    F = pd.concat(pd.read_parquet(f) for f in glob.glob('xs/ext/stretch/*.parquet')).rename(columns={'known': 'entry'})
    T = T.merge(F, on=['tk', 'entry'], how='left'); print(f'{len(T):,} trades, features on {T.risk.notna().mean():.0%}')
    for b in ('SPY', 'QQQ'):
        D = pd.read_parquet(f'etf_daily/{b}.parquet'); D['d'] = pd.to_datetime(D.datetime).dt.tz_localize(None).dt.normalize(); c = D.drop_duplicates('d').set_index('d').adjclose
        ok = (c > c.rolling(50).mean()).shift(1); T[b + '50'] = ok.reindex(T.entry.dt.normalize(), method='ffill').fillna(True).values.astype(bool)
    for f in (('risk', 'run', 'z') if sys.argv[1] == '2' else ()):
        q = pd.qcut(T[f], 5); g = T.groupby([q, T.entry.dt.year >= 2022]).ret.mean().unstack() * 100
        print(f'\navg trade % by {f} quintile (2019-21 | 2022-26):'); print(g.round(2).to_string())
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
    print(f'\n{"6 slots, ratio >= 2.5, sticky":<40}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}  trades')
    def show(nm, X):
        res = [st(sim(X[(X.entry >= a) & (X.entry < b)])) for a, b in P]
        print(f'  {nm:<38}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in res) + f'{len(X):>8,}', flush=True)
    if sys.argv[1] == '3':
        for cut in (1.5, 1.75, 2.0, 2.25, 2.5, 3.0):
            X = T[T.ratio >= cut]
            for n in (6, 8):
                sim.__defaults__ = (n,); show(f'ratio >= {cut}, {n} slots' + ('  (live)' if cut == 2.5 and n == 6 else ''), X)
                show(f'   + SPY > 50d', X[X.SPY50])
        sys.exit()
    show('live', T)
    qr = {k: T[k].quantile([.6, .8, .9]).values for k in ('risk', 'run')}
    for k in ('risk', 'run'):
        for lab, th in zip(('top 40%', 'top 20%', 'top 10%'), qr[k]): show(f'skip {k} {lab} (> {th:.1f} ATR)', T[~(T[k] > th)])
    for th in (1.5, 2.0, 2.5): show(f'skip VWAP z > {th}', T[~(T.z > th)])
    show('only when SPY > 50d', T[T.SPY50]); show('only when QQQ > 50d', T[T.QQQ50])
    show('QQQ>50d + skip run top 20%', T[T.QQQ50 & ~(T.run > qr['run'][1])])
    show('SPY>50d + skip run top 20%', T[T.SPY50 & ~(T.run > qr['run'][1])])
