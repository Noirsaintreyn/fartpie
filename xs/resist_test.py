"""Is the flip buying INTO resistance? Levels known before the entry open: prior-day RTH high (PDH), today's premarket high (PMH, RTH entries),
5-day RTH high (H5). room = (nearest level above the entry - entry) / ATR(14 bars); 'above' = entry already above all three.
Live trades (xs/ext/ahx 'base+sticky', 15m track), ratio >= 2.5, 6 slots. Stage 1 TK..: features; stage 2: buckets + sims."""
import sys, glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'xs')
if sys.argv[1] == '1':
    fs = {os.path.basename(f).split('_15m_')[0]: f for f in sorted(glob.glob('lse_ohlcv_cache/*_15m_2018-01-01_*.parquet'))}
    for tk in sys.argv[2:]:
        H = pd.read_parquet(fs[tk]); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
        H['m'] = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(H.m >= 240) & (H.m < 960) & (H.volume > 0) & (H.dt.dt.dayofweek < 5)].drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
        H['d'] = H.dt.dt.normalize(); rth = H[H.m >= 570]
        dh = rth.groupby('d').high.max(); pdh = dh.shift(1); h5 = dh.rolling(5).max().shift(1)
        pmh = H[H.m < 570].groupby('d').high.max()
        tr = np.maximum(H.high - H.low, np.maximum((H.high - H.close.shift()).abs(), (H.low - H.close.shift()).abs())); H['atr'] = tr.rolling(14).mean()
        H['run_hi'] = H.groupby('d').high.cummax()                                 # today's high so far (incl. the bar) - shifted below
        F = pd.DataFrame({'entry': H.dt, 'px': H.open, 'atr': H.atr.shift(1), 'pdh': H.d.map(pdh), 'h5': H.d.map(h5),
                          'pmh': np.where(H.m >= 570, H.d.map(pmh), np.nan), 'tdh': H.groupby('d').high.shift(1).groupby(H.d).cummax()})
        lv = F[['pdh', 'pmh', 'h5', 'tdh']].values; px = F.px.values[:, None]
        up = np.where(lv > px, lv, np.nan); F['room'] = (np.nanmin(up, 1) - F.px) / F.atr
        F['above'] = np.all((lv <= px) | np.isnan(lv), 1)
        F['under_pdh'] = (F.pdh > F.px) & ((F.pdh - F.px) / F.atr < 1)
        F[['entry', 'room', 'above', 'under_pdh']].assign(tk=tk).to_parquet(f'xs/ext/resx/{tk}.parquet'); print(tk, end=' ', flush=True)
else:
    exec(open('xs/ctx_sim.py').read().split('T = pd.concat')[0])                  # RAT, last
    T = pd.concat(pd.read_parquet(f) for f in glob.glob('xs/ext/ahx/*.parquet')); T = T[T['var'] == 'base+sticky']
    T['entry'] = pd.to_datetime(T.entry); T['exit'] = pd.to_datetime(T.exit)
    T = T[(T.entry >= '2019-01-01') & (T.entry <= T.tk.map(last) + pd.Timedelta(days=30))]
    T['ratio'] = [RAT[t].asof(d.normalize()) for t, d in zip(T.tk, T.entry)]; T = T[T.ratio >= 2.5]
    T = T.merge(pd.concat(pd.read_parquet(f) for f in glob.glob('xs/ext/resx/*.parquet')), on=['tk', 'entry'], how='left')
    L = T[~T.h1].copy(); L['room'] = L.room.fillna(99)
    b = pd.cut(L.room, [-1, .25, .5, 1, 2, 4, 98, 100], labels=['<0.25 ATR', '0.25-0.5', '0.5-1', '1-2', '2-4', '>4', 'above all levels'])
    g = L.groupby([b, L.entry.dt.year >= 2022]).ret.agg(['mean', 'count', lambda s: (s > 0).mean()]); g.columns = ['avg', 'n', 'win']
    g['avg'] *= 100; print('15m flips by room to the nearest resistance above the entry (2019-21 | 2022-26):'); print(g.unstack().round(2).to_string())
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
    show('live', T); R = T.room.fillna(99)
    for th in (.25, .5, 1.0): show(f'skip 15m flips < {th} ATR under resistance', T[T.h1 | ~(R < th)])
    show('skip flips < 1 ATR under PDH', T[T.h1 | ~T.under_pdh.fillna(False).astype(bool)])
