"""Flip-candle features: size and relative volume of the bar that flipped the vector (15m trades: the 15m flip bar; 1h trades: the hour).
  body = |close-open| / ATR14,  rng = (high-low) / ATR14   (ATR on the same timeframe's bars)
  rvol = flip-bar volume / mean volume of the same time-of-day slot over the prior 20 sessions
Stage 1 TK..: xs/ext/fcandle/<tk>.parquet ; stage 2: averages by tercile + book tests (skip / tie-break)."""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'xs')
def feats(B):
    B = B.copy(); pc = B.close.shift(1)
    tr = np.maximum(B.high - B.low, np.maximum((B.high - pc).abs(), (B.low - pc).abs())); atr = tr.rolling(14).mean().shift(1)
    B['body'] = (B.close - B.open).abs() / atr; B['rng'] = (B.high - B.low) / atr; B['up'] = (B.close > B.open).astype(float)
    slot = B.dt.dt.hour * 60 + B.dt.dt.minute
    B['rvol'] = B.volume / B.groupby(slot).volume.transform(lambda v: v.shift(1).rolling(20, min_periods=10).mean())
    return B
if sys.argv[1] == '1':
    os.makedirs('xs/ext/fcandle', exist_ok=True)
    T = pd.read_parquet('xs/ext/wide_T.parquet'); T = T[T.sp & (T.ratio >= 1.5)]
    fs = {os.path.basename(f).split('_15m_')[0]: f for f in sorted(glob.glob('lse_ohlcv_cache/*_15m_2018-01-01_*.parquet'))}
    for tk in sorted(set(T.tk)):
        out = f'xs/ext/fcandle/{tk}.parquet'
        if os.path.exists(out) or tk not in fs: continue
        H = pd.read_parquet(fs[tk]); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
        m = H.dt.dt.hour * 60 + H.dt.dt.minute
        H = H[(m >= 240) & (m < 960) & (H.volume > 0) & (H.dt.dt.dayofweek < 5)].drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
        F15 = feats(H[['dt', 'open', 'high', 'low', 'close', 'volume']])
        x = H[H.dt.dt.hour * 60 + H.dt.dt.minute >= 570].set_index('dt')
        g = x.groupby([x.index.normalize(), pd.Grouper(freq='1h', origin='start_day', offset='9h30min')])
        r = g.agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last'), volume=('volume', 'sum')).dropna()
        r.index = r.index.get_level_values(1); r = r.reset_index().rename(columns={'index': 'dt'}); r.columns = ['dt', 'open', 'high', 'low', 'close', 'volume']
        F1 = feats(r); F1['end'] = F1.dt + pd.Timedelta('1h'); F15['end'] = F15.dt + pd.Timedelta('15min')
        rows = []
        for e, h1 in zip(T[T.tk == tk].entry, T[T.tk == tk].h1):
            F = F1 if h1 else F15; k = F.end.searchsorted(e, 'right') - 1         # last bar that closed at or before the entry
            rows.append((tk, h1, e) + ((np.nan,) * 4 if k < 0 else tuple(float(F[c].iloc[k]) for c in ('body', 'rng', 'rvol', 'up'))))
        pd.DataFrame(rows, columns=['tk', 'h1', 'entry', 'body', 'rng', 'rvol', 'up']).to_parquet(out); print(tk, end=' ', flush=True)
else:
    exec(open('xs/batch_oct25.py').read().split("PART = os.getenv('PART', 'ALL')")[0])
    F = pd.concat(pd.read_parquet(f) for f in glob.glob('xs/ext/fcandle/*.parquet'))
    T = T.merge(F, on=['tk', 'h1', 'entry'], how='left').sort_values('entry').reset_index(drop=True)
    T['per'] = np.where(T.entry.dt.year >= 2022, '22-26', '19-21')
    print(f"coverage: {T.rvol.notna().mean():.0%}  green flip bar: {T.up.mean():.0%}")
    for c in ('body', 'rng', 'rvol'):
        T['q'] = T.groupby('h1')[c].transform(lambda v: pd.qcut(v, 3, labels=['low', 'mid', 'high']))
        g = T.groupby(['q', 'per']).ret.agg(['mean', 'count', lambda r: (r >= .03).mean()]); g.columns = ['avg%', 'n', 'P(>=3%)']; g['avg%'] *= 100
        print(f'\n{c} terciles (within track):'); print(g.unstack().round(3).to_string())
        T[c + '_q'] = T.q
    head('FLIP CANDLE book tests')
    for c in ('body', 'rng', 'rvol'):
        for q in ('low', 'high'): show(f'skip {c} {q} third', T[T[c + '_q'] != q])
        for asc in (False, True):
            Z = T.copy(); Z['k'] = Z.groupby('entry')[c].rank(pct=True, ascending=not asc).fillna(.5)
            show(f'tie-break: {"biggest" if not asc else "smallest"} {c} first', Z, key='k')
            Z['k2'] = Z.groupby('entry').ratio.rank(pct=True) + Z.k; show(f'  ratio rank + that rank', Z, key='k2')
