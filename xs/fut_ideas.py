"""Nov 4 2026: two quick futures checks. DECLARED BEFORE RESULTS.
(A) ORDER FLOW on MNQ 1-minute data (Dec 2024 - Nov 2025; buy/sell labels swapped in the file, so delta = -signed_vol). At every 5-minute close from 09:35 to 11:30, features from data up to that close:
    D5 (5m delta / 20-day same-time std) | CVD since 09:30 (in units of the session's typical 5m |delta|) | DIV (30m price change sign minus 30m delta sign: price up on selling = -2 ...)
    IMB (bid size - ask size)/(sum), mean of the last 5 minutes | ABS absorption = volume z >= 2 with a 5m range below its median | TR trade-count z
    target: NQ points over the next 30 minutes. Rank correlation (IC) in Dec-May and Jun-Nov separately; a feature counts if |IC| >= .03 with the same sign in both halves.
(B) INTRADAY MOMENTUM (Gao, Han, Li & Zhou 2018): the return from the prior 16:00 close to 10:00 predicts the 15:30-16:00 return. QQQ proxy 2018 - Sep 2026 in NQ points, 1 pt cost:
    trade the last 30 minutes in the direction of the first window, every day | only when |first window| > its 1-year median. Both halves (2018-21 / 2022-26) and by year."""
import numpy as np, pandas as pd, glob, warnings; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
ROOT = '/mnt/user-data/uploads/fartpie'
M = pd.read_parquet(f'{ROOT}/mnq_orderflow_1min.parquet'); M['dt'] = pd.to_datetime(M.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
M = M.dropna(subset=['last_price']).sort_values('dt').drop_duplicates('dt').set_index('dt'); M['delta'] = -M.signed_vol.fillna(0); M['vol'] = M.buy_vol.fillna(0) + M.sell_vol.fillna(0)
f5 = M.resample('5min', label='right', closed='right').agg({'last_price': ['last', 'max', 'min'], 'delta': 'sum', 'vol': 'sum', 'n_trades': 'sum', 'mean_bid_sz': 'mean', 'mean_ask_sz': 'mean'}).dropna()
f5.columns = ['c', 'h', 'l', 'delta', 'vol', 'ntr', 'bid', 'ask']; f5['m'] = f5.index.hour * 60 + f5.index.minute; f5['d'] = f5.index.normalize()
f5['rng'] = f5.h - f5.l; f5['fwd'] = f5.c.shift(-6) - f5.c                                  # next 30 minutes
f5.loc[f5.d.shift(-6) != f5.d, 'fwd'] = np.nan
g = f5.groupby('m')
f5['D5'] = f5.delta / g.delta.transform(lambda s: s.shift(1).rolling(20, min_periods=10).std())
f5['volz'] = (f5.vol - g.vol.transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())) / g.vol.transform(lambda s: s.shift(1).rolling(20, min_periods=10).std())
f5['TR'] = (f5.ntr - g.ntr.transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())) / g.ntr.transform(lambda s: s.shift(1).rolling(20, min_periods=10).std())
rth = f5[(f5.m >= 575) & (f5.m <= 960)].copy()
rth['CVD'] = rth.groupby('d').delta.cumsum() / rth.groupby('d').delta.transform(lambda s: s.abs().expanding().mean().shift(1))
rth['DIV'] = np.sign(rth.c - rth.groupby('d').c.shift(6)) - np.sign(rth.groupby('d').delta.transform(lambda s: s.rolling(6).sum()))
rth['IMB'] = (rth.bid - rth.ask) / (rth.bid + rth.ask)
rth['ABS'] = ((rth.volz >= 2) & (rth.rng < rth.groupby('m').rng.transform(lambda s: s.shift(1).rolling(20, min_periods=10).median()))).astype(float) * np.sign(rth.delta)
A = rth[(rth.m >= 575) & (rth.m <= 690)].dropna(subset=['fwd'])
print(f'(A) MNQ order flow: {A.d.nunique()} sessions, {len(A)} 5m bars 09:35-11:30; forward 30m |move| median {A.fwd.abs().median():.1f} pts', flush=True)
h1 = A.index < '2025-06-01'
for k in ('D5', 'CVD', 'DIV', 'IMB', 'ABS', 'TR'):
    a, b = A[h1].dropna(subset=[k]), A[~h1].dropna(subset=[k]); i1, i2 = spearmanr(a[k], a.fwd)[0], spearmanr(b[k], b.fwd)[0]
    q = pd.qcut(A[k].rank(method='first'), 5, labels=False) if A[k].nunique() > 5 else None
    ext = (f' | top fifth fwd {A[q == 4].fwd.mean():+.1f} pts, bottom fifth {A[q == 0].fwd.mean():+.1f}' if q is not None else f' | +1: {A[A[k] > 0].fwd.mean():+.1f} pts (n{(A[k] > 0).sum()}), -1: {A[A[k] < 0].fwd.mean():+.1f} (n{(A[k] < 0).sum()})')
    ok = abs(i1) >= .03 and abs(i2) >= .03 and np.sign(i1) == np.sign(i2)
    print(f'  {k:<4} IC Dec-May {i1:+.3f} / Jun-Nov {i2:+.3f}' + ext + ('   <- consistent' if ok else ''), flush=True)
# ---------------------------------------------------------------- (B) intraday momentum on QQQ (NQ points)
import yfinance as yf
H = pd.read_parquet(glob.glob(f'{ROOT}/lse_ohlcv_cache/QQQ_5m_*.parquet')[0]); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
H = H.drop_duplicates('dt').set_index('dt').sort_index(); H['m'] = H.index.hour * 60 + H.index.minute; H['d'] = H.index.normalize()
R_ = H[(H.m >= 570) & (H.m < 960)]
day = pd.DataFrame({'c16': R_.groupby('d').close.last(), 'c10': R_[R_.m < 600].groupby('d').close.last(), 'o1530': R_[R_.m >= 930].groupby('d').open.first()}).dropna()
day['pc'] = day.c16.shift(1); day['r1'] = day.c10 / day.pc - 1; day['rl'] = day.c16 / day.o1530 - 1
NQd = yf.download('NQ=F', start='2017-12-01', interval='1d', progress=False, auto_adjust=False)['Close'].squeeze(); NQd.index = NQd.index.normalize()
day['lvl'] = NQd.reindex(day.index - pd.Timedelta(days=1), method='ffill').values; day = day.dropna()
day['pts'] = np.sign(day.r1) * day.rl * day.lvl - 1.0; day['big'] = day.r1.abs() > day.r1.abs().rolling(252, min_periods=120).median().shift(1)
print(f'\n(B) intraday momentum, QQQ proxy -> NQ points, {len(day)} days', flush=True)
for nm, z in (('every day', day), ('|first window| above its 1y median', day[day.big])):
    a, b = z[z.index < '2022-01-01'], z[z.index >= '2022-01-01']
    print(f'  {nm:<36} n{len(z)} win {z.pts.gt(0).mean():.0%} avg {z.pts.mean():+.1f} pts | 2018-21 {a.pts.mean():+.1f} / 2022-26 {b.pts.mean():+.1f} | by year ' + str(z.groupby(z.index.year).pts.sum().round(0).astype(int).to_dict()), flush=True)
