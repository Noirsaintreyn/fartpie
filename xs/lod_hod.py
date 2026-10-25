"""Oct 25 2026: once the low of day looks in, how far does the high go? (and the mirror for a high that looks in)
Daily vol sigma = std of the prior 20 daily close-to-close returns. Checkpoints 10:30 / 12:00 / 14:00 ET.
'Low looks in' = the day's low so far was set >= 30 min before the checkpoint and price is >= 0.5 sigma above it.
Measured: P(the low holds to the close), and the final high's distance above the low in sigma units (quantiles).
Calibrated on 2019-2021, then checked on 2022-2026: does the 80% / 95% band hold the high as often as it should?"""
import glob, numpy as np, pandas as pd, sys, warnings; warnings.filterwarnings('ignore')
TKS = sys.argv[1:] or ['QQQ', 'SPY', 'AAPL', 'MSFT', 'NVDA', 'AMD', 'TSLA', 'META', 'AMZN', 'GOOGL']
rows = []
for tk in TKS:
    f = (glob.glob(f'lse_clean/{tk}_15m_2018-01-01_*.parquet') or glob.glob(f'lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet'))
    if not f: print('no bars', tk); continue
    B = pd.read_parquet(f[0], columns=['datetime', 'open', 'high', 'low', 'close']); B['dt'] = pd.to_datetime(B.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = B.dt.dt.hour * 60 + B.dt.dt.minute; B = B[(m >= 570) & (m < 960)].drop_duplicates('dt').sort_values('dt'); B['d'] = B.dt.dt.normalize(); B['m'] = B.dt.dt.hour * 60 + B.dt.dt.minute
    C = B.groupby('d').close.last(); sig = C.pct_change().rolling(20).std().shift(1)
    for d, g in B.groupby('d'):
        s = sig.get(d)
        if not s or s != s or len(g) < 24 or g.d.iloc[0] < pd.Timestamp('2019-01-01'): continue
        H, L = g.high.max(), g.low.min()
        for cp in (630, 720, 840):
            k = g[g.m < cp]
            if len(k) < 2: continue
            lo, hi, last = k.low.min(), k.high.max(), k.close.iloc[-1]
            tl = k.m[k.low.idxmin()]; th = k.m[k.high.idxmax()]
            if cp - tl >= 30 and last >= lo * (1 + .5 * s):
                rows.append(dict(tk=tk, d=d, cp=cp, side='low in', holds=L >= lo, ext=(H / lo - 1) / s, sofar=(hi / lo - 1) / s))
            if cp - th >= 30 and last <= hi * (1 - .5 * s):
                rows.append(dict(tk=tk, d=d, cp=cp, side='high in', holds=H <= hi, ext=(1 - L / hi) / s, sofar=(1 - lo / hi) / s))
X = pd.DataFrame(rows); X['per'] = np.where(X.d.dt.year >= 2022, '22-26', '19-21')
print(f'{len(X)} checkpoints, {X.tk.nunique()} names\n')
for side in ('low in', 'high in'):
    print(f'== {side}: range from the {side.split()[0]} to the close-of-day extreme, in daily-sigma units')
    for cp in (630, 720, 840):
        a = X[(X.side == side) & (X.cp == cp)]; tr = a[a.per == '19-21']; te = a[a.per == '22-26']
        q = tr.ext.quantile([.5, .8, .95]).values
        cov80 = (te.ext <= q[1]).mean(); cov95 = (te.ext <= q[2]).mean()
        print(f'  {cp // 60}:{cp % 60:02d}  n {len(a):5d}  {side.split()[0]} holds {a.holds.mean():.0%}   '
              f'median {q[0]:.2f}  80% {q[1]:.2f}  95% {q[2]:.2f} sigma (fit 19-21)  ->  22-26 inside 80% band: {cov80:.0%}, 95% band: {cov95:.0%}'
              f'   (already {tr.sofar.median():.2f} sigma by then)')
    print()
