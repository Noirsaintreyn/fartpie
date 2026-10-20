"""Non-dividend-adjusted (Yahoo 'Close', split-adjusted only) daily closes for market-cap calcs.
Outputs xs/ext/close_px_sp.parquet and close_px_mid.parquet: date,ticker,close"""
import glob, os, pandas as pd
X = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(X)
for name, pat in (('sp', os.path.join(R, 'sp500_cache_raw2', '*.parquet')), ('mid', os.path.join(X, 'raw', 'ext_prices', '*.parquet'))):
    L = []
    for f in glob.glob(pat):
        if os.path.getsize(f) == 0: continue
        try: d = pd.read_parquet(f, columns=['datetime', 'close'])
        except Exception: continue
        d = d.rename(columns={'datetime': 'date'}); d['date'] = pd.to_datetime(d['date']).dt.tz_localize(None).dt.normalize()
        d = d[d.date >= '2016-01-01']; d['ticker'] = os.path.basename(f)[:-8]; L.append(d)
    out = pd.concat(L); out.to_parquet(os.path.join(X, 'ext', f'close_px_{name}.parquet'), index=False)
    print(name, out.shape, out.ticker.nunique())
