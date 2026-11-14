"""Daily OHLCV 2018-01 -> now for the wide mid/small universe (xs/ext/close_px_mid tickers + midsmall_universe.txt) via yfinance, in batches. Output scratchpad/ext/wide_daily_ohlcv.pkl (dict ticker -> DataFrame). SURVIVORSHIP: only names still listed; flagged in the report."""
import pandas as pd, yfinance as yf, re, time, warnings; warnings.filterwarnings('ignore')
OUT = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'
a = set(pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/close_px_mid.parquet', columns=['ticker']).ticker.unique()); b = set(re.findall(r'[A-Z][A-Z\.\-]{0,5}', open('/mnt/user-data/uploads/fartpie/midsmall_universe.txt').read())); U = sorted(a | b); print('universe', len(a), len(b), len(U), flush=True)
res = {}
for i in range(0, len(U), 60):
    batch = U[i:i + 60]
    try:
        d = yf.download(batch, start='2018-01-01', interval='1d', auto_adjust=False, progress=False, threads=True, group_by='ticker')
        for tk in batch:
            try:
                x = d[tk][['Open', 'High', 'Low', 'Close', 'Volume']].dropna(); x.columns = ['open', 'high', 'low', 'close', 'volume']
                if len(x) > 300: res[tk] = x
            except Exception: pass
    except Exception as e: print('batch err', i, str(e)[:80], flush=True)
    print(i, len(res), flush=True)
pd.to_pickle(res, OUT + 'wide_daily_ohlcv.pkl'); print('done', len(res))
