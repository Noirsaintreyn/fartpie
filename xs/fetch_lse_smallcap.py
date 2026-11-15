"""Pull 15m candles from London Strategic Edge for the sub-$5 flip tickers (xs/smallcap_tickers_px5.txt), to test the TWAP / accumulation footprint where it was meant to work.
Run on the Mac (needs LSE_API_KEY in .env). Step 1: python3 xs/fetch_lse_smallcap.py probe   -> prints, for 8 tickers, first date / bars per day / pre-2022 coverage (does LSE keep small-cap intraday history?)
Step 2: python3 xs/fetch_lse_smallcap.py all [N]   -> caches parquet in lse_ohlcv_cache/ for the first N tickers (default all, ordered by flip count). Resumable (cache hits skip)."""
import sys, os, pandas as pd; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fetch_lse_ohlcv import fetch_ohlcv
T = open(os.path.join(os.path.dirname(__file__), 'smallcap_tickers_px5.txt')).read().split()
A, B = pd.Timestamp('2019-01-01'), pd.Timestamp(os.getenv('LSE_END', '2026-11-13'))
if sys.argv[1] == 'probe':
    for tk in T[:8]:
        try:
            d = fetch_ohlcv(tk, '15m', A, B, use_cache=False)
            if d.empty: print(tk, 'EMPTY'); continue
            d['day'] = d.datetime.dt.date; print(tk, len(d), 'rows', d.datetime.min(), '->', d.datetime.max(), '| bars/day median', int(d.groupby('day').size().median()), '| pre-2022 rows', int((d.datetime < '2022-01-01').sum()), '| zero-volume share %.2f' % (d.volume == 0).mean())
        except Exception as e: print(tk, 'ERR', str(e)[:100])
else:
    n = int(sys.argv[2]) if len(sys.argv) > 2 else len(T)
    for i, tk in enumerate(T[:n]):
        try: d = fetch_ohlcv(tk, '15m', A, B); print(i, tk, len(d), flush=True)
        except Exception as e: print(i, tk, 'ERR', str(e)[:100], flush=True)
