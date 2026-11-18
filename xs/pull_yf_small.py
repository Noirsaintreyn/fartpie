"""yfinance earnings dates (EPS est/actual/surprise) + analyst rating changes for the 753-name universe. Free; ~1 name/s."""
import os, time, warnings, pandas as pd, yfinance as yf; warnings.filterwarnings('ignore')
U = [l.strip() for l in open('/mnt/user-data/uploads/fartpie/midsmall_universe.txt') if l.strip() and not l.startswith('#')]
for t in U:
    f = f'data/fund/yf_small/{t}_earn.parquet'
    if os.path.exists(f): continue
    k = yf.Ticker(t); n = 0
    try:
        e = k.get_earnings_dates(limit=80); e.to_parquet(f); n = len(e)
        try: k.upgrades_downgrades.to_parquet(f'data/fund/yf_small/{t}_ud.parquet')
        except Exception: pass
    except Exception as x: pd.DataFrame().to_parquet(f); print(t, 'ERR', str(x)[:60], flush=True); continue
    print(t, n, e.index.min() if n else '', flush=True); time.sleep(0.6)
