"""Historical earnings dates + EPS surprise from Yahoo for the screener's names -> xs/ext/earn/<tk>.csv (resumable)."""
import os, time, yfinance as yf, pandas as pd
YF = {'BRKB': 'BRK-B', 'BFB': 'BF-B'}
for tk in open('xs/ext/earn/_tickers.txt').read().split():
    f = f'xs/ext/earn/{tk}.csv'
    if os.path.exists(f): continue
    for k in range(3):
        try:
            e = yf.Ticker(YF.get(tk, tk)).get_earnings_dates(limit=60)
            (e if e is not None else pd.DataFrame()).to_csv(f); print(tk, 0 if e is None else len(e), flush=True); break
        except Exception as ex:
            print(tk, 'ERR', str(ex)[:60], flush=True); time.sleep(5 * (k + 1))
    else: pd.DataFrame().to_csv(f)
    time.sleep(0.6)
print('DONE')
