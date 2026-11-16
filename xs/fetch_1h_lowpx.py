"""Fetch ~3 years of 1H bars (Yahoo, period=730d, interval=1h, unadjusted) for every name in the wide daily universe that closed below $5 on at least 20 days since 2023-12-01. Saved as a dict of DataFrames."""
import yfinance as yf, pandas as pd, pickle, time
E='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; D=pd.read_pickle(E+'wide_daily_ohlcv.pkl')
tks=[t for t,B in D.items() if (B[B.index>='2023-12-01'].close<5).sum()>=20]; print('names',len(tks),flush=True); out={}
for k,t in enumerate(tks):
    for a in range(2):
        try:
            h=yf.Ticker(t.replace('.','-')).history(period='730d',interval='1h',auto_adjust=False)
            if len(h): out[t]=h[['Open','High','Low','Close','Volume']]
            break
        except Exception as e: time.sleep(2)
    if k%50==0: print(k,len(out),flush=True); pickle.dump(out,open(E+'lowpx_1h_3y.pkl','wb'))
    time.sleep(.2)
pickle.dump(out,open(E+'lowpx_1h_3y.pkl','wb')); print('done',len(out))
