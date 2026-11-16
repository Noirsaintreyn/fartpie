"""Fetch split history (Yahoo) for every ticker in the wide daily universe, to rebuild the ACTUAL (unadjusted) price on each signal date. Yahoo prices are split-adjusted, so a name that later reverse-split shows a higher historical price than it really traded at."""
import yfinance as yf, pandas as pd, pickle, time
E='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; D=pd.read_pickle(E+'wide_daily_ohlcv.pkl'); out={}
for k,t in enumerate(D):
    try:
        s=yf.Ticker(t.replace('.','-')).splits
        out[t]=s if s is not None else pd.Series(dtype=float)
    except Exception as e: out[t]=None
    if k%100==0: print(k,flush=True); pickle.dump(out,open(E+'splits_wide.pkl','wb'))
    time.sleep(.12)
pickle.dump(out,open(E+'splits_wide.pkl','wb')); print('done',len(out),sum(1 for v in out.values() if v is not None and len(v)))
