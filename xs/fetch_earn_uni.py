"""Earnings dates + EPS estimate/reported/surprise (Yahoo) for every ticker in the 1h universe cache -> scratchpad earn_uni.pkl"""
import pickle, pandas as pd, yfinance as yf, warnings; warnings.filterwarnings('ignore')
from concurrent.futures import ThreadPoolExecutor
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; tks = list(pickle.load(open(SC + 'uni_cache.pkl', 'rb')).keys())
def g(tk):
    for _ in range(2):
        try:
            d = yf.Ticker(tk).get_earnings_dates(limit=24)
            if d is not None and len(d): return tk, d
        except Exception: pass
    return tk, None
with ThreadPoolExecutor(8) as ex: R = dict(ex.map(g, tks))
ok = {k: v for k, v in R.items() if v is not None}; print('earnings for', len(ok), 'of', len(tks)); pickle.dump(ok, open(SC + 'earn_uni.pkl', 'wb'))
