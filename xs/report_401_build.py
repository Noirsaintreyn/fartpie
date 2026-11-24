"""Report 401 build: daily close + vector regime panels for the 54 research names + SPY (from LSE 15m cache)."""
import sys, glob, numpy as np, pandas as pd, multiprocessing as mp, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs'); import report_295_flip_quality as q
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; U = '/mnt/user-data/uploads/fartpie/'
def work(tk):
    try:
        f = sorted(glob.glob(U + f'lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet'))[-1]; H = q.build(f); m = H.m.values; ok = (m >= 570) & (m < 960)
        d = pd.DataFrame({'close': H.close.values, 'reg': H.reg.values, 'day': pd.to_datetime(H.dt.values).normalize()})[ok]
        g = d.groupby('day').last(); g['tk'] = tk; return g
    except Exception as e: print('ERR', tk, str(e)[:80], flush=True); return None
if __name__ == '__main__':
    tks = sorted(pd.read_parquet(SP + 'b302.parquet').tk.unique()) + ['SPY']
    with mp.Pool(2) as P: out = [g for g in P.map(work, tks) if g is not None]
    pd.concat(out).reset_index().to_parquet(SP + 'panel401.parquet'); print('done', len(out))
