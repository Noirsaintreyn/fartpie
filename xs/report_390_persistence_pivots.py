"""Nov 23 2026 report 390. Rey: model market SHAPE, rebuild the vector in a scale-free way.  Hypothesis: the vector works because it ratchets on the market's own pivots, but its pivots are chosen by a fixed 8-bar window (ta.pivothigh/low len 8) - an arbitrary scale.  Replace it with the causal form of 0-dim persistent homology: a swing high is confirmed at the first bar where price has retraced >= k x ATR below the running high (its persistence/prominence reached k ATR), a swing low mirror (the zig-zag filtration).  Pivot value delivered at the confirmation bar, exactly like the Pine pivot (so no look-ahead).  Everything else identical: break/confirm/ratchet logic, exits (report 295 live exits), 15m entries at next RTH open, 54 names, ratio gate >= 1.5 where ratio = rolling-252 median of the daily options ratio shifted one day (reproduces b302 exactly), 6 slots, 8% breaker, ratio-first order, 2+2bp.
Variants (pre-registered, k only; no other parameter touched): BASE (original pivots, rebuilt in this pipeline) | k=1.0 | k=2.0 | k=3.0 ATR.   Pass: Sharpe above BASE in BOTH halves for the same k, flip count not collapsing, and the trade-level mean flip return same sign train and test.   Multiple comparison: 3 variants -> one k must win both halves; a single winner of three is reported as such."""
import sys, glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/claude/fartpie/xs')
import multiprocessing as mp, vector_pine as vp, report_295_flip_quality as q
from vector_core import compute_atr_series, ATR_LEN
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; U = '/mnt/user-data/uploads/fartpie/'; _orig_piv = vp._piv; _orig_run = vp.run_state_machine
def zz(v, hi, thr):
    n = len(v); out = np.full(n, np.nan); m, mi = v[0], 0
    for i in range(1, n):
        t = thr[i]
        if hi:
            if v[i] > m: m, mi = v[i], i
            elif not np.isnan(t) and m - v[i] >= t: out[i] = m; m, mi = v[i], i
        else:
            if v[i] < m: m, mi = v[i], i
            elif not np.isnan(t) and v[i] - m >= t: out[i] = m; m, mi = v[i], i
    return out
def make_run(k):
    def run(df, **kw):
        if k is None: vp._piv = _orig_piv
        else:
            atr = compute_atr_series(df['high'].values, df['low'].values, df['close'].values, ATR_LEN); vp._piv = lambda v, L, hi: zz(v, hi, atr * k)
        return _orig_run(df, **kw)
    return run
fs = {os.path.basename(f).split('_15m_')[0]: f for f in sorted(glob.glob(U + 'lse_ohlcv_cache/*_15m_2018-01-01_*.parquet'))}
def work(a):
    var, tk = a; k = {'BASE': None, 'k1': 1.0, 'k2': 2.0, 'k3': 3.0}[var]; q.run_state_machine = make_run(k); vp.run_state_machine = make_run(k)
    H = q.build(fs[tk]); return var, q.trades(H, tk)
if __name__ == '__main__':
    b = pd.read_parquet(SP + 'b302.parquet'); names = sorted(b.tk.unique()); vars_ = ['BASE', 'k1', 'k2', 'k3']; jobs = [(v, t) for v in vars_ for t in names]; res = {v: [] for v in vars_}
    with mp.Pool(2) as P:
        for i, (v, T) in enumerate(P.imap_unordered(work, jobs)):
            res[v] += T
            if i % 20 == 0: print(i, len(jobs), flush=True)
    for v in vars_: pd.DataFrame(res[v]).to_parquet(SP + f'tr390_{v}.parquet')
    print('trades built', {v: len(res[v]) for v in vars_}, flush=True)
