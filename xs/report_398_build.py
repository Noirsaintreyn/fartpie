"""Nov 23 2026 report 398 (part 1): window tensors for a CNN-BiLSTM on vector flips.  For every one of the 31,170 live-exit 15m flips of the 54 large-cap names (2018-2026, the report-390 BASE trade set, NO options gate), take the 64 15m bars ending at the flip bar (strictly pre-entry).  7 channels, all scale-free: open/high/low/close relative to the flip-bar close divided by the flip-bar ATR; log volume relative to its 64-bar median (clipped +-3); vector regime (+1/-1); (vector level - close)/ATR (clipped +-8).  Output: scratchpad/win398.npz with X (n,64,7), ret, entry, tk."""
import sys, glob, os, numpy as np, pandas as pd, warnings, multiprocessing as mp; warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/claude/fartpie/xs')
import report_295_flip_quality as q
from vector_pine import run_state_machine
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; U = '/mnt/user-data/uploads/fartpie/'; L = 64
def work(tk):
    f = sorted(glob.glob(U + f'lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet'))[-1]; H = q.build(f); r_, l_, a_, fl = run_state_machine(H[['open', 'high', 'low', 'close']].assign(datetime=H.dt)); r_ = np.asarray(r_); l_ = np.asarray(l_); a_ = np.asarray(a_)
    o, h, lo, c, v = (H[x].values.astype(float) for x in ('open', 'high', 'low', 'close', 'volume')); lv = np.log1p(v); med = pd.Series(lv).rolling(L, min_periods=20).median().values; pos = {t: i for i, t in enumerate(H.dt.values)}
    T = pd.read_parquet(SP + 'tr390_BASE.parquet'); T = T[T.tk == tk]; T['entry'] = pd.to_datetime(T.entry); X = []; M = []
    for _, e in T.iterrows():
        j = pos.get(np.datetime64(e.entry))
        if j is None: continue
        k = j - 1; n = 0
        while k > 0 and not (r_[k] == 1 and r_[k - 1] != 1):
            k -= 1; n += 1
            if n > 40: k = None; break
        if k is None or k < L or not a_[k] > 0: continue
        s = slice(k - L + 1, k + 1); a = a_[k]; ref = c[k]
        x = np.stack([(o[s] - ref) / a, (h[s] - ref) / a, (lo[s] - ref) / a, (c[s] - ref) / a, np.clip(lv[s] - med[s], -3, 3), r_[s].astype(float), np.clip((l_[s] - c[s]) / a, -8, 8)], 1)
        if not np.isfinite(x).all(): continue
        X.append(np.clip(x, -20, 20).astype(np.float32)); M.append((tk, e.entry, e.ret))
    return X, M
if __name__ == '__main__':
    names = sorted(pd.read_parquet(SP + 'tr390_BASE.parquet').tk.unique())
    with mp.Pool(2) as P: R = P.map(work, names)
    X = np.concatenate([np.stack(r[0]) for r in R if r[0]]); M = [m for r in R for m in r[1]]
    np.savez_compressed(SP + 'win398.npz', X=X, ret=np.array([m[2] for m in M], dtype=np.float32), entry=np.array([np.datetime64(m[1]) for m in M]).astype('datetime64[ns]').astype('int64'), tk=np.array([m[0] for m in M])); print('windows', X.shape, flush=True)
