"""Nov 23 2026 report 399 (part 1): POOLED windows with cross-market channels.  Books: A = 54 large caps (31k flips, 2018-26, no gate), B = 108 sub-$5 small caps (50.8k flips, 2019-26, live exits).  Per flip, the 64 15m bars ending at the flip bar, 9 scale-free channels: (1-4) OHLC vs flip close / ATR, (5) log volume vs 64-bar median, (6) vector regime, (7) vector level distance in ATR, (8) SPY cumulative log return since the flip bar's time scaled by SPY's own sigma over the window (market context, co-movement), (9) rolling 20-bar correlation of the stock's 15m returns with SPY's.  All strictly pre-entry.  SPY file ends 2026-08-15 and has two corrupt days (May 16, 18 2026): flips with entries from 2026-05-14 on are dropped."""
import sys, glob, os, numpy as np, pandas as pd, warnings, multiprocessing as mp; warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/claude/fartpie/xs')
import report_295_flip_quality as q
from vector_pine import run_state_machine
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; U = '/mnt/user-data/uploads/fartpie/'; L = 64
S0 = pd.read_parquet(sorted(glob.glob(U + 'lse_ohlcv_cache/SPY_15m_2018-01-01_*.parquet'))[-1]); S0.index = pd.to_datetime(S0.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); SPYc = S0.sort_index().close; SPYc = SPYc[~SPYc.index.duplicated()]
def work(a):
    ds, tk = a
    try:
        if ds == 'A': f = sorted(glob.glob(U + f'lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet'))[-1]; H = q.build(f); T = pd.read_parquet(SP + 'tr390_BASE.parquet'); T = T[T.tk == tk].copy()
        else:
            d = pd.read_parquet(SP + f'lse15s/{tk}.parquet').rename(columns={'ts': 'datetime'}); d['datetime'] = pd.to_datetime(d.datetime).dt.tz_localize('UTC'); tmp = SP + f'tmp399_{tk}.parquet'; d.to_parquet(tmp); H = q.build(tmp); os.remove(tmp); T = pd.DataFrame(q.trades(H, tk))
        T['entry'] = pd.to_datetime(T.entry); T = T[T.entry < '2026-05-14']
        r_, l_, a_, fl = run_state_machine(H[['open', 'high', 'low', 'close']].assign(datetime=H.dt)); r_ = np.asarray(r_); l_ = np.asarray(l_); a_ = np.asarray(a_)
        o, h, lo, c, v = (H[x].values.astype(float) for x in ('open', 'high', 'low', 'close', 'volume')); lv = np.log1p(v); med = pd.Series(lv).rolling(L, min_periods=20).median().values; pos = {t: i for i, t in enumerate(H.dt.values)}
        sp = SPYc.reindex(H.dt.values, method='ffill').bfill().values; lsp = np.log(sp); ls = np.log(c); rs = np.r_[np.nan, np.diff(ls)]; rm = np.r_[np.nan, np.diff(lsp)]; cor = pd.Series(rs).rolling(20, min_periods=15).corr(pd.Series(rm)).values
        X = []; M = []
        for _, e in T.iterrows():
            j = pos.get(np.datetime64(e.entry))
            if j is None: continue
            k = j - 1; n = 0
            while k > 0 and not (r_[k] == 1 and r_[k - 1] != 1):
                k -= 1; n += 1
                if n > 40: k = None; break
            if k is None or k < L or not a_[k] > 0: continue
            s = slice(k - L + 1, k + 1); aa = a_[k]; ref = c[k]; sg = np.nanstd(rm[s])
            if not np.isfinite(sg) or sg <= 0: continue
            x = np.stack([(o[s] - ref) / aa, (h[s] - ref) / aa, (lo[s] - ref) / aa, (c[s] - ref) / aa, np.clip(lv[s] - med[s], -3, 3), r_[s].astype(float), np.clip((l_[s] - c[s]) / aa, -8, 8), np.clip((lsp[s] - lsp[k]) / (sg * 8), -5, 5), np.nan_to_num(cor[s], nan=0.0)], 1)
            if not np.isfinite(x).all(): continue
            X.append(np.clip(x, -20, 20).astype(np.float32)); M.append((ds, tk, e.entry, e.ret, o[j]))
        return X, M
    except Exception as ex: print('ERR', ds, tk, str(ex)[:100], flush=True); return [], []
if __name__ == '__main__':
    namesA = sorted(pd.read_parquet(SP + 'tr390_BASE.parquet').tk.unique()); namesB = sorted(f[:-8] for f in os.listdir(SP + 'lse15s') if f.endswith('.parquet')); jobs = [('A', t) for t in namesA] + [('B', t) for t in namesB]; XS = []; MS = []
    with mp.Pool(2) as P:
        for i, (X, M) in enumerate(P.imap_unordered(work, jobs)):
            if X: XS.append(np.stack(X)); MS += M
            if i % 10 == 0: print(i, len(jobs), flush=True)
    X = np.concatenate(XS); np.savez_compressed(SP + 'win399.npz', X=X, ret=np.array([m[3] for m in MS], dtype=np.float32), entry=np.array([np.datetime64(m[2]) for m in MS]).astype('datetime64[ns]').astype('int64'), tk=np.array([m[1] for m in MS]), ds=np.array([m[0] for m in MS]), px=np.array([m[4] for m in MS], dtype=np.float32)); print('windows', X.shape, flush=True)
