"""Nov 23 2026 report 397. Rey: run the consolidation / AMD-sequence feature (report 396) on ALL books to see if it helps any algo.  Definition, parameters and timeframes are exactly report 396 (fixed there before running); the pooled higher-timeframe flag S_HTF = sequence on 1h OR 4h OR daily is now PRE-DECLARED as the primary test (positive direction), everything else descriptive.
Books: (A) large-cap vector, ALL 31,170 live-exit 15m flips on the 54 names 2018-2026 WITHOUT the options gate (so 2018, the held-out year, is included; report 396 had only the gated 12,360 from 2019); (B) sub-$5 small-cap vector (the mid/small book trades entries < $5): 108 names of 15m LSE data 2019-2026, same live exits, entry price < $5; (C) mid/small Yahoo 1h bars and NQ 1h cover only Apr 2025 / Jun 2024 onward (too short for a train/test split) - NOT tested.
Periods: A: 2018 (held out) | 2019-21 | 2022-26; B: 2019-21 | 2022-26.  Test = yes-minus-no mean flip return, week-clustered OLS t.  Admit (primary S_HTF): same sign in every period and |t| >= 2 in the test periods."""
import sys, glob, os, numpy as np, pandas as pd, warnings, statsmodels.formula.api as smf, multiprocessing as mp; warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/claude/fartpie/xs')
import report_295_flip_quality as q
from vector_pine import run_state_machine
from vector_core import compute_atr_series, ATR_LEN
_s = open('/home/claude/fartpie/xs/report_396_consolidation_mtf.py').read(); exec(_s[_s.index("N = 20; KW"):_s.index("b = pd.read_parquet")].replace("SP = ", "_x = ")) if False else None
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; U = '/mnt/user-data/uploads/fartpie/'; N = 20; KW = 4.0
exec(_s[_s.index("def tf_bars"):_s.index("rows = []; fs =")])
def feats(H, T):
    r_ = np.asarray(run_state_machine(H[['open', 'high', 'low', 'close']].assign(datetime=H.dt))[0]); pos = {t: i for i, t in enumerate(H.dt.values)}; TFS = {rule: analyse(tf_bars(H, rule)).sort_values('known') for rule in ('15m', '1h', '4h', 'D')}; ks = []; meta = []
    for _, e in T.iterrows():
        j = pos.get(np.datetime64(e.entry))
        if j is None: continue
        k = j - 1; n = 0
        while k > 0 and not (r_[k] == 1 and r_[k - 1] != 1):
            k -= 1; n += 1
            if n > 40: k = None; break
        if k is None: continue
        ks.append(H.dt.values[k] + np.timedelta64(15, 'm')); meta.append((e.entry, e.ret, H.open.values[j]))
    if not ks: return None
    Q = pd.DataFrame({'t': ks, 'entry': [m[0] for m in meta], 'ret': [m[1] for m in meta], 'px': [m[2] for m in meta]}).sort_values('t')
    for rule, S in TFS.items():
        m = pd.merge_asof(Q[['t']], S, left_on='t', right_on='known', direction='backward'); Q[f'C_{rule}'] = m.cons.values.astype(float); Q[f'S_{rule}'] = m.seq.values.astype(float)
    return Q.drop(columns='t')
def work(a):
    ds, tk = a
    try:
        if ds == 'A':
            f = sorted(glob.glob(U + f'lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet'))[-1]; H = q.build(f); T = pd.read_parquet(SP + 'tr390_BASE.parquet'); T = T[T.tk == tk]; T['entry'] = pd.to_datetime(T.entry)
        else:
            d = pd.read_parquet(SP + f'lse15s/{tk}.parquet').rename(columns={'ts': 'datetime'}); d['datetime'] = pd.to_datetime(d.datetime).dt.tz_localize('UTC'); tmp = SP + f'tmp397_{tk}.parquet'; d.to_parquet(tmp); H = q.build(tmp); os.remove(tmp); T = pd.DataFrame(q.trades(H, tk)); T['entry'] = pd.to_datetime(T.entry)
        Q = feats(H, T); 
        if Q is None: return ds, None
        Q['tk'] = tk; Q['ds'] = ds; return ds, Q
    except Exception as e: print('ERR', ds, tk, str(e)[:100], flush=True); return ds, None
if __name__ == '__main__':
    namesA = sorted(pd.read_parquet(SP + 'tr390_BASE.parquet').tk.unique()); namesB = sorted(f[:-8] for f in os.listdir(SP + 'lse15s') if f.endswith('.parquet')); jobs = [('A', t) for t in namesA] + [('B', t) for t in namesB]; out = []
    with mp.Pool(2) as P:
        for i, (ds, Q) in enumerate(P.imap_unordered(work, jobs)):
            if Q is not None: out.append(Q)
            if i % 10 == 0: print(i, len(jobs), flush=True)
    F = pd.concat(out, ignore_index=True); F.to_parquet(SP + 'amd397.parquet'); print('rows', F.groupby('ds').size().to_dict(), flush=True)
