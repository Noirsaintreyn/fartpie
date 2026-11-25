"""Nov 25 2026: cumulative share of the live-group flips (sub-$5, 4-feature top 30%, 2022+) that touch +10% above the next-open entry within d trading days, before the vector exit. Stored in daily_base5_ref.json['curve'] for the report page."""
import numpy as np, pandas as pd, json, sys; sys.path.insert(0, 'xs')
from vector_pine import run_state_machine
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; O = pd.read_pickle(SP + 'ext/structure_small.pkl'); O['d'] = pd.to_datetime(O.d); W = pd.read_pickle(SP + 'ext/wide_daily_ohlcv.pkl'); R = json.load(open('daily_base5_ref.json')); Q = np.linspace(0, 1, 101)
def sc(r):
    v = []
    for f, s in R['signs'].items():
        x = r[f]
        if x == x: p = np.interp(x, R['ref'][f], Q); v.append(p if s > 0 else 1 - p)
    return np.mean(v)
S = O[(O.px < 5) & (O.d >= '2022-01-01')].copy(); S['s'] = S.apply(sc, axis=1); G = S[S.s >= R['cut']]; days = []; n = 0
for tk, g in G.groupby('tk'):
    df = W.get(tk)
    if df is None: continue
    df = df[~df.index.duplicated()]; ix = df.index; reg = run_state_machine(df.assign(datetime=ix))[0]; h = df.high.values; o = df.open.values
    for d in g.d:
        i = ix.searchsorted(d)
        if i + 2 >= len(ix) or ix[i] != d: continue
        e = o[i + 1]; n += 1; j = i + 1
        while j < len(ix):
            if h[j] >= 1.10 * e: days.append(j - i); break
            if reg[j] == -1 and j > i + 1: break
            j += 1
days = np.array(days); curve = {str(k): float((days <= k).sum() / n) for k in (1, 2, 3, 5, 10, 20, 40)}; curve['all'] = float(len(days) / n); curve['n'] = int(n); curve['median_days'] = float(np.median(days)); print(curve)
R['curve'] = curve; json.dump(R, open('daily_base5_ref.json', 'w'))
