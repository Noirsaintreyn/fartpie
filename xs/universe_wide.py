"""Oct 28 2026: universe breadth. wide_T = live-rule trades (base+sticky) on 168 names with ratio >= 1.0. Same book (6 slots, ratio tie-break, 8% breaker).
Vary the ratio cutoff and S&P-only vs all names; 'liq' = liquidity flag."""
import os, numpy as np, pandas as pd
from collections import deque
src = open('/home/claude/fartpie/xs/stop_book.py').read(); exec(src[src.index('def sim('):src.index("B = S[S['var']")])
W = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet'); W['w'] = 1.0
W = W[(W.entry >= '2019-01-01') & ((W.entry.dt.hour * 60 + W.entry.dt.minute) < 840)].reset_index(drop=True); W['sp'] = W.sp.fillna(False).astype(bool)
print('names per year by cutoff (S&P):'); print({c: int(W[(W.ratio >= c) & W.sp].groupby(W.entry.dt.year).tk.nunique().median()) for c in (1.0, 1.2, 1.5, 2.0, 2.5)})
print('liq values', W.liq.value_counts().to_dict())
P = [(np.datetime64('2019-01-01'), np.datetime64('2022-01-01')), (np.datetime64('2022-01-01'), np.datetime64('2027-01-01'))]
def sh(nm, Z): print(f'  {nm:<36}' + ''.join(f'{c:>+8.1%} /{d:>4.0%}   ' for c, d in (sim(Z, a, b) for a, b in P)) + f' {Z.tk.nunique()} names', flush=True)
for sp in (True, False):
    for c in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0):
        Z = W[(W.ratio >= c) & (W.sp if sp else True)]
        sh(f'{"S&P" if sp else "all"} ratio >= {c}', Z)
for n in (6, 8, 10):
    Z = W[(W.ratio >= 1.0)]; print(f'  all >=1.0, {n} slots @1/n ' + ''.join(f'{c:>+8.1%} /{d:>4.0%}   ' for c, d in (sim(Z, a, b, n=n) for a, b in P)))
