"""Higher options-ratio cuts x universes (S&P live, Nasdaq-100, Dow, S&P+NDX, ALL ex-ETF). Same engine as universe_test.py / stop_1h_ev.py: 40 matched runs, 6 slots, ratio tie-break, 8% breaker. Cuts 1.5 (live), 2, 2.5, 3, 4."""
import numpy as np, pandas as pd, zlib, warnings; warnings.filterwarnings('ignore')
from collections import deque
a = open('xs/stop_1h_ev.py').read(); a = a[a.index('IDX = '):a.index('B0 = ev(XL)')].replace("Xo.exit.max()", "'2026-10-21'")
exec(a)
u = open('xs/universe_test.py').read(); u = u[u.index('NDX = set'):u.index("print('parity")]
exec(u.replace("print('unclassified", "pass #"))
def stats(nm, Y):
    a_, b_ = Y[Y.entry < '2022-01-01'], Y[Y.entry >= '2022-01-01']
    print(f'  {nm}: {Y.tk.nunique()} tickers, {len(Y)} trades ({len(a_)}/{len(b_)}), avg {a_.ret.mean()*100:+.2f}% / {b_.ret.mean()*100:+.2f}%, win {a_.ret.gt(0).mean():.0%} / {b_.ret.gt(0).mean():.0%}', flush=True)
B0 = ev(W[W.sp].reset_index(drop=True)); print('reference live (S&P, ratio>=1.5): 40 matched runs, 2019-21 / 2022-26'); line('L live', B0)
UN = [('S&P (live)', W.sp), ('Nasdaq-100', W.ndx), ('Dow 30', W.dow), ('S&P+NDX', W.sp | W.ndx), ('ALL ex-ETF', ~W.etf)]
for cut in (1.5, 2, 2.5, 3, 4):
    print(f'\n=== ratio >= {cut}')
    for nm, m in UN:
        Y = W[m & (W.ratio >= cut)].reset_index(drop=True); stats(f'{nm} cut {cut}', Y)
        if len(Y) >= 300: line(f'  book {nm} {cut}', ev(Y), B0)
