"""Oct 28 2026: slot count on the live book (live-engine trades, ratio tie-break, 8% breaker). n slots at 1/n each, and n slots at 1/6 each (leverage)."""
import os, glob, numpy as np, pandas as pd
os.chdir('/tmp/claude-0/ml'); exec(open('/home/claude/fartpie/xs/book_mtm.py').read().split("if __name__")[0])
src = open('/home/claude/fartpie/xs/stop_book.py').read(); exec(src[src.index('def sim('):src.index("B = S[S['var']")])
import inspect; print(inspect.signature(sim))
L = prep(glob.glob('tm/*.parquet')); L = L[L['var'] == 'vec'].reset_index(drop=True)
P = [(np.datetime64('2019-01-01'), np.datetime64('2022-01-01')), (np.datetime64('2022-01-01'), np.datetime64('2027-01-01'))]
for n in (4, 6, 8, 10, 12, 16, 99):
    print(f'{n:>3} slots @1/n ' + ''.join(f'{c:>+8.1%} /{d:>4.0%}  ' for c, d in (sim(L, a, b, n=n) for a, b in P)))
print('same 1/6 size per trade, more slots (more money in the market, up to n/6 = leverage)')
for n in (6, 8, 10, 12):
    Z = L.copy(); Z['w'] = n / 6
    print(f'{n:>3} slots @1/6 ' + ''.join(f'{c:>+8.1%} /{d:>4.0%}  ' for c, d in (sim(Z, a, b, n=n) for a, b in P)) + f'  max exposure {n / 6:.0%}')
