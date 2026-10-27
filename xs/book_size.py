"""Oct 26 2026: how many slots? Same live signals and tie-break, n slots each 1/n of the book (no leverage), 8% breaker.
Also 6 slots at 1/6 plus up to 2 extra slots at 1/6 each (up to 133% gross) for signals that arrive while full."""
import glob, numpy as np, pandas as pd
exec(open('xs/book_mtm.py').read().split("if __name__")[0])
src = open('xs/stop_book.py').read(); exec(src[src.index('def sim('):src.index("B = S[S['var']")])
ss = src[src.index('def sim('):src.index("P = [(np.datetime64")].replace('def sim(', 'def sim_w(').replace('def sim_w(Z, a, b, n=6, brk=.08)', 'def sim_w(Z, a, b, n=6, brk=.08, wn=None)').replace("pend.append((x, r_, ww / n))", "pend.append((x, r_, ww / (wn or n)))")
exec(ss)
L = prep(glob.glob('xs/ext/tm/*.parquet')); L = L[L['var'] == 'vec']
PER = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
X = pd.read_parquet('xs/ext/vt/series.parquet')['tm:vec']
print(f'{"":<44}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}')
for n in (3, 4, 5, 6, 7, 8, 10, 12, 15):
    R = [sim(L, a, b, n=n) for a, b in PER]; print(f'  {n:>2} slots, {1/n:.1%} each' + ' ' * 23 + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in R), flush=True)
for n in (7, 8):
    R = [sim_w(L, a, b, n=n, wn=6) for a, b in PER]; print(f'  {n} slots at 1/6 each (up to {n/6:.0%} gross)' + ' ' * 9 + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in R), flush=True)
