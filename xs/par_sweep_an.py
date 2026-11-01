"""Nov 1 2026: the vector settings sweep through the live book (6 slots, ratio tie-break, 8% breaker)."""
import os, glob, sys, numpy as np, pandas as pd
os.chdir('/tmp/claude-0/ml'); exec(open('/home/claude/fartpie/xs/book_mtm.py').read().split("if __name__")[0])
src = open('/home/claude/fartpie/xs/stop_book.py').read(); exec(src[src.index('def sim('):src.index("B = S[S['var']")])
VS = sys.argv[1:] or ['vec', 'piv5', 'piv6', 'piv10', 'piv12', 'brk0.2', 'brk0.25', 'brk0.45', 'brk0.5', 'atr10', 'atr20', 'mm0.5', 'mm1.2', 'conf2', 'ema13_34', 'ema34_89']
P = [(np.datetime64('2019-01-01'), np.datetime64('2022-01-01')), (np.datetime64('2022-01-01'), np.datetime64('2027-01-01'))]
for v in VS:
    S = prep([f for f in glob.glob('par/*.parquet')]) if False else None
    Z = prep_v(v) if False else None
    D = pd.concat(pd.read_parquet(f).query('var == @v') for f in glob.glob('par/*.parquet')).assign(var='vec')
    tmpf = f'/tmp/claude-0/ml/_par_{v}.parquet'; D.to_parquet(tmpf); Z = prep([tmpf])
    print(f'  {v:<12}' + ''.join(f'{c:>+8.1%} /{d:>4.0%}  ' for c, d in (sim(Z, a, b) for a, b in P)) + f'  {len(Z):>6} trades  avg {Z.ret.mean():+.2%}', flush=True)
