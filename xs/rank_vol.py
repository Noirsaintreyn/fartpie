"""Oct 28 2026: keep the options ratio as the universe (>= 1.5) but rank simultaneous signals by something else.
Traits from daily RTH closes up to the prior session: rv20 (20d realized vol), rv60, mom20, mom60, dist from 20d high; and blends with the ratio (rank average)."""
import os, glob, numpy as np, pandas as pd
os.chdir('/tmp/claude-0/ml'); exec(open('/home/claude/fartpie/xs/book_mtm.py').read().split("if __name__")[0])
src = open('/home/claude/fartpie/xs/stop_book.py').read(); exec(src[src.index('def sim('):src.index("B = S[S['var']")])
L = prep(glob.glob('tm/*.parquet')); L = L[L['var'] == 'vec'].reset_index(drop=True)
F = []
for tk, g in L.groupby('tk'):
    b = bars(tk)
    if b is None: continue
    dc = b[1]; r = np.log(dc).diff()
    T = pd.DataFrame({'rv20': r.rolling(20).std(), 'rv60': r.rolling(60).std(), 'mom20': dc / dc.shift(20) - 1, 'mom60': dc / dc.shift(60) - 1,
                      'h20': dc / dc.rolling(20).max() - 1}).shift(1)            # known at the prior close
    x = g[['entry']].copy(); x['d'] = x.entry.dt.normalize(); x = x.join(T, on='d'); F.append(x.drop(columns=['entry', 'd']))
L = L.join(pd.concat(F))
P = [(np.datetime64('2019-01-01'), np.datetime64('2022-01-01')), (np.datetime64('2022-01-01'), np.datetime64('2027-01-01'))]
def show(nm, key):
    Z = L.copy(); Z['ratio'] = key.fillna(key.median())
    print(f'  {nm:<34}' + ''.join(f'{c:>+8.1%} /{d:>4.0%}   ' for c, d in (sim(Z, a, b) for a, b in P)), flush=True)
pr = lambda s: s.groupby(L.entry.dt.year).rank(pct=True)          # percentile within the year (no look-ahead in scale)
show('live: options ratio', L.ratio)
for f in ('rv20', 'rv60', 'mom20', 'mom60', 'h20'):
    show(f'{f} high first', L[f]); show(f'{f} low first', -L[f])
for f in ('rv20', 'rv60', 'mom20', 'mom60'):
    for w in (.5, 1.0):
        show(f'ratio + {w} x {f} (high)', pr(L.ratio) + w * pr(L[f])); show(f'ratio + {w} x {f} (low)', pr(L.ratio) - w * pr(L[f]))
rng = np.random.default_rng(0); R = []
for k in range(20):
    Z = L.copy(); Z['ratio'] = rng.random(len(Z)); R.append([sim(Z, a, b)[0] for a, b in P])
R = np.array(R); print(f'  {"random (20 runs) median":<34}{np.median(R[:, 0]):>+8.1%}         {np.median(R[:, 1]):>+8.1%}    range {R[:, 1].min():+.1%}..{R[:, 1].max():+.1%}')
