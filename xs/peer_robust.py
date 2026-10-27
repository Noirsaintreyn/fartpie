import numpy as np, pandas as pd, glob
exec(open('xs/book_mtm.py').read().split("if __name__")[0])
src = open('xs/stop_book.py').read(); exec(src[src.index('def sim('):src.index("B = S[S['var']")])
L = prep(glob.glob('xs/ext/tm/*.parquet')); L = L[L['var'] == 'vec']
L = L.merge(pd.read_parquet('xs/ext/px/feat.parquet'), on=['tk', 'h1', 'entry'], how='left')
P_ = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
Y = [(np.datetime64(f'{y}-01-01'), np.datetime64(f'{y + 1}-01-01')) for y in range(2019, 2027)]; base = [sim(L, a, b)[0] for a, b in Y]
def show(nm, Z):
    yr = [sim(Z, a, b)[0] for a, b in Y]
    print(f'  {nm:<52}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in (sim(Z, a, b) for a, b in P_)) + f'   beats live {sum(x > y + 1e-9 for x, y in zip(yr, base))}/8, ties {sum(abs(x - y) < 1e-9 for x, y in zip(yr, base))}', flush=True)
g = L.groupby('entry').size(); print('entry times with 2+ signals at once:', (g > 1).sum(), 'of', len(g), '| trades in them', g[g > 1].sum())
print(f'{"":<54}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}'); show('live', L)
for w in (.1, .25, .5, 1.0, 5.0):
    z = L.copy(); z['ratio'] = z.ratio + z.peer_bull.fillna(z.peer_bull.median()) * w; show(f'peer_bull tie-break, weight {w}', z)
z = L.copy(); z['ratio'] = z.peer_bull.fillna(.5); show('peer_bull only (ratio ignored)', z)
z = L.copy(); z['ratio'] = z.ratio + (z.peer_bull.fillna(.5) > .5) * .5; show('peers mostly bull first (binary)', z)
z = L.copy(); z['ratio'] = z.ratio + (z.peer_bull.fillna(.5) + z.lead.fillna(.5)) * .25; show('peer_bull + leader', z)
