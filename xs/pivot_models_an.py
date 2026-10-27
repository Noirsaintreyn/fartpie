import numpy as np, pandas as pd, glob
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
exec(open('xs/book_mtm.py').read().split("if __name__")[0])
src = open('xs/stop_book.py').read(); exec(src[src.index('def sim('):src.index("B = S[S['var']")])
P_ = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
def show(nm, Z): print(f'  {nm:<62}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in (sim(Z, a, b) for a, b in P_)) + f'  n {len(Z)} avg {Z.ret.mean()*100:+.2f}%', flush=True)
print(f'{"":<64}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}')
S = prep(glob.glob('xs/ext/adv/tr/*.parquet'))
print('== new detectors / fib (through the live engine + book)')
for v in ('live', 'dow5', 'dow3', 'dow5_atr', 'pivclust', 'fib_tp_half'): show(v, S[S['var'] == v])
show('fib_pullback alone', S[S['var'] == 'fib_pullback']); L = S[S['var'] == 'live']
show('live + fib_pullback (same slots)', pd.concat([L.assign(ratio=L.ratio + 100), S[S['var'] == 'fib_pullback']]))
F = pd.concat(pd.read_parquet(f) for f in glob.glob('xs/ext/adv/ft/*.parquet')).drop(columns=['exit', 'ret'])
L = L.merge(F, on=['tk', 'h1', 'entry'], how='left'); L['half'] = np.where(L.d.dt.year >= 2022, '22-26', '19-21'); tr = L.half == '19-21'
def tab(lab, q, Z):
    g = Z.groupby([q, 'half']).ret.agg(['mean', 'size']).unstack(); g[[('mean', '19-21'), ('mean', '22-26')]] *= 100; print(f'\n{lab}'); print(g.round(3).to_string())
# 1 swing language clusters (fit on 2019-21 only)
SW = [f'sw{i}' for i in range(10)]; ok = L[SW].notna().all(axis=1)
sc = StandardScaler().fit(L.loc[ok & tr, SW].clip(-20, 20)); km = KMeans(8, n_init=10, random_state=0).fit(sc.transform(L.loc[ok & tr, SW].clip(-20, 20)))
L.loc[ok, 'cl'] = km.predict(sc.transform(L.loc[ok, SW].clip(-20, 20)))
tab('swing-language cluster (k-means on the last 3 ATR swings, fit 2019-21)', L.loc[ok, 'cl'], L[ok])
by = L[ok & tr].groupby('cl').ret.mean(); bad = by.nsmallest(2).index.tolist(); good = by.nlargest(2).index.tolist()
show(f'skip the 2 worst 2019-21 clusters {bad}', L[~L.cl.isin(bad)])
z = L.copy(); z['ratio'] = z.ratio + z.cl.map(by.rank(pct=True)).fillna(.5) * .5; show('tie-break by 2019-21 cluster rank', z)
# 2 duration (swings since the 1h flip)
tab('completed up-swings since the 1h flip (semi-Markov age)', pd.cut(L.nsw, [-1, 0, 1, 2, 4, 99], labels=['0', '1', '2', '3-4', '5+']), L[L.nsw.notna()])
h = L[tr & L.nsw.notna()].groupby(pd.cut(L.nsw, [-1, 0, 1, 2, 4, 99])).ret.mean(); wb = h.idxmin()
show(f'skip swing-age bucket {wb} (worst 2019-21)', L[~((L.nsw > wb.left) & (L.nsw <= wb.right))])
# 3 nesting
L['nest'] = L[['n2', 'n5', 'n12', 'n30']].sum(axis=1)
tab('fractal nesting score (sum of Dow states at pivot scales 2/5/12/30)', L.nest, L)
tab('30-scale structure at entry', L.n30, L)
show('skip when the 30-scale structure is bear', L[~(L.n30 == -1)]); show('skip nesting score <= -2', L[~(L.nest <= -2)])
show('only when 30-scale bull and 2-scale bull', L[(L.n30 == 1) & (L.n2 == 1)])
# 4 analog / motif model: nearest neighbours on the ATR-normalized 32-bar path, walk-forward by year
PC = [f'p{i}' for i in range(32)]; L['analog'] = np.nan
for Y in range(2020, 2027):
    lib = L[(L.exit < pd.Timestamp(f'{Y}-01-01')) & L[PC].notna().all(axis=1)]; te = (L.entry.dt.year == Y) & L[PC].notna().all(axis=1)
    if len(lib) < 500 or not te.any(): continue
    nn = NearestNeighbors(n_neighbors=100).fit(lib[PC].clip(-30, 30).values); _, ix = nn.kneighbors(L.loc[te, PC].clip(-30, 30).values)
    L.loc[te, 'analog'] = lib.ret.values[ix].mean(axis=1)
Z = L[L.analog.notna()]; print('\nanalog model rank IC by year:', ' '.join(f'{y}: {Z[Z.entry.dt.year == y].analog.corr(Z[Z.entry.dt.year == y].ret, method="spearman"):+.3f}' for y in range(2020, 2027)))
tab('analog forecast quintile (2020-26 out of sample)', pd.qcut(Z.analog.rank(method='first'), 5, labels=[1, 2, 3, 4, 5]), Z)
B = L[L.entry >= '2020-01-01']
show('(2020+) live', B); show('(2020+) skip bottom 20% analog forecasts', B[~(B.analog < B.analog.quantile(.2))])
z = B.copy(); z['ratio'] = z.ratio + z.analog.rank(pct=True).fillna(.5) * .5; show('(2020+) tie-break by analog forecast', z)
