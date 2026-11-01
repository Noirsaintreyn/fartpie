"""Nov 1 2026: trim the losing contexts. Walk-forward: for each year Y, using only flips that exited before Y, find the 'dead zones' -
single-feature quintile bins and two-feature bin pairs whose average trade was negative with n >= 150 (pairs) / 300 (singles), negative
in at least 2/3 of past years, then skip flips in those zones during year Y. Out-of-sample 2020-26, judged with the 40-run perturbed yardstick."""
import numpy as np, pandas as pd, itertools
exec(open('/home/claude/fartpie/xs/ml_ranker.py').read().split("y = L.ret.clip")[0])
B = pd.DataFrame(index=L.index)
for c in X_COLS:
    v = L[c]
    if v.nunique() <= 5: B[c] = v.fillna(-9).astype(str)
    else: B[c] = pd.qcut(v.rank(method='first'), 5, labels=False).astype('Int64').astype(str).where(v.notna(), 'na')   # quintile on full sample = labels only
yr = L.entry.dt.year; ret = L.ret
def zones(tr, pairs):
    Z = []
    cols = list(itertools.combinations(X_COLS, 2)) if pairs else [(c,) for c in X_COLS]
    for cs in cols:
        key = B.loc[tr, list(cs)].astype(str).agg('|'.join, axis=1)
        g = pd.DataFrame({'k': key, 'r': ret[tr], 'y': yr[tr]}).groupby('k')
        st = g.r.agg(['size', 'mean']); py = g.apply(lambda d: (d.groupby('y').r.mean() < 0).mean())
        ok = st[(st['size'] >= (150 if pairs else 300)) & (st['mean'] < 0) & (py.reindex(st.index) >= 2 / 3)]
        Z += [(cs, k) for k in ok.index]
    return Z
SKIP = pd.Series(False, index=L.index); log = []
for Y in range(2020, 2027):
    tr = L.exit < pd.Timestamp(f'{Y}-01-01'); te = yr == Y; z = zones(tr, False) + zones(tr, True)
    m = pd.Series(False, index=L.index)
    for cs, k in z: m |= te & (B[list(cs)].astype(str).agg('|'.join, axis=1) == k)
    SKIP |= m; log.append((Y, len(z), int(m.sum()), int(te.sum()), ret[m].mean() if m.any() else np.nan, ret[te & ~m].mean()))
    print(f'{Y}: {len(z)} dead zones from past data -> skips {int(m.sum())} of {int(te.sum())} flips; skipped avg {ret[m].mean() if m.any() else np.nan:+.2%} vs kept {ret[te & ~m].mean():+.2%}', flush=True)
L['skip'] = SKIP
P = [(np.datetime64('2020-01-01'), np.datetime64('2022-01-01')), (np.datetime64('2022-01-01'), np.datetime64('2027-01-01'))]
rng = np.random.default_rng(7); DROPS = [rng.random(len(L)) < .05 for _ in range(40)]
def runs(Z): return np.array([[sim(Z[~dm[Z.index]], a, b)[0] for a, b in P] for dm in [pd.Series(np.zeros(len(L), bool), L.index)] + [pd.Series(d, L.index) for d in DROPS]])
A = runs(L); Bk = runs(L[~L.skip])
for nm, r in (('live', A), ('trim dead zones', Bk)):
    print(f'  {nm:<16} exact {r[0, 0]:+.1%} / {r[0, 1]:+.1%}   median {np.median(r[1:, 0]):+.1%} / {np.median(r[1:, 1]):+.1%}')
w = (Bk[1:] > A[1:]).mean(axis=0); print(f'  trim beats live in the same perturbed run: 2020-21 {w[0]:.0%}, 2022-26 {w[1]:.0%}')
