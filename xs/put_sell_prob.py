# Oct 25 2026: short-put proxy - after a screener flip, P(stock ends above entry - k*sigma*sqrt(n)) vs random days in the same names
exec(open('xs/options_model.py').read().split('\ndef run(')[0])
"""short-put proxy: after a flip, does the stock finish above entry - k*sigma*sqrt(t) more often than on a random day in the same names?"""
d = T.entry.dt.normalize(); sig_d = (CL.pct_change().rolling(20).std()).shift(1)
rows = []
for tk, g in T[T.ratio >= 1.5].groupby('tk'):
    if tk not in CL.columns: continue
    c = CL[tk].dropna(); s = sig_d[tk]
    idx = c.index
    for e in g.entry.dt.normalize().unique():
        k = idx.searchsorted(e)
        if k >= len(idx) - 11 or k < 1: continue
        p0 = c.iloc[k - 1]; sd = s.iloc[k]                                 # prior close as the reference
        for n in (5, 10):
            rows.append(dict(tk=tk, d=e, n=n, z=(c.iloc[k - 1 + n] / p0 - 1) / (sd * np.sqrt(n)) if sd == sd and sd > 0 else np.nan, flip=True))
    for k in range(260, len(idx) - 11, 3):                                  # baseline: every 3rd day, same names
        p0 = c.iloc[k - 1]; sd = s.iloc[k]
        for n in (5, 10):
            rows.append(dict(tk=tk, d=idx[k], n=n, z=(c.iloc[k - 1 + n] / p0 - 1) / (sd * np.sqrt(n)) if sd == sd and sd > 0 else np.nan, flip=False))
X = pd.DataFrame(rows).dropna(); X = X[X.d >= '2019-01-01']; X['per'] = np.where(X.d.dt.year >= 2022, '22-26', '19-21')
for kk in (0.5, 1.0, 1.5):
    g = X.groupby(['n', 'per', 'flip']).z.apply(lambda z: (z > -kk).mean() * 100).unstack()
    print(f'P(stock ends above entry - {kk} sigma) %, flips vs random days:'); print(g.round(1).to_string()); print()
