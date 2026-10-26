import pandas as pd, numpy as np
X = pd.read_parquet('xs/ext/range_vector_daily.parquet'); X['rem'] = X.ext - X.sofar
X['ctx'] = np.where(X.c15 & X.c1, 'both', np.where(X.c15, '15m', np.where(X.c1, '1h', np.where(X.age >= 3, 'none, low 45m+ old', 'none, low fresh'))))
X['tod'] = pd.cut(X.t, [599, 659, 779, 899, 960], labels=['10-11', '11-13', '13-15', '15-15:30'])
print(X.pivot_table(index='tod', columns='ctx', values='holds', aggfunc='mean').round(2).to_string())
tr, te = X[X.d.dt.year < 2022], X[X.d.dt.year >= 2022].copy()
q1 = tr.groupby(['tod']).rem.median(); q2 = tr.groupby(['tod', 'ctx']).rem.median()
te['p_site'] = np.maximum(te.sofar, 1.0); te['p_tod'] = te.sofar + te.tod.map(q1).astype(float); te['p_vec'] = te.sofar + te.join(q2.rename('q'), on=['tod', 'ctx']).q
te['p_ext'] = te.join(tr.groupby(['tod', 'ctx']).ext.median().rename('q'), on=['tod', 'ctx']).q.clip(lower=0); te['p_ext'] = np.maximum(te.sofar, te.p_ext)
print('\nmean abs error predicting the final high (budget units), 2022-26, by context:')
r = te.groupby('ctx').apply(lambda g: pd.Series({k: (g[k] - g.ext).abs().mean() for k in ('p_site', 'p_tod', 'p_vec', 'p_ext')} | {'n': len(g)}))
print(r.round(3).to_string())
print('\nwhen the 15m vector confirmed the low before 11:00: final high above low, quantiles (budget units):')
print(X[(X.ctx.isin(['15m', 'both'])) & (X.tod == '10-11')].ext.quantile([.1, .25, .5, .75, .9]).round(2).to_string())
