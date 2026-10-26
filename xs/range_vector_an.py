import pandas as pd, numpy as np
X = pd.read_parquet('xs/ext/range_vector_daily.parquet'); X['per'] = np.where(X.d.dt.year >= 2022, 'te', 'tr')
X = X[X.sofar >= 0]  # all
X['ctx'] = np.where(X.c15 & X.c1, 'both', np.where(X.c15, '15m', np.where(X.c1, '1h', 'none')))
X['tod'] = pd.cut(X.t, [599, 659, 779, 899, 960], labels=['10-11', '11-13', '13-15', '15-15:30'])
print('DAILY  n', len(X))
print('\nP(low of day already in) by time and vector context (all years):')
print(X.pivot_table(index='tod', columns='ctx', values='holds', aggfunc='mean').round(2).to_string())
print('\nshare of checkpoints in each context:'); print(X.ctx.value_counts(normalize=True).round(2).to_string())
print('\nmedian final high above the low (budget units) by time x ctx:')
print(X.pivot_table(index='tod', columns='ctx', values='ext', aggfunc='median').round(2).to_string())
print('\nmedian remaining upside beyond high so far (budget units):')
X['rem'] = X.ext - X.sofar
print(X.pivot_table(index='tod', columns='ctx', values='rem', aggfunc='median').round(2).to_string())
# models
tr, te = X[X.per == 'tr'], X[X.per == 'te'].copy()
def score(nm, pred, lo80=None, hi80=None):
    e = (pred - te.ext).abs(); s = f'{nm:<34} median abs err {e.median():.3f}   mean {e.mean():.3f}'
    if hi80 is not None: s += f'   80% band covers {((te.ext >= lo80) & (te.ext <= hi80)).mean():.0%}'
    print(s)
print('\nPredicting the final high (budget units above the low), fit 2019-21, scored 2022-26:')
score('site: max(sofar, 1.0)', np.maximum(te.sofar, 1.0))
for keys, nm in ((['tod'], 'time of day'), (['tod', 'ctx'], 'time of day x vector'), (['ctx'], 'vector only')):
    # predict remaining upside quantiles, add to sofar
    q = tr.groupby(keys).rem.quantile([.1, .5, .9]).unstack()
    j = te.join(q, on=keys)
    score(nm, te.sofar + j[.5], te.sofar + j[.1], te.sofar + j[.9])
# finer: also by sofar bucket
for X_ in (tr, te): X_['sb'] = pd.cut(X_.sofar, [-1, .25, .5, .75, 1, 1.5, 99])
for keys, nm in ((['tod', 'sb'], 'time x already-traveled'), (['tod', 'sb', 'ctx'], 'time x traveled x vector')):
    q = tr.groupby(keys).rem.quantile([.1, .5, .9]).unstack(); j = te.join(q, on=keys)
    score(nm, te.sofar + j[.5], te.sofar + j[.1], te.sofar + j[.9])
print('\nP(low holds) out of sample check: fit 19-21 vs realized 22-26 by tod x ctx')
a = tr.pivot_table(index='tod', columns='ctx', values='holds', aggfunc='mean'); b = te.pivot_table(index='tod', columns='ctx', values='holds', aggfunc='mean')
print(pd.concat({'19-21': a, '22-26': b}, axis=1).round(2).to_string())

Y = pd.read_parquet('xs/ext/range_vector_weekly.parquet'); Y['per'] = np.where(Y.d.dt.year >= 2022, 'te', 'tr'); Y['rem'] = Y.ext - Y.sofar
Y['day'] = Y.dow.map({0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu'})
print('\n\nWEEKLY  n', len(Y), ' (checkpoint = that day\'s close; context = 1h vector bull with its flip on/after the week-low day)')
print('P(week low already in):'); print(Y.pivot_table(index='day', columns='c1', values='holds', aggfunc='mean').round(2).to_string())
print('share with vector context:', round(Y.c1.mean(), 2))
print('median final week high above the low (weekly budget units):'); print(Y.pivot_table(index='day', columns='c1', values='ext', aggfunc='median').round(2).to_string())
tr, te = Y[Y.per == 'tr'], Y[Y.per == 'te'].copy()
print('Predicting final week high, fit 19-21, scored 22-26:')
score('site-style: max(sofar, 1.0)', np.maximum(te.sofar, 1.0))
for X_ in (tr, te): X_['sb'] = pd.cut(X_.sofar, [-1, .25, .5, .75, 1, 1.5, 99])
for keys, nm in ((['day'], 'day of week'), (['day', 'c1'], 'day x vector'), (['day', 'sb'], 'day x traveled'), (['day', 'sb', 'c1'], 'day x traveled x vector')):
    q = tr.groupby(keys).rem.quantile([.1, .5, .9]).unstack(); j = te.join(q, on=keys)
    score(nm, te.sofar + j[.5], te.sofar + j[.1], te.sofar + j[.9])
