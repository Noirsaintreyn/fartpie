"""Oct 26 2026: meta-labeling v2 - a walk-forward model that scores every vector signal at entry from everything we have collected,
used to choose WHICH signals get the 6 slots (not to change stops). Features (all known before entry):
 options ratio, track (15m/1h), entry time, weekday | 1h uptrend age, daily uptrend age, daily vector state | market / sector / stock-specific
 20-day trends, raw 20-day trend | changepoints in ~5 sessions | days since earnings, return since earnings | share of today's HAR range used,
 15m+1h both bull, week low confirmed, bars since the day's low
Model: LightGBM regression on the trade return (clipped +-10%), small trees; walk-forward by year: train on trades that exited before
Jan 1 of the test year (2019 seed: trained on 2018H2-only is too thin, so 2019 uses an expanding fit on 2019H1 for H2; 2019H1 left unscored)
Uses: tie-break by score, skip the bottom 20% / 33% of scores (cut from the training years), both."""
import numpy as np, pandas as pd, glob, warnings; warnings.filterwarnings('ignore')
import lightgbm as lgb
exec(open('xs/book_mtm.py').read().split("if __name__")[0])
src = open('xs/stop_book.py').read(); exec(src[src.index('def sim('):src.index("B = S[S['var']")])
L = prep(glob.glob('xs/ext/tm/*.parquet')); L = L[L['var'] == 'vec']
F = pd.concat(pd.read_parquet(f) for f in glob.glob('xs/ext/fx/*.parquet')).drop(columns=['exit', 'ret'])
RV = pd.concat(pd.read_parquet(f) for f in glob.glob('xs/ext/rv/*.parquet')); RV = RV[RV['var'] == 'live'][['tk', 'h1', 'entry', 'used', 'both', 'wk', 'age']]
L = L.merge(F, on=['tk', 'h1', 'entry'], how='left').merge(RV, on=['tk', 'h1', 'entry'], how='left')
L['tmin'] = L.entry.dt.hour * 60 + L.entry.dt.minute; L['dow'] = L.entry.dt.dayofweek; L['h1i'] = L.h1.astype(int)
for c in ('both', 'wk'): L[c] = L[c].astype(float)
X = ['ratio', 'h1i', 'tmin', 'dow', 'age1h', 'dage', 'dstate', 'mkt20', 'sec20', 'res20', 'stk20', 'cp5', 'dse', 'rse', 'used', 'both', 'wk', 'age']
L['score'] = np.nan; IC = {}
pc = dict(n_estimators=200, learning_rate=.03, num_leaves=15, min_child_samples=200, subsample=.8, subsample_freq=1, colsample_bytree=.8, verbose=-1)
for Y in range(2020, 2027):
    t0 = pd.Timestamp(f'{Y}-01-01'); tr = L[L.exit < t0]; te = L.entry.dt.year == Y
    m = lgb.LGBMRegressor(**pc).fit(tr[X], tr.ret.clip(-.1, .1)); L.loc[te, 'score'] = m.predict(L.loc[te, X])
    trs = pd.Series(m.predict(tr[X]), index=tr.index); L.loc[te, 'c20'] = trs.quantile(.2); L.loc[te, 'c33'] = trs.quantile(1 / 3)
    IC[Y] = L.loc[te, 'score'].corr(L.loc[te, 'ret'], method='spearman')
print('out-of-sample rank IC by year (score vs trade return):', ' '.join(f'{y}: {v:+.3f}' for y, v in IC.items()))
Z = L[L.score.notna()].copy(); q = pd.qcut(Z.score.rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
print('avg trade by score quintile (2020-26, out of sample):'); print((Z.groupby([q, np.where(Z.d.dt.year >= 2022, '22-26', '2020-21')]).ret.mean().unstack() * 100).round(3).to_string())
imp = pd.Series(m.booster_.feature_importance('gain'), index=X).sort_values(ascending=False); print('2026 model importance:', ', '.join(f'{k} {v / imp.sum():.0%}' for k, v in imp.head(8).items()))
P_ = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2020-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2020-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
def show(nm, Z): print(f'  {nm:<50}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in (sim(Z, a, b) for a, b in P_)), flush=True)
print(f'\n{"(scored years only)":<52}{"2020-21":>16}{"2022-26":>16}{"2020-26":>16}{"2026 YTD":>16}')
B = L[L.entry >= '2020-01-01']; show('live', B)
z = B.copy(); z['ratio'] = z.score.rank(pct=True); show('slots go to the highest score (tie-break)', z)
z = B.copy(); z['ratio'] = z.ratio + z.score.rank(pct=True) * .5; show('ratio first, score nudges', z)
show('skip bottom 20% of scores', B[~(B.score < B.c20)]); show('skip bottom 33% of scores', B[~(B.score < B.c33)])
z = B[~(B.score < B.c20)].copy(); z['ratio'] = z.score.rank(pct=True); show('skip bottom 20% + highest score first', z)
