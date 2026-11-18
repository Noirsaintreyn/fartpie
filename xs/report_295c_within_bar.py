"""Report 295c: WITHIN-BAR (bar fixed effects) regression: when several flips arrive in the same bar, does the higher-ratio name earn more, after beta, YZ20 vol, log $volume, 20d momentum? Train 2019-22 | test 2023-26. Bars with >= 3 flips."""
import numpy as np, pandas as pd, statsmodels.api as sm, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
A = pd.read_parquet(SP + 'flips295.parquet'); W = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet'); W = W[W.sp & ~W.h1 & (W['var'] == 'base+sticky')][['tk', 'entry', 'ratio']]
B = A.merge(W, on=['tk', 'entry']); B = B[(B.ratio >= 1.5) & (B.entry >= '2019-01-01') & (B.exit < '2026-09-26')].copy(); B['day'] = B.entry.dt.normalize()
J = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/site_wide_daily.parquet'); spy = J.SPY; FE = []
for tk in B.tk.unique():
    Y = pd.read_parquet(f'data/yahoo/daily_v/{tk}.parquet'); Y.index = pd.DatetimeIndex(Y.index).tz_localize(None).normalize(); r = np.log(Y.close).diff(); sr = np.log1p(spy.reindex(Y.index).fillna(0))
    beta = r.rolling(252, min_periods=120).cov(sr) / sr.rolling(252, min_periods=120).var(); yz = r.rolling(20).std()
    f = pd.DataFrame({'beta': beta, 'vol': yz, 'ldv': np.log((Y.close * Y.volume).rolling(50).mean()), 'mom20': np.log(Y.close).diff(20)}).shift(1); f['tk'] = tk; f['day'] = f.index; FE.append(f)
B = B.merge(pd.concat(FE), on=['tk', 'day'], how='left'); B['lr'] = np.log(B.ratio); B['n'] = B.groupby('entry').tk.transform('size'); B = B[B.n >= 3].dropna(subset=['beta', 'vol', 'ldv', 'mom20']).copy()
def fe(S, xs):
    Z = S[['ret'] + xs].copy(); Z = Z - S.groupby('entry')[['ret'] + xs].transform('mean'); Z[xs] = Z[xs] / Z[xs].std()
    m = sm.OLS(Z.ret.values, Z[xs].values).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(S.day)[0]}); return dict(zip(xs, zip(m.params * 100, m.tvalues))), len(S)
for lab, a, b in (('train 2019-22', '2019-01-01', '2023-01-01'), ('test 2023-26', '2023-01-01', '2027-01-01')):
    S = B[(B.entry >= a) & (B.entry < b)]
    for nm, xs in (('ratio alone', ['lr']), ('ratio + controls', ['lr', 'beta', 'vol', 'ldv', 'mom20']), ('controls only', ['beta', 'vol', 'ldv', 'mom20'])):
        r, n = fe(S, xs); print(f'{lab:14s} {nm:17s} n={n:5d} ' + ' '.join(f'{k}:{v[0]:+.2f}[{v[1]:+.1f}]' for k, v in r.items()))
