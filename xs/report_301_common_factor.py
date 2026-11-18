"""Nov 18 2026 report 301 (Rey: what do the boosted signals SHARE; what's the equation; why do they make flips bigger; what's the theory and where does it break).
Stage 1 (no ranking yet): decompose each flip's return into  market part (beta x SPY over the hold)  +  rest, and into  vol x hold-length x direction.
Then: do the signals (ratio rank, dollar volume, 12-1 mom, 200d trend, near-52w-high, 20d mom, rel-strength) collapse to ONE latent factor (PCA fit on 2019-22 only)?
What does that factor measure: beta? vol? persistence (variance ratio / autocorr of daily returns)? Everything point-in-time (shift 1). Split 2019-22 | 2023-26."""
import numpy as np, pandas as pd, warnings, glob; warnings.filterwarnings('ignore')
import statsmodels.api as sm
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
A = pd.read_parquet(SP + 'flips295.parquet'); W = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet'); W = W[W.sp & ~W.h1 & (W['var'] == 'base+sticky')][['tk', 'entry', 'ratio']]
B = A.merge(W, on=['tk', 'entry']); B = B[(B.ratio >= 1.5) & (B.entry >= '2019-01-01')].copy(); B['day'] = B.entry.dt.normalize()
# --- market path (SPY 15m, ET)
f = glob.glob('/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/SPY_15m*')[0]; Q = pd.read_parquet(f); Q['dt'] = pd.to_datetime(Q.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); Q = Q.set_index('dt').close.sort_index()
qi = Q.index; qv = Q.values
def qat(t): i = qi.searchsorted(t, side='right') - 1; return qv[np.clip(i, 0, len(qv) - 1)]
B['mkt'] = qat(B.exit.values) / qat(B.entry.values) - 1; B['hold_h'] = (B.exit - B.entry).dt.total_seconds() / 3600
# --- daily point-in-time features
Sd = pd.read_parquet('data/yahoo/daily_v/SPY.parquet'); Sd.index = pd.DatetimeIndex(Sd.index).tz_localize(None).normalize(); rm = np.log(Sd.close).diff(); sm20 = Sd.close / Sd.close.shift(20) - 1
FE = []
for tk in sorted(B.tk.unique()):
    Y = pd.read_parquet(f'data/yahoo/daily_v/{tk}.parquet'); Y.index = pd.DatetimeIndex(Y.index).tz_localize(None).normalize(); c = Y.close; r = np.log(c).diff(); rmm = rm.reindex(c.index)
    cov = r.rolling(60, min_periods=40).cov(rmm); beta = cov / rmm.rolling(60, min_periods=40).var()
    r5 = np.log(c / c.shift(5)); vr = r5.rolling(120, min_periods=80).var() / (5 * r.rolling(120, min_periods=80).var())
    ac1 = r.rolling(60, min_periods=40).apply(lambda z: pd.Series(z).autocorr(1), raw=True)
    d = pd.DataFrame({'trend': (c > c.rolling(200, min_periods=150).mean()).astype(float), 'mom': np.log(c.shift(21) / c.shift(252)), 'h52b': (c / c.rolling(252, min_periods=150).max() >= .9).astype(float),
                      'ldv': np.log((c * Y.volume).rolling(50).mean()), 'mom20': c / c.shift(20) - 1, 'beta': beta, 'vol20': r.rolling(20).std(), 'vr': vr, 'ac1': ac1}); d['rs20'] = d.mom20 - sm20.reindex(c.index)
    d = d.shift(1); d['tk'] = tk; d['day'] = d.index; FE.append(d)
FE = pd.concat(FE)
for k in ('mom', 'ldv', 'mom20', 'rs20', 'beta', 'vol20', 'vr'): FE[k + 'p'] = FE.groupby('day')[k].rank(pct=True)
B = B.merge(FE, on=['tk', 'day'], how='left')
O = pd.read_csv('/mnt/user-data/uploads/fartpie/xs/ext/optvol_ratio_live.csv', parse_dates=['date']); O = O[O.sh > 0]; O['r'] = O.opt_contracts * 100 / O.sh
RKm = O.pivot_table(index='date', columns='tk', values='r').sort_index().rolling(252, min_periods=60).median().shift(1); RKr = RKm.rank(axis=1, ascending=False, method='min'); RKp = RKm.rank(axis=1, pct=True)
B['brank'] = [RKr[t].asof(d) if t in RKr.columns else np.nan for t, d in zip(B.tk, B.day)]; B['ratiop'] = [RKp[t].asof(d) if t in RKp.columns else np.nan for t, d in zip(B.tk, B.day)]
B = B.dropna(subset=['momp', 'ldvp', 'trend', 'beta', 'vr', 'brank', 'mom20p', 'rs20p']).copy(); B['yr'] = B.entry.dt.year; B['tr'] = B.entry < '2023-01-01'
B['betamkt'] = B.beta * B.mkt; B['rest'] = B.ret - B.betamkt
SIG = ['ratiop', 'ldvp', 'momp', 'trend', 'h52b', 'mom20p', 'rs20p']; B.to_parquet(SP + 'b301.parquet')
print(f'trades {len(B)}  train {B.tr.sum()}  test {(~B.tr).sum()}')
# ---------- 1. what is a "bigger" flip made of
print('\n[1] return decomposition by number of aligned signals (mean per trade, %):')
B['ns'] = (B.ratiop >= 2 / 3).astype(int) + (B.ldvp >= 2 / 3) + (B.momp >= 2 / 3) + B.trend.astype(int) + B.h52b.astype(int) + (B.mom20p >= 2 / 3)
for lab, m in (('train', B.tr), ('test ', ~B.tr)):
    g = B[m].groupby(pd.cut(B[m].ns, [-1, 1, 2, 3, 4, 6]))
    print(lab, pd.DataFrame({'n': g.size(), 'ret%': g.ret.mean() * 100, 'beta*mkt%': g.betamkt.mean() * 100, 'rest%': g.rest.mean() * 100, 'mkt%': g.mkt.mean() * 100, 'beta': g.beta.mean(), 'hold_h': g.hold_h.median(), 'vol20%': g.vol20.mean() * 100, 'vr': g.vr.mean(), 'win%': g.ret.apply(lambda z: (z > 0).mean() * 100)}).round(2).to_string())
# ---------- 2. do signals collapse to one factor?
Z = B[SIG].astype(float); mu = Z[B.tr].mean(); sd = Z[B.tr].std(); Zs = (Z - mu) / sd
C = Zs[B.tr].corr(); print('\n[2] signal correlation (train):'); print(C.round(2).to_string())
u, s, vt = np.linalg.svd(Zs[B.tr].values, full_matrices=False); ev = s ** 2 / (s ** 2).sum(); print('\nPCA (train) explained:', np.round(ev, 2)); print('PC1 loadings:', dict(zip(SIG, np.round(vt[0] * np.sign(vt[0][1]), 2)))); print('PC2 loadings:', dict(zip(SIG, np.round(vt[1], 2))))
B['pc1'] = Zs.values @ (vt[0] * np.sign(vt[0][1])); B['pc2'] = Zs.values @ vt[1]
print('\nWhat PC1 is made of (train corr with):', {k: round(float(B.loc[B.tr, 'pc1'].corr(B.loc[B.tr, k])), 2) for k in ('beta', 'vol20', 'vr', 'ac1', 'betamkt')})
print('PC1 vs ticker (mean, top/bottom):'); mt = B.groupby('tk').pc1.mean().sort_values(); print(' low ', list(mt.index[:6]), '\n high', list(mt.index[-8:]))
# ---------- 3. within-bar fixed-effects: which of {PC1, beta, vr, vol} carries the return?
def fe(cols, y='ret', m=None):
    D = B if m is None else B[m]; D = D.dropna(subset=cols + [y]); g = D.groupby('entry'); n = g[y].transform('size'); D = D[n >= 2]
    X = D[cols].astype(float); X = (X - X.mean()) / X.std(); X = X - X.groupby(D.entry).transform('mean'); yy = D[y] - D.groupby('entry')[y].transform('mean')
    res = sm.OLS(yy.values * 100, X.values).fit(cov_type='cluster', cov_kwds={'groups': D.day.dt.strftime('%Y-%m-%d').values}); return {c: f'{b:+.3f}(t{t:+.1f})' for c, b, t in zip(cols, res.params, res.tvalues)}
print('\n[3] within-bar FE (coef = %ret per 1 sd; t clustered by day)')
for lab, m in (('train', B.tr), ('test ', ~B.tr)):
    print(lab, 'PC1 alone:', fe(['pc1'], m=m)); print(lab, 'PC1 + ratio:', fe(['pc1', 'ratiop'], m=m)); print(lab, 'beta, vr, vol20, ldv:', fe(['beta', 'vr', 'vol20', 'ldv'], m=m)); print(lab, 'PC1 + beta + vr + vol20:', fe(['pc1', 'beta', 'vr', 'vol20'], m=m))
    print(lab, 'rest (ret - beta*mkt) on PC1:', fe(['pc1'], y='rest', m=m))
B.to_parquet(SP + 'b301.parquet')
