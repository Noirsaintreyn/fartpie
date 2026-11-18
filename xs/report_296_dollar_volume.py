"""Nov 18 2026 report 296 (Rey: research dollar volume). The screener's ratio edge shrinks to dollar volume (corr 0.62; ldv carried the 2023-26 return +0.21%/sd, t 2.7 in report 294). DECLARED BEFORE RESULTS:
Features (prior close, per name): LDV = log 50d mean $ volume, converted to a CROSS-SECTIONAL percentile among the 54 names that day (price-level free); SHOCK = prior-day $vol / 50d mean; TREND = 50d $vol / 252d-median of it (attention rising).
H1 per-trade return by LDV third (train 2019-22 | test 2023-26, and by year).  H2 6-slot pick order: LDV high-first / low-first / ratio / ratio+LDV rank-sum.  H3 UNIVERSE: replace the ratio >= 1.5 filter by 'top-share of names by LDV' with the same
trade count, still ordered by ratio.  H4 mechanism: within-bar regression of ret on LDV pct and ratio pct together.  H5 is it just megacaps? returns for the 10 highest-LDV names vs the rest."""
import numpy as np, pandas as pd, statsmodels.api as sm, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
_s = open('/home/claude/fartpie/xs/scr_slots.py').read(); exec(_s[:_s.index("if __name__ == '__main__':")])
A = pd.read_parquet(SP + 'flips295.parquet'); W = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet'); W = W[W.sp & ~W.h1 & (W['var'] == 'base+sticky')][['tk', 'entry', 'ratio']]
B0 = A.merge(W, on=['tk', 'entry']); B0 = B0[(B0.entry >= '2019-01-01') & (B0.exit < END)].copy(); B0['day'] = B0.entry.dt.normalize()
DVD = {}
for tk in sorted(A.tk.unique()):
    try: Y = pd.read_parquet(f'data/yahoo/daily_v/{tk}.parquet')
    except Exception: continue
    Y.index = pd.DatetimeIndex(Y.index).tz_localize(None).normalize(); dv = Y.close * Y.volume; DVD[tk] = pd.DataFrame({'ldv': np.log(dv.rolling(50).mean()), 'shock': dv / dv.rolling(50).mean(), 'trend': dv.rolling(50).mean() / dv.rolling(50).mean().rolling(252, min_periods=120).median()}).shift(1)
L = pd.concat(DVD, names=['tk', 'day']).reset_index(); L['ldvp'] = L.groupby('day').ldv.rank(pct=True)
B0 = B0.merge(L, on=['tk', 'day'], how='left').dropna(subset=['ldvp']); B0['per'] = np.where(B0.entry < '2023-01-01', 'train', 'test'); B = B0[B0.ratio >= 1.5].copy(); B['h1'] = False
print('trades (ratio>=1.5)', len(B), '| all-ratio trades', len(B0), '| corr(ratio, ldvp)', round(B.ratio.corr(B.ldvp, method='spearman'), 2))
print('\nH1 mean return per trade by LDV percentile third (cuts fixed 1/3, 2/3 of the cross-section):')
for f in ('ldvp', 'shock', 'trend'):
    cut = B[B.per == 'train'][f].quantile([1 / 3, 2 / 3]).values; B['_t'] = np.where(B[f] <= cut[0], 'lo', np.where(B[f] <= cut[1], 'mid', 'hi')); r = B.groupby(['per', '_t']).ret.agg(['mean', 'size'])
    print(f'  {f:6s} ' + ' | '.join(f'{p}: ' + ' '.join(f"{t} {r.loc[(p, t), 'mean']:+.2%}(n{int(r.loc[(p, t), 'size'])})" for t in ('lo', 'mid', 'hi')) for p in ('train', 'test')))
B['_t'] = pd.qcut(B.ldvp, 3, labels=['lo', 'mid', 'hi']); print('  LDV third by year (mean ret, n):'); print((B.groupby([B.entry.dt.year, '_t']).ret.mean().unstack() * 100).round(2).to_string())
print('\nH5 megacap check: 10 highest average-LDV names vs others, mean ret train | test'); top = B.groupby('tk').ldvp.mean().nlargest(10).index; B['mega'] = B.tk.isin(top)
print(B.groupby(['per', 'mega']).ret.agg(['mean', 'size']).round(4).to_string()); print('  top-10 names:', list(top))
def st2(s):
    out = []
    for a, b in (('2019-01-01', '2023-01-01'), ('2023-01-01', '2027-01-01')):
        z = s[(s.index >= a) & (s.index < b)]; e = (1 + z).cumprod(); y = (z.index[-1] - z.index[0]).days / 365.25; out.append(f'{e.iloc[-1] ** (1 / y) - 1:6.1%} / DD {(e / e.cummax() - 1).min():6.1%}')
    return ' | '.join(out)
print('\nH2 6-slot pick order (CAGR/DD 2019-22 | 2023-26):'); g = B.groupby('entry')
for lab, key in (('ratio (live)', -B.ratio), ('LDV high first', -B.ldvp), ('LDV low first', B.ldvp), ('ratio + LDV ranks', g.ratio.rank(ascending=False) + g.ldvp.rank(ascending=False)), ('SHOCK high first', -B.shock), ('TREND high first', -B.trend)):
    T0 = B.assign(_k=np.asarray(key)).sort_values(['entry', '_k', 'tk'], kind='stable'); tk_, _ = taken(); s_, _ = daily(tk_, 6); print(f'  {lab:20s} {st2(s_)}', flush=True)
print('\nH3 universe by LDV instead of ratio (ordered by ratio; same trade count in each period):')
for lab, flt in (('ratio>=1.5 (live)', None), ('top-LDV share', 'ldv')):
    if flt is None: U = B
    else:
        n_ = len(B); U = B0.sort_values('ldvp', ascending=False).head(n_).copy(); U['h1'] = False
    T0 = U.sort_values(['entry', 'ratio', 'tk'], ascending=[True, False, True], kind='stable'); tk_, _ = taken(); s_, _ = daily(tk_, 6); print(f'  {lab:20s} {st2(s_)} | trades {len(U)} min ldvp {U.ldvp.min():.2f}', flush=True)
print('\nH4 within-bar FE regression (bars with >=3 flips), per sd, cluster t:'); X = B.copy(); X['n'] = X.groupby('entry').tk.transform('size'); X = X[X.n >= 3].copy(); X['rp'] = X.ratio.rank(pct=True)
for lab, a, b in (('train', '2019-01-01', '2023-01-01'), ('test', '2023-01-01', '2027-01-01')):
    S = X[(X.entry >= a) & (X.entry < b)]
    for nm, xs in (('LDV alone', ['ldvp']), ('ratio alone', ['rp']), ('both', ['ldvp', 'rp'])):
        Z = S[['ret'] + xs] - S.groupby('entry')[['ret'] + xs].transform('mean'); Z[xs] = Z[xs] / Z[xs].std(); m = sm.OLS(Z.ret.values, Z[xs].values).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(S.day)[0]})
        print(f'  {lab:5s} {nm:11s} n={len(S):5d} ' + ' '.join(f'{k}:{v * 100:+.2f}[{t:+.1f}]' for k, v, t in zip(xs, m.params, m.tvalues)))
