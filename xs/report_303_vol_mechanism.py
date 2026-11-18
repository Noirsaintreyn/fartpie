"""Nov 18 2026 report 303. MECHANISM TEST. From 301/302: flip hit-rate is ~31% for every signal count; mean return per trade is FLAT across name-volatility quintiles (+0.2..+0.4%) while its std doubles (2.6% -> 5.1%).
THEORY (declared before running): a flip is a convex trend bet: ~31% hit, win:loss ~3:1, with a stop proportional to the name's own noise. Edge per trade in % is roughly the same in every name (what pays is the persistence of the move after the flip), but RISK scales with sigma.
 => (1) the boost signals (trend, 52w-high, size, ratio) mostly choose LOWER-sigma names, which is why drawdown fell (-16.2% -> -10.9%) and CAGR barely moved.
 => (2) the right expression is a RISK allocation (size ~ 1/sigma), not a ranking.
 Why it should work: lottery-preferring buyers overpay for high-vol names (low-vol anomaly), so their risk is not rewarded; stop-based exits scale with sigma while the post-flip drift does not.
 Why it would stop: when speculative/high-beta names LEAD (2020-21 style melt-ups, early 2023 rebounds) the same flip in a high-vol name pays far more in % -- a low-vol tilt then lags; also low-vol names can be crowded (rates shock).
TESTS: (a) which signals change sigma vs change return-per-sigma (within-bar FE); (b) ordering by low vol alone vs the 300 boost; (c) inverse-vol sizing on the live order; (d) boost + inverse-vol. CAGR/DD/ann vol 2019-22 | 2023-26."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm
_s = open('/home/claude/fartpie/xs/scr_slots.py').read(); exec(_s[:_s.index("if __name__ == '__main__':")])
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
B = pd.read_parquet(SP + 'b302.parquet'); B = B[B.exit < END].copy(); B['h1'] = False
B['R'] = B.ret / B.vol20; B['MOM'] = B.momp >= 2 / 3; B['SIZE'] = B.ldvp >= 2 / 3; B['TREND'] = B.trend.astype(bool); B['HI52'] = B.h52b.astype(bool)
# ---- (a) per-signal: effect on sigma vs on return/sigma (within-bar FE, day-clustered)
def fe(col, y, m):
    D = B[m].dropna(subset=[col, y]); D = D[D.groupby('entry')[y].transform('size') >= 2]; X = D[[col]].astype(float); X = (X - X.mean()) / X.std(); X = X - X.groupby(D.entry).transform('mean'); yy = D[y] - D.groupby('entry')[y].transform('mean')
    r = sm.OLS(yy.values, X.values).fit(cov_type='cluster', cov_kwds={'groups': D.day.dt.strftime('%Y-%m-%d').values}); return f'{r.params[0]:+.4f}(t{r.tvalues[0]:+.1f})'
print('(a) within-bar FE, coef per +1sd of signal; columns: vol20 (sigma) | ret (%) | R = ret/sigma | hold hours')
for s in ('ratiop', 'ldvp', 'momp', 'trend', 'h52b', 'mom20p', 'beta', 'vr'):
    for lab, m in (('train', B.tr), ('test ', ~B.tr)): print(f'  {s:7s} {lab} sigma {fe(s, "vol20", m):17s} ret {fe(s, "ret", m):17s} R {fe(s, "R", m):17s}')
# ---- slot sim helpers
def sim(order_key, wfun=None, N=6):
    global T0
    X = B.assign(_k=order_key).sort_values(['entry', '_k', 'tk'], kind='stable'); T0 = X; tk_, _ = taken(N); X2 = X.set_index(['tk', 'entry'])
    R = np.zeros(len(D)); Wt = np.zeros(len(D))
    for tk, e, x, r, ra in tk_:
        a, b = D.searchsorted(e.normalize()), D.searchsorted(x.normalize())
        if a >= len(D): continue
        b = min(b, len(D) - 1); n = b - a + 1; w = (1 / N) * (wfun(X2.loc[(tk, e)]) if wfun else 1.0)
        R[a:b + 1] += w * ((1 + r) * (1 - 2e-4)) ** (1 / n) - w; Wt[a:b + 1] += w
    sc = np.where(Wt > 1, 1 / Wt, 1.0); R = R * sc; return pd.Series(R, index=D), np.minimum(Wt, 1.0)
def st2(s):
    out = []
    for a, b in (('2019-01-01', '2023-01-01'), ('2023-01-01', '2027-01-01')):
        z = s[(s.index >= a) & (s.index < b)]; e = (1 + z).cumprod(); y = (z.index[-1] - z.index[0]).days / 365.25; out.append(f'{e.iloc[-1] ** (1 / y) - 1:6.1%} / DD {(e / e.cummax() - 1).min():6.1%} / vol {z.std() * np.sqrt(252):4.1%}')
    return ' | '.join(out)
brank = B.brank.values + 1e-6 * B.ratio.rank(ascending=False).values; allS = ['TREND', 'MOM', 'HI52', 'SIZE']; ns = B[allS].sum(axis=1).values
volr = B.vol20.rank(pct=True).values; MEDV = B.loc[B.tr, 'vol20'].median()
inv = lambda r: float(np.clip(MEDV / r.vol20, .5, 1.5))
print('\n(b)-(d) slot book, CAGR / DD / ann.vol   [2019-22  |  2023-26]')
print(' live order (ratio), equal size   ', st2(sim(brank)[0]), flush=True)
print(' 300 boost K=2                   ', st2(sim(brank - 2 * ns)[0]), flush=True)
for K in (5, 10, 20): print(f' ratio-rank minus {K:2d}*(low-vol pct)   ', st2(sim(brank - K * (1 - volr))[0]), flush=True)
print(' order by lowest vol only         ', st2(sim(volr)[0]), flush=True)
print(' live order + inverse-vol size    ', st2(sim(brank, inv)[0]), flush=True)
print(' 300 boost + inverse-vol size     ', st2(sim(brank - 2 * ns, inv)[0]), flush=True)
r = sim(brank - 2 * ns, inv); r0 = sim(brank)[0]; print(' avg invested, live vs boost+invvol:', round(sim(brank)[1][D >= '2019-01-01'].mean(), 2), round(r[1][D >= '2019-01-01'].mean(), 2))
