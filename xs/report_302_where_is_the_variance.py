"""Nov 18 2026 report 302. After 301: the seven selection signals do NOT collapse to one factor (ratio+dollar volume cluster, trend/52w-high/12-1 cluster, 20d mom separate; PC1 return t = -0.7 train, +1.8 test).
So: where does a flip's size actually come from?  DECLARED BEFORE RUNNING (direction + why it works + why it would stop):
 A. Variance budget: R2 of flip return on entry-date effects (market timing) vs name effects vs within-bar spread (the only thing a RANKER can use).
 B. Tail: a flip system wins with ~31% hit rate; mean is made by the right tail. Does anything change hit rate, or only tail size?
 C. Market-state gates (point-in-time, shift 1). Theory: a trailing-stop trend entry earns when the INDEX is persistent (autocorrelated multi-day moves) and broad participation exists; it bleeds in choppy/mean-reverting tape.
    C1 idx_vr    SPY 5d/1d variance ratio over 120d  (+: persistence; fails when a vol shock flips the sign of persistence suddenly)
    C2 breadth   share of the 54 names above their 200d    (+: broad participation; fails in narrow megacap tapes where only few names trend -> then SIZE matters instead)
    C3 corr60    mean pairwise 60d correlation of the 54    (+ means single-factor tape: flips ride the market leg; - means rotation: flips whipsaw)
    C4 spy_trend SPY > 200d                                  (+; fails at V-bottoms)
 Reported by tercile, train 2019-22 | test 2023-26, t on month-clustered means."""
import numpy as np, pandas as pd, warnings, glob; warnings.filterwarnings('ignore')
import statsmodels.api as sm, statsmodels.formula.api as smf
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
B = pd.read_parquet(SP + 'b301.parquet'); B['mo'] = B.entry.dt.to_period('M').astype(str); B['wk'] = B.entry.dt.to_period('W').astype(str)
# ---------------- A. variance budget
print('[A] share of flip-return variance explained (adjusted R2, %), in-sample by block:')
for lab, m in (('train', B.tr), ('test ', ~B.tr)):
    D = B[m]; out = {}
    for nm, fm in (('week', 'ret ~ C(wk)'), ('month', 'ret ~ C(mo)'), ('name', 'ret ~ C(tk)'), ('name x year', 'ret ~ C(tk):C(yr)'), ('entry bar', 'ret ~ C(entry)')):
        out[nm] = round(100 * smf.ols(fm, D).fit().rsquared_adj, 1)
    print(lab, out, '| bars with >=2 candidates: %.0f%% of trades' % (100 * (D.groupby('entry').ret.transform('size') >= 2).mean()))
# ---------------- B. tail
print('\n[B] hit rate and tail by number of aligned signals (test | train):')
for lab, m in (('train', B.tr), ('test ', ~B.tr)):
    D = B[m]; q = D.ret.quantile(.95); g = D.groupby(pd.cut(D.ns, [-1, 1, 2, 3, 4, 6]))
    print(lab, pd.DataFrame({'win%': g.ret.apply(lambda z: (z > 0).mean() * 100), 'avgwin%': g.ret.apply(lambda z: z[z > 0].mean() * 100), 'avgloss%': g.ret.apply(lambda z: z[z <= 0].mean() * 100), 'p(>+5%)': g.ret.apply(lambda z: (z > .05).mean() * 100), 'top5%share': g.ret.apply(lambda z: z[z >= q].sum() / z.sum() * 100)}).round(2).to_string())
# ---------------- C. market state
tks = sorted(B.tk.unique()); C = {}
for tk in tks: Y = pd.read_parquet(f'data/yahoo/daily_v/{tk}.parquet'); Y.index = pd.DatetimeIndex(Y.index).tz_localize(None).normalize(); C[tk] = Y.close
C = pd.DataFrame(C).sort_index(); R = np.log(C).diff()
Sd = pd.read_parquet('data/yahoo/daily_v/SPY.parquet'); Sd.index = pd.DatetimeIndex(Sd.index).tz_localize(None).normalize(); rs = np.log(Sd.close).diff()
idx_vr = np.log(Sd.close / Sd.close.shift(5)).rolling(120).var() / (5 * rs.rolling(120).var())
breadth = (C > C.rolling(200, min_periods=150).mean()).where(C.rolling(200, min_periods=150).mean().notna()).mean(axis=1)
corr60 = R.rolling(60).apply(lambda z: 0, raw=True) if False else None
cm = (R.rolling(60, min_periods=40).mean()); ew = R.mean(axis=1); var_ew = ew.rolling(60).var(); var_i = R.rolling(60).var().mean(axis=1)
corr60 = (var_ew - var_i / R.shape[1]) / (var_i * (1 - 1 / R.shape[1]))  # mean pairwise corr proxy via equal-weight variance identity
M = pd.DataFrame({'idx_vr': idx_vr, 'breadth': breadth, 'corr60': corr60, 'spy_trend': (Sd.close > Sd.close.rolling(200).mean()).astype(float)}).shift(1); M['day'] = M.index
B = B.merge(M, on='day', how='left').dropna(subset=['idx_vr', 'breadth', 'corr60'])
print('\n[C] flip return by market-state tercile (cut on 2019-22), mean % per trade; month-clustered t of top minus bottom tercile')
def cl(D, col, cut):
    d = D.assign(g=np.where(D[col] <= cut[0], 'lo', np.where(D[col] >= cut[1], 'hi', 'mid'))); r = {k: round(d[d.g == k].ret.mean() * 100, 2) for k in ('lo', 'mid', 'hi')}
    x = d[d.g.isin(['lo', 'hi'])].assign(h=lambda z: (z.g == 'hi').astype(float)); res = smf.ols('ret ~ h', x).fit(cov_type='cluster', cov_kwds={'groups': x.mo.values}); n = d.g.value_counts().to_dict()
    return r, round(res.tvalues['h'], 1), (n.get('lo', 0), n.get('mid', 0), n.get('hi', 0))
for col in ('idx_vr', 'breadth', 'corr60', 'spy_trend'):
    if col == 'spy_trend': cut = (.5, .5)
    else: cut = tuple(B.loc[B.tr, col].quantile([1 / 3, 2 / 3]))
    a = cl(B[B.tr], col, cut); b = cl(B[~B.tr], col, cut); print(f'  {col:9s} cut {np.round(cut, 2)}  train {a[0]} t{a[1]:+.1f} n{a[2]} | test {b[0]} t{b[1]:+.1f} n{b[2]}')
B.to_parquet(SP + 'b302.parquet')
# ---------------- D. does the selection edge depend on the state?  ns>=3 minus ns<=2 inside each breadth/corr tercile
print('\n[D] selection edge (ns>=3 minus ns<=2, mean % per trade) inside each market-state bucket:')
for col in ('breadth', 'corr60', 'idx_vr'):
    cut = tuple(B.loc[B.tr, col].quantile([1 / 3, 2 / 3]))
    for lab, m in (('train', B.tr), ('test ', ~B.tr)):
        D = B[m]; r = []
        for k, sel in (('lo', D[col] <= cut[0]), ('mid', (D[col] > cut[0]) & (D[col] < cut[1])), ('hi', D[col] >= cut[1])):
            d = D[sel]; r.append(f'{k} {(d[d.ns >= 3].ret.mean() - d[d.ns <= 2].ret.mean()) * 100:+.2f} (n{len(d)})')
        print(f'  {col:8s} {lab} ' + ' | '.join(r))
