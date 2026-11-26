"""Report 502 (Nov 26 2026): Rey - 'my strategy is breakout, not trend; find models that pair with breakouts'. Two UNTESTED breakout pairings on the sub-$5/<$10/<$20 daily vector flips (the flip = the breakout; tested and rejected already: pre-flip compression [expansion wins], 52w-high proximity, volume spike, candle patterns, efficiency-ratio consolidation, ORB).
DECLARED BEFORE RESULTS. Known at the signal close, cuts frozen on train 2019-21, test 2018 + 2022-26 (flips start 2019), controls ATRP tercile x log-mcap tercile, SE clustered by month, outcome HIT (PEAK>=+10%) and winsorised r.
OVH overhead supply (mechanism: holders who bought higher and are underwater sell at breakeven into a breakout; clear air = none): share of the last 252 trading days' volume (hlc3-weighted) traded at prices ABOVE the signal close and within +50% of it. Prediction: LOW OVH better (sign - on OVH_high flag).
BRD breakout breadth (mechanism: breakouts cluster in risk-on tapes, shared liquidity/flows): number of flips of the whole flip universe on the same signal day (share of days' maximum), top tercile vs rest. Prediction + (two-sided read).
LEAD = same sign train & test + pooled |t|>=2; REAL? = also test |t|>=2 at 2 of 3 price caps."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
SPd = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
src = open('xs/report_464_catalyst_flips.py').read(); exec(src[:src.index('def mp_(')])
W = pd.read_pickle(SPd + 'ext/wide_daily_ohlcv.pkl'); print(f'daily bars for {len(W)} tickers; flips with bars: {O.tk.isin(W).mean():.0%} (px<5: {O[O.px<5].tk.isin(W).mean():.0%})')
ovh = np.full(len(O), np.nan)
for tk, g in O.groupby('tk'):
    b = W.get(tk)
    if b is None or len(b) < 260: continue
    b = b[~b.index.duplicated()].sort_index(); tp = ((b.high + b.low + b.close) / 3).values; vo = b.volume.values.astype(float); cl = b.close.values; ix = b.index
    for i in g.index:
        j = ix.searchsorted(pd.Timestamp(O.at[i, 'd']), side='right') - 1
        if j < 252 or j >= len(ix): continue
        t = tp[j - 252:j + 1]; v = vo[j - 252:j + 1]; c = cl[j]; tot = v.sum()
        if tot > 0: ovh[i] = v[(t > c) & (t <= 1.5 * c)].sum() / tot
O['OVH'] = ovh; cnt = O.groupby('d').size(); O['BRD'] = O.d.map(cnt).values.astype(float)
O['hit'] = (O.PEAK >= .10).astype(float); lo, hi = O.r.quantile([.01, .99]); O['rw'] = O.r.clip(lo, hi); O['mo_'] = O.d.dt.to_period('M').astype(str)
def eff(D, flag, y):
    if D[flag].sum() < 10 or (~D[flag]).sum() < 10: return (np.nan, np.nan, int(D[flag].sum()))
    m = smf.ols(f'{y} ~ {flag}.astype(float) + C(a3) + C(c3)', D).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(D.mo_)[0]}); k = [q for q in m.params.index if q.startswith(flag)][0]; return (m.params[k], m.tvalues[k], int(D[flag].sum()))
for CAP in (5, 10, 20):
    S = O[(O.px < CAP) & O.ATRP.notna() & O.LMC.notna()].copy(); tr = S.tr
    S['a3'] = pd.cut(S.ATRP, [-np.inf, *np.quantile(S.ATRP[tr], [1/3, 2/3]), np.inf], labels=False); S['c3'] = pd.cut(S.LMC, [-np.inf, *np.quantile(S.LMC[tr], [1/3, 2/3]), np.inf], labels=False)
    S['OVH_HI'] = S.OVH >= np.nanquantile(S.OVH[tr], 2/3); S.loc[S.OVH.isna(), 'OVH_HI'] = False; S['OVH_LO'] = S.OVH <= np.nanquantile(S.OVH[tr], 1/3); S.loc[S.OVH.isna(), 'OVH_LO'] = False
    S['BRD_HI'] = S.BRD >= np.quantile(S.BRD[tr], 2/3); S['BRD_LO'] = S.BRD <= np.quantile(S.BRD[tr], 1/3)
    print(f"\n######## px<{CAP}: n {len(S)} (OVH known {S.OVH.notna().mean():.0%}) train {int(tr.sum())} test {int((~tr).sum())} | base reach {S[tr].hit.mean():.0%}/{S[~tr].hit.mean():.0%}")
    print(f"{'flag':8s}| n tr/te | raw test reach on vs off | ADJ reach train test all (t) | ADJ mean r train test all (t) | verdict")
    for f in ('OVH_HI', 'OVH_LO', 'BRD_HI', 'BRD_LO'):
        te = S[~S.tr]; a = [eff(D, f, 'hit') for D in (S[S.tr], te, S)]; b = [eff(D, f, 'rw') for D in (S[S.tr], te, S)]
        same = lambda z: (np.sign(z[0][0]) == np.sign(z[1][0])) and abs(z[2][1]) >= 2; v = 'LEAD' if (same(a) or same(b)) else '-'
        print(f"{f:8s}| {a[0][2]:3d}/{a[1][2]:3d} | {te[te[f]].hit.mean():.0%} vs {te[~te[f]].hit.mean():.0%} | {a[0][0]:+.2f} {a[1][0]:+.2f} {a[2][0]:+.2f} (t {a[2][1]:+.1f}) | {b[0][0]:+.3f} {b[1][0]:+.3f} {b[2][0]:+.3f} (t {b[2][1]:+.1f}) | {v}")
print('\n######## decisive: inside the high-ATRP slice (top 30% ATRP, the live regime) with CONTINUOUS ATRP + LMC + LTURN controls')
for CAP in (5, 10, 20):
    S = O[(O.px < CAP) & O.ATRP.notna() & O.LMC.notna() & O.LTURN.notna()].copy(); tr = S.tr; S['a3'] = 0; S['c3'] = 0
    S['OVH_LO'] = S.OVH <= np.nanquantile(S.OVH[tr], 1/3); S['OVH_HI'] = S.OVH >= np.nanquantile(S.OVH[tr], 2/3); S['BRD_HI'] = S.BRD >= np.quantile(S.BRD[tr], 2/3); S['BRD_LO'] = S.BRD <= np.quantile(S.BRD[tr], 1/3)
    G = S[S.ATRP >= np.quantile(S[S.tr].ATRP, .7)].copy()
    for f in ('OVH_LO', 'OVH_HI', 'BRD_HI', 'BRD_LO'):
        out = []
        for lab, D in (('ALL', G), ('train', G[G.tr]), ('test', G[~G.tr])):
            m = smf.ols(f'hit ~ {f}.astype(float) + ATRP + LMC + LTURN', D).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(D.mo_)[0]}); out.append(f"{lab} {m.params[f'{f}.astype(float)']:+.3f} (t {m.tvalues[f'{f}.astype(float)']:+.1f}, n flag {int(D[f].sum())})")
        print(f"  px<{CAP} {f:7s}: " + ' | '.join(out))
