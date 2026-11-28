"""Report 511 (Nov 27 2026): Rey - 'throw a few things in the basket that should create buying or selling pressure, macro and fundamental, then screen only the ones that flip the vector and have the highest vol first'.
DECLARED BEFORE RESULTS. Basket = a stock-day with >=1 pressure reason known before the flip (each has a pressure mechanism): DEAL (8-K 1.01/2.01 in 7d: deal flow), INS (insider open-market buy 30d), SECMAC (sector macro driver favourable), SQ (cash-burning, runway<12m: financing pending), RAISE90 (8-K 3.02 in 90d: fresh supply/funding), SI15 (short >=15% of shares: cover/squeeze pressure; FINRA settle+9d), OVH_LOW (overhead supply in the bottom train third: clear air).
Rule: flips only, volatility first = ATRP top third (train cuts per price bucket). Compare in the top-vol third: all flips vs basket>=1 vs basket>=2 vs not-in-basket. Prediction: basket top-vol beats all top-vol on REACH and mean r (sign +). Pass = same sign train and test, test |t|>=2 month-clustered, controls log ATRP.
Unit: daily bull vector flips px<5 / <10 / <20 (report 464 set, 2019-26Q1, 8,408 flips), train 2019-21 / test 2022+."""
import numpy as np, pandas as pd, json, os, warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
s = open('xs/report_469_reason_to_move.py').read(); exec(s[s.index('import numpy as np'):s.index('# ---- base score ----')])
SPd = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
# SI15
S = pd.read_pickle(SPd + 'ext/shortint.pkl')[['symbolCode', 'settlementDate', 'currentShortPositionQuantity']]; S['use'] = pd.to_datetime(S.settlementDate).astype('datetime64[ns]') + pd.Timedelta(days=9); S = S.sort_values('use')
O['shares'] = np.exp(O.MCAP) / O.px; O['d'] = O.d.astype('datetime64[ns]'); M = pd.merge_asof(O.sort_values('d')[['tk', 'd', 'shares']], S.rename(columns={'symbolCode': 'tk'}), left_on='d', right_on='use', by='tk', direction='backward', tolerance=pd.Timedelta(days=45))
O = O.merge(M[['tk', 'd', 'currentShortPositionQuantity']].drop_duplicates(['tk', 'd']), on=['tk', 'd'], how='left'); O['SI15'] = np.where(O.currentShortPositionQuantity.isna(), np.nan, (O.currentShortPositionQuantity / O.shares >= .15).astype(float))
# OVH
Wd = pd.read_pickle(SPd + 'ext/wide_daily_ohlcv.pkl'); ovh = np.full(len(O), np.nan)
for tk, g in O.groupby('tk'):
    b = Wd.get(tk)
    if b is None or len(b) < 260: continue
    b = b[~b.index.duplicated()].sort_index(); tp = ((b.high + b.low + b.close) / 3).values; vo = b.volume.values.astype(float); cl = b.close.values; ix = b.index
    for i in g.index:
        j = ix.searchsorted(pd.Timestamp(O.at[i, 'd']), side='right') - 1
        if j < 252 or j >= len(ix): continue
        t = tp[j - 252:j + 1]; v = vo[j - 252:j + 1]; c = cl[j]; tot = v.sum()
        if tot > 0: ovh[i] = v[(t > c) & (t <= 1.5 * c)].sum() / tot
O['OVH'] = ovh; O['hit'] = (O.PEAK >= .10).astype(float); lo, hi = O.r.quantile([.01, .99]); O['rw'] = O.r.clip(lo, hi); O['mo_'] = O.d.dt.to_period('M').astype(str)
res = []
for cap, name in ((5, 'px<5'), (10, 'px<10'), (20, 'px<20')):
    D = O[O.px < cap].copy(); tr = D.tr
    D['OVHL'] = np.where(D.OVH.isna(), np.nan, (D.OVH <= D.OVH[tr].quantile(1 / 3)).astype(float)); D['lA'] = np.log(D.ATRP)
    FL = ['DEALf', 'INSf', 'SECMAC', 'SQ', 'RAISE90', 'SI15', 'OVHL']; D['nb'] = D[FL].fillna(0).eq(1).sum(axis=1)
    cut = D.ATRP[tr].quantile(2 / 3); D['topvol'] = D.ATRP >= cut
    print(f'\n=== {name}: {len(D)} flips | top-vol third ATRP >= {cut:.3f} | flag rates ' + ' '.join(f'{f} {D[f].mean():.2f}' for f in FL))
    for era, E in (('train 19-21', D[D.tr]), ('test 22-26', D[~D.tr])):
        T = E[E.topvol]
        rows = {'ALL top-vol': T, 'basket>=1': T[T.nb >= 1], 'basket>=2': T[T.nb >= 2], 'not in basket': T[T.nb == 0]}
        print(f'  {era}: ' + ' | '.join(f"{k} n{len(v)} reach {100 * v.hit.mean():.0f}% mean r {100 * v.rw.mean():+.1f}% med {100 * v.rw.median():+.1f}%" for k, v in rows.items()))
        for lab, f in (('basket>=1', T.nb >= 1), ('basket>=2', T.nb >= 2)):
            X = T.assign(b=f.astype(float))
            if X.b.sum() < 10 or (X.b == 0).sum() < 10: continue
            for y in ('hit', 'rw'):
                m = smf.ols(f'{y} ~ b + lA', X).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(X.mo_)[0]}); res.append((name, era, lab, y, m.params.b * 100, m.tvalues.b, int(X.b.sum())))
R = pd.DataFrame(res, columns=['bucket', 'era', 'basket', 'y', 'adj_pp', 't', 'n_basket']); print('\nadjusted basket effect inside the top-vol third (pp, month-clustered t):'); print(R.round(2).to_string(index=False))
