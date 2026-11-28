"""Report 512 (Nov 27 2026): INSTITUTIONAL reasons for the basket, from the literature (Asness-Frazzini-Israel-Moskowitz-Pedersen 'Size matters if you control your junk': the small-cap premium needs a quality control, profitability the strongest piece; Pontiff-Woodgate / Dimensional: net share issuers underperform, repurchasers outperform; Drechsler: anomaly returns sit in the SHORT leg - issuers, distress, max-return / high idio vol stocks - not in the long leg).
DECLARED BEFORE RESULTS (point-in-time, latest 10-K filed BEFORE the flip, <=540d old). PROFIT: operating income > 0 (EBIT/assets as the profitability piece). NOISSUE: net share issuance <= 2% (ln of annual weighted shares vs prior FY; no dilution). BUYBACK: cash paid to repurchase stock > 0 in the latest FY. INST = count of the three. Prediction: INST>=2 raises MEAN and MEDIAN r (junk/short-leg names are what drags the lottery payoff), sign +; reach is not the target. Then stacked with the report-511 pressure basket. Same top-vol third rule, train 2019-21 / test 2022+, month-clustered, controls log ATRP."""
import numpy as np, pandas as pd, json, os, warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
s511 = open('xs/report_511_pressure_basket.py').read(); exec(s511[s511.index('import numpy as np'):s511.index('res = []')])
def ent(us, tags, instant=False):
    for t in tags:
        x = us.get(t)
        if x and 'USD' in x.get('units', {}) or (x and 'shares' in x.get('units', {})):
            r = pd.DataFrame(x['units'].get('USD') or x['units']['shares']); r = r[(r.form == '10-K') & (r.fp == 'FY')]
            if not instant and 'start' in r: r = r[(pd.to_datetime(r.end) - pd.to_datetime(r.start)).dt.days.between(300, 380)]
            if len(r): r['filed'] = pd.to_datetime(r.filed); r['end'] = pd.to_datetime(r.end); return r.sort_values(['end', 'filed']).drop_duplicates('end', keep='first').set_index('end')[['val', 'filed']]
    return None
rows = []
for tk in sorted(set(O.tk)):
    f = f'data/fund/sec_small/{tk}.json'
    if not os.path.exists(f): continue
    us = json.load(open(f))['facts'].get('us-gaap', {})
    ebit = ent(us, ['OperatingIncomeLoss']); sh = ent(us, ['WeightedAverageNumberOfSharesOutstandingBasic', 'WeightedAverageNumberOfDilutedSharesOutstanding']); bb = ent(us, ['PaymentsForRepurchaseOfCommonStock'])
    if ebit is None: continue
    X = ebit.rename(columns={'val': 'ebit'}); X['PROFIT'] = (X.ebit > 0).astype(float)
    if sh is not None:
        g = np.log(sh.val / sh.val.shift(1).where((sh.index.to_series().diff().dt.days.between(300, 430)))); X['NSI'] = g.reindex(X.index)
    if bb is not None: X['BB'] = (bb.val.reindex(X.index) > 0).astype(float).where(bb.val.reindex(X.index).notna())
    else: X['BB'] = 0.0
    X['tk'] = tk; rows.append(X.reset_index(drop=True))
F = pd.concat(rows); F['filed'] = F.filed.astype('datetime64[ns]'); F = F.sort_values('filed')
O2 = pd.merge_asof(O.sort_values('d'), F[['tk', 'filed', 'PROFIT', 'NSI', 'BB']], left_on='d', right_on='tk' if False else 'filed', by='tk', direction='backward', allow_exact_matches=False, tolerance=pd.Timedelta(days=540))
O2['NOISSUE'] = np.where(O2.NSI.isna(), np.nan, (O2.NSI <= .02).astype(float)); O2['BUYBACK'] = O2.BB
O2 = O2.sort_values(['tk', 'd']).reset_index(drop=True); print('flips', len(O2), 'with PROFIT', O2.PROFIT.notna().mean().round(2), 'NSI', O2.NOISSUE.notna().mean().round(2), 'BB', O2.BUYBACK.notna().mean().round(2),
      '| rates PROFIT', O2.PROFIT.mean().round(2), 'NOISSUE', O2.NOISSUE.mean().round(2), 'BUYBACK', O2.BUYBACK.mean().round(2))
res = []
for cap, name in ((5, 'px<5'), (10, 'px<10'), (20, 'px<20')):
    D = O2[O2.px < cap].copy(); tr = D.tr; D['lA'] = np.log(D.ATRP); D['OVHL'] = np.where(D.OVH.isna(), np.nan, (D.OVH <= D.OVH[tr].quantile(1 / 3)).astype(float))
    FL = ['DEALf', 'INSf', 'SECMAC', 'SQ', 'RAISE90', 'SI15', 'OVHL']; D['nb'] = D[FL].fillna(0).eq(1).sum(axis=1)
    D['INST'] = D[['PROFIT', 'NOISSUE', 'BUYBACK']].fillna(0).sum(axis=1); D['known'] = D[['PROFIT', 'NOISSUE']].notna().all(axis=1)
    cut = D.ATRP[tr].quantile(2 / 3); D['topvol'] = D.ATRP >= cut; T0 = D[D.topvol & D.known]
    print(f'\n=== {name}: top-vol flips with data {len(T0)} (train {int(T0.tr.sum())}) | INST dist ' + str(T0.INST.value_counts().sort_index().to_dict()))
    for era, E in (('train', T0[T0.tr]), ('test', T0[~T0.tr])):
        print(f'  {era:5}: ' + ' | '.join(f"{k} n{len(v)} reach {100 * v.hit.mean():.0f}% mean {100 * v.rw.mean():+.1f}% med {100 * v.rw.median():+.1f}%" for k, v in {'all': E, 'INST>=2': E[E.INST >= 2], 'INST<=1': E[E.INST <= 1], 'INST=3': E[E.INST == 3], 'INST=0': E[E.INST == 0], 'INST>=2 & basket>=2': E[(E.INST >= 2) & (E.nb >= 2)]}.items()))
        for lab, f in (('PROFIT', E.PROFIT == 1), ('NOISSUE', E.NOISSUE == 1), ('BUYBACK', E.BUYBACK == 1), ('INST>=2', E.INST >= 2), ('INST>=2&bkt>=2', (E.INST >= 2) & (E.nb >= 2))):
            X = E.assign(b=f.astype(float))
            if X.b.sum() < 10 or (X.b == 0).sum() < 10: continue
            for y in ('hit', 'rw'):
                m = smf.ols(f'{y} ~ b + lA', X).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(X.mo_)[0]}); res.append((name, era, lab, y, m.params.b * 100, m.tvalues.b, int(X.b.sum())))
R = pd.DataFrame(res, columns=['bucket', 'era', 'flag', 'y', 'adj_pp', 't', 'n']); print('\nadjusted effect inside the top-vol third (pp, month-clustered t):')
print(R.pivot_table(index=['bucket', 'flag', 'y'], columns='era', values=['adj_pp', 't', 'n'], aggfunc='first').round(2).to_string())
