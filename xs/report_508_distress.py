"""Report 508 (Nov 27 2026): distress score vs reach on sub-$5 flips. Hypothesis declared before looking (Moyer, Distressed Debt Analysis ch. 8; Merton): equity of a financially distressed firm is a call on the assets, so at the SAME volatility a distressed name has a fatter upside tail, i.e. higher reach (hit = PEAK >= +10%); the price is a fatter bankruptcy tail (worse median r).
Score = Altman Z'' (non-manufacturer, no market value): 6.56 WC/TA + 3.26 RE/TA + 6.72 EBIT/TA + 1.05 BookEquity/Liabilities, from the latest 10-K filed BEFORE the flip (<=540d). Low = distressed. Control: ATRP tercile (train cut points). Train 2019-21 / test 2022+, month-clustered SE."""
import json, os, pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
H = pd.read_csv('flip_history.csv', parse_dates=['d'])
def ann(g, tags, instant):
    for t in tags:
        x = g.get(t)
        if x and 'USD' in x.get('units', {}):
            r = pd.DataFrame(x['units']['USD']); r = r[(r.form == '10-K') & (r.fp == 'FY')]
            if not instant and 'start' in r: r = r[(pd.to_datetime(r.end) - pd.to_datetime(r.start)).dt.days > 300]
            if len(r): r['filed'] = pd.to_datetime(r.filed); return r.sort_values('filed').drop_duplicates('end', keep='last').set_index('end').val.rename(t)
    return None
rows = []
for tk in sorted(set(H.tk)):
    try: g = json.load(open(f'data/fund/sec_small/{tk}.json'))['facts']['us-gaap']
    except Exception: continue
    S = {'TA': ann(g, ['Assets'], 1), 'TL': ann(g, ['Liabilities'], 1), 'CA': ann(g, ['AssetsCurrent'], 1), 'CL': ann(g, ['LiabilitiesCurrent'], 1), 'RE': ann(g, ['RetainedEarningsAccumulatedDeficit'], 1),
         'EQ': ann(g, ['StockholdersEquity'], 1), 'EBIT': ann(g, ['OperatingIncomeLoss'], 0), 'INT': ann(g, ['InterestExpense'], 0)}
    if S['TA'] is None or S['EBIT'] is None: continue
    F = pd.concat({k: v for k, v in S.items() if v is not None}, axis=1)
    fl = pd.concat([ann(g, ['Assets'], 1)], axis=1); filed = {}
    x = g['Assets']; r = pd.DataFrame(x['units']['USD']); r = r[(r.form == '10-K') & (r.fp == 'FY')]; r['filed'] = pd.to_datetime(r.filed); filed = r.sort_values('filed').drop_duplicates('end', keep='last').set_index('end').filed
    F['filed'] = filed.reindex(F.index); F['tk'] = tk; rows.append(F.reset_index(drop=True))
F = pd.concat(rows).dropna(subset=['filed', 'TA', 'EBIT']); F = F[F.TA > 0]
F['WC'] = (F.CA - F.CL) if 'CA' in F and 'CL' in F else np.nan; TLv = F.TL.where(F.TL.notna(), F.TA - F.EQ)
F['Z'] = 6.56 * F.WC / F.TA + 3.26 * F.RE / F.TA + 6.72 * F.EBIT / F.TA + 1.05 * F.EQ / TLv
F['NEGEQ'] = F.EQ < 0; F['COV'] = F.EBIT / F.INT.abs().replace(0, np.nan)
F = F.dropna(subset=['Z']).sort_values('filed'); F['filed'] = F.filed.astype('datetime64[ns]')
H['d'] = H.d.astype('datetime64[ns]'); M = pd.merge_asof(H.sort_values('d'), F[['tk', 'filed', 'Z', 'NEGEQ', 'COV']], left_on='d', right_on='filed', by='tk', direction='backward', tolerance=pd.Timedelta(days=540))
print('flips', len(H), 'with a score', M.Z.notna().sum(), 'names', M[M.Z.notna()].tk.nunique())
M = M.dropna(subset=['Z']).copy(); M['mon'] = M.d.dt.strftime('%Y-%m'); M['tr'] = M.d < '2022-01-01'; M['era'] = np.where(M.tr, 'train 19-21', 'test 22-26')
zq = M[M.tr].Z.quantile([1 / 3, 2 / 3]).values; aq = M[M.tr].ATRP.quantile([1 / 3, 2 / 3]).values
M['zb'] = pd.cut(M.Z, [-1e9, zq[0], zq[1], 1e9], labels=['distressed', 'mid', 'healthy']); M['ab'] = pd.cut(M.ATRP, [-1, aq[0], aq[1], 9], labels=['low vol', 'mid vol', 'high vol'])
print('Z cut points (train thirds)', np.round(zq, 2), ' ATRP cuts', np.round(aq, 3))
print('\nreach (hit %) / median r % / n by era x distress, all vol'); print(M.groupby(['era', 'zb'], observed=True).agg(n=('hit', 'size'), reach=('hit', lambda x: round(100 * x.mean())), med_r=('r', lambda x: round(100 * x.median(), 1))).to_string())
print('\nSAME-VOL comparison: reach by era x vol tercile x distress'); print(M.pivot_table(index=['era', 'ab'], columns='zb', values='hit', aggfunc=lambda x: round(100 * x.mean()), observed=True).to_string())
print('\nn'); print(M.pivot_table(index=['era', 'ab'], columns='zb', values='hit', aggfunc='size', observed=True).to_string())
for era, D in M.groupby('era'):
    D = D.assign(low=(D.zb == 'distressed').astype(int), high=(D.zb == 'healthy').astype(int), lA=np.log(D.ATRP))
    m = smf.ols('hit ~ low + high + lA + OVH + BRD', D).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(D.mon)[0]})
    print(f'\n{era}: hit ~ distressed + healthy + log ATRP + OVH + BRD (month-clustered): distressed {m.params.low * 100:+.1f}pp (t {m.tvalues.low:.2f})  healthy {m.params.high * 100:+.1f}pp (t {m.tvalues.high:.2f})  n {int(m.nobs)}')
    m2 = smf.ols('r ~ low + high + lA', D).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(D.mon)[0]}); print(f'   r (flip return to exit) distressed {m2.params.low * 100:+.1f}pp (t {m2.tvalues.low:.2f})')
print('\nnegative book equity flag:'); print(M.groupby(['era', 'NEGEQ']).agg(n=('hit', 'size'), reach=('hit', lambda x: round(100 * x.mean())), med_r=('r', lambda x: round(100 * x.median(), 1))).to_string())
M.to_csv('xs/report_508_distress.csv', index=False)
