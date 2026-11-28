"""Report 513 (Nov 27 2026): Rey - 'the buyback lead was in my old mid/small-cap screener, try that with the top-ATRP flips'. Report 438 tested BUYER (>=10% of operating cash flow) on ALL sub-$5 flips: +0.09pp, t +1.0 (no admission). 512 found buyback lifts REACH inside the top-vol third. This tests the screener form: BUYER flag x top-ATRP flips, as a trading rule, net of 40bp, by year.
Declared: buyer in the top ATRP third (also top fifth / tenth) beats non-buyers in the same top-vol set on mean AND median net r, same sign in train 2019-21 and test 2022+; BUYER defs: BB>0 (any repurchase cash) and BUYER10 (repurchase / operating cash flow >= 10%, 438's def). Unit: report-464 daily flips px<5/<10/<20 (8,408, 2019-26Q1), PIT 10-K filed before the flip."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
s512 = open('xs/report_512_institutional_basket.py').read(); exec(s512[s512.index('import numpy as np'):s512.index('\nres = []')])
import json
rows = []
for tk in sorted(set(O.tk)):
    f = f'data/fund/sec_small/{tk}.json'
    if not os.path.exists(f): continue
    us = json.load(open(f))['facts'].get('us-gaap', {}); bb = ent(us, ['PaymentsForRepurchaseOfCommonStock']); oc = ent(us, ['NetCashProvidedByUsedInOperatingActivities'])
    if oc is None: continue
    X = oc.rename(columns={'val': 'ocf'}); X['bb'] = bb.val.reindex(X.index) if bb is not None else np.nan; X['tk'] = tk; rows.append(X.reset_index(drop=True))
G = pd.concat(rows); G['filed'] = G.filed.astype('datetime64[ns]'); G = G.sort_values('filed')
O3 = pd.merge_asof(O2.sort_values('d'), G[['tk', 'filed', 'ocf', 'bb']].rename(columns={'filed': 'filed2'}), left_on='d', right_on='filed2', by='tk', direction='backward', allow_exact_matches=False, tolerance=pd.Timedelta(days=540))
O3['BUYER10'] = np.where(O3.ocf.isna() | O3.bb.isna(), np.where(O3.bb.isna() & O3.ocf.notna(), 0.0, np.nan), ((O3.ocf > 0) & (O3.bb / O3.ocf.where(O3.ocf > 0) >= .10)).astype(float))
O3['BBANY'] = np.where(O3.ocf.isna(), np.nan, (O3.bb.fillna(0) > 0).astype(float)); O3['net'] = O3.r - 0.004; O3['yr'] = O3.d.dt.year
def line(v): return f"n{len(v):4} reach {100 * v.hit.mean():3.0f}%  mean {100 * v.net.mean():+6.1f}%  trim-mean {100 * v.net.sort_values().iloc[3:-3].mean() if len(v) > 12 else np.nan:+6.1f}%  median {100 * v.net.median():+6.1f}%  win {100 * (v.net > 0).mean():3.0f}%"
print('coverage: BUYER10 known', O3.BUYER10.notna().mean().round(2), ' share buyer', O3.BUYER10.mean().round(2), '| BBANY share', O3.BBANY.mean().round(2))
for cap in (5, 10, 20):
    D = O3[(O3.px < cap) & O3.BUYER10.notna()].copy(); tr = D.tr
    for q, nm in ((2 / 3, 'top third'), (.8, 'top fifth'), (.9, 'top tenth')):
        D['top'] = D.ATRP >= D.ATRP[tr].quantile(q)
        print(f'\n=== px<{cap}, {nm} ATRP (cut {D.ATRP[tr].quantile(q):.3f})')
        for era, E in (('train 19-21', D[D.tr & D.top]), ('test 22-26', D[~D.tr & D.top])):
            for lab, v in (('all', E), ('BUYER10', E[E.BUYER10 == 1]), ('BBANY', E[E.BBANY == 1]), ('non-buyer', E[E.BBANY == 0])): print(f'  {era} {lab:10} ' + line(v))
print('\n### BY YEAR, px<10, top third: BUYER10 vs non-buyer (net of 40bp)')
D = O3[(O3.px < 10) & O3.BUYER10.notna()].copy(); D['top'] = D.ATRP >= D.ATRP[D.tr].quantile(2 / 3)
for y, g in D[D.top].groupby('yr'): print(f'  {y}  BUYER10 ' + line(g[g.BUYER10 == 1]) + '   | non ' + line(g[g.BBANY == 0]))
print('\n### BY YEAR, px<20, top third')
D = O3[(O3.px < 20) & O3.BUYER10.notna()].copy(); D['top'] = D.ATRP >= D.ATRP[D.tr].quantile(2 / 3)
for y, g in D[D.top].groupby('yr'): print(f'  {y}  BUYER10 ' + line(g[g.BUYER10 == 1]) + '   | non ' + line(g[g.BBANY == 0]))
