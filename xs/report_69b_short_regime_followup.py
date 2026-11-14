"""Follow-up to report_69 (declared before results): is the high-short edge a speculative-risk-on REGIME (explanation) rather than a flip effect? (1) per half-year: universe return, high-minus-low-short basket spread, mean of all flips, mean of SQ10 flips -> correlation across half-years (does SQ10 edge track the spread?). (2) is high-short = biotech? biotech share of SQ10 flips and of HUGE winners by era; SQ10 & LEAD excluding biotech SIC. (3) what separated 2022-24 (LEAD but no money): breadth of the universe vector (share of names in bull regime) and universe 60d return at flip bar for SQ10&LEAD flips by era."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('report_69_short_regime.py').read(); exec(src.split("print('PART A")[0]); 
ret = cl.pct_change(fill_method=None).clip(-0.5, 1.0); hi = (dtc >= 10).shift(1).fillna(False); lo = (dtc < 10).shift(1).fillna(False) & dtc.shift(1).notna(); bk = ret.where(hi).mean(axis=1); un = ret.mean(axis=1); lw = ret.where(lo).mean(axis=1)
lead = ((1 + bk.fillna(0)).rolling(40).apply(np.prod, raw=True) - (1 + un.fillna(0)).rolling(40).apply(np.prod, raw=True)); P['LEAD'] = lead.reindex(P.d0).values > 0
P['BIO'] = ((P.sic >= 2833) & (P.sic <= 2836)) | (P.sic == 8731); P['hy'] = P.t.dt.year.astype(str) + 'H' + ((P.t.dt.month > 6) + 1).astype(str)
W = P[P.DTC.notna()]; rows = []
for h, g in W.groupby('hy'):
    m = (un.index.year.astype(str) + 'H' + ((un.index.month > 6) + 1).astype(str)) == h; s = g[g.SQ10]
    rows.append(dict(h=h, uni=un[m].mean() * 252 * 100, spread=(bk[m] - lw[m]).mean() * 252 * 100, allflip=g.rn.mean() * 100, sq10=s.rn.mean() * 100 if len(s) > 20 else np.nan, sq10_lead=g[g.SQ10 & g.LEAD].rn.mean() * 100 if (g.SQ10 & g.LEAD).sum() > 20 else np.nan, nlead=int((g.SQ10 & g.LEAD).sum())))
R = pd.DataFrame(rows).dropna(subset=['sq10']); print('(1) half-year table: universe ann%, high-minus-low-short spread ann%, all-flip mean%, SQ10 mean%, SQ10&LEAD mean% (n)'); [print(f'  {r.h}: uni {r.uni:+6.1f} spread {r.spread:+6.1f} | all {r.allflip:+6.2f} SQ10 {r.sq10:+7.2f} SQ10&LEAD {r.sq10_lead:+7.2f} ({r.nlead})') for r in R.itertuples()]
print(f'  corr across half-years: SQ10 mean vs spread {R.sq10.corr(R.spread):+.2f} | SQ10 vs universe return {R.sq10.corr(R.uni):+.2f} | all-flip mean vs universe {R.allflip.corr(R.uni):+.2f} | SQ10 minus all-flip vs spread {(R.sq10-R.allflip).corr(R.spread):+.2f}  (n={len(R)} half-years)')
print('\n(2) biotech share and SQ10&LEAD without biotech, by era')
for en, a, b in (('2019-21', '2019', '2022'), ('2022-24', '2022', '2025'), ('2025-26Q1', '2025', '2026-04')):
    S = W[(W.t >= a) & (W.t < b)]; sq = S[S.SQ10]; hg = S[S.HUGE == 1]; base = S.HUGE.mean()
    nb = S[S.SQ10 & S.LEAD & ~S.BIO]; bb = S[S.SQ10 & S.LEAD & S.BIO]
    print(f'  {en}: biotech share of all flips {S.BIO.mean()*100:.0f}% | of SQ10 flips {sq.BIO.mean()*100:.0f}% | of HUGE winners {hg.BIO.mean()*100:.0f}% | SQ10&LEAD non-biotech n {len(nb)} mean {nb.rn.mean()*100:+.2f}% HUGE lift {nb.HUGE.mean()/base:.2f} | biotech n {len(bb)} mean {bb.rn.mean()*100:+.2f}% lift {bb.HUGE.mean()/base:.2f}')
print('\n(3) market context at the flip bar for SQ10&LEAD flips (mean breadth of universe vector bull share, universe 20d median return) and their mean return')
for en, a, b in (('2019-21', '2019', '2022'), ('2022-24', '2022', '2025'), ('2025-26Q1', '2025', '2026-04')):
    S = W[(W.t >= a) & (W.t < b)]; x = S[S.SQ10 & S.LEAD]; y = S[S.SQ10 & ~S.LEAD]
    print(f'  {en}: LEAD-on breadth {x.BREADTH.astype(float).mean():.2f} mkt20 {x.MKT20.mean()*100:+.1f}% mean {x.rn.mean()*100:+.2f}% | LEAD-off breadth {y.BREADTH.astype(float).mean():.2f} mkt20 {y.MKT20.mean()*100:+.1f}% mean {y.rn.mean()*100:+.2f}%')
    for lo_, hi_ in ((0, .35), (.35, .6), (.6, 1.01)):
        z = S[S.SQ10 & S.LEAD & (S.BREADTH.astype(float) >= lo_) & (S.BREADTH.astype(float) < hi_)]
        if len(z) >= 15: print(f'      LEAD & breadth {lo_:.2f}-{min(hi_,1):.2f}: n {len(z)} mean {z.rn.mean()*100:+.2f}% HUGE/100 {z.HUGE.mean()*100:.2f}')
