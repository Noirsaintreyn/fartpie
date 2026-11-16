"""Report 209: capital structure as the universe variable for sub-$5 flips (a non-price axis). DECLARED before results.
Theory: Merton (1974): equity is a call option on the firm's assets struck at its debt. (a) Leverage raises the elasticity of equity to asset value, so when the vector flips on a levered 'stub' the equity response is convex -> more BIG flips (H_LEV, sign + on P(BIG); expected to carry bigger losses too, so it must ALSO beat on mean ex-best-3 to count). (b) Graham net-nets: a firm whose current assets exceed ALL liabilities and whose market cap is small relative to that surplus has a liquidation floor under the price -> smaller losses / higher median without losing the tail (H_NCAV, sign + on median and ex-best-3 mean).
Data: SEC XBRL frames (Liabilities, or L&SE - SE when missing; AssetsCurrent; Assets; dei shares outstanding), quarter-ends 2018Q2-2026Q3; a quarter is usable 90 days after its end (frames hold the latest reported value, so restatements can leak: caveat). Market cap = shares x signal-day close. LEV = Liabilities / market cap; NCAV = (AssetsCurrent - Liabilities) / market cap; BLEV = Liabilities / Assets.
Sample: report-203 flips (train 2019-21 | test 2022-Feb26 | late 2026). Cuts on train. PASS = same sign in train and test with |t|>=2 on the top-vs-rest difference (BIG rate z-test; mean ex-best3 Welch t on returns winsorized at the 99th pct), n>=30 each. Two hypotheses = two tests. Also: is it new information? rank corr with name ATRP, and the BIG-rate split inside ATRP terciles."""
import json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy import stats
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; E=SP+'ext/'
F=pd.read_pickle(E+'sec_frames.pkl'); F['end']=pd.to_datetime(F.end); m=json.load(open(E+'tk_cik.json')); Z=pd.read_pickle(SP+'selectors203.pkl'); D=pd.read_pickle(E+'wide_daily_ohlcv.pkl')
P=F.pivot_table(index=['cik','end'],columns='tag',values='val',aggfunc='last').reset_index(); P['L']=P.Liabilities.fillna(P.LiabilitiesAndStockholdersEquity-P.StockholdersEquity)
BS={c:g.sort_values('end') for c,g in P.dropna(subset=['L']).groupby('cik')}; SH={c:g.dropna(subset=['EntityCommonStockSharesOutstanding']).sort_values('end') for c,g in P.groupby('cik')}
rows=[]
for ix,tk,d in zip(Z.index,Z.tk,Z.d):
    c=m.get(tk); 
    if c is None or c not in BS: continue
    b=BS[c]; b=b[(b.end<=d-pd.Timedelta(days=90))&(b.end>=d-pd.Timedelta(days=500))]
    if not len(b): continue
    b=b.iloc[-1]; s=SH.get(c); s=s[(s.end<=d)&(s.end>=d-pd.Timedelta(days=400))] if s is not None else []
    B=D[tk]; B=B[~B.index.duplicated()]; px=B.close.get(d,np.nan); sh=s.EntityCommonStockSharesOutstanding.iloc[-1] if len(s) else np.nan; mc=sh*px
    rows.append(dict(ix=ix,L=b.L,AC=b.AssetsCurrent,A=b.Assets,MC=mc,LEV=b.L/mc if mc>0 else np.nan,NCAV=(b.AssetsCurrent-b.L)/mc if mc>0 else np.nan,BLEV=b.L/b.Assets if b.Assets and b.Assets>0 else np.nan))
X=Z.join(pd.DataFrame(rows).set_index('ix'),how='inner'); print('flips',len(Z),'with balance sheet',len(X),'with market cap',X.LEV.notna().sum(),X.groupby('era').size().to_dict())
def wins(r): return np.minimum(r,np.quantile(r,.99))
def cmp(a,b,tag):
    ra=a.r.sort_values(); rb=b.r.sort_values(); p1,p2=a.big.mean(),b.big.mean(); p=(a.big.sum()+b.big.sum())/(len(a)+len(b)); z=(p1-p2)/np.sqrt(p*(1-p)*(1/len(a)+1/len(b))+1e-12); t=stats.ttest_ind(wins(a.r.values),wins(b.r.values),equal_var=False).statistic
    return f'{tag}: n{len(a):4d} BIG {p1*100:5.1f}% mean {a.r.mean()*100:6.1f}% x3 {ra.iloc[:-3].mean()*100:6.1f}% med {a.r.median()*100:6.1f}% win {np.mean(a.r>0)*100:3.0f}% | rest n{len(b):4d} BIG {p2*100:5.1f}% x3 {rb.iloc[:-3].mean()*100:6.1f}% med {b.r.median()*100:6.1f}% win {np.mean(b.r>0)*100:3.0f}% | zBIG {z:+.1f} t(mean,wins) {t:+.1f}'
tr=X[X.era=='train']
print('\nH_LEV: top quintile of Liabilities/market cap (train cut %.2f)'%tr.LEV.quantile(.8))
for e in ('train','test','late26'):
    s=X[(X.era==e)&X.LEV.notna()]; c=tr.LEV.quantile(.8); a=s[s.LEV>=c]; b=s[s.LEV<c]
    if len(a)>=5: print('  ',cmp(a,b,e))
print('  quintile BIG rate / median (train | test):',' ; '.join(f'Q{k+1} '+' | '.join(f'{x.big.mean()*100:4.1f}%/{x.r.median()*100:5.1f}%' for x in [X[(X.era==e)&(X.LEV>lo)&(X.LEV<=hi)] for e in ('train','test')]) for k,(lo,hi) in enumerate(zip([-np.inf,*tr.LEV.quantile([.2,.4,.6,.8]).values],[*tr.LEV.quantile([.2,.4,.6,.8]).values,np.inf]))))
print('\nH_NCAV: net current asset value / market cap')
for nm,f in (('NCAV>0.5 (net-net)',lambda s:s.NCAV>.5),('NCAV>0',lambda s:s.NCAV>0),('NCAV>1 (below liquidation value)',lambda s:s.NCAV>1)):
    for e in ('train','test','late26'):
        s=X[(X.era==e)&X.NCAV.notna()]; a=s[f(s)]; b=s[~f(s)]
        if len(a)>=5 and len(b)>=5: print(f'  {nm:32s}',cmp(a,b,e))
print('\nnew information? Spearman with name ATRP (ATRP col):',{k:round(X[[k,'ATRP']].dropna().corr(method='spearman').iloc[0,1],2) for k in ('LEV','NCAV','BLEV')})
for e in ('train','test'):
    s=X[(X.era==e)&X.LEV.notna()].copy(); s['at']=pd.qcut(s.ATRP,3,labels=['loATR','midATR','hiATR']); c=tr.LEV.quantile(.8)
    print(f'  {e} BIG rate hi-LEV vs rest inside ATRP terciles:',' ; '.join(f'{k}: {g[g.LEV>=c].big.mean()*100:4.1f}% (n{(g.LEV>=c).sum()}) vs {g[g.LEV<c].big.mean()*100:4.1f}%' for k,g in s.groupby('at')))
    s2=X[(X.era==e)&X.NCAV.notna()].copy(); s2['at']=pd.qcut(s2.ATRP,3,labels=['loATR','midATR','hiATR']); print(f'  {e} median r NCAV>0.5 vs rest inside ATRP terciles:',' ; '.join(f'{k}: {g[g.NCAV>.5].r.median()*100:5.1f}% (n{(g.NCAV>.5).sum()}) vs {g[g.NCAV<=.5].r.median()*100:5.1f}%' for k,g in s2.groupby('at')))
X.to_pickle(SP+'capstruct209.pkl')
