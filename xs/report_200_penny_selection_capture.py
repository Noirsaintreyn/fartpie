"""Report 200: of the 2026 clean sub-$5 vector flips (report 199), how many of the big ones would a 2-feature outlier screener (ATRP + EXT train-ECDF rank-sum, cut = train 80th pct; turnover component omitted: no shares data for 2026) have selected?
Features at the signal-day close (day before entry): ATRP = 14d ATR / close, EXT = close / 60d low - 1. Declared as the report_118 model minus turnover. 2026 = test era."""
import pandas as pd, numpy as np
X='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
Of=pd.read_pickle(X+'outlier_features.pkl'); D=pd.read_pickle(X+'wide_daily_ohlcv.pkl'); R=pd.read_pickle(SP+'penny2026_199.pkl'); R=R[~R.disc].copy()
tr=Of[~Of.test].dropna(subset=['ATRP','EXT']); ec=lambda col:(lambda s:(lambda x:np.searchsorted(s,x,side='right')/len(s)))(np.sort(tr[col].values)); fa,fe=ec('ATRP'),ec('EXT'); cut=np.quantile(fa(tr.ATRP.values)+fe(tr.EXT.values),.8); sc=[]
for tk,e in zip(R.tk,R.entry):
    B=D[tk]; B=B[~B.index.duplicated()]; i=B.index.searchsorted(e)-1
    if i<60: sc.append(np.nan); continue
    h,l,c=B.high.values,B.low.values,B.close.values; pc=np.r_[c[0],c[:-1]]; t_=np.maximum(h-l,np.maximum(abs(h-pc),abs(l-pc))); sc.append(fa(t_[i-13:i+1].mean()/c[i])+fe(c[i]/l[i-59:i+1].min()-1))
R['sc']=sc; R=R.dropna(subset=['sc']); R['sel']=R.sc>=cut
for nm,S in (('ALL',R),('SELECTED',R[R.sel]),('NOT selected',R[~R.sel])): print(f'{nm:14s} n{len(S):4d} mean {S.ret.mean()*100:6.1f}% median {S.ret.median()*100:6.1f}% win {np.mean(S.ret>0)*100:3.0f}% >=+50%: {int((S.ret>=.5).sum())} >=+100%: {int((S.ret>=1).sum())}')
