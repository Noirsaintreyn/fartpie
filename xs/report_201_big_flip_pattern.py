"""Report 201: pattern search among 2026 big sub-$5 flips (descriptive; n is small: 8 flips >=+100%, 20 >=+50% of 330 clean). Features at the signal-day close (day before entry), from daily bars: ATRP, EXT (close/60d low-1), R20/R60 (trailing returns), D52 (distance below 52w high), BASE (60d high/low range / close), VX (signal-day volume / 20d mean), VX5 (5d mean vol / 60d mean vol), DV (20d mean dollar volume), GAP (signal-day return), DSB (trading days since previous bear state), PX (entry px). Compare big (>=+50%) vs rest: medians and AUC. Anything with AUC far from .5 is then checked on 2019-25 sub-$5 flips (outlier_features.pkl, 25k flips, same sign needed in train 2019-21 and test 2022-25) before it is believed."""
import pandas as pd, numpy as np
from scipy import stats
from vector_pine import run_state_machine
X='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
D=pd.read_pickle(X+'wide_daily_ohlcv.pkl'); R=pd.read_pickle(SP+'penny2026_199.pkl'); R=R[~R.disc].copy(); rows=[]
for tk,e in zip(R.tk,R.entry):
    B=D[tk]; B=B[~B.index.duplicated()]; B=B[B.index>='2024-06-01']; i=B.index.searchsorted(e)-1
    if i<130: rows.append({}); continue
    h,l,c,v=B.high.values,B.low.values,B.close.values,B.volume.values; pc=np.r_[c[0],c[:-1]]; t_=np.maximum(h-l,np.maximum(abs(h-pc),abs(l-pc)))
    reg,_,_,_=run_state_machine(B.assign(datetime=B.index)); reg=np.asarray(reg); k=i; 
    while k>0 and reg[k-1]==1: k-=1
    # days since previous bear state before this bull run
    j=k-1; 
    while j>0 and reg[j]!=-1: j-=1
    rows.append(dict(ATRP=t_[i-13:i+1].mean()/c[i],EXT=c[i]/l[i-59:i+1].min()-1,R20=c[i]/c[i-20]-1,R60=c[i]/c[i-60]-1,D52=1-c[i]/h[max(0,i-251):i+1].max(),BASE=(h[i-59:i+1].max()-l[i-59:i+1].min())/c[i],VX=v[i]/v[i-20:i].mean(),VX5=v[i-4:i+1].mean()/v[i-59:i-4].mean(),DV=np.log10((v[i-19:i+1]*c[i-19:i+1]).mean()),GAP=c[i]/c[i-1]-1,PRIOR_DD=1-c[max(0,i-130):i+1].min()/c[max(0,i-130):i+1].max(),PX=c[i]))
F=pd.DataFrame(rows,index=R.index); R=R.join(F).dropna(subset=['ATRP']); R['big']=R.ret>=.5; R['huge']=R.ret>=1
print('n',len(R),'big',R.big.sum(),'huge',R.huge.sum())
print(f'{"feature":10s}{"big median":>12s}{"rest median":>13s}{"AUC":>7s}{"p":>8s}')
res=[]
for f in F.columns:
    a=R[R.big][f]; b=R[~R.big][f]; u=stats.mannwhitneyu(a,b); auc=u.statistic/(len(a)*len(b)); res.append((f,a.median(),b.median(),auc,u.pvalue)); print(f'{f:10s}{a.median():12.3f}{b.median():13.3f}{auc:7.2f}{u.pvalue:8.3f}')
pd.set_option('display.width',220); print(R[R.huge][['tk','entry','px','ret','ATRP','EXT','R60','D52','VX','DV','PRIOR_DD']].round(2).to_string(index=False))
R.to_pickle(SP+'big_flip_feats201.pkl')
