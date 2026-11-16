"""Report 216: is forecasting the SIZE of a sub-$5 move a signal by itself? For the report-203 flips with a full 40-day window (flips ~ random days per report 214), by OS2 quintile (volatility+extension score, train cuts): share reaching +30% first, -15% first, neither; barrier P&L (target +30, stop -15, timeout = 40-day open-to-open return), net of 100bp. A driftless price hits +30 before -15 one time in three (15/45), where expectancy is zero."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; X=pd.read_pickle(SP+'capturable213.pkl'); D=pd.read_pickle(SP+'ext/wide_daily_ohlcv.pkl'); rows=[]
for tk,g in X.groupby('tk'):
    B=D[tk]; B=B[~B.index.duplicated()]; idx=B.index; o,h,l=[B[k].values.astype(float) for k in ('open','high','low')]; n=len(o)
    for ix,d in zip(g.index,g.d):
        i=idx.searchsorted(d); i=i if (i<n and idx[i]==d) else i-1; e=i+1
        if e+40>=n: continue
        pe=o[e]; out=2
        for j in range(e,e+40):
            if l[j]<=pe*.85: out=0; break
            if h[j]>=pe*1.3: out=1; break
        rows.append((ix,out,{1:.30,0:-.15}.get(out,o[e+40]/pe-1)-.01))
F=X.join(pd.DataFrame(rows,columns=['ix','T','pnl']).set_index('ix'),how='inner'); tr=F[F.era=='train']; q=tr.OS2.quantile([.2,.4,.6,.8]).values; F['Q']=np.digitize(F.OS2,q)+1
for e in ('train','test','late26'):
    print(e)
    for k in range(1,6):
        s=F[(F.era==e)&(F.Q==k)]
        if len(s)<10: continue
        t,st=np.mean(s['T']==1),np.mean(s['T']==0); print(f'  OS2 Q{k} n{len(s):4d} target {t*100:4.1f}% stop {st*100:4.1f}% timeout {(1-t-st)*100:4.1f}% | of resolved, target share {t/(t+st)*100:4.1f}% (zero-edge line 33.3%) | barrier P&L mean {s.pnl.mean()*100:5.2f}% | actual vector exit mean {s.r.mean()*100:6.1f}% med {s.r.median()*100:5.1f}%')
