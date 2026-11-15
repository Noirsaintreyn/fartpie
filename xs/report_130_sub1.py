"""report_130 (Nov 15 2026): SUB-$1 DAILY FLIPS, can a signal separate big from small? Universe = wide daily universe (1,656 names), flips whose flip-bar close is $0.10-$1.00 and 20d dollar volume >= $300k; entry next open, exit open after bear flip, cost 100bp (also 300bp: sub-$1 spreads are wide); train 2019-21 / test 2022+. SURVIVORSHIP: worst here (names that delisted or went to zero are mostly absent) - all levels are upper bounds; only the SORTING (within sub-$1, same universe) is the claim.
Features at flip-bar close: ATRP, EXT (60d low), LTURN (flip-bar volume / shares, SEC point-in-time; missing -> train median), LDV, LPX, VOLX (flip volume / 20d mean), F2 (flip range / ATR), F11 ((close-level)/close), RES (flip-day return minus EW market), RET (bear-leg retrace), AD5. Declared: ATRP, EXT, LTURN, VOLX, F2, F11, RES, RET all [+] (same mechanism as $1-5: supply/volatility/attention/break size). Test (1) transfer: $1-5 frozen OS (train ECDF from the $1-5 flips) applied to sub-$1; (2) sub-$1 own ridge fit on train for BIG (>=+30%) / HUGE (>=+100%); AUC, top-20% capture, per-trade mean/median. Then 1H pennies (60d; select Aug31-Oct5, judge Oct6-Nov12) with the same score restricted to sub-$1."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from vector_pine import run_state_machine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
X='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; D=pd.read_pickle(X+'wide_daily_ohlcv.pkl'); O=pd.read_pickle(X+'combined_scores.pkl')
SH=pd.read_pickle(X+'shares_pit.pkl'); SH['filed']=pd.to_datetime(SH.filed); SH=SH.sort_values('filed'); shg={k:g for k,g in SH.groupby('tk')}
Q={}; mrets=[]
for tk,q in D.items():
    q=q[['open','high','low','close','volume']].copy(); q=q[~q.index.duplicated()].dropna(); q.index=pd.to_datetime(q.index).tz_localize(None).normalize()
    if len(q)>=300: Q[tk]=q
C=pd.DataFrame({k:v.close for k,v in Q.items()}); mret=C.pct_change().clip(-.5,.5).mean(axis=1); rows=[]
for tk,q in Q.items():
    reg,lev,atr,_=run_state_machine(q.assign(datetime=q.index)); reg=np.asarray(reg); lev=np.asarray(lev,float)
    o,h,l,c,v=[q[k].values.astype(float) for k in ('open','high','low','close','volume')]; n=len(q); dts=q.index
    pc=np.r_[c[0],c[:-1]]; trr=np.maximum(h-l,np.maximum(abs(h-pc),abs(l-pc))); a14=pd.Series(trr).rolling(14).mean().shift(1).values; lo60=pd.Series(l).rolling(60).min().shift(1).values
    dv20=pd.Series(c*v).rolling(20).mean().shift(1).values; v20=pd.Series(v).rolling(20).mean().shift(1).values; clv=np.where(h>l,((c-l)-(h-c))/(h-l),0.)
    g=shg.get(tk)
    for i in np.where((reg[1:]==1)&(reg[:-1]!=1))[0]+1:
        if i<130 or i+2>=n or dts[i]<pd.Timestamp('2019-01-01') or not (.10<=c[i]<1.0) or not (dv20[i]>=3e5): continue
        j=i+1
        while j<n-1 and reg[j]==1: j+=1
        if j+1>=n: continue   # still open
        ent=o[i+1]; ex=o[j+1]
        if not (.05<=ent<=1.3) or ent<=0: continue
        k=i-1
        while k>0 and reg[k]!=1: k-=1
        tr_=l[k+1:i].min() if i-1>k else np.nan; RET=(c[i]-tr_)/(c[k]-tr_) if (i-1>k and c[k]>tr_) else np.nan
        sh=np.nan
        if g is not None:
            gg=g[g.filed<=dts[i]]
            if len(gg): sh=gg.val.iloc[-1]
        rows.append(dict(tk=tk,d=dts[i],px=c[i],r=ex/ent-1,hold=j-i,ATRP=a14[i]/c[i-1] if c[i-1]>0 else np.nan,EXT=c[i-1]/lo60[i]-1 if lo60[i]>0 else np.nan,TURN=v[i]/sh if sh==sh and sh>0 else np.nan,LDV=np.log(dv20[i]),LPX=np.log(c[i]),
            VOLX=v[i]/v20[i] if v20[i]>0 else np.nan,F2=(h[i]-l[i])/a14[i-1] if a14[i-1]>0 else np.nan,F11=(c[i]-lev[i])/c[i] if lev[i]==lev[i] else np.nan,RES=(c[i]/pc[i]-1)-mret.get(dts[i],np.nan),RET=RET,AD5=(clv[i-4:i+1]*v[i-4:i+1]).sum()/max(v[i-4:i+1].sum(),1)))
S=pd.DataFrame(rows); S['test']=S.d>='2022-01-01'; S['LTURN']=np.log(S.TURN.clip(lower=1e-9)); S.to_pickle(X+'sub1_flips.pkl'); print('sub-$1 flips',len(S),'train',(~S.test).sum(),'test',S.test.sum(),'tickers',S.tk.nunique(),'| shares coverage',S.TURN.notna().mean().round(2))
for cost in (.01,.03):
    S['n']=S.r-cost
    for era,e in (('TRAIN',~S.test),('TEST',S.test)):
        z=S[e]; print(f'cost {cost*100:.0f}bp {era}: n {len(z)} mean {z.n.mean()*100:+6.1f}% med {z.n.median()*100:+5.1f}% win {np.mean(z.n>0)*100:4.1f}% big30 {np.mean(z.n>=.3)*100:4.1f}% huge100 {np.mean(z.n>=1)*100:4.1f}% hold med {z.hold.median():.0f}d')
S['n']=S.r-.01; S['BIG']=S.n>=.3; S['HUGE']=S.n>=1.0; tr=S[~S.test]
O['LTURN']=np.log(O.TURN.clip(lower=1e-9)); T1=O[(~O.test)&(O.px<5)]
def ecdf(df,c): s=np.sort(df[c].dropna().values); return lambda x: np.searchsorted(s,x,side='right')/len(s)
# (1) transfer: frozen $1-5 OS
S['OS']=ecdf(T1,'ATRP')(S.ATRP.values)+ecdf(T1,'EXT')(S.EXT.values)+np.where(S.LTURN.notna(),ecdf(T1,'LTURN')(S.LTURN.fillna(0).values),ecdf(T1,'ATRP')(S.ATRP.values))  # missing turnover -> ATRP rank (neutral-ish)
def rep(name,col):
    out=f'  {name:26s}'
    for era,e in (('TR',~S.test),('TE',S.test)):
        z=S[e].dropna(subset=[col]); cut=S[~S.test][col].quantile(.8); top=z[z[col]>=cut]
        out+=f' | {era} AUC big {roc_auc_score(z.BIG,z[col]):.3f} huge {roc_auc_score(z.HUGE,z[col]):.3f} top20 ({len(top)/len(z)*100:3.0f}%): mean {top.n.mean()*100:+6.1f}% med {top.n.median()*100:+5.1f}% catch big {top.BIG.sum()/max(1,z.BIG.sum())*100:4.1f}% huge {top.HUGE.sum()/max(1,z.HUGE.sum())*100:4.1f}%'
    print(out)
print('\n(1) $1-5 frozen OS on sub-$1'); rep('OS ($1-5 ECDFs)','OS')
for f in ('ATRP','EXT','LTURN','VOLX','F2','F11','RES','RET','LDV','LPX','AD5'): rep('single '+f,f)
# (2) own ridge, train only
feats=['ATRP','EXT','LTURN','LDV','LPX','VOLX','F2','F11','RES','RET','AD5']; Rk=pd.DataFrame({f:ecdf(tr,f)(S[f].fillna(tr[f].median()).values) for f in feats},index=S.index)
for tgt in ('BIG','HUGE'):
    m=LogisticRegression(C=.3,max_iter=2000).fit(Rk[~S.test],S[~S.test][tgt]); S['M_'+tgt]=m.decision_function(Rk); print(tgt,'coefs',dict(zip(feats,m.coef_[0].round(2)))); rep('own ridge '+tgt,'M_'+tgt)
S.to_pickle(X+'sub1_flips.pkl')
