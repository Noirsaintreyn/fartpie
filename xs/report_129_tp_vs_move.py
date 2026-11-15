"""report_129 (Nov 15 2026): TAKE-PROFIT FLIPS vs MOVE FLIPS + the three untested ideas + flip correlation. Rey's mechanism: a bull flip inside a larger downtrend, or on no buying volume, is shorts/longs taking profit / position closing (exhaustion), not new buyers establishing a new fair value; price returns to the start after a pump unless news/earnings reset value. Declared signs [+ = better returns] fixed BEFORE results, all features at flip-bar close (PIT):
IDEA 1 WEEKLY NESTING: W_BULL (weekly vector bull at the last completed week) [+]; W_FRESH (weekly flip within 4 weeks) [+]. IDEA 2 RATCHET SHAPE: RET = fraction of the prior bear leg retraced at the flip close [+]; BRK = RET>=1 (above where the downleg began) [+]; LSLOPE = ratchet-level change over last 20 bars / ATR [-] (still falling = downtrend context); DTC = close<SMA100 & SMA100 falling [-]. IDEA 3 PATTERN LIBRARY: KMeans (k=10, fit on train paths: 40 closes before the flip normalized by ATR); train-best 2 clusters judged on test. BUY/SELL VOLUME: AD5 = sum(CLV*vol)/sum(vol) over last 5 bars [+] (buyers stepping in), ADBEAR = same over the prior bear leg [+] (accumulation during decline), VRB = flip-bar volume / mean bear-leg volume [+]. CORRELATION: CORR60 (60d corr of daily returns with equal-weight universe) and RES = flip-day return minus market return [+ idiosyncratic = new information], PEERS (existing). 
Train 2019-21 thresholds, test 2022+, 30bp, hold to bear flip; universes A all, B px<5, C px<5 & frozen top20. Pass: same sign both eras, month-clustered |t|>=2 on test."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from vector_pine import run_state_machine
from sklearn.cluster import KMeans
X='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; D=pd.read_pickle(X+'wide_daily_ohlcv.pkl'); O=pd.read_pickle(X+'combined_scores.pkl'); COST=.003
Q={}
for tk,q in D.items():
    q=q[['open','high','low','close','volume']].copy(); q=q[~q.index.duplicated()].dropna(); q.index=pd.to_datetime(q.index).tz_localize(None).normalize()
    if len(q)>=300: Q[tk]=q
C=pd.DataFrame({k:v.close for k,v in Q.items()}); mret=C.pct_change().clip(-.5,.5).mean(axis=1)
info={}; paths={}
for tk,g in O.groupby('tk'):
    q=Q.get(tk)
    if q is None: continue
    reg,lev,atr,_=run_state_machine(q.assign(datetime=q.index)); reg=np.asarray(reg); lev=np.asarray(lev,float)
    wq=q.resample('W-FRI').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    if len(wq)>60: wr=pd.Series(np.asarray(run_state_machine(wq.assign(datetime=wq.index))[0]),index=wq.index)
    else: wr=None
    o,h,l,c,v=[q[k].values.astype(float) for k in ('open','high','low','close','volume')]; n=len(q); pos={d:i for i,d in enumerate(q.index)}
    pc=np.r_[c[0],c[:-1]]; trr=np.maximum(h-l,np.maximum(abs(h-pc),abs(l-pc))); a14=pd.Series(trr).rolling(14).mean().shift(1).values
    clv=np.where(h>l,((c-l)-(h-c))/(h-l),0.); sma100=pd.Series(c).rolling(100).mean().values
    r_=pd.Series(c,index=q.index).pct_change(); corr=r_.rolling(60).corr(mret.reindex(q.index)).shift(1).values
    for _,r in g.iterrows():
        i=pos.get(r.d)
        if i is None or i<130: continue
        k=i-1
        while k>0 and reg[k]!=1: k-=1
        if i-1<=k: continue
        tr_=l[k+1:i].min(); top=c[k]; RET=(c[i]-tr_)/(top-tr_) if top>tr_ else np.nan
        wk=np.nan;wf=np.nan
        if wr is not None:
            a=wr.index.searchsorted(r.d,side='right')-1
            if a>=5: wk=float(wr.iloc[a]==1); wf=float((wr.iloc[a-3:a+1]==1).any() and (wr.iloc[max(0,a-12):a-3]!=1).all()) if a>=12 else np.nan
        ad5=(clv[i-4:i+1]*v[i-4:i+1]).sum()/max(v[i-4:i+1].sum(),1); adb=(clv[k+1:i]*v[k+1:i]).sum()/max(v[k+1:i].sum(),1); vrb=v[i]/max(v[k+1:i].mean(),1)
        ls=(lev[i-1]-lev[i-21])/a14[i-1] if a14[i-1]>0 and lev[i-1]==lev[i-1] and lev[i-21]==lev[i-21] else np.nan
        dtc=float(c[i]<sma100[i] and sma100[i]<sma100[i-20])
        res=(c[i]/pc[i]-1)-mret.get(r.d,np.nan)
        info[(tk,r.d)]=dict(W_BULL=wk,W_FRESH=wf,RET=RET,BRK=float(RET>=1) if RET==RET else np.nan,LSLOPE=ls,DTC=dtc,AD5=ad5,ADBEAR=adb,VRB=vrb,CORR60=corr[i],RES=res)
        if a14[i-1]>0: paths[(tk,r.d)]=np.clip((c[i-40:i+1]-c[i])/a14[i-1],-30,30)
I=pd.DataFrame.from_dict(info,orient='index'); I.index=pd.MultiIndex.from_tuples(I.index,names=['tk','d']); A=O.merge(I.reset_index(),on=['tk','d'])
# pattern library
keys=[(a,b) for a,b in zip(A.tk,A.d)]; P=np.array([paths.get(k,np.full(41,np.nan)) for k in keys]); okp=~np.isnan(P).any(1)
trm=(~A.test.values)&okp; km=KMeans(10,n_init=5,random_state=0).fit(P[trm]); A['CL']=np.nan; A.loc[okp,'CL']=km.predict(P[okp])
A['mo']=A.d.dt.to_period('M').astype(str); A=A[A.d>='2019-01-01']; A['top']=A.lr>=A[~A.test].lr.quantile(.8); A.to_pickle(X+'tp_move_features.pkl')
sign=dict(W_BULL='+',W_FRESH='+',RET='+',BRK='+',LSLOPE='-',DTC='-',AD5='+',ADBEAR='+',VRB='+',CORR60='-',RES='+',PEERS='-')
UN={'A all':A,'B px<5':A[A.px<5],'C px<5&top20':A[(A.px<5)&A.top]}
def t_(d): return d.mean()/(d.std()/np.sqrt(len(d))) if len(d)>5 and d.std()>0 else np.nan
for un,U in UN.items():
    print(f'\n#### {un}: train n {(~U.test).sum()} test n {U.test.sum()}')
    for f,sg in sign.items():
        tr=U[~U.test].dropna(subset=[f]); binf=tr[f].nunique()<=2
        if binf: hiF=lambda z:z[z[f]==1]; loF=lambda z:z[z[f]==0]; lab='1/0'
        else: q1,q2=tr[f].quantile([1/3,2/3]).values; hiF=lambda z,q2=q2:z[z[f]>=q2]; loF=lambda z,q1=q1:z[z[f]<=q1]; lab='T3/T1'
        out=[];ok=True
        for era,z in (('TR',U[~U.test]),('TE',U[U.test])):
            z=z.dropna(subset=[f]); hi=hiF(z); lo=loF(z); dd=(hi.groupby('mo').n.mean()-lo.groupby('mo').n.mean()).dropna(); tt=t_(dd); sp=hi.n.mean()-lo.n.mean()
            out.append(f'{era}: {lab} {hi.n.mean()*100:+6.1f}%(n{len(hi)})/{lo.n.mean()*100:+6.1f}%(n{len(lo)}) med {hi.n.median()*100:+5.1f}/{lo.n.median()*100:+5.1f} big30 {np.mean(hi.n>=.3)*100:4.1f}/{np.mean(lo.n>=.3)*100:4.1f} t {tt:+.2f}')
            ok&=((sp>0)==(sg=='+')) and (era=='TR' or (tt==tt and abs(tt)>=2 and (tt>0)==(sg=='+')))
        print(f'  {f:8s}[{sg}] '+' | '.join(out)+('  <== PASS' if ok else ''))
    # pattern library: train-best 2 clusters by train mean (>=30 trades)
    trz=U[~U.test].dropna(subset=['CL']); cm=trz.groupby('CL').n.agg(['mean','count']); cm=cm[cm['count']>=30].sort_values('mean',ascending=False)
    if len(cm)>=3:
        best=list(cm.index[:2]); worst=list(cm.index[-2:])
        for era,z in (('TR',U[~U.test]),('TE',U[U.test])):
            z=z.dropna(subset=['CL']); b=z[z.CL.isin(best)]; w=z[z.CL.isin(worst)]; print(f'  PATTERN train-best clusters {best} vs worst {worst} {era}: {b.n.mean()*100:+6.1f}%(n{len(b)}) vs {w.n.mean()*100:+6.1f}%(n{len(w)}) med {b.n.median()*100:+5.1f}/{w.n.median()*100:+5.1f}')
# declared MOVE vs TP composite
for un,U in UN.items():
    U=U.dropna(subset=['DTC','AD5']); mv=(U.DTC==0)&(U.AD5>0); tp=(U.DTC==1)&(U.AD5<=0)
    for era,e in (('TR',~U.test),('TE',U.test)):
        a=U[e&mv]; b=U[e&tp]; c=U[e&~mv&~tp]; print(f'COMPOSITE {un:13s} {era}: MOVE(no downtrend & buying) {a.n.mean()*100:+6.1f}% med {a.n.median()*100:+5.1f} n{len(a)} | TP(downtrend & selling) {b.n.mean()*100:+6.1f}% med {b.n.median()*100:+5.1f} n{len(b)} | other {c.n.mean()*100:+6.1f}% n{len(c)}')
