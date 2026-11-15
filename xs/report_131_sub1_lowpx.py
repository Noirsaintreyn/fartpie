"""report_131 (Nov 15 2026): SUB-$1 flips on the low-priced daily set (lowpx_daily: 2,024 names, 501 bars Nov 2024-Nov 2026; report_130 showed the wide universe has only 36 sub-$1 flips, i.e. it cannot answer this). Flip bar close $0.10-$1.00, dv20 >= $200k, entry next open, exit open after bear flip (open trades marked to the last close), cost 1% and 3%. SPLIT: with 2 years of data and 130 bars of warm-up, select = flips Jun 2025-Feb 2026 (frozen cuts from here), judge = Mar 2026-Nov 2026. Survivorship/selection bias: the list is names that are low-priced NOW (delisted names absent) -> levels are NOT trustworthy, only within-sample sorting is the claim; one regime. Declared [+]: ATRP, EXT, LTURN, VOLX, F2, F11. Score OS = $1-5 train ECDF ranks of ATRP+EXT (+LTURN where shares known). Judge: AUC for BIG (>=+30%)/HUGE (>=+100%), top-20% capture, per-trade mean/median/ex-top3."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from vector_pine import run_state_machine
from sklearn.metrics import roc_auc_score
S0='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; X=S0+'ext/'; L=pd.read_pickle(S0+'lowpx_daily.pkl'); O=pd.read_pickle(X+'combined_scores.pkl')
SH=pd.read_pickle(X+'shares_pit.pkl'); SH['filed']=pd.to_datetime(SH.filed); SH=SH.sort_values('filed'); shg={k:g for k,g in SH.groupby('tk')}
Q={}
for tk,q in L.items():
    q=q.copy(); q.columns=[c.lower() for c in q.columns]; q.index=pd.to_datetime(q.index); q.index=(q.index.tz_localize(None) if q.index.tz is not None else q.index).normalize(); q=q[~q.index.duplicated()][['open','high','low','close','volume']].dropna()
    if len(q)>=200: Q[tk]=q
C=pd.DataFrame({k:v.close for k,v in Q.items()}); mret=C.pct_change().clip(-.5,.5).mean(axis=1); rows=[]
for tk,q in Q.items():
    reg,lev,atr,_=run_state_machine(q.assign(datetime=q.index)); reg=np.asarray(reg); lev=np.asarray(lev,float)
    o,h,l,c,v=[q[k].values.astype(float) for k in ('open','high','low','close','volume')]; n=len(q); dts=q.index
    pc=np.r_[c[0],c[:-1]]; trr=np.maximum(h-l,np.maximum(abs(h-pc),abs(l-pc))); a14=pd.Series(trr).rolling(14).mean().shift(1).values; lo60=pd.Series(l).rolling(60).min().shift(1).values
    dv20=pd.Series(c*v).rolling(20).mean().shift(1).values; v20=pd.Series(v).rolling(20).mean().shift(1).values; g=shg.get(tk)
    for i in np.where((reg[1:]==1)&(reg[:-1]!=1))[0]+1:
        if i<75 or i+2>=n or not (.10<=c[i]<1.0) or not (dv20[i]>=2e5): continue
        j=i+1
        while j<n-1 and reg[j]==1: j+=1
        ent=o[i+1]; ex=o[j+1] if j+1<n else c[n-1]; opn=(j+1>=n)
        if not (.05<=ent<=1.3): continue
        sh=np.nan
        if g is not None:
            gg=g[g.filed<=dts[i]]
            if len(gg): sh=gg.val.iloc[-1]
        rows.append(dict(tk=tk,d=dts[i],px=c[i],open=opn,r=ex/ent-1,hold=j-i,ATRP=a14[i]/c[i-1],EXT=c[i-1]/lo60[i]-1,LTURN=np.log(max(v[i]/sh,1e-9)) if sh==sh and sh>0 else np.nan,VOLX=v[i]/v20[i],F2=(h[i]-l[i])/a14[i-1],F11=(c[i]-lev[i])/c[i] if lev[i]==lev[i] else np.nan,RES=(c[i]/pc[i]-1)-mret.get(dts[i],np.nan)))
S=pd.DataFrame(rows); S=S[S.d>='2025-06-01'].copy(); S['judge']=S.d>='2026-03-01'; print('flips',len(S),'select',(~S.judge).sum(),'judge',S.judge.sum(),'tickers',S.tk.nunique(),'turnover coverage',S.LTURN.notna().mean().round(2))
for cost in (.01,.03):
    S['n']=S.r-cost
    for lab,e in (('SELECT',~S.judge),('JUDGE ',S.judge)):
        z=S[e]; ex3=z.n.sort_values(ascending=False).iloc[3:].mean()
        print(f'cost {cost*100:.0f}% {lab}: n {len(z)} mean {z.n.mean()*100:+6.1f}% med {z.n.median()*100:+5.1f}% ex-top3 {ex3*100:+6.1f}% win {np.mean(z.n>0)*100:4.1f}% big30 {np.mean(z.n>=.3)*100:4.1f}% huge {np.mean(z.n>=1)*100:4.1f}% hold med {z.hold.median():.0f}d')
S['n']=S.r-.01; S['BIG']=S.n>=.3; S['HUGE']=S.n>=1.0; T1=O[(~O.test)&(O.px<5)].copy(); T1['LTURN']=np.log(T1.TURN.clip(lower=1e-9))
ec=lambda c:(lambda x,s=np.sort(T1[c].dropna().values):np.searchsorted(s,x,side='right')/len(s))
S['OS']=ec('ATRP')(S.ATRP.values)+ec('EXT')(S.EXT.values)+np.where(S.LTURN.notna(),ec('LTURN')(S.LTURN.fillna(0).values),(ec('ATRP')(S.ATRP.values)+ec('EXT')(S.EXT.values))/2)
S['OSa']=ec('ATRP')(S.ATRP.values)+ec('EXT')(S.EXT.values)
def rep(name,col):
    out=f'  {name:10s}'; cut=S[~S.judge][col].quantile(.8)
    for lab,e in (('SEL',~S.judge),('JDG',S.judge)):
        z=S[e].dropna(subset=[col]); top=z[z[col]>=cut]; rest=z[z[col]<cut]
        if z.BIG.nunique()<2 or z.HUGE.nunique()<2: out+=f' | {lab} n{len(z)} top20 n{len(top)} mean {top.n.mean()*100:+6.1f}% vs rest {rest.n.mean()*100:+6.1f}% (one-class, AUC n/a)'; continue
        out+=f' | {lab} AUC big {roc_auc_score(z.BIG,z[col]):.3f} huge {roc_auc_score(z.HUGE,z[col]):.3f} top20 n{len(top):3d}: mean {top.n.mean()*100:+6.1f}% med {top.n.median()*100:+5.1f}% (rest {rest.n.mean()*100:+6.1f}% / {rest.n.median()*100:+5.1f}%) catch big {top.BIG.sum()/max(1,z.BIG.sum())*100:4.1f}% huge {top.HUGE.sum()/max(1,z.HUGE.sum())*100:4.1f}%'
    print(out)
print('\nscores (cut from SELECT half)'); 
for col in ('OS','OSa','ATRP','EXT','LTURN','VOLX','F2','F11','RES'): rep(col,col)
S.to_pickle(X+'sub1_lowpx_flips.pkl')
