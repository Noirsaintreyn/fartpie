"""Report 218: filters on 1H flips of sub-$5 names, three years (Rey: 'daily is too big, filter from 1H'). DECLARED before results.
Data = report 210 set (Yahoo 1H 2023-12..2026-11, 299 survivor names; flips entry >= 2024-02, entry < $5, split trades dropped; net 100bp). Every 1H bull flip gets two outcomes: R1H (exit open after the 1H bear flip) and RHY (hybrid: once the daily vector is bull at a daily close, hold to the daily bear flip; a 1H bear flip before that exits).
Four filters only, each with a mechanism, measured at the flip bar with completed bars:
 F1 DSTATE: daily vector state at the prior daily close (bull = with the daily trend, bear = early reversal). Two-sided (report 214 saw bear-state days do better; trend logic says bull).
 F2 ASY1H: price-response asymmetry, same formula as report 213 on the last 140 completed 1H bars (= 20 sessions): log(up-bar return per dollar / down-bar return per dollar). Sign +.
 F3 RISK: (flip close - ratchet level) / close = distance to the vector's own stop. Sign: smaller is better (cheaper failed probes).
 F4 VOLX: flip-bar volume / mean of the previous 100 bars. Sign +.
Cuts = H1 terciles (H1 2024-02..2025-06, H2 2025-07..2026-11). PASS = same sign in both halves and day-clustered |t| >= 2.7 (Bonferroni, 8 tests) on top-minus-bottom tercile of the net return (winsorized at the 99th pct); also shown: mean, ex-best-3, median, win, share >= +30%. Reference row: all flips."""
import pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from vector_pine import run_state_machine
E='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; SP=E[:-4]
P=pickle.load(open(E+'lowpx_1h_3y.pkl','rb')); D=pd.read_pickle(E+'wide_daily_ohlcv.pkl'); S0=pd.Timestamp('2024-02-01'); MID=pd.Timestamp('2025-07-01'); rows=[]
for tk,h in P.items():
    x=h.copy(); x.columns=['open','high','low','close','volume']; x.index=x.index.tz_convert('America/New_York').tz_localize(None); x=x[~x.index.duplicated()].dropna(); m=x.index.hour*60+x.index.minute; x=x[(m>=570)&(m<960)]
    if len(x)<1500 or tk not in D: continue
    d=D[tk]; d=d[~d.index.duplicated()]; d=d[d.index>='2022-06-01'][['open','high','low','close','volume']].dropna()
    res=run_state_machine(x.assign(datetime=x.index)); r1=np.asarray(res[0]); lev=np.asarray(res[1],float); rD=pd.Series(np.asarray(run_state_machine(d.assign(datetime=d.index))[0]),index=d.index)
    day=x.index.normalize(); o,c,v=x.open.values,x.close.values,x.volume.values.astype(float); n=len(x); last=np.r_[day[1:]!=day[:-1],True]; rd=rD.reindex(day,method='ffill').values; rdp=rD.shift(1).reindex(day,method='ffill').values; ret=np.r_[0,c[1:]/c[:-1]-1]; dv=c*v
    for i in range(141,n-2):
        if not (r1[i]==1 and r1[i-1]!=1): continue
        e=i+1
        if x.index[e]<S0 or o[e]>=5: continue
        j=e
        while j<n and r1[j]!=-1: j+=1
        x1=min(j+1,n-1); p1=c[n-1] if j>=n-1 else o[x1]
        conf=False; xi=None; j=e
        while j<n-1:
            if last[j]:
                if rd[j]==1: conf=True
                elif conf and rd[j]==-1: xi=j+1; break
            if not conf and r1[j]==-1: xi=j+1; break
            j+=1
        x2=n-1 if xi is None else xi; p2=c[n-1] if xi is None else o[x2]; xm=max(x1,x2); g=o[e+1:xm+1]/c[e:xm] if xm>e else np.array([1.])
        if len(g) and (g.max()>2 or g.min()<.5): continue
        rr=ret[i-139:i+1]; dd=dv[i-139:i+1]; u=rr>0; dn=rr<0; asy=np.log((rr[u].sum()/dd[u].sum())/(abs(rr[dn]).sum()/dd[dn].sum())) if u.sum()>=10 and dn.sum()>=10 and dd[u].sum()>0 and dd[dn].sum()>0 else np.nan
        rows.append(dict(tk=tk,entry=x.index[e],R1H=p1/o[e]-1-.01,RHY=p2/o[e]-1-.01,DSTATE=rdp[e],ASY1H=asy,RISK=(c[i]-lev[i])/c[i] if lev[i]==lev[i] else np.nan,VOLX=v[i]/max(v[i-100:i].mean(),1)))
F=pd.DataFrame(rows); F['half']=np.where(F.entry<MID,'H1','H2'); print('1H flips',len(F),F.groupby('half').size().to_dict(),'| daily state at flip: bull',int((F.DSTATE==1).sum()),'bear',int((F.DSTATE==-1).sum()))
def st(s,col):
    r=s[col]; rs=r.sort_values(); return f'n{len(s):5d} mean {r.mean()*100:5.2f}% x3 {rs.iloc[:-3].mean()*100:5.2f}% med {r.median()*100:5.2f}% win {np.mean(r>0)*100:3.0f}% >=+30% {np.mean(r>=.3)*100:4.1f}%'
def tt(a,b,col):
    cap=np.quantile(np.r_[a[col].values,b[col].values],.99); A=a.assign(w=np.minimum(a[col],cap)).groupby(a.entry.dt.normalize()).w.mean(); B=b.assign(w=np.minimum(b[col],cap)).groupby(b.entry.dt.normalize()).w.mean(); return (A.mean()-B.mean())/np.sqrt(A.var(ddof=1)/len(A)+B.var(ddof=1)/len(B))
for col in ('R1H','RHY'):
    print(f'\n=== outcome {col} (net 100bp)'); 
    for hf in ('H1','H2'): print(f'  ALL {hf}: '+st(F[F.half==hf],col))
    for hf in ('H1','H2'):
        s=F[F.half==hf]; a=s[s.DSTATE==1]; b=s[s.DSTATE==-1]; print(f'  F1 DSTATE {hf}: daily BULL {st(a,col)} | daily BEAR {st(b,col)} | t(bull-bear) {tt(a,b,col):+.1f}')
    for f,sgn in (('ASY1H','+'),('RISK','-'),('VOLX','+')):
        q=F[F.half=='H1'][f].quantile([1/3,2/3]).values
        for hf in ('H1','H2'):
            s=F[(F.half==hf)&F[f].notna()]; lo=s[s[f]<=q[0]]; hi=s[s[f]>=q[1]]; print(f'  {f} ({sgn}) {hf}: low third {st(lo,col)} | high third {st(hi,col)} | t(high-low) {tt(hi,lo,col):+.1f}')
F.to_pickle(SP+'flips1h_feats218.pkl')
