"""1H 1:1 vector trades for the Lucid sim (report 231). Entry = open of the 5m bar after the 1H flip bar closes; stop = vector level, target = same distance; flat 15:55; cost 1 pt; one trade at a time; entries 09:35-15:30 ET. Returns day, m, dir, pts, mae (<=0), neg (prior-close QQQ net GEX < 0), low3."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/claude/fartpie/xs'); from vector_pine import run_state_machine
def vx(k=1.0, rmin=3.0, be=None):
    b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); b=b[b.dt>='2017-06-01'].reset_index(drop=True); o,h,l,c=[b[x].values for x in ('open','high','low','close')]; n=len(b)
    m=(b.dt.dt.hour*60+b.dt.dt.minute).values; day=b.dt.dt.normalize().values; t5=b.dt.values
    G=pd.read_csv('/home/claude/fartpie/xs/ext/gex/QQQ.csv',parse_dates=['date']).set_index('date').net_gex.sort_index(); G.index=G.index.normalize(); GR=G.rolling(252,min_periods=120).rank(pct=True)
    g=b.set_index('dt').resample('60min',label='right',closed='left').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna(); res=run_state_machine(g.assign(datetime=g.index)); reg=np.asarray(res[0]); lev=np.asarray(res[1],float); ends=g.index.values
    sig=[]
    for i in range(1,len(g)):
        s=1 if (reg[i]==1 and reg[i-1]!=1) else -1 if (reg[i]==-1 and reg[i-1]!=-1) else 0
        if s==0 or not np.isfinite(lev[i]): continue
        e=np.searchsorted(t5,ends[i])
        if e>=n or t5[e]!=ends[i]: continue
        R=abs(o[e]-lev[i])
        if R>=rmin: sig.append((e,s,R))
    out=[]; last=-1
    for e,s,R in sig:
        if e<=last or e>=n-2 or not (575<=m[e]<=930) or day[e]<np.datetime64('2018-01-01'): continue
        pe=o[e]; tg=k*R; res_=None; j=e; mae=0.0; sd=R
        while j<n:
            adv=(pe-l[j]) if s==1 else (h[j]-pe); fav=(h[j]-pe) if s==1 else (pe-l[j])
            if adv>=sd: res_=-sd; mae=max(mae,sd); break
            if fav>=tg: res_=tg; mae=max(mae,adv); break
            mae=max(mae,adv)
            if m[j]>=955 or day[j]!=day[e]: res_=s*(c[j]-pe); break
            if be is not None and fav>=be*R: sd=0.0
            j+=1
        if res_ is None: res_=s*(c[min(j,n-1)]-pe)
        d=pd.Timestamp(day[e]); kk=G.index.searchsorted(d)-1; a=G.iloc[kk] if kk>=0 else np.nan; r=GR.iloc[kk] if kk>=0 else np.nan
        out.append((d,float(m[e]),s,res_-1.0,-mae,R,a<0 if np.isfinite(a) else False,(r<=1/3) if np.isfinite(r) else False)); last=j
    return pd.DataFrame(out,columns=['day','m','dir','pts','mae','R','neg','low3'])
if __name__=='__main__':
    d=vx(); print(len(d), d.groupby(d.day.dt.year).pts.agg(['count','mean']).round(1).T.to_string()); print(d.neg.mean(), d.low3.mean())
