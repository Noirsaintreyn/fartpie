"""Report 228: 1m and 5m vector flips on NQ, filtered by higher-timeframe alignment (Rey: 'even 1m or 5m vectors if we just filter those for a small consistent edge'). DECLARED before results.
Data: NQ 1m (ET = timestamp - 6h) and 5m bars, 2018-01..2026-07, RTH entries 09:35-15:25, entry at the open of the bar after the flip bar, one position at a time, flat by 15:55. Vector = the same state machine on the bar series of that timeframe (full session history).
Filters: NONE; V15 = the 15m vector agrees with the flip direction; V15+V60 = both the 15m and 1H vectors agree. (States taken from higher-timeframe bars completed at the flip bar's close.)
Exits: VEC = hold until the same-timeframe vector flips against the trade; A1/2 = target 1 x ATR14, stop 2 x ATR14 of that timeframe; A2/2 = target 2 x ATR, stop 2 x ATR; time stop 60 minutes for the ATR exits. Stop wins a bar that touches both. Cost 1.0 pt round trip (shown gross and net).
Split train 2018-21 | test 2022-26. PASS = net mean same sign both eras, day-clustered |t| >= 2, n >= 30. 18 cells (Bonferroni |t| >= 3.0)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from vector_pine import run_state_machine
b5=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); b5=b5[b5.dt>='2017-06-01'].reset_index(drop=True)
d1=pd.read_csv('/home/claude/nqdrive/NQ_1m_ETH.csv',parse_dates=['timestamp']); d1['dt']=d1.timestamp-pd.Timedelta(hours=6); d1=d1.sort_values('dt').drop_duplicates('dt'); d1=d1[d1.dt>='2017-06-01'][['dt','open','high','low','close','volume']].reset_index(drop=True)
def htf(rule):
    g=b5.set_index('dt').resample(rule,label='right',closed='left').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna(); return pd.Series(np.asarray(run_state_machine(g.assign(datetime=g.index))[0]),index=g.index)
v15=htf('15min'); v60=htf('60min')
def study(b,mins,tag):
    reg=np.asarray(run_state_machine(b.assign(datetime=b.dt))[0]); ce=b.dt+pd.Timedelta(minutes=mins); V15=v15.reindex(ce,method='ffill').values; V60=v60.reindex(ce,method='ffill').values
    o,h,l,c=[b[k].values for k in ('open','high','low','close')]; pc=np.r_[c[0],c[:-1]]; atr=pd.Series(np.maximum(h-l,np.maximum(abs(h-pc),abs(l-pc)))).rolling(14).mean().values
    m=(b.dt.dt.hour*60+b.dt.dt.minute).values; day=b.dt.dt.normalize().values; yr=b.dt.dt.year.values; n=len(b); flip=np.zeros(n,int); flip[1:]=np.where((reg[1:]==1)&(reg[:-1]!=1),1,np.where((reg[1:]==-1)&(reg[:-1]!=-1),-1,0)); ok=(m>=575)&(m<=925)&(yr>=2018); tb=60//mins
    nd={'train':len(np.unique(day[ok&(yr<2022)])),'test':len(np.unique(day[ok&(yr>=2022)]))}
    for fl in ('NONE','V15','V15+V60'):
        d=np.where(ok,flip,0)
        if fl!='NONE': d=np.where(d==V15,d,0)
        if fl=='V15+V60': d=np.where(d==V60,d,0)
        for ex in ('VEC','A1/2','A2/2'):
            out=[]; i=0
            while i<n-3:
                if d[i]==0: i+=1; continue
                e=i+1; s=d[i]; pe=o[e]
                if day[e]!=day[i] or not np.isfinite(atr[i]): i+=1; continue
                tg,st=(1e9,1e9) if ex=='VEC' else ((atr[i],2*atr[i]) if ex=='A1/2' else (2*atr[i],2*atr[i])); res=None; j=e
                while j<n:
                    adv=(pe-l[j]) if s==1 else (h[j]-pe); fav=(h[j]-pe) if s==1 else (pe-l[j])
                    if adv>=st: res=-st; break
                    if fav>=tg: res=tg; break
                    if m[j]>=955 or day[j]!=day[e]: res=s*(c[j]-pe); break
                    if ex=='VEC' and reg[j]==-s: res=s*((o[j+1] if j+1<n and day[j+1]==day[e] else c[j])-pe); break
                    if ex!='VEC' and j-e+1>=tb: res=s*(c[j]-pe); break
                    j+=1
                if res is None: res=s*(c[min(j,n-1)]-pe)
                out.append((day[e],res)); i=j+1
            R=pd.DataFrame(out,columns=['day','g']); cells=[]
            for nm,mk in (('train',R.day<np.datetime64('2022-01-01')),('test',R.day>=np.datetime64('2022-01-01'))):
                x=R[mk]; net=x.g-1.0; dd=net.groupby(x.day).sum(); t=dd.mean()/(dd.std(ddof=1)/np.sqrt(len(dd))); cells.append(f'{nm}: n{len(x):6d} ({len(x)/nd[nm]:4.1f}/day) win {np.mean(net>0)*100:3.0f}% gross {x.g.mean():+5.2f} net {net.mean():+5.2f} pts t{t:+5.1f} | day net {dd.sum()/nd[nm]:+6.1f} pts')
            print(f'{tag} flips {fl:8s}{ex:5s}',' || '.join(cells),flush=True)
study(b5,5,'5m'); study(d1,1,'1m')
