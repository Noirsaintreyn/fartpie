"""report 168: IFVG take-profit comparison (Rey: VWAP or ICT liquidity?). DECLARED BEFORE RESULTS. Same IFVG trades (ext/uni_ifvg.pkl: entry, stop = entry -+ risk, fill bar, dir), same stop, flat 15:55, cost .5; only the target changes. 1R/2R/3R fixed; VWAP = RTH VWAP (09:30 cumulative hlc3 x volume) at the fill bar, used only if >= 1R away in the profit direction else 2R fallback; VWAPonly = only trades where VWAP is >=1R away; LIQ = nearest of {prior-day high/low, overnight (18:00-09:29) high/low, today's RTH high/low so far} that is >= 1R away in the profit direction, else 2R; LIQ_FAR = the FARTHEST of those up to 4R, else 2R; LIQ2 = nearest of those >= 1.5R. A | B, PASS = same sign |t|>=2 n>=30."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; COST=.5; BL=pd.Timestamp('2022-01-01')
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet').sort_values('dt').drop_duplicates('dt').reset_index(drop=True); b['d']=b.dt.dt.normalize(); b['m']=b.dt.dt.hour*60+b.dt.dt.minute
by=dict(tuple(b.groupby('d'))); days=sorted(by); RTd=b[(b.m>=570)&(b.m<960)].groupby('d').agg(h=('high','max'),l=('low','min'))
t=pd.DataFrame(pd.read_pickle(S+'ext/uni_ifvg.pkl')); t=t[t.model=='IFVG'].copy(); t['d']=pd.to_datetime(t.d)
def sim(x,f,ent,stop,dr,tgt):
    h,l,c=x.high.values,x.low.values,x.close.values; xe=np.where(x.m.values==955)[0][-1]
    for q in range(f,xe+1):
        if (l[q]<=stop) if dr>0 else (h[q]>=stop): return -abs(ent-stop)-COST
        if (h[q]>=tgt) if dr>0 else (l[q]<=tgt): return abs(tgt-ent)-COST
    return (c[xe]-ent)*dr-COST
R=[]
for _,r in t.iterrows():
    D=r.d
    if D not in by or D not in RTd.index: continue
    i=days.index(D); 
    if i==0 or days[i-1] not in RTd.index: continue
    p_=by[days[i-1]]; p_=p_[p_.m>=1080]; x=pd.concat([p_,by[D][by[D].m<=955]]).reset_index(drop=True); mm=x.m.values; ms=np.where(mm>=1080,mm-1440,mm)
    fi=np.where(ms==int(r.m))[0]; 
    if not len(fi): continue
    f=int(fi[0]); i930=int(np.where(ms==570)[0][0]); dr=int(r.dr); ent=r.ent; rk=r.risk; stop=ent-dr*rk
    rt=x.iloc[i930:f+1]; vw=((rt.high+rt.low+rt.close)/3*rt.volume).sum()/max(rt.volume.sum(),1)
    on=x.iloc[:i930]; lv=[RTd.h[days[i-1]],RTd.l[days[i-1]],on.high.max(),on.low.min(),rt.high.max(),rt.low.min()]
    prof=[(v-ent)*dr for v in lv]; far=[v for v,dd in zip(lv,prof) if dd>=rk]; ok15=[v for v,dd in zip(lv,prof) if dd>=1.5*rk]
    row=dict(d=D)
    for nm,k in (('R1',1),('R2',2),('R3',3)): row[nm]=sim(x,f,ent,stop,dr,ent+k*rk*dr)
    vd=(vw-ent)*dr; row['VWAP']=sim(x,f,ent,stop,dr,vw if vd>=rk else ent+2*rk*dr); row['VWAPonly']=sim(x,f,ent,stop,dr,vw) if vd>=rk else np.nan
    row['LIQ']=sim(x,f,ent,stop,dr,min(far,key=lambda v:(v-ent)*dr) if far else ent+2*rk*dr)
    cap=[v for v in far if (v-ent)*dr<=4*rk]; row['LIQ_FAR']=sim(x,f,ent,stop,dr,max(cap,key=lambda v:(v-ent)*dr) if cap else ent+2*rk*dr)
    row['LIQ2']=sim(x,f,ent,stop,dr,min(ok15,key=lambda v:(v-ent)*dr) if ok15 else ent+2*rk*dr); row['n_liq']=len(far); row['vw_ok']=vd>=rk; R.append(row)
T=pd.DataFrame(R); T.to_pickle(S+'tp168.pkl'); T['era']=np.where(T.d<BL,'A','B')
def st(s):
    s=s.dropna(); n=len(s); 
    return f'n{n:4d} {s.mean():+6.2f} t{(s.mean()/(s.std()/np.sqrt(n)) if n>2 and s.std()>0 else 0):+5.1f} w{(s>0).mean()*100:3.0f}%' if n else 'n   0'
for c in ('R1','R2','R3','VWAP','VWAPonly','LIQ','LIQ_FAR','LIQ2'):
    a,bb=T[T.era=='A'][c],T[T.era=='B'][c]; ok=len(a.dropna())>=30 and len(bb.dropna())>=30 and np.sign(a.mean())==np.sign(bb.mean()) and abs(a.mean()/(a.std()/np.sqrt(a.count())))>=2 and abs(bb.mean()/(bb.std()/np.sqrt(bb.count())))>=2
    print(f'{c:9s} A {st(a)} | B {st(bb)} {"PASS" if ok else ""}')
print('VWAP >=1R away on',T.vw_ok.mean().round(2),'of trades; LIQ level >=1R available on',(T.n_liq>0).mean().round(2))
