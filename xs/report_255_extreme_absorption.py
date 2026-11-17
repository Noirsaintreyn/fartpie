"""Report 255: TRADE THE EXTREME ITSELF (no day-type prediction). DECLARED before results (Rey Nov 17: volume building on later pushes, absorption at extremes, CVD).
Event: 5m bar after 10:00 and before 15:00 that makes a new RTH session high (short fade) or low (long fade), price >= .5 ATR14 (daily RTH range) from the 09:30 open, and REJECTS: closes back with >= 50% of its range as the wick on the push side. Entry next 5m open; stop = bar extreme +/-1 pt; R = |stop-entry| >= 10 pts (cap 120); target 1R and 2R; stop first if both in one bar; flat 15:55; cost 1 pt; max 2 per day per direction.
Variants: V0 rejection only. V1 + volume building (event-bar volume >= mean volume of the earlier new-extreme bars the same day, >=2 of them, AND >= 1.3x prior-12-bar mean). V2 + absorption (est. delta of the event bar + 2 before, in the push direction, > 0 while price rejected; with volume >= 1.3x). V3 + CVD divergence (price makes a new extreme but session cumulative est. delta at the event bar is below its value at the earlier new-extreme bars the same day, needs >= 1 earlier). V4 = V1 + V3. V5 = V1 + V2 + V3.
Est. delta = vol x (2c-h-l)/(h-l). Real tick CVD (buy_vol - sell_vol, MNQ 1m, Dec 2024-Nov 2025) is used for a check: V3 on real CVD vs est CVD in that window.
Split: train 2019-21, test 2018 + 2022-26."""
import numpy as np, pandas as pd, pickle
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); b['dt']=pd.to_datetime(b.dt); b=b.set_index('dt').sort_index().between_time('09:30','15:55'); b['d']=b.index.normalize()
D=b.groupby('d'); dd=pd.DataFrame({'h':D.high.max(),'l':D.low.min(),'n':D.size()}); dd=dd[dd.n>=70]; dd['atr']=(dd.h-dd.l).rolling(14).mean().shift(1)
# real CVD on 5m (ET)
rf=pd.read_parquet('/mnt/user-data/uploads/fartpie/mnq_orderflow_1min.parquet'); rf['datetime']=pd.to_datetime(rf.datetime); rf=rf.set_index('datetime'); rf.index=rf.index.tz_convert('America/New_York').tz_localize(None)
rc=rf.signed_vol.resample('5min',label='left',closed='left').sum()
out=[]
for d,g in D:
    if d not in dd.index or not np.isfinite(dd.atr[d]): continue
    a=dd.atr[d]; o=g.open.values;h=g.high.values;l=g.low.values;c=g.close.values;v=g.volume.values.astype(float);t=g.index; mm=t.hour*60+t.minute
    dl=v*(2*c-h-l)/np.maximum(h-l,1e-9); cum=np.cumsum(dl); rcd=rc.reindex(t).values; rcum=np.nancumsum(rcd) if np.isfinite(rcd).any() else None
    o0=o[0]; ext_vols={1:[],-1:[]}; ext_cum={1:[],-1:[]}; ext_rc={1:[],-1:[]}; cnt={1:0,-1:0}; rh=-np.inf; rl=np.inf; cmax=-np.inf; cmin=np.inf; rcmax=-np.inf; rcmin=np.inf
    for i in range(len(g)-2):
        newh=h[i]>rh; newl=l[i]<rl; pcmax,pcmin=cmax,cmin
        prh,prl=rh,rl
        if newh: rh=h[i]
        if newl: rl=l[i]
        cmax=max(cmax,cum[i]); cmin=min(cmin,cum[i])
        prcmax,prcmin=rcmax,rcmin
        if rcum is not None: rcmax=max(rcmax,rcum[i]); rcmin=min(rcmin,rcum[i])
        for dr,new in ((-1,newh),(1,newl)):   # dr=-1 short fade at new high
            if not new: continue
            pv=ext_vols[dr][:]; pcs=ext_cum[dr][:]; prs=ext_rc[dr][:]; ext_vols[dr].append(v[i]); ext_cum[dr].append(cum[i]); ext_rc[dr].append(rcum[i] if rcum is not None else np.nan)
            if not (600<=mm[i]<900) or cnt[dr]>=2: continue
            rng=h[i]-l[i]
            if rng<=0: continue
            if dr==-1:
                if h[i]-o0<.5*a or (h[i]-c[i])<.5*rng: continue
                stop=h[i]+1
            else:
                if o0-l[i]<.5*a or (c[i]-l[i])<.5*rng: continue
                stop=l[i]-1
            e=o[i+1]; R=abs(stop-e)
            if R<10 or R>120: continue
            cnt[dr]+=1
            pm=v[max(0,i-12):i].mean() if i>0 else np.nan
            rel=v[i]/pm if pm>0 else np.nan
            vb=(len(pv)>=2 and v[i]>=np.mean(pv) and rel>=1.3)
            dsum=dl[max(0,i-2):i+1].sum(); ab=(-dr*dsum>0 and rel>=1.3)   # buyers aggressive into a high that failed
            cd=(len(pcs)>=1 and ((cum[i]<max(pcs)) if dr==-1 else (cum[i]>min(pcs))))
            rcd_=np.nan
            if rcum is not None and len(prs)>=1 and np.isfinite(prs).all(): rcd_=((rcum[i]<max(prs)) if dr==-1 else (rcum[i]>min(prs)))
            r1=r2=None
            for tgm,name in ((1,'r1'),(2,'r2')):
                tg=e+(-dr)*0+( -dr*-1 if False else 0)
                tgt=e+(-1 if dr==-1 else 1)*tgm*R; res=None
                for j in range(i+1,len(g)):
                    hit_s=(h[j]>=stop) if dr==-1 else (l[j]<=stop); hit_t=(l[j]<=tgt) if dr==-1 else (h[j]>=tgt)
                    if hit_s: res=-R;break
                    if hit_t: res=tgm*R;break
                    if mm[j]>=955: res=(e-c[j]) if dr==-1 else (c[j]-e);break
                if res is None: res=(e-c[-1]) if dr==-1 else (c[-1]-e)
                if name=='r1': r1=res-1
                else: r2=res-1
            out.append((d,dr,R,r1,r2,vb,ab,bool(cd),rcd_))
T=pd.DataFrame(out,columns=['d','dr','R','r1','r2','vb','ab','cd','rcd']); T['yr']=T.d.dt.year; T['split']=np.where((T.yr>=2019)&(T.yr<=2021),'train','test')
VAR={'V0 reject only':np.ones(len(T),bool),'V1 +vol building':T.vb.values,'V2 +absorption':T.ab.values,'V3 +CVD divergence':T.cd.values,'V4 V1+V3':(T.vb&T.cd).values,'V5 V1+V2+V3':(T.vb&T.ab&T.cd).values,'V6 V2+V3':(T.ab&T.cd).values}
def tt(x): return x.mean()/(x.std()/np.sqrt(len(x))) if len(x)>2 and x.std()>0 else np.nan
yrs=8.0
for nm,m in VAR.items():
    g=T[m]; print(f'{nm:20s} n {len(g)} ({len(g)/yrs:.0f}/yr)',end='')
    for sp in ('train','test'):
        x=g[g.split==sp]; print(f' | {sp} n{len(x)} 1R {x.r1.mean():+5.1f} (t{tt(x.r1):+.1f}) 2R {x.r2.mean():+5.1f} (t{tt(x.r2):+.1f})',end='')
    print()
print('by year V4 (1R):',T[VAR['V4 V1+V3']].groupby('yr').r1.agg(['count','mean']).round(1).T.to_dict())
print('by year V0 (1R):',T.groupby('yr').r1.agg(['count','mean']).round(1).T.to_dict())
x=T[T.rcd.notna()].copy(); x['rcd']=x.rcd.astype(bool); print('real-CVD window n',len(x),'est vs real divergence agree',round((x.cd==x.rcd).mean(),2))
for nm,col in (('est-CVD div','cd'),('real-CVD div','rcd')):
    for on in (True,False):
        y=x[x[col]==on]; print(f'  {nm} {on}: n {len(y)} 1R {y.r1.mean():+.1f} 2R {y.r2.mean():+.1f}')
pickle.dump(T,open('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext255.pkl','wb'))
