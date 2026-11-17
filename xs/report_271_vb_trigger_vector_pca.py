"""Report 271 (declared before results).
(A) Does pairing PCA with the vector state help? Same PCA first-2h-shape scores as report 270 (fit on train 2019-21) PLUS vector features at 11:25: regime aligned with the first-2h drift (+1 same direction, -1 opposite), number of regime flips 09:30-11:25, signed distance of close to vector level (in R20, positive = on the regime side). Target: rest-of-day continuation cont. OLS fitted on train, R2 train|test (test = 2018+2022-26): PCA only, vector only, both.
(B) Standalone volume-building trigger (Rey's 'run the next test'). Event: new RTH extreme 10:00-15:00, >=2 earlier new-extreme bars same side that day, >=0.5 R20 from open, event-bar volume >= mean of earlier extreme-bar volumes AND >=1.3x prior-12-bar mean (VB); no rejection-wick requirement (255 required one and had n=194). Entries: E0 next open; E1 confirmation = within 6 bars first 5m close beyond the event bar's far side (short: below its low), enter next open. Stop = extreme +-1pt (E1: extreme since event). Targets 1R, 2R, and session VWAP (if >=0.5R away). 10<R<120, max 2 events per side per day, net of 1pt. Comparators: same entry without VB (placebo). Cuts: gamma sign (prior-close QQQ net GEX), vector regime opposite to the push (fade aligned) vs not. Split train 2019-21 | test 2018+2022-26. Pts per contract."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0,'xs')
from vector_pine import run_state_machine
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); b['dt']=pd.to_datetime(b.dt); b=b.set_index('dt').sort_index()
res=run_state_machine(b.assign(datetime=b.index)); b['vreg']=np.asarray(res[0]); b['vlev']=np.asarray(res[1])
rth=b.between_time('09:30','15:55'); rth=rth.assign(d=rth.index.normalize()); G=rth.groupby('d')
dd=pd.DataFrame({'o':G.open.first(),'h':G.high.max(),'l':G.low.min(),'c':G.close.last(),'n':G.size()}); dd=dd[dd.n>=70]; dd['rng']=dd.h-dd.l; dd['R20']=dd.rng.shift(1).rolling(20).mean()
gx=pd.read_csv('xs/ext/gex/QQQ.csv',parse_dates=['date']).set_index('date').sort_index(); gx.index=gx.index.normalize()
kg=np.clip(gx.index.searchsorted(dd.index.values.astype('datetime64[ns]'))-1,0,len(gx)-1); dd['gpos']=gx.net_gex.values[kg]>0
# ---- A
P=[];Y=[];V=[]
for d,g in G:
    if d not in dd.index or not np.isfinite(dd.R20[d]): continue
    a=g.iloc[:24]; rest=g.between_time('11:30','15:55')
    if len(a)<24 or len(rest)<50: continue
    o=dd.o[d];R=dd.R20[d];p=a.close.values; pe=p[-1]; sg=np.sign(pe-o) or 1
    cont=sg*(dd.c[d]-pe)/R; reg=a.vreg.values; fl=int((np.diff(reg)!=0).sum()); lev=a.vlev.values[-1]
    ali=sg*reg[-1]; dist=(reg[-1]*(pe-lev)/R) if np.isfinite(lev) else np.nan
    P.append(pd.Series((p-o)/R,name=d)); Y.append(cont); V.append((ali,fl,dist))
Pm=pd.DataFrame(P); Y=pd.Series(Y,index=Pm.index); Vf=pd.DataFrame(V,index=Pm.index,columns=['ali','flips','dist']); ok=Pm.notna().all(1)&Vf.notna().all(1)&Y.notna(); Pm,Y,Vf=Pm[ok],Y[ok],Vf[ok]
yr=Pm.index.year; tr=(yr>=2019)&(yr<=2021); te=(yr==2018)|(yr>=2022)
S=PCA(3).fit(Pm[tr]).transform(Pm); Vz=((Vf-Vf[tr].mean())/Vf[tr].std()).values
print('(A) days',tr.sum(),te.sum())
for nm,X in (('PCA 3',S),('vector only',Vz),('PCA+vector',np.c_[S,Vz])):
    m=LinearRegression().fit(X[tr],Y[tr]); pr=m.predict(X); o=[]
    for msk in (tr,te): yv=Y[msk].values; o.append(f'R2 {1-((yv-pr[msk])**2).sum()/((yv-yv.mean())**2).sum():+.3f} corr {np.corrcoef(yv,pr[msk])[0,1]:+.3f}')
    print(f'  cont ~ {nm:11s} train {o[0]} | test {o[1]}')
for lb,msk_ in (('vector aligned with drift',Vf.ali>0),('vector opposite drift',Vf.ali<0)):
    s=[]
    for msk in (tr,te): v=Y[msk&msk_.values]; s.append(f'n{len(v):4d} {v.mean():+.3f} (t{v.mean()/(v.std()/np.sqrt(len(v))):+.1f})')
    print(f'  mean cont, {lb:26s} {s[0]} | {s[1]}')
# ---- B
rows=[]
for d,g in G:
    if d not in dd.index or not np.isfinite(dd.R20[d]): continue
    o=g.open.values;h=g.high.values;l=g.low.values;c=g.close.values;v=g.volume.values.astype(float);mm=g.index.hour*60+g.index.minute;reg=g.vreg.values
    tp=(h+l+c)/3; vw=np.cumsum(tp*v)/np.maximum(np.cumsum(v),1); o0=o[0];R20=dd.R20[d]; gp=dd.gpos[d]
    rh=-np.inf; rl=np.inf; ev={1:[],-1:[]}; cnt={1:0,-1:0}
    for i in range(len(g)-8):
        for dr,new in ((-1,h[i]>rh),(1,l[i]<rl)):
            if not new: continue
            pv=list(ev[dr]); ev[dr].append(v[i])
            if not(600<=mm[i]<900) or cnt[dr]>=2 or len(pv)<2: continue
            ext=h[i] if dr==-1 else l[i]
            if abs(ext-o0)<.5*R20 or (dr==-1)!=(ext>o0): continue
            pm=v[max(0,i-12):i].mean(); rel=v[i]/pm if pm>0 else np.nan; vb=bool(v[i]>=np.mean(pv) and rel>=1.3)
            cnt[dr]+=1
            for ent in ('E0','E1'):
                if ent=='E0': j0=i+1; stopbase=ext
                else:
                    j0=None
                    for j in range(i+1,min(i+7,len(g)-2)):
                        if (dr==-1 and c[j]<l[i]) or (dr==1 and c[j]>h[i]): j0=j+1; break
                    if j0 is None: continue
                    stopbase=h[i+1:j0].max() if dr==-1 else l[i+1:j0].min()
                e=o[j0]; stop=stopbase+(1 if dr==-1 else -1); R=abs(stop-e)
                if R<10 or R>120 or (dr==-1 and stop<=e) or (dr==1 and stop>=e): continue
                out={}
                for nm,tg in (('1R',e-dr*R*1.0*-1*-1),('2R',None),('VW',None)): pass
                for nm in ('1R','2R','VW'):
                    if nm=='VW':
                        tgt=vw[j0-1]
                        if abs(e-tgt)<.5*R or (dr==-1 and tgt>=e) or (dr==1 and tgt<=e): out[nm]=np.nan; continue
                    else: tgt=e+dr*(-1)*0 + (-dr)*(1 if nm=='1R' else 2)*R*-1*-1 if False else (e-(1 if nm=='1R' else 2)*R if dr==-1 else e+(1 if nm=='1R' else 2)*R)
                    r=None
                    for j in range(j0,len(g)):
                        hs=(h[j]>=stop) if dr==-1 else (l[j]<=stop); ht=(l[j]<=tgt) if dr==-1 else (h[j]>=tgt)
                        if hs: r=-R;break
                        if ht: r=abs(e-tgt);break
                        if mm[j]>=955: r=(e-c[j]) if dr==-1 else (c[j]-e);break
                    if r is None: r=(e-c[-1]) if dr==-1 else (c[-1]-e)
                    out[nm]=r-1
                rows.append((d,ent,vb,gp,(-dr)*reg[i]>0*1 if False else (reg[i]==-dr*-1*-1*-1),out['1R'],out['2R'],out['VW'],R))
T=pd.DataFrame(rows,columns=['d','ent','vb','gpos','vopp','r1','r2','rv','R'])
# vopp: vector regime opposite to the push direction (fade aligned with vector). push dir = -dr; reg==dr means regime in fade direction
T['split']=np.where((T.d.dt.year>=2019)&(T.d.dt.year<=2021),'train','test')
def st(x): x=x.dropna(); return f'n{len(x):4d} {x.mean():+6.1f} (t{(x.mean()/(x.std()/np.sqrt(len(x)))) if len(x)>2 and x.std()>0 else np.nan:+.1f})'
print('\n(B) pts net, train | test')
for ent in ('E0','E1'):
    for lb,m in (('VB',T.vb),('no-VB placebo',~T.vb)):
        z=T[(T.ent==ent)&m]
        for col in ('r1','r2','rv'): print(f'  {ent} {lb:14s} {col}: {st(z[z.split=="train"][col])} | {st(z[z.split=="test"][col])}')
print('  --- VB cuts (E1), r2')
z=T[(T.ent=='E1')&T.vb]
for lb,m in (('gamma+',z.gpos),('gamma-',~z.gpos),('vector w/ fade',z.vopp),('vector vs fade',~z.vopp)):
    q=z[m]; print(f'  {lb:15s} {st(q[q.split=="train"].r2)} | {st(q[q.split=="test"].r2)}')
print('  --- VB cuts (E0), r2')
z=T[(T.ent=='E0')&T.vb]
for lb,m in (('gamma+',z.gpos),('gamma-',~z.gpos),('vector w/ fade',z.vopp),('vector vs fade',~z.vopp)):
    q=z[m]; print(f'  {lb:15s} {st(q[q.split=="train"].r2)} | {st(q[q.split=="test"].r2)}')
T.to_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/vb271.pkl')
