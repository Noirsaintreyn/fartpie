"""DAY-TYPE: (1) how early can we tell a rotation day from a trend day, (2) mean-reversion / volume-profile fades gated by that call.
Label ROT = full-day RTH efficiency |close-open|/range < 0.30 (rotation / balance), TREND = efficiency >= 0.30. IDLE = core legs flat (separate label).
Decision times: pre-open, 09:45, 10:00, 10:30, 11:00, 12:00, 13:00 (features use only bars before that time). Walk-forward L2 logistic, refit each year from 2018 data (first test year 2019).
Gated trades: sweep-reclaim fades (model A), prior-range-edge fade (D) and prior-day volume-profile VA-edge fade to POC (V) taken only when P(rot) at the 10:00 call >= cut, entries after 10:05."""
import numpy as np, pandas as pd, pickle, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet').sort_values('dt').drop_duplicates('dt').reset_index(drop=True)
mn=b.dt.dt.hour*60+b.dt.dt.minute; b['d']=b.dt.dt.normalize(); rth=b[(mn>=570)&(mn<960)].copy()
idl=pd.read_pickle(SC+'idle_state.pkl'); idle=~idl['trade'].astype(bool); IS=idl
D=rth.groupby('d').agg(o=('open','first'),h=('high','max'),l=('low','min'),c=('close','last'),v=('volume','sum')); D['rng']=D.h-D.l; D['eff']=(D.c-D.o).abs()/D.rng
D['adr']=D.rng.rolling(20).mean().shift(); D['pdh']=D.h.shift(); D['pdl']=D.l.shift(); D['peff']=D.eff.shift(); D['pv']=D.v.shift(); D['rr']=D.rng.shift()/D.adr
G={d:g.reset_index(drop=True) for d,g in rth.groupby('d')}; days=list(D.index)
# prior-day volume profile -> POC, VAH, VAL
def vp(g,bin_=2.0):
    tp=((g.high+g.low+g.close)/3).values; v=g.volume.values; lo=np.floor(tp.min()/bin_)*bin_; k=((tp-lo)/bin_).astype(int); h=np.bincount(k,weights=v); poc=int(h.argmax()); tot=h.sum(); a=b_=poc; acc=h[poc]
    while acc<.7*tot and (a>0 or b_<len(h)-1):
        up=h[b_+1] if b_<len(h)-1 else -1; dn=h[a-1] if a>0 else -1
        if up>=dn: b_+=1; acc+=up
        else: a-=1; acc+=dn
    return lo+poc*bin_+bin_/2, lo+(b_+1)*bin_, lo+a*bin_
VP={days[i]:vp(G[days[i-1]]) for i in range(1,len(days)) if days[i-1] in G}
TS=[0,15,30,60,90,150,210]            # minutes after 09:30; 0 = pre-open
rows=[]
for i,d in enumerate(days):
    if i<21 or d not in G or d not in VP or not D.adr[d]==D.adr[d]: continue
    g=G[d]; adr=D.adr[d]; pdh,pdl=D.pdh[d],D.pdl[d]; poc,vah,val=VP[d]; op=g.open.iloc[0]; m_=(g.dt.dt.hour*60+g.dt.dt.minute-570).values
    base=dict(day=d,rot=float(D.eff[d]<.30),idle=float(idle.get(d,np.nan)),gap=(op-D.pdh[d]*0-b[b.d==d].open.iloc[0]*0) if False else 0)
    pc=D.c.shift()[d]; base.update(gap_rel=abs(op-pc)/adr,ins=float(pdl<op<pdh),peff=D.peff[d],rr=D.rr[d],opos=(op-pdl)/(pdh-pdl),dpoc=abs(op-poc)/adr,inva=float(val<=op<=vah),varng=(vah-val)/adr)
    for T in TS:
        x=g[m_<T] if T>0 else g.iloc[:0]; f=dict(base)
        if len(x)>=2:
            hh,ll,cc=x.high.max(),x.low.min(),x.close.iloc[-1]; rg=max(hh-ll,1e-6); f.update(rng=rg/adr,eff=abs(cc-op)/rg,disp=(cc-op)/adr*np.sign(op-pc if op!=pc else 1)*0+abs(cc-op)/adr,
                pos=(cc-pdl)/(pdh-pdl),tch=float(hh>=pdh)+float(ll<=pdl),cross=float(((x.close.values[1:]>op)!=(x.close.values[:-1]>op)).sum()),vpace=x.volume.sum()/D.pv[d],incl=float(val<=cc<=vah),dpoc_t=abs(cc-poc)/adr)
        else: f.update(rng=0,eff=0,disp=0,pos=base['opos'],tch=0,cross=0,vpace=0,incl=base['inva'],dpoc_t=base['dpoc'])
        f['T']=T; rows.append(f)
F=pd.DataFrame(rows).dropna(subset=['peff','rr','gap_rel']); F['yr']=pd.to_datetime(F.day).dt.year
cols=['gap_rel','ins','peff','rr','opos','dpoc','inva','varng','rng','eff','disp','pos','tch','cross','vpace','incl','dpoc_t']
PROB={}; PROBI={}
print('=== DAY-TYPE call: walk-forward AUC / accuracy by decision time (label ROT = efficiency<0.30; base rate = share ROT)')
for lab in ('rot','idle'):
    print(f'-- label {lab}')
    for T in TS:
        z=F[F['T']==T].dropna(subset=[lab]); pr=pd.Series(np.nan,index=z.index)
        for y in range(2019,2027):
            tr=z[z.yr<y]; te=z[z.yr==y]
            if len(te)==0 or len(tr)<200: continue
            sc=StandardScaler().fit(tr[cols]); m=LogisticRegression(C=.3,max_iter=500).fit(sc.transform(tr[cols]),tr[lab]); pr[te.index]=m.predict_proba(sc.transform(te[cols]))[:,1]
        z=z.assign(p=pr).dropna(subset=['p']); 
        if lab=='rot': PROB[T]=z.set_index('day').p
        if lab=='idle': PROBI[T]=z.set_index('day').p
        out=[]
        for e,mm in (('19-21',z.yr<=2021),('22-26',z.yr>=2022)):
            q=z[mm]; hi=q[(q.p>=.7)|(q.p<=.3)]; out.append(f'{e}: AUC {roc_auc_score(q[lab],q.p):.3f} acc {((q.p>.5)==q[lab]).mean():.0%} (base {max(q[lab].mean(),1-q[lab].mean()):.0%}) conf>=70%: {len(hi)/len(q):.0%} of days, right {((hi.p>.5)==hi[lab]).mean():.0%}')
        print(f'  T=+{T:3d}m  '+' | '.join(out))
# ---- gated fades
Tm=pd.read_pickle(SC+'mr_trades.pkl'); Tm=Tm[Tm.m.isin(['A','D'])].copy()
V=[]
for d in days:
    if d not in VP or d not in G or not D.adr.get(d)==D.adr.get(d): continue
    g=G[d]; adr=D.adr[d]; poc,vah,val=VP[d]; hi=g.high.values; lo=g.low.values; m_=(g.dt.dt.hour*60+g.dt.dt.minute-570).values
    for i in range(7,len(g)-3):
        if m_[i]>330: break
        if hi[i]>=vah-.03*adr or lo[i]<=val+.03*adr:
            dr=-1 if hi[i]>=vah-.03*adr else 1; e=g.open.iloc[i+1]; edge=vah if dr<0 else val; st=edge+(.15*adr if dr<0 else -.15*adr)
            if ((dr<0 and poc<e<st) or (dr>0 and st<e<poc)) and abs(e-edge)<.12*adr:
                n=min(len(g),i+1+24); res=None
                for j in range(i+1,n):
                    if dr>0:
                        if lo[j]<=st: res=st-e-.5;break
                        if hi[j]>=poc: res=poc-e-.5;break
                    else:
                        if hi[j]>=st: res=e-st-.5;break
                        if lo[j]<=poc: res=e-poc-.5;break
                if res is None: res=(g.close.iloc[n-1]-e)*dr-.5
                V.append(dict(day=d,m='V',sub='VA-edge',ex='POC',t=g.dt.iloc[i+1],dir=dr,pts=res,R=abs(e-st)))
            break
V=pd.DataFrame(V); Tm=pd.concat([Tm[['day','m','sub','ex','t','dir','pts','R']],V],ignore_index=True); Tm['yr']=pd.to_datetime(Tm.day).dt.year
Tm=Tm[Tm.t.dt.hour*60+Tm.t.dt.minute>=605]              # entries after 10:05 so the 10:00 call is known
p10=PROB[30] if False else PROB[30]                      # T=+30m = 10:00 call
Tm['p']=Tm.day.map(p10); Tm['pi']=Tm.day.map(PROBI[30]); Tm=Tm.dropna(subset=['p','pi'])
Tm['era']=np.where(Tm.yr<=2021,'19-21','22-26')
def ln(x): return 'n %4d' % len(x) if len(x)<20 else f'n {len(x):4d} avg {x.pts.mean():+6.2f} win {(x.pts>0).mean():.0%} t {x.pts.mean()/(x.pts.std()/len(x)**.5):+.1f}'
QC=[0.0]
def grid(col):
    qs=Tm[col].quantile([.5,.67]).tolist(); return [(f'all',0.0),(f'top50% {col}>={qs[0]:.2f}',qs[0]),(f'top33% {col}>={qs[1]:.2f}',qs[1])]
for col in ('p','pi'):
    print(f'\n=== fades after 10:05 gated by {"P(rotation)" if col=="p" else "P(idle)"} at the 10:00 call (rank cuts)')
    for (m,sub,ex),x in Tm.groupby(['m','sub','ex']):
        if m=='A' and (sub not in ('PDL','ONL','PDH','ONH') or ex=='2R'): continue
        for nm,cut in grid(col): print(f'{m} {sub:7s} {ex:4s} {nm:28s}: '+' || '.join(f'{e} {ln(x[(x.era==e)&(x[col]>=cut)])}' for e in ('19-21','22-26')))
    print(f'--- pooled, first signal per day ({col})')
    for (m,ex),x in Tm.groupby(['m','ex']):
        x1=x.sort_values('t').drop_duplicates('day')
        for nm,cut in grid(col): print(f'{m} {ex:4s} {nm:28s}: '+' || '.join(f'{e} {ln(x1[(x1.era==e)&(x1[col]>=cut)])}' for e in ('19-21','22-26')))
