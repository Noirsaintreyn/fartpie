"""VALUE-SHAPE models (declared before results): where are the edges of accepted value, found by data-science structure, and does fading them pay?
Point cloud = fractal pivots (5-bar, 5m RTH) of the PRIOR 5 sessions (price only). Four structure finders: TDA (0-dim persistent homology via ripser; clusters cut where persistence gap is largest), OPTICS, spectral clustering (eigengap k), affinity propagation. Baseline: prior-5-day volume-profile value area.
Edges U/L = outer bounds of clusters holding >= 8% of pivots; centre = nearest cluster centre/median. HAR-RV (daily realized vol from 5m returns, HAR(1,5,22) refit yearly) -> expected daily sigma; extreme = open +- 1.0 sigma_hat.
Trade (after 10:00, first per day): price touches within 0.03 ADR of U (short) / L (long) from inside -> fade at next open, stop edge+0.15 ADR (must be beyond entry), exit at centre (MID) or 1R, 24 bars, cost .5.  Variants: EDGE alone, HAR-extreme alone, CONFLUENCE (both)."""
import numpy as np, pandas as pd, warnings, math; warnings.filterwarnings('ignore')
from ripser import ripser
from sklearn.cluster import OPTICS, AffinityPropagation, KMeans
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet').sort_values('dt').drop_duplicates('dt').reset_index(drop=True)
mn=b.dt.dt.hour*60+b.dt.dt.minute; b['d']=b.dt.dt.normalize(); rth=b[(mn>=570)&(mn<960)].copy()
D=rth.groupby('d').agg(o=('open','first'),h=('high','max'),l=('low','min'),c=('close','last'),v=('volume','sum')); D['rng']=D.h-D.l; D['adr']=D.rng.rolling(20).mean().shift()
G={d:g.reset_index(drop=True) for d,g in rth.groupby('d')}; days=list(D.index)
RV=pd.Series({d:(np.diff(np.log(g.close.values))**2).sum() for d,g in G.items()})
H=pd.DataFrame({'y':np.sqrt(RV)}); H['x1']=H.y.shift(); H['x5']=H.y.rolling(5).mean().shift(); H['x22']=H.y.rolling(22).mean().shift(); H['yr']=H.index.year
SIG=pd.Series(np.nan,index=H.index)
for y in range(2019,2027):
    tr=H[(H.yr<y)].dropna(); te=H[H.yr==y].dropna(subset=['x1','x5','x22'])
    if len(tr)<200 or len(te)==0: continue
    A=np.c_[np.ones(len(tr)),tr.x1,tr.x5,tr.x22]; w=np.linalg.lstsq(A,tr.y.values,rcond=None)[0]; SIG[te.index]=np.c_[np.ones(len(te)),te.x1,te.x5,te.x22]@w
def pivots(g):
    h=g.high.values; l=g.low.values; ph=[h[i] for i in range(2,len(g)-2) if h[i]>=h[i-2:i+3].max()]; pl=[l[i] for i in range(2,len(g)-2) if l[i]<=l[i-2:i+3].min()]; return ph+pl
def split_by_gap(x,eps):
    x=np.sort(x); cuts=np.where(np.diff(x)>eps)[0]; return np.split(x,cuts+1)
def edges_from_groups(groups,N):
    keep=[g for g in groups if len(g)>=max(3,.08*N)]
    if not keep: return None
    allp=np.concatenate(keep); return allp.min(),allp.max(),float(np.median(allp))
def m_tda(x,adr):
    if len(x)<12: return None
    dg=ripser(np.sort(x).reshape(-1,1),maxdim=0)['dgms'][0]; death=np.sort(dg[np.isfinite(dg[:,1]),1])[::-1]
    # persistence gap rule: largest ratio drop among the top-6 persistences, clipped to [.05,.40] adr
    cand=death[:7]; eps=.10*adr
    if len(cand)>=3:
        r=cand[:-1]/np.maximum(cand[1:],1e-9); j=int(np.argmax(r)); eps=float(np.clip(math.sqrt(cand[j]*cand[j+1]),.05*adr,.40*adr))
    return edges_from_groups(split_by_gap(x,eps),len(x))
def m_optics(x,adr):
    if len(x)<15: return None
    xs=np.sort(x).reshape(-1,1); lab=OPTICS(min_samples=max(4,len(x)//25),xi=.05,min_cluster_size=.08).fit(xs).labels_
    groups=[xs[lab==k,0] for k in set(lab) if k>=0]; return edges_from_groups(groups,len(x)) if groups else None
def m_spectral(x,adr):
    if len(x)<15: return None
    xs=np.sort(x); sg=.08*adr; W=np.exp(-((xs[:,None]-xs[None,:])**2)/(2*sg**2)); d=W.sum(1); L=np.eye(len(xs))-W/np.sqrt(np.outer(d,d)); ev,vec=np.linalg.eigh(L)
    gaps=np.diff(ev[:7]); k=int(np.argmax(gaps))+1; k=max(1,min(k,6)); 
    if k==1: return edges_from_groups([xs],len(x))
    U=vec[:,:k]; U=U/np.maximum(np.linalg.norm(U,axis=1,keepdims=True),1e-9); lab=KMeans(k,n_init=5,random_state=0).fit_predict(U); return edges_from_groups([xs[lab==j] for j in range(k)],len(x))
def m_ap(x,adr):
    if len(x)<15: return None
    xs=np.sort(x).reshape(-1,1); ap=AffinityPropagation(damping=.8,random_state=0,max_iter=300,preference=-(.12*adr)**2*len(xs)**0*1.0).fit(xs); lab=ap.labels_
    if lab.min()<0: return None
    return edges_from_groups([xs[lab==k,0] for k in set(lab)],len(x))
def m_vp(g5,adr):
    tp=((g5.high+g5.low+g5.close)/3).values; v=g5.volume.values; lo=np.floor(tp.min()/2)*2; k=((tp-lo)/2).astype(int); h=np.bincount(k,weights=v); poc=int(h.argmax()); tot=h.sum(); a=b_=poc; acc=h[poc]
    while acc<.7*tot and (a>0 or b_<len(h)-1):
        up=h[b_+1] if b_<len(h)-1 else -1; dn=h[a-1] if a>0 else -1
        if up>=dn: b_+=1; acc+=up
        else: a-=1; acc+=dn
    return lo+a*2,lo+(b_+1)*2,lo+poc*2+1
def resolve(g,i,dr,st,tgt):
    hi=g.high.values;lo=g.low.values;e=g.open.iloc[i];n=min(len(g),i+24)
    for j in range(i,n):
        if dr>0:
            if lo[j]<=st: return st-e-.5
            if hi[j]>=tgt: return tgt-e-.5
        else:
            if hi[j]>=st: return e-st-.5
            if lo[j]<=tgt: return e-tgt-.5
    return (g.close.iloc[n-1]-e)*dr-.5
METH={'TDA':m_tda,'OPTICS':m_optics,'Spectral':m_spectral,'AffProp':m_ap}
rows=[]; SH=[]
for i,d in enumerate(days):
    if i<25 or d not in G or not D.adr[d]==D.adr[d] or d.year<2019: continue
    adr=D.adr[d]; g=G[d]; op=g.open.iloc[0]
    prior=[days[j] for j in range(i-5,i) if days[j] in G]; x=np.array(sum((pivots(G[q]) for q in prior),[]))
    g5=pd.concat([G[q] for q in prior]); E={'TDA':None,'OPTICS':None,'Spectral':None,'AffProp':None,'VP5':None}
    for nm,f in METH.items():
        try: E[nm]=f(x,adr)
        except Exception as ex: E[nm]=None
    E['VP5']=m_vp(g5,adr)
    sg=SIG.get(d,np.nan); sig_pts=sg*op if sg==sg else np.nan   # HAR sigma (log-return units) -> pts
    # sigma_hat from 5m-return RV is per-day total; extreme = open +- 1 sigma
    SH.append(dict(day=d,**{f'ncl_{k}':(None if v is None else (v[1]-v[0])/adr) for k,v in E.items()}))
    hi=g.high.values;lo=g.low.values;m_=(g.dt.dt.hour*60+g.dt.dt.minute-570).values
    for nm,e_ in E.items():
        if e_ is None: continue
        Lw,Up,ctr=e_
        if not (Up-Lw)>.15*adr: continue
        for i2 in range(6,len(g)-3):
            if m_[i2]>330: break
            sU=hi[i2]>=Up-.03*adr and g.close.values[i2-1]<Up; sL=lo[i2]<=Lw+.03*adr and g.close.values[i2-1]>Lw
            if not (sU or sL): continue
            dr=-1 if sU else 1; e=g.open.iloc[i2+1]; edge=Up if dr<0 else Lw; st=edge+(.15*adr if dr<0 else -.15*adr)
            if not ((dr<0 and ctr<e<st) or (dr>0 and st<e<ctr)): break
            har=(sig_pts==sig_pts) and ((dr<0 and e>=op+1.0*sig_pts) or (dr>0 and e<=op-1.0*sig_pts)); R=abs(e-st)
            for ex,t in (('MID',ctr),('1R',e+dr*R)):
                if (dr<0 and t<e) or (dr>0 and t>e): rows.append(dict(day=d,meth=nm,ex=ex,har=bool(har),dr=dr,pts=resolve(g,i2+1,dr,st,t),R=R,width=(Up-Lw)/adr))
            break
T=pd.DataFrame(rows); T['era']=np.where(T.day<'2022-01-01','19-21','22-26'); T.to_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/value_shape.pkl')
def ln(x): return f'n {len(x):4d}' if len(x)<15 else f'n {len(x):4d} avg {x.pts.mean():+6.2f} win {(x.pts>0).mean():.0%} t {x.pts.mean()/(x.pts.std()/len(x)**.5):+.1f}'
S=pd.DataFrame(SH).set_index('day'); print('median value-area width / ADR by method:',{k[4:]:round(S[k].median(),2) for k in S.columns})
for (meth,ex),x in T.groupby(['meth','ex']):
    print(f'{meth:9s} {ex:3s} EDGE  : '+' || '.join(f'{e} {ln(x[x.era==e])}' for e in ('19-21','22-26')))
    print(f'{"":13s} +HAR  : '+' || '.join(f'{e} {ln(x[(x.era==e)&x.har])}' for e in ('19-21','22-26'))+'   (confluence: also beyond open+-1 HAR sigma)')
